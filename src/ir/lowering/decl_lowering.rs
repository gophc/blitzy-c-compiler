//! # Declaration Lowering
//!
//! Handles the conversion of AST declarations into IR constructs:
//!
//! - **Global variables**: Converted to `GlobalVariable` entries in the `IrModule`
//!   with constant initializers (evaluated at compile time)
//! - **Function definitions**: Creates `IrFunction` with entry block, parameter
//!   allocas, local variable allocas, prologue/epilogue
//! - **Function declarations** (extern): Creates `FunctionDeclaration` entries
//! - **Static local variables**: Lowered as globals with internal linkage
//! - **Thread-local variables**: Lowered with `is_thread_local` flag
//!
//! ## Alloca-First Pattern
//!
//! The central architectural decision: ALL local variables start as `alloca`
//! instructions in the function's entry block, regardless of whether they
//! could be registers. This simplifies lowering enormously:
//!
//! 1. Scan function body for local variable declarations
//! 2. Emit alloca for each in the entry block
//! 3. Store initial values (if any) after the allocas
//! 4. The mem2reg pass (Phase 7) promotes eligible allocas to SSA registers
//!
//! ## Dependencies
//! - `crate::ir::*` — IR types, instructions, function, module, builder
//! - `crate::frontend::parser::ast` — AST declaration nodes
//! - `crate::common::*` — Types, diagnostics, target

use crate::common::diagnostics::{DiagnosticEngine, Span};
use crate::common::fx_hash::FxHashMap;
use crate::common::target::Target;
use crate::common::type_builder::TypeBuilder;
use crate::common::types::{CType, StructField};
use crate::frontend::parser::ast;
use crate::ir::builder::IrBuilder;
use crate::ir::function::{
    CallingConvention, FunctionParam, IrFunction, Linkage as FnLinkage, Visibility as FnVisibility,
};
use crate::ir::instructions::{BlockId, Instruction, Value};
use crate::ir::module::{
    Constant, FunctionDeclaration, GlobalVariable, IrModule, Linkage, Visibility,
};
use crate::ir::types::IrType;

use super::expr_lowering;
use super::stmt_lowering;

// ---------------------------------------------------------------------------
// Symbol resolution helper
// ---------------------------------------------------------------------------

/// Resolve an AST `Symbol` to its string representation using the name table.
/// Returns an empty string if the symbol index is out of bounds.
#[inline]
fn resolve_sym<'a>(
    name_table: &'a [String],
    sym: &crate::common::string_interner::Symbol,
) -> &'a str {
    let idx = sym.as_u32() as usize;
    if idx < name_table.len() {
        &name_table[idx]
    } else {
        ""
    }
}

// ---------------------------------------------------------------------------
// Local variable information struct
// ---------------------------------------------------------------------------

/// Metadata for a local variable discovered during function body scanning.
#[allow(dead_code)]
struct LocalVarInfo {
    name: String,
    c_type: CType,
    is_static: bool,
    has_initializer: bool,
    alignment: Option<usize>,
    span: Span,
    /// For static locals with initializers, store the AST initializer
    /// for compile-time constant evaluation.
    static_init: Option<ast::Initializer>,
}

// ===========================================================================
// Public API — lower_global_variable
// ===========================================================================

/// Lower a global variable declaration/definition to an IR GlobalVariable.
///
/// Extracts the variable name, type, storage class, and initializer from the
/// AST declaration, converts types, evaluates constant initializers, determines
/// linkage/visibility/section/alignment from attributes, and adds the resulting
/// `GlobalVariable` to the `IrModule`.
pub fn lower_global_variable(
    decl: &ast::Declaration,
    module: &mut IrModule,
    target: &Target,
    type_builder: &TypeBuilder,
    diagnostics: &mut DiagnosticEngine,
    name_table: &[String],
) {
    let specifiers = &decl.specifiers;
    let storage_class = specifiers.storage_class;
    let attributes = &specifiers.attributes;

    let is_thread_local = matches!(storage_class, Some(ast::StorageClass::ThreadLocal));
    let is_const = specifiers
        .type_qualifiers
        .iter()
        .any(|q| matches!(q, ast::TypeQualifier::Const));

    for init_decl in &decl.declarators {
        let declarator = &init_decl.declarator;

        let var_name = match extract_declarator_name(declarator, name_table) {
            Some(name) => name,
            None => continue,
        };

        let c_type = resolve_declaration_type(specifiers, declarator, target, name_table);
        let ir_type = IrType::from_ctype(&c_type, target);

        let (initializer, is_definition) = if let Some(ref init) = init_decl.initializer {
            let constant = evaluate_initializer_constant(
                init,
                &c_type,
                target,
                type_builder,
                diagnostics,
                name_table,
            );
            (constant, true)
        } else if matches!(storage_class, Some(ast::StorageClass::Extern)) {
            (None, false)
        } else {
            (Some(Constant::ZeroInit), true)
        };

        let linkage = determine_linkage(storage_class, attributes, name_table);
        let _visibility = determine_visibility(attributes, name_table);
        let section = extract_section_attribute(attributes, name_table);
        let alignment = extract_alignment_attribute(attributes, name_table);

        let mut global = GlobalVariable::new(var_name, ir_type, initializer);
        global.is_definition = is_definition;
        global.linkage = linkage;
        global.is_thread_local = is_thread_local;
        global.section = section;
        global.alignment = alignment;
        global.is_constant = is_const;

        module.add_global(global);
    }
}

// ===========================================================================
// Public API — lower_function_definition
// ===========================================================================

/// Lower a complete function definition: create `IrFunction`, allocate
/// parameters, scan for local variables, create allocas, verify termination,
/// and add the function to the `IrModule`.
///
/// Implements the mandated alloca-first pattern.
pub fn lower_function_definition(
    func_def: &ast::FunctionDefinition,
    module: &mut IrModule,
    target: &Target,
    type_builder: &TypeBuilder,
    diagnostics: &mut DiagnosticEngine,
    name_table: &[String],
    enum_constants: &FxHashMap<String, i128>,
    struct_defs: &FxHashMap<String, CType>,
) {
    let specifiers = &func_def.specifiers;
    let declarator = &func_def.declarator;

    let func_name = match extract_declarator_name(declarator, name_table) {
        Some(name) => name,
        None => {
            diagnostics.emit_error(func_def.span, "function definition without a name");
            return;
        }
    };

    let return_c_type = resolve_base_type(specifiers, target);
    let return_ir_type = IrType::from_ctype(&return_c_type, target);

    let (param_declarations, is_variadic) = extract_function_params(declarator);

    let mut builder = IrBuilder::new();

    // Build parameter list with SSA values.
    let mut ir_params = Vec::with_capacity(param_declarations.len());
    for param_decl in &param_declarations {
        let param_name = extract_param_name(param_decl, name_table);
        let param_c_type = resolve_param_type(param_decl, target, name_table);
        let param_ir_type = IrType::from_ctype(&param_c_type, target);
        let param_value = builder.fresh_value();
        ir_params.push(FunctionParam::new(param_name, param_ir_type, param_value));
    }

    // Create IrFunction.
    let mut ir_function = IrFunction::new(func_name.clone(), ir_params, return_ir_type.clone());
    ir_function.calling_convention = CallingConvention::C;
    ir_function.is_variadic = is_variadic;
    ir_function.is_definition = true;

    // Collect all attributes from specifiers, function attrs, and declarator.
    let all_attributes = collect_all_attributes(specifiers, &func_def.attributes, declarator);

    // Set linkage.
    let mod_linkage = determine_linkage(specifiers.storage_class, &all_attributes, name_table);
    ir_function.linkage = convert_linkage_to_fn(mod_linkage);

    // Set visibility.
    let mod_vis = determine_visibility(&all_attributes, name_table);
    ir_function.visibility = convert_visibility_to_fn(mod_vis);

    // Set noreturn.
    ir_function.is_noreturn = specifiers
        .function_specifiers
        .iter()
        .any(|fs| matches!(fs, ast::FunctionSpecifier::Noreturn))
        || has_attribute(&all_attributes, "noreturn", name_table);

    // Set section and alignment.
    ir_function.section = extract_section_attribute(&all_attributes, name_table);
    ir_function.alignment = extract_alignment_attribute(&all_attributes, name_table);

    // --- Alloca-first: parameter allocation in entry block ---
    // Consume BlockId(0) from the builder so that subsequent create_block()
    // calls return BlockId(1), BlockId(2), etc.  The entry block already
    // exists (created by IrFunction::new) at blocks[0] = BlockId(0).
    let entry_block_id = builder.create_block(); // Returns BlockId(0), bumps next_block to 1
    debug_assert_eq!(entry_block_id.index(), 0);
    builder.set_insert_point(entry_block_id);

    let mut local_vars: FxHashMap<String, Value> = FxHashMap::default();

    allocate_parameters(
        &mut builder,
        &mut ir_function,
        &param_declarations,
        target,
        &mut local_vars,
        name_table,
    );

    // --- Alloca-first: scan for ALL local variable declarations ---
    let body_stmt = ast::Statement::Compound(func_def.body.clone());
    let mut locals = collect_local_variables(&body_stmt, name_table);

    // Resolve struct/union forward references: when a local variable is
    // declared as `struct S s;` and the struct definition came from a
    // top-level declaration, the collected CType has empty fields.
    // Replace with the full definition from struct_defs.
    for local_info in &mut locals {
        resolve_struct_forward_ref(&mut local_info.c_type, struct_defs);
    }

    // Handle static locals separately — they become globals.
    for local_info in &locals {
        if local_info.is_static {
            lower_static_local(
                &local_info.name,
                &func_name,
                &local_info.c_type,
                local_info.has_initializer,
                local_info.static_init.as_ref(),
                module,
                target,
                type_builder,
                diagnostics,
            );
        }
    }

    // Allocate stack-local variables in entry block.
    let stack_locals: Vec<&LocalVarInfo> = locals.iter().filter(|l| !l.is_static).collect();
    allocate_local_variables(
        &mut builder,
        &mut ir_function,
        &stack_locals,
        target,
        &mut local_vars,
    );

    ir_function.local_count = local_vars.len() as u32;

    // --- Function prologue (IR-level, minimal) ---
    setup_function_prologue(&mut builder, &mut ir_function);

    // --- Build param_values from allocated parameter allocas ---
    // Parameters were stored in local_vars by allocate_parameters. Create
    // a parallel map so that expression lowering can find parameters.
    let mut param_values: FxHashMap<String, Value> = FxHashMap::default();
    for param_decl in &param_declarations {
        let pname = extract_param_name(param_decl, name_table);
        if !pname.is_empty() {
            if let Some(&alloca_val) = local_vars.get(&pname) {
                param_values.insert(pname, alloca_val);
            }
        }
    }

    // --- Build local_types from collected local variable info ---
    let mut local_types: FxHashMap<String, CType> = FxHashMap::default();
    for local_info in &locals {
        local_types.insert(local_info.name.clone(), local_info.c_type.clone());
    }
    // Add parameter types as well.
    for param_decl in &param_declarations {
        let pname = extract_param_name(param_decl, name_table);
        if !pname.is_empty() {
            let ptype = resolve_param_type(param_decl, target, name_table);
            local_types.insert(pname, ptype);
        }
    }

    // --- Build static_locals map from collected locals ---
    let mut static_locals: FxHashMap<String, String> = FxHashMap::default();
    for local_info in &locals {
        if local_info.is_static {
            let mangled = format!("{}.{}", func_name, local_info.name);
            static_locals.insert(local_info.name.clone(), mangled);
        }
    }

    // --- Lower the function body statements ---
    {
        let mut label_blocks: FxHashMap<String, BlockId> = FxHashMap::default();
        let body_stmt = ast::Statement::Compound(func_def.body.clone());
        let mut stmt_ctx = stmt_lowering::StmtLoweringContext {
            builder: &mut builder,
            function: &mut ir_function,
            module,
            target,
            diagnostics,
            local_vars: &mut local_vars,
            label_blocks: &mut label_blocks,
            loop_stack: Vec::new(),
            switch_ctx: None,
            recursion_depth: 0,
            type_builder,
            param_values: &param_values,
            name_table,
            local_types: &local_types,
            enum_constants,
            static_locals: &static_locals,
        };
        stmt_lowering::lower_statement(&mut stmt_ctx, &body_stmt);
    }

    // --- Verify function termination ---
    verify_function_termination(&mut ir_function, &return_ir_type, &mut builder, diagnostics);

    // Sync value_count from builder.
    ir_function.value_count = builder.fresh_value().0;

    module.add_function(ir_function);
}

// ===========================================================================
// Public API — lower_function_declaration
// ===========================================================================

/// Lower an extern function declaration (no body) to a `FunctionDeclaration`.
pub fn lower_function_declaration(
    decl: &ast::Declaration,
    module: &mut IrModule,
    target: &Target,
    _diagnostics: &mut DiagnosticEngine,
    name_table: &[String],
) {
    let specifiers = &decl.specifiers;

    for init_decl in &decl.declarators {
        let declarator = &init_decl.declarator;

        let func_name = match extract_declarator_name(declarator, name_table) {
            Some(name) => name,
            None => continue,
        };

        let return_c_type = resolve_base_type(specifiers, target);
        let return_ir_type = IrType::from_ctype(&return_c_type, target);

        let (param_decls, is_variadic) = extract_function_params(declarator);
        let param_types: Vec<IrType> = param_decls
            .iter()
            .map(|pd| {
                let param_c_type = resolve_param_type(pd, target, name_table);
                IrType::from_ctype(&param_c_type, target)
            })
            .collect();

        let attributes = &specifiers.attributes;
        let linkage = determine_linkage(specifiers.storage_class, attributes, name_table);
        let visibility = determine_visibility(attributes, name_table);

        let mut func_decl = FunctionDeclaration::new(func_name, return_ir_type, param_types);
        func_decl.is_variadic = is_variadic;
        func_decl.linkage = linkage;
        func_decl.visibility = visibility;

        module.add_declaration(func_decl);
    }
}

// ===========================================================================
// Public API - lower_local_initializer
// ===========================================================================

/// Lower a local variable initializer and emit store instructions.
///
/// Called after all allocas are created. For scalar types, lowers the
/// initializer expression and stores to the alloca. For aggregates,
/// handles initializer lists with per-element GEP + store.
pub fn lower_local_initializer(
    var_alloca: Value,
    initializer: &ast::Initializer,
    var_type: &CType,
    ctx: &mut expr_lowering::ExprLoweringContext<'_>,
) {
    match initializer {
        ast::Initializer::Expression(expr) => {
            let init_val = expr_lowering::lower_expression(ctx, expr);
            let store_inst = ctx.builder.build_store(init_val, var_alloca, Span::dummy());
            emit_inst_to_ctx(ctx, store_inst);
        }
        ast::Initializer::List {
            designators_and_initializers,
            span,
            ..
        } => {
            lower_aggregate_local_init(
                var_alloca,
                designators_and_initializers,
                var_type,
                ctx,
                *span,
            );
        }
    }
}

// ===========================================================================
// Aggregate local initializer lowering
// ===========================================================================

/// Lower an aggregate initializer list (struct or array) into GEP + store
/// instructions for a local variable.
fn lower_aggregate_local_init(
    base_alloca: Value,
    init_list: &[ast::DesignatedInitializer],
    target_type: &CType,
    ctx: &mut expr_lowering::ExprLoweringContext<'_>,
    span: Span,
) {
    let resolved = crate::common::types::resolve_typedef(target_type);

    match resolved {
        CType::Array(element_type, size_opt) => {
            let array_len = size_opt.unwrap_or(init_list.len());
            for (idx, desig_init) in init_list.iter().enumerate() {
                if idx >= array_len {
                    break;
                }
                let actual_idx = resolve_designator_index(&desig_init.designators, idx);
                let idx_val = make_index_value(ctx, actual_idx);
                let (elem_ptr, gep_inst) =
                    ctx.builder
                        .build_gep(base_alloca, vec![idx_val], IrType::Ptr, span);
                emit_inst_to_ctx(ctx, gep_inst);
                lower_single_init_element(elem_ptr, &desig_init.initializer, element_type, ctx);
            }
        }
        CType::Struct { ref fields, .. } => {
            for (idx, desig_init) in init_list.iter().enumerate() {
                let field_idx =
                    resolve_field_designator(&desig_init.designators, fields, idx, ctx.name_table);
                if field_idx < fields.len() {
                    let field = &fields[field_idx];
                    let field_idx_val = make_index_value(ctx, field_idx);
                    let (field_ptr, gep_inst) =
                        ctx.builder
                            .build_gep(base_alloca, vec![field_idx_val], IrType::Ptr, span);
                    emit_inst_to_ctx(ctx, gep_inst);
                    lower_single_init_element(field_ptr, &desig_init.initializer, &field.ty, ctx);
                }
            }
        }
        CType::Union { ref fields, .. } => {
            if let Some(first) = init_list.first() {
                let field_idx =
                    resolve_field_designator(&first.designators, fields, 0, ctx.name_table);
                if field_idx < fields.len() {
                    lower_single_init_element(
                        base_alloca,
                        &first.initializer,
                        &fields[field_idx].ty,
                        ctx,
                    );
                }
            }
        }
        _ => {
            // Scalar with brace-enclosed init: int x = {42};
            if let Some(first) = init_list.first() {
                lower_single_init_element(base_alloca, &first.initializer, target_type, ctx);
            }
        }
    }
}

/// Lower a single initializer element -- recursively handles nested lists.
fn lower_single_init_element(
    ptr: Value,
    init: &ast::Initializer,
    elem_type: &CType,
    ctx: &mut expr_lowering::ExprLoweringContext<'_>,
) {
    match init {
        ast::Initializer::Expression(expr) => {
            let val = expr_lowering::lower_expression(ctx, expr);
            let store_inst = ctx.builder.build_store(val, ptr, Span::dummy());
            emit_inst_to_ctx(ctx, store_inst);
        }
        ast::Initializer::List {
            designators_and_initializers,
            span,
            ..
        } => {
            lower_aggregate_local_init(ptr, designators_and_initializers, elem_type, ctx, *span);
        }
    }
}

// ===========================================================================
// Constant initializer evaluation (for globals)
// ===========================================================================

/// Evaluate an AST initializer as a compile-time constant for global variables.
fn evaluate_initializer_constant(
    init: &ast::Initializer,
    expected_type: &CType,
    target: &Target,
    type_builder: &TypeBuilder,
    diagnostics: &mut DiagnosticEngine,
    name_table: &[String],
) -> Option<Constant> {
    match init {
        ast::Initializer::Expression(expr) => evaluate_constant_expr(
            expr,
            expected_type,
            target,
            type_builder,
            diagnostics,
            name_table,
        ),
        ast::Initializer::List {
            designators_and_initializers,
            ..
        } => lower_designated_initializer(
            designators_and_initializers,
            expected_type,
            target,
            type_builder,
            diagnostics,
            name_table,
        ),
    }
}

/// Evaluate a compile-time constant expression for a global variable initializer.
fn evaluate_constant_expr(
    expr: &ast::Expression,
    _expected_type: &CType,
    target: &Target,
    type_builder: &TypeBuilder,
    diagnostics: &mut DiagnosticEngine,
    name_table: &[String],
) -> Option<Constant> {
    match expr {
        ast::Expression::IntegerLiteral { value, .. } => Some(Constant::Integer(*value as i128)),
        ast::Expression::FloatLiteral { value, .. } => Some(Constant::Float(*value)),
        ast::Expression::StringLiteral { segments, .. } => {
            // Concatenate all segments for the string value.
            let mut bytes = Vec::new();
            for seg in segments {
                bytes.extend_from_slice(&seg.value);
            }
            Some(Constant::String(bytes))
        }
        ast::Expression::CharLiteral { value, .. } => Some(Constant::Integer(*value as i128)),
        ast::Expression::Identifier { name, .. } => {
            let name_str = resolve_sym(name_table, name).to_string();
            Some(Constant::GlobalRef(name_str))
        }
        ast::Expression::UnaryOp { op, operand, .. } => match op {
            ast::UnaryOp::AddressOf => {
                if let ast::Expression::Identifier { name, .. } = operand.as_ref() {
                    let name_str = resolve_sym(name_table, name).to_string();
                    Some(Constant::GlobalRef(name_str))
                } else {
                    Some(Constant::Undefined)
                }
            }
            ast::UnaryOp::Negate => {
                let inner = evaluate_constant_expr(
                    operand,
                    _expected_type,
                    target,
                    type_builder,
                    diagnostics,
                    name_table,
                )?;
                match inner {
                    Constant::Integer(v) => Some(Constant::Integer(-v)),
                    Constant::Float(v) => Some(Constant::Float(-v)),
                    _ => Some(Constant::Undefined),
                }
            }
            ast::UnaryOp::BitwiseNot => {
                let inner = evaluate_constant_expr(
                    operand,
                    _expected_type,
                    target,
                    type_builder,
                    diagnostics,
                    name_table,
                )?;
                match inner {
                    Constant::Integer(v) => Some(Constant::Integer(!v)),
                    _ => Some(Constant::Undefined),
                }
            }
            ast::UnaryOp::LogicalNot => {
                let inner = evaluate_constant_expr(
                    operand,
                    _expected_type,
                    target,
                    type_builder,
                    diagnostics,
                    name_table,
                )?;
                match inner {
                    Constant::Integer(v) => Some(Constant::Integer(if v == 0 { 1 } else { 0 })),
                    _ => Some(Constant::Undefined),
                }
            }
            _ => Some(Constant::Undefined),
        },
        ast::Expression::Binary {
            op, left, right, ..
        } => {
            let lhs = evaluate_constant_expr(
                left,
                _expected_type,
                target,
                type_builder,
                diagnostics,
                name_table,
            )?;
            let rhs = evaluate_constant_expr(
                right,
                _expected_type,
                target,
                type_builder,
                diagnostics,
                name_table,
            )?;
            evaluate_const_binop(op, &lhs, &rhs)
        }
        ast::Expression::Cast { operand, .. } => evaluate_constant_expr(
            operand,
            _expected_type,
            target,
            type_builder,
            diagnostics,
            name_table,
        ),
        ast::Expression::SizeofExpr { .. } | ast::Expression::SizeofType { .. } => Some(
            Constant::Integer(evaluate_sizeof_expr(expr, target, type_builder) as i128),
        ),
        ast::Expression::AlignofType { .. } => {
            Some(Constant::Integer(
                evaluate_alignof_expr(expr, target, type_builder) as i128,
            ))
        }
        ast::Expression::Conditional {
            condition,
            then_expr,
            else_expr,
            ..
        } => {
            let cond = evaluate_constant_expr(
                condition,
                _expected_type,
                target,
                type_builder,
                diagnostics,
                name_table,
            )?;
            match cond {
                Constant::Integer(v) if v != 0 => {
                    if let Some(ref te) = then_expr {
                        evaluate_constant_expr(
                            te,
                            _expected_type,
                            target,
                            type_builder,
                            diagnostics,
                            name_table,
                        )
                    } else {
                        // GCC extension: x ?: y — if condition is true, value is condition
                        Some(Constant::Integer(v))
                    }
                }
                Constant::Integer(_) => evaluate_constant_expr(
                    else_expr,
                    _expected_type,
                    target,
                    type_builder,
                    diagnostics,
                    name_table,
                ),
                _ => Some(Constant::Undefined),
            }
        }
        ast::Expression::Parenthesized { inner, .. } => evaluate_constant_expr(
            inner,
            _expected_type,
            target,
            type_builder,
            diagnostics,
            name_table,
        ),
        _ => {
            diagnostics.emit_error(
                Span::dummy(),
                "initializer element is not a compile-time constant",
            );
            Some(Constant::Undefined)
        }
    }
}

/// Evaluate a constant binary operation.
fn evaluate_const_binop(op: &ast::BinaryOp, lhs: &Constant, rhs: &Constant) -> Option<Constant> {
    match (lhs, rhs) {
        (Constant::Integer(a), Constant::Integer(b)) => {
            let result = match op {
                ast::BinaryOp::Add => a.wrapping_add(*b),
                ast::BinaryOp::Sub => a.wrapping_sub(*b),
                ast::BinaryOp::Mul => a.wrapping_mul(*b),
                ast::BinaryOp::Div if *b != 0 => a.wrapping_div(*b),
                ast::BinaryOp::Mod if *b != 0 => a.wrapping_rem(*b),
                ast::BinaryOp::BitwiseAnd => *a & *b,
                ast::BinaryOp::BitwiseOr => *a | *b,
                ast::BinaryOp::BitwiseXor => *a ^ *b,
                ast::BinaryOp::ShiftLeft => a.wrapping_shl(*b as u32),
                ast::BinaryOp::ShiftRight => a.wrapping_shr(*b as u32),
                ast::BinaryOp::Equal => {
                    if a == b {
                        1
                    } else {
                        0
                    }
                }
                ast::BinaryOp::NotEqual => {
                    if a != b {
                        1
                    } else {
                        0
                    }
                }
                ast::BinaryOp::Less => {
                    if a < b {
                        1
                    } else {
                        0
                    }
                }
                ast::BinaryOp::LessEqual => {
                    if a <= b {
                        1
                    } else {
                        0
                    }
                }
                ast::BinaryOp::Greater => {
                    if a > b {
                        1
                    } else {
                        0
                    }
                }
                ast::BinaryOp::GreaterEqual => {
                    if a >= b {
                        1
                    } else {
                        0
                    }
                }
                ast::BinaryOp::LogicalAnd => {
                    if *a != 0 && *b != 0 {
                        1
                    } else {
                        0
                    }
                }
                ast::BinaryOp::LogicalOr => {
                    if *a != 0 || *b != 0 {
                        1
                    } else {
                        0
                    }
                }
                _ => return Some(Constant::Undefined),
            };
            Some(Constant::Integer(result))
        }
        (Constant::Float(a), Constant::Float(b)) => {
            let result = match op {
                ast::BinaryOp::Add => a + b,
                ast::BinaryOp::Sub => a - b,
                ast::BinaryOp::Mul => a * b,
                ast::BinaryOp::Div if *b != 0.0 => a / b,
                _ => return Some(Constant::Undefined),
            };
            Some(Constant::Float(result))
        }
        _ => Some(Constant::Undefined),
    }
}

// ===========================================================================
// Designated initializer lowering (constant, for globals)
// ===========================================================================

/// Lower a designated initializer list for a struct or array global.
fn lower_designated_initializer(
    init_list: &[ast::DesignatedInitializer],
    target_type: &CType,
    target: &Target,
    type_builder: &TypeBuilder,
    diagnostics: &mut DiagnosticEngine,
    name_table: &[String],
) -> Option<Constant> {
    let resolved = crate::common::types::resolve_typedef(target_type);

    match resolved {
        CType::Array(element_type, size_opt) => {
            let array_len = size_opt.unwrap_or(init_list.len());
            let mut elements = vec![Constant::ZeroInit; array_len];
            let mut current_idx = 0usize;
            for desig_init in init_list {
                current_idx = resolve_designator_index(&desig_init.designators, current_idx);
                if current_idx < array_len {
                    if let Some(c) = evaluate_initializer_constant(
                        &desig_init.initializer,
                        element_type,
                        target,
                        type_builder,
                        diagnostics,
                        name_table,
                    ) {
                        elements[current_idx] = c;
                    }
                }
                current_idx += 1;
            }
            Some(Constant::Array(elements))
        }
        CType::Struct { ref fields, .. } => {
            let field_count = fields.len();
            let mut field_values = vec![Constant::ZeroInit; field_count];
            let mut current_idx = 0usize;
            for desig_init in init_list {
                current_idx = resolve_field_designator(
                    &desig_init.designators,
                    fields,
                    current_idx,
                    name_table,
                );
                if current_idx < field_count {
                    let field_type = &fields[current_idx].ty;
                    if let Some(c) = evaluate_initializer_constant(
                        &desig_init.initializer,
                        field_type,
                        target,
                        type_builder,
                        diagnostics,
                        name_table,
                    ) {
                        field_values[current_idx] = c;
                    }
                }
                current_idx += 1;
            }
            Some(Constant::Struct(field_values))
        }
        CType::Union { ref fields, .. } => {
            if let Some(first) = init_list.first() {
                let field_idx = resolve_field_designator(&first.designators, fields, 0, name_table);
                if field_idx < fields.len() {
                    return evaluate_initializer_constant(
                        &first.initializer,
                        &fields[field_idx].ty,
                        target,
                        type_builder,
                        diagnostics,
                        name_table,
                    );
                }
            }
            Some(Constant::ZeroInit)
        }
        _ => {
            if let Some(first) = init_list.first() {
                evaluate_initializer_constant(
                    &first.initializer,
                    target_type,
                    target,
                    type_builder,
                    diagnostics,
                    name_table,
                )
            } else {
                Some(Constant::ZeroInit)
            }
        }
    }
}

// ===========================================================================
// Parameter allocation
// ===========================================================================

/// Create allocas for all function parameters in the entry block.
fn allocate_parameters(
    builder: &mut IrBuilder,
    function: &mut IrFunction,
    params: &[ast::ParameterDeclaration],
    target: &Target,
    local_vars: &mut FxHashMap<String, Value>,
    name_table: &[String],
) {
    for (idx, param_decl) in params.iter().enumerate() {
        let param_name = extract_param_name(param_decl, name_table);
        if param_name.is_empty() {
            continue;
        }

        let param_c_type = resolve_param_type(param_decl, target, name_table);
        let param_ir_type = IrType::from_ctype(&param_c_type, target);

        let (alloca_val, alloca_inst) = builder.build_alloca(param_ir_type, Span::dummy());
        push_inst_to_entry(function, alloca_inst);

        let param_value = if idx < function.params.len() {
            function.params[idx].value
        } else {
            Value::UNDEF
        };

        let store_inst = builder.build_store(param_value, alloca_val, Span::dummy());
        push_inst_to_entry(function, store_inst);

        local_vars.insert(param_name, alloca_val);
    }
}

// ===========================================================================
// Local variable scanning
// ===========================================================================

/// Recursively scan the function body for ALL local variable declarations.
fn collect_local_variables(body: &ast::Statement, name_table: &[String]) -> Vec<LocalVarInfo> {
    let mut locals = Vec::new();
    collect_locals_recursive(body, &mut locals, name_table);
    locals
}

fn collect_locals_recursive(
    stmt: &ast::Statement,
    locals: &mut Vec<LocalVarInfo>,
    name_table: &[String],
) {
    match stmt {
        ast::Statement::Compound(compound) => {
            for item in &compound.items {
                match item {
                    ast::BlockItem::Declaration(decl) => {
                        collect_locals_from_declaration(decl, locals, name_table);
                    }
                    ast::BlockItem::Statement(inner_stmt) => {
                        collect_locals_recursive(inner_stmt, locals, name_table);
                    }
                }
            }
        }
        ast::Statement::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_locals_recursive(then_branch, locals, name_table);
            if let Some(ref else_stmt) = else_branch {
                collect_locals_recursive(else_stmt, locals, name_table);
            }
        }
        ast::Statement::While { body, .. } | ast::Statement::DoWhile { body, .. } => {
            collect_locals_recursive(body, locals, name_table);
        }
        ast::Statement::For { init, body, .. } => {
            if let Some(ref for_init) = init {
                match for_init {
                    ast::ForInit::Declaration(decl) => {
                        collect_locals_from_declaration(decl, locals, name_table);
                    }
                    ast::ForInit::Expression(_) => {}
                }
            }
            collect_locals_recursive(body, locals, name_table);
        }
        ast::Statement::Switch { body, .. } => {
            collect_locals_recursive(body, locals, name_table);
        }
        ast::Statement::Labeled { statement, .. }
        | ast::Statement::Case { statement, .. }
        | ast::Statement::Default { statement, .. } => {
            collect_locals_recursive(statement, locals, name_table);
        }
        ast::Statement::CaseRange { statement, .. } => {
            collect_locals_recursive(statement, locals, name_table);
        }
        _ => {}
    }
}

fn collect_locals_from_declaration(
    decl: &ast::Declaration,
    locals: &mut Vec<LocalVarInfo>,
    name_table: &[String],
) {
    let specifiers = &decl.specifiers;
    let storage_class = specifiers.storage_class;

    if matches!(storage_class, Some(ast::StorageClass::Extern)) {
        return;
    }

    let is_static = matches!(storage_class, Some(ast::StorageClass::Static))
        || matches!(storage_class, Some(ast::StorageClass::ThreadLocal));

    for init_decl in &decl.declarators {
        let declarator = &init_decl.declarator;
        let var_name = match extract_declarator_name(declarator, name_table) {
            Some(name) => name,
            None => continue,
        };

        let c_type = resolve_declaration_type(specifiers, declarator, &Target::X86_64, name_table);
        let alignment = extract_alignment_attribute(&specifiers.attributes, name_table);

        let static_init = if is_static {
            init_decl.initializer.clone()
        } else {
            None
        };

        locals.push(LocalVarInfo {
            name: var_name,
            c_type,
            is_static,
            has_initializer: init_decl.initializer.is_some(),
            alignment,
            span: decl.span,
            static_init,
        });
    }
}

// ===========================================================================
// Local variable allocation
// ===========================================================================

fn allocate_local_variables(
    builder: &mut IrBuilder,
    function: &mut IrFunction,
    locals: &[&LocalVarInfo],
    target: &Target,
    local_vars: &mut FxHashMap<String, Value>,
) {
    for local in locals {
        let ir_type = IrType::from_ctype(&local.c_type, target);
        let (alloca_val, alloca_inst) = builder.build_alloca(ir_type, local.span);
        push_inst_to_entry(function, alloca_inst);
        local_vars.insert(local.name.clone(), alloca_val);
    }
}

// ===========================================================================
// Static local and thread-local variable lowering
// ===========================================================================

fn lower_static_local(
    name: &str,
    func_name: &str,
    c_type: &CType,
    has_initializer: bool,
    init_expr: Option<&ast::Initializer>,
    module: &mut IrModule,
    target: &Target,
    _type_builder: &TypeBuilder,
    _diagnostics: &mut DiagnosticEngine,
) {
    let mangled_name = format!("{}.{}", func_name, name);
    let ir_type = IrType::from_ctype(c_type, target);

    // Attempt to extract a compile-time constant from the initializer.
    let constant = if has_initializer {
        if let Some(ast::Initializer::Expression(expr)) = init_expr {
            eval_static_init_expr(expr, c_type)
        } else {
            Some(Constant::ZeroInit)
        }
    } else {
        Some(Constant::ZeroInit)
    };

    let mut global = GlobalVariable::new(mangled_name, ir_type, constant);
    global.linkage = Linkage::Internal;
    global.is_definition = true;
    module.add_global(global);
}

/// Evaluate a simple constant expression for static variable initialization.
fn eval_static_init_expr(expr: &ast::Expression, _c_type: &CType) -> Option<Constant> {
    match expr {
        ast::Expression::IntegerLiteral { value, .. } => Some(Constant::Integer(*value as i128)),
        ast::Expression::UnaryOp { op, operand, .. } => {
            if matches!(op, ast::UnaryOp::Negate) {
                if let ast::Expression::IntegerLiteral { value, .. } = operand.as_ref() {
                    return Some(Constant::Integer(-(*value as i128)));
                }
            }
            Some(Constant::ZeroInit)
        }
        _ => Some(Constant::ZeroInit),
    }
}

// ===========================================================================
// Function prologue/epilogue
// ===========================================================================

fn setup_function_prologue(_builder: &mut IrBuilder, _function: &mut IrFunction) {
    // Prologue is implicitly defined by alloca instructions in the entry block.
}

fn verify_function_termination(
    function: &mut IrFunction,
    return_type: &IrType,
    builder: &mut IrBuilder,
    diagnostics: &mut DiagnosticEngine,
) {
    let block_count = function.block_count();
    for idx in 0..block_count {
        if let Some(block) = function.get_block_mut(idx) {
            if !block.has_terminator() {
                if return_type.is_void() {
                    let ret_inst = builder.build_return(None, Span::dummy());
                    block.push_instruction(ret_inst);
                } else {
                    diagnostics
                        .emit_warning(Span::dummy(), "control reaches end of non-void function");
                    let ret_inst = builder.build_return(Some(Value::UNDEF), Span::dummy());
                    block.push_instruction(ret_inst);
                }
            }
        }
    }
}

// ===========================================================================
// Linkage and visibility helpers
// ===========================================================================

fn determine_linkage(
    storage_class: Option<ast::StorageClass>,
    attributes: &[ast::Attribute],
    name_table: &[String],
) -> Linkage {
    if has_attribute(attributes, "weak", name_table) {
        return Linkage::Weak;
    }
    match storage_class {
        Some(ast::StorageClass::Static) => Linkage::Internal,
        Some(ast::StorageClass::Extern) => Linkage::External,
        _ => Linkage::External,
    }
}

fn determine_visibility(attributes: &[ast::Attribute], name_table: &[String]) -> Visibility {
    for attr in attributes {
        if resolve_sym(name_table, &attr.name) == "visibility" {
            if let Some(first_arg) = attr.args.first() {
                match first_arg {
                    ast::AttributeArg::String(bytes, _) => {
                        let val = String::from_utf8_lossy(bytes);
                        match val.as_ref() {
                            "hidden" => return Visibility::Hidden,
                            "protected" => return Visibility::Protected,
                            "default" => return Visibility::Default,
                            _ => {}
                        }
                    }
                    ast::AttributeArg::Identifier(sym, _) => match resolve_sym(name_table, sym) {
                        "hidden" => return Visibility::Hidden,
                        "protected" => return Visibility::Protected,
                        "default" => return Visibility::Default,
                        _ => {}
                    },
                    _ => {}
                }
            }
        }
    }
    Visibility::Default
}

/// Convert module::Linkage to function::Linkage.
fn convert_linkage_to_fn(linkage: Linkage) -> FnLinkage {
    match linkage {
        Linkage::External => FnLinkage::External,
        Linkage::Internal => FnLinkage::Internal,
        Linkage::Weak => FnLinkage::Weak,
        Linkage::Common => FnLinkage::External,
    }
}

/// Convert module::Visibility to function::Visibility.
fn convert_visibility_to_fn(vis: Visibility) -> FnVisibility {
    match vis {
        Visibility::Default => FnVisibility::Default,
        Visibility::Hidden => FnVisibility::Hidden,
        Visibility::Protected => FnVisibility::Protected,
    }
}

// ===========================================================================
// Attribute extraction helpers
// ===========================================================================

fn has_attribute(attributes: &[ast::Attribute], name: &str, name_table: &[String]) -> bool {
    attributes
        .iter()
        .any(|a| resolve_sym(name_table, &a.name) == name)
}

fn extract_section_attribute(
    attributes: &[ast::Attribute],
    name_table: &[String],
) -> Option<String> {
    for attr in attributes {
        if resolve_sym(name_table, &attr.name) == "section" {
            if let Some(first_arg) = attr.args.first() {
                match first_arg {
                    ast::AttributeArg::String(bytes, _) => {
                        return Some(String::from_utf8_lossy(bytes).into_owned());
                    }
                    ast::AttributeArg::Identifier(sym, _) => {
                        return Some(resolve_sym(name_table, sym).to_string());
                    }
                    _ => {}
                }
            }
        }
    }
    None
}

/// Extract struct/union fields from the AST `StructOrUnionSpecifier` into
/// `StructField` entries that can be used in `CType::Struct` / `CType::Union`.
pub fn extract_struct_union_fields(spec: &ast::StructOrUnionSpecifier) -> Vec<StructField> {
    // Use thread-local name table for symbol resolution
    let name_table =
        super::INTERNER_SNAPSHOT.with(|snap| snap.borrow().as_ref().cloned().unwrap_or_default());
    let mut fields = Vec::new();
    if let Some(ref members) = spec.members {
        for member in members {
            // Resolve base type from the member's specifier-qualifier list
            let member_base = resolve_base_type_from_sqlist(&member.specifiers);
            if member.declarators.is_empty() {
                // Anonymous struct/union member (no declarators)
                fields.push(StructField {
                    name: None,
                    ty: member_base.clone(),
                    bit_width: None,
                });
            } else {
                for sd in &member.declarators {
                    let bit_width = sd
                        .bit_width
                        .as_ref()
                        .and_then(|e| evaluate_const_int_expr(e).map(|v| v as u32));
                    if let Some(ref declarator) = sd.declarator {
                        let name = extract_declarator_name(declarator, &name_table);
                        // Apply pointer/array modifiers from declarator
                        let member_type =
                            apply_declarator_type(member_base.clone(), declarator, &name_table);
                        fields.push(StructField {
                            name,
                            ty: member_type,
                            bit_width,
                        });
                    } else {
                        // Anonymous bitfield: `int : 3;`
                        fields.push(StructField {
                            name: None,
                            ty: member_base.clone(),
                            bit_width,
                        });
                    }
                }
            }
        }
    }
    fields
}

/// Resolve forward-referenced struct/union types: if a CType::Struct (or Union)
/// has a tag name but empty fields, replace it with the full definition from the
/// struct definitions registry.
fn resolve_struct_forward_ref(ctype: &mut CType, struct_defs: &FxHashMap<String, CType>) {
    match ctype {
        CType::Struct {
            name: Some(ref tag),
            ref fields,
            ..
        } if fields.is_empty() => {
            if let Some(full_def) = struct_defs.get(tag) {
                *ctype = full_def.clone();
            }
        }
        CType::Union {
            name: Some(ref tag),
            ref fields,
            ..
        } if fields.is_empty() => {
            if let Some(full_def) = struct_defs.get(tag) {
                *ctype = full_def.clone();
            }
        }
        CType::Pointer(inner, _) => resolve_struct_forward_ref(inner, struct_defs),
        CType::Array(inner, _) => resolve_struct_forward_ref(inner, struct_defs),
        _ => {}
    }
}

/// Resolve a base C type from a specifier-qualifier list (used for struct members).
pub fn resolve_base_type_from_sqlist(sqlist: &ast::SpecifierQualifierList) -> CType {
    let type_specs = &sqlist.type_specifiers;
    if type_specs.is_empty() {
        return CType::Int;
    }
    if type_specs.len() == 1 {
        return map_single_type_specifier(&type_specs[0]);
    }
    resolve_multi_word_type(type_specs)
}

fn extract_alignment_attribute(
    attributes: &[ast::Attribute],
    name_table: &[String],
) -> Option<usize> {
    for attr in attributes {
        if resolve_sym(name_table, &attr.name) == "aligned" {
            if let Some(ast::AttributeArg::Expression(expr)) = attr.args.first() {
                if let ast::Expression::IntegerLiteral { value, .. } = expr.as_ref() {
                    return Some(*value as usize);
                }
            }
            // aligned without argument defaults to max alignment (usually 16).
            return Some(16);
        }
    }
    None
}

fn collect_all_attributes(
    specifiers: &ast::DeclarationSpecifiers,
    func_attrs: &[ast::Attribute],
    declarator: &ast::Declarator,
) -> Vec<ast::Attribute> {
    let mut all = Vec::new();
    all.extend_from_slice(&specifiers.attributes);
    all.extend_from_slice(func_attrs);
    all.extend_from_slice(&declarator.attributes);
    all
}

// ===========================================================================
// Type resolution helpers
// ===========================================================================

fn resolve_base_type(specifiers: &ast::DeclarationSpecifiers, _target: &Target) -> CType {
    let type_specs = &specifiers.type_specifiers;
    if type_specs.is_empty() {
        return CType::Int;
    }
    if type_specs.len() == 1 {
        return map_single_type_specifier(&type_specs[0]);
    }
    resolve_multi_word_type(type_specs)
}

/// Resolve a Symbol to its actual string name using the name_table.
/// Returns the symbol string if the index is in range, otherwise the
/// debug representation.
fn sym_to_string(sym: &crate::common::string_interner::Symbol, name_table: &[String]) -> String {
    let idx = sym.as_u32() as usize;
    if idx < name_table.len() {
        name_table[idx].clone()
    } else {
        sym.to_string()
    }
}

fn map_single_type_specifier(spec: &ast::TypeSpecifier) -> CType {
    // Use thread-local name table for symbol resolution
    let name_table =
        super::INTERNER_SNAPSHOT.with(|snap| snap.borrow().as_ref().cloned().unwrap_or_default());
    map_single_type_specifier_with_names(spec, &name_table)
}

fn map_single_type_specifier_with_names(spec: &ast::TypeSpecifier, name_table: &[String]) -> CType {
    match spec {
        ast::TypeSpecifier::Void => CType::Void,
        ast::TypeSpecifier::Char => CType::Char,
        ast::TypeSpecifier::Short => CType::Short,
        ast::TypeSpecifier::Int => CType::Int,
        ast::TypeSpecifier::Long => CType::Long,
        ast::TypeSpecifier::Float => CType::Float,
        ast::TypeSpecifier::Double => CType::Double,
        ast::TypeSpecifier::Bool => CType::Bool,
        ast::TypeSpecifier::Signed => CType::Int,
        ast::TypeSpecifier::Unsigned => CType::UInt,
        ast::TypeSpecifier::Struct(s) => {
            let fields = extract_struct_union_fields(s);
            CType::Struct {
                name: s.tag.as_ref().map(|t| sym_to_string(t, name_table)),
                fields,
                packed: false,
                aligned: None,
            }
        }
        ast::TypeSpecifier::Union(u) => {
            let fields = extract_struct_union_fields(u);
            CType::Union {
                name: u.tag.as_ref().map(|t| sym_to_string(t, name_table)),
                fields,
                packed: false,
                aligned: None,
            }
        }
        ast::TypeSpecifier::Enum(e) => CType::Enum {
            name: e.tag.as_ref().map(|t| sym_to_string(t, name_table)),
            underlying_type: Box::new(CType::Int),
        },
        ast::TypeSpecifier::TypedefName(name) => CType::Typedef {
            name: sym_to_string(name, name_table),
            underlying: Box::new(CType::Int),
        },
        ast::TypeSpecifier::Typeof(_) => CType::Int,
        ast::TypeSpecifier::Atomic(_) => CType::Atomic(Box::new(CType::Int)),
        ast::TypeSpecifier::Complex => CType::Complex(Box::new(CType::Double)),
    }
}

fn resolve_multi_word_type(specs: &[ast::TypeSpecifier]) -> CType {
    let mut has_unsigned = false;
    let mut has_signed = false;
    let mut long_count = 0u32;
    let mut has_short = false;
    let mut has_char = false;
    let mut has_double = false;
    let mut has_float = false;
    let mut has_complex = false;

    for spec in specs {
        match spec {
            ast::TypeSpecifier::Unsigned => has_unsigned = true,
            ast::TypeSpecifier::Signed => has_signed = true,
            ast::TypeSpecifier::Long => long_count += 1,
            ast::TypeSpecifier::Short => has_short = true,
            ast::TypeSpecifier::Char => has_char = true,
            ast::TypeSpecifier::Int => {}
            ast::TypeSpecifier::Double => has_double = true,
            ast::TypeSpecifier::Float => has_float = true,
            ast::TypeSpecifier::Complex => has_complex = true,
            other => return map_single_type_specifier(other),
        }
    }

    if has_complex {
        let base = if has_float {
            CType::Float
        } else if long_count > 0 {
            CType::LongDouble
        } else {
            CType::Double
        };
        return CType::Complex(Box::new(base));
    }
    if has_char {
        return if has_unsigned {
            CType::UChar
        } else if has_signed {
            CType::SChar
        } else {
            CType::Char
        };
    }
    if has_short {
        return if has_unsigned {
            CType::UShort
        } else {
            CType::Short
        };
    }
    if long_count >= 2 {
        return if has_unsigned {
            CType::ULongLong
        } else {
            CType::LongLong
        };
    }
    if long_count == 1 {
        if has_double {
            return CType::LongDouble;
        }
        return if has_unsigned {
            CType::ULong
        } else {
            CType::Long
        };
    }
    if has_double {
        return CType::Double;
    }
    if has_float {
        return CType::Float;
    }
    if has_unsigned {
        CType::UInt
    } else {
        CType::Int
    }
}

pub fn resolve_declaration_type(
    specifiers: &ast::DeclarationSpecifiers,
    declarator: &ast::Declarator,
    target: &Target,
    name_table: &[String],
) -> CType {
    let base = resolve_base_type(specifiers, target);
    apply_declarator_type(base, declarator, name_table)
}

fn apply_declarator_type(
    base: CType,
    declarator: &ast::Declarator,
    name_table: &[String],
) -> CType {
    let mut result = apply_direct_declarator(base, &declarator.direct, name_table);
    if let Some(ref pointer) = declarator.pointer {
        result = apply_pointer_layers(result, pointer);
    }
    result
}

fn apply_pointer_layers(base: CType, pointer: &ast::Pointer) -> CType {
    let quals = crate::common::types::TypeQualifiers::default();
    let mut current = CType::Pointer(Box::new(base), quals);
    if let Some(ref inner) = pointer.inner {
        current = apply_pointer_layers(current, inner);
    }
    current
}

fn apply_direct_declarator(
    base: CType,
    direct: &ast::DirectDeclarator,
    name_table: &[String],
) -> CType {
    match direct {
        ast::DirectDeclarator::Identifier(_, _) => base,
        ast::DirectDeclarator::Parenthesized(inner) => {
            apply_declarator_type(base, inner, name_table)
        }
        ast::DirectDeclarator::Array {
            base: inner_dd,
            size,
            ..
        } => {
            let inner_type = apply_direct_declarator(base, inner_dd, name_table);
            let array_size = size.as_ref().and_then(|e| evaluate_const_int_expr(e));
            CType::Array(Box::new(inner_type), array_size)
        }
        ast::DirectDeclarator::Function {
            base: inner_dd,
            params,
            is_variadic,
            ..
        } => {
            let return_type = apply_direct_declarator(base, inner_dd, name_table);
            let param_types: Vec<CType> = params
                .iter()
                .map(|p| resolve_param_type(p, &Target::X86_64, name_table))
                .collect();
            CType::Function {
                return_type: Box::new(return_type),
                params: param_types,
                variadic: *is_variadic,
            }
        }
    }
}

// ===========================================================================
// Parameter and name helpers
// ===========================================================================

fn extract_function_params(declarator: &ast::Declarator) -> (Vec<ast::ParameterDeclaration>, bool) {
    extract_function_params_from_dd(&declarator.direct)
}

fn extract_function_params_from_dd(
    dd: &ast::DirectDeclarator,
) -> (Vec<ast::ParameterDeclaration>, bool) {
    match dd {
        ast::DirectDeclarator::Function {
            params,
            is_variadic,
            ..
        } => (params.clone(), *is_variadic),
        ast::DirectDeclarator::Parenthesized(inner) => extract_function_params(inner),
        _ => (Vec::new(), false),
    }
}

fn extract_param_name(param: &ast::ParameterDeclaration, name_table: &[String]) -> String {
    if let Some(ref declarator) = param.declarator {
        extract_declarator_name(declarator, name_table).unwrap_or_default()
    } else {
        String::new()
    }
}

fn resolve_param_type(
    param: &ast::ParameterDeclaration,
    target: &Target,
    name_table: &[String],
) -> CType {
    let base = resolve_base_type(&param.specifiers, target);
    if let Some(ref declarator) = param.declarator {
        apply_declarator_type(base, declarator, name_table)
    } else {
        base
    }
}

fn extract_declarator_name(declarator: &ast::Declarator, name_table: &[String]) -> Option<String> {
    extract_name_from_dd(&declarator.direct, name_table)
}

fn extract_name_from_dd(dd: &ast::DirectDeclarator, name_table: &[String]) -> Option<String> {
    match dd {
        ast::DirectDeclarator::Identifier(sym, _) => Some(resolve_sym(name_table, sym).to_string()),
        ast::DirectDeclarator::Parenthesized(inner) => extract_declarator_name(inner, name_table),
        ast::DirectDeclarator::Array { base, .. } => extract_name_from_dd(base, name_table),
        ast::DirectDeclarator::Function { base, .. } => extract_name_from_dd(base, name_table),
    }
}

// ===========================================================================
// Constant evaluation utilities
// ===========================================================================

/// Evaluate a constant integer expression at compile time.
/// Returns None if the expression cannot be evaluated as a constant.
fn evaluate_const_int_expr(expr: &ast::Expression) -> Option<usize> {
    match expr {
        ast::Expression::IntegerLiteral { value, .. } => Some(*value as usize),
        ast::Expression::UnaryOp {
            op: ast::UnaryOp::Negate,
            operand,
            ..
        } => {
            let inner = evaluate_const_int_expr(operand)?;
            Some((-(inner as i64)) as usize)
        }
        ast::Expression::Binary {
            op, left, right, ..
        } => {
            let l = evaluate_const_int_expr(left)?;
            let r = evaluate_const_int_expr(right)?;
            match op {
                ast::BinaryOp::Add => Some(l.wrapping_add(r)),
                ast::BinaryOp::Sub => Some(l.wrapping_sub(r)),
                ast::BinaryOp::Mul => Some(l.wrapping_mul(r)),
                ast::BinaryOp::Div if r != 0 => Some(l / r),
                ast::BinaryOp::Mod if r != 0 => Some(l % r),
                ast::BinaryOp::ShiftLeft => Some(l << (r & 63)),
                ast::BinaryOp::ShiftRight => Some(l >> (r & 63)),
                ast::BinaryOp::BitwiseAnd => Some(l & r),
                ast::BinaryOp::BitwiseOr => Some(l | r),
                ast::BinaryOp::BitwiseXor => Some(l ^ r),
                _ => None,
            }
        }
        ast::Expression::Cast { operand, .. } => evaluate_const_int_expr(operand),
        ast::Expression::Parenthesized { inner, .. } => evaluate_const_int_expr(inner),
        _ => None,
    }
}

/// Evaluate sizeof for a type expression (simplified heuristic).
fn evaluate_sizeof_expr(
    _expr: &ast::Expression,
    _target: &Target,
    _type_builder: &TypeBuilder,
) -> u64 {
    // In a full implementation, this would evaluate sizeof(type) using the target.
    // For now, return a reasonable default for the most common case (pointer size).
    8
}

/// Evaluate alignof for a type expression (simplified heuristic).
fn evaluate_alignof_expr(
    _expr: &ast::Expression,
    _target: &Target,
    _type_builder: &TypeBuilder,
) -> u64 {
    // In a full implementation, this would evaluate alignof(type) using the target.
    // For now, return a reasonable default.
    8
}

// ===========================================================================
// Designator resolution helpers
// ===========================================================================

/// Resolve array index from designator, or fall back to default index.
fn resolve_designator_index(designators: &[ast::Designator], default_idx: usize) -> usize {
    if let Some(di) = designators.first() {
        match di {
            ast::Designator::Index(expr, _) => evaluate_const_int_expr(expr).unwrap_or(default_idx),
            _ => default_idx,
        }
    } else {
        default_idx
    }
}

/// Resolve struct/union field index from designator, or fall back to default.
fn resolve_field_designator(
    designators: &[ast::Designator],
    fields: &[crate::common::types::StructField],
    default_idx: usize,
    name_table: &[String],
) -> usize {
    if let Some(di) = designators.first() {
        match di {
            ast::Designator::Field(sym, _) => {
                let field_name = resolve_sym(name_table, sym);
                fields
                    .iter()
                    .position(|f| f.name.as_deref() == Some(field_name))
                    .unwrap_or(default_idx)
            }
            _ => default_idx,
        }
    } else {
        default_idx
    }
}

// ===========================================================================
// Instruction emission helpers
// ===========================================================================

/// Push an instruction into the entry block of the given function.
fn push_inst_to_entry(function: &mut IrFunction, inst: Instruction) {
    function.entry_block_mut().push_instruction(inst);
}

/// Emit an instruction into the current insertion point of the builder context.
fn emit_inst_to_ctx(ctx: &mut expr_lowering::ExprLoweringContext<'_>, inst: Instruction) {
    if let Some(block_id) = ctx.builder.get_insert_block() {
        let block_idx = block_id.0 as usize;
        if let Some(block) = ctx.function.get_block_mut(block_idx) {
            block.push_instruction(inst);
        }
    }
}

/// Create a Value representing an integer constant for GEP indices.
/// Uses the builder to allocate a fresh value identifier.
fn make_index_value(ctx: &mut expr_lowering::ExprLoweringContext<'_>, _index: usize) -> Value {
    ctx.builder.fresh_value()
}
