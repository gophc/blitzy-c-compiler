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
    enum_constants: &FxHashMap<String, i128>,
) {
    let specifiers = &decl.specifiers;
    let storage_class = specifiers.storage_class;
    let attributes = &specifiers.attributes;

    let is_thread_local = matches!(storage_class, Some(ast::StorageClass::ThreadLocal));
    let spec_has_const = specifiers
        .type_qualifiers
        .iter()
        .any(|q| matches!(q, ast::TypeQualifier::Const));

    for init_decl in &decl.declarators {
        // GCC global register variable: `register T *name __asm__("reg");`
        // These bind a C identifier to a specific hardware register (e.g.
        // `tp` on RISC-V for `current`).  They must NOT produce any
        // storage in the ELF output — no .bss, .data, or .comm entry.
        // Instead, store the mapping in `register_globals` so expression
        // lowering can emit register reads/writes.
        if let Some(ref reg_name) = init_decl.asm_register {
            if let Some(var_name) = extract_declarator_name(&init_decl.declarator, name_table) {
                let c_type =
                    resolve_declaration_type(specifiers, &init_decl.declarator, target, name_table);
                module
                    .register_globals
                    .insert(var_name.to_string(), (reg_name.clone(), c_type));
            }
            continue;
        }

        let declarator = &init_decl.declarator;

        // Determine whether the *variable* itself is const.  When the
        // declarator contains a pointer derivation (e.g. `const char *p`),
        // the specifier-level `const` qualifies the pointee, not the
        // pointer variable — so the variable is mutable and must go in
        // `.data`, not `.rodata`.  Only when the declarator adds no pointer
        // (e.g. `const int x`) does the specifier `const` apply to the
        // variable itself.
        let is_const = if declarator.pointer.is_some() {
            false
        } else {
            spec_has_const
        };

        let var_name = match extract_declarator_name(declarator, name_table) {
            Some(name) => name,
            None => continue,
        };

        let c_type = resolve_declaration_type(specifiers, declarator, target, name_table);

        // Declarations whose resolved type is a function type (e.g. using
        // a function typedef: `fs_param_type fs_param_is_bool;`) are function
        // declarations, NOT variable tentative definitions.  They must never
        // produce BSS storage — they are implicitly extern.
        // Use is_function() which resolves through typedefs.
        let is_function_type = crate::common::types::is_function(&c_type);

        let (initializer, is_definition) = if let Some(ref init) = init_decl.initializer {
            let constant = evaluate_initializer_constant(
                init,
                &c_type,
                target,
                type_builder,
                diagnostics,
                name_table,
                enum_constants,
            );
            (constant, true)
        } else if matches!(storage_class, Some(ast::StorageClass::Extern)) || is_function_type {
            (None, false)
        } else {
            (Some(Constant::ZeroInit), true)
        };

        // Infer array size from initializer when the declaration uses [].
        // For `char data[] = "hello"`, the C type is Array(Char, None) but
        // the actual size is determined by the string literal length + 1
        // (for the null terminator). Similarly for brace-init lists.
        let c_type = match (&c_type, &initializer) {
            (CType::Array(elem, None), Some(Constant::String(bytes))) => {
                // String initializer: bytes already includes null terminator.
                CType::Array(elem.clone(), Some(bytes.len()))
            }
            (CType::Array(elem, None), Some(Constant::Array(elems))) => {
                // Brace-init list: size = number of elements
                CType::Array(elem.clone(), Some(elems.len()))
            }
            (CType::Array(elem, None), Some(Constant::Struct(elems))) => {
                // Struct-style list initializer
                CType::Array(elem.clone(), Some(elems.len()))
            }
            _ => c_type,
        };

        // Resolve forward-referenced struct/union types so the IR type
        // carries the full field list (needed for correct initializer byte
        // layout and struct size computation in the backend).
        let c_type = {
            let mut ct = c_type;
            super::SIZEOF_STRUCT_DEFS.with(|defs| {
                if let Some(ref sd) = *defs.borrow() {
                    resolve_struct_forward_ref(&mut ct, sd);
                }
            });
            ct
        };
        let ir_type = IrType::from_ctype(&c_type, target);

        let linkage = determine_linkage(storage_class, attributes, name_table);
        let _visibility = determine_visibility(attributes, name_table);
        let section = extract_section_attribute(attributes, name_table);
        let alignment = extract_alignment_attribute(attributes, name_table);

        // Post-process: convert any Constant::String values at
        // pointer-typed positions into Constant::GlobalRef pointing to
        // interned string literals in .rodata.  Without this, string
        // literal initializers for `const char *` struct fields would be
        // written inline (as raw bytes) rather than as a pointer +
        // relocation to a .rodata entry.
        let initializer =
            initializer.map(|init| fixup_string_ptrs_in_constant(init, &ir_type, module));

        let var_name_owned = var_name.clone();
        let mut global = GlobalVariable::new(var_name, ir_type, initializer);
        global.is_definition = is_definition;
        global.linkage = linkage;
        global.is_thread_local = is_thread_local;
        global.section = section;
        global.alignment = alignment;
        global.is_constant = is_const;

        // Store the original C type so that expression lowering can
        // recover full struct field information (including field names
        // and function pointer signatures) for member access operations.
        module.global_c_types.insert(var_name_owned, c_type.clone());

        module.add_global(global);
    }
}

// ===========================================================================
// String-in-pointer fixup for static initializers
// ===========================================================================

/// Recursively walks a `Constant` initializer alongside its IR type tree
/// and replaces `Constant::String` values that appear at `IrType::Ptr`
/// positions with `Constant::GlobalRef` entries.  The raw string bytes
/// are interned into the module's string pool (→ `.rodata`), and the
/// resulting label (e.g. `.L.str.N`) is used for the `GlobalRef`.
///
/// This is necessary because `evaluate_initializer_constant` always
/// produces `Constant::String(bytes)` for string literals — correct for
/// `char arr[] = "hello"` (array) but wrong for `const char *p = "hello"`
/// (pointer), where the struct field must hold an address, not inline data.
fn fixup_string_ptrs_in_constant(
    constant: Constant,
    ir_ty: &IrType,
    module: &mut IrModule,
) -> Constant {
    match (&constant, ir_ty) {
        // Core case: a String constant at a Ptr-typed position.
        // Intern the bytes and produce a GlobalRef.
        (Constant::String(bytes), IrType::Ptr) => {
            let id = module.intern_string(bytes.clone());
            let label = format!(".L.str.{}", id);
            Constant::GlobalRef(label)
        }
        // Recurse into struct fields.
        (Constant::Struct(fields), IrType::Struct(st)) => {
            let new_fields: Vec<Constant> = fields
                .iter()
                .enumerate()
                .map(|(i, field)| {
                    let ft = st.fields.get(i).unwrap_or(ir_ty);
                    fixup_string_ptrs_in_constant(field.clone(), ft, module)
                })
                .collect();
            Constant::Struct(new_fields)
        }
        // Recurse into array elements.
        (Constant::Array(elements), IrType::Array(elem_ty, _)) => {
            let new_elems: Vec<Constant> = elements
                .iter()
                .map(|e| fixup_string_ptrs_in_constant(e.clone(), elem_ty, module))
                .collect();
            Constant::Array(new_elems)
        }
        // Everything else passes through unchanged.
        _ => constant,
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

    // Build the complete function type using the full declarator.
    // This correctly handles complex cases like function-returning-function-pointer:
    //   void (*memdbDlSym(sqlite3_vfs *pVfs, void *p, const char *zSym))(void)
    // where the return type is `void (*)(void)` and the simple approach of only
    // applying top-level pointer from the declarator would yield bare `void`.
    let base_spec_type = resolve_base_type_fast(specifiers, name_table);
    let full_type = apply_declarator_type(base_spec_type.clone(), declarator, name_table);
    let mut return_c_type = if let CType::Function {
        ref return_type, ..
    } = full_type
    {
        return_type.as_ref().clone()
    } else {
        // Fallback: simple pointer application (shouldn't happen for func defs).
        let mut rt = base_spec_type;
        if let Some(ref ptr) = declarator.pointer {
            rt = apply_pointer_layers(rt, ptr);
        }
        rt
    };
    // Resolve forward-referenced struct/union in the return type so that
    // ABI classification can see the actual struct fields (e.g., for
    // small-struct register return).
    resolve_struct_forward_ref(&mut return_c_type, struct_defs);
    let return_ir_type = IrType::from_ctype(&return_c_type, target);

    // Store the C-level return type so `lower_function_call` can recover
    // it even when the callee IR value is typed as opaque `Pointer(Void)`.
    module
        .func_c_return_types
        .insert(func_name.clone(), return_c_type.clone());

    let (param_declarations, is_variadic) = extract_function_params(declarator);

    let mut builder = IrBuilder::new();

    // Build parameter list with SSA values, and collect C-level param types
    // for correct implicit conversions in `lower_function_call`.
    let mut ir_params = Vec::with_capacity(param_declarations.len());
    let mut c_param_types = Vec::with_capacity(param_declarations.len());
    for param_decl in &param_declarations {
        let param_name = extract_param_name(param_decl, name_table);
        let mut param_c_type = resolve_param_type(param_decl, target, name_table);
        // C11 §6.7.6.3p7: Parameter type adjustments — array→pointer, function→pointer.
        param_c_type = match param_c_type {
            CType::Array(elem, _) => {
                CType::Pointer(elem, crate::common::types::TypeQualifiers::default())
            }
            CType::Function { .. } => CType::Pointer(
                Box::new(param_c_type),
                crate::common::types::TypeQualifiers::default(),
            ),
            other => other,
        };
        // Resolve struct forward references so the IR type has the full
        // field layout — required for correct ABI classification (e.g.,
        // 16-byte structs passed in register pairs).
        resolve_struct_forward_ref(&mut param_c_type, struct_defs);
        c_param_types.push(param_c_type.clone());
        let param_ir_type = IrType::from_ctype(&param_c_type, target);
        let param_value = builder.fresh_value();
        ir_params.push(FunctionParam::new(param_name, param_ir_type, param_value));
    }

    // Store C-level parameter types so `lower_function_call` can insert
    // correct implicit conversions (e.g. sign-extend int → long long).
    module
        .func_c_param_types
        .insert(func_name.clone(), c_param_types);

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

    // --- Seed the typeof resolution context with parameter types ---
    // This allows `typeof(param_name)` inside the function body to resolve
    // correctly during the local variable collection pass.
    super::TYPEOF_CONTEXT.with(|ctx| {
        let mut map = FxHashMap::default();
        for param_decl in &param_declarations {
            let pname = extract_param_name(param_decl, name_table);
            if !pname.is_empty() {
                let ptype = resolve_param_type(param_decl, target, name_table);
                map.insert(pname, ptype);
            }
        }
        *ctx.borrow_mut() = Some(map);
    });

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
                enum_constants,
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
    // Add parameter types as well — resolve struct forward references so
    // that `struct Item *arr` parameter types carry the full struct layout,
    // enabling correct sizeof computation for pointer arithmetic.
    // C11 §6.7.6.3p7: parameter type adjustments (array→pointer, function→pointer).
    for param_decl in &param_declarations {
        let pname = extract_param_name(param_decl, name_table);
        if !pname.is_empty() {
            let mut ptype = resolve_param_type(param_decl, target, name_table);
            // Array-to-pointer / function-to-pointer decay for parameter types.
            ptype = match ptype {
                CType::Array(elem, _) => {
                    CType::Pointer(elem, crate::common::types::TypeQualifiers::default())
                }
                CType::Function { .. } => CType::Pointer(
                    Box::new(ptype),
                    crate::common::types::TypeQualifiers::default(),
                ),
                other => other,
            };
            resolve_struct_forward_ref(&mut ptype, struct_defs);
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

    // --- Pre-scan for labels to support forward references in asm goto ---
    let mut label_blocks: FxHashMap<String, BlockId> = FxHashMap::default();
    {
        let body_stmt = ast::Statement::Compound(func_def.body.clone());
        let mut label_names: Vec<String> = Vec::new();
        collect_label_names(&body_stmt, &mut label_names, name_table);
        for lname in label_names {
            let block_id = builder.create_block();
            ir_function.ensure_block(block_id);
            label_blocks.insert(lname, block_id);
        }
    }

    // --- Register a forward declaration for the current function ---
    // This allows self-referential patterns (recursion, taking own address,
    // passing self as function pointer) within the function body.  The
    // final `module.add_function()` will add the full definition, but
    // identifier lookup needs to find *something* during body lowering.
    {
        let fwd_param_types: Vec<IrType> =
            ir_function.params.iter().map(|p| p.ty.clone()).collect();
        let fwd_decl = crate::ir::module::FunctionDeclaration::new(
            func_name.clone(),
            return_ir_type.clone(),
            fwd_param_types,
        );
        module.add_declaration(fwd_decl);
    }

    // --- Lower the function body statements ---
    {
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
            static_locals: &mut static_locals,
            struct_defs,
            current_function_name: Some(&func_name),
            scope_type_overrides: FxHashMap::default(),
            return_ctype: Some(return_c_type.clone()),
            layout_cache: FxHashMap::default(),
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

        let mut return_c_type = resolve_base_type_fast(specifiers, name_table);
        // Apply pointer indirection from the declarator (e.g., `void *foo()`)
        if let Some(ref ptr) = declarator.pointer {
            return_c_type = apply_pointer_layers(return_c_type, ptr);
        }
        let return_ir_type = IrType::from_ctype(&return_c_type, target);

        // Store the C-level return type for `lower_function_call`.
        module
            .func_c_return_types
            .entry(func_name.clone())
            .or_insert_with(|| return_c_type.clone());

        let (param_decls, is_variadic) = extract_function_params(declarator);
        let mut c_param_types_decl = Vec::with_capacity(param_decls.len());
        let param_types: Vec<IrType> = param_decls
            .iter()
            .map(|pd| {
                let param_c_type = resolve_param_type(pd, target, name_table);
                c_param_types_decl.push(param_c_type.clone());
                IrType::from_ctype(&param_c_type, target)
            })
            .collect();

        // Store C-level parameter types so `lower_function_call` can insert
        // correct implicit conversions (e.g. sign-extend int → long long).
        module
            .func_c_param_types
            .entry(func_name.clone())
            .or_insert(c_param_types_decl);

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
            // ---------------------------------------------------------------
            // Special case: char/unsigned-char array initialized from a
            // string literal — `char a[4] = "abc"`.
            //
            // The expression evaluates to a *pointer* to the .rodata string
            // constant, but C semantics require *copying* the string bytes
            // into the local array (with zero-fill for any remaining
            // elements).  We detect this pattern here and emit byte-by-byte
            // stores (coalesced into word-size stores for efficiency).
            // ---------------------------------------------------------------
            let mut resolved_var_ty = var_type.clone();
            expr_lowering::resolve_forward_ref_type(&mut resolved_var_ty, ctx.struct_defs);

            if let Some((elem_ty, arr_size)) = is_char_array_type(&resolved_var_ty) {
                if let Some(str_bytes) = extract_string_literal_bytes(expr) {
                    lower_char_array_from_string(
                        var_alloca, &str_bytes, arr_size, &elem_ty, ctx,
                    );
                    return;
                }
            }

            // For aggregate types (struct/union) larger than 8 bytes,
            // the backend cannot load/store the entire aggregate through
            // a single register.  Instead we get the source as an lvalue
            // (pointer) and emit chunk-by-chunk load/store pairs, exactly
            // as the assignment path does in expr_lowering::lower_assignment.
            let is_agg = crate::common::types::is_struct_or_union(&resolved_var_ty);
            let struct_size = if is_agg {
                crate::common::types::sizeof_ctype(&resolved_var_ty, ctx.target)
            } else {
                0
            };
            if is_agg && struct_size > 8 {
                // Get the RHS as an lvalue (pointer) — do NOT dereference.
                let rhs_ptr = expr_lowering::lower_lvalue(ctx, expr);

                // Emit individual 8-byte load/store pairs to copy the
                // aggregate from source to destination.
                let span = Span::dummy();
                let mut offset: usize = 0;
                while offset < struct_size {
                    let remaining = struct_size - offset;
                    let (chunk_sz, chunk_ir) = if remaining >= 8 {
                        (8, IrType::I64)
                    } else if remaining >= 4 {
                        (4, IrType::I32)
                    } else if remaining >= 2 {
                        (2, IrType::I16)
                    } else {
                        (1, IrType::I8)
                    };

                    // Compute source address: rhs_ptr + offset
                    let src_addr = if offset == 0 {
                        rhs_ptr
                    } else {
                        let off_val = make_index_value(ctx, offset as i64);
                        let (gep_val, gep_inst) =
                            ctx.builder
                                .build_gep(rhs_ptr, vec![off_val], chunk_ir.clone(), span);
                        emit_inst_to_ctx(ctx, gep_inst);
                        gep_val
                    };

                    // Load chunk from source.
                    let (loaded, li) = ctx.builder.build_load(src_addr, chunk_ir.clone(), span);
                    emit_inst_to_ctx(ctx, li);

                    // Compute destination address: var_alloca + offset
                    let dst_addr = if offset == 0 {
                        var_alloca
                    } else {
                        let off_val2 = make_index_value(ctx, offset as i64);
                        let (gep_val2, gep_inst2) = ctx.builder.build_gep(
                            var_alloca,
                            vec![off_val2],
                            chunk_ir.clone(),
                            span,
                        );
                        emit_inst_to_ctx(ctx, gep_inst2);
                        gep_val2
                    };

                    // Store chunk to destination.
                    let si = ctx.builder.build_store(loaded, dst_addr, span);
                    emit_inst_to_ctx(ctx, si);

                    offset += chunk_sz;
                }
            } else {
                // Scalar or small-aggregate (≤8 bytes): lower the expression
                // and emit a single store.  This handles cases like
                // `char *p = 0;` where the literal 0 has type `int` (I32) but
                // the variable needs a pointer-width (I64) store.
                let typed_val = expr_lowering::lower_expression_typed(ctx, expr);
                let init_val = expr_lowering::insert_implicit_conversion(
                    ctx,
                    typed_val.value,
                    &typed_val.ty,
                    var_type,
                    Span::dummy(),
                );
                let store_inst = ctx.builder.build_store(init_val, var_alloca, Span::dummy());
                emit_inst_to_ctx(ctx, store_inst);
            }
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
            let elem_size = ctx.type_builder.sizeof_type(element_type);

            // Detect brace elision: if element type is a struct and ALL
            // initializers are flat scalar expressions (no nested init-lists
            // or designators), we consume multiple scalars per element.
            let elem_resolved = crate::common::types::resolve_typedef(element_type);
            let scalars_per_elem = count_aggregate_scalar_fields(elem_resolved, ctx.type_builder);
            let has_any_designators = init_list.iter().any(|di| !di.designators.is_empty());
            let all_flat_scalars = scalars_per_elem > 1
                && !has_any_designators
                && init_list
                    .iter()
                    .all(|di| matches!(di.initializer, ast::Initializer::Expression(_)));

            let array_len = if all_flat_scalars {
                size_opt.unwrap_or((init_list.len() + scalars_per_elem - 1) / scalars_per_elem)
            } else {
                size_opt.unwrap_or(init_list.len())
            };

            // C99 §6.7.9/19: zero-initialize elements that are NOT
            // explicitly covered by the init list.  Skip when all
            // elements are supplied to avoid excessive IR generation.
            let effective_init_count = if all_flat_scalars {
                (init_list.len() + scalars_per_elem - 1) / scalars_per_elem
            } else {
                init_list.len()
            };
            let needs_array_zero = effective_init_count < array_len || has_any_designators;
            if needs_array_zero {
                for i in 0..array_len {
                    let byte_offset = (i * elem_size) as i64;
                    zero_init_field(base_alloca, byte_offset, element_type, ctx, span);
                }
            }

            if all_flat_scalars {
                // Brace elision: consume scalars_per_elem initializers per
                // array element, writing them into the struct fields.
                let mut init_cursor = 0usize;
                for arr_idx in 0..array_len {
                    if init_cursor >= init_list.len() {
                        break;
                    }
                    let byte_offset = (arr_idx * elem_size) as i64;
                    let idx_val = make_index_value(ctx, byte_offset);
                    let (elem_ptr, gep_inst) =
                        ctx.builder
                            .build_gep(base_alloca, vec![idx_val], IrType::Ptr, span);
                    emit_inst_to_ctx(ctx, gep_inst);
                    // Fill aggregate fields from flat scalars
                    init_cursor = lower_brace_elision_into_aggregate(
                        elem_ptr,
                        init_list,
                        init_cursor,
                        elem_resolved,
                        ctx,
                        span,
                    );
                }
            } else {
                // Normal (non-elision) array init.
                for (idx, desig_init) in init_list.iter().enumerate() {
                    if idx >= array_len {
                        break;
                    }
                    let actual_idx = resolve_designator_index(&desig_init.designators, idx);
                    let byte_offset = (actual_idx * elem_size) as i64;
                    let idx_val = make_index_value(ctx, byte_offset);
                    let (elem_ptr, gep_inst) =
                        ctx.builder
                            .build_gep(base_alloca, vec![idx_val], IrType::Ptr, span);
                    emit_inst_to_ctx(ctx, gep_inst);
                    lower_single_init_element(elem_ptr, &desig_init.initializer, element_type, ctx);
                }
            }
        }
        CType::Struct {
            ref fields,
            packed: is_packed,
            aligned: explicit_align,
            ..
        } => {
            // Compute full struct layout (handles bitfields properly).
            let layout =
                ctx.type_builder
                    .compute_struct_layout_with_fields(fields, *is_packed, *explicit_align);
            let field_offsets: Vec<usize> = layout.fields.iter().map(|fl| fl.offset).collect();

            // C99 §6.7.9/19: members not explicitly initialized shall
            // be zero-initialized.  Only emit zero-init stores for
            // fields that are NOT covered by the explicit init list.
            // This avoids generating excessive IR instructions for the
            // common case where all fields are supplied.
            let has_designators = init_list.iter().any(|di| !di.designators.is_empty());

            // Detect struct-level brace elision: when the init list is a
            // flat list of scalar expressions with no designators, and
            // the number of items exceeds the number of struct fields,
            // one or more fields must be aggregate sub-objects (arrays
            // or nested structs) that should consume multiple
            // initializers via brace elision per C99 §6.7.9/20.
            //
            // Also detect: struct has fewer fields than init items, AND
            // at least one field is an aggregate type.  This handles
            // `struct { int f[4]; } s = {1,2,3,4};`.
            let all_flat_exprs = !has_designators
                && init_list
                    .iter()
                    .all(|di| matches!(di.initializer, ast::Initializer::Expression(_)));
            let any_aggregate_field = fields.iter().any(|f| {
                let ft = crate::common::types::resolve_typedef(&f.ty);
                matches!(
                    ft,
                    CType::Struct { .. } | CType::Array(..) | CType::Union { .. }
                )
            });
            let use_brace_elision =
                all_flat_exprs && any_aggregate_field && init_list.len() > fields.len();

            if use_brace_elision {
                // Zero-initialize the entire struct first.
                for (fi, field) in fields.iter().enumerate() {
                    let byte_offset = field_offsets.get(fi).copied().unwrap_or(0) as i64;
                    zero_init_field(base_alloca, byte_offset, &field.ty, ctx, span);
                }
                // Use the recursive brace-elision lowering to consume the
                // correct number of flat initializers per aggregate field.
                lower_brace_elision_into_aggregate(
                    base_alloca, init_list, 0, resolved, ctx, span,
                );
            } else {
                let needs_zero_init = init_list.len() < fields.len() || has_designators;
                if needs_zero_init {
                    for (fi, field) in fields.iter().enumerate() {
                        let byte_offset = field_offsets.get(fi).copied().unwrap_or(0) as i64;
                        zero_init_field(base_alloca, byte_offset, &field.ty, ctx, span);
                    }
                }

                // Now store explicitly initialized fields, handling nested
                // designators (e.g., `.origin.x = 1`).
                for (idx, desig_init) in init_list.iter().enumerate() {
                    lower_designated_struct_init(
                        base_alloca,
                        &desig_init.designators,
                        &desig_init.initializer,
                        fields,
                        &field_offsets,
                        &layout.fields,
                        0, // base byte offset
                        idx,
                        ctx,
                        span,
                    );
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
            // Insert an implicit conversion (truncation) when the expression
            // value is wider than the target element type. For example, `0x80`
            // (I32) stored into a `char` element (I8) needs Trunc(I32 → I8)
            // so the backend emits a byte store instead of a 32-bit store.
            let val = expr_lowering::convert_for_store(ctx, val, elem_type);
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
pub(super) fn evaluate_initializer_constant(
    init: &ast::Initializer,
    expected_type: &CType,
    target: &Target,
    type_builder: &TypeBuilder,
    diagnostics: &mut DiagnosticEngine,
    name_table: &[String],
    enum_constants: &FxHashMap<String, i128>,
) -> Option<Constant> {
    match init {
        ast::Initializer::Expression(expr) => evaluate_constant_expr(
            expr,
            expected_type,
            target,
            type_builder,
            diagnostics,
            name_table,
            enum_constants,
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
            enum_constants,
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
    enum_constants: &FxHashMap<String, i128>,
) -> Option<Constant> {
    match expr {
        ast::Expression::IntegerLiteral { value, .. } => Some(Constant::Integer(*value as i128)),
        ast::Expression::FloatLiteral { value, .. } => Some(Constant::Float(*value)),
        ast::Expression::StringLiteral { segments, .. } => {
            // Concatenate all segments for the string value.
            // C string literals include a null terminator — append it
            // so that the byte vector matches the declared array size.
            let mut bytes = Vec::new();
            for seg in segments {
                bytes.extend_from_slice(&seg.value);
            }
            bytes.push(0); // null terminator
            Some(Constant::String(bytes))
        }
        ast::Expression::CharLiteral { value, .. } => Some(Constant::Integer(*value as i128)),
        ast::Expression::Identifier { name, .. } => {
            let name_str = resolve_sym(name_table, name).to_string();
            // Check if this identifier is an enum constant — resolve to
            // its integer value rather than emitting a symbol reference.
            if let Some(&val) = enum_constants.get(&name_str) {
                return Some(Constant::Integer(val));
            }
            Some(Constant::GlobalRef(name_str))
        }
        ast::Expression::UnaryOp { op, operand, .. } => match op {
            ast::UnaryOp::AddressOf => {
                // &identifier
                if let ast::Expression::Identifier { name, .. } = operand.as_ref() {
                    let name_str = resolve_sym(name_table, name).to_string();
                    return Some(Constant::GlobalRef(name_str));
                }
                // &array[index] — address of array element
                if let ast::Expression::ArraySubscript { base, index, .. } = operand.as_ref() {
                    if let Some((sym, byte_off)) =
                        evaluate_address_of_subscript(base, index, target, name_table)
                    {
                        if byte_off == 0 {
                            return Some(Constant::GlobalRef(sym));
                        }
                        return Some(Constant::GlobalRefOffset(sym, byte_off));
                    }
                }
                // &struct->member or &ptr[N] chains
                if let Some((sym, byte_off)) =
                    evaluate_address_of_member_chain(operand, target, name_table)
                {
                    if byte_off == 0 {
                        return Some(Constant::GlobalRef(sym));
                    }
                    return Some(Constant::GlobalRefOffset(sym, byte_off));
                }
                Some(Constant::Undefined)
            }
            ast::UnaryOp::Negate => {
                let inner = evaluate_constant_expr(
                    operand,
                    _expected_type,
                    target,
                    type_builder,
                    diagnostics,
                    name_table,
                    enum_constants,
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
                    enum_constants,
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
                    enum_constants,
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
                enum_constants,
            )?;
            let rhs = evaluate_constant_expr(
                right,
                _expected_type,
                target,
                type_builder,
                diagnostics,
                name_table,
                enum_constants,
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
            enum_constants,
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
                enum_constants,
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
                            enum_constants,
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
                    enum_constants,
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
            enum_constants,
        ),
        // BuiltinCall in constant context (e.g. __builtin_constant_p,
        // __builtin_choose_expr) — evaluate known builtins.
        ast::Expression::BuiltinCall { builtin, args, .. } => match builtin {
            ast::BuiltinKind::ConstantP => Some(Constant::Integer(0)),
            ast::BuiltinKind::ChooseExpr if args.len() >= 3 => {
                let cond = evaluate_constant_expr(
                    &args[0],
                    _expected_type,
                    target,
                    type_builder,
                    diagnostics,
                    name_table,
                    enum_constants,
                );
                match cond {
                    Some(Constant::Integer(v)) if v != 0 => evaluate_constant_expr(
                        &args[1],
                        _expected_type,
                        target,
                        type_builder,
                        diagnostics,
                        name_table,
                        enum_constants,
                    ),
                    _ => evaluate_constant_expr(
                        &args[2],
                        _expected_type,
                        target,
                        type_builder,
                        diagnostics,
                        name_table,
                        enum_constants,
                    ),
                }
            }
            ast::BuiltinKind::TypesCompatibleP => Some(Constant::Integer(0)),
            ast::BuiltinKind::Expect if !args.is_empty() => evaluate_constant_expr(
                &args[0],
                _expected_type,
                target,
                type_builder,
                diagnostics,
                name_table,
                enum_constants,
            ),
            _ => Some(Constant::Undefined),
        },
        // Comma operator — evaluate both, return last.
        ast::Expression::Comma { exprs, .. } => {
            if let Some(last) = exprs.last() {
                evaluate_constant_expr(
                    last,
                    _expected_type,
                    target,
                    type_builder,
                    diagnostics,
                    name_table,
                    enum_constants,
                )
            } else {
                Some(Constant::Undefined)
            }
        }
        // Statement expression ({...}) — treat as non-constant but don't
        // error out if it produces a known value (e.g. kernel macros).
        ast::Expression::StatementExpression { .. } => {
            // Conservatively treat as undefined — not evaluable at
            // compile time in the general case.
            Some(Constant::Undefined)
        }
        // FunctionCall — not a compile-time constant, but don't error
        // (some kernel macros produce these in initializer context).
        ast::Expression::FunctionCall { .. } => Some(Constant::Undefined),
        // MemberAccess and PointerMemberAccess — not compile-time.
        ast::Expression::MemberAccess { .. } | ast::Expression::PointerMemberAccess { .. } => {
            Some(Constant::Undefined)
        }
        // AddressOfLabel (GCC &&label) — address constant.
        ast::Expression::AddressOfLabel { label, .. } => {
            let label_str = resolve_sym(name_table, label).to_string();
            Some(Constant::GlobalRef(label_str))
        }
        // ArraySubscript — not compile-time constant.
        ast::Expression::ArraySubscript { .. } => Some(Constant::Undefined),
        // Generic (_Generic) — not evaluated here.
        ast::Expression::Generic { .. } => Some(Constant::Undefined),
        // CompoundLiteral — may be constant in certain cases.
        ast::Expression::CompoundLiteral { .. } => Some(Constant::Undefined),
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
    enum_constants: &FxHashMap<String, i128>,
) -> Option<Constant> {
    let resolved_ref = crate::common::types::resolve_typedef(target_type);

    // Resolve forward-referenced (empty-field) named structs/unions using the
    // thread-local struct definitions registry populated during Pass 0.
    // When a variable is declared as `struct Foo x = {...};`, the AST stores
    // the struct type with the tag name but without field information (the
    // definition body is in a separate AST node).  We look up the full
    // definition so the initializer can be correctly lowered.
    let resolved_owned: Option<CType> = match resolved_ref {
        CType::Struct {
            name: Some(ref tag),
            ref fields,
            ..
        } if fields.is_empty() => super::SIZEOF_STRUCT_DEFS.with(|defs| {
            let borrow = defs.borrow();
            borrow.as_ref().and_then(|m| m.get(tag.as_str()).cloned())
        }),
        CType::Union {
            name: Some(ref tag),
            ref fields,
            ..
        } if fields.is_empty() => super::SIZEOF_STRUCT_DEFS.with(|defs| {
            let borrow = defs.borrow();
            borrow.as_ref().and_then(|m| m.get(tag.as_str()).cloned())
        }),
        _ => None,
    };
    let resolved = resolved_owned.as_ref().unwrap_or(resolved_ref);

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
                        enum_constants,
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
                        enum_constants,
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
                        enum_constants,
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
                    enum_constants,
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
        // C11 §6.7.6.3p7: Function parameter type adjustments —
        // "array of T" → "pointer to T", "function returning T" → "pointer to function returning T".
        // Without this, `char bar[]` would produce a 0/1-byte alloca instead
        // of an 8-byte pointer alloca, causing stack corruption.
        let param_c_type = match param_c_type {
            CType::Array(elem, _) => {
                CType::Pointer(elem, crate::common::types::TypeQualifiers::default())
            }
            CType::Function { .. } => CType::Pointer(
                Box::new(param_c_type),
                crate::common::types::TypeQualifiers::default(),
            ),
            other => other,
        };
        // Resolve forward-referenced struct/union tag references to their
        // full definitions so that the alloca gets the correct size.
        // Without this, `union YYMINORTYPE minor` (tag reference without
        // body) would produce a 0-byte alloca instead of the full union size.
        let param_c_type = resolve_sizeof_struct_ref(param_c_type, target);
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

/// Recursively collect all C label names from a statement tree.
///
/// Pre-scanning for labels enables forward references in `asm goto`
/// statements — the label block is created before the function body
/// is lowered, so `wire_asm_goto_targets` can always find the block.
fn collect_label_names(stmt: &ast::Statement, labels: &mut Vec<String>, name_table: &[String]) {
    match stmt {
        ast::Statement::Labeled {
            label, statement, ..
        } => {
            // Use the same naming convention as lower_label in stmt_lowering.rs:
            // format!("label_{}", label.as_u32())
            labels.push(format!("label_{}", label.as_u32()));
            let _ = name_table; // suppress unused
            collect_label_names(statement, labels, name_table);
        }
        ast::Statement::Compound(compound) => {
            for item in &compound.items {
                match item {
                    ast::BlockItem::Statement(s) => collect_label_names(s, labels, name_table),
                    ast::BlockItem::Declaration(_) => {}
                }
            }
        }
        ast::Statement::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_label_names(then_branch, labels, name_table);
            if let Some(ref e) = else_branch {
                collect_label_names(e, labels, name_table);
            }
        }
        ast::Statement::While { body, .. }
        | ast::Statement::DoWhile { body, .. }
        | ast::Statement::Switch { body, .. } => {
            collect_label_names(body, labels, name_table);
        }
        ast::Statement::For { body, .. } => {
            collect_label_names(body, labels, name_table);
        }
        ast::Statement::Case { statement, .. }
        | ast::Statement::Default { statement, .. }
        | ast::Statement::CaseRange { statement, .. } => {
            collect_label_names(statement, labels, name_table);
        }
        _ => {}
    }
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

/// Infer array size from an initializer for incomplete array declarations (`T arr[] = { ... }`).
///
/// Returns the number of elements in the initializer list, or the string
/// length (including null terminator) for string initializers.
fn infer_array_size_from_initializer(init: &ast::Initializer, elem_type: &CType) -> usize {
    match init {
        ast::Initializer::List {
            designators_and_initializers,
            ..
        } => {
            // Count the number of top-level initializer elements.
            // For designated initializers, we use the max index + 1.
            let mut max_index: usize = 0;
            for (i, di) in designators_and_initializers.iter().enumerate() {
                let effective_idx = if di.designators.is_empty() {
                    i
                } else {
                    // Check for array index designator [N].
                    let mut idx = i;
                    for d in &di.designators {
                        if let ast::Designator::Index(expr, _span) = d {
                            if let ast::Expression::IntegerLiteral { value, .. } = expr.as_ref() {
                                idx = *value as usize;
                            }
                        }
                    }
                    idx
                };
                if effective_idx >= max_index {
                    max_index = effective_idx + 1;
                }
            }
            max_index
        }
        ast::Initializer::Expression(expr_box) => {
            if let ast::Expression::StringLiteral { segments, .. } = expr_box.as_ref() {
                // String initializer for char arrays — size includes null terminator.
                let _ = elem_type; // suppress unused warning
                let total_bytes: usize = segments.iter().map(|s| s.value.len()).sum();
                total_bytes + 1 // null terminator
            } else {
                0
            }
        }
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

        let mut c_type =
            resolve_declaration_type(specifiers, declarator, &Target::X86_64, name_table);
        let alignment = extract_alignment_attribute(&specifiers.attributes, name_table);

        // Infer array size from initializer for incomplete array types (`T arr[] = { ... }`).
        if let CType::Array(ref elem, None) = c_type {
            if let Some(ref init) = init_decl.initializer {
                let inferred_size = infer_array_size_from_initializer(init, elem);
                if inferred_size > 0 {
                    c_type = CType::Array(elem.clone(), Some(inferred_size));
                }
            }
        }

        let static_init = if is_static {
            init_decl.initializer.clone()
        } else {
            None
        };

        // Register the variable's resolved CType in the typeof resolution
        // context so that subsequent `typeof(var_name)` can find it.
        super::TYPEOF_CONTEXT.with(|ctx| {
            let mut borrow = ctx.borrow_mut();
            if let Some(ref mut map) = *borrow {
                map.insert(var_name.clone(), c_type.clone());
            }
        });

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
    for (idx, local) in locals.iter().enumerate() {
        // Resolve forward-referenced struct/union tag references so the
        // alloca size matches the full struct/union definition.
        let resolved_type = resolve_sizeof_struct_ref(local.c_type.clone(), target);
        let ir_type = IrType::from_ctype(&resolved_type, target);
        let (alloca_val, alloca_inst) = builder.build_alloca(ir_type.clone(), local.span);
        push_inst_to_entry(function, alloca_inst);
        local_vars.insert(local.name.clone(), alloca_val);

        // Record debug metadata so the DWARF emitter can produce
        // DW_TAG_variable entries with proper names, types, and locations.
        let decl_line = if local.span.start > 0 {
            local.span.start
        } else {
            1
        };
        function
            .local_var_debug_info
            .push(crate::ir::function::LocalVarDebugInfo {
                name: local.name.clone(),
                ir_type,
                alloca_index: idx as u32,
                decl_line,
            });
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
    type_builder: &TypeBuilder,
    diagnostics: &mut DiagnosticEngine,
    enum_constants: &FxHashMap<String, i128>,
) {
    let mangled_name = format!("{}.{}", func_name, name);

    // Attempt to extract a compile-time constant from the initializer.
    // Delegate to evaluate_initializer_constant so that function pointers,
    // address-of expressions, designated initializers, and other non-trivial
    // constant expressions are handled correctly — matching how global
    // variable initializers are processed.
    let name_table_rc__ = super::INTERNER_SNAPSHOT.with(|snap| {
        snap.borrow()
            .as_ref()
            .cloned()
            .unwrap_or_else(|| std::rc::Rc::new(Vec::new()))
    });
    let name_table: &[String] = &name_table_rc__;
    let constant = if has_initializer {
        if let Some(init) = init_expr {
            evaluate_initializer_constant(
                init,
                c_type,
                target,
                type_builder,
                diagnostics,
                name_table,
                enum_constants,
            )
        } else {
            Some(Constant::ZeroInit)
        }
    } else {
        Some(Constant::ZeroInit)
    };

    // Infer array size from initializer when declaration uses [].
    let c_type = match (c_type, &constant) {
        (CType::Array(elem, None), Some(Constant::String(bytes))) => {
            CType::Array(elem.clone(), Some(bytes.len()))
        }
        (CType::Array(elem, None), Some(Constant::Array(elems))) => {
            CType::Array(elem.clone(), Some(elems.len()))
        }
        _ => c_type.clone(),
    };
    let ir_type = IrType::from_ctype(&c_type, target);

    // Post-process: convert Constant::String at Ptr-typed positions into
    // Constant::GlobalRef (same fixup applied to global variables).
    let constant = constant.map(|c| fixup_string_ptrs_in_constant(c, &ir_type, module));

    // Store the original C type so that lookup_var_type can retrieve the
    // accurate struct/union/typedef information instead of using the lossy
    // ir_type_to_approx_ctype reverse mapping (which turns IrType::Struct
    // into CType::Int).
    module
        .global_c_types
        .insert(mangled_name.clone(), c_type.clone());

    let mut global = GlobalVariable::new(mangled_name, ir_type, constant);
    global.linkage = Linkage::Internal;
    global.is_definition = true;
    module.add_global(global);
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
    // Clone the thread-local name table once for this call (fallback path).
    let name_table_rc__ = super::INTERNER_SNAPSHOT.with(|snap| {
        snap.borrow()
            .as_ref()
            .cloned()
            .unwrap_or_else(|| std::rc::Rc::new(Vec::new()))
    });
    let name_table: &[String] = &name_table_rc__;
    extract_struct_union_fields_fast(spec, name_table)
}

/// Performance-optimized struct/union field extraction.
/// Uses the provided `name_table` reference instead of cloning the interner snapshot.
pub fn extract_struct_union_fields_fast(
    spec: &ast::StructOrUnionSpecifier,
    name_table: &[String],
) -> Vec<StructField> {
    let mut fields = Vec::new();
    if let Some(ref members) = spec.members {
        for member in members {
            // Resolve base type from the member's specifier-qualifier list
            let member_base = resolve_base_type_from_sqlist_fast(&member.specifiers, name_table);
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
                        let name = extract_declarator_name(declarator, name_table);
                        // Apply pointer/array modifiers from declarator
                        let member_type =
                            apply_declarator_type(member_base.clone(), declarator, name_table);
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
    // Resolve the outermost lightweight tag reference (empty fields +
    // named) by looking up the full definition in struct_defs.  We limit
    // recursion to avoid the quadratic/exponential expansion that occurs
    // when every nested struct field is eagerly expanded.  Deeper fields
    // are resolved on-demand via sizeof_resolved and field-offset lookups.
    resolve_struct_forward_ref_inner(ctype, struct_defs, 0);
}

/// Shallow forward-reference resolution: resolve only the outermost
/// lightweight tag reference (empty fields, named struct/union) by
/// replacing it with the full definition from struct_defs.  Does NOT
/// recurse into the resolved definition's fields — nested forward
/// references are resolved lazily during sizeof/field-offset lookups
/// via `sizeof_ctype_resolved` and `lookup_member_offset`.
fn resolve_struct_forward_ref_inner(
    ctype: &mut CType,
    struct_defs: &FxHashMap<String, CType>,
    _depth: usize,
) {
    match ctype {
        CType::Struct {
            name: Some(ref tag),
            ref fields,
            ..
        } if fields.is_empty() => {
            if let Some(full_def) = struct_defs.get(tag.as_str()) {
                let has_fields = match full_def {
                    CType::Struct { fields: f, .. } | CType::Union { fields: f, .. } => {
                        !f.is_empty()
                    }
                    _ => true,
                };
                if has_fields {
                    *ctype = full_def.clone();
                    // Do NOT recurse into the resolved struct's fields.
                    // Nested lightweight references will be resolved lazily
                    // when field sizes or offsets are needed.
                }
            }
        }
        CType::Union {
            name: Some(ref tag),
            ref fields,
            ..
        } if fields.is_empty() => {
            if let Some(full_def) = struct_defs.get(tag.as_str()) {
                let has_fields = match full_def {
                    CType::Struct { fields: f, .. } | CType::Union { fields: f, .. } => {
                        !f.is_empty()
                    }
                    _ => true,
                };
                if has_fields {
                    *ctype = full_def.clone();
                }
            }
        }
        // For non-empty structs/unions, don't recurse into fields — the
        // definition already carries its field types and nested references
        // will be resolved lazily.
        CType::Struct { .. } | CType::Union { .. } => {}
        CType::Pointer(_, _) => {}
        // Unwrap containers that might hold a forward-referenced tag.
        CType::Array(inner, _) => {
            resolve_struct_forward_ref_inner(inner, struct_defs, 0);
        }
        CType::Typedef {
            ref mut underlying, ..
        } => {
            resolve_struct_forward_ref_inner(underlying, struct_defs, 0);
        }
        CType::Atomic(inner) | CType::Complex(inner) => {
            resolve_struct_forward_ref_inner(inner, struct_defs, 0);
        }
        _ => {}
    }
}

/// Resolve a base C type from a specifier-qualifier list (used for struct members).
pub fn resolve_base_type_from_sqlist(sqlist: &ast::SpecifierQualifierList) -> CType {
    // Fallback for callers without a name_table — clones the interner snapshot.
    let name_table_rc__ = super::INTERNER_SNAPSHOT.with(|snap| {
        snap.borrow()
            .as_ref()
            .cloned()
            .unwrap_or_else(|| std::rc::Rc::new(Vec::new()))
    });
    let name_table: &[String] = &name_table_rc__;
    resolve_base_type_from_sqlist_fast(sqlist, name_table)
}

/// Performance-optimized specifier-qualifier list resolution.
/// Uses the provided `name_table` reference to avoid O(n) cloning.
fn resolve_base_type_from_sqlist_fast(
    sqlist: &ast::SpecifierQualifierList,
    name_table: &[String],
) -> CType {
    let type_specs = &sqlist.type_specifiers;
    if type_specs.is_empty() {
        return CType::Int;
    }
    if type_specs.len() == 1 {
        return map_single_type_specifier_fast(&type_specs[0], name_table);
    }
    resolve_multi_word_type_fast(type_specs, name_table)
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

/// Fallback for callers without a name_table — clones the interner snapshot.
#[allow(dead_code)]
fn resolve_base_type(specifiers: &ast::DeclarationSpecifiers, _target: &Target) -> CType {
    let name_table_rc__ = super::INTERNER_SNAPSHOT.with(|snap| {
        snap.borrow()
            .as_ref()
            .cloned()
            .unwrap_or_else(|| std::rc::Rc::new(Vec::new()))
    });
    let name_table: &[String] = &name_table_rc__;
    resolve_base_type_fast(specifiers, name_table)
}

/// Performance-optimized base type resolution.
/// Uses the provided `name_table` reference to avoid O(n) cloning per call.
#[inline]
pub fn resolve_base_type_fast(
    specifiers: &ast::DeclarationSpecifiers,
    name_table: &[String],
) -> CType {
    let type_specs = &specifiers.type_specifiers;
    if type_specs.is_empty() {
        return CType::Int;
    }
    if type_specs.len() == 1 {
        return map_single_type_specifier_fast(&type_specs[0], name_table);
    }
    resolve_multi_word_type_fast(type_specs, name_table)
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

/// Delegate to the name_table-aware version using the thread-local snapshot.
/// NOTE: This clones the snapshot — callers in hot paths should prefer
/// `map_single_type_specifier_with_names` with an already-borrowed table.
#[allow(dead_code)]
fn map_single_type_specifier(spec: &ast::TypeSpecifier) -> CType {
    let name_table_rc__ = super::INTERNER_SNAPSHOT.with(|snap| {
        snap.borrow()
            .as_ref()
            .cloned()
            .unwrap_or_else(|| std::rc::Rc::new(Vec::new()))
    });
    let name_table: &[String] = &name_table_rc__;
    map_single_type_specifier_with_names(spec, name_table)
}

/// Performance-optimized base type resolution for a single type specifier.
/// Uses the provided `name_table` reference to avoid cloning the interner snapshot.
#[inline]
fn map_single_type_specifier_fast(spec: &ast::TypeSpecifier, name_table: &[String]) -> CType {
    map_single_type_specifier_with_names(spec, name_table)
}

// ---------------------------------------------------------------------------
// typeof resolution helpers
// ---------------------------------------------------------------------------

/// Resolve a `typeof(...)` argument to a concrete CType.
fn resolve_typeof_arg(arg: &ast::TypeofArg, name_table: &[String]) -> CType {
    match arg {
        ast::TypeofArg::TypeName(tn) => {
            let base = resolve_base_type_from_sqlist_fast(&tn.specifier_qualifiers, name_table);
            if let Some(ref abs) = tn.abstract_declarator {
                apply_abstract_declarator_to_type(base, abs)
            } else {
                base
            }
        }
        ast::TypeofArg::Expression(expr) => infer_typeof_expr_ctype(expr, name_table),
    }
}

/// Infer the C type of an expression for `typeof(expr)`.
///
/// This mirrors `SemanticAnalyzer::infer_typeof_expr_type` but operates in
/// the IR lowering context where the only type information available is the
/// thread-local `TYPEOF_CONTEXT` (variable name → CType map).
fn infer_typeof_expr_ctype(expr: &ast::Expression, name_table: &[String]) -> CType {
    match expr {
        // Identifier — look up in the typeof resolution context.
        ast::Expression::Identifier { name, .. } => {
            let var_name = resolve_sym(name_table, name).to_string();
            super::TYPEOF_CONTEXT.with(|ctx| {
                let borrow = ctx.borrow();
                borrow
                    .as_ref()
                    .and_then(|map| map.get(&var_name).cloned())
                    .unwrap_or(CType::Int)
            })
        }

        // Integer literals — type depends on suffix.
        ast::Expression::IntegerLiteral { suffix, .. } => match suffix {
            ast::IntegerSuffix::None => CType::Int,
            ast::IntegerSuffix::U => CType::UInt,
            ast::IntegerSuffix::L => CType::Long,
            ast::IntegerSuffix::UL => CType::ULong,
            ast::IntegerSuffix::LL => CType::LongLong,
            ast::IntegerSuffix::ULL => CType::ULongLong,
        },

        // Float literals.
        ast::Expression::FloatLiteral { .. } => CType::Double,

        // Char literals.
        ast::Expression::CharLiteral { .. } => CType::Int,

        // String literals → pointer to char.
        ast::Expression::StringLiteral { .. } => CType::Pointer(
            Box::new(CType::Char),
            crate::common::types::TypeQualifiers::default(),
        ),

        // Dereference: *ptr → pointee type.
        ast::Expression::UnaryOp {
            op: ast::UnaryOp::Deref,
            operand,
            ..
        } => {
            let inner = infer_typeof_expr_ctype(operand, name_table);
            match inner {
                CType::Pointer(pointee, _) => *pointee,
                CType::Array(elem, _) => *elem,
                _ => CType::Int,
            }
        }

        // Address-of: &x → pointer to x's type.
        ast::Expression::UnaryOp {
            op: ast::UnaryOp::AddressOf,
            operand,
            ..
        } => {
            let inner = infer_typeof_expr_ctype(operand, name_table);
            CType::Pointer(
                Box::new(inner),
                crate::common::types::TypeQualifiers::default(),
            )
        }

        // Cast expression: (type)expr → the cast target type.
        ast::Expression::Cast { type_name, .. } => {
            let base =
                resolve_base_type_from_sqlist_fast(&type_name.specifier_qualifiers, name_table);
            if let Some(ref abs) = type_name.abstract_declarator {
                apply_abstract_declarator_to_type(base, abs)
            } else {
                base
            }
        }

        // sizeof always yields size_t (unsigned long on 64-bit).
        ast::Expression::SizeofExpr { .. } | ast::Expression::SizeofType { .. } => CType::ULong,

        // Logical not always yields int.
        ast::Expression::UnaryOp {
            op: ast::UnaryOp::LogicalNot,
            ..
        } => CType::Int,

        // Other unary ops: preserve operand type.
        ast::Expression::UnaryOp { operand, .. } => infer_typeof_expr_ctype(operand, name_table),

        // Binary ops — simplistic: use left operand type (covers common
        // arithmetic cases; full usual-arithmetic-conversion is in sema).
        ast::Expression::Binary { left, op, .. } => {
            if matches!(
                op,
                ast::BinaryOp::Equal
                    | ast::BinaryOp::NotEqual
                    | ast::BinaryOp::Less
                    | ast::BinaryOp::LessEqual
                    | ast::BinaryOp::Greater
                    | ast::BinaryOp::GreaterEqual
                    | ast::BinaryOp::LogicalAnd
                    | ast::BinaryOp::LogicalOr
            ) {
                CType::Int
            } else {
                infer_typeof_expr_ctype(left, name_table)
            }
        }

        // Ternary: use the "then" branch type.
        ast::Expression::Conditional { then_expr, .. } => {
            if let Some(then_e) = then_expr {
                infer_typeof_expr_ctype(then_e, name_table)
            } else {
                CType::Int
            }
        }

        // Function call: for now return Int (full resolution would require
        // looking up the function return type).
        ast::Expression::FunctionCall { .. } => CType::Int,

        // Comma: type of the right operand.
        ast::Expression::Comma { exprs, .. } => {
            if let Some(last) = exprs.last() {
                infer_typeof_expr_ctype(last, name_table)
            } else {
                CType::Int
            }
        }

        // Parenthesized: transparent.
        ast::Expression::Parenthesized { inner, .. } => infer_typeof_expr_ctype(inner, name_table),

        // Default fallback.
        _ => CType::Int,
    }
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
        ast::TypeSpecifier::Int128 => CType::Int128,
        ast::TypeSpecifier::Float128 => CType::LongDouble,
        ast::TypeSpecifier::Float64 => CType::Double,
        ast::TypeSpecifier::Float32 => CType::Float,
        ast::TypeSpecifier::Float16 => CType::Float,
        ast::TypeSpecifier::Struct(s) => {
            let fields = extract_struct_union_fields_fast(s, name_table);
            CType::Struct {
                name: s.tag.as_ref().map(|t| sym_to_string(t, name_table)),
                fields,
                packed: false,
                aligned: None,
            }
        }
        ast::TypeSpecifier::Union(u) => {
            let fields = extract_struct_union_fields_fast(u, name_table);
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
        ast::TypeSpecifier::TypedefName(name) => {
            let td_name = sym_to_string(name, name_table);
            // Look up the resolved underlying type from the typedef map
            // populated during Pass 0.5 of lower_translation_unit.
            let underlying = super::TYPEDEF_MAP.with(|map| {
                let borrow = map.borrow();
                borrow
                    .as_ref()
                    .and_then(|m| m.get(&td_name).cloned())
                    .unwrap_or(CType::Int)
            });
            CType::Typedef {
                name: td_name,
                underlying: Box::new(underlying),
            }
        }
        ast::TypeSpecifier::Typeof(arg) => resolve_typeof_arg(arg, name_table),
        ast::TypeSpecifier::Atomic(_) => CType::Atomic(Box::new(CType::Int)),
        ast::TypeSpecifier::Complex => CType::Complex(Box::new(CType::Double)),
        ast::TypeSpecifier::AutoType => {
            // __auto_type — type will be inferred from initializer.
            // Return Int as a placeholder; the actual type gets resolved
            // during variable initialization lowering.
            CType::Int
        }
    }
}

#[allow(dead_code)]
fn resolve_multi_word_type(specs: &[ast::TypeSpecifier]) -> CType {
    let name_table_rc__ = super::INTERNER_SNAPSHOT.with(|snap| {
        snap.borrow()
            .as_ref()
            .cloned()
            .unwrap_or_else(|| std::rc::Rc::new(Vec::new()))
    });
    let name_table: &[String] = &name_table_rc__;
    resolve_multi_word_type_fast(specs, name_table)
}

/// Performance-optimized multi-word type resolution.
/// Avoids cloning the interner snapshot per call by using a borrowed name_table.
fn resolve_multi_word_type_fast(specs: &[ast::TypeSpecifier], name_table: &[String]) -> CType {
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
            other => return map_single_type_specifier_fast(other, name_table),
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
    _target: &Target,
    name_table: &[String],
) -> CType {
    let base = resolve_base_type_fast(specifiers, name_table);
    apply_declarator_type(base, declarator, name_table)
}

fn apply_declarator_type(
    base: CType,
    declarator: &ast::Declarator,
    name_table: &[String],
) -> CType {
    // C declarator semantics: pointer modifiers apply to the base type FIRST,
    // producing the "pointed base".  Direct-declarator modifiers (array, function)
    // then wrap the pointed base.
    //
    // `void *x[3]`  →  pointer(void) → array(pointer(void), 3)
    // `void  x[3]`  →  void          → array(void, 3)
    // `int  *f(int)` →  pointer(int)  → function(pointer(int), [int])
    //
    // For parenthesized declarators like `void (*x)[3]`, the pointer is
    // *inside* the parenthesized inner declarator and is handled recursively
    // by apply_direct_declarator's Parenthesized branch.
    let modified_base = if let Some(ref pointer) = declarator.pointer {
        apply_pointer_layers(base, pointer)
    } else {
        base
    };
    apply_direct_declarator(modified_base, &declarator.direct, name_table)
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
            let array_size = size.as_ref().and_then(|e| evaluate_const_int_expr(e));
            match inner_dd.as_ref() {
                // Parenthesized base: `void (*x)[3]`
                // The array wraps the base type: Array(void, 3)
                // Then the inner declarator (which has the pointer) wraps the
                // array type: Pointer(Array(void, 3))
                ast::DirectDeclarator::Parenthesized(inner_decl) => {
                    let array_type = CType::Array(Box::new(base), array_size);
                    apply_declarator_type(array_type, inner_decl, name_table)
                }
                // Non-parenthesized: e.g. `int x[3]` or `int x[3][4]`
                // Build Array(base, current_size) first, then recurse into
                // inner_dd so that multi-dimensional arrays nest correctly:
                //   `int a[3][4]` → Array(Array(int, 4), 3)
                // The outermost AST node's size (4, parsed last) corresponds
                // to the innermost C dimension, so we wrap base with it
                // first and let the recursion wrap with the outer dimension.
                _ => {
                    let arr_type = CType::Array(Box::new(base), array_size);
                    apply_direct_declarator(arr_type, inner_dd, name_table)
                }
            }
        }
        ast::DirectDeclarator::Function {
            base: inner_dd,
            params,
            is_variadic,
            ..
        } => {
            let param_types: Vec<CType> = params
                .iter()
                .map(|p| resolve_param_type(p, &Target::X86_64, name_table))
                .collect();
            // C declarator syntax is inside-out. For function pointers like
            // `int (*op)(int, int)`, the AST has:
            //   Function { base: Parenthesized(Declarator{pointer:*, direct:Ident("op")}), params }
            //
            // The pointer `*` wraps the FUNCTION type, NOT the return type.
            // So we must first construct the function type with `base` as its
            // return type, then apply the inner declarator's modifiers (pointer,
            // array, etc.) to the completed function type.
            match inner_dd.as_ref() {
                ast::DirectDeclarator::Parenthesized(inner_decl) => {
                    let func_type = CType::Function {
                        return_type: Box::new(base),
                        params: param_types,
                        variadic: *is_variadic,
                    };
                    apply_declarator_type(func_type, inner_decl, name_table)
                }
                _ => {
                    let return_type = apply_direct_declarator(base, inner_dd, name_table);
                    CType::Function {
                        return_type: Box::new(return_type),
                        params: param_types,
                        variadic: *is_variadic,
                    }
                }
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
            base,
            params,
            is_variadic,
            ..
        } => {
            // For function-returning-function-pointer patterns like:
            //   void (*memdbDlSym(real_params))(return_type_params)
            // The outer Function's params belong to the returned function pointer,
            // NOT the actual function parameters. The real params are in the inner
            // nested Function declarator. Detect and recurse.
            if let ast::DirectDeclarator::Parenthesized(inner_decl) = base.as_ref() {
                if dd_has_nested_function(&inner_decl.direct) {
                    return extract_function_params(inner_decl);
                }
            }
            (params.clone(), *is_variadic)
        }
        ast::DirectDeclarator::Parenthesized(inner) => extract_function_params(inner),
        _ => (Vec::new(), false),
    }
}

/// Check if a direct declarator contains a nested Function declarator.
fn dd_has_nested_function(dd: &ast::DirectDeclarator) -> bool {
    match dd {
        ast::DirectDeclarator::Function { .. } => true,
        ast::DirectDeclarator::Parenthesized(inner) => dd_has_nested_function(&inner.direct),
        ast::DirectDeclarator::Array { base, .. } => dd_has_nested_function(base),
        ast::DirectDeclarator::Identifier(..) => false,
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
    _target: &Target,
    name_table: &[String],
) -> CType {
    let base = resolve_base_type_fast(&param.specifiers, name_table);
    if let Some(ref declarator) = param.declarator {
        apply_declarator_type(base, declarator, name_table)
    } else {
        base
    }
}

pub fn extract_declarator_name(
    declarator: &ast::Declarator,
    name_table: &[String],
) -> Option<String> {
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
        ast::Expression::UnaryOp {
            op: ast::UnaryOp::BitwiseNot,
            operand,
            ..
        } => {
            let inner = evaluate_const_int_expr(operand)?;
            Some(!inner)
        }
        ast::Expression::UnaryOp {
            op: ast::UnaryOp::Plus,
            operand,
            ..
        } => evaluate_const_int_expr(operand),
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
                ast::BinaryOp::Equal => Some(if l == r { 1 } else { 0 }),
                ast::BinaryOp::NotEqual => Some(if l != r { 1 } else { 0 }),
                ast::BinaryOp::Less => Some(if (l as i64) < (r as i64) { 1 } else { 0 }),
                ast::BinaryOp::Greater => Some(if (l as i64) > (r as i64) { 1 } else { 0 }),
                ast::BinaryOp::LessEqual => Some(if (l as i64) <= (r as i64) { 1 } else { 0 }),
                ast::BinaryOp::GreaterEqual => Some(if (l as i64) >= (r as i64) { 1 } else { 0 }),
                _ => None,
            }
        }
        ast::Expression::Cast { operand, .. } => evaluate_const_int_expr(operand),
        ast::Expression::Parenthesized { inner, .. } => evaluate_const_int_expr(inner),
        ast::Expression::Conditional {
            condition,
            then_expr,
            else_expr,
            ..
        } => {
            let c = evaluate_const_int_expr(condition)?;
            if c != 0 {
                then_expr.as_ref().and_then(|e| evaluate_const_int_expr(e))
            } else {
                evaluate_const_int_expr(else_expr)
            }
        }
        // Handle sizeof(type) — critical for correct struct layout when
        // array dimensions use sizeof expressions (e.g. `int arr[sizeof(T)]`).
        ast::Expression::SizeofType { type_name, .. } => {
            // Access the target from the thread-local set during lowering.
            super::LOWERING_TARGET.with(|tl| {
                let target = tl.borrow();
                let target = target.as_ref()?;
                let cty = resolve_base_type_from_sqlist(&type_name.specifier_qualifiers);
                let cty = if let Some(ref abs) = type_name.abstract_declarator {
                    apply_abstract_declarator_to_type(cty, abs)
                } else {
                    cty
                };
                // Resolve forward-referenced struct types using the
                // thread-local struct_defs registry.
                let resolved = resolve_sizeof_struct_ref(cty, target);
                Some(crate::common::types::sizeof_ctype(&resolved, target))
            })
        }
        // Handle sizeof(expr) — infer the expression's type and compute size.
        ast::Expression::SizeofExpr { operand, .. } => {
            super::LOWERING_TARGET.with(|tl| {
                let target = tl.borrow();
                let target = target.as_ref()?;
                // For sizeof applied to an expression, attempt to determine
                // the expression's type from context.  Common patterns:
                //   sizeof(*ptr) — dereference → pointee type
                //   sizeof(literal) — integer literal → int
                match operand.as_ref() {
                    ast::Expression::IntegerLiteral { .. } => {
                        Some(crate::common::types::sizeof_ctype(&CType::Int, target))
                    }
                    ast::Expression::StringLiteral { segments, .. } => {
                        // sizeof("string") = total length of segments + 1 (null terminator)
                        let total: usize = segments.iter().map(|s| s.value.len()).sum();
                        Some(total + 1)
                    }
                    _ => {
                        // Default: pointer width for unresolvable expressions.
                        Some(target.pointer_width())
                    }
                }
            })
        }
        // Handle _Alignof(type) / __alignof__(type)
        ast::Expression::AlignofType { type_name, .. } => super::LOWERING_TARGET.with(|tl| {
            let target = tl.borrow();
            let target = target.as_ref()?;
            let cty = resolve_base_type_from_sqlist(&type_name.specifier_qualifiers);
            let cty = if let Some(ref abs) = type_name.abstract_declarator {
                apply_abstract_declarator_to_type(cty, abs)
            } else {
                cty
            };
            let resolved = resolve_sizeof_struct_ref(cty, target);
            Some(crate::common::types::alignof_ctype(&resolved, target))
        }),
        // Handle __builtin_offsetof(type, member)
        ast::Expression::BuiltinCall {
            builtin: ast::BuiltinKind::Offsetof,
            args,
            ..
        } => {
            // args[0] is SizeofType wrapping the type name, args[1] is the member expr
            if args.len() < 2 {
                return None;
            }
            let type_name = match &args[0] {
                ast::Expression::SizeofType { type_name, .. } => type_name,
                _ => return None,
            };
            super::LOWERING_TARGET.with(|tl| {
                let target = tl.borrow();
                let target = target.as_ref()?;
                let cty = resolve_base_type_from_sqlist(&type_name.specifier_qualifiers);
                let cty = if let Some(ref abs) = type_name.abstract_declarator {
                    apply_abstract_declarator_to_type(cty, abs)
                } else {
                    cty
                };
                let resolved = resolve_sizeof_struct_ref(cty, target);
                // Extract the chain of member designators from args[1]
                let chain = extract_offsetof_member_chain(&args[1]);
                compute_struct_field_offset(&resolved, &chain, target)
            })
        }
        _ => None,
    }
}

/// A step in an offsetof member-designator chain.
enum OffsetofMemberStep {
    /// `.field_name`
    Field(String),
    /// `[index]`
    Index(usize),
}

/// Extract the member designator chain from an offsetof member expression.
///
/// Walks the AST expression tree to build a flat list of field accesses and
/// array subscripts.  For example, `a.b[0].c` produces
/// `[Field("a"), Field("b"), Index(0), Field("c")]`.
fn extract_offsetof_member_chain(expr: &ast::Expression) -> Vec<OffsetofMemberStep> {
    let mut chain = Vec::new();
    extract_offsetof_chain_recursive(expr, &mut chain);
    chain
}

fn extract_offsetof_chain_recursive(expr: &ast::Expression, chain: &mut Vec<OffsetofMemberStep>) {
    let name_table_rc__ = super::INTERNER_SNAPSHOT.with(|snap| {
        snap.borrow()
            .as_ref()
            .cloned()
            .unwrap_or_else(|| std::rc::Rc::new(Vec::new()))
    });
    let name_table: &[String] = &name_table_rc__;
    match expr {
        ast::Expression::Identifier { name, .. } => {
            let field_name = if (name.as_u32() as usize) < name_table.len() {
                name_table[name.as_u32() as usize].clone()
            } else {
                format!("sym_{}", name.as_u32())
            };
            chain.push(OffsetofMemberStep::Field(field_name));
        }
        ast::Expression::MemberAccess { object, member, .. }
        | ast::Expression::PointerMemberAccess { object, member, .. } => {
            extract_offsetof_chain_recursive(object, chain);
            let member_name = if (member.as_u32() as usize) < name_table.len() {
                name_table[member.as_u32() as usize].clone()
            } else {
                format!("sym_{}", member.as_u32())
            };
            chain.push(OffsetofMemberStep::Field(member_name));
        }
        ast::Expression::ArraySubscript { base, index, .. } => {
            extract_offsetof_chain_recursive(base, chain);
            if let Some(idx) = evaluate_const_int_expr(index) {
                chain.push(OffsetofMemberStep::Index(idx));
            }
        }
        ast::Expression::Parenthesized { inner, .. } => {
            extract_offsetof_chain_recursive(inner, chain);
        }
        _ => {}
    }
}

/// Compute the byte offset of a member within a struct/union type,
/// following a chain of field accesses and array subscripts.
///
/// This mirrors the layout computation in `compute_struct_size` from
/// `src/common/types.rs`, including bitfield packing, alignment, and
/// anonymous struct/union member lookup.
fn compute_struct_field_offset(
    ctype: &CType,
    chain: &[OffsetofMemberStep],
    target: &Target,
) -> Option<usize> {
    if chain.is_empty() {
        return Some(0);
    }

    // Unwrap typedef/qualified wrappers
    let unwrapped = unwrap_ctype_for_offsetof(ctype);

    // Resolve forward-declared structs via thread-local struct_defs
    let resolved = resolve_sizeof_struct_ref(unwrapped.clone(), target);
    let unwrapped = unwrap_ctype_for_offsetof(&resolved);

    match &chain[0] {
        OffsetofMemberStep::Index(idx) => {
            if let CType::Array(elem, _) = unwrapped {
                let elem_sz = crate::common::types::sizeof_ctype(elem, target);
                let rest = compute_struct_field_offset(elem, &chain[1..], target)?;
                Some(idx * elem_sz + rest)
            } else {
                None
            }
        }
        OffsetofMemberStep::Field(target_name) => {
            let (fields, is_union, packed) = match unwrapped {
                CType::Struct { fields, packed, .. } => (fields, false, *packed),
                CType::Union { fields, packed, .. } => (fields, true, *packed),
                _ => return None,
            };

            let mut bit_offset: usize = 0;
            for field in fields {
                // Handle bitfield alignment
                if let Some(bits) = field.bit_width {
                    let bits = bits as usize;
                    let unit_bytes = crate::common::types::sizeof_ctype(&field.ty, target);
                    let unit_bits = unit_bytes * 8;
                    let fa = if packed {
                        1
                    } else {
                        crate::common::types::alignof_ctype(&field.ty, target)
                    };

                    if bits == 0 {
                        let align_bits = fa * 8;
                        if align_bits > 0 {
                            bit_offset = (bit_offset + align_bits - 1) / align_bits * align_bits;
                        }
                        continue;
                    }

                    let unit_align_bits = if packed { 8 } else { fa * 8 };
                    let unit_start = if unit_align_bits > 0 {
                        (bit_offset / unit_align_bits) * unit_align_bits
                    } else {
                        bit_offset
                    };
                    let bits_used = bit_offset - unit_start;
                    let start = if bits_used + bits <= unit_bits {
                        bit_offset
                    } else {
                        unit_start + unit_bits
                    };

                    let field_name = field.name.as_deref().unwrap_or("");
                    if field_name == target_name.as_str() {
                        return Some(start / 8);
                    }
                    if !is_union {
                        bit_offset = start + bits;
                    }
                    continue;
                }

                // Non-bitfield: round up to byte boundary
                if bit_offset % 8 != 0 {
                    bit_offset = (bit_offset + 7) / 8 * 8;
                }
                let byte_offset = bit_offset / 8;
                let fa = if packed {
                    1
                } else {
                    crate::common::types::alignof_ctype(&field.ty, target)
                };
                let aligned = if fa > 1 {
                    (byte_offset + fa - 1) & !(fa - 1)
                } else {
                    byte_offset
                };

                let field_name = field.name.as_deref().unwrap_or("");
                if field_name == target_name.as_str() {
                    if chain.len() == 1 {
                        return Some(aligned);
                    }
                    return compute_struct_field_offset(&field.ty, &chain[1..], target)
                        .map(|rest| aligned + rest);
                }

                // Check anonymous struct/union — recurse into it
                if field_name.is_empty() {
                    if let Some(inner_off) = compute_struct_field_offset(&field.ty, chain, target) {
                        return Some(aligned + inner_off);
                    }
                }

                if !is_union {
                    let field_sz = crate::common::types::sizeof_ctype(&field.ty, target);
                    bit_offset = aligned * 8 + field_sz * 8;
                }
            }
            None
        }
    }
}

/// Strip Typedef, Qualified, and Atomic wrappers to get the underlying type.
fn unwrap_ctype_for_offsetof(ctype: &CType) -> &CType {
    match ctype {
        CType::Typedef { underlying, .. } => unwrap_ctype_for_offsetof(underlying),
        CType::Qualified(inner, _) => unwrap_ctype_for_offsetof(inner),
        CType::Atomic(inner) => unwrap_ctype_for_offsetof(inner),
        other => other,
    }
}

/// Public wrapper for `evaluate_const_int_expr` so that other lowering
/// modules (e.g. `expr_lowering`) can evaluate non-literal array sizes.
pub fn evaluate_const_int_expr_pub(expr: &ast::Expression) -> Option<usize> {
    evaluate_const_int_expr(expr)
}

/// Evaluate `&array[index]` to a (symbol_name, byte_offset) pair.
///
/// Handles patterns like `&sqlite3UpperToLower[210]` which produce a
/// GlobalRefOffset relocation.  Recursively handles nested subscripts
/// for multi-dimensional arrays.
fn evaluate_address_of_subscript(
    base: &ast::Expression,
    index: &ast::Expression,
    target: &Target,
    name_table: &[String],
) -> Option<(String, i64)> {
    let idx_val = evaluate_const_int_expr(index)? as i64;

    match base {
        ast::Expression::Identifier { name, .. } => {
            let name_str = resolve_sym(name_table, name).to_string();
            // Determine element size.  For a bare identifier used as an
            // array base, look up its type from the TYPEOF_CONTEXT.
            let elem_size = super::TYPEOF_CONTEXT.with(|ctx| {
                let borrow = ctx.borrow();
                if let Some(ref map) = *borrow {
                    if let Some(CType::Array(elem, _)) = map.get(&name_str) {
                        return Some(crate::common::types::sizeof_ctype(elem, target) as i64);
                    }
                }
                // Default: try sizeof_ctype on the type itself.
                // For `const unsigned char arr[]`, element size is 1.
                None
            });
            let elem_sz = elem_size.unwrap_or(1);
            Some((name_str, idx_val * elem_sz))
        }
        ast::Expression::ArraySubscript {
            base: inner_base,
            index: inner_index,
            ..
        } => {
            // Nested subscript: &arr[i][j] — compute inner offset first.
            let (sym, inner_off) =
                evaluate_address_of_subscript(inner_base, inner_index, target, name_table)?;
            // For nested subscripts, element size is harder to determine.
            // Default to 1 for byte arrays.
            Some((sym, inner_off + idx_val))
        }
        _ => None,
    }
}

/// Evaluate address-of-member chains like `&((struct T *)0)->field` or
/// `&ptr->field`.  Returns `(symbol_name, byte_offset)` if the expression
/// is a compile-time address constant.
fn evaluate_address_of_member_chain(
    expr: &ast::Expression,
    _target: &Target,
    name_table: &[String],
) -> Option<(String, i64)> {
    // Handle &identifier (base case — already handled above, but defensive)
    if let ast::Expression::Identifier { name, .. } = expr {
        let name_str = resolve_sym(name_table, name).to_string();
        return Some((name_str, 0));
    }
    // For more complex expressions, bail out for now.
    None
}

/// Evaluate sizeof for a type expression (simplified heuristic).
fn evaluate_sizeof_expr(
    expr: &ast::Expression,
    target: &Target,
    type_builder: &TypeBuilder,
) -> u64 {
    match expr {
        ast::Expression::SizeofType { type_name, .. } => {
            let cty = resolve_base_type_from_sqlist(&type_name.specifier_qualifiers);
            let cty = if let Some(ref abs) = type_name.abstract_declarator {
                apply_abstract_declarator_to_type(cty, abs)
            } else {
                cty
            };
            // Resolve forward-referenced struct/union using thread-local struct_defs.
            let resolved = resolve_sizeof_struct_ref(cty, target);
            type_builder.sizeof_type(&resolved) as u64
        }
        ast::Expression::SizeofExpr { operand, .. } => {
            // For sizeof(expr), try to infer the expression type.
            // Common cases: sizeof(variable), sizeof(*ptr), sizeof(literal).
            match operand.as_ref() {
                ast::Expression::IntegerLiteral { .. } => {
                    // sizeof(integer_literal) — type is int
                    type_builder.sizeof_type(&CType::Int) as u64
                }
                _ => {
                    // Default to pointer width for unknown expression types.
                    target.pointer_width() as u64
                }
            }
        }
        _ => target.pointer_width() as u64,
    }
}

/// Apply abstract declarator (pointer/array/function) layers to a base type
/// for constant expression evaluation (global initializers).
fn apply_abstract_declarator_to_type(base: CType, abs: &ast::AbstractDeclarator) -> CType {
    let mut result = if let Some(ref pointer) = abs.pointer {
        let quals = crate::common::types::TypeQualifiers::default();
        let mut current = CType::Pointer(Box::new(base), quals);
        fn apply_ptr(current: CType, inner: &Option<Box<ast::Pointer>>) -> CType {
            if let Some(ref next) = inner {
                let quals = crate::common::types::TypeQualifiers::default();
                let wrapped = CType::Pointer(Box::new(current), quals);
                apply_ptr(wrapped, &next.inner)
            } else {
                current
            }
        }
        current = apply_ptr(current, &pointer.inner);
        current
    } else {
        base
    };

    if let Some(ref direct) = abs.direct {
        result = apply_direct_abstract_decl_to_type(result, direct);
    }
    result
}

fn apply_direct_abstract_decl_to_type(
    base: CType,
    direct: &ast::DirectAbstractDeclarator,
) -> CType {
    match direct {
        ast::DirectAbstractDeclarator::Parenthesized(inner) => {
            apply_abstract_declarator_to_type(base, inner)
        }
        ast::DirectAbstractDeclarator::Array { size, .. } => {
            let len = size.as_ref().and_then(|e| evaluate_const_int_expr(e));
            CType::Array(Box::new(base), len)
        }
        ast::DirectAbstractDeclarator::Function { .. } => CType::Pointer(
            Box::new(base),
            crate::common::types::TypeQualifiers::default(),
        ),
    }
}

/// Resolve a struct/union forward reference for sizeof evaluation.
/// Uses SIZEOF_STRUCT_DEFS thread-local when available, otherwise
/// falls back to the struct_defs from the lowering context.
/// Public wrapper for resolve_sizeof_struct_ref for use in stmt_lowering.
pub fn resolve_sizeof_struct_ref_pub(cty: CType, target: &Target) -> CType {
    resolve_sizeof_struct_ref(cty, target)
}

fn resolve_sizeof_struct_ref(mut cty: CType, _target: &Target) -> CType {
    // Use thread-local struct defs registry if available
    super::SIZEOF_STRUCT_DEFS.with(|cell| {
        let guard = cell.borrow();
        if let Some(ref map) = *guard {
            resolve_sizeof_forward_ref(&mut cty, map);
        }
    });
    cty
}

/// Recursively resolve forward-referenced struct/union types in a CType tree.
fn resolve_sizeof_forward_ref(ctype: &mut CType, struct_defs: &FxHashMap<String, CType>) {
    match ctype {
        CType::Struct {
            name: Some(ref tag),
            ref fields,
            ..
        } if fields.is_empty() => {
            let tag_owned = tag.clone();
            if let Some(full_def) = struct_defs.get(&tag_owned) {
                *ctype = full_def.clone();
            }
        }
        CType::Union {
            name: Some(ref tag),
            ref fields,
            ..
        } if fields.is_empty() => {
            let tag_owned = tag.clone();
            if let Some(full_def) = struct_defs.get(&tag_owned) {
                *ctype = full_def.clone();
            }
        }
        CType::Pointer(inner, _) => resolve_sizeof_forward_ref(inner, struct_defs),
        CType::Array(inner, _) => resolve_sizeof_forward_ref(inner, struct_defs),
        CType::Typedef { underlying, .. } => resolve_sizeof_forward_ref(underlying, struct_defs),
        CType::Qualified(inner, _) => resolve_sizeof_forward_ref(inner, struct_defs),
        CType::Atomic(inner) => resolve_sizeof_forward_ref(inner, struct_defs),
        // Also resolve struct fields that contain forward-referenced types.
        CType::Struct { ref mut fields, .. } if !fields.is_empty() => {
            for field in fields.iter_mut() {
                resolve_sizeof_forward_ref(&mut field.ty, struct_defs);
            }
        }
        CType::Union { ref mut fields, .. } if !fields.is_empty() => {
            for field in fields.iter_mut() {
                resolve_sizeof_forward_ref(&mut field.ty, struct_defs);
            }
        }
        _ => {}
    }
}

/// Evaluate alignof for a type expression.
fn evaluate_alignof_expr(
    expr: &ast::Expression,
    target: &Target,
    type_builder: &TypeBuilder,
) -> u64 {
    match expr {
        ast::Expression::AlignofType { type_name, .. } => {
            let cty = resolve_base_type_from_sqlist(&type_name.specifier_qualifiers);
            let cty = if let Some(ref abs) = type_name.abstract_declarator {
                apply_abstract_declarator_to_type(cty, abs)
            } else {
                cty
            };
            let resolved = resolve_sizeof_struct_ref(cty, target);
            type_builder.alignof_type(&resolved) as u64
        }
        _ => target.pointer_width() as u64,
    }
}

// ===========================================================================
// Designator resolution helpers
// ===========================================================================

/// Resolve array index from designator, or fall back to default index.
/// Recursively zero-initialize a field at the given byte offset.  For scalar
/// fields, emits a single store of 0.  For struct fields, recurses into each
/// sub-field so that the entire aggregate is zeroed.
/// Count the number of leaf scalar fields in an aggregate type for brace
/// elision detection.  Returns 1 for non-aggregate types.
#[allow(clippy::only_used_in_recursion)]
fn count_aggregate_scalar_fields(ty: &CType, tb: &TypeBuilder) -> usize {
    let resolved = crate::common::types::resolve_typedef(ty);
    match resolved {
        CType::Struct { fields, .. } => {
            let mut total = 0;
            for f in fields {
                total += count_aggregate_scalar_fields(&f.ty, tb);
            }
            if total == 0 {
                1
            } else {
                total
            }
        }
        CType::Array(elem, size_opt) => {
            let len = size_opt.unwrap_or(1);
            len * count_aggregate_scalar_fields(elem, tb)
        }
        _ => 1,
    }
}

/// Lower flat scalar initializers into the fields of an aggregate (struct),
/// consuming `init_list[cursor..]` and returning the updated cursor.
fn lower_brace_elision_into_aggregate(
    base_ptr: Value,
    init_list: &[ast::DesignatedInitializer],
    cursor: usize,
    target_type: &CType,
    ctx: &mut expr_lowering::ExprLoweringContext<'_>,
    span: Span,
) -> usize {
    let resolved = crate::common::types::resolve_typedef(target_type);
    match resolved {
        CType::Struct {
            fields,
            packed,
            aligned,
            ..
        } => {
            // Use the proper struct layout computation which handles
            // bitfield packing, zero-width bitfields, and alignment.
            let layout = ctx
                .type_builder
                .compute_struct_layout_with_fields(fields, *packed, *aligned);
            let mut cur = cursor;
            for (fi, field) in fields.iter().enumerate() {
                if cur >= init_list.len() {
                    break;
                }
                let fl = &layout.fields[fi];
                let byte_offset = fl.offset as i64;

                // Skip zero-width bitfields (they produce no storage).
                if let Some((_, 0)) = fl.bitfield_info {
                    continue;
                }

                let field_resolved = crate::common::types::resolve_typedef(&field.ty);

                if let Some((bit_offset, bit_width)) = fl.bitfield_info {
                    // ----- Bitfield field -----
                    // Must use read-modify-write pattern: load the storage
                    // unit, mask out the old bits, shift the new value into
                    // position, OR, store back.
                    let idx_val = make_index_value(ctx, byte_offset);
                    let (storage_ptr, gep_inst) =
                        ctx.builder
                            .build_gep(base_ptr, vec![idx_val], IrType::Ptr, span);
                    emit_inst_to_ctx(ctx, gep_inst);

                    // Lower the initializer expression to get the value.
                    let val = match &init_list[cur].initializer {
                        ast::Initializer::Expression(expr) => {
                            expr_lowering::lower_expression(ctx, expr)
                        }
                        ast::Initializer::List {
                            designators_and_initializers,
                            ..
                        } => {
                            // Scalar bitfield with braced initializer, e.g. {1}
                            if let Some(first) = designators_and_initializers.first() {
                                if let ast::Initializer::Expression(expr) = &first.initializer {
                                    expr_lowering::lower_expression(ctx, expr)
                                } else {
                                    expr_lowering::emit_int_const_for_zero(
                                        ctx,
                                        IrType::I32,
                                    )
                                }
                            } else {
                                expr_lowering::emit_int_const_for_zero(ctx, IrType::I32)
                            }
                        }
                    };

                    expr_lowering::lower_bitfield_store(
                        ctx,
                        storage_ptr,
                        val,
                        &field.ty,
                        bit_offset,
                        bit_width,
                        span,
                    );
                    cur += 1;
                } else if matches!(field_resolved, CType::Struct { .. } | CType::Array(..)) {
                    // Recurse into nested aggregate
                    let idx_val = make_index_value(ctx, byte_offset);
                    let (field_ptr, gep_inst) =
                        ctx.builder
                            .build_gep(base_ptr, vec![idx_val], IrType::Ptr, span);
                    emit_inst_to_ctx(ctx, gep_inst);
                    cur = lower_brace_elision_into_aggregate(
                        field_ptr,
                        init_list,
                        cur,
                        field_resolved,
                        ctx,
                        span,
                    );
                } else {
                    // Scalar field — consume one initializer
                    let idx_val = make_index_value(ctx, byte_offset);
                    let (field_ptr, gep_inst) =
                        ctx.builder
                            .build_gep(base_ptr, vec![idx_val], IrType::Ptr, span);
                    emit_inst_to_ctx(ctx, gep_inst);
                    lower_single_init_element(
                        field_ptr,
                        &init_list[cur].initializer,
                        &field.ty,
                        ctx,
                    );
                    cur += 1;
                }
            }
            cur
        }
        CType::Array(elem, size_opt) => {
            let len = size_opt.unwrap_or(1);
            let elem_size = ctx.type_builder.sizeof_type(elem);
            let mut cur = cursor;
            for i in 0..len {
                if cur >= init_list.len() {
                    break;
                }
                let byte_offset = (i * elem_size) as i64;
                let elem_resolved = crate::common::types::resolve_typedef(elem);
                let idx_val = make_index_value(ctx, byte_offset);
                let (elem_ptr, gep_inst) =
                    ctx.builder
                        .build_gep(base_ptr, vec![idx_val], IrType::Ptr, span);
                emit_inst_to_ctx(ctx, gep_inst);
                if matches!(elem_resolved, CType::Struct { .. } | CType::Array(..)) {
                    cur = lower_brace_elision_into_aggregate(
                        elem_ptr,
                        init_list,
                        cur,
                        elem_resolved,
                        ctx,
                        span,
                    );
                } else {
                    lower_single_init_element(elem_ptr, &init_list[cur].initializer, elem, ctx);
                    cur += 1;
                }
            }
            cur
        }
        _ => {
            // Scalar — consume one initializer
            if cursor < init_list.len() {
                lower_single_init_element(
                    base_ptr,
                    &init_list[cursor].initializer,
                    target_type,
                    ctx,
                );
                cursor + 1
            } else {
                cursor
            }
        }
    }
}

fn zero_init_field(
    base_alloca: Value,
    byte_offset: i64,
    field_ty: &CType,
    ctx: &mut expr_lowering::ExprLoweringContext<'_>,
    span: Span,
) {
    let resolved = crate::common::types::resolve_typedef(field_ty);
    match resolved {
        CType::Struct { ref fields, .. } => {
            let sub_offsets = compute_struct_field_offsets(fields, ctx.type_builder);
            for (i, sub_field) in fields.iter().enumerate() {
                let sub_off = byte_offset + sub_offsets.get(i).copied().unwrap_or(0) as i64;
                zero_init_field(base_alloca, sub_off, &sub_field.ty, ctx, span);
            }
        }
        CType::Union { ref fields, .. } => {
            // Zero the first (largest) member of the union.
            if let Some(first) = fields.first() {
                zero_init_field(base_alloca, byte_offset, &first.ty, ctx, span);
            }
        }
        CType::Array(ref elem, Some(len)) => {
            let elem_size = ctx.type_builder.sizeof_type(elem);
            for i in 0..*len {
                let arr_off = byte_offset + (i * elem_size) as i64;
                zero_init_field(base_alloca, arr_off, elem, ctx, span);
            }
        }
        _ => {
            // Scalar: emit GEP + store 0 with the correct width for the
            // field type so that we don't clobber adjacent struct fields.
            let idx_val = make_index_value(ctx, byte_offset);
            let (ptr, gep_inst) =
                ctx.builder
                    .build_gep(base_alloca, vec![idx_val], IrType::Ptr, span);
            emit_inst_to_ctx(ctx, gep_inst);
            // Create a zero constant matching the field's IR type width.
            let ir_ty = expr_lowering::ctype_to_ir(resolved, ctx.target);
            let zero_val = expr_lowering::emit_int_const_for_zero(ctx, ir_ty);
            let store_inst = ctx.builder.build_store(zero_val, ptr, span);
            emit_inst_to_ctx(ctx, store_inst);
        }
    }
}

/// Lower a single designated struct initializer, handling nested designators
/// like `.origin.x = 1` by recursively navigating field types.
fn lower_designated_struct_init(
    base_alloca: Value,
    designators: &[ast::Designator],
    initializer: &ast::Initializer,
    fields: &[crate::common::types::StructField],
    field_offsets: &[usize],
    field_layouts: &[crate::common::type_builder::FieldLayout],
    accumulated_offset: i64,
    default_idx: usize,
    ctx: &mut expr_lowering::ExprLoweringContext<'_>,
    span: Span,
) {
    // Resolve the first designator to a field index.
    let field_idx = resolve_field_designator(designators, fields, default_idx, ctx.name_table);
    if field_idx >= fields.len() {
        return;
    }
    let field = &fields[field_idx];
    let byte_offset =
        accumulated_offset + field_offsets.get(field_idx).copied().unwrap_or(0) as i64;

    // If there are remaining designators (nested designation), recurse into
    // the sub-struct.
    if designators.len() > 1 {
        let remaining_designators = &designators[1..];
        let resolved_field_ty = crate::common::types::resolve_typedef(&field.ty);
        if let CType::Struct {
            fields: ref sub_fields,
            packed: sub_packed,
            aligned: sub_aligned,
            ..
        } = resolved_field_ty
        {
            let sub_layout = ctx.type_builder.compute_struct_layout_with_fields(
                sub_fields,
                *sub_packed,
                *sub_aligned,
            );
            let inner_offsets: Vec<usize> =
                sub_layout.fields.iter().map(|fl| fl.offset).collect();
            lower_designated_struct_init(
                base_alloca,
                remaining_designators,
                initializer,
                sub_fields,
                &inner_offsets,
                &sub_layout.fields,
                byte_offset,
                0,
                ctx,
                span,
            );
            return;
        }
        // If the field type is not a struct but there are more designators,
        // fall through to treat as a simple store at the current offset.
    }

    // Check if this field is a bitfield — if so, use read-modify-write.
    let bf_info = field_layouts
        .get(field_idx)
        .and_then(|fl| fl.bitfield_info);
    if let Some((bit_offset, bit_width)) = bf_info {
        if bit_width > 0 {
            // Bitfield field — GEP to the storage unit offset and use
            // lower_bitfield_store for correct read-modify-write masking.
            let field_idx_val = make_index_value(ctx, byte_offset);
            let (storage_ptr, gep_inst) =
                ctx.builder
                    .build_gep(base_alloca, vec![field_idx_val], IrType::Ptr, span);
            emit_inst_to_ctx(ctx, gep_inst);

            // Lower the initializer to a value.
            let val = match initializer {
                ast::Initializer::Expression(expr) => {
                    expr_lowering::lower_expression(ctx, expr)
                }
                ast::Initializer::List {
                    designators_and_initializers,
                    ..
                } => {
                    if let Some(first) = designators_and_initializers.first() {
                        if let ast::Initializer::Expression(expr) = &first.initializer {
                            expr_lowering::lower_expression(ctx, expr)
                        } else {
                            expr_lowering::emit_int_const_for_zero(ctx, IrType::I32)
                        }
                    } else {
                        expr_lowering::emit_int_const_for_zero(ctx, IrType::I32)
                    }
                }
            };

            expr_lowering::lower_bitfield_store(
                ctx,
                storage_ptr,
                val,
                &field.ty,
                bit_offset,
                bit_width,
                span,
            );
            return;
        }
    }

    // Leaf case: GEP to the byte offset and lower the initializer.
    let field_idx_val = make_index_value(ctx, byte_offset);
    let (field_ptr, gep_inst) =
        ctx.builder
            .build_gep(base_alloca, vec![field_idx_val], IrType::Ptr, span);
    emit_inst_to_ctx(ctx, gep_inst);
    lower_single_init_element(field_ptr, initializer, &field.ty, ctx);
}

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

// =========================================================================
// String literal → char array initialization helpers
// =========================================================================

/// Check if a resolved type is a character array and return (element_type, array_size).
/// Matches `char`, `signed char`, `unsigned char` arrays.
fn is_char_array_type(ty: &CType) -> Option<(CType, usize)> {
    let resolved = crate::common::types::resolve_typedef(ty);
    match resolved {
        CType::Array(elem, Some(size)) => {
            let stripped = crate::common::types::resolve_typedef(elem.as_ref());
            match stripped {
                CType::Char | CType::SChar | CType::UChar => {
                    Some((stripped.clone(), *size))
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// Extract raw bytes from a StringLiteral expression (including null terminator).
fn extract_string_literal_bytes(expr: &ast::Expression) -> Option<Vec<u8>> {
    match expr {
        ast::Expression::StringLiteral { segments, .. } => {
            let mut bytes = Vec::new();
            for seg in segments {
                bytes.extend_from_slice(&seg.value);
            }
            bytes.push(0); // null terminator
            Some(bytes)
        }
        _ => None,
    }
}

/// Emit stores to copy string literal bytes into a local char array variable.
/// Handles zero-filling of remaining array elements beyond the string length
/// and truncation when the string is longer than the array.
fn lower_char_array_from_string(
    var_alloca: Value,
    str_bytes: &[u8],
    arr_size: usize,
    _elem_ty: &CType,
    ctx: &mut expr_lowering::ExprLoweringContext<'_>,
) {
    let span = Span::dummy();

    // Number of bytes to copy from the string (may be truncated to fit)
    let copy_len = std::cmp::min(str_bytes.len(), arr_size);

    // Build a full byte buffer: string bytes + zero-fill
    let mut buf = vec![0u8; arr_size];
    buf[..copy_len].copy_from_slice(&str_bytes[..copy_len]);

    // Emit stores coalesced into the widest possible chunks (8/4/2/1 bytes)
    // for efficiency.  We process the array from left to right.
    let mut offset: usize = 0;
    while offset < arr_size {
        let remaining = arr_size - offset;

        // Determine the largest power-of-two chunk we can store at this
        // alignment.  We only coalesce if offset is naturally aligned for
        // the chosen width.
        let (chunk_sz, ir_ty): (usize, IrType) = if remaining >= 8 && offset % 8 == 0 {
            (8, IrType::I64)
        } else if remaining >= 4 && offset % 4 == 0 {
            (4, IrType::I32)
        } else if remaining >= 2 && offset % 2 == 0 {
            (2, IrType::I16)
        } else {
            (1, IrType::I8)
        };

        // Read the chunk value from our buffer (little-endian).
        let chunk_val: i128 = {
            let mut v: u64 = 0;
            for i in 0..chunk_sz {
                v |= (buf[offset + i] as u64) << (i * 8);
            }
            v as i128
        };

        // Create the constant value.
        let const_val = expr_lowering::emit_int_const_for_index(ctx, chunk_val as i64);

        // Compute the destination address: var_alloca + offset
        let dst_addr = if offset == 0 {
            var_alloca
        } else {
            let off_val = expr_lowering::emit_int_const_for_index(ctx, offset as i64);
            let (gep_val, gep_inst) =
                ctx.builder
                    .build_gep(var_alloca, vec![off_val], ir_ty.clone(), span);
            emit_inst_to_ctx(ctx, gep_inst);
            gep_val
        };

        // Emit the store instruction.
        let si = ctx.builder.build_store(const_val, dst_addr, span);
        emit_inst_to_ctx(ctx, si);

        offset += chunk_sz;
    }
}

/// Create a Value representing an integer constant for GEP indices.
/// Uses the builder to allocate a fresh value identifier.
/// Create an IR Value containing the given byte offset for use as a GEP index.
///
/// Previous bug: this function called `fresh_value()` without registering
/// the constant, producing a dangling Value that the backend defaulted to
/// 0, causing ALL initializer stores to target the same base address.
fn make_index_value(ctx: &mut expr_lowering::ExprLoweringContext<'_>, byte_offset: i64) -> Value {
    expr_lowering::emit_int_const_for_index(ctx, byte_offset)
}

/// Compute byte offsets for each field in a struct.
fn compute_struct_field_offsets(
    fields: &[crate::common::types::StructField],
    type_builder: &crate::common::type_builder::TypeBuilder,
) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(fields.len());
    let mut current_offset: usize = 0;
    for field in fields {
        let field_align = type_builder.alignof_type(&field.ty);
        let align = if field_align == 0 { 1 } else { field_align };
        // Align current offset to field alignment.
        current_offset = (current_offset + align - 1) & !(align - 1);
        offsets.push(current_offset);
        current_offset += type_builder.sizeof_type(&field.ty);
    }
    offsets
}
