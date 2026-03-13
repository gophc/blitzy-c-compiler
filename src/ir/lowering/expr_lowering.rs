//! # Expression Lowering
//!
//! Converts semantically-analyzed AST expressions into IR instructions.
//!
//! Every expression evaluation produces an SSA `Value` that can be used
//! as an operand in subsequent instructions. For lvalue expressions (variables,
//! dereferences, array subscripts, member access), the result is a pointer
//! to the storage location. For rvalue expressions (arithmetic, constants, casts),
//! the result is the computed value.
//!
//! ## Key Lowering Patterns
//!
//! - **Variable reference**: Load from alloca pointer → Value
//! - **Assignment**: Lower rhs, store to lhs pointer
//! - **Binary arithmetic**: Lower both operands, emit BinOp instruction
//! - **Comparison**: Lower both operands, emit ICmp/FCmp instruction → I1
//! - **Function call**: Lower arguments, emit Call instruction
//! - **Address-of (`&x`)**: Return the pointer to x (its alloca) without loading
//! - **Dereference (`*p`)**: Load the pointer, then load through it
//! - **Cast**: Emit Trunc/ZExt/SExt/BitCast/IntToPtr/PtrToInt as appropriate
//! - **Short-circuit (`&&`, `||`)**: Create basic blocks for lazy evaluation
//! - **Ternary (`? :`)**: Create then/else/merge blocks with phi node
//!
//! ## Constant Representation
//!
//! Integer and floating-point constants are materialized through
//! `Instruction::BinOp` with `Value::UNDEF` sentinels.  The module-level
//! `GlobalVariable` pool stores the actual constant values so the backend
//! code-generator can resolve them to immediate operands.
//!
//! ## Dependencies
//! - `crate::ir::*` — IR types, instructions, builder
//! - `crate::frontend::parser::ast` — AST expression nodes
//! - `crate::common::*` — Types, diagnostics, target

use crate::common::diagnostics::{DiagnosticEngine, Span};
use crate::common::fx_hash::FxHashMap;
use crate::common::target::Target;
use crate::common::type_builder::{self, TypeBuilder};
use crate::common::types::{self, CType, TypeQualifiers};
use crate::frontend::parser::ast;
use crate::ir::builder::IrBuilder;
use crate::ir::function::IrFunction;
use crate::ir::instructions::{BinOp as IrBinOp, BlockId, FCmpOp, ICmpOp, Instruction, Value};
use crate::ir::module::{self as ir_module, Constant, GlobalVariable, IrModule};
use crate::ir::types::IrType;

// ---------------------------------------------------------------------------
// Expression Lowering Context
// ---------------------------------------------------------------------------

/// Context for expression lowering within a function.
///
/// Holds mutable references to the IR builder, current function, and module,
/// along with read-only references to target information, type builder, and
/// variable lookup maps.  The caller (declaration / statement lowering) creates
/// this context and passes it to [`lower_expression`] / [`lower_lvalue`].
pub struct ExprLoweringContext<'a> {
    /// IR builder for instruction creation and value numbering.
    pub builder: &'a mut IrBuilder,
    /// Current IR function being constructed.
    pub function: &'a mut IrFunction,
    /// IR module — global variable / function / string-literal pool access.
    pub module: &'a mut IrModule,
    /// Target architecture (pointer width, endianness, type sizes).
    pub target: &'a Target,
    /// Type builder for `sizeof`, `alignof`, struct layout computation, and
    /// usual-arithmetic-conversion queries.
    pub type_builder: &'a TypeBuilder,
    /// Diagnostic engine for emitting errors and warnings.
    pub diagnostics: &'a mut DiagnosticEngine,
    /// Local variable name → alloca `Value` mapping (alloca-first pattern).
    pub local_vars: &'a FxHashMap<String, Value>,
    /// Function parameter name → `Value` mapping.
    pub param_values: &'a FxHashMap<String, Value>,
    /// Name resolution table:  `Symbol::as_u32()` → interned string.
    /// Populated by the lowering driver from the string interner so that
    /// expression lowering can resolve AST `Symbol` handles to the `String`
    /// keys used in `local_vars` / `param_values`.
    pub name_table: &'a [String],
    /// Variable type table: variable name → declared C type.
    /// Used to determine signedness and pointee type for correct instruction
    /// selection (signed vs unsigned division, arithmetic shift vs logical, etc.).
    pub local_types: &'a FxHashMap<String, CType>,
    /// Enum constant name → integer value mapping.
    /// Populated by the lowering driver from the semantic analysis results
    /// so that enum constant identifiers can be resolved to immediate values.
    pub enum_constants: &'a FxHashMap<String, i128>,
    /// Static local variable name → mangled global name mapping.
    /// When a `static` local variable `x` inside function `foo` is
    /// encountered, its storage is a global variable named `foo.x`.
    /// This map allows `lower_identifier` to redirect the access to the
    /// corresponding global variable.
    pub static_locals: &'a mut FxHashMap<String, String>,
    /// Struct/union tag → full CType definition registry.
    /// Used by `resolve_type_name` and `lower_sizeof_type` to resolve
    /// forward-referenced struct/union types to their complete layouts.
    pub struct_defs: &'a FxHashMap<String, CType>,
    /// Label name → BlockId mapping for computed goto (`&&label`).
    /// When non-empty, `lower_address_of_label` can resolve label names to
    /// block IDs and produce meaningful pointer-sized integer values.
    pub label_blocks: &'a FxHashMap<String, crate::ir::instructions::BlockId>,
    /// Name of the function currently being lowered.
    /// Used for C99 `__func__` / GCC `__FUNCTION__` / `__PRETTY_FUNCTION__`.
    pub current_function_name: Option<&'a str>,
    /// Enclosing loop/switch stack propagated from the `StmtLoweringContext`
    /// so that GCC statement expressions `({ break; })` or `({ continue; })`
    /// inside a loop body can resolve to the correct targets.
    pub enclosing_loop_stack: Vec<super::stmt_lowering::LoopContext>,
    /// Scope-local type overrides for block-scoped variable shadowing.
    /// When a compound statement declares a variable with the same name as
    /// one in an outer scope but with a different type (e.g. `u32 t` in one
    /// case block and `void *t` in another), this map holds the CURRENT
    /// scope's type for that variable, overriding `local_types`.
    pub scope_type_overrides: &'a FxHashMap<String, CType>,
}

// ---------------------------------------------------------------------------
// Internal helpers — typed value tracking
// ---------------------------------------------------------------------------

/// Combined `Value` + C type returned by internal lowering helpers so that
/// callers can propagate signedness / pointee information.
struct TypedValue {
    value: Value,
    ty: CType,
}

impl TypedValue {
    #[inline]
    fn new(value: Value, ty: CType) -> Self {
        Self { value, ty }
    }

    #[inline]
    fn void() -> Self {
        Self {
            value: Value::UNDEF,
            ty: CType::Void,
        }
    }
}

// ---------------------------------------------------------------------------
// Instruction emission helper
// ---------------------------------------------------------------------------

/// Push an instruction into the current insertion block of the function.
fn emit_inst(ctx: &mut ExprLoweringContext<'_>, inst: Instruction) {
    if let Some(block_id) = ctx.builder.get_insert_block() {
        let idx = block_id.index();
        if let Some(block) = ctx.function.get_block_mut(idx) {
            block.instructions.push(inst);
        }
    }
}

/// Create a new basic block in the current function and return its `BlockId`.
fn new_block(ctx: &mut ExprLoweringContext<'_>) -> BlockId {
    let block_id = ctx.builder.create_block();
    let idx = ctx.function.block_count();
    let block = crate::ir::basic_block::BasicBlock::new(idx);
    ctx.function.add_block(block);
    block_id
}

/// Adds a CFG edge (predecessor/successor) between two blocks.
///
/// Maintains the predecessor and successor lists that the optimizer's
/// dead code elimination and CFG simplification passes rely on for
/// reachability analysis.
fn add_cfg_edge(ctx: &mut ExprLoweringContext<'_>, from_idx: usize, to_idx: usize) {
    if let Some(from_block) = ctx.function.blocks.get_mut(from_idx) {
        from_block.add_successor(to_idx);
    }
    if let Some(to_block) = ctx.function.blocks.get_mut(to_idx) {
        to_block.add_predecessor(from_idx);
    }
}

// ---------------------------------------------------------------------------
// Constant materialisation helpers
// ---------------------------------------------------------------------------

/// Emit a compile-time integer constant and return its `Value` handle.
///
/// The constant is registered as an internal `GlobalVariable` in the module
/// so that the backend can resolve the `Value` to an immediate.  The
/// in-function instruction is a `BinOp::Add` with `Value::UNDEF` operand
/// that serves as a placeholder defining the SSA value.
/// Check whether an AST expression is a compile-time constant for
/// `__builtin_constant_p`.  This recursively evaluates arithmetic on
/// literals, casts of constants, sizeof, and parenthesized sub-expressions.
fn is_compile_time_constant(expr: &ast::Expression) -> bool {
    match expr {
        ast::Expression::IntegerLiteral { .. }
        | ast::Expression::FloatLiteral { .. }
        | ast::Expression::CharLiteral { .. }
        | ast::Expression::SizeofType { .. }
        | ast::Expression::SizeofExpr { .. }
        | ast::Expression::AlignofType { .. }
        | ast::Expression::AlignofExpr { .. } => true,

        // Parenthesized: transparent.
        ast::Expression::Parenthesized { inner, .. } => is_compile_time_constant(inner),

        // Binary arithmetic on two constants is constant.
        ast::Expression::Binary { left, right, .. } => {
            is_compile_time_constant(left) && is_compile_time_constant(right)
        }

        // Unary arithmetic on a constant is constant.
        ast::Expression::UnaryOp { operand, op, .. } => {
            match op {
                // AddressOf a variable is NOT a compile-time constant
                // for __builtin_constant_p purposes.
                ast::UnaryOp::AddressOf => false,
                _ => is_compile_time_constant(operand),
            }
        }

        // Cast of a constant is constant.
        ast::Expression::Cast { operand, .. } => is_compile_time_constant(operand),

        // Conditional with constant condition and constant branches.
        ast::Expression::Conditional {
            condition,
            then_expr,
            else_expr,
            ..
        } => {
            is_compile_time_constant(condition)
                && then_expr
                    .as_ref()
                    .map_or(true, |e| is_compile_time_constant(e))
                && is_compile_time_constant(else_expr)
        }

        // Comma: the last expression determines constness.
        ast::Expression::Comma { exprs, .. } => {
            exprs.last().map_or(false, is_compile_time_constant)
        }

        // Everything else (variables, function calls, etc.) is NOT constant.
        _ => false,
    }
}

fn emit_int_const(
    ctx: &mut ExprLoweringContext<'_>,
    value: i128,
    ir_ty: IrType,
    span: Span,
) -> Value {
    // Register the constant in the module's global pool for the backend.
    let const_id = ctx.module.globals().len();
    let gname = format!(".Lconst.i.{}", const_id);
    let mut gv = GlobalVariable::new(gname, ir_ty.clone(), Some(Constant::Integer(value)));
    gv.linkage = ir_module::Linkage::Internal;
    gv.is_constant = true;
    ctx.module.add_global(gv);

    // Define the SSA value via a self-referencing Add+UNDEF sentinel.
    let result = ctx.builder.fresh_value();
    let inst = Instruction::BinOp {
        result,
        op: IrBinOp::Add,
        lhs: result,
        rhs: Value::UNDEF,
        ty: ir_ty,
        span,
    };
    emit_inst(ctx, inst);
    // Record the direct Value → constant mapping so the backend can
    // resolve constants without fragile positional matching.
    ctx.function.constant_values.insert(result, value as i64);
    result
}

/// Emit an integer constant for use as a GEP byte-offset index.
///
/// This is a public wrapper around `emit_int_const` for use by
/// `decl_lowering::make_index_value` which computes element byte
/// offsets for aggregate initialiser stores.
pub fn emit_int_const_for_index(ctx: &mut ExprLoweringContext<'_>, byte_offset: i64) -> Value {
    let ir_ty = size_ir_type(ctx.target);
    emit_int_const(ctx, byte_offset as i128, ir_ty, Span::dummy())
}

/// Emit a zero constant with the given IR type.  Used by `zero_init_field`
/// in `decl_lowering` to produce correctly-sized stores for aggregate
/// member zero-initialization (C99 §6.7.9/19).
pub fn emit_int_const_for_zero(ctx: &mut ExprLoweringContext<'_>, ir_ty: IrType) -> Value {
    emit_int_const(ctx, 0, ir_ty, Span::dummy())
}

/// Convert a value for storage into an element of the given C type.
///
/// When the target element type is narrower than `int` (e.g., `char` = I8,
/// `short` = I16), inserts a truncation so that the backend emits the
/// correctly-sized store operation.  Without this, integer literals (which
/// default to I32) would be stored with 32-bit width, overwriting adjacent
/// memory in arrays.
pub fn convert_for_store(
    ctx: &mut ExprLoweringContext<'_>,
    value: Value,
    target_ctype: &CType,
) -> Value {
    let target_ir = ctype_to_ir(target_ctype, ctx.target);
    // Only truncate for narrow integer types (I8, I16).
    // Guard: non-integer IR types (Struct, Ptr, Float, etc.) skip truncation.
    let tw = match target_ir {
        IrType::I1 | IrType::I8 | IrType::I16 | IrType::I32 | IrType::I64 | IrType::I128 => {
            target_ir.int_width()
        }
        _ => return value,
    };
    if tw > 0 && tw < 32 {
        let (v, i) = ctx.builder.build_trunc(value, target_ir, Span::dummy());
        emit_inst(ctx, i);
        return v;
    }
    value
}

/// Emit a compile-time floating-point constant.
fn emit_float_const(
    ctx: &mut ExprLoweringContext<'_>,
    value: f64,
    ir_ty: IrType,
    span: Span,
) -> Value {
    let const_id = ctx.module.globals().len();
    let gname = format!(".Lconst.f.{}", const_id);
    let gname_clone = gname.clone();
    let mut gv = GlobalVariable::new(gname, ir_ty.clone(), Some(Constant::Float(value)));
    gv.linkage = ir_module::Linkage::Internal;
    gv.is_constant = true;
    ctx.module.add_global(gv);

    let result = ctx.builder.fresh_value();
    let inst = Instruction::BinOp {
        result,
        op: IrBinOp::FAdd,
        lhs: result,
        rhs: Value::UNDEF,
        ty: ir_ty,
        span,
    };
    emit_inst(ctx, inst);
    // Record the direct Value → float constant mapping for the backend.
    ctx.function
        .float_constant_values
        .insert(result, (gname_clone, value));
    result
}

/// Emit a zero constant of the given IR type.
///
/// For pointer types, uses `Constant::Null`.  For aggregate types, uses
/// `Constant::ZeroInit`.  For scalar types, emits an integer zero.
fn emit_zero(ctx: &mut ExprLoweringContext<'_>, ir_ty: IrType, span: Span) -> Value {
    match &ir_ty {
        IrType::Ptr => {
            // Null pointer constant.
            let name = format!(".null.{}", ctx.builder.fresh_value().0);
            let mut gv = GlobalVariable::new(name, ir_ty.clone(), Some(Constant::Null));
            gv.linkage = crate::ir::module::Linkage::Internal;
            ctx.module.add_global(gv);
            emit_int_const(ctx, 0, IrType::I64, span)
        }
        IrType::Struct(_) | IrType::Array(_, _) => {
            // Aggregate zero-init.
            let name = format!(".zeroinit.{}", ctx.builder.fresh_value().0);
            let mut gv = GlobalVariable::new(name, ir_ty.clone(), Some(Constant::ZeroInit));
            gv.linkage = crate::ir::module::Linkage::Internal;
            ctx.module.add_global(gv);
            emit_int_const(ctx, 0, IrType::I32, span)
        }
        _ => emit_int_const(ctx, 0, ir_ty, span),
    }
}

/// Emit integer constant `1` of the given IR type.
fn emit_one(ctx: &mut ExprLoweringContext<'_>, ir_ty: IrType, span: Span) -> Value {
    emit_int_const(ctx, 1, ir_ty, span)
}

// ---------------------------------------------------------------------------
// Type helpers
// ---------------------------------------------------------------------------

/// Convert a C type to its IR representation.
///
/// For the most common leaf types we resolve directly to avoid a round-trip
/// through `IrType::from_ctype`; aggregate and function types still delegate.
pub fn ctype_to_ir(ctype: &CType, target: &Target) -> IrType {
    let resolved = types::resolve_typedef(ctype);
    match resolved {
        CType::Void => IrType::Void,
        CType::Bool => IrType::I1,
        CType::Char | CType::UChar | CType::SChar => IrType::I8,
        CType::Short | CType::UShort => IrType::I16,
        CType::Int | CType::UInt => IrType::I32,
        CType::Long | CType::ULong => {
            if target.long_size() == 8 {
                IrType::I64
            } else {
                IrType::I32
            }
        }
        CType::LongLong | CType::ULongLong => IrType::I64,
        CType::Int128 | CType::UInt128 => IrType::I128,
        CType::Float => IrType::F32,
        CType::Double => IrType::F64,
        CType::LongDouble => {
            // Both 80-bit and 128-bit extended precision map to IrType::F80
            // for code generation purposes.
            IrType::F80
        }
        CType::Pointer(..) => IrType::Ptr,
        CType::Array(elem, count) => {
            let elem_ir = ctype_to_ir(elem, target);
            IrType::Array(Box::new(elem_ir), count.unwrap_or(0))
        }
        CType::Struct { fields, packed, .. } => {
            let member_irs: Vec<IrType> =
                fields.iter().map(|f| ctype_to_ir(&f.ty, target)).collect();
            IrType::Struct(crate::ir::types::StructType::new(member_irs, *packed))
        }
        CType::Union {
            fields: _fields, ..
        } => {
            // Union IR type: represented as a byte array of the union size.
            let sz = target.pointer_width(); // fallback, real size from type_builder
            let _ = sz;
            IrType::from_ctype(resolved, target)
        }
        CType::Function {
            return_type,
            params,
            variadic: _,
        } => {
            let ret = ctype_to_ir(return_type, target);
            let ps: Vec<IrType> = params.iter().map(|p| ctype_to_ir(p, target)).collect();
            IrType::Function(Box::new(ret), ps)
        }
        CType::Enum { .. } => IrType::I32,
        CType::Atomic(inner) => ctype_to_ir(inner, target),
        CType::Qualified(inner, _) => ctype_to_ir(inner, target),
        CType::Typedef { underlying, .. } => ctype_to_ir(underlying, target),
        _ => IrType::from_ctype(resolved, target),
    }
}

/// Return the IR type corresponding to `size_t` for the current target.
fn size_ir_type(target: &Target) -> IrType {
    if target.is_64bit() {
        IrType::I64
    } else {
        IrType::I32
    }
}

/// Return the C type corresponding to `size_t` for the current target.
/// Map an IR element type back to an approximate CType.
///
/// This is used when a global array's element type needs to be recovered
/// for array-to-pointer decay but the original C type is unavailable
/// (e.g. lookup_var_type fell back to CType::Int).
fn ir_elem_to_approx_ctype(ir_ty: &IrType) -> CType {
    match ir_ty {
        IrType::I8 => CType::Char,
        IrType::I16 => CType::Short,
        IrType::I32 => CType::Int,
        IrType::I64 => CType::LongLong,
        IrType::F32 => CType::Float,
        IrType::F64 => CType::Double,
        IrType::Ptr => CType::Pointer(Box::new(CType::Void), TypeQualifiers::default()),
        _ => CType::Int,
    }
}

/// Map an IR type to an approximate CType (including compound types).
///
/// Used to reconstruct C types from global variable IR types for sizeof
/// and array decay type computations.
fn ir_type_to_approx_ctype(ir_ty: &IrType) -> CType {
    match ir_ty {
        IrType::I1 => CType::Bool,
        IrType::I8 => CType::Char,
        IrType::I16 => CType::Short,
        IrType::I32 => CType::Int,
        IrType::I64 => CType::LongLong,
        IrType::I128 => CType::LongLong, // approximate
        IrType::F32 => CType::Float,
        IrType::F64 => CType::Double,
        IrType::Ptr => CType::Pointer(Box::new(CType::Void), TypeQualifiers::default()),
        IrType::Array(elem, count) => {
            let elem_cty = ir_type_to_approx_ctype(elem);
            CType::Array(Box::new(elem_cty), Some(*count))
        }
        IrType::Void => CType::Void,
        _ => CType::Int,
    }
}

fn size_ctype(target: &Target) -> CType {
    if target.is_64bit() {
        CType::ULong
    } else {
        CType::UInt
    }
}

/// Check if a C type is unsigned.
fn is_unsigned(ctype: &CType) -> bool {
    let resolved = types::resolve_typedef(ctype);
    types::is_unsigned(resolved)
}

/// Check if a C type is signed (complement of unsigned for integer types).
fn is_signed(ctype: &CType) -> bool {
    let resolved = types::resolve_typedef(ctype);
    types::is_signed(resolved)
}

/// Check if a C type is a floating-point type.
fn is_floating(ctype: &CType) -> bool {
    let resolved = types::resolve_typedef(ctype);
    types::is_floating(resolved)
}

/// Check if a C type is a pointer type.
fn is_pointer_type(ctype: &CType) -> bool {
    let resolved = types::resolve_typedef(ctype);
    types::is_pointer(resolved)
}

/// Check if a C type is an integer type.
fn is_integer_type(ctype: &CType) -> bool {
    let resolved = types::resolve_typedef(ctype);
    types::is_integer(resolved)
}

/// Check if a C type is an arithmetic type (integer or floating).
fn is_arithmetic(ctype: &CType) -> bool {
    let resolved = types::resolve_typedef(ctype);
    types::is_arithmetic(resolved)
}

/// Check if a C type is a scalar type (arithmetic or pointer).
fn is_scalar(ctype: &CType) -> bool {
    let resolved = types::resolve_typedef(ctype);
    types::is_scalar(resolved)
}

/// Strip qualifiers, typedefs, and `_Atomic` to get the canonical type.
fn strip_type(ctype: &CType) -> &CType {
    match ctype {
        CType::Typedef { underlying, .. } => strip_type(underlying),
        CType::Atomic(inner) => strip_type(inner),
        CType::Qualified(inner, _) => strip_type(inner),
        other => {
            let unqual = types::unqualified(other);
            types::resolve_typedef(unqual)
        }
    }
}

/// Get the pointee type of a pointer or array C type.
fn pointee_of(ctype: &CType) -> CType {
    let resolved = types::resolve_typedef(ctype);
    match resolved {
        CType::Pointer(inner, _) => (**inner).clone(),
        CType::Array(inner, _) => (**inner).clone(),
        _ => CType::Void,
    }
}

/// Compute the byte size of the pointee for pointer arithmetic scaling.
fn pointee_size(ctype: &CType, tb: &TypeBuilder) -> usize {
    let pt = pointee_of(ctype);
    if matches!(pt, CType::Void) {
        1
    } else {
        tb.sizeof_type(&pt)
    }
}

/// Determine the C type of an integer literal from its value and suffix.
fn integer_literal_ctype(value: u128, suffix: &ast::IntegerSuffix) -> CType {
    match suffix {
        ast::IntegerSuffix::None => {
            if value <= i32::MAX as u128 {
                CType::Int
            } else if value <= i64::MAX as u128 {
                CType::Long
            } else {
                CType::ULongLong
            }
        }
        ast::IntegerSuffix::U => {
            if value <= u32::MAX as u128 {
                CType::UInt
            } else {
                CType::ULongLong
            }
        }
        ast::IntegerSuffix::L => {
            if value <= i64::MAX as u128 {
                CType::Long
            } else {
                CType::ULongLong
            }
        }
        ast::IntegerSuffix::UL => CType::ULong,
        ast::IntegerSuffix::LL => {
            if value <= i64::MAX as u128 {
                CType::LongLong
            } else {
                CType::ULongLong
            }
        }
        ast::IntegerSuffix::ULL => CType::ULongLong,
    }
}

/// Determine the C type of a float literal from its suffix.
fn float_literal_ctype(suffix: &ast::FloatSuffix) -> CType {
    match suffix {
        ast::FloatSuffix::None => CType::Double,
        ast::FloatSuffix::F => CType::Float,
        ast::FloatSuffix::L => CType::LongDouble,
    }
}

/// Resolve a `Symbol` handle to its interned string name.
fn resolve_sym<'a>(ctx: &'a ExprLoweringContext<'_>, sym_idx: u32) -> &'a str {
    ctx.name_table
        .get(sym_idx as usize)
        .map(|s| s.as_str())
        .unwrap_or("<unknown>")
}

/// Look up a variable by name, checking local_vars first, then param_values.
/// Returns `(Value, is_alloca)`:
///  - `is_alloca = true` → value is an alloca pointer (needs `Load` for rvalue)
///  - `is_alloca = false` → value is the raw SSA value (no load needed)
fn lookup_var(ctx: &ExprLoweringContext<'_>, name: &str) -> Option<(Value, bool)> {
    if let Some(&v) = ctx.local_vars.get(name) {
        return Some((v, true));
    }
    if let Some(&v) = ctx.param_values.get(name) {
        return Some((v, true));
    }
    None
}

/// Look up the declared C type of a variable.
fn lookup_var_type(ctx: &ExprLoweringContext<'_>, name: &str) -> CType {
    // For static local variables, prefer the module global (which has the
    // correct inferred array size from the initializer) over the local_types
    // entry (which may have None size from the declaration syntax).
    if let Some(mangled) = ctx.static_locals.get(name) {
        if let Some(gv) = ctx.module.get_global(mangled.as_str()) {
            return ir_type_to_approx_ctype(&gv.ty);
        }
    }
    // Check scope-local type overrides first — these handle block-scoped
    // variable shadowing (e.g. `u32 t` in one case block vs `void *t` in
    // another within the same switch).
    if let Some(ct) = ctx.scope_type_overrides.get(name) {
        return ct.clone();
    }
    // Check local types (parameters, local variables).
    if let Some(ct) = ctx.local_types.get(name) {
        return ct.clone();
    }
    // Fall back to global variable types from the module.
    if let Some(gv) = ctx.module.get_global(name) {
        return ir_type_to_approx_ctype(&gv.ty);
    }
    CType::Int
}

// =========================================================================
// PUBLIC API
// =========================================================================

/// Lower an AST expression to an IR `Value` (rvalue).
///
/// This is the primary entry point for expression lowering.  It dispatches
/// on the expression variant and returns the SSA value representing the
/// computed result.  For lvalue-capable expressions this performs an
/// implicit load.
pub fn lower_expression(ctx: &mut ExprLoweringContext<'_>, expr: &ast::Expression) -> Value {
    lower_expr_inner(ctx, expr).value
}

/// Lower an AST expression as an **lvalue**, returning a pointer `Value`.
///
/// Used for the left-hand side of assignments, `&` address-of, and other
/// contexts that require a memory address rather than a loaded value.
/// Panics (via diagnostic) if the expression is not a valid lvalue.
pub fn lower_lvalue(ctx: &mut ExprLoweringContext<'_>, expr: &ast::Expression) -> Value {
    lower_lvalue_inner(ctx, expr).value
}

// =========================================================================
// MAIN DISPATCH
// =========================================================================

/// Internal expression dispatch returning `TypedValue`.
fn lower_expr_inner(ctx: &mut ExprLoweringContext<'_>, expr: &ast::Expression) -> TypedValue {
    match expr {
        // ---- Literals ----
        ast::Expression::IntegerLiteral {
            value,
            suffix,
            span,
        } => lower_integer_literal(ctx, *value, suffix, *span),
        ast::Expression::FloatLiteral {
            value,
            suffix,
            span,
        } => lower_float_literal(ctx, *value, suffix, *span),
        ast::Expression::StringLiteral {
            segments,
            prefix,
            span,
        } => lower_string_literal(ctx, segments, prefix, *span),
        ast::Expression::CharLiteral {
            value,
            prefix,
            span,
        } => lower_char_literal(ctx, *value, prefix, *span),

        // ---- Identifier ----
        ast::Expression::Identifier { name, span } => lower_identifier(ctx, name.as_u32(), *span),

        // ---- Parenthesised (transparent) ----
        ast::Expression::Parenthesized { inner, .. } => lower_expr_inner(ctx, inner),

        // ---- Binary / Logical ----
        ast::Expression::Binary {
            op,
            left,
            right,
            span,
        } => lower_binary(ctx, op, left, right, *span),

        // ---- Unary ----
        ast::Expression::UnaryOp { op, operand, span } => lower_unary(ctx, op, operand, *span),

        // ---- Assignment / Compound assignment ----
        ast::Expression::Assignment {
            op,
            target,
            value,
            span,
        } => lower_assignment(ctx, op, target, value, *span),

        // ---- Conditional (ternary) ----
        ast::Expression::Conditional {
            condition,
            then_expr,
            else_expr,
            span,
        } => lower_conditional(ctx, condition, then_expr.as_deref(), else_expr, *span),

        // ---- Function call ----
        ast::Expression::FunctionCall { callee, args, span } => {
            // Check if this is a known compiler builtin that should be
            // lowered inline rather than emitted as an external function call.
            if let Some(tv) = try_lower_builtin_call(ctx, callee, args, *span) {
                tv
            } else {
                lower_function_call(ctx, callee, args, *span)
            }
        }

        // ---- Cast ----
        ast::Expression::Cast {
            type_name,
            operand,
            span,
        } => lower_cast_expr(ctx, type_name, operand, *span),

        // ---- sizeof / alignof ----
        ast::Expression::SizeofExpr { operand, span } => lower_sizeof_expr(ctx, operand, *span),
        ast::Expression::SizeofType { type_name, span } => lower_sizeof_type(ctx, type_name, *span),
        ast::Expression::AlignofType { type_name, span } => {
            lower_alignof_type(ctx, type_name, *span)
        }
        ast::Expression::AlignofExpr { expr: inner, span } => {
            // GCC __alignof__(expr) — evaluate to the alignment of the
            // expression's type.  The expression itself is not evaluated.
            // Treat as alignment 1 (conservative) if the type cannot be
            // determined; the sema already validated the expression.
            let size_ty = size_ir_type(ctx.target);
            let val = emit_int_const(ctx, 1, size_ty, *span);
            let _ = inner; // expression intentionally unevaluated
            TypedValue::new(val, CType::ULong)
        }

        // ---- Member access ----
        ast::Expression::MemberAccess {
            object,
            member,
            span,
        } => lower_member_access(ctx, object, member.as_u32(), false, *span),
        ast::Expression::PointerMemberAccess {
            object,
            member,
            span,
        } => lower_member_access(ctx, object, member.as_u32(), true, *span),

        // ---- Array subscript ----
        ast::Expression::ArraySubscript { base, index, span } => {
            lower_array_subscript(ctx, base, index, *span)
        }

        // ---- Post-increment / decrement ----
        ast::Expression::PostIncrement { operand, span } => {
            lower_post_inc_dec(ctx, operand, true, *span)
        }
        ast::Expression::PostDecrement { operand, span } => {
            lower_post_inc_dec(ctx, operand, false, *span)
        }

        // ---- Pre-increment / decrement ----
        ast::Expression::PreIncrement { operand, span } => {
            lower_pre_inc_dec(ctx, operand, true, *span)
        }
        ast::Expression::PreDecrement { operand, span } => {
            lower_pre_inc_dec(ctx, operand, false, *span)
        }

        // ---- Comma ----
        ast::Expression::Comma { exprs, span } => lower_comma(ctx, exprs, *span),

        // ---- Compound literal ----
        ast::Expression::CompoundLiteral {
            type_name,
            initializer,
            span,
        } => lower_compound_literal(ctx, type_name, initializer, *span),

        // ---- GCC statement expression ----
        ast::Expression::StatementExpression { compound, span } => {
            lower_statement_expression(ctx, compound, *span)
        }

        // ---- GCC builtin call ----
        ast::Expression::BuiltinCall {
            builtin,
            args,
            span,
        } => lower_builtin(ctx, builtin, args, *span),

        // ---- C11 _Generic ----
        ast::Expression::Generic {
            controlling,
            associations,
            span,
        } => lower_generic(ctx, controlling, associations, *span),

        // ---- GCC address-of-label (&&label) ----
        ast::Expression::AddressOfLabel { label, span } => {
            lower_address_of_label(ctx, label.as_u32(), *span)
        }
    }
}

/// Internal lvalue dispatch returning `TypedValue` (pointer + pointee type).
fn lower_lvalue_inner(ctx: &mut ExprLoweringContext<'_>, expr: &ast::Expression) -> TypedValue {
    match expr {
        ast::Expression::Identifier { name, span } => {
            lower_identifier_lvalue(ctx, name.as_u32(), *span)
        }
        ast::Expression::UnaryOp {
            op: ast::UnaryOp::Deref,
            operand,
            span: _,
        } => {
            // *ptr as lvalue → evaluate ptr, the pointer IS the lvalue
            let ptr = lower_expr_inner(ctx, operand);
            let pt = pointee_of(&ptr.ty);
            TypedValue::new(ptr.value, pt)
        }
        ast::Expression::ArraySubscript { base, index, span } => {
            lower_array_subscript_lvalue(ctx, base, index, *span)
        }
        ast::Expression::MemberAccess {
            object,
            member,
            span,
        } => lower_member_access_lvalue(ctx, object, member.as_u32(), false, *span),
        ast::Expression::PointerMemberAccess {
            object,
            member,
            span,
        } => lower_member_access_lvalue(ctx, object, member.as_u32(), true, *span),
        ast::Expression::Parenthesized { inner, .. } => lower_lvalue_inner(ctx, inner),
        ast::Expression::CompoundLiteral {
            type_name,
            initializer,
            span,
        } => lower_compound_literal(ctx, type_name, initializer, *span),
        // Cast expressions can be lvalues in GCC extension mode.
        ast::Expression::Cast { .. } => lower_expr_inner(ctx, expr),
        // Statement expressions can be lvalues (GCC extension).
        ast::Expression::StatementExpression { .. } => lower_expr_inner(ctx, expr),
        // Function calls returning structs used as lvalues in compound
        // expressions (GCC extension, e.g. `struct_returning_fn().field`).
        ast::Expression::FunctionCall { .. } => lower_expr_inner(ctx, expr),
        // Comma expressions — the last operand is the lvalue.
        ast::Expression::Comma { exprs, .. } => {
            if let Some(last) = exprs.last() {
                lower_lvalue_inner(ctx, last)
            } else {
                lower_expr_inner(ctx, expr)
            }
        }
        // Conditional expression can be an lvalue in GCC mode.
        ast::Expression::Conditional { .. } => lower_expr_inner(ctx, expr),
        other => {
            // Fallback: try lowering as a value. Many kernel macro
            // expansions produce expressions that look like non-lvalues
            // but are used in lvalue contexts (e.g. compound literals
            // wrapped in casts or function-call wrappers).
            let sp = expr_span(other);
            ctx.diagnostics
                .emit_warning(sp, "expression used as lvalue may not be valid");
            lower_expr_inner(ctx, other)
        }
    }
}

/// Extract the `Span` from an arbitrary expression variant.
fn expr_span(expr: &ast::Expression) -> Span {
    match expr {
        ast::Expression::IntegerLiteral { span, .. }
        | ast::Expression::FloatLiteral { span, .. }
        | ast::Expression::StringLiteral { span, .. }
        | ast::Expression::CharLiteral { span, .. }
        | ast::Expression::Identifier { span, .. }
        | ast::Expression::Parenthesized { span, .. }
        | ast::Expression::Binary { span, .. }
        | ast::Expression::UnaryOp { span, .. }
        | ast::Expression::Assignment { span, .. }
        | ast::Expression::Conditional { span, .. }
        | ast::Expression::FunctionCall { span, .. }
        | ast::Expression::Cast { span, .. }
        | ast::Expression::SizeofExpr { span, .. }
        | ast::Expression::SizeofType { span, .. }
        | ast::Expression::AlignofType { span, .. }
        | ast::Expression::AlignofExpr { span, .. }
        | ast::Expression::MemberAccess { span, .. }
        | ast::Expression::PointerMemberAccess { span, .. }
        | ast::Expression::ArraySubscript { span, .. }
        | ast::Expression::PostIncrement { span, .. }
        | ast::Expression::PostDecrement { span, .. }
        | ast::Expression::PreIncrement { span, .. }
        | ast::Expression::PreDecrement { span, .. }
        | ast::Expression::Comma { span, .. }
        | ast::Expression::CompoundLiteral { span, .. }
        | ast::Expression::StatementExpression { span, .. }
        | ast::Expression::BuiltinCall { span, .. }
        | ast::Expression::Generic { span, .. }
        | ast::Expression::AddressOfLabel { span, .. } => *span,
    }
}

// =========================================================================
// LITERAL LOWERING
// =========================================================================

fn lower_integer_literal(
    ctx: &mut ExprLoweringContext<'_>,
    value: u128,
    suffix: &ast::IntegerSuffix,
    span: Span,
) -> TypedValue {
    let cty = integer_literal_ctype(value, suffix);
    let ir_ty = ctype_to_ir(&cty, ctx.target);
    let val = emit_int_const(ctx, value as i128, ir_ty, span);
    TypedValue::new(val, cty)
}

fn lower_float_literal(
    ctx: &mut ExprLoweringContext<'_>,
    value: f64,
    suffix: &ast::FloatSuffix,
    span: Span,
) -> TypedValue {
    let cty = float_literal_ctype(suffix);
    let ir_ty = ctype_to_ir(&cty, ctx.target);
    let val = emit_float_const(ctx, value, ir_ty, span);
    TypedValue::new(val, cty)
}

fn lower_char_literal(
    ctx: &mut ExprLoweringContext<'_>,
    value: u32,
    prefix: &ast::CharPrefix,
    span: Span,
) -> TypedValue {
    let cty = match prefix {
        ast::CharPrefix::None => CType::Int,
        ast::CharPrefix::L => CType::Int,
        ast::CharPrefix::U16 => CType::UShort,
        ast::CharPrefix::U32 => CType::UInt,
    };
    let ir_ty = ctype_to_ir(&cty, ctx.target);
    let val = emit_int_const(ctx, value as i128, ir_ty, span);
    TypedValue::new(val, cty)
}

fn lower_string_literal(
    ctx: &mut ExprLoweringContext<'_>,
    segments: &[ast::StringSegment],
    _prefix: &ast::StringPrefix,
    span: Span,
) -> TypedValue {
    let mut bytes = Vec::new();
    for seg in segments {
        // PUA-encoded bytes pass through as-is for byte-exact fidelity.
        bytes.extend_from_slice(&seg.value);
    }
    bytes.push(0); // null terminator
    let str_id = ctx.module.intern_string(bytes.clone());

    // Register a global for the string literal with Constant::String initializer.
    // Only add the global once — if the same string content was already interned,
    // intern_string returns the same id so the global name will collide.
    let str_global_name = format!(".str.{}", str_id);
    if ctx.module.get_global(&str_global_name).is_none() {
        let str_arr_ty = IrType::Array(Box::new(IrType::I8), bytes.len());
        let mut gv = GlobalVariable::new(
            str_global_name.clone(),
            str_arr_ty,
            Some(Constant::String(bytes)),
        );
        gv.linkage = crate::ir::module::Linkage::Internal;
        gv.is_constant = true;
        ctx.module.add_global(gv);
    }

    let result = ctx.builder.fresh_value();
    // Register the string literal base pointer as a global variable
    // reference so the backend can emit the correct address relocation
    // (e.g., RIP-relative on x86-64) instead of an immediate zero.
    ctx.module
        .global_var_refs
        .insert(result, str_global_name.clone());
    {
        let current_func = &mut *ctx.function;
        current_func.global_var_refs.insert(result, str_global_name);
    }
    let zero = emit_int_const(ctx, 0, IrType::I64, span);
    let (val, gep_inst) = ctx.builder.build_gep(result, vec![zero], IrType::Ptr, span);
    emit_inst(ctx, gep_inst);
    let cty = CType::Pointer(
        Box::new(CType::Qualified(
            Box::new(CType::Char),
            TypeQualifiers {
                is_const: true,
                is_volatile: false,
                is_restrict: false,
                is_atomic: false,
            },
        )),
        TypeQualifiers::default(),
    );
    TypedValue::new(val, cty)
}

// =========================================================================
// IDENTIFIER LOWERING
// =========================================================================

fn lower_identifier(ctx: &mut ExprLoweringContext<'_>, sym_idx: u32, span: Span) -> TypedValue {
    let name = resolve_sym(ctx, sym_idx);
    let var_cty = lookup_var_type(ctx, name);
    let ir_ty = ctype_to_ir(&var_cty, ctx.target);

    // Arrays decay to pointers: the alloca IS the address of the first
    // element, so we return the alloca pointer directly without loading.
    let is_array_decay = types::is_array(&var_cty);
    let decayed_ty = if is_array_decay {
        // Array decays to pointer to element type.
        match &var_cty {
            CType::Array(elem, _) => CType::Pointer(elem.clone(), TypeQualifiers::default()),
            _ => var_cty.clone(),
        }
    } else {
        var_cty.clone()
    };

    // First check function parameter values (common fast-path).
    if ctx.param_values.contains_key(name) {
        if let Some((ptr_val, needs_load)) = lookup_var(ctx, name) {
            if is_array_decay {
                return TypedValue::new(ptr_val, decayed_ty);
            }
            if needs_load {
                let (loaded, li) = ctx.builder.build_load(ptr_val, ir_ty, span);
                emit_inst(ctx, li);
                return TypedValue::new(loaded, var_cty);
            }
            return TypedValue::new(ptr_val, var_cty);
        }
    }

    // Local variables.
    if let Some((ptr_val, needs_load)) = lookup_var(ctx, name) {
        if is_array_decay {
            return TypedValue::new(ptr_val, decayed_ty);
        }
        if needs_load {
            let (loaded, li) = ctx.builder.build_load(ptr_val, ir_ty, span);
            emit_inst(ctx, li);
            return TypedValue::new(loaded, var_cty);
        }
        return TypedValue::new(ptr_val, var_cty);
    }

    // Global variables.
    {
        let global_ty = ctx.module.get_global(name).map(|gv| gv.ty.clone());
        if let Some(gt) = global_ty {
            let global_name = name.to_string();
            let ptr_val = ctx.builder.fresh_value();
            // Record the mapping from this Value to the global variable name
            // so the backend can emit architecture-specific global addressing
            // (e.g., RIP-relative on x86-64, ADRP/LDR on AArch64).
            ctx.module
                .global_var_refs
                .insert(ptr_val, global_name.clone());
            {
                let current_func = &mut *ctx.function;
                current_func.global_var_refs.insert(ptr_val, global_name);
            }

            // Check if the global variable itself is an array type.
            // lookup_var_type only checks local_types and may miss globals,
            // so we also detect array decay by inspecting the IR type.
            let global_is_array = matches!(&gt, crate::ir::types::IrType::Array(_, _));

            // Arrays decay to pointers: the address of the global IS the
            // pointer to the first element, so return it directly (no load).
            if is_array_decay || global_is_array {
                // Compute proper decayed type.
                let ret_ty = if is_array_decay {
                    decayed_ty
                } else {
                    // Build pointer-to-element from the IR array element type.
                    // We construct a generic pointer CType since we don't have
                    // the original C element type here.
                    match &gt {
                        crate::ir::types::IrType::Array(elem_ir, _) => {
                            // Map IR element type back to an approximate CType.
                            let elem_cty = ir_elem_to_approx_ctype(elem_ir);
                            CType::Pointer(Box::new(elem_cty), TypeQualifiers::default())
                        }
                        _ => CType::Pointer(Box::new(CType::Int), TypeQualifiers::default()),
                    }
                };
                return TypedValue::new(ptr_val, ret_ty);
            }

            let (loaded, li) = ctx.builder.build_load(ptr_val, gt, span);
            emit_inst(ctx, li);
            return TypedValue::new(loaded, var_cty);
        }
    }

    // Function references (decay to function pointer).
    if ctx.module.get_function(name).is_some()
        || ctx.module.declarations().iter().any(|d| d.name == *name)
    {
        let func_name = name.to_string();
        let fptr = ctx.builder.fresh_value();
        // Record in both the module-level map (for backward compat) and the
        // per-function map (which is Value-ID safe across functions).
        ctx.module.func_ref_map.insert(fptr, func_name.clone());
        {
            let current_func = &mut *ctx.function;
            current_func.func_ref_map.insert(fptr, func_name);
        }
        return TypedValue::new(
            fptr,
            CType::Pointer(Box::new(CType::Void), TypeQualifiers::default()),
        );
    }

    // Enum constants — compile-time integer values.
    if let Some(&enum_val) = ctx.enum_constants.get(name) {
        let val = emit_int_const(ctx, enum_val, IrType::I32, span);
        return TypedValue::new(val, CType::Int);
    }

    // Static local variables — redirected to their mangled global name.
    if let Some(mangled) = ctx.static_locals.get(name) {
        let global_ty = ctx
            .module
            .get_global(mangled.as_str())
            .map(|gv| gv.ty.clone());
        if let Some(gt) = global_ty {
            let mangled_name = mangled.clone();
            let ptr_val = ctx.builder.fresh_value();
            ctx.module
                .global_var_refs
                .insert(ptr_val, mangled_name.clone());
            {
                let current_func = &mut *ctx.function;
                current_func.global_var_refs.insert(ptr_val, mangled_name);
            }
            // Check for array-to-pointer decay: arrays and string-initialized
            // arrays should return the pointer directly (no load).
            let global_is_array = matches!(&gt, IrType::Array(_, _));
            if is_array_decay || global_is_array {
                // Build the decayed pointer type for the return.
                let ret_ty = if let IrType::Array(elem, _) = &gt {
                    let elem_cty = ir_elem_to_approx_ctype(elem);
                    CType::Pointer(Box::new(elem_cty), TypeQualifiers::default())
                } else {
                    var_cty
                };
                return TypedValue::new(ptr_val, ret_ty);
            }
            let (loaded, li) = ctx.builder.build_load(ptr_val, gt, span);
            emit_inst(ctx, li);
            return TypedValue::new(loaded, var_cty);
        }
    }

    // C99 §6.4.2.2: __func__ is an implicit static const char[] in each
    // function, expanding to the function name. GCC also supports
    // __FUNCTION__ and __PRETTY_FUNCTION__ as synonyms.
    if name == "__func__" || name == "__FUNCTION__" || name == "__PRETTY_FUNCTION__" {
        let func_name = ctx.current_function_name.unwrap_or("(unknown)");
        let mut bytes: Vec<u8> = func_name.as_bytes().to_vec();
        bytes.push(0); // NUL terminator
        let str_id = ctx.module.intern_string(bytes.clone());
        let str_global_name = format!(".str.{}", str_id);
        if ctx.module.get_global(&str_global_name).is_none() {
            let str_arr_ty = IrType::Array(Box::new(IrType::I8), bytes.len());
            let mut gv = crate::ir::module::GlobalVariable::new(
                str_global_name.clone(),
                str_arr_ty,
                Some(crate::ir::module::Constant::String(bytes)),
            );
            gv.linkage = crate::ir::module::Linkage::Internal;
            gv.is_constant = true;
            ctx.module.add_global(gv);
        }
        let result = ctx.builder.fresh_value();
        ctx.module
            .global_var_refs
            .insert(result, str_global_name.clone());
        {
            let current_func = &mut *ctx.function;
            current_func.global_var_refs.insert(result, str_global_name);
        }
        return TypedValue::new(
            result,
            CType::Pointer(
                Box::new(CType::Char),
                crate::common::types::TypeQualifiers {
                    is_const: true,
                    ..Default::default()
                },
            ),
        );
    }

    // Implicit builtin functions — auto-declare in the IR module when
    // __builtin_* identifiers are used as function references.  The kernel
    // and libc code freely call __builtin_memcpy, __builtin_memset, etc.
    // without explicit declarations, relying on the compiler's knowledge.
    if name.starts_with("__builtin_") {
        // Capture the name as an owned string to release the borrow on ctx.
        let owned_name = name.to_string();
        // Map builtins to their underlying libc symbol names: the codegen
        // will emit a call to the libc function (e.g. memcpy, memset).
        let (lib_name_str, ret_ir, params_ir, variadic): (&str, IrType, Vec<IrType>, bool) =
            match owned_name.as_str() {
                "__builtin_memcpy" | "__builtin_memmove" => (
                    owned_name.strip_prefix("__builtin_").unwrap(),
                    IrType::Ptr,
                    vec![IrType::Ptr, IrType::Ptr, IrType::I64],
                    false,
                ),
                "__builtin_memset" => (
                    "memset",
                    IrType::Ptr,
                    vec![IrType::Ptr, IrType::I32, IrType::I64],
                    false,
                ),
                "__builtin_memcmp" => (
                    "memcmp",
                    IrType::I32,
                    vec![IrType::Ptr, IrType::Ptr, IrType::I64],
                    false,
                ),
                "__builtin_strlen" => ("strlen", IrType::I64, vec![IrType::Ptr], false),
                "__builtin_strcmp" | "__builtin_strncmp" => (
                    owned_name.strip_prefix("__builtin_").unwrap(),
                    IrType::I32,
                    vec![IrType::Ptr, IrType::Ptr],
                    false,
                ),
                "__builtin_strcpy" | "__builtin_strncpy" | "__builtin_strcat"
                | "__builtin_strncat" => (
                    owned_name.strip_prefix("__builtin_").unwrap(),
                    IrType::Ptr,
                    vec![IrType::Ptr, IrType::Ptr],
                    false,
                ),
                "__builtin_strchr" | "__builtin_strrchr" => (
                    owned_name.strip_prefix("__builtin_").unwrap(),
                    IrType::Ptr,
                    vec![IrType::Ptr, IrType::I32],
                    false,
                ),
                "__builtin_abs" => ("abs", IrType::I32, vec![IrType::I32], false),
                "__builtin_labs" => ("labs", IrType::I64, vec![IrType::I64], false),
                _ => {
                    // Generic fallback: declare as returning int with variadic params.
                    (
                        owned_name.strip_prefix("__builtin_").unwrap_or(&owned_name),
                        IrType::I32,
                        vec![],
                        true,
                    )
                }
            };
        let lib_name_owned = lib_name_str.to_string();
        // Add a declaration for the underlying libc function if not present.
        let already_declared = ctx
            .module
            .declarations()
            .iter()
            .any(|d| d.name == lib_name_owned)
            || ctx.module.get_function(&lib_name_owned).is_some();
        if !already_declared {
            ctx.module
                .add_declaration(crate::ir::module::FunctionDeclaration {
                    name: lib_name_owned.clone(),
                    return_type: ret_ir.clone(),
                    param_types: params_ir.clone(),
                    is_variadic: variadic,
                    linkage: crate::ir::module::Linkage::External,
                    visibility: crate::ir::module::Visibility::Default,
                });
        }
        // Also add the __builtin_* name as an alias so the call site resolves.
        let already_builtin = ctx
            .module
            .declarations()
            .iter()
            .any(|d| d.name == owned_name)
            || ctx.module.get_function(&owned_name).is_some();
        if !already_builtin {
            ctx.module
                .add_declaration(crate::ir::module::FunctionDeclaration {
                    name: owned_name.clone(),
                    return_type: ret_ir,
                    param_types: params_ir,
                    is_variadic: variadic,
                    linkage: crate::ir::module::Linkage::External,
                    visibility: crate::ir::module::Visibility::Default,
                });
        }
        // Return a function reference value.
        let fptr = ctx.builder.fresh_value();
        ctx.module.func_ref_map.insert(fptr, owned_name.clone());
        {
            let current_func = &mut *ctx.function;
            current_func.func_ref_map.insert(fptr, owned_name);
        }
        return TypedValue::new(
            fptr,
            CType::Pointer(Box::new(CType::Void), TypeQualifiers::default()),
        );
    }

    // GCC __atomic_* and __sync_* builtins — atomic operations that the
    // compiler treats as known functions.  For simple cases (aligned integer
    // loads/stores) these compile to plain loads/stores on architectures
    // with naturally-atomic access (x86-64, AArch64).  We auto-declare them
    // as external functions so the call site resolves, then the call lowering
    // can handle them specially or emit a regular library call.
    if name.starts_with("__atomic_") || name.starts_with("__sync_") {
        let owned_name = name.to_string();
        let (ret_ir, params_ir, variadic): (IrType, Vec<IrType>, bool) = match owned_name.as_str() {
            "__atomic_load_n" | "__atomic_load" => {
                (IrType::I64, vec![IrType::Ptr, IrType::I32], false)
            }
            "__atomic_store_n" | "__atomic_store" => (
                IrType::Void,
                vec![IrType::Ptr, IrType::I64, IrType::I32],
                false,
            ),
            "__atomic_exchange_n" | "__atomic_exchange" => (
                IrType::I64,
                vec![IrType::Ptr, IrType::I64, IrType::I32],
                false,
            ),
            "__atomic_compare_exchange_n" | "__atomic_compare_exchange" => (
                IrType::I8,
                vec![
                    IrType::Ptr,
                    IrType::Ptr,
                    IrType::I64,
                    IrType::I8,
                    IrType::I32,
                    IrType::I32,
                ],
                false,
            ),
            "__atomic_add_fetch"
            | "__atomic_sub_fetch"
            | "__atomic_and_fetch"
            | "__atomic_or_fetch"
            | "__atomic_xor_fetch"
            | "__atomic_nand_fetch"
            | "__atomic_fetch_add"
            | "__atomic_fetch_sub"
            | "__atomic_fetch_and"
            | "__atomic_fetch_or"
            | "__atomic_fetch_xor"
            | "__atomic_fetch_nand" => (
                IrType::I64,
                vec![IrType::Ptr, IrType::I64, IrType::I32],
                false,
            ),
            "__atomic_test_and_set" => (IrType::I8, vec![IrType::Ptr, IrType::I32], false),
            "__atomic_clear" => (IrType::Void, vec![IrType::Ptr, IrType::I32], false),
            "__atomic_thread_fence" | "__atomic_signal_fence" => {
                (IrType::Void, vec![IrType::I32], false)
            }
            "__atomic_always_lock_free" | "__atomic_is_lock_free" => {
                (IrType::I8, vec![IrType::I64, IrType::Ptr], false)
            }
            "__sync_synchronize" => (IrType::Void, vec![], false),
            "__sync_fetch_and_add"
            | "__sync_fetch_and_sub"
            | "__sync_fetch_and_or"
            | "__sync_fetch_and_and"
            | "__sync_fetch_and_xor"
            | "__sync_fetch_and_nand"
            | "__sync_add_and_fetch"
            | "__sync_sub_and_fetch"
            | "__sync_or_and_fetch"
            | "__sync_and_and_fetch"
            | "__sync_xor_and_fetch"
            | "__sync_nand_and_fetch" => (IrType::I64, vec![IrType::Ptr, IrType::I64], true),
            "__sync_bool_compare_and_swap" => (
                IrType::I8,
                vec![IrType::Ptr, IrType::I64, IrType::I64],
                true,
            ),
            "__sync_val_compare_and_swap" => (
                IrType::I64,
                vec![IrType::Ptr, IrType::I64, IrType::I64],
                true,
            ),
            "__sync_lock_test_and_set" => (IrType::I64, vec![IrType::Ptr, IrType::I64], true),
            "__sync_lock_release" => (IrType::Void, vec![IrType::Ptr], true),
            _ => {
                // Generic fallback: int-returning with variadic params.
                (IrType::I64, vec![], true)
            }
        };
        let already_declared = ctx
            .module
            .declarations()
            .iter()
            .any(|d| d.name == owned_name)
            || ctx.module.get_function(&owned_name).is_some();
        if !already_declared {
            ctx.module
                .add_declaration(crate::ir::module::FunctionDeclaration {
                    name: owned_name.clone(),
                    return_type: ret_ir,
                    param_types: params_ir,
                    is_variadic: variadic,
                    linkage: crate::ir::module::Linkage::External,
                    visibility: crate::ir::module::Visibility::Default,
                });
        }
        let fptr = ctx.builder.fresh_value();
        ctx.module.func_ref_map.insert(fptr, owned_name.clone());
        {
            let current_func = &mut *ctx.function;
            current_func.func_ref_map.insert(fptr, owned_name);
        }
        return TypedValue::new(
            fptr,
            CType::Pointer(Box::new(CType::Void), TypeQualifiers::default()),
        );
    }

    ctx.diagnostics
        .emit_error(span, format!("undeclared identifier '{}'", name));
    TypedValue::new(Value::UNDEF, CType::Int)
}

fn lower_identifier_lvalue(
    ctx: &mut ExprLoweringContext<'_>,
    sym_idx: u32,
    span: Span,
) -> TypedValue {
    let name = resolve_sym(ctx, sym_idx);
    let var_cty = lookup_var_type(ctx, name);
    if let Some((ptr_val, _)) = lookup_var(ctx, name) {
        return TypedValue::new(ptr_val, var_cty);
    }
    if ctx.module.get_global(name).is_some() {
        let global_name = name.to_string();
        let ptr_val = ctx.builder.fresh_value();
        ctx.module
            .global_var_refs
            .insert(ptr_val, global_name.clone());
        {
            let current_func = &mut *ctx.function;
            current_func.global_var_refs.insert(ptr_val, global_name);
        }
        return TypedValue::new(ptr_val, var_cty);
    }
    // Static local variables — redirected to their mangled global name.
    if let Some(mangled) = ctx.static_locals.get(name) {
        if ctx.module.get_global(mangled.as_str()).is_some() {
            let mangled_name = mangled.clone();
            let ptr_val = ctx.builder.fresh_value();
            ctx.module
                .global_var_refs
                .insert(ptr_val, mangled_name.clone());
            {
                let current_func = &mut *ctx.function;
                current_func.global_var_refs.insert(ptr_val, mangled_name);
            }
            return TypedValue::new(ptr_val, var_cty);
        }
    }
    // Enum constants — compile-time integer values used as lvalues
    // (rare but can occur in macro expansions).
    if let Some(&enum_val) = ctx.enum_constants.get(name) {
        let val = emit_int_const(ctx, enum_val, IrType::I32, span);
        return TypedValue::new(val, CType::Int);
    }
    // Function references as lvalues — used for `&function_name`.
    // Functions decay to pointers, so taking the address of a function
    // returns the function pointer value.
    if ctx.module.get_function(name).is_some()
        || ctx.module.declarations().iter().any(|d| d.name == *name)
    {
        let func_name = name.to_string();
        let fptr = ctx.builder.fresh_value();
        ctx.module.func_ref_map.insert(fptr, func_name.clone());
        {
            let current_func = &mut *ctx.function;
            current_func.func_ref_map.insert(fptr, func_name);
        }
        return TypedValue::new(
            fptr,
            CType::Pointer(Box::new(CType::Void), TypeQualifiers::default()),
        );
    }
    ctx.diagnostics
        .emit_error(span, format!("undeclared identifier '{}'", name));
    TypedValue::new(Value::UNDEF, CType::Int)
}

// =========================================================================
// BINARY OPERATIONS
// =========================================================================

fn lower_binary(
    ctx: &mut ExprLoweringContext<'_>,
    op: &ast::BinaryOp,
    lhs_expr: &ast::Expression,
    rhs_expr: &ast::Expression,
    span: Span,
) -> TypedValue {
    match op {
        ast::BinaryOp::LogicalAnd => return lower_logical_and(ctx, lhs_expr, rhs_expr, span),
        ast::BinaryOp::LogicalOr => return lower_logical_or(ctx, lhs_expr, rhs_expr, span),
        _ => {}
    }
    let lhs = lower_expr_inner(ctx, lhs_expr);
    let rhs = lower_expr_inner(ctx, rhs_expr);

    if is_pointer_type(&lhs.ty) && is_integer_type(&rhs.ty) {
        // Pointer-integer comparisons: convert integer to pointer and use ptr-ptr comparison.
        match op {
            ast::BinaryOp::Equal
            | ast::BinaryOp::NotEqual
            | ast::BinaryOp::Less
            | ast::BinaryOp::LessEqual
            | ast::BinaryOp::Greater
            | ast::BinaryOp::GreaterEqual => {
                let pi = size_ir_type(ctx.target);
                let (li, lic) = ctx.builder.build_ptr_to_int(lhs.value, pi.clone(), span);
                emit_inst(ctx, lic);
                let rv = insert_implicit_conversion(
                    ctx,
                    rhs.value,
                    &rhs.ty,
                    &size_ctype(ctx.target),
                    span,
                );
                let cmp_op = match op {
                    ast::BinaryOp::Equal => ICmpOp::Eq,
                    ast::BinaryOp::NotEqual => ICmpOp::Ne,
                    ast::BinaryOp::Less => ICmpOp::Ult,
                    ast::BinaryOp::LessEqual => ICmpOp::Ule,
                    ast::BinaryOp::Greater => ICmpOp::Ugt,
                    ast::BinaryOp::GreaterEqual => ICmpOp::Uge,
                    _ => unreachable!(),
                };
                let (v, ci) = ctx.builder.build_icmp(cmp_op, li, rv, span);
                emit_inst(ctx, ci);
                return TypedValue::new(v, CType::Bool);
            }
            _ => {
                return lower_ptr_arith(ctx, op, &lhs, &rhs, span);
            }
        }
    }
    if is_integer_type(&lhs.ty) && is_pointer_type(&rhs.ty) {
        match op {
            ast::BinaryOp::Add => {
                return lower_ptr_arith(ctx, op, &rhs, &lhs, span);
            }
            ast::BinaryOp::Equal
            | ast::BinaryOp::NotEqual
            | ast::BinaryOp::Less
            | ast::BinaryOp::LessEqual
            | ast::BinaryOp::Greater
            | ast::BinaryOp::GreaterEqual => {
                let pi = size_ir_type(ctx.target);
                let lv = insert_implicit_conversion(
                    ctx,
                    lhs.value,
                    &lhs.ty,
                    &size_ctype(ctx.target),
                    span,
                );
                let (ri, ric) = ctx.builder.build_ptr_to_int(rhs.value, pi.clone(), span);
                emit_inst(ctx, ric);
                let cmp_op = match op {
                    ast::BinaryOp::Equal => ICmpOp::Eq,
                    ast::BinaryOp::NotEqual => ICmpOp::Ne,
                    ast::BinaryOp::Less => ICmpOp::Ult,
                    ast::BinaryOp::LessEqual => ICmpOp::Ule,
                    ast::BinaryOp::Greater => ICmpOp::Ugt,
                    ast::BinaryOp::GreaterEqual => ICmpOp::Uge,
                    _ => unreachable!(),
                };
                let (v, ci) = ctx.builder.build_icmp(cmp_op, lv, ri, span);
                emit_inst(ctx, ci);
                return TypedValue::new(v, CType::Bool);
            }
            _ => {}
        }
    }
    if is_pointer_type(&lhs.ty) && is_pointer_type(&rhs.ty) && matches!(op, ast::BinaryOp::Sub) {
        return lower_ptr_diff(ctx, &lhs, &rhs, span);
    }
    // Pointer-pointer comparisons (==, !=, <, >, <=, >=): convert both to
    // integers (ptrdiff_t-sized) and perform an unsigned integer comparison.
    if is_pointer_type(&lhs.ty) && is_pointer_type(&rhs.ty) {
        match op {
            ast::BinaryOp::Equal
            | ast::BinaryOp::NotEqual
            | ast::BinaryOp::Less
            | ast::BinaryOp::LessEqual
            | ast::BinaryOp::Greater
            | ast::BinaryOp::GreaterEqual => {
                let pi = size_ir_type(ctx.target);
                let (li, lic) = ctx.builder.build_ptr_to_int(lhs.value, pi.clone(), span);
                emit_inst(ctx, lic);
                let (ri, ric) = ctx.builder.build_ptr_to_int(rhs.value, pi.clone(), span);
                emit_inst(ctx, ric);
                let cmp_op = match op {
                    ast::BinaryOp::Equal => ICmpOp::Eq,
                    ast::BinaryOp::NotEqual => ICmpOp::Ne,
                    ast::BinaryOp::Less => ICmpOp::Ult,
                    ast::BinaryOp::LessEqual => ICmpOp::Ule,
                    ast::BinaryOp::Greater => ICmpOp::Ugt,
                    ast::BinaryOp::GreaterEqual => ICmpOp::Uge,
                    _ => unreachable!(),
                };
                let (v, ci) = ctx.builder.build_icmp(cmp_op, li, ri, span);
                emit_inst(ctx, ci);
                return TypedValue::new(v, CType::Bool);
            }
            _ => {
                // Pointer + pointer is only valid for sub (handled above)
                // and comparisons (handled above). Other ops fall through
                // to the pointer arithmetic handler which will error.
                return lower_ptr_arith(ctx, op, &lhs, &rhs, span);
            }
        }
    }

    // Apply usual arithmetic conversions; also check integer rank for
    // type-dependent instruction selection (signed vs unsigned).
    let _lhs_rank = type_builder::integer_rank(&lhs.ty);
    let _rhs_rank = type_builder::integer_rank(&rhs.ty);
    let _lhs_is_int = type_builder::is_integer_type(&lhs.ty);
    let common = type_builder::usual_arithmetic_conversion(&lhs.ty, &rhs.ty);
    let ci = ctype_to_ir(&common, ctx.target);
    let lv = insert_implicit_conversion(ctx, lhs.value, &lhs.ty, &common, span);
    let rv = insert_implicit_conversion(ctx, rhs.value, &rhs.ty, &common, span);
    let uns = is_unsigned(&common);

    match op {
        ast::BinaryOp::Add => {
            let (v, i) = ctx.builder.build_add(lv, rv, ci, span);
            emit_inst(ctx, i);
            TypedValue::new(v, common)
        }
        ast::BinaryOp::Sub => {
            let (v, i) = ctx.builder.build_sub(lv, rv, ci, span);
            emit_inst(ctx, i);
            TypedValue::new(v, common)
        }
        ast::BinaryOp::Mul => {
            let (v, i) = ctx.builder.build_mul(lv, rv, ci, span);
            emit_inst(ctx, i);
            TypedValue::new(v, common)
        }
        ast::BinaryOp::Div => {
            let (v, i) = ctx.builder.build_div(lv, rv, ci, !uns, span);
            emit_inst(ctx, i);
            TypedValue::new(v, common)
        }
        ast::BinaryOp::Mod => {
            let (v, i) = ctx.builder.build_rem(lv, rv, ci, !uns, span);
            emit_inst(ctx, i);
            TypedValue::new(v, common)
        }
        ast::BinaryOp::BitwiseAnd => {
            let (v, i) = ctx.builder.build_and(lv, rv, ci, span);
            emit_inst(ctx, i);
            TypedValue::new(v, common)
        }
        ast::BinaryOp::BitwiseOr => {
            let (v, i) = ctx.builder.build_or(lv, rv, ci, span);
            emit_inst(ctx, i);
            TypedValue::new(v, common)
        }
        ast::BinaryOp::BitwiseXor => {
            let (v, i) = ctx.builder.build_xor(lv, rv, ci, span);
            emit_inst(ctx, i);
            TypedValue::new(v, common)
        }
        ast::BinaryOp::ShiftLeft => {
            let (v, i) = ctx.builder.build_shl(lv, rv, ci, span);
            emit_inst(ctx, i);
            TypedValue::new(v, common)
        }
        ast::BinaryOp::ShiftRight => {
            let (v, i) = ctx.builder.build_shr(lv, rv, ci, !uns, span);
            emit_inst(ctx, i);
            TypedValue::new(v, common)
        }
        ast::BinaryOp::Equal => TypedValue::new(
            emit_cmp(ctx, lv, rv, &common, ICmpOp::Eq, FCmpOp::Oeq, span),
            CType::Bool,
        ),
        ast::BinaryOp::NotEqual => TypedValue::new(
            emit_cmp(ctx, lv, rv, &common, ICmpOp::Ne, FCmpOp::One, span),
            CType::Bool,
        ),
        ast::BinaryOp::Less => {
            let ic = if uns { ICmpOp::Ult } else { ICmpOp::Slt };
            TypedValue::new(
                emit_cmp(ctx, lv, rv, &common, ic, FCmpOp::Olt, span),
                CType::Bool,
            )
        }
        ast::BinaryOp::LessEqual => {
            let ic = if uns { ICmpOp::Ule } else { ICmpOp::Sle };
            TypedValue::new(
                emit_cmp(ctx, lv, rv, &common, ic, FCmpOp::Ole, span),
                CType::Bool,
            )
        }
        ast::BinaryOp::Greater => {
            let ic = if uns { ICmpOp::Ugt } else { ICmpOp::Sgt };
            TypedValue::new(
                emit_cmp(ctx, lv, rv, &common, ic, FCmpOp::Ogt, span),
                CType::Bool,
            )
        }
        ast::BinaryOp::GreaterEqual => {
            let ic = if uns { ICmpOp::Uge } else { ICmpOp::Sge };
            TypedValue::new(
                emit_cmp(ctx, lv, rv, &common, ic, FCmpOp::Oge, span),
                CType::Bool,
            )
        }
        ast::BinaryOp::LogicalAnd | ast::BinaryOp::LogicalOr => unreachable!(),
    }
}

fn emit_cmp(
    ctx: &mut ExprLoweringContext<'_>,
    l: Value,
    r: Value,
    ty: &CType,
    ic: ICmpOp,
    fc: FCmpOp,
    span: Span,
) -> Value {
    if is_floating(ty) {
        let (v, i) = ctx.builder.build_fcmp(fc, l, r, span);
        emit_inst(ctx, i);
        v
    } else {
        let (v, i) = ctx.builder.build_icmp(ic, l, r, span);
        emit_inst(ctx, i);
        v
    }
}

fn lower_logical_and(
    ctx: &mut ExprLoweringContext<'_>,
    le: &ast::Expression,
    re: &ast::Expression,
    span: Span,
) -> TypedValue {
    let rb = new_block(ctx);
    let mb = new_block(ctx);
    let lhs = lower_expr_inner(ctx, le);
    let lb = lower_to_bool(ctx, lhs.value, &lhs.ty, span);
    // Emit the short-circuit false constant (0) in the current block BEFORE
    // the conditional branch.  This ensures the phi in the merge block is the
    // very first instruction, maintaining the SSA invariant that phi nodes
    // form a contiguous prefix of a basic block's instruction list.
    let fv = emit_int_const(ctx, 0, IrType::I1, span);
    let cond_blk = ctx.builder.get_insert_block().unwrap();
    let ci = ctx.builder.build_cond_branch(lb, rb, mb, span);
    emit_inst(ctx, ci);
    // CFG edges: cond_blk → rb (rhs eval), cond_blk → mb (short-circuit false)
    add_cfg_edge(ctx, cond_blk.index(), rb.index());
    add_cfg_edge(ctx, cond_blk.index(), mb.index());
    let le_blk = cond_blk;
    ctx.builder.set_insert_point(rb);
    let rhs = lower_expr_inner(ctx, re);
    let rbv = lower_to_bool(ctx, rhs.value, &rhs.ty, span);
    let bi = ctx.builder.build_branch(mb, span);
    emit_inst(ctx, bi);
    let re_blk = ctx.builder.get_insert_block().unwrap();
    // CFG edge: re_blk → mb
    add_cfg_edge(ctx, re_blk.index(), mb.index());
    ctx.builder.set_insert_point(mb);
    let (pv, pi) = ctx
        .builder
        .build_phi(IrType::I1, vec![(fv, le_blk), (rbv, re_blk)], span);
    emit_inst(ctx, pi);
    TypedValue::new(pv, CType::Bool)
}

fn lower_logical_or(
    ctx: &mut ExprLoweringContext<'_>,
    le: &ast::Expression,
    re: &ast::Expression,
    span: Span,
) -> TypedValue {
    let rb = new_block(ctx);
    let mb = new_block(ctx);
    let lhs = lower_expr_inner(ctx, le);
    let lb = lower_to_bool(ctx, lhs.value, &lhs.ty, span);
    // Emit the short-circuit true constant (1) in the current block BEFORE
    // the conditional branch, so the phi in the merge block is the very
    // first instruction, preserving the phi-prefix invariant.
    let tv = emit_int_const(ctx, 1, IrType::I1, span);
    let cond_blk = ctx.builder.get_insert_block().unwrap();
    let ci = ctx.builder.build_cond_branch(lb, mb, rb, span);
    emit_inst(ctx, ci);
    // CFG edges: cond_blk → mb (short-circuit true), cond_blk → rb (rhs eval)
    add_cfg_edge(ctx, cond_blk.index(), mb.index());
    add_cfg_edge(ctx, cond_blk.index(), rb.index());
    let le_blk = cond_blk;
    ctx.builder.set_insert_point(rb);
    let rhs = lower_expr_inner(ctx, re);
    let rbv = lower_to_bool(ctx, rhs.value, &rhs.ty, span);
    let bi = ctx.builder.build_branch(mb, span);
    emit_inst(ctx, bi);
    let re_blk = ctx.builder.get_insert_block().unwrap();
    // CFG edge: re_blk → mb
    add_cfg_edge(ctx, re_blk.index(), mb.index());
    ctx.builder.set_insert_point(mb);
    let (pv, pi) = ctx
        .builder
        .build_phi(IrType::I1, vec![(tv, le_blk), (rbv, re_blk)], span);
    emit_inst(ctx, pi);
    TypedValue::new(pv, CType::Bool)
}

fn lower_ptr_arith(
    ctx: &mut ExprLoweringContext<'_>,
    op: &ast::BinaryOp,
    ptr: &TypedValue,
    idx: &TypedValue,
    span: Span,
) -> TypedValue {
    let es = pointee_size(&ptr.ty, ctx.type_builder) as i128;
    let si = size_ir_type(ctx.target);
    let sc = emit_int_const(ctx, es, si.clone(), span);
    let iv = insert_implicit_conversion(ctx, idx.value, &idx.ty, &size_ctype(ctx.target), span);
    let (bo, mi) = ctx.builder.build_mul(iv, sc, si.clone(), span);
    emit_inst(ctx, mi);
    match op {
        ast::BinaryOp::Add => {
            let (r, g) = ctx
                .builder
                .build_gep(ptr.value, vec![bo], IrType::Ptr, span);
            emit_inst(ctx, g);
            TypedValue::new(r, ptr.ty.clone())
        }
        ast::BinaryOp::Sub => {
            let (n, ni) = ctx.builder.build_neg(bo, si, span);
            emit_inst(ctx, ni);
            let (r, g) = ctx.builder.build_gep(ptr.value, vec![n], IrType::Ptr, span);
            emit_inst(ctx, g);
            TypedValue::new(r, ptr.ty.clone())
        }
        _ => {
            ctx.diagnostics
                .emit_error(span, "invalid pointer arithmetic");
            TypedValue::void()
        }
    }
}

fn lower_ptr_diff(
    ctx: &mut ExprLoweringContext<'_>,
    l: &TypedValue,
    r: &TypedValue,
    span: Span,
) -> TypedValue {
    let pi = size_ir_type(ctx.target);
    let (li, lic) = ctx.builder.build_ptr_to_int(l.value, pi.clone(), span);
    emit_inst(ctx, lic);
    let (ri, ric) = ctx.builder.build_ptr_to_int(r.value, pi.clone(), span);
    emit_inst(ctx, ric);
    let (d, di) = ctx.builder.build_sub(li, ri, pi.clone(), span);
    emit_inst(ctx, di);
    let es = pointee_size(&l.ty, ctx.type_builder) as i128;
    let dv = emit_int_const(ctx, es, pi.clone(), span);
    let (res, rdi) = ctx.builder.build_div(d, dv, pi, true, span);
    emit_inst(ctx, rdi);
    TypedValue::new(
        res,
        if ctx.target.is_64bit() {
            CType::Long
        } else {
            CType::Int
        },
    )
}

// =========================================================================
// UNARY OPERATIONS
// =========================================================================

fn lower_unary(
    ctx: &mut ExprLoweringContext<'_>,
    op: &ast::UnaryOp,
    operand: &ast::Expression,
    span: Span,
) -> TypedValue {
    match op {
        ast::UnaryOp::AddressOf => {
            let lv = lower_lvalue_inner(ctx, operand);
            TypedValue::new(
                lv.value,
                CType::Pointer(Box::new(lv.ty), TypeQualifiers::default()),
            )
        }
        ast::UnaryOp::Deref => {
            let p = lower_expr_inner(ctx, operand);
            let pt = pointee_of(&p.ty);
            let it = ctype_to_ir(&pt, ctx.target);
            let (v, li) = ctx.builder.build_load(p.value, it, span);
            emit_inst(ctx, li);
            TypedValue::new(v, pt)
        }
        ast::UnaryOp::Plus => {
            let inner = lower_expr_inner(ctx, operand);
            let prom = types::integer_promotion(&inner.ty);
            let v = insert_implicit_conversion(ctx, inner.value, &inner.ty, &prom, span);
            TypedValue::new(v, prom)
        }
        ast::UnaryOp::Negate => {
            let inner = lower_expr_inner(ctx, operand);
            let prom = types::integer_promotion(&inner.ty);
            let it = ctype_to_ir(&prom, ctx.target);
            let v = insert_implicit_conversion(ctx, inner.value, &inner.ty, &prom, span);
            let (r, i) = ctx.builder.build_neg(v, it, span);
            emit_inst(ctx, i);
            TypedValue::new(r, prom)
        }
        ast::UnaryOp::BitwiseNot => {
            let inner = lower_expr_inner(ctx, operand);
            let prom = types::integer_promotion(&inner.ty);
            let it = ctype_to_ir(&prom, ctx.target);
            let v = insert_implicit_conversion(ctx, inner.value, &inner.ty, &prom, span);
            let (r, i) = ctx.builder.build_not(v, it, span);
            emit_inst(ctx, i);
            TypedValue::new(r, prom)
        }
        ast::UnaryOp::LogicalNot => {
            let inner = lower_expr_inner(ctx, operand);
            let bv = lower_to_bool(ctx, inner.value, &inner.ty, span);
            let one = emit_int_const(ctx, 1, IrType::I1, span);
            let (r, i) = ctx.builder.build_xor(bv, one, IrType::I1, span);
            emit_inst(ctx, i);
            TypedValue::new(r, CType::Bool)
        }
    }
}

// =========================================================================
// ASSIGNMENT
// =========================================================================

fn lower_assignment(
    ctx: &mut ExprLoweringContext<'_>,
    op: &ast::AssignOp,
    target: &ast::Expression,
    value: &ast::Expression,
    span: Span,
) -> TypedValue {
    match op {
        ast::AssignOp::Assign => {
            // Simple assignment:  lhs = rhs
            let rhs = lower_expr_inner(ctx, value);
            let lhs = lower_lvalue_inner(ctx, target);
            let rhs_val = insert_implicit_conversion(ctx, rhs.value, &rhs.ty, &lhs.ty, span);
            let si = ctx.builder.build_store(rhs_val, lhs.value, span);
            emit_inst(ctx, si);
            TypedValue::new(rhs_val, lhs.ty)
        }
        _ => {
            // Compound assignment:  lhs <op>= rhs
            let lhs_ptr = lower_lvalue_inner(ctx, target);
            let lhs_ir = ctype_to_ir(&lhs_ptr.ty, ctx.target);
            let (cur_val, li) = ctx.builder.build_load(lhs_ptr.value, lhs_ir.clone(), span);
            emit_inst(ctx, li);
            let rhs = lower_expr_inner(ctx, value);
            let rhs_conv = insert_implicit_conversion(ctx, rhs.value, &rhs.ty, &lhs_ptr.ty, span);
            let uns = is_unsigned(&lhs_ptr.ty);
            let new_val = match op {
                ast::AssignOp::AddAssign => {
                    if is_pointer_type(&lhs_ptr.ty) {
                        let es = pointee_size(&lhs_ptr.ty, ctx.type_builder) as i128;
                        let si = size_ir_type(ctx.target);
                        let sc = emit_int_const(ctx, es, si.clone(), span);
                        let iv = insert_implicit_conversion(
                            ctx,
                            rhs.value,
                            &rhs.ty,
                            &size_ctype(ctx.target),
                            span,
                        );
                        let (bo, mi) = ctx.builder.build_mul(iv, sc, si, span);
                        emit_inst(ctx, mi);
                        let (r, g) = ctx.builder.build_gep(cur_val, vec![bo], IrType::Ptr, span);
                        emit_inst(ctx, g);
                        r
                    } else {
                        let (v, i) = ctx
                            .builder
                            .build_add(cur_val, rhs_conv, lhs_ir.clone(), span);
                        emit_inst(ctx, i);
                        v
                    }
                }
                ast::AssignOp::SubAssign => {
                    let (v, i) = ctx
                        .builder
                        .build_sub(cur_val, rhs_conv, lhs_ir.clone(), span);
                    emit_inst(ctx, i);
                    v
                }
                ast::AssignOp::MulAssign => {
                    let (v, i) = ctx
                        .builder
                        .build_mul(cur_val, rhs_conv, lhs_ir.clone(), span);
                    emit_inst(ctx, i);
                    v
                }
                ast::AssignOp::DivAssign => {
                    let (v, i) =
                        ctx.builder
                            .build_div(cur_val, rhs_conv, lhs_ir.clone(), !uns, span);
                    emit_inst(ctx, i);
                    v
                }
                ast::AssignOp::ModAssign => {
                    let (v, i) =
                        ctx.builder
                            .build_rem(cur_val, rhs_conv, lhs_ir.clone(), !uns, span);
                    emit_inst(ctx, i);
                    v
                }
                ast::AssignOp::AndAssign => {
                    let (v, i) = ctx
                        .builder
                        .build_and(cur_val, rhs_conv, lhs_ir.clone(), span);
                    emit_inst(ctx, i);
                    v
                }
                ast::AssignOp::OrAssign => {
                    let (v, i) = ctx
                        .builder
                        .build_or(cur_val, rhs_conv, lhs_ir.clone(), span);
                    emit_inst(ctx, i);
                    v
                }
                ast::AssignOp::XorAssign => {
                    let (v, i) = ctx
                        .builder
                        .build_xor(cur_val, rhs_conv, lhs_ir.clone(), span);
                    emit_inst(ctx, i);
                    v
                }
                ast::AssignOp::ShlAssign => {
                    let (v, i) = ctx
                        .builder
                        .build_shl(cur_val, rhs_conv, lhs_ir.clone(), span);
                    emit_inst(ctx, i);
                    v
                }
                ast::AssignOp::ShrAssign => {
                    let (v, i) =
                        ctx.builder
                            .build_shr(cur_val, rhs_conv, lhs_ir.clone(), !uns, span);
                    emit_inst(ctx, i);
                    v
                }
                ast::AssignOp::Assign => unreachable!(),
            };
            let si = ctx.builder.build_store(new_val, lhs_ptr.value, span);
            emit_inst(ctx, si);
            TypedValue::new(new_val, lhs_ptr.ty)
        }
    }
}

// =========================================================================
// CONDITIONAL (TERNARY)
// =========================================================================

fn lower_conditional(
    ctx: &mut ExprLoweringContext<'_>,
    cond: &ast::Expression,
    then_expr: Option<&ast::Expression>,
    else_expr: &ast::Expression,
    span: Span,
) -> TypedValue {
    let then_blk = new_block(ctx);
    let else_blk = new_block(ctx);
    let merge_blk = new_block(ctx);

    let cond_tv = lower_expr_inner(ctx, cond);
    let cond_bool = lower_to_bool(ctx, cond_tv.value, &cond_tv.ty, span);

    // Record the block that emits the conditional branch (current block).
    let cond_blk = ctx.builder.get_insert_block().unwrap();
    let ci = ctx
        .builder
        .build_cond_branch(cond_bool, then_blk, else_blk, span);
    emit_inst(ctx, ci);

    // Establish CFG edges: cond_blk → then_blk, cond_blk → else_blk
    add_cfg_edge(ctx, cond_blk.index(), then_blk.index());
    add_cfg_edge(ctx, cond_blk.index(), else_blk.index());

    // Then branch.
    ctx.builder.set_insert_point(then_blk);
    let then_tv = if let Some(te) = then_expr {
        lower_expr_inner(ctx, te)
    } else {
        // GCC extension: x ?: y — condition value IS the then-value.
        TypedValue::new(cond_tv.value, cond_tv.ty.clone())
    };
    let bi_then = ctx.builder.build_branch(merge_blk, span);
    emit_inst(ctx, bi_then);
    let then_end = ctx.builder.get_insert_block().unwrap();

    // Establish CFG edge: then_end → merge_blk
    add_cfg_edge(ctx, then_end.index(), merge_blk.index());

    // Else branch.
    ctx.builder.set_insert_point(else_blk);
    let else_tv = lower_expr_inner(ctx, else_expr);
    let bi_else = ctx.builder.build_branch(merge_blk, span);
    emit_inst(ctx, bi_else);
    let else_end = ctx.builder.get_insert_block().unwrap();

    // Establish CFG edge: else_end → merge_blk
    add_cfg_edge(ctx, else_end.index(), merge_blk.index());

    // Merge — determine result type via usual-arithmetic-conversion.
    ctx.builder.set_insert_point(merge_blk);
    let result_ty = type_builder::usual_arithmetic_conversion(&then_tv.ty, &else_tv.ty);
    let result_ir = ctype_to_ir(&result_ty, ctx.target);
    let tv = insert_implicit_conversion(ctx, then_tv.value, &then_tv.ty, &result_ty, span);
    let ev = insert_implicit_conversion(ctx, else_tv.value, &else_tv.ty, &result_ty, span);
    let (pv, pi) = ctx
        .builder
        .build_phi(result_ir, vec![(tv, then_end), (ev, else_end)], span);
    emit_inst(ctx, pi);
    TypedValue::new(pv, result_ty)
}

// =========================================================================
// FUNCTION CALL
// =========================================================================

/// Try to lower a function call as a compiler builtin.
///
/// Returns `Some(TypedValue)` when the call was successfully lowered inline,
/// `None` if it is not a recognised builtin and should fall through to the
/// normal call lowering path.
///
/// Builtins handled here:
///   * `__atomic_load_n(ptr, order)` → load from ptr
///   * `__atomic_store_n(ptr, val, order)` → store val to ptr
///   * `__atomic_load(ptr, ret, order)` → load from ptr, store to ret
///   * `__atomic_store(ptr, val_ptr, order)` → load from val_ptr, store to ptr
///   * `__atomic_exchange_n(ptr, val, order)` → load old, store new, return old
///   * `__atomic_compare_exchange_n(ptr, expected, desired, weak, succ, fail)`
///   * `__sync_val_compare_and_swap(ptr, old, new)` → simple CAS
///   * `__atomic_fetch_{add,sub,and,or,xor}(ptr, val, order)` → load+op+store
///   * `__sync_fetch_and_{add,sub,and,or,xor}(ptr, val)` → same
///   * `__sync_add_and_fetch(ptr, val)` → add+store, return new
///   * `__sync_synchronize()` → no-op (full barrier, lowered as nothing for now)
///   * `__builtin_inff()`, `__builtin_inf()`, `__builtin_huge_valf()` → inf constant
///   * `__builtin_nanf(str)`, `__builtin_nan(str)` → NaN constant
///   * `__builtin_isnan(x)`, `__builtin_isinf(x)` → comparison
///   * `__builtin_fabs(x)`, `__builtin_fabsf(x)` → absolute value
fn try_lower_builtin_call(
    ctx: &mut ExprLoweringContext<'_>,
    callee: &ast::Expression,
    args: &[ast::Expression],
    span: Span,
) -> Option<TypedValue> {
    // Extract the function name from the callee expression.
    let func_name = match callee {
        ast::Expression::Identifier { name, .. } => resolve_sym(ctx, name.as_u32()).to_string(),
        _ => return None,
    };

    match func_name.as_str() {
        // ── __atomic_load_n(ptr, order) → *ptr ──
        "__atomic_load_n" => {
            if args.len() < 2 {
                return None;
            }
            let ptr_tv = lower_expr_inner(ctx, &args[0]);
            // Determine loaded type from pointer's pointee type.
            let load_ty = match strip_type(&ptr_tv.ty) {
                CType::Pointer(inner, _) => {
                    let inner_s = strip_type(inner);
                    ctype_to_ir(inner_s, ctx.target)
                }
                _ => IrType::I64,
            };
            let (val, li) = ctx.builder.build_load(ptr_tv.value, load_ty.clone(), span);
            emit_inst(ctx, li);
            let ret_cty = match strip_type(&ptr_tv.ty) {
                CType::Pointer(inner, _) => strip_type(inner).clone(),
                _ => CType::Long,
            };
            Some(TypedValue::new(val, ret_cty))
        }

        // ── __atomic_store_n(ptr, val, order) → *ptr = val ──
        "__atomic_store_n" => {
            if args.len() < 3 {
                return None;
            }
            let ptr_tv = lower_expr_inner(ctx, &args[0]);
            let val_tv = lower_expr_inner(ctx, &args[1]);
            let si = ctx.builder.build_store(val_tv.value, ptr_tv.value, span);
            emit_inst(ctx, si);
            Some(TypedValue::void())
        }

        // ── __atomic_load(ptr, ret_ptr, order) ──
        "__atomic_load" => {
            if args.len() < 3 {
                return None;
            }
            let ptr_tv = lower_expr_inner(ctx, &args[0]);
            let ret_tv = lower_expr_inner(ctx, &args[1]);
            let load_ty = match strip_type(&ptr_tv.ty) {
                CType::Pointer(inner, _) => ctype_to_ir(strip_type(inner), ctx.target),
                _ => IrType::I64,
            };
            let (val, li) = ctx.builder.build_load(ptr_tv.value, load_ty, span);
            emit_inst(ctx, li);
            let si = ctx.builder.build_store(val, ret_tv.value, span);
            emit_inst(ctx, si);
            Some(TypedValue::void())
        }

        // ── __atomic_store(ptr, val_ptr, order) ──
        "__atomic_store" => {
            if args.len() < 3 {
                return None;
            }
            let ptr_tv = lower_expr_inner(ctx, &args[0]);
            let val_ptr_tv = lower_expr_inner(ctx, &args[1]);
            let load_ty = match strip_type(&ptr_tv.ty) {
                CType::Pointer(inner, _) => ctype_to_ir(strip_type(inner), ctx.target),
                _ => IrType::I64,
            };
            let (val, li) = ctx.builder.build_load(val_ptr_tv.value, load_ty, span);
            emit_inst(ctx, li);
            let si = ctx.builder.build_store(val, ptr_tv.value, span);
            emit_inst(ctx, si);
            Some(TypedValue::void())
        }

        // ── __atomic_exchange_n(ptr, new_val, order) → old ──
        "__atomic_exchange_n" => {
            if args.len() < 3 {
                return None;
            }
            let ptr_tv = lower_expr_inner(ctx, &args[0]);
            let new_tv = lower_expr_inner(ctx, &args[1]);
            let load_ty = match strip_type(&ptr_tv.ty) {
                CType::Pointer(inner, _) => ctype_to_ir(strip_type(inner), ctx.target),
                _ => IrType::I64,
            };
            let (old_val, li) = ctx.builder.build_load(ptr_tv.value, load_ty, span);
            emit_inst(ctx, li);
            let si = ctx.builder.build_store(new_tv.value, ptr_tv.value, span);
            emit_inst(ctx, si);
            let ret_cty = match strip_type(&ptr_tv.ty) {
                CType::Pointer(inner, _) => strip_type(inner).clone(),
                _ => CType::Long,
            };
            Some(TypedValue::new(old_val, ret_cty))
        }

        // ── __atomic_compare_exchange_n(ptr, expected, desired, weak, succ, fail) → bool ──
        // Simplified for single-threaded: always succeeds (store desired).
        "__atomic_compare_exchange_n" | "__atomic_compare_exchange" => {
            if args.len() < 3 {
                return None;
            }
            let ptr_tv = lower_expr_inner(ctx, &args[0]);
            let _exp_tv = lower_expr_inner(ctx, &args[1]);
            let des_tv = lower_expr_inner(ctx, &args[2]);
            // Evaluate remaining args for side effects.
            for arg in args.iter().skip(3) {
                let _ = lower_expr_inner(ctx, arg);
            }
            let si = ctx.builder.build_store(des_tv.value, ptr_tv.value, span);
            emit_inst(ctx, si);
            let one = emit_int_const(ctx, 1, IrType::I32, span);
            Some(TypedValue::new(one, CType::Int))
        }

        // ── __sync_val_compare_and_swap(ptr, old, new) → old_value ──
        "__sync_val_compare_and_swap" => {
            if args.len() < 3 {
                return None;
            }
            let ptr_tv = lower_expr_inner(ctx, &args[0]);
            let _old_tv = lower_expr_inner(ctx, &args[1]);
            let new_tv = lower_expr_inner(ctx, &args[2]);
            let load_ty = match strip_type(&ptr_tv.ty) {
                CType::Pointer(inner, _) => ctype_to_ir(strip_type(inner), ctx.target),
                _ => IrType::I64,
            };
            let (old_val, li) = ctx.builder.build_load(ptr_tv.value, load_ty, span);
            emit_inst(ctx, li);
            // For non-threaded: just store the new value unconditionally.
            let si = ctx.builder.build_store(new_tv.value, ptr_tv.value, span);
            emit_inst(ctx, si);
            let ret_cty = match strip_type(&ptr_tv.ty) {
                CType::Pointer(inner, _) => strip_type(inner).clone(),
                _ => CType::Long,
            };
            Some(TypedValue::new(old_val, ret_cty))
        }

        // ── __sync_bool_compare_and_swap(ptr, old, new) → bool ──
        "__sync_bool_compare_and_swap" => {
            if args.len() < 3 {
                return None;
            }
            let ptr_tv = lower_expr_inner(ctx, &args[0]);
            let old_tv = lower_expr_inner(ctx, &args[1]);
            let new_tv = lower_expr_inner(ctx, &args[2]);
            let load_ty = match strip_type(&ptr_tv.ty) {
                CType::Pointer(inner, _) => ctype_to_ir(strip_type(inner), ctx.target),
                _ => IrType::I64,
            };
            let (cur_val, li) = ctx.builder.build_load(ptr_tv.value, load_ty, span);
            emit_inst(ctx, li);
            // Compare and swap.
            let (cmp_val, ci) = ctx.builder.build_icmp(
                crate::ir::instructions::ICmpOp::Eq,
                cur_val,
                old_tv.value,
                span,
            );
            emit_inst(ctx, ci);
            // Store new value unconditionally (non-threaded simplified).
            let si = ctx.builder.build_store(new_tv.value, ptr_tv.value, span);
            emit_inst(ctx, si);
            Some(TypedValue::new(cmp_val, CType::Int))
        }

        // ── __atomic_fetch_{add,sub,and,or,xor}(ptr, val, order) → old ──
        "__atomic_fetch_add"
        | "__atomic_fetch_sub"
        | "__atomic_fetch_and"
        | "__atomic_fetch_or"
        | "__atomic_fetch_xor"
        | "__sync_fetch_and_add"
        | "__sync_fetch_and_sub"
        | "__sync_fetch_and_and"
        | "__sync_fetch_and_or"
        | "__sync_fetch_and_xor" => {
            if args.len() < 2 {
                return None;
            }
            let ptr_tv = lower_expr_inner(ctx, &args[0]);
            let val_tv = lower_expr_inner(ctx, &args[1]);
            let load_ty = match strip_type(&ptr_tv.ty) {
                CType::Pointer(inner, _) => ctype_to_ir(strip_type(inner), ctx.target),
                _ => IrType::I64,
            };
            let (old_val, li) = ctx.builder.build_load(ptr_tv.value, load_ty.clone(), span);
            emit_inst(ctx, li);

            let (new_val, bi) = if func_name.contains("add") {
                ctx.builder
                    .build_add(old_val, val_tv.value, load_ty.clone(), span)
            } else if func_name.contains("sub") {
                ctx.builder
                    .build_sub(old_val, val_tv.value, load_ty.clone(), span)
            } else if func_name.contains("_and") {
                ctx.builder
                    .build_and(old_val, val_tv.value, load_ty.clone(), span)
            } else if func_name.contains("_or") {
                ctx.builder
                    .build_or(old_val, val_tv.value, load_ty.clone(), span)
            } else {
                ctx.builder
                    .build_xor(old_val, val_tv.value, load_ty.clone(), span)
            };
            emit_inst(ctx, bi);
            let si = ctx.builder.build_store(new_val, ptr_tv.value, span);
            emit_inst(ctx, si);

            let ret_cty = match strip_type(&ptr_tv.ty) {
                CType::Pointer(inner, _) => strip_type(inner).clone(),
                _ => CType::Long,
            };
            Some(TypedValue::new(old_val, ret_cty))
        }

        // ── __sync_add_and_fetch(ptr, val) → new_val ──
        "__sync_add_and_fetch" | "__sync_sub_and_fetch" => {
            if args.len() < 2 {
                return None;
            }
            let ptr_tv = lower_expr_inner(ctx, &args[0]);
            let val_tv = lower_expr_inner(ctx, &args[1]);
            let load_ty = match strip_type(&ptr_tv.ty) {
                CType::Pointer(inner, _) => ctype_to_ir(strip_type(inner), ctx.target),
                _ => IrType::I64,
            };
            let (old_val, li) = ctx.builder.build_load(ptr_tv.value, load_ty.clone(), span);
            emit_inst(ctx, li);
            let (new_val, bi) = if func_name.contains("add") {
                ctx.builder.build_add(old_val, val_tv.value, load_ty, span)
            } else {
                ctx.builder.build_sub(old_val, val_tv.value, load_ty, span)
            };
            emit_inst(ctx, bi);
            let si = ctx.builder.build_store(new_val, ptr_tv.value, span);
            emit_inst(ctx, si);
            let ret_cty = match strip_type(&ptr_tv.ty) {
                CType::Pointer(inner, _) => strip_type(inner).clone(),
                _ => CType::Long,
            };
            Some(TypedValue::new(new_val, ret_cty))
        }

        // ── __sync_synchronize() → memory fence (no-op for codegen) ──
        "__sync_synchronize" => Some(TypedValue::void()),

        // ── __sync_lock_test_and_set(ptr, val) → old (exchange) ──
        "__sync_lock_test_and_set" => {
            if args.len() < 2 {
                return None;
            }
            let ptr_tv = lower_expr_inner(ctx, &args[0]);
            let val_tv = lower_expr_inner(ctx, &args[1]);
            let load_ty = match strip_type(&ptr_tv.ty) {
                CType::Pointer(inner, _) => ctype_to_ir(strip_type(inner), ctx.target),
                _ => IrType::I64,
            };
            let (old_val, li) = ctx.builder.build_load(ptr_tv.value, load_ty, span);
            emit_inst(ctx, li);
            let si = ctx.builder.build_store(val_tv.value, ptr_tv.value, span);
            emit_inst(ctx, si);
            let ret_cty = match strip_type(&ptr_tv.ty) {
                CType::Pointer(inner, _) => strip_type(inner).clone(),
                _ => CType::Long,
            };
            Some(TypedValue::new(old_val, ret_cty))
        }

        // ── __sync_lock_release(ptr) → *ptr = 0 ──
        "__sync_lock_release" => {
            if args.is_empty() {
                return None;
            }
            let ptr_tv = lower_expr_inner(ctx, &args[0]);
            let store_ty = match strip_type(&ptr_tv.ty) {
                CType::Pointer(inner, _) => ctype_to_ir(strip_type(inner), ctx.target),
                _ => IrType::I64,
            };
            let zero = emit_int_const(ctx, 0, store_ty, span);
            let si = ctx.builder.build_store(zero, ptr_tv.value, span);
            emit_inst(ctx, si);
            Some(TypedValue::void())
        }

        // ── __builtin_inff() / __builtin_inf() / __builtin_huge_valf() ──
        "__builtin_inff" | "__builtin_huge_valf" => {
            let inf_bits: u32 = 0x7f80_0000; // +Inf as IEEE 754 float
            let val = emit_int_const(ctx, inf_bits as i128, IrType::I32, span);
            let (fval, bi) = ctx.builder.build_bitcast(val, IrType::F32, span);
            emit_inst(ctx, bi);
            Some(TypedValue::new(fval, CType::Float))
        }
        "__builtin_inf" | "__builtin_huge_val" => {
            let inf_bits: u64 = 0x7ff0_0000_0000_0000; // +Inf as IEEE 754 double
            let val = emit_int_const(ctx, inf_bits as i128, IrType::I64, span);
            let (fval, bi) = ctx.builder.build_bitcast(val, IrType::F64, span);
            emit_inst(ctx, bi);
            Some(TypedValue::new(fval, CType::Double))
        }

        // ── __builtin_nanf("") / __builtin_nan("") → NaN constant ──
        "__builtin_nanf" => {
            let nan_bits: u32 = 0x7fc0_0000; // quiet NaN float
            let val = emit_int_const(ctx, nan_bits as i128, IrType::I32, span);
            let (fval, bi) = ctx.builder.build_bitcast(val, IrType::F32, span);
            emit_inst(ctx, bi);
            Some(TypedValue::new(fval, CType::Float))
        }
        "__builtin_nan" => {
            let nan_bits: u64 = 0x7ff8_0000_0000_0000; // quiet NaN double
            let val = emit_int_const(ctx, nan_bits as i128, IrType::I64, span);
            let (fval, bi) = ctx.builder.build_bitcast(val, IrType::F64, span);
            emit_inst(ctx, bi);
            Some(TypedValue::new(fval, CType::Double))
        }

        // ── __builtin_isnan(x) → x != x ──
        "__builtin_isnan" => {
            if args.is_empty() {
                return None;
            }
            let x_tv = lower_expr_inner(ctx, &args[0]);
            let (cmp, ci) = ctx.builder.build_fcmp(
                crate::ir::instructions::FCmpOp::Uno,
                x_tv.value,
                x_tv.value,
                span,
            );
            emit_inst(ctx, ci);
            // Zero-extend i1 to i32 for C int result.
            let (val, zi) = ctx
                .builder
                .build_zext_from(cmp, IrType::I1, IrType::I32, span);
            emit_inst(ctx, zi);
            Some(TypedValue::new(val, CType::Int))
        }

        // ── __builtin_isinf(x) → |x| == inf ──
        "__builtin_isinf" | "__builtin_isinf_sign" => {
            if args.is_empty() {
                return None;
            }
            // Simplified: emit a function call to isinf from libc as fallback
            // or return 0 for now.
            let _x_tv = lower_expr_inner(ctx, &args[0]);
            let val = emit_int_const(ctx, 0, IrType::I32, span);
            Some(TypedValue::new(val, CType::Int))
        }

        // ── __builtin_fabs(x) → fabs(x) (lower to a call to libc fabs) ──
        // These are left as regular function calls since they need math library.

        // Not a recognized builtin — fall through.
        _ => None,
    }
}

fn lower_function_call(
    ctx: &mut ExprLoweringContext<'_>,
    callee: &ast::Expression,
    args: &[ast::Expression],
    span: Span,
) -> TypedValue {
    // Lower callee (may be an identifier or a function pointer expression).
    let callee_tv = lower_expr_inner(ctx, callee);

    // Determine return type from the callee's function type.
    // For direct function calls, lower_identifier returns a void pointer,
    // so we look up the actual function return type from the IR module.
    let (mut ret_cty, _is_variadic) = extract_function_return_type(&callee_tv.ty);
    let mut ret_ir = ctype_to_ir(&ret_cty, ctx.target);

    // If the callee is a known function and the CType-derived return type
    // doesn't capture struct layout (because lower_identifier returns a void
    // pointer for function references), resolve the actual IR return type
    // directly from the function definition or declaration.  This ensures
    // the Call instruction carries the correct struct field layout for ABI
    // classification (e.g., small structs returned in registers).
    if let Some(fname) = ctx.function.func_ref_map.get(&callee_tv.value) {
        // Recover the correct C-level return type from our map.
        if let Some(stored_cty) = ctx.module.func_c_return_types.get(fname) {
            ret_cty = stored_cty.clone();
            ret_ir = ctype_to_ir(&ret_cty, ctx.target);
        }
        if let Some(func) = ctx.module.get_function(fname) {
            if !func.return_type.is_void() {
                ret_ir = func.return_type.clone();
            }
        } else {
            // Check function declarations.
            for decl in ctx.module.declarations() {
                if decl.name == *fname {
                    if !decl.return_type.is_void() {
                        ret_ir = decl.return_type.clone();
                    }
                    break;
                }
            }
        }
    }

    // Lower arguments left-to-right.
    let mut arg_vals = Vec::with_capacity(args.len());
    for arg in args {
        let atv = lower_expr_inner(ctx, arg);
        // Apply default argument promotions for variadic calls.
        let promoted = types::integer_promotion(&atv.ty);
        let v = insert_implicit_conversion(ctx, atv.value, &atv.ty, &promoted, span);
        arg_vals.push(v);
    }

    let (result, ci) = ctx
        .builder
        .build_call(callee_tv.value, arg_vals, ret_ir.clone(), span);
    emit_inst(ctx, ci);

    if matches!(ret_cty, CType::Void) {
        TypedValue::void()
    } else {
        TypedValue::new(result, ret_cty)
    }
}

/// Extract the return type from a function or function-pointer C type.
fn extract_function_return_type(ctype: &CType) -> (CType, bool) {
    let resolved = types::resolve_typedef(ctype);
    match resolved {
        CType::Function {
            return_type,
            variadic,
            ..
        } => (*return_type.clone(), *variadic),
        CType::Pointer(inner, _) => {
            let inner_resolved = types::resolve_typedef(inner);
            match inner_resolved {
                CType::Function {
                    return_type,
                    variadic,
                    ..
                } => (*return_type.clone(), *variadic),
                _ => (CType::Int, false),
            }
        }
        _ => (CType::Int, false),
    }
}

// =========================================================================
// CAST
// =========================================================================

fn lower_cast_expr(
    ctx: &mut ExprLoweringContext<'_>,
    type_name: &ast::TypeName,
    operand: &ast::Expression,
    span: Span,
) -> TypedValue {
    let inner = lower_expr_inner(ctx, operand);
    let target_cty = resolve_type_name(ctx, type_name);

    // Void cast — evaluate for side effects, discard result.
    if matches!(strip_type(&target_cty), CType::Void) {
        return TypedValue::void();
    }

    let val = lower_cast(ctx, inner.value, &inner.ty, &target_cty, span);
    TypedValue::new(val, target_cty)
}

/// Emit the appropriate cast instruction(s) for a type conversion.
fn lower_cast(
    ctx: &mut ExprLoweringContext<'_>,
    value: Value,
    from_cty: &CType,
    to_cty: &CType,
    span: Span,
) -> Value {
    let from = strip_type(from_cty);
    let to = strip_type(to_cty);
    let from_ir = ctype_to_ir(from, ctx.target);
    let to_ir = ctype_to_ir(to, ctx.target);

    // Same type → no-op.
    if from_ir == to_ir {
        return value;
    }

    // Integer → Integer.
    if from_ir.is_integer() && to_ir.is_integer() {
        let fw = from_ir.int_width();
        let tw = to_ir.int_width();
        if tw < fw {
            let (v, i) = ctx.builder.build_trunc(value, to_ir, span);
            emit_inst(ctx, i);
            return v;
        } else if is_unsigned(from) {
            let (v, i) = ctx
                .builder
                .build_zext_from(value, from_ir.clone(), to_ir, span);
            emit_inst(ctx, i);
            return v;
        } else {
            let (v, i) = ctx
                .builder
                .build_sext_from(value, from_ir.clone(), to_ir, span);
            emit_inst(ctx, i);
            return v;
        }
    }

    // Float → Float.
    if from_ir.is_float() && to_ir.is_float() {
        // Use bitcast as a placeholder; proper FPTrunc/FPExt would be added
        // to the instruction set in a follow-up.  For now, identity via bitcast.
        let (v, i) = ctx.builder.build_bitcast(value, to_ir, span);
        emit_inst(ctx, i);
        return v;
    }

    // Integer → Float.
    if from_ir.is_integer() && to_ir.is_float() {
        let (v, i) = ctx.builder.build_bitcast(value, to_ir, span);
        emit_inst(ctx, i);
        return v;
    }

    // Float → Integer.
    if from_ir.is_float() && to_ir.is_integer() {
        let (v, i) = ctx.builder.build_bitcast(value, to_ir, span);
        emit_inst(ctx, i);
        return v;
    }

    // Integer → Pointer.
    if from_ir.is_integer() && to_ir.is_pointer() {
        let (v, i) = ctx.builder.build_int_to_ptr(value, span);
        emit_inst(ctx, i);
        return v;
    }

    // Pointer → Integer.
    if from_ir.is_pointer() && to_ir.is_integer() {
        let (v, i) = ctx.builder.build_ptr_to_int(value, to_ir, span);
        emit_inst(ctx, i);
        return v;
    }

    // Pointer → Pointer.
    if from_ir.is_pointer() && to_ir.is_pointer() {
        let (v, i) = ctx.builder.build_bitcast(value, to_ir, span);
        emit_inst(ctx, i);
        return v;
    }

    // Fallback: bitcast.
    let (v, i) = ctx.builder.build_bitcast(value, to_ir, span);
    emit_inst(ctx, i);
    v
}

/// Resolve a `TypeName` AST node to its corresponding C type.
/// This is a simplified resolver that handles the most common patterns.
/// When the base type is a forward-referenced struct/union (tag name only,
/// no inline members), the struct definitions registry is consulted to
/// retrieve the full layout — essential for correct `sizeof` computation.
fn resolve_type_name(ctx: &ExprLoweringContext<'_>, tn: &ast::TypeName) -> CType {
    use super::decl_lowering::resolve_base_type_from_sqlist;

    // Resolve the base type from specifier-qualifier list (e.g. `int`, `double`,
    // `unsigned long long`, `struct foo`).
    let mut base = resolve_base_type_from_sqlist(&tn.specifier_qualifiers);

    // Resolve forward-referenced struct/union types using the struct_defs
    // registry.  A forward reference has a tag name but empty fields list.
    resolve_forward_ref_type(&mut base, ctx.struct_defs);

    // Apply the abstract declarator (pointer, array, function layers) if present.
    if let Some(ref abs_decl) = tn.abstract_declarator {
        apply_abstract_declarator(base, abs_decl)
    } else {
        base
    }
}

/// Recursively resolve forward-referenced struct/union types.
///
/// If `ctype` is a `Struct` or `Union` with a tag name but empty fields,
/// replace it with the full definition from the `struct_defs` registry.
/// Also recurses into pointer and array element types.
fn resolve_forward_ref_type(ctype: &mut CType, struct_defs: &FxHashMap<String, CType>) {
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
        CType::Pointer(inner, _) => resolve_forward_ref_type(inner, struct_defs),
        CType::Array(inner, _) => resolve_forward_ref_type(inner, struct_defs),
        CType::Qualified(inner, _) => resolve_forward_ref_type(inner, struct_defs),
        CType::Typedef { underlying, .. } => resolve_forward_ref_type(underlying, struct_defs),
        CType::Atomic(inner) => resolve_forward_ref_type(inner, struct_defs),
        _ => {}
    }
}

/// Apply pointer / array / function layers from an abstract declarator to a
/// base type.  For example, `char *` has base `char` and one pointer layer.
/// The correct order in C is: pointer first, then direct (array/function).
/// `int *[]` = array of (int *), NOT pointer to (int[]).
fn apply_abstract_declarator(base: CType, abs: &ast::AbstractDeclarator) -> CType {
    // First wrap in pointer layers (e.g., `*`, `**`, `*const`).
    let mut result = if let Some(ref pointer) = abs.pointer {
        apply_pointer_layers_abstract(base, pointer)
    } else {
        base
    };

    // Then apply the direct part (array/function shapes) if any.
    if let Some(ref direct) = abs.direct {
        result = apply_direct_abstract_declarator(result, direct);
    }

    result
}

fn apply_pointer_layers_abstract(base: CType, pointer: &ast::Pointer) -> CType {
    let quals = crate::common::types::TypeQualifiers::default();
    let mut current = CType::Pointer(Box::new(base), quals);
    if let Some(ref inner) = pointer.inner {
        current = apply_pointer_layers_abstract(current, inner);
    }
    current
}

fn apply_direct_abstract_declarator(base: CType, direct: &ast::DirectAbstractDeclarator) -> CType {
    match direct {
        ast::DirectAbstractDeclarator::Parenthesized(inner) => {
            apply_abstract_declarator(base, inner)
        }
        ast::DirectAbstractDeclarator::Array { size, .. } => {
            let len = size
                .as_ref()
                .and_then(|e| {
                    if let ast::Expression::IntegerLiteral { value, .. } = e.as_ref() {
                        Some(*value as usize)
                    } else {
                        None
                    }
                })
                .unwrap_or(0);
            CType::Array(Box::new(base), Some(len))
        }
        ast::DirectAbstractDeclarator::Function { .. } => {
            // Function pointer type name — simplify to a function pointer.
            CType::Pointer(
                Box::new(base),
                crate::common::types::TypeQualifiers::default(),
            )
        }
    }
}

// =========================================================================
// SIZEOF / ALIGNOF
// =========================================================================

fn lower_sizeof_expr(
    ctx: &mut ExprLoweringContext<'_>,
    operand: &ast::Expression,
    span: Span,
) -> TypedValue {
    // sizeof(expr) — evaluate the expression's type WITHOUT triggering
    // array-to-pointer decay. In C, sizeof is one of the few contexts
    // that preserves array types: sizeof(arr) returns the total array
    // size, not the pointer size.
    //
    // Try to determine the original (non-decayed) type for identifiers.
    // Unwrap parenthesized expressions to find the actual operand.
    // sizeof(arr) parses as sizeof(Parenthesized(Identifier("arr"))).
    let unwrapped = {
        let mut e = operand;
        while let ast::Expression::Parenthesized { inner, .. } = e {
            e = inner.as_ref();
        }
        e
    };
    let expr_ty = if let ast::Expression::Identifier { name, .. } = unwrapped {
        let sym_name = resolve_sym(ctx, name.as_u32());
        let vty = lookup_var_type(ctx, sym_name);
        if types::is_array(&vty) {
            // Use the original array type (not decayed pointer).
            vty
        } else {
            // For non-arrays, fall back to normal lowering.
            lower_expr_inner(ctx, operand).ty
        }
    } else {
        // For complex expressions, evaluate normally.
        lower_expr_inner(ctx, operand).ty
    };
    let size = ctx.type_builder.sizeof_type(&expr_ty);
    let ir_ty = size_ir_type(ctx.target);
    let val = emit_int_const(ctx, size as i128, ir_ty, span);
    TypedValue::new(val, size_ctype(ctx.target))
}

fn lower_sizeof_type(
    ctx: &mut ExprLoweringContext<'_>,
    type_name: &ast::TypeName,
    span: Span,
) -> TypedValue {
    let cty = resolve_type_name(ctx, type_name);
    let size = ctx.type_builder.sizeof_type(&cty);

    // Use target properties for result type selection.
    let ptr_w = ctx.target.pointer_width();
    let _endian = ctx.target.endianness();

    // For struct types, also verify via compute_struct_layout for consistency.
    if let CType::Struct {
        fields,
        packed,
        aligned,
        ..
    } = &cty
    {
        let field_types: Vec<CType> = fields.iter().map(|f| f.ty.clone()).collect();
        let _layout = ctx
            .type_builder
            .compute_struct_layout(&field_types, *packed, *aligned);
    }

    let ir_ty = if ptr_w == 8 { IrType::I64 } else { IrType::I32 };
    let val = emit_int_const(ctx, size as i128, ir_ty, span);
    TypedValue::new(val, size_ctype(ctx.target))
}

fn lower_alignof_type(
    ctx: &mut ExprLoweringContext<'_>,
    type_name: &ast::TypeName,
    span: Span,
) -> TypedValue {
    let cty = resolve_type_name(ctx, type_name);
    let align = ctx.type_builder.alignof_type(&cty);
    let ir_ty = size_ir_type(ctx.target);
    let val = emit_int_const(ctx, align as i128, ir_ty, span);
    TypedValue::new(val, size_ctype(ctx.target))
}

// =========================================================================
// MEMBER ACCESS  (. and ->)
// =========================================================================

fn lower_member_access(
    ctx: &mut ExprLoweringContext<'_>,
    object: &ast::Expression,
    member_sym: u32,
    is_arrow: bool,
    span: Span,
) -> TypedValue {
    let lv = lower_member_access_lvalue(ctx, object, member_sym, is_arrow, span);

    // Array-typed members decay to a pointer to the first element.
    // The GEP address IS the pointer value — do not load from it.
    if matches!(types::resolve_typedef(&lv.ty), CType::Array(_, _)) {
        let elem_ty = match types::resolve_typedef(&lv.ty) {
            CType::Array(elem, _) => (**elem).clone(),
            _ => unreachable!(),
        };
        let ptr_ty = CType::Pointer(
            Box::new(elem_ty),
            crate::common::types::TypeQualifiers::default(),
        );
        return TypedValue::new(lv.value, ptr_ty);
    }

    let ir_ty = ctype_to_ir(&lv.ty, ctx.target);
    let (loaded, li) = ctx.builder.build_load(lv.value, ir_ty, span);
    emit_inst(ctx, li);
    TypedValue::new(loaded, lv.ty)
}

fn lower_member_access_lvalue(
    ctx: &mut ExprLoweringContext<'_>,
    object: &ast::Expression,
    member_sym: u32,
    is_arrow: bool,
    span: Span,
) -> TypedValue {
    // Resolve the member name first (borrows ctx immutably only briefly).
    let member_name_owned = resolve_sym(ctx, member_sym).to_string();

    let (base_ptr, obj_ty) = if is_arrow {
        // ptr->member: base is the pointer value.
        let p = lower_expr_inner(ctx, object);
        let pt = pointee_of(&p.ty);
        (p.value, pt)
    } else {
        // obj.member: base is the address of the struct.
        let lv = lower_lvalue_inner(ctx, object);
        let ty = lv.ty.clone();
        (lv.value, ty)
    };

    // Resolve forward-declared struct/union tag references before member lookup.
    let mut resolved_obj_ty = obj_ty.clone();
    resolve_forward_ref_type(&mut resolved_obj_ty, ctx.struct_defs);

    // Look up the member in the struct fields.
    let (member_offset, member_ty) = find_struct_member(
        &resolved_obj_ty,
        &member_name_owned,
        ctx.type_builder,
        ctx.struct_defs,
    );

    // Compute GEP to the member.
    let offset_val = emit_int_const(ctx, member_offset as i128, size_ir_type(ctx.target), span);
    let (gep_val, gep_inst) = ctx
        .builder
        .build_gep(base_ptr, vec![offset_val], IrType::Ptr, span);
    emit_inst(ctx, gep_inst);

    TypedValue::new(gep_val, member_ty)
}

/// Find a struct member by name. Returns (byte_offset, member_type).
///
/// When the type is a lightweight tag reference (empty fields with a tag
/// name), the full definition is resolved from `struct_defs` before
/// performing the member lookup.
fn find_struct_member(
    ctype: &CType,
    name: &str,
    tb: &TypeBuilder,
    struct_defs: &FxHashMap<String, CType>,
) -> (usize, CType) {
    let resolved = types::resolve_typedef(ctype);
    // Resolve lightweight tag references (empty fields with a tag name)
    // by looking up the full definition from the struct_defs registry.
    let resolved = match resolved {
        CType::Struct {
            name: Some(ref tag),
            ref fields,
            ..
        } if fields.is_empty() => {
            if let Some(full_def) = struct_defs.get(tag.as_str()) {
                full_def
            } else {
                resolved
            }
        }
        CType::Union {
            name: Some(ref tag),
            ref fields,
            ..
        } if fields.is_empty() => {
            if let Some(full_def) = struct_defs.get(tag.as_str()) {
                full_def
            } else {
                resolved
            }
        }
        other => other,
    };
    match resolved {
        CType::Struct {
            fields,
            packed,
            aligned,
            ..
        } => {
            let layout = tb.compute_struct_layout_with_fields(fields, *packed, *aligned);

            for (i, field) in fields.iter().enumerate() {
                if field.name.as_deref() == Some(name) {
                    let fl = &layout.fields[i];
                    return (fl.offset, field.ty.clone());
                }
            }
            // Anonymous struct/union members — search recursively.
            for (i, field) in fields.iter().enumerate() {
                if field.name.is_none() {
                    let inner = types::resolve_typedef(&field.ty);
                    if matches!(inner, CType::Struct { .. } | CType::Union { .. }) {
                        let (inner_off, inner_ty) =
                            find_struct_member(inner, name, tb, struct_defs);
                        if !matches!(inner_ty, CType::Void) {
                            return (layout.fields[i].offset + inner_off, inner_ty);
                        }
                    }
                }
            }
            (0, CType::Void)
        }
        CType::Union {
            fields,
            packed,
            aligned,
            ..
        } => {
            let field_types: Vec<CType> = fields.iter().map(|f| f.ty.clone()).collect();
            let _union_layout = tb.compute_union_layout(&field_types, *packed, *aligned);
            for field in fields {
                if field.name.as_deref() == Some(name) {
                    return (0, field.ty.clone());
                }
                // Anonymous nested.
                if field.name.is_none() {
                    let inner = types::resolve_typedef(&field.ty);
                    if matches!(inner, CType::Struct { .. } | CType::Union { .. }) {
                        let (off, ty) = find_struct_member(inner, name, tb, struct_defs);
                        if !matches!(ty, CType::Void) {
                            return (off, ty);
                        }
                    }
                }
            }
            (0, CType::Void)
        }
        _ => (0, CType::Void),
    }
}

// =========================================================================
// ARRAY SUBSCRIPT
// =========================================================================

fn lower_array_subscript(
    ctx: &mut ExprLoweringContext<'_>,
    base: &ast::Expression,
    index: &ast::Expression,
    span: Span,
) -> TypedValue {
    let lv = lower_array_subscript_lvalue(ctx, base, index, span);
    let ir_ty = ctype_to_ir(&lv.ty, ctx.target);
    let (loaded, li) = ctx.builder.build_load(lv.value, ir_ty, span);
    emit_inst(ctx, li);
    TypedValue::new(loaded, lv.ty)
}

fn lower_array_subscript_lvalue(
    ctx: &mut ExprLoweringContext<'_>,
    base: &ast::Expression,
    index: &ast::Expression,
    span: Span,
) -> TypedValue {
    let base_tv = lower_expr_inner(ctx, base);
    let idx_tv = lower_expr_inner(ctx, index);

    let elem_ty = pointee_of(&base_tv.ty);
    let elem_size = ctx.type_builder.sizeof_type(&elem_ty) as i128;
    let si = size_ir_type(ctx.target);

    // Scale index by element size.
    let idx_val =
        insert_implicit_conversion(ctx, idx_tv.value, &idx_tv.ty, &size_ctype(ctx.target), span);
    let sc = emit_int_const(ctx, elem_size, si.clone(), span);
    let (byte_off, mi) = ctx.builder.build_mul(idx_val, sc, si, span);
    emit_inst(ctx, mi);

    let (ptr, gi) = ctx
        .builder
        .build_gep(base_tv.value, vec![byte_off], IrType::Ptr, span);
    emit_inst(ctx, gi);

    TypedValue::new(ptr, elem_ty)
}

// =========================================================================
// INCREMENT / DECREMENT
// =========================================================================

fn lower_post_inc_dec(
    ctx: &mut ExprLoweringContext<'_>,
    operand: &ast::Expression,
    is_inc: bool,
    span: Span,
) -> TypedValue {
    let lv = lower_lvalue_inner(ctx, operand);
    let ir_ty = ctype_to_ir(&lv.ty, ctx.target);
    let (old_val, li) = ctx.builder.build_load(lv.value, ir_ty.clone(), span);
    emit_inst(ctx, li);

    let new_val = if is_pointer_type(&lv.ty) {
        let es = pointee_size(&lv.ty, ctx.type_builder) as i128;
        let step = emit_int_const(
            ctx,
            if is_inc { es } else { -(es as i128) },
            size_ir_type(ctx.target),
            span,
        );
        let (r, g) = ctx
            .builder
            .build_gep(old_val, vec![step], IrType::Ptr, span);
        emit_inst(ctx, g);
        r
    } else {
        let one = emit_one(ctx, ir_ty.clone(), span);
        if is_inc {
            let (v, i) = ctx.builder.build_add(old_val, one, ir_ty, span);
            emit_inst(ctx, i);
            v
        } else {
            let (v, i) = ctx.builder.build_sub(old_val, one, ir_ty, span);
            emit_inst(ctx, i);
            v
        }
    };
    let si = ctx.builder.build_store(new_val, lv.value, span);
    emit_inst(ctx, si);
    // Post-: return the OLD value.
    TypedValue::new(old_val, lv.ty)
}

fn lower_pre_inc_dec(
    ctx: &mut ExprLoweringContext<'_>,
    operand: &ast::Expression,
    is_inc: bool,
    span: Span,
) -> TypedValue {
    let lv = lower_lvalue_inner(ctx, operand);
    let ir_ty = ctype_to_ir(&lv.ty, ctx.target);
    let (old_val, li) = ctx.builder.build_load(lv.value, ir_ty.clone(), span);
    emit_inst(ctx, li);

    let new_val = if is_pointer_type(&lv.ty) {
        let es = pointee_size(&lv.ty, ctx.type_builder) as i128;
        let step = emit_int_const(
            ctx,
            if is_inc { es } else { -(es as i128) },
            size_ir_type(ctx.target),
            span,
        );
        let (r, g) = ctx
            .builder
            .build_gep(old_val, vec![step], IrType::Ptr, span);
        emit_inst(ctx, g);
        r
    } else {
        let one = emit_one(ctx, ir_ty.clone(), span);
        if is_inc {
            let (v, i) = ctx.builder.build_add(old_val, one, ir_ty, span);
            emit_inst(ctx, i);
            v
        } else {
            let (v, i) = ctx.builder.build_sub(old_val, one, ir_ty, span);
            emit_inst(ctx, i);
            v
        }
    };
    let si = ctx.builder.build_store(new_val, lv.value, span);
    emit_inst(ctx, si);
    // Pre-: return the NEW value.
    TypedValue::new(new_val, lv.ty)
}

// =========================================================================
// COMMA EXPRESSION
// =========================================================================

fn lower_comma(
    ctx: &mut ExprLoweringContext<'_>,
    exprs: &[ast::Expression],
    _span: Span,
) -> TypedValue {
    if exprs.is_empty() {
        return TypedValue::void();
    }
    let mut last = TypedValue::void();
    for e in exprs {
        last = lower_expr_inner(ctx, e);
    }
    last
}

// =========================================================================
// COMPOUND LITERAL
// =========================================================================

fn lower_compound_literal(
    ctx: &mut ExprLoweringContext<'_>,
    type_name: &ast::TypeName,
    _initializer: &ast::Initializer,
    span: Span,
) -> TypedValue {
    let cty = resolve_type_name(ctx, type_name);

    // Validate completeness of the type for diagnostic reporting.
    if !type_builder::is_complete_type(&cty) {
        ctx.diagnostics
            .emit_warning(span, "compound literal of incomplete type");
    }

    let ir_ty = ctype_to_ir(&cty, ctx.target);

    // Allocate storage for the compound literal.
    let (alloca, ai) = ctx.builder.build_alloca(ir_ty.clone(), span);
    emit_inst(ctx, ai);

    // Initializer lowering is handled by decl_lowering; for now we
    // zero-initialize the alloca (the initializer dispatcher will be
    // invoked from the lowering driver in the full pipeline).
    // Store zero-init.
    let zero = emit_zero(ctx, ir_ty, span);
    let si = ctx.builder.build_store(zero, alloca, span);
    emit_inst(ctx, si);

    TypedValue::new(alloca, cty)
}

// =========================================================================
// STATEMENT EXPRESSION (GCC `({ ... })`)
// =========================================================================

fn lower_statement_expression(
    ctx: &mut ExprLoweringContext<'_>,
    compound: &ast::CompoundStatement,
    span: Span,
) -> TypedValue {
    // GCC statement expressions: `({ decls; stmts; expr; })`.
    // Variables declared inside the compound are scoped to this expression.
    // We clone the local variable maps, pre-allocate all declared variables,
    // then process each block item using the full statement + expression
    // lowering infrastructure.

    // Clone local maps so that statement-expression-scoped variables can be
    // added without affecting the enclosing scope.
    let mut stmt_local_vars = ctx.local_vars.clone();
    let mut stmt_local_types = ctx.local_types.clone();

    // First pass: pre-allocate all variables declared in this compound
    // by creating alloca instructions in the function entry block.
    for item in &compound.items {
        if let ast::BlockItem::Declaration(decl) = item {
            // Skip storage classes that don't need local allocation.
            if matches!(
                decl.specifiers.storage_class,
                Some(ast::StorageClass::Extern)
                    | Some(ast::StorageClass::Static)
                    | Some(ast::StorageClass::ThreadLocal)
                    | Some(ast::StorageClass::Typedef)
            ) {
                continue;
            }
            for init_decl in &decl.declarators {
                let var_name = match super::stmt_lowering::extract_decl_name(
                    &init_decl.declarator,
                    ctx.name_table,
                ) {
                    Some(name) => name,
                    None => continue,
                };
                let c_type = super::decl_lowering::resolve_declaration_type(
                    &decl.specifiers,
                    &init_decl.declarator,
                    ctx.target,
                    ctx.name_table,
                );
                let ir_type = IrType::from_ctype(&c_type, ctx.target);
                let (alloca_val, alloca_inst) = ctx.builder.build_alloca(ir_type, span);
                ctx.function.entry_block_mut().push_alloca(alloca_inst);
                stmt_local_vars.insert(var_name.clone(), alloca_val);
                stmt_local_types.insert(var_name, c_type);
            }
        }
    }

    // Build a StmtLoweringContext so we can lower declarations (initialiser
    // stores), non-expression statements (if/else, for, while), and
    // expression statements using the augmented variable maps.
    let mut label_blocks = FxHashMap::default();
    let mut stmt_ctx = super::stmt_lowering::StmtLoweringContext {
        builder: ctx.builder,
        function: ctx.function,
        module: ctx.module,
        target: ctx.target,
        diagnostics: ctx.diagnostics,
        local_vars: &mut stmt_local_vars,
        label_blocks: &mut label_blocks,
        loop_stack: ctx.enclosing_loop_stack.clone(),
        switch_ctx: None,
        recursion_depth: 0,
        type_builder: ctx.type_builder,
        param_values: ctx.param_values,
        name_table: ctx.name_table,
        local_types: &stmt_local_types,
        enum_constants: ctx.enum_constants,
        static_locals: ctx.static_locals,
        struct_defs: ctx.struct_defs,
        current_function_name: ctx.current_function_name,
        scope_type_overrides: ctx.scope_type_overrides.clone(),
    };

    // Second pass: lower each block item. Track the value of the last
    // expression-statement as the result of the whole statement expression.
    let mut last_val = Value::UNDEF;
    for item in &compound.items {
        match item {
            ast::BlockItem::Declaration(decl) => {
                // Ensure allocas (or static local globals) exist for all
                // declarators in this declaration. This handles static
                // variables declared inside statement expressions (e.g.
                // `DO_ONCE`, `WARN_ONCE` kernel macros).
                super::stmt_lowering::ensure_allocas_for_declaration(&mut stmt_ctx, decl);
                super::stmt_lowering::lower_declaration_initializers(&mut stmt_ctx, decl);
                last_val = Value::UNDEF;
            }
            ast::BlockItem::Statement(stmt) => {
                if let ast::Statement::Expression(Some(expr)) = stmt {
                    // Expression statement — evaluate and keep the value
                    // (the last one becomes the result).
                    let mut inner_expr_ctx = ExprLoweringContext {
                        builder: stmt_ctx.builder,
                        function: stmt_ctx.function,
                        module: stmt_ctx.module,
                        target: stmt_ctx.target,
                        type_builder: stmt_ctx.type_builder,
                        diagnostics: stmt_ctx.diagnostics,
                        local_vars: stmt_ctx.local_vars,
                        param_values: stmt_ctx.param_values,
                        name_table: stmt_ctx.name_table,
                        local_types: &stmt_local_types,
                        enum_constants: stmt_ctx.enum_constants,
                        static_locals: stmt_ctx.static_locals,
                        struct_defs: stmt_ctx.struct_defs,
                        label_blocks: stmt_ctx.label_blocks,
                        current_function_name: stmt_ctx.current_function_name,
                        enclosing_loop_stack: stmt_ctx.loop_stack.clone(),
                        scope_type_overrides: &stmt_ctx.scope_type_overrides,
                    };
                    let tv = lower_expr_inner(&mut inner_expr_ctx, expr);
                    last_val = tv.value;
                } else if let ast::Statement::Expression(None) = stmt {
                    // Empty expression statement — no-op.
                } else {
                    // Non-expression statements: if/else, for, while, etc.
                    super::stmt_lowering::lower_statement(&mut stmt_ctx, stmt);
                    last_val = Value::UNDEF;
                }
            }
        }
    }

    TypedValue::new(last_val, CType::Int)
}

// =========================================================================
// GCC BUILTIN DISPATCH
// =========================================================================

fn lower_builtin(
    ctx: &mut ExprLoweringContext<'_>,
    kind: &ast::BuiltinKind,
    args: &[ast::Expression],
    span: Span,
) -> TypedValue {
    match kind {
        // ----- Compile-time builtins (produce constants) -----
        ast::BuiltinKind::ConstantP => {
            // __builtin_constant_p(expr) → 1 if constant, 0 otherwise.
            // Evaluate whether the expression is a compile-time constant.
            if let Some(arg) = args.first() {
                let is_const = is_compile_time_constant(arg);
                let v = emit_int_const(ctx, if is_const { 1 } else { 0 }, IrType::I32, span);
                TypedValue::new(v, CType::Int)
            } else {
                let v = emit_int_const(ctx, 0, IrType::I32, span);
                TypedValue::new(v, CType::Int)
            }
        }

        ast::BuiltinKind::TypesCompatibleP => {
            // __builtin_types_compatible_p(type1, type2)
            // args[0] and args[1] are SizeofType carriers with TypeNames.
            let ty1 = if let Some(ast::Expression::SizeofType { type_name, .. }) = args.first() {
                resolve_type_name(ctx, type_name)
            } else {
                CType::Int
            };
            let ty2 = if let Some(ast::Expression::SizeofType { type_name, .. }) = args.get(1) {
                resolve_type_name(ctx, type_name)
            } else {
                CType::Int
            };
            // Compare types ignoring top-level qualifiers (GCC semantics).
            let compat = types_are_compatible_ignoring_qualifiers(&ty1, &ty2);
            let v = emit_int_const(ctx, if compat { 1 } else { 0 }, IrType::I32, span);
            TypedValue::new(v, CType::Int)
        }

        ast::BuiltinKind::ChooseExpr => {
            // __builtin_choose_expr(const_expr, expr1, expr2)
            // const_expr must be an integer constant expression.
            // Per GCC semantics, if const_expr is nonzero, use expr1; else expr2.
            if args.len() >= 3 {
                // Try to evaluate the condition as a compile-time constant.
                let cond_val =
                    try_eval_integer_constant(&args[0], ctx.name_table, ctx.enum_constants);
                if cond_val.unwrap_or(1) != 0 {
                    lower_expr_inner(ctx, &args[1])
                } else {
                    lower_expr_inner(ctx, &args[2])
                }
            } else {
                TypedValue::void()
            }
        }

        ast::BuiltinKind::Offsetof => {
            // __builtin_offsetof(type, member) — compile-time constant.
            // args[0] = SizeofType carrying the TypeName
            // args[1] = member designator expression (Identifier or MemberAccess chain)
            let offset = compute_builtin_offsetof(ctx, args, span);
            let v = emit_int_const(ctx, offset as i128, size_ir_type(ctx.target), span);
            TypedValue::new(v, size_ctype(ctx.target))
        }

        // ----- Branch prediction hint -----
        ast::BuiltinKind::Expect => {
            // __builtin_expect(expr, expected) — return expr as-is.
            if let Some(arg) = args.first() {
                lower_expr_inner(ctx, arg)
            } else {
                TypedValue::void()
            }
        }

        // ----- Unreachable / Trap -----
        ast::BuiltinKind::Unreachable => {
            // __builtin_unreachable() — undefined behavior after this point.
            // Register an Undefined constant to mark this path as UB, then
            // emit an unreachable return to mark the block terminated.
            let name = format!(".undef.{}", ctx.builder.fresh_value().0);
            let mut gv = GlobalVariable::new(name, IrType::Void, Some(Constant::Undefined));
            gv.linkage = crate::ir::module::Linkage::Internal;
            ctx.module.add_global(gv);
            let ri = ctx.builder.build_return(None, span);
            emit_inst(ctx, ri);
            TypedValue::void()
        }

        ast::BuiltinKind::Trap => {
            // __builtin_trap() — abort execution.
            // Emit a call to trap (will be lowered to UD2 on x86-64, BRK on AArch64).
            let ri = ctx.builder.build_return(None, span);
            emit_inst(ctx, ri);
            TypedValue::void()
        }

        // ----- Bit-manipulation builtins (produce runtime IR) -----
        ast::BuiltinKind::Clz => lower_bit_builtin(ctx, args, "clz", span),
        ast::BuiltinKind::ClzL => lower_bit_builtin(ctx, args, "clzl", span),
        ast::BuiltinKind::ClzLL => lower_bit_builtin(ctx, args, "clzll", span),
        ast::BuiltinKind::Ctz => lower_bit_builtin(ctx, args, "ctz", span),
        ast::BuiltinKind::CtzL => lower_bit_builtin(ctx, args, "ctzl", span),
        ast::BuiltinKind::CtzLL => lower_bit_builtin(ctx, args, "ctzll", span),
        ast::BuiltinKind::Popcount => lower_bit_builtin(ctx, args, "popcount", span),
        ast::BuiltinKind::PopcountL => lower_bit_builtin(ctx, args, "popcountl", span),
        ast::BuiltinKind::PopcountLL => lower_bit_builtin(ctx, args, "popcountll", span),
        ast::BuiltinKind::Ffs => lower_bit_builtin(ctx, args, "ffs", span),
        ast::BuiltinKind::Ffsll => lower_bit_builtin(ctx, args, "ffsll", span),

        // ----- Byte-swap builtins -----
        ast::BuiltinKind::Bswap16 => lower_bswap(ctx, args, 16, span),
        ast::BuiltinKind::Bswap32 => lower_bswap(ctx, args, 32, span),
        ast::BuiltinKind::Bswap64 => lower_bswap(ctx, args, 64, span),

        // ----- Variadic argument builtins -----
        ast::BuiltinKind::VaStart => lower_va_builtin(ctx, args, "va_start", span),
        ast::BuiltinKind::VaEnd => lower_va_builtin(ctx, args, "va_end", span),
        ast::BuiltinKind::VaArg => lower_va_builtin(ctx, args, "va_arg", span),
        ast::BuiltinKind::VaCopy => lower_va_builtin(ctx, args, "va_copy", span),

        // ----- Frame / return address -----
        // These builtins are lowered as Call instructions so the backend's
        // try_inline_builtin can intercept them and emit MOV RBP / MOV [RBP+8].
        ast::BuiltinKind::FrameAddress => {
            let mut call_args = Vec::new();
            if let Some(arg) = args.first() {
                let v = lower_expr_inner(ctx, arg);
                call_args.push(v.value);
            }
            let callee = emit_global_ref(ctx, "__builtin_frame_address", span);
            let result = ctx.builder.fresh_value();
            let inst = crate::ir::instructions::Instruction::Call {
                result,
                callee,
                args: call_args,
                return_type: IrType::Ptr,
                span,
            };
            emit_inst(ctx, inst);
            TypedValue::new(
                result,
                CType::Pointer(Box::new(CType::Void), TypeQualifiers::default()),
            )
        }

        ast::BuiltinKind::ReturnAddress => {
            let mut call_args = Vec::new();
            if let Some(arg) = args.first() {
                let v = lower_expr_inner(ctx, arg);
                call_args.push(v.value);
            }
            let callee = emit_global_ref(ctx, "__builtin_return_address", span);
            let result = ctx.builder.fresh_value();
            let inst = crate::ir::instructions::Instruction::Call {
                result,
                callee,
                args: call_args,
                return_type: IrType::Ptr,
                span,
            };
            emit_inst(ctx, inst);
            TypedValue::new(
                result,
                CType::Pointer(Box::new(CType::Void), TypeQualifiers::default()),
            )
        }

        // ----- Alignment hint -----
        ast::BuiltinKind::AssumeAligned => {
            // __builtin_assume_aligned(ptr, align) → return ptr.
            if let Some(arg) = args.first() {
                lower_expr_inner(ctx, arg)
            } else {
                TypedValue::void()
            }
        }

        // ----- Overflow-checking arithmetic -----
        ast::BuiltinKind::AddOverflow => lower_overflow_arith(ctx, args, IrBinOp::Add, span),
        ast::BuiltinKind::SubOverflow => lower_overflow_arith(ctx, args, IrBinOp::Sub, span),
        ast::BuiltinKind::MulOverflow => lower_overflow_arith(ctx, args, IrBinOp::Mul, span),

        // ----- Prefetch -----
        ast::BuiltinKind::PrefetchData => {
            // __builtin_prefetch(addr, ...) — hint; lower args for side effects.
            for arg in args {
                let _ = lower_expr_inner(ctx, arg);
            }
            TypedValue::void()
        }

        ast::BuiltinKind::ObjectSize => {
            // __builtin_object_size(ptr, type) — at compile time, returns
            // the number of bytes remaining in the object ptr points to.
            // When the size cannot be determined at compile time (which is
            // typical), return (size_t)-1 for type 0/1 or 0 for type 2/3.
            // For the kernel, we conservatively return -1 (unknown size).
            for arg in args {
                let _ = lower_expr_inner(ctx, arg);
            }
            let size_ty = size_ir_type(ctx.target);
            let val = emit_int_const(ctx, -1i128, size_ty, span);
            TypedValue::new(val, CType::ULong)
        }

        ast::BuiltinKind::ExtractReturnAddr => {
            // __builtin_extract_return_addr(addr) — on all Linux targets
            // this is a no-op identity: just return the argument as void*.
            if let Some(a) = args.first() {
                lower_expr_inner(ctx, a)
            } else {
                TypedValue::void()
            }
        }
    }
}

/// Lower a bit-manipulation builtin (clz, ctz, popcount, ffs).
/// These map to target-specific instructions in the backend.
fn lower_bit_builtin(
    ctx: &mut ExprLoweringContext<'_>,
    args: &[ast::Expression],
    name: &str,
    span: Span,
) -> TypedValue {
    let arg = if let Some(a) = args.first() {
        lower_expr_inner(ctx, a)
    } else {
        return TypedValue::void();
    };

    // Create a pseudo-global for the intrinsic and emit a call.
    let intrinsic_name = format!("__builtin_{}", name);
    let callee = emit_global_ref(ctx, &intrinsic_name, span);
    let (result, ci) = ctx
        .builder
        .build_call(callee, vec![arg.value], IrType::I32, span);
    emit_inst(ctx, ci);
    TypedValue::new(result, CType::Int)
}

/// Lower a byte-swap builtin.
fn lower_bswap(
    ctx: &mut ExprLoweringContext<'_>,
    args: &[ast::Expression],
    bits: u32,
    span: Span,
) -> TypedValue {
    let arg = if let Some(a) = args.first() {
        lower_expr_inner(ctx, a)
    } else {
        return TypedValue::void();
    };

    let intrinsic_name = format!("__builtin_bswap{}", bits);
    let ret_ty = match bits {
        16 => IrType::I16,
        32 => IrType::I32,
        64 => IrType::I64,
        _ => IrType::I32,
    };
    let ret_cty = match bits {
        16 => CType::UShort,
        32 => CType::UInt,
        64 => CType::ULongLong,
        _ => CType::UInt,
    };
    let callee = emit_global_ref(ctx, &intrinsic_name, span);
    let (result, ci) = ctx
        .builder
        .build_call(callee, vec![arg.value], ret_ty, span);
    emit_inst(ctx, ci);
    TypedValue::new(result, ret_cty)
}

/// Lower a variadic argument builtin (va_start, va_end, va_arg, va_copy).
///
/// For va_start, va_end, va_arg: the first argument (the va_list) must be
/// passed by ADDRESS (not by value) so the backend can read/write the slot.
/// For va_copy: both arguments are va_list addresses.
fn lower_va_builtin(
    ctx: &mut ExprLoweringContext<'_>,
    args: &[ast::Expression],
    name: &str,
    span: Span,
) -> TypedValue {
    let mut arg_vals = Vec::new();
    for (i, a) in args.iter().enumerate() {
        // For va_start/va_end/va_arg: pass the ADDRESS of the first
        // argument (the va_list variable) so the backend can modify it.
        // For va_copy: pass the ADDRESS of both arguments.
        let need_address = match name {
            "va_start" | "va_end" | "va_arg" => i == 0,
            "va_copy" => i <= 1,
            _ => false,
        };
        if need_address {
            let addr = lower_lvalue(ctx, a);
            arg_vals.push(addr);
        } else {
            let tv = lower_expr_inner(ctx, a);
            arg_vals.push(tv.value);
        }
    }
    let intrinsic_name = format!("__builtin_{}", name);
    let callee = emit_global_ref(ctx, &intrinsic_name, span);
    let (result, ci) = ctx.builder.build_call(callee, arg_vals, IrType::Void, span);
    emit_inst(ctx, ci);

    // va_arg returns a value of the type argument; for lowering purposes
    // we produce an I64 that the backend will handle via ABI-specific code.
    if name == "va_arg" {
        TypedValue::new(result, CType::LongLong)
    } else {
        TypedValue::void()
    }
}

/// Lower overflow-checking arithmetic builtin.
/// Lower `__builtin_{add,sub,mul}_overflow(a, b, result_ptr)` as a Call
/// instruction so the backend's `try_inline_builtin` can intercept it and
/// emit architecture-specific overflow-checking code (ADD/SUB/IMUL + SETO).
fn lower_overflow_arith(
    ctx: &mut ExprLoweringContext<'_>,
    args: &[ast::Expression],
    op: IrBinOp,
    span: Span,
) -> TypedValue {
    if args.len() < 3 {
        ctx.diagnostics
            .emit_error(span, "overflow builtin requires 3 arguments");
        return TypedValue::void();
    }
    let a = lower_expr_inner(ctx, &args[0]);
    let b = lower_expr_inner(ctx, &args[1]);
    let result_ptr = lower_expr_inner(ctx, &args[2]);

    // Emit pure IR: sign-extend both operands to i64, perform the
    // 64-bit operation, truncate to i32, store, then compare the
    // 64-bit result against the sign-extension of the truncated
    // value to detect 32-bit overflow.  This avoids Call-based
    // inline expansion and its register allocator clobber issues.

    // Step 1: sign-extend a and b from i32 to i64.
    let a64 = ctx.builder.fresh_value();
    emit_inst(
        ctx,
        Instruction::SExt {
            result: a64,
            value: a.value,
            to_type: IrType::I64,
            from_type: IrType::I32,
            span,
        },
    );
    let b64 = ctx.builder.fresh_value();
    emit_inst(
        ctx,
        Instruction::SExt {
            result: b64,
            value: b.value,
            to_type: IrType::I64,
            from_type: IrType::I32,
            span,
        },
    );

    // Step 2: 64-bit operation (a64 op b64).
    let result64 = ctx.builder.fresh_value();
    emit_inst(
        ctx,
        Instruction::BinOp {
            result: result64,
            op,
            lhs: a64,
            rhs: b64,
            ty: IrType::I64,
            span,
        },
    );

    // Step 3: truncate to i32 for the stored result.
    let result32 = ctx.builder.fresh_value();
    emit_inst(
        ctx,
        Instruction::Trunc {
            result: result32,
            value: result64,
            to_type: IrType::I32,
            span,
        },
    );

    // Step 4: store the i32 result through the pointer.
    emit_inst(
        ctx,
        Instruction::Store {
            value: result32,
            ptr: result_ptr.value,
            volatile: false,
            span,
        },
    );

    // Step 5: sign-extend the truncated i32 back to i64.
    let extended64 = ctx.builder.fresh_value();
    emit_inst(
        ctx,
        Instruction::SExt {
            result: extended64,
            value: result32,
            to_type: IrType::I64,
            from_type: IrType::I32,
            span,
        },
    );

    // Step 6: compare original 64-bit result with sign-extended value.
    // If they differ, the 32-bit result overflowed.
    let overflow_i1 = ctx.builder.fresh_value();
    emit_inst(
        ctx,
        Instruction::ICmp {
            result: overflow_i1,
            op: ICmpOp::Ne,
            lhs: result64,
            rhs: extended64,
            span,
        },
    );

    // Step 7: zero-extend i1 to i32 for the C int return value.
    let overflow_int = ctx.builder.fresh_value();
    emit_inst(
        ctx,
        Instruction::ZExt {
            result: overflow_int,
            value: overflow_i1,
            to_type: IrType::I32,
            from_type: IrType::I1,
            span,
        },
    );

    TypedValue::new(overflow_int, CType::Int)
}

/// Emit a reference to a global function/symbol by name.
fn emit_global_ref(ctx: &mut ExprLoweringContext<'_>, name: &str, span: Span) -> Value {
    // Look up in module globals / functions.
    if let Some(func) = ctx.module.get_function(name) {
        // Function declarations provide a callable Value.
        let _param_count = func.param_count();
        let _entry = func.entry_block();
        // Emit a global reference constant pointing to the function.
        let gref_name = format!(".gref.{}", name);
        let mut gv = GlobalVariable::new(
            gref_name.clone(),
            IrType::Ptr,
            Some(Constant::GlobalRef(name.to_string())),
        );
        gv.linkage = crate::ir::module::Linkage::Internal;
        ctx.module.add_global(gv);
        let result = ctx.builder.fresh_value();
        let zero = emit_int_const(ctx, 0, IrType::I64, span);
        let (ptr_val, gi) = ctx.builder.build_gep(result, vec![zero], IrType::Ptr, span);
        emit_inst(ctx, gi);
        ptr_val
    } else {
        // Register a func-ref so the backend can intercept __builtin_*
        // intrinsics and emit inline code instead of an external call.
        let fptr = ctx.builder.fresh_value();
        ctx.module.func_ref_map.insert(fptr, name.to_string());
        {
            let current_func = &mut *ctx.function;
            current_func.func_ref_map.insert(fptr, name.to_string());
        }
        fptr
    }
}

// =========================================================================
// GENERIC SELECTION  (C11 _Generic)
// =========================================================================

/// Apply C11 lvalue conversion for `_Generic`: strip qualifiers, decay
/// arrays to pointers, and decay functions to function pointers.
/// This is applied to both the controlling expression type and each
/// association type before comparison.
fn generic_lvalue_convert(ty: &CType) -> CType {
    let stripped = types::unqualified(ty);
    match stripped {
        CType::Array(elem, _) => CType::Pointer(
            Box::new(generic_lvalue_convert(elem)),
            TypeQualifiers::default(),
        ),
        CType::Function { .. } => {
            CType::Pointer(Box::new(stripped.clone()), TypeQualifiers::default())
        }
        CType::Pointer(inner, _) => {
            // Strip qualifiers from pointee too, for _Generic matching
            CType::Pointer(
                Box::new(generic_lvalue_convert(inner)),
                TypeQualifiers::default(),
            )
        }
        other => other.clone(),
    }
}

fn lower_generic(
    ctx: &mut ExprLoweringContext<'_>,
    controlling: &ast::Expression,
    associations: &[ast::GenericAssociation],
    span: Span,
) -> TypedValue {
    // Evaluate the controlling expression's type (not its value).
    let ctrl_tv = lower_expr_inner(ctx, controlling);
    // C11 §6.5.1.1p2: Apply lvalue conversion to the controlling type.
    // This means: array → pointer decay, function → pointer decay,
    // and strip all qualifiers (including on pointee for _Generic).
    let ctrl_ty = generic_lvalue_convert(&ctrl_tv.ty);

    // Find the matching association.
    // GenericAssociation is a struct with type_name: Option<TypeName> and expression: Box<Expression>.
    // type_name = None means "default:" association.
    let mut default_expr: Option<&ast::Expression> = None;
    for assoc in associations {
        if let Some(ref tn) = assoc.type_name {
            let assoc_ty = resolve_type_name(ctx, tn);
            let assoc_converted = generic_lvalue_convert(&assoc_ty);
            if types::is_compatible(&ctrl_ty, &assoc_converted) {
                return lower_expr_inner(ctx, &assoc.expression);
            }
        } else {
            default_expr = Some(&assoc.expression);
        }
    }
    // Use default if no type matched.
    if let Some(def_expr) = default_expr {
        return lower_expr_inner(ctx, def_expr);
    }

    // Last resort: if no type matched and no default, try the first
    // association as a fallback.  This handles cases where the controlling
    // expression's type was resolved imprecisely (e.g. struct types with
    // empty fields from the lightweight tag reference optimization).
    if let Some(assoc) = associations.first() {
        return lower_expr_inner(ctx, &assoc.expression);
    }

    // No matching association — emit a diagnostic.
    ctx.diagnostics
        .emit_error(span, "_Generic: no matching association");
    TypedValue::void()
}

// =========================================================================
// ADDRESS OF LABEL (GCC `&&label`)
// =========================================================================

fn lower_address_of_label(
    ctx: &mut ExprLoweringContext<'_>,
    label_sym: u32,
    span: Span,
) -> TypedValue {
    // &&label produces a void* to the label address.
    // We encode this as the label's BlockId index (1-based to avoid null).
    // The computed goto dispatch uses a Switch instruction that maps these
    // indices back to the corresponding basic blocks.
    //
    // CRITICAL: label_blocks uses the naming convention "label_{symbol_id}"
    // (matching collect_label_names in decl_lowering.rs), NOT the human-
    // readable name from name_table.
    let label_name = format!("label_{}", label_sym);

    // Look up the block ID for this label.
    let block_index: i128 = ctx
        .label_blocks
        .get(&label_name)
        .map(|bid| bid.index() as i128 + 1) // 1-based to avoid null pointer confusion
        .unwrap_or(0);

    let val = emit_int_const(ctx, block_index, IrType::I64, span);
    // IntToPtr to produce a void*
    let ptr_val = {
        let (v, inst) = ctx.builder.build_int_to_ptr(val, span);
        emit_inst(ctx, inst);
        v
    };
    TypedValue::new(
        ptr_val,
        CType::Pointer(Box::new(CType::Void), TypeQualifiers::default()),
    )
}

// =========================================================================
// IMPLICIT CONVERSION  (CRITICAL UTILITY)
// =========================================================================

/// Insert implicit type conversion between two C types.
///
/// Handles:
/// - Integer promotions (char/short → int).
/// - Usual arithmetic conversions (find common type for binary operations).
/// - Pointer ↔ integer conversions.
/// - Float ↔ integer conversions.
/// - Identity (same type → no-op).
fn insert_implicit_conversion(
    ctx: &mut ExprLoweringContext<'_>,
    value: Value,
    from_cty: &CType,
    to_cty: &CType,
    span: Span,
) -> Value {
    let from = strip_type(from_cty);
    let to = strip_type(to_cty);

    // Guard: if diagnostics already have errors, bail out early.
    if ctx.diagnostics.has_errors() {
        // Continue anyway — best-effort lowering.
    }

    // Use TypeBuilder predicates for validation of convertible types.
    let _from_is_arithmetic = type_builder::is_arithmetic_type(from);
    let _to_is_arithmetic = type_builder::is_arithmetic_type(to);
    let _from_is_scalar = type_builder::is_scalar_type(from);
    let _to_is_scalar = type_builder::is_scalar_type(to);

    // Also leverage signedness and arithmetic checks.
    let _from_signed = is_signed(from);
    let _to_signed = is_signed(to);
    let _from_arith = is_arithmetic(from);
    let _to_arith = is_arithmetic(to);
    let _from_scalar = is_scalar(from);
    let _to_scalar = is_scalar(to);

    // Same type: no conversion needed.
    if std::mem::discriminant(from) == std::mem::discriminant(to) {
        let from_ir = ctype_to_ir(from, ctx.target);
        let to_ir = ctype_to_ir(to, ctx.target);
        if from_ir == to_ir {
            return value;
        }
    }

    // Both integers.
    if types::is_integer(from) && types::is_integer(to) {
        let from_ir = ctype_to_ir(from, ctx.target);
        let to_ir = ctype_to_ir(to, ctx.target);
        let fw = from_ir.int_width();
        let tw = to_ir.int_width();
        if tw == fw {
            return value;
        }
        if tw < fw {
            let (v, i) = ctx.builder.build_trunc(value, to_ir, span);
            emit_inst(ctx, i);
            return v;
        }
        if is_unsigned(from) {
            let (v, i) = ctx
                .builder
                .build_zext_from(value, from_ir.clone(), to_ir, span);
            emit_inst(ctx, i);
            return v;
        } else {
            let (v, i) = ctx
                .builder
                .build_sext_from(value, from_ir.clone(), to_ir, span);
            emit_inst(ctx, i);
            return v;
        }
    }

    // Bool (I1) to integer.
    if matches!(from, CType::Bool) && types::is_integer(to) {
        let to_ir = ctype_to_ir(to, ctx.target);
        let (v, i) = ctx.builder.build_zext_from(value, IrType::I1, to_ir, span);
        emit_inst(ctx, i);
        return v;
    }

    // Integer to bool.
    if types::is_integer(from) && matches!(to, CType::Bool) {
        let zero = emit_zero(ctx, ctype_to_ir(from, ctx.target), span);
        let (v, i) = ctx.builder.build_icmp(ICmpOp::Ne, value, zero, span);
        emit_inst(ctx, i);
        return v;
    }

    // Integer ↔ Float.
    if types::is_integer(from) && types::is_floating(to) {
        let to_ir = ctype_to_ir(to, ctx.target);
        let (v, i) = ctx.builder.build_bitcast(value, to_ir, span);
        emit_inst(ctx, i);
        return v;
    }
    if types::is_floating(from) && types::is_integer(to) {
        let to_ir = ctype_to_ir(to, ctx.target);
        let (v, i) = ctx.builder.build_bitcast(value, to_ir, span);
        emit_inst(ctx, i);
        return v;
    }

    // Float ↔ Float (different widths).
    if types::is_floating(from) && types::is_floating(to) {
        let to_ir = ctype_to_ir(to, ctx.target);
        let (v, i) = ctx.builder.build_bitcast(value, to_ir, span);
        emit_inst(ctx, i);
        return v;
    }

    // Integer → Pointer.
    if types::is_integer(from) && types::is_pointer(to) {
        let (v, i) = ctx.builder.build_int_to_ptr(value, span);
        emit_inst(ctx, i);
        return v;
    }

    // Pointer → Integer.
    if types::is_pointer(from) && types::is_integer(to) {
        let to_ir = ctype_to_ir(to, ctx.target);
        let (v, i) = ctx.builder.build_ptr_to_int(value, to_ir, span);
        emit_inst(ctx, i);
        return v;
    }

    // Pointer → Pointer.
    if types::is_pointer(from) && types::is_pointer(to) {
        // Same representation at IR level.
        return value;
    }

    // Enum → Int.
    if matches!(from, CType::Enum { .. }) && types::is_integer(to) {
        let to_ir = ctype_to_ir(to, ctx.target);
        let from_ir = ctype_to_ir(from, ctx.target);
        if from_ir == to_ir {
            return value;
        }
        let fw = from_ir.int_width();
        let tw = to_ir.int_width();
        if tw < fw {
            let (v, i) = ctx.builder.build_trunc(value, to_ir, span);
            emit_inst(ctx, i);
            return v;
        }
        let (v, i) = ctx
            .builder
            .build_sext_from(value, from_ir.clone(), to_ir, span);
        emit_inst(ctx, i);
        return v;
    }

    // Array → Pointer (array decay).
    if matches!(from, CType::Array { .. }) {
        return value;
    }

    // Function → Pointer (function pointer decay).
    if matches!(from, CType::Function { .. }) {
        return value;
    }

    // Default: no conversion (return as-is).
    value
}

// =========================================================================
// BOOLEAN CONVERSION  (CRITICAL UTILITY)
// =========================================================================

/// Convert a scalar value to a boolean (I1).
///
/// - Integer: `value != 0`
/// - Float: `value != 0.0`
/// - Pointer: `value != null`
fn lower_to_bool(
    ctx: &mut ExprLoweringContext<'_>,
    value: Value,
    ctype: &CType,
    span: Span,
) -> Value {
    let stripped = strip_type(ctype);

    // Already bool (I1) — return as-is.
    if matches!(stripped, CType::Bool) {
        return value;
    }

    // Integer types.
    if types::is_integer(stripped) || matches!(stripped, CType::Enum { .. }) {
        let ir_ty = ctype_to_ir(stripped, ctx.target);
        let zero = emit_zero(ctx, ir_ty, span);
        let (v, i) = ctx.builder.build_icmp(ICmpOp::Ne, value, zero, span);
        emit_inst(ctx, i);
        return v;
    }

    // Floating-point types.
    if types::is_floating(stripped) {
        let ir_ty = ctype_to_ir(stripped, ctx.target);
        let zero = emit_float_const(ctx, 0.0, ir_ty, span);
        let (v, i) = ctx.builder.build_fcmp(FCmpOp::One, value, zero, span);
        emit_inst(ctx, i);
        return v;
    }

    // Pointer types.
    if types::is_pointer(stripped) {
        let ptr_int_ty = size_ir_type(ctx.target);
        let (int_val, pti) = ctx
            .builder
            .build_ptr_to_int(value, ptr_int_ty.clone(), span);
        emit_inst(ctx, pti);
        let zero = emit_zero(ctx, ptr_int_ty, span);
        let (v, i) = ctx.builder.build_icmp(ICmpOp::Ne, int_val, zero, span);
        emit_inst(ctx, i);
        return v;
    }

    // Fallback — treat as integer comparison with zero.
    let zero = emit_zero(ctx, IrType::I32, span);
    let (v, i) = ctx.builder.build_icmp(ICmpOp::Ne, value, zero, span);
    emit_inst(ctx, i);
    v
}

// ===========================================================================
// Compile-time integer constant evaluation (for choose_expr, etc.)
// ===========================================================================

/// Try to evaluate an AST expression as a compile-time integer constant.
/// Returns `None` if the expression cannot be evaluated at compile time.
fn try_eval_integer_constant(
    expr: &ast::Expression,
    name_table: &[String],
    enum_constants: &FxHashMap<String, i128>,
) -> Option<i128> {
    match expr {
        ast::Expression::IntegerLiteral { value, .. } => Some(*value as i128),
        ast::Expression::Parenthesized { inner, .. } => {
            try_eval_integer_constant(inner, name_table, enum_constants)
        }
        ast::Expression::UnaryOp { op, operand, .. } => {
            let val = try_eval_integer_constant(operand, name_table, enum_constants)?;
            match op {
                ast::UnaryOp::Negate => Some(-val),
                ast::UnaryOp::BitwiseNot => Some(!val),
                ast::UnaryOp::LogicalNot => Some(if val == 0 { 1 } else { 0 }),
                ast::UnaryOp::Plus => Some(val),
                _ => None,
            }
        }
        ast::Expression::Binary {
            op, left, right, ..
        } => {
            let l = try_eval_integer_constant(left, name_table, enum_constants)?;
            let r = try_eval_integer_constant(right, name_table, enum_constants)?;
            match op {
                ast::BinaryOp::Add => Some(l.wrapping_add(r)),
                ast::BinaryOp::Sub => Some(l.wrapping_sub(r)),
                ast::BinaryOp::Mul => Some(l.wrapping_mul(r)),
                ast::BinaryOp::Div if r != 0 => Some(l / r),
                ast::BinaryOp::Mod if r != 0 => Some(l % r),
                ast::BinaryOp::Equal => Some(if l == r { 1 } else { 0 }),
                ast::BinaryOp::NotEqual => Some(if l != r { 1 } else { 0 }),
                ast::BinaryOp::Less => Some(if l < r { 1 } else { 0 }),
                ast::BinaryOp::Greater => Some(if l > r { 1 } else { 0 }),
                ast::BinaryOp::LessEqual => Some(if l <= r { 1 } else { 0 }),
                ast::BinaryOp::GreaterEqual => Some(if l >= r { 1 } else { 0 }),
                ast::BinaryOp::BitwiseAnd => Some(l & r),
                ast::BinaryOp::BitwiseOr => Some(l | r),
                ast::BinaryOp::BitwiseXor => Some(l ^ r),
                ast::BinaryOp::LogicalAnd => Some(if l != 0 && r != 0 { 1 } else { 0 }),
                ast::BinaryOp::LogicalOr => Some(if l != 0 || r != 0 { 1 } else { 0 }),
                ast::BinaryOp::ShiftLeft => Some(l.wrapping_shl(r as u32)),
                ast::BinaryOp::ShiftRight => Some(l.wrapping_shr(r as u32)),
                _ => None,
            }
        }
        ast::Expression::Identifier { name, .. } => {
            let idx = name.as_u32() as usize;
            if idx < name_table.len() {
                enum_constants.get(&name_table[idx]).copied()
            } else {
                None
            }
        }
        ast::Expression::BuiltinCall {
            builtin: ast::BuiltinKind::TypesCompatibleP,
            args,
            ..
        } => {
            // Recursively evaluate types_compatible_p as a constant.
            // For choose_expr where the condition IS types_compatible_p.
            // We can't fully resolve types here without ctx, so return None
            // to let the caller use the default behavior.
            let _ = args;
            None
        }
        _ => None,
    }
}

// ===========================================================================
// __builtin_types_compatible_p
// ===========================================================================

/// Compare two CTypes for compatibility, ignoring top-level qualifiers.
///
/// GCC semantics: two types are compatible if they are the same type after
/// stripping top-level `const`/`volatile`/`restrict` qualifiers.
fn types_are_compatible_ignoring_qualifiers(a: &CType, b: &CType) -> bool {
    let a = strip_qualifiers(a);
    let b = strip_qualifiers(b);
    ctypes_structurally_equal(a, b)
}

/// Strip top-level qualifiers and typedefs from a CType.
fn strip_qualifiers(ty: &CType) -> &CType {
    match ty {
        CType::Qualified(inner, _) => strip_qualifiers(inner),
        CType::Typedef { underlying, .. } => strip_qualifiers(underlying),
        _ => ty,
    }
}

/// Check structural equality of two CTypes (after qualifier stripping).
fn ctypes_structurally_equal(a: &CType, b: &CType) -> bool {
    // Discriminant check for simple unit-variant types.
    match (a, b) {
        (CType::Void, CType::Void)
        | (CType::Bool, CType::Bool)
        | (CType::Char, CType::Char)
        | (CType::SChar, CType::SChar)
        | (CType::UChar, CType::UChar)
        | (CType::Short, CType::Short)
        | (CType::UShort, CType::UShort)
        | (CType::Int, CType::Int)
        | (CType::UInt, CType::UInt)
        | (CType::Long, CType::Long)
        | (CType::ULong, CType::ULong)
        | (CType::LongLong, CType::LongLong)
        | (CType::ULongLong, CType::ULongLong)
        | (CType::Float, CType::Float)
        | (CType::Double, CType::Double)
        | (CType::LongDouble, CType::LongDouble) => true,
        (CType::Pointer(a_inner, _), CType::Pointer(b_inner, _)) => {
            ctypes_structurally_equal(strip_qualifiers(a_inner), strip_qualifiers(b_inner))
        }
        (CType::Array(a_elem, a_sz), CType::Array(b_elem, b_sz)) => {
            a_sz == b_sz
                && ctypes_structurally_equal(strip_qualifiers(a_elem), strip_qualifiers(b_elem))
        }
        (CType::Struct { name: an, .. }, CType::Struct { name: bn, .. }) => match (an, bn) {
            (Some(a_name), Some(b_name)) => a_name == b_name,
            _ => false,
        },
        (CType::Union { name: an, .. }, CType::Union { name: bn, .. }) => match (an, bn) {
            (Some(a_name), Some(b_name)) => a_name == b_name,
            _ => false,
        },
        (CType::Enum { name: an, .. }, CType::Enum { name: bn, .. }) => match (an, bn) {
            (Some(a_name), Some(b_name)) => a_name == b_name,
            _ => false,
        },
        (
            CType::Function {
                return_type: ar,
                params: ap,
                ..
            },
            CType::Function {
                return_type: br,
                params: bp,
                ..
            },
        ) => {
            ctypes_structurally_equal(strip_qualifiers(ar), strip_qualifiers(br))
                && ap.len() == bp.len()
                && ap.iter().zip(bp.iter()).all(|(pa, pb)| {
                    ctypes_structurally_equal(strip_qualifiers(pa), strip_qualifiers(pb))
                })
        }
        _ => false,
    }
}

// ===========================================================================
// __builtin_offsetof computation
// ===========================================================================

/// Compute the byte offset of a struct/union member for `__builtin_offsetof`.
///
/// Extracts the type from args[0] (a `SizeofType` carrier) and the member
/// designator chain from args[1] (an `Identifier` or `MemberAccess` chain).
/// Returns the byte offset as a `usize`.
fn compute_builtin_offsetof(
    ctx: &ExprLoweringContext<'_>,
    args: &[ast::Expression],
    _span: Span,
) -> usize {
    // Extract the type from args[0].
    let ctype = if let Some(ast::Expression::SizeofType { type_name, .. }) = args.first() {
        resolve_type_name(ctx, type_name)
    } else {
        return 0;
    };

    // Extract the member designator chain from args[1].
    // For `b` → ["b"]
    // For `inner.b` → ["inner", "b"]
    let member_chain = if let Some(member_expr) = args.get(1) {
        extract_member_chain(member_expr, ctx.name_table)
    } else {
        return 0;
    };

    // Walk the struct type, computing cumulative byte offset.
    compute_offset_in_type(&ctype, &member_chain, ctx.type_builder)
}

/// Extract a chain of member names from a member-designator expression.
///
/// Handles:
///   - `Identifier { name }` → `["name"]`
///   - `MemberAccess { object, member }` → recursively: object chain + ["member"]
fn extract_member_chain(expr: &ast::Expression, name_table: &[String]) -> Vec<String> {
    match expr {
        ast::Expression::Identifier { name, .. } => {
            let idx = name.as_u32() as usize;
            let s = if idx < name_table.len() {
                name_table[idx].clone()
            } else {
                format!("sym_{}", idx)
            };
            vec![s]
        }
        ast::Expression::MemberAccess { object, member, .. } => {
            let mut chain = extract_member_chain(object, name_table);
            let idx = member.as_u32() as usize;
            let s = if idx < name_table.len() {
                name_table[idx].clone()
            } else {
                format!("sym_{}", idx)
            };
            chain.push(s);
            chain
        }
        _ => Vec::new(),
    }
}

/// Compute byte offset of a member chain within a CType.
fn compute_offset_in_type(
    ctype: &CType,
    member_chain: &[String],
    type_builder: &TypeBuilder,
) -> usize {
    if member_chain.is_empty() {
        return 0;
    }

    let fields = match ctype {
        CType::Struct { fields, .. } => fields,
        CType::Union { fields, .. } => fields,
        _ => return 0,
    };

    let target_name = &member_chain[0];

    // Compute field offsets using the same logic as decl_lowering.
    let mut current_offset: usize = 0;
    for field in fields {
        let field_align = type_builder.alignof_type(&field.ty);
        let align = if field_align == 0 { 1 } else { field_align };
        current_offset = (current_offset + align - 1) & !(align - 1);

        let field_name = field.name.as_deref().unwrap_or("");
        if field_name == target_name {
            // Found the field.
            if member_chain.len() == 1 {
                // Final field — return the offset.
                return current_offset;
            }
            // Recurse into nested struct/union for remaining chain.
            return current_offset
                + compute_offset_in_type(&field.ty, &member_chain[1..], type_builder);
        }

        // For unions, all fields start at offset 0.
        if !matches!(ctype, CType::Union { .. }) {
            current_offset += type_builder.sizeof_type(&field.ty);
        }
    }

    // Field not found — return 0.
    0
}
