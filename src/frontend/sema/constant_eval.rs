//! Compile-time constant expression evaluation for the BCC C11 compiler.
//!
//! Implements C11 §6.6 integer constant expression semantics for Phase 5
//! (semantic analysis). Evaluates constant expressions required for:
//!
//! - **Array sizes:** `int arr[N]` — `N` must be a positive integer constant expression
//! - **Case labels:** `case V:` — `V` must be an integer constant expression
//! - **Enum values:** `enum { A = V }` — `V` must be an integer constant expression
//! - **Bitfield widths:** `int x : W` — `W` must be a non-negative integer constant expression
//! - **`_Static_assert`:** `_Static_assert(C, "msg")` — `C` must be an integer constant expression
//! - **`sizeof` / `_Alignof`:** always constant expressions (except VLAs)
//!
//! # Supported Operations
//!
//! All C11 integer arithmetic, bitwise, shift, comparison, and logical operators
//! are evaluated at compile time. The ternary `? :` and comma operators are also
//! supported. `sizeof` and `_Alignof` produce architecture-dependent constants
//! parameterised by [`Target`].
//!
//! # Overflow Behaviour
//!
//! - **Signed:** implementation-defined (wrapping `i128` arithmetic, matching GCC)
//! - **Unsigned:** wrapping per C11 §6.2.5 ¶9
//!
//! # Zero-Dependency Compliance
//!
//! This module depends only on the Rust standard library and internal crate
//! modules (`common`, `frontend::parser::ast`). It does NOT depend on `ir`,
//! `passes`, or `backend`.

use crate::common::fx_hash::FxHashMap;

use crate::common::diagnostics::{Diagnostic, DiagnosticEngine, Severity, Span};
use crate::common::string_interner::Symbol;
use crate::common::target::Target;
use crate::common::types::{alignof_ctype, is_integer, is_unsigned, sizeof_ctype, CType};
use crate::frontend::parser::ast::*;

// ===========================================================================
// ConstValue — Result of Constant Expression Evaluation
// ===========================================================================

/// Represents the result of evaluating a compile-time constant expression.
///
/// The variants cover all possible compile-time constant values in C11:
/// signed/unsigned integers, floating-point values, address constants
/// (for global symbol references with offsets), and string literals
/// (for initialiser contexts).
///
/// Integer constant expressions (C11 §6.6p6) always resolve to
/// [`SignedInt`](ConstValue::SignedInt) or [`UnsignedInt`](ConstValue::UnsignedInt).
#[derive(Debug, Clone)]
pub enum ConstValue {
    /// Signed integer value — covers all signed C integer types up to
    /// `long long` (and `__int128` when needed). Uses `i128` for
    /// full-range coverage without overflow during intermediate computation.
    SignedInt(i128),

    /// Unsigned integer value — covers all unsigned C integer types.
    /// Uses `u128` for full-range coverage.
    UnsignedInt(u128),

    /// Floating-point value — used for non-integer constant expressions
    /// where applicable (e.g., `3.14` in a `double` initialiser).
    /// Uses `f64` as the intermediate representation.
    Float(f64),

    /// Address constant — the address of a global variable or function,
    /// optionally with a byte offset. Represents expressions like
    /// `&global_var` or `&array[5]` in static initialiser contexts.
    Address {
        /// Interned symbol name of the global variable or function.
        symbol: Symbol,
        /// Byte offset from the symbol's base address.
        offset: i64,
    },

    /// String literal reference — raw byte content for initialiser contexts.
    /// Stores PUA-decoded bytes for exact fidelity.
    StringLiteral(Vec<u8>),
}

impl ConstValue {
    /// Extract a signed integer value, converting unsigned values if they
    /// fit within the `i128` range.
    ///
    /// Returns `None` for non-integer variants (Float, Address, StringLiteral)
    /// or if an unsigned value exceeds `i128::MAX`.
    pub fn to_i128(&self) -> Option<i128> {
        match self {
            ConstValue::SignedInt(v) => Some(*v),
            ConstValue::UnsignedInt(v) => {
                if *v <= i128::MAX as u128 {
                    Some(*v as i128)
                } else {
                    None
                }
            }
            ConstValue::Float(f) => Some(*f as i128),
            _ => None,
        }
    }

    /// Extract an unsigned integer value, converting signed values if they
    /// are non-negative.
    ///
    /// Returns `None` for non-integer variants or negative signed values.
    pub fn to_u128(&self) -> Option<u128> {
        match self {
            ConstValue::SignedInt(v) => {
                if *v >= 0 {
                    Some(*v as u128)
                } else {
                    None
                }
            }
            ConstValue::UnsignedInt(v) => Some(*v),
            ConstValue::Float(f) => {
                if *f >= 0.0 {
                    Some(*f as u128)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Truthiness check — returns `true` if the value is non-zero.
    ///
    /// For integer types, non-zero is truthy. For floats, non-zero and
    /// non-NaN is truthy. Address constants are always truthy (non-null).
    /// String literals are always truthy (non-null pointer).
    pub fn to_bool(&self) -> bool {
        !self.is_zero()
    }

    /// Returns `true` if the value is zero (or null).
    ///
    /// - `SignedInt(0)` → true
    /// - `UnsignedInt(0)` → true
    /// - `Float(0.0)` → true
    /// - `Address` → false (addresses are non-null by definition)
    /// - `StringLiteral` → false (string literals are non-null pointers)
    pub fn is_zero(&self) -> bool {
        match self {
            ConstValue::SignedInt(v) => *v == 0,
            ConstValue::UnsignedInt(v) => *v == 0,
            ConstValue::Float(f) => *f == 0.0,
            ConstValue::Address { .. } => false,
            ConstValue::StringLiteral(_) => false,
        }
    }

    /// Negate this constant value.
    ///
    /// - `SignedInt(v)` → `SignedInt(-v)` (wrapping)
    /// - `UnsignedInt(v)` → `SignedInt(-(v as i128))` (converts to signed)
    /// - `Float(v)` → `Float(-v)`
    /// - Other variants return themselves unchanged (negation is not
    ///   meaningful for addresses or string literals).
    pub fn negate(&self) -> ConstValue {
        match self {
            ConstValue::SignedInt(v) => ConstValue::SignedInt(v.wrapping_neg()),
            ConstValue::UnsignedInt(v) => {
                // Convert to signed and negate
                ConstValue::SignedInt(-(*v as i128))
            }
            ConstValue::Float(f) => ConstValue::Float(-f),
            other => other.clone(),
        }
    }

    /// Returns `true` if this value is a signed integer.
    fn is_signed(&self) -> bool {
        matches!(self, ConstValue::SignedInt(_))
    }

    /// Returns `true` if this value is an unsigned integer.
    fn is_unsigned_val(&self) -> bool {
        matches!(self, ConstValue::UnsignedInt(_))
    }
}

// ===========================================================================
// ConstantEvaluator — Compile-Time Expression Evaluator
// ===========================================================================

/// Evaluates C11 constant expressions at compile time during semantic analysis.
///
/// The evaluator is parameterised by a [`Target`] for architecture-dependent
/// computations (`sizeof`, `_Alignof`, pointer width) and holds a mutable
/// reference to the [`DiagnosticEngine`] for error/warning reporting.
///
/// # Usage
///
/// ```ignore
/// let mut diags = DiagnosticEngine::new();
/// let mut eval = ConstantEvaluator::new(&mut diags, Target::X86_64);
///
/// // Evaluate an integer constant expression
/// let value = eval.evaluate_integer_constant(&expr, expr.span())?;
///
/// // Validate an array size
/// let size = eval.validate_array_size(&size_expr, size_expr.span())?;
///
/// // Validate a _Static_assert
/// eval.evaluate_static_assert(&cond_expr, Some("assertion message"), span)?;
/// ```
pub struct ConstantEvaluator<'a> {
    /// Multi-error diagnostic engine for error/warning emission.
    diagnostics: &'a mut DiagnosticEngine,
    /// Target architecture for sizeof/alignof/pointer-width computations.
    target: Target,
    /// Registry of known enum constant values, populated by the semantic
    /// analyser before constant expression evaluation. Maps interned
    /// enum constant names to their integer values.
    enum_values: FxHashMap<Symbol, i128>,
    /// Registry of known struct/union tag types with complete field
    /// information, populated by the semantic analyser. Maps tag Symbol
    /// to the fully-resolved CType (with fields), enabling correct sizeof
    /// computation for struct/union types in constant expressions.
    tag_types: FxHashMap<Symbol, CType>,
    /// Registry of known variable types, populated by the semantic
    /// analyser before evaluating block-scope constant expressions.
    /// Maps variable name Symbol to its CType, enabling correct sizeof
    /// computation for `sizeof(arr)` where `arr` is a local array.
    variable_types: FxHashMap<Symbol, CType>,
}

impl<'a> ConstantEvaluator<'a> {
    /// Create a new constant expression evaluator.
    ///
    /// # Arguments
    ///
    /// * `diagnostics` — Mutable reference to the diagnostic engine for
    ///   error/warning reporting during evaluation.
    /// * `target` — The compilation target architecture, used for
    ///   architecture-dependent size and alignment computations.
    pub fn new(diagnostics: &'a mut DiagnosticEngine, target: Target) -> Self {
        ConstantEvaluator {
            diagnostics,
            target,
            enum_values: FxHashMap::default(),
            tag_types: FxHashMap::default(),
            variable_types: FxHashMap::default(),
        }
    }

    /// Register a known enum constant value for lookup during evaluation.
    ///
    /// This must be called by the semantic analyser for each enum constant
    /// before evaluating expressions that may reference them.
    pub fn register_enum_value(&mut self, name: Symbol, value: i128) {
        self.enum_values.insert(name, value);
    }

    /// Register a known struct/union tag type for sizeof/alignof lookups.
    ///
    /// This allows the constant evaluator to correctly compute
    /// `sizeof(struct tag)` for previously-defined struct/union types.
    pub fn register_tag_type(&mut self, tag: Symbol, ty: CType) {
        self.tag_types.insert(tag, ty);
    }

    /// Register a known variable type for sizeof/typeof lookups.
    ///
    /// This allows the constant evaluator to correctly compute
    /// `sizeof(var)` for local variables (e.g., arrays) in block-scope
    /// `_Static_assert` expressions.
    pub fn register_variable_type(&mut self, name: Symbol, ty: CType) {
        self.variable_types.insert(name, ty);
    }

    // ===================================================================
    // Top-Level Public Entry Points
    // ===================================================================

    /// Evaluate an expression as an integer constant expression (C11 §6.6p6).
    #[allow(clippy::result_unit_err)]
    ///
    /// An integer constant expression must consist only of:
    /// - Integer constants, enum constants, character constants
    /// - `sizeof` / `_Alignof` expressions
    /// - Casts to integer types
    /// - Arithmetic, bitwise, shift, comparison, and logical operators
    /// - Ternary operator `? :`
    ///
    /// # Returns
    ///
    /// The evaluated integer value as `i128`, or `Err(())` if the expression
    /// is not a valid integer constant expression (with diagnostics emitted).
    pub fn evaluate_integer_constant(&mut self, expr: &Expression, span: Span) -> Result<i128, ()> {
        let value = self.evaluate_constant_expr(expr)?;
        match value.to_i128() {
            Some(v) => Ok(v),
            None => {
                // Value exists but cannot be represented as i128
                // (e.g., very large unsigned value)
                match &value {
                    ConstValue::UnsignedInt(v) => {
                        // Allow large unsigned values by wrapping to i128
                        Ok(*v as i128)
                    }
                    _ => {
                        self.diagnostics
                            .emit_error(span, "expression is not an integer constant expression");
                        Err(())
                    }
                }
            }
        }
    }

    /// Evaluate a general constant expression, returning any constant value.
    ///
    /// This is the main recursive evaluation entry point. It dispatches
    /// based on the expression variant and recursively evaluates sub-expressions.
    ///
    /// # Returns
    ///
    /// A [`ConstValue`] representing the evaluated constant, or `Err(())`
    /// if the expression cannot be evaluated at compile time (with diagnostics
    /// emitted).
    #[allow(clippy::result_unit_err)]
    pub fn evaluate_constant_expr(&mut self, expr: &Expression) -> Result<ConstValue, ()> {
        match expr {
            // Integer literal: 42, 0xFF, 100ULL
            Expression::IntegerLiteral {
                value,
                suffix,
                span,
            } => Ok(self.evaluate_integer_literal(*value, suffix, *span)),

            // Floating-point literal: 3.14, 1.0f, 2.0L
            Expression::FloatLiteral { value, .. } => Ok(ConstValue::Float(*value)),

            // Character literal: 'a', L'\x00'
            Expression::CharLiteral { value, .. } => {
                // Character constants are int-width in C
                Ok(ConstValue::SignedInt(*value as i128))
            }

            // Identifier — may be an enum constant
            Expression::Identifier { name, span } => self.evaluate_enum_constant(*name, *span),

            // Parenthesised expression — recurse on inner
            Expression::Parenthesized { inner, .. } => self.evaluate_constant_expr(inner),

            // Binary operator: left op right
            Expression::Binary {
                op,
                left,
                right,
                span,
            } => self.evaluate_binary_op(*op, left, right, *span),

            // Unary operator: +x, -x, ~x, !x
            Expression::UnaryOp { op, operand, span } => {
                self.evaluate_unary_op(*op, operand, *span)
            }

            // Ternary: condition ? then : else
            Expression::Conditional {
                condition,
                then_expr,
                else_expr,
                span,
            } => self.evaluate_ternary(condition, then_expr.as_deref(), else_expr, *span),

            // Comma operator: expr1, expr2
            Expression::Comma { exprs, span } => self.evaluate_comma_exprs(exprs, *span),

            // sizeof(expr)
            Expression::SizeofExpr { operand, span } => self.evaluate_sizeof_expr(operand, *span),

            // sizeof(type-name)
            Expression::SizeofType { type_name, span } => {
                self.evaluate_sizeof_type(type_name, *span)
            }

            // _Alignof(type-name)
            Expression::AlignofType { type_name, span } => self.evaluate_alignof(type_name, *span),

            // (type-name) expr — cast
            Expression::Cast {
                type_name,
                operand,
                span,
            } => self.evaluate_cast_expr(type_name, operand, *span),

            // String literal — valid in some initialiser contexts
            Expression::StringLiteral { segments, .. } => {
                let mut bytes = Vec::new();
                for seg in segments {
                    bytes.extend_from_slice(&seg.value);
                }
                Ok(ConstValue::StringLiteral(bytes))
            }

            // Non-constant expressions — emit diagnostic
            _ => {
                let span = expr.span();
                self.diagnostics
                    .emit_error(span, "expression is not a compile-time constant");
                Err(())
            }
        }
    }

    /// Check whether an expression CAN be evaluated at compile time.
    ///
    /// This is a pure predicate — it does not emit any diagnostics.
    /// Returns `true` if the expression is structurally a constant
    /// expression, `false` otherwise.
    pub fn is_constant_expression(&self, expr: &Expression) -> bool {
        match expr {
            Expression::IntegerLiteral { .. }
            | Expression::FloatLiteral { .. }
            | Expression::CharLiteral { .. }
            | Expression::StringLiteral { .. } => true,

            Expression::Identifier { name, .. } => {
                // Enum constants are compile-time constants
                self.enum_values.contains_key(name)
            }

            Expression::Parenthesized { inner, .. } => self.is_constant_expression(inner),

            Expression::Binary { left, right, .. } => {
                self.is_constant_expression(left) && self.is_constant_expression(right)
            }

            Expression::UnaryOp {
                op: UnaryOp::Plus | UnaryOp::Negate | UnaryOp::BitwiseNot | UnaryOp::LogicalNot,
                operand,
                ..
            } => self.is_constant_expression(operand),

            // Address-of and deref are not constant expressions in general
            Expression::UnaryOp { .. } => false,

            Expression::Conditional {
                condition,
                then_expr,
                else_expr,
                ..
            } => {
                self.is_constant_expression(condition)
                    && then_expr
                        .as_ref()
                        .map_or(true, |e| self.is_constant_expression(e))
                    && self.is_constant_expression(else_expr)
            }

            Expression::Comma { exprs, .. } => exprs.iter().all(|e| self.is_constant_expression(e)),

            Expression::SizeofExpr { .. }
            | Expression::SizeofType { .. }
            | Expression::AlignofType { .. } => true,

            Expression::Cast { operand, .. } => self.is_constant_expression(operand),

            _ => false,
        }
    }

    /// Validate an array size expression.
    ///
    /// Evaluates the expression as an integer constant expression and
    /// validates that the result is positive (> 0) and fits within `usize`.
    /// Zero-length arrays are reported as errors (GCC extension for
    /// zero-length arrays should be handled separately by the caller).
    ///
    /// # Returns
    ///
    /// The validated array size as `usize`, or `Err(())` on failure.
    #[allow(clippy::result_unit_err)]
    pub fn validate_array_size(&mut self, expr: &Expression, span: Span) -> Result<usize, ()> {
        let value = self.evaluate_integer_constant(expr, span)?;

        if value < 0 {
            self.diagnostics
                .emit_error(span, "array size must be positive");
            return Err(());
        }

        if value == 0 {
            // Zero-length arrays: GCC extension. Emit a warning but allow.
            self.diagnostics
                .emit_warning(span, "zero-length array is a GCC extension");
            return Ok(0);
        }

        // Check that the value fits in usize
        if value as u128 > usize::MAX as u128 {
            self.diagnostics.emit_error(
                span,
                format!("array size {} exceeds maximum addressable size", value),
            );
            return Err(());
        }

        Ok(value as usize)
    }

    /// Validate a bitfield width expression.
    ///
    /// Evaluates the expression as an integer constant expression and
    /// validates that:
    /// - The width is >= 0
    /// - Width 0 is only valid for anonymous bitfields (padding)
    /// - The width does not exceed the bit width of the base type
    ///
    /// # Arguments
    ///
    /// * `expr` — The width expression to evaluate.
    /// * `base_type` — The base type of the bitfield (e.g., `int`, `unsigned int`).
    /// * `span` — Source location for diagnostics.
    ///
    /// # Returns
    ///
    /// The validated bitfield width as `u32`, or `Err(())` on failure.
    #[allow(clippy::result_unit_err)]
    pub fn validate_bitfield_width(
        &mut self,
        expr: &Expression,
        base_type: &CType,
        span: Span,
    ) -> Result<u32, ()> {
        let value = self.evaluate_integer_constant(expr, span)?;

        if value < 0 {
            self.diagnostics
                .emit_error(span, "bitfield width must be non-negative");
            return Err(());
        }

        // Compute the maximum allowed width based on the base type
        let max_width = (sizeof_ctype(base_type, &self.target) * 8) as i128;

        if value > max_width {
            self.diagnostics.emit(Diagnostic::error(
                span,
                format!(
                    "bitfield width {} exceeds the width of type ({} bits)",
                    value, max_width
                ),
            ));
            return Err(());
        }

        // Width 0 is valid (for anonymous bitfields / padding), caller checks context
        Ok(value as u32)
    }

    /// Evaluate a `_Static_assert` declaration.
    ///
    /// Evaluates the condition as an integer constant expression. If the
    /// result is zero (false), emits an error diagnostic with the provided
    /// message (or a generic message if none is provided).
    ///
    /// # Arguments
    ///
    /// * `expr` — The assertion condition expression.
    /// * `message` — Optional error message string from the `_Static_assert`.
    /// * `span` — Source location for diagnostics.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the assertion passes, `Err(())` if it fails.
    #[allow(clippy::result_unit_err)]
    pub fn evaluate_static_assert(
        &mut self,
        expr: &Expression,
        message: Option<&str>,
        span: Span,
    ) -> Result<(), ()> {
        let value = self.evaluate_integer_constant(expr, span)?;

        if value == 0 {
            // Static assertion failed
            let msg = match message {
                Some(m) => format!("static assertion failed: {}", m),
                None => "static assertion failed".to_string(),
            };
            self.diagnostics.emit(Diagnostic::error(span, msg));
            Err(())
        } else {
            // Assertion passed
            Ok(())
        }
    }

    // ===================================================================
    // Integer Literal Evaluation
    // ===================================================================

    /// Evaluate an integer literal with its suffix to determine the
    /// appropriate [`ConstValue`] variant (signed vs unsigned).
    ///
    /// Follows C11 §6.4.4.1 for type determination:
    /// - Decimal literals without suffix: `int`, `long`, `long long` (first that fits)
    /// - Hex/octal without suffix: also try unsigned variants
    /// - `u`/`U` suffix: unsigned variant of the smallest fitting type
    /// - `l`/`L` suffix: `long`, `long long` (first that fits)
    /// - `ul`/`UL`: `unsigned long`, `unsigned long long`
    /// - `ll`/`LL`: `long long`
    /// - `ull`/`ULL`: `unsigned long long`
    fn evaluate_integer_literal(
        &self,
        value: u128,
        suffix: &IntegerSuffix,
        _span: Span,
    ) -> ConstValue {
        match suffix {
            IntegerSuffix::None => {
                // No suffix — decimal: try signed types in order
                // C11 §6.4.4.1: int → long → long long (first that fits)
                if value <= i128::MAX as u128 {
                    ConstValue::SignedInt(value as i128)
                } else {
                    // Doesn't fit in any signed type — use unsigned
                    ConstValue::UnsignedInt(value)
                }
            }
            IntegerSuffix::U => {
                // Unsigned: unsigned int → unsigned long → unsigned long long
                ConstValue::UnsignedInt(value)
            }
            IntegerSuffix::L => {
                // Long: long → long long (signed)
                // On LP64 targets, long is 64-bit; on ILP32, long is 32-bit.
                // The value is stored in i128 regardless, but we record the
                // architecture-dependent threshold for proper type tracking.
                let long_max: u128 = if self.target.is_64bit() {
                    i64::MAX as u128
                } else {
                    i32::MAX as u128
                };
                if value <= long_max {
                    // Fits in long
                    ConstValue::SignedInt(value as i128)
                } else if value <= i128::MAX as u128 {
                    // Promotes to long long
                    ConstValue::SignedInt(value as i128)
                } else {
                    ConstValue::UnsignedInt(value)
                }
            }
            IntegerSuffix::UL => {
                // Unsigned long
                ConstValue::UnsignedInt(value)
            }
            IntegerSuffix::LL => {
                // Long long (signed) — always 64 bits, so i64::MAX
                if value <= i128::MAX as u128 {
                    ConstValue::SignedInt(value as i128)
                } else {
                    ConstValue::UnsignedInt(value)
                }
            }
            IntegerSuffix::ULL => {
                // Unsigned long long
                ConstValue::UnsignedInt(value)
            }
        }
    }

    // ===================================================================
    // Binary Operator Evaluation
    // ===================================================================

    /// Evaluate a binary operator expression on two constant operands.
    ///
    /// Handles arithmetic (`+`, `-`, `*`, `/`, `%`), bitwise (`&`, `|`, `^`),
    /// shift (`<<`, `>>`), comparison (`==`, `!=`, `<`, `>`, `<=`, `>=`),
    /// and logical (`&&`, `||`) operators.
    ///
    /// Division by zero emits an error diagnostic. Shift amounts outside
    /// the valid range emit warnings.
    fn evaluate_binary_op(
        &mut self,
        op: BinaryOp,
        lhs: &Expression,
        rhs: &Expression,
        span: Span,
    ) -> Result<ConstValue, ()> {
        // Logical operators use short-circuit evaluation
        match op {
            BinaryOp::LogicalAnd => {
                let left_val = self.evaluate_constant_expr(lhs)?;
                if !left_val.to_bool() {
                    return Ok(ConstValue::SignedInt(0));
                }
                let right_val = self.evaluate_constant_expr(rhs)?;
                return Ok(ConstValue::SignedInt(if right_val.to_bool() {
                    1
                } else {
                    0
                }));
            }
            BinaryOp::LogicalOr => {
                let left_val = self.evaluate_constant_expr(lhs)?;
                if left_val.to_bool() {
                    return Ok(ConstValue::SignedInt(1));
                }
                let right_val = self.evaluate_constant_expr(rhs)?;
                return Ok(ConstValue::SignedInt(if right_val.to_bool() {
                    1
                } else {
                    0
                }));
            }
            _ => {}
        }

        // Non-short-circuit operators: evaluate both operands
        let left_val = self.evaluate_constant_expr(lhs)?;
        let right_val = self.evaluate_constant_expr(rhs)?;

        // Perform usual arithmetic conversions
        let (lv, rv, result_unsigned) = self.usual_arithmetic_conversions(&left_val, &right_val);

        match op {
            // Arithmetic operators
            BinaryOp::Add => {
                if result_unsigned {
                    Ok(ConstValue::UnsignedInt(
                        (lv as u128).wrapping_add(rv as u128),
                    ))
                } else {
                    Ok(ConstValue::SignedInt(lv.wrapping_add(rv)))
                }
            }
            BinaryOp::Sub => {
                if result_unsigned {
                    Ok(ConstValue::UnsignedInt(
                        (lv as u128).wrapping_sub(rv as u128),
                    ))
                } else {
                    Ok(ConstValue::SignedInt(lv.wrapping_sub(rv)))
                }
            }
            BinaryOp::Mul => {
                if result_unsigned {
                    Ok(ConstValue::UnsignedInt(
                        (lv as u128).wrapping_mul(rv as u128),
                    ))
                } else {
                    Ok(ConstValue::SignedInt(lv.wrapping_mul(rv)))
                }
            }
            BinaryOp::Div => {
                if rv == 0 {
                    self.diagnostics
                        .emit_error(span, "division by zero in constant expression");
                    return Err(());
                }
                if result_unsigned {
                    Ok(ConstValue::UnsignedInt((lv as u128) / (rv as u128)))
                } else {
                    // Signed division: wrapping to handle i128::MIN / -1
                    Ok(ConstValue::SignedInt(lv.wrapping_div(rv)))
                }
            }
            BinaryOp::Mod => {
                if rv == 0 {
                    self.diagnostics
                        .emit_error(span, "division by zero in constant expression");
                    return Err(());
                }
                if result_unsigned {
                    Ok(ConstValue::UnsignedInt((lv as u128) % (rv as u128)))
                } else {
                    Ok(ConstValue::SignedInt(lv.wrapping_rem(rv)))
                }
            }

            // Bitwise operators
            BinaryOp::BitwiseAnd => {
                if result_unsigned {
                    Ok(ConstValue::UnsignedInt((lv as u128) & (rv as u128)))
                } else {
                    Ok(ConstValue::SignedInt(lv & rv))
                }
            }
            BinaryOp::BitwiseOr => {
                if result_unsigned {
                    Ok(ConstValue::UnsignedInt((lv as u128) | (rv as u128)))
                } else {
                    Ok(ConstValue::SignedInt(lv | rv))
                }
            }
            BinaryOp::BitwiseXor => {
                if result_unsigned {
                    Ok(ConstValue::UnsignedInt((lv as u128) ^ (rv as u128)))
                } else {
                    Ok(ConstValue::SignedInt(lv ^ rv))
                }
            }

            // Shift operators
            BinaryOp::ShiftLeft => self.evaluate_shift(lv, rv, result_unsigned, true, span),
            BinaryOp::ShiftRight => self.evaluate_shift(lv, rv, result_unsigned, false, span),

            // Comparison operators — always return int (0 or 1)
            BinaryOp::Equal => Ok(ConstValue::SignedInt(if lv == rv { 1 } else { 0 })),
            BinaryOp::NotEqual => Ok(ConstValue::SignedInt(if lv != rv { 1 } else { 0 })),
            BinaryOp::Less => {
                let result = if result_unsigned {
                    (lv as u128) < (rv as u128)
                } else {
                    lv < rv
                };
                Ok(ConstValue::SignedInt(if result { 1 } else { 0 }))
            }
            BinaryOp::Greater => {
                let result = if result_unsigned {
                    (lv as u128) > (rv as u128)
                } else {
                    lv > rv
                };
                Ok(ConstValue::SignedInt(if result { 1 } else { 0 }))
            }
            BinaryOp::LessEqual => {
                let result = if result_unsigned {
                    (lv as u128) <= (rv as u128)
                } else {
                    lv <= rv
                };
                Ok(ConstValue::SignedInt(if result { 1 } else { 0 }))
            }
            BinaryOp::GreaterEqual => {
                let result = if result_unsigned {
                    (lv as u128) >= (rv as u128)
                } else {
                    lv >= rv
                };
                Ok(ConstValue::SignedInt(if result { 1 } else { 0 }))
            }

            // Logical operators already handled above
            BinaryOp::LogicalAnd | BinaryOp::LogicalOr => {
                unreachable!("logical operators handled above")
            }
        }
    }

    /// Evaluate a shift operation with validation.
    ///
    /// Shift amount must be non-negative and less than the bit width.
    /// Invalid shifts emit a warning and produce a zero result.
    fn evaluate_shift(
        &mut self,
        lv: i128,
        rv: i128,
        result_unsigned: bool,
        is_left: bool,
        span: Span,
    ) -> Result<ConstValue, ()> {
        // Validate shift amount
        if rv < 0 {
            self.emit_diag(
                Severity::Warning,
                span,
                "negative shift amount in constant expression",
            );
            return Ok(if result_unsigned {
                ConstValue::UnsignedInt(0)
            } else {
                ConstValue::SignedInt(0)
            });
        }

        if rv >= 128 {
            self.emit_diag(
                Severity::Warning,
                span,
                format!("shift amount {} is too large for type width", rv),
            );
            return Ok(if result_unsigned {
                ConstValue::UnsignedInt(0)
            } else {
                ConstValue::SignedInt(0)
            });
        }

        let shift = rv as u32;

        if is_left {
            if result_unsigned {
                Ok(ConstValue::UnsignedInt((lv as u128).wrapping_shl(shift)))
            } else {
                Ok(ConstValue::SignedInt(lv.wrapping_shl(shift)))
            }
        } else {
            // Right shift: arithmetic for signed, logical for unsigned
            if result_unsigned {
                Ok(ConstValue::UnsignedInt((lv as u128) >> shift))
            } else {
                // Arithmetic right shift preserves sign
                Ok(ConstValue::SignedInt(lv >> shift))
            }
        }
    }

    // ===================================================================
    // Unary Operator Evaluation
    // ===================================================================

    /// Evaluate a unary operator expression.
    ///
    /// Handles `+` (unary plus), `-` (negation), `~` (bitwise NOT),
    /// and `!` (logical NOT).
    fn evaluate_unary_op(
        &mut self,
        op: UnaryOp,
        operand: &Expression,
        span: Span,
    ) -> Result<ConstValue, ()> {
        let val = self.evaluate_constant_expr(operand)?;

        match op {
            UnaryOp::Plus => {
                // Unary plus: integer promotion (no-op for our i128/u128 representation)
                Ok(val)
            }
            UnaryOp::Negate => Ok(val.negate()),
            UnaryOp::BitwiseNot => match val {
                ConstValue::SignedInt(v) => Ok(ConstValue::SignedInt(!v)),
                ConstValue::UnsignedInt(v) => Ok(ConstValue::UnsignedInt(!v)),
                _ => {
                    self.diagnostics
                        .emit_error(span, "bitwise NOT requires an integer operand");
                    Err(())
                }
            },
            UnaryOp::LogicalNot => {
                // !x → 1 if x is 0, else 0
                Ok(ConstValue::SignedInt(if val.is_zero() { 1 } else { 0 }))
            }
            _ => {
                // Address-of, dereference — not valid in constant context
                self.diagnostics
                    .emit_error(span, "operator not valid in constant expression");
                Err(())
            }
        }
    }

    // ===================================================================
    // Ternary and Comma Evaluation
    // ===================================================================

    /// Evaluate a ternary (conditional) expression: `cond ? then : else`.
    ///
    /// The condition is evaluated first. Based on its truthiness, the
    /// appropriate branch is returned. Both branches must be valid
    /// constant expressions (even if one is not taken).
    ///
    /// Supports the GCC conditional operand omission extension: `x ?: y`
    /// where `then_expr` is `None` and the condition value is used as
    /// the "then" result.
    fn evaluate_ternary(
        &mut self,
        cond: &Expression,
        then_expr: Option<&Expression>,
        else_expr: &Expression,
        _span: Span,
    ) -> Result<ConstValue, ()> {
        let cond_val = self.evaluate_constant_expr(cond)?;

        if cond_val.to_bool() {
            // Condition is true — evaluate and return "then" branch
            match then_expr {
                Some(expr) => self.evaluate_constant_expr(expr),
                // GCC extension: x ?: y — use the condition value as the result
                None => Ok(cond_val),
            }
        } else {
            // Condition is false — evaluate and return "else" branch
            self.evaluate_constant_expr(else_expr)
        }
    }

    /// Evaluate a comma expression list: `expr1, expr2, ..., exprN`.
    ///
    /// Evaluates all expressions in order and returns the value of the
    /// last expression. In strict C11 integer constant expression mode,
    /// comma is not allowed, but we support it for generality.
    fn evaluate_comma_exprs(&mut self, exprs: &[Expression], span: Span) -> Result<ConstValue, ()> {
        if exprs.is_empty() {
            // Use the provided span, or a dummy span if the original is
            // compiler-generated (e.g., desugared expression)
            let diag_span = if span.is_dummy() {
                Self::dummy_span()
            } else {
                Self::make_span(span.file_id, span.start, span.end)
            };
            self.diagnostics
                .emit_error(diag_span, "empty comma expression");
            return Err(());
        }

        let mut result = Err(());
        for expr in exprs {
            result = self.evaluate_constant_expr(expr);
        }
        result
    }

    // ===================================================================
    // sizeof and _Alignof Evaluation
    // ===================================================================

    /// Evaluate `sizeof(type-name)` as a constant expression.
    ///
    /// Converts the AST type name to a [`CType`] and computes its size
    /// using the target-dependent [`sizeof_ctype`] function.
    ///
    /// The result type is `size_t` (unsigned, pointer-width).
    fn evaluate_sizeof_type(&mut self, type_name: &TypeName, span: Span) -> Result<ConstValue, ()> {
        let ctype = self.resolve_type_name(type_name, span)?;
        let size = sizeof_ctype(&ctype, &self.target);
        Ok(ConstValue::UnsignedInt(size as u128))
    }

    /// Evaluate `sizeof expr` as a constant expression.
    ///
    /// For a sizeof-expression, we need to determine the type of the
    /// expression. In a full semantic analyser, this would use the type
    /// checker. Here we handle common literal cases and produce an error
    /// for expressions whose type cannot be determined statically.
    fn evaluate_sizeof_expr(&mut self, operand: &Expression, span: Span) -> Result<ConstValue, ()> {
        // Try to infer the type of the expression for sizeof
        let ctype = self.infer_expr_type(operand, span)?;
        let size = sizeof_ctype(&ctype, &self.target);
        Ok(ConstValue::UnsignedInt(size as u128))
    }

    /// Evaluate `_Alignof(type-name)` as a constant expression.
    ///
    /// Converts the AST type name to a [`CType`] and computes its alignment
    /// using the target-dependent [`alignof_ctype`] function.
    fn evaluate_alignof(&mut self, type_name: &TypeName, span: Span) -> Result<ConstValue, ()> {
        let ctype = self.resolve_type_name(type_name, span)?;
        let align = alignof_ctype(&ctype, &self.target);
        Ok(ConstValue::UnsignedInt(align as u128))
    }

    // ===================================================================
    // Cast Evaluation
    // ===================================================================

    /// Evaluate a cast expression: `(type-name) expr`.
    ///
    /// Evaluates the operand as a constant expression and applies the
    /// type conversion. Supports:
    /// - Integer-to-integer (truncation, extension, sign changes)
    /// - Float-to-integer (truncation)
    /// - Integer-to-float (conversion)
    /// - Null pointer constant (integer 0 to pointer)
    fn evaluate_cast_expr(
        &mut self,
        type_name: &TypeName,
        operand: &Expression,
        span: Span,
    ) -> Result<ConstValue, ()> {
        let val = self.evaluate_constant_expr(operand)?;
        let target_type = self.resolve_type_name(type_name, span)?;
        self.apply_cast(&target_type, &val, span)
    }

    /// Apply a type cast to a constant value.
    ///
    /// Handles integer-to-integer truncation and extension, float-to-integer
    /// truncation, integer-to-float conversion, and pointer casts.
    /// Resolves through [`CType::Typedef`] wrappers to reach the underlying type.
    fn apply_cast(
        &mut self,
        target_type: &CType,
        val: &ConstValue,
        span: Span,
    ) -> Result<ConstValue, ()> {
        // Resolve through Typedef wrappers to the underlying type
        let resolved_type = Self::resolve_typedef(target_type);

        // Determine if the target type is unsigned
        let target_is_unsigned = is_unsigned(resolved_type);
        let target_is_int = is_integer(resolved_type);
        let target_type = resolved_type;

        // Integer-to-integer cast
        if target_is_int {
            let bit_width = sizeof_ctype(target_type, &self.target) * 8;

            let raw_opt = match val {
                ConstValue::SignedInt(v) => Some(*v),
                ConstValue::UnsignedInt(v) => Some(*v as i128),
                ConstValue::Float(f) => Some(*f as i128),
                _ => None,
            };

            if let Some(raw) = raw_opt {
                return Ok(self.truncate_to_width(raw, bit_width, target_is_unsigned));
            }
        }

        // Integer/Float to float cast
        if matches!(
            target_type,
            CType::Float | CType::Double | CType::LongDouble
        ) {
            match val {
                ConstValue::SignedInt(v) => {
                    return Ok(ConstValue::Float(*v as f64));
                }
                ConstValue::UnsignedInt(v) => {
                    return Ok(ConstValue::Float(*v as f64));
                }
                ConstValue::Float(_) => {
                    return Ok(val.clone());
                }
                _ => {}
            }
        }

        // Pointer cast: only null pointer constant (0 to pointer) is allowed
        if matches!(target_type, CType::Pointer(_, _)) {
            if val.is_zero() {
                return Ok(ConstValue::UnsignedInt(0));
            }
            // Non-zero to pointer is not a constant expression
            self.diagnostics
                .emit_error(span, "non-null pointer cast is not a constant expression");
            return Err(());
        }

        // Void cast — discard value
        if matches!(target_type, CType::Void) {
            return Ok(ConstValue::SignedInt(0));
        }

        // Fallback — just pass through the value
        Ok(val.clone())
    }

    /// Truncate an integer value to a specific bit width.
    ///
    /// For unsigned targets, masks to the appropriate width.
    /// For signed targets, sign-extends after masking.
    fn truncate_to_width(
        &self,
        value: i128,
        bit_width: usize,
        is_unsigned_target: bool,
    ) -> ConstValue {
        if bit_width == 0 || bit_width >= 128 {
            // No truncation needed
            if is_unsigned_target {
                return ConstValue::UnsignedInt(value as u128);
            } else {
                return ConstValue::SignedInt(value);
            }
        }

        let mask = if bit_width >= 128 {
            u128::MAX
        } else {
            (1u128 << bit_width) - 1
        };
        let masked = (value as u128) & mask;

        if is_unsigned_target {
            ConstValue::UnsignedInt(masked)
        } else {
            // Sign-extend: check the sign bit
            let sign_bit = 1u128 << (bit_width - 1);
            if masked & sign_bit != 0 {
                // Negative: sign-extend by filling upper bits with 1s
                let extended = masked | !mask;
                ConstValue::SignedInt(extended as i128)
            } else {
                ConstValue::SignedInt(masked as i128)
            }
        }
    }

    // ===================================================================
    // Enum Constant Evaluation
    // ===================================================================

    /// Look up an enum constant value by its interned symbol name.
    ///
    /// If the symbol is registered as an enum constant (via
    /// [`register_enum_value`](ConstantEvaluator::register_enum_value)),
    /// returns its value. Otherwise, emits a diagnostic error.
    fn evaluate_enum_constant(&mut self, name: Symbol, span: Span) -> Result<ConstValue, ()> {
        if let Some(&value) = self.enum_values.get(&name) {
            Ok(ConstValue::SignedInt(value))
        } else {
            self.diagnostics.emit_error(
                span,
                "identifier is not a compile-time constant (not an enum constant)",
            );
            Err(())
        }
    }

    // ===================================================================
    // Type Name Resolution (AST → CType)
    // ===================================================================

    /// Convert an AST [`TypeName`] to a [`CType`] for sizeof/alignof
    /// and cast evaluation.
    ///
    /// This is a simplified type resolver sufficient for constant expression
    /// evaluation. It handles basic type specifiers, pointers, and arrays.
    /// Complex cases (typedefs, struct members) would be resolved by the
    /// full semantic analyser.
    fn resolve_type_name(&mut self, type_name: &TypeName, span: Span) -> Result<CType, ()> {
        // Resolve the base type from specifiers
        let base_type =
            self.resolve_type_specifiers(&type_name.specifier_qualifiers.type_specifiers, span)?;

        // Apply abstract declarator (pointer/array) if present
        match &type_name.abstract_declarator {
            Some(decl) => self.apply_abstract_declarator(base_type, decl, span),
            None => Ok(base_type),
        }
    }

    /// Resolve a list of type specifiers into a [`CType`].
    ///
    /// Handles the C11 rules for combining multiple type specifiers:
    /// - `unsigned long long` = three specifiers combining into `ULongLong`
    /// - `signed char` = two specifiers combining into `SChar`
    /// - etc.
    fn resolve_type_specifiers(
        &mut self,
        specifiers: &[TypeSpecifier],
        span: Span,
    ) -> Result<CType, ()> {
        if specifiers.is_empty() {
            // Default to int if no type specifiers (C11 implicit int)
            return Ok(CType::Int);
        }

        // Track specifier flags
        let mut has_void = false;
        let mut has_char = false;
        let mut has_short = false;
        let mut has_int = false;
        let mut long_count: usize = 0;
        let mut has_float = false;
        let mut has_double = false;
        let mut has_signed = false;
        let mut has_unsigned = false;
        let mut has_bool = false;
        let mut has_complex = false;

        for spec in specifiers {
            match spec {
                TypeSpecifier::Void => has_void = true,
                TypeSpecifier::Char => has_char = true,
                TypeSpecifier::Short => has_short = true,
                TypeSpecifier::Int => has_int = true,
                TypeSpecifier::Long => long_count += 1,
                TypeSpecifier::Float => has_float = true,
                TypeSpecifier::Double => has_double = true,
                TypeSpecifier::Signed => has_signed = true,
                TypeSpecifier::Unsigned => has_unsigned = true,
                TypeSpecifier::Bool => has_bool = true,
                TypeSpecifier::Complex => has_complex = true,

                // Struct/Union/Enum — look up the tag type from the
                // registered tag map if available, otherwise fall back to
                // an empty struct (forward declaration).
                TypeSpecifier::Struct(s) => {
                    if let Some(tag_sym) = s.tag {
                        if let Some(resolved) = self.tag_types.get(&tag_sym) {
                            return Ok(resolved.clone());
                        }
                    }
                    return Ok(CType::Struct {
                        name: s.tag.map(|sym| format!("struct_{}", sym.as_u32())),
                        fields: Vec::new(),
                        packed: false,
                        aligned: None,
                    });
                }
                TypeSpecifier::Union(u) => {
                    if let Some(tag_sym) = u.tag {
                        if let Some(resolved) = self.tag_types.get(&tag_sym) {
                            return Ok(resolved.clone());
                        }
                    }
                    return Ok(CType::Union {
                        name: u.tag.map(|sym| format!("union_{}", sym.as_u32())),
                        fields: Vec::new(),
                        packed: false,
                        aligned: None,
                    });
                }
                TypeSpecifier::Enum(e) => {
                    return Ok(CType::Enum {
                        name: e.tag.map(|sym| format!("enum_{}", sym.as_u32())),
                        underlying_type: Box::new(CType::Int),
                    });
                }

                // Typedef name — cannot resolve without full symbol table
                TypeSpecifier::TypedefName(_sym) => {
                    // Best effort: treat as int for sizeof purposes
                    // The full sema will resolve this properly
                    return Ok(CType::Int);
                }

                // _Atomic(type) — recurse on inner type
                TypeSpecifier::Atomic(inner_tn) => {
                    let inner = self.resolve_type_name(inner_tn, span)?;
                    return Ok(CType::Atomic(Box::new(inner)));
                }

                // typeof — cannot evaluate without expression type inference
                TypeSpecifier::Typeof(_) => {
                    self.diagnostics.emit_error(
                        span,
                        "typeof in constant expression context is not supported",
                    );
                    return Err(());
                }
            }
        }

        // Combine specifier flags into a CType
        if has_void {
            return Ok(CType::Void);
        }
        if has_bool {
            return Ok(CType::Bool);
        }

        // Complex types
        if has_complex {
            let base = if has_float {
                CType::Float
            } else if long_count > 0 {
                CType::LongDouble
            } else {
                CType::Double
            };
            return Ok(CType::Complex(Box::new(base)));
        }

        // Floating-point types
        if has_float {
            return Ok(CType::Float);
        }
        if has_double {
            if long_count > 0 {
                return Ok(CType::LongDouble);
            }
            return Ok(CType::Double);
        }

        // Character types
        if has_char {
            if has_unsigned {
                return Ok(CType::UChar);
            } else if has_signed {
                return Ok(CType::SChar);
            } else {
                return Ok(CType::Char);
            }
        }

        // Short types
        if has_short {
            if has_unsigned {
                return Ok(CType::UShort);
            } else {
                return Ok(CType::Short);
            }
        }

        // Long long types
        if long_count >= 2 {
            if has_unsigned {
                return Ok(CType::ULongLong);
            } else {
                return Ok(CType::LongLong);
            }
        }

        // Long types
        if long_count == 1 {
            if has_unsigned {
                return Ok(CType::ULong);
            } else {
                return Ok(CType::Long);
            }
        }

        // Int types (including bare `signed`/`unsigned`)
        if has_unsigned {
            return Ok(CType::UInt);
        }
        if has_signed || has_int {
            return Ok(CType::Int);
        }

        // Default: int
        Ok(CType::Int)
    }

    /// Apply an abstract declarator to a base type, building pointer
    /// and array types.
    fn apply_abstract_declarator(
        &mut self,
        base: CType,
        decl: &AbstractDeclarator,
        span: Span,
    ) -> Result<CType, ()> {
        let mut result = base;

        // Apply direct abstract declarator first (array, function)
        if let Some(ref direct) = decl.direct {
            result = self.apply_direct_abstract_declarator(result, direct, span)?;
        }

        // Apply pointer chain
        if let Some(ref ptr) = decl.pointer {
            result = self.apply_pointer_chain(result, ptr);
        }

        Ok(result)
    }

    /// Apply a direct abstract declarator (array or function shape).
    fn apply_direct_abstract_declarator(
        &mut self,
        base: CType,
        direct: &DirectAbstractDeclarator,
        span: Span,
    ) -> Result<CType, ()> {
        match direct {
            DirectAbstractDeclarator::Parenthesized(inner) => {
                self.apply_abstract_declarator(base, inner, span)
            }
            DirectAbstractDeclarator::Array { size, .. } => {
                let array_size = match size {
                    Some(expr) => {
                        let val = self.evaluate_integer_constant(expr, span)?;
                        if val >= 0 {
                            Some(val as usize)
                        } else {
                            None
                        }
                    }
                    None => None,
                };
                Ok(CType::Array(Box::new(base), array_size))
            }
            DirectAbstractDeclarator::Function { .. } => {
                // Function type: sizeof(function) == 1 in GCC
                Ok(CType::Function {
                    return_type: Box::new(base),
                    params: Vec::new(),
                    variadic: false,
                })
            }
        }
    }

    /// Apply a pointer chain to a type.
    fn apply_pointer_chain(&self, base: CType, ptr: &Pointer) -> CType {
        let mut result = CType::Pointer(
            Box::new(base),
            crate::common::types::TypeQualifiers::default(),
        );

        // Recurse for multi-level pointers
        if let Some(ref inner) = ptr.inner {
            result = self.apply_pointer_chain(result, inner);
        }

        result
    }

    // ===================================================================
    // Expression Type Inference (for sizeof(expr))
    // ===================================================================

    /// Infer the C type of an expression for `sizeof(expr)` evaluation.
    ///
    /// This is a simplified type inference for common cases. The full
    /// semantic analyser provides comprehensive type inference, but for
    /// constant evaluation we handle the most common patterns.
    fn infer_expr_type(&mut self, expr: &Expression, span: Span) -> Result<CType, ()> {
        match expr {
            Expression::IntegerLiteral { suffix, .. } => Ok(self.integer_suffix_to_type(suffix)),
            Expression::FloatLiteral { suffix, .. } => match suffix {
                FloatSuffix::F => Ok(CType::Float),
                FloatSuffix::L => Ok(CType::LongDouble),
                FloatSuffix::None => Ok(CType::Double),
            },
            Expression::CharLiteral { .. } => Ok(CType::Int),
            Expression::StringLiteral { .. } => {
                // String literal: pointer to char
                Ok(CType::Pointer(
                    Box::new(CType::Char),
                    crate::common::types::TypeQualifiers {
                        is_const: true,
                        ..Default::default()
                    },
                ))
            }
            Expression::Parenthesized { inner, .. } => self.infer_expr_type(inner, span),
            Expression::Cast { type_name, .. } => self.resolve_type_name(type_name, span),
            Expression::SizeofExpr { .. }
            | Expression::SizeofType { .. }
            | Expression::AlignofType { .. } => {
                // sizeof/alignof return size_t
                if self.target.is_64bit() {
                    Ok(CType::ULong)
                } else {
                    Ok(CType::UInt)
                }
            }
            Expression::UnaryOp { op, operand, .. } => {
                match op {
                    UnaryOp::AddressOf => {
                        let inner_type = self.infer_expr_type(operand, span)?;
                        Ok(CType::Pointer(
                            Box::new(inner_type),
                            crate::common::types::TypeQualifiers::default(),
                        ))
                    }
                    UnaryOp::Deref => {
                        let inner_type = self.infer_expr_type(operand, span)?;
                        match inner_type {
                            CType::Pointer(pointee, _) => Ok(*pointee),
                            _ => Ok(CType::Int), // fallback
                        }
                    }
                    _ => self.infer_expr_type(operand, span),
                }
            }
            Expression::Identifier { name, .. } => {
                // Look up variable type from the registry (populated by sema
                // for block-scope _Static_assert evaluation).
                if let Some(ty) = self.variable_types.get(name) {
                    Ok(ty.clone())
                } else {
                    // Unknown identifier — fallback to int
                    Ok(CType::Int)
                }
            }
            Expression::Binary { left, right, .. } => {
                // Binary ops: infer type from left operand
                let lt = self.infer_expr_type(left, span)?;
                let _rt = self.infer_expr_type(right, span)?;
                Ok(lt)
            }
            _ => {
                // Default: assume int for expressions we can't easily type
                // The full sema will provide accurate types
                Ok(CType::Int)
            }
        }
    }

    /// Convert an integer suffix to the corresponding C type.
    fn integer_suffix_to_type(&self, suffix: &IntegerSuffix) -> CType {
        match suffix {
            IntegerSuffix::None => CType::Int,
            IntegerSuffix::U => CType::UInt,
            IntegerSuffix::L => CType::Long,
            IntegerSuffix::UL => CType::ULong,
            IntegerSuffix::LL => CType::LongLong,
            IntegerSuffix::ULL => CType::ULongLong,
        }
    }

    // ===================================================================
    // Usual Arithmetic Conversions
    // ===================================================================

    /// Perform the usual arithmetic conversions on two constant values.
    ///
    /// Returns the values as `i128` (for uniform computation) and a flag
    /// indicating whether the result should be treated as unsigned.
    ///
    /// C11 §6.3.1.8: If either operand is unsigned, the result is unsigned.
    /// If both are signed, the result is signed. This follows the standard
    /// integer promotion and usual arithmetic conversion rules.
    fn usual_arithmetic_conversions(
        &self,
        left: &ConstValue,
        right: &ConstValue,
    ) -> (i128, i128, bool) {
        let lv = match left {
            ConstValue::SignedInt(v) => *v,
            ConstValue::UnsignedInt(v) => *v as i128,
            ConstValue::Float(f) => *f as i128,
            _ => 0,
        };

        let rv = match right {
            ConstValue::SignedInt(v) => *v,
            ConstValue::UnsignedInt(v) => *v as i128,
            ConstValue::Float(f) => *f as i128,
            _ => 0,
        };

        // C11 §6.3.1.8: If either operand is unsigned, the result is unsigned.
        // If both operands are signed, the result is signed.
        let result_unsigned = if left.is_unsigned_val() || right.is_unsigned_val() {
            true
        } else {
            // Both signed — result is signed
            let _both_signed = left.is_signed() && right.is_signed();
            false
        };

        (lv, rv, result_unsigned)
    }

    /// Evaluate a [`StaticAssert`] AST node directly.
    ///
    /// Convenience method that extracts the condition and optional message
    /// from a `StaticAssert` struct and delegates to
    /// [`evaluate_static_assert`](ConstantEvaluator::evaluate_static_assert).
    #[allow(clippy::result_unit_err)]
    pub fn evaluate_static_assert_node(&mut self, sa: &StaticAssert) -> Result<(), ()> {
        let message = sa.message.as_ref().map(|m| {
            // Convert raw bytes to string (best effort UTF-8 decoding)
            String::from_utf8_lossy(m).into_owned()
        });
        self.evaluate_static_assert(&sa.condition, message.as_deref(), sa.span)
    }

    /// Get the pointer width in bytes for the target architecture.
    ///
    /// Returns 4 for i686 (ILP32) and 8 for x86-64/AArch64/RISC-V 64 (LP64).
    /// Used for sizeof(pointer) and sizeof(size_t) constant evaluation.
    #[inline]
    pub fn target_pointer_width(&self) -> usize {
        self.target.pointer_width()
    }

    /// Get the size of `long` in bytes for the target architecture.
    ///
    /// Returns 4 for i686 (ILP32) and 8 for LP64 targets.
    #[inline]
    pub fn target_long_size(&self) -> usize {
        self.target.long_size()
    }

    /// Get the size of `long double` in bytes for the target architecture.
    #[inline]
    pub fn target_long_double_size(&self) -> usize {
        self.target.long_double_size()
    }

    /// Get the alignment of `long double` in bytes for the target architecture.
    #[inline]
    pub fn target_long_double_align(&self) -> usize {
        self.target.long_double_align()
    }

    /// Resolve through [`CType::Typedef`] chains to the underlying type.
    ///
    /// Recursively strips `Typedef` wrappers until a non-typedef type is
    /// reached. Also strips `CType::Atomic` wrappers for consistency.
    fn resolve_typedef(ty: &CType) -> &CType {
        match ty {
            CType::Typedef { underlying, .. } => Self::resolve_typedef(underlying),
            CType::Atomic(inner) => Self::resolve_typedef(inner),
            other => other,
        }
    }

    /// Create a span covering a range of bytes in a given source file.
    ///
    /// Convenience wrapper around [`Span::new`] for creating spans during
    /// constant evaluation.
    #[inline]
    fn make_span(file_id: u32, start: u32, end: u32) -> Span {
        Span::new(file_id, start, end)
    }

    /// Create a dummy span for compiler-generated constructs that have
    /// no corresponding source location.
    ///
    /// Convenience wrapper around [`Span::dummy`].
    #[inline]
    fn dummy_span() -> Span {
        Span::dummy()
    }

    /// Emit a diagnostic with explicit severity control.
    ///
    /// This is a convenience helper that directly constructs a [`Diagnostic`]
    /// with the specified [`Severity`], providing fine-grained control over
    /// whether a message is emitted as an error or warning.
    fn emit_diag(&mut self, severity: Severity, span: Span, msg: impl Into<String>) {
        let diag = match severity {
            Severity::Error => Diagnostic::error(span, msg),
            Severity::Warning => Diagnostic::warning(span, msg),
            Severity::Note => Diagnostic::note(span, msg),
        };
        self.diagnostics.emit(diag);
    }
}
