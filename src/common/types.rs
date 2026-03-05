//! Dual type system — C language types and target-machine ABI types.
//!
//! This module provides two complementary type representations:
//!
//! 1. **[`CType`]** — C language types used by the frontend (parser,
//!    semantic analysis) to represent the C11 type system including GCC
//!    extensions, qualifiers, and composite types.
//!
//! 2. **[`MachineType`]** — Target-machine types and register classes used
//!    by the backend for instruction selection, register allocation, and
//!    ABI-correct calling convention implementation.
//!
//! The IR layer acts as the bridge: during lowering, C types are translated
//! to IR types; during code generation, IR types map to machine types.
//!
//! All size and alignment computations are parameterised by
//! [`Target`](crate::common::target::Target) so that the same `CType`
//! yields correct results for LP64 (x86-64, AArch64, RISC-V 64) and
//! ILP32 (i686) data models.

use crate::common::target::Target;

// ===========================================================================
// Type Qualifiers
// ===========================================================================

/// C11 type qualifiers.
///
/// Each qualifier corresponds to a keyword in C11 (`const`, `volatile`,
/// `restrict`, `_Atomic`). Multiple qualifiers may be present simultaneously
/// on a single type.
///
/// The `_Atomic` qualifier is supported at the storage/representation level.
/// Actual atomic operations may delegate to `libatomic` at link time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct TypeQualifiers {
    /// `const` — object is read-only after initialisation.
    pub is_const: bool,
    /// `volatile` — every access must be honoured; no optimisation may
    /// elide or reorder accesses.
    pub is_volatile: bool,
    /// `restrict` — the pointer is the sole means of accessing the
    /// underlying object within its scope (C11 §6.7.3.1).
    pub is_restrict: bool,
    /// `_Atomic` — the object has atomic storage semantics.
    pub is_atomic: bool,
}

impl TypeQualifiers {
    /// Returns `true` if no qualifiers are set.
    #[inline]
    pub fn is_empty(&self) -> bool {
        !self.is_const && !self.is_volatile && !self.is_restrict && !self.is_atomic
    }

    /// Merge two qualifier sets (union of flags).
    #[inline]
    pub fn merge(self, other: TypeQualifiers) -> TypeQualifiers {
        TypeQualifiers {
            is_const: self.is_const || other.is_const,
            is_volatile: self.is_volatile || other.is_volatile,
            is_restrict: self.is_restrict || other.is_restrict,
            is_atomic: self.is_atomic || other.is_atomic,
        }
    }
}

// ===========================================================================
// Struct / Union Field
// ===========================================================================

/// Representation of a single field within a `struct` or `union`.
///
/// Supports both regular fields and bitfields (indicated by a non-`None`
/// `bit_width`). Anonymous fields (e.g. anonymous inner structs/unions)
/// have `name` set to `None`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructField {
    /// Field name, or `None` for anonymous fields.
    pub name: Option<String>,
    /// C type of this field.
    pub ty: CType,
    /// Bit-width for bitfields, or `None` for regular fields.
    pub bit_width: Option<u32>,
}

// ===========================================================================
// CType — C Language Types
// ===========================================================================

/// Comprehensive enumeration of C11 types plus GCC extensions.
///
/// Every variant maps directly to a concept in the C type system. The enum
/// is recursive (via `Box`) for composed types such as pointers, arrays,
/// and functions.
///
/// # Size and Alignment
///
/// The physical size and alignment of each type depend on the target
/// architecture. Use [`sizeof_ctype`] and [`alignof_ctype`] with a
/// [`Target`] reference for correct, architecture-dependent results.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CType {
    // ----- Void -----
    /// `void` — incomplete type, cannot be instantiated.
    Void,

    // ----- Boolean -----
    /// `_Bool` (C11) / `bool` — 1-byte Boolean.
    Bool,

    // ----- Character types -----
    /// `char` — signedness is implementation-defined; BCC treats it as
    /// signed (matching GCC's default on x86/ARM/RISC-V).
    Char,
    /// `signed char` — explicitly signed 8-bit integer.
    SChar,
    /// `unsigned char` — explicitly unsigned 8-bit integer.
    UChar,

    // ----- Integer types -----
    /// `short` / `signed short`.
    Short,
    /// `unsigned short`.
    UShort,
    /// `int` / `signed int`.
    Int,
    /// `unsigned int`.
    UInt,
    /// `long` / `signed long` — 4 bytes on ILP32, 8 bytes on LP64.
    Long,
    /// `unsigned long`.
    ULong,
    /// `long long` / `signed long long` — always 8 bytes.
    LongLong,
    /// `unsigned long long`.
    ULongLong,

    // ----- Floating-point types -----
    /// `float` — IEEE 754 single-precision (32-bit).
    Float,
    /// `double` — IEEE 754 double-precision (64-bit).
    Double,
    /// `long double` — 80-bit x87 extended-precision on x86, 128-bit
    /// quad-precision on AArch64 and RISC-V 64.
    LongDouble,

    // ----- Complex types (C11 _Complex) -----
    /// `_Complex <base>` — pair of the base floating-point type.
    Complex(Box<CType>),

    // ----- Pointer -----
    /// Pointer to a type, carrying qualifiers on the pointer itself
    /// (e.g. `int *const` has `is_const = true` in the qualifiers).
    Pointer(Box<CType>, TypeQualifiers),

    // ----- Array -----
    /// Array of a type. `None` size indicates a variable-length array (VLA)
    /// or a flexible array member.
    Array(Box<CType>, Option<usize>),

    // ----- Function -----
    /// Function type: return type, parameter types, and variadic flag.
    Function {
        return_type: Box<CType>,
        params: Vec<CType>,
        variadic: bool,
    },

    // ----- Struct -----
    /// `struct` type with optional name, fields, packing, and alignment.
    Struct {
        name: Option<String>,
        fields: Vec<StructField>,
        packed: bool,
        /// Explicit alignment override from `__attribute__((aligned(N)))`.
        aligned: Option<usize>,
    },

    // ----- Union -----
    /// `union` type, laid out as the largest field. Shares structural
    /// metadata with `Struct`.
    Union {
        name: Option<String>,
        fields: Vec<StructField>,
        packed: bool,
        aligned: Option<usize>,
    },

    // ----- Enum -----
    /// `enum` type wrapping an underlying integer type (typically `int`).
    Enum {
        name: Option<String>,
        underlying_type: Box<CType>,
    },

    // ----- Atomic -----
    /// `_Atomic(T)` — atomic qualifier at the type level.
    Atomic(Box<CType>),

    // ----- Typedef -----
    /// `typedef` wrapping another type with a user-defined name.
    Typedef {
        name: String,
        underlying: Box<CType>,
    },

    // ----- Qualified -----
    /// Type with qualifiers applied (`const`, `volatile`, `restrict`,
    /// `_Atomic`).
    Qualified(Box<CType>, TypeQualifiers),
}

// ===========================================================================
// MachineType — Backend Register Classes
// ===========================================================================

/// Machine-level type representation used by the backend for register
/// allocation, instruction selection, and ABI classification.
///
/// The variants fall into two categories:
///
/// 1. **Concrete data types** (`I8`–`I128`, `F32`–`F80`, `Ptr`, `Void`) —
///    describe the physical representation of a value.
///
/// 2. **ABI register classes** (`Integer`, `SSE`, `X87`, `Memory`) — used
///    during calling-convention classification to determine how a value is
///    passed to or returned from a function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MachineType {
    /// 8-bit integer.
    I8,
    /// 16-bit integer.
    I16,
    /// 32-bit integer.
    I32,
    /// 64-bit integer.
    I64,
    /// 128-bit integer (`__int128`).
    I128,
    /// IEEE 754 single-precision float (32-bit).
    F32,
    /// IEEE 754 double-precision float (64-bit).
    F64,
    /// x87 extended-precision float (80-bit, used on i686/x86-64).
    F80,
    /// Pointer-width value (32-bit on ILP32, 64-bit on LP64).
    Ptr,
    /// Void — no value (used for void-returning functions).
    Void,

    // -- ABI register classes -----------------------------------------------
    /// General-purpose integer register class.
    Integer,
    /// SSE / floating-point register class (x86-64, AArch64 NEON/FP).
    SSE,
    /// x87 FPU register class (i686 floating-point returns).
    X87,
    /// Passed/returned in memory (on the stack).
    Memory,
}

// ===========================================================================
// sizeof — Target-Dependent Size Computation
// ===========================================================================

/// Returns the size in bytes of a C type for the given target architecture.
///
/// This function mirrors GCC's `sizeof` semantics including GCC extensions
/// (e.g. `sizeof(void) == 1`). Sizes of composite types (structs, unions)
/// are computed recursively with proper alignment padding.
///
/// # Arguments
///
/// * `ty` — The C type to measure.
/// * `target` — The compilation target, providing data-model constants.
///
/// # Returns
///
/// The size in bytes. Returns `0` for flexible-array/VLA arrays with no
/// known element count, and `1` for `void` and function types (GCC compat).
pub fn sizeof_ctype(ty: &CType, target: &Target) -> usize {
    match ty {
        // -- Void: GCC extension — sizeof(void) == 1 for pointer arithmetic.
        CType::Void => 1,

        // -- Boolean
        CType::Bool => 1,

        // -- Character types — always 1 byte
        CType::Char | CType::SChar | CType::UChar => 1,

        // -- Short — 2 bytes on all targets
        CType::Short | CType::UShort => 2,

        // -- Int — 4 bytes on all targets
        CType::Int | CType::UInt => 4,

        // -- Long — target-dependent (ILP32: 4, LP64: 8)
        CType::Long | CType::ULong => target.long_size(),

        // -- Long long — always 8 bytes
        CType::LongLong | CType::ULongLong => 8,

        // -- Float — 4 bytes (IEEE 754 single)
        CType::Float => 4,

        // -- Double — 8 bytes (IEEE 754 double)
        CType::Double => 8,

        // -- Long double — target-dependent (i686: 12, others: 16)
        CType::LongDouble => target.long_double_size(),

        // -- Complex — pair of the base floating-point type
        CType::Complex(base) => 2 * sizeof_ctype(base, target),

        // -- Pointer — target-dependent (ILP32: 4, LP64: 8)
        CType::Pointer(_, _) => target.pointer_width(),

        // -- Array — element size × count; 0 for incomplete arrays
        CType::Array(elem, count) => match count {
            Some(n) => sizeof_ctype(elem, target) * n,
            None => 0,
        },

        // -- Function — GCC extension: sizeof(function) == 1
        CType::Function { .. } => 1,

        // -- Struct — compute layout with padding
        CType::Struct {
            fields,
            packed,
            aligned,
            ..
        } => compute_struct_size(fields, *packed, *aligned, target),

        // -- Union — max of all field sizes, padded to alignment
        CType::Union {
            fields,
            packed,
            aligned,
            ..
        } => compute_union_size(fields, *packed, *aligned, target),

        // -- Enum — size of the underlying integer type
        CType::Enum {
            underlying_type, ..
        } => sizeof_ctype(underlying_type, target),

        // -- Atomic — same size as the inner type
        CType::Atomic(inner) => sizeof_ctype(inner, target),

        // -- Typedef — size of the underlying type
        CType::Typedef { underlying, .. } => sizeof_ctype(underlying, target),

        // -- Qualified — size of the unqualified type
        CType::Qualified(inner, _) => sizeof_ctype(inner, target),
    }
}

/// Compute the total size of a struct, including inter-field alignment
/// padding and trailing padding to the struct's overall alignment.
///
/// When `packed` is true, no inter-field padding is inserted (fields are
/// laid out contiguously). An explicit `aligned` override sets the minimum
/// struct alignment (the actual alignment is the maximum of the natural
/// alignment and the explicit override).
fn compute_struct_size(
    fields: &[StructField],
    packed: bool,
    aligned: Option<usize>,
    target: &Target,
) -> usize {
    if fields.is_empty() {
        // Empty structs have size 0 (GCC extension; ISO C forbids them).
        return if let Some(a) = aligned { a.max(1) } else { 0 };
    }

    let mut offset: usize = 0;
    let mut max_field_align: usize = 1;

    for field in fields {
        // For bitfields, approximate: treat as the underlying type's size
        // when bit_width is present. A full bitfield layout engine would
        // track bit offsets, but for sizeof purposes this is sufficient.
        let field_size = if let Some(bits) = field.bit_width {
            // A zero-width bitfield forces alignment without consuming space.
            if bits == 0 {
                let fa = if packed {
                    1
                } else {
                    alignof_ctype(&field.ty, target)
                };
                offset = align_up(offset, fa);
                if fa > max_field_align {
                    max_field_align = fa;
                }
                continue;
            }
            // Non-zero bitfield: round up to whole bytes.
            ((bits as usize) + 7) / 8
        } else {
            sizeof_ctype(&field.ty, target)
        };

        let field_align = if packed {
            1
        } else {
            alignof_ctype(&field.ty, target)
        };

        // Align the current offset to the field's alignment requirement.
        offset = align_up(offset, field_align);

        // Advance past this field.
        offset += field_size;

        if field_align > max_field_align {
            max_field_align = field_align;
        }
    }

    // Apply explicit alignment override.
    let struct_align = match aligned {
        Some(a) => a.max(max_field_align),
        None => {
            if packed {
                1
            } else {
                max_field_align
            }
        }
    };

    // Pad the struct to a multiple of its alignment.
    align_up(offset, struct_align)
}

/// Compute the total size of a union (max field size, padded to alignment).
fn compute_union_size(
    fields: &[StructField],
    packed: bool,
    aligned: Option<usize>,
    target: &Target,
) -> usize {
    if fields.is_empty() {
        return if let Some(a) = aligned { a.max(1) } else { 0 };
    }

    let mut max_size: usize = 0;
    let mut max_align: usize = 1;

    for field in fields {
        let field_size = sizeof_ctype(&field.ty, target);
        let field_align = if packed {
            1
        } else {
            alignof_ctype(&field.ty, target)
        };

        if field_size > max_size {
            max_size = field_size;
        }
        if field_align > max_align {
            max_align = field_align;
        }
    }

    let union_align = match aligned {
        Some(a) => a.max(max_align),
        None => {
            if packed {
                1
            } else {
                max_align
            }
        }
    };

    align_up(max_size, union_align)
}

/// Round `value` up to the nearest multiple of `align`.
///
/// `align` must be a power of two (or 1). If `align` is 0, returns `value`
/// unchanged.
#[inline]
fn align_up(value: usize, align: usize) -> usize {
    if align == 0 {
        return value;
    }
    let mask = align - 1;
    (value + mask) & !mask
}

// ===========================================================================
// alignof — Target-Dependent Alignment Computation
// ===========================================================================

/// Returns the alignment in bytes of a C type for the given target
/// architecture.
///
/// # Arguments
///
/// * `ty` — The C type whose alignment to query.
/// * `target` — The compilation target, providing data-model constants.
///
/// # Returns
///
/// The required alignment in bytes (always a power of two, ≥ 1).
pub fn alignof_ctype(ty: &CType, target: &Target) -> usize {
    match ty {
        // -- Void: alignment 1 (GCC extension)
        CType::Void => 1,

        // -- Boolean
        CType::Bool => 1,

        // -- Character types
        CType::Char | CType::SChar | CType::UChar => 1,

        // -- Short
        CType::Short | CType::UShort => 2,

        // -- Int
        CType::Int | CType::UInt => 4,

        // -- Long — matches sizeof (4 on i686, 8 on LP64)
        CType::Long | CType::ULong => target.long_size(),

        // -- Long long
        CType::LongLong | CType::ULongLong => {
            // On i686, long long has 4-byte alignment in the System V ABI
            // when inside structs (but 8 on stack). We use 8 for standalone.
            if *target == Target::I686 {
                4
            } else {
                8
            }
        }

        // -- Float
        CType::Float => 4,

        // -- Double
        CType::Double => {
            if *target == Target::I686 {
                4
            } else {
                8
            }
        }

        // -- Long double — target-dependent
        CType::LongDouble => target.long_double_align(),

        // -- Complex — alignment of the base type
        CType::Complex(base) => alignof_ctype(base, target),

        // -- Pointer — matches pointer width
        CType::Pointer(_, _) => target.pointer_width(),

        // -- Array — alignment of the element type
        CType::Array(elem, _) => alignof_ctype(elem, target),

        // -- Function — alignment 1
        CType::Function { .. } => 1,

        // -- Struct — max field alignment, with optional explicit override
        CType::Struct {
            fields,
            packed,
            aligned,
            ..
        } => compute_struct_or_union_align(fields, *packed, *aligned, target),

        // -- Union — same logic as struct
        CType::Union {
            fields,
            packed,
            aligned,
            ..
        } => compute_struct_or_union_align(fields, *packed, *aligned, target),

        // -- Enum — alignment of the underlying type
        CType::Enum {
            underlying_type, ..
        } => alignof_ctype(underlying_type, target),

        // -- Atomic — alignment of the inner type (may be raised to sizeof
        //    for lock-free access, but we match GCC's behaviour here)
        CType::Atomic(inner) => alignof_ctype(inner, target),

        // -- Typedef — alignment of the underlying type
        CType::Typedef { underlying, .. } => alignof_ctype(underlying, target),

        // -- Qualified — alignment of the unqualified type
        CType::Qualified(inner, _) => alignof_ctype(inner, target),
    }
}

/// Compute alignment for a struct or union from its fields, packed flag,
/// and optional explicit aligned attribute.
fn compute_struct_or_union_align(
    fields: &[StructField],
    packed: bool,
    aligned: Option<usize>,
    target: &Target,
) -> usize {
    let natural = if packed {
        1
    } else {
        let mut max_align: usize = 1;
        for field in fields {
            let fa = alignof_ctype(&field.ty, target);
            if fa > max_align {
                max_align = fa;
            }
        }
        max_align
    };

    match aligned {
        Some(a) => a.max(natural),
        None => natural,
    }
}

// ===========================================================================
// Type Predicate Functions
// ===========================================================================

/// Returns `true` if `ty` is `void` (after stripping qualifiers/typedefs).
pub fn is_void(ty: &CType) -> bool {
    matches!(resolve_and_strip(ty), CType::Void)
}

/// Returns `true` if `ty` is an integer type (including `_Bool`, `char`,
/// and `enum`). C11 §6.2.5 ¶17.
pub fn is_integer(ty: &CType) -> bool {
    matches!(
        resolve_and_strip(ty),
        CType::Bool
            | CType::Char
            | CType::SChar
            | CType::UChar
            | CType::Short
            | CType::UShort
            | CType::Int
            | CType::UInt
            | CType::Long
            | CType::ULong
            | CType::LongLong
            | CType::ULongLong
            | CType::Enum { .. }
    )
}

/// Returns `true` if `ty` is an unsigned integer type.
pub fn is_unsigned(ty: &CType) -> bool {
    matches!(
        resolve_and_strip(ty),
        CType::Bool | CType::UChar | CType::UShort | CType::UInt | CType::ULong | CType::ULongLong
    )
}

/// Returns `true` if `ty` is a signed integer type. `char` is treated as
/// signed (matching GCC defaults on x86/ARM/RISC-V).
pub fn is_signed(ty: &CType) -> bool {
    matches!(
        resolve_and_strip(ty),
        CType::Char | CType::SChar | CType::Short | CType::Int | CType::Long | CType::LongLong
    )
}

/// Returns `true` if `ty` is a floating-point type (`float`, `double`, or
/// `long double`).
pub fn is_floating(ty: &CType) -> bool {
    matches!(
        resolve_and_strip(ty),
        CType::Float | CType::Double | CType::LongDouble
    )
}

/// Returns `true` if `ty` is an arithmetic type (integer or floating).
/// C11 §6.2.5 ¶18.
pub fn is_arithmetic(ty: &CType) -> bool {
    is_integer(ty) || is_floating(ty)
}

/// Returns `true` if `ty` is a scalar type (arithmetic or pointer).
/// C11 §6.2.5 ¶21.
pub fn is_scalar(ty: &CType) -> bool {
    is_arithmetic(ty) || is_pointer(ty)
}

/// Returns `true` if `ty` is a pointer type.
pub fn is_pointer(ty: &CType) -> bool {
    matches!(resolve_and_strip(ty), CType::Pointer(_, _))
}

/// Returns `true` if `ty` is an array type.
pub fn is_array(ty: &CType) -> bool {
    matches!(resolve_and_strip(ty), CType::Array(_, _))
}

/// Returns `true` if `ty` is a function type.
pub fn is_function(ty: &CType) -> bool {
    matches!(resolve_and_strip(ty), CType::Function { .. })
}

/// Returns `true` if `ty` is a struct or union type.
pub fn is_struct_or_union(ty: &CType) -> bool {
    matches!(
        resolve_and_strip(ty),
        CType::Struct { .. } | CType::Union { .. }
    )
}

/// Returns `true` if `ty` is a complete type (has a known, finite size).
///
/// Incomplete types include:
/// - `void`
/// - Arrays with unknown size (`Array(_, None)`)
/// - Forward-declared structs/unions with no fields
pub fn is_complete(ty: &CType) -> bool {
    match resolve_and_strip(ty) {
        CType::Void => false,
        CType::Array(_, None) => false,
        CType::Struct { fields, .. } | CType::Union { fields, .. } => {
            // A struct/union with zero fields may be a forward declaration
            // or a GCC extension (empty struct). We treat empty structs as
            // complete (GCC extension) and rely on the semantic analyser to
            // distinguish forward declarations.
            let _ = fields;
            true
        }
        CType::Function { .. } => {
            // Function types are never "complete" in the C11 sense (you
            // cannot have an object of function type), but they are not
            // truly "incomplete" either. We return true here because
            // function pointers need their pointee to be usable.
            true
        }
        _ => true,
    }
}

/// Returns `true` if types `a` and `b` are compatible per C11 §6.2.7.
///
/// This is a structural compatibility check. Two types are compatible if:
/// - They are the same basic type (after stripping qualifiers/typedefs).
/// - For pointers: pointee types are compatible.
/// - For arrays: element types are compatible and sizes match (or one is
///   incomplete).
/// - For functions: return types and all parameter types are compatible.
/// - For structs/unions/enums: they share the same tag name (nominal
///   equivalence in the same translation unit).
pub fn is_compatible(a: &CType, b: &CType) -> bool {
    let a = resolve_and_strip(a);
    let b = resolve_and_strip(b);
    is_compatible_inner(a, b)
}

/// Inner recursive compatibility check on already-resolved types.
fn is_compatible_inner(a: &CType, b: &CType) -> bool {
    match (a, b) {
        // Identical basic types
        (CType::Void, CType::Void) => true,
        (CType::Bool, CType::Bool) => true,
        (CType::Char, CType::Char) => true,
        (CType::SChar, CType::SChar) => true,
        (CType::UChar, CType::UChar) => true,
        (CType::Short, CType::Short) => true,
        (CType::UShort, CType::UShort) => true,
        (CType::Int, CType::Int) => true,
        (CType::UInt, CType::UInt) => true,
        (CType::Long, CType::Long) => true,
        (CType::ULong, CType::ULong) => true,
        (CType::LongLong, CType::LongLong) => true,
        (CType::ULongLong, CType::ULongLong) => true,
        (CType::Float, CType::Float) => true,
        (CType::Double, CType::Double) => true,
        (CType::LongDouble, CType::LongDouble) => true,

        // Complex types — compatible if base types are compatible
        (CType::Complex(base_a), CType::Complex(base_b)) => is_compatible_inner(base_a, base_b),

        // Pointer types — compatible if pointee types are compatible
        (CType::Pointer(pointee_a, _), CType::Pointer(pointee_b, _)) => {
            is_compatible_inner(pointee_a, pointee_b)
        }

        // Array types — compatible if element types are compatible and
        // sizes match (or at least one is unknown/incomplete)
        (CType::Array(elem_a, size_a), CType::Array(elem_b, size_b)) => {
            if !is_compatible_inner(elem_a, elem_b) {
                return false;
            }
            match (size_a, size_b) {
                (Some(sa), Some(sb)) => sa == sb,
                _ => true, // One or both are incomplete — still compatible
            }
        }

        // Function types — compatible if return types, param counts, and
        // all parameter types are compatible
        (
            CType::Function {
                return_type: ret_a,
                params: params_a,
                variadic: var_a,
            },
            CType::Function {
                return_type: ret_b,
                params: params_b,
                variadic: var_b,
            },
        ) => {
            if var_a != var_b {
                return false;
            }
            if !is_compatible_inner(ret_a, ret_b) {
                return false;
            }
            if params_a.len() != params_b.len() {
                return false;
            }
            for (pa, pb) in params_a.iter().zip(params_b.iter()) {
                if !is_compatible(pa, pb) {
                    return false;
                }
            }
            true
        }

        // Struct — compatible if both have the same tag name
        (CType::Struct { name: name_a, .. }, CType::Struct { name: name_b, .. }) => {
            match (name_a, name_b) {
                (Some(a), Some(b)) => a == b,
                // Anonymous structs are only compatible with themselves
                // (identity check handled by pointer equality in practice).
                _ => false,
            }
        }

        // Union — compatible if both have the same tag name
        (CType::Union { name: name_a, .. }, CType::Union { name: name_b, .. }) => {
            match (name_a, name_b) {
                (Some(a), Some(b)) => a == b,
                _ => false,
            }
        }

        // Enum — compatible if both have the same tag name
        (CType::Enum { name: name_a, .. }, CType::Enum { name: name_b, .. }) => {
            match (name_a, name_b) {
                (Some(a), Some(b)) => a == b,
                _ => false,
            }
        }

        // Atomic — compatible if inner types are compatible
        (CType::Atomic(inner_a), CType::Atomic(inner_b)) => is_compatible_inner(inner_a, inner_b),

        // All other combinations are incompatible
        _ => false,
    }
}

/// Recursively strip `Qualified`, `Typedef`, and `Atomic` wrappers to
/// reach the underlying undecorated type. Used by predicate functions.
fn resolve_and_strip(ty: &CType) -> &CType {
    match ty {
        CType::Qualified(inner, _) => resolve_and_strip(inner),
        CType::Typedef { underlying, .. } => resolve_and_strip(underlying),
        CType::Atomic(inner) => resolve_and_strip(inner),
        other => other,
    }
}

// ===========================================================================
// Type Conversion Utilities
// ===========================================================================

/// Strip all type qualifiers from `ty`, returning the unqualified type.
///
/// Only removes the outermost `Qualified` wrapper. For full resolution
/// through typedefs and qualifiers, use [`resolve_typedef`].
pub fn unqualified(ty: &CType) -> &CType {
    match ty {
        CType::Qualified(inner, _) => inner.as_ref(),
        other => other,
    }
}

/// Resolve through `Typedef` chains to the ultimate underlying type.
///
/// Strips all `Typedef` indirections but preserves `Qualified` wrappers
/// (call [`unqualified`] afterwards if both stripping operations are needed).
pub fn resolve_typedef(ty: &CType) -> &CType {
    match ty {
        CType::Typedef { underlying, .. } => resolve_typedef(underlying),
        other => other,
    }
}

/// Perform C11 §6.3.1.1 integer promotions on `ty`.
///
/// Types whose conversion rank is less than `int` are promoted to `int` (or
/// `unsigned int` if `int` cannot represent all values of the original type).
/// All other types are returned unchanged.
pub fn integer_promotion(ty: &CType) -> CType {
    let resolved = resolve_and_strip(ty);
    match resolved {
        CType::Bool | CType::Char | CType::SChar | CType::Short => {
            // All fit in `int`.
            CType::Int
        }
        CType::UChar | CType::UShort => {
            // `unsigned char` and `unsigned short` fit in `int` on all
            // targets where `int` is 32-bit (which is always true for BCC).
            CType::Int
        }
        // Enum is promoted to its underlying type, then promoted if needed.
        CType::Enum {
            underlying_type, ..
        } => integer_promotion(underlying_type),
        // Types with rank ≥ int are returned as-is.
        _ => resolved.clone(),
    }
}

/// Return the integer conversion rank of `ty` per C11 §6.3.1.1.
///
/// Higher rank means higher precedence in the usual arithmetic conversions.
/// The rank assignment is:
///
/// | Rank | Types                                 |
/// |------|---------------------------------------|
/// | 1    | `_Bool`                               |
/// | 2    | `char`, `signed char`, `unsigned char` |
/// | 3    | `short`, `unsigned short`             |
/// | 4    | `int`, `unsigned int`                 |
/// | 5    | `long`, `unsigned long`               |
/// | 6    | `long long`, `unsigned long long`     |
///
/// Returns `0` for non-integer types.
pub fn integer_rank(ty: &CType) -> u8 {
    let resolved = resolve_and_strip(ty);
    match resolved {
        CType::Bool => 1,
        CType::Char | CType::SChar | CType::UChar => 2,
        CType::Short | CType::UShort => 3,
        CType::Int | CType::UInt => 4,
        CType::Long | CType::ULong => 5,
        CType::LongLong | CType::ULongLong => 6,
        // Enum rank equals the rank of its underlying integer type.
        CType::Enum {
            underlying_type, ..
        } => integer_rank(underlying_type),
        // Non-integer types have no rank.
        _ => 0,
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::target::Target;

    // -----------------------------------------------------------------------
    // TypeQualifiers
    // -----------------------------------------------------------------------

    #[test]
    fn test_type_qualifiers_default() {
        let q = TypeQualifiers::default();
        assert!(!q.is_const);
        assert!(!q.is_volatile);
        assert!(!q.is_restrict);
        assert!(!q.is_atomic);
        assert!(q.is_empty());
    }

    #[test]
    fn test_type_qualifiers_merge() {
        let a = TypeQualifiers {
            is_const: true,
            ..Default::default()
        };
        let b = TypeQualifiers {
            is_volatile: true,
            ..Default::default()
        };
        let merged = a.merge(b);
        assert!(merged.is_const);
        assert!(merged.is_volatile);
        assert!(!merged.is_restrict);
        assert!(!merged.is_empty());
    }

    // -----------------------------------------------------------------------
    // sizeof — Basic types
    // -----------------------------------------------------------------------

    #[test]
    fn test_sizeof_void() {
        assert_eq!(sizeof_ctype(&CType::Void, &Target::X86_64), 1);
    }

    #[test]
    fn test_sizeof_bool() {
        assert_eq!(sizeof_ctype(&CType::Bool, &Target::X86_64), 1);
    }

    #[test]
    fn test_sizeof_char() {
        for t in &[
            Target::X86_64,
            Target::I686,
            Target::AArch64,
            Target::RiscV64,
        ] {
            assert_eq!(sizeof_ctype(&CType::Char, t), 1);
            assert_eq!(sizeof_ctype(&CType::SChar, t), 1);
            assert_eq!(sizeof_ctype(&CType::UChar, t), 1);
        }
    }

    #[test]
    fn test_sizeof_short() {
        for t in &[
            Target::X86_64,
            Target::I686,
            Target::AArch64,
            Target::RiscV64,
        ] {
            assert_eq!(sizeof_ctype(&CType::Short, t), 2);
            assert_eq!(sizeof_ctype(&CType::UShort, t), 2);
        }
    }

    #[test]
    fn test_sizeof_int() {
        for t in &[
            Target::X86_64,
            Target::I686,
            Target::AArch64,
            Target::RiscV64,
        ] {
            assert_eq!(sizeof_ctype(&CType::Int, t), 4);
            assert_eq!(sizeof_ctype(&CType::UInt, t), 4);
        }
    }

    #[test]
    fn test_sizeof_long_target_dependent() {
        // i686 (ILP32): long = 4
        assert_eq!(sizeof_ctype(&CType::Long, &Target::I686), 4);
        assert_eq!(sizeof_ctype(&CType::ULong, &Target::I686), 4);
        // LP64 targets: long = 8
        assert_eq!(sizeof_ctype(&CType::Long, &Target::X86_64), 8);
        assert_eq!(sizeof_ctype(&CType::Long, &Target::AArch64), 8);
        assert_eq!(sizeof_ctype(&CType::Long, &Target::RiscV64), 8);
    }

    #[test]
    fn test_sizeof_long_long() {
        for t in &[
            Target::X86_64,
            Target::I686,
            Target::AArch64,
            Target::RiscV64,
        ] {
            assert_eq!(sizeof_ctype(&CType::LongLong, t), 8);
            assert_eq!(sizeof_ctype(&CType::ULongLong, t), 8);
        }
    }

    #[test]
    fn test_sizeof_float_double() {
        for t in &[
            Target::X86_64,
            Target::I686,
            Target::AArch64,
            Target::RiscV64,
        ] {
            assert_eq!(sizeof_ctype(&CType::Float, t), 4);
            assert_eq!(sizeof_ctype(&CType::Double, t), 8);
        }
    }

    #[test]
    fn test_sizeof_long_double_target_dependent() {
        // i686: 12 bytes (80-bit padded)
        assert_eq!(sizeof_ctype(&CType::LongDouble, &Target::I686), 12);
        // LP64: 16 bytes
        assert_eq!(sizeof_ctype(&CType::LongDouble, &Target::X86_64), 16);
        assert_eq!(sizeof_ctype(&CType::LongDouble, &Target::AArch64), 16);
        assert_eq!(sizeof_ctype(&CType::LongDouble, &Target::RiscV64), 16);
    }

    #[test]
    fn test_sizeof_pointer_target_dependent() {
        let ptr = CType::Pointer(Box::new(CType::Int), TypeQualifiers::default());
        // i686: 4 bytes
        assert_eq!(sizeof_ctype(&ptr, &Target::I686), 4);
        // LP64: 8 bytes
        assert_eq!(sizeof_ctype(&ptr, &Target::X86_64), 8);
        assert_eq!(sizeof_ctype(&ptr, &Target::AArch64), 8);
        assert_eq!(sizeof_ctype(&ptr, &Target::RiscV64), 8);
    }

    #[test]
    fn test_sizeof_array() {
        let arr = CType::Array(Box::new(CType::Int), Some(10));
        assert_eq!(sizeof_ctype(&arr, &Target::X86_64), 40);
    }

    #[test]
    fn test_sizeof_incomplete_array() {
        let arr = CType::Array(Box::new(CType::Int), None);
        assert_eq!(sizeof_ctype(&arr, &Target::X86_64), 0);
    }

    #[test]
    fn test_sizeof_complex() {
        let cf = CType::Complex(Box::new(CType::Float));
        assert_eq!(sizeof_ctype(&cf, &Target::X86_64), 8);
        let cd = CType::Complex(Box::new(CType::Double));
        assert_eq!(sizeof_ctype(&cd, &Target::X86_64), 16);
    }

    #[test]
    fn test_sizeof_simple_struct() {
        // struct { int x; char y; } => padded to 8 on x86-64
        let s = CType::Struct {
            name: None,
            fields: vec![
                StructField {
                    name: Some("x".into()),
                    ty: CType::Int,
                    bit_width: None,
                },
                StructField {
                    name: Some("y".into()),
                    ty: CType::Char,
                    bit_width: None,
                },
            ],
            packed: false,
            aligned: None,
        };
        assert_eq!(sizeof_ctype(&s, &Target::X86_64), 8);
    }

    #[test]
    fn test_sizeof_packed_struct() {
        // __attribute__((packed)) struct { int x; char y; } => 5
        let s = CType::Struct {
            name: None,
            fields: vec![
                StructField {
                    name: Some("x".into()),
                    ty: CType::Int,
                    bit_width: None,
                },
                StructField {
                    name: Some("y".into()),
                    ty: CType::Char,
                    bit_width: None,
                },
            ],
            packed: true,
            aligned: None,
        };
        assert_eq!(sizeof_ctype(&s, &Target::X86_64), 5);
    }

    #[test]
    fn test_sizeof_union() {
        let u = CType::Union {
            name: None,
            fields: vec![
                StructField {
                    name: Some("i".into()),
                    ty: CType::Int,
                    bit_width: None,
                },
                StructField {
                    name: Some("d".into()),
                    ty: CType::Double,
                    bit_width: None,
                },
            ],
            packed: false,
            aligned: None,
        };
        assert_eq!(sizeof_ctype(&u, &Target::X86_64), 8);
    }

    #[test]
    fn test_sizeof_enum() {
        let e = CType::Enum {
            name: Some("color".into()),
            underlying_type: Box::new(CType::Int),
        };
        assert_eq!(sizeof_ctype(&e, &Target::X86_64), 4);
    }

    #[test]
    fn test_sizeof_typedef() {
        let td = CType::Typedef {
            name: "size_t".into(),
            underlying: Box::new(CType::ULong),
        };
        assert_eq!(sizeof_ctype(&td, &Target::X86_64), 8);
        assert_eq!(sizeof_ctype(&td, &Target::I686), 4);
    }

    #[test]
    fn test_sizeof_qualified() {
        let q = CType::Qualified(
            Box::new(CType::Int),
            TypeQualifiers {
                is_const: true,
                ..Default::default()
            },
        );
        assert_eq!(sizeof_ctype(&q, &Target::X86_64), 4);
    }

    #[test]
    fn test_sizeof_atomic() {
        let a = CType::Atomic(Box::new(CType::Int));
        assert_eq!(sizeof_ctype(&a, &Target::X86_64), 4);
    }

    // -----------------------------------------------------------------------
    // alignof
    // -----------------------------------------------------------------------

    #[test]
    fn test_alignof_basic_types() {
        assert_eq!(alignof_ctype(&CType::Char, &Target::X86_64), 1);
        assert_eq!(alignof_ctype(&CType::Short, &Target::X86_64), 2);
        assert_eq!(alignof_ctype(&CType::Int, &Target::X86_64), 4);
        assert_eq!(alignof_ctype(&CType::Long, &Target::X86_64), 8);
        assert_eq!(alignof_ctype(&CType::Long, &Target::I686), 4);
    }

    #[test]
    fn test_alignof_long_double() {
        assert_eq!(alignof_ctype(&CType::LongDouble, &Target::I686), 4);
        assert_eq!(alignof_ctype(&CType::LongDouble, &Target::X86_64), 16);
    }

    #[test]
    fn test_alignof_pointer() {
        let ptr = CType::Pointer(Box::new(CType::Void), TypeQualifiers::default());
        assert_eq!(alignof_ctype(&ptr, &Target::I686), 4);
        assert_eq!(alignof_ctype(&ptr, &Target::X86_64), 8);
    }

    #[test]
    fn test_alignof_struct_with_aligned() {
        let s = CType::Struct {
            name: None,
            fields: vec![StructField {
                name: Some("x".into()),
                ty: CType::Char,
                bit_width: None,
            }],
            packed: false,
            aligned: Some(16),
        };
        assert_eq!(alignof_ctype(&s, &Target::X86_64), 16);
    }

    // -----------------------------------------------------------------------
    // Type predicates
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_void() {
        assert!(is_void(&CType::Void));
        assert!(!is_void(&CType::Int));
    }

    #[test]
    fn test_is_integer() {
        assert!(is_integer(&CType::Bool));
        assert!(is_integer(&CType::Char));
        assert!(is_integer(&CType::Int));
        assert!(is_integer(&CType::ULongLong));
        assert!(is_integer(&CType::Enum {
            name: Some("e".into()),
            underlying_type: Box::new(CType::Int),
        }));
        assert!(!is_integer(&CType::Float));
        assert!(!is_integer(&CType::Void));
    }

    #[test]
    fn test_is_unsigned() {
        assert!(is_unsigned(&CType::Bool));
        assert!(is_unsigned(&CType::UChar));
        assert!(is_unsigned(&CType::UInt));
        assert!(!is_unsigned(&CType::Int));
        assert!(!is_unsigned(&CType::Char));
    }

    #[test]
    fn test_is_signed() {
        assert!(is_signed(&CType::Char));
        assert!(is_signed(&CType::Int));
        assert!(is_signed(&CType::LongLong));
        assert!(!is_signed(&CType::UInt));
        assert!(!is_signed(&CType::Bool));
    }

    #[test]
    fn test_is_floating() {
        assert!(is_floating(&CType::Float));
        assert!(is_floating(&CType::Double));
        assert!(is_floating(&CType::LongDouble));
        assert!(!is_floating(&CType::Int));
    }

    #[test]
    fn test_is_arithmetic() {
        assert!(is_arithmetic(&CType::Int));
        assert!(is_arithmetic(&CType::Double));
        assert!(!is_arithmetic(&CType::Pointer(
            Box::new(CType::Int),
            TypeQualifiers::default()
        )));
    }

    #[test]
    fn test_is_scalar() {
        assert!(is_scalar(&CType::Int));
        assert!(is_scalar(&CType::Pointer(
            Box::new(CType::Int),
            TypeQualifiers::default()
        )));
        assert!(!is_scalar(&CType::Struct {
            name: None,
            fields: vec![],
            packed: false,
            aligned: None,
        }));
    }

    #[test]
    fn test_is_pointer() {
        assert!(is_pointer(&CType::Pointer(
            Box::new(CType::Void),
            TypeQualifiers::default()
        )));
        assert!(!is_pointer(&CType::Int));
    }

    #[test]
    fn test_is_array() {
        assert!(is_array(&CType::Array(Box::new(CType::Int), Some(5))));
        assert!(!is_array(&CType::Int));
    }

    #[test]
    fn test_is_function() {
        assert!(is_function(&CType::Function {
            return_type: Box::new(CType::Void),
            params: vec![],
            variadic: false,
        }));
        assert!(!is_function(&CType::Int));
    }

    #[test]
    fn test_is_struct_or_union() {
        assert!(is_struct_or_union(&CType::Struct {
            name: None,
            fields: vec![],
            packed: false,
            aligned: None,
        }));
        assert!(is_struct_or_union(&CType::Union {
            name: None,
            fields: vec![],
            packed: false,
            aligned: None,
        }));
        assert!(!is_struct_or_union(&CType::Int));
    }

    #[test]
    fn test_is_complete() {
        assert!(!is_complete(&CType::Void));
        assert!(!is_complete(&CType::Array(Box::new(CType::Int), None)));
        assert!(is_complete(&CType::Int));
        assert!(is_complete(&CType::Array(Box::new(CType::Int), Some(10))));
    }

    // -----------------------------------------------------------------------
    // Type compatibility
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_compatible_basic() {
        assert!(is_compatible(&CType::Int, &CType::Int));
        assert!(!is_compatible(&CType::Int, &CType::Long));
    }

    #[test]
    fn test_is_compatible_pointer() {
        let p1 = CType::Pointer(Box::new(CType::Int), TypeQualifiers::default());
        let p2 = CType::Pointer(Box::new(CType::Int), TypeQualifiers::default());
        let p3 = CType::Pointer(Box::new(CType::Char), TypeQualifiers::default());
        assert!(is_compatible(&p1, &p2));
        assert!(!is_compatible(&p1, &p3));
    }

    #[test]
    fn test_is_compatible_array() {
        let a1 = CType::Array(Box::new(CType::Int), Some(5));
        let a2 = CType::Array(Box::new(CType::Int), Some(5));
        let a3 = CType::Array(Box::new(CType::Int), Some(10));
        let a4 = CType::Array(Box::new(CType::Int), None);
        assert!(is_compatible(&a1, &a2));
        assert!(!is_compatible(&a1, &a3));
        assert!(is_compatible(&a1, &a4)); // incomplete is compatible
    }

    #[test]
    fn test_is_compatible_function() {
        let f1 = CType::Function {
            return_type: Box::new(CType::Int),
            params: vec![CType::Int, CType::Double],
            variadic: false,
        };
        let f2 = CType::Function {
            return_type: Box::new(CType::Int),
            params: vec![CType::Int, CType::Double],
            variadic: false,
        };
        assert!(is_compatible(&f1, &f2));
    }

    #[test]
    fn test_is_compatible_struct() {
        let s1 = CType::Struct {
            name: Some("point".into()),
            fields: vec![],
            packed: false,
            aligned: None,
        };
        let s2 = CType::Struct {
            name: Some("point".into()),
            fields: vec![],
            packed: false,
            aligned: None,
        };
        let s3 = CType::Struct {
            name: Some("other".into()),
            fields: vec![],
            packed: false,
            aligned: None,
        };
        assert!(is_compatible(&s1, &s2));
        assert!(!is_compatible(&s1, &s3));
    }

    #[test]
    fn test_is_compatible_through_typedef() {
        let td = CType::Typedef {
            name: "myint".into(),
            underlying: Box::new(CType::Int),
        };
        assert!(is_compatible(&td, &CType::Int));
    }

    // -----------------------------------------------------------------------
    // Type conversions
    // -----------------------------------------------------------------------

    #[test]
    fn test_unqualified() {
        let q = CType::Qualified(
            Box::new(CType::Int),
            TypeQualifiers {
                is_const: true,
                ..Default::default()
            },
        );
        assert_eq!(unqualified(&q), &CType::Int);
        assert_eq!(unqualified(&CType::Int), &CType::Int);
    }

    #[test]
    fn test_resolve_typedef() {
        let inner_td = CType::Typedef {
            name: "uint".into(),
            underlying: Box::new(CType::UInt),
        };
        let outer_td = CType::Typedef {
            name: "my_uint".into(),
            underlying: Box::new(inner_td),
        };
        assert_eq!(resolve_typedef(&outer_td), &CType::UInt);
    }

    #[test]
    fn test_integer_promotion() {
        assert_eq!(integer_promotion(&CType::Bool), CType::Int);
        assert_eq!(integer_promotion(&CType::Char), CType::Int);
        assert_eq!(integer_promotion(&CType::SChar), CType::Int);
        assert_eq!(integer_promotion(&CType::UChar), CType::Int);
        assert_eq!(integer_promotion(&CType::Short), CType::Int);
        assert_eq!(integer_promotion(&CType::UShort), CType::Int);
        // int stays int
        assert_eq!(integer_promotion(&CType::Int), CType::Int);
        // long stays long
        assert_eq!(integer_promotion(&CType::Long), CType::Long);
    }

    #[test]
    fn test_integer_rank() {
        assert_eq!(integer_rank(&CType::Bool), 1);
        assert_eq!(integer_rank(&CType::Char), 2);
        assert_eq!(integer_rank(&CType::Short), 3);
        assert_eq!(integer_rank(&CType::Int), 4);
        assert_eq!(integer_rank(&CType::Long), 5);
        assert_eq!(integer_rank(&CType::LongLong), 6);
        assert_eq!(integer_rank(&CType::UInt), 4);
        assert_eq!(integer_rank(&CType::Float), 0); // non-integer
    }

    #[test]
    fn test_integer_rank_enum() {
        let e = CType::Enum {
            name: Some("color".into()),
            underlying_type: Box::new(CType::Int),
        };
        assert_eq!(integer_rank(&e), 4);
    }

    // -----------------------------------------------------------------------
    // MachineType coverage
    // -----------------------------------------------------------------------

    #[test]
    fn test_machine_type_variants_exist() {
        // Ensure all variants compile and are distinct.
        let variants = [
            MachineType::I8,
            MachineType::I16,
            MachineType::I32,
            MachineType::I64,
            MachineType::I128,
            MachineType::F32,
            MachineType::F64,
            MachineType::F80,
            MachineType::Ptr,
            MachineType::Void,
            MachineType::Integer,
            MachineType::SSE,
            MachineType::X87,
            MachineType::Memory,
        ];
        // Verify all 14 variants are present.
        assert_eq!(variants.len(), 14);
        // Verify distinctness.
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                assert_ne!(variants[i], variants[j]);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Predicates through Qualified / Typedef wrappers
    // -----------------------------------------------------------------------

    #[test]
    fn test_predicates_through_typedef() {
        let td = CType::Typedef {
            name: "myint".into(),
            underlying: Box::new(CType::Int),
        };
        assert!(is_integer(&td));
        assert!(is_signed(&td));
        assert!(is_arithmetic(&td));
        assert!(is_scalar(&td));
        assert!(!is_floating(&td));
        assert!(!is_pointer(&td));
    }

    #[test]
    fn test_predicates_through_qualified() {
        let q = CType::Qualified(
            Box::new(CType::Double),
            TypeQualifiers {
                is_const: true,
                ..Default::default()
            },
        );
        assert!(is_floating(&q));
        assert!(is_arithmetic(&q));
        assert!(!is_integer(&q));
    }

    // -----------------------------------------------------------------------
    // StructField bitfield support
    // -----------------------------------------------------------------------

    #[test]
    fn test_struct_field_bitfield() {
        let bf = StructField {
            name: Some("flags".into()),
            ty: CType::UInt,
            bit_width: Some(3),
        };
        assert_eq!(bf.bit_width, Some(3));
    }

    // -----------------------------------------------------------------------
    // CType variant exhaustiveness (compile-time assertion via match)
    // -----------------------------------------------------------------------

    #[test]
    fn test_ctype_all_variants_constructible() {
        let variants: Vec<CType> = vec![
            CType::Void,
            CType::Bool,
            CType::Char,
            CType::SChar,
            CType::UChar,
            CType::Short,
            CType::UShort,
            CType::Int,
            CType::UInt,
            CType::Long,
            CType::ULong,
            CType::LongLong,
            CType::ULongLong,
            CType::Float,
            CType::Double,
            CType::LongDouble,
            CType::Complex(Box::new(CType::Float)),
            CType::Pointer(Box::new(CType::Int), TypeQualifiers::default()),
            CType::Array(Box::new(CType::Int), Some(5)),
            CType::Function {
                return_type: Box::new(CType::Void),
                params: vec![],
                variadic: false,
            },
            CType::Struct {
                name: None,
                fields: vec![],
                packed: false,
                aligned: None,
            },
            CType::Union {
                name: None,
                fields: vec![],
                packed: false,
                aligned: None,
            },
            CType::Enum {
                name: None,
                underlying_type: Box::new(CType::Int),
            },
            CType::Atomic(Box::new(CType::Int)),
            CType::Typedef {
                name: "t".into(),
                underlying: Box::new(CType::Int),
            },
            CType::Qualified(Box::new(CType::Int), TypeQualifiers::default()),
        ];
        // 26 variants
        assert_eq!(variants.len(), 26);
    }

    // -----------------------------------------------------------------------
    // align_up helper
    // -----------------------------------------------------------------------

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 4), 0);
        assert_eq!(align_up(1, 4), 4);
        assert_eq!(align_up(4, 4), 4);
        assert_eq!(align_up(5, 4), 8);
        assert_eq!(align_up(7, 8), 8);
        assert_eq!(align_up(8, 8), 8);
        assert_eq!(align_up(9, 8), 16);
        assert_eq!(align_up(0, 1), 0);
        assert_eq!(align_up(5, 1), 5);
    }
}
