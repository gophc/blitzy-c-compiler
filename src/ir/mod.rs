//! # BCC Intermediate Representation
//!
//! This module defines BCC's intermediate representation (IR) and the transformations
//! that operate on it.
//!
//! ## Core IR Definitions
//! - [`types`] — IR type system (Void, I1–I128, F32/F64/F80, Ptr, Array, Struct, Function)

// Core IR definitions
pub mod types;

// Re-export core IR types for convenient access
pub use types::IrType;
pub use types::StructType;
