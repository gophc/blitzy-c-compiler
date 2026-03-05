//! # IR Lowering — Phase 6
//!
//! This module implements Phase 6 of the BCC compilation pipeline: lowering the
//! semantically-validated, type-annotated AST into BCC's intermediate representation (IR).

pub mod asm_lowering;
pub mod expr_lowering;
