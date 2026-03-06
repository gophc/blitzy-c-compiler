//! # Built-in AArch64 Assembler
//!
//! Self-contained assembler for the AArch64 (ARM 64-bit) architecture.
//! Converts A64 machine instructions into binary machine code and produces
//! ELF relocatable object sections.
//!
//! ## Architecture
//! - Accepts `A64Instruction` sequences from `codegen.rs`
//! - Dispatches to `encoder` for A64 instruction format binary encoding
//! - Collects `relocations` for unresolved symbol references
//! - Produces `.text` section bytes + relocation entries
//!
//! ## Key Characteristic: Fixed 32-bit Instruction Width
//! Every AArch64 instruction encodes to exactly 4 bytes. There is no
//! variable-length encoding, simplifying offset computation and branch targeting.
//!
//! ## Standalone Backend Mode
//! No external assembler is invoked. This module, together with `encoder.rs` and
//! `relocations.rs`, is entirely self-contained per BCC's zero-dependency mandate.

pub mod encoder;
pub mod relocations;
