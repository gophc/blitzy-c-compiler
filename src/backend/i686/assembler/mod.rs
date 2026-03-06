//! # Built-in i686 Assembler
//!
//! This module provides the built-in assembler for the i686 (32-bit x86)
//! backend. It encodes machine instructions from `MachineFunction` into
//! raw bytes suitable for inclusion in ELF object files.
//!
//! ## Submodules
//! - [`encoder`] — i686 instruction encoder (ModR/M, SIB, opcode emission)
//! - [`relocations`] — i686 ELF relocation type definitions (`R_386_*`)

pub mod encoder;
pub mod relocations;
