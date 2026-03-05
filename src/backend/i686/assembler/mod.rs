//! # Built-in i686 Assembler
//!
//! This module provides the built-in assembler for the i686 (32-bit x86)
//! backend. It encodes machine instructions from `MachineFunction` into
//! raw bytes suitable for inclusion in ELF object files.
//!
//! ## Submodules
//! - [`relocations`] — i686 ELF relocation type definitions (`R_386_*`)

pub mod relocations;
