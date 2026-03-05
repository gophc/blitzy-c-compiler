//! # AArch64 Built-in ELF Linker
//!
//! Built-in ELF linker for the AArch64 architecture, producing ET_EXEC (static
//! executables) and ET_DYN (shared objects). This is part of BCC's standalone
//! backend — NO external linker is invoked.

pub mod relocations;
