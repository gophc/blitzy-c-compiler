//! # RISC-V 64 ELF Linker
//!
//! Built-in RISC-V 64 linker producing ET_EXEC and ET_DYN ELF binaries.
//! Handles RISC-V specific relocation application with linker relaxation support.

pub mod relocations;
