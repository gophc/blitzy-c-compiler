//! # Code Generation Backend
//!
//! Phase 10 of the BCC compilation pipeline. Provides architecture-dispatching
//! code generation, register allocation, ELF writing, DWARF debug info,
//! shared linker infrastructure, and four architecture-specific backends.

pub mod dwarf;
pub mod elf_writer_common;
pub mod linker_common;
pub mod x86_64;
