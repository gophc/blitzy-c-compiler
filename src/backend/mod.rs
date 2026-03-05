//! # Code Generation Backend
//!
//! Phase 10 of the BCC compilation pipeline. Provides architecture-dispatching
//! code generation, register allocation, ELF writing, DWARF debug info,
//! shared linker infrastructure, and four architecture-specific backends.

pub mod traits;

pub mod aarch64;
pub mod dwarf;
pub mod elf_writer_common;
pub mod i686;
pub mod linker_common;
pub mod register_allocator;
pub mod riscv64;
pub mod x86_64;
