//! # BCC Linker Common — Shared Linker Infrastructure
//!
//! This module provides architecture-agnostic linker infrastructure shared by all
//! four target architecture linkers (x86-64, i686, AArch64, RISC-V 64).
//!
//! ## Standalone Backend Mode
//! BCC includes its own built-in linker. NO external linker is invoked
//! (no `ld`, `lld`, `gold`). This module implements the core linker logic.
//!
//! ## Submodules
//! - [`dynamic`] — Dynamic linking section generation (.dynamic, .dynsym, .dynstr,
//!   .gnu.hash, .got, .got.plt, .plt, .rela.dyn, .rela.plt, PT_INTERP, PT_DYNAMIC).
//! - [`section_merger`] — Input section aggregation from multiple object files into output
//!   sections, with alignment padding, COMDAT group deduplication, and standard section ordering.

pub mod dynamic;
pub mod section_merger;
