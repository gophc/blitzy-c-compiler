//! # x86-64 Built-in ELF Linker
//!
//! Built-in ELF linker for the x86-64 architecture, producing ET_EXEC (static
//! executables) and ET_DYN (shared objects) with full GOT/PLT support and
//! GOTPCRELX relaxation.  This is part of BCC's standalone backend — NO
//! external linker is invoked.

pub mod relocations;
