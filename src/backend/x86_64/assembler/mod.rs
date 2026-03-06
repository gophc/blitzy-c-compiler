//! x86-64 built-in assembler — encodes machine instructions into relocatable
//! ELF `.o` object files.
//!
//! This module serves as the assembler driver:
//! 1. Receives `MachineFunction` from x86-64 instruction selection
//! 2. Iterates machine instructions, dispatching to the encoder
//! 3. Collects relocations for external symbol references
//! 4. Produces ELF `.o` files with `.text`, `.data`, `.bss`, `.rodata`,
//!    `.symtab`, `.rela.text`
//!
//! No external assembler (`as`, `nasm`, `llvm-mc`) is invoked — everything
//! is built-in.

pub mod encoder;
pub mod relocations;
