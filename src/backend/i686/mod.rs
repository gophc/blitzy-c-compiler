//! # BCC i686 (32-bit x86) Backend
//!
//! This module implements the i686 architecture backend for BCC, targeting
//! 32-bit x86 Linux systems with the ILP32 data model.
//!
//! ## Architecture Characteristics
//! - **8 General-Purpose Registers**: EAX, ECX, EDX, EBX, ESP, EBP, ESI, EDI
//! - **x87 FPU**: Stack-based floating-point (ST(0)–ST(7))
//! - **ILP32 Data Model**: `int`, `long`, and pointers are all 32-bit (4 bytes)
//! - **cdecl Calling Convention**: ALL function arguments passed on the stack
//! - **32-bit Instruction Encoding**: No REX prefix, purely IA-32 ISA
//!
//! ## Submodules
//! - [`registers`] — Register definitions (GPRs, x87 FPU stack)

pub mod registers;
