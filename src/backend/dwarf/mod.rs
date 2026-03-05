//! # DWARF v4 Debug Information Generation
//!
//! This module generates DWARF v4 debug sections for BCC output binaries
//! when the `-g` flag is specified. It replaces the external `gimli` crate.
//!
//! ## Generated Sections
//! - `.debug_info` — Compilation unit, subprogram, and variable DIEs
//! - `.debug_abbrev` — Abbreviation table encoding
//! - `.debug_line` — Line number program for source mapping
//! - `.debug_str` — String table for debug names

pub mod abbrev;
pub mod str;
