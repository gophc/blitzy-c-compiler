//! Common infrastructure for BCC.
//!
//! This module provides foundational types and utilities used by every layer
//! of the compiler: frontend, IR, optimization passes, and backend.
//! It has no dependencies on other BCC modules.

pub mod encoding;
pub mod fx_hash;
pub mod long_double;
pub mod source_map;
pub mod target;
pub mod temp_files;
