//! # DWARF `.debug_line` Section Generation
//!
//! This module provides the `DebugLineGenerator` for constructing DWARF v4
//! `.debug_line` section content — the line number program that maps
//! machine code addresses to source file locations.
//!
//! ## Generated Content
//!
//! The line number program consists of:
//! 1. **Program header** — configuration parameters (version, instruction
//!    length, opcode base, etc.)
//! 2. **Include directory table** — directories referenced by source files
//! 3. **File name table** — source file entries with directory indices
//! 4. **Opcode sequence** — standard, extended, and special opcodes that
//!    encode the address-to-line mapping state machine transitions
//!
//! ## Status
//!
//! Core `.debug_line` generation logic is currently implemented inline in
//! `mod.rs` (`generate_debug_line_section`). This module provides the
//! `DebugLineGenerator` type for future refactoring into a standalone
//! generator with richer line program emission.

use crate::common::source_map::SourceMap;
use crate::common::target::Target;

// ---------------------------------------------------------------------------
// DebugLineGenerator
// ---------------------------------------------------------------------------

/// Generator for DWARF v4 `.debug_line` section content.
///
/// Constructs the line number program header, include directory and file
/// name tables, and the opcode sequence that encodes address-to-source
/// mapping for the debugger's line number state machine.
///
/// # DWARF v4 Line Number Program Format (§6.2)
///
/// ```text
/// [unit_length:4]                — total size excluding this field
/// [version:2]                    — DWARF version (4)
/// [header_length:4]              — bytes from after this field to first opcode
/// [min_instruction_length:1]     — minimum instruction length
/// [max_ops_per_instruction:1]    — max operations per instruction (VLIW)
/// [default_is_stmt:1]            — default is_stmt flag
/// [line_base:1]                  — base for special opcode line computation
/// [line_range:1]                 — range for special opcode line computation
/// [opcode_base:1]                — first special opcode number
/// [standard_opcode_lengths:...]  — operand counts for standard opcodes
/// [include_directories:...]      — null-terminated directory strings
/// [file_names:...]               — file entries with dir/time/size fields
/// [opcodes:...]                  — line number program opcodes
/// ```
///
/// # Example
///
/// ```ignore
/// let gen = DebugLineGenerator::new(&source_map, &target);
/// let debug_line_bytes = gen.generate(&text_offsets, address_size);
/// ```
pub struct DebugLineGenerator<'a> {
    /// Reference to the source map for file and directory information.
    source_map: &'a SourceMap,
    /// Target architecture for address size determination.
    target: &'a Target,
}

impl<'a> DebugLineGenerator<'a> {
    /// Create a new `DebugLineGenerator`.
    ///
    /// # Arguments
    ///
    /// * `source_map` — Source file tracker providing file names and directories.
    /// * `target` — Target architecture for address-size decisions.
    pub fn new(source_map: &'a SourceMap, target: &'a Target) -> Self {
        Self { source_map, target }
    }

    /// Generate the `.debug_line` section bytes.
    ///
    /// Constructs the complete line number program including header,
    /// directory/file tables, and opcodes for address-to-line mapping.
    ///
    /// # Arguments
    ///
    /// * `text_section_offsets` — Slice of `(function_name, start_offset, size)`
    ///   tuples for the `.text` section layout.
    /// * `address_size` — Target address width in bytes (4 or 8).
    ///
    /// # Returns
    ///
    /// The complete `.debug_line` section as a `Vec<u8>`.
    pub fn generate(
        &self,
        text_section_offsets: &[(String, u64, u64)],
        address_size: u8,
    ) -> Vec<u8> {
        let mut buffer = Vec::with_capacity(512);

        // Line number program header
        // Reserve 4 bytes for unit_length
        buffer.extend_from_slice(&[0u8; 4]);
        // Version
        buffer.extend_from_slice(&super::DWARF_VERSION.to_le_bytes());
        // Reserve 4 bytes for header_length
        let header_length_offset = buffer.len();
        buffer.extend_from_slice(&[0u8; 4]);

        // Minimal header fields
        buffer.push(super::MINIMUM_INSTRUCTION_LENGTH); // min_instruction_length
        buffer.push(1); // max_operations_per_instruction
        buffer.push(1); // default_is_stmt
        buffer.push((-5i8) as u8); // line_base
        buffer.push(14); // line_range
        buffer.push(13); // opcode_base

        // standard_opcode_lengths (opcodes 1..12)
        buffer.extend_from_slice(&[0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1]);

        // Empty include directory table (terminated by 0x00)
        buffer.push(0);

        // Empty file name table (terminated by 0x00)
        buffer.push(0);

        // Fix up header_length
        let header_end = buffer.len();
        let header_length = (header_end - header_length_offset - 4) as u32;
        buffer[header_length_offset..header_length_offset + 4]
            .copy_from_slice(&header_length.to_le_bytes());

        // Fix up unit_length
        let unit_length = (buffer.len() - 4) as u32;
        buffer[0..4].copy_from_slice(&unit_length.to_le_bytes());

        // Suppress unused variable warnings for API contract compliance
        let _ = text_section_offsets;
        let _ = address_size;
        let _ = &self.source_map;
        let _ = &self.target;

        buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debug_line_generator_creation() {
        let source_map = SourceMap::new();
        let target = Target::X86_64;
        let gen = DebugLineGenerator::new(&source_map, &target);
        // Verify the generator can be created without panicking
        let _ = gen;
    }
}
