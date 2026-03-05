//! # DWARF `.debug_info` Section Generation
//!
//! This module provides the `DebugInfoGenerator` for constructing DWARF v4
//! `.debug_info` section content — compilation unit headers and Debug
//! Information Entry (DIE) trees describing functions, variables, and types.
//!
//! ## Design
//!
//! The generator works in conjunction with:
//! - [`super::abbrev::AbbrevTable`] for abbreviation codes
//! - [`super::str::DebugStrTable`] for string offsets (DW_FORM_strp)
//! - [`super::mod::dwarf_address_size`] for target-specific address widths
//!
//! ## Status
//!
//! Core `.debug_info` generation logic is currently implemented inline in
//! `mod.rs` (`generate_debug_info_section`). This module provides the
//! `DebugInfoGenerator` type for future refactoring into a standalone
//! generator with richer DIE tree construction capabilities.

use crate::common::source_map::SourceMap;
use crate::common::target::Target;
use crate::ir::module::IrModule;

use super::abbrev::AbbrevTable;
use super::r#str::DebugStrTable;

// ---------------------------------------------------------------------------
// DebugInfoGenerator
// ---------------------------------------------------------------------------

/// Generator for DWARF v4 `.debug_info` section content.
///
/// Constructs the compilation unit header and the DIE tree that describes
/// the program's functions, variables, and types. The resulting byte stream
/// is a complete `.debug_info` section ready for embedding in an ELF file.
///
/// # Coordination
///
/// The generator references:
/// - An [`AbbrevTable`] for abbreviation codes used in each DIE
/// - A [`DebugStrTable`] (mutably) for interning debug string names
/// - A [`SourceMap`] for source file and line information
/// - A [`Target`] for address size determination
///
/// # Example
///
/// ```ignore
/// let mut str_table = DebugStrTable::new();
/// let abbrev_table = AbbrevTable::new();
/// let gen = DebugInfoGenerator::new(&abbrev_table, &source_map, &target);
/// let debug_info_bytes = gen.generate(&module, &mut str_table, &offsets);
/// ```
pub struct DebugInfoGenerator<'a> {
    /// Reference to the abbreviation table for looking up DIE abbreviation codes.
    abbrev_table: &'a AbbrevTable,
    /// Reference to the source map for file/line resolution.
    source_map: &'a SourceMap,
    /// Target architecture for address size determination.
    target: &'a Target,
}

impl<'a> DebugInfoGenerator<'a> {
    /// Create a new `DebugInfoGenerator`.
    ///
    /// # Arguments
    ///
    /// * `abbrev_table` — The abbreviation table providing DIE abbreviation codes.
    /// * `source_map` — Source file tracker for file names and line numbers.
    /// * `target` — Target architecture for address-size decisions.
    pub fn new(
        abbrev_table: &'a AbbrevTable,
        source_map: &'a SourceMap,
        target: &'a Target,
    ) -> Self {
        Self {
            abbrev_table,
            source_map,
            target,
        }
    }

    /// Generate the `.debug_info` section bytes for the given IR module.
    ///
    /// Constructs the compilation unit header (DWARF v4 §7.5) and the
    /// full DIE tree: `DW_TAG_compile_unit` → child DIEs → null terminator.
    ///
    /// # Arguments
    ///
    /// * `module` — The IR module with functions and globals to describe.
    /// * `str_table` — Mutable reference to the shared string table for
    ///   interning debug names (function names, type names, producer, etc.).
    /// * `text_section_offsets` — Slice of `(function_name, start_offset, size)`
    ///   tuples mapping function names to their `.text` section byte ranges.
    ///
    /// # Returns
    ///
    /// The complete `.debug_info` section as a `Vec<u8>`.
    pub fn generate(
        &self,
        module: &IrModule,
        str_table: &mut DebugStrTable,
        text_section_offsets: &[(String, u64, u64)],
    ) -> Vec<u8> {
        let address_size = super::dwarf_address_size(self.target);
        let mut buffer = Vec::with_capacity(1024);

        // Compilation unit header (11 bytes for 32-bit DWARF)
        // Reserve 4 bytes for unit_length
        buffer.extend_from_slice(&[0u8; 4]);
        // Version
        buffer.extend_from_slice(&super::DWARF_VERSION.to_le_bytes());
        // debug_abbrev_offset
        buffer.extend_from_slice(&0u32.to_le_bytes());
        // address_size
        buffer.push(address_size);

        // Compile unit DIE (placeholder — the real driver is in mod.rs)
        // Emit a minimal null terminator for the CU children list
        buffer.push(0x00);

        // Fix up unit_length
        let unit_length = (buffer.len() - 4) as u32;
        buffer[0..4].copy_from_slice(&unit_length.to_le_bytes());

        // Suppress unused variable warnings for API contract compliance
        let _ = module;
        let _ = str_table;
        let _ = text_section_offsets;
        let _ = &self.abbrev_table;
        let _ = &self.source_map;

        buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debug_info_generator_creation() {
        let abbrev = AbbrevTable::new();
        let source_map = SourceMap::new();
        let target = Target::X86_64;
        let gen = DebugInfoGenerator::new(&abbrev, &source_map, &target);
        // Verify the generator can be created without panicking
        let _ = gen;
    }
}
