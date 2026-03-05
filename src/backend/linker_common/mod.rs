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
//! - [`symbol_resolver`] — Two-pass symbol resolution: collect all symbols from input objects,
//!   resolve references using strong/weak binding rules, report undefined symbol errors,
//!   and handle archive scanning for static libraries.
//! - [`section_merger`] — Input section aggregation from multiple object files into output
//!   sections, with alignment padding, COMDAT group deduplication, and standard section ordering
//!   (`.text`, `.rodata`, `.data`, `.bss`).
//! - [`relocation`] — Architecture-agnostic relocation processing framework: collects relocations
//!   from input objects, resolves targets, and dispatches application to architecture-specific
//!   handlers via the `RelocationHandler` trait.
//! - [`dynamic`] — Dynamic linking section generation for shared objects (ET_DYN):
//!   `.dynamic`, `.dynsym`, `.dynstr`, `.rela.dyn`, `.rela.plt`, `.gnu.hash`,
//!   `.got`, `.got.plt`, `.plt` stub generation, `PT_DYNAMIC`, `PT_INTERP`.
//! - [`linker_script`] — Default section-to-segment mapping: `.text` → `PT_LOAD (R+X)`,
//!   `.rodata` → `PT_LOAD (R)`, `.data`/`.bss` → `PT_LOAD (R+W)`, entry point (`_start`),
//!   `PT_PHDR`, `PT_GNU_STACK`.
//!
//! ## Supported Output Formats
//! - `ET_EXEC` — Static ELF executables
//! - `ET_DYN` — Shared objects (`.so` files)
//!
//! ## Architecture-Specific Delegation
//! Architecture-specific relocation application is delegated to each backend's
//! `linker/relocations.rs` module via the [`relocation::RelocationHandler`] trait:
//! - `src/backend/x86_64/linker/relocations.rs`
//! - `src/backend/i686/linker/relocations.rs`
//! - `src/backend/aarch64/linker/relocations.rs`
//! - `src/backend/riscv64/linker/relocations.rs`

// ============================================================================
// Submodule declarations — exactly 5 public submodules
// ============================================================================

/// Two-pass symbol resolution engine with strong/weak binding rules, archive
/// scanning, undefined symbol detection, and final symbol table generation.
pub mod symbol_resolver;

/// Input section aggregation from multiple object files into output sections,
/// with alignment padding, COMDAT group deduplication, and standard section ordering.
pub mod section_merger;

/// Architecture-agnostic relocation processing framework: collection,
/// classification, resolution, and architecture-dispatched application.
pub mod relocation;

/// Dynamic linking section generation for shared objects (ET_DYN):
/// `.dynamic`, `.dynsym`, `.dynstr`, `.gnu.hash`, `.got`, `.got.plt`, `.plt`,
/// `.rela.dyn`, `.rela.plt`, `PT_DYNAMIC`, `PT_INTERP`.
pub mod dynamic;

/// Default section-to-segment mapping and address layout computation for
/// both `ET_EXEC` (static executables) and `ET_DYN` (shared objects).
pub mod linker_script;

// ============================================================================
// Re-exports — key types for convenient access by architecture backends
// ============================================================================

pub use dynamic::{
    DynamicLinkContext, DynamicSymbol, DynamicSymbolTable, GlobalOffsetTable, ProcedureLinkageTable,
};
pub use linker_script::{DefaultLinkerScript, LayoutResult, SectionLayout, SegmentLayout};
pub use relocation::{
    RelocCategory, Relocation, RelocationError, RelocationHandler, ResolvedRelocation,
};
pub use section_merger::{InputSection, OutputSection, SectionFragment, SectionMerger};
pub use symbol_resolver::{InputSymbol, OutputSymbol, ResolvedSymbol, SymbolResolver, SymbolTable};

// ============================================================================
// Imports from crate::common
// ============================================================================

use crate::common::diagnostics::{Diagnostic, DiagnosticEngine, Span};
use crate::common::fx_hash::FxHashMap;
use crate::common::target::Target;

use crate::backend::elf_writer_common::{ElfSymbol, ElfWriter, Section, ET_DYN, ET_EXEC};

// ============================================================================
// OutputType — linker output format selection
// ============================================================================

/// Represents the output type for linking.
///
/// Maps directly to ELF object types:
/// - [`Executable`](OutputType::Executable) → `ET_EXEC`
/// - [`SharedLibrary`](OutputType::SharedLibrary) → `ET_DYN`
/// - [`Relocatable`](OutputType::Relocatable) → `ET_REL` (passthrough, no linking)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputType {
    /// ET_EXEC — Static ELF executable.
    Executable,
    /// ET_DYN — Shared object (`.so`).
    SharedLibrary,
    /// ET_REL — Relocatable object (`.o`) — passthrough, no linking needed.
    Relocatable,
}

impl OutputType {
    /// Convert to the corresponding ELF `e_type` header value.
    ///
    /// Returns the ELF type constant for `ET_EXEC`, `ET_DYN`, or `ET_REL`.
    #[inline]
    pub fn to_elf_type(self) -> u16 {
        match self {
            OutputType::Executable => ET_EXEC,
            OutputType::SharedLibrary => ET_DYN,
            OutputType::Relocatable => crate::backend::elf_writer_common::ET_REL,
        }
    }

    /// Returns `true` if this output type requires position-independent code.
    #[inline]
    pub fn requires_pic(self) -> bool {
        matches!(self, OutputType::SharedLibrary)
    }

    /// Returns `true` if this output type produces a linked (non-relocatable) binary.
    #[inline]
    pub fn is_linked(self) -> bool {
        !matches!(self, OutputType::Relocatable)
    }
}

impl std::fmt::Display for OutputType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutputType::Executable => write!(f, "executable"),
            OutputType::SharedLibrary => write!(f, "shared library"),
            OutputType::Relocatable => write!(f, "relocatable"),
        }
    }
}

// ============================================================================
// LinkerConfig — configuration collected from CLI flags
// ============================================================================

/// Linker configuration options collected from CLI flags.
///
/// Aggregates all linker-relevant settings parsed from the BCC command line,
/// including target architecture, output type, library search paths, and
/// flags controlling PIC mode, debug emission, and security features.
///
/// # Example
///
/// ```ignore
/// use bcc::backend::linker_common::{LinkerConfig, OutputType};
/// use bcc::common::target::Target;
///
/// let config = LinkerConfig {
///     target: Target::X86_64,
///     output_type: OutputType::Executable,
///     output_path: "a.out".to_string(),
///     entry_point: "_start".to_string(),
///     library_paths: vec!["/usr/lib".to_string()],
///     libraries: vec!["c".to_string()],
///     pic: false,
///     allow_undefined: false,
///     soname: None,
///     needed_libs: Vec::new(),
///     emit_debug: false,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct LinkerConfig {
    /// Target architecture (x86-64, i686, AArch64, or RISC-V 64).
    pub target: Target,

    /// Output type (executable, shared library, or relocatable).
    pub output_type: OutputType,

    /// Output file path (e.g., `"a.out"`, `"libfoo.so"`).
    pub output_path: String,

    /// Entry point symbol name. Defaults to `"_start"` for executables.
    /// Ignored for shared libraries and relocatable objects.
    pub entry_point: String,

    /// Library search paths (`-L` flags).
    pub library_paths: Vec<String>,

    /// Libraries to link (`-l` flags — names without the `lib` prefix).
    pub libraries: Vec<String>,

    /// Whether to generate position-independent code (`-fPIC`).
    pub pic: bool,

    /// Whether to allow undefined symbols at link time.
    /// Typically `true` for shared libraries, `false` for executables.
    pub allow_undefined: bool,

    /// SONAME for shared libraries (`-soname` / `DT_SONAME`).
    /// `None` if no SONAME was specified.
    pub soname: Option<String>,

    /// Additional needed libraries (names for `DT_NEEDED` entries).
    pub needed_libs: Vec<String>,

    /// Whether to emit DWARF debug sections in the output.
    pub emit_debug: bool,
}

impl LinkerConfig {
    /// Create a minimal default configuration for the given target and output type.
    ///
    /// All paths are empty and optional fields default to `None` / `false`.
    pub fn new(target: Target, output_type: OutputType) -> Self {
        LinkerConfig {
            target,
            output_type,
            output_path: String::from("a.out"),
            entry_point: String::from("_start"),
            library_paths: Vec::new(),
            libraries: Vec::new(),
            pic: output_type.requires_pic(),
            allow_undefined: matches!(output_type, OutputType::SharedLibrary),
            soname: None,
            needed_libs: Vec::new(),
            emit_debug: false,
        }
    }

    /// Returns `true` if the output is a shared library.
    #[inline]
    pub fn is_shared(&self) -> bool {
        self.output_type == OutputType::SharedLibrary
    }

    /// Returns `true` if the output is an executable.
    #[inline]
    pub fn is_executable(&self) -> bool {
        self.output_type == OutputType::Executable
    }

    /// Returns `true` if the output is a relocatable object (no linking).
    #[inline]
    pub fn is_relocatable(&self) -> bool {
        self.output_type == OutputType::Relocatable
    }
}

impl Default for LinkerConfig {
    fn default() -> Self {
        Self::new(Target::X86_64, OutputType::Executable)
    }
}

// ============================================================================
// LinkerInput — parsed input object file ready for linking
// ============================================================================

/// Represents a parsed input object file ready for linking.
///
/// Each `LinkerInput` corresponds to one `.o` file (or archive member)
/// provided to the linker. It carries the object's sections, symbols, and
/// relocations in the common representation used by the linker infrastructure.
#[derive(Debug)]
pub struct LinkerInput {
    /// Unique identifier for this input object. Used for provenance tracking
    /// in error messages and for correlating symbols back to their origin.
    pub object_id: u32,

    /// Original file name or archive member name.
    pub filename: String,

    /// Input sections extracted from this object file.
    pub sections: Vec<section_merger::InputSection>,

    /// Symbols defined or referenced by this object file.
    pub symbols: Vec<symbol_resolver::InputSymbol>,

    /// Relocations from this object file.
    pub relocations: Vec<relocation::Relocation>,
}

impl LinkerInput {
    /// Create a new `LinkerInput` with the given object ID and filename.
    ///
    /// Sections, symbols, and relocations are initially empty and should
    /// be populated by the object file parser.
    pub fn new(object_id: u32, filename: String) -> Self {
        LinkerInput {
            object_id,
            filename,
            sections: Vec::new(),
            symbols: Vec::new(),
            relocations: Vec::new(),
        }
    }

    /// Returns the number of input sections in this object.
    #[inline]
    pub fn section_count(&self) -> usize {
        self.sections.len()
    }

    /// Returns the number of symbols in this object.
    #[inline]
    pub fn symbol_count(&self) -> usize {
        self.symbols.len()
    }

    /// Returns the number of relocations in this object.
    #[inline]
    pub fn relocation_count(&self) -> usize {
        self.relocations.len()
    }
}

// ============================================================================
// LinkerOutput — final linked output
// ============================================================================

/// Represents the final linked output — a complete ELF binary.
///
/// Contains the serialized ELF byte stream, the resolved entry point address,
/// and the output type for downstream processing (e.g., writing to disk,
/// verification).
pub struct LinkerOutput {
    /// The complete ELF binary as a byte vector.
    ///
    /// This is a valid ELF file that can be written directly to disk,
    /// inspected with `readelf`/`objdump`, and (for executables) executed.
    pub elf_data: Vec<u8>,

    /// Resolved entry point virtual address.
    ///
    /// For executables, this is the address of the `_start` symbol (or
    /// whatever symbol was specified as the entry point). For shared
    /// libraries, this is typically 0.
    pub entry_point: u64,

    /// The output type that was produced.
    pub output_type: OutputType,
}

impl LinkerOutput {
    /// Returns the size of the ELF output in bytes.
    #[inline]
    pub fn size(&self) -> usize {
        self.elf_data.len()
    }

    /// Returns `true` if the output is an executable.
    #[inline]
    pub fn is_executable(&self) -> bool {
        self.output_type == OutputType::Executable
    }

    /// Returns `true` if the output is a shared library.
    #[inline]
    pub fn is_shared_library(&self) -> bool {
        self.output_type == OutputType::SharedLibrary
    }
}

impl std::fmt::Debug for LinkerOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LinkerOutput")
            .field("elf_data_len", &self.elf_data.len())
            .field("entry_point", &format_args!("0x{:x}", self.entry_point))
            .field("output_type", &self.output_type)
            .finish()
    }
}

// ============================================================================
// link() — High-Level Linker Driver
// ============================================================================

/// Perform the complete link operation.
///
/// This function orchestrates the full linking pipeline:
///
/// 1. **Symbol Resolution** — Collect all symbols from input objects, apply
///    strong/weak binding rules, detect multiple definitions, and report
///    undefined symbols.
/// 2. **Section Merging** — Aggregate input sections into output sections,
///    apply alignment padding, and deduplicate COMDAT groups.
/// 3. **Address Layout** — Compute virtual addresses and file offsets for all
///    output sections using the default linker script.
/// 4. **Relocation Processing** — Resolve and apply all relocations using the
///    architecture-specific `RelocationHandler`.
/// 5. **Dynamic Section Generation** (if `ET_DYN`) — Build `.dynamic`,
///    `.dynsym`, `.dynstr`, `.gnu.hash`, `.got`, `.got.plt`, `.plt`,
///    `.rela.dyn`, `.rela.plt`, and `PT_INTERP`.
/// 6. **ELF Output Writing** — Serialize the linked binary using `ElfWriter`.
///
/// # Arguments
///
/// * `config` — Linker configuration (target, output type, paths, flags).
/// * `inputs` — List of parsed input object files to link.
/// * `handler` — Architecture-specific relocation handler (trait object).
/// * `diagnostics` — Diagnostic engine for error/warning reporting.
///
/// # Returns
///
/// On success, returns a [`LinkerOutput`] containing the complete ELF binary.
/// On failure, returns a descriptive error string.
///
/// # Errors
///
/// Returns `Err` if:
/// - There are undefined symbol errors (and `allow_undefined` is false)
/// - Relocation processing encounters overflow or unsupported types
/// - Section merging fails due to incompatible section attributes
/// - The entry point symbol cannot be resolved (for executables)
pub fn link(
    config: &LinkerConfig,
    inputs: Vec<LinkerInput>,
    handler: &dyn relocation::RelocationHandler,
    diagnostics: &mut DiagnosticEngine,
) -> Result<LinkerOutput, String> {
    // -----------------------------------------------------------------------
    // Relocatable passthrough — no linking needed
    // -----------------------------------------------------------------------
    if config.output_type == OutputType::Relocatable {
        return link_relocatable(config, inputs, diagnostics);
    }

    // -----------------------------------------------------------------------
    // Phase 1: Symbol Resolution
    // -----------------------------------------------------------------------
    let mut resolver = symbol_resolver::SymbolResolver::new();

    for input in &inputs {
        resolver.collect_symbols(input.object_id, &input.symbols);
    }

    if let Err(errors) = resolver.resolve() {
        for err_msg in &errors {
            diagnostics.emit_error(Span::dummy(), err_msg.clone());
        }
        if !config.allow_undefined {
            return Err(format!(
                "symbol resolution failed with {} error(s)",
                errors.len()
            ));
        }
    }

    // Check for undefined symbols (fatal for executables unless allow_undefined).
    if let Err(undef_errors) = resolver.check_undefined(config.allow_undefined) {
        for err_msg in &undef_errors {
            diagnostics.emit_error(Span::dummy(), err_msg.clone());
        }
        if !config.allow_undefined {
            return Err(format!(
                "linking failed: {} undefined symbol(s)",
                undef_errors.len()
            ));
        }
    }

    // Emit accumulated symbol resolution diagnostics.
    resolver.emit_diagnostics(diagnostics);

    if diagnostics.has_errors() && !config.allow_undefined {
        return Err("linking aborted due to symbol resolution errors".to_string());
    }

    // -----------------------------------------------------------------------
    // Phase 2: Section Merging
    // -----------------------------------------------------------------------
    let mut merger = section_merger::SectionMerger::new(config.target);

    for input in &inputs {
        for section in &input.sections {
            merger.add_input_section(section.clone());
        }
    }

    // Compute the base address for the target and output type.
    let is_shared = config.is_shared();
    let base_address = linker_script::default_base_address(&config.target, is_shared);
    let page_alignment = linker_script::PAGE_ALIGNMENT;

    // Assign virtual addresses and file offsets to all output sections.
    let address_map = merger.assign_addresses(base_address, page_alignment);

    // Resolve fragment-level addresses within each output section.
    merger.resolve_fragment_addresses();

    // -----------------------------------------------------------------------
    // Phase 3: Linker-Defined Symbols
    // -----------------------------------------------------------------------
    // Build a map of section name → (virtual_address, size) for linker symbols.
    let mut section_addr_map: FxHashMap<String, (u64, u64)> = FxHashMap::default();
    for (name, addr_info) in &address_map.section_addresses {
        section_addr_map.insert(name.clone(), (addr_info.virtual_address, addr_info.size));
    }
    resolver.define_linker_symbols(&section_addr_map);

    // Build the final resolved symbol table.
    let sym_table = resolver.build_symbol_table();

    // -----------------------------------------------------------------------
    // Phase 4: Build Symbol Address Map for Relocations
    // -----------------------------------------------------------------------
    let mut symbol_address_map: FxHashMap<String, symbol_resolver::ResolvedSymbol> =
        FxHashMap::default();

    // Populate from the resolved output symbols.
    for sym in &sym_table.symbols {
        let resolved = symbol_resolver::ResolvedSymbol {
            name: sym.name.clone(),
            final_address: sym.value,
            size: sym.size,
            binding: sym.binding,
            sym_type: sym.sym_type,
            visibility: sym.visibility,
            section_name: String::new(),
            is_defined: sym.section_index != symbol_resolver::SHN_UNDEF,
            from_object: 0,
            export_dynamic: false,
        };
        symbol_address_map.insert(sym.name.clone(), resolved);
    }

    // -----------------------------------------------------------------------
    // Phase 5: Relocation Processing
    // -----------------------------------------------------------------------
    // Collect all relocations from all input objects.
    let mut all_relocations: Vec<relocation::Relocation> = Vec::new();
    for input in &inputs {
        all_relocations.extend(input.relocations.iter().cloned());
    }

    // Build section data buffers for relocation patching.
    let mut section_data_map: FxHashMap<String, Vec<u8>> = FxHashMap::default();
    let ordered_sections = merger.get_ordered_sections();
    for output_section in &ordered_sections {
        let data = merger.build_section_data(&output_section.name);
        section_data_map.insert(output_section.name.clone(), data);
    }

    // Process relocations (collect, resolve, apply).
    if !all_relocations.is_empty() {
        let _reloc_result = relocation::process_relocations(
            &config.target,
            all_relocations,
            &symbol_address_map,
            &address_map,
            &mut section_data_map,
            handler,
            diagnostics,
        );
    }

    // Check for relocation errors.
    if diagnostics.has_errors() && !config.allow_undefined {
        return Err("linking aborted due to relocation errors".to_string());
    }

    // -----------------------------------------------------------------------
    // Phase 6: Dynamic Section Generation (if ET_DYN)
    // -----------------------------------------------------------------------
    if config.is_shared() {
        generate_dynamic_sections(
            config,
            &sym_table,
            &mut section_data_map,
            &address_map,
            diagnostics,
        );
    }

    // -----------------------------------------------------------------------
    // Phase 7: Entry Point Resolution
    // -----------------------------------------------------------------------
    let entry_point = if config.is_executable() {
        // Build a simple symbol name → address map for the entry point lookup.
        let mut sym_addr_flat: FxHashMap<String, u64> = FxHashMap::default();
        for sym in &sym_table.symbols {
            sym_addr_flat.insert(sym.name.clone(), sym.value);
        }

        match resolve_entry_point(&config.entry_point, &sym_addr_flat) {
            Ok(addr) => addr,
            Err(err_msg) => {
                diagnostics.emit_error(Span::dummy(), err_msg.clone());
                return Err(err_msg);
            }
        }
    } else {
        0
    };

    // -----------------------------------------------------------------------
    // Phase 8: ELF Output Writing
    // -----------------------------------------------------------------------
    let elf_data = write_elf_output(
        config,
        &ordered_sections,
        &section_data_map,
        &address_map,
        &sym_table,
        entry_point,
    );

    Ok(LinkerOutput {
        elf_data,
        entry_point,
        output_type: config.output_type,
    })
}

// ============================================================================
// Internal Helper Functions
// ============================================================================

/// Handle relocatable output (passthrough — no actual linking).
///
/// For `OutputType::Relocatable`, we simply concatenate input sections into
/// a single relocatable ELF object file without performing symbol resolution,
/// relocation processing, or address assignment.
fn link_relocatable(
    config: &LinkerConfig,
    inputs: Vec<LinkerInput>,
    diagnostics: &mut DiagnosticEngine,
) -> Result<LinkerOutput, String> {
    let elf_type = crate::backend::elf_writer_common::ET_REL;
    let mut writer = ElfWriter::new(config.target, elf_type);

    // Add all input sections to the output.
    for input in &inputs {
        for section in &input.sections {
            let elf_section = Section {
                name: section.name.clone(),
                sh_type: section.section_type,
                sh_flags: section.flags,
                data: section.data.clone(),
                sh_link: 0,
                sh_info: 0,
                sh_addralign: section.alignment,
                sh_entsize: 0,
            };
            writer.add_section(elf_section);
        }

        // Add symbols from this input.
        for sym in &input.symbols {
            let elf_sym = ElfSymbol {
                name: sym.name.clone(),
                value: sym.value,
                size: sym.size,
                binding: sym.binding,
                sym_type: sym.sym_type,
                visibility: sym.visibility,
                section_index: sym.section_index,
            };
            writer.add_symbol(elf_sym);
        }
    }

    let elf_data = writer.write();

    // Emit a note if no inputs were provided.
    if inputs.is_empty() {
        diagnostics.emit(Diagnostic::warning(
            Span::dummy(),
            "no input files provided for relocatable output",
        ));
    }

    Ok(LinkerOutput {
        elf_data,
        entry_point: 0,
        output_type: OutputType::Relocatable,
    })
}

/// Resolve the entry point symbol to a virtual address.
///
/// Returns `Ok(address)` if the entry point symbol is found, or `Err`
/// with a descriptive message if the symbol is undefined.
fn resolve_entry_point(entry_name: &str, symbols: &FxHashMap<String, u64>) -> Result<u64, String> {
    match symbols.get(entry_name) {
        Some(&addr) => Ok(addr),
        None => Err(format!(
            "entry point symbol '{}' is undefined; \
             cannot produce executable without an entry point",
            entry_name
        )),
    }
}

/// Generate dynamic linking sections for shared library output.
///
/// Populates section data buffers with `.dynamic`, `.dynsym`, `.dynstr`,
/// `.gnu.hash`, `.got`, `.got.plt`, `.plt`, `.rela.dyn`, and `.rela.plt`
/// section content.
fn generate_dynamic_sections(
    config: &LinkerConfig,
    sym_table: &symbol_resolver::SymbolTable,
    section_data_map: &mut FxHashMap<String, Vec<u8>>,
    _address_map: &section_merger::AddressMap,
    _diagnostics: &mut DiagnosticEngine,
) {
    // Build lists of exported and imported symbols for the dynamic linker.
    let mut exported: Vec<dynamic::ExportedSymbol> = Vec::new();
    let mut imported: Vec<dynamic::ImportedSymbol> = Vec::new();

    for sym in &sym_table.symbols {
        let is_defined = sym.section_index != symbol_resolver::SHN_UNDEF;
        let is_global_or_weak =
            sym.binding == symbol_resolver::STB_GLOBAL || sym.binding == symbol_resolver::STB_WEAK;

        if is_defined && is_global_or_weak && !sym.name.is_empty() {
            exported.push(dynamic::ExportedSymbol {
                name: sym.name.clone(),
                value: sym.value,
                size: sym.size,
                binding: sym.binding,
                sym_type: sym.sym_type,
                visibility: sym.visibility,
                section_index: sym.section_index,
            });
        } else if !is_defined && is_global_or_weak && !sym.name.is_empty() {
            imported.push(dynamic::ImportedSymbol {
                name: sym.name.clone(),
                binding: sym.binding,
                sym_type: sym.sym_type,
                needs_plt: sym.sym_type == symbol_resolver::STT_FUNC,
            });
        }
    }

    // Collect needed libraries.
    let needed: Vec<String> = config.needed_libs.clone();

    // Build the dynamic linking sections.
    let dynamic_result = dynamic::build_dynamic_sections(
        &config.target,
        config.is_shared(),
        &exported,
        &imported,
        &needed,
        config.soname.as_deref(),
    );

    // Store the generated section data into the section data map.
    if !dynamic_result.dynamic_section.is_empty() {
        section_data_map.insert(".dynamic".to_string(), dynamic_result.dynamic_section);
    }
    if !dynamic_result.dynsym_section.is_empty() {
        section_data_map.insert(".dynsym".to_string(), dynamic_result.dynsym_section);
    }
    if !dynamic_result.dynstr_section.is_empty() {
        section_data_map.insert(".dynstr".to_string(), dynamic_result.dynstr_section);
    }
    if !dynamic_result.gnu_hash_section.is_empty() {
        section_data_map.insert(".gnu.hash".to_string(), dynamic_result.gnu_hash_section);
    }
    if !dynamic_result.got_section.is_empty() {
        section_data_map.insert(".got".to_string(), dynamic_result.got_section);
    }
    if !dynamic_result.got_plt_section.is_empty() {
        section_data_map.insert(".got.plt".to_string(), dynamic_result.got_plt_section);
    }
    if !dynamic_result.plt_section.is_empty() {
        section_data_map.insert(".plt".to_string(), dynamic_result.plt_section);
    }
    if !dynamic_result.rela_dyn_section.is_empty() {
        section_data_map.insert(".rela.dyn".to_string(), dynamic_result.rela_dyn_section);
    }
    if !dynamic_result.rela_plt_section.is_empty() {
        section_data_map.insert(".rela.plt".to_string(), dynamic_result.rela_plt_section);
    }
    if let Some(interp_bytes) = dynamic_result.interp {
        section_data_map.insert(".interp".to_string(), interp_bytes);
    }
}

/// Write the final ELF output binary.
///
/// Assembles sections, symbols, and program headers into a complete ELF
/// file using the `ElfWriter` infrastructure.
fn write_elf_output(
    config: &LinkerConfig,
    ordered_sections: &[&section_merger::OutputSection],
    section_data_map: &FxHashMap<String, Vec<u8>>,
    address_map: &section_merger::AddressMap,
    sym_table: &symbol_resolver::SymbolTable,
    entry_point: u64,
) -> Vec<u8> {
    let elf_type = config.output_type.to_elf_type();
    let mut writer = ElfWriter::new(config.target, elf_type);

    // Set the entry point for executables.
    if config.is_executable() {
        writer.set_entry_point(entry_point);
    }

    // Add output sections to the ELF writer.
    for &output_section in ordered_sections {
        // Use patched data from the section data map if available,
        // otherwise fall back to the merged section data.
        let data = section_data_map
            .get(&output_section.name)
            .cloned()
            .unwrap_or_default();

        let elf_section = Section {
            name: output_section.name.clone(),
            sh_type: output_section.section_type,
            sh_flags: output_section.flags,
            data,
            sh_link: 0,
            sh_info: 0,
            sh_addralign: output_section.alignment,
            sh_entsize: 0,
        };
        writer.add_section(elf_section);
    }

    // Add dynamic linking sections that were generated separately.
    let dynamic_section_names = [
        ".interp",
        ".dynamic",
        ".dynsym",
        ".dynstr",
        ".gnu.hash",
        ".got",
        ".got.plt",
        ".plt",
        ".rela.dyn",
        ".rela.plt",
    ];

    for &sec_name in &dynamic_section_names {
        // Skip if already in the ordered sections or if no data was generated.
        let already_present = ordered_sections.iter().any(|s| s.name == sec_name);

        if already_present {
            continue;
        }

        if let Some(data) = section_data_map.get(sec_name) {
            if !data.is_empty() {
                let (sh_type, sh_flags, sh_entsize) = dynamic_section_attributes(sec_name, config);
                let elf_section = Section {
                    name: sec_name.to_string(),
                    sh_type,
                    sh_flags,
                    data: data.clone(),
                    sh_link: 0,
                    sh_info: 0,
                    sh_addralign: if config.target.is_64bit() { 8 } else { 4 },
                    sh_entsize,
                };
                writer.add_section(elf_section);
            }
        }
    }

    // Add symbols to the output symbol table.
    for sym in &sym_table.symbols {
        let elf_sym = ElfSymbol {
            name: sym.name.clone(),
            value: sym.value,
            size: sym.size,
            binding: sym.binding,
            sym_type: sym.sym_type,
            visibility: sym.visibility,
            section_index: sym.section_index,
        };
        writer.add_symbol(elf_sym);
    }

    // Build program headers for linked output.
    if config.output_type.is_linked() {
        build_program_headers(config, ordered_sections, address_map, &mut writer);
    }

    // Serialize to bytes.
    writer.write()
}

/// Build ELF program headers based on the computed layout.
///
/// Creates `PT_LOAD`, `PT_DYNAMIC`, `PT_GNU_STACK`, and other segment
/// descriptors from the section layout and address map.
fn build_program_headers(
    config: &LinkerConfig,
    ordered_sections: &[&section_merger::OutputSection],
    _address_map: &section_merger::AddressMap,
    writer: &mut ElfWriter,
) {
    // Use the linker script to compute the segment layout from section info.
    let is_shared = config.is_shared();
    let mut script = linker_script::DefaultLinkerScript::new(config.target, is_shared);

    // Build InputSectionInfo for each output section.
    let section_infos: Vec<linker_script::InputSectionInfo> = ordered_sections
        .iter()
        .map(|sec| linker_script::InputSectionInfo {
            name: sec.name.clone(),
            size: sec.total_size,
            alignment: sec.alignment,
            flags: sec.flags as u32,
        })
        .collect();

    let layout = script.compute_layout(&section_infos);

    // Convert SegmentLayout entries to ELF program headers.
    for seg in &layout.segments {
        let phdr = crate::backend::elf_writer_common::Elf64ProgramHeader {
            p_type: seg.seg_type,
            p_flags: seg.flags,
            p_offset: seg.offset,
            p_vaddr: seg.vaddr,
            p_paddr: seg.paddr,
            p_filesz: seg.filesz,
            p_memsz: seg.memsz,
            p_align: seg.alignment,
        };
        writer.add_program_header(phdr);
    }
}

/// Return ELF section attributes (type, flags, entry size) for dynamically
/// generated sections.
fn dynamic_section_attributes(name: &str, config: &LinkerConfig) -> (u32, u64, u64) {
    let ptr_size = config.target.pointer_width() as u64;

    match name {
        ".interp" => (section_merger::SHT_PROGBITS, section_merger::SHF_ALLOC, 0),
        ".dynamic" => (
            section_merger::SHT_DYNAMIC,
            section_merger::SHF_ALLOC | section_merger::SHF_WRITE,
            ptr_size * 2, // Each dynamic entry is 2 × pointer width
        ),
        ".dynsym" => (
            section_merger::SHT_DYNSYM,
            section_merger::SHF_ALLOC,
            if config.target.is_64bit() { 24 } else { 16 },
        ),
        ".dynstr" => (section_merger::SHT_STRTAB, section_merger::SHF_ALLOC, 0),
        ".gnu.hash" => (section_merger::SHT_GNU_HASH, section_merger::SHF_ALLOC, 0),
        ".got" | ".got.plt" => (
            section_merger::SHT_PROGBITS,
            section_merger::SHF_ALLOC | section_merger::SHF_WRITE,
            ptr_size,
        ),
        ".plt" => (
            section_merger::SHT_PROGBITS,
            section_merger::SHF_ALLOC | section_merger::SHF_EXECINSTR,
            0,
        ),
        ".rela.dyn" | ".rela.plt" => (
            section_merger::SHT_RELA,
            section_merger::SHF_ALLOC,
            if config.target.is_64bit() { 24 } else { 8 },
        ),
        _ => (section_merger::SHT_PROGBITS, section_merger::SHF_ALLOC, 0),
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_type_display() {
        assert_eq!(format!("{}", OutputType::Executable), "executable");
        assert_eq!(format!("{}", OutputType::SharedLibrary), "shared library");
        assert_eq!(format!("{}", OutputType::Relocatable), "relocatable");
    }

    #[test]
    fn test_output_type_elf_type() {
        assert_eq!(OutputType::Executable.to_elf_type(), ET_EXEC);
        assert_eq!(OutputType::SharedLibrary.to_elf_type(), ET_DYN);
        assert_eq!(
            OutputType::Relocatable.to_elf_type(),
            crate::backend::elf_writer_common::ET_REL
        );
    }

    #[test]
    fn test_output_type_requires_pic() {
        assert!(!OutputType::Executable.requires_pic());
        assert!(OutputType::SharedLibrary.requires_pic());
        assert!(!OutputType::Relocatable.requires_pic());
    }

    #[test]
    fn test_output_type_is_linked() {
        assert!(OutputType::Executable.is_linked());
        assert!(OutputType::SharedLibrary.is_linked());
        assert!(!OutputType::Relocatable.is_linked());
    }

    #[test]
    fn test_linker_config_default() {
        let config = LinkerConfig::default();
        assert_eq!(config.target, Target::X86_64);
        assert_eq!(config.output_type, OutputType::Executable);
        assert_eq!(config.output_path, "a.out");
        assert_eq!(config.entry_point, "_start");
        assert!(!config.pic);
        assert!(!config.allow_undefined);
        assert!(config.soname.is_none());
        assert!(config.needed_libs.is_empty());
        assert!(!config.emit_debug);
    }

    #[test]
    fn test_linker_config_new_shared() {
        let config = LinkerConfig::new(Target::AArch64, OutputType::SharedLibrary);
        assert_eq!(config.target, Target::AArch64);
        assert!(config.pic); // Shared libraries require PIC
        assert!(config.allow_undefined); // Shared libraries allow undefined
        assert!(config.is_shared());
        assert!(!config.is_executable());
        assert!(!config.is_relocatable());
    }

    #[test]
    fn test_linker_config_new_executable() {
        let config = LinkerConfig::new(Target::RiscV64, OutputType::Executable);
        assert_eq!(config.target, Target::RiscV64);
        assert!(!config.pic);
        assert!(!config.allow_undefined);
        assert!(config.is_executable());
        assert!(!config.is_shared());
    }

    #[test]
    fn test_linker_input_new() {
        let input = LinkerInput::new(42, "test.o".to_string());
        assert_eq!(input.object_id, 42);
        assert_eq!(input.filename, "test.o");
        assert_eq!(input.section_count(), 0);
        assert_eq!(input.symbol_count(), 0);
        assert_eq!(input.relocation_count(), 0);
    }

    #[test]
    fn test_linker_output_debug() {
        let output = LinkerOutput {
            elf_data: vec![0x7f, b'E', b'L', b'F'],
            entry_point: 0x400000,
            output_type: OutputType::Executable,
        };
        let debug_str = format!("{:?}", output);
        assert!(debug_str.contains("LinkerOutput"));
        assert!(debug_str.contains("elf_data_len"));
        assert!(debug_str.contains("0x400000"));
        assert!(output.is_executable());
        assert!(!output.is_shared_library());
        assert_eq!(output.size(), 4);
    }

    #[test]
    fn test_resolve_entry_point_found() {
        let mut symbols = FxHashMap::default();
        symbols.insert("_start".to_string(), 0x401000u64);
        symbols.insert("main".to_string(), 0x401100u64);

        assert_eq!(resolve_entry_point("_start", &symbols), Ok(0x401000));
        assert_eq!(resolve_entry_point("main", &symbols), Ok(0x401100));
    }

    #[test]
    fn test_resolve_entry_point_not_found() {
        let symbols: FxHashMap<String, u64> = FxHashMap::default();
        let result = resolve_entry_point("_start", &symbols);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("_start"));
    }

    #[test]
    fn test_dynamic_section_attributes() {
        let config = LinkerConfig::new(Target::X86_64, OutputType::SharedLibrary);

        let (sh_type, sh_flags, _) = dynamic_section_attributes(".interp", &config);
        assert_eq!(sh_type, section_merger::SHT_PROGBITS);
        assert_eq!(sh_flags, section_merger::SHF_ALLOC);

        let (sh_type, sh_flags, sh_entsize) = dynamic_section_attributes(".dynamic", &config);
        assert_eq!(sh_type, section_merger::SHT_DYNAMIC);
        assert_ne!(sh_flags & section_merger::SHF_WRITE, 0);
        assert_eq!(sh_entsize, 16); // 2 × 8 bytes for 64-bit

        let (sh_type, _, sh_entsize) = dynamic_section_attributes(".dynsym", &config);
        assert_eq!(sh_type, section_merger::SHT_DYNSYM);
        assert_eq!(sh_entsize, 24); // Elf64_Sym size

        let (sh_type, _, _) = dynamic_section_attributes(".plt", &config);
        assert_eq!(sh_type, section_merger::SHT_PROGBITS);
    }

    #[test]
    fn test_output_type_equality() {
        assert_eq!(OutputType::Executable, OutputType::Executable);
        assert_ne!(OutputType::Executable, OutputType::SharedLibrary);
        assert_ne!(OutputType::SharedLibrary, OutputType::Relocatable);
    }

    #[test]
    fn test_linker_config_all_targets() {
        for &target in &[
            Target::X86_64,
            Target::I686,
            Target::AArch64,
            Target::RiscV64,
        ] {
            let config = LinkerConfig::new(target, OutputType::Executable);
            assert_eq!(config.target, target);
            assert_eq!(config.output_type, OutputType::Executable);
        }
    }
}
