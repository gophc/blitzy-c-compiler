//! # x86-64 Built-in ELF Linker
//!
//! Produces x86-64 ELF binaries (`ET_EXEC` and `ET_DYN`) using BCC's standalone
//! backend — no external linker is invoked.
//!
//! ## Architecture Details
//! - ELF machine type: `EM_X86_64` (0x3E / 62)
//! - Variable-length instruction patching during relocation application
//! - RIP-relative addressing for GOT/PLT (`GOTPCREL`, `GOTPCRELX`)
//! - PLT entries: `jmp *GOT[n](%rip)` / `push n` / `jmp PLT[0]`
//!
//! ## Features
//! - Static linking: symbol resolution, section merging, relocation patching → ET_EXEC
//! - Dynamic linking: GOT, PLT, .dynamic, .dynsym, .gnu.hash → ET_DYN
//! - PIC code generation support via `-fPIC` / `-shared`
//! - Symbol visibility control (STV_DEFAULT, STV_HIDDEN, STV_PROTECTED)
//! - DWARF debug section passthrough (when `-g` is active)
//!
//! ## Delegation
//! Shared linker infrastructure is provided by [`crate::backend::linker_common`]:
//! - Symbol resolution: [`crate::backend::linker_common::symbol_resolver`]
//! - Section merging: [`crate::backend::linker_common::section_merger`]
//! - Relocation framework: [`crate::backend::linker_common::relocation`]
//! - Dynamic sections: [`crate::backend::linker_common::dynamic`]
//! - Linker script: [`crate::backend::linker_common::linker_script`]

pub mod relocations;

// ---------------------------------------------------------------------------
// Imports — crate-internal only (zero external crate dependencies)
// ---------------------------------------------------------------------------

use crate::common::diagnostics::{DiagnosticEngine, Span};
use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::common::target::Target;

use crate::backend::elf_writer_common::{
    Elf64ProgramHeader, ElfSymbol, ElfWriter, Section, ET_DYN, ET_EXEC, PT_DYNAMIC, PT_GNU_STACK,
    PT_INTERP, SHF_ALLOC, SHF_EXECINSTR, SHF_WRITE, SHT_DYNAMIC, SHT_DYNSYM, SHT_GNU_HASH,
    SHT_NOBITS, SHT_PROGBITS, SHT_RELA, SHT_STRTAB,
};
use crate::backend::linker_common::dynamic::DynamicLinkContext;
use crate::backend::linker_common::linker_script::{DefaultLinkerScript, InputSectionInfo};
use crate::backend::linker_common::relocation::{
    annotate_relocations, apply_all_relocations, build_input_to_output_map, resolve_relocations,
    RelocationCollector, RelocationHandler,
};
use crate::backend::linker_common::section_merger::{OutputSection, SectionMerger};
use crate::backend::linker_common::symbol_resolver::{ResolvedSymbol, SymbolResolver, SHN_UNDEF};
use crate::backend::linker_common::{LinkerConfig, LinkerInput, LinkerOutput, OutputType};

// Re-export the x86-64 relocation handler for consumers of this module.
pub use self::relocations::X86_64RelocationHandler;

// ---------------------------------------------------------------------------
// x86-64 ELF Constants
// ---------------------------------------------------------------------------

/// ELF machine type for x86-64 (AMD64).
pub const EM_X86_64: u16 = 62; // 0x3E

/// Default virtual base address for x86-64 ET_EXEC executables.
///
/// Matches the conventional Linux default used by GNU ld for static x86-64
/// executables. Shared objects (ET_DYN) use base 0 (PIC).
pub const DEFAULT_BASE_ADDRESS: u64 = 0x0040_0000;

/// Standard page size for x86-64 Linux (4 KiB).
///
/// Used for segment alignment in the ELF program header table and for
/// guard-page stack probing thresholds.
pub const PAGE_SIZE: u64 = 0x1000;

/// Default entry-point symbol name for x86-64 executables.
pub const DEFAULT_ENTRY: &str = "_start";

/// Path to the x86-64 dynamic linker on Linux.
///
/// Written into the `.interp` section and referenced by the `PT_INTERP`
/// program header when producing dynamically-linked executables.
pub const DYNAMIC_LINKER: &str = "/lib64/ld-linux-x86-64.so.2";

/// Size of the PLT header stub (PLT\[0\]) in bytes.
///
/// PLT\[0\] is the resolver trampoline: it pushes `link_map` from GOT\[1\]
/// and jumps to `_dl_runtime_resolve` via GOT\[2\].
pub const PLT0_SIZE: usize = 16;

/// Size of each per-function PLT entry (PLT\[n\]) in bytes.
///
/// Standard x86-64 PLT entry layout:
/// - `jmp *GOT(%rip)` (6 bytes) — indirect jump through GOT slot
/// - `push <index>`    (5 bytes) — push relocation index for lazy binding
/// - `jmp PLT[0]`      (5 bytes) — fall through to resolver
pub const PLT_ENTRY_SIZE: usize = 16;

/// Number of reserved entries at the start of `.got.plt`.
///
/// - GOT\[0\]: address of `.dynamic` section (filled by dynamic linker)
/// - GOT\[1\]: `link_map` pointer (filled by dynamic linker)
/// - GOT\[2\]: `_dl_runtime_resolve` address (filled by dynamic linker)
pub const GOT_PLT_RESERVED: usize = 3;

/// Size of a single GOT entry in bytes (8 bytes for 64-bit pointers).
pub const GOT_ENTRY_SIZE: usize = 8;

// ---------------------------------------------------------------------------
// x86-64 Linker Driver
// ---------------------------------------------------------------------------

/// x86-64 ELF linker driver.
///
/// Orchestrates the complete linking process for x86-64 targets, producing
/// `ET_EXEC` (static executables) or `ET_DYN` (shared objects).
///
/// ## Pipeline
///
/// 1. Collect input objects via [`add_input`](X86_64Linker::add_input).
/// 2. Call [`link`](X86_64Linker::link) which executes:
///    - Symbol resolution (strong/weak binding, undefined detection)
///    - Section merging (aggregation, alignment, ordering)
///    - GOT/PLT requirement scanning
///    - GOT/PLT stub generation (for PIC / shared libraries)
///    - Address layout computation (linker script)
///    - Relocation resolution and application
///    - Dynamic section generation (for shared libraries)
///    - Final ELF output assembly
pub struct X86_64Linker {
    /// Linker configuration from CLI flags.
    config: LinkerConfig,
    /// Two-pass symbol resolver.
    symbol_resolver: SymbolResolver,
    /// Section merging engine.
    section_merger: SectionMerger,
    /// x86-64-specific relocation handler implementing `RelocationHandler`.
    reloc_handler: X86_64RelocationHandler,
    /// Dynamic linking context (created for shared libs or when `-l` is used).
    dynamic_context: Option<DynamicLinkContext>,
    /// Default linker script for section-to-segment mapping.
    linker_script: DefaultLinkerScript,
    /// Accumulated input object files.
    inputs: Vec<LinkerInput>,
    /// Whether PIC mode is active (`-fPIC`).
    pic_enabled: bool,
    /// Whether the output is a shared object (`-shared`).
    is_shared: bool,
    /// Whether DWARF debug sections should be passed through (`-g`).
    emit_debug: bool,
}

impl X86_64Linker {
    /// Create a new x86-64 linker instance.
    ///
    /// The constructor reads the `config` to determine the output type,
    /// PIC mode, debug emission, and needed libraries. It pre-creates
    /// the dynamic linking context when producing shared objects or
    /// when external shared libraries are referenced.
    ///
    /// # Arguments
    ///
    /// * `config` — Linker configuration derived from CLI flags.
    pub fn new(config: LinkerConfig) -> Self {
        let target = Target::X86_64;
        let is_shared = config.output_type == OutputType::SharedLibrary;
        let pic_enabled = config.pic;
        let emit_debug = config.emit_debug;

        // Create a dynamic link context when:
        // - producing a shared library, OR
        // - the executable needs shared libraries (DT_NEEDED entries)
        let dynamic_context = if is_shared || !config.needed_libs.is_empty() {
            let mut ctx = DynamicLinkContext::new(target, is_shared);
            // Set PT_INTERP for dynamically-linked executables.
            if !is_shared {
                ctx.set_interp();
            }
            // Record DT_NEEDED entries.
            for lib in &config.needed_libs {
                ctx.add_needed_library(lib);
            }
            // Set SONAME if specified.
            if let Some(ref soname) = config.soname {
                ctx.soname = Some(soname.clone());
            }
            Some(ctx)
        } else {
            None
        };

        Self {
            pic_enabled,
            is_shared,
            emit_debug,
            symbol_resolver: SymbolResolver::new(),
            section_merger: SectionMerger::new(target),
            reloc_handler: X86_64RelocationHandler::new(),
            dynamic_context,
            linker_script: DefaultLinkerScript::new(target, is_shared),
            inputs: Vec::new(),
            config,
        }
    }

    /// Add a relocatable object file (`.o`) to the link.
    ///
    /// Each input is registered with the symbol resolver (for cross-file
    /// symbol resolution), the section merger (for output section
    /// construction), and stored for relocation processing.
    pub fn add_input(&mut self, input: LinkerInput) {
        // Feed symbols to the symbol resolver.
        self.symbol_resolver
            .collect_symbols(input.object_id, &input.symbols);

        // Feed sections to the section merger.
        for section in &input.sections {
            self.section_merger.add_input_section(section.clone());
        }

        // Retain the full input for relocation processing.
        self.inputs.push(input);
    }

    /// Perform the complete x86-64 link operation.
    ///
    /// Executes the full linking pipeline:
    ///
    /// 1. Resolve symbols (strong/weak binding, undefined detection)
    /// 2. Merge input sections into output sections
    /// 3. Scan relocations for GOT/PLT requirements
    /// 4. Generate GOT/PLT stubs (if PIC/shared)
    /// 5. Compute address layout (linker script)
    /// 6. Resolve and apply relocations (x86-64-specific patching)
    /// 7. Generate dynamic sections (if shared library)
    /// 8. Write final ELF output
    ///
    /// # Arguments
    ///
    /// * `diagnostics` — Diagnostic engine for error/warning reporting.
    ///
    /// # Returns
    ///
    /// On success, returns a [`LinkerOutput`] containing the serialized ELF
    /// bytes, the resolved entry point, and the output type.
    pub fn link(&mut self, diagnostics: &mut DiagnosticEngine) -> Result<LinkerOutput, String> {
        // ===================================================================
        // Phase 1: Symbol Resolution
        // ===================================================================
        if let Err(errors) = self.symbol_resolver.resolve() {
            for err_msg in &errors {
                diagnostics.emit_error(Span::dummy(), err_msg.clone());
            }
            if !self.config.allow_undefined {
                return Err(format!(
                    "symbol resolution failed with {} error(s)",
                    errors.len()
                ));
            }
        }

        // Check for undefined symbols (fatal for executables).
        if let Err(undef_errors) = self
            .symbol_resolver
            .check_undefined(self.config.allow_undefined)
        {
            for err_msg in &undef_errors {
                diagnostics.emit_error(Span::dummy(), err_msg.clone());
            }
            if !self.config.allow_undefined {
                return Err(format!(
                    "linking failed: {} undefined symbol(s)",
                    undef_errors.len()
                ));
            }
        }

        // Emit accumulated diagnostics from symbol resolution.
        self.symbol_resolver.emit_diagnostics(diagnostics);

        if diagnostics.has_errors() && !self.config.allow_undefined {
            return Err("linking aborted due to symbol resolution errors".to_string());
        }

        // ===================================================================
        // Phase 2: Section Merging
        // ===================================================================
        let base_address = if self.is_shared {
            0u64
        } else {
            DEFAULT_BASE_ADDRESS
        };
        let page_alignment = PAGE_SIZE;

        let address_map = self
            .section_merger
            .assign_addresses(base_address, page_alignment);
        self.section_merger.resolve_fragment_addresses();

        // ===================================================================
        // Phase 3: Linker-Defined Symbols
        // ===================================================================
        let mut section_addr_map: FxHashMap<String, (u64, u64)> = FxHashMap::default();
        for (name, addr_info) in &address_map.section_addresses {
            section_addr_map.insert(name.clone(), (addr_info.virtual_address, addr_info.size));
        }
        self.symbol_resolver
            .define_linker_symbols(&section_addr_map);

        // ===================================================================
        // Phase 4: Relocation Collection & GOT/PLT Scanning
        // ===================================================================
        let output_sections = self.section_merger.get_ordered_sections();
        let output_section_vec: Vec<OutputSection> = output_sections.into_iter().cloned().collect();
        let input_to_output_map = build_input_to_output_map(&output_section_vec);

        // Collect all relocations from input objects.
        let mut reloc_collector = RelocationCollector::new();
        for input in &self.inputs {
            for (sec_idx, section) in input.sections.iter().enumerate() {
                let relocs: Vec<crate::backend::linker_common::relocation::Relocation> = section
                    .relocations
                    .iter()
                    .map(|sr| crate::backend::linker_common::relocation::Relocation {
                        offset: sr.offset,
                        symbol_name: String::new(), // resolved below
                        sym_index: sr.sym_index,
                        rel_type: sr.rel_type,
                        addend: sr.addend,
                        object_id: input.object_id,
                        section_index: sec_idx as u32,
                        output_section_name: None,
                    })
                    .collect();
                if !relocs.is_empty() {
                    reloc_collector.add_relocations(input.object_id, sec_idx as u32, relocs);
                }
            }
            // Also process top-level relocations from the input.
            if !input.relocations.is_empty() {
                let mut relocs = input.relocations.clone();
                for r in &mut relocs {
                    r.object_id = input.object_id;
                }
                reloc_collector.add_relocations(input.object_id, 0, relocs);
            }
        }

        // Annotate relocations with output section names and resolve symbol
        // names from the symbol indices.
        let mut all_relocs = reloc_collector.into_relocations();
        annotate_relocations(&mut all_relocs, &input_to_output_map);

        // Resolve symbol names for relocations that have sym_index references.
        self.resolve_relocation_symbol_names(&mut all_relocs);

        // Scan for GOT/PLT requirements.
        let mut got_symbols: FxHashSet<String> = FxHashSet::default();
        let mut plt_symbols: FxHashSet<String> = FxHashSet::default();
        for rel in &all_relocs {
            if self.reloc_handler.needs_got(rel.rel_type) && !rel.symbol_name.is_empty() {
                got_symbols.insert(rel.symbol_name.clone());
            }
            if self.reloc_handler.needs_plt(rel.rel_type) && !rel.symbol_name.is_empty() {
                plt_symbols.insert(rel.symbol_name.clone());
            }
        }

        // ===================================================================
        // Phase 5: GOT/PLT Stub Generation
        // ===================================================================
        let mut got_entries: FxHashMap<String, u64> = FxHashMap::default();
        let mut plt_entries: FxHashMap<String, u64> = FxHashMap::default();
        let mut got_base: u64 = 0;

        if self.dynamic_context.is_some() || self.pic_enabled || !got_symbols.is_empty() {
            let ctx = self
                .dynamic_context
                .get_or_insert_with(|| DynamicLinkContext::new(Target::X86_64, self.is_shared));

            // Allocate GOT entries for symbols that need them.
            for sym_name in &got_symbols {
                let offset = ctx.got.add_got_entry(sym_name, 0);
                got_entries.insert(sym_name.clone(), offset);
            }

            // Allocate GOT.PLT entries and PLT stubs for symbols that need PLT.
            for sym_name in &plt_symbols {
                let (got_plt_offset, _plt_idx) = ctx.got.add_got_plt_entry(sym_name);
                // PLT entry address will be resolved after layout; store the
                // index for now.
                plt_entries.insert(sym_name.clone(), got_plt_offset);
            }
        }

        // ===================================================================
        // Phase 6: Address Layout (Linker Script)
        // ===================================================================
        // Gather section information for the linker script layout engine.
        let ordered_sections = self.section_merger.get_ordered_sections();
        let mut layout_input: Vec<InputSectionInfo> = Vec::new();
        for sec in &ordered_sections {
            layout_input.push(InputSectionInfo {
                name: sec.name.clone(),
                size: sec.total_size,
                alignment: sec.alignment,
                flags: sec.flags as u32,
            });
        }

        // Add synthetic sections for GOT/PLT if they were generated.
        if let Some(ref ctx) = self.dynamic_context {
            let got_size = ctx.got.encode_got().len() as u64;
            if got_size > 0 {
                layout_input.push(InputSectionInfo {
                    name: ".got".to_string(),
                    size: got_size,
                    alignment: 8,
                    flags: (SHF_ALLOC | SHF_WRITE) as u32,
                });
            }

            let got_plt_size = ctx.got.got_plt_size() as u64;
            if got_plt_size > 0 {
                layout_input.push(InputSectionInfo {
                    name: ".got.plt".to_string(),
                    size: got_plt_size,
                    alignment: 8,
                    flags: (SHF_ALLOC | SHF_WRITE) as u32,
                });
            }

            if !plt_symbols.is_empty() {
                let plt_size = (PLT0_SIZE + plt_symbols.len() * PLT_ENTRY_SIZE) as u64;
                layout_input.push(InputSectionInfo {
                    name: ".plt".to_string(),
                    size: plt_size,
                    alignment: 16,
                    flags: (SHF_ALLOC | SHF_EXECINSTR) as u32,
                });
            }
        }

        let layout = self.linker_script.compute_layout(&layout_input);

        // Build a section-name → virtual-address map from the layout.
        let mut section_vaddr_map: FxHashMap<String, u64> = FxHashMap::default();
        for sl in &layout.sections {
            section_vaddr_map.insert(sl.name.clone(), sl.virtual_address);
        }

        // Update GOT entries to absolute addresses based on layout.
        if let Some(got_addr) = section_vaddr_map.get(".got") {
            got_base = *got_addr;
            let mut updated: FxHashMap<String, u64> = FxHashMap::default();
            for (sym_name, offset) in &got_entries {
                updated.insert(sym_name.clone(), got_base + offset);
            }
            got_entries = updated;
        }

        // Update PLT entries to absolute addresses based on layout.
        if let Some(plt_addr) = section_vaddr_map.get(".plt") {
            let got_plt_addr = section_vaddr_map.get(".got.plt").copied().unwrap_or(0);
            let _ = got_plt_addr; // GOT.PLT address tracked for PLT stub encoding
            let mut updated: FxHashMap<String, u64> = FxHashMap::default();
            for (idx, sym_name) in plt_symbols.iter().enumerate() {
                let entry_addr = plt_addr + PLT0_SIZE as u64 + idx as u64 * PLT_ENTRY_SIZE as u64;
                updated.insert(sym_name.clone(), entry_addr);
            }
            plt_entries = updated;
        }

        // ===================================================================
        // Phase 7: Symbol Address Assignment
        // ===================================================================
        // Build the resolved symbol table and construct a
        // FxHashMap<String, ResolvedSymbol> for relocation resolution.
        // This mirrors the approach in linker_common::link().
        let symbol_table = self.symbol_resolver.build_symbol_table();
        let mut symbol_addresses: FxHashMap<String, ResolvedSymbol> = FxHashMap::default();

        for sym in &symbol_table.symbols {
            let resolved = ResolvedSymbol {
                name: sym.name.clone(),
                final_address: sym.value,
                size: sym.size,
                binding: sym.binding,
                sym_type: sym.sym_type,
                visibility: sym.visibility,
                section_name: String::new(),
                is_defined: sym.section_index != SHN_UNDEF,
                from_object: 0,
                export_dynamic: false,
            };
            symbol_addresses.insert(sym.name.clone(), resolved);
        }

        // Resolve entry point.
        let entry_point = if self.is_shared {
            0u64
        } else {
            let mut entry_symbols: FxHashMap<String, u64> = FxHashMap::default();
            for (sym_name, sym) in &symbol_addresses {
                entry_symbols.insert(sym_name.clone(), sym.final_address);
            }
            match self.linker_script.resolve_entry_point(&entry_symbols) {
                Ok(addr) => addr,
                Err(e) => {
                    diagnostics.emit_error(Span::dummy(), e);
                    // Fall back to .text start if available.
                    section_vaddr_map
                        .get(".text")
                        .copied()
                        .unwrap_or(DEFAULT_BASE_ADDRESS)
                }
            }
        };

        // ===================================================================
        // Phase 8: Relocation Resolution and Application
        // ===================================================================
        // Build the section address map for the relocation resolver.
        let mut sec_addr_map = crate::backend::linker_common::section_merger::AddressMap {
            section_addresses: FxHashMap::default(),
        };
        for sl in &layout.sections {
            sec_addr_map.section_addresses.insert(
                sl.name.clone(),
                crate::backend::linker_common::section_merger::SectionAddress {
                    virtual_address: sl.virtual_address,
                    file_offset: sl.file_offset,
                    size: sl.size,
                    mem_size: sl.mem_size,
                },
            );
        }

        let resolved_relocs = resolve_relocations(
            &all_relocs,
            &symbol_addresses,
            &sec_addr_map,
            &got_entries,
            &plt_entries,
            got_base,
            &self.reloc_handler,
        );

        // Build mutable section data buffers for patching.
        let mut section_data_map: FxHashMap<String, Vec<u8>> = FxHashMap::default();
        for sec in &ordered_sections {
            let data = self.section_merger.build_section_data(&sec.name);
            section_data_map.insert(sec.name.clone(), data);
        }

        // Apply relocations.
        match resolved_relocs {
            Ok(resolved) => {
                if let Err(errors) = apply_all_relocations(
                    &resolved,
                    &mut section_data_map,
                    &self.reloc_handler,
                    diagnostics,
                ) {
                    for err in &errors {
                        diagnostics.emit_error(Span::dummy(), err.to_string());
                    }
                    if !self.config.allow_undefined {
                        return Err(format!(
                            "relocation application failed with {} error(s)",
                            errors.len()
                        ));
                    }
                }
            }
            Err(errors) => {
                for err in &errors {
                    diagnostics.emit_error(Span::dummy(), err.to_string());
                }
                if !self.config.allow_undefined {
                    return Err(format!(
                        "relocation resolution failed with {} error(s)",
                        errors.len()
                    ));
                }
            }
        }

        // ===================================================================
        // Phase 9: Dynamic Section Finalization
        // ===================================================================
        if let Some(ref mut ctx) = self.dynamic_context {
            ctx.finalize();
        }

        // ===================================================================
        // Phase 10: ELF Output Assembly
        // ===================================================================
        let elf_type = if self.is_shared { ET_DYN } else { ET_EXEC };
        let mut writer = ElfWriter::new(Target::X86_64, elf_type);
        writer.set_entry_point(entry_point);

        // Add program headers for loadable segments.
        for seg in &layout.segments {
            writer.add_program_header(Elf64ProgramHeader {
                p_type: seg.seg_type,
                p_flags: seg.flags,
                p_offset: seg.offset,
                p_vaddr: seg.vaddr,
                p_paddr: seg.paddr,
                p_filesz: seg.filesz,
                p_memsz: seg.memsz,
                p_align: seg.alignment,
            });
        }

        // Add PT_INTERP for dynamically-linked executables.
        if let Some(ref ctx) = self.dynamic_context {
            if let Some(interp_bytes) = ctx.get_interp_bytes() {
                // Find the .interp section layout.
                let interp_addr = section_vaddr_map.get(".interp").copied().unwrap_or(0);
                let interp_offset = layout
                    .sections
                    .iter()
                    .find(|s| s.name == ".interp")
                    .map(|s| s.file_offset)
                    .unwrap_or(0);
                let interp_size = interp_bytes.len() as u64;

                writer.add_section(Section {
                    name: ".interp".to_string(),
                    sh_type: SHT_PROGBITS,
                    sh_flags: SHF_ALLOC,
                    data: interp_bytes,
                    sh_addralign: 1,
                    ..Section::default()
                });

                writer.add_program_header(Elf64ProgramHeader {
                    p_type: PT_INTERP,
                    p_flags: 4, // PF_R
                    p_offset: interp_offset,
                    p_vaddr: interp_addr,
                    p_paddr: interp_addr,
                    p_filesz: interp_size,
                    p_memsz: interp_size,
                    p_align: 1,
                });
            }
        }

        // Add merged output sections with patched data.
        for sec in &ordered_sections {
            // Skip debug sections when debug info was not requested.
            if !self.emit_debug && sec.name.starts_with(".debug") {
                continue;
            }

            let data = section_data_map
                .remove(&sec.name)
                .unwrap_or_else(|| self.section_merger.build_section_data(&sec.name));

            let sh_type = if sec.name == ".bss" {
                SHT_NOBITS
            } else {
                sec.section_type
            };

            writer.add_section(Section {
                name: sec.name.clone(),
                sh_type,
                sh_flags: sec.flags,
                data,
                sh_addralign: sec.alignment,
                ..Section::default()
            });
        }

        // Add GOT/PLT/dynamic sections for shared libraries.
        if let Some(ref ctx) = self.dynamic_context {
            // .got section
            let got_data = ctx.got.encode_got();
            if !got_data.is_empty() {
                writer.add_section(Section {
                    name: ".got".to_string(),
                    sh_type: SHT_PROGBITS,
                    sh_flags: SHF_ALLOC | SHF_WRITE,
                    data: got_data,
                    sh_addralign: 8,
                    sh_entsize: GOT_ENTRY_SIZE as u64,
                    logical_size: 0,
                    ..Section::default()
                });
            }

            // .got.plt section
            let plt_addr = section_vaddr_map.get(".plt").copied().unwrap_or(0);
            let got_plt_data = ctx.got.encode_got_plt(plt_addr);
            if !got_plt_data.is_empty() {
                writer.add_section(Section {
                    name: ".got.plt".to_string(),
                    sh_type: SHT_PROGBITS,
                    sh_flags: SHF_ALLOC | SHF_WRITE,
                    data: got_plt_data,
                    sh_addralign: 8,
                    sh_entsize: GOT_ENTRY_SIZE as u64,
                    logical_size: 0,
                    ..Section::default()
                });
            }

            // .plt section
            let got_plt_addr = section_vaddr_map.get(".got.plt").copied().unwrap_or(0);
            let plt_data = ctx.plt.encode(got_plt_addr, plt_addr);
            if !plt_data.is_empty() {
                writer.add_section(Section {
                    name: ".plt".to_string(),
                    sh_type: SHT_PROGBITS,
                    sh_flags: SHF_ALLOC | SHF_EXECINSTR,
                    data: plt_data,
                    sh_addralign: 16,
                    sh_entsize: PLT_ENTRY_SIZE as u64,
                    logical_size: 0,
                    ..Section::default()
                });
            }

            // .dynsym section
            let dynsym_data = ctx.dynsym.encode_dynsym(&Target::X86_64);
            if !dynsym_data.is_empty() {
                writer.add_section(Section {
                    name: ".dynsym".to_string(),
                    sh_type: SHT_DYNSYM,
                    sh_flags: SHF_ALLOC,
                    data: dynsym_data,
                    sh_addralign: 8,
                    sh_entsize: 24, // Elf64Sym size
                    logical_size: 0,
                    ..Section::default()
                });
            }

            // .dynstr section
            let dynstr_data = ctx.dynsym.encode_dynstr();
            if dynstr_data.len() > 1 {
                writer.add_section(Section {
                    name: ".dynstr".to_string(),
                    sh_type: SHT_STRTAB,
                    sh_flags: SHF_ALLOC,
                    data: dynstr_data,
                    sh_addralign: 1,
                    ..Section::default()
                });
            }

            // .gnu.hash section
            let gnu_hash_data = ctx.gnu_hash.encode();
            if !gnu_hash_data.is_empty() {
                writer.add_section(Section {
                    name: ".gnu.hash".to_string(),
                    sh_type: SHT_GNU_HASH,
                    sh_flags: SHF_ALLOC,
                    data: gnu_hash_data,
                    sh_addralign: 8,
                    ..Section::default()
                });
            }

            // .rela.dyn section
            let rela_dyn_data = ctx.rela.encode_rela_dyn(true);
            if !rela_dyn_data.is_empty() {
                writer.add_section(Section {
                    name: ".rela.dyn".to_string(),
                    sh_type: SHT_RELA,
                    sh_flags: SHF_ALLOC,
                    data: rela_dyn_data,
                    sh_addralign: 8,
                    sh_entsize: 24, // Elf64Rela size
                    logical_size: 0,
                    ..Section::default()
                });
            }

            // .rela.plt section
            let rela_plt_data = ctx.rela.encode_rela_plt(true);
            if !rela_plt_data.is_empty() {
                writer.add_section(Section {
                    name: ".rela.plt".to_string(),
                    sh_type: SHT_RELA,
                    sh_flags: SHF_ALLOC,
                    data: rela_plt_data,
                    sh_addralign: 8,
                    sh_entsize: 24,
                    logical_size: 0,
                    ..Section::default()
                });
            }

            // .dynamic section
            let dynamic_data = ctx.dynamic.encode(true);
            if !dynamic_data.is_empty() {
                writer.add_section(Section {
                    name: ".dynamic".to_string(),
                    sh_type: SHT_DYNAMIC,
                    sh_flags: SHF_ALLOC | SHF_WRITE,
                    data: dynamic_data,
                    sh_addralign: 8,
                    sh_entsize: 16, // Elf64Dyn size
                    logical_size: 0,
                    ..Section::default()
                });

                // Add PT_DYNAMIC program header.
                if let Some(dyn_sl) = layout.sections.iter().find(|s| s.name == ".dynamic") {
                    writer.add_program_header(Elf64ProgramHeader {
                        p_type: PT_DYNAMIC,
                        p_flags: 6, // PF_R | PF_W
                        p_offset: dyn_sl.file_offset,
                        p_vaddr: dyn_sl.virtual_address,
                        p_paddr: dyn_sl.virtual_address,
                        p_filesz: dyn_sl.size,
                        p_memsz: dyn_sl.mem_size,
                        p_align: 8,
                    });
                }
            }
        }

        // Add PT_GNU_STACK (non-executable stack).
        writer.add_program_header(Elf64ProgramHeader {
            p_type: PT_GNU_STACK,
            p_flags: 6, // PF_R | PF_W (no PF_X — NX stack)
            p_offset: 0,
            p_vaddr: 0,
            p_paddr: 0,
            p_filesz: 0,
            p_memsz: 0,
            p_align: 16,
        });

        // Write the resolved symbols into the ELF symbol table.
        for sym in &symbol_table.symbols {
            let final_addr = symbol_addresses
                .get(&sym.name)
                .map(|s| s.final_address)
                .unwrap_or(sym.value);
            writer.add_symbol(ElfSymbol {
                name: sym.name.clone(),
                value: final_addr,
                size: sym.size,
                binding: sym.binding,
                sym_type: sym.sym_type,
                visibility: sym.visibility,
                section_index: sym.section_index,
            });
        }

        // Serialize the complete ELF binary.
        let elf_data = writer.write();

        Ok(LinkerOutput {
            elf_data,
            entry_point,
            output_type: self.config.output_type,
        })
    }

    // -----------------------------------------------------------------------
    // Private Helpers
    // -----------------------------------------------------------------------

    /// Generate the PT_INTERP section content for x86-64 Linux.
    ///
    /// Returns the dynamic linker path as a null-terminated byte vector:
    /// `/lib64/ld-linux-x86-64.so.2\0`
    #[allow(dead_code)]
    fn generate_interp_section(&self) -> Vec<u8> {
        let mut data = DYNAMIC_LINKER.as_bytes().to_vec();
        data.push(0); // null terminator
        data
    }

    /// Resolve symbol names for relocations by looking up `sym_index` against
    /// each input object's symbol list.
    ///
    /// Relocations collected from section-level relocation tables have
    /// `sym_index` set but `symbol_name` empty. This method fills in the
    /// symbol name by looking up each input object's symbol table.
    fn resolve_relocation_symbol_names(
        &self,
        relocs: &mut [crate::backend::linker_common::relocation::Relocation],
    ) {
        // Build a map from (object_id, sym_index) → symbol name.
        let mut sym_name_map: FxHashMap<(u32, u32), String> = FxHashMap::default();
        for input in &self.inputs {
            for (idx, sym) in input.symbols.iter().enumerate() {
                sym_name_map.insert((input.object_id, idx as u32), sym.name.clone());
            }
        }

        for rel in relocs.iter_mut() {
            if rel.symbol_name.is_empty() {
                if let Some(name) = sym_name_map.get(&(rel.object_id, rel.sym_index)) {
                    rel.symbol_name = name.clone();
                }
            }
        }
    }

    /// Generate x86-64 PLT header stub (PLT\[0\]).
    ///
    /// PLT\[0\] pushes the `link_map` pointer from GOT\[1\] and jumps to
    /// `_dl_runtime_resolve` via GOT\[2\]:
    ///
    /// ```text
    /// ff 35 XX XX XX XX    push [rip + GOT[1]]   ; 6 bytes
    /// ff 25 XX XX XX XX    jmp  [rip + GOT[2]]   ; 6 bytes
    /// 0f 1f 40 00          nop dword [rax+0x0]   ; 4 bytes (padding)
    /// ```
    ///
    /// Total: 16 bytes. Uses RIP-relative addressing. The displacement is
    /// computed as `GOT_addr - (PLT_addr + instruction_length)`.
    #[allow(dead_code)]
    fn generate_plt0(&self, got_plt_addr: u64, plt_addr: u64) -> Vec<u8> {
        let mut buf = Vec::with_capacity(PLT0_SIZE);

        // push [rip + displacement_to_GOT[1]]
        // RIP points to the byte AFTER this 6-byte instruction.
        let rip_after_push = plt_addr + 6;
        let disp_push = ((got_plt_addr + 8) as i64 - rip_after_push as i64) as i32;
        buf.extend_from_slice(&[0xff, 0x35]);
        buf.extend_from_slice(&disp_push.to_le_bytes());

        // jmp [rip + displacement_to_GOT[2]]
        let rip_after_jmp = plt_addr + 12;
        let disp_jmp = ((got_plt_addr + 16) as i64 - rip_after_jmp as i64) as i32;
        buf.extend_from_slice(&[0xff, 0x25]);
        buf.extend_from_slice(&disp_jmp.to_le_bytes());

        // 4-byte NOP padding (0f 1f 40 00)
        buf.extend_from_slice(&[0x0f, 0x1f, 0x40, 0x00]);

        debug_assert_eq!(buf.len(), PLT0_SIZE);
        buf
    }

    /// Generate a per-function PLT entry (PLT\[n\]).
    ///
    /// Standard x86-64 PLT entry layout (16 bytes total):
    ///
    /// ```text
    /// ff 25 XX XX XX XX    jmp *GOT[n](%rip)     ; 6 bytes — indirect jump
    /// 68 NN NN NN NN       push <reloc_index>     ; 5 bytes — lazy binding index
    /// e9 XX XX XX XX       jmp PLT[0]             ; 5 bytes — fall to resolver
    /// ```
    #[allow(dead_code)]
    fn generate_plt_entry(
        &self,
        got_entry_addr: u64,
        plt_entry_addr: u64,
        plt0_addr: u64,
        reloc_index: u32,
    ) -> Vec<u8> {
        let mut buf = Vec::with_capacity(PLT_ENTRY_SIZE);

        // jmp *GOT[n](%rip)
        let rip_after_jmp = plt_entry_addr + 6;
        let disp = (got_entry_addr as i64 - rip_after_jmp as i64) as i32;
        buf.extend_from_slice(&[0xff, 0x25]);
        buf.extend_from_slice(&disp.to_le_bytes());

        // push <reloc_index>
        buf.push(0x68);
        buf.extend_from_slice(&reloc_index.to_le_bytes());

        // jmp PLT[0]
        let rip_after_jmp2 = plt_entry_addr + 16;
        let disp2 = (plt0_addr as i64 - rip_after_jmp2 as i64) as i32;
        buf.push(0xe9);
        buf.extend_from_slice(&disp2.to_le_bytes());

        debug_assert_eq!(buf.len(), PLT_ENTRY_SIZE);
        buf
    }
}

// ---------------------------------------------------------------------------
// Public Convenience Function
// ---------------------------------------------------------------------------

/// Link x86-64 object files into an ELF binary.
///
/// This is the main entry point for x86-64 linking from the code generation
/// driver ([`crate::backend::generation`]). It creates an [`X86_64Linker`],
/// feeds all inputs, and executes the link pipeline.
///
/// # Arguments
///
/// * `config`      — Linker configuration derived from CLI flags.
/// * `inputs`      — Relocatable object files to link.
/// * `diagnostics` — Diagnostic engine for error/warning reporting.
///
/// # Returns
///
/// On success, a [`LinkerOutput`] containing the serialized ELF binary.
/// On failure, a descriptive error string.
pub fn link_x86_64(
    config: LinkerConfig,
    inputs: Vec<LinkerInput>,
    diagnostics: &mut DiagnosticEngine,
) -> Result<LinkerOutput, String> {
    let mut linker = X86_64Linker::new(config);
    for input in inputs {
        linker.add_input(input);
    }
    linker.link(diagnostics)
}

// ---------------------------------------------------------------------------
// Unit Tests (inline — can access private methods)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_exec_config() -> LinkerConfig {
        LinkerConfig {
            target: Target::X86_64,
            output_type: OutputType::Executable,
            output_path: "test_output".to_string(),
            entry_point: "_start".to_string(),
            library_paths: Vec::new(),
            libraries: Vec::new(),
            pic: false,
            allow_undefined: false,
            soname: None,
            needed_libs: Vec::new(),
            emit_debug: false,
        }
    }

    // -- PLT0 stub tests --------------------------------------------------

    #[test]
    fn plt0_size_is_16_bytes() {
        let linker = X86_64Linker::new(make_exec_config());
        let plt0 = linker.generate_plt0(0x601000, 0x400000);
        assert_eq!(plt0.len(), PLT0_SIZE);
    }

    #[test]
    fn plt0_opcodes_correct() {
        let linker = X86_64Linker::new(make_exec_config());
        let plt0 = linker.generate_plt0(0x601000, 0x400000);
        assert_eq!(plt0[0], 0xff);
        assert_eq!(plt0[1], 0x35);
        assert_eq!(plt0[6], 0xff);
        assert_eq!(plt0[7], 0x25);
        assert_eq!(&plt0[12..16], &[0x0f, 0x1f, 0x40, 0x00]);
    }

    #[test]
    fn plt0_rip_relative_displacements() {
        let linker = X86_64Linker::new(make_exec_config());
        let plt0 = linker.generate_plt0(0x3000, 0x1000);
        let disp_push = i32::from_le_bytes([plt0[2], plt0[3], plt0[4], plt0[5]]);
        assert_eq!(disp_push, 0x2002);
        let disp_jmp = i32::from_le_bytes([plt0[8], plt0[9], plt0[10], plt0[11]]);
        assert_eq!(disp_jmp, 0x2004);
    }

    // -- PLT entry tests ---------------------------------------------------

    #[test]
    fn plt_entry_size_is_16_bytes() {
        let linker = X86_64Linker::new(make_exec_config());
        let entry = linker.generate_plt_entry(0x601018, 0x400010, 0x400000, 0);
        assert_eq!(entry.len(), PLT_ENTRY_SIZE);
    }

    #[test]
    fn plt_entry_opcodes_correct() {
        let linker = X86_64Linker::new(make_exec_config());
        let entry = linker.generate_plt_entry(0x601018, 0x400010, 0x400000, 42);
        assert_eq!(entry[0], 0xff);
        assert_eq!(entry[1], 0x25);
        assert_eq!(entry[6], 0x68);
        assert_eq!(
            u32::from_le_bytes([entry[7], entry[8], entry[9], entry[10]]),
            42
        );
        assert_eq!(entry[11], 0xe9);
    }

    #[test]
    fn plt_entry_jmp_displacement() {
        let linker = X86_64Linker::new(make_exec_config());
        let got_addr = 0x601018u64;
        let plt_addr = 0x400010u64;
        let entry = linker.generate_plt_entry(got_addr, plt_addr, 0x400000, 0);
        let disp = i32::from_le_bytes([entry[2], entry[3], entry[4], entry[5]]);
        let expected = (got_addr as i64 - (plt_addr as i64 + 6)) as i32;
        assert_eq!(disp, expected);
    }

    #[test]
    fn plt_entry_jmp_plt0_displacement() {
        let linker = X86_64Linker::new(make_exec_config());
        let plt0_addr = 0x400000u64;
        let plt_addr = 0x400010u64;
        let entry = linker.generate_plt_entry(0x601018, plt_addr, plt0_addr, 0);
        let disp = i32::from_le_bytes([entry[12], entry[13], entry[14], entry[15]]);
        let expected = (plt0_addr as i64 - (plt_addr as i64 + 16)) as i32;
        assert_eq!(disp, expected);
    }

    // -- Interp section test -----------------------------------------------

    #[test]
    fn interp_section_content() {
        let linker = X86_64Linker::new(make_exec_config());
        let interp = linker.generate_interp_section();
        assert_eq!(interp, b"/lib64/ld-linux-x86-64.so.2\0");
        assert_eq!(*interp.last().unwrap(), 0u8);
    }

    // -- Constant self-consistency -----------------------------------------

    #[test]
    fn got_entry_size_matches_pointer_width() {
        assert_eq!(GOT_ENTRY_SIZE, 8);
    }

    #[test]
    fn em_x86_64_matches_target() {
        let target = Target::X86_64;
        assert_eq!(target.elf_machine(), EM_X86_64);
    }
}
