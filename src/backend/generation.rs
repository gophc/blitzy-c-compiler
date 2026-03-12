//! # Phase 10 — Code Generation Driver
//!
//! This module is the central dispatch point for BCC's backend, orchestrating
//! the entire machine-code emission pipeline:
//!
//! 1. Receives phi-eliminated IR from Phase 9
//! 2. Dispatches to the correct architecture backend based on `--target` flag
//! 3. Injects security mitigations (retpoline, CET/IBT, stack probe) for x86-64
//! 4. Orchestrates relocatable object file (`.o`) emission
//! 5. Coordinates the built-in assembler and built-in linker for final ELF
//!    production (ET_EXEC or ET_DYN)
//!
//! ## Standalone Backend Mode
//!
//! BCC does **not** invoke any external toolchain component (no `as`, `ld`,
//! `gcc`, `llvm-mc`, `lld`).  Every step — instruction selection, register
//! allocation, assembly encoding, relocatable object writing, linking — is
//! performed by built-in modules.
//!
//! ## Security Mitigations (x86-64 Only)
//!
//! - **Retpoline** (`-mretpoline`): Indirect call/jump thunks
//! - **CET/IBT** (`-fcf-protection`): `endbr64` insertion
//! - **Stack Probe**: Guard-page probe loop for frames > 4096 bytes
//!
//! These mitigations are injected **only** for the x86-64 target.  Other
//! architectures are unaffected regardless of CLI flags.
//!
//! ## DWARF Debug Information
//!
//! When `-g` is active (and `-O0`), DWARF v4 sections are generated.
//! Without `-g`, **zero** `.debug_*` sections appear in the output.
//!
//! ## Zero-Dependency
//!
//! This module depends only on `std` and `crate::` references.  No external
//! crates are used.

use crate::backend::dwarf::{
    dwarf_address_size, generate_dwarf_sections, should_emit_dwarf, DwarfSections, DWARF_VERSION,
};
use crate::backend::elf_writer_common::{
    Elf32Rel, Elf64Rela, ElfSymbol, ElfWriter, Section, ET_REL, SHF_ALLOC, SHF_EXECINSTR,
    SHF_INFO_LINK, SHF_WRITE, SHT_NOBITS, SHT_PROGBITS, SHT_REL, SHT_RELA, STB_GLOBAL, STB_LOCAL,
    STT_FUNC, STT_NOTYPE, STT_OBJECT, STV_DEFAULT, STV_HIDDEN, STV_PROTECTED,
};
use crate::backend::linker_common::relocation::RelocationHandler;
use crate::backend::linker_common::{link, LinkerConfig, LinkerInput, LinkerOutput, OutputType};
use crate::backend::register_allocator::{
    self, allocate_registers, compute_live_intervals, AllocationResult,
};
use crate::backend::traits::{
    ArchCodegen, MachineFunction, MachineInstruction, MachineOperand,
    RegisterInfo as TraitsRegisterInfo,
};
use crate::common::diagnostics::{DiagnosticEngine, Span};
use crate::common::fx_hash::FxHashMap;
use crate::common::source_map::SourceMap;
use crate::common::target::Target;
use crate::ir::function::{IrFunction, Linkage as FunctionLinkage, Visibility};
use crate::ir::module::IrModule;

// ===========================================================================
// CodegenContext — compilation options for the code generation phase
// ===========================================================================

/// Compilation options collected from the CLI flags that control the
/// code generation phase.
///
/// `CodegenContext` is constructed by the CLI driver (`src/main.rs`) and
/// passed through the pipeline to every backend component — instruction
/// selector, register allocator, ELF writer, DWARF emitter, and linker.
///
/// # Security Mitigations
///
/// The `retpoline` and `cf_protection` fields only affect x86-64 targets.
/// When set for other architectures, they are silently ignored by the
/// code generation driver (no error — this matches GCC's behavior of
/// ignoring target-inapplicable flags).
///
/// # Examples
///
/// ```ignore
/// use bcc::backend::generation::CodegenContext;
/// use bcc::common::target::Target;
///
/// let ctx = CodegenContext {
///     target: Target::X86_64,
///     debug_info: true,
///     optimization_level: 0,
///     pic: false,
///     shared: false,
///     retpoline: true,
///     cf_protection: true,
///     output_path: "hello".to_string(),
///     compile_only: false,
///     emit_assembly: false,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct CodegenContext {
    /// Target architecture (x86-64, i686, AArch64, or RISC-V 64).
    ///
    /// Determines which `ArchCodegen` implementation is instantiated, which
    /// ELF header constants are used, and which ABI conventions apply.
    pub target: Target,

    /// Whether to emit DWARF v4 debug information (`-g` flag).
    ///
    /// When `true` and `optimization_level == 0`, the backend generates
    /// `.debug_info`, `.debug_abbrev`, `.debug_line`, and `.debug_str`
    /// sections.  When `false`, **zero** debug sections are present in the
    /// output binary.
    pub debug_info: bool,

    /// Optimization level (`-O0`, `-O1`, etc.).
    ///
    /// Currently only `-O0` is fully supported.  DWARF debug information
    /// is only emitted at `-O0`.  The value 0 corresponds to no optimization.
    pub optimization_level: u8,

    /// Whether to generate position-independent code (`-fPIC` flag).
    ///
    /// When `true`, global variable access uses GOT-relative addressing
    /// and function calls use PLT stubs.  Required for shared libraries.
    pub pic: bool,

    /// Whether to produce a shared object / shared library (`-shared` flag).
    ///
    /// When `true`, the output is an ET_DYN ELF with `.dynamic`, `.dynsym`,
    /// `.gnu.hash`, GOT, and PLT sections.  Implies PIC code generation.
    pub shared: bool,

    /// Whether to enable retpoline indirect-call thunks (`-mretpoline` flag).
    ///
    /// **x86-64 only.**  Transforms indirect call/jump instructions to use
    /// `__x86_indirect_thunk_*` thunks instead of direct `call *%reg`.
    /// Silently ignored for non-x86-64 targets.
    pub retpoline: bool,

    /// Whether to enable Intel CET / IBT (`-fcf-protection` flag).
    ///
    /// **x86-64 only.**  Inserts `endbr64` at function entries and indirect
    /// branch targets, and generates a `.note.gnu.property` section with
    /// CET flags.  Silently ignored for non-x86-64 targets.
    pub cf_protection: bool,

    /// Output file path (`-o` flag).
    ///
    /// The path where the final ELF binary (or assembly text, or
    /// relocatable object) is written.
    pub output_path: String,

    /// Whether to produce only a relocatable object file (`-c` flag).
    ///
    /// When `true`, the built-in linker is **not** invoked.  The output
    /// is a `.o` file (ET_REL) suitable for later linking.
    pub compile_only: bool,

    /// Whether to emit human-readable assembly text (`-S` flag).
    ///
    /// When `true`, the output is AT&T-syntax assembly (x86-64/i686) or
    /// standard assembly (AArch64/RISC-V 64) instead of binary machine code.
    pub emit_assembly: bool,

    /// Library search paths (`-L` flags).
    pub library_paths: Vec<String>,

    /// Libraries to link (`-l` flags — names without the `lib` prefix).
    pub libraries: Vec<String>,
}

impl CodegenContext {
    /// Returns `true` if DWARF debug sections should be generated.
    ///
    /// Delegates to [`should_emit_dwarf`] — DWARF is emitted only when
    /// both `-g` is active and the optimization level is 0.
    #[inline]
    pub fn should_emit_dwarf(&self) -> bool {
        should_emit_dwarf(self.debug_info, self.optimization_level)
    }

    /// Returns `true` if security mitigations are applicable.
    ///
    /// Security mitigations (retpoline, CET/IBT, stack probe) are only
    /// supported on x86-64.  This helper is used by the code generation
    /// driver to guard mitigation injection.
    #[inline]
    pub fn has_security_mitigations(&self) -> bool {
        self.target == Target::X86_64 && (self.retpoline || self.cf_protection)
    }

    /// Determines the linker output type based on CLI flags.
    ///
    /// - `-c` → `Relocatable` (no linking)
    /// - `-shared` → `SharedLibrary` (ET_DYN)
    /// - otherwise → `Executable` (ET_EXEC)
    pub fn output_type(&self) -> OutputType {
        if self.compile_only {
            OutputType::Relocatable
        } else if self.shared {
            OutputType::SharedLibrary
        } else {
            OutputType::Executable
        }
    }
}

impl Default for CodegenContext {
    fn default() -> Self {
        CodegenContext {
            target: Target::X86_64,
            debug_info: false,
            optimization_level: 0,
            pic: false,
            shared: false,
            retpoline: false,
            cf_protection: false,
            output_path: String::from("a.out"),
            compile_only: false,
            emit_assembly: false,
            library_paths: Vec::new(),
            libraries: Vec::new(),
        }
    }
}

// ===========================================================================
// generate_code — main entry point for Phase 10
// ===========================================================================

/// Main entry point for Phase 10 code generation.
///
/// Receives an IR module (post phi-elimination from Phase 9), dispatches
/// to the correct architecture backend based on the target specified in
/// `ctx`, and produces the final output — either a relocatable object,
/// an ELF executable, a shared library, or assembly text.
///
/// # Pipeline
///
/// 1. **Architecture dispatch**: Instantiate the correct `ArchCodegen`
///    implementation based on `ctx.target`.
/// 2. **Per-function code generation**: For each function definition in the
///    module, perform instruction selection, register allocation, and
///    assembly encoding.
/// 3. **Security mitigation injection** (x86-64 only): Apply retpoline,
///    CET/IBT, and stack probe transformations.
/// 4. **Global data emission**: Emit global variables and string literal
///    pool into `.data` / `.rodata` / `.bss` sections.
/// 5. **DWARF debug info** (conditional on `-g`): Generate `.debug_info`,
///    `.debug_abbrev`, `.debug_line`, `.debug_str` sections.
/// 6. **ELF object writing**: Produce a relocatable `.o` file.
/// 7. **Linking** (unless `-c`): Invoke the built-in linker to produce
///    the final ET_EXEC or ET_DYN ELF binary.
///
/// # Parameters
///
/// - `module`: The IR module to generate code for.
/// - `ctx`: Compilation options (target, flags, output path).
/// - `diagnostics`: Diagnostic engine for error/warning reporting.
///
/// # Returns
///
/// On success, returns `Ok(Vec<u8>)` containing the final output bytes
/// (ELF binary, relocatable object, or assembly text).
/// On failure, returns `Err(String)` with a description of the fatal error.
///
/// # Errors
///
/// - Instruction selection failure (unsupported IR constructs)
/// - Register allocation failure (internal consistency error)
/// - Assembly encoding failure (immediate out of range)
/// - DWARF generation errors
/// - Linking errors (undefined symbols, relocation overflow)
pub fn generate_code(
    module: &IrModule,
    ctx: &CodegenContext,
    diagnostics: &mut DiagnosticEngine,
    source_map: &SourceMap,
) -> Result<Vec<u8>, String> {
    // Dispatch to the appropriate architecture backend.
    match ctx.target {
        Target::X86_64 => {
            let security_config = crate::backend::x86_64::SecurityConfig {
                retpoline: ctx.retpoline,
                cf_protection: ctx.cf_protection,
                stack_probe: true, // Always enabled for x86-64
            };
            let codegen = crate::backend::x86_64::X86_64Backend::new(
                Target::X86_64,
                security_config,
                ctx.pic,
                ctx.shared,
                ctx.debug_info,
            );
            generate_for_arch(&codegen, module, ctx, diagnostics, source_map)
        }
        Target::I686 => {
            let codegen = crate::backend::i686::I686Codegen::new(ctx.pic, ctx.debug_info);
            generate_for_arch(&codegen, module, ctx, diagnostics, source_map)
        }
        Target::AArch64 => {
            let codegen = crate::backend::aarch64::AArch64Codegen::new(ctx.pic, ctx.debug_info);
            generate_for_arch(&codegen, module, ctx, diagnostics, source_map)
        }
        Target::RiscV64 => {
            let mut codegen = crate::backend::riscv64::RiscV64Codegen::new(ctx.pic, ctx.debug_info);
            // Populate the variadic function set from module declarations.
            // RISC-V LP64D ABI requires variadic FP arguments to be passed
            // in integer registers instead of FP registers.
            let variadic: crate::common::fx_hash::FxHashSet<String> = module
                .declarations()
                .iter()
                .filter(|d| d.is_variadic)
                .map(|d| d.name.clone())
                .collect();
            codegen.set_variadic_functions(variadic);
            generate_for_arch(&codegen, module, ctx, diagnostics, source_map)
        }
    }
}

// ===========================================================================
// generate_for_arch — architecture-agnostic code generation pipeline
// ===========================================================================

/// Architecture-agnostic code generation pipeline.
///
/// Given an `ArchCodegen` implementation, processes the entire IR module
/// through instruction selection, register allocation, assembly encoding,
/// optional DWARF generation, ELF object writing, and optional linking.
///
/// This function is the core workhorse of Phase 10.  It is called once per
/// compilation with the architecture-specific backend selected by
/// [`generate_code`].
///
/// # Type Parameter
///
/// - `A`: An architecture backend implementing [`ArchCodegen`].
///
/// # Pipeline Steps
///
/// 1. For each function definition in the module:
///    a. `arch.lower_function()` → instruction selection → `MachineFunction`
///    b. `allocate_registers()` → register allocation
///    c. `arch.emit_prologue()` / `arch.emit_epilogue()` → frame setup/teardown
///    d. `arch.emit_assembly()` → encode to raw bytes
/// 2. Emit globals into `.data` / `.rodata` / `.bss` sections.
/// 3. Emit string literal pool into `.rodata`.
/// 4. If `-g`: generate DWARF sections; else: zero debug sections.
/// 5. Write the relocatable object via `ElfWriter`.
/// 6. If not `-c`: invoke the built-in linker.
fn generate_for_arch<A: ArchCodegen>(
    arch: &A,
    module: &IrModule,
    ctx: &CodegenContext,
    diagnostics: &mut DiagnosticEngine,
    source_map: &SourceMap,
) -> Result<Vec<u8>, String> {
    // If `-S` is specified, emit assembly text instead of binary.
    if ctx.emit_assembly {
        return emit_assembly_text(arch, module, ctx, diagnostics);
    }

    // Validate that the architecture backend's target matches the context.
    let backend_target = arch.target();
    if backend_target != ctx.target {
        diagnostics.emit_error(
            Span::dummy(),
            format!(
                "codegen: backend target mismatch: expected {}, got {}",
                ctx.target, backend_target
            ),
        );
        return Err("code generation aborted: target mismatch".to_string());
    }

    // Warn if security mitigations are requested on non-x86-64 targets.
    if (ctx.retpoline || ctx.cf_protection) && ctx.target != Target::X86_64 {
        diagnostics.emit_warning(
            Span::dummy(),
            format!(
                "codegen: security mitigations (retpoline={}, cf_protection={}) are only supported on x86-64; ignoring for {}",
                ctx.retpoline, ctx.cf_protection, ctx.target
            ),
        );
    }

    // Get the register info for this architecture (used for register allocation).
    let reg_info: TraitsRegisterInfo = arch.register_info();

    // Convert to the register allocator's internal RegisterInfo representation.
    let alloc_reg_info: register_allocator::RegisterInfo = reg_info.into();

    // -----------------------------------------------------------------------
    // Phase A: Per-Function Code Generation
    // -----------------------------------------------------------------------
    // For each function definition: instruction selection → register allocation
    // → prologue/epilogue → assembly encoding.

    let mut compiled_functions: Vec<CompiledFunction> = Vec::new();
    let mut text_section_data: Vec<u8> = Vec::new();
    let mut text_section_offsets: Vec<(String, u64, u64)> = Vec::new();
    let mut text_relocations: Vec<crate::backend::traits::FunctionRelocation> = Vec::new();

    let functions: &[IrFunction] = module.functions();
    for func in functions {
        // Skip declaration-only functions (no body to compile).
        if !func.is_definition {
            continue;
        }

        // Step 1: Instruction selection — IR → MachineFunction
        //
        // Use the per-function ref maps (scoped to this function's Value IDs)
        // rather than the module-level maps which contain Values from ALL
        // functions and suffer from cross-function Value-ID collisions.
        let mut mf = match arch.lower_function(
            func,
            diagnostics,
            module.globals(),
            &func.func_ref_map,
            &func.global_var_refs,
        ) {
            Ok(mf) => mf,
            Err(e) => {
                diagnostics.emit_error(
                    Span::dummy(),
                    format!(
                        "codegen: instruction selection failed for '{}': {}",
                        func.name, e
                    ),
                );
                if diagnostics.has_errors() {
                    return Err(format!(
                        "code generation aborted: instruction selection failed for '{}'",
                        func.name
                    ));
                }
                continue;
            }
        };

        // Step 2: Register allocation
        //
        // Compute live intervals from the IR function's blocks() and
        // parameter/return-type metadata, then allocate physical registers.
        let _num_blocks = func.blocks().len();
        let _num_params = func.params.len();
        let _ret_type = &func.return_type;

        let mut intervals = compute_live_intervals(func);
        let alloc_result = allocate_registers(&mut intervals, &alloc_reg_info, &ctx.target);

        // Insert spill/reload code for any virtual registers that couldn't
        // be assigned a physical register.  The spill code is recorded in the
        // allocation result and applied to the MachineFunction below —
        // modifying the IR function is not required.
        insert_spill_code_from_result(&mut mf, &alloc_result);

        // Update MachineFunction with allocation results.
        apply_allocation_result(&mut mf, &alloc_result, &ctx.target);

        // Resolve integer-division register conflicts.
        // After register allocation, the divisor may be assigned to RAX or
        // RDX — which get clobbered by the dividend setup sequence (MOV RAX
        // + CQO/CDQ for signed, XOR RDX + MOV RAX for unsigned).
        if matches!(ctx.target, Target::X86_64) {
            resolve_div_conflicts(&mut mf);
        }

        // Resolve call-argument register move conflicts.
        // After register allocation, argument setup moves like
        //   MOV RSI, <vreg_x>  ; MOV RDX, <vreg_y>
        // may conflict when vreg_y was allocated to RSI (clobbered
        // by the first MOV before the second reads it).  This pass
        // detects and fixes such conflicts using a target-appropriate
        // scratch register (R11 for x86-64, T0 for RISC-V, X16 for
        // AArch64).
        resolve_call_arg_conflicts(&mut mf, &ctx.target);

        // Resolve parameter-loading MOV conflicts at the function entry.
        // After register allocation, parameter MOVs (e.g., MOV RCX, RSI)
        // may clobber ABI registers that are sources for later parameter
        // loads.  This pass applies parallel-move sequentialisation.
        if matches!(ctx.target, Target::X86_64) {
            resolve_param_load_conflicts(&mut mf);
        }

        // Step 3: Insert prologue and epilogue instructions.
        let prologue = arch.emit_prologue(&mf);
        let epilogue = arch.emit_epilogue(&mf);
        insert_prologue_epilogue(&mut mf, prologue, epilogue);

        // Step 4: Assembly encoding — machine instructions → raw bytes + relocations
        let asm_result = match arch.emit_assembly(&mf) {
            Ok(asm) => asm,
            Err(e) => {
                diagnostics.emit_error(
                    Span::dummy(),
                    format!(
                        "codegen: assembly encoding failed for '{}': {}",
                        func.name, e
                    ),
                );
                if diagnostics.has_errors() {
                    return Err(format!(
                        "code generation aborted: assembly encoding failed for '{}'",
                        func.name
                    ));
                }
                continue;
            }
        };

        let func_bytes = asm_result.bytes;

        // Track the function's position within the .text section.
        let func_offset = text_section_data.len() as u64;
        let func_size = func_bytes.len() as u64;

        // Collect relocations, adjusting offsets to be relative to the
        // combined .text section rather than this individual function.
        for mut reloc in asm_result.relocations {
            reloc.offset += func_offset;
            text_relocations.push(reloc);
        }
        text_section_offsets.push((func.name.clone(), func_offset, func_size));

        // Map IR visibility enum to ELF STV_* constant for symbol table emission.
        let elf_visibility = match func.visibility {
            Visibility::Hidden => STV_HIDDEN,
            Visibility::Protected => STV_PROTECTED,
            Visibility::Default => STV_DEFAULT,
        };

        compiled_functions.push(CompiledFunction {
            name: func.name.clone(),
            bytes: func_bytes.clone(),
            offset: func_offset,
            size: func_size,
            is_global: func.linkage == FunctionLinkage::External
                || func.linkage == FunctionLinkage::Weak,
            visibility: elf_visibility,
        });

        text_section_data.extend_from_slice(&func_bytes);
    }

    // Check for errors after function compilation.
    if diagnostics.has_errors() {
        return Err("code generation aborted due to errors".to_string());
    }

    // -----------------------------------------------------------------------
    // Phase A.4b: Append retpoline thunks to .text (x86-64 only)
    // -----------------------------------------------------------------------
    // When -mretpoline is active, the assembler transforms indirect
    // call/jmp instructions into `CALL __x86_indirect_thunk_<reg>`.
    // The thunk bodies must be physically present in the .text section
    // and registered in the function offset map so that Phase A.5 can
    // resolve the CALL relocations targeting them.
    if ctx.retpoline && ctx.target == Target::X86_64 {
        let thunks = crate::backend::x86_64::assembler::assemble_retpoline_thunks();
        for (thunk_name, thunk_bytes) in &thunks {
            let thunk_offset = text_section_data.len() as u64;
            let thunk_size = thunk_bytes.len() as u64;
            text_section_offsets.push((thunk_name.clone(), thunk_offset, thunk_size));
            compiled_functions.push(CompiledFunction {
                name: thunk_name.clone(),
                bytes: thunk_bytes.clone(),
                offset: thunk_offset,
                size: thunk_size,
                is_global: true,
                visibility: STV_DEFAULT,
            });
            text_section_data.extend_from_slice(thunk_bytes);
        }
    }

    // -----------------------------------------------------------------------
    // Phase A.5: Resolve intra-module text relocations
    // -----------------------------------------------------------------------
    // Build a map from function name → offset in combined .text section.
    // For relocations targeting defined functions, patch the .text bytes
    // directly (PC-relative CALL/JMP).  Remaining relocations (external
    // symbols) are carried forward to the ELF object for the linker.
    //
    // IMPORTANT: The patching format is architecture-dependent.
    // - x86-64 / i686: Write raw 4-byte little-endian PC-relative offset
    //   (R_X86_64_PC32, R_X86_64_PLT32, R_386_PC32 all use 32-bit addend).
    // - AArch64: Encode offset/4 into the instruction's immediate field
    //   while preserving the opcode bits (BL=imm26, B.cond=imm19, etc.).
    // - RISC-V 64: Architecture-specific encoding per relocation type.
    {
        let mut func_offset_map = crate::common::fx_hash::FxHashMap::default();
        for (fname, foff, _fsz) in &text_section_offsets {
            func_offset_map.insert(fname.as_str(), *foff);
        }
        let mut unresolved = Vec::new();
        for reloc in text_relocations.drain(..) {
            if let Some(&target_offset) = func_offset_map.get(reloc.symbol.as_str()) {
                // Intra-module relocation — patch text_section_data directly.
                // Standard ELF PC-relative formula: S + A - P
                let s = target_offset as i64;
                let a = reloc.addend;
                let p = reloc.offset as i64;
                let value = s + a - p;
                let fixup_off = reloc.offset as usize;
                if fixup_off + 4 <= text_section_data.len() {
                    patch_intra_module_reloc(
                        &mut text_section_data,
                        fixup_off,
                        value,
                        reloc.rel_type_id,
                        &ctx.target,
                    );
                }
            } else {
                // External symbol — keep for the linker.
                unresolved.push(reloc);
            }
        }
        text_relocations = unresolved;
    }

    // -----------------------------------------------------------------------
    // Phase B: Global Data Emission
    // -----------------------------------------------------------------------

    let mut data_section: Vec<u8> = Vec::new();
    let mut rodata_section: Vec<u8> = Vec::new();
    let mut bss_size: u64 = 0;
    let mut global_symbols: Vec<GlobalSymbolInfo> = Vec::new();

    for global in module.globals() {
        let sym_info = emit_global_variable(
            global,
            &mut data_section,
            &mut rodata_section,
            &mut bss_size,
            ctx,
        );
        global_symbols.push(sym_info);
    }

    // -----------------------------------------------------------------------
    // Phase C: String Literal Pool → .rodata
    // -----------------------------------------------------------------------

    let string_pool_offset = rodata_section.len() as u64;
    for string_lit in module.string_pool() {
        let str_offset = rodata_section.len() as u64;
        rodata_section.extend_from_slice(&string_lit.bytes);

        global_symbols.push(GlobalSymbolInfo {
            name: string_lit.label.clone(),
            section: SectionPlacement::Rodata,
            offset: str_offset,
            size: string_lit.bytes.len() as u64,
            is_global: false,
        });
    }
    // Suppress unused-variable warning on string_pool_offset — it's tracked
    // for future use (e.g., relocation generation for string references).
    let _ = string_pool_offset;

    // -----------------------------------------------------------------------
    // Phase D: DWARF Debug Information (conditional on -g)
    // -----------------------------------------------------------------------

    let dwarf_sections = if ctx.should_emit_dwarf() {
        // Use the source_map passed from the frontend, which contains all files
        // registered during preprocessing (file ID 0 = the primary source file).

        // Verify DWARF parameters for this target:
        // - Address size is 4 bytes on 32-bit targets, 8 bytes on 64-bit targets
        // - DWARF version is always v4 for BCC
        let addr_size = dwarf_address_size(&ctx.target);
        debug_assert!(
            (ctx.target.is_64bit() && addr_size == 8) || (!ctx.target.is_64bit() && addr_size == 4),
            "DWARF address size mismatch for target"
        );
        debug_assert_eq!(DWARF_VERSION, 4, "BCC targets DWARF v4");

        let dwarf = generate_dwarf_sections(
            module,
            source_map,
            &ctx.target,
            &text_section_offsets,
            diagnostics,
        );
        Some(dwarf)
    } else {
        // No -g flag: ZERO debug sections in the output.
        None
    };

    // -----------------------------------------------------------------------
    // Phase E: ELF Object Writing
    // -----------------------------------------------------------------------

    let object_bytes = write_relocatable_object(
        &text_section_data,
        &data_section,
        &rodata_section,
        bss_size,
        &compiled_functions,
        &global_symbols,
        &text_relocations,
        &dwarf_sections,
        module,
        ctx,
        diagnostics,
    )?;

    // If -c (compile-only), return the .o bytes directly.
    if ctx.compile_only {
        return Ok(object_bytes);
    }

    // -----------------------------------------------------------------------
    // Phase F: Linking Orchestration
    // -----------------------------------------------------------------------

    link_to_final_output(&object_bytes, &text_relocations, ctx, diagnostics)
}

// ===========================================================================
// Internal data structures
// ===========================================================================

/// Information about a compiled function within the .text section.
#[allow(dead_code)]
struct CompiledFunction {
    /// Function symbol name.
    name: String,
    /// Encoded machine code bytes.
    bytes: Vec<u8>,
    /// Byte offset within .text.
    offset: u64,
    /// Size of the function in bytes.
    size: u64,
    /// Whether this function has external/global linkage.
    is_global: bool,
    /// ELF symbol visibility (STV_DEFAULT, STV_HIDDEN, STV_PROTECTED).
    /// Propagated from the IR function's visibility attribute.
    visibility: u8,
}

/// Placement of a global symbol within an ELF section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SectionPlacement {
    /// `.data` section (initialized, writable).
    Data,
    /// `.rodata` section (initialized, read-only).
    Rodata,
    /// `.bss` section (zero-initialized, writable).
    Bss,
}

/// Information about a global variable or string literal for the symbol table.
struct GlobalSymbolInfo {
    /// Symbol name.
    name: String,
    /// Which section the symbol resides in.
    section: SectionPlacement,
    /// Byte offset within the section.
    offset: u64,
    /// Size of the data in bytes.
    size: u64,
    /// Whether the symbol has external linkage.
    is_global: bool,
}

// ===========================================================================
// insert_spill_code_from_result — record spill metadata on MachineFunction
// ===========================================================================

/// Records spill slot information from the register allocation result
/// directly on the `MachineFunction`.
///
/// Instead of cloning the IR function and modifying it, this function
/// updates the `MachineFunction`'s frame size to account for all spill
/// slots.  Actual spill loads/stores are handled by the architecture's
/// prologue/epilogue emission which reads the frame size.
fn insert_spill_code_from_result(mf: &mut MachineFunction, alloc: &AllocationResult) {
    if alloc.spill_map.is_empty() {
        return;
    }
    // Each spill slot occupies 8 bytes (pointer-width).  Accumulate the
    // total spill area size and update the frame size so prologue/epilogue
    // allocate sufficient stack space.
    let spill_area = alloc.spill_slots.len() * 8;
    mf.frame_size = mf.frame_size.max(alloc.frame_size).max(spill_area);
}

// ===========================================================================
// apply_allocation_result — update MachineFunction with regalloc results
// ===========================================================================

/// Applies register allocation results to a `MachineFunction`.
///
/// Replaces `VirtualRegister` operands with physical `Register` operands
/// based on the allocation mapping.  Updates the function's callee-saved
/// register list with registers that were actually used.
///
/// For **spilled** virtual registers (those assigned a stack slot instead
/// of a physical register), this function inserts machine-level spill
/// loads and stores:
/// - **Before each USE** of a spilled vreg: inserts `MOV scratch, [RBP-offset]`
///   and replaces the operand with the scratch register.
/// - **After each DEF** of a spilled vreg: replaces the result with a scratch
///   register and inserts `MOV [RBP-offset], scratch`.
///
/// The scratch register is chosen dynamically from caller-saved registers
/// that are not otherwise used by the current instruction.
fn apply_allocation_result(mf: &mut MachineFunction, alloc: &AllocationResult, target: &Target) {
    // Record callee-saved registers used by this function.
    mf.callee_saved_regs = alloc.callee_saved_used.iter().map(|pr| pr.0).collect();

    // Update frame size with spill slot requirements.
    mf.frame_size = mf.frame_size.max(alloc.frame_size);

    // Build a set of codegen vreg IDs from vreg_to_ir_value.
    // CRITICAL: This set is used to guard ALL fallback loops below that
    // use val.index() as a vreg number.  Without this guard, val.index()
    // can collide with codegen vreg IDs, causing physical-register vregs
    // to be incorrectly treated as spilled (or vice versa).
    let codegen_vreg_ids: crate::common::fx_hash::FxHashSet<u32> =
        mf.vreg_to_ir_value.keys().copied().collect();

    // Build a complete vreg → physical register lookup table upfront.
    // Primary path: map codegen vregs via vreg_to_ir_value.
    let mut vreg_phys: crate::common::fx_hash::FxHashMap<u32, u16> =
        crate::common::fx_hash::FxHashMap::default();
    for (&vreg, ir_val) in &mf.vreg_to_ir_value {
        if let Some(phys) = alloc.assignments.get(ir_val) {
            vreg_phys.insert(vreg, phys.0);
        }
    }
    // Fallback: for vregs not in vreg_to_ir_value, use val.index().
    // entry().or_insert() ensures codegen vregs already in the map
    // (from the primary loop) are NOT overwritten.
    for (val, phys) in &alloc.assignments {
        let vreg = val.index();
        vreg_phys.entry(vreg).or_insert(phys.0);
    }

    // Debug: dump vreg→phys mapping and vreg_to_ir_value
    if std::env::var("BCC_DEBUG_REGALLOC").is_ok() {
        eprintln!("=== VREG_TO_IR_VALUE ===");
        let mut pairs: Vec<_> = mf.vreg_to_ir_value.iter().collect();
        pairs.sort_by_key(|(&k, _)| k);
        for (&vreg, ir_val) in &pairs {
            let phys = vreg_phys.get(&vreg);
            eprintln!(
                "  vreg={} -> ir_v{} -> phys={:?}",
                vreg,
                ir_val.index(),
                phys
            );
        }
        eprintln!("=== VREG_PHYS (full) ===");
        let mut vp: Vec<_> = vreg_phys.iter().collect();
        vp.sort_by_key(|(&k, _)| k);
        for (&vreg, &phys) in &vp {
            eprintln!("  vreg={} -> r{}", vreg, phys);
        }
    }

    // Build spilled vreg → frame offset mapping.
    //
    // Frame layout conventions differ by architecture:
    //
    // **x86-64 / i686:** FP (RBP) points to the TOP of the frame.  Locals
    //   and spill slots live at *negative* offsets from RBP.
    //
    // **AArch64 / RISC-V 64:** FP points to the BOTTOM of the frame
    //   (FP = SP after the prologue).  The prologue allocates:
    //     total = 16 (FP/LR) + callee_save_bytes + mf.frame_size
    //   **AArch64:** FP/LR are at [FP+0..+15].  Locals occupy
    //   [FP+16 .. FP+16+locals_size).  Callee-saved regs are placed
    //   AFTER locals at [FP+16+locals_size .. +callee_save_bytes).
    //   Spill slots go right after locals (within mf.frame_size), so
    //   they use FP + 16 + existing_locals as their base.
    //
    //   **RISC-V 64:** FP points to old_SP (top of frame).  Locals and
    //   spills are at negative offsets, but the spill base computation
    //   still uses 16 + callee_save_bytes + base_offset as positive.
    let base_offset = mf.frame_size;
    let is_aarch64_or_rv = matches!(target, Target::AArch64 | Target::RiscV64);
    let a64_rv_spill_base: i64 = if is_aarch64_or_rv {
        if matches!(target, Target::AArch64) {
            // AArch64: spills start at FP + 16 + existing locals_size.
            // No callee-saved offset because callee-saved regs are AFTER
            // both locals and spills in the AArch64 frame layout.
            (16 + base_offset) as i64
        } else {
            // RISC-V: keep the original formula.
            let num_gpr = mf
                .callee_saved_regs
                .iter()
                .filter(|&&r| (r as u32) < 32)
                .count();
            let num_fpr = mf
                .callee_saved_regs
                .iter()
                .filter(|&&r| (r as u32) >= 32)
                .count();
            let gpr_pairs = (num_gpr + 1) / 2;
            let fpr_pairs = (num_fpr + 1) / 2;
            let callee_save_bytes = (gpr_pairs + fpr_pairs) * 16;
            (16 + callee_save_bytes + base_offset) as i64
        }
    } else {
        0
    };

    let mut vreg_spill_offset: crate::common::fx_hash::FxHashMap<u32, i64> =
        crate::common::fx_hash::FxHashMap::default();

    // Primary path: map codegen vregs to spill offsets via vreg_to_ir_value.
    for (&vreg, ir_val) in &mf.vreg_to_ir_value {
        if let Some(&slot_idx) = alloc.spill_map.get(ir_val) {
            let offset = if is_aarch64_or_rv {
                // Positive offset from FP for AArch64/RV.
                a64_rv_spill_base + 8 * (slot_idx as i64)
            } else {
                // Negative offset from RBP for x86.
                -((base_offset as i64) + 8 * (slot_idx as i64 + 1))
            };
            vreg_spill_offset.insert(vreg, offset);
        }
    }
    // Fallback path: for IR values whose val.index() is used directly
    // as a vreg in machine instructions (not via vreg_to_ir_value).
    // CRITICAL: Skip entries where val.index() collides with an existing
    // codegen vreg ID — those codegen vregs may refer to completely
    // different IR values and must NOT be marked as spilled.
    for (val, &slot_idx) in &alloc.spill_map {
        let vreg = val.index();
        // Only add if this vreg ID is NOT a known codegen vreg.
        if !codegen_vreg_ids.contains(&vreg) {
            let offset = if is_aarch64_or_rv {
                a64_rv_spill_base + 8 * (slot_idx as i64)
            } else {
                -((base_offset as i64) + 8 * (slot_idx as i64 + 1))
            };
            vreg_spill_offset.entry(vreg).or_insert(offset);
        }
    }

    // Expand frame size to include spill area.
    if !vreg_spill_offset.is_empty() {
        let spill_area = alloc.spill_slots.len() * 8;
        mf.frame_size += spill_area;
    }

    // Architecture-specific constants for spill code.
    //
    // Frame pointer register:
    // - x86-64 / i686: RBP = 5
    // - AArch64: X29 (FP) = 29
    // - RISC-V 64: X8 (s0/fp) = 8
    let frame_ptr_reg: u16 = match target {
        Target::X86_64 | Target::I686 => 5, // RBP
        Target::AArch64 => 29,              // X29 / FP
        Target::RiscV64 => 8,               // X8 / s0 / fp
    };
    // MOV/load opcodes are architecture-dependent:
    // x86-64: X86Opcode::Mov = 0, X86Opcode::Movsd = 86
    // i686:   I686_MOV = 0x100
    // AArch64: LDR_imm = 59 (A64Opcode::LDR_imm enum discriminant)
    // RISC-V 64: LD = 13 (RvOpcode::LD)
    //
    // IMPORTANT: AArch64 opcode values MUST match A64Opcode enum
    // discriminants exactly.  If variants are added to the enum
    // (e.g. MUL, SMULL), all subsequent discriminants shift.
    // Current mapping: LDR_imm=59, STR_imm=72.
    let mov_opcode: u32 = match target {
        Target::I686 => 0x100, // I686_MOV
        Target::AArch64 => 59, // A64Opcode::LDR_imm (64-bit load)
        Target::RiscV64 => 13, // RvOpcode::LD
        _ => 0,                // X86Opcode::Mov
    };
    let movsd_opcode: u32 = match target {
        Target::I686 => 0x100, // i686 uses MOV for all spills (no SSE)
        Target::AArch64 => 59, // AArch64: same LDR_imm for FP spills
        Target::RiscV64 => 13, // RISC-V: LD for FP spills (stored as 64-bit)
        _ => 86,               // X86Opcode::Movsd
    };
    // Store opcodes for spill stores:
    // x86-64 / i686: same opcode as load (MOV is bidirectional via operand layout)
    // AArch64: STR_imm = 72 (A64Opcode::STR_imm enum discriminant)
    // RISC-V 64: SD = 20 (RvOpcode::SD)
    let store_opcode: u32 = match target {
        Target::AArch64 => 72, // A64Opcode::STR_imm
        Target::RiscV64 => 20, // RvOpcode::SD
        _ => mov_opcode,       // x86: same opcode for load/store
    };
    let store_fp_opcode: u32 = match target {
        Target::AArch64 => 72, // AArch64 STR_imm
        Target::RiscV64 => 20, // RISC-V SD
        _ => movsd_opcode,     // x86: MOVSD for float stores
    };
    // Spill scratch register is architecture-dependent:
    // - x86-64: R11 (index 11) — reserved from GPR allocation
    // - i686: ECX (index 1) — caller-saved, used as scratch for spills
    // - AArch64: X16 (IP0, index 16) — scratch register
    // - RISC-V 64: X5 (T0, index 5) — temporary register
    let spill_scratch: u16 = match target {
        Target::I686 => 1,     // ECX
        Target::AArch64 => 16, // X16 / IP0
        Target::RiscV64 => 5,  // X5 / T0
        _ => 11,               // R11 (x86-64)
    };
    let float_spill_scratch: u16 = match target {
        Target::I686 => 1, // ECX (i686 uses x87 stack, no XMM15)
        _ => 31,           // XMM15
    };

    // Build a set of codegen vregs that hold float types.
    // CRITICAL: Must use BOTH the vreg_to_ir_value mapping AND direct
    // Value.index() to ensure coverage, since codegen vreg numbering
    // may differ from IR Value indices.
    let mut vreg_is_float: crate::common::fx_hash::FxHashSet<u32> =
        crate::common::fx_hash::FxHashSet::default();
    // Path 1: Map through vreg_to_ir_value (codegen vreg → IR Value → type).
    for (&vreg, ir_val) in &mf.vreg_to_ir_value {
        // Check if this IR value is spilled AND is float.
        if alloc.spill_map.contains_key(ir_val) {
            if let Some(ty) = alloc.value_types.get(ir_val) {
                if ty.is_float() {
                    vreg_is_float.insert(vreg);
                }
            }
        }
        // Check if assigned to an SSE register.
        if let Some(phys) = alloc.assignments.get(ir_val) {
            if phys.0 >= 16 && phys.0 < 32 {
                vreg_is_float.insert(vreg);
            }
        }
    }
    // Path 2: Direct fallback using Value.index() as vreg (for vregs
    // not in vreg_to_ir_value — e.g. the second pass of the mapping).
    // CRITICAL: Skip entries where val.index() collides with a known
    // codegen vreg to prevent marking an integer codegen vreg as float.
    for val in alloc.spill_map.keys() {
        if !codegen_vreg_ids.contains(&val.index()) {
            if let Some(ty) = alloc.value_types.get(val) {
                if ty.is_float() {
                    vreg_is_float.insert(val.index());
                }
            }
        }
    }
    for (val, phys) in &alloc.assignments {
        if !codegen_vreg_ids.contains(&val.index()) && phys.0 >= 16 && phys.0 < 32 {
            vreg_is_float.insert(val.index());
        }
    }

    // Helper: select scratch register and opcode based on float/int class.
    // Helper: select scratch register and LOAD opcode based on float/int class.
    let spill_scratch_for = |is_float: bool| -> (u16, u32) {
        if is_float {
            (float_spill_scratch, movsd_opcode)
        } else {
            (spill_scratch, mov_opcode)
        }
    };
    // Helper: select scratch register and STORE opcode based on float/int class.
    // On x86, load/store use the same opcode (MOV/MOVSD are bidirectional).
    // On RISC-V/AArch64, stores have separate opcodes (SD/STR vs LD/LDR).
    let spill_store_for = |is_float: bool| -> (u16, u32) {
        if is_float {
            (float_spill_scratch, store_fp_opcode)
        } else {
            (spill_scratch, store_opcode)
        }
    };

    // Helper: make a load instruction for spill reload.
    //
    // Architecture-specific layouts:
    // - i686:     result = scratch, operands = [Memory{base=RBP, disp=offset}]
    // - x86-64:   operands = [Register(scratch), Memory{base=RBP, disp=offset}]
    // - AArch64:  result = Register(scratch), operands = [Memory{base=FP, disp=offset}]
    // - RISC-V:   result = Register(scratch), operands = [Memory{base=FP, disp=offset}]
    //             → machine_to_rv_instruction: rd=scratch, rs1=fp, imm=offset → LD scratch, offset(fp)
    let is_i686 = matches!(target, Target::I686);
    let is_risc_or_arm = matches!(target, Target::RiscV64 | Target::AArch64);
    let make_spill_load = |scratch: u16, offset: i64, opcode: u32| -> MachineInstruction {
        let mem = MachineOperand::Memory {
            base: Some(frame_ptr_reg),
            index: None,
            scale: 1,
            displacement: offset,
        };
        if is_i686 || is_risc_or_arm {
            // i686 / RISC-V / AArch64: result = rd, operand = Memory(base+disp)
            let mut inst = MachineInstruction::new(opcode);
            inst.operands.push(mem);
            inst.result = Some(MachineOperand::Register(scratch));
            inst
        } else {
            // x86-64: operands = [Register(scratch), Memory]
            let mut inst = MachineInstruction::new(opcode);
            inst.operands.push(MachineOperand::Register(scratch));
            inst.operands.push(mem);
            inst
        }
    };

    // Helper: make a store instruction for spill save.
    //
    // Architecture-specific layouts:
    // - x86-64/i686: operands = [Memory{base=RBP, disp=offset}, Register(scratch)]
    // - AArch64:     operands = [Memory{base=FP, disp=offset}, Register(scratch)]
    //                → machine_to_a64_instruction: maps to STR scratch, [fp, #offset]
    // - RISC-V:      operands = [Memory{base=FP, disp=offset}, Register(scratch)]
    //                → machine_to_rv_instruction: rs1=fp, rs2=scratch, imm=offset → SD scratch, offset(fp)
    let make_spill_store = |scratch: u16, offset: i64, opcode: u32| -> MachineInstruction {
        let mut inst = MachineInstruction::new(opcode);
        inst.operands.push(MachineOperand::Memory {
            base: Some(frame_ptr_reg),
            index: None,
            scale: 1,
            displacement: offset,
        });
        inst.operands.push(MachineOperand::Register(scratch));
        inst
    };

    // Process each block: replace VRs with physical registers or insert spill code.
    //
    // ## Critical x86 Spill Strategy
    //
    // x86 ALU instructions are **destructive**: `ADD dst, src` computes
    // `dst = dst + src`.  The BCC codegen emits these as:
    //
    //   result = Some(dst_vreg), operands = [src_vreg]
    //
    // The encoder prepends `result` into operands[0], yielding `[dst, src]`.
    // The destination register is therefore BOTH a source (implicit read)
    // and a target (write).
    //
    // When the result vreg is spilled we must:
    //   1. LOAD  R11 ← [RBP + result_offset]   (get the old value — implicit source)
    //   2. Instruction: e.g. ADD R11, <src>      (R11 = old + src)
    //   3. STORE [RBP + result_offset] ← R11    (save new value)
    //
    // Because R11 is occupied by the result, any spilled *explicit operand*
    // must use a **Memory** operand so there is no scratch-register conflict.
    //
    // When the result is NOT spilled (or absent), R11 is free and can be
    // used for the first spilled explicit operand.

    for block in &mut mf.blocks {
        let old_insts = std::mem::take(&mut block.instructions);
        let mut new_insts: Vec<MachineInstruction> =
            Vec::with_capacity(old_insts.len() + old_insts.len() / 2);

        for mut inst in old_insts {
            // ----------------------------------------------------------
            // Step 1: Determine if the result is spilled.
            // ----------------------------------------------------------
            let result_spill_offset: Option<i64> = inst.result.as_ref().and_then(|r| {
                if let MachineOperand::VirtualRegister(vreg) = r {
                    vreg_spill_offset.get(vreg).copied()
                } else {
                    None
                }
            });

            // Detect if the result vreg is float (needs XMM15 scratch).
            let result_is_float = inst.result.as_ref().map_or(false, |r| {
                if let MachineOperand::VirtualRegister(vreg) = r {
                    vreg_is_float.contains(vreg)
                } else {
                    false
                }
            });

            // ----------------------------------------------------------
            // Step 2: Insert a load for the spilled result BEFORE the
            //         instruction so that the scratch reg holds the old
            //         value. Uses R11 for int, XMM15 for float.
            // ----------------------------------------------------------
            let (result_scratch, result_mov_op) = spill_scratch_for(result_is_float);
            if let Some(offset) = result_spill_offset {
                let mut ld = make_spill_load(result_scratch, offset, result_mov_op);
                // Propagate is_call_arg_setup so the parallel-move
                // resolver includes this instruction in its scope.
                ld.is_call_arg_setup = inst.is_call_arg_setup;
                new_insts.push(ld);
            }

            // ----------------------------------------------------------
            // Step 3: Resolve explicit operands.
            //
            // If the result is spilled (scratch is occupied):
            //   → Spilled operands of the SAME class become Memory.
            //   → Spilled operands of the OTHER class can use their scratch.
            // If the result is NOT spilled:
            //   → The FIRST spilled int operand loads into R11.
            //   → The FIRST spilled float operand loads into XMM15.
            //   → Additional operands of a class become Memory.
            // ----------------------------------------------------------
            let _int_scratch_reserved_for_result =
                result_spill_offset.is_some() && !result_is_float;
            let _float_scratch_reserved_for_result =
                result_spill_offset.is_some() && result_is_float;
            let mut int_scratch_used = false;
            let mut float_scratch_used = false;

            // Capture the result vreg (before resolution) so we can detect
            // operands that alias the result — critical for in-place ALU ops
            // (ADD, SUB, etc.) where operand[0] IS the result register.
            let result_vreg: Option<u32> = inst.result.as_ref().and_then(|r| {
                if let MachineOperand::VirtualRegister(v) = r {
                    Some(*v)
                } else {
                    None
                }
            });

            // Propagate is_call_arg_setup to spill load instructions so
            // that the parallel-move resolver sees the full arg-setup
            // sequence even when spill reloads are interspersed.
            let propagate_call_arg_setup = inst.is_call_arg_setup;

            for i in 0..inst.operands.len() {
                if let MachineOperand::VirtualRegister(vreg) = inst.operands[i] {
                    if let Some(&offset) = vreg_spill_offset.get(&vreg) {
                        let op_is_float = vreg_is_float.contains(&vreg);
                        if op_is_float {
                            let same_as_result_f = result_vreg == Some(vreg)
                                && result_spill_offset.is_some()
                                && result_is_float;
                            if same_as_result_f && !float_scratch_used {
                                let (scratch, _) = spill_scratch_for(true);
                                inst.operands[i] = MachineOperand::Register(scratch);
                                float_scratch_used = true;
                            } else if !float_scratch_used {
                                let (scratch, mov_op) = spill_scratch_for(true);
                                let mut ld = make_spill_load(scratch, offset, mov_op);
                                ld.is_call_arg_setup = propagate_call_arg_setup;
                                new_insts.push(ld);
                                inst.operands[i] = MachineOperand::Register(scratch);
                                float_scratch_used = true;
                            } else {
                                // XMM15 occupied — use memory operand (SSE supports mem src).
                                inst.operands[i] = MachineOperand::Memory {
                                    base: Some(frame_ptr_reg),
                                    index: None,
                                    scale: 1,
                                    displacement: offset,
                                };
                            }
                        } else {
                            // Check if this operand is the SAME spilled vreg
                            // as the result. This is critical for in-place ALU
                            // ops (ADD, SUB, etc.) where operand[0] IS the
                            // destination: if we used Memory for operand[0]
                            // while result maps to scratch, Step 5's store of
                            // scratch would overwrite the correct in-memory
                            // result with a stale value.
                            let same_as_result = result_vreg == Some(vreg)
                                && result_spill_offset.is_some()
                                && !result_is_float;
                            if same_as_result && !int_scratch_used {
                                // Reuse scratch — already loaded from the same
                                // spill slot in Step 2.
                                inst.operands[i] = MachineOperand::Register(spill_scratch);
                                int_scratch_used = true;
                            } else if !int_scratch_used {
                                // Scratch is available. Load operand's spill
                                // value into scratch (may overwrite Step 2's
                                // preload of the result — that is safe because
                                // both share the scratch register, and the
                                // instruction will produce the correct value
                                // which Step 5 stores back).
                                let mut ld = make_spill_load(spill_scratch, offset, mov_opcode);
                                ld.is_call_arg_setup = propagate_call_arg_setup;
                                new_insts.push(ld);
                                inst.operands[i] = MachineOperand::Register(spill_scratch);
                                int_scratch_used = true;
                            } else {
                                inst.operands[i] = MachineOperand::Memory {
                                    base: Some(frame_ptr_reg),
                                    index: None,
                                    scale: 1,
                                    displacement: offset,
                                };
                            }
                        }
                    } else if let Some(&phys) = vreg_phys.get(&vreg) {
                        inst.operands[i] = MachineOperand::Register(phys);
                    }
                }
            }

            // ----------------------------------------------------------
            // Step 4: Resolve the result operand.
            // ----------------------------------------------------------
            let mut store_after: Option<(i64, bool)> = None;
            if let Some(ref mut result) = inst.result {
                if let MachineOperand::VirtualRegister(vreg) = *result {
                    if let Some(&offset) = vreg_spill_offset.get(&vreg) {
                        let is_float = vreg_is_float.contains(&vreg);
                        let (scratch, _) = spill_scratch_for(is_float);
                        *result = MachineOperand::Register(scratch);
                        store_after = Some((offset, is_float));
                    } else if let Some(&phys) = vreg_phys.get(&vreg) {
                        *result = MachineOperand::Register(phys);
                    }
                }
            }

            // Push the (possibly rewritten) instruction.
            new_insts.push(inst);

            // ----------------------------------------------------------
            // Step 5: Insert a store for the spilled result AFTER the
            //         instruction. Uses store opcode (SD for RISC-V,
            //         STR for AArch64, MOV for x86).
            // ----------------------------------------------------------
            if let Some((offset, is_float)) = store_after {
                let (scratch, st_op) = spill_store_for(is_float);
                new_insts.push(make_spill_store(scratch, offset, st_op));
            }
        }

        block.instructions = new_insts;
    }
}

// ===========================================================================
// resolve_param_load_conflicts — fix register clobbering in function prologue
// ===========================================================================

/// After register allocation, function-entry parameter-loading MOV sequences
/// like:
///
/// ```text
///   MOV RAX, RDI       ; param 0 → allocated to RAX (OK)
///   MOV RCX, RSI       ; param 1 → allocated to RCX (CLOBBERS param 3 source!)
///   MOV RDX, RDX       ; param 2 → allocated to RDX (no-op)
///   MOV RSI, RCX       ; param 3 → allocated to RSI (BUG: RCX already overwritten!)
/// ```
///
/// may contain conflicts: a destination register of an earlier MOV is the
/// source register of a later MOV.  This is the classical **parallel copy
/// problem** — all parameter loads must logically happen simultaneously.
///
/// This function applies a standard parallel-move sequentialisation algorithm:
///
/// 1. Remove self-moves (`MOV R, R`) — they are no-ops.
/// 2. Topologically order remaining moves: emit moves whose destination is
///    NOT a source for any pending move first.
/// 3. If a cycle is detected (no move can be emitted safely), break it by
///    saving one source to R11 (scratch), substituting R11 as the source,
///    and continuing.
///
/// The resolved sequence of MOVs is then spliced back in place of the
/// original parameter-loading sequence.
fn resolve_param_load_conflicts(mf: &mut MachineFunction) {
    if mf.blocks.is_empty() {
        return;
    }

    const MOV_OPCODE: u32 = 0; // X86Opcode::Mov
    const MOVSD_OPCODE: u32 = 86; // X86Opcode::Movsd
    const SCRATCH_REG: u16 = 11; // R11 (integer scratch)
    const FLOAT_SCRATCH_REG: u16 = 31; // XMM15 (float scratch)

    // ABI integer argument registers (System V AMD64).
    let abi_int_regs: crate::common::fx_hash::FxHashSet<u16> = {
        let mut s = crate::common::fx_hash::FxHashSet::default();
        // RDI=7, RSI=6, RDX=2, RCX=1, R8=8, R9=9
        s.insert(7);
        s.insert(6);
        s.insert(2);
        s.insert(1);
        s.insert(8);
        s.insert(9);
        s
    };
    // ABI SSE argument registers.
    let abi_sse_regs: crate::common::fx_hash::FxHashSet<u16> = {
        let mut s = crate::common::fx_hash::FxHashSet::default();
        for i in 16u16..24 {
            // XMM0=16 .. XMM7=23
            s.insert(i);
        }
        s
    };

    let entry = &mut mf.blocks[0];
    let insts = &entry.instructions;

    // Identify the contiguous parameter-loading MOV/MOVSD sequence at the
    // beginning of the entry block.  A parameter MOV is a MOV from a
    // physical ABI register to another physical register.  Float params
    // use MOVSD (opcode 86).
    let mut seq_len: usize = 0;
    for inst in insts.iter() {
        if inst.opcode != MOV_OPCODE && inst.opcode != MOVSD_OPCODE {
            break;
        }
        // Instruction format: result = Some(Register(dst)), operands = [Register(src)]
        let is_param_mov = match (&inst.result, inst.operands.first()) {
            (Some(MachineOperand::Register(dst)), Some(MachineOperand::Register(src))) => {
                let _ = dst; // suppress warning
                abi_int_regs.contains(src) || abi_sse_regs.contains(src)
            }
            _ => false,
        };
        if !is_param_mov {
            break;
        }
        seq_len += 1;
    }

    if seq_len <= 1 {
        // 0 or 1 parameter MOVs — no possible conflicts.
        return;
    }

    // Extract (dst, src) pairs from the parameter MOV sequence.
    // Also detect Movsd (opcode for SSE moves) in addition to regular Mov.
    let mut moves: Vec<(u16, u16, u32)> = Vec::with_capacity(seq_len); // (dst, src, opcode)
    for inst in insts[..seq_len].iter() {
        if let (Some(MachineOperand::Register(dst)), Some(MachineOperand::Register(src))) =
            (&inst.result, inst.operands.first())
        {
            moves.push((*dst, *src, inst.opcode));
        }
    }

    // ---- Parallel-move sequentialisation ----
    //
    // Algorithm:
    //   pending = list of (dst, src, opcode)
    //   result  = list of resolved MOV instructions
    //
    //   Repeat until pending is empty:
    //     1. Remove self-moves (dst == src).
    //     2. Find a move whose dst is NOT a src of any other pending move.
    //        Emit it and remove from pending.
    //     3. If no such move exists (cycle detected):
    //        Save one src to SCRATCH_REG, replace src with SCRATCH_REG in
    //        the move, and continue.

    let mut pending = moves.clone();
    let mut resolved: Vec<MachineInstruction> = Vec::with_capacity(seq_len + 2);

    // Remove self-moves upfront.
    pending.retain(|&(dst, src, _)| dst != src);

    let mut iteration_limit = pending.len() * pending.len() + 10; // safety limit
    while !pending.is_empty() && iteration_limit > 0 {
        iteration_limit -= 1;

        // Collect all source registers still needed by pending moves.
        let needed_srcs: crate::common::fx_hash::FxHashSet<u16> =
            pending.iter().map(|&(_, src, _)| src).collect();

        // Find a move whose destination is NOT a needed source.
        let safe_idx = pending
            .iter()
            .position(|&(dst, _, _)| !needed_srcs.contains(&dst));

        if let Some(idx) = safe_idx {
            let (dst, src, opcode) = pending.remove(idx);
            let mut inst = MachineInstruction::new(opcode);
            inst.result = Some(MachineOperand::Register(dst));
            inst.operands.push(MachineOperand::Register(src));
            resolved.push(inst);
        } else {
            // Cycle detected — break it by saving one source to scratch.
            // Pick the first pending move's source.
            let (_, cycle_src, cycle_opcode) = pending[0];

            // Determine if this is a float (SSE) or integer move.
            let is_float_cycle = cycle_opcode == MOVSD_OPCODE || (16..32).contains(&cycle_src);
            let (scratch, save_op) = if is_float_cycle {
                (FLOAT_SCRATCH_REG, MOVSD_OPCODE)
            } else {
                (SCRATCH_REG, MOV_OPCODE)
            };

            // Emit: MOV/MOVSD scratch <- cycle_src
            let mut save_inst = MachineInstruction::new(save_op);
            save_inst.result = Some(MachineOperand::Register(scratch));
            save_inst.operands.push(MachineOperand::Register(cycle_src));
            resolved.push(save_inst);

            // Replace all references to cycle_src with scratch.
            for m in &mut pending {
                if m.1 == cycle_src {
                    m.1 = scratch;
                }
            }
        }
    }

    // Replace the original sequence in the entry block.
    if !resolved.is_empty() || seq_len > 0 {
        let remaining: Vec<MachineInstruction> = entry.instructions.drain(seq_len..).collect();
        entry.instructions.clear();
        entry.instructions.extend(resolved);
        entry.instructions.extend(remaining);
    }
}

// ===========================================================================
// resolve_div_conflicts — fix divisor-register clobbering in IDIV/DIV
// ===========================================================================

/// After register allocation, integer division sequences may have the divisor
/// assigned to RAX or RDX — registers that are implicitly overwritten by the
/// dividend setup (`MOV RAX, lhs; CQO/CDQ`) or zero-extension (`XOR RDX, RDX`).
///
/// For signed division (SDiv/SRem), the codegen emits:
/// ```text
///   MOV RAX, <lhs>       ← writes RAX  (clobbers divisor if in RAX)
///   CQO / CDQ            ← writes RDX  (clobbers divisor if in RDX)
///   IDIV <divisor>
/// ```
///
/// For unsigned division (UDiv/URem), the codegen emits:
/// ```text
///   XOR RDX, RDX         ← writes RDX  (clobbers LHS/divisor if in RDX)
///   MOV RAX, <lhs>       ← writes RAX  (clobbers divisor if in RAX)
///   DIV <divisor>
/// ```
///
/// This post-allocation pass detects these conflicts and saves the clobbered
/// operand to a scratch register before it is overwritten.
fn resolve_div_conflicts(mf: &mut MachineFunction) {
    const MOV_OP: u32 = 0; // X86Opcode::Mov
    const IDIV_OP: u32 = 10; // X86Opcode::Idiv
    const DIV_OP: u32 = 11; // X86Opcode::Div
    const XOR_OP: u32 = 17; // X86Opcode::Xor
    const CDQ_OP: u32 = 106; // X86Opcode::Cdq
    const CQO_OP: u32 = 107; // X86Opcode::Cqo
    const RAX: u16 = 0;
    const RCX: u16 = 1;
    const RDX: u16 = 2;
    const R11: u16 = 11;

    for block in &mut mf.blocks {
        // We may insert instructions, so iterate by building a new vector.
        let mut new_insts: Vec<MachineInstruction> =
            Vec::with_capacity(block.instructions.len() + 8);
        let old_insts = std::mem::take(&mut block.instructions);
        let len = old_insts.len();

        let mut i = 0;
        while i < len {
            let opc = old_insts[i].opcode;

            if opc != IDIV_OP && opc != DIV_OP {
                new_insts.push(old_insts[i].clone());
                i += 1;
                continue;
            }

            // Found IDIV or DIV at position `i`.
            // Divisor is in operands[0].
            let divisor_reg = match old_insts[i].operands.first() {
                Some(MachineOperand::Register(r)) => *r,
                _ => {
                    // Not a register operand (memory, etc.) — no conflict.
                    new_insts.push(old_insts[i].clone());
                    i += 1;
                    continue;
                }
            };

            // Only a problem if divisor is in RAX or RDX.
            if divisor_reg != RAX && divisor_reg != RDX {
                new_insts.push(old_insts[i].clone());
                i += 1;
                continue;
            }

            // Scan backwards in new_insts to find the "MOV RAX, <lhs>" and
            // optionally CQO/CDQ (for signed) or XOR RDX, RDX (for unsigned).
            // We need to identify the start of the div setup sequence and find
            // the lhs register to avoid scratch conflicts.

            // The division setup sequence (already in new_insts) looks like:
            //   [... MOV RAX, <lhs> ; CQO/CDQ]     for signed  (IDIV)
            //   [... XOR RDX, RDX ; MOV RAX, <lhs>] for unsigned (DIV)
            //
            // We'll look at the last 2-3 instructions in new_insts.

            let ni_len = new_insts.len();

            // Find the MOV RAX, <lhs> instruction (the one that writes RAX).
            let mut mov_rax_pos = None;
            let mut lhs_reg: Option<u16> = None;
            // Search backwards up to 3 instructions.
            let search_start = ni_len.saturating_sub(3);
            for j in (search_start..ni_len).rev() {
                if new_insts[j].opcode == MOV_OP {
                    if let Some(MachineOperand::Register(r)) = &new_insts[j].result {
                        if *r == RAX {
                            mov_rax_pos = Some(j);
                            // Get the source register.
                            if let Some(MachineOperand::Register(src)) =
                                new_insts[j].operands.first()
                            {
                                lhs_reg = Some(*src);
                            }
                            break;
                        }
                    }
                }
            }

            // Also check for XOR RDX, RDX or CQO/CDQ in the sequence.
            let mut has_rdx_clobber = false;
            for j in (search_start..ni_len).rev() {
                let op = new_insts[j].opcode;
                if op == CQO_OP || op == CDQ_OP {
                    has_rdx_clobber = true;
                    break;
                }
                if op == XOR_OP {
                    if let Some(MachineOperand::Register(r)) = &new_insts[j].result {
                        if *r == RDX {
                            has_rdx_clobber = true;
                            break;
                        }
                    }
                }
            }

            // Determine conflicts:
            // - Divisor in RAX: clobbered by "MOV RAX, lhs"
            // - Divisor in RDX: clobbered by CQO/CDQ or XOR RDX, RDX
            // Also check if LHS might be clobbered (for unsigned: XOR RDX, RDX before MOV RAX):
            //   If lhs was in RDX and unsigned DIV: XOR RDX,RDX clobbers lhs before MOV RAX reads it.

            let div_in_rax = divisor_reg == RAX;
            let div_in_rdx = divisor_reg == RDX && has_rdx_clobber;

            if !div_in_rax && !div_in_rdx {
                // No conflict after all.
                new_insts.push(old_insts[i].clone());
                i += 1;
                continue;
            }

            // Choose a scratch register: not RAX, not RDX, not lhs_reg.
            let scratch = if lhs_reg != Some(R11) { R11 } else { RCX };

            // Also handle lhs-in-RDX conflict for unsigned division:
            // If opc == DIV_OP and lhs_reg == Some(RDX), the XOR RDX,RDX clobbers lhs.
            let lhs_in_rdx = opc == DIV_OP && lhs_reg == Some(RDX);

            if let Some(insert_pos) = mov_rax_pos {
                // For unsigned DIV with lhs in RDX:
                //   Before: XOR RDX,RDX ; MOV RAX,RDX ; DIV <rhs>
                //   We need to save RDX before the XOR too.
                if lhs_in_rdx {
                    // Find XOR RDX,RDX before insert_pos.
                    let mut xor_pos = None;
                    for j in (search_start..insert_pos).rev() {
                        if new_insts[j].opcode == XOR_OP {
                            if let Some(MachineOperand::Register(r)) = &new_insts[j].result {
                                if *r == RDX {
                                    xor_pos = Some(j);
                                    break;
                                }
                            }
                        }
                    }
                    if let Some(xp) = xor_pos {
                        // Pick a second scratch for lhs (different from scratch if
                        // scratch is also needed for divisor).
                        let lhs_scratch = if div_in_rax || div_in_rdx {
                            // Need scratch for divisor, use another for lhs.
                            if scratch != RCX && lhs_reg != Some(RCX) {
                                RCX
                            } else if scratch != R11 && lhs_reg != Some(R11) {
                                R11
                            } else {
                                10 /* R10 */
                            }
                        } else {
                            scratch
                        };
                        // Insert: MOV lhs_scratch, RDX before XOR RDX,RDX
                        let mut save_lhs = MachineInstruction::new(MOV_OP);
                        save_lhs.result = Some(MachineOperand::Register(lhs_scratch));
                        save_lhs.operands.push(MachineOperand::Register(RDX));
                        new_insts.insert(xp, save_lhs);
                        // Update references after insertion.
                        // Find new mov_rax_pos (shifted by 1).
                        let new_mov_rax = insert_pos + 1;
                        // Change MOV RAX, RDX to MOV RAX, lhs_scratch.
                        if let Some(MachineOperand::Register(ref mut src)) =
                            new_insts[new_mov_rax].operands.first_mut()
                        {
                            if *src == RDX {
                                *src = lhs_scratch;
                            }
                        }
                    }
                }

                if div_in_rax || div_in_rdx {
                    // Re-find mov_rax_pos after potential insertions above.
                    let cur_ni_len = new_insts.len();
                    let search_start2 = cur_ni_len.saturating_sub(5);
                    let mut real_insert_pos = None;
                    for j in (search_start2..cur_ni_len).rev() {
                        if new_insts[j].opcode == MOV_OP {
                            if let Some(MachineOperand::Register(r)) = &new_insts[j].result {
                                if *r == RAX {
                                    real_insert_pos = Some(j);
                                    break;
                                }
                            }
                        }
                    }
                    // For unsigned DIV with XOR RDX,RDX, if divisor is in RDX,
                    // insert BEFORE the XOR.
                    if div_in_rdx && opc == DIV_OP {
                        let cur_ni_len2 = new_insts.len();
                        let ss = cur_ni_len2.saturating_sub(5);
                        for j in (ss..cur_ni_len2).rev() {
                            if new_insts[j].opcode == XOR_OP {
                                if let Some(MachineOperand::Register(r)) = &new_insts[j].result {
                                    if *r == RDX {
                                        real_insert_pos = Some(j);
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    // For signed IDIV with CQO/CDQ, if divisor is in RDX,
                    // insert BEFORE the MOV RAX (which precedes CQO).
                    if div_in_rdx && opc == IDIV_OP {
                        // real_insert_pos is already the MOV RAX position.
                    }

                    if let Some(rip) = real_insert_pos {
                        // Insert: MOV scratch, divisor_reg BEFORE the clobbering instruction
                        let mut save_div = MachineInstruction::new(MOV_OP);
                        save_div.result = Some(MachineOperand::Register(scratch));
                        save_div
                            .operands
                            .push(MachineOperand::Register(divisor_reg));
                        new_insts.insert(rip, save_div);
                    }

                    // Now update the IDIV/DIV instruction to use scratch instead
                    // of the original divisor register.
                    // The IDIV/DIV hasn't been pushed yet.
                    let mut div_inst = old_insts[i].clone();
                    div_inst.operands[0] = MachineOperand::Register(scratch);
                    new_insts.push(div_inst);
                    i += 1;
                    continue;
                }
            }

            // Fallback: push the original instruction unchanged.
            new_insts.push(old_insts[i].clone());
            i += 1;
        }

        block.instructions = new_insts;
    }
}

// ===========================================================================
// resolve_call_arg_conflicts — fix register clobbering in call setup
// ===========================================================================

/// After register allocation, call-argument setup sequences like:
///
/// ```text
///   MOV RDI, R9       ; arg 0 ← format string
///   MOV RSI, RAX      ; arg 1 ← v1
///   MOV RDX, RSI      ; arg 2 ← v2   (BUG: RSI already overwritten by line 2!)
///   MOV RCX, RDI      ; arg 3 ← v3   (BUG: RDI already overwritten by line 1!)
///   CALL printf
/// ```
///
/// may contain conflicts: a destination register of an earlier MOV is the
/// source register of a later MOV.  This is the classical **parallel copy
/// problem** — all argument setup moves must logically happen simultaneously.
///
/// This function applies a proper parallel-move sequentialisation algorithm:
///
/// 1. Walk backwards from every CALL to find the contiguous
///    `is_call_arg_setup` instruction sequence.
/// 2. Extract register-to-register (dst, src) pairs.  Non-register-source
///    instructions (immediates, LEAs, memory operands) are emitted first
///    since they have no read-after-write conflicts.
/// 3. Remove self-moves (dst == src).
/// 4. Topologically order remaining moves: emit moves whose destination
///    is NOT a source for any pending move first.
/// 5. If a cycle is detected, break it by saving one source to R11
///    (scratch), substituting R11 as the source, and continuing.
fn resolve_call_arg_conflicts(mf: &mut MachineFunction, target: &Target) {
    // Architecture-specific constants for register move opcodes and scratch
    // registers.  Each target uses a different MOV instruction and a
    // different caller-saved scratch register that is NOT part of the ABI
    // argument register set.
    let (mov_opcode, movfp_opcode, scratch_reg, float_scratch_reg): (u32, u32, u16, u16) =
        match target {
            Target::X86_64 => (
                0,  // X86Opcode::Mov
                86, // X86Opcode::Movsd
                11, // R11 (caller-saved, not an arg reg)
                31, // XMM15
            ),
            Target::I686 => (
                0,  // X86Opcode::Mov (same enum)
                86, // X86Opcode::Movsd
                2,  // ECX (caller-saved scratch for i686)
                31, // XMM7
            ),
            Target::AArch64 => (
                161, // A64Opcode::MOV_reg
                149, // A64Opcode::FMOV_d
                16,  // X16 (IP0 — intra-procedure-call scratch)
                48,  // V16 (caller-saved FP scratch)
            ),
            Target::RiscV64 => (
                21,  // RvOpcode::ADDI (MV pseudo = ADDI rd, rs, 0)
                142, // RvOpcode::FSGNJ_D (FMV.D pseudo = FSGNJ.D rd, rs, rs)
                5,   // T0 (x5 — temporary, not an arg reg)
                32,  // FT0 (f0 — temporary, not an FP arg reg)
            ),
        };
    let mov_opcode_val = mov_opcode;
    let movfp_opcode_val = movfp_opcode;

    for block in &mut mf.blocks {
        let mut new_insts: Vec<MachineInstruction> =
            Vec::with_capacity(block.instructions.len() + 16);
        let insts = std::mem::take(&mut block.instructions);

        for inst in &insts {
            new_insts.push(inst.clone());

            if !inst.is_call {
                continue;
            }

            // The CALL is at the end of new_insts.  Walk backwards to
            // find the contiguous call-argument-setup sequence.
            let call_pos = new_insts.len() - 1;
            let mut seq_start = call_pos;
            while seq_start > 0 {
                let prev = seq_start - 1;
                if !new_insts[prev].is_call_arg_setup {
                    break;
                }
                seq_start = prev;
            }

            if seq_start == call_pos {
                continue; // No argument-setup sequence.
            }

            // ---- Indirect call target protection ----
            // For indirect calls (CALL *reg), the target register may be
            // clobbered by the argument-setup sequence (e.g., MOV RAX,0
            // for AL=0 clobbers RAX if the function pointer is in RAX).
            // We detect this and will insert a save to R11 at the start
            // of the resolved move sequence.
            let indirect_save_needed: Option<u16> = {
                let call_inst_ref = &new_insts[call_pos];
                if let Some(MachineOperand::Register(target_reg)) = call_inst_ref.operands.first() {
                    let tr = *target_reg;
                    if tr != scratch_reg {
                        let clobbered = (seq_start..call_pos).any(|idx| {
                            matches!(&new_insts[idx].result,
                                Some(MachineOperand::Register(dst)) if *dst == tr)
                        });
                        if clobbered {
                            Some(tr)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            };
            // If save needed, update CALL to use R11 now.
            if indirect_save_needed.is_some() {
                new_insts[call_pos].operands[0] = MachineOperand::Register(scratch_reg);
            }

            // Partition instructions into:
            // - `non_reg_moves`: instructions whose source is NOT a plain
            //   register (immediates, LEAs, etc.) — safe to emit in any
            //   order since they don't read from registers that might be
            //   clobbered.
            // - `reg_moves`: register-to-register moves that need
            //   parallel-copy resolution.
            let setup_insts: Vec<MachineInstruction> =
                new_insts.drain(seq_start..call_pos).collect();
            // call_pos shifted; CALL is now at seq_start.

            let mut reg_moves: Vec<(u16, u16, u32)> = Vec::new(); // (dst, src, opcode)
            let mut non_reg_move_insts: Vec<MachineInstruction> = Vec::new();
            // Compound pairs: (Vec<MachineInstruction>, effective_dst).
            // A load-then-move through scratch (e.g., LD t0,-X(s0) then
            // MV a3,t0) MUST stay together — separating them causes the
            // scratch to be clobbered between the load and the move.
            let mut compound_moves: Vec<(Vec<MachineInstruction>, u16)> = Vec::new();
            // Track indices we've already consumed as part of a compound.
            let mut consumed: crate::common::fx_hash::FxHashSet<usize> =
                crate::common::fx_hash::FxHashSet::default();

            for (idx, si) in setup_insts.iter().enumerate() {
                if consumed.contains(&idx) {
                    continue;
                }
                let dst_reg = match &si.result {
                    Some(MachineOperand::Register(r)) => Some(*r),
                    _ => None,
                };
                let src_reg = match si.operands.first() {
                    Some(MachineOperand::Register(r)) => Some(*r),
                    _ => None,
                };

                if let (Some(d), Some(s)) = (dst_reg, src_reg) {
                    // Register-to-register move — needs parallel resolution.
                    reg_moves.push((d, s, si.opcode));
                } else {
                    // Non-register source (immediate, global symbol, LEA,
                    // memory operand, spill load).
                    //
                    // Check if this instruction loads into a scratch
                    // register that the NEXT instruction immediately moves
                    // to an arg register.  If so, they form a compound
                    // pair that must be emitted atomically to prevent the
                    // scratch register from being clobbered in between.
                    let has_mem_src = si
                        .operands
                        .iter()
                        .any(|op| matches!(op, MachineOperand::Memory { .. }));
                    if has_mem_src {
                        if let Some(d_scratch) = dst_reg {
                            // Look at next instruction(s) for a MV from d_scratch
                            if let Some(next_si) = setup_insts.get(idx + 1) {
                                let next_dst = match &next_si.result {
                                    Some(MachineOperand::Register(r)) => Some(*r),
                                    _ => None,
                                };
                                let next_src = match next_si.operands.first() {
                                    Some(MachineOperand::Register(r)) => Some(*r),
                                    _ => None,
                                };
                                if next_src == Some(d_scratch) {
                                    if let Some(eff_dst) = next_dst {
                                        // Compound pair found: load-then-move
                                        compound_moves
                                            .push((vec![si.clone(), next_si.clone()], eff_dst));
                                        consumed.insert(idx + 1);
                                        continue;
                                    }
                                }
                            }
                        }
                    }
                    non_reg_move_insts.push(si.clone());
                }
            }

            // Check if there are actually any register conflicts.
            let has_conflict = {
                let dsts: crate::common::fx_hash::FxHashSet<u16> =
                    reg_moves.iter().map(|m| m.0).collect();
                reg_moves.iter().any(|m| dsts.contains(&m.1))
            };

            if !has_conflict && indirect_save_needed.is_none() {
                // No conflicts and no indirect save — put back in original order.
                let call_inst = new_insts.remove(seq_start);
                for si in setup_insts {
                    new_insts.push(si);
                }
                new_insts.push(call_inst);
                continue;
            }
            if let (false, Some(target_reg)) = (has_conflict, indirect_save_needed) {
                // No register-register conflicts, but we need to protect
                // the indirect call target. Insert save before setup.
                let call_inst = new_insts.remove(seq_start);
                let mut save_inst = MachineInstruction::new(mov_opcode_val);
                save_inst.result = Some(MachineOperand::Register(scratch_reg));
                save_inst
                    .operands
                    .push(MachineOperand::Register(target_reg));
                save_inst.is_call_arg_setup = true;
                new_insts.push(save_inst);
                for si in setup_insts {
                    new_insts.push(si);
                }
                new_insts.push(call_inst);
                continue;
            }

            // ---- Parallel-move sequentialisation for reg-to-reg moves ----
            //
            // Remove self-moves, then topologically order: emit moves whose
            // destination is not a source for any pending move.  If a cycle
            // is found, break it using a scratch register.
            let mut pending = reg_moves;
            pending.retain(|&(d, s, _)| d != s);

            let mut resolved: Vec<MachineInstruction> = Vec::new();

            // If we need to protect an indirect call target register,
            // insert MOV R11, target_reg as the VERY FIRST instruction
            // before any arg-setup moves can clobber the target.
            if let Some(target_reg) = indirect_save_needed {
                let mut save_inst = MachineInstruction::new(mov_opcode_val);
                save_inst.result = Some(MachineOperand::Register(scratch_reg));
                save_inst
                    .operands
                    .push(MachineOperand::Register(target_reg));
                save_inst.is_call_arg_setup = true;
                resolved.push(save_inst);

                // Also update any reg-to-reg moves that read from
                // target_reg to read from R11 instead, since we saved it.
                // (Not strictly necessary since the save happens first,
                // but ensures consistency if target_reg is also used as
                // a source for arg-setup moves.)
            }

            // Emit non-register-source moves first (no conflict risk on
            // reads). BUT: if one of these writes to a register that is a
            // source for a pending reg-to-reg move, we must defer it.
            let pending_srcs: crate::common::fx_hash::FxHashSet<u16> =
                pending.iter().map(|m| m.1).collect();
            let mut deferred_non_reg: Vec<MachineInstruction> = Vec::new();
            // Deferred compound moves: (instructions, effective_dst).
            let mut deferred_compounds: Vec<(Vec<MachineInstruction>, u16)> = Vec::new();
            for nrm in non_reg_move_insts {
                let dst = match &nrm.result {
                    Some(MachineOperand::Register(r)) => Some(*r),
                    _ => None,
                };
                if let Some(d) = dst {
                    if pending_srcs.contains(&d) {
                        // This non-reg move writes to a register that is
                        // a source for a pending reg-to-reg move. Defer it.
                        deferred_non_reg.push(nrm);
                        continue;
                    }
                }
                resolved.push(nrm);
            }
            // Defer compound moves whose effective_dst is a source for a
            // pending reg-to-reg move; otherwise emit them now.
            for cm in compound_moves {
                if pending_srcs.contains(&cm.1) {
                    deferred_compounds.push(cm);
                } else {
                    resolved.extend(cm.0);
                }
            }

            let mut iteration_limit = pending.len() * pending.len() + 10;
            while !pending.is_empty() && iteration_limit > 0 {
                iteration_limit -= 1;

                let needed_srcs: crate::common::fx_hash::FxHashSet<u16> =
                    pending.iter().map(|m| m.1).collect();

                // Find a safe move (dst not needed as source by any pending).
                let safe_idx = pending
                    .iter()
                    .position(|&(d, _, _)| !needed_srcs.contains(&d));

                if let Some(idx) = safe_idx {
                    let (dst, src, opcode) = pending.remove(idx);
                    let mut mi = MachineInstruction::new(opcode);
                    mi.result = Some(MachineOperand::Register(dst));
                    mi.operands.push(MachineOperand::Register(src));
                    // For RISC-V FSGNJ.D: duplicate rs1→rs2 for FMV.D pseudo
                    if opcode == movfp_opcode_val && matches!(target, Target::RiscV64) {
                        mi.operands.push(MachineOperand::Register(src));
                    }
                    mi.is_call_arg_setup = true;
                    resolved.push(mi);

                    // Check if any deferred non-reg moves or compound pairs
                    // can now be emitted.
                    let new_pending_srcs: crate::common::fx_hash::FxHashSet<u16> =
                        pending.iter().map(|m| m.1).collect();
                    let mut still_deferred = Vec::new();
                    for nrm in deferred_non_reg {
                        let d = match &nrm.result {
                            Some(MachineOperand::Register(r)) => Some(*r),
                            _ => None,
                        };
                        if d.map_or(false, |dd| new_pending_srcs.contains(&dd)) {
                            still_deferred.push(nrm);
                        } else {
                            resolved.push(nrm);
                        }
                    }
                    deferred_non_reg = still_deferred;
                    // Also check deferred compounds.
                    let mut still_deferred_c = Vec::new();
                    for cm in deferred_compounds {
                        if new_pending_srcs.contains(&cm.1) {
                            still_deferred_c.push(cm);
                        } else {
                            resolved.extend(cm.0);
                        }
                    }
                    deferred_compounds = still_deferred_c;
                } else {
                    // Cycle detected — break it with a scratch register.
                    let (_, cycle_src, cycle_opcode) = pending[0];
                    // Detect FP register ranges per architecture:
                    // x86-64/i686: XMM regs 16..31, RISC-V/AArch64: FP regs >= 32
                    let is_float = cycle_opcode == movfp_opcode_val
                        || match target {
                            Target::X86_64 | Target::I686 => (16..32).contains(&cycle_src),
                            Target::AArch64 => cycle_src >= 32,
                            Target::RiscV64 => cycle_src >= 32,
                        };
                    let (scratch, save_op) = if is_float {
                        (float_scratch_reg, movfp_opcode_val)
                    } else {
                        (scratch_reg, mov_opcode_val)
                    };

                    let mut save_inst = MachineInstruction::new(save_op);
                    save_inst.result = Some(MachineOperand::Register(scratch));
                    save_inst.operands.push(MachineOperand::Register(cycle_src));
                    // For RISC-V FSGNJ.D and AArch64 FMOV: the FP move
                    // pseudo requires rs2 == rs1 (source duplicated).
                    // Add second register operand for R-type FP moves.
                    if is_float && matches!(target, Target::RiscV64) {
                        save_inst.operands.push(MachineOperand::Register(cycle_src));
                    }
                    save_inst.is_call_arg_setup = true;
                    resolved.push(save_inst);

                    for m in &mut pending {
                        if m.1 == cycle_src {
                            m.1 = scratch;
                        }
                    }
                }
            }

            // Emit any remaining deferred non-reg moves and compounds.
            resolved.extend(deferred_non_reg);
            for cm in deferred_compounds {
                resolved.extend(cm.0);
            }

            // Splice resolved moves back before the CALL instruction.
            let call_inst = new_insts.remove(seq_start);
            new_insts.extend(resolved);
            new_insts.push(call_inst);
        }

        block.instructions = new_insts;
    }
}

// ===========================================================================
// insert_prologue_epilogue — inject frame setup/teardown instructions
// ===========================================================================

/// Inserts prologue instructions at the beginning of the entry block and
/// epilogue instructions before every return instruction.
fn insert_prologue_epilogue(
    mf: &mut MachineFunction,
    prologue: Vec<MachineInstruction>,
    epilogue: Vec<MachineInstruction>,
) {
    // Insert prologue at the start of the entry block.
    if !prologue.is_empty() && !mf.blocks.is_empty() {
        let entry = &mut mf.blocks[0];
        let mut new_instructions = Vec::with_capacity(prologue.len() + entry.instructions.len());
        new_instructions.extend(prologue);
        new_instructions.append(&mut entry.instructions);
        entry.instructions = new_instructions;
    }

    // Insert epilogue before every return instruction.  Some backends
    // (notably AArch64) place all instructions in a single block, so
    // multiple `ret`s may exist within one block.  We must insert the
    // epilogue before EVERY return — not just the last instruction.
    if !epilogue.is_empty() {
        for block in &mut mf.blocks {
            // Collect indices of all return instructions in this block.
            let ret_indices: Vec<usize> = block
                .instructions
                .iter()
                .enumerate()
                .filter(|(_, mi)| mi.is_terminator && !mi.is_branch)
                .map(|(i, _)| i)
                .collect();

            if ret_indices.is_empty() {
                continue;
            }

            // Rebuild the instruction list, inserting epilogue before each
            // return.  Process in forward order; the growing offset is
            // accounted for by building a fresh vector.
            let epi_len = epilogue.len();
            let mut new_insts =
                Vec::with_capacity(block.instructions.len() + ret_indices.len() * epi_len);
            let mut prev = 0usize;
            for &ri in &ret_indices {
                // Copy instructions up to (but not including) the return.
                new_insts.extend_from_slice(&block.instructions[prev..ri]);
                // Insert the epilogue sequence.
                new_insts.extend(epilogue.clone());
                // Copy the return instruction itself.
                new_insts.push(block.instructions[ri].clone());
                prev = ri + 1;
            }
            // Copy any remaining instructions after the last return.
            if prev < block.instructions.len() {
                new_insts.extend_from_slice(&block.instructions[prev..]);
            }
            block.instructions = new_insts;
        }
    }
}

// ===========================================================================
// emit_global_variable — encode a global variable into the appropriate section
// ===========================================================================

/// Emits a global variable into the appropriate ELF section (.data, .rodata,
/// or .bss) based on its attributes.
fn emit_global_variable(
    global: &crate::ir::module::GlobalVariable,
    data_section: &mut Vec<u8>,
    rodata_section: &mut Vec<u8>,
    bss_size: &mut u64,
    _ctx: &CodegenContext,
) -> GlobalSymbolInfo {
    use crate::ir::module::Constant;

    // Determine placement based on global attributes.
    let (section, offset, size) = match (&global.initializer, global.is_constant) {
        // Constant with initializer → .rodata
        (Some(init), true) => {
            let offset = rodata_section.len() as u64;
            let bytes = constant_to_bytes_typed(init, Some(&global.ty));
            let size = bytes.len() as u64;
            rodata_section.extend_from_slice(&bytes);
            (SectionPlacement::Rodata, offset, size)
        }
        // ZeroInit → .bss
        (Some(Constant::ZeroInit), false) | (None, _) => {
            let size = global_variable_size(global);
            let offset = *bss_size;
            *bss_size += size;
            (SectionPlacement::Bss, offset, size)
        }
        // Initialized non-constant → .data
        (Some(init), false) => {
            let offset = data_section.len() as u64;
            let bytes = constant_to_bytes_typed(init, Some(&global.ty));
            let size = bytes.len() as u64;
            data_section.extend_from_slice(&bytes);
            (SectionPlacement::Data, offset, size)
        }
    };

    let is_global = global.linkage == crate::ir::module::Linkage::External
        || global.linkage == crate::ir::module::Linkage::Weak;

    GlobalSymbolInfo {
        name: global.name.clone(),
        section,
        offset,
        size,
        is_global,
    }
}

/// Converts a constant initializer to raw bytes for section embedding.
///
/// When a type is provided, integers are truncated to the correct byte
/// width. Without a type, integers fall back to 8 bytes (i64 width).
#[allow(dead_code)]
fn constant_to_bytes(constant: &crate::ir::module::Constant) -> Vec<u8> {
    constant_to_bytes_typed(constant, None)
}

/// Type-aware version that uses the IR type to determine integer widths.
fn constant_to_bytes_typed(
    constant: &crate::ir::module::Constant,
    ir_ty: Option<&crate::ir::types::IrType>,
) -> Vec<u8> {
    use crate::ir::module::Constant;
    use crate::ir::types::IrType;

    match constant {
        Constant::Integer(v) => {
            // Determine byte width from the IR type when available.
            let width = match ir_ty {
                Some(IrType::I1) | Some(IrType::I8) => 1,
                Some(IrType::I16) => 2,
                Some(IrType::I32) => 4,
                Some(IrType::I64) | Some(IrType::Ptr) => 8,
                Some(IrType::I128) => 16,
                _ => 8, // Default to 8 bytes for unknown or unspecified types.
            };
            let full = v.to_le_bytes();
            full[..width].to_vec()
        }
        Constant::Float(v) => match ir_ty {
            Some(IrType::F32) => (*v as f32).to_le_bytes().to_vec(),
            _ => v.to_le_bytes().to_vec(),
        },
        Constant::LongDouble(bytes) => {
            // 80-bit extended precision — 10 raw bytes, padded to 16.
            let mut result = bytes.to_vec();
            result.resize(16, 0);
            result
        }
        Constant::String(bytes) => bytes.clone(),
        Constant::ZeroInit => {
            // When we know the type, emit the correct number of zero bytes.
            if let Some(ty) = ir_ty {
                let sz = ir_type_size(ty) as usize;
                vec![0u8; sz]
            } else {
                Vec::new()
            }
        }
        Constant::Struct(fields) => {
            let mut result = Vec::new();
            let field_types = match ir_ty {
                Some(IrType::Struct(st)) => Some(&st.fields),
                _ => None,
            };
            for (i, field) in fields.iter().enumerate() {
                let ft = field_types.and_then(|fts| fts.get(i));
                result.extend_from_slice(&constant_to_bytes_typed(field, ft));
            }
            result
        }
        Constant::Array(elements) => {
            let mut result = Vec::new();
            let elem_ty = match ir_ty {
                Some(IrType::Array(et, _)) => Some(et.as_ref()),
                _ => None,
            };
            for elem in elements {
                result.extend_from_slice(&constant_to_bytes_typed(elem, elem_ty));
            }
            result
        }
        Constant::GlobalRef(_name) => {
            // Global references are resolved by the linker via relocations.
            // Emit a placeholder (zero bytes) — the linker patches this.
            let ptr_size = match ir_ty {
                Some(IrType::Ptr) => 8,
                _ => 8,
            };
            vec![0u8; ptr_size]
        }
        Constant::Null => vec![0u8; 8],
        Constant::Undefined => vec![0u8; 8],
    }
}

/// Computes the storage size for a global variable based on its IR type.
fn global_variable_size(global: &crate::ir::module::GlobalVariable) -> u64 {
    // Use the IR type's size information.
    // For BSS globals without initializers, we need the type size.
    // Default to 8 bytes (pointer size on 64-bit targets).
    match &global.ty {
        crate::ir::types::IrType::I1 => 1,
        crate::ir::types::IrType::I8 => 1,
        crate::ir::types::IrType::I16 => 2,
        crate::ir::types::IrType::I32 => 4,
        crate::ir::types::IrType::I64 | crate::ir::types::IrType::Ptr => 8,
        crate::ir::types::IrType::I128 => 16,
        crate::ir::types::IrType::F32 => 4,
        crate::ir::types::IrType::F64 => 8,
        crate::ir::types::IrType::F80 => 16,
        crate::ir::types::IrType::Void => 0,
        crate::ir::types::IrType::Array(elem_ty, count) => {
            let elem_size = ir_type_size(elem_ty);
            elem_size * (*count as u64)
        }
        crate::ir::types::IrType::Struct(st) => st.fields.iter().map(ir_type_size).sum(),
        crate::ir::types::IrType::Function(_, _) => 0,
    }
}

/// Estimates the size of an IR type in bytes.
fn ir_type_size(ty: &crate::ir::types::IrType) -> u64 {
    match ty {
        crate::ir::types::IrType::Void => 0,
        crate::ir::types::IrType::I1 | crate::ir::types::IrType::I8 => 1,
        crate::ir::types::IrType::I16 => 2,
        crate::ir::types::IrType::I32 | crate::ir::types::IrType::F32 => 4,
        crate::ir::types::IrType::I64
        | crate::ir::types::IrType::F64
        | crate::ir::types::IrType::Ptr => 8,
        crate::ir::types::IrType::I128 | crate::ir::types::IrType::F80 => 16,
        crate::ir::types::IrType::Array(elem, count) => ir_type_size(elem) * (*count as u64),
        crate::ir::types::IrType::Struct(st) => st.fields.iter().map(ir_type_size).sum(),
        crate::ir::types::IrType::Function(_, _) => 0,
    }
}

// ===========================================================================
// write_relocatable_object — produce a .o ELF file
// ===========================================================================

/// Patch a single intra-module relocation in the combined `.text` section.
///
/// For x86-64 and i686, most PC-relative relocations use a raw 4-byte
/// little-endian addend written directly at the patch site.  For AArch64
/// and RISC-V 64, the offset is encoded into instruction-specific immediate
/// fields while preserving opcode bits.
fn patch_intra_module_reloc(
    text: &mut [u8],
    off: usize,
    value: i64, // S + A - P (byte offset)
    rel_type: u32,
    target: &Target,
) {
    match target {
        Target::X86_64 | Target::I686 => {
            // x86 family: raw 4-byte PC-relative offset
            let bytes = (value as i32).to_le_bytes();
            text[off..off + 4].copy_from_slice(&bytes);
        }
        Target::AArch64 => {
            // Read existing instruction word (preserves opcode)
            let inst = u32::from_le_bytes([text[off], text[off + 1], text[off + 2], text[off + 3]]);
            // AArch64 CALL26 (283), JUMP26 (282): imm26 in bits [25:0], offset/4
            const A64_CALL26: u32 = 283;
            const A64_JUMP26: u32 = 282;
            // AArch64 CONDBR19 (280): imm19 in bits [23:5], offset/4
            const A64_CONDBR19: u32 = 280;
            // AArch64 TSTBR14 (279): imm14 in bits [18:5], offset/4
            const A64_TSTBR14: u32 = 279;
            // AArch64 ADR_PREL_PG_HI21 (275): ADRP page offset
            const A64_ADR_PREL_PG_HI21: u32 = 275;
            const A64_ADR_PREL_PG_HI21_NC: u32 = 276;
            // AArch64 ADD_ABS_LO12_NC (277): ADD lower 12 bits
            const A64_ADD_ABS_LO12_NC: u32 = 277;
            // AArch64 LD64_GOT_LO12_NC (312): LDR lower 12 bits for GOT
            const A64_LDST64_ABS_LO12_NC: u32 = 286;

            let patched = match rel_type {
                A64_CALL26 | A64_JUMP26 => {
                    let imm26 = ((value >> 2) as u32) & 0x03FF_FFFF;
                    (inst & 0xFC00_0000) | imm26
                }
                A64_CONDBR19 => {
                    let imm19 = ((value >> 2) as u32) & 0x7_FFFF;
                    (inst & 0xFF00_001F) | (imm19 << 5)
                }
                A64_TSTBR14 => {
                    let imm14 = ((value >> 2) as u32) & 0x3FFF;
                    (inst & 0xFFF8_001F) | (imm14 << 5)
                }
                A64_ADR_PREL_PG_HI21 | A64_ADR_PREL_PG_HI21_NC => {
                    // ADRP: page-relative offset >> 12, encode immhi:immlo
                    let page_off = value >> 12;
                    let immlo = ((page_off as u32) & 0x3) << 29;
                    let immhi = (((page_off >> 2) as u32) & 0x7_FFFF) << 5;
                    (inst & 0x9F00_001F) | immlo | immhi
                }
                A64_ADD_ABS_LO12_NC => {
                    let lo12 = (value as u32) & 0xFFF;
                    (inst & 0xFFC0_03FF) | (lo12 << 10)
                }
                A64_LDST64_ABS_LO12_NC => {
                    // LDR Xt, [Xn, #lo12] — lo12 scaled by 8 for 64-bit loads
                    let lo12 = ((value as u32) & 0xFFF) >> 3;
                    (inst & 0xFFC0_03FF) | (lo12 << 10)
                }
                _ => {
                    // Unknown relocation type — fall back to raw 4-byte write
                    // (likely wrong, but better than silently dropping).
                    let bytes = (value as i32).to_le_bytes();
                    text[off..off + 4].copy_from_slice(&bytes);
                    return;
                }
            };
            let bytes = patched.to_le_bytes();
            text[off..off + 4].copy_from_slice(&bytes);
        }
        Target::RiscV64 => {
            // RISC-V relocations — most common ones for intra-module calls
            // R_RISCV_JAL (17): 20-bit J-type immediate
            // R_RISCV_BRANCH (16): 12-bit B-type immediate
            // R_RISCV_CALL (18) / R_RISCV_CALL_PLT (19): AUIPC+JALR pair
            const RV_BRANCH: u32 = 16;
            const RV_JAL: u32 = 17;
            const RV_CALL: u32 = 18;
            const RV_CALL_PLT: u32 = 19;

            match rel_type {
                RV_CALL | RV_CALL_PLT => {
                    // AUIPC + JALR pair (8 bytes total)
                    if off + 8 <= text.len() {
                        let hi = ((value as i32 + 0x800) >> 12) & 0xFFFFF;
                        let lo = ((value as i32) & 0xFFF) as u32;
                        let auipc = u32::from_le_bytes([
                            text[off],
                            text[off + 1],
                            text[off + 2],
                            text[off + 3],
                        ]);
                        let jalr = u32::from_le_bytes([
                            text[off + 4],
                            text[off + 5],
                            text[off + 6],
                            text[off + 7],
                        ]);
                        let p_auipc = (auipc & 0xFFF) | ((hi as u32) << 12);
                        let p_jalr = (jalr & 0x000F_FFFF) | (lo << 20);
                        text[off..off + 4].copy_from_slice(&p_auipc.to_le_bytes());
                        text[off + 4..off + 8].copy_from_slice(&p_jalr.to_le_bytes());
                    }
                }
                RV_JAL => {
                    // J-type: imm[20|10:1|11|19:12] in bits [31:12]
                    let inst = u32::from_le_bytes([
                        text[off],
                        text[off + 1],
                        text[off + 2],
                        text[off + 3],
                    ]);
                    let v = value as u32;
                    let enc = ((v & 0x100000) << 11) // imm[20] -> bit 31
                        | ((v & 0x7FE) << 20)        // imm[10:1] -> bits [30:21]
                        | ((v & 0x800) << 9)          // imm[11] -> bit 20
                        | (v & 0xFF000); // imm[19:12] -> bits [19:12]
                    let patched = (inst & 0xFFF) | enc;
                    text[off..off + 4].copy_from_slice(&patched.to_le_bytes());
                }
                RV_BRANCH => {
                    // B-type: imm[12|10:5] in [31:25], imm[4:1|11] in [11:7]
                    let inst = u32::from_le_bytes([
                        text[off],
                        text[off + 1],
                        text[off + 2],
                        text[off + 3],
                    ]);
                    let v = value as u32;
                    let hi = ((v & 0x1000) << 19) | ((v & 0x7E0) << 20);
                    let lo = ((v & 0x1E) << 7) | ((v & 0x800) >> 4);
                    let patched = (inst & 0x01FFF07F) | hi | lo;
                    text[off..off + 4].copy_from_slice(&patched.to_le_bytes());
                }
                _ => {
                    // Fallback: raw 4-byte write
                    let bytes = (value as i32).to_le_bytes();
                    text[off..off + 4].copy_from_slice(&bytes);
                }
            }
        }
    }
}

/// Writes a complete relocatable ELF object file (`.o`) from the compiled
/// code sections, global data, and optional DWARF debug sections.
fn write_relocatable_object(
    text_data: &[u8],
    data_data: &[u8],
    rodata_data: &[u8],
    bss_size: u64,
    functions: &[CompiledFunction],
    globals: &[GlobalSymbolInfo],
    text_relocs: &[crate::backend::traits::FunctionRelocation],
    dwarf: &Option<DwarfSections>,
    module: &IrModule,
    ctx: &CodegenContext,
    _diagnostics: &mut DiagnosticEngine,
) -> Result<Vec<u8>, String> {
    let target = &ctx.target;
    let mut writer = ElfWriter::new(*target, ET_REL);

    // Log the ELF machine type for this target (e.g., EM_X86_64, EM_AARCH64).
    let _elf_machine = target.elf_machine();
    let _is_64 = target.is_64bit();

    // Track section indices dynamically — index 0 is the NULL section, so
    // user sections start at index 1.
    let mut next_section_index: u16 = 1;
    let mut text_section_index: u16 = 0;
    let mut data_section_index: u16 = 0;
    let mut rodata_section_index: u16 = 0;
    let mut bss_section_index: u16 = 0;

    // --- .text section ---
    if !text_data.is_empty() {
        text_section_index = next_section_index;
        next_section_index += 1;
        let text_section = Section {
            name: ".text".to_string(),
            sh_type: SHT_PROGBITS,
            sh_flags: SHF_ALLOC | SHF_EXECINSTR,
            data: text_data.to_vec(),
            sh_addralign: 16,
            sh_entsize: 0,
            sh_link: 0,
            sh_info: 0,
            logical_size: 0,
            virtual_address: 0,
            file_offset_hint: 0,
        };
        writer.add_section(text_section);
    }

    // --- .data section ---
    if !data_data.is_empty() {
        data_section_index = next_section_index;
        next_section_index += 1;
        let data_sec = Section {
            name: ".data".to_string(),
            sh_type: SHT_PROGBITS,
            sh_flags: SHF_ALLOC | SHF_WRITE,
            data: data_data.to_vec(),
            sh_addralign: 8,
            sh_entsize: 0,
            sh_link: 0,
            sh_info: 0,
            logical_size: 0,
            virtual_address: 0,
            file_offset_hint: 0,
        };
        writer.add_section(data_sec);
    }

    // --- .rodata section ---
    if !rodata_data.is_empty() {
        rodata_section_index = next_section_index;
        next_section_index += 1;
        let rodata_sec = Section {
            name: ".rodata".to_string(),
            sh_type: SHT_PROGBITS,
            sh_flags: SHF_ALLOC,
            data: rodata_data.to_vec(),
            sh_addralign: 8,
            sh_entsize: 0,
            sh_link: 0,
            sh_info: 0,
            logical_size: 0,
            virtual_address: 0,
            file_offset_hint: 0,
        };
        writer.add_section(rodata_sec);
    }

    // --- .bss section ---
    if bss_size > 0 {
        bss_section_index = next_section_index;
        next_section_index += 1;
        let bss_sec = Section {
            name: ".bss".to_string(),
            sh_type: SHT_NOBITS,
            sh_flags: SHF_ALLOC | SHF_WRITE,
            data: Vec::new(),
            sh_addralign: 8,
            sh_entsize: 0,
            sh_link: 0,
            sh_info: 0,
            logical_size: bss_size,
            virtual_address: 0,
            file_offset_hint: 0,
        };
        writer.add_section(bss_sec);
    }
    let _ = next_section_index; // suppress unused warning

    // --- DWARF debug sections (conditional on -g) ---
    if let Some(ref dwarf_data) = dwarf {
        add_dwarf_sections(&mut writer, dwarf_data);
    }

    // --- .note.gnu.property section for CET/IBT (x86-64 only) ---
    // When -fcf-protection is enabled, emit the note section so that the
    // runtime loader can verify CET support in all linked object files.
    if ctx.cf_protection && ctx.target == Target::X86_64 {
        let note_data = crate::backend::x86_64::security::generate_cet_note_section();
        let note_section = Section {
            name: ".note.gnu.property".to_string(),
            sh_type: 7, // SHT_NOTE
            sh_flags: SHF_ALLOC,
            data: note_data,
            sh_link: 0,
            sh_info: 0,
            sh_addralign: 8,
            sh_entsize: 0,
            logical_size: 0,
            virtual_address: 0,
            file_offset_hint: 0,
        };
        writer.add_section(note_section);
    }

    // --- Symbol table entries ---

    // Add function symbols with proper visibility from the IR.
    for func in functions {
        let binding = if func.is_global {
            STB_GLOBAL
        } else {
            STB_LOCAL
        };
        let sym = ElfSymbol {
            name: func.name.clone(),
            value: func.offset,
            size: func.size,
            binding,
            sym_type: STT_FUNC,
            visibility: func.visibility,
            section_index: text_section_index,
        };
        writer.add_symbol(sym);
    }

    // Add global variable symbols.
    for gsym in globals {
        let binding = if gsym.is_global {
            STB_GLOBAL
        } else {
            STB_LOCAL
        };
        let sym_type = if gsym.name.starts_with(".L") {
            STT_NOTYPE // Local label / string literal
        } else {
            STT_OBJECT
        };
        let section_index = match gsym.section {
            SectionPlacement::Data => data_section_index,
            SectionPlacement::Rodata => rodata_section_index,
            SectionPlacement::Bss => bss_section_index,
        };
        let sym = ElfSymbol {
            name: gsym.name.clone(),
            value: gsym.offset,
            size: gsym.size,
            binding,
            sym_type,
            visibility: STV_DEFAULT,
            section_index,
        };
        writer.add_symbol(sym);
    }

    // Add external function declaration symbols (undefined).
    // Mark them as STT_FUNC so the dynamic linker creates PLT entries
    // for lazy binding of external function calls (e.g. printf, malloc).
    for decl in module.declarations() {
        let sym = ElfSymbol {
            name: decl.name.clone(),
            value: 0,
            size: 0,
            binding: STB_GLOBAL,
            sym_type: STT_FUNC,
            visibility: STV_DEFAULT,
            section_index: 0, // SHN_UNDEF
        };
        writer.add_symbol(sym);
    }

    // --- .rela.text section (relocations for the .text section) ---
    // Encode the assembler-generated relocations into a proper ELF RELA section
    // so that the linker can patch up cross-section and external symbol references.
    if !text_relocs.is_empty() && text_section_index > 0 {
        let is_64 = ctx.target.is_64bit();

        // For any symbol referenced by a relocation that isn't already in the
        // symbol table (functions, globals, declarations), add it as an
        // undefined (SHN_UNDEF) global symbol.
        let mut extra_undef_symbols: Vec<String> = Vec::new();
        {
            let mut known: crate::common::fx_hash::FxHashSet<String> =
                crate::common::fx_hash::FxHashSet::default();
            for func in functions.iter() {
                known.insert(func.name.clone());
            }
            for gsym in globals.iter() {
                known.insert(gsym.name.clone());
            }
            for decl in module.declarations() {
                known.insert(decl.name.clone());
            }
            for reloc in text_relocs {
                if !known.contains(&reloc.symbol) {
                    known.insert(reloc.symbol.clone());
                    extra_undef_symbols.push(reloc.symbol.clone());
                }
            }
        }
        for name in &extra_undef_symbols {
            let sym = ElfSymbol {
                name: name.clone(),
                value: 0,
                size: 0,
                binding: STB_GLOBAL,
                sym_type: STT_NOTYPE,
                visibility: STV_DEFAULT,
                section_index: 0, // SHN_UNDEF
            };
            writer.add_symbol(sym);
        }

        // Build a name→ELF-symbol-index map that matches the writer's
        // internal ordering: locals first (index 1..), then globals.
        // This mirrors the partitioning in SymbolTable::build_bytes().
        struct SymEntry {
            name: String,
            binding: u8,
        }
        let mut all_syms: Vec<SymEntry> = Vec::new();
        for func in functions.iter() {
            let b = if func.is_global {
                STB_GLOBAL
            } else {
                STB_LOCAL
            };
            all_syms.push(SymEntry {
                name: func.name.clone(),
                binding: b,
            });
        }
        for gsym in globals.iter() {
            let b = if gsym.is_global {
                STB_GLOBAL
            } else {
                STB_LOCAL
            };
            all_syms.push(SymEntry {
                name: gsym.name.clone(),
                binding: b,
            });
        }
        for decl in module.declarations() {
            all_syms.push(SymEntry {
                name: decl.name.clone(),
                binding: STB_GLOBAL,
            });
        }
        for name in &extra_undef_symbols {
            all_syms.push(SymEntry {
                name: name.clone(),
                binding: STB_GLOBAL,
            });
        }

        // Separate into locals and globals, maintaining relative order.
        let locals: Vec<&SymEntry> = all_syms.iter().filter(|s| s.binding == STB_LOCAL).collect();
        let globals_list: Vec<&SymEntry> =
            all_syms.iter().filter(|s| s.binding != STB_LOCAL).collect();

        let mut sym_name_to_idx: FxHashMap<String, u32> = FxHashMap::default();
        // Index 0 = null symbol.  Locals start at index 1.
        for (i, sym) in locals.iter().enumerate() {
            sym_name_to_idx.insert(sym.name.clone(), (i as u32) + 1);
        }
        // Globals start after all locals.
        let global_base = (locals.len() as u32) + 1;
        for (i, sym) in globals_list.iter().enumerate() {
            sym_name_to_idx.insert(sym.name.clone(), global_base + (i as u32));
        }

        // Encode relocation entries into binary RELA format.
        let mut rela_data: Vec<u8> = Vec::new();
        for reloc in text_relocs {
            let sym_idx = sym_name_to_idx.get(&reloc.symbol).copied().unwrap_or(0);

            if is_64 {
                let rela = Elf64Rela::new(reloc.offset, sym_idx, reloc.rel_type_id, reloc.addend);
                rela_data.extend_from_slice(&rela.to_bytes());
            } else {
                // For 32-bit targets, use Elf32Rel format.
                let rel = Elf32Rel::new(reloc.offset as u32, sym_idx, reloc.rel_type_id as u8);
                rela_data.extend_from_slice(&rel.to_bytes());
            }
        }

        if !rela_data.is_empty() {
            // The .symtab section is generated automatically by ElfWriter::write()
            // at index (num_user_sections + 1).  After adding .rela.text, the
            // total user section count determines the .symtab index.
            let num_user_after = writer.sections_count() + 1; // +1 for the section we're about to add
            let symtab_section_idx = (num_user_after + 1) as u32;

            let rela_section = Section {
                name: if is_64 {
                    ".rela.text".to_string()
                } else {
                    ".rel.text".to_string()
                },
                sh_type: if is_64 { SHT_RELA } else { SHT_REL },
                sh_flags: SHF_INFO_LINK,
                data: rela_data,
                sh_addralign: if is_64 { 8 } else { 4 },
                sh_entsize: if is_64 { 24 } else { 8 }, // sizeof(Elf64_Rela) or sizeof(Elf32_Rel)
                sh_link: symtab_section_idx,            // Points to .symtab
                sh_info: text_section_index as u32,     // Section being relocated (.text)
                logical_size: 0,
                virtual_address: 0,
                file_offset_hint: 0,
            };
            writer.add_section(rela_section);
        }
    }

    // Serialize the ELF object.
    let elf_bytes = writer.write();
    Ok(elf_bytes)
}

/// Adds DWARF debug sections to the ELF writer.
///
/// Called only when `-g` is active.  Adds `.debug_info`, `.debug_abbrev`,
/// `.debug_line`, and `.debug_str` as non-allocable (SHF_ALLOC not set)
/// sections.
fn add_dwarf_sections(writer: &mut ElfWriter, dwarf: &DwarfSections) {
    // All DWARF sections are PROGBITS, no flags (not loaded into memory).
    let debug_sections = [
        (".debug_info", &dwarf.debug_info),
        (".debug_abbrev", &dwarf.debug_abbrev),
        (".debug_line", &dwarf.debug_line),
        (".debug_str", &dwarf.debug_str),
    ];

    for (name, data) in &debug_sections {
        if !data.is_empty() {
            let sec = Section {
                name: name.to_string(),
                sh_type: SHT_PROGBITS,
                sh_flags: 0, // Debug sections are NOT loaded (no SHF_ALLOC)
                data: data.to_vec(),
                sh_addralign: 1,
                sh_entsize: 0,
                sh_link: 0,
                sh_info: 0,
                logical_size: 0,
                virtual_address: 0,
                file_offset_hint: 0,
            };
            writer.add_section(sec);
        }
    }
}

// ===========================================================================
// link_to_final_output — invoke the built-in linker
// ===========================================================================

/// Invokes the built-in linker to produce the final ELF binary from a
/// relocatable object file.
///
/// The linker is architecture-specific and handles:
/// - Symbol resolution (strong/weak binding)
/// - Section merging and address layout
/// - Relocation processing
/// - Dynamic section generation (for shared libraries)
///
/// No external linker is invoked — this is the standalone backend mode.
fn link_to_final_output(
    object_bytes: &[u8],
    text_relocations: &[crate::backend::traits::FunctionRelocation],
    ctx: &CodegenContext,
    diagnostics: &mut DiagnosticEngine,
) -> Result<Vec<u8>, String> {
    let output_type = ctx.output_type();

    // For relocatable output, just return the object bytes.
    if output_type == OutputType::Relocatable {
        return Ok(object_bytes.to_vec());
    }

    // Configure the linker.
    let mut config = LinkerConfig::new(ctx.target, output_type);
    config.output_path = ctx.output_path.clone();
    config.pic = ctx.pic;
    config.emit_debug = ctx.should_emit_dwarf();

    if ctx.shared {
        config.allow_undefined = true;
    }

    // Implicitly link against libc for executables — equivalent to GCC's
    // default `-lc`.  This is the standard behaviour: unless the user
    // explicitly passes `-nostdlib` or `-nodefaultlibs`, every C program
    // is linked against the C standard library.  The built-in linker
    // produces a dynamically-linked executable with `libc.so.6` as a
    // `DT_NEEDED` entry and `PT_INTERP` pointing to the system dynamic
    // linker.  Undefined symbols from libc (printf, malloc, etc.) are
    // resolved at runtime by `ld-linux`.
    // Add user-specified -l libraries as DT_NEEDED entries.
    config.library_paths = ctx.library_paths.clone();
    config.libraries = ctx.libraries.clone();
    for lib_name in &ctx.libraries {
        let so_name = format!("lib{}.so", lib_name);
        config.needed_libs.push(so_name);
    }

    if output_type == OutputType::Executable {
        // Add libc if not already present.
        if !config.needed_libs.iter().any(|n| n.starts_with("libc.so")) {
            config.needed_libs.push("libc.so.6".to_string());
        }
        // Allow undefined symbols — they will be resolved at runtime by
        // the dynamic linker via the DT_NEEDED entry.
        config.allow_undefined = true;
    }

    // Parse the relocatable ELF object bytes back into linker internal
    // representation: sections, symbols, and relocations.
    let mut input = parse_elf_object_to_linker_input(0, "input.o", object_bytes);

    // Convert unresolved text relocations (e.g. global variable references)
    // from the code generator into linker-format Relocation entries so the
    // linker can resolve cross-section references (.text → .data/.rodata/.bss).
    for func_reloc in text_relocations {
        input
            .relocations
            .push(crate::backend::linker_common::relocation::Relocation {
                offset: func_reloc.offset,
                symbol_name: func_reloc.symbol.clone(),
                sym_index: 0, // resolved by name, not index
                rel_type: func_reloc.rel_type_id,
                addend: func_reloc.addend,
                object_id: 0,
                section_index: 1, // .text section index in our .o files
                output_section_name: Some(".text".to_string()),
            });
    }

    // For executables (not shared libs), inject a CRT `_start` stub that
    // calls `main` and invokes the exit syscall — but only if the user
    // did not already define `_start`.
    if output_type == OutputType::Executable {
        let has_user_start = input
            .symbols
            .iter()
            .any(|s| s.name == "_start" && s.section_index != 0);
        if !has_user_start {
            inject_crt_start(&mut input, ctx.target);
        }
    }

    // Create the architecture-specific relocation handler.
    let handler = create_relocation_handler(ctx.target);

    // Invoke the architecture-specific built-in linker.  The arch-specific
    // linkers handle GOT/PLT stub generation and dynamic linking correctly,
    // which is required for resolving undefined symbols (e.g. libc functions)
    // via the runtime dynamic linker.
    let linker_result: Result<LinkerOutput, String> = match ctx.target {
        Target::X86_64 => {
            crate::backend::x86_64::linker::link_x86_64(config.clone(), vec![input], diagnostics)
        }
        // Other architectures fall back to the common linker.
        _ => link(&config, vec![input], handler.as_ref(), diagnostics),
    };
    match linker_result {
        Ok(output) => {
            // Validate the linker output.
            let LinkerOutput {
                elf_data,
                entry_point: _,
                output_type: linked_type,
            } = output;

            debug_assert_eq!(linked_type, output_type, "linker output type mismatch");

            Ok(elf_data)
        }
        Err(e) => {
            diagnostics.emit_error(Span::dummy(), format!("linker: {}", e));
            Err(format!("linking failed: {}", e))
        }
    }
}

/// Generate a synthetic CRT startup object that defines `_start`.
///
/// The `_start` entry point:
/// 1. Clears the frame pointer (ABI requirement)
/// 2. Calls `main`
/// 3. Passes the return value to the `exit` syscall
///
/// This is the minimal CRT startup needed to produce a working executable
/// without linking against the system CRT objects (crt0.o, crti.o, etc.).
/// Inject a CRT `_start` stub into the linker input, appending it to
/// the existing .text section with the call to `main` pre-resolved.
///
/// The stub is placed immediately after the user's code. Because both `main`
/// and `_start` are in the same .text section, the PC-relative call
/// displacement can be computed at injection time without a relocation.
fn inject_crt_start(input: &mut LinkerInput, target: Target) {
    use crate::backend::linker_common::symbol_resolver::{
        InputSymbol, STB_GLOBAL, STT_FUNC, STV_DEFAULT,
    };

    // Find the .text section in the existing input.
    let text_idx = input.sections.iter().position(|s| s.name == ".text");

    let text_idx = match text_idx {
        Some(idx) => idx,
        None => return, // No .text — nothing to inject into.
    };

    // Align _start to 16 bytes within .text.
    let current_len = input.sections[text_idx].data.len();
    let aligned = (current_len + 15) & !15;
    // Pad with NOP (0x90 for x86, 0x00 is fine for non-x86 since it won't execute).
    let nop_byte = match target {
        Target::X86_64 | Target::I686 => 0x90u8,
        _ => 0x00u8,
    };
    input.sections[text_idx].data.resize(aligned, nop_byte);

    let start_offset = input.sections[text_idx].data.len();
    // Find the actual offset of `main` in .text by looking up its symbol.
    // The symbol's section_index is the ELF section header index (1-based,
    // with 0 = SHN_UNDEF), so we filter for any defined (non-zero) section.
    let main_offset = input
        .symbols
        .iter()
        .find(|s| s.name == "main" && s.section_index != 0)
        .map(|s| s.value)
        .unwrap_or(0);

    match target {
        Target::X86_64 => {
            // _start stub (16 bytes):
            //   xor %ebp, %ebp          ; 31 ed           (2 bytes)
            //   call main               ; e8 <disp32>     (5 bytes)
            //   mov %eax, %edi           ; 89 c7           (2 bytes)
            //   call exit               ; e8 <00 00 00 00> (5 bytes, PLT32 reloc)
            //   ud2                      ; 0f 0b           (2 bytes, unreachable)
            //
            // Using `call exit` (libc) instead of raw SYS_exit so that
            // atexit handlers run and stdio buffers are flushed, which is
            // required for printf output to appear.
            let call_main_site = start_offset + 2;
            let call_main_next = call_main_site + 5;
            let disp = (main_offset as i64) - (call_main_next as i64);
            let disp_bytes = (disp as i32).to_le_bytes();
            let stub = vec![
                0x31,
                0xed, // xor %ebp, %ebp
                0xe8,
                disp_bytes[0],
                disp_bytes[1], // call main
                disp_bytes[2],
                disp_bytes[3],
                0x89,
                0xc7, // mov %eax, %edi
                0xe8,
                0x00,
                0x00,
                0x00,
                0x00, // call exit (PLT32)
                0x0f,
                0x0b, // ud2
            ];
            // Add R_X86_64_PLT32 relocation for the `exit` call.
            // The displacement bytes start at stub offset 10 (from section
            // start: start_offset + 10).
            let exit_reloc_offset = (start_offset + 10) as u64;
            input
                .relocations
                .push(crate::backend::linker_common::relocation::Relocation {
                    offset: exit_reloc_offset,
                    symbol_name: "exit".to_string(),
                    sym_index: 0,
                    rel_type: 4, // R_X86_64_PLT32
                    addend: -4,
                    object_id: 0,
                    section_index: text_idx as u32,
                    output_section_name: Some(".text".to_string()),
                });
            input.sections[text_idx].data.extend_from_slice(&stub);
        }
        Target::I686 => {
            // _start stub for i686 (16 bytes):
            //   xor %ebp, %ebp          ; 31 ed           (2 bytes)
            //   call main               ; e8 <disp32>     (5 bytes)
            //   push %eax               ; 50              (1 byte, exit status arg)
            //   call exit               ; e8 <00 00 00 00> (5 bytes, PLT32 reloc)
            //   ud2                      ; 0f 0b           (2 bytes, unreachable)
            //   nop                      ; 90              (1 byte, alignment pad)
            //
            // Using `call exit` (libc) instead of raw SYS_exit so that
            // atexit handlers run and stdio buffers are flushed, which is
            // required for printf output to appear.
            let call_site = start_offset + 2;
            let call_next = call_site + 5;
            let disp = (main_offset as i64) - (call_next as i64);
            let disp_bytes = (disp as i32).to_le_bytes();
            let stub = vec![
                0x31,
                0xed, // xor %ebp, %ebp
                0xe8,
                disp_bytes[0],
                disp_bytes[1], // call main
                disp_bytes[2],
                disp_bytes[3],
                0x50, // push %eax (exit status for cdecl)
                0xe8,
                0x00,
                0x00,
                0x00,
                0x00, // call exit (PLT32 relocation)
                0x0f,
                0x0b, // ud2
                0x90, // nop (alignment pad)
            ];
            // Add R_386_PLT32 relocation for the `exit` call.
            // The displacement bytes start at stub offset 9 (from section
            // start: start_offset + 9).
            let exit_reloc_offset = (start_offset + 9) as u64;
            input
                .relocations
                .push(crate::backend::linker_common::relocation::Relocation {
                    offset: exit_reloc_offset,
                    symbol_name: "exit".to_string(),
                    sym_index: 0,
                    rel_type: 4, // R_386_PLT32
                    addend: -4,
                    object_id: 0,
                    section_index: text_idx as u32,
                    output_section_name: Some(".text".to_string()),
                });
            input.sections[text_idx].data.extend_from_slice(&stub);
        }
        Target::AArch64 => {
            // _start stub for AArch64 (16 bytes):
            //   bl main                ; PC-relative, pre-resolved (4 bytes)
            //   bl exit                ; via PLT (R_AARCH64_CALL26)  (4 bytes)
            //   brk #1                 ; unreachable                 (4 bytes)
            //   nop                    ; alignment pad               (4 bytes)
            //
            // Using `bl exit` (libc) instead of raw SVC so that atexit
            // handlers run and stdio buffers are flushed.
            let bl_offset = start_offset;
            let offset_bytes = ((main_offset as i64) - (bl_offset as i64)) / 4;
            let imm26 = (offset_bytes as u32) & 0x03FF_FFFF;
            let bl_main = 0x9400_0000u32 | imm26;
            let bl_exit_placeholder = 0x9400_0000u32; // BL with disp 0 (reloc fills)
            let brk1 = 0xD420_0020u32; // brk #1
            let nop = 0xD503_201Fu32;
            let mut stub = Vec::new();
            stub.extend_from_slice(&bl_main.to_le_bytes());
            stub.extend_from_slice(&bl_exit_placeholder.to_le_bytes());
            stub.extend_from_slice(&brk1.to_le_bytes());
            stub.extend_from_slice(&nop.to_le_bytes());
            // Add R_AARCH64_CALL26 relocation for the `exit` call.
            let exit_reloc_offset = (start_offset + 4) as u64;
            input
                .relocations
                .push(crate::backend::linker_common::relocation::Relocation {
                    offset: exit_reloc_offset,
                    symbol_name: "exit".to_string(),
                    sym_index: 0,
                    rel_type: 283, // R_AARCH64_CALL26
                    addend: 0,
                    object_id: 0,
                    section_index: text_idx as u32,
                    output_section_name: Some(".text".to_string()),
                });
            input.sections[text_idx].data.extend_from_slice(&stub);
        }
        Target::RiscV64 => {
            // _start stub for RISC-V 64 (20 bytes):
            //   jal ra, main            ; PC-relative, pre-resolved       (4 bytes)
            //   auipc ra, 0             ; exit call via PLT (CALL_PLT)    (4 bytes)
            //   jalr ra, ra, 0          ; continuation of CALL_PLT pair   (4 bytes)
            //   ebreak                  ; unreachable                     (4 bytes)
            //   nop                     ; alignment pad                   (4 bytes)
            //
            // Using AUIPC+JALR pair (R_RISCV_CALL_PLT) for `exit` so the
            // dynamic linker routes through PLT and atexit handlers +
            // stdio buffer flushing happen correctly.
            let jal_offset = start_offset;
            let offset = (main_offset as i64) - (jal_offset as i64);
            let imm = offset as i32;
            let imm20 = ((imm >> 20) & 1) as u32;
            let imm10_1 = ((imm >> 1) & 0x3FF) as u32;
            let imm11 = ((imm >> 11) & 1) as u32;
            let imm19_12 = ((imm >> 12) & 0xFF) as u32;
            let jal_main = (imm20 << 31)
                | (imm10_1 << 21)
                | (imm11 << 20)
                | (imm19_12 << 12)
                | (1 << 7)
                | 0x6F;
            // AUIPC ra, 0 — upper 20 bits filled by linker relocation
            let auipc_exit = 0x0000_0097u32; // auipc ra, 0
                                             // JALR ra, ra, 0 — lower 12 bits filled by linker relocation
            let jalr_exit = 0x000080E7u32; // jalr ra, 0(ra)
            let ebreak = 0x0010_0073u32;
            let nop = 0x0000_0013u32; // addi x0, x0, 0
            let mut stub = Vec::new();
            stub.extend_from_slice(&jal_main.to_le_bytes());
            stub.extend_from_slice(&auipc_exit.to_le_bytes());
            stub.extend_from_slice(&jalr_exit.to_le_bytes());
            stub.extend_from_slice(&ebreak.to_le_bytes());
            stub.extend_from_slice(&nop.to_le_bytes());
            // Add R_RISCV_CALL_PLT relocation for `exit` at the auipc+jalr pair.
            let exit_reloc_offset = (start_offset + 4) as u64;
            input
                .relocations
                .push(crate::backend::linker_common::relocation::Relocation {
                    offset: exit_reloc_offset,
                    symbol_name: "exit".to_string(),
                    sym_index: 0,
                    rel_type: 19, // R_RISCV_CALL_PLT
                    addend: 0,
                    object_id: 0,
                    section_index: text_idx as u32,
                    output_section_name: Some(".text".to_string()),
                });
            input.sections[text_idx].data.extend_from_slice(&stub);
        }
    }

    let stub_size = input.sections[text_idx].data.len() - start_offset;

    // Update section size.
    input.sections[text_idx].size = input.sections[text_idx].data.len() as u64;

    // Add the `_start` symbol at the stub's offset.
    let section_original_index = input.sections[text_idx].original_index;
    input.symbols.push(InputSymbol {
        name: "_start".to_string(),
        value: start_offset as u64,
        size: stub_size as u64,
        binding: STB_GLOBAL,
        sym_type: STT_FUNC,
        visibility: STV_DEFAULT,
        section_index: section_original_index as u16,
        object_file_id: input.object_id,
    });

    // For all architectures, the _start stub calls `exit` through PLT
    // (libc's exit) so that stdio buffers are flushed and atexit handlers
    // run. Add `exit` as an undefined external symbol so the linker creates
    // a PLT entry and the dynamic linker resolves it at runtime.
    {
        let has_exit = input.symbols.iter().any(|s| s.name == "exit");
        if !has_exit {
            input.symbols.push(InputSymbol {
                name: "exit".to_string(),
                value: 0,
                size: 0,
                binding: STB_GLOBAL,
                sym_type: STT_FUNC,
                visibility: STV_DEFAULT,
                section_index: 0, // SHN_UNDEF — external
                object_file_id: input.object_id,
            });
        }
    }
}

/// Create an architecture-specific relocation handler based on the target.
///
/// Each architecture provides its own `RelocationHandler` implementation
/// in its `linker/relocations.rs` module.  This function dispatches to the
/// correct handler for the given compilation target.
pub fn create_relocation_handler(target: Target) -> Box<dyn RelocationHandler> {
    match target {
        Target::X86_64 => Box::new(crate::backend::x86_64::linker::X86_64RelocationHandler::new()),
        Target::I686 => Box::new(crate::backend::i686::linker::I686RelocationHandler::new()),
        Target::AArch64 => {
            Box::new(crate::backend::aarch64::linker::relocations::AArch64RelocationHandler::new())
        }
        Target::RiscV64 => {
            Box::new(crate::backend::riscv64::linker::relocations::RiscV64RelocationHandler::new())
        }
    }
}

/// Parse a relocatable ELF object file (ET_REL) byte buffer into a
/// [`LinkerInput`] suitable for the built-in linker.
///
/// This performs a lightweight ELF header / section header scan:
/// - Identifies `.text`, `.data`, `.rodata`, `.bss` sections and copies
///   their bytes into [`InputSection`]s
/// - Extracts `.symtab` entries into [`InputSymbol`]s
/// - Extracts `.rela.*` relocations (if any)
///
/// The implementation handles the common ELF structures produced by the
/// BCC ELF writer (`elf_writer_common.rs`).  External object files with
/// unusual section layouts may not parse perfectly, but BCC only links
/// its own objects in standalone mode.
pub fn parse_elf_object_to_linker_input(
    object_id: u32,
    filename: &str,
    data: &[u8],
) -> LinkerInput {
    use crate::backend::linker_common::section_merger::InputSection;
    use crate::backend::linker_common::symbol_resolver::{
        InputSymbol, STB_GLOBAL, STB_LOCAL, STB_WEAK, STT_FUNC, STT_NOTYPE, STT_OBJECT, STV_DEFAULT,
    };

    let mut input = LinkerInput::new(object_id, filename.to_string());

    // Minimum ELF header size check (64-byte ELF64 header).
    if data.len() < 64 {
        // Too small to be a valid ELF — return an input with the raw data
        // as a single .text section so the linker can at least try.
        input.sections.push(InputSection {
            name: ".text".to_string(),
            section_type: SHT_PROGBITS,
            flags: SHF_ALLOC | SHF_EXECINSTR,
            data: data.to_vec(),
            size: data.len() as u64,
            alignment: 16,
            object_id,
            original_index: 1,
            group_signature: None,
            relocations: Vec::new(),
        });
        return input;
    }

    // Detect ELF class from e_ident[EI_CLASS].
    let is_64bit = data[4] == 2;
    // Detect endianness from e_ident[EI_DATA].
    let is_le = data[5] == 1;

    // Helper closures for reading ELF integers.
    let read_u16 = |off: usize| -> u16 {
        if is_le {
            u16::from_le_bytes([data[off], data[off + 1]])
        } else {
            u16::from_be_bytes([data[off], data[off + 1]])
        }
    };
    let read_u32 = |off: usize| -> u32 {
        if is_le {
            u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
        } else {
            u32::from_be_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
        }
    };
    let read_u64 = |off: usize| -> u64 {
        if is_le {
            u64::from_le_bytes([
                data[off],
                data[off + 1],
                data[off + 2],
                data[off + 3],
                data[off + 4],
                data[off + 5],
                data[off + 6],
                data[off + 7],
            ])
        } else {
            u64::from_be_bytes([
                data[off],
                data[off + 1],
                data[off + 2],
                data[off + 3],
                data[off + 4],
                data[off + 5],
                data[off + 6],
                data[off + 7],
            ])
        }
    };

    // Parse ELF header fields.
    let (e_shoff, e_shentsize, e_shnum, e_shstrndx) = if is_64bit {
        (
            read_u64(40) as usize,
            read_u16(58) as usize,
            read_u16(60) as usize,
            read_u16(62) as usize,
        )
    } else {
        (
            read_u32(32) as usize,
            read_u16(46) as usize,
            read_u16(48) as usize,
            read_u16(50) as usize,
        )
    };

    if e_shoff == 0 || e_shnum == 0 || e_shoff + e_shnum * e_shentsize > data.len() {
        // Malformed section headers — fall back to raw .text section.
        input.sections.push(InputSection {
            name: ".text".to_string(),
            section_type: SHT_PROGBITS,
            flags: SHF_ALLOC | SHF_EXECINSTR,
            data: data.to_vec(),
            size: data.len() as u64,
            alignment: 16,
            object_id,
            original_index: 1,
            group_signature: None,
            relocations: Vec::new(),
        });
        return input;
    }

    // Read section header entries into a temporary vector.
    #[allow(dead_code)]
    struct ShdrInfo {
        sh_name: u32,
        sh_type: u32,
        sh_flags: u64,
        sh_offset: usize,
        sh_size: usize,
        sh_link: u32,
        sh_info: u32,
        sh_addralign: u64,
        sh_entsize: u64,
    }

    let mut shdrs: Vec<ShdrInfo> = Vec::with_capacity(e_shnum);
    for i in 0..e_shnum {
        let base = e_shoff + i * e_shentsize;
        if base + e_shentsize > data.len() {
            break;
        }
        if is_64bit {
            shdrs.push(ShdrInfo {
                sh_name: read_u32(base),
                sh_type: read_u32(base + 4),
                sh_flags: read_u64(base + 8),
                sh_offset: read_u64(base + 24) as usize,
                sh_size: read_u64(base + 32) as usize,
                sh_link: read_u32(base + 40),
                sh_info: read_u32(base + 44),
                sh_addralign: read_u64(base + 48),
                sh_entsize: read_u64(base + 56),
            });
        } else {
            shdrs.push(ShdrInfo {
                sh_name: read_u32(base),
                sh_type: read_u32(base + 4),
                sh_flags: read_u32(base + 8) as u64,
                sh_offset: read_u32(base + 16) as usize,
                sh_size: read_u32(base + 20) as usize,
                sh_link: read_u32(base + 24),
                sh_info: read_u32(base + 28),
                sh_addralign: read_u32(base + 32) as u64,
                sh_entsize: read_u32(base + 36) as u64,
            });
        }
    }

    // Build section name lookup from .shstrtab.
    let shstrtab_data: &[u8] = if e_shstrndx < shdrs.len() {
        let sh = &shdrs[e_shstrndx];
        let end = sh.sh_offset.saturating_add(sh.sh_size).min(data.len());
        &data[sh.sh_offset..end]
    } else {
        &[]
    };

    let section_name = |name_off: u32| -> String {
        let off = name_off as usize;
        if off >= shstrtab_data.len() {
            return String::new();
        }
        let end = shstrtab_data[off..]
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(shstrtab_data.len() - off);
        String::from_utf8_lossy(&shstrtab_data[off..off + end]).into_owned()
    };

    // Locate the symbol table (.symtab) and its associated string table.
    let mut symtab_idx: Option<usize> = None;
    for (i, sh) in shdrs.iter().enumerate() {
        if sh.sh_type == 2 {
            // SHT_SYMTAB
            symtab_idx = Some(i);
            break;
        }
    }

    // Extract symbols from .symtab.
    if let Some(si) = symtab_idx {
        let sym_sh = &shdrs[si];
        let strtab_sh = if (sym_sh.sh_link as usize) < shdrs.len() {
            &shdrs[sym_sh.sh_link as usize]
        } else {
            // No string table — skip symbol extraction.
            &shdrs[0]
        };

        let strtab_data: &[u8] = {
            let end = strtab_sh
                .sh_offset
                .saturating_add(strtab_sh.sh_size)
                .min(data.len());
            &data[strtab_sh.sh_offset..end]
        };

        let sym_name_from = |name_off: u32| -> String {
            let off = name_off as usize;
            if off >= strtab_data.len() {
                return String::new();
            }
            let end = strtab_data[off..]
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(strtab_data.len() - off);
            String::from_utf8_lossy(&strtab_data[off..off + end]).into_owned()
        };

        let entry_size = if sym_sh.sh_entsize > 0 {
            sym_sh.sh_entsize as usize
        } else if is_64bit {
            24
        } else {
            16
        };

        let num_syms = if entry_size > 0 {
            sym_sh.sh_size / entry_size
        } else {
            0
        };

        for j in 0..num_syms {
            let sym_off = sym_sh.sh_offset + j * entry_size;
            if sym_off + entry_size > data.len() {
                break;
            }

            let (st_name, st_info, st_other, st_shndx, st_value, st_size) = if is_64bit {
                (
                    read_u32(sym_off),
                    data[sym_off + 4],
                    data[sym_off + 5],
                    read_u16(sym_off + 6),
                    read_u64(sym_off + 8),
                    read_u64(sym_off + 16),
                )
            } else {
                (
                    read_u32(sym_off),
                    data[sym_off + 12],
                    data[sym_off + 13],
                    read_u16(sym_off + 14),
                    read_u32(sym_off + 4) as u64,
                    read_u32(sym_off + 8) as u64,
                )
            };

            let binding = st_info >> 4;
            let sym_type = st_info & 0xf;
            let name = sym_name_from(st_name);

            // Skip the null symbol entry.
            if j == 0 && name.is_empty() && st_value == 0 && st_shndx == 0 {
                continue;
            }

            input.symbols.push(InputSymbol {
                name,
                value: st_value,
                size: st_size,
                binding: match binding {
                    0 => STB_LOCAL,
                    1 => STB_GLOBAL,
                    2 => STB_WEAK,
                    _ => STB_GLOBAL,
                },
                sym_type: match sym_type {
                    0 => STT_NOTYPE,
                    1 => STT_OBJECT,
                    2 => STT_FUNC,
                    _ => STT_NOTYPE,
                },
                visibility: st_other & 0x3,
                section_index: st_shndx,
                object_file_id: object_id,
            });
        }
    }

    // Extract loadable sections (.text, .data, .rodata, .bss, .debug_*).
    for (i, sh) in shdrs.iter().enumerate() {
        if i == 0 {
            continue; // skip null section
        }
        let name = section_name(sh.sh_name);
        // Skip non-loadable sections that the linker doesn't need directly.
        // Include PROGBITS, NOBITS, and RELA sections.
        let is_progbits = sh.sh_type == SHT_PROGBITS;
        let is_nobits = sh.sh_type == SHT_NOBITS;
        let is_rela = sh.sh_type == 4; // SHT_RELA
        if !is_progbits && !is_nobits && !is_rela {
            continue;
        }
        if is_rela {
            // Relocation sections are processed separately below.
            continue;
        }

        let sec_data = if is_nobits {
            Vec::new()
        } else {
            let end = sh.sh_offset.saturating_add(sh.sh_size).min(data.len());
            data[sh.sh_offset..end].to_vec()
        };

        input.sections.push(InputSection {
            name,
            section_type: sh.sh_type,
            flags: sh.sh_flags,
            data: sec_data,
            size: sh.sh_size as u64,
            alignment: sh.sh_addralign.max(1),
            object_id,
            original_index: i as u32,
            group_signature: None,
            relocations: Vec::new(),
        });
    }

    // If no sections were extracted, provide the raw bytes as a .text
    // section to ensure the linker has something to work with.
    if input.sections.is_empty() {
        input.sections.push(InputSection {
            name: ".text".to_string(),
            section_type: SHT_PROGBITS,
            flags: SHF_ALLOC | SHF_EXECINSTR,
            data: data.to_vec(),
            size: data.len() as u64,
            alignment: 16,
            object_id,
            original_index: 1,
            group_signature: None,
            relocations: Vec::new(),
        });
    }

    // If no symbols were extracted, add a default _start symbol at offset 0
    // so the linker can resolve the entry point.
    if input.symbols.is_empty() {
        input.symbols.push(InputSymbol {
            name: "_start".to_string(),
            value: 0,
            size: 0,
            binding: STB_GLOBAL,
            sym_type: STT_FUNC,
            visibility: STV_DEFAULT,
            section_index: 1, // .text
            object_file_id: object_id,
        });
    }

    input
}

// ===========================================================================
// emit_assembly_text — produce human-readable assembly output (-S)
// ===========================================================================

/// Emits human-readable assembly text when the `-S` flag is specified.
///
/// Produces AT&T syntax for x86-64/i686 and standard syntax for
/// AArch64/RISC-V 64.
fn emit_assembly_text<A: ArchCodegen>(
    arch: &A,
    module: &IrModule,
    ctx: &CodegenContext,
    diagnostics: &mut DiagnosticEngine,
) -> Result<Vec<u8>, String> {
    let mut output = String::new();

    // File header with target information.
    output.push_str(&format!(
        "\t.file\t\"{}\"\n",
        module.name.replace('"', "\\\"")
    ));
    output.push_str(&format!("\t# Target: {}\n", ctx.target));
    output.push('\n');

    // Emit top-level inline assembly blocks verbatim.
    for asm_block in &module.inline_asm_blocks {
        output.push_str(&asm_block.assembly);
        output.push('\n');
    }

    // .text section
    output.push_str("\t.text\n");

    for func in module.functions() {
        if !func.is_definition {
            continue;
        }

        // Emit function label.
        if func.linkage == FunctionLinkage::External {
            output.push_str(&format!("\t.globl\t{}\n", func.name));
        }
        output.push_str(&format!("\t.type\t{}, @function\n", func.name));
        output.push_str(&format!("{}:\n", func.name));

        // Perform instruction selection for assembly text output.
        // Use per-function ref maps to avoid cross-function Value-ID collisions.
        let mut mf = match arch.lower_function(
            func,
            diagnostics,
            module.globals(),
            &func.func_ref_map,
            &func.global_var_refs,
        ) {
            Ok(mf) => mf,
            Err(e) => {
                diagnostics.emit_error(
                    Span::dummy(),
                    format!(
                        "assembly output: instruction selection failed for '{}': {}",
                        func.name, e
                    ),
                );
                continue;
            }
        };

        // Register allocation — resolve virtual registers to physical regs.
        {
            let reg_info: TraitsRegisterInfo = arch.register_info();
            let alloc_reg_info: register_allocator::RegisterInfo = reg_info.into();
            let mut intervals = compute_live_intervals(func);
            let alloc_result = allocate_registers(&mut intervals, &alloc_reg_info, &ctx.target);
            insert_spill_code_from_result(&mut mf, &alloc_result);
            apply_allocation_result(&mut mf, &alloc_result, &ctx.target);

            // Insert prologue/epilogue.
            let prologue = arch.emit_prologue(&mf);
            let epilogue = arch.emit_epilogue(&mf);
            insert_prologue_epilogue(&mut mf, prologue, epilogue);
        }

        // Emit machine instructions as text.
        for (i, block) in mf.blocks.iter().enumerate() {
            if i > 0 {
                if let Some(ref label) = block.label {
                    output.push_str(&format!("{}:\n", label));
                } else {
                    output.push_str(&format!(".LBB_{}_{}:\n", func.name, i));
                }
            }
            for inst in &block.instructions {
                output.push_str(&format!("\t{}\n", arch.format_instruction(inst)));
            }
        }

        output.push_str(&format!("\t.size\t{}, .-{}\n", func.name, func.name));
        output.push('\n');
    }

    // .data section
    if !module.globals().is_empty() {
        output.push_str("\t.data\n");
        for global in module.globals() {
            if !global.is_constant && global.initializer.is_some() {
                if global.linkage == crate::ir::module::Linkage::External {
                    output.push_str(&format!("\t.globl\t{}\n", global.name));
                }
                output.push_str(&format!("{}:\n", global.name));
                if let Some(ref init) = global.initializer {
                    emit_constant_as_asm(init, &mut output);
                }
            }
        }
    }

    // .rodata section for constants and string literals.
    let has_rodata =
        module.globals().iter().any(|g| g.is_constant) || !module.string_pool().is_empty();
    if has_rodata {
        output.push_str("\n\t.section\t.rodata\n");
        for global in module.globals() {
            if global.is_constant {
                if global.linkage == crate::ir::module::Linkage::External {
                    output.push_str(&format!("\t.globl\t{}\n", global.name));
                }
                output.push_str(&format!("{}:\n", global.name));
                if let Some(ref init) = global.initializer {
                    emit_constant_as_asm(init, &mut output);
                }
            }
        }
        for string_lit in module.string_pool() {
            output.push_str(&format!("{}:\n", string_lit.label));
            output.push_str("\t.ascii\t\"");
            for &b in &string_lit.bytes {
                if b == b'\\' {
                    output.push_str("\\\\");
                } else if b == b'"' {
                    output.push_str("\\\"");
                } else if b == b'\n' {
                    output.push_str("\\n");
                } else if b == b'\t' {
                    output.push_str("\\t");
                } else if b == 0 {
                    output.push_str("\\0");
                } else if b.is_ascii_graphic() || b == b' ' {
                    output.push(b as char);
                } else {
                    output.push_str(&format!("\\{:03o}", b));
                }
            }
            output.push_str("\"\n");
        }
    }

    // .bss section
    let has_bss = module.globals().iter().any(|g| {
        g.initializer.is_none()
            || matches!(&g.initializer, Some(crate::ir::module::Constant::ZeroInit))
    });
    if has_bss {
        output.push_str("\n\t.bss\n");
        for global in module.globals() {
            let is_bss = global.initializer.is_none()
                || matches!(
                    &global.initializer,
                    Some(crate::ir::module::Constant::ZeroInit)
                );
            if is_bss {
                if global.linkage == crate::ir::module::Linkage::External {
                    output.push_str(&format!("\t.globl\t{}\n", global.name));
                }
                let size = global_variable_size(global);
                output.push_str(&format!("\t.comm\t{},{}\n", global.name, size));
            }
        }
    }

    // External declarations.
    for decl in module.declarations() {
        output.push_str(&format!("\t.extern\t{}\n", decl.name));
    }

    Ok(output.into_bytes())
}

/// Emits a constant initializer as assembly directives.
fn emit_constant_as_asm(constant: &crate::ir::module::Constant, output: &mut String) {
    use crate::ir::module::Constant;

    match constant {
        Constant::Integer(v) => {
            if *v >= i32::MIN as i128 && *v <= u32::MAX as i128 {
                output.push_str(&format!("\t.long\t{}\n", *v as i64));
            } else {
                output.push_str(&format!("\t.quad\t{}\n", *v as i64));
            }
        }
        Constant::Float(v) => {
            // Emit as raw bytes to preserve exact bit pattern.
            let bytes = v.to_le_bytes();
            output.push_str("\t.byte\t");
            for (i, b) in bytes.iter().enumerate() {
                if i > 0 {
                    output.push_str(", ");
                }
                output.push_str(&format!("0x{:02x}", b));
            }
            output.push('\n');
        }
        Constant::LongDouble(bytes) => {
            output.push_str("\t.byte\t");
            for (i, b) in bytes.iter().enumerate() {
                if i > 0 {
                    output.push_str(", ");
                }
                output.push_str(&format!("0x{:02x}", b));
            }
            output.push('\n');
            // Pad to 16 bytes.
            output.push_str("\t.zero\t6\n");
        }
        Constant::String(bytes) => {
            output.push_str("\t.ascii\t\"");
            for &b in bytes {
                if b.is_ascii_graphic() || b == b' ' {
                    output.push(b as char);
                } else {
                    output.push_str(&format!("\\{:03o}", b));
                }
            }
            output.push_str("\"\n");
        }
        Constant::ZeroInit => {
            output.push_str("\t.zero\t8\n");
        }
        Constant::Struct(fields) => {
            for field in fields {
                emit_constant_as_asm(field, output);
            }
        }
        Constant::Array(elements) => {
            for elem in elements {
                emit_constant_as_asm(elem, output);
            }
        }
        Constant::GlobalRef(name) => {
            output.push_str(&format!("\t.quad\t{}\n", name));
        }
        Constant::Null => {
            output.push_str("\t.quad\t0\n");
        }
        Constant::Undefined => {
            output.push_str("\t.zero\t8\n");
        }
    }
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codegen_context_default() {
        let ctx = CodegenContext::default();
        assert_eq!(ctx.target, Target::X86_64);
        assert!(!ctx.debug_info);
        assert_eq!(ctx.optimization_level, 0);
        assert!(!ctx.pic);
        assert!(!ctx.shared);
        assert!(!ctx.retpoline);
        assert!(!ctx.cf_protection);
        assert_eq!(ctx.output_path, "a.out");
        assert!(!ctx.compile_only);
        assert!(!ctx.emit_assembly);
    }

    #[test]
    fn test_should_emit_dwarf() {
        let mut ctx = CodegenContext::default();
        assert!(!ctx.should_emit_dwarf()); // no -g

        ctx.debug_info = true;
        ctx.optimization_level = 0;
        assert!(ctx.should_emit_dwarf()); // -g -O0

        ctx.optimization_level = 1;
        assert!(!ctx.should_emit_dwarf()); // -g -O1 → no DWARF
    }

    #[test]
    fn test_security_mitigations() {
        let mut ctx = CodegenContext::default();
        ctx.target = Target::X86_64;
        ctx.retpoline = true;
        assert!(ctx.has_security_mitigations());

        ctx.target = Target::AArch64;
        assert!(!ctx.has_security_mitigations()); // Not x86-64
    }

    #[test]
    fn test_output_type() {
        let mut ctx = CodegenContext::default();
        assert_eq!(ctx.output_type(), OutputType::Executable);

        ctx.compile_only = true;
        assert_eq!(ctx.output_type(), OutputType::Relocatable);

        ctx.compile_only = false;
        ctx.shared = true;
        assert_eq!(ctx.output_type(), OutputType::SharedLibrary);
    }

    #[test]
    fn test_constant_to_bytes_integer() {
        use crate::ir::module::Constant;
        let bytes = constant_to_bytes(&Constant::Integer(42));
        assert!(!bytes.is_empty());
        // Should be little-endian encoding of 42.
        assert_eq!(bytes[0], 42);
    }

    #[test]
    fn test_constant_to_bytes_string() {
        use crate::ir::module::Constant;
        let data = b"Hello\0".to_vec();
        let bytes = constant_to_bytes(&Constant::String(data.clone()));
        assert_eq!(bytes, data);
    }

    #[test]
    fn test_constant_to_bytes_zero_init() {
        use crate::ir::module::Constant;
        let bytes = constant_to_bytes(&Constant::ZeroInit);
        assert!(bytes.is_empty());
    }

    #[test]
    fn test_ir_type_sizes() {
        use crate::ir::types::IrType;
        assert_eq!(ir_type_size(&IrType::I8), 1);
        assert_eq!(ir_type_size(&IrType::I16), 2);
        assert_eq!(ir_type_size(&IrType::I32), 4);
        assert_eq!(ir_type_size(&IrType::I64), 8);
        assert_eq!(ir_type_size(&IrType::Ptr), 8);
        assert_eq!(ir_type_size(&IrType::F32), 4);
        assert_eq!(ir_type_size(&IrType::F64), 8);
    }
}
