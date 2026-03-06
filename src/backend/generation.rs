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
    ElfSymbol, ElfWriter, Section, ET_REL, SHF_ALLOC, SHF_EXECINSTR, SHF_WRITE, SHT_NOBITS,
    SHT_PROGBITS, STB_GLOBAL, STB_LOCAL, STT_FUNC, STT_NOTYPE, STT_OBJECT, STV_DEFAULT,
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
use crate::common::source_map::SourceMap;
use crate::common::target::Target;
use crate::ir::function::{IrFunction, Linkage as FunctionLinkage};
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
            generate_for_arch(&codegen, module, ctx, diagnostics)
        }
        Target::I686 => {
            let codegen = crate::backend::i686::I686Codegen::new(ctx.pic, ctx.debug_info);
            generate_for_arch(&codegen, module, ctx, diagnostics)
        }
        Target::AArch64 => {
            let codegen = crate::backend::aarch64::AArch64Codegen::new(ctx.pic, ctx.debug_info);
            generate_for_arch(&codegen, module, ctx, diagnostics)
        }
        Target::RiscV64 => {
            let codegen = crate::backend::riscv64::RiscV64Codegen::new(ctx.pic, ctx.debug_info);
            generate_for_arch(&codegen, module, ctx, diagnostics)
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

    let functions: &[IrFunction] = module.functions();
    for func in functions {
        // Skip declaration-only functions (no body to compile).
        if !func.is_definition {
            continue;
        }

        // Step 1: Instruction selection — IR → MachineFunction
        let mut mf = match arch.lower_function(func, diagnostics, module.globals()) {
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
        apply_allocation_result(&mut mf, &alloc_result);

        // Step 3: Insert prologue and epilogue instructions.
        let prologue = arch.emit_prologue(&mf);
        let epilogue = arch.emit_epilogue(&mf);
        insert_prologue_epilogue(&mut mf, prologue, epilogue);

        // Step 4: Assembly encoding — machine instructions → raw bytes
        let func_bytes = match arch.emit_assembly(&mf) {
            Ok(bytes) => bytes,
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

        // Track the function's position within the .text section.
        let func_offset = text_section_data.len() as u64;
        let func_size = func_bytes.len() as u64;
        text_section_offsets.push((func.name.clone(), func_offset, func_size));

        compiled_functions.push(CompiledFunction {
            name: func.name.clone(),
            bytes: func_bytes.clone(),
            offset: func_offset,
            size: func_size,
            is_global: func.linkage == FunctionLinkage::External
                || func.linkage == FunctionLinkage::Weak,
        });

        text_section_data.extend_from_slice(&func_bytes);
    }

    // Check for errors after function compilation.
    if diagnostics.has_errors() {
        return Err("code generation aborted due to errors".to_string());
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
        // Create a minimal SourceMap for DWARF generation.
        // In a full pipeline, the SourceMap would be passed from the frontend.
        // Here we pass a reference to generate debug info from function/module metadata.
        let source_map = SourceMap::new();

        // Verify DWARF parameters for this target:
        // - Address size is 4 bytes on 32-bit targets, 8 bytes on 64-bit targets
        // - DWARF version is always v4 for BCC
        let addr_size = dwarf_address_size(&ctx.target);
        debug_assert!(
            (ctx.target.is_64bit() && addr_size == 8) || (!ctx.target.is_64bit() && addr_size == 4),
            "DWARF address size mismatch for target"
        );
        debug_assert_eq!(DWARF_VERSION, 4, "BCC targets DWARF v4");

        // Use SourceMap API to resolve file information for the module's source.
        // In a full pipeline, the SourceMap would be pre-populated with file
        // entries from the frontend; here we probe the API to ensure it's
        // connected for DWARF generation.
        if let Some(file) = source_map.get_file(0) {
            let _ = file; // File entry available for DWARF compilation unit DIE
        }
        if let Some(loc) = source_map.lookup_location(0, 0) {
            let _ = loc; // Location available for DWARF line mapping
        }
        if let Some(filename) = source_map.get_filename(0) {
            let _ = filename; // Filename for DWARF DW_AT_name attribute
        }

        let dwarf = generate_dwarf_sections(
            module,
            &source_map,
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

    link_to_final_output(&object_bytes, ctx, diagnostics)
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
fn apply_allocation_result(mf: &mut MachineFunction, alloc: &AllocationResult) {
    // Record callee-saved registers used by this function.
    mf.callee_saved_regs = alloc.callee_saved_used.iter().map(|pr| pr.0).collect();

    // Update frame size with spill slot requirements.
    mf.frame_size = mf.frame_size.max(alloc.frame_size);

    // Replace virtual registers with physical registers in all instructions.
    for block in &mut mf.blocks {
        for inst in &mut block.instructions {
            // Replace operands.
            for operand in &mut inst.operands {
                if let MachineOperand::VirtualRegister(vreg) = operand {
                    let value = crate::ir::instructions::Value(*vreg);
                    if let Some(phys) = alloc.assignments.get(&value) {
                        *operand = MachineOperand::Register(phys.0);
                    }
                }
            }
            // Replace result operand.
            if let Some(ref mut result) = inst.result {
                if let MachineOperand::VirtualRegister(vreg) = result {
                    let value = crate::ir::instructions::Value(*vreg);
                    if let Some(phys) = alloc.assignments.get(&value) {
                        *result = MachineOperand::Register(phys.0);
                    }
                }
            }
        }
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

    // Insert epilogue before every terminator that is a return.
    if !epilogue.is_empty() {
        for block in &mut mf.blocks {
            if let Some(last) = block.instructions.last() {
                if last.is_terminator && !last.is_branch {
                    // This looks like a return instruction — insert epilogue before it.
                    let last_idx = block.instructions.len() - 1;
                    let mut new_insts =
                        Vec::with_capacity(block.instructions.len() + epilogue.len());
                    new_insts.extend_from_slice(&block.instructions[..last_idx]);
                    new_insts.extend(epilogue.clone());
                    new_insts.push(block.instructions[last_idx].clone());
                    block.instructions = new_insts;
                }
            }
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
            let bytes = constant_to_bytes(init);
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
            let bytes = constant_to_bytes(init);
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
fn constant_to_bytes(constant: &crate::ir::module::Constant) -> Vec<u8> {
    use crate::ir::module::Constant;

    match constant {
        Constant::Integer(v) => {
            // Emit as the smallest representation that fits.
            // Default to 8 bytes for general use.
            v.to_le_bytes().to_vec()
        }
        Constant::Float(v) => v.to_le_bytes().to_vec(),
        Constant::LongDouble(bytes) => {
            // 80-bit extended precision — 10 raw bytes, padded to 16.
            let mut result = bytes.to_vec();
            result.resize(16, 0);
            result
        }
        Constant::String(bytes) => bytes.clone(),
        Constant::ZeroInit => Vec::new(),
        Constant::Struct(fields) => {
            let mut result = Vec::new();
            for field in fields {
                result.extend_from_slice(&constant_to_bytes(field));
            }
            result
        }
        Constant::Array(elements) => {
            let mut result = Vec::new();
            for elem in elements {
                result.extend_from_slice(&constant_to_bytes(elem));
            }
            result
        }
        Constant::GlobalRef(_name) => {
            // Global references are resolved by the linker via relocations.
            // Emit a placeholder (zero bytes) — the linker patches this.
            vec![0u8; 8]
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

/// Writes a complete relocatable ELF object file (`.o`) from the compiled
/// code sections, global data, and optional DWARF debug sections.
fn write_relocatable_object(
    text_data: &[u8],
    data_data: &[u8],
    rodata_data: &[u8],
    bss_size: u64,
    functions: &[CompiledFunction],
    globals: &[GlobalSymbolInfo],
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

    // --- .text section ---
    if !text_data.is_empty() {
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

    // --- DWARF debug sections (conditional on -g) ---
    if let Some(ref dwarf_data) = dwarf {
        add_dwarf_sections(&mut writer, dwarf_data);
    }

    // --- Symbol table entries ---

    // Add function symbols.
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
            visibility: STV_DEFAULT,
            section_index: 1, // .text section (index determined by writer)
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
            SectionPlacement::Data => 2,   // .data
            SectionPlacement::Rodata => 3, // .rodata
            SectionPlacement::Bss => 4,    // .bss
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
    for decl in module.declarations() {
        let sym = ElfSymbol {
            name: decl.name.clone(),
            value: 0,
            size: 0,
            binding: STB_GLOBAL,
            sym_type: STT_NOTYPE,
            visibility: STV_DEFAULT,
            section_index: 0, // SHN_UNDEF
        };
        writer.add_symbol(sym);
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

    // Parse the relocatable ELF object bytes back into linker internal
    // representation: sections, symbols, and relocations.
    let mut input = parse_elf_object_to_linker_input(0, "input.o", object_bytes);

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

    // Invoke the built-in linker with a single unified input.
    match link(&config, vec![input], handler.as_ref(), diagnostics) {
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
    // main is at offset 0 in .text (the first function).
    let main_offset = 0u64;

    match target {
        Target::X86_64 => {
            // _start stub (16 bytes):
            //   xor %ebp, %ebp          ; 31 ed
            //   call main               ; e8 <disp32>
            //   mov %eax, %edi           ; 89 c7
            //   mov $60, %eax            ; b8 3c 00 00 00
            //   syscall                  ; 0f 05
            let call_site = start_offset + 2; // offset of the E8 opcode
            let call_next = call_site + 5; // IP after the CALL
            let disp = (main_offset as i64) - (call_next as i64);
            let disp_bytes = (disp as i32).to_le_bytes();
            let stub = vec![
                0x31,
                0xed,
                0xe8,
                disp_bytes[0],
                disp_bytes[1],
                disp_bytes[2],
                disp_bytes[3],
                0x89,
                0xc7,
                0xb8,
                0x3c,
                0x00,
                0x00,
                0x00,
                0x0f,
                0x05,
            ];
            input.sections[text_idx].data.extend_from_slice(&stub);
        }
        Target::I686 => {
            let call_site = start_offset + 2;
            let call_next = call_site + 5;
            let disp = (main_offset as i64) - (call_next as i64);
            let disp_bytes = (disp as i32).to_le_bytes();
            let stub = vec![
                0x31,
                0xed,
                0xe8,
                disp_bytes[0],
                disp_bytes[1],
                disp_bytes[2],
                disp_bytes[3],
                0x89,
                0xc3,
                0xb8,
                0x01,
                0x00,
                0x00,
                0x00,
                0xcd,
                0x80,
            ];
            input.sections[text_idx].data.extend_from_slice(&stub);
        }
        Target::AArch64 => {
            // bl main: encoding is (imm26 << 0) | 0x94000000
            // imm26 = (main_offset - start_offset) / 4
            let bl_offset = start_offset;
            let offset_bytes = ((main_offset as i64) - (bl_offset as i64)) / 4;
            let imm26 = (offset_bytes as u32) & 0x03FF_FFFF;
            let bl_inst = 0x9400_0000u32 | imm26;
            let mov_x8_93 = 0xD280_0BA8u32; // mov x8, #93
            let svc = 0xD400_0001u32; // svc #0
            let mut stub = Vec::new();
            stub.extend_from_slice(&bl_inst.to_le_bytes());
            stub.extend_from_slice(&mov_x8_93.to_le_bytes());
            stub.extend_from_slice(&svc.to_le_bytes());
            input.sections[text_idx].data.extend_from_slice(&stub);
        }
        Target::RiscV64 => {
            // For RISC-V, use a JAL instruction to call main.
            // jal ra, offset  where offset = main_offset - start_offset
            let jal_offset = start_offset;
            let offset = (main_offset as i64) - (jal_offset as i64);
            // JAL encoding: imm[20|10:1|11|19:12] rd=1(ra) opcode=1101111
            let imm = offset as i32;
            let imm20 = ((imm >> 20) & 1) as u32;
            let imm10_1 = ((imm >> 1) & 0x3FF) as u32;
            let imm11 = ((imm >> 11) & 1) as u32;
            let imm19_12 = ((imm >> 12) & 0xFF) as u32;
            let jal = (imm20 << 31)
                | (imm10_1 << 21)
                | (imm11 << 20)
                | (imm19_12 << 12)
                | (1 << 7) // rd = ra (x1)
                | 0x6F; // opcode = JAL
                        // li a7, 93: addi a7, zero, 93
            let li_a7_93 = 0x05D0_0893u32;
            // ecall
            let ecall = 0x0000_0073u32;
            let mut stub = Vec::new();
            stub.extend_from_slice(&jal.to_le_bytes());
            stub.extend_from_slice(&li_a7_93.to_le_bytes());
            stub.extend_from_slice(&ecall.to_le_bytes());
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
        let mf = match arch.lower_function(func, diagnostics, module.globals()) {
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
                output.push_str(&format!("\t{}\n", inst));
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
