//! # AArch64 Backend
//!
//! Complete AArch64 (ARM 64-bit) code generation backend for BCC.
//! Implements the [`ArchCodegen`] trait for the A64 ISA with AAPCS64 ABI.
//!
//! ## Submodules
//! - `codegen` — Instruction selection (IR → AArch64 machine instructions)
//! - `registers` — Register file definitions (X0–X30, SP, V0–V31, NZCV)
//! - `abi` — AAPCS64 calling convention, HFA/HVA handling, stack frame layout
//! - `assembler` — Built-in AArch64 assembler (A64 instruction encoding, relocations)
//! - `linker` — Built-in AArch64 ELF linker (relocation application)
//!
//! ## Architecture Characteristics
//! - Fixed 32-bit instruction width (no variable-length encoding)
//! - Load/store architecture — ALU ops on registers only
//! - 31 general-purpose registers: X0–X30 (64-bit) / W0–W30 (32-bit)
//! - SP (stack pointer) and XZR (zero register) share encoding
//! - 32 SIMD/FP registers: V0–V31 (128-bit), with D/S/H/B views
//! - NZCV condition flags for conditional execution
//! - Little-endian byte order (default)
//! - ELF machine type: EM_AARCH64 (183)
//! - Page size: 4KB (default), 16KB, or 64KB
//!
//! ## Key: ADRP+ADD for PIC addressing, STP/LDP for efficient stack ops.

// ============================================================================
// Submodule Declarations
// ============================================================================

pub mod abi;
pub mod assembler;
pub mod codegen;
pub mod linker;
pub mod registers;

// ============================================================================
// Crate-level Imports
// ============================================================================

use crate::backend::traits::{
    ArchCodegen, ArgLocation, MachineFunction, MachineInstruction,
    MachineOperand, RegisterInfo, RelocationTypeInfo,
};
use crate::backend::elf_writer_common::{
    ElfWriter, Section, ElfSymbol,
    ET_REL, SHT_PROGBITS, SHT_NOBITS, SHT_RELA,
    SHF_ALLOC, SHF_EXECINSTR, SHF_WRITE, SHF_INFO_LINK,
    STB_GLOBAL, STT_FUNC, STT_NOTYPE, STV_DEFAULT,
};
use crate::ir::function::IrFunction;
use crate::ir::module::{IrModule, Constant};
use crate::ir::types::IrType;
use crate::common::target::Target;
// CType and MachineType are used for ABI classification in submodules
// and kept available for future extensions.
#[allow(unused_imports)]
use crate::common::types::{CType, MachineType};
use crate::common::diagnostics::{DiagnosticEngine, Span};

// ============================================================================
// Public Re-exports
// ============================================================================

/// Re-export instruction selection types for external access.
pub use self::codegen::{A64Instruction, A64Opcode, AArch64InstructionSelector, CondCode};

/// Re-export register file definitions.
pub use self::registers::{AArch64RegisterInfo, RegClass};

/// Re-export ABI types.
pub use self::abi::{AArch64Abi, ArgClass, FrameLayout};

// ============================================================================
// AArch64 ELF Constants
// ============================================================================

/// ELF machine type for AArch64.
///
/// Value 183 (0xB7) as defined in the ELF specification for AArch64.
/// Written to the `e_machine` field of the ELF header.
pub const EM_AARCH64: u16 = 183;

/// ELF flags for standard AArch64 (no special flags).
///
/// Unlike RISC-V which encodes ISA extensions in flags, AArch64 ELF
/// files use flags = 0 for the standard configuration.
pub const ELF_FLAGS: u32 = 0;

/// Default base virtual address for AArch64 static executables (ET_EXEC).
///
/// Position-independent executables (PIE) and shared libraries use a
/// base address of 0x0; the dynamic linker handles ASLR placement.
pub const DEFAULT_BASE_ADDRESS: u64 = 0x400000;

/// Default page size for AArch64 (4 KiB).
///
/// AArch64 supports 4 KiB, 16 KiB, and 64 KiB page sizes, but 4 KiB
/// is the standard default for Linux. Used for ADRP page-relative
/// addressing calculations and ELF segment alignment.
pub const PAGE_SIZE: u64 = 4096;

// ============================================================================
// STP/LDP Opcode Constants for Prologue/Epilogue
// ============================================================================

/// Opcode identifier for STP (Store Pair) — used in prologue generation.
const OPCODE_STP: u32 = 0xA9_00_00_00;

/// Opcode identifier for LDP (Load Pair) — used in epilogue generation.
const OPCODE_LDP: u32 = 0xA9_40_00_00;

/// Opcode identifier for MOV (register) — used for setting frame pointer.
const OPCODE_MOV_REG: u32 = 0xAA_00_00_00;

/// Opcode identifier for RET — return from subroutine.
const OPCODE_RET: u32 = 0xD6_5F_00_00;

/// Opcode identifier for SUB immediate — stack pointer adjustment.
const OPCODE_SUB_IMM: u32 = 0xD1_00_00_00;

/// Opcode identifier for ADD immediate — stack pointer restore.
const OPCODE_ADD_IMM: u32 = 0x91_00_00_00;

/// Opcode identifier for STP (FP pair) — callee-saved FP regs.
const OPCODE_STP_FP: u32 = 0xAD_00_00_00;

/// Opcode identifier for LDP (FP pair) — callee-saved FP reg restore.
const OPCODE_LDP_FP: u32 = 0xAD_40_00_00;

// ============================================================================
// AArch64Codegen — Primary Backend Struct
// ============================================================================

/// AArch64 code generation backend.
///
/// Implements the [`ArchCodegen`] trait, providing AArch64 instruction selection,
/// register allocation interface, prologue/epilogue generation, and machine code
/// emission. This struct is the single entry point for all AArch64-specific code
/// generation, instantiated by `crate::backend::generation` for `Target::AArch64`.
///
/// # Architecture Summary
///
/// - **ISA**: A64 (AArch64) — fixed 32-bit instruction width
/// - **ABI**: AAPCS64 — X0–X7 for integer args, V0–V7 for FP args
/// - **Registers**: 31 GPRs (X0–X30) + SP + ZR, 32 SIMD/FP (V0–V31)
/// - **Frame**: STP/LDP for paired register saves, FP = X29, LR = X30
/// - **PIC**: ADRP+ADD for PC-relative addressing, ADRP+LDR for GOT access
pub struct AArch64Codegen {
    /// Target architecture information — always `Target::AArch64`.
    target: Target,
    /// Register information provider for the AArch64 register file.
    reg_info: AArch64RegisterInfo,
    /// ABI handler for AAPCS64 calling convention classification.
    abi: AArch64Abi,
    /// Position-independent code mode (`-fPIC`).
    pic_mode: bool,
    /// Debug information generation (`-g` flag).
    /// Used by `compile_to_object` to conditionally emit DWARF sections.
    #[allow(dead_code)]
    debug_info: bool,
    /// Relocation type descriptors for AArch64 ELF relocations.
    relocation_types: Vec<RelocationTypeInfo>,
}

// ============================================================================
// AArch64Codegen — Constructor and Helpers
// ============================================================================

impl AArch64Codegen {
    /// Create a new AArch64 code generation backend.
    ///
    /// # Arguments
    ///
    /// * `pic_mode` — If `true`, generate position-independent code suitable
    ///   for shared libraries. Uses GOT-relative addressing for global symbols.
    /// * `debug_info` — If `true`, emit DWARF v4 debug sections.
    pub fn new(pic_mode: bool, debug_info: bool) -> Self {
        Self {
            target: Target::AArch64,
            reg_info: AArch64RegisterInfo::new(),
            abi: AArch64Abi::new(),
            pic_mode,
            debug_info,
            relocation_types: Self::build_relocation_types(),
        }
    }

    /// Build the complete list of AArch64 ELF relocation type descriptors.
    fn build_relocation_types() -> Vec<RelocationTypeInfo> {
        vec![
            RelocationTypeInfo::new("R_AARCH64_NONE", 0, 0, false),
            // Absolute data relocations
            RelocationTypeInfo::new("R_AARCH64_ABS64", 257, 8, false),
            RelocationTypeInfo::new("R_AARCH64_ABS32", 258, 4, false),
            RelocationTypeInfo::new("R_AARCH64_ABS16", 259, 2, false),
            // PC-relative data relocations
            RelocationTypeInfo::new("R_AARCH64_PREL64", 260, 8, true),
            RelocationTypeInfo::new("R_AARCH64_PREL32", 261, 4, true),
            RelocationTypeInfo::new("R_AARCH64_PREL16", 262, 2, true),
            // ADR/ADRP page-relative relocations
            RelocationTypeInfo::new("R_AARCH64_ADR_PREL_LO21", 274, 4, true),
            RelocationTypeInfo::new("R_AARCH64_ADR_PREL_PG_HI21", 275, 4, true),
            // ADD/LDR :lo12: relocations
            RelocationTypeInfo::new("R_AARCH64_ADD_ABS_LO12_NC", 277, 4, false),
            RelocationTypeInfo::new("R_AARCH64_LDST8_ABS_LO12_NC", 278, 4, false),
            // TBZ/TBNZ relocation
            RelocationTypeInfo::new("R_AARCH64_TSTBR14", 279, 4, true),
            // B.cond relocation
            RelocationTypeInfo::new("R_AARCH64_CONDBR19", 280, 4, true),
            // B/BL relocations
            RelocationTypeInfo::new("R_AARCH64_JUMP26", 282, 4, true),
            RelocationTypeInfo::new("R_AARCH64_CALL26", 283, 4, true),
            // Load/Store :lo12: relocations (scaled)
            RelocationTypeInfo::new("R_AARCH64_LDST16_ABS_LO12_NC", 284, 4, false),
            RelocationTypeInfo::new("R_AARCH64_LDST32_ABS_LO12_NC", 285, 4, false),
            RelocationTypeInfo::new("R_AARCH64_LDST64_ABS_LO12_NC", 286, 4, false),
            RelocationTypeInfo::new("R_AARCH64_LDST128_ABS_LO12_NC", 299, 4, false),
            // GOT-relative relocations (PIC)
            RelocationTypeInfo::new("R_AARCH64_ADR_GOT_PAGE", 311, 4, true),
            RelocationTypeInfo::new("R_AARCH64_LD64_GOT_LO12_NC", 312, 4, false),
            // TLS relocations
            RelocationTypeInfo::new("R_AARCH64_TLSLE_ADD_TPREL_HI12", 549, 4, false),
            RelocationTypeInfo::new("R_AARCH64_TLSLE_ADD_TPREL_LO12_NC", 551, 4, false),
            // Dynamic relocations
            RelocationTypeInfo::new("R_AARCH64_COPY", 1024, 8, false),
            RelocationTypeInfo::new("R_AARCH64_GLOB_DAT", 1025, 8, false),
            RelocationTypeInfo::new("R_AARCH64_JUMP_SLOT", 1026, 8, false),
            RelocationTypeInfo::new("R_AARCH64_RELATIVE", 1027, 8, false),
            // TLS dynamic relocations
            RelocationTypeInfo::new("R_AARCH64_TLS_DTPMOD64", 1028, 8, false),
            RelocationTypeInfo::new("R_AARCH64_TLS_DTPREL64", 1029, 8, false),
            RelocationTypeInfo::new("R_AARCH64_TLS_TPREL64", 1030, 8, false),
            RelocationTypeInfo::new("R_AARCH64_TLSDESC", 1031, 8, false),
        ]
    }

    /// Convert a list of AArch64-specific `A64Instruction`s into
    /// architecture-agnostic `MachineInstruction`s.
    fn convert_instructions(
        a64_instructions: &[A64Instruction],
    ) -> Vec<MachineInstruction> {
        let mut result = Vec::with_capacity(a64_instructions.len());
        for inst in a64_instructions {
            let mut mi = MachineInstruction::new(inst.opcode as u32);

            // Set operand size based on 32-bit vs 64-bit forms.
            mi.operand_size = if inst.is_32bit { 4 } else { 8 };

            // Map destination register as the result operand.
            if let Some(rd) = inst.rd {
                mi.result = Some(MachineOperand::Register(rd as u16));
            }

            // Map source registers as input operands.
            if let Some(rn) = inst.rn {
                mi.operands.push(MachineOperand::Register(rn as u16));
            }
            if let Some(rm) = inst.rm {
                mi.operands.push(MachineOperand::Register(rm as u16));
            }
            if let Some(ra) = inst.ra {
                mi.operands.push(MachineOperand::Register(ra as u16));
            }

            // Map immediate values.
            if inst.imm != 0
                || matches!(
                    inst.opcode,
                    A64Opcode::MOVZ | A64Opcode::MOVK | A64Opcode::MOVN
                )
            {
                mi.operands.push(MachineOperand::Immediate(inst.imm));
            }

            // Map symbol references.
            if let Some(ref sym) = inst.symbol {
                mi.operands.push(MachineOperand::GlobalSymbol(sym.clone()));
            }

            // Classify control-flow properties.
            match inst.opcode {
                A64Opcode::B | A64Opcode::BR => {
                    mi.is_terminator = true;
                    mi.is_branch = true;
                }
                A64Opcode::B_cond
                | A64Opcode::CBZ
                | A64Opcode::CBNZ
                | A64Opcode::TBZ
                | A64Opcode::TBNZ => {
                    mi.is_branch = true;
                }
                A64Opcode::BL | A64Opcode::BLR => {
                    mi.is_call = true;
                }
                A64Opcode::RET => {
                    mi.is_terminator = true;
                }
                _ => {}
            }

            result.push(mi);
        }
        result
    }

    /// Create a `MachineInstruction` for a paired store (STP) operation.
    fn make_stp(rt1: u16, rt2: u16, base: u16, offset: i64, is_fp: bool) -> MachineInstruction {
        let opcode = if is_fp { OPCODE_STP_FP } else { OPCODE_STP };
        let mut mi = MachineInstruction::new(opcode);
        mi.operands.push(MachineOperand::Register(rt1));
        mi.operands.push(MachineOperand::Register(rt2));
        mi.operands.push(MachineOperand::Memory {
            base: Some(base),
            index: None,
            scale: 1,
            displacement: offset,
        });
        mi.operand_size = 8;
        mi
    }

    /// Create a `MachineInstruction` for a paired load (LDP) operation.
    fn make_ldp(rt1: u16, rt2: u16, base: u16, offset: i64, is_fp: bool) -> MachineInstruction {
        let opcode = if is_fp { OPCODE_LDP_FP } else { OPCODE_LDP };
        let mut mi = MachineInstruction::new(opcode);
        mi.result = Some(MachineOperand::Register(rt1));
        mi.operands.push(MachineOperand::Register(rt2));
        mi.operands.push(MachineOperand::Memory {
            base: Some(base),
            index: None,
            scale: 1,
            displacement: offset,
        });
        mi.operand_size = 8;
        mi
    }

    /// Create a `MachineInstruction` for a MOV register operation.
    fn make_mov_reg(rd: u16, rn: u16) -> MachineInstruction {
        let mut mi = MachineInstruction::new(OPCODE_MOV_REG);
        mi.result = Some(MachineOperand::Register(rd));
        mi.operands.push(MachineOperand::Register(rn));
        mi.operand_size = 8;
        mi
    }

    /// Create a `MachineInstruction` for RET.
    fn make_ret() -> MachineInstruction {
        let mut mi = MachineInstruction::new(OPCODE_RET);
        mi.operands
            .push(MachineOperand::Register(registers::LR as u16));
        mi.is_terminator = true;
        mi.operand_size = 8;
        mi
    }

    /// Align a value up to the given alignment.
    #[inline]
    fn align_up(value: usize, align: usize) -> usize {
        if align == 0 {
            return value;
        }
        (value + align - 1) & !(align - 1)
    }

    /// Serialize an IR [`Constant`] to raw little-endian bytes.
    ///
    /// Used during object file generation to produce initialized data
    /// for `.data` and `.rodata` sections.
    #[allow(clippy::only_used_in_recursion)]
    fn serialize_constant(constant: &Constant, target: &Target) -> Vec<u8> {
        match constant {
            Constant::Integer(val) => {
                // Default integer serialization: 8 bytes (i64-width).
                (*val as i64).to_le_bytes().to_vec()
            }
            Constant::Float(val) => val.to_le_bytes().to_vec(),
            Constant::LongDouble(bytes) => bytes.to_vec(),
            Constant::String(bytes) => bytes.clone(),
            Constant::ZeroInit => Vec::new(),
            Constant::Struct(fields) => {
                let mut data = Vec::new();
                for field in fields {
                    data.extend(Self::serialize_constant(field, target));
                }
                data
            }
            Constant::Array(elems) => {
                let mut data = Vec::new();
                for elem in elems {
                    data.extend(Self::serialize_constant(elem, target));
                }
                data
            }
            Constant::GlobalRef(_name) => {
                // Emit a placeholder 8-byte address; a relocation will patch it.
                vec![0u8; 8]
            }
            Constant::Null => vec![0u8; 8],
            Constant::Undefined => Vec::new(),
        }
    }

    /// Compile an IR module to an AArch64 ELF relocatable object file (.o).
    ///
    /// Orchestrates the complete code generation pipeline:
    /// 1. Lower each IR function to AArch64 machine instructions.
    /// 2. Assemble machine instructions into binary code.
    /// 3. Build the ELF relocatable object with sections and symbols.
    /// 4. Optionally emit DWARF debug sections.
    ///
    /// # Arguments
    ///
    /// * `module` — The IR module containing functions, globals, and strings.
    /// * `diagnostics` — Diagnostic engine for error/warning reporting.
    ///
    /// # Returns
    ///
    /// `Ok(Vec<u8>)` containing the complete ELF object bytes on success,
    /// or `Err(())` if code generation fails.
    #[allow(clippy::result_unit_err)]
    pub fn compile_to_object(
        &self,
        module: &IrModule,
        diagnostics: &mut DiagnosticEngine,
    ) -> Result<Vec<u8>, ()> {
        let mut asm = assembler::AArch64Assembler::new(self.pic_mode);
        let mut all_functions: Vec<(String, Vec<A64Instruction>)> = Vec::new();

        // Phase 1: Lower each function definition to AArch64 instructions.
        for func in module.functions() {
            if !func.is_definition {
                continue;
            }

            let mut selector = codegen::AArch64InstructionSelector::new(self.target, self.pic_mode);
            let a64_instructions = selector.select_function(func, &self.abi);
            all_functions.push((func.name.clone(), a64_instructions));
        }

        // Phase 2: Assemble all functions into machine code.
        let asm_result = match asm.assemble_module(&all_functions) {
            Ok(result) => result,
            Err(e) => {
                diagnostics.emit_error(
                    Span::dummy(),
                    format!("AArch64 assembly failed: {}", e),
                );
                return Err(());
            }
        };

        // Phase 3: Build ELF relocatable object.
        let mut elf = ElfWriter::new(self.target, ET_REL);

        // Add .text section.
        let text_section = Section {
            name: ".text".to_string(),
            sh_type: SHT_PROGBITS,
            sh_flags: SHF_ALLOC | SHF_EXECINSTR,
            data: asm_result.code.clone(),
            sh_link: 0,
            sh_info: 0,
            sh_addralign: 4,
            sh_entsize: 0,
            logical_size: 0,
        };
        let text_idx = elf.add_section(text_section);

        // Collect .data bytes from non-BSS, non-const global definitions.
        let mut data_bytes = Vec::new();
        for global in module.globals() {
            if !global.is_definition {
                continue;
            }
            let is_bss = matches!(&global.initializer, Some(Constant::ZeroInit) | None);
            if is_bss || global.is_constant {
                continue;
            }
            if let Some(ref init) = global.initializer {
                let align = global.alignment.unwrap_or(
                    global.ty.align_bytes(&self.target),
                );
                let padding = Self::align_up(data_bytes.len(), align) - data_bytes.len();
                data_bytes.extend(std::iter::repeat(0u8).take(padding));
                data_bytes.extend(Self::serialize_constant(init, &self.target));
            }
        }
        if !data_bytes.is_empty() {
            let data_section = Section {
                name: ".data".to_string(),
                sh_type: SHT_PROGBITS,
                sh_flags: SHF_ALLOC | SHF_WRITE,
                data: data_bytes,
                sh_link: 0,
                sh_info: 0,
                sh_addralign: 8,
                sh_entsize: 0,
                logical_size: 0,
            };
            elf.add_section(data_section);
        }

        // Collect .rodata bytes from string pool and constant globals.
        let mut rodata_bytes = Vec::new();
        for string_lit in module.string_pool() {
            let padding = Self::align_up(rodata_bytes.len(), 1) - rodata_bytes.len();
            rodata_bytes.extend(std::iter::repeat(0u8).take(padding));
            rodata_bytes.extend_from_slice(&string_lit.bytes);
        }
        for global in module.globals() {
            if !global.is_definition || !global.is_constant {
                continue;
            }
            let is_bss = matches!(&global.initializer, Some(Constant::ZeroInit) | None);
            if is_bss {
                continue;
            }
            if let Some(ref init) = global.initializer {
                let align = global.alignment.unwrap_or(
                    global.ty.align_bytes(&self.target),
                );
                let padding = Self::align_up(rodata_bytes.len(), align) - rodata_bytes.len();
                rodata_bytes.extend(std::iter::repeat(0u8).take(padding));
                rodata_bytes.extend(Self::serialize_constant(init, &self.target));
            }
        }
        if !rodata_bytes.is_empty() {
            let rodata_section = Section {
                name: ".rodata".to_string(),
                sh_type: SHT_PROGBITS,
                sh_flags: SHF_ALLOC,
                data: rodata_bytes,
                sh_link: 0,
                sh_info: 0,
                sh_addralign: 8,
                sh_entsize: 0,
                logical_size: 0,
            };
            elf.add_section(rodata_section);
        }

        // Add .bss section for zero-initialized/undefined globals.
        let mut bss_size: u64 = 0;
        for global in module.globals() {
            if !global.is_definition {
                continue;
            }
            let is_bss = matches!(&global.initializer, Some(Constant::ZeroInit) | None);
            if !is_bss {
                continue;
            }
            let align = global.alignment.unwrap_or(
                global.ty.align_bytes(&self.target),
            ) as u64;
            bss_size = (bss_size + align - 1) & !(align - 1);
            bss_size += global.ty.size_bytes(&self.target) as u64;
        }
        if bss_size > 0 {
            let bss_section = Section {
                name: ".bss".to_string(),
                sh_type: SHT_NOBITS,
                sh_flags: SHF_ALLOC | SHF_WRITE,
                data: Vec::new(),
                sh_link: 0,
                sh_info: 0,
                sh_addralign: 8,
                sh_entsize: 0,
                logical_size: bss_size,
            };
            elf.add_section(bss_section);
        }

        // Add function symbols.
        for (name, _) in &all_functions {
            let offset = asm_result.symbols.get(name.as_str()).copied().unwrap_or(0);
            elf.add_symbol(ElfSymbol {
                name: name.clone(),
                value: offset,
                size: 0,
                binding: STB_GLOBAL,
                sym_type: STT_FUNC,
                visibility: STV_DEFAULT,
                section_index: text_idx as u16,
            });
        }

        // Add external function declaration symbols (undefined references).
        for decl in module.declarations() {
            elf.add_symbol(ElfSymbol {
                name: decl.name.clone(),
                value: 0,
                size: 0,
                binding: STB_GLOBAL,
                sym_type: STT_NOTYPE,
                visibility: STV_DEFAULT,
                section_index: 0, // SHN_UNDEF
            });
        }

        // Add .rela.text section for relocations.
        if !asm_result.relocations.is_empty() {
            let mut rela_data = Vec::new();
            for reloc in &asm_result.relocations {
                // Elf64_Rela: offset(8) + info(8) + addend(8) = 24 bytes
                rela_data.extend_from_slice(&reloc.offset.to_le_bytes());
                let r_info: u64 = reloc.reloc_type as u64;
                rela_data.extend_from_slice(&r_info.to_le_bytes());
                rela_data.extend_from_slice(&reloc.addend.to_le_bytes());
            }
            let rela_section = Section {
                name: ".rela.text".to_string(),
                sh_type: SHT_RELA,
                sh_flags: SHF_INFO_LINK,
                data: rela_data,
                sh_link: 0,
                sh_info: text_idx as u32,
                sh_addralign: 8,
                sh_entsize: 24,
                logical_size: 0,
            };
            elf.add_section(rela_section);
        }

        // Phase 4: Write the ELF object.
        let elf_bytes = elf.write();

        if diagnostics.has_errors() {
            return Err(());
        }

        Ok(elf_bytes)
    }
}

// ============================================================================
// ArchCodegen Trait Implementation
// ============================================================================

impl ArchCodegen for AArch64Codegen {
    /// Lower an IR function to AArch64 machine instructions.
    fn lower_function(
        &self,
        func: &IrFunction,
        _diag: &mut DiagnosticEngine,
    ) -> Result<MachineFunction, String> {
        if !func.is_definition {
            return Err(format!(
                "Cannot lower function '{}': not a definition",
                func.name
            ));
        }

        // Perform instruction selection: IR → A64 instructions.
        let mut selector = codegen::AArch64InstructionSelector::new(self.target, self.pic_mode);
        let a64_instructions = selector.select_function(func, &self.abi);

        // Convert to common MachineInstructions.
        let machine_instructions = Self::convert_instructions(&a64_instructions);

        // Build the MachineFunction with a single entry block.
        let mut mf = MachineFunction::new(func.name.clone());

        // Track whether we have calls to set is_leaf.
        let has_calls = machine_instructions.iter().any(|mi| mi.is_call);
        if has_calls {
            mf.mark_has_calls();
        }

        // Populate the entry block.
        let entry = mf.entry_block_mut();
        for inst in machine_instructions {
            entry.push_instruction(inst);
        }

        mf.frame_size = 0;

        Ok(mf)
    }

    /// Encode a machine function's instructions to raw AArch64 binary bytes.
    fn emit_assembly(&self, mf: &MachineFunction) -> Result<Vec<u8>, String> {
        let mut asm = assembler::AArch64Assembler::new(self.pic_mode);

        // Collect all instructions from all basic blocks and convert
        // MachineInstruction back to A64Instruction for the assembler.
        let mut all_instructions: Vec<A64Instruction> = Vec::new();
        for block in &mf.blocks {
            for mi in &block.instructions {
                let a64_inst = Self::mi_to_a64(mi);
                all_instructions.push(a64_inst);
            }
        }

        let result = asm.assemble_function(&all_instructions)?;
        Ok(result.code)
    }

    /// Returns the target architecture: `Target::AArch64`.
    #[inline]
    fn target(&self) -> Target {
        Target::AArch64
    }

    /// Returns the AArch64 register file information.
    fn register_info(&self) -> RegisterInfo {
        self.reg_info.register_info()
    }

    /// Returns AArch64 relocation type descriptors.
    fn relocation_types(&self) -> &[RelocationTypeInfo] {
        &self.relocation_types
    }

    /// Generate AArch64 function prologue instructions.
    ///
    /// ## AArch64 Prologue Pattern
    ///
    /// ```text
    /// stp x29, x30, [sp, #-frame_size]!  // Save FP+LR, pre-decrement SP
    /// mov x29, sp                          // Set frame pointer
    /// stp x19, x20, [sp, #offset]         // Save callee-saved GPR pair
    /// stp d8, d9, [sp, #fp_offset]         // Save callee-saved FP pair
    /// ```
    fn emit_prologue(&self, mf: &MachineFunction) -> Vec<MachineInstruction> {
        let mut prologue = Vec::new();
        let sp = registers::SP_REG as u16;
        let fp = registers::FP_REG as u16;
        let lr = registers::LR as u16;

        let frame_size = Self::align_up(mf.frame_size.max(16), 16);

        // Step 1: STP X29, X30, [SP, #-frame_size]! (pre-decrement)
        prologue.push(Self::make_stp(fp, lr, sp, -(frame_size as i64), false));

        // Step 2: MOV X29, SP (establish frame pointer)
        prologue.push(Self::make_mov_reg(fp, sp));

        // Step 3: Save callee-saved registers in pairs.
        let mut gpr_saved: Vec<u16> = Vec::new();
        let mut fpr_saved: Vec<u16> = Vec::new();

        for &reg in &mf.callee_saved_regs {
            if reg < 32 {
                gpr_saved.push(reg);
            } else {
                fpr_saved.push(reg);
            }
        }

        // Save GPRs in pairs.
        let mut offset = 16i64;
        let mut i = 0;
        while i + 1 < gpr_saved.len() {
            prologue.push(Self::make_stp(gpr_saved[i], gpr_saved[i + 1], sp, offset, false));
            offset += 16;
            i += 2;
        }
        if i < gpr_saved.len() {
            prologue.push(Self::make_stp(gpr_saved[i], gpr_saved[i], sp, offset, false));
            offset += 16;
        }

        // Save FPRs in pairs.
        let mut j = 0;
        while j + 1 < fpr_saved.len() {
            prologue.push(Self::make_stp(fpr_saved[j], fpr_saved[j + 1], sp, offset, true));
            offset += 16;
            j += 2;
        }
        if j < fpr_saved.len() {
            prologue.push(Self::make_stp(fpr_saved[j], fpr_saved[j], sp, offset, true));
        }

        prologue
    }

    /// Generate AArch64 function epilogue instructions.
    ///
    /// Restores callee-saved registers in reverse order, then returns.
    fn emit_epilogue(&self, mf: &MachineFunction) -> Vec<MachineInstruction> {
        let mut epilogue = Vec::new();
        let sp = registers::SP_REG as u16;
        let fp = registers::FP_REG as u16;
        let lr = registers::LR as u16;

        let frame_size = Self::align_up(mf.frame_size.max(16), 16);

        // Classify callee-saved registers.
        let mut gpr_saved: Vec<u16> = Vec::new();
        let mut fpr_saved: Vec<u16> = Vec::new();
        for &reg in &mf.callee_saved_regs {
            if reg < 32 {
                gpr_saved.push(reg);
            } else {
                fpr_saved.push(reg);
            }
        }

        // Compute offsets matching prologue layout.
        let mut offset = 16i64;
        let mut gpr_offsets: Vec<(usize, i64)> = Vec::new();
        let mut i = 0;
        while i + 1 < gpr_saved.len() {
            gpr_offsets.push((i, offset));
            offset += 16;
            i += 2;
        }
        if i < gpr_saved.len() {
            gpr_offsets.push((i, offset));
            offset += 16;
        }

        let mut fpr_offsets: Vec<(usize, i64)> = Vec::new();
        let mut j = 0;
        while j + 1 < fpr_saved.len() {
            fpr_offsets.push((j, offset));
            offset += 16;
            j += 2;
        }
        if j < fpr_saved.len() {
            fpr_offsets.push((j, offset));
        }

        // Restore FPRs in reverse order.
        for &(idx, off) in fpr_offsets.iter().rev() {
            if idx + 1 < fpr_saved.len() {
                epilogue.push(Self::make_ldp(
                    fpr_saved[idx],
                    fpr_saved[idx + 1],
                    sp,
                    off,
                    true,
                ));
            } else {
                epilogue.push(Self::make_ldp(fpr_saved[idx], fpr_saved[idx], sp, off, true));
            }
        }

        // Restore GPRs in reverse order.
        for &(idx, off) in gpr_offsets.iter().rev() {
            if idx + 1 < gpr_saved.len() {
                epilogue.push(Self::make_ldp(
                    gpr_saved[idx],
                    gpr_saved[idx + 1],
                    sp,
                    off,
                    false,
                ));
            } else {
                epilogue.push(Self::make_ldp(
                    gpr_saved[idx],
                    gpr_saved[idx],
                    sp,
                    off,
                    false,
                ));
            }
        }

        // Restore FP and LR, post-increment SP.
        epilogue.push(Self::make_ldp(fp, lr, sp, frame_size as i64, false));

        // Return via X30 (LR).
        epilogue.push(Self::make_ret());

        epilogue
    }

    /// Returns the physical register index of the frame pointer (X29).
    #[inline]
    fn frame_pointer_reg(&self) -> u16 {
        registers::FP_REG as u16
    }

    /// Returns the physical register index of the stack pointer (SP = 31).
    #[inline]
    fn stack_pointer_reg(&self) -> u16 {
        registers::SP_REG as u16
    }

    /// Returns the physical register index of the return address register (X30/LR).
    #[inline]
    fn return_address_reg(&self) -> Option<u16> {
        Some(registers::LR as u16)
    }

    /// Classify where a function argument should be placed per AAPCS64.
    fn classify_argument(&self, ty: &IrType) -> ArgLocation {
        match ty {
            IrType::F32 | IrType::F64 | IrType::F80 => {
                ArgLocation::Register(registers::V0 as u16)
            }
            IrType::Struct(_fields) => {
                let size = ty.size_bytes(&self.target);
                if size <= 8 {
                    ArgLocation::Register(registers::X0 as u16)
                } else if size <= 16 {
                    ArgLocation::RegisterPair(registers::X0 as u16, registers::X1 as u16)
                } else {
                    ArgLocation::Stack(0)
                }
            }
            IrType::Array(_, _) => ArgLocation::Register(registers::X0 as u16),
            IrType::Void => ArgLocation::Stack(0),
            _ => ArgLocation::Register(registers::X0 as u16),
        }
    }

    /// Classify where a function return value should be placed per AAPCS64.
    fn classify_return(&self, ty: &IrType) -> ArgLocation {
        match ty {
            IrType::Void => ArgLocation::Stack(0),
            IrType::F32 | IrType::F64 | IrType::F80 => {
                ArgLocation::Register(registers::V0 as u16)
            }
            IrType::Struct(_fields) => {
                let size = ty.size_bytes(&self.target);
                if size <= 8 {
                    ArgLocation::Register(registers::X0 as u16)
                } else if size <= 16 {
                    ArgLocation::RegisterPair(registers::X0 as u16, registers::X1 as u16)
                } else {
                    // Large return: indirect via X8.
                    ArgLocation::Register(registers::X8 as u16)
                }
            }
            _ => ArgLocation::Register(registers::X0 as u16),
        }
    }
}

// ============================================================================
// MachineInstruction ↔ A64Instruction Conversion
// ============================================================================

impl AArch64Codegen {
    /// Convert a `MachineInstruction` back to an `A64Instruction` for
    /// the assembler.
    fn mi_to_a64(mi: &MachineInstruction) -> A64Instruction {
        let mut inst = A64Instruction::new(A64Opcode::NOP);

        // Map the opcode back to an A64Opcode.
        inst.opcode = Self::u32_to_a64_opcode(mi.opcode);

        // Map result as destination register.
        if let Some(ref result) = mi.result {
            if let Some(reg) = result.as_register() {
                inst.rd = Some(reg as u8);
            }
        }

        // Map input operands.
        let mut reg_idx = 0;
        for op in &mi.operands {
            match op {
                MachineOperand::Register(r) => {
                    match reg_idx {
                        0 => inst.rn = Some(*r as u8),
                        1 => inst.rm = Some(*r as u8),
                        2 => inst.ra = Some(*r as u8),
                        _ => {}
                    }
                    reg_idx += 1;
                }
                MachineOperand::Immediate(imm) => {
                    inst.imm = *imm;
                }
                MachineOperand::GlobalSymbol(ref name) => {
                    inst.symbol = Some(name.clone());
                }
                MachineOperand::Memory {
                    base, displacement, ..
                } => {
                    if let Some(b) = base {
                        if inst.rn.is_none() {
                            inst.rn = Some(*b as u8);
                        }
                    }
                    inst.imm = *displacement;
                }
                _ => {}
            }
        }

        // Set the 32-bit flag from operand_size.
        inst.is_32bit = mi.operand_size == 4;

        inst
    }

    /// Map a `u32` opcode back to an `A64Opcode`.
    fn u32_to_a64_opcode(opcode: u32) -> A64Opcode {
        match opcode {
            x if x == OPCODE_STP => A64Opcode::STP,
            x if x == OPCODE_LDP => A64Opcode::LDP,
            x if x == OPCODE_MOV_REG => A64Opcode::ORR_reg,
            x if x == OPCODE_RET => A64Opcode::RET,
            x if x == OPCODE_SUB_IMM => A64Opcode::SUB_imm,
            x if x == OPCODE_ADD_IMM => A64Opcode::ADD_imm,
            x if x == OPCODE_STP_FP => A64Opcode::STP_fp,
            x if x == OPCODE_LDP_FP => A64Opcode::LDP_fp,
            other => Self::discriminant_to_opcode(other),
        }
    }

    /// Map a discriminant value back to an A64Opcode via linear scan.
    fn discriminant_to_opcode(d: u32) -> A64Opcode {
        let opcodes: &[A64Opcode] = &[
            A64Opcode::ADD_imm,
            A64Opcode::ADDS_imm,
            A64Opcode::SUB_imm,
            A64Opcode::SUBS_imm,
            A64Opcode::MOVZ,
            A64Opcode::MOVK,
            A64Opcode::MOVN,
            A64Opcode::AND_imm,
            A64Opcode::ORR_imm,
            A64Opcode::EOR_imm,
            A64Opcode::ANDS_imm,
            A64Opcode::ADRP,
            A64Opcode::ADR,
            A64Opcode::ADD_reg,
            A64Opcode::ADDS_reg,
            A64Opcode::SUB_reg,
            A64Opcode::SUBS_reg,
            A64Opcode::AND_reg,
            A64Opcode::ORR_reg,
            A64Opcode::EOR_reg,
            A64Opcode::ANDS_reg,
            A64Opcode::ORN_reg,
            A64Opcode::BIC_reg,
            A64Opcode::MADD,
            A64Opcode::MSUB,
            A64Opcode::SMULH,
            A64Opcode::UMULH,
            A64Opcode::SDIV,
            A64Opcode::UDIV,
            A64Opcode::LSL_reg,
            A64Opcode::LSR_reg,
            A64Opcode::ASR_reg,
            A64Opcode::ROR_reg,
            A64Opcode::LSL_imm,
            A64Opcode::LSR_imm,
            A64Opcode::ASR_imm,
            A64Opcode::CSEL,
            A64Opcode::CSINC,
            A64Opcode::CSINV,
            A64Opcode::CSNEG,
            A64Opcode::CSET,
            A64Opcode::CSETM,
            A64Opcode::CLZ,
            A64Opcode::CLS,
            A64Opcode::RBIT,
            A64Opcode::REV,
            A64Opcode::REV16,
            A64Opcode::REV32,
            A64Opcode::EXTR,
            A64Opcode::BFM,
            A64Opcode::SBFM,
            A64Opcode::UBFM,
            A64Opcode::SXTB,
            A64Opcode::SXTH,
            A64Opcode::SXTW,
            A64Opcode::UXTB,
            A64Opcode::UXTH,
            A64Opcode::LDR_imm,
            A64Opcode::LDRB_imm,
            A64Opcode::LDRH_imm,
            A64Opcode::LDRSB_imm,
            A64Opcode::LDRSH_imm,
            A64Opcode::LDRSW_imm,
            A64Opcode::LDR_reg,
            A64Opcode::LDR_literal,
            A64Opcode::LDP,
            A64Opcode::LDPSW,
            A64Opcode::LDR_pre,
            A64Opcode::LDR_post,
            A64Opcode::STR_imm,
            A64Opcode::STRB_imm,
            A64Opcode::STRH_imm,
            A64Opcode::STR_reg,
            A64Opcode::STP,
            A64Opcode::STR_pre,
            A64Opcode::STR_post,
            A64Opcode::LDR_fp_imm,
            A64Opcode::STR_fp_imm,
            A64Opcode::LDP_fp,
            A64Opcode::STP_fp,
            A64Opcode::B,
            A64Opcode::BL,
            A64Opcode::B_cond,
            A64Opcode::BR,
            A64Opcode::BLR,
            A64Opcode::RET,
            A64Opcode::CBZ,
            A64Opcode::CBNZ,
            A64Opcode::TBZ,
            A64Opcode::TBNZ,
            A64Opcode::NOP,
        ];

        for &op in opcodes {
            if op as u32 == d {
                return op;
            }
        }

        // Defensive fallback for unrecognized opcodes.
        A64Opcode::NOP
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codegen_new() {
        let codegen = AArch64Codegen::new(false, false);
        assert_eq!(codegen.target(), Target::AArch64);
        assert!(!codegen.pic_mode);
        assert!(!codegen.debug_info);
    }

    #[test]
    fn test_codegen_new_pic() {
        let codegen = AArch64Codegen::new(true, true);
        assert!(codegen.pic_mode);
        assert!(codegen.debug_info);
        assert_eq!(codegen.target(), Target::AArch64);
    }

    #[test]
    fn test_frame_pointer_reg() {
        let codegen = AArch64Codegen::new(false, false);
        assert_eq!(codegen.frame_pointer_reg(), 29);
        assert_eq!(codegen.frame_pointer_reg(), registers::FP_REG as u16);
    }

    #[test]
    fn test_stack_pointer_reg() {
        let codegen = AArch64Codegen::new(false, false);
        assert_eq!(codegen.stack_pointer_reg(), 31);
        assert_eq!(codegen.stack_pointer_reg(), registers::SP_REG as u16);
    }

    #[test]
    fn test_return_address_reg() {
        let codegen = AArch64Codegen::new(false, false);
        assert_eq!(codegen.return_address_reg(), Some(30));
        assert_eq!(codegen.return_address_reg(), Some(registers::LR as u16));
    }

    #[test]
    fn test_register_info() {
        let codegen = AArch64Codegen::new(false, false);
        let info = codegen.register_info();
        assert_eq!(info.allocatable_gpr.len(), 29);
        assert_eq!(info.allocatable_fpr.len(), 32);
        assert_eq!(info.callee_saved.len(), 18);
        assert_eq!(info.argument_gpr.len(), 8);
        assert_eq!(info.argument_fpr.len(), 8);
        assert_eq!(info.return_gpr.len(), 2);
        assert_eq!(info.return_fpr.len(), 4);
    }

    #[test]
    fn test_relocation_types() {
        let codegen = AArch64Codegen::new(false, false);
        let relocs = codegen.relocation_types();
        assert!(relocs.len() >= 31);

        assert_eq!(relocs[0].name, "R_AARCH64_NONE");
        assert_eq!(relocs[0].type_id, 0);

        let abs64 = relocs.iter().find(|r| r.name == "R_AARCH64_ABS64");
        assert!(abs64.is_some());
        assert_eq!(abs64.unwrap().type_id, 257);
        assert_eq!(abs64.unwrap().size, 8);
        assert!(!abs64.unwrap().is_pc_relative);

        let call26 = relocs.iter().find(|r| r.name == "R_AARCH64_CALL26");
        assert!(call26.is_some());
        assert_eq!(call26.unwrap().type_id, 283);
        assert!(call26.unwrap().is_pc_relative);

        let adrp = relocs
            .iter()
            .find(|r| r.name == "R_AARCH64_ADR_PREL_PG_HI21");
        assert!(adrp.is_some());
        assert_eq!(adrp.unwrap().type_id, 275);

        let jslot = relocs.iter().find(|r| r.name == "R_AARCH64_JUMP_SLOT");
        assert!(jslot.is_some());
        assert_eq!(jslot.unwrap().type_id, 1026);
    }

    #[test]
    fn test_elf_constants() {
        assert_eq!(EM_AARCH64, 183);
        assert_eq!(ELF_FLAGS, 0);
        assert_eq!(DEFAULT_BASE_ADDRESS, 0x400000);
        assert_eq!(PAGE_SIZE, 4096);
    }

    #[test]
    fn test_emit_prologue_basic() {
        let codegen = AArch64Codegen::new(false, false);
        let mf = MachineFunction::new("test_func".to_string());
        let prologue = codegen.emit_prologue(&mf);
        // Minimum: STP (FP, LR) + MOV (FP, SP)
        assert!(
            prologue.len() >= 2,
            "Prologue should have at least 2 instructions"
        );
    }

    #[test]
    fn test_emit_epilogue_basic() {
        let codegen = AArch64Codegen::new(false, false);
        let mf = MachineFunction::new("test_func".to_string());
        let epilogue = codegen.emit_epilogue(&mf);
        // Minimum: LDP (FP, LR) + RET
        assert!(
            epilogue.len() >= 2,
            "Epilogue should have at least 2 instructions"
        );
        let last = epilogue.last().unwrap();
        assert!(
            last.is_terminator,
            "Last epilogue instruction should be a terminator (RET)"
        );
    }

    #[test]
    fn test_classify_argument_integer() {
        let codegen = AArch64Codegen::new(false, false);
        let loc = codegen.classify_argument(&IrType::I64);
        assert!(loc.is_register());
    }

    #[test]
    fn test_classify_argument_float() {
        let codegen = AArch64Codegen::new(false, false);
        let loc = codegen.classify_argument(&IrType::F64);
        assert!(loc.is_register());
    }

    #[test]
    fn test_classify_argument_pointer() {
        let codegen = AArch64Codegen::new(false, false);
        let loc = codegen.classify_argument(&IrType::Ptr);
        assert!(loc.is_register());
    }

    #[test]
    fn test_classify_return_void() {
        let codegen = AArch64Codegen::new(false, false);
        let loc = codegen.classify_return(&IrType::Void);
        assert!(loc.is_stack());
    }

    #[test]
    fn test_classify_return_integer() {
        let codegen = AArch64Codegen::new(false, false);
        let loc = codegen.classify_return(&IrType::I32);
        assert!(loc.is_register());
    }

    #[test]
    fn test_align_up() {
        assert_eq!(AArch64Codegen::align_up(0, 16), 0);
        assert_eq!(AArch64Codegen::align_up(1, 16), 16);
        assert_eq!(AArch64Codegen::align_up(15, 16), 16);
        assert_eq!(AArch64Codegen::align_up(16, 16), 16);
        assert_eq!(AArch64Codegen::align_up(17, 16), 32);
        assert_eq!(AArch64Codegen::align_up(0, 4), 0);
        assert_eq!(AArch64Codegen::align_up(3, 4), 4);
    }

    #[test]
    fn test_convert_instructions() {
        let a64_insts = vec![
            A64Instruction::new(A64Opcode::ADD_imm)
                .with_rd(0)
                .with_rn(1)
                .with_imm(42),
            A64Instruction::new(A64Opcode::RET),
        ];

        let mis = AArch64Codegen::convert_instructions(&a64_insts);
        assert_eq!(mis.len(), 2);
        assert!(mis[0].result.is_some());
        assert!(!mis[0].is_terminator);
        assert!(mis[1].is_terminator);
    }

    #[test]
    fn test_serialize_constant_integer() {
        let target = Target::AArch64;
        let bytes = AArch64Codegen::serialize_constant(&Constant::Integer(42), &target);
        assert_eq!(bytes.len(), 8);
        assert_eq!(i64::from_le_bytes(bytes[..8].try_into().unwrap()), 42);
    }

    #[test]
    fn test_serialize_constant_float() {
        let target = Target::AArch64;
        let bytes = AArch64Codegen::serialize_constant(&Constant::Float(3.14), &target);
        assert_eq!(bytes.len(), 8);
        let val = f64::from_le_bytes(bytes[..8].try_into().unwrap());
        assert!((val - 3.14).abs() < f64::EPSILON);
    }

    #[test]
    fn test_serialize_constant_string() {
        let target = Target::AArch64;
        let bytes = AArch64Codegen::serialize_constant(
            &Constant::String(b"Hello\0".to_vec()),
            &target,
        );
        assert_eq!(bytes, b"Hello\0");
    }

    #[test]
    fn test_serialize_constant_zero_init() {
        let target = Target::AArch64;
        let bytes = AArch64Codegen::serialize_constant(&Constant::ZeroInit, &target);
        assert!(bytes.is_empty());
    }

    #[test]
    fn test_serialize_constant_null() {
        let target = Target::AArch64;
        let bytes = AArch64Codegen::serialize_constant(&Constant::Null, &target);
        assert_eq!(bytes.len(), 8);
        assert!(bytes.iter().all(|&b| b == 0));
    }
}
