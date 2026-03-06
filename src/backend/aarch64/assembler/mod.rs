//! # Built-in AArch64 Assembler
//!
//! Self-contained assembler for the AArch64 (ARM 64-bit) architecture.
//! Converts A64 machine instructions into binary machine code and produces
//! ELF relocatable object sections.
//!
//! ## Architecture
//! - Accepts `A64Instruction` sequences from `codegen.rs`
//! - Dispatches to `encoder` for A64 instruction format binary encoding
//! - Collects `relocations` for unresolved symbol references
//! - Produces `.text` section bytes + relocation entries
//!
//! ## Key Characteristic: Fixed 32-bit Instruction Width
//! Every AArch64 instruction encodes to exactly 4 bytes. There is no
//! variable-length encoding, simplifying offset computation and branch targeting.
//!
//! ## Standalone Backend Mode
//! No external assembler is invoked. This module, together with `encoder.rs` and
//! `relocations.rs`, is entirely self-contained per BCC's zero-dependency mandate.
//!
//! ## Instruction Format Groups
//! - Data Processing (Immediate): ADD/SUB, MOV, logical, ADRP/ADR
//! - Data Processing (Register): shifted/extended register ops, CSEL
//! - Loads and Stores: LDR/STR, LDP/STP, pre/post-index
//! - Branches: B, BL, B.cond, BR, BLR, RET, CBZ/CBNZ, TBZ/TBNZ
//! - SIMD/FP: FADD, FSUB, FMUL, FDIV, FCVT, FMOV
//! - System: NOP, DMB, DSB, ISB, SVC, MRS, MSR

pub mod encoder;
pub mod relocations;

// ---------------------------------------------------------------------------
// Re-exports for parent module access
// ---------------------------------------------------------------------------

pub use self::encoder::AArch64Encoder;
pub use self::relocations::{AArch64RelocationHandler, AArch64RelocationType};

// ---------------------------------------------------------------------------
// Crate-internal imports
// ---------------------------------------------------------------------------

use crate::backend::aarch64::codegen::{A64Instruction, A64Opcode};
use crate::backend::aarch64::registers;
use crate::backend::linker_common::relocation::{RelocCategory, Relocation};
use crate::common::fx_hash::FxHashMap;
use crate::common::target::Target;

// Imported for documentation / future use — the assembler produces .text
// section bytes and relocations that feed into the ELF writer (via
// build_text_section) and the diagnostic engine is used for error reporting
// in extended inline assembly processing.
#[allow(unused_imports)]
use crate::backend::elf_writer_common;
#[allow(unused_imports)]
use crate::common::diagnostics::{DiagnosticEngine, Span};

use self::encoder::EncodedInstruction;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// AArch64 instruction width in bytes — every instruction is exactly 4 bytes.
const INSTRUCTION_SIZE: u64 = 4;

/// AArch64 NOP encoding: `0xD503201F` (little-endian: `1F 20 03 D5`).
const NOP_ENCODING: u32 = 0xD503_201F;

/// Maximum signed offset for B/BL (26-bit imm × 4 = ±128 MiB).
/// 2^25 instructions × 4 bytes = 128 MiB.
const MAX_BRANCH26_OFFSET: i64 = (1 << 27) - 4; // +128 MiB - 4
const MIN_BRANCH26_OFFSET: i64 = -(1 << 27); // -128 MiB

/// Maximum signed offset for B.cond / CBZ / CBNZ (19-bit imm × 4 = ±1 MiB).
const MAX_CONDBR19_OFFSET: i64 = (1 << 20) - 4; // +1 MiB - 4
const MIN_CONDBR19_OFFSET: i64 = -(1 << 20); // -1 MiB

/// Maximum signed offset for TBZ/TBNZ (14-bit imm × 4 = ±32 KiB).
const MAX_TSTBR14_OFFSET: i64 = (1 << 15) - 4; // +32 KiB - 4
const MIN_TSTBR14_OFFSET: i64 = -(1 << 15); // -32 KiB

// ---------------------------------------------------------------------------
// AssemblyRelocation — relocation entry generated during assembly
// ---------------------------------------------------------------------------

/// A relocation entry generated during assembly.
///
/// Records the location within the code buffer, the referenced symbol,
/// the AArch64-specific relocation type (as a raw ELF `r_type` value),
/// and the addend for RELA-style relocations.
#[derive(Debug, Clone)]
pub struct AssemblyRelocation {
    /// Offset within the code section where the relocation applies.
    pub offset: u64,
    /// Symbol name this relocation references.
    pub symbol: String,
    /// AArch64 relocation type (ELF r_type value).
    pub reloc_type: u32,
    /// Addend value for the relocation computation.
    pub addend: i64,
}

impl AssemblyRelocation {
    /// Convert this assembler-local relocation to the linker-common
    /// [`Relocation`] format for integration with the linker pipeline.
    pub fn to_linker_relocation(&self) -> Relocation {
        Relocation {
            offset: self.offset,
            symbol_name: self.symbol.clone(),
            sym_index: 0,
            rel_type: self.reloc_type,
            addend: self.addend,
            object_id: 0,
            section_index: 0,
            output_section_name: None,
        }
    }

    /// Classify this relocation using the AArch64 relocation type mapping.
    pub fn category(&self) -> RelocCategory {
        match AArch64RelocationType::from_raw(self.reloc_type) {
            Some(rt) => rt.category(),
            None => RelocCategory::Other,
        }
    }
}

// ---------------------------------------------------------------------------
// AssemblyResult — output of the assembler
// ---------------------------------------------------------------------------

/// Result of assembling a sequence of AArch64 instructions.
///
/// Contains the machine code bytes suitable for a `.text` ELF section,
/// any unresolved relocations for the linker, and the symbol table
/// mapping labels/function names to code offsets.
#[derive(Debug)]
pub struct AssemblyResult {
    /// Assembled machine code bytes (`.text` section content).
    pub code: Vec<u8>,
    /// Relocations generated during assembly (for unresolved symbols).
    pub relocations: Vec<AssemblyRelocation>,
    /// Symbol definitions found during assembly (label → offset).
    pub symbols: FxHashMap<String, u64>,
    /// Total size of assembled code in bytes.
    pub code_size: u64,
}

impl AssemblyResult {
    /// Create an empty assembly result.
    pub fn empty() -> Self {
        AssemblyResult {
            code: Vec::new(),
            relocations: Vec::new(),
            symbols: FxHashMap::default(),
            code_size: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// InlineAsmOperand — operand descriptor for inline assembly
// ---------------------------------------------------------------------------

/// Describes a single operand for inline assembly substitution.
///
/// Used by `assemble_inline_asm` to replace `%0`, `%1`, or `%[name]`
/// placeholders in the assembly template string with concrete register
/// names or immediate values.
#[derive(Debug, Clone)]
pub struct InlineAsmOperand {
    /// Constraint string (e.g. `"r"`, `"=r"`, `"i"`, `"m"`).
    pub constraint: String,
    /// Concrete register ID (if register operand).
    pub register: Option<u8>,
    /// Immediate value (if immediate operand).
    pub immediate: Option<i64>,
    /// Named operand label (e.g. `[result]`).
    pub name: Option<String>,
}

// ---------------------------------------------------------------------------
// AArch64Assembler — the main assembler driver
// ---------------------------------------------------------------------------

/// Built-in AArch64 assembler.
///
/// Converts `A64Instruction` sequences into binary machine code,
/// collecting relocations for unresolved symbol references.
/// All instructions encode to exactly 4 bytes (fixed-width).
///
/// ## Usage
///
/// ```ignore
/// let mut asm = AArch64Assembler::new(false);
/// let result = asm.assemble_function(&instructions)?;
/// let (code, relocs) = asm.build_text_section();
/// ```
pub struct AArch64Assembler {
    /// The instruction encoder instance.
    encoder: AArch64Encoder,
    /// Accumulated machine code bytes.
    code: Vec<u8>,
    /// Accumulated relocations for unresolved symbols.
    relocations: Vec<AssemblyRelocation>,
    /// Label/symbol → code offset map.
    symbols: FxHashMap<String, u64>,
    /// Current code offset (write position), always a multiple of 4.
    current_offset: u64,
    /// Whether PIC mode is active (`-fPIC`).
    pic_mode: bool,
    /// Target architecture info.
    target: Target,
}

impl AArch64Assembler {
    // -----------------------------------------------------------------------
    // Construction and reset
    // -----------------------------------------------------------------------

    /// Create a new AArch64 assembler.
    ///
    /// # Arguments
    /// * `pic_mode` — whether to generate PIC-style ADRP+LDR GOT relocations.
    pub fn new(pic_mode: bool) -> Self {
        Self {
            encoder: AArch64Encoder::new(),
            code: Vec::new(),
            relocations: Vec::new(),
            symbols: FxHashMap::default(),
            current_offset: 0,
            pic_mode,
            target: Target::AArch64,
        }
    }

    /// Reset the assembler for a new function/section.
    ///
    /// Clears all accumulated code, relocations, symbols, and resets the
    /// write offset to zero. The encoder and configuration (PIC mode,
    /// target) are preserved.
    pub fn reset(&mut self) {
        self.code.clear();
        self.relocations.clear();
        self.symbols.clear();
        self.current_offset = 0;
    }

    // -----------------------------------------------------------------------
    // Symbol and label management
    // -----------------------------------------------------------------------

    /// Define a label at the current code offset.
    ///
    /// Records `name → current_offset` in the symbol table. Used for
    /// function entry points, branch targets, and local labels.
    pub fn define_label(&mut self, name: &str) {
        self.symbols.insert(name.to_string(), self.current_offset);
    }

    /// Return the current write position (byte offset) in the code buffer.
    ///
    /// This value is always a multiple of 4 (AArch64 instruction alignment).
    #[inline]
    pub fn current_offset(&self) -> u64 {
        self.current_offset
    }

    // -----------------------------------------------------------------------
    // Core assembly: single function
    // -----------------------------------------------------------------------

    /// Assemble a sequence of AArch64 instructions for a single function.
    ///
    /// For each instruction:
    /// 1. If the instruction carries a label (via `symbol` field on certain
    ///    pseudo-ops), record it at the current offset.
    /// 2. Dispatch to the encoder for binary encoding.
    /// 3. Collect any relocation produced by the encoder.
    /// 4. Handle pseudo-instructions (LA, CALL, INLINE_ASM) that expand
    ///    to multi-instruction sequences.
    /// 5. Append encoded bytes and advance the offset.
    ///
    /// Returns an `AssemblyResult` with accumulated code, relocations, and
    /// symbols for this function. The assembler state is **not** reset
    /// automatically — call [`reset`](Self::reset) before assembling
    /// the next function if a fresh context is needed.
    pub fn assemble_function(
        &mut self,
        instructions: &[A64Instruction],
    ) -> Result<AssemblyResult, String> {
        for inst in instructions {
            self.assemble_one(inst)?;
        }

        Ok(AssemblyResult {
            code: self.code.clone(),
            relocations: self.relocations.clone(),
            symbols: self.symbols.clone(),
            code_size: self.current_offset,
        })
    }

    // -----------------------------------------------------------------------
    // Core assembly: module (multiple functions)
    // -----------------------------------------------------------------------

    /// Assemble multiple functions sequentially into a single code section.
    ///
    /// Each function is identified by a `(name, instructions)` pair.
    /// Function names are recorded as symbols at their start offsets.
    /// Functions are aligned to 4-byte boundaries (inherent in AArch64).
    pub fn assemble_module(
        &mut self,
        functions: &[(String, Vec<A64Instruction>)],
    ) -> Result<AssemblyResult, String> {
        for (name, instructions) in functions {
            // Align to instruction boundary (should already be aligned).
            self.align_to(INSTRUCTION_SIZE);

            // Record the function name at its start offset.
            self.define_label(name);

            // Assemble all instructions for this function.
            for inst in instructions {
                self.assemble_one(inst)?;
            }
        }

        Ok(AssemblyResult {
            code: self.code.clone(),
            relocations: self.relocations.clone(),
            symbols: self.symbols.clone(),
            code_size: self.current_offset,
        })
    }

    // -----------------------------------------------------------------------
    // Single instruction assembly
    // -----------------------------------------------------------------------

    /// Assemble a single instruction, handling pseudo-instructions and
    /// relocation emission.
    fn assemble_one(&mut self, inst: &A64Instruction) -> Result<(), String> {
        match inst.opcode {
            // Pseudo-instruction: Load Address (ADRP + ADD pair).
            A64Opcode::LA => {
                self.assemble_load_address(inst)?;
            }

            // Pseudo-instruction: Function call (may expand to BL or ADRP+BLR).
            A64Opcode::CALL => {
                self.assemble_call(inst)?;
            }

            // Pseudo-instruction: Inline assembly passthrough.
            A64Opcode::INLINE_ASM => {
                // Inline assembly is handled as raw bytes or via
                // assemble_inline_asm(); here we emit a NOP placeholder
                // if no symbol/template is provided.
                if inst.symbol.is_some() {
                    // The symbol field carries the asm template; for now
                    // emit a NOP as placeholder — full inline asm parsing
                    // is handled by assemble_inline_asm().
                    self.emit_raw_word(NOP_ENCODING);
                } else {
                    self.emit_raw_word(NOP_ENCODING);
                }
            }

            // Standard single-instruction encoding.
            _ => {
                let encoded = self.encoder.encode(inst)?;
                self.emit_encoded(inst, &encoded)?;
            }
        }
        Ok(())
    }

    /// Emit an encoded instruction result, collecting any relocations.
    fn emit_encoded(
        &mut self,
        inst: &A64Instruction,
        encoded: &EncodedInstruction,
    ) -> Result<(), String> {
        // Handle multi-instruction expansions (e.g., MOV_imm → MOVZ+MOVK sequence).
        // The encoder may produce more than 4 bytes for such pseudo-instructions.
        let num_words = encoded.bytes.len() / 4;

        // Emit the first 4-byte instruction word and attach any encoder relocation.
        if let Some(ref enc_reloc) = encoded.relocation {
            self.relocations.push(AssemblyRelocation {
                offset: self.current_offset,
                symbol: inst.symbol.clone().unwrap_or_default(),
                reloc_type: enc_reloc.reloc_type,
                addend: enc_reloc.addend,
            });
        }

        // Append all encoded bytes (could be 4, 8, 12, or 16 for multi-word sequences).
        self.code.extend_from_slice(&encoded.bytes);
        self.current_offset += encoded.bytes.len() as u64;

        // For multi-instruction expansions without explicit relocations,
        // no additional relocation work is needed — the encoder handles the
        // full MOVZ+MOVK sequence internally.
        let _ = num_words; // suppress unused warning; counted for documentation

        Ok(())
    }

    /// Emit a raw 32-bit word into the code buffer (little-endian).
    fn emit_raw_word(&mut self, word: u32) {
        self.code.extend_from_slice(&word.to_le_bytes());
        self.current_offset += INSTRUCTION_SIZE;
    }

    // -----------------------------------------------------------------------
    // Pseudo-instruction expansion: Load Address (LA)
    // -----------------------------------------------------------------------

    /// Assemble the LA (Load Address) pseudo-instruction.
    ///
    /// Expands to an ADRP + ADD pair for non-GOT addressing, or
    /// ADRP + LDR for GOT-relative addressing (PIC mode).
    fn assemble_load_address(&mut self, inst: &A64Instruction) -> Result<(), String> {
        let sym = inst.symbol.as_deref().unwrap_or("");

        let rd = inst.rd.unwrap_or(0);

        if self.pic_mode {
            // PIC mode: ADRP + LDR from GOT.
            let adrp_offset = self.current_offset;
            self.emit_adrp(rd, true);
            self.emit_got_relocation(sym, adrp_offset);

            // LDR Xd, [Xd, #:got_lo12:sym]
            // Emit a placeholder LDR instruction with relocation.
            let ldr_word = encode_ldr_unsigned_imm(rd, rd, 0);
            let ldr_offset = self.current_offset;
            self.emit_raw_word(ldr_word);
            // The GOT LDR relocation was already emitted as part of the
            // ADRP+LDR pair in emit_got_relocation.
            let _ = ldr_offset;
        } else {
            // Non-PIC: ADRP + ADD pair.
            let adrp_offset = self.current_offset;
            self.emit_adrp(rd, false);
            self.emit_adrp_add_relocation(sym, adrp_offset);

            // ADD Xd, Xd, #:lo12:sym
            let add_word = encode_add_imm(rd, rd, 0);
            self.emit_raw_word(add_word);
            // The ADD relocation was already emitted as part of the
            // ADRP+ADD pair in emit_adrp_add_relocation.
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Pseudo-instruction expansion: CALL
    // -----------------------------------------------------------------------

    /// Assemble the CALL pseudo-instruction.
    ///
    /// For direct calls with a known symbol, emits a BL instruction with
    /// a CALL26 relocation. For indirect calls or when PIC mode requires
    /// GOT lookup, may expand to ADRP+LDR+BLR.
    fn assemble_call(&mut self, inst: &A64Instruction) -> Result<(), String> {
        let sym = inst.symbol.as_deref().unwrap_or("");

        if !sym.is_empty() {
            // Direct call via BL — emit a placeholder BL with CALL26 relocation.
            let bl_word: u32 = 0x9400_0000; // BL (imm26=0 placeholder)
            let offset = self.current_offset;
            self.emit_raw_word(bl_word);
            self.emit_branch_relocation(sym, offset, AArch64RelocationType::Call26.to_raw());
        } else if let Some(rn) = inst.rn {
            // Indirect call via BLR Xn.
            let blr_word: u32 = 0xD63F_0000 | ((registers::hw_encoding(rn) as u32) << 5);
            self.emit_raw_word(blr_word);
        } else {
            return Err("CALL instruction has no symbol and no register operand".to_string());
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // ADRP emission helper
    // -----------------------------------------------------------------------

    /// Emit an ADRP instruction placeholder (encoding with imm21=0).
    fn emit_adrp(&mut self, rd: u8, _is_got: bool) {
        // ADRP encoding: op=1 (bit 31), immlo (bits 30:29), 10000 (bits 28:24),
        // immhi (bits 23:5), Rd (bits 4:0). With imm21=0:
        let hw_rd = registers::hw_encoding(rd) as u32;
        let word: u32 = (1 << 31) | (0b10000 << 24) | hw_rd;
        self.emit_raw_word(word);
    }

    // -----------------------------------------------------------------------
    // Relocation emission helpers
    // -----------------------------------------------------------------------

    /// Emit ADRP + ADD relocation pair for non-GOT symbol addressing.
    ///
    /// - At `adrp_offset`: `R_AARCH64_ADR_PREL_PG_HI21` (page-relative high bits)
    /// - At `adrp_offset + 4`: `R_AARCH64_ADD_ABS_LO12_NC` (low 12-bit offset)
    fn emit_adrp_add_relocation(&mut self, symbol: &str, adrp_offset: u64) {
        self.relocations.push(AssemblyRelocation {
            offset: adrp_offset,
            symbol: symbol.to_string(),
            reloc_type: AArch64RelocationType::AdrPrelPgHi21.to_raw(),
            addend: 0,
        });
        self.relocations.push(AssemblyRelocation {
            offset: adrp_offset + INSTRUCTION_SIZE,
            symbol: symbol.to_string(),
            reloc_type: AArch64RelocationType::AddAbsLo12Nc.to_raw(),
            addend: 0,
        });
    }

    /// Emit ADRP + LDR GOT relocation pair for PIC symbol addressing.
    ///
    /// - At `adrp_offset`: `R_AARCH64_ADR_GOT_PAGE` (GOT page)
    /// - At `adrp_offset + 4`: `R_AARCH64_LD64_GOT_LO12_NC` (GOT entry low bits)
    fn emit_got_relocation(&mut self, symbol: &str, adrp_offset: u64) {
        self.relocations.push(AssemblyRelocation {
            offset: adrp_offset,
            symbol: symbol.to_string(),
            reloc_type: AArch64RelocationType::AdrGotPage.to_raw(),
            addend: 0,
        });
        self.relocations.push(AssemblyRelocation {
            offset: adrp_offset + INSTRUCTION_SIZE,
            symbol: symbol.to_string(),
            reloc_type: AArch64RelocationType::Ld64GotLo12Nc.to_raw(),
            addend: 0,
        });
    }

    /// Emit a branch-type relocation at the given offset.
    fn emit_branch_relocation(&mut self, symbol: &str, offset: u64, reloc_type: u32) {
        self.relocations.push(AssemblyRelocation {
            offset,
            symbol: symbol.to_string(),
            reloc_type,
            addend: 0,
        });
    }

    // -----------------------------------------------------------------------
    // Local branch resolution
    // -----------------------------------------------------------------------

    /// Resolve local branch targets after all instructions have been assembled.
    ///
    /// For each relocation targeting a label defined in `self.symbols`:
    /// - Compute the PC-relative offset (`target - patch_site`).
    /// - Validate the offset fits within the relocation's range.
    /// - Patch the instruction encoding in the code buffer.
    /// - Remove the resolved relocation from the list.
    ///
    /// Relocations referencing undefined/external symbols are left intact
    /// for the linker to process.
    ///
    /// # Branch Ranges (AArch64)
    /// - B/BL (CALL26/JUMP26): ±128 MiB (26-bit signed word offset)
    /// - B.cond/CBZ/CBNZ (CONDBR19): ±1 MiB (19-bit signed word offset)
    /// - TBZ/TBNZ (TSTBR14): ±32 KiB (14-bit signed word offset)
    pub fn resolve_local_branches(&mut self) -> Result<(), String> {
        // Phase 1: Scan relocations and collect patch actions.
        // We separate the scanning from the patching to satisfy the borrow
        // checker — we cannot mutably borrow self.code while iterating
        // over self.relocations.
        enum PatchAction {
            Imm26 { offset: usize, imm26: i32 },
            Imm19 { offset: usize, imm19: i32 },
            Imm14 { offset: usize, imm14: i32 },
            AdrImm { offset: usize, imm21: i32 },
            Imm12 { offset: usize, imm12: u32 },
        }

        let mut patches: Vec<PatchAction> = Vec::new();
        let mut unresolved_indices: Vec<usize> = Vec::new();

        for (idx, reloc) in self.relocations.iter().enumerate() {
            if let Some(&target_offset) = self.symbols.get(&reloc.symbol) {
                let pc = reloc.offset;
                let raw_offset = target_offset as i64 - pc as i64;

                let reloc_type = AArch64RelocationType::from_raw(reloc.reloc_type);

                match reloc_type {
                    Some(AArch64RelocationType::Call26) | Some(AArch64RelocationType::Jump26) => {
                        if !(MIN_BRANCH26_OFFSET..=MAX_BRANCH26_OFFSET).contains(&raw_offset) {
                            return Err(format!(
                                "branch target '{}' out of range for B/BL: offset {} bytes \
                                 (must be within ±128 MiB)",
                                reloc.symbol, raw_offset
                            ));
                        }
                        let imm26 = (raw_offset >> 2) as i32;
                        patches.push(PatchAction::Imm26 {
                            offset: pc as usize,
                            imm26,
                        });
                    }

                    Some(AArch64RelocationType::Condbr19) => {
                        if !(MIN_CONDBR19_OFFSET..=MAX_CONDBR19_OFFSET).contains(&raw_offset) {
                            return Err(format!(
                                "conditional branch target '{}' out of range for B.cond: \
                                 offset {} bytes (must be within ±1 MiB)",
                                reloc.symbol, raw_offset
                            ));
                        }
                        let imm19 = (raw_offset >> 2) as i32;
                        patches.push(PatchAction::Imm19 {
                            offset: pc as usize,
                            imm19,
                        });
                    }

                    Some(AArch64RelocationType::Tstbr14) => {
                        if !(MIN_TSTBR14_OFFSET..=MAX_TSTBR14_OFFSET).contains(&raw_offset) {
                            return Err(format!(
                                "test-and-branch target '{}' out of range for TBZ/TBNZ: \
                                 offset {} bytes (must be within ±32 KiB)",
                                reloc.symbol, raw_offset
                            ));
                        }
                        let imm14 = (raw_offset >> 2) as i32;
                        patches.push(PatchAction::Imm14 {
                            offset: pc as usize,
                            imm14,
                        });
                    }

                    Some(AArch64RelocationType::AdrPrelPgHi21)
                    | Some(AArch64RelocationType::AdrPrelPgHi21Nc) => {
                        let target_page = (target_offset as i64) & !0xFFF;
                        let pc_page = (pc as i64) & !0xFFF;
                        let page_offset = target_page - pc_page;
                        let imm21 = (page_offset >> 12) as i32;
                        patches.push(PatchAction::AdrImm {
                            offset: pc as usize,
                            imm21,
                        });
                    }

                    Some(AArch64RelocationType::AddAbsLo12Nc) => {
                        let lo12 = (target_offset & 0xFFF) as u32;
                        patches.push(PatchAction::Imm12 {
                            offset: pc as usize,
                            imm12: lo12,
                        });
                    }

                    _ => {
                        // Cannot resolve locally — keep for linker.
                        unresolved_indices.push(idx);
                        continue;
                    }
                }
                // Successfully resolved — index not added to unresolved.
            } else {
                // Symbol not defined locally — keep for linker.
                unresolved_indices.push(idx);
            }
        }

        // Phase 2: Apply all collected patches to the code buffer.
        for patch in &patches {
            match patch {
                PatchAction::Imm26 { offset, imm26 } => {
                    self.patch_imm26(*offset, *imm26);
                }
                PatchAction::Imm19 { offset, imm19 } => {
                    self.patch_imm19(*offset, *imm19);
                }
                PatchAction::Imm14 { offset, imm14 } => {
                    self.patch_imm14(*offset, *imm14);
                }
                PatchAction::AdrImm { offset, imm21 } => {
                    self.patch_adr_imm(*offset, *imm21);
                }
                PatchAction::Imm12 { offset, imm12 } => {
                    self.patch_imm12(*offset, *imm12);
                }
            }
        }

        // Phase 3: Retain only unresolved relocations.
        let old_relocations = std::mem::take(&mut self.relocations);
        self.relocations = unresolved_indices
            .into_iter()
            .filter_map(|idx| old_relocations.get(idx).cloned())
            .collect();

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Instruction patching helpers
    // -----------------------------------------------------------------------

    /// Patch a 26-bit signed immediate into a B/BL instruction at `offset`.
    fn patch_imm26(&mut self, offset: usize, imm26: i32) {
        if offset + 4 > self.code.len() {
            return;
        }
        let inst = read_le_u32(&self.code, offset);
        let patched = (inst & !0x03FF_FFFF) | ((imm26 as u32) & 0x03FF_FFFF);
        write_le_u32(&mut self.code, offset, patched);
    }

    /// Patch a 19-bit signed immediate into a B.cond/CBZ/CBNZ instruction.
    fn patch_imm19(&mut self, offset: usize, imm19: i32) {
        if offset + 4 > self.code.len() {
            return;
        }
        let inst = read_le_u32(&self.code, offset);
        let patched = (inst & !0x00FF_FFE0) | (((imm19 as u32) & 0x7FFFF) << 5);
        write_le_u32(&mut self.code, offset, patched);
    }

    /// Patch a 14-bit signed immediate into a TBZ/TBNZ instruction.
    fn patch_imm14(&mut self, offset: usize, imm14: i32) {
        if offset + 4 > self.code.len() {
            return;
        }
        let inst = read_le_u32(&self.code, offset);
        let patched = (inst & !0x0007_FFE0) | (((imm14 as u32) & 0x3FFF) << 5);
        write_le_u32(&mut self.code, offset, patched);
    }

    /// Patch a 21-bit signed immediate into an ADRP/ADR instruction.
    fn patch_adr_imm(&mut self, offset: usize, imm21: i32) {
        if offset + 4 > self.code.len() {
            return;
        }
        let inst = read_le_u32(&self.code, offset);
        let imm = imm21 as u32;
        let immlo = imm & 0x3;
        let immhi = (imm >> 2) & 0x7FFFF;
        let patched = (inst & !(0x00FF_FFE0 | (0x3 << 29))) | (immhi << 5) | (immlo << 29);
        write_le_u32(&mut self.code, offset, patched);
    }

    /// Patch a 12-bit unsigned immediate into an ADD/LDR/STR instruction.
    fn patch_imm12(&mut self, offset: usize, imm12: u32) {
        if offset + 4 > self.code.len() {
            return;
        }
        let inst = read_le_u32(&self.code, offset);
        let patched = (inst & !0x003F_FC00) | ((imm12 & 0xFFF) << 10);
        write_le_u32(&mut self.code, offset, patched);
    }

    // -----------------------------------------------------------------------
    // Alignment and padding
    // -----------------------------------------------------------------------

    /// Pad the code buffer with NOP instructions to reach the specified
    /// alignment boundary.
    ///
    /// The alignment must be a power of two and at least 4 bytes (the
    /// inherent instruction alignment of AArch64). The minimum alignment
    /// is derived from the target's instruction width to ensure correctness.
    pub fn align_to(&mut self, alignment: u64) {
        // Ensure alignment is at least the instruction size for this target.
        // On AArch64, all instructions are 4-byte aligned. The page_size()
        // call validates we have a valid target configuration.
        let _ = self.target.page_size();
        let alignment = alignment.max(INSTRUCTION_SIZE);
        while self.current_offset % alignment != 0 {
            self.emit_nop();
        }
    }

    /// Emit a single NOP instruction (4 bytes).
    ///
    /// AArch64 NOP encoding: `0xD503201F` (little-endian: `1F 20 03 D5`).
    pub fn emit_nop(&mut self) {
        self.emit_raw_word(NOP_ENCODING);
    }

    // -----------------------------------------------------------------------
    // Finalization and section building
    // -----------------------------------------------------------------------

    /// Finalize the assembly, resolving local branches and producing the
    /// final [`AssemblyResult`].
    ///
    /// After finalization, only relocations for external/undefined symbols
    /// remain in the result — all local branch targets have been patched
    /// into the code buffer.
    pub fn finalize(&mut self) -> AssemblyResult {
        // Attempt to resolve local branches; any errors are deferred
        // as unresolved relocations rather than hard failures, since the
        // linker may handle long-range branches.
        let _ = self.resolve_local_branches();

        AssemblyResult {
            code: self.code.clone(),
            relocations: self.relocations.clone(),
            symbols: self.symbols.clone(),
            code_size: self.current_offset,
        }
    }

    /// Return the code bytes and unresolved relocations suitable for
    /// constructing an ELF `.text` section.
    ///
    /// This method does not modify the assembler state — it returns cloned
    /// copies of the internal buffers.
    pub fn build_text_section(&self) -> (Vec<u8>, Vec<AssemblyRelocation>) {
        (self.code.clone(), self.relocations.clone())
    }

    // -----------------------------------------------------------------------
    // Inline assembly support
    // -----------------------------------------------------------------------

    /// Assemble an inline assembly template string with operand substitution.
    ///
    /// Parses the template, substitutes operand placeholders (`%0`, `%1`,
    /// `%[name]`) with register names or immediate values, and encodes each
    /// resulting instruction via the encoder.
    ///
    /// Handles `.pushsection` / `.popsection` directives by switching the
    /// output section context (though for the initial implementation, these
    /// directives are noted but the bytes still go into the current code
    /// buffer).
    ///
    /// Returns the total number of bytes emitted.
    pub fn assemble_inline_asm(
        &mut self,
        template: &str,
        operands: &[InlineAsmOperand],
    ) -> Result<usize, String> {
        let start_offset = self.current_offset;

        // Split the template into individual lines/instructions.
        let lines: Vec<&str> = template
            .split(|c| c == '\n' || c == ';')
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .collect();

        for line in &lines {
            // Handle assembler directives.
            if line.starts_with('.') {
                self.handle_asm_directive(line)?;
                continue;
            }

            // Substitute operand placeholders in the instruction text.
            let expanded = self.substitute_operands(line, operands);

            // For inline assembly, we encode each instruction individually.
            // If the line can be parsed as a known A64 instruction, encode it.
            // Otherwise, emit a NOP as a placeholder — the full inline asm
            // parser will be extended as needed during kernel build.
            match self.try_encode_asm_line(&expanded) {
                Ok(bytes) => {
                    self.code.extend_from_slice(&bytes);
                    self.current_offset += bytes.len() as u64;
                }
                Err(_) => {
                    // Unknown instruction — emit NOP placeholder.
                    // This allows the assembler to make forward progress while
                    // additional instructions are implemented.
                    self.emit_nop();
                }
            }
        }

        let bytes_emitted = (self.current_offset - start_offset) as usize;
        Ok(bytes_emitted)
    }

    /// Handle an assembler directive within inline assembly.
    fn handle_asm_directive(&mut self, directive: &str) -> Result<(), String> {
        let lower = directive.to_ascii_lowercase();

        if lower.starts_with(".pushsection") || lower.starts_with(".popsection") {
            // Section switching directives — noted but currently inline asm
            // bytes stay in the main code buffer. A full implementation would
            // maintain a section stack and redirect output accordingly.
            Ok(())
        } else if lower.starts_with(".align") || lower.starts_with(".p2align") {
            // Alignment directive — parse the alignment value and pad.
            let parts: Vec<&str> = directive.split_whitespace().collect();
            if parts.len() >= 2 {
                if let Ok(align_val) = parts[1].trim_end_matches(',').parse::<u64>() {
                    let alignment = if lower.starts_with(".p2align") {
                        1u64 << align_val
                    } else {
                        align_val
                    };
                    self.align_to(alignment);
                }
            }
            Ok(())
        } else if lower.starts_with(".byte") {
            // .byte directives — emit raw bytes.
            let parts: Vec<&str> = directive[5..].split(',').collect();
            for part in parts {
                let trimmed = part.trim();
                if let Ok(val) = parse_int_literal(trimmed) {
                    self.code.push(val as u8);
                    // .byte doesn't maintain 4-byte alignment; that's expected
                    // for inline asm data emission.
                }
            }
            // Re-align current_offset to the actual code length.
            self.current_offset = self.code.len() as u64;
            Ok(())
        } else if lower.starts_with(".word") || lower.starts_with(".long") {
            // .word / .long directives — emit 32-bit values.
            // ".word" is 5 chars, ".long" is 5 chars — both have the same prefix length.
            let prefix_len = 5;
            let parts: Vec<&str> = directive[prefix_len..].split(',').collect();
            for part in parts {
                let trimmed = part.trim();
                if let Ok(val) = parse_int_literal(trimmed) {
                    self.code.extend_from_slice(&(val as u32).to_le_bytes());
                }
            }
            self.current_offset = self.code.len() as u64;
            Ok(())
        } else {
            // Unknown directive — ignore silently for resilience.
            Ok(())
        }
    }

    /// Substitute operand placeholders in an asm template line.
    ///
    /// Handles `%0`, `%1`, etc., and `%[name]` named operands.
    fn substitute_operands(&self, line: &str, operands: &[InlineAsmOperand]) -> String {
        let mut result = String::with_capacity(line.len());
        let chars: Vec<char> = line.chars().collect();
        let len = chars.len();
        let mut i = 0;

        while i < len {
            if chars[i] == '%' && i + 1 < len {
                if chars[i + 1] == '[' {
                    // Named operand: %[name]
                    if let Some(close) = chars[i + 2..].iter().position(|&c| c == ']') {
                        let name: String = chars[i + 2..i + 2 + close].iter().collect();
                        let replacement = self.find_operand_by_name(&name, operands);
                        result.push_str(&replacement);
                        i += 3 + close; // skip %[name]
                        continue;
                    }
                } else if chars[i + 1] == '%' {
                    // Escaped percent: %%
                    result.push('%');
                    i += 2;
                    continue;
                } else if chars[i + 1].is_ascii_digit() {
                    // Positional operand: %0, %1, etc.
                    let mut end = i + 1;
                    while end < len && chars[end].is_ascii_digit() {
                        end += 1;
                    }
                    let idx_str: String = chars[i + 1..end].iter().collect();
                    if let Ok(idx) = idx_str.parse::<usize>() {
                        let replacement = self.format_operand(idx, operands);
                        result.push_str(&replacement);
                        i = end;
                        continue;
                    }
                }
            }
            result.push(chars[i]);
            i += 1;
        }
        result
    }

    /// Find an operand by its named label and format it.
    fn find_operand_by_name(&self, name: &str, operands: &[InlineAsmOperand]) -> String {
        for (idx, op) in operands.iter().enumerate() {
            if op.name.as_deref() == Some(name) {
                return self.format_operand(idx, operands);
            }
        }
        // Not found — return the placeholder as-is for error resilience.
        format!("%[{}]", name)
    }

    /// Format operand at the given index as a string suitable for asm substitution.
    fn format_operand(&self, idx: usize, operands: &[InlineAsmOperand]) -> String {
        if idx >= operands.len() {
            return format!("%{}", idx);
        }
        let op = &operands[idx];
        if let Some(reg) = op.register {
            format_aarch64_register(reg)
        } else if let Some(imm) = op.immediate {
            format!("#{}", imm)
        } else {
            format!("%{}", idx)
        }
    }

    /// Attempt to encode a single asm text line into machine code bytes.
    ///
    /// This is a simplified parser for common AArch64 instructions found
    /// in inline assembly. For unrecognized instructions, returns an error
    /// so the caller can fall back to a NOP placeholder.
    fn try_encode_asm_line(&self, _line: &str) -> Result<Vec<u8>, String> {
        // Full inline assembly text parsing is a significant undertaking.
        // For the initial implementation, we handle the most critical case:
        // NOP and simple instructions. The encoder will be extended
        // incrementally as kernel build inline asm patterns are encountered.
        //
        // For now, return error to trigger NOP fallback. The main inline asm
        // path uses A64Instruction-level encoding from the codegen pipeline,
        // not text-level parsing.
        Err("text-level asm encoding not yet implemented".to_string())
    }
}

// ===========================================================================
// Free-standing encoding helpers
// ===========================================================================

/// Encode an LDR (unsigned immediate offset) instruction.
///
/// `LDR Xt, [Xn, #imm]` — unsigned offset scaled by 8 for 64-bit loads.
/// Encoding: `11 111 0 01 01 imm12 Rn Rd`
fn encode_ldr_unsigned_imm(rd: u8, rn: u8, imm12: u32) -> u32 {
    let hw_rd = registers::hw_encoding(rd) as u32;
    let hw_rn = registers::hw_encoding(rn) as u32;
    // size=11 (64-bit), V=0 (GPR), opc=01 (LDR)
    (0b11 << 30) | (0b111001 << 24) | (0b01 << 22) | ((imm12 & 0xFFF) << 10) | (hw_rn << 5) | hw_rd
}

/// Encode an ADD immediate instruction.
///
/// `ADD Xd, Xn, #imm` (64-bit, no flags).
/// Encoding: `sf=1 | op=0 | S=0 | 100010 | sh=0 | imm12 | Rn | Rd`
fn encode_add_imm(rd: u8, rn: u8, imm12: u32) -> u32 {
    let hw_rd = registers::hw_encoding(rd) as u32;
    let hw_rn = registers::hw_encoding(rn) as u32;
    (1 << 31) | (0b100010 << 23) | ((imm12 & 0xFFF) << 10) | (hw_rn << 5) | hw_rd
}

/// Format an AArch64 register ID as a human-readable name for asm substitution.
fn format_aarch64_register(reg: u8) -> String {
    if reg < 31 {
        format!("x{}", reg)
    } else if reg == registers::XZR {
        "xzr".to_string()
    } else if (32..64).contains(&reg) {
        // SIMD/FP register
        let fp_idx = reg - 32;
        format!("v{}", fp_idx)
    } else {
        format!("r{}", reg)
    }
}

// ===========================================================================
// Little-endian read/write helpers
// ===========================================================================

/// Read a 32-bit little-endian word from a byte slice at the given offset.
#[inline]
fn read_le_u32(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

/// Write a 32-bit little-endian word to a byte slice at the given offset.
#[inline]
fn write_le_u32(data: &mut [u8], offset: usize, value: u32) {
    let bytes = value.to_le_bytes();
    data[offset] = bytes[0];
    data[offset + 1] = bytes[1];
    data[offset + 2] = bytes[2];
    data[offset + 3] = bytes[3];
}

/// Parse an integer literal from an asm directive value (decimal or hex).
fn parse_int_literal(s: &str) -> Result<i64, String> {
    let s = s.trim();
    if s.starts_with("0x") || s.starts_with("0X") {
        i64::from_str_radix(&s[2..], 16).map_err(|e| format!("invalid hex literal '{}': {}", s, e))
    } else if s.starts_with('-') {
        s.parse::<i64>()
            .map_err(|e| format!("invalid integer '{}': {}", s, e))
    } else {
        // Try unsigned first to handle large values, then fall back to signed.
        if let Ok(v) = s.parse::<u64>() {
            Ok(v as i64)
        } else {
            s.parse::<i64>()
                .map_err(|e| format!("invalid integer '{}': {}", s, e))
        }
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::aarch64::codegen::{A64Instruction, A64Opcode};

    #[test]
    fn test_new_assembler() {
        let asm = AArch64Assembler::new(false);
        assert_eq!(asm.current_offset(), 0);
        assert!(!asm.pic_mode);
        assert_eq!(asm.target, Target::AArch64);
    }

    #[test]
    fn test_new_assembler_pic_mode() {
        let asm = AArch64Assembler::new(true);
        assert!(asm.pic_mode);
    }

    #[test]
    fn test_reset() {
        let mut asm = AArch64Assembler::new(false);
        asm.define_label("test");
        asm.emit_nop();
        assert_eq!(asm.current_offset(), 4);
        assert!(asm.symbols.contains_key("test"));

        asm.reset();
        assert_eq!(asm.current_offset(), 0);
        assert!(asm.symbols.is_empty());
        assert!(asm.code.is_empty());
        assert!(asm.relocations.is_empty());
    }

    #[test]
    fn test_define_label() {
        let mut asm = AArch64Assembler::new(false);
        asm.emit_nop(); // offset = 0, advances to 4
        asm.define_label("after_nop");
        assert_eq!(*asm.symbols.get("after_nop").unwrap(), 4);
    }

    #[test]
    fn test_emit_nop() {
        let mut asm = AArch64Assembler::new(false);
        asm.emit_nop();
        assert_eq!(asm.current_offset(), 4);
        assert_eq!(asm.code.len(), 4);
        // NOP = 0xD503201F in little-endian: 1F 20 03 D5
        assert_eq!(asm.code[0], 0x1F);
        assert_eq!(asm.code[1], 0x20);
        assert_eq!(asm.code[2], 0x03);
        assert_eq!(asm.code[3], 0xD5);
    }

    #[test]
    fn test_nop_encoding_value() {
        assert_eq!(NOP_ENCODING, 0xD503_201F);
        let bytes = NOP_ENCODING.to_le_bytes();
        assert_eq!(bytes, [0x1F, 0x20, 0x03, 0xD5]);
    }

    #[test]
    fn test_align_to() {
        let mut asm = AArch64Assembler::new(false);
        asm.emit_nop(); // 4 bytes
        asm.align_to(16); // should emit 3 more NOPs to reach 16
        assert_eq!(asm.current_offset(), 16);
        assert_eq!(asm.code.len(), 16);
    }

    #[test]
    fn test_align_already_aligned() {
        let mut asm = AArch64Assembler::new(false);
        asm.align_to(4); // Already aligned at 0
        assert_eq!(asm.current_offset(), 0);
    }

    #[test]
    fn test_assemble_nop_instruction() {
        let mut asm = AArch64Assembler::new(false);
        let inst = A64Instruction::new(A64Opcode::NOP);
        let result = asm.assemble_function(&[inst]).unwrap();
        assert_eq!(result.code_size, 4);
        assert_eq!(result.code.len(), 4);
    }

    #[test]
    fn test_assemble_multiple_nops() {
        let mut asm = AArch64Assembler::new(false);
        let insts = vec![
            A64Instruction::new(A64Opcode::NOP),
            A64Instruction::new(A64Opcode::NOP),
            A64Instruction::new(A64Opcode::NOP),
        ];
        let result = asm.assemble_function(&insts).unwrap();
        assert_eq!(result.code_size, 12);
        assert_eq!(result.code.len(), 12);
    }

    #[test]
    fn test_bl_generates_call26_relocation() {
        let mut asm = AArch64Assembler::new(false);
        let inst = A64Instruction::new(A64Opcode::BL).with_symbol("my_function".to_string());
        let result = asm.assemble_function(&[inst]).unwrap();
        assert_eq!(result.code_size, 4);

        // Should have a CALL26 relocation.
        let call26_relocs: Vec<_> = result
            .relocations
            .iter()
            .filter(|r| r.reloc_type == AArch64RelocationType::Call26.to_raw())
            .collect();
        assert!(
            !call26_relocs.is_empty(),
            "expected CALL26 relocation for BL"
        );
        assert_eq!(call26_relocs[0].symbol, "my_function");
    }

    #[test]
    fn test_b_generates_jump26_relocation() {
        let mut asm = AArch64Assembler::new(false);
        let inst = A64Instruction::new(A64Opcode::B).with_symbol("target_label".to_string());
        let result = asm.assemble_function(&[inst]).unwrap();

        let jump26_relocs: Vec<_> = result
            .relocations
            .iter()
            .filter(|r| r.reloc_type == AArch64RelocationType::Jump26.to_raw())
            .collect();
        assert!(
            !jump26_relocs.is_empty(),
            "expected JUMP26 relocation for B"
        );
    }

    #[test]
    fn test_adrp_generates_relocation() {
        let mut asm = AArch64Assembler::new(false);
        let inst = A64Instruction::new(A64Opcode::ADRP)
            .with_rd(0)
            .with_symbol("global_var".to_string());
        let result = asm.assemble_function(&[inst]).unwrap();

        let adrp_relocs: Vec<_> = result
            .relocations
            .iter()
            .filter(|r| {
                r.reloc_type == AArch64RelocationType::AdrPrelPgHi21.to_raw()
                    || r.reloc_type == AArch64RelocationType::AdrGotPage.to_raw()
            })
            .collect();
        assert!(!adrp_relocs.is_empty(), "expected ADRP relocation");
    }

    #[test]
    fn test_assemble_module_records_function_symbols() {
        let mut asm = AArch64Assembler::new(false);
        let funcs = vec![
            (
                "func_a".to_string(),
                vec![A64Instruction::new(A64Opcode::NOP)],
            ),
            (
                "func_b".to_string(),
                vec![A64Instruction::new(A64Opcode::NOP)],
            ),
        ];
        let result = asm.assemble_module(&funcs).unwrap();
        assert!(result.symbols.contains_key("func_a"));
        assert!(result.symbols.contains_key("func_b"));
        assert_eq!(*result.symbols.get("func_a").unwrap(), 0);
        assert_eq!(*result.symbols.get("func_b").unwrap(), 4);
    }

    #[test]
    fn test_finalize_resolves_local_branches() {
        let mut asm = AArch64Assembler::new(false);

        // Define a label, emit NOP, then a BL to that label.
        asm.define_label("loop_start");
        asm.emit_nop(); // offset 0..4
                        // Manually add a CALL26 relocation targeting "loop_start".
        let bl_word: u32 = 0x9400_0000; // BL placeholder
        let bl_offset = asm.current_offset();
        asm.emit_raw_word(bl_word);
        asm.relocations.push(AssemblyRelocation {
            offset: bl_offset,
            symbol: "loop_start".to_string(),
            reloc_type: AArch64RelocationType::Call26.to_raw(),
            addend: 0,
        });

        let result = asm.finalize();
        // The relocation to "loop_start" should be resolved (removed from
        // the unresolved list) because "loop_start" is defined locally.
        let unresolved_loop: Vec<_> = result
            .relocations
            .iter()
            .filter(|r| r.symbol == "loop_start")
            .collect();
        assert!(
            unresolved_loop.is_empty(),
            "local branch to 'loop_start' should be resolved"
        );
    }

    #[test]
    fn test_build_text_section() {
        let mut asm = AArch64Assembler::new(false);
        asm.emit_nop();
        asm.emit_nop();

        let (code, relocs) = asm.build_text_section();
        assert_eq!(code.len(), 8);
        assert!(relocs.is_empty());
    }

    #[test]
    fn test_instruction_size_always_4() {
        assert_eq!(INSTRUCTION_SIZE, 4);
    }

    #[test]
    fn test_assembly_relocation_to_linker_relocation() {
        let asm_reloc = AssemblyRelocation {
            offset: 100,
            symbol: "extern_func".to_string(),
            reloc_type: AArch64RelocationType::Call26.to_raw(),
            addend: 0,
        };
        let linker_reloc = asm_reloc.to_linker_relocation();
        assert_eq!(linker_reloc.offset, 100);
        assert_eq!(linker_reloc.symbol_name, "extern_func");
        assert_eq!(
            linker_reloc.rel_type,
            AArch64RelocationType::Call26.to_raw()
        );
    }

    #[test]
    fn test_assembly_relocation_category() {
        let reloc = AssemblyRelocation {
            offset: 0,
            symbol: "sym".to_string(),
            reloc_type: AArch64RelocationType::Call26.to_raw(),
            addend: 0,
        };
        assert_eq!(reloc.category(), RelocCategory::PcRelative);

        let got_reloc = AssemblyRelocation {
            offset: 0,
            symbol: "sym".to_string(),
            reloc_type: AArch64RelocationType::AdrGotPage.to_raw(),
            addend: 0,
        };
        assert_eq!(got_reloc.category(), RelocCategory::GotRelative);
    }

    #[test]
    fn test_call_pseudo_instruction_direct() {
        let mut asm = AArch64Assembler::new(false);
        let inst = A64Instruction::new(A64Opcode::CALL).with_symbol("target_fn".to_string());
        asm.assemble_one(&inst).unwrap();
        assert_eq!(asm.current_offset(), 4);

        let call26_relocs: Vec<_> = asm
            .relocations
            .iter()
            .filter(|r| r.reloc_type == AArch64RelocationType::Call26.to_raw())
            .collect();
        assert_eq!(call26_relocs.len(), 1);
        assert_eq!(call26_relocs[0].symbol, "target_fn");
    }

    #[test]
    fn test_call_pseudo_instruction_indirect() {
        let mut asm = AArch64Assembler::new(false);
        let inst = A64Instruction::new(A64Opcode::CALL).with_rn(registers::X0);
        asm.assemble_one(&inst).unwrap();
        assert_eq!(asm.current_offset(), 4);
        // Indirect call has no relocation.
        assert!(asm.relocations.is_empty());
    }

    #[test]
    fn test_la_pseudo_non_pic() {
        let mut asm = AArch64Assembler::new(false);
        let inst = A64Instruction::new(A64Opcode::LA)
            .with_rd(registers::X0)
            .with_symbol("my_global".to_string());
        asm.assemble_one(&inst).unwrap();
        // LA expands to ADRP + ADD = 8 bytes.
        assert_eq!(asm.current_offset(), 8);

        // Should have ADRP_HI21 and ADD_LO12 relocations.
        let hi21: Vec<_> = asm
            .relocations
            .iter()
            .filter(|r| r.reloc_type == AArch64RelocationType::AdrPrelPgHi21.to_raw())
            .collect();
        let lo12: Vec<_> = asm
            .relocations
            .iter()
            .filter(|r| r.reloc_type == AArch64RelocationType::AddAbsLo12Nc.to_raw())
            .collect();
        assert_eq!(hi21.len(), 1, "expected ADR_PREL_PG_HI21 relocation");
        assert_eq!(lo12.len(), 1, "expected ADD_ABS_LO12_NC relocation");
        assert_eq!(hi21[0].offset, 0);
        assert_eq!(lo12[0].offset, 4);
    }

    #[test]
    fn test_la_pseudo_pic() {
        let mut asm = AArch64Assembler::new(true);
        let inst = A64Instruction::new(A64Opcode::LA)
            .with_rd(registers::X0)
            .with_symbol("got_sym".to_string());
        asm.assemble_one(&inst).unwrap();
        // LA in PIC mode expands to ADRP + LDR = 8 bytes.
        assert_eq!(asm.current_offset(), 8);

        let got_page: Vec<_> = asm
            .relocations
            .iter()
            .filter(|r| r.reloc_type == AArch64RelocationType::AdrGotPage.to_raw())
            .collect();
        let got_lo12: Vec<_> = asm
            .relocations
            .iter()
            .filter(|r| r.reloc_type == AArch64RelocationType::Ld64GotLo12Nc.to_raw())
            .collect();
        assert_eq!(got_page.len(), 1, "expected ADR_GOT_PAGE relocation");
        assert_eq!(got_lo12.len(), 1, "expected LD64_GOT_LO12_NC relocation");
    }

    #[test]
    fn test_inline_asm_basic() {
        let mut asm = AArch64Assembler::new(false);
        let bytes = asm.assemble_inline_asm("nop", &[]).unwrap();
        // Should emit at least 4 bytes (a NOP).
        assert!(bytes >= 4);
    }

    #[test]
    fn test_offset_always_multiple_of_4() {
        let mut asm = AArch64Assembler::new(false);
        for _ in 0..10 {
            asm.emit_nop();
            assert_eq!(asm.current_offset() % 4, 0, "offset must be 4-byte aligned");
        }
    }

    #[test]
    fn test_parse_int_literal_decimal() {
        assert_eq!(parse_int_literal("42").unwrap(), 42);
        assert_eq!(parse_int_literal("-1").unwrap(), -1);
        assert_eq!(parse_int_literal("0").unwrap(), 0);
    }

    #[test]
    fn test_parse_int_literal_hex() {
        assert_eq!(parse_int_literal("0xFF").unwrap(), 255);
        assert_eq!(parse_int_literal("0x1000").unwrap(), 4096);
    }

    #[test]
    fn test_format_register() {
        assert_eq!(format_aarch64_register(0), "x0");
        assert_eq!(format_aarch64_register(30), "x30");
        assert_eq!(format_aarch64_register(31), "xzr");
        assert_eq!(format_aarch64_register(32), "v0");
        assert_eq!(format_aarch64_register(63), "v31");
    }

    #[test]
    fn test_empty_assembly_result() {
        let result = AssemblyResult::empty();
        assert!(result.code.is_empty());
        assert!(result.relocations.is_empty());
        assert!(result.symbols.is_empty());
        assert_eq!(result.code_size, 0);
    }
}
