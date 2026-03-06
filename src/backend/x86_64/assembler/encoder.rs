//! x86-64 instruction encoder — binary encoding engine for the built-in assembler.
//!
//! This module converts x86-64 machine instructions (represented as
//! [`MachineInstruction`] with [`X86Opcode`] opcodes) into their binary byte
//! representation. It handles:
//!
//! - **REX prefix generation** (0x40–0x4F): REX.W for 64-bit operand size,
//!   REX.R/X/B for extended register encoding (R8–R15, XMM8–XMM15)
//! - **ModR/M byte construction**: 2-bit mod + 3-bit reg + 3-bit r/m fields
//! - **SIB byte construction**: 2-bit scale + 3-bit index + 3-bit base fields
//! - **Displacement encoding**: 8-bit or 32-bit signed displacements
//! - **Immediate encoding**: 8/16/32/64-bit immediate values
//! - **Relocation emission**: Creates [`RelocationEntry`] records for symbol references
//! - **SSE/SSE2 instruction encoding**: Mandatory prefix handling (0xF2/0xF3/0x66)
//!
//! ## x86-64 Instruction Format (in order)
//!
//! 1. Legacy prefixes (0–4 bytes): 0x66, 0x67, 0xF2, 0xF3, segment overrides
//! 2. REX prefix (0–1 byte): 0100 WRXB
//! 3. Opcode (1–3 bytes)
//! 4. ModR/M (0–1 byte): mod(2) + reg(3) + r/m(3)
//! 5. SIB (0–1 byte): scale(2) + index(3) + base(3)
//! 6. Displacement (0/1/4 bytes)
//! 7. Immediate (0/1/2/4/8 bytes)
//!
//! ## Zero-Dependency Mandate
//!
//! Only `std` and `crate::` references. No external crates.

use super::relocations::{RelocationEntry, X86_64RelocationType};
use crate::backend::traits::{MachineInstruction, MachineOperand};
use crate::backend::x86_64::codegen::X86Opcode;
use crate::backend::x86_64::registers::{
    hw_encoding, is_gpr, is_sse, needs_rex, OPERAND_SIZE_PREFIX,
    RAX, RBP, RBX, RCX, RDI, RDX, RSI, RSP,
    R8, R9, R10, R11, R12, R13, R14, R15,
    XMM0, XMM8,
};
use crate::common::fx_hash::FxHashMap;

// ===========================================================================
// EncodedInstruction — encoder output
// ===========================================================================

/// Result of encoding a single x86-64 instruction.
///
/// Contains the raw machine code bytes (1–15 bytes per x86-64 instruction)
/// and any relocation entries generated for symbol references.
#[derive(Debug, Clone)]
pub struct EncodedInstruction {
    /// The raw bytes of the encoded instruction (1–15 bytes for x86-64).
    pub bytes: Vec<u8>,
    /// Relocation entries generated during encoding (e.g., for external
    /// symbol references in call/jmp/mov instructions).
    pub relocations: Vec<RelocationEntry>,
}

impl EncodedInstruction {
    /// Create a new encoded instruction with the given bytes and no relocations.
    fn new(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            relocations: Vec::new(),
        }
    }

    /// Create a new encoded instruction with bytes and relocations.
    fn with_relocations(bytes: Vec<u8>, relocations: Vec<RelocationEntry>) -> Self {
        Self { bytes, relocations }
    }
}

// ===========================================================================
// MemoryEncoding — memory operand encoding result
// ===========================================================================

/// Result of encoding a memory operand into ModR/M, optional SIB, and
/// displacement bytes.
#[derive(Debug, Clone)]
pub struct MemoryEncoding {
    /// The ModR/M byte encoding the memory addressing mode.
    pub modrm: u8,
    /// Optional SIB byte (present when base is RSP/R12 or scaled-index
    /// addressing is used).
    pub sib: Option<u8>,
    /// Displacement bytes (0, 1, or 4 bytes depending on the mod field).
    pub displacement: Vec<u8>,
}

// ===========================================================================
// EncodingForm — instruction encoding formats
// ===========================================================================

/// x86-64 instruction encoding forms.
///
/// Determines how operands are mapped into the ModR/M byte fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodingForm {
    /// Register in reg field, register/memory in r/m field.
    RegRm,
    /// Register/memory in r/m field, register in reg field.
    RmReg,
    /// Register/memory in r/m field, immediate follows.
    RmImm,
    /// R/m only — opcode extension in reg field.
    RmOnly,
    /// Register encoded in lower 3 bits of opcode (e.g., PUSH r64).
    OpcodeReg,
    /// No operands (e.g., RET, NOP, CDQ).
    NoOperands,
    /// Relative offset (e.g., JMP rel32, CALL rel32).
    Relative,
    /// Special encoding handled case-by-case.
    Special,
}

// ===========================================================================
// OpcodeEntry — opcode table entry
// ===========================================================================

/// Encoding metadata for an x86-64 opcode.
///
/// Maps an [`X86Opcode`] variant to its binary encoding information,
/// enabling table-driven instruction encoding.
#[derive(Debug, Clone)]
pub struct OpcodeEntry {
    /// Primary opcode byte(s) (1–3 bytes).
    pub opcode: &'static [u8],
    /// Opcode extension in ModR/M reg field (`/0`–`/7`), if used.
    pub reg_opext: Option<u8>,
    /// Whether this instruction defaults to 64-bit operand size (REX.W).
    pub default_64bit: bool,
    /// The instruction encoding form.
    pub form: EncodingForm,
}

// ===========================================================================
// REX Prefix Construction
// ===========================================================================

/// Construct a REX prefix byte.
///
/// REX format: 0100 WRXB
/// - W: 1 = 64-bit operand size
/// - R: extension of ModR/M reg field (selects R8–R15)
/// - X: extension of SIB index field
/// - B: extension of ModR/M r/m field or SIB base field
#[inline]
fn rex_byte(w: bool, r: bool, x: bool, b: bool) -> u8 {
    let mut rex: u8 = 0x40;
    if w { rex |= 0x08; }
    if r { rex |= 0x04; }
    if x { rex |= 0x02; }
    if b { rex |= 0x01; }
    rex
}

/// Determine if a REX prefix is needed and construct it.
///
/// Returns `Some(rex_byte)` if any REX bit is set, `None` otherwise.
/// REX.W forces 64-bit operand size, REX.R extends the reg field,
/// REX.X extends the SIB index field, and REX.B extends the r/m or
/// SIB base field.
fn compute_rex(
    operand_size_64: bool,
    reg: Option<u16>,
    index: Option<u16>,
    rm_or_base: Option<u16>,
) -> Option<u8> {
    let w = operand_size_64;
    let r = reg.map_or(false, needs_rex);
    let x = index.map_or(false, needs_rex);
    let b = rm_or_base.map_or(false, needs_rex);
    if w || r || x || b {
        Some(rex_byte(w, r, x, b))
    } else {
        None
    }
}

// ===========================================================================
// ModR/M Byte Construction
// ===========================================================================

/// Construct a ModR/M byte.
///
/// ModR/M format: [mod(2)][reg(3)][r/m(3)]
/// - mod: 00=no disp, 01=disp8, 10=disp32, 11=reg-reg
/// - reg: register operand or opcode extension
/// - r/m: register or memory operand
#[inline]
fn modrm_byte(mod_bits: u8, reg: u8, rm: u8) -> u8 {
    ((mod_bits & 0x3) << 6) | ((reg & 0x7) << 3) | (rm & 0x7)
}

// ===========================================================================
// SIB Byte Construction
// ===========================================================================

/// Construct a SIB byte.
///
/// SIB format: [scale(2)][index(3)][base(3)]
/// - scale: 00=×1, 01=×2, 10=×4, 11=×8
/// - index: index register (100b = no index)
/// - base: base register (101b with mod=00 = disp32 only)
#[inline]
fn sib_byte(scale: u8, index: u8, base: u8) -> u8 {
    ((scale & 0x3) << 6) | ((index & 0x7) << 3) | (base & 0x7)
}

/// Convert scale factor (1, 2, 4, 8) to SIB scale encoding (0, 1, 2, 3).
#[inline]
fn scale_encoding(scale: u8) -> u8 {
    match scale {
        1 => 0b00,
        2 => 0b01,
        4 => 0b10,
        8 => 0b11,
        _ => 0b00, // default to ×1 for invalid scales
    }
}

// ===========================================================================
// Displacement and Immediate Encoding
// ===========================================================================

/// Encode a displacement value as 1 or 4 bytes.
///
/// Uses disp8 (sign-extended) if the value fits in [-128, 127],
/// otherwise disp32 (4 bytes, little-endian).
#[allow(dead_code)]
fn encode_displacement(disp: i64) -> Vec<u8> {
    if (-128..=127).contains(&disp) {
        vec![disp as i8 as u8]
    } else {
        (disp as i32).to_le_bytes().to_vec()
    }
}

/// Encode an immediate value in the specified number of bytes (little-endian).
fn encode_immediate(imm: i64, size: u8) -> Vec<u8> {
    match size {
        1 => vec![imm as u8],
        2 => (imm as i16).to_le_bytes().to_vec(),
        4 => (imm as i32).to_le_bytes().to_vec(),
        8 => imm.to_le_bytes().to_vec(),
        _ => vec![imm as u8],
    }
}

/// Compute mod bits and displacement bytes for a memory operand.
///
/// - If displacement is 0 and base encoding is not 5 (RBP/R13): mod=00, no disp
/// - If displacement fits in i8: mod=01, 1-byte disp
/// - Otherwise: mod=10, 4-byte disp
///
/// Note: base_enc==5 with mod=00 means RIP-relative addressing, so we
/// must use mod=01 with disp8=0 when the actual displacement is 0 but
/// the base is RBP or R13.
fn compute_mod_disp(displacement: i64, base_enc: u8) -> (u8, Vec<u8>) {
    if displacement == 0 && base_enc != 5 {
        (0b00, vec![])
    } else if (-128..=127).contains(&displacement) {
        (0b01, vec![displacement as i8 as u8])
    } else {
        (0b10, (displacement as i32).to_le_bytes().to_vec())
    }
}

// ===========================================================================
// Memory Operand Encoding
// ===========================================================================

/// Encode a memory operand into ModR/M + optional SIB + displacement.
///
/// Handles all x86-64 memory addressing modes:
/// - `[base]` — simple base register
/// - `[base + disp8/disp32]` — base + displacement
/// - `[base + index*scale + disp]` — SIB addressing
/// - `[disp32]` — absolute addressing (via SIB escape)
///
/// Special encoding rules:
/// - RSP/R12 (hw_encoding & 7 == 4) as base always require SIB byte
/// - RBP/R13 (hw_encoding & 7 == 5) as base with 0 displacement use mod=01
/// - RSP (hw_encoding 4) as SIB index means "no index register"
fn encode_memory_operand(
    reg_or_opext: u8,
    base: Option<u16>,
    index: Option<u16>,
    scale: u8,
    displacement: i64,
) -> MemoryEncoding {
    let base_enc_low = base.map(|b| hw_encoding(b) & 0x7);
    let has_real_index = index.is_some()
        && index.map(|i| hw_encoding(i) & 0x7).unwrap_or(4) != 4;

    let need_sib = has_real_index
        || base_enc_low == Some(SIB_REQUIRED_ENC)
        || base.is_none();

    if need_sib {
        let index_enc = match index {
            Some(idx) if hw_encoding(idx) & 0x7 != 4 => hw_encoding(idx) & 0x7,
            _ => 0b100, // 100 = no index
        };
        let scale_enc = if has_real_index { scale_encoding(scale) } else { 0b00 };

        match base {
            None => {
                // No base: mod=00, rm=100 (SIB), SIB base=101 = disp32 only
                MemoryEncoding {
                    modrm: modrm_byte(0b00, reg_or_opext, 0b100),
                    sib: Some(sib_byte(scale_enc, index_enc, 0b101)),
                    displacement: (displacement as i32).to_le_bytes().to_vec(),
                }
            }
            Some(b) => {
                let b_enc = hw_encoding(b) & 0x7;
                let (mod_bits, disp_bytes) = compute_mod_disp(displacement, b_enc);
                MemoryEncoding {
                    modrm: modrm_byte(mod_bits, reg_or_opext, 0b100),
                    sib: Some(sib_byte(scale_enc, index_enc, b_enc)),
                    displacement: disp_bytes,
                }
            }
        }
    } else {
        // No SIB byte needed
        let base_reg = base.expect("Base register required without SIB");
        let rm = hw_encoding(base_reg) & 0x7;
        let (mod_bits, disp_bytes) = compute_mod_disp(displacement, rm);
        MemoryEncoding {
            modrm: modrm_byte(mod_bits, reg_or_opext, rm),
            sib: None,
            displacement: disp_bytes,
        }
    }
}

// ===========================================================================
// Condition Code and ALU Helpers
// ===========================================================================

/// Map an X86Opcode conditional variant to its x86-64 condition code (cc).
fn condition_code(op: &X86Opcode) -> u8 {
    match op {
        X86Opcode::Je | X86Opcode::Cmove | X86Opcode::Sete => 0x04,
        X86Opcode::Jne | X86Opcode::Cmovne | X86Opcode::Setne => 0x05,
        X86Opcode::Jl => 0x0C,
        X86Opcode::Jle => 0x0E,
        X86Opcode::Jg => 0x0F,
        X86Opcode::Jge => 0x0D,
        _ => 0x04,
    }
}

/// Return (opcode_extension, rm_reg_opcode, reg_rm_opcode) for ALU ops.
fn alu_info(op: &X86Opcode) -> Option<(u8, u8, u8)> {
    match op {
        X86Opcode::Add => Some((0, 0x01, 0x03)),
        X86Opcode::Or  => Some((1, 0x09, 0x0B)),
        X86Opcode::And => Some((4, 0x21, 0x23)),
        X86Opcode::Sub => Some((5, 0x29, 0x2B)),
        X86Opcode::Xor => Some((6, 0x31, 0x33)),
        X86Opcode::Cmp => Some((7, 0x39, 0x3B)),
        _ => None,
    }
}

/// Return the opcode extension for shift instructions.
fn shift_opext(op: &X86Opcode) -> u8 {
    match op {
        X86Opcode::Shl => 4,
        X86Opcode::Shr => 5,
        X86Opcode::Sar => 7,
        _ => 4,
    }
}

// ===========================================================================
// Opcode Table Construction
// ===========================================================================

/// Build the opcode table mapping X86Opcode discriminants to their
/// binary encoding metadata using FxHashMap for O(1) lookup.
fn build_opcode_table() -> FxHashMap<u32, OpcodeEntry> {
    let mut table: FxHashMap<u32, OpcodeEntry> = FxHashMap::default();

    table.insert(X86Opcode::Ret.as_u32(), OpcodeEntry {
        opcode: &[0xC3], reg_opext: None, default_64bit: false,
        form: EncodingForm::NoOperands,
    });
    table.insert(X86Opcode::Nop.as_u32(), OpcodeEntry {
        opcode: &[0x90], reg_opext: None, default_64bit: false,
        form: EncodingForm::NoOperands,
    });
    table.insert(X86Opcode::Cdq.as_u32(), OpcodeEntry {
        opcode: &[0x99], reg_opext: None, default_64bit: false,
        form: EncodingForm::NoOperands,
    });
    table.insert(X86Opcode::Cqo.as_u32(), OpcodeEntry {
        opcode: &[0x99], reg_opext: None, default_64bit: true,
        form: EncodingForm::NoOperands,
    });
    table.insert(X86Opcode::Endbr64.as_u32(), OpcodeEntry {
        opcode: &[0xF3, 0x0F, 0x1E, 0xFA], reg_opext: None, default_64bit: false,
        form: EncodingForm::NoOperands,
    });
    table.insert(X86Opcode::Pause.as_u32(), OpcodeEntry {
        opcode: &[0xF3, 0x90], reg_opext: None, default_64bit: false,
        form: EncodingForm::NoOperands,
    });
    table.insert(X86Opcode::Lfence.as_u32(), OpcodeEntry {
        opcode: &[0x0F, 0xAE, 0xE8], reg_opext: None, default_64bit: false,
        form: EncodingForm::NoOperands,
    });
    table.insert(X86Opcode::Neg.as_u32(), OpcodeEntry {
        opcode: &[0xF7], reg_opext: Some(3), default_64bit: true,
        form: EncodingForm::RmOnly,
    });
    table.insert(X86Opcode::Not.as_u32(), OpcodeEntry {
        opcode: &[0xF7], reg_opext: Some(2), default_64bit: true,
        form: EncodingForm::RmOnly,
    });
    table.insert(X86Opcode::Idiv.as_u32(), OpcodeEntry {
        opcode: &[0xF7], reg_opext: Some(7), default_64bit: true,
        form: EncodingForm::RmOnly,
    });
    table.insert(X86Opcode::Div.as_u32(), OpcodeEntry {
        opcode: &[0xF7], reg_opext: Some(6), default_64bit: true,
        form: EncodingForm::RmOnly,
    });

    table
}

// ===========================================================================
// Register Encoding Constants
// ===========================================================================

/// Registers whose hardware encoding & 7 == 4 require SIB byte.
const SIB_REQUIRED_ENC: u8 = 4;

/// Registers whose hardware encoding & 7 == 5 conflict with RIP-relative.
const RIP_CONFLICT_ENC: u8 = 5;

/// Compile-time verification of register encoding properties.
#[allow(dead_code)]
const _: () = {
    assert!(RSP as u8 % 8 == SIB_REQUIRED_ENC);
    assert!(R12 as u8 % 8 == SIB_REQUIRED_ENC);
    assert!(RBP as u8 % 8 == RIP_CONFLICT_ENC);
    assert!(R13 as u8 % 8 == RIP_CONFLICT_ENC);
};

// ===========================================================================
// X86_64Encoder — main encoder struct
// ===========================================================================

/// x86-64 instruction encoder.
///
/// Encodes x86-64 machine instructions into their binary representation,
/// handling REX prefixes, ModR/M bytes, SIB bytes, displacements, and
/// immediates. Tracks the current offset within the `.text` section for
/// relocation offset computation.
pub struct X86_64Encoder {
    /// Current byte offset in the `.text` section. Updated after each
    /// instruction is encoded so that relocation offsets are correct.
    pub current_offset: usize,
    /// Opcode table for O(1) lookup of encoding metadata.
    opcode_table: FxHashMap<u32, OpcodeEntry>,
}

impl X86_64Encoder {
    /// Create a new encoder starting at the given text section offset.
    pub fn new(current_offset: usize) -> Self {
        // Validate register encoding properties at runtime (debug only).
        debug_assert!(!needs_rex(RAX) && !needs_rex(RBX));
        debug_assert!(!needs_rex(RCX) && !needs_rex(RDX));
        debug_assert!(!needs_rex(RSI) && !needs_rex(RDI));
        debug_assert!(needs_rex(R8) && needs_rex(R9));
        debug_assert!(needs_rex(R10) && needs_rex(R11));
        debug_assert!(needs_rex(R14) && needs_rex(R15));
        debug_assert!(is_gpr(RAX) && is_sse(XMM0) && is_sse(XMM8));
        debug_assert_eq!(OPERAND_SIZE_PREFIX, 0x66);

        Self {
            current_offset,
            opcode_table: build_opcode_table(),
        }
    }

    // ===================================================================
    // Main Instruction Encoding Entry Point
    // ===================================================================

    /// Encode a single machine instruction into bytes.
    ///
    /// This is the main dispatch function called from `mod.rs` for each
    /// [`MachineInstruction`] in the function. It converts the opcode `u32`
    /// back to [`X86Opcode`], examines operand types, and delegates to
    /// the appropriate encoding helper.
    pub fn encode_instruction(
        &mut self,
        inst: &MachineInstruction,
    ) -> EncodedInstruction {
        let opcode = match X86Opcode::from_u32(inst.opcode) {
            Some(op) => op,
            None => {
                let result = EncodedInstruction::new(vec![0x90]);
                self.current_offset += result.bytes.len();
                return result;
            }
        };

        // Handle no-operand instructions via opcode table first.
        if let Some(entry) = self.opcode_table.get(&inst.opcode) {
            if entry.form == EncodingForm::NoOperands {
                let mut bytes = Vec::with_capacity(8);
                if entry.default_64bit {
                    bytes.push(rex_byte(true, false, false, false));
                }
                bytes.extend_from_slice(entry.opcode);
                let result = EncodedInstruction::new(bytes);
                self.current_offset += result.bytes.len();
                return result;
            }
        }

        let result = match opcode {
            // ALU binary ops
            X86Opcode::Add | X86Opcode::Sub | X86Opcode::And
            | X86Opcode::Or | X86Opcode::Xor | X86Opcode::Cmp => {
                let (opext, rm_reg_opc, reg_rm_opc) = alu_info(&opcode).unwrap();
                self.encode_alu_op(inst, opext, rm_reg_opc, reg_rm_opc)
            }
            X86Opcode::Test => self.encode_test(inst),
            X86Opcode::Mov => self.encode_mov(inst),
            X86Opcode::Lea => self.encode_lea(inst),
            X86Opcode::Shl | X86Opcode::Shr | X86Opcode::Sar => {
                self.encode_shift(inst, &opcode)
            }
            X86Opcode::Neg | X86Opcode::Not | X86Opcode::Idiv | X86Opcode::Div => {
                self.encode_unary(inst, &opcode)
            }
            X86Opcode::Imul => self.encode_imul(inst),
            X86Opcode::Push => {
                let reg = self.extract_register(&inst.operands, 0);
                self.encode_push_pop(0x50, reg)
            }
            X86Opcode::Pop => {
                let reg = self.extract_register(&inst.operands, 0);
                self.encode_push_pop(0x58, reg)
            }
            X86Opcode::MovZX => {
                self.encode_movzx_movsx(inst, &[0x0F, 0xB6], &[0x0F, 0xB7])
            }
            X86Opcode::MovSX => self.encode_movsx(inst),
            X86Opcode::Je | X86Opcode::Jne | X86Opcode::Jl
            | X86Opcode::Jle | X86Opcode::Jg | X86Opcode::Jge => {
                let cc = condition_code(&opcode);
                self.encode_jcc(inst, cc)
            }
            X86Opcode::Jmp => self.encode_jmp(inst),
            X86Opcode::Call => self.encode_call(inst),
            X86Opcode::Cmove | X86Opcode::Cmovne => {
                let cc = condition_code(&opcode);
                self.encode_cmovcc(inst, cc)
            }
            X86Opcode::Sete | X86Opcode::Setne => {
                let cc = condition_code(&opcode);
                let dst = self.extract_register(&inst.operands, 0);
                self.encode_setcc(inst, cc, dst)
            }
            X86Opcode::Movsd => self.encode_movsd(inst),
            X86Opcode::Movss => self.encode_movss(inst),
            X86Opcode::Addsd => self.encode_sse_op(inst, 0xF2, 0x58),
            X86Opcode::Subsd => self.encode_sse_op(inst, 0xF2, 0x5C),
            X86Opcode::Mulsd => self.encode_sse_op(inst, 0xF2, 0x59),
            X86Opcode::Divsd => self.encode_sse_op(inst, 0xF2, 0x5E),
            X86Opcode::Ucomisd => self.encode_ucomisd(inst),
            X86Opcode::Cvtsi2sd => self.encode_cvtsi2sd(inst),
            X86Opcode::Cvtsd2si => self.encode_cvtsd2si(inst),
            // Fixed-encoding instructions (fallback if not caught above)
            X86Opcode::Ret | X86Opcode::Nop | X86Opcode::Cdq
            | X86Opcode::Cqo | X86Opcode::Endbr64 | X86Opcode::Pause
            | X86Opcode::Lfence => {
                if let Some(entry) = self.opcode_table.get(&inst.opcode) {
                    let mut bytes = Vec::new();
                    if entry.default_64bit {
                        bytes.push(rex_byte(true, false, false, false));
                    }
                    bytes.extend_from_slice(entry.opcode);
                    EncodedInstruction::new(bytes)
                } else {
                    EncodedInstruction::new(vec![0x90])
                }
            }
            // Catch-all
            _ => EncodedInstruction::new(vec![0x90]),
        };

        self.current_offset += result.bytes.len();
        result
    }

    // ===================================================================
    // Public Encoding Helpers
    // ===================================================================

    /// Encode a two-register instruction (e.g., `add rax, rcx`).
    ///
    /// The source register goes in the ModR/M `reg` field and the
    /// destination register goes in the `r/m` field (mod=11).
    pub fn encode_reg_reg_op(
        &mut self,
        opcode: &[u8],
        dst: u16,
        src: u16,
        operand_size: u8,
    ) -> EncodedInstruction {
        let mut bytes = Vec::with_capacity(8);
        if operand_size == 2 {
            bytes.push(OPERAND_SIZE_PREFIX);
        }
        let need_64 = operand_size == 8;
        if let Some(rex) = compute_rex(need_64, Some(src), None, Some(dst)) {
            bytes.push(rex);
        }
        bytes.extend_from_slice(opcode);
        bytes.push(modrm_byte(0b11, hw_encoding(src) & 0x7, hw_encoding(dst) & 0x7));
        EncodedInstruction::new(bytes)
    }

    /// Encode a register-immediate instruction (e.g., `add rax, 42`).
    pub fn encode_reg_imm_op(
        &mut self,
        opcode: &[u8],
        opext: u8,
        dst: u16,
        imm: i64,
        operand_size: u8,
    ) -> EncodedInstruction {
        let mut bytes = Vec::with_capacity(12);
        if operand_size == 2 {
            bytes.push(OPERAND_SIZE_PREFIX);
        }
        let need_64 = operand_size == 8;
        if let Some(rex) = compute_rex(need_64, None, None, Some(dst)) {
            bytes.push(rex);
        }
        bytes.extend_from_slice(opcode);
        bytes.push(modrm_byte(0b11, opext, hw_encoding(dst) & 0x7));
        let imm_size = if opcode.last() == Some(&0x83) { 1 }
            else if operand_size <= 4 { operand_size.max(1) }
            else { 4 };
        bytes.extend_from_slice(&encode_immediate(imm, imm_size));
        EncodedInstruction::new(bytes)
    }

    /// Encode a register-memory instruction (e.g., `mov rax, [rbp-8]`).
    pub fn encode_reg_mem_op(
        &mut self,
        opcode: &[u8],
        reg: u16,
        base: Option<u16>,
        index: Option<u16>,
        scale: u8,
        displacement: i64,
        operand_size: u8,
    ) -> EncodedInstruction {
        let mut bytes = Vec::with_capacity(12);
        if operand_size == 2 {
            bytes.push(OPERAND_SIZE_PREFIX);
        }
        let reg_enc = hw_encoding(reg);
        let mem_enc = encode_memory_operand(reg_enc & 0x7, base, index, scale, displacement);
        let need_64 = operand_size == 8;
        if let Some(rex) = compute_rex(need_64, Some(reg), index, base) {
            bytes.push(rex);
        }
        bytes.extend_from_slice(opcode);
        bytes.push(mem_enc.modrm);
        if let Some(s) = mem_enc.sib {
            bytes.push(s);
        }
        bytes.extend_from_slice(&mem_enc.displacement);
        EncodedInstruction::new(bytes)
    }

    /// Encode a memory-register instruction (e.g., `mov [rbp-8], rax`).
    pub fn encode_mem_reg_op(
        &mut self,
        opcode: &[u8],
        base: Option<u16>,
        index: Option<u16>,
        scale: u8,
        displacement: i64,
        reg: u16,
        operand_size: u8,
    ) -> EncodedInstruction {
        let mut bytes = Vec::with_capacity(12);
        if operand_size == 2 {
            bytes.push(OPERAND_SIZE_PREFIX);
        }
        let reg_enc = hw_encoding(reg);
        let mem_enc = encode_memory_operand(reg_enc & 0x7, base, index, scale, displacement);
        let need_64 = operand_size == 8;
        if let Some(rex) = compute_rex(need_64, Some(reg), index, base) {
            bytes.push(rex);
        }
        bytes.extend_from_slice(opcode);
        bytes.push(mem_enc.modrm);
        if let Some(s) = mem_enc.sib {
            bytes.push(s);
        }
        bytes.extend_from_slice(&mem_enc.displacement);
        EncodedInstruction::new(bytes)
    }

    /// Encode a relative branch/call instruction (e.g., `jmp label`).
    pub fn encode_relative_op(
        &mut self,
        opcode: &[u8],
        target_offset: i64,
    ) -> EncodedInstruction {
        let mut bytes = Vec::with_capacity(8);
        bytes.extend_from_slice(opcode);
        let inst_size = (opcode.len() + 4) as i64;
        let rel32 = target_offset - (self.current_offset as i64 + inst_size);
        bytes.extend_from_slice(&(rel32 as i32).to_le_bytes());
        EncodedInstruction::new(bytes)
    }

    /// Encode a RIP-relative memory access (for PIC code).
    pub fn encode_rip_relative(
        &mut self,
        opcode: &[u8],
        reg: u16,
        symbol: &str,
        addend: i64,
    ) -> EncodedInstruction {
        let mut bytes = Vec::with_capacity(12);
        if let Some(rex) = compute_rex(true, Some(reg), None, None) {
            bytes.push(rex);
        }
        bytes.extend_from_slice(opcode);
        let reg_enc = hw_encoding(reg) & 0x7;
        bytes.push(modrm_byte(0b00, reg_enc, 0b101));
        let reloc_offset = self.current_offset + bytes.len();
        bytes.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
        let reloc = RelocationEntry::new(
            reloc_offset as u64,
            symbol.to_string(),
            X86_64RelocationType::Pc32,
            addend - 4,
            ".text".to_string(),
        );
        EncodedInstruction::with_relocations(bytes, vec![reloc])
    }

    /// Encode a PUSH or POP instruction using the opcode+rd form.
    pub fn encode_push_pop(
        &mut self,
        base_opcode: u8,
        reg: u16,
    ) -> EncodedInstruction {
        let mut bytes = Vec::with_capacity(2);
        if needs_rex(reg) {
            bytes.push(rex_byte(false, false, false, true));
        }
        bytes.push(base_opcode + (hw_encoding(reg) & 0x7));
        EncodedInstruction::new(bytes)
    }

    /// Encode an SSE register-register instruction with mandatory prefix.
    pub fn encode_sse_reg_reg(
        &mut self,
        prefix: u8,
        opcode: &[u8],
        dst: u16,
        src: u16,
    ) -> EncodedInstruction {
        let mut bytes = Vec::with_capacity(8);
        bytes.push(prefix);
        if let Some(rex) = compute_rex(false, Some(dst), None, Some(src)) {
            bytes.push(rex);
        }
        bytes.extend_from_slice(opcode);
        bytes.push(modrm_byte(0b11, hw_encoding(dst) & 0x7, hw_encoding(src) & 0x7));
        EncodedInstruction::new(bytes)
    }

    // ===================================================================
    // Private Encoding Helpers
    // ===================================================================

    /// Extract a register from the operand list at the given index.
    fn extract_register(&self, operands: &[MachineOperand], idx: usize) -> u16 {
        match operands.get(idx) {
            Some(MachineOperand::Register(r)) => *r,
            Some(MachineOperand::FrameSlot(_)) => RBP,
            _ => RAX,
        }
    }

    /// Encode an ALU binary operation (ADD, SUB, AND, OR, XOR, CMP).
    fn encode_alu_op(
        &mut self,
        inst: &MachineInstruction,
        opext: u8,
        rm_reg_opc: u8,
        reg_rm_opc: u8,
    ) -> EncodedInstruction {
        let ops = &inst.operands;
        match (ops.first(), ops.get(1)) {
            (Some(MachineOperand::Register(dst)), Some(MachineOperand::Register(src))) => {
                self.encode_reg_reg_op(&[rm_reg_opc], *dst, *src, 8)
            }
            (Some(MachineOperand::Register(dst)), Some(MachineOperand::Immediate(imm))) => {
                let imm = *imm;
                if (-128..=127).contains(&imm) {
                    self.encode_reg_imm_op(&[0x83], opext, *dst, imm, 8)
                } else {
                    self.encode_reg_imm_op(&[0x81], opext, *dst, imm, 8)
                }
            }
            (Some(MachineOperand::Register(reg)), Some(MachineOperand::Memory { base, index, scale, displacement })) => {
                self.encode_reg_mem_op(&[reg_rm_opc], *reg, *base, *index, *scale, *displacement, 8)
            }
            (Some(MachineOperand::Register(reg)), Some(MachineOperand::FrameSlot(offset))) => {
                self.encode_reg_mem_op(&[reg_rm_opc], *reg, Some(RBP), None, 1, *offset as i64, 8)
            }
            (Some(MachineOperand::Memory { base, index, scale, displacement }), Some(MachineOperand::Register(reg))) => {
                self.encode_mem_reg_op(&[rm_reg_opc], *base, *index, *scale, *displacement, *reg, 8)
            }
            (Some(MachineOperand::Memory { base, index, scale, displacement }), Some(MachineOperand::Immediate(imm))) => {
                let imm = *imm;
                let (base, index, scale, displacement) = (*base, *index, *scale, *displacement);
                let mut bytes = Vec::with_capacity(16);
                let opc = if (-128..=127).contains(&imm) { 0x83u8 } else { 0x81u8 };
                let mem_enc = encode_memory_operand(opext, base, index, scale, displacement);
                if let Some(rex) = compute_rex(true, None, index, base) {
                    bytes.push(rex);
                }
                bytes.push(opc);
                bytes.push(mem_enc.modrm);
                if let Some(s) = mem_enc.sib { bytes.push(s); }
                bytes.extend_from_slice(&mem_enc.displacement);
                let imm_size = if opc == 0x83 { 1 } else { 4 };
                bytes.extend_from_slice(&encode_immediate(imm, imm_size));
                EncodedInstruction::new(bytes)
            }
            _ => EncodedInstruction::new(vec![0x90]),
        }
    }

    /// Encode TEST instruction.
    fn encode_test(&mut self, inst: &MachineInstruction) -> EncodedInstruction {
        let ops = &inst.operands;
        match (ops.first(), ops.get(1)) {
            (Some(MachineOperand::Register(dst)), Some(MachineOperand::Register(src))) => {
                self.encode_reg_reg_op(&[0x85], *dst, *src, 8)
            }
            (Some(MachineOperand::Register(dst)), Some(MachineOperand::Immediate(imm))) => {
                self.encode_reg_imm_op(&[0xF7], 0, *dst, *imm, 8)
            }
            _ => EncodedInstruction::new(vec![0x90]),
        }
    }

    /// Encode MOV instruction — handles multiple forms.
    fn encode_mov(&mut self, inst: &MachineInstruction) -> EncodedInstruction {
        let ops = &inst.operands;
        match (ops.first(), ops.get(1)) {
            (Some(MachineOperand::Register(dst)), Some(MachineOperand::Register(src))) => {
                self.encode_reg_reg_op(&[0x89], *dst, *src, 8)
            }
            (Some(MachineOperand::Register(dst)), Some(MachineOperand::Immediate(imm))) => {
                let imm = *imm;
                let dst = *dst;
                if (i32::MIN as i64..=i32::MAX as i64).contains(&imm) {
                    self.encode_reg_imm_op(&[0xC7], 0, dst, imm, 8)
                } else {
                    // MOV r64, imm64: REX.W + 0xB8+rd + imm64
                    let mut bytes = Vec::with_capacity(10);
                    let rex = compute_rex(true, None, None, Some(dst))
                        .unwrap_or(rex_byte(true, false, false, false));
                    bytes.push(rex);
                    bytes.push(0xB8 + (hw_encoding(dst) & 0x7));
                    bytes.extend_from_slice(&imm.to_le_bytes());
                    EncodedInstruction::new(bytes)
                }
            }
            (Some(MachineOperand::Register(reg)), Some(MachineOperand::Memory { base, index, scale, displacement })) => {
                self.encode_reg_mem_op(&[0x8B], *reg, *base, *index, *scale, *displacement, 8)
            }
            (Some(MachineOperand::Register(reg)), Some(MachineOperand::FrameSlot(offset))) => {
                self.encode_reg_mem_op(&[0x8B], *reg, Some(RBP), None, 1, *offset as i64, 8)
            }
            (Some(MachineOperand::Memory { base, index, scale, displacement }), Some(MachineOperand::Register(reg))) => {
                self.encode_mem_reg_op(&[0x89], *base, *index, *scale, *displacement, *reg, 8)
            }
            (Some(MachineOperand::FrameSlot(offset)), Some(MachineOperand::Register(reg))) => {
                self.encode_mem_reg_op(&[0x89], Some(RBP), None, 1, *offset as i64, *reg, 8)
            }
            (Some(MachineOperand::Memory { base, index, scale, displacement }), Some(MachineOperand::Immediate(imm))) => {
                let (base, index, scale, displacement) = (*base, *index, *scale, *displacement);
                let mut bytes = Vec::with_capacity(16);
                let mem_enc = encode_memory_operand(0, base, index, scale, displacement);
                if let Some(rex) = compute_rex(true, None, index, base) {
                    bytes.push(rex);
                }
                bytes.push(0xC7);
                bytes.push(mem_enc.modrm);
                if let Some(s) = mem_enc.sib { bytes.push(s); }
                bytes.extend_from_slice(&mem_enc.displacement);
                bytes.extend_from_slice(&encode_immediate(*imm, 4));
                EncodedInstruction::new(bytes)
            }
            (Some(MachineOperand::Register(reg)), Some(MachineOperand::GlobalSymbol(sym))) => {
                self.encode_rip_relative(&[0x8B], *reg, sym, 0)
            }
            (Some(MachineOperand::GlobalSymbol(sym)), Some(MachineOperand::Register(reg))) => {
                let sym_clone = sym.clone();
                let reg = *reg;
                let mut bytes = Vec::with_capacity(12);
                if let Some(rex) = compute_rex(true, Some(reg), None, None) {
                    bytes.push(rex);
                }
                bytes.push(0x89);
                let reg_enc = hw_encoding(reg) & 0x7;
                bytes.push(modrm_byte(0b00, reg_enc, 0b101));
                let reloc_offset = self.current_offset + bytes.len();
                bytes.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
                let reloc = RelocationEntry::new(
                    reloc_offset as u64, sym_clone,
                    X86_64RelocationType::Pc32, -4, ".text".to_string(),
                );
                EncodedInstruction::with_relocations(bytes, vec![reloc])
            }
            _ => EncodedInstruction::new(vec![0x90]),
        }
    }

    /// Encode LEA instruction.
    fn encode_lea(&mut self, inst: &MachineInstruction) -> EncodedInstruction {
        let ops = &inst.operands;
        match (ops.first(), ops.get(1)) {
            (Some(MachineOperand::Register(reg)), Some(MachineOperand::Memory { base, index, scale, displacement })) => {
                self.encode_reg_mem_op(&[0x8D], *reg, *base, *index, *scale, *displacement, 8)
            }
            (Some(MachineOperand::Register(reg)), Some(MachineOperand::FrameSlot(offset))) => {
                self.encode_reg_mem_op(&[0x8D], *reg, Some(RBP), None, 1, *offset as i64, 8)
            }
            (Some(MachineOperand::Register(reg)), Some(MachineOperand::GlobalSymbol(sym))) => {
                self.encode_rip_relative(&[0x8D], *reg, sym, 0)
            }
            _ => EncodedInstruction::new(vec![0x90]),
        }
    }

    /// Encode shift instruction (SHL, SHR, SAR).
    fn encode_shift(&mut self, inst: &MachineInstruction, opcode: &X86Opcode) -> EncodedInstruction {
        let opext = shift_opext(opcode);
        let ops = &inst.operands;
        match (ops.first(), ops.get(1)) {
            (Some(MachineOperand::Register(dst)), Some(MachineOperand::Register(cl)))
                if *cl == RCX =>
            {
                let mut bytes = Vec::with_capacity(4);
                if let Some(rex) = compute_rex(true, None, None, Some(*dst)) {
                    bytes.push(rex);
                }
                bytes.push(0xD3);
                bytes.push(modrm_byte(0b11, opext, hw_encoding(*dst) & 0x7));
                EncodedInstruction::new(bytes)
            }
            (Some(MachineOperand::Register(dst)), Some(MachineOperand::Immediate(imm))) => {
                let mut bytes = Vec::with_capacity(5);
                if let Some(rex) = compute_rex(true, None, None, Some(*dst)) {
                    bytes.push(rex);
                }
                if *imm == 1 {
                    bytes.push(0xD1);
                    bytes.push(modrm_byte(0b11, opext, hw_encoding(*dst) & 0x7));
                } else {
                    bytes.push(0xC1);
                    bytes.push(modrm_byte(0b11, opext, hw_encoding(*dst) & 0x7));
                    bytes.push(*imm as u8);
                }
                EncodedInstruction::new(bytes)
            }
            _ => EncodedInstruction::new(vec![0x90]),
        }
    }

    /// Encode unary instruction (NEG, NOT, IDIV, DIV).
    fn encode_unary(&mut self, inst: &MachineInstruction, opcode: &X86Opcode) -> EncodedInstruction {
        let entry = self.opcode_table.get(&opcode.as_u32());
        let (opc_byte, opext) = match entry {
            Some(e) => (e.opcode[0], e.reg_opext.unwrap_or(0)),
            None => (0xF7, 0),
        };
        let ops = &inst.operands;
        match ops.first() {
            Some(MachineOperand::Register(reg)) => {
                let mut bytes = Vec::with_capacity(4);
                if let Some(rex) = compute_rex(true, None, None, Some(*reg)) {
                    bytes.push(rex);
                }
                bytes.push(opc_byte);
                bytes.push(modrm_byte(0b11, opext, hw_encoding(*reg) & 0x7));
                EncodedInstruction::new(bytes)
            }
            Some(MachineOperand::Memory { base, index, scale, displacement }) => {
                let mut bytes = Vec::with_capacity(12);
                let mem_enc = encode_memory_operand(opext, *base, *index, *scale, *displacement);
                if let Some(rex) = compute_rex(true, None, *index, *base) {
                    bytes.push(rex);
                }
                bytes.push(opc_byte);
                bytes.push(mem_enc.modrm);
                if let Some(s) = mem_enc.sib { bytes.push(s); }
                bytes.extend_from_slice(&mem_enc.displacement);
                EncodedInstruction::new(bytes)
            }
            _ => EncodedInstruction::new(vec![0x90]),
        }
    }

    /// Encode IMUL instruction.
    fn encode_imul(&mut self, inst: &MachineInstruction) -> EncodedInstruction {
        let ops = &inst.operands;
        match (ops.first(), ops.get(1), ops.get(2)) {
            (Some(MachineOperand::Register(dst)), Some(MachineOperand::Register(src)), None) => {
                let mut bytes = Vec::with_capacity(5);
                if let Some(rex) = compute_rex(true, Some(*dst), None, Some(*src)) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(0xAF);
                bytes.push(modrm_byte(0b11, hw_encoding(*dst) & 0x7, hw_encoding(*src) & 0x7));
                EncodedInstruction::new(bytes)
            }
            (Some(MachineOperand::Register(dst)), Some(MachineOperand::Register(src)), Some(MachineOperand::Immediate(imm))) => {
                let imm = *imm;
                let mut bytes = Vec::with_capacity(8);
                if let Some(rex) = compute_rex(true, Some(*dst), None, Some(*src)) {
                    bytes.push(rex);
                }
                if (-128..=127).contains(&imm) {
                    bytes.push(0x6B);
                    bytes.push(modrm_byte(0b11, hw_encoding(*dst) & 0x7, hw_encoding(*src) & 0x7));
                    bytes.push(imm as u8);
                } else {
                    bytes.push(0x69);
                    bytes.push(modrm_byte(0b11, hw_encoding(*dst) & 0x7, hw_encoding(*src) & 0x7));
                    bytes.extend_from_slice(&(imm as i32).to_le_bytes());
                }
                EncodedInstruction::new(bytes)
            }
            (Some(MachineOperand::Register(dst)), Some(MachineOperand::Memory { base, index, scale, displacement }), None) => {
                let mut bytes = Vec::with_capacity(10);
                let mem_enc = encode_memory_operand(hw_encoding(*dst) & 0x7, *base, *index, *scale, *displacement);
                if let Some(rex) = compute_rex(true, Some(*dst), *index, *base) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(0xAF);
                bytes.push(mem_enc.modrm);
                if let Some(s) = mem_enc.sib { bytes.push(s); }
                bytes.extend_from_slice(&mem_enc.displacement);
                EncodedInstruction::new(bytes)
            }
            _ => EncodedInstruction::new(vec![0x90]),
        }
    }

    /// Encode MOVZX.
    fn encode_movzx_movsx(&mut self, inst: &MachineInstruction, byte_opc: &[u8], _word_opc: &[u8]) -> EncodedInstruction {
        let ops = &inst.operands;
        match (ops.first(), ops.get(1)) {
            (Some(MachineOperand::Register(dst)), Some(MachineOperand::Register(src))) => {
                let mut bytes = Vec::with_capacity(5);
                if let Some(rex) = compute_rex(false, Some(*dst), None, Some(*src)) {
                    bytes.push(rex);
                }
                bytes.extend_from_slice(byte_opc);
                bytes.push(modrm_byte(0b11, hw_encoding(*dst) & 0x7, hw_encoding(*src) & 0x7));
                EncodedInstruction::new(bytes)
            }
            (Some(MachineOperand::Register(dst)), Some(MachineOperand::Memory { base, index, scale, displacement })) => {
                let mem_enc = encode_memory_operand(hw_encoding(*dst) & 0x7, *base, *index, *scale, *displacement);
                let mut bytes = Vec::with_capacity(10);
                if let Some(rex) = compute_rex(false, Some(*dst), *index, *base) {
                    bytes.push(rex);
                }
                bytes.extend_from_slice(byte_opc);
                bytes.push(mem_enc.modrm);
                if let Some(s) = mem_enc.sib { bytes.push(s); }
                bytes.extend_from_slice(&mem_enc.displacement);
                EncodedInstruction::new(bytes)
            }
            _ => EncodedInstruction::new(vec![0x90]),
        }
    }

    /// Encode MOVSX / MOVSXD.
    fn encode_movsx(&mut self, inst: &MachineInstruction) -> EncodedInstruction {
        let ops = &inst.operands;
        match (ops.first(), ops.get(1)) {
            (Some(MachineOperand::Register(dst)), Some(MachineOperand::Register(src))) => {
                let mut bytes = Vec::with_capacity(5);
                if let Some(rex) = compute_rex(true, Some(*dst), None, Some(*src)) {
                    bytes.push(rex);
                }
                bytes.push(0x63); // MOVSXD
                bytes.push(modrm_byte(0b11, hw_encoding(*dst) & 0x7, hw_encoding(*src) & 0x7));
                EncodedInstruction::new(bytes)
            }
            (Some(MachineOperand::Register(dst)), Some(MachineOperand::Memory { base, index, scale, displacement })) => {
                let mem_enc = encode_memory_operand(hw_encoding(*dst) & 0x7, *base, *index, *scale, *displacement);
                let mut bytes = Vec::with_capacity(10);
                if let Some(rex) = compute_rex(true, Some(*dst), *index, *base) {
                    bytes.push(rex);
                }
                bytes.push(0x63);
                bytes.push(mem_enc.modrm);
                if let Some(s) = mem_enc.sib { bytes.push(s); }
                bytes.extend_from_slice(&mem_enc.displacement);
                EncodedInstruction::new(bytes)
            }
            _ => EncodedInstruction::new(vec![0x90]),
        }
    }

    /// Encode conditional jump (Jcc rel32).
    fn encode_jcc(&mut self, inst: &MachineInstruction, cc: u8) -> EncodedInstruction {
        let ops = &inst.operands;
        match ops.first() {
            Some(MachineOperand::Immediate(target)) => {
                // Jcc rel32: 0x0F 0x80+cc rel32
                let mut bytes = Vec::with_capacity(6);
                bytes.push(0x0F);
                bytes.push(0x80 + cc);
                let rel = *target - (self.current_offset as i64 + 6); // 6 = instruction length
                bytes.extend_from_slice(&(rel as i32).to_le_bytes());
                EncodedInstruction::new(bytes)
            }
            Some(MachineOperand::BlockLabel(label)) => {
                // Jcc rel32 with placeholder — will need patching
                let mut bytes = Vec::with_capacity(6);
                bytes.push(0x0F);
                bytes.push(0x80 + cc);
                bytes.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // placeholder
                let reloc = RelocationEntry::new(
                    (self.current_offset + 2) as u64,
                    format!(".L{}", label),
                    X86_64RelocationType::Pc32,
                    -4,
                    ".text".to_string(),
                );
                EncodedInstruction::with_relocations(bytes, vec![reloc])
            }
            Some(MachineOperand::GlobalSymbol(sym)) => {
                let mut bytes = Vec::with_capacity(6);
                bytes.push(0x0F);
                bytes.push(0x80 + cc);
                bytes.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
                let reloc = RelocationEntry::new(
                    (self.current_offset + 2) as u64,
                    sym.clone(),
                    X86_64RelocationType::Pc32,
                    -4,
                    ".text".to_string(),
                );
                EncodedInstruction::with_relocations(bytes, vec![reloc])
            }
            _ => EncodedInstruction::new(vec![0x90]),
        }
    }

    /// Encode JMP instruction.
    fn encode_jmp(&mut self, inst: &MachineInstruction) -> EncodedInstruction {
        let ops = &inst.operands;
        match ops.first() {
            Some(MachineOperand::Immediate(target)) => {
                // JMP rel32: 0xE9 rel32
                let mut bytes = Vec::with_capacity(5);
                bytes.push(0xE9);
                let rel = *target - (self.current_offset as i64 + 5);
                bytes.extend_from_slice(&(rel as i32).to_le_bytes());
                EncodedInstruction::new(bytes)
            }
            Some(MachineOperand::BlockLabel(label)) => {
                let mut bytes = Vec::with_capacity(5);
                bytes.push(0xE9);
                bytes.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
                let reloc = RelocationEntry::new(
                    (self.current_offset + 1) as u64,
                    format!(".L{}", label),
                    X86_64RelocationType::Pc32,
                    -4,
                    ".text".to_string(),
                );
                EncodedInstruction::with_relocations(bytes, vec![reloc])
            }
            Some(MachineOperand::GlobalSymbol(sym)) => {
                let mut bytes = Vec::with_capacity(5);
                bytes.push(0xE9);
                bytes.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
                let reloc = RelocationEntry::new(
                    (self.current_offset + 1) as u64,
                    sym.clone(),
                    X86_64RelocationType::Pc32,
                    -4,
                    ".text".to_string(),
                );
                EncodedInstruction::with_relocations(bytes, vec![reloc])
            }
            Some(MachineOperand::Register(reg)) => {
                // JMP r/m64: 0xFF /4
                let mut bytes = Vec::with_capacity(3);
                if needs_rex(*reg) {
                    bytes.push(rex_byte(false, false, false, true));
                }
                bytes.push(0xFF);
                bytes.push(modrm_byte(0b11, 4, hw_encoding(*reg) & 0x7));
                EncodedInstruction::new(bytes)
            }
            _ => EncodedInstruction::new(vec![0x90]),
        }
    }

    /// Encode CALL instruction.
    fn encode_call(&mut self, inst: &MachineInstruction) -> EncodedInstruction {
        let ops = &inst.operands;
        match ops.first() {
            Some(MachineOperand::GlobalSymbol(sym)) => {
                // CALL rel32: 0xE8 rel32
                let mut bytes = Vec::with_capacity(5);
                bytes.push(0xE8);
                let reloc_offset = self.current_offset + 1;
                bytes.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
                let reloc = RelocationEntry::pc32_call(
                    reloc_offset as u64,
                    sym.clone(),
                    true, // use PLT for PIC safety
                );
                EncodedInstruction::with_relocations(bytes, vec![reloc])
            }
            Some(MachineOperand::Register(reg)) => {
                // CALL r/m64: 0xFF /2
                let mut bytes = Vec::with_capacity(3);
                if needs_rex(*reg) {
                    bytes.push(rex_byte(false, false, false, true));
                }
                bytes.push(0xFF);
                bytes.push(modrm_byte(0b11, 2, hw_encoding(*reg) & 0x7));
                EncodedInstruction::new(bytes)
            }
            Some(MachineOperand::Immediate(target)) => {
                // CALL rel32 with absolute target
                let mut bytes = Vec::with_capacity(5);
                bytes.push(0xE8);
                let rel = *target - (self.current_offset as i64 + 5);
                bytes.extend_from_slice(&(rel as i32).to_le_bytes());
                EncodedInstruction::new(bytes)
            }
            Some(MachineOperand::Memory { base, index, scale, displacement }) => {
                // CALL [mem]: 0xFF /2
                let mem_enc = encode_memory_operand(2, *base, *index, *scale, *displacement);
                let mut bytes = Vec::with_capacity(10);
                if let Some(rex) = compute_rex(false, None, *index, *base) {
                    bytes.push(rex);
                }
                bytes.push(0xFF);
                bytes.push(mem_enc.modrm);
                if let Some(s) = mem_enc.sib { bytes.push(s); }
                bytes.extend_from_slice(&mem_enc.displacement);
                EncodedInstruction::new(bytes)
            }
            _ => EncodedInstruction::new(vec![0x90]),
        }
    }

    /// Encode CMOVcc instruction.
    fn encode_cmovcc(&mut self, inst: &MachineInstruction, cc: u8) -> EncodedInstruction {
        let ops = &inst.operands;
        match (ops.first(), ops.get(1)) {
            (Some(MachineOperand::Register(dst)), Some(MachineOperand::Register(src))) => {
                let mut bytes = Vec::with_capacity(5);
                if let Some(rex) = compute_rex(true, Some(*dst), None, Some(*src)) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(0x40 + cc);
                bytes.push(modrm_byte(0b11, hw_encoding(*dst) & 0x7, hw_encoding(*src) & 0x7));
                EncodedInstruction::new(bytes)
            }
            (Some(MachineOperand::Register(dst)), Some(MachineOperand::Memory { base, index, scale, displacement })) => {
                let mem_enc = encode_memory_operand(hw_encoding(*dst) & 0x7, *base, *index, *scale, *displacement);
                let mut bytes = Vec::with_capacity(10);
                if let Some(rex) = compute_rex(true, Some(*dst), *index, *base) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(0x40 + cc);
                bytes.push(mem_enc.modrm);
                if let Some(s) = mem_enc.sib { bytes.push(s); }
                bytes.extend_from_slice(&mem_enc.displacement);
                EncodedInstruction::new(bytes)
            }
            _ => EncodedInstruction::new(vec![0x90]),
        }
    }

    /// Encode SETcc instruction.
    fn encode_setcc(&mut self, _inst: &MachineInstruction, cc: u8, dst: u16) -> EncodedInstruction {
        let mut bytes = Vec::with_capacity(5);
        // SETcc needs REX if the destination is one of the high byte regs or R8-R15
        if needs_rex(dst) {
            bytes.push(rex_byte(false, false, false, true));
        } else {
            // Need REX.=0 for SPL/BPL/SIL/DIL access in byte operations
            let hw = hw_encoding(dst);
            if (4..=7).contains(&hw) {
                bytes.push(0x40); // plain REX to access low byte of RSP/RBP/RSI/RDI
            }
        }
        bytes.push(0x0F);
        bytes.push(0x90 + cc);
        bytes.push(modrm_byte(0b11, 0, hw_encoding(dst) & 0x7));
        EncodedInstruction::new(bytes)
    }

    /// Encode MOVSD (scalar double).
    fn encode_movsd(&mut self, inst: &MachineInstruction) -> EncodedInstruction {
        let ops = &inst.operands;
        match (ops.first(), ops.get(1)) {
            (Some(MachineOperand::Register(dst)), Some(MachineOperand::Register(src))) if is_sse(*dst) && is_sse(*src) => {
                self.encode_sse_reg_reg(0xF2, &[0x0F, 0x10], *dst, *src)
            }
            (Some(MachineOperand::Register(xmm)), Some(MachineOperand::Memory { base, index, scale, displacement })) if is_sse(*xmm) => {
                // MOVSD xmm, [mem]: F2 0F 10 /r
                let mem_enc = encode_memory_operand(hw_encoding(*xmm) & 0x7, *base, *index, *scale, *displacement);
                let mut bytes = Vec::with_capacity(12);
                bytes.push(0xF2);
                if let Some(rex) = compute_rex(false, Some(*xmm), *index, *base) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(0x10);
                bytes.push(mem_enc.modrm);
                if let Some(s) = mem_enc.sib { bytes.push(s); }
                bytes.extend_from_slice(&mem_enc.displacement);
                EncodedInstruction::new(bytes)
            }
            (Some(MachineOperand::Memory { base, index, scale, displacement }), Some(MachineOperand::Register(xmm))) if is_sse(*xmm) => {
                // MOVSD [mem], xmm: F2 0F 11 /r
                let mem_enc = encode_memory_operand(hw_encoding(*xmm) & 0x7, *base, *index, *scale, *displacement);
                let mut bytes = Vec::with_capacity(12);
                bytes.push(0xF2);
                if let Some(rex) = compute_rex(false, Some(*xmm), *index, *base) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(0x11);
                bytes.push(mem_enc.modrm);
                if let Some(s) = mem_enc.sib { bytes.push(s); }
                bytes.extend_from_slice(&mem_enc.displacement);
                EncodedInstruction::new(bytes)
            }
            (Some(MachineOperand::Register(xmm)), Some(MachineOperand::FrameSlot(off))) if is_sse(*xmm) => {
                let mem_enc = encode_memory_operand(hw_encoding(*xmm) & 0x7, Some(RBP), None, 1, *off as i64);
                let mut bytes = Vec::with_capacity(12);
                bytes.push(0xF2);
                if let Some(rex) = compute_rex(false, Some(*xmm), None, Some(RBP)) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(0x10);
                bytes.push(mem_enc.modrm);
                if let Some(s) = mem_enc.sib { bytes.push(s); }
                bytes.extend_from_slice(&mem_enc.displacement);
                EncodedInstruction::new(bytes)
            }
            (Some(MachineOperand::FrameSlot(off)), Some(MachineOperand::Register(xmm))) if is_sse(*xmm) => {
                let mem_enc = encode_memory_operand(hw_encoding(*xmm) & 0x7, Some(RBP), None, 1, *off as i64);
                let mut bytes = Vec::with_capacity(12);
                bytes.push(0xF2);
                if let Some(rex) = compute_rex(false, Some(*xmm), None, Some(RBP)) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(0x11);
                bytes.push(mem_enc.modrm);
                if let Some(s) = mem_enc.sib { bytes.push(s); }
                bytes.extend_from_slice(&mem_enc.displacement);
                EncodedInstruction::new(bytes)
            }
            _ => EncodedInstruction::new(vec![0x90]),
        }
    }

    /// Encode MOVSS (scalar single).
    fn encode_movss(&mut self, inst: &MachineInstruction) -> EncodedInstruction {
        let ops = &inst.operands;
        match (ops.first(), ops.get(1)) {
            (Some(MachineOperand::Register(dst)), Some(MachineOperand::Register(src))) if is_sse(*dst) && is_sse(*src) => {
                self.encode_sse_reg_reg(0xF3, &[0x0F, 0x10], *dst, *src)
            }
            (Some(MachineOperand::Register(xmm)), Some(MachineOperand::Memory { base, index, scale, displacement })) if is_sse(*xmm) => {
                let mem_enc = encode_memory_operand(hw_encoding(*xmm) & 0x7, *base, *index, *scale, *displacement);
                let mut bytes = Vec::with_capacity(12);
                bytes.push(0xF3);
                if let Some(rex) = compute_rex(false, Some(*xmm), *index, *base) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(0x10);
                bytes.push(mem_enc.modrm);
                if let Some(s) = mem_enc.sib { bytes.push(s); }
                bytes.extend_from_slice(&mem_enc.displacement);
                EncodedInstruction::new(bytes)
            }
            (Some(MachineOperand::Memory { base, index, scale, displacement }), Some(MachineOperand::Register(xmm))) if is_sse(*xmm) => {
                let mem_enc = encode_memory_operand(hw_encoding(*xmm) & 0x7, *base, *index, *scale, *displacement);
                let mut bytes = Vec::with_capacity(12);
                bytes.push(0xF3);
                if let Some(rex) = compute_rex(false, Some(*xmm), *index, *base) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(0x11);
                bytes.push(mem_enc.modrm);
                if let Some(s) = mem_enc.sib { bytes.push(s); }
                bytes.extend_from_slice(&mem_enc.displacement);
                EncodedInstruction::new(bytes)
            }
            _ => EncodedInstruction::new(vec![0x90]),
        }
    }

    /// Encode a generic SSE arithmetic operation (addsd, subsd, mulsd, divsd, etc.).
    fn encode_sse_op(&mut self, inst: &MachineInstruction, prefix: u8, opc_suffix: u8) -> EncodedInstruction {
        let ops = &inst.operands;
        match (ops.first(), ops.get(1)) {
            (Some(MachineOperand::Register(dst)), Some(MachineOperand::Register(src))) if is_sse(*dst) && is_sse(*src) => {
                self.encode_sse_reg_reg(prefix, &[0x0F, opc_suffix], *dst, *src)
            }
            (Some(MachineOperand::Register(xmm)), Some(MachineOperand::Memory { base, index, scale, displacement })) if is_sse(*xmm) => {
                let mem_enc = encode_memory_operand(hw_encoding(*xmm) & 0x7, *base, *index, *scale, *displacement);
                let mut bytes = Vec::with_capacity(12);
                bytes.push(prefix);
                if let Some(rex) = compute_rex(false, Some(*xmm), *index, *base) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(opc_suffix);
                bytes.push(mem_enc.modrm);
                if let Some(s) = mem_enc.sib { bytes.push(s); }
                bytes.extend_from_slice(&mem_enc.displacement);
                EncodedInstruction::new(bytes)
            }
            _ => EncodedInstruction::new(vec![0x90]),
        }
    }

    /// Encode UCOMISD.
    fn encode_ucomisd(&mut self, inst: &MachineInstruction) -> EncodedInstruction {
        let ops = &inst.operands;
        match (ops.first(), ops.get(1)) {
            (Some(MachineOperand::Register(dst)), Some(MachineOperand::Register(src))) if is_sse(*dst) && is_sse(*src) => {
                let mut bytes = Vec::with_capacity(6);
                bytes.push(0x66);
                if let Some(rex) = compute_rex(false, Some(*dst), None, Some(*src)) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(0x2E);
                bytes.push(modrm_byte(0b11, hw_encoding(*dst) & 0x7, hw_encoding(*src) & 0x7));
                EncodedInstruction::new(bytes)
            }
            (Some(MachineOperand::Register(xmm)), Some(MachineOperand::Memory { base, index, scale, displacement })) if is_sse(*xmm) => {
                let mem_enc = encode_memory_operand(hw_encoding(*xmm) & 0x7, *base, *index, *scale, *displacement);
                let mut bytes = Vec::with_capacity(12);
                bytes.push(0x66);
                if let Some(rex) = compute_rex(false, Some(*xmm), *index, *base) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(0x2E);
                bytes.push(mem_enc.modrm);
                if let Some(s) = mem_enc.sib { bytes.push(s); }
                bytes.extend_from_slice(&mem_enc.displacement);
                EncodedInstruction::new(bytes)
            }
            _ => EncodedInstruction::new(vec![0x90]),
        }
    }

    /// Encode CVTSI2SD (integer to double).
    fn encode_cvtsi2sd(&mut self, inst: &MachineInstruction) -> EncodedInstruction {
        let ops = &inst.operands;
        match (ops.first(), ops.get(1)) {
            (Some(MachineOperand::Register(xmm)), Some(MachineOperand::Register(gpr))) if is_sse(*xmm) => {
                let mut bytes = Vec::with_capacity(6);
                bytes.push(0xF2);
                if let Some(rex) = compute_rex(true, Some(*xmm), None, Some(*gpr)) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(0x2A);
                bytes.push(modrm_byte(0b11, hw_encoding(*xmm) & 0x7, hw_encoding(*gpr) & 0x7));
                EncodedInstruction::new(bytes)
            }
            _ => EncodedInstruction::new(vec![0x90]),
        }
    }

    /// Encode CVTSD2SI (double to integer).
    fn encode_cvtsd2si(&mut self, inst: &MachineInstruction) -> EncodedInstruction {
        let ops = &inst.operands;
        match (ops.first(), ops.get(1)) {
            (Some(MachineOperand::Register(gpr)), Some(MachineOperand::Register(xmm))) if is_sse(*xmm) => {
                let mut bytes = Vec::with_capacity(6);
                bytes.push(0xF2);
                if let Some(rex) = compute_rex(true, Some(*gpr), None, Some(*xmm)) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(0x2D);
                bytes.push(modrm_byte(0b11, hw_encoding(*gpr) & 0x7, hw_encoding(*xmm) & 0x7));
                EncodedInstruction::new(bytes)
            }
            _ => EncodedInstruction::new(vec![0x90]),
        }
    }

} // end impl X86_64Encoder

// =====================================================================
// Unit Tests
// =====================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rex_byte() {
        assert_eq!(rex_byte(false, false, false, false), 0x40);
        assert_eq!(rex_byte(true, false, false, false), 0x48);
        assert_eq!(rex_byte(false, true, false, false), 0x44);
        assert_eq!(rex_byte(false, false, true, false), 0x42);
        assert_eq!(rex_byte(false, false, false, true), 0x41);
        assert_eq!(rex_byte(true, true, true, true), 0x4F);
    }

    #[test]
    fn test_modrm_byte() {
        // mod=11 reg=000 r/m=001 => 0b11_000_001 = 0xC1
        assert_eq!(modrm_byte(0b11, 0, 1), 0xC1);
        // mod=00 reg=101 r/m=100 => 0b00_101_100 = 0x2C
        assert_eq!(modrm_byte(0b00, 5, 4), 0x2C);
        // mod=01 reg=011 r/m=101 => 0b01_011_101 = 0x5D
        assert_eq!(modrm_byte(0b01, 3, 5), 0x5D);
    }

    #[test]
    fn test_sib_byte() {
        // scale=00 index=100 base=101 => 0b00_100_101 = 0x25
        assert_eq!(sib_byte(0b00, 4, 5), 0x25);
        // scale=10 index=001 base=000 => 0b10_001_000 = 0x88
        assert_eq!(sib_byte(0b10, 1, 0), 0x88);
    }

    #[test]
    fn test_scale_encoding() {
        assert_eq!(scale_encoding(1), 0b00);
        assert_eq!(scale_encoding(2), 0b01);
        assert_eq!(scale_encoding(4), 0b10);
        assert_eq!(scale_encoding(8), 0b11);
    }

    #[test]
    fn test_encode_displacement_i8() {
        let d = encode_displacement(10);
        assert_eq!(d, vec![10u8]);
    }

    #[test]
    fn test_encode_displacement_i32() {
        let d = encode_displacement(256);
        assert_eq!(d, (256i32).to_le_bytes().to_vec());
    }

    #[test]
    fn test_encode_immediate_sizes() {
        assert_eq!(encode_immediate(0x12, 1), vec![0x12]);
        assert_eq!(encode_immediate(0x1234, 2), vec![0x34, 0x12]);
        assert_eq!(encode_immediate(0x12345678, 4), vec![0x78, 0x56, 0x34, 0x12]);
    }

    #[test]
    fn test_compute_rex_none_needed() {
        assert!(compute_rex(false, Some(RAX), None, Some(RBX)).is_none());
    }

    #[test]
    fn test_compute_rex_w_only() {
        let rex = compute_rex(true, Some(RAX), None, Some(RBX));
        assert_eq!(rex, Some(0x48));
    }

    #[test]
    fn test_compute_rex_b_set() {
        // R8 needs REX.B
        let rex = compute_rex(true, Some(RAX), None, Some(R8));
        assert_eq!(rex, Some(0x49)); // W + B
    }

    #[test]
    fn test_compute_rex_r_set() {
        // R10 in reg field needs REX.R
        let rex = compute_rex(true, Some(R10), None, Some(RAX));
        assert_eq!(rex, Some(0x4C)); // W + R
    }

    #[test]
    fn test_encoder_nop() {
        let mut enc = X86_64Encoder::new(0);
        let inst = MachineInstruction::new(X86Opcode::Nop.as_u32());
        let result = enc.encode_instruction(&inst);
        assert_eq!(result.bytes, vec![0x90]);
    }

    #[test]
    fn test_encoder_ret() {
        let mut enc = X86_64Encoder::new(0);
        let inst = MachineInstruction::new(X86Opcode::Ret.as_u32());
        let result = enc.encode_instruction(&inst);
        assert_eq!(result.bytes, vec![0xC3]);
    }

    #[test]
    fn test_encoder_push_rax() {
        let mut enc = X86_64Encoder::new(0);
        let mut inst = MachineInstruction::new(X86Opcode::Push.as_u32());
        inst.operands.push(MachineOperand::Register(RAX));
        let result = enc.encode_instruction(&inst);
        assert_eq!(result.bytes, vec![0x50]);
    }

    #[test]
    fn test_encoder_push_r12() {
        let mut enc = X86_64Encoder::new(0);
        let mut inst = MachineInstruction::new(X86Opcode::Push.as_u32());
        inst.operands.push(MachineOperand::Register(R12));
        let result = enc.encode_instruction(&inst);
        // R12 hw_encoding = 12, lower 3 bits = 4, REX.B needed
        assert_eq!(result.bytes, vec![0x41, 0x50 + 4]);
    }

    #[test]
    fn test_encoder_pop_rbp() {
        let mut enc = X86_64Encoder::new(0);
        let mut inst = MachineInstruction::new(X86Opcode::Pop.as_u32());
        inst.operands.push(MachineOperand::Register(RBP));
        let result = enc.encode_instruction(&inst);
        assert_eq!(result.bytes, vec![0x58 + 5]); // 0x5D
    }

    #[test]
    fn test_encoder_cdq() {
        let mut enc = X86_64Encoder::new(0);
        let inst = MachineInstruction::new(X86Opcode::Cdq.as_u32());
        let result = enc.encode_instruction(&inst);
        assert_eq!(result.bytes, vec![0x99]);
    }

    #[test]
    fn test_encoder_cqo() {
        let mut enc = X86_64Encoder::new(0);
        let inst = MachineInstruction::new(X86Opcode::Cqo.as_u32());
        let result = enc.encode_instruction(&inst);
        assert_eq!(result.bytes, vec![0x48, 0x99]);
    }

    #[test]
    fn test_encoder_endbr64() {
        let mut enc = X86_64Encoder::new(0);
        let inst = MachineInstruction::new(X86Opcode::Endbr64.as_u32());
        let result = enc.encode_instruction(&inst);
        assert_eq!(result.bytes, vec![0xF3, 0x0F, 0x1E, 0xFA]);
    }

    #[test]
    fn test_encoder_pause() {
        let mut enc = X86_64Encoder::new(0);
        let inst = MachineInstruction::new(X86Opcode::Pause.as_u32());
        let result = enc.encode_instruction(&inst);
        assert_eq!(result.bytes, vec![0xF3, 0x90]);
    }

    #[test]
    fn test_encoder_lfence() {
        let mut enc = X86_64Encoder::new(0);
        let inst = MachineInstruction::new(X86Opcode::Lfence.as_u32());
        let result = enc.encode_instruction(&inst);
        assert_eq!(result.bytes, vec![0x0F, 0xAE, 0xE8]);
    }

    #[test]
    fn test_encoder_add_reg_reg() {
        let mut enc = X86_64Encoder::new(0);
        let mut inst = MachineInstruction::new(X86Opcode::Add.as_u32());
        inst.operands.push(MachineOperand::Register(RAX));
        inst.operands.push(MachineOperand::Register(RCX));
        let result = enc.encode_instruction(&inst);
        // ADD RAX, RCX: REX.W 0x01 ModRM(11,001,000)
        assert_eq!(result.bytes[0], 0x48); // REX.W
        assert_eq!(result.bytes[1], 0x01); // ADD r/m,r
        assert_eq!(result.bytes[2], modrm_byte(0b11, 1, 0)); // 0xC8
    }

    #[test]
    fn test_encoder_add_reg_imm8() {
        let mut enc = X86_64Encoder::new(0);
        let mut inst = MachineInstruction::new(X86Opcode::Add.as_u32());
        inst.operands.push(MachineOperand::Register(RAX));
        inst.operands.push(MachineOperand::Immediate(42));
        let result = enc.encode_instruction(&inst);
        // ADD RAX, 42: REX.W 0x83 ModRM(11, /0, RAX) imm8
        assert_eq!(result.bytes[0], 0x48); // REX.W
        assert_eq!(result.bytes[1], 0x83);
        assert_eq!(result.bytes[2], modrm_byte(0b11, 0, 0)); // 0xC0
        assert_eq!(result.bytes[3], 42);
    }

    #[test]
    fn test_encoder_mov_reg_reg() {
        let mut enc = X86_64Encoder::new(0);
        let mut inst = MachineInstruction::new(X86Opcode::Mov.as_u32());
        inst.operands.push(MachineOperand::Register(RDI));
        inst.operands.push(MachineOperand::Register(RSI));
        let result = enc.encode_instruction(&inst);
        // MOV RDI, RSI: REX.W 0x89 ModRM(11, RSI=6, RDI=7)
        assert_eq!(result.bytes[0], 0x48);
        assert_eq!(result.bytes[1], 0x89);
        assert_eq!(result.bytes[2], modrm_byte(0b11, 6, 7));
    }

    #[test]
    fn test_encoder_call_symbol() {
        let mut enc = X86_64Encoder::new(0);
        let mut inst = MachineInstruction::new(X86Opcode::Call.as_u32());
        inst.operands.push(MachineOperand::GlobalSymbol("printf".to_string()));
        let result = enc.encode_instruction(&inst);
        assert_eq!(result.bytes[0], 0xE8);
        assert_eq!(result.bytes.len(), 5);
        assert_eq!(result.relocations.len(), 1);
        assert_eq!(result.relocations[0].symbol, "printf");
    }
}
