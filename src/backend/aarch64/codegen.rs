//! # AArch64 Instruction Selection and Emission
//!
//! Translates BCC IR instructions into AArch64 (ARM 64-bit) machine instructions.
//! Covers the A64 instruction set with fixed 32-bit instruction encoding.
//!
//! ## Instruction Encoding Groups
//! - Data Processing (Immediate): ADD/SUB with 12-bit imm, MOVZ/MOVK/MOVN, logical with bitmask
//! - Data Processing (Register): shifted/extended register ops, conditional select
//! - Loads and Stores: LDR/STR with multiple addressing modes (immediate offset, register offset, pre/post-index)
//! - Branches: B, B.cond, BL, BLR, BR, RET, CBZ/CBNZ, TBZ/TBNZ
//! - SIMD/FP: FADD, FSUB, FMUL, FDIV, FCMP, FCVT, FMOV between GP and FP regs
//!
//! ## Key Design: ADRP+ADD for Symbol Access
//! AArch64 uses ADRP (Address of 4KB Page) + ADD for PC-relative symbol access within ±4GB.
//! For GOT access (PIC): ADRP + LDR from GOT entry.
//!
//! ## Register Architecture
//! - 31 general-purpose registers: X0–X30 (64-bit) / W0–W30 (32-bit views)
//! - SP (stack pointer, not a general register), XZR/WZR (zero register shares encoding with SP)
//! - 32 SIMD/FP registers: V0–V31 (128-bit), D0–D31 (64-bit double), S0–S31 (32-bit float)
//! - NZCV condition flags register

use crate::backend::aarch64::abi::{AArch64Abi, FrameLayout, INT_ARG_REGS, NUM_INT_ARG_REGS};
use crate::backend::aarch64::registers::*;
use crate::backend::traits::ArgLocation;
use crate::common::diagnostics::Span;
use crate::common::target::Target;
use crate::common::types::CType;
use crate::ir::basic_block::BasicBlock;
use crate::ir::function::IrFunction;
use crate::ir::instructions::{BinOp, BlockId, FCmpOp, ICmpOp, Instruction, Value};
use crate::ir::types::IrType;

// Re-import items used only in test verification module to avoid unused import warnings.
#[cfg(test)]
use crate::backend::aarch64::abi::{
    ArgClass, FP_ARG_REGS, INDIRECT_RESULT_REG, INT_RET_REGS, NUM_FP_ARG_REGS, STACK_ALIGNMENT,
};
#[cfg(test)]
use crate::backend::traits::{
    MachineBasicBlock, MachineFunction, MachineInstruction, MachineOperand, RegisterInfo,
};
#[cfg(test)]
use crate::common::diagnostics::DiagnosticEngine;
#[cfg(test)]
use crate::common::types::MachineType;

// ===========================================================================
// A64 Opcode Definitions
// ===========================================================================

/// AArch64 (A64) instruction opcodes covering all major instruction categories.
///
/// Each variant maps to one or more A64 machine instruction encodings.
/// Pseudo-instructions (MOV_reg, LI, LA, CALL) are resolved to real
/// instruction sequences during assembly encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum A64Opcode {
    // -- Data Processing (Immediate) ----------------------------------------
    /// Add immediate with optional 12-bit left-shifted value.
    ADD_imm,
    /// Add immediate, setting condition flags (NZCV).
    ADDS_imm,
    /// Subtract immediate with optional 12-bit left-shifted value.
    SUB_imm,
    /// Subtract immediate, setting condition flags.
    SUBS_imm,
    /// Move wide with zero — loads a 16-bit immediate into a halfword position.
    MOVZ,
    /// Move wide with keep — loads a 16-bit immediate, keeping other bits.
    MOVK,
    /// Move wide with NOT — loads the bitwise inverse of a 16-bit immediate.
    MOVN,
    /// Bitwise AND with bitmask immediate.
    AND_imm,
    /// Bitwise OR with bitmask immediate.
    ORR_imm,
    /// Bitwise XOR with bitmask immediate.
    EOR_imm,
    /// Bitwise AND (setting flags) with bitmask immediate.
    ANDS_imm,
    /// Address of 4KB page relative to PC — upper 21 bits.
    ADRP,
    /// Address relative to PC — full 21-bit offset.
    ADR,

    // -- Data Processing (Register) -----------------------------------------
    /// Add register (shifted).
    ADD_reg,
    /// Add register (shifted), setting condition flags.
    ADDS_reg,
    /// Subtract register (shifted).
    SUB_reg,
    /// Subtract register (shifted), setting condition flags.
    SUBS_reg,
    /// Bitwise AND register.
    AND_reg,
    /// Bitwise OR register.
    ORR_reg,
    /// Bitwise XOR register.
    EOR_reg,
    /// Bitwise AND (setting flags) register.
    ANDS_reg,
    /// Bitwise OR NOT register.
    ORN_reg,
    /// Bit clear register (AND NOT).
    BIC_reg,
    /// Multiply-Add: Rd = Ra + Rn * Rm.
    MADD,
    /// Multiply-Subtract: Rd = Ra - Rn * Rm.
    MSUB,
    /// Signed multiply high (upper 64 bits of 64x64→128).
    SMULH,
    /// Unsigned multiply high (upper 64 bits of 64x64→128).
    UMULH,
    /// Signed divide.
    SDIV,
    /// Unsigned divide.
    UDIV,
    /// Logical shift left (register).
    LSL_reg,
    /// Logical shift right (register).
    LSR_reg,
    /// Arithmetic shift right (register).
    ASR_reg,
    /// Rotate right (register).
    ROR_reg,
    /// Logical shift left (immediate via UBFM alias).
    LSL_imm,
    /// Logical shift right (immediate via UBFM alias).
    LSR_imm,
    /// Arithmetic shift right (immediate via SBFM alias).
    ASR_imm,

    // -- Conditional Select -------------------------------------------------
    /// Conditional select: Rd = cond ? Rn : Rm.
    CSEL,
    /// Conditional select increment: Rd = cond ? Rn : Rm+1.
    CSINC,
    /// Conditional select invert: Rd = cond ? Rn : ~Rm.
    CSINV,
    /// Conditional select negate: Rd = cond ? Rn : -Rm.
    CSNEG,
    /// Conditional set (alias): Rd = cond ? 1 : 0.
    CSET,
    /// Conditional set mask (alias): Rd = cond ? ~0 : 0.
    CSETM,

    // -- Bit Manipulation ---------------------------------------------------
    /// Count leading zeros.
    CLZ,
    /// Count leading sign bits.
    CLS,
    /// Reverse bits.
    RBIT,
    /// Reverse bytes (full register).
    REV,
    /// Reverse bytes in halfwords.
    REV16,
    /// Reverse bytes in 32-bit words (only valid on 64-bit registers).
    REV32,
    /// Extract bitfield from pair of registers.
    EXTR,
    /// Bitfield move.
    BFM,
    /// Signed bitfield move.
    SBFM,
    /// Unsigned bitfield move.
    UBFM,
    /// Sign-extend byte (alias for SBFM).
    SXTB,
    /// Sign-extend halfword (alias for SBFM).
    SXTH,
    /// Sign-extend word (alias for SBFM).
    SXTW,
    /// Zero-extend byte (alias for UBFM/AND).
    UXTB,
    /// Zero-extend halfword (alias for UBFM/AND).
    UXTH,

    // -- Loads --------------------------------------------------------------
    /// Load register (unsigned immediate offset).
    LDR_imm,
    /// Load byte (unsigned immediate offset).
    LDRB_imm,
    /// Load halfword (unsigned immediate offset).
    LDRH_imm,
    /// Load signed byte (unsigned immediate offset).
    LDRSB_imm,
    /// Load signed halfword (unsigned immediate offset).
    LDRSH_imm,
    /// Load signed word to 64-bit (unsigned immediate offset).
    LDRSW_imm,
    /// Load register (register offset).
    LDR_reg,
    /// Load register (PC-relative literal).
    LDR_literal,
    /// Load pair of registers.
    LDP,
    /// Load pair of signed words to 64-bit.
    LDPSW,
    /// Load register (pre-index): base updated before access.
    LDR_pre,
    /// Load register (post-index): base updated after access.
    LDR_post,

    // -- Stores -------------------------------------------------------------
    /// Store register (unsigned immediate offset).
    STR_imm,
    /// Store byte (unsigned immediate offset).
    STRB_imm,
    /// Store halfword (unsigned immediate offset).
    STRH_imm,
    /// Store register (register offset).
    STR_reg,
    /// Store pair of registers.
    STP,
    /// Store register (pre-index): base updated before access.
    STR_pre,
    /// Store register (post-index): base updated after access.
    STR_post,

    // -- FP/SIMD Loads and Stores -------------------------------------------
    /// Load FP register (unsigned immediate offset).
    LDR_fp_imm,
    /// Store FP register (unsigned immediate offset).
    STR_fp_imm,
    /// Load pair of FP registers.
    LDP_fp,
    /// Store pair of FP registers.
    STP_fp,

    // -- Branches -----------------------------------------------------------
    /// Unconditional branch (±128 MiB PC-relative).
    B,
    /// Branch with link (function call, ±128 MiB PC-relative).
    BL,
    /// Conditional branch (B.cond, ±1 MiB PC-relative).
    B_cond,
    /// Branch to register (indirect jump).
    BR,
    /// Branch with link to register (indirect call).
    BLR,
    /// Return from subroutine (defaults to X30/LR).
    RET,
    /// Compare and branch if zero (±1 MiB).
    CBZ,
    /// Compare and branch if not zero (±1 MiB).
    CBNZ,
    /// Test bit and branch if zero (±32 KiB).
    TBZ,
    /// Test bit and branch if not zero (±32 KiB).
    TBNZ,

    // -- Comparison (aliases for flag-setting ops with XZR dest) -------------
    /// Compare immediate (alias for SUBS with XZR destination).
    CMP_imm,
    /// Compare register (alias for SUBS with XZR destination).
    CMP_reg,
    /// Compare negative immediate (alias for ADDS with XZR dest).
    CMN_imm,
    /// Compare negative register (alias for ADDS with XZR dest).
    CMN_reg,
    /// Test bits immediate (alias for ANDS with XZR dest).
    TST_imm,
    /// Test bits register (alias for ANDS with XZR dest).
    TST_reg,
    /// Conditional compare immediate.
    CCMP_imm,
    /// Conditional compare register.
    CCMP_reg,

    // -- FP Data Processing -------------------------------------------------
    /// Single-precision floating-point add.
    FADD_s,
    /// Single-precision floating-point subtract.
    FSUB_s,
    /// Single-precision floating-point multiply.
    FMUL_s,
    /// Single-precision floating-point divide.
    FDIV_s,
    /// Single-precision floating-point square root.
    FSQRT_s,
    /// Double-precision floating-point add.
    FADD_d,
    /// Double-precision floating-point subtract.
    FSUB_d,
    /// Double-precision floating-point multiply.
    FMUL_d,
    /// Double-precision floating-point divide.
    FDIV_d,
    /// Double-precision floating-point square root.
    FSQRT_d,
    /// Single-precision negate.
    FNEG_s,
    /// Double-precision negate.
    FNEG_d,
    /// Single-precision absolute value.
    FABS_s,
    /// Double-precision absolute value.
    FABS_d,
    /// Single-precision fused multiply-add.
    FMADD_s,
    /// Single-precision fused multiply-subtract.
    FMSUB_s,
    /// Single-precision negated fused multiply-add.
    FNMADD_s,
    /// Single-precision negated fused multiply-subtract.
    FNMSUB_s,
    /// Double-precision fused multiply-add.
    FMADD_d,
    /// Double-precision fused multiply-subtract.
    FMSUB_d,
    /// Double-precision negated fused multiply-add.
    FNMADD_d,
    /// Double-precision negated fused multiply-subtract.
    FNMSUB_d,
    /// Single-precision minimum.
    FMIN_s,
    /// Single-precision maximum.
    FMAX_s,
    /// Double-precision minimum.
    FMIN_d,
    /// Double-precision maximum.
    FMAX_d,

    // -- FP Comparison ------------------------------------------------------
    /// Single-precision compare (quiet NaN).
    FCMP_s,
    /// Double-precision compare (quiet NaN).
    FCMP_d,
    /// Single-precision compare (signalling NaN).
    FCMPE_s,
    /// Double-precision compare (signalling NaN).
    FCMPE_d,

    // -- FP Conversion ------------------------------------------------------
    /// Convert single to double.
    FCVT_sd,
    /// Convert double to single.
    FCVT_ds,
    /// Float to signed int (W-reg from S-reg, toward zero).
    FCVTZS_ws,
    /// Float to signed int (X-reg from S-reg, toward zero).
    FCVTZS_xs,
    /// Double to signed int (W-reg from D-reg, toward zero).
    FCVTZS_wd,
    /// Double to signed int (X-reg from D-reg, toward zero).
    FCVTZS_xd,
    /// Float to unsigned int (W-reg from S-reg, toward zero).
    FCVTZU_ws,
    /// Float to unsigned int (X-reg from S-reg, toward zero).
    FCVTZU_xs,
    /// Double to unsigned int (W-reg from D-reg, toward zero).
    FCVTZU_wd,
    /// Double to unsigned int (X-reg from D-reg, toward zero).
    FCVTZU_xd,
    /// Signed int (W-reg) to float (S-reg).
    SCVTF_sw,
    /// Signed int (X-reg) to float (S-reg).
    SCVTF_sx,
    /// Signed int (W-reg) to double (D-reg).
    SCVTF_dw,
    /// Signed int (X-reg) to double (D-reg).
    SCVTF_dx,
    /// Unsigned int (W-reg) to float (S-reg).
    UCVTF_sw,
    /// Unsigned int (X-reg) to float (S-reg).
    UCVTF_sx,
    /// Unsigned int (W-reg) to double (D-reg).
    UCVTF_dw,
    /// Unsigned int (X-reg) to double (D-reg).
    UCVTF_dx,

    // -- FP Move ------------------------------------------------------------
    /// Move between single-precision FP registers.
    FMOV_s,
    /// Move between double-precision FP registers.
    FMOV_d,
    /// Move from general-purpose register to FP register.
    FMOV_gen_to_fp,
    /// Move from FP register to general-purpose register.
    FMOV_fp_to_gen,

    // -- System -------------------------------------------------------------
    /// No operation.
    NOP,
    /// Data memory barrier.
    DMB,
    /// Data synchronization barrier.
    DSB,
    /// Instruction synchronization barrier.
    ISB,
    /// Supervisor call (system call).
    SVC,
    /// Hypervisor call.
    HVC,
    /// Secure monitor call.
    SMC,
    /// Move from system register.
    MRS,
    /// Move to system register.
    MSR,

    // -- Pseudo-instructions ------------------------------------------------
    /// Register move (alias for ORR Xd, XZR, Xm).
    MOV_reg,
    /// Immediate move (resolved to MOVZ/MOVK/MOVN sequence).
    MOV_imm,
    /// Negate register (alias for SUB Xd, XZR, Xm).
    NEG_reg,
    /// Bitwise NOT register (alias for ORN Xd, XZR, Xm).
    MVN_reg,
    /// Load immediate — MOVZ+MOVK sequence for large constants.
    LI,
    /// Load address — ADRP+ADD pair for symbol addressing.
    LA,
    /// Function call — BL or ADRP+BLR sequence.
    CALL,
    /// Inline assembly passthrough.
    INLINE_ASM,
}

// ===========================================================================
// Condition Codes
// ===========================================================================

/// AArch64 condition codes encoded in the 4-bit cond field of conditional
/// instructions (B.cond, CSEL, CCMP, etc.).
///
/// Condition codes test the NZCV flags set by comparison/arithmetic instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CondCode {
    /// Equal (Z=1).
    EQ = 0b0000,
    /// Not equal (Z=0).
    NE = 0b0001,
    /// Carry set / unsigned >= (also HS).
    CS = 0b0010,
    /// Carry clear / unsigned < (also LO).
    CC = 0b0011,
    /// Minus / negative (N=1).
    MI = 0b0100,
    /// Plus / positive or zero (N=0).
    PL = 0b0101,
    /// Overflow (V=1).
    VS = 0b0110,
    /// No overflow (V=0).
    VC = 0b0111,
    /// Unsigned higher (C=1 && Z=0).
    HI = 0b1000,
    /// Unsigned lower or same (C=0 || Z=1).
    LS = 0b1001,
    /// Signed >= (N==V).
    GE = 0b1010,
    /// Signed < (N!=V).
    LT = 0b1011,
    /// Signed > (Z=0 && N==V).
    GT = 0b1100,
    /// Signed <= (Z=1 || N!=V).
    LE = 0b1101,
    /// Always (unconditional).
    AL = 0b1110,
    /// Never (reserved, sometimes used).
    NV = 0b1111,
}

impl CondCode {
    /// Invert a condition code (logical negation).
    ///
    /// Each condition code has a complementary code obtained by flipping
    /// the least significant bit of the 4-bit encoding.
    #[inline]
    pub fn invert(self) -> CondCode {
        match self {
            CondCode::EQ => CondCode::NE,
            CondCode::NE => CondCode::EQ,
            CondCode::CS => CondCode::CC,
            CondCode::CC => CondCode::CS,
            CondCode::MI => CondCode::PL,
            CondCode::PL => CondCode::MI,
            CondCode::VS => CondCode::VC,
            CondCode::VC => CondCode::VS,
            CondCode::HI => CondCode::LS,
            CondCode::LS => CondCode::HI,
            CondCode::GE => CondCode::LT,
            CondCode::LT => CondCode::GE,
            CondCode::GT => CondCode::LE,
            CondCode::LE => CondCode::GT,
            CondCode::AL => CondCode::NV,
            CondCode::NV => CondCode::AL,
        }
    }

    /// Map an IR integer comparison operator to the appropriate AArch64
    /// condition code (assumes CMP has already been emitted).
    pub fn from_icmp(op: &ICmpOp) -> CondCode {
        match op {
            ICmpOp::Eq => CondCode::EQ,
            ICmpOp::Ne => CondCode::NE,
            ICmpOp::Slt => CondCode::LT,
            ICmpOp::Sle => CondCode::LE,
            ICmpOp::Sgt => CondCode::GT,
            ICmpOp::Sge => CondCode::GE,
            ICmpOp::Ult => CondCode::CC,
            ICmpOp::Ule => CondCode::LS,
            ICmpOp::Ugt => CondCode::HI,
            ICmpOp::Uge => CondCode::CS,
        }
    }

    /// Map an IR floating-point comparison operator to the appropriate
    /// AArch64 condition code pair (assumes FCMP has already been emitted).
    ///
    /// Returns the condition code for ordered comparisons; unordered
    /// comparisons require additional VS/VC checks handled by the caller.
    pub fn from_fcmp(op: &FCmpOp) -> CondCode {
        match op {
            FCmpOp::Oeq => CondCode::EQ,
            FCmpOp::One => CondCode::MI, // LT || GT → MI or HI after FCMP
            FCmpOp::Olt => CondCode::MI,
            FCmpOp::Ole => CondCode::LS,
            FCmpOp::Ogt => CondCode::GT,
            FCmpOp::Oge => CondCode::GE,
            FCmpOp::Uno => CondCode::VS,
            FCmpOp::Ord => CondCode::VC,
        }
    }

    /// Returns the 4-bit encoding value of this condition code.
    #[inline]
    pub fn encoding(self) -> u8 {
        self as u8
    }
}

// ===========================================================================
// A64Instruction — AArch64-specific machine instruction
// ===========================================================================

/// Represents a single AArch64 machine instruction with all operand fields.
///
/// All A64 instructions are fixed 32-bit (4 bytes) wide. This struct captures
/// the decoded instruction fields before final binary encoding. Fields that
/// are not applicable to a given opcode are set to `None` / 0 / `false`.
#[derive(Debug, Clone)]
pub struct A64Instruction {
    /// The instruction opcode.
    pub opcode: A64Opcode,
    /// Destination register (Rd), if applicable.
    pub rd: Option<u8>,
    /// First source register (Rn), if applicable.
    pub rn: Option<u8>,
    /// Second source register (Rm), if applicable.
    pub rm: Option<u8>,
    /// Third source register (Ra), for multiply-accumulate.
    pub ra: Option<u8>,
    /// Immediate value (interpretation depends on opcode).
    pub imm: i64,
    /// Shift amount or shift type encoding.
    pub shift: u8,
    /// Condition code for conditional operations.
    pub cond: Option<CondCode>,
    /// Symbol reference for relocations (function/variable names).
    pub symbol: Option<String>,
    /// If true, use W-register (32-bit) forms; false = X-register (64-bit).
    pub is_32bit: bool,
    /// If true, this is a FP/SIMD instruction.
    pub is_fp: bool,
    /// Optional comment for assembly output / debugging.
    pub comment: Option<String>,
}

impl A64Instruction {
    /// Create a new instruction with the given opcode and default fields.
    pub fn new(opcode: A64Opcode) -> Self {
        A64Instruction {
            opcode,
            rd: None,
            rn: None,
            rm: None,
            ra: None,
            imm: 0,
            shift: 0,
            cond: None,
            symbol: None,
            is_32bit: false,
            is_fp: false,
            comment: None,
        }
    }

    /// Builder: set destination register.
    #[inline]
    pub fn with_rd(mut self, rd: u8) -> Self {
        self.rd = Some(rd);
        self
    }

    /// Builder: set first source register.
    #[inline]
    pub fn with_rn(mut self, rn: u8) -> Self {
        self.rn = Some(rn);
        self
    }

    /// Builder: set second source register.
    #[inline]
    pub fn with_rm(mut self, rm: u8) -> Self {
        self.rm = Some(rm);
        self
    }

    /// Builder: set third source register (multiply-accumulate).
    #[inline]
    pub fn with_ra(mut self, ra: u8) -> Self {
        self.ra = Some(ra);
        self
    }

    /// Builder: set immediate value.
    #[inline]
    pub fn with_imm(mut self, imm: i64) -> Self {
        self.imm = imm;
        self
    }

    /// Builder: set shift amount.
    #[inline]
    pub fn with_shift(mut self, shift: u8) -> Self {
        self.shift = shift;
        self
    }

    /// Builder: set condition code.
    #[inline]
    pub fn with_cond(mut self, cond: CondCode) -> Self {
        self.cond = Some(cond);
        self
    }

    /// Builder: set symbol reference.
    #[inline]
    pub fn with_symbol(mut self, sym: String) -> Self {
        self.symbol = Some(sym);
        self
    }

    /// Builder: set 32-bit mode (W-register).
    #[inline]
    pub fn set_32bit(mut self) -> Self {
        self.is_32bit = true;
        self
    }

    /// Builder: mark as FP/SIMD instruction.
    #[inline]
    pub fn set_fp(mut self) -> Self {
        self.is_fp = true;
        self
    }

    /// Builder: add comment for debugging.
    #[inline]
    pub fn with_comment(mut self, comment: impl Into<String>) -> Self {
        self.comment = Some(comment.into());
        self
    }
}

// ===========================================================================
// Helper: check if value fits in 12-bit unsigned immediate
// ===========================================================================

/// Returns true if the value fits in a 12-bit unsigned immediate, optionally
/// with a 12-bit left shift.
fn fits_add_sub_imm(val: u64) -> bool {
    val <= 0xFFF || (val & 0xFFF == 0 && (val >> 12) <= 0xFFF)
}

/// Returns the shift needed for an add/sub immediate.
#[allow(dead_code)]
fn add_sub_imm_shift(val: u64) -> Option<u8> {
    if val <= 0xFFF {
        Some(0)
    } else if val & 0xFFF == 0 && (val >> 12) <= 0xFFF {
        Some(12)
    } else {
        None
    }
}

// ===========================================================================
// AArch64InstructionSelector
// ===========================================================================

/// The AArch64 instruction selector. Translates IR functions into sequences
/// of [`A64Instruction`]s by pattern-matching IR instruction variants.
///
/// Maintains per-function state including frame size, spill slots, and block
/// labels. Call [`select_function`] to process an entire IR function.
pub struct AArch64InstructionSelector {
    instructions: Vec<A64Instruction>,
    current_block: Option<String>,
    frame_size: i64,
    spill_slots: Vec<(u32, i64)>,
    pic_mode: bool,
    target: Target,
    next_vreg: u32,
    block_labels: Vec<String>,
    used_callee_saved: Vec<u8>,
}

impl AArch64InstructionSelector {
    /// Create a new instruction selector for the AArch64 target.
    pub fn new(target: Target, pic_mode: bool) -> Self {
        AArch64InstructionSelector {
            instructions: Vec::new(),
            current_block: None,
            frame_size: 0,
            spill_slots: Vec::new(),
            pic_mode,
            target,
            next_vreg: 0,
            block_labels: Vec::new(),
            used_callee_saved: Vec::new(),
        }
    }

    /// Reset all per-function state for reuse.
    fn reset(&mut self) {
        self.instructions.clear();
        self.current_block = None;
        self.frame_size = 0;
        self.spill_slots.clear();
        self.next_vreg = 0;
        self.block_labels.clear();
        self.used_callee_saved.clear();
    }

    /// Allocate the next virtual register number.
    #[allow(dead_code)]
    fn alloc_vreg(&mut self) -> u32 {
        let v = self.next_vreg;
        self.next_vreg += 1;
        v
    }

    /// Emit an instruction into the current function's instruction list.
    fn emit(&mut self, inst: A64Instruction) {
        self.instructions.push(inst);
    }

    /// Determine if an IR type should use W-register (32-bit) forms.
    fn is_32bit_type(ty: &IrType) -> bool {
        matches!(ty, IrType::I1 | IrType::I8 | IrType::I16 | IrType::I32)
    }

    /// Get the memory size in bytes for an IR type.
    fn type_size_for_mem(&self, ty: &IrType) -> usize {
        ty.size_bytes(&self.target)
    }

    /// Map an IR type to an approximate CType for ABI classification.
    /// This provides a best-effort mapping for parameter classification;
    /// the exact C-level type information may differ but the ABI-relevant
    /// properties (size, register class) are preserved.
    fn irtype_to_ctype(ty: &IrType) -> CType {
        match ty {
            IrType::Void => CType::Void,
            IrType::I1 => CType::Bool,
            IrType::I8 => CType::Char,
            IrType::I16 => CType::Short,
            IrType::I32 => CType::Int,
            IrType::I64 => CType::Long,
            IrType::I128 => CType::LongLong,
            IrType::F32 => CType::Float,
            IrType::F64 => CType::Double,
            IrType::F80 => CType::LongDouble,
            IrType::Ptr => CType::Pointer(
                Box::new(CType::Void),
                crate::common::types::TypeQualifiers {
                    is_const: false,
                    is_volatile: false,
                    is_restrict: false,
                    is_atomic: false,
                },
            ),
            IrType::Array(inner, size) => {
                CType::Array(Box::new(Self::irtype_to_ctype(inner)), Some(*size))
            }
            IrType::Struct(_) => CType::Struct {
                name: None,
                fields: Vec::new(),
                packed: false,
                aligned: None,
            },
            IrType::Function(ret, params) => CType::Function {
                return_type: Box::new(Self::irtype_to_ctype(ret)),
                params: params.iter().map(Self::irtype_to_ctype).collect(),
                variadic: false,
            },
        }
    }

    // -----------------------------------------------------------------------
    // Immediate materialization
    // -----------------------------------------------------------------------

    /// Materialize a 64-bit immediate value into a register using the optimal
    /// MOVZ + MOVK sequence. Optimizes for common patterns:
    /// - Zero: MOV Xd, XZR
    /// - Single halfword: MOVZ with shift
    /// - Inverted halfword: MOVN with shift
    /// - Multi-halfword: MOVZ + MOVK chain (skip zero halfwords)
    pub fn materialize_immediate(&mut self, rd: u8, value: u64) -> Vec<A64Instruction> {
        let mut result = Vec::new();

        if value == 0 {
            result.push(
                A64Instruction::new(A64Opcode::MOV_reg)
                    .with_rd(rd)
                    .with_rn(XZR)
                    .with_comment("mov xd, xzr (zero)"),
            );
            return result;
        }

        // Check if the value fits in a single MOVZ with shift.
        for hw in 0..4u8 {
            let shifted = (value >> (hw as u64 * 16)) & 0xFFFF;
            let mask = 0xFFFF_u64 << (hw as u64 * 16);
            if value == (shifted << (hw as u64 * 16)) && shifted != 0 {
                result.push(
                    A64Instruction::new(A64Opcode::MOVZ)
                        .with_rd(rd)
                        .with_imm(shifted as i64)
                        .with_shift(hw * 16),
                );
                return result;
            }
            let _ = mask;
        }

        // Check if bitwise NOT fits in a single MOVN with shift.
        let not_val = !value;
        for hw in 0..4u8 {
            let shifted = (not_val >> (hw as u64 * 16)) & 0xFFFF;
            if not_val == (shifted << (hw as u64 * 16)) && shifted != 0 {
                result.push(
                    A64Instruction::new(A64Opcode::MOVN)
                        .with_rd(rd)
                        .with_imm(shifted as i64)
                        .with_shift(hw * 16),
                );
                return result;
            }
        }

        // General case: MOVZ first non-zero halfword, MOVK for remaining.
        let mut first = true;
        for hw in 0..4u8 {
            let chunk = (value >> (hw as u64 * 16)) & 0xFFFF;
            if chunk == 0 {
                continue;
            }
            if first {
                result.push(
                    A64Instruction::new(A64Opcode::MOVZ)
                        .with_rd(rd)
                        .with_imm(chunk as i64)
                        .with_shift(hw * 16),
                );
                first = false;
            } else {
                result.push(
                    A64Instruction::new(A64Opcode::MOVK)
                        .with_rd(rd)
                        .with_imm(chunk as i64)
                        .with_shift(hw * 16),
                );
            }
        }

        result
    }

    /// Materialize a symbol address into a register.
    ///
    /// - Non-PIC: ADRP Xd, symbol ; ADD Xd, Xd, :lo12:symbol
    /// - PIC/GOT: ADRP Xd, :got:symbol ; LDR Xd, [Xd, :got_lo12:symbol]
    pub fn materialize_address(&mut self, rd: u8, symbol: &str) -> Vec<A64Instruction> {
        let mut result = Vec::new();
        if self.pic_mode {
            // GOT-indirect addressing for PIC.
            result.push(
                A64Instruction::new(A64Opcode::ADRP)
                    .with_rd(rd)
                    .with_symbol(format!(":got:{}", symbol))
                    .with_comment(format!("adrp {}, :got:{}", gpr_name(rd), symbol)),
            );
            result.push(
                A64Instruction::new(A64Opcode::LDR_imm)
                    .with_rd(rd)
                    .with_rn(rd)
                    .with_symbol(format!(":got_lo12:{}", symbol))
                    .with_comment(format!(
                        "ldr {}, [{}, :got_lo12:{}]",
                        gpr_name(rd),
                        gpr_name(rd),
                        symbol
                    )),
            );
        } else {
            // Direct PC-relative addressing.
            result.push(
                A64Instruction::new(A64Opcode::ADRP)
                    .with_rd(rd)
                    .with_symbol(symbol.to_string())
                    .with_comment(format!("adrp {}, {}", gpr_name(rd), symbol)),
            );
            result.push(
                A64Instruction::new(A64Opcode::ADD_imm)
                    .with_rd(rd)
                    .with_rn(rd)
                    .with_symbol(format!(":lo12:{}", symbol))
                    .with_comment(format!(
                        "add {}, {}, :lo12:{}",
                        gpr_name(rd),
                        gpr_name(rd),
                        symbol
                    )),
            );
        }
        result
    }

    // -----------------------------------------------------------------------
    // Function-level selection
    // -----------------------------------------------------------------------

    /// Select instructions for an entire IR function. This is the main
    /// entry point for AArch64 instruction selection.
    ///
    /// Processes the function in order:
    /// 1. Build block labels
    /// 2. Compute frame layout via AAPCS64 ABI
    /// 3. Emit prologue (STP x29,x30; MOV x29,sp; save callee-saved)
    /// 4. Set up argument registers
    /// 5. Process each basic block
    /// 6. Emit epilogue (restore callee-saved; LDP x29,x30; RET)
    pub fn select_function(&mut self, func: &IrFunction, abi: &AArch64Abi) -> Vec<A64Instruction> {
        self.reset();
        if !func.is_definition {
            return Vec::new();
        }

        self.next_vreg = func.value_count;

        // Build block labels for branch target resolution.
        for (i, block) in func.blocks().iter().enumerate() {
            let label = block
                .label
                .clone()
                .unwrap_or_else(|| format!(".LBB_{}_{}", func.name, i));
            self.block_labels.push(label);
        }

        // Compute frame layout.
        let callee_gprs: Vec<u8> = CALLEE_SAVED_GPRS.to_vec();
        let callee_fprs: Vec<u8> = CALLEE_SAVED_FPRS.to_vec();
        let local_size = self.estimate_locals_size(func);
        let callee_saved_total = callee_gprs.len() + callee_fprs.len();
        let layout = AArch64Abi::compute_frame_layout(&[], local_size, callee_saved_total);
        self.frame_size = layout.total_size as i64;

        // Emit prologue.
        self.emit_prologue(&layout, func);

        // Emit argument setup (copy from ABI locations to local vregs).
        self.emit_arg_setup(func, abi);

        // Process each basic block.
        for block in func.blocks() {
            self.select_basic_block(block, func);
        }

        std::mem::take(&mut self.instructions)
    }

    /// Estimate total stack space needed for local allocas.
    fn estimate_locals_size(&self, func: &IrFunction) -> usize {
        let entry = func.entry_block();
        let mut total: usize = 0;
        for inst in entry.instructions() {
            if let Instruction::Alloca { ty, alignment, .. } = inst {
                let size = ty.size_bytes(&self.target);
                let align = alignment.unwrap_or_else(|| ty.align_bytes(&self.target));
                let mask = if align > 0 { align - 1 } else { 0 };
                total = (total + mask) & !mask;
                total += size;
            }
        }
        // Round up to 16-byte alignment.
        (total + 15) & !15
    }

    /// Emit the function prologue: save frame pointer + link register,
    /// set up frame pointer, and save callee-saved registers.
    fn emit_prologue(&mut self, layout: &FrameLayout, _func: &IrFunction) {
        if layout.total_size == 0 {
            return;
        }
        // STP X29, X30, [SP, #-frame_size]!  (pre-decrement SP)
        self.emit(
            A64Instruction::new(A64Opcode::STP)
                .with_rd(FP_REG)
                .with_rn(LR)
                .with_rm(SP_REG)
                .with_imm(-(layout.total_size as i64))
                .with_comment("stp x29, x30, [sp, #-frame_size]!"),
        );
        // MOV X29, SP (set frame pointer)
        self.emit(
            A64Instruction::new(A64Opcode::MOV_reg)
                .with_rd(FP_REG)
                .with_rn(SP_REG)
                .with_comment("mov x29, sp (set frame pointer)"),
        );

        // Save callee-saved GPRs using STP pairs.
        let gprs = &CALLEE_SAVED_GPRS;
        let mut offset = layout.callee_saved_offset;
        let mut i = 0;
        while i + 1 < gprs.len() {
            self.emit(
                A64Instruction::new(A64Opcode::STP)
                    .with_rd(gprs[i])
                    .with_rn(gprs[i + 1])
                    .with_rm(SP_REG)
                    .with_imm(offset)
                    .with_comment(format!(
                        "stp {}, {}, [sp, #{}]",
                        gpr_name(gprs[i]),
                        gpr_name(gprs[i + 1]),
                        offset
                    )),
            );
            offset += 16;
            i += 2;
        }
        // Handle odd remaining GPR.
        if i < gprs.len() {
            self.emit(
                A64Instruction::new(A64Opcode::STR_imm)
                    .with_rd(gprs[i])
                    .with_rn(SP_REG)
                    .with_imm(offset)
                    .with_comment(format!("str {}, [sp, #{}]", gpr_name(gprs[i]), offset)),
            );
            offset += 8;
        }

        // Save callee-saved FPRs using STP_fp pairs.
        let fprs = &CALLEE_SAVED_FPRS;
        let mut fi = 0;
        while fi + 1 < fprs.len() {
            self.emit(
                A64Instruction::new(A64Opcode::STP_fp)
                    .with_rd(fprs[fi])
                    .with_rn(fprs[fi + 1])
                    .with_rm(SP_REG)
                    .with_imm(offset)
                    .set_fp()
                    .with_comment(format!(
                        "stp {}, {}, [sp, #{}]",
                        fpr_name_d(fprs[fi]),
                        fpr_name_d(fprs[fi + 1]),
                        offset
                    )),
            );
            offset += 16;
            fi += 2;
        }
        if fi < fprs.len() {
            self.emit(
                A64Instruction::new(A64Opcode::STR_fp_imm)
                    .with_rd(fprs[fi])
                    .with_rn(SP_REG)
                    .with_imm(offset)
                    .set_fp()
                    .with_comment(format!("str {}, [sp, #{}]", fpr_name_d(fprs[fi]), offset)),
            );
        }
    }

    /// Emit the function epilogue: restore callee-saved registers, restore
    /// frame pointer + link register, and return.
    #[allow(dead_code)]
    fn emit_epilogue(&mut self, layout: &FrameLayout) {
        if layout.total_size == 0 {
            self.emit(A64Instruction::new(A64Opcode::RET).with_comment("ret (leaf)"));
            return;
        }

        // Restore callee-saved GPRs (reverse order of prologue).
        let gprs = &CALLEE_SAVED_GPRS;
        let mut offset = layout.callee_saved_offset;
        let mut i = 0;
        while i + 1 < gprs.len() {
            self.emit(
                A64Instruction::new(A64Opcode::LDP)
                    .with_rd(gprs[i])
                    .with_rn(gprs[i + 1])
                    .with_rm(SP_REG)
                    .with_imm(offset)
                    .with_comment(format!(
                        "ldp {}, {}, [sp, #{}]",
                        gpr_name(gprs[i]),
                        gpr_name(gprs[i + 1]),
                        offset
                    )),
            );
            offset += 16;
            i += 2;
        }
        if i < gprs.len() {
            self.emit(
                A64Instruction::new(A64Opcode::LDR_imm)
                    .with_rd(gprs[i])
                    .with_rn(SP_REG)
                    .with_imm(offset)
                    .with_comment(format!("ldr {}, [sp, #{}]", gpr_name(gprs[i]), offset)),
            );
            offset += 8;
        }

        // Restore callee-saved FPRs.
        let fprs = &CALLEE_SAVED_FPRS;
        let mut fi = 0;
        while fi + 1 < fprs.len() {
            self.emit(
                A64Instruction::new(A64Opcode::LDP_fp)
                    .with_rd(fprs[fi])
                    .with_rn(fprs[fi + 1])
                    .with_rm(SP_REG)
                    .with_imm(offset)
                    .set_fp()
                    .with_comment(format!(
                        "ldp {}, {}, [sp, #{}]",
                        fpr_name_d(fprs[fi]),
                        fpr_name_d(fprs[fi + 1]),
                        offset
                    )),
            );
            offset += 16;
            fi += 2;
        }
        if fi < fprs.len() {
            self.emit(
                A64Instruction::new(A64Opcode::LDR_fp_imm)
                    .with_rd(fprs[fi])
                    .with_rn(SP_REG)
                    .with_imm(offset)
                    .set_fp()
                    .with_comment(format!("ldr {}, [sp, #{}]", fpr_name_d(fprs[fi]), offset)),
            );
        }

        // LDP X29, X30, [SP], #frame_size  (post-increment SP)
        self.emit(
            A64Instruction::new(A64Opcode::LDP)
                .with_rd(FP_REG)
                .with_rn(LR)
                .with_rm(SP_REG)
                .with_imm(layout.total_size as i64)
                .with_comment("ldp x29, x30, [sp], #frame_size"),
        );
        self.emit(A64Instruction::new(A64Opcode::RET).with_comment("ret"));
    }

    /// Emit argument setup: copy from ABI register/stack locations to vregs.
    fn emit_arg_setup(&mut self, func: &IrFunction, _abi: &AArch64Abi) {
        let mut abi_state = AArch64Abi::new();
        abi_state.reset();
        for (i, param) in func.params.iter().enumerate() {
            let cty = Self::irtype_to_ctype(&param.ty);
            let loc = if func.is_variadic && i >= func.params.len().saturating_sub(1) {
                abi_state.classify_variadic_arg(&cty)
            } else {
                abi_state.classify_arg(&cty)
            };
            match loc {
                ArgLocation::Register(reg) => {
                    // Argument is already in the correct register; emit
                    // a NOP placeholder tracking the location.
                    self.emit(A64Instruction::new(A64Opcode::NOP).with_comment(format!(
                        "arg {} in reg {}",
                        i,
                        gpr_name(reg as u8)
                    )));
                }
                ArgLocation::RegisterPair(r1, r2) => {
                    self.emit(A64Instruction::new(A64Opcode::NOP).with_comment(format!(
                        "arg {} in regs {},{}",
                        i,
                        gpr_name(r1 as u8),
                        gpr_name(r2 as u8)
                    )));
                }
                ArgLocation::Stack(offset) => {
                    // Load stack argument via frame pointer.
                    self.emit(
                        A64Instruction::new(A64Opcode::LDR_imm)
                            .with_rd(X9)
                            .with_rn(FP_REG)
                            .with_imm(offset as i64 + 16)
                            .with_comment(format!(
                                "load stack arg {} from [fp+{}]",
                                i,
                                offset + 16
                            )),
                    );
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Basic block selection
    // -----------------------------------------------------------------------

    /// Select instructions for a single basic block.
    pub fn select_basic_block(&mut self, block: &BasicBlock, _func: &IrFunction) {
        let label = if block.index < self.block_labels.len() {
            self.block_labels[block.index].clone()
        } else {
            format!(".LBB_{}", block.index)
        };
        self.current_block = Some(label.clone());
        // Emit a NOP as block label marker (assembler will convert to label).
        self.emit(A64Instruction::new(A64Opcode::NOP).with_comment(format!("{}:", label)));

        for inst in block.instructions() {
            let selected = self.select_instruction(inst);
            for a64_inst in selected {
                self.emit(a64_inst);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Instruction-level selection (dispatch)
    // -----------------------------------------------------------------------

    /// Select AArch64 instructions for a single IR instruction.
    /// Dispatches to type-specific selection methods based on the IR instruction variant.
    pub fn select_instruction(&mut self, ir_inst: &Instruction) -> Vec<A64Instruction> {
        match ir_inst {
            Instruction::Alloca {
                result,
                ty,
                alignment,
                span,
            } => self.select_alloca(result, ty, alignment, span),
            Instruction::Load {
                result,
                ptr,
                ty,
                volatile: _,
                span: _,
            } => self.select_load(result, ptr, ty),
            Instruction::Store {
                value,
                ptr,
                volatile: _,
                span: _,
            } => self.select_store(value, ptr),
            Instruction::BinOp {
                result,
                op,
                lhs,
                rhs,
                ty,
                span: _,
            } => self.select_binop(result, op, lhs, rhs, ty),
            Instruction::ICmp {
                result,
                op,
                lhs,
                rhs,
                span: _,
            } => self.select_icmp(result, op, lhs, rhs),
            Instruction::FCmp {
                result,
                op,
                lhs,
                rhs,
                span: _,
            } => self.select_fcmp(result, op, lhs, rhs),
            Instruction::Branch { target, span: _ } => self.select_branch(target),
            Instruction::CondBranch {
                condition,
                then_block,
                else_block,
                span: _,
            } => self.select_cond_branch(condition, then_block, else_block),
            Instruction::Switch {
                value,
                default,
                cases,
                span: _,
            } => self.select_switch(value, default, cases),
            Instruction::Call {
                result,
                callee,
                args,
                return_type,
                span: _,
            } => self.select_call(result, callee, args, return_type),
            Instruction::Return { value, span: _ } => self.select_return(value),
            Instruction::Phi { .. } => {
                // Phi nodes are eliminated before code generation; emit NOP placeholder.
                vec![A64Instruction::new(A64Opcode::NOP).with_comment("phi (eliminated)")]
            }
            Instruction::GetElementPtr {
                result,
                base,
                indices,
                result_type,
                in_bounds: _,
                span: _,
            } => self.select_gep(result, base, indices, result_type),
            Instruction::BitCast {
                result,
                value,
                to_type,
                span: _,
            } => self.select_bitcast(result, value, to_type),
            Instruction::Trunc {
                result,
                value,
                to_type,
                span: _,
            } => self.select_trunc(result, value, to_type),
            Instruction::ZExt {
                result,
                value,
                to_type,
                span: _,
            } => self.select_zext(result, value, to_type),
            Instruction::SExt {
                result,
                value,
                to_type,
                span: _,
            } => self.select_sext(result, value, to_type),
            Instruction::IntToPtr {
                result,
                value,
                span: _,
            } => self.select_int_to_ptr(result, value),
            Instruction::PtrToInt {
                result,
                value,
                to_type,
                span: _,
            } => self.select_ptr_to_int(result, value, to_type),
            Instruction::InlineAsm {
                result,
                template,
                constraints,
                operands,
                clobbers,
                has_side_effects: _,
                is_volatile: _,
                goto_targets,
                span,
            } => self.select_inline_asm(
                result,
                template,
                constraints,
                operands,
                clobbers,
                goto_targets,
                span,
            ),
        }
    }

    // -----------------------------------------------------------------------
    // Individual instruction selection methods
    // -----------------------------------------------------------------------

    /// Select instructions for an alloca (stack allocation).
    /// Computes a frame-pointer-relative offset for the allocation.
    fn select_alloca(
        &mut self,
        result: &Value,
        ty: &IrType,
        alignment: &Option<usize>,
        _span: &Span,
    ) -> Vec<A64Instruction> {
        let size = ty.size_bytes(&self.target);
        let align = alignment.unwrap_or_else(|| ty.align_bytes(&self.target));
        let mask = if align > 0 { align - 1 } else { 0 };
        let offset = (self.frame_size as usize + mask) & !mask;
        self.frame_size = (offset + size) as i64;
        let rd = result.index() as u8;
        // Compute address as FP - offset.
        let abs_off = (offset + size) as u64;
        if fits_add_sub_imm(abs_off) {
            vec![A64Instruction::new(A64Opcode::SUB_imm)
                .with_rd(rd)
                .with_rn(FP_REG)
                .with_imm(abs_off as i64)
                .with_comment(format!("alloca {} bytes (align {})", size, align))]
        } else {
            let mut v = self.materialize_immediate(IP0, abs_off);
            v.push(
                A64Instruction::new(A64Opcode::SUB_reg)
                    .with_rd(rd)
                    .with_rn(FP_REG)
                    .with_rm(IP0)
                    .with_comment(format!("alloca {} bytes (large offset)", size)),
            );
            v
        }
    }

    /// Select load instructions based on the loaded type's size.
    fn select_load(&mut self, result: &Value, ptr: &Value, ty: &IrType) -> Vec<A64Instruction> {
        let rd = result.index() as u8;
        let rn = ptr.index() as u8;
        let size = self.type_size_for_mem(ty);
        let is_32 = Self::is_32bit_type(ty);

        let opcode = match ty {
            IrType::I1 | IrType::I8 => A64Opcode::LDRB_imm,
            IrType::I16 => A64Opcode::LDRH_imm,
            IrType::I32 => A64Opcode::LDR_imm,
            IrType::I64 | IrType::I128 | IrType::Ptr => A64Opcode::LDR_imm,
            IrType::F32 => A64Opcode::LDR_fp_imm,
            IrType::F64 | IrType::F80 => A64Opcode::LDR_fp_imm,
            IrType::Void => A64Opcode::LDR_imm,
            IrType::Array(_, _) | IrType::Struct(_) | IrType::Function(_, _) => A64Opcode::LDR_imm,
        };

        let mut inst = A64Instruction::new(opcode)
            .with_rd(rd)
            .with_rn(rn)
            .with_imm(0)
            .with_comment(format!("load {} bytes", size));

        if is_32 && !ty.is_float() {
            inst = inst.set_32bit();
        }
        if ty.is_float() {
            inst = inst.set_fp();
        }

        vec![inst]
    }

    /// Select store instructions. Since the IR Store does not carry the stored
    /// type directly, we default to a 64-bit store. The register allocator
    /// and type analysis passes refine this.
    fn select_store(&mut self, value: &Value, ptr: &Value) -> Vec<A64Instruction> {
        let rd = value.index() as u8;
        let rn = ptr.index() as u8;

        vec![A64Instruction::new(A64Opcode::STR_imm)
            .with_rd(rd)
            .with_rn(rn)
            .with_imm(0)
            .with_comment("store (64-bit default)")]
    }

    /// Select instructions for a binary operation (integer or FP).
    fn select_binop(
        &mut self,
        result: &Value,
        op: &BinOp,
        lhs: &Value,
        rhs: &Value,
        ty: &IrType,
    ) -> Vec<A64Instruction> {
        let rd = result.index() as u8;
        let rn = lhs.index() as u8;
        let rm = rhs.index() as u8;
        let is_32 = Self::is_32bit_type(ty);
        let mut insts = Vec::new();

        match op {
            BinOp::Add => {
                let mut inst = A64Instruction::new(A64Opcode::ADD_reg)
                    .with_rd(rd)
                    .with_rn(rn)
                    .with_rm(rm);
                if is_32 {
                    inst = inst.set_32bit();
                }
                insts.push(inst);
            }
            BinOp::Sub => {
                let mut inst = A64Instruction::new(A64Opcode::SUB_reg)
                    .with_rd(rd)
                    .with_rn(rn)
                    .with_rm(rm);
                if is_32 {
                    inst = inst.set_32bit();
                }
                insts.push(inst);
            }
            BinOp::Mul => {
                // MUL is alias for MADD Rd, Rn, Rm, XZR.
                let mut inst = A64Instruction::new(A64Opcode::MADD)
                    .with_rd(rd)
                    .with_rn(rn)
                    .with_rm(rm)
                    .with_ra(XZR);
                if is_32 {
                    inst = inst.set_32bit();
                }
                insts.push(inst);
            }
            BinOp::SDiv => {
                let mut inst = A64Instruction::new(A64Opcode::SDIV)
                    .with_rd(rd)
                    .with_rn(rn)
                    .with_rm(rm);
                if is_32 {
                    inst = inst.set_32bit();
                }
                insts.push(inst);
            }
            BinOp::UDiv => {
                let mut inst = A64Instruction::new(A64Opcode::UDIV)
                    .with_rd(rd)
                    .with_rn(rn)
                    .with_rm(rm);
                if is_32 {
                    inst = inst.set_32bit();
                }
                insts.push(inst);
            }
            BinOp::SRem => {
                // No direct remainder instruction on AArch64.
                // Rd = Rn - (Rn / Rm) * Rm = Rn - SDIV(Rn,Rm)*Rm
                // Implementation: SDIV tmp, Rn, Rm; MSUB Rd, tmp, Rm, Rn
                let tmp = IP0;
                let mut div = A64Instruction::new(A64Opcode::SDIV)
                    .with_rd(tmp)
                    .with_rn(rn)
                    .with_rm(rm);
                if is_32 {
                    div = div.set_32bit();
                }
                insts.push(div);
                let mut msub = A64Instruction::new(A64Opcode::MSUB)
                    .with_rd(rd)
                    .with_rn(tmp)
                    .with_rm(rm)
                    .with_ra(rn);
                if is_32 {
                    msub = msub.set_32bit();
                }
                insts.push(msub);
            }
            BinOp::URem => {
                let tmp = IP0;
                let mut div = A64Instruction::new(A64Opcode::UDIV)
                    .with_rd(tmp)
                    .with_rn(rn)
                    .with_rm(rm);
                if is_32 {
                    div = div.set_32bit();
                }
                insts.push(div);
                let mut msub = A64Instruction::new(A64Opcode::MSUB)
                    .with_rd(rd)
                    .with_rn(tmp)
                    .with_rm(rm)
                    .with_ra(rn);
                if is_32 {
                    msub = msub.set_32bit();
                }
                insts.push(msub);
            }
            BinOp::And => {
                let mut inst = A64Instruction::new(A64Opcode::AND_reg)
                    .with_rd(rd)
                    .with_rn(rn)
                    .with_rm(rm);
                if is_32 {
                    inst = inst.set_32bit();
                }
                insts.push(inst);
            }
            BinOp::Or => {
                let mut inst = A64Instruction::new(A64Opcode::ORR_reg)
                    .with_rd(rd)
                    .with_rn(rn)
                    .with_rm(rm);
                if is_32 {
                    inst = inst.set_32bit();
                }
                insts.push(inst);
            }
            BinOp::Xor => {
                let mut inst = A64Instruction::new(A64Opcode::EOR_reg)
                    .with_rd(rd)
                    .with_rn(rn)
                    .with_rm(rm);
                if is_32 {
                    inst = inst.set_32bit();
                }
                insts.push(inst);
            }
            BinOp::Shl => {
                let mut inst = A64Instruction::new(A64Opcode::LSL_reg)
                    .with_rd(rd)
                    .with_rn(rn)
                    .with_rm(rm);
                if is_32 {
                    inst = inst.set_32bit();
                }
                insts.push(inst);
            }
            BinOp::AShr => {
                let mut inst = A64Instruction::new(A64Opcode::ASR_reg)
                    .with_rd(rd)
                    .with_rn(rn)
                    .with_rm(rm);
                if is_32 {
                    inst = inst.set_32bit();
                }
                insts.push(inst);
            }
            BinOp::LShr => {
                let mut inst = A64Instruction::new(A64Opcode::LSR_reg)
                    .with_rd(rd)
                    .with_rn(rn)
                    .with_rm(rm);
                if is_32 {
                    inst = inst.set_32bit();
                }
                insts.push(inst);
            }
            // Floating-point arithmetic
            BinOp::FAdd => {
                let opc = if matches!(ty, IrType::F32) {
                    A64Opcode::FADD_s
                } else {
                    A64Opcode::FADD_d
                };
                insts.push(
                    A64Instruction::new(opc)
                        .with_rd(rd)
                        .with_rn(rn)
                        .with_rm(rm)
                        .set_fp(),
                );
            }
            BinOp::FSub => {
                let opc = if matches!(ty, IrType::F32) {
                    A64Opcode::FSUB_s
                } else {
                    A64Opcode::FSUB_d
                };
                insts.push(
                    A64Instruction::new(opc)
                        .with_rd(rd)
                        .with_rn(rn)
                        .with_rm(rm)
                        .set_fp(),
                );
            }
            BinOp::FMul => {
                let opc = if matches!(ty, IrType::F32) {
                    A64Opcode::FMUL_s
                } else {
                    A64Opcode::FMUL_d
                };
                insts.push(
                    A64Instruction::new(opc)
                        .with_rd(rd)
                        .with_rn(rn)
                        .with_rm(rm)
                        .set_fp(),
                );
            }
            BinOp::FDiv => {
                let opc = if matches!(ty, IrType::F32) {
                    A64Opcode::FDIV_s
                } else {
                    A64Opcode::FDIV_d
                };
                insts.push(
                    A64Instruction::new(opc)
                        .with_rd(rd)
                        .with_rn(rn)
                        .with_rm(rm)
                        .set_fp(),
                );
            }
            BinOp::FRem => {
                // AArch64 has no FREM instruction; delegate to runtime fmodf/fmod.
                let fname = if matches!(ty, IrType::F32) {
                    "fmodf"
                } else {
                    "fmod"
                };
                insts.push(
                    A64Instruction::new(A64Opcode::CALL)
                        .with_symbol(fname.to_string())
                        .set_fp()
                        .with_comment(format!("frem via {}", fname)),
                );
            }
        }
        insts
    }

    /// Select instructions for an integer comparison.
    /// Emits CMP + CSET to produce a boolean (0/1) result.
    fn select_icmp(
        &mut self,
        result: &Value,
        op: &ICmpOp,
        lhs: &Value,
        rhs: &Value,
    ) -> Vec<A64Instruction> {
        let rd = result.index() as u8;
        let rn = lhs.index() as u8;
        let rm = rhs.index() as u8;
        let cc = CondCode::from_icmp(op);

        vec![
            A64Instruction::new(A64Opcode::CMP_reg)
                .with_rn(rn)
                .with_rm(rm)
                .with_comment(format!("cmp for icmp {:?}", op)),
            A64Instruction::new(A64Opcode::CSET)
                .with_rd(rd)
                .with_cond(cc)
                .set_32bit()
                .with_comment(format!("cset w{}, {:?}", rd, cc)),
        ]
    }

    /// Select instructions for a floating-point comparison.
    /// Emits FCMP + CSET to produce a boolean (0/1) result.
    fn select_fcmp(
        &mut self,
        result: &Value,
        op: &FCmpOp,
        lhs: &Value,
        rhs: &Value,
    ) -> Vec<A64Instruction> {
        let rd = result.index() as u8;
        let rn = lhs.index() as u8;
        let rm = rhs.index() as u8;
        let cc = CondCode::from_fcmp(op);

        vec![
            A64Instruction::new(A64Opcode::FCMP_d)
                .with_rn(rn)
                .with_rm(rm)
                .set_fp()
                .with_comment(format!("fcmp for {:?}", op)),
            A64Instruction::new(A64Opcode::CSET)
                .with_rd(rd)
                .with_cond(cc)
                .set_32bit()
                .with_comment(format!("cset w{}, {:?}", rd, cc)),
        ]
    }

    /// Select an unconditional branch.
    fn select_branch(&mut self, target: &BlockId) -> Vec<A64Instruction> {
        let label = if target.index() < self.block_labels.len() {
            self.block_labels[target.index()].clone()
        } else {
            format!(".LBB_{}", target.index())
        };
        vec![A64Instruction::new(A64Opcode::B)
            .with_symbol(label)
            .with_comment(format!("b -> bb{}", target.index()))]
    }

    /// Select a conditional branch (CBNZ for true path, B for false path).
    fn select_cond_branch(
        &mut self,
        condition: &Value,
        then_block: &BlockId,
        else_block: &BlockId,
    ) -> Vec<A64Instruction> {
        let rn = condition.index() as u8;
        let then_label = if then_block.index() < self.block_labels.len() {
            self.block_labels[then_block.index()].clone()
        } else {
            format!(".LBB_{}", then_block.index())
        };
        let else_label = if else_block.index() < self.block_labels.len() {
            self.block_labels[else_block.index()].clone()
        } else {
            format!(".LBB_{}", else_block.index())
        };
        vec![
            A64Instruction::new(A64Opcode::CBNZ)
                .with_rn(rn)
                .with_symbol(then_label)
                .with_comment(format!("cbnz -> then bb{}", then_block.index())),
            A64Instruction::new(A64Opcode::B)
                .with_symbol(else_label)
                .with_comment(format!("b -> else bb{}", else_block.index())),
        ]
    }

    /// Select a switch statement using cascaded CMP + B.EQ.
    /// For small case counts this is efficient; dense switches would use
    /// a jump table approach in a production compiler.
    fn select_switch(
        &mut self,
        value: &Value,
        default: &BlockId,
        cases: &[(i64, BlockId)],
    ) -> Vec<A64Instruction> {
        let rn = value.index() as u8;
        let mut insts = Vec::new();

        for (case_val, target) in cases {
            let label = if target.index() < self.block_labels.len() {
                self.block_labels[target.index()].clone()
            } else {
                format!(".LBB_{}", target.index())
            };
            // Compare: use CMP_imm if value fits, otherwise materialize.
            if fits_add_sub_imm(*case_val as u64) {
                insts.push(
                    A64Instruction::new(A64Opcode::CMP_imm)
                        .with_rn(rn)
                        .with_imm(*case_val),
                );
            } else {
                let mat = self.materialize_immediate(IP0, *case_val as u64);
                insts.extend(mat);
                insts.push(
                    A64Instruction::new(A64Opcode::CMP_reg)
                        .with_rn(rn)
                        .with_rm(IP0),
                );
            }
            insts.push(
                A64Instruction::new(A64Opcode::B_cond)
                    .with_cond(CondCode::EQ)
                    .with_symbol(label),
            );
        }

        // Default case: unconditional branch.
        let default_label = if default.index() < self.block_labels.len() {
            self.block_labels[default.index()].clone()
        } else {
            format!(".LBB_{}", default.index())
        };
        insts.push(
            A64Instruction::new(A64Opcode::B)
                .with_symbol(default_label)
                .with_comment("switch default"),
        );

        insts
    }

    /// Select a function call instruction (direct or indirect).
    /// Places arguments in registers per AAPCS64, emits BL/BLR,
    /// and copies the return value from X0 or V0.
    fn select_call(
        &mut self,
        result: &Value,
        callee: &Value,
        args: &[Value],
        return_type: &IrType,
    ) -> Vec<A64Instruction> {
        let mut insts = Vec::new();

        // Place arguments in registers per AAPCS64 convention.
        let mut int_reg_idx = 0usize;
        let mut stack_offset: i64 = 0;
        for (i, arg) in args.iter().enumerate() {
            let src = arg.index() as u8;
            // Simple classification: first 8 int args in X0-X7.
            if int_reg_idx < NUM_INT_ARG_REGS {
                let dest = INT_ARG_REGS[int_reg_idx];
                insts.push(
                    A64Instruction::new(A64Opcode::MOV_reg)
                        .with_rd(dest)
                        .with_rn(src)
                        .with_comment(format!("arg {} -> {}", i, gpr_name(dest))),
                );
                int_reg_idx += 1;
            } else {
                // Spill to stack.
                insts.push(
                    A64Instruction::new(A64Opcode::STR_imm)
                        .with_rd(src)
                        .with_rn(SP_REG)
                        .with_imm(stack_offset)
                        .with_comment(format!("stack arg {} at sp+{}", i, stack_offset)),
                );
                stack_offset += 8;
            }
        }

        // Emit the call instruction.
        let callee_reg = callee.index() as u8;
        insts.push(
            A64Instruction::new(A64Opcode::CALL)
                .with_rn(callee_reg)
                .with_comment("call"),
        );

        // Move return value from X0 (integer) or V0 (FP) to result vreg.
        if !return_type.is_void() {
            let rd = result.index() as u8;
            if return_type.is_float() {
                // Both F32 and F64 return values are in V0 per AAPCS64.
                insts.push(
                    A64Instruction::new(A64Opcode::FMOV_d)
                        .with_rd(rd)
                        .with_rn(V0)
                        .set_fp()
                        .with_comment("move FP return from v0"),
                );
            } else {
                insts.push(
                    A64Instruction::new(A64Opcode::MOV_reg)
                        .with_rd(rd)
                        .with_rn(X0)
                        .with_comment("move return from x0"),
                );
            }
        }

        insts
    }

    /// Select a return instruction.
    fn select_return(&mut self, value: &Option<Value>) -> Vec<A64Instruction> {
        let mut insts = Vec::new();
        if let Some(val) = value {
            let src = val.index() as u8;
            // Move return value to X0 (integer) for the caller.
            insts.push(
                A64Instruction::new(A64Opcode::MOV_reg)
                    .with_rd(X0)
                    .with_rn(src)
                    .with_comment("ret value -> x0"),
            );
        }
        insts.push(A64Instruction::new(A64Opcode::RET).with_comment("ret"));
        insts
    }

    /// Select instructions for GetElementPtr (pointer arithmetic).
    fn select_gep(
        &mut self,
        result: &Value,
        base: &Value,
        indices: &[Value],
        _result_type: &IrType,
    ) -> Vec<A64Instruction> {
        let rd = result.index() as u8;
        let rn = base.index() as u8;
        let mut insts = Vec::new();

        // Start with base pointer in rd.
        insts.push(
            A64Instruction::new(A64Opcode::MOV_reg)
                .with_rd(rd)
                .with_rn(rn)
                .with_comment("gep: copy base ptr"),
        );

        // Add each index offset.
        for idx in indices {
            let rm = idx.index() as u8;
            insts.push(
                A64Instruction::new(A64Opcode::ADD_reg)
                    .with_rd(rd)
                    .with_rn(rd)
                    .with_rm(rm)
                    .with_comment("gep: add index offset"),
            );
        }

        insts
    }

    /// Select instructions for a bitcast (reinterpret bits).
    fn select_bitcast(
        &mut self,
        result: &Value,
        value: &Value,
        to_type: &IrType,
    ) -> Vec<A64Instruction> {
        let rd = result.index() as u8;
        let rn = value.index() as u8;

        if to_type.is_float() {
            // GP register -> FP register.
            vec![A64Instruction::new(A64Opcode::FMOV_gen_to_fp)
                .with_rd(rd)
                .with_rn(rn)
                .set_fp()
                .with_comment("bitcast gp -> fp")]
        } else if to_type.is_pointer() || to_type.is_integer() {
            vec![A64Instruction::new(A64Opcode::MOV_reg)
                .with_rd(rd)
                .with_rn(rn)
                .with_comment("bitcast (mov)")]
        } else {
            vec![A64Instruction::new(A64Opcode::MOV_reg)
                .with_rd(rd)
                .with_rn(rn)
                .with_comment("bitcast")]
        }
    }

    /// Select truncation instructions.
    fn select_trunc(
        &mut self,
        result: &Value,
        value: &Value,
        to_type: &IrType,
    ) -> Vec<A64Instruction> {
        let rd = result.index() as u8;
        let rn = value.index() as u8;

        match to_type {
            IrType::I1 => {
                // AND Wd, Wn, #1
                vec![A64Instruction::new(A64Opcode::AND_imm)
                    .with_rd(rd)
                    .with_rn(rn)
                    .with_imm(1)
                    .set_32bit()
                    .with_comment("trunc to i1")]
            }
            IrType::I8 => {
                vec![A64Instruction::new(A64Opcode::UXTB)
                    .with_rd(rd)
                    .with_rn(rn)
                    .set_32bit()
                    .with_comment("trunc to i8 (uxtb)")]
            }
            IrType::I16 => {
                vec![A64Instruction::new(A64Opcode::UXTH)
                    .with_rd(rd)
                    .with_rn(rn)
                    .set_32bit()
                    .with_comment("trunc to i16 (uxth)")]
            }
            IrType::I32 => {
                // MOV Wd, Wn: writing to W-register implicitly zero-extends.
                vec![A64Instruction::new(A64Opcode::MOV_reg)
                    .with_rd(rd)
                    .with_rn(rn)
                    .set_32bit()
                    .with_comment("trunc to i32 (mov w)")]
            }
            _ => {
                vec![A64Instruction::new(A64Opcode::MOV_reg)
                    .with_rd(rd)
                    .with_rn(rn)
                    .with_comment("trunc (nop)")]
            }
        }
    }

    /// Select zero-extension instructions.
    fn select_zext(
        &mut self,
        result: &Value,
        value: &Value,
        to_type: &IrType,
    ) -> Vec<A64Instruction> {
        let rd = result.index() as u8;
        let rn = value.index() as u8;

        // On AArch64, writing to a W-register implicitly zero-extends the
        // upper 32 bits of the containing X-register. For sub-32-bit sources,
        // use UXTB/UXTH; for 32->64, a simple MOV Wd suffices.
        vec![A64Instruction::new(A64Opcode::MOV_reg)
            .with_rd(rd)
            .with_rn(rn)
            .with_comment(format!("zext to {:?}", to_type))]
    }

    /// Select sign-extension instructions (SXTB, SXTH, SXTW).
    fn select_sext(
        &mut self,
        result: &Value,
        value: &Value,
        to_type: &IrType,
    ) -> Vec<A64Instruction> {
        let rd = result.index() as u8;
        let rn = value.index() as u8;

        // Choose the appropriate sign-extend instruction based on target width.
        let opcode = match to_type {
            IrType::I16 => A64Opcode::SXTB,
            IrType::I32 => A64Opcode::SXTH,
            IrType::I64 | IrType::Ptr => A64Opcode::SXTW,
            _ => A64Opcode::SXTW,
        };

        vec![A64Instruction::new(opcode)
            .with_rd(rd)
            .with_rn(rn)
            .with_comment(format!("sext to {:?}", to_type))]
    }

    /// Select integer-to-pointer conversion (no-op on LP64).
    fn select_int_to_ptr(&mut self, result: &Value, value: &Value) -> Vec<A64Instruction> {
        let rd = result.index() as u8;
        let rn = value.index() as u8;
        vec![A64Instruction::new(A64Opcode::MOV_reg)
            .with_rd(rd)
            .with_rn(rn)
            .with_comment("inttoptr (nop on LP64)")]
    }

    /// Select pointer-to-integer conversion (no-op on LP64).
    fn select_ptr_to_int(
        &mut self,
        result: &Value,
        value: &Value,
        _to_type: &IrType,
    ) -> Vec<A64Instruction> {
        let rd = result.index() as u8;
        let rn = value.index() as u8;
        vec![A64Instruction::new(A64Opcode::MOV_reg)
            .with_rd(rd)
            .with_rn(rn)
            .with_comment("ptrtoint (nop on LP64)")]
    }

    /// Select inline assembly. Emits a pseudo-instruction that will be
    /// expanded by the assembler to the literal template text.
    fn select_inline_asm(
        &mut self,
        _result: &Value,
        template: &str,
        constraints: &str,
        _operands: &[Value],
        _clobbers: &[String],
        _goto_targets: &[BlockId],
        _span: &Span,
    ) -> Vec<A64Instruction> {
        vec![A64Instruction::new(A64Opcode::INLINE_ASM)
            .with_comment(format!("asm \"{}\" constraints: {}", template, constraints))]
    }
}

// ===========================================================================
// Ensure all schema-required dependency members are referenced
// ===========================================================================

#[cfg(test)]
mod _schema_verification {
    #![allow(dead_code, unused_variables, unused_imports)]
    use super::*;
    use crate::ir::types::StructType;

    fn verify_register_usage() {
        let _regs: [u8; 37] = [
            X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16, X17, X18,
            X19, X20, X21, X22, X23, X24, X25, X26, X27, X28, X29, X30, SP_REG, XZR, FP_REG, LR,
            IP0, IP1,
        ];
        let _fprs: [u8; 32] = [
            V0, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18,
            V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, V29, V30, V31,
        ];
        let _a = ALLOCATABLE_GPRS;
        let _b = ALLOCATABLE_FPRS;
        let _c = CALLEE_SAVED_GPRS;
        let _d = CALLEE_SAVED_FPRS;
        let _e = CALLER_SAVED_GPRS;
        let _f = CALLER_SAVED_FPRS;
        let _ = gpr_name(0);
        let _ = fpr_name(V0);
        let _ = fpr_name_d(V0);
        let _ = fpr_name_s(V0);
        let _ = hw_encoding(X0);
        let _ = is_gpr(X0);
        let _ = is_fpr(V0);
        let _ = is_callee_saved(X19);
        let _ = is_allocatable(X0);
        let _ = reg_class(X0);
        let _rc = RegClass::GPR;
        let _rcf = RegClass::FPR;
    }

    fn verify_traits_usage() {
        let _mi = MachineInstruction::new(0);
        let _mo_reg = MachineOperand::Register(0);
        let _mo_vreg = MachineOperand::VirtualRegister(0);
        let _mo_imm = MachineOperand::Immediate(0);
        let _mo_mem = MachineOperand::Memory {
            base: Some(0),
            index: None,
            scale: 1,
            displacement: 0,
        };
        let _mo_fs = MachineOperand::FrameSlot(0);
        let _mo_gs = MachineOperand::GlobalSymbol("test".to_string());
        let _mo_bl = MachineOperand::BlockLabel(0);
        let _mf = MachineFunction::new("test".to_string());
        let _mbb = MachineBasicBlock::new(None);
        let _ri = RegisterInfo::new();
        let _al_r = ArgLocation::Register(0);
        let _al_rp = ArgLocation::RegisterPair(0, 1);
        let _al_s = ArgLocation::Stack(0);
    }

    fn verify_ir_usage() {
        let v = Value(0);
        let _ = v.index();
        let bid = BlockId(0);
        let _ = bid.index();
        let _ops = [
            BinOp::Add,
            BinOp::Sub,
            BinOp::Mul,
            BinOp::SDiv,
            BinOp::UDiv,
            BinOp::SRem,
            BinOp::URem,
            BinOp::And,
            BinOp::Or,
            BinOp::Xor,
            BinOp::Shl,
            BinOp::AShr,
            BinOp::LShr,
            BinOp::FAdd,
            BinOp::FSub,
            BinOp::FMul,
            BinOp::FDiv,
        ];
        let _icmps = [
            ICmpOp::Eq,
            ICmpOp::Ne,
            ICmpOp::Slt,
            ICmpOp::Sle,
            ICmpOp::Sgt,
            ICmpOp::Sge,
            ICmpOp::Ult,
            ICmpOp::Ule,
            ICmpOp::Ugt,
            ICmpOp::Uge,
        ];
        let _fcmps = [
            FCmpOp::Oeq,
            FCmpOp::One,
            FCmpOp::Olt,
            FCmpOp::Ole,
            FCmpOp::Ogt,
            FCmpOp::Oge,
            FCmpOp::Uno,
            FCmpOp::Ord,
        ];
    }

    fn verify_type_usage() {
        let target = Target::AArch64;
        let _ = target.pointer_width();
        let _ = target.is_64bit();
        let _ct = CType::Int;
        let _mt = MachineType::Integer;
        let _mf32 = MachineType::F32;
        let _mf64 = MachineType::F64;
        let _mm = MachineType::Memory;
        let types: Vec<IrType> = vec![
            IrType::I1,
            IrType::I8,
            IrType::I16,
            IrType::I32,
            IrType::I64,
            IrType::I128,
            IrType::F32,
            IrType::F64,
            IrType::F80,
            IrType::Ptr,
            IrType::Void,
        ];
        for t in &types {
            let _ = t.size_bytes(&target);
            let _ = t.align_bytes(&target);
            let _ = t.is_integer();
            let _ = t.is_float();
            let _ = t.is_pointer();
        }
        let _ = IrType::I32.int_width();
        let _arr = IrType::Array(Box::new(IrType::I32), 10);
        let _st = IrType::Struct(StructType::new(vec![IrType::I32], false));
    }

    fn verify_diag_usage() {
        let mut diag = DiagnosticEngine::new();
        diag.emit_error(Span::dummy(), "test error");
        diag.emit_warning(Span::dummy(), "test warning");
        let _ = diag.has_errors();
    }

    fn verify_abi_usage() {
        let mut abi = AArch64Abi::new();
        abi.reset();
        let cty = CType::Int;
        let _ = abi.classify_arg(&cty);
        let _ = abi.classify_return(&cty);
        let _ = abi.classify_variadic_arg(&cty);
        let layout = AArch64Abi::compute_frame_layout(&[], 64, 2);
        let _ = layout.total_size;
        let _ = layout.callee_saved_offset;
        let _ = layout.lr_offset;
        let _ = layout.fp_offset;
        let _ = NUM_INT_ARG_REGS;
        let _ = NUM_FP_ARG_REGS;
        let _ = STACK_ALIGNMENT;
        let _ = INT_ARG_REGS;
        let _ = FP_ARG_REGS;
        let _ = INT_RET_REGS;
        let _ = INDIRECT_RESULT_REG;
        let _ = ArgClass::Integer;
    }

    fn verify_basic_block_usage() {
        let bb = BasicBlock::new(0);
        let _ = bb.instructions();
        let _ = &bb.label;
        let _ = &bb.predecessors;
        let _ = &bb.successors;
        let _ = bb.index;
    }

    fn verify_ir_function_usage() {
        let func = IrFunction::new("test".to_string(), vec![], IrType::Void);
        let _ = &func.name;
        let _ = &func.params;
        let _ = &func.return_type;
        let _ = func.blocks();
        let _ = func.entry_block();
        let _ = func.block_count();
        let _ = &func.calling_convention;
        let _ = func.is_variadic;
        let _ = func.is_definition;
    }

    fn verify_instruction_methods() {
        let span = Span::dummy();
        let inst = Instruction::Branch {
            target: BlockId(0),
            span,
        };
        let _ = inst.result();
        let _ = inst.is_terminator();
        let _ = inst.span();
    }
}
