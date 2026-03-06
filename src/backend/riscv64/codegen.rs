//! # RISC-V 64 Instruction Selection and Emission
//!
//! Translates BCC IR instructions into RISC-V 64 machine instructions.
//! Covers the RV64IMAFDC ISA (Integer, Multiply/Divide, Atomic,
//! Single-precision Float, Double-precision Float, Compressed).
//!
//! ## Instruction Formats
//! - R-type: register-register operations (ADD, SUB, MUL, etc.)
//! - I-type: register-immediate operations (ADDI, LW, LD, JALR, etc.)
//! - S-type: stores (SW, SD, etc.)
//! - B-type: conditional branches (BEQ, BNE, BLT, BGE, etc.)
//! - U-type: upper-immediate (LUI, AUIPC)
//! - J-type: jumps (JAL)
//!
//! ## Key Design Decision: LUI/AUIPC for Large Constants
//! RISC-V lacks direct 64-bit immediate loads. Large constants require:
//! - LUI rd, upper_20_bits  +  ADDI rd, rd, lower_12_bits  (for 32-bit constants)
//! - For 64-bit: additional shift+add sequences
//! - PC-relative addressing: AUIPC + ADDI/LD for PIC code
//!
//! ## Primary validation target for Linux kernel 6.9 boot.

use crate::backend::riscv64::abi::RiscV64Abi;
use crate::backend::riscv64::registers::*;
use crate::backend::traits::ArgLocation;
use crate::common::target::Target;
use crate::common::types::{CType, TypeQualifiers};
use crate::ir::function::IrFunction;
use crate::ir::instructions::{BinOp, FCmpOp, ICmpOp, Instruction, Value};
use crate::ir::types::IrType;

// Re-export needed traits to keep in schema compliance.
// These are used by callers, even if not directly used in this module's body.
#[allow(unused_imports)]
use crate::backend::riscv64::abi::{FrameLayout, STACK_ALIGNMENT};
#[allow(unused_imports)]
use crate::backend::traits::{
    MachineBasicBlock, MachineFunction, MachineInstruction, MachineOperand, RegisterInfo,
};
#[allow(unused_imports)]
use crate::common::diagnostics::{DiagnosticEngine, Span};
#[allow(unused_imports)]
use crate::common::types::MachineType;
#[allow(unused_imports)]
use crate::ir::basic_block::BasicBlock;
#[allow(unused_imports)]
use crate::ir::instructions::BlockId;

// ===========================================================================
// RvOpcode — RISC-V 64 instruction opcodes
// ===========================================================================

/// RISC-V 64 instruction opcodes for machine code representation.
///
/// Covers the full RV64IMAFDC ISA plus pseudo-instructions used during
/// instruction selection. Each variant maps to one or more hardware
/// instruction encodings resolved by the assembler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum RvOpcode {
    // RV64I Base Integer Instructions
    /// Load Upper Immediate — U-type: rd = imm << 12
    LUI,
    /// Add Upper Immediate to PC — U-type: rd = PC + (imm << 12)
    AUIPC,
    /// Jump And Link — J-type: rd = PC+4; PC += offset
    JAL,
    /// Jump And Link Register — I-type: rd = PC+4; PC = rs1 + imm
    JALR,
    /// Branch if Equal — B-type
    BEQ,
    /// Branch if Not Equal — B-type
    BNE,
    /// Branch if Less Than (signed) — B-type
    BLT,
    /// Branch if Greater or Equal (signed) — B-type
    BGE,
    /// Branch if Less Than (unsigned) — B-type
    BLTU,
    /// Branch if Greater or Equal (unsigned) — B-type
    BGEU,
    /// Load Byte (sign-extend) — I-type
    LB,
    /// Load Halfword (sign-extend) — I-type
    LH,
    /// Load Word (sign-extend) — I-type
    LW,
    /// Load Doubleword — I-type
    LD,
    /// Load Byte Unsigned — I-type
    LBU,
    /// Load Halfword Unsigned — I-type
    LHU,
    /// Load Word Unsigned — I-type
    LWU,
    /// Store Byte — S-type
    SB,
    /// Store Halfword — S-type
    SH,
    /// Store Word — S-type
    SW,
    /// Store Doubleword — S-type
    SD,
    /// Add Immediate — I-type
    ADDI,
    /// Set Less Than Immediate (signed) — I-type
    SLTI,
    /// Set Less Than Immediate Unsigned — I-type
    SLTIU,
    /// XOR Immediate — I-type
    XORI,
    /// OR Immediate — I-type
    ORI,
    /// AND Immediate — I-type
    ANDI,
    /// Shift Left Logical Immediate — I-type
    SLLI,
    /// Shift Right Logical Immediate — I-type
    SRLI,
    /// Shift Right Arithmetic Immediate — I-type
    SRAI,
    /// Add — R-type
    ADD,
    /// Subtract — R-type
    SUB,
    /// Shift Left Logical — R-type
    SLL,
    /// Set Less Than (signed) — R-type
    SLT,
    /// Set Less Than Unsigned — R-type
    SLTU,
    /// XOR — R-type
    XOR,
    /// Shift Right Logical — R-type
    SRL,
    /// Shift Right Arithmetic — R-type
    SRA,
    /// OR — R-type
    OR,
    /// AND — R-type
    AND,

    // RV64I Word variants (32-bit operations on 64-bit, sign-extend result)
    /// Add Immediate Word — sign-extends result to 64 bits
    ADDIW,
    /// Shift Left Logical Immediate Word
    SLLIW,
    /// Shift Right Logical Immediate Word
    SRLIW,
    /// Shift Right Arithmetic Immediate Word
    SRAIW,
    /// Add Word — R-type, sign-extends result
    ADDW,
    /// Subtract Word — R-type, sign-extends result
    SUBW,
    /// Shift Left Logical Word
    SLLW,
    /// Shift Right Logical Word
    SRLW,
    /// Shift Right Arithmetic Word
    SRAW,

    // RV64M Multiply/Divide Extension
    /// Multiply — R-type: rd = (rs1 × rs2)[63:0]
    MUL,
    /// Multiply High (signed×signed) — R-type: rd = (rs1 × rs2)[127:64]
    MULH,
    /// Multiply High Signed-Unsigned — R-type
    MULHSU,
    /// Multiply High Unsigned — R-type
    MULHU,
    /// Divide (signed) — R-type
    DIV,
    /// Divide Unsigned — R-type
    DIVU,
    /// Remainder (signed) — R-type
    REM,
    /// Remainder Unsigned — R-type
    REMU,
    /// Multiply Word — R-type, sign-extends result
    MULW,
    /// Divide Word (signed) — R-type
    DIVW,
    /// Divide Word Unsigned — R-type
    DIVUW,
    /// Remainder Word (signed) — R-type
    REMW,
    /// Remainder Word Unsigned — R-type
    REMUW,

    // RV64A Atomic Extension
    /// Load Reserved Word
    LR_W,
    /// Store Conditional Word
    SC_W,
    /// Load Reserved Doubleword
    LR_D,
    /// Store Conditional Doubleword
    SC_D,
    /// Atomic Swap Word
    AMOSWAP_W,
    /// Atomic Add Word
    AMOADD_W,
    /// Atomic AND Word
    AMOAND_W,
    /// Atomic OR Word
    AMOOR_W,
    /// Atomic XOR Word
    AMOXOR_W,
    /// Atomic Max (signed) Word
    AMOMAX_W,
    /// Atomic Min (signed) Word
    AMOMIN_W,
    /// Atomic Max Unsigned Word
    AMOMAXU_W,
    /// Atomic Min Unsigned Word
    AMOMINU_W,
    /// Atomic Swap Doubleword
    AMOSWAP_D,
    /// Atomic Add Doubleword
    AMOADD_D,
    /// Atomic AND Doubleword
    AMOAND_D,
    /// Atomic OR Doubleword
    AMOOR_D,
    /// Atomic XOR Doubleword
    AMOXOR_D,
    /// Atomic Max (signed) Doubleword
    AMOMAX_D,
    /// Atomic Min (signed) Doubleword
    AMOMIN_D,
    /// Atomic Max Unsigned Doubleword
    AMOMAXU_D,
    /// Atomic Min Unsigned Doubleword
    AMOMINU_D,

    // RV64F Single-Precision Float Extension
    /// Load Float Word — I-type: fd = mem[rs1+imm]
    FLW,
    /// Store Float Word — S-type: mem[rs1+imm] = fs2
    FSW,
    /// Float Add Single
    FADD_S,
    /// Float Subtract Single
    FSUB_S,
    /// Float Multiply Single
    FMUL_S,
    /// Float Divide Single
    FDIV_S,
    /// Float Square Root Single
    FSQRT_S,
    /// Float Minimum Single
    FMIN_S,
    /// Float Maximum Single
    FMAX_S,
    /// Convert Float to Signed 32-bit Int
    FCVT_W_S,
    /// Convert Float to Unsigned 32-bit Int
    FCVT_WU_S,
    /// Convert Float to Signed 64-bit Int
    FCVT_L_S,
    /// Convert Float to Unsigned 64-bit Int
    FCVT_LU_S,
    /// Convert Signed 32-bit Int to Float
    FCVT_S_W,
    /// Convert Unsigned 32-bit Int to Float
    FCVT_S_WU,
    /// Convert Signed 64-bit Int to Float
    FCVT_S_L,
    /// Convert Unsigned 64-bit Int to Float
    FCVT_S_LU,
    /// Move Float to Integer Register
    FMV_X_W,
    /// Move Integer to Float Register
    FMV_W_X,
    /// Float Equal Single
    FEQ_S,
    /// Float Less Than Single
    FLT_S,
    /// Float Less or Equal Single
    FLE_S,
    /// Fused Multiply-Add Single: rd = rs1×rs2 + rs3
    FMADD_S,
    /// Fused Multiply-Subtract Single: rd = rs1×rs2 − rs3
    FMSUB_S,
    /// Negated Fused Multiply-Subtract Single: rd = −(rs1×rs2 − rs3)
    FNMSUB_S,
    /// Negated Fused Multiply-Add Single: rd = −(rs1×rs2 + rs3)
    FNMADD_S,
    /// Float Sign-Inject Single (copy sign)
    FSGNJ_S,
    /// Float Sign-Inject Negate Single
    FSGNJN_S,
    /// Float Sign-Inject XOR Single (abs via self-xor)
    FSGNJX_S,
    /// Float Classify Single
    FCLASS_S,

    // RV64D Double-Precision Float Extension
    /// Load Float Double — I-type: fd = mem[rs1+imm]
    FLD,
    /// Store Float Double — S-type: mem[rs1+imm] = fs2
    FSD,
    /// Float Add Double
    FADD_D,
    /// Float Subtract Double
    FSUB_D,
    /// Float Multiply Double
    FMUL_D,
    /// Float Divide Double
    FDIV_D,
    /// Float Square Root Double
    FSQRT_D,
    /// Float Minimum Double
    FMIN_D,
    /// Float Maximum Double
    FMAX_D,
    /// Convert Double to Signed 32-bit Int
    FCVT_W_D,
    /// Convert Double to Unsigned 32-bit Int
    FCVT_WU_D,
    /// Convert Double to Signed 64-bit Int
    FCVT_L_D,
    /// Convert Double to Unsigned 64-bit Int
    FCVT_LU_D,
    /// Convert Signed 32-bit Int to Double
    FCVT_D_W,
    /// Convert Unsigned 32-bit Int to Double
    FCVT_D_WU,
    /// Convert Signed 64-bit Int to Double
    FCVT_D_L,
    /// Convert Unsigned 64-bit Int to Double
    FCVT_D_LU,
    /// Convert Single to Double
    FCVT_S_D,
    /// Convert Double to Single
    FCVT_D_S,
    /// Move Double to Integer Register (64-bit)
    FMV_X_D,
    /// Move Integer (64-bit) to Double Register
    FMV_D_X,
    /// Float Equal Double
    FEQ_D,
    /// Float Less Than Double
    FLT_D,
    /// Float Less or Equal Double
    FLE_D,
    /// Fused Multiply-Add Double
    FMADD_D,
    /// Fused Multiply-Subtract Double
    FMSUB_D,
    /// Negated Fused Multiply-Subtract Double
    FNMSUB_D,
    /// Negated Fused Multiply-Add Double
    FNMADD_D,
    /// Float Sign-Inject Double
    FSGNJ_D,
    /// Float Sign-Inject Negate Double
    FSGNJN_D,
    /// Float Sign-Inject XOR Double
    FSGNJX_D,
    /// Float Classify Double
    FCLASS_D,

    // Pseudo-instructions (expanded by assembler)
    /// No operation: ADDI x0, x0, 0
    NOP,
    /// Move register: ADDI rd, rs, 0
    MV,
    /// Load immediate (multi-instruction LUI+ADDI sequence)
    LI,
    /// Load address (AUIPC+ADDI sequence for PC-relative)
    LA,
    /// Function call (AUIPC+JALR or JAL for near calls)
    CALL,
    /// Return from function: JALR x0, ra, 0
    RET,
    /// Negate: SUB rd, x0, rs
    NEG,
    /// Bitwise NOT: XORI rd, rs, -1
    NOT,
    /// Set if Equal to Zero: SLTIU rd, rs, 1
    SEQZ,
    /// Set if Not Equal to Zero: SLTU rd, x0, rs
    SNEZ,
    /// Unconditional jump: JAL x0, offset
    J,
    /// Inline assembly passthrough marker
    INLINE_ASM,
}

// ===========================================================================
// RvInstruction — RISC-V 64 machine instruction
// ===========================================================================

/// A RISC-V 64 machine instruction produced by instruction selection.
///
/// Fields `rd`, `rs1`, `rs2`, `rs3` hold register IDs from the register
/// namespace defined in `registers.rs` (0–31 for GPRs, 32–63 for FPRs).
/// The `symbol` field carries relocation references resolved by the linker.
#[derive(Debug, Clone)]
pub struct RvInstruction {
    /// The instruction opcode identifying the operation.
    pub opcode: RvOpcode,
    /// Destination register, or `None` for instructions without a result.
    pub rd: Option<u8>,
    /// First source register.
    pub rs1: Option<u8>,
    /// Second source register.
    pub rs2: Option<u8>,
    /// Third source register (fused multiply-add only).
    pub rs3: Option<u8>,
    /// Immediate value (12-bit signed for I-type, 20-bit for U-type, etc.).
    pub imm: i64,
    /// Symbol reference for linker relocations (function/global names).
    pub symbol: Option<String>,
    /// Whether this instruction operates on floating-point registers.
    pub is_fp: bool,
    /// Optional debug annotation for assembly listing.
    pub comment: Option<String>,
}

impl RvInstruction {
    /// Create a new R-type instruction (register-register).
    fn r_type(opcode: RvOpcode, rd: u8, rs1: u8, rs2: u8) -> Self {
        RvInstruction {
            opcode,
            rd: Some(rd),
            rs1: Some(rs1),
            rs2: Some(rs2),
            rs3: None,
            imm: 0,
            symbol: None,
            is_fp: false,
            comment: None,
        }
    }

    /// Create a new I-type instruction (register-immediate).
    fn i_type(opcode: RvOpcode, rd: u8, rs1: u8, imm: i64) -> Self {
        RvInstruction {
            opcode,
            rd: Some(rd),
            rs1: Some(rs1),
            rs2: None,
            rs3: None,
            imm,
            symbol: None,
            is_fp: false,
            comment: None,
        }
    }

    /// Create a new S-type instruction (store).
    fn s_type(opcode: RvOpcode, rs1: u8, rs2: u8, imm: i64) -> Self {
        RvInstruction {
            opcode,
            rd: None,
            rs1: Some(rs1),
            rs2: Some(rs2),
            rs3: None,
            imm,
            symbol: None,
            is_fp: false,
            comment: None,
        }
    }

    /// Create a new B-type instruction (conditional branch).
    fn b_type(opcode: RvOpcode, rs1: u8, rs2: u8, imm: i64) -> Self {
        RvInstruction {
            opcode,
            rd: None,
            rs1: Some(rs1),
            rs2: Some(rs2),
            rs3: None,
            imm,
            symbol: None,
            is_fp: false,
            comment: None,
        }
    }

    /// Create a new U-type instruction (upper immediate).
    fn u_type(opcode: RvOpcode, rd: u8, imm: i64) -> Self {
        RvInstruction {
            opcode,
            rd: Some(rd),
            rs1: None,
            rs2: None,
            rs3: None,
            imm,
            symbol: None,
            is_fp: false,
            comment: None,
        }
    }

    /// Create a no-operand instruction (e.g., NOP, RET).
    fn no_op(opcode: RvOpcode) -> Self {
        RvInstruction {
            opcode,
            rd: None,
            rs1: None,
            rs2: None,
            rs3: None,
            imm: 0,
            symbol: None,
            is_fp: false,
            comment: None,
        }
    }

    /// Set the floating-point flag on this instruction.
    fn with_fp(mut self) -> Self {
        self.is_fp = true;
        self
    }

    /// Set a symbol reference on this instruction.
    #[allow(dead_code)]
    fn with_symbol(mut self, sym: String) -> Self {
        self.symbol = Some(sym);
        self
    }

    /// Set a debug comment on this instruction.
    #[allow(dead_code)]
    fn with_comment(mut self, comment: impl Into<String>) -> Self {
        self.comment = Some(comment.into());
        self
    }
}

// ===========================================================================
// Helper: 12-bit signed immediate range check
// ===========================================================================

/// Check whether an immediate value fits in a 12-bit signed field (I-type).
#[inline]
fn fits_i12(val: i64) -> bool {
    (-2048..=2047).contains(&val)
}

/// Check whether an immediate fits in a 32-bit signed value (LUI+ADDI).
#[inline]
fn fits_i32(val: i64) -> bool {
    (i32::MIN as i64..=i32::MAX as i64).contains(&val)
}

/// Decompose a 32-bit value into LUI upper 20 bits and ADDI lower 12 bits,
/// handling the sign-extension gotcha: if bit 11 of the lower portion is set,
/// the ADDI will sign-extend and subtract 0x1000, so we compensate by adding
/// 1 to the upper portion.
fn split_i32_lui_addi(val: i64) -> (i64, i64) {
    let val32 = val as i32;
    let lo12 = ((val32 as i64) << 52 >> 52) as i32; // sign-extended lower 12
    let hi20 = ((val32 as u32).wrapping_sub(lo12 as u32) >> 12) as i32;
    let hi20_masked = hi20 & 0xFFFFF_i32;
    (hi20_masked as i64, lo12 as i64)
}

// ===========================================================================
// Virtual register tracking
// ===========================================================================

/// Maps IR Value indices to physical/virtual register IDs or stack offsets.
/// During instruction selection, IR values are mapped to virtual registers
/// that are later resolved by the register allocator.
struct ValueMap {
    /// Maps IR Value index → register ID (physical or virtual).
    regs: std::collections::HashMap<u32, u8>,
    /// Maps IR Value index → stack frame offset for spilled/alloca'd values.
    stack_offsets: std::collections::HashMap<u32, i64>,
    /// Next virtual register number for allocation.
    next_vreg: u32,
}

impl ValueMap {
    fn new() -> Self {
        ValueMap {
            regs: std::collections::HashMap::new(),
            stack_offsets: std::collections::HashMap::new(),
            next_vreg: 64, // above physical register space (0-63)
        }
    }

    /// Allocate a GPR for an IR value from the allocatable pool.
    fn alloc_gpr(&mut self, val: Value) -> u8 {
        let idx = val.index();
        if let Some(&reg) = self.regs.get(&idx) {
            return reg;
        }
        let pool = &ALLOCATABLE_GPRS;
        let reg = pool[(self.next_vreg as usize) % pool.len()];
        self.next_vreg += 1;
        self.regs.insert(idx, reg);
        reg
    }

    /// Allocate an FPR for an IR value from the allocatable pool.
    fn alloc_fpr(&mut self, val: Value) -> u8 {
        let idx = val.index();
        if let Some(&reg) = self.regs.get(&idx) {
            return reg;
        }
        let pool = &ALLOCATABLE_FPRS;
        let reg = pool[(self.next_vreg as usize) % pool.len()];
        self.next_vreg += 1;
        self.regs.insert(idx, reg);
        reg
    }

    /// Get the register for an IR value, defaulting to T0.
    fn get_reg(&self, val: Value) -> u8 {
        *self.regs.get(&val.index()).unwrap_or(&T0)
    }

    /// Get the stack offset for an IR value (alloca'd value).
    #[allow(dead_code)]
    fn get_stack_offset(&self, val: Value) -> Option<i64> {
        self.stack_offsets.get(&val.index()).copied()
    }

    /// Record a stack offset for an IR value.
    fn set_stack_offset(&mut self, val: Value, offset: i64) {
        self.stack_offsets.insert(val.index(), offset);
    }

    /// Record a register mapping for an IR value.
    fn set_reg(&mut self, val: Value, reg: u8) {
        self.regs.insert(val.index(), reg);
    }
}

// ===========================================================================
// RiscV64InstructionSelector — Main instruction selection engine
// ===========================================================================

/// RISC-V 64 instruction selector.
///
/// Translates phi-eliminated IR functions into sequences of `RvInstruction`s
/// that conform to the RV64IMAFDC ISA. Handles:
/// - Integer and floating-point arithmetic
/// - Memory loads and stores (byte through doubleword)
/// - Comparisons and branches (with ±4 KiB offset awareness)
/// - Large immediate materialization (LUI+ADDI, multi-instruction 64-bit)
/// - Function call lowering per LP64D ABI
/// - Prologue/epilogue generation with callee-saved register preservation
/// - Inline assembly passthrough
/// - PIC (position-independent code) addressing via GOT/PLT
/// - Switch statements via cascaded comparisons or jump tables
pub struct RiscV64InstructionSelector {
    /// Accumulated machine instructions for the current function.
    pub instructions: Vec<RvInstruction>,
    /// Current basic block label.
    current_block: Option<String>,
    /// Computed frame size in bytes (aligned to 16).
    pub frame_size: i64,
    /// Spill slots: (virtual_reg_id, stack_offset) pairs.
    spill_slots: Vec<(u32, i64)>,
    /// Whether PIC mode is enabled (`-fPIC`).
    pub pic_mode: bool,
    /// Target architecture reference.
    target: Target,
    /// Value-to-register and value-to-stack mapping.
    vmap: ValueMap,
    /// Current stack allocation offset (grows downward from FP).
    alloca_offset: i64,
    /// Callee-saved GPRs actually used in this function.
    used_callee_saved_gprs: Vec<u8>,
    /// Callee-saved FPRs actually used in this function.
    used_callee_saved_fprs: Vec<u8>,
    /// Block label map: IR BlockId index → string label.
    block_labels: std::collections::HashMap<usize, String>,
    /// Whether the current function makes any calls.
    has_calls: bool,
}

impl RiscV64InstructionSelector {
    /// Create a new RISC-V 64 instruction selector.
    ///
    /// # Arguments
    /// - `target` — Must be `Target::RiscV64`.
    /// - `pic_mode` — Whether to generate position-independent code.
    pub fn new(target: Target, pic_mode: bool) -> Self {
        RiscV64InstructionSelector {
            instructions: Vec::new(),
            current_block: None,
            frame_size: 0,
            spill_slots: Vec::new(),
            pic_mode,
            target,
            vmap: ValueMap::new(),
            alloca_offset: 0,
            used_callee_saved_gprs: Vec::new(),
            used_callee_saved_fprs: Vec::new(),
            block_labels: std::collections::HashMap::new(),
            has_calls: false,
        }
    }

    /// Reset selector state for a new function.
    fn reset(&mut self) {
        self.instructions.clear();
        self.current_block = None;
        self.frame_size = 0;
        self.spill_slots.clear();
        self.vmap = ValueMap::new();
        self.alloca_offset = 0;
        self.used_callee_saved_gprs.clear();
        self.used_callee_saved_fprs.clear();
        self.block_labels.clear();
        self.has_calls = false;
    }

    // -----------------------------------------------------------------------
    // Block label management
    // -----------------------------------------------------------------------

    /// Get or create the label string for a block index.
    fn block_label(&mut self, block_idx: usize) -> String {
        if let Some(lbl) = self.block_labels.get(&block_idx) {
            lbl.clone()
        } else {
            let lbl = format!(".LBB_{}", block_idx);
            self.block_labels.insert(block_idx, lbl.clone());
            lbl
        }
    }

    /// Emit a label pseudo-instruction for a basic block.
    fn emit_block_label(&mut self, label: &str) {
        self.current_block = Some(label.to_string());
    }

    // -----------------------------------------------------------------------
    // Register selection helpers
    // -----------------------------------------------------------------------

    /// Determine whether an IR type should use floating-point registers.
    fn uses_fpr(ty: &IrType) -> bool {
        matches!(ty, IrType::F32 | IrType::F64)
    }

    /// Get a destination register for an IR value given its type.
    fn dest_reg(&mut self, val: Value, ty: &IrType) -> u8 {
        if Self::uses_fpr(ty) {
            self.vmap.alloc_fpr(val)
        } else {
            self.vmap.alloc_gpr(val)
        }
    }

    /// Get the source register for an IR value.
    fn src_reg(&self, val: Value) -> u8 {
        self.vmap.get_reg(val)
    }

    /// Track usage of a callee-saved register.
    fn mark_callee_saved(&mut self, reg: u8) {
        if is_callee_saved(reg) {
            if is_gpr(reg) && !self.used_callee_saved_gprs.contains(&reg) {
                self.used_callee_saved_gprs.push(reg);
            } else if is_fpr(reg) && !self.used_callee_saved_fprs.contains(&reg) {
                self.used_callee_saved_fprs.push(reg);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Emit helpers — push instructions to the output
    // -----------------------------------------------------------------------

    fn emit(&mut self, inst: RvInstruction) {
        if let Some(rd) = inst.rd {
            self.mark_callee_saved(rd);
        }
        self.instructions.push(inst);
    }

    fn emit_r(&mut self, op: RvOpcode, rd: u8, rs1: u8, rs2: u8) {
        self.emit(RvInstruction::r_type(op, rd, rs1, rs2));
    }

    fn emit_i(&mut self, op: RvOpcode, rd: u8, rs1: u8, imm: i64) {
        self.emit(RvInstruction::i_type(op, rd, rs1, imm));
    }

    fn emit_s(&mut self, op: RvOpcode, base: u8, src: u8, offset: i64) {
        self.emit(RvInstruction::s_type(op, base, src, offset));
    }

    fn emit_b(&mut self, op: RvOpcode, rs1: u8, rs2: u8, target_label: &str) {
        let mut inst = RvInstruction::b_type(op, rs1, rs2, 0);
        inst.symbol = Some(target_label.to_string());
        self.emit(inst);
    }

    #[allow(dead_code)]
    fn emit_u(&mut self, op: RvOpcode, rd: u8, imm: i64) {
        self.emit(RvInstruction::u_type(op, rd, imm));
    }

    fn emit_fp_r(&mut self, op: RvOpcode, rd: u8, rs1: u8, rs2: u8) {
        self.emit(RvInstruction::r_type(op, rd, rs1, rs2).with_fp());
    }

    fn emit_fp_unary(&mut self, op: RvOpcode, rd: u8, rs1: u8) {
        let mut inst = RvInstruction::i_type(op, rd, rs1, 0);
        inst.is_fp = true;
        inst.rs2 = None;
        self.emit(inst);
    }

    // ===================================================================
    // Immediate materialization
    // ===================================================================

    /// Materialize a constant integer into a register.
    ///
    /// Handles the full range of 64-bit values:
    /// - 12-bit signed: single ADDI
    /// - 32-bit: LUI + ADDI (with sign-extension correction)
    /// - 64-bit: multi-instruction sequence (LUI+ADDI+SLLI+ADDI chains)
    pub fn materialize_immediate(&mut self, rd: u8, value: i64) -> Vec<RvInstruction> {
        let mut result = Vec::new();

        if value == 0 {
            result.push(RvInstruction::i_type(RvOpcode::ADDI, rd, ZERO, 0));
            return result;
        }

        if fits_i12(value) {
            result.push(RvInstruction::i_type(RvOpcode::ADDI, rd, ZERO, value));
            return result;
        }

        if fits_i32(value) {
            let (hi, lo) = split_i32_lui_addi(value);
            if hi != 0 {
                result.push(RvInstruction::u_type(RvOpcode::LUI, rd, hi));
            }
            if lo != 0 {
                let base = if hi != 0 { rd } else { ZERO };
                result.push(RvInstruction::i_type(RvOpcode::ADDI, rd, base, lo));
            }
            return result;
        }

        // 64-bit value: decompose into shift-add sequences.
        Self::materialize_i64_into(&mut result, rd, value);
        result
    }

    /// Internal helper for materializing 64-bit immediates.
    fn materialize_i64_into(result: &mut Vec<RvInstruction>, rd: u8, value: i64) {
        let upper32 = (value >> 32) as i32 as i64;
        let lower32_u = (value as u64) & 0xFFFF_FFFF;

        // Load upper 32 bits as a 32-bit constant
        if fits_i12(upper32) {
            result.push(RvInstruction::i_type(RvOpcode::ADDI, rd, ZERO, upper32));
        } else {
            let (hi, lo) = split_i32_lui_addi(upper32);
            if hi != 0 {
                result.push(RvInstruction::u_type(RvOpcode::LUI, rd, hi));
            }
            if lo != 0 {
                let base = if hi != 0 { rd } else { ZERO };
                result.push(RvInstruction::i_type(RvOpcode::ADDI, rd, base, lo));
            } else if hi == 0 {
                result.push(RvInstruction::i_type(RvOpcode::ADDI, rd, ZERO, 0));
            }
        }

        // Shift left by 32
        result.push(RvInstruction::i_type(RvOpcode::SLLI, rd, rd, 32));

        // Add lower 32 bits if non-zero
        if lower32_u != 0 {
            // Decompose lower 32 bits: upper 20 + lower 12
            let lo_raw = lower32_u as i64;
            let lo_lo = (lo_raw << 52) >> 52; // sign-extend low 12
            let lo_hi = ((lower32_u.wrapping_sub(lo_lo as u64)) >> 12) as i64 & 0xFFFFF;

            if lo_hi != 0 {
                // Use T0 as temporary to build lower 32 bits
                result.push(RvInstruction::u_type(RvOpcode::LUI, T0, lo_hi));
                if lo_lo != 0 {
                    result.push(RvInstruction::i_type(RvOpcode::ADDI, T0, T0, lo_lo));
                }
                result.push(RvInstruction::r_type(RvOpcode::ADD, rd, rd, T0));
            } else if lo_lo != 0 {
                result.push(RvInstruction::i_type(RvOpcode::ADDI, rd, rd, lo_lo));
            }
        }
    }

    /// Emit immediate materialization inline (adds to self.instructions).
    fn emit_immediate(&mut self, rd: u8, value: i64) {
        let insts = self.materialize_immediate(rd, value);
        for inst in insts {
            self.instructions.push(inst);
        }
    }

    // ===================================================================
    // Address materialization
    // ===================================================================

    /// Materialize a symbol address into a register.
    ///
    /// - **Non-PIC**: `LUI rd, %hi(sym)` + `ADDI rd, rd, %lo(sym)`
    /// - **PIC (GOT)**: `AUIPC rd, %got_pcrel_hi(sym)` + `LD rd, rd, %pcrel_lo(.)`
    pub fn materialize_address(&mut self, rd: u8, symbol: &str) -> Vec<RvInstruction> {
        let mut result = Vec::new();

        if self.pic_mode {
            result.push(RvInstruction {
                opcode: RvOpcode::AUIPC,
                rd: Some(rd),
                rs1: None,
                rs2: None,
                rs3: None,
                imm: 0,
                symbol: Some(format!("%got_pcrel_hi({})", symbol)),
                is_fp: false,
                comment: Some(format!("GOT hi: {}", symbol)),
            });
            result.push(RvInstruction {
                opcode: RvOpcode::LD,
                rd: Some(rd),
                rs1: Some(rd),
                rs2: None,
                rs3: None,
                imm: 0,
                symbol: Some(format!("%pcrel_lo(.L_got_{})", symbol)),
                is_fp: false,
                comment: Some(format!("GOT ld: {}", symbol)),
            });
        } else {
            result.push(RvInstruction {
                opcode: RvOpcode::LUI,
                rd: Some(rd),
                rs1: None,
                rs2: None,
                rs3: None,
                imm: 0,
                symbol: Some(format!("%hi({})", symbol)),
                is_fp: false,
                comment: Some(format!("addr hi: {}", symbol)),
            });
            result.push(RvInstruction {
                opcode: RvOpcode::ADDI,
                rd: Some(rd),
                rs1: Some(rd),
                rs2: None,
                rs3: None,
                imm: 0,
                symbol: Some(format!("%lo({})", symbol)),
                is_fp: false,
                comment: Some(format!("addr lo: {}", symbol)),
            });
        }

        result
    }

    /// Emit address materialization inline.
    fn emit_address(&mut self, rd: u8, symbol: &str) {
        let insts = self.materialize_address(rd, symbol);
        for inst in insts {
            self.instructions.push(inst);
        }
    }

    /// Materialize a PC-relative address (for local PIC symbols).
    fn emit_pcrel_address(&mut self, rd: u8, symbol: &str) {
        self.emit(RvInstruction {
            opcode: RvOpcode::AUIPC,
            rd: Some(rd),
            rs1: None,
            rs2: None,
            rs3: None,
            imm: 0,
            symbol: Some(format!("%pcrel_hi({})", symbol)),
            is_fp: false,
            comment: None,
        });
        self.emit(RvInstruction {
            opcode: RvOpcode::ADDI,
            rd: Some(rd),
            rs1: Some(rd),
            rs2: None,
            rs3: None,
            imm: 0,
            symbol: Some(format!("%pcrel_lo(.L_pcrel_{})", symbol)),
            is_fp: false,
            comment: None,
        });
    }

    // ===================================================================
    // Load instruction selection
    // ===================================================================

    /// Emit a load instruction from memory to a register.
    ///
    /// Selects the appropriate load opcode based on the type size and
    /// sign-extension requirements:
    /// - I1/I8 signed → LB, unsigned → LBU
    /// - I16 signed → LH, unsigned → LHU
    /// - I32 signed → LW, unsigned → LWU
    /// - I64/Ptr → LD
    /// - F32 → FLW
    /// - F64 → FLD
    pub fn emit_load(&mut self, rd: u8, base: u8, offset: i64, ty: &IrType) {
        if !fits_i12(offset) {
            // Large offset: materialize into T0, then ADD to base
            self.emit_immediate(T0, offset);
            self.emit_r(RvOpcode::ADD, T0, base, T0);
            self.emit_load(rd, T0, 0, ty);
            return;
        }

        let opcode = match ty {
            IrType::I1 | IrType::I8 => RvOpcode::LBU,
            IrType::I16 => RvOpcode::LH,
            IrType::I32 => RvOpcode::LW,
            IrType::I64 | IrType::I128 | IrType::Ptr => RvOpcode::LD,
            IrType::F32 => RvOpcode::FLW,
            IrType::F64 | IrType::F80 => RvOpcode::FLD,
            IrType::Void => return,
            IrType::Array(_, _) | IrType::Struct(_) | IrType::Function(_, _) => {
                // Aggregate types: load as pointer (address)
                RvOpcode::LD
            }
        };

        let is_fp = matches!(ty, IrType::F32 | IrType::F64 | IrType::F80);
        let mut inst = RvInstruction::i_type(opcode, rd, base, offset);
        inst.is_fp = is_fp;
        self.emit(inst);
    }

    // ===================================================================
    // Store instruction selection
    // ===================================================================

    /// Emit a store instruction from a register to memory.
    ///
    /// Selects the appropriate store opcode based on the type size:
    /// - I1/I8 → SB
    /// - I16 → SH
    /// - I32 → SW
    /// - I64/Ptr → SD
    /// - F32 → FSW
    /// - F64 → FSD
    pub fn emit_store(&mut self, src: u8, base: u8, offset: i64, ty: &IrType) {
        if !fits_i12(offset) {
            self.emit_immediate(T0, offset);
            self.emit_r(RvOpcode::ADD, T0, base, T0);
            self.emit_store(src, T0, 0, ty);
            return;
        }

        let opcode = match ty {
            IrType::I1 | IrType::I8 => RvOpcode::SB,
            IrType::I16 => RvOpcode::SH,
            IrType::I32 => RvOpcode::SW,
            IrType::I64 | IrType::I128 | IrType::Ptr => RvOpcode::SD,
            IrType::F32 => RvOpcode::FSW,
            IrType::F64 | IrType::F80 => RvOpcode::FSD,
            IrType::Void => return,
            IrType::Array(_, _) | IrType::Struct(_) | IrType::Function(_, _) => RvOpcode::SD,
        };

        let is_fp = matches!(ty, IrType::F32 | IrType::F64 | IrType::F80);
        let mut inst = RvInstruction::s_type(opcode, base, src, offset);
        inst.is_fp = is_fp;
        self.emit(inst);
    }

    // ===================================================================
    // Comparison instruction selection
    // ===================================================================

    /// Emit an integer comparison, materializing the boolean result into `rd`.
    ///
    /// For conditional branches, callers should use `emit_branch` instead,
    /// which generates direct branch instructions. This method produces a
    /// 0/1 integer result for use as a value.
    pub fn emit_comparison(&mut self, rd: u8, rs1: u8, rs2: u8, op: &ICmpOp) {
        match op {
            ICmpOp::Eq => {
                // rd = (rs1 == rs2) → SUB tmp, rs1, rs2; SEQZ rd, tmp
                self.emit_r(RvOpcode::SUB, rd, rs1, rs2);
                self.emit_i(RvOpcode::SLTIU, rd, rd, 1); // SEQZ
            }
            ICmpOp::Ne => {
                // rd = (rs1 != rs2) → SUB tmp, rs1, rs2; SNEZ rd, tmp
                self.emit_r(RvOpcode::SUB, rd, rs1, rs2);
                self.emit_r(RvOpcode::SLTU, rd, ZERO, rd); // SNEZ
            }
            ICmpOp::Slt => {
                self.emit_r(RvOpcode::SLT, rd, rs1, rs2);
            }
            ICmpOp::Sle => {
                // rd = (rs1 <= rs2) → rd = !(rs2 < rs1) → SLT tmp, rs2, rs1; XORI rd, tmp, 1
                self.emit_r(RvOpcode::SLT, rd, rs2, rs1);
                self.emit_i(RvOpcode::XORI, rd, rd, 1);
            }
            ICmpOp::Sgt => {
                // rd = (rs1 > rs2) → SLT rd, rs2, rs1
                self.emit_r(RvOpcode::SLT, rd, rs2, rs1);
            }
            ICmpOp::Sge => {
                // rd = (rs1 >= rs2) → !(rs1 < rs2) → SLT tmp, rs1, rs2; XORI rd, tmp, 1
                self.emit_r(RvOpcode::SLT, rd, rs1, rs2);
                self.emit_i(RvOpcode::XORI, rd, rd, 1);
            }
            ICmpOp::Ult => {
                self.emit_r(RvOpcode::SLTU, rd, rs1, rs2);
            }
            ICmpOp::Ule => {
                self.emit_r(RvOpcode::SLTU, rd, rs2, rs1);
                self.emit_i(RvOpcode::XORI, rd, rd, 1);
            }
            ICmpOp::Ugt => {
                self.emit_r(RvOpcode::SLTU, rd, rs2, rs1);
            }
            ICmpOp::Uge => {
                self.emit_r(RvOpcode::SLTU, rd, rs1, rs2);
                self.emit_i(RvOpcode::XORI, rd, rd, 1);
            }
        }
    }

    /// Emit a floating-point comparison, putting 0/1 into integer `rd`.
    fn emit_fcmp(&mut self, rd: u8, rs1: u8, rs2: u8, op: &FCmpOp, is_double: bool) {
        match (op, is_double) {
            (FCmpOp::Oeq, false) => self.emit_fp_r(RvOpcode::FEQ_S, rd, rs1, rs2),
            (FCmpOp::Oeq, true) => self.emit_fp_r(RvOpcode::FEQ_D, rd, rs1, rs2),
            (FCmpOp::Olt, false) => self.emit_fp_r(RvOpcode::FLT_S, rd, rs1, rs2),
            (FCmpOp::Olt, true) => self.emit_fp_r(RvOpcode::FLT_D, rd, rs1, rs2),
            (FCmpOp::Ole, false) => self.emit_fp_r(RvOpcode::FLE_S, rd, rs1, rs2),
            (FCmpOp::Ole, true) => self.emit_fp_r(RvOpcode::FLE_D, rd, rs1, rs2),
            (FCmpOp::Ogt, false) => self.emit_fp_r(RvOpcode::FLT_S, rd, rs2, rs1),
            (FCmpOp::Ogt, true) => self.emit_fp_r(RvOpcode::FLT_D, rd, rs2, rs1),
            (FCmpOp::Oge, false) => self.emit_fp_r(RvOpcode::FLE_S, rd, rs2, rs1),
            (FCmpOp::Oge, true) => self.emit_fp_r(RvOpcode::FLE_D, rd, rs2, rs1),
            (FCmpOp::One, false) => {
                // !(a == b) — FEQ then invert
                self.emit_fp_r(RvOpcode::FEQ_S, rd, rs1, rs2);
                self.emit_i(RvOpcode::XORI, rd, rd, 1);
            }
            (FCmpOp::One, true) => {
                self.emit_fp_r(RvOpcode::FEQ_D, rd, rs1, rs2);
                self.emit_i(RvOpcode::XORI, rd, rd, 1);
            }
            (FCmpOp::Uno, false) => {
                // Unordered: !(a == a) || !(b == b)
                self.emit_fp_r(RvOpcode::FEQ_S, rd, rs1, rs1);
                self.emit_fp_r(RvOpcode::FEQ_S, T0, rs2, rs2);
                self.emit_r(RvOpcode::AND, rd, rd, T0);
                self.emit_i(RvOpcode::XORI, rd, rd, 1);
            }
            (FCmpOp::Uno, true) => {
                self.emit_fp_r(RvOpcode::FEQ_D, rd, rs1, rs1);
                self.emit_fp_r(RvOpcode::FEQ_D, T0, rs2, rs2);
                self.emit_r(RvOpcode::AND, rd, rd, T0);
                self.emit_i(RvOpcode::XORI, rd, rd, 1);
            }
            (FCmpOp::Ord, false) => {
                // Ordered: (a == a) && (b == b)
                self.emit_fp_r(RvOpcode::FEQ_S, rd, rs1, rs1);
                self.emit_fp_r(RvOpcode::FEQ_S, T0, rs2, rs2);
                self.emit_r(RvOpcode::AND, rd, rd, T0);
            }
            (FCmpOp::Ord, true) => {
                self.emit_fp_r(RvOpcode::FEQ_D, rd, rs1, rs1);
                self.emit_fp_r(RvOpcode::FEQ_D, T0, rs2, rs2);
                self.emit_r(RvOpcode::AND, rd, rd, T0);
            }
        }
    }

    // ===================================================================
    // Branch instruction selection
    // ===================================================================

    /// Emit a conditional or unconditional branch.
    ///
    /// Conditional branches use B-type instructions with ±4 KiB range.
    /// For far branches (detected post-layout), trampolines may be needed.
    pub fn emit_branch(&mut self, cond: Option<(u8, u8, &ICmpOp)>, target_label: &str) {
        match cond {
            None => {
                // Unconditional jump: J target (pseudo for JAL x0, offset)
                let mut inst = RvInstruction::no_op(RvOpcode::J);
                inst.symbol = Some(target_label.to_string());
                self.emit(inst);
            }
            Some((rs1, rs2, op)) => {
                let branch_op = match op {
                    ICmpOp::Eq => RvOpcode::BEQ,
                    ICmpOp::Ne => RvOpcode::BNE,
                    ICmpOp::Slt => RvOpcode::BLT,
                    ICmpOp::Sge => RvOpcode::BGE,
                    ICmpOp::Ult => RvOpcode::BLTU,
                    ICmpOp::Uge => RvOpcode::BGEU,
                    // For other predicates, invert or swap operands
                    ICmpOp::Sgt => {
                        self.emit_b(RvOpcode::BLT, rs2, rs1, target_label);
                        return;
                    }
                    ICmpOp::Sle => {
                        self.emit_b(RvOpcode::BGE, rs2, rs1, target_label);
                        return;
                    }
                    ICmpOp::Ugt => {
                        self.emit_b(RvOpcode::BLTU, rs2, rs1, target_label);
                        return;
                    }
                    ICmpOp::Ule => {
                        self.emit_b(RvOpcode::BGEU, rs2, rs1, target_label);
                        return;
                    }
                };
                self.emit_b(branch_op, rs1, rs2, target_label);
            }
        }
    }

    // ===================================================================
    // Type conversion instruction selection
    // ===================================================================

    /// Emit type conversion (cast) instructions.
    ///
    /// Handles integer width conversions, integer↔float conversions,
    /// and pointer↔integer conversions.
    pub fn emit_conversion(&mut self, rd: u8, rs: u8, from_ty: &IrType, to_ty: &IrType) {
        match (from_ty, to_ty) {
            // Integer truncation: just use the lower bits via ANDI/shift
            (_, IrType::I1) => {
                self.emit_i(RvOpcode::ANDI, rd, rs, 1);
            }
            (_, IrType::I8) if !matches!(from_ty, IrType::I1) => {
                self.emit_i(RvOpcode::ANDI, rd, rs, 0xFF);
            }
            (_, IrType::I16) if !matches!(from_ty, IrType::I1 | IrType::I8) => {
                // Zero-extend 16-bit: shift left 48, then logical shift right 48
                self.emit_i(RvOpcode::SLLI, rd, rs, 48);
                self.emit_i(RvOpcode::SRLI, rd, rd, 48);
            }
            (_, IrType::I32) if matches!(from_ty, IrType::I64 | IrType::Ptr | IrType::I128) => {
                // Truncate to 32 bits: ADDIW sign-extends 32→64
                self.emit_i(RvOpcode::ADDIW, rd, rs, 0);
            }

            // Zero extension
            (IrType::I1, _) if to_ty.is_integer() || to_ty.is_pointer() => {
                self.emit_i(RvOpcode::ANDI, rd, rs, 1);
            }
            (IrType::I8, _) if to_ty.is_integer() || to_ty.is_pointer() => {
                self.emit_i(RvOpcode::ANDI, rd, rs, 0xFF);
            }
            (IrType::I16, _) if to_ty.is_integer() || to_ty.is_pointer() => {
                self.emit_i(RvOpcode::SLLI, rd, rs, 48);
                self.emit_i(RvOpcode::SRLI, rd, rd, 48);
            }
            (IrType::I32, IrType::I64) | (IrType::I32, IrType::Ptr) => {
                // Zero-extend word: shift left 32, logical shift right 32
                self.emit_i(RvOpcode::SLLI, rd, rs, 32);
                self.emit_i(RvOpcode::SRLI, rd, rd, 32);
            }

            // Sign extension
            (IrType::I8, IrType::I32) => {
                self.emit_i(RvOpcode::SLLI, rd, rs, 56);
                self.emit_i(RvOpcode::SRAI, rd, rd, 56);
            }
            (IrType::I16, IrType::I32) => {
                self.emit_i(RvOpcode::SLLI, rd, rs, 48);
                self.emit_i(RvOpcode::SRAI, rd, rd, 48);
            }

            // Integer ↔ Float conversions
            (IrType::I32, IrType::F32) => {
                self.emit_fp_unary(RvOpcode::FCVT_S_W, rd, rs);
            }
            (IrType::I32, IrType::F64) => {
                self.emit_fp_unary(RvOpcode::FCVT_D_W, rd, rs);
            }
            (IrType::I64, IrType::F32) => {
                self.emit_fp_unary(RvOpcode::FCVT_S_L, rd, rs);
            }
            (IrType::I64, IrType::F64) => {
                self.emit_fp_unary(RvOpcode::FCVT_D_L, rd, rs);
            }
            (IrType::F32, IrType::I32) => {
                self.emit_fp_unary(RvOpcode::FCVT_W_S, rd, rs);
            }
            (IrType::F32, IrType::I64) => {
                self.emit_fp_unary(RvOpcode::FCVT_L_S, rd, rs);
            }
            (IrType::F64, IrType::I32) => {
                self.emit_fp_unary(RvOpcode::FCVT_W_D, rd, rs);
            }
            (IrType::F64, IrType::I64) => {
                self.emit_fp_unary(RvOpcode::FCVT_L_D, rd, rs);
            }
            // Float ↔ Float conversions
            (IrType::F32, IrType::F64) => {
                self.emit_fp_unary(RvOpcode::FCVT_D_S, rd, rs);
            }
            (IrType::F64, IrType::F32) => {
                self.emit_fp_unary(RvOpcode::FCVT_S_D, rd, rs);
            }
            // Pointer ↔ Integer: no-op on RV64 (both 64 bits)
            (IrType::Ptr, _) if to_ty.is_integer() => {
                if rd != rs {
                    self.emit_i(RvOpcode::ADDI, rd, rs, 0); // MV
                }
            }
            (_, IrType::Ptr) if from_ty.is_integer() => {
                if rd != rs {
                    self.emit_i(RvOpcode::ADDI, rd, rs, 0); // MV
                }
            }
            // Default: MV (copy register)
            _ => {
                if rd != rs {
                    self.emit_i(RvOpcode::ADDI, rd, rs, 0);
                }
            }
        }
    }

    // ===================================================================
    // Function call lowering
    // ===================================================================

    /// Emit a function call instruction sequence.
    ///
    /// Follows the LP64D ABI:
    /// - Integer args in a0–a7, FP args in fa0–fa7
    /// - Stack-passed args aligned to 8 bytes
    /// - Return value in a0 (integer) or fa0 (FP)
    /// - Caller saves: t0–t6, a0–a7, ft0–ft11, fa0–fa7
    pub fn emit_call(
        &mut self,
        callee: &str,
        args: &[(u8, bool)], // (register, is_fp)
        result_reg: Option<u8>,
        result_is_fp: bool,
    ) {
        self.has_calls = true;

        // Move arguments into ABI registers using separate counters for
        // integer and floating-point register banks per LP64D calling convention.
        // Integer args go to a0–a7, FP args go to fa0–fa7; each bank maintains
        // its own allocation counter.
        let mut int_idx: usize = 0;
        let mut fp_idx: usize = 0;
        let mut stack_offset: i64 = 0;

        for &(reg, is_fp) in args.iter() {
            if is_fp {
                if fp_idx < 8 {
                    let fa = FA0 + fp_idx as u8;
                    fp_idx += 1;
                    if reg != fa {
                        // FSGNJ.D fa_i, reg, reg  (FP MV)
                        self.emit_fp_r(RvOpcode::FSGNJ_D, fa, reg, reg);
                    }
                } else {
                    // FP spill to stack
                    self.emit_store(reg, SP, stack_offset, &IrType::F64);
                    stack_offset += 8;
                }
            } else if int_idx < 8 {
                let ai = A0 + int_idx as u8;
                int_idx += 1;
                if reg != ai {
                    self.emit_i(RvOpcode::ADDI, ai, reg, 0); // MV
                }
            } else {
                // Integer spill to stack
                self.emit_s(RvOpcode::SD, SP, reg, stack_offset);
                stack_offset += 8;
            }
        }

        // Generate the call instruction
        if self.pic_mode {
            // PIC: AUIPC+JALR through PLT
            self.emit(RvInstruction {
                opcode: RvOpcode::CALL,
                rd: Some(RA),
                rs1: None,
                rs2: None,
                rs3: None,
                imm: 0,
                symbol: Some(format!("{}@plt", callee)),
                is_fp: false,
                comment: Some(format!("call {}", callee)),
            });
        } else {
            // Direct call
            self.emit(RvInstruction {
                opcode: RvOpcode::CALL,
                rd: Some(RA),
                rs1: None,
                rs2: None,
                rs3: None,
                imm: 0,
                symbol: Some(callee.to_string()),
                is_fp: false,
                comment: Some(format!("call {}", callee)),
            });
        }

        // Move result from ABI return register to destination
        if let Some(dst) = result_reg {
            if result_is_fp {
                if dst != FA0 {
                    self.emit_fp_r(RvOpcode::FSGNJ_D, dst, FA0, FA0);
                }
            } else if dst != A0 {
                self.emit_i(RvOpcode::ADDI, dst, A0, 0); // MV
            }
        }
    }

    // ===================================================================
    // Inline assembly handling
    // ===================================================================

    /// Emit inline assembly as a passthrough instruction.
    ///
    /// The template string and constraints are preserved for the assembler
    /// to process. This is critical for Linux kernel builds which use
    /// extensive inline assembly.
    pub fn emit_inline_asm(
        &mut self,
        template: &str,
        constraints: &str,
        operand_regs: &[u8],
        clobbers: &[String],
        goto_targets: &[String],
    ) {
        let mut inst = RvInstruction {
            opcode: RvOpcode::INLINE_ASM,
            rd: None,
            rs1: None,
            rs2: None,
            rs3: None,
            imm: 0,
            symbol: Some(template.to_string()),
            is_fp: false,
            comment: Some(format!(
                "asm: constraints={}, clobbers=[{}], operands={}, gotos={}",
                constraints,
                clobbers.join(","),
                operand_regs.len(),
                goto_targets.len(),
            )),
        };

        // Encode operand registers in the instruction for the assembler
        if let Some(&r) = operand_regs.first() {
            inst.rs1 = Some(r);
        }
        if operand_regs.len() > 1 {
            inst.rs2 = Some(operand_regs[1]);
        }
        if operand_regs.len() > 2 {
            inst.rs3 = Some(operand_regs[2]);
        }

        self.emit(inst);
    }

    // ===================================================================
    // Switch statement lowering
    // ===================================================================

    /// Emit a switch statement.
    ///
    /// For small case counts (≤8): cascaded BEQ comparisons.
    /// For dense ranges: jump table with bounds check.
    /// For sparse cases: binary search tree.
    pub fn emit_switch(&mut self, val_reg: u8, default_label: &str, cases: &[(i64, String)]) {
        if cases.is_empty() {
            // Just jump to default
            self.emit_branch(None, default_label);
            return;
        }

        // Threshold: use cascaded comparisons for small case counts
        if cases.len() <= 8 {
            self.emit_switch_cascade(val_reg, default_label, cases);
            return;
        }

        // Check for dense range: if (max - min + 1) <= 2 * len, use jump table
        let min_val = cases.iter().map(|(v, _)| *v).min().unwrap_or(0);
        let max_val = cases.iter().map(|(v, _)| *v).max().unwrap_or(0);
        let range = (max_val - min_val + 1) as usize;

        if range <= cases.len() * 2 && range <= 1024 {
            self.emit_switch_jump_table(val_reg, default_label, cases, min_val, range);
        } else {
            // Sparse: use cascaded comparisons (can be extended to binary search)
            self.emit_switch_cascade(val_reg, default_label, cases);
        }
    }

    /// Emit cascaded BEQ comparisons for a switch statement.
    fn emit_switch_cascade(&mut self, val_reg: u8, default_label: &str, cases: &[(i64, String)]) {
        for (case_val, case_label) in cases {
            if fits_i12(*case_val) {
                // Compare: ADDI T0, x0, case_val; BEQ val_reg, T0, label
                self.emit_i(RvOpcode::ADDI, T0, ZERO, *case_val);
            } else {
                self.emit_immediate(T0, *case_val);
            }
            self.emit_b(RvOpcode::BEQ, val_reg, T0, case_label);
        }
        // Fall through to default
        self.emit_branch(None, default_label);
    }

    /// Emit a jump table for a dense switch statement.
    fn emit_switch_jump_table(
        &mut self,
        val_reg: u8,
        default_label: &str,
        cases: &[(i64, String)],
        min_val: i64,
        range: usize,
    ) {
        // Subtract min_val to get zero-based index
        if min_val != 0 {
            if fits_i12(-min_val) {
                self.emit_i(RvOpcode::ADDI, T0, val_reg, -min_val);
            } else {
                self.emit_immediate(T0, -min_val);
                self.emit_r(RvOpcode::ADD, T0, val_reg, T0);
            }
        } else {
            self.emit_i(RvOpcode::ADDI, T0, val_reg, 0); // MV
        }

        // Bounds check: if index >= range, jump to default
        self.emit_immediate(T1, range as i64);
        self.emit_b(RvOpcode::BGEU, T0, T1, default_label);

        // Scale index by 8 (pointer size) for table entry
        self.emit_i(RvOpcode::SLLI, T0, T0, 3);

        // Load jump table base address
        let table_label = format!(".L_switch_table_{}", self.instructions.len());
        if self.pic_mode {
            self.emit_pcrel_address(T1, &table_label);
        } else {
            self.emit_address(T1, &table_label);
        }

        // Load target address from table: LD T0, T1(T0)
        self.emit_r(RvOpcode::ADD, T0, T1, T0);
        self.emit_i(RvOpcode::LD, T0, T0, 0);

        // Jump to target: JALR x0, T0, 0
        self.emit_i(RvOpcode::JALR, ZERO, T0, 0);

        // Build the case map for table generation (done by assembler/linker)
        // Emit the table entries as data pseudo-instructions
        let mut table: Vec<String> = vec![default_label.to_string(); range];
        for (case_val, case_label) in cases {
            let idx = (*case_val - min_val) as usize;
            if idx < range {
                table[idx] = case_label.clone();
            }
        }

        // Emit table references as symbol-bearing pseudo-instructions
        for entry in &table {
            self.emit(RvInstruction {
                opcode: RvOpcode::LA,
                rd: None,
                rs1: None,
                rs2: None,
                rs3: None,
                imm: 0,
                symbol: Some(entry.clone()),
                is_fp: false,
                comment: Some(format!("switch table entry: {}", entry)),
            });
        }
    }

    // ===================================================================
    // Prologue / Epilogue generation
    // ===================================================================

    /// Emit function prologue.
    ///
    /// Allocates the stack frame, saves the return address and frame pointer,
    /// and saves all callee-saved registers used in the function body.
    ///
    /// Stack layout (growing downward):
    /// ```text
    /// [old SP] ← caller's stack pointer
    ///   RA save slot
    ///   FP (s0) save slot
    ///   s1 save slot (if used)
    ///   s2 save slot (if used)
    ///   ...
    ///   fs0 save slot (if used)
    ///   ...
    ///   local variables / spill area
    /// [new SP] ← current stack pointer
    /// ```
    pub fn emit_prologue(&mut self) {
        let frame = self.frame_size;
        if frame == 0 {
            return;
        }

        // Allocate stack frame: addi sp, sp, -frame_size
        if fits_i12(-frame) {
            self.emit_i(RvOpcode::ADDI, SP, SP, -frame);
        } else {
            // Large frame: materialize offset in T0 then sub
            self.emit_immediate(T0, frame);
            self.emit_r(RvOpcode::SUB, SP, SP, T0);
        }

        // Save return address
        let ra_offset = frame - 8;
        self.emit_s(RvOpcode::SD, SP, RA, ra_offset);

        // Save frame pointer
        let fp_offset = frame - 16;
        self.emit_s(RvOpcode::SD, SP, FP, fp_offset);

        // Set up frame pointer: addi s0, sp, frame_size
        if fits_i12(frame) {
            self.emit_i(RvOpcode::ADDI, FP, SP, frame);
        } else {
            self.emit_immediate(T0, frame);
            self.emit_r(RvOpcode::ADD, FP, SP, T0);
        }

        // Save callee-saved GPRs
        let mut offset = frame - 24; // after RA and FP
        for &reg in &self.used_callee_saved_gprs.clone() {
            if reg == FP {
                continue; // FP already saved above
            }
            if fits_i12(offset) {
                self.emit_s(RvOpcode::SD, SP, reg, offset);
            } else {
                self.emit_immediate(T0, offset);
                self.emit_r(RvOpcode::ADD, T0, SP, T0);
                self.emit_s(RvOpcode::SD, T0, reg, 0);
            }
            offset -= 8;
        }

        // Save callee-saved FPRs
        for &reg in &self.used_callee_saved_fprs.clone() {
            if fits_i12(offset) {
                let mut inst = RvInstruction::s_type(RvOpcode::FSD, SP, reg, offset);
                inst.is_fp = true;
                self.emit(inst);
            } else {
                self.emit_immediate(T0, offset);
                self.emit_r(RvOpcode::ADD, T0, SP, T0);
                let mut inst = RvInstruction::s_type(RvOpcode::FSD, T0, reg, 0);
                inst.is_fp = true;
                self.emit(inst);
            }
            offset -= 8;
        }
    }

    /// Emit function epilogue.
    ///
    /// Restores callee-saved registers, deallocates the stack frame,
    /// and returns to the caller.
    pub fn emit_epilogue(&mut self) {
        let frame = self.frame_size;
        if frame == 0 {
            self.emit(RvInstruction::no_op(RvOpcode::RET));
            return;
        }

        // Restore callee-saved FPRs (reverse order of prologue)
        // Count includes FP which we skip
        let gprs_to_restore: Vec<u8> = self
            .used_callee_saved_gprs
            .iter()
            .copied()
            .filter(|&r| r != FP)
            .collect();
        let fprs_to_restore: Vec<u8> = self.used_callee_saved_fprs.clone();

        // Restore FPRs
        let mut offset = frame - 24 - (gprs_to_restore.len() as i64 * 8);
        for &reg in fprs_to_restore.iter().rev() {
            offset -= 8;
            if fits_i12(offset) {
                let mut inst = RvInstruction::i_type(RvOpcode::FLD, reg, SP, offset);
                inst.is_fp = true;
                self.emit(inst);
            } else {
                self.emit_immediate(T0, offset);
                self.emit_r(RvOpcode::ADD, T0, SP, T0);
                let mut inst = RvInstruction::i_type(RvOpcode::FLD, reg, T0, 0);
                inst.is_fp = true;
                self.emit(inst);
            }
        }

        // Restore callee-saved GPRs (reverse order)
        offset = frame - 24;
        for &reg in gprs_to_restore.iter().rev() {
            if fits_i12(offset) {
                self.emit_i(RvOpcode::LD, reg, SP, offset);
            } else {
                self.emit_immediate(T0, offset);
                self.emit_r(RvOpcode::ADD, T0, SP, T0);
                self.emit_i(RvOpcode::LD, reg, T0, 0);
            }
            offset -= 8;
        }

        // Restore frame pointer
        let fp_offset = frame - 16;
        self.emit_i(RvOpcode::LD, FP, SP, fp_offset);

        // Restore return address
        let ra_offset = frame - 8;
        self.emit_i(RvOpcode::LD, RA, SP, ra_offset);

        // Deallocate stack frame
        if fits_i12(frame) {
            self.emit_i(RvOpcode::ADDI, SP, SP, frame);
        } else {
            self.emit_immediate(T0, frame);
            self.emit_r(RvOpcode::ADD, SP, SP, T0);
        }

        // Return
        self.emit(RvInstruction::no_op(RvOpcode::RET));
    }

    // ===================================================================
    // Arithmetic instruction selection helpers
    // ===================================================================

    /// Select arithmetic instructions for an IR BinOp.
    fn select_binop(&mut self, rd: u8, lhs_reg: u8, rhs_reg: u8, op: &BinOp, ty: &IrType) {
        // Decide whether to use W-variants for 32-bit int ops
        let use_word = matches!(ty, IrType::I32);

        match op {
            BinOp::Add => {
                if use_word {
                    self.emit_r(RvOpcode::ADDW, rd, lhs_reg, rhs_reg);
                } else {
                    self.emit_r(RvOpcode::ADD, rd, lhs_reg, rhs_reg);
                }
            }
            BinOp::Sub => {
                if use_word {
                    self.emit_r(RvOpcode::SUBW, rd, lhs_reg, rhs_reg);
                } else {
                    self.emit_r(RvOpcode::SUB, rd, lhs_reg, rhs_reg);
                }
            }
            BinOp::Mul => {
                if use_word {
                    self.emit_r(RvOpcode::MULW, rd, lhs_reg, rhs_reg);
                } else {
                    self.emit_r(RvOpcode::MUL, rd, lhs_reg, rhs_reg);
                }
            }
            BinOp::SDiv => {
                if use_word {
                    self.emit_r(RvOpcode::DIVW, rd, lhs_reg, rhs_reg);
                } else {
                    self.emit_r(RvOpcode::DIV, rd, lhs_reg, rhs_reg);
                }
            }
            BinOp::UDiv => {
                if use_word {
                    self.emit_r(RvOpcode::DIVUW, rd, lhs_reg, rhs_reg);
                } else {
                    self.emit_r(RvOpcode::DIVU, rd, lhs_reg, rhs_reg);
                }
            }
            BinOp::SRem => {
                if use_word {
                    self.emit_r(RvOpcode::REMW, rd, lhs_reg, rhs_reg);
                } else {
                    self.emit_r(RvOpcode::REM, rd, lhs_reg, rhs_reg);
                }
            }
            BinOp::URem => {
                if use_word {
                    self.emit_r(RvOpcode::REMUW, rd, lhs_reg, rhs_reg);
                } else {
                    self.emit_r(RvOpcode::REMU, rd, lhs_reg, rhs_reg);
                }
            }
            BinOp::And => {
                self.emit_r(RvOpcode::AND, rd, lhs_reg, rhs_reg);
            }
            BinOp::Or => {
                self.emit_r(RvOpcode::OR, rd, lhs_reg, rhs_reg);
            }
            BinOp::Xor => {
                self.emit_r(RvOpcode::XOR, rd, lhs_reg, rhs_reg);
            }
            BinOp::Shl => {
                if use_word {
                    self.emit_r(RvOpcode::SLLW, rd, lhs_reg, rhs_reg);
                } else {
                    self.emit_r(RvOpcode::SLL, rd, lhs_reg, rhs_reg);
                }
            }
            BinOp::AShr => {
                if use_word {
                    self.emit_r(RvOpcode::SRAW, rd, lhs_reg, rhs_reg);
                } else {
                    self.emit_r(RvOpcode::SRA, rd, lhs_reg, rhs_reg);
                }
            }
            BinOp::LShr => {
                if use_word {
                    self.emit_r(RvOpcode::SRLW, rd, lhs_reg, rhs_reg);
                } else {
                    self.emit_r(RvOpcode::SRL, rd, lhs_reg, rhs_reg);
                }
            }
            BinOp::FAdd => {
                if matches!(ty, IrType::F32) {
                    self.emit_fp_r(RvOpcode::FADD_S, rd, lhs_reg, rhs_reg);
                } else {
                    self.emit_fp_r(RvOpcode::FADD_D, rd, lhs_reg, rhs_reg);
                }
            }
            BinOp::FSub => {
                if matches!(ty, IrType::F32) {
                    self.emit_fp_r(RvOpcode::FSUB_S, rd, lhs_reg, rhs_reg);
                } else {
                    self.emit_fp_r(RvOpcode::FSUB_D, rd, lhs_reg, rhs_reg);
                }
            }
            BinOp::FMul => {
                if matches!(ty, IrType::F32) {
                    self.emit_fp_r(RvOpcode::FMUL_S, rd, lhs_reg, rhs_reg);
                } else {
                    self.emit_fp_r(RvOpcode::FMUL_D, rd, lhs_reg, rhs_reg);
                }
            }
            BinOp::FDiv => {
                if matches!(ty, IrType::F32) {
                    self.emit_fp_r(RvOpcode::FDIV_S, rd, lhs_reg, rhs_reg);
                } else {
                    self.emit_fp_r(RvOpcode::FDIV_D, rd, lhs_reg, rhs_reg);
                }
            }
            BinOp::FRem => {
                // RISC-V has no FREM instruction; implement via call to fmod/fmodf.
                // This requires the target binary to be linked against libm (-lm).
                // The linker must add -lm to resolve fmod/fmodf symbols.
                // Store operands in fa0/fa1, call fmod, result in fa0
                if matches!(ty, IrType::F32) {
                    if lhs_reg != FA0 {
                        self.emit_fp_r(RvOpcode::FSGNJ_S, FA0, lhs_reg, lhs_reg);
                    }
                    if rhs_reg != FA0 + 1 {
                        self.emit_fp_r(RvOpcode::FSGNJ_S, FA0 + 1, rhs_reg, rhs_reg);
                    }
                    self.emit(RvInstruction {
                        opcode: RvOpcode::CALL,
                        rd: Some(RA),
                        rs1: None,
                        rs2: None,
                        rs3: None,
                        imm: 0,
                        symbol: Some("fmodf".to_string()),
                        is_fp: false,
                        comment: Some("FRem via fmodf".to_string()),
                    });
                    if rd != FA0 {
                        self.emit_fp_r(RvOpcode::FSGNJ_S, rd, FA0, FA0);
                    }
                } else {
                    if lhs_reg != FA0 {
                        self.emit_fp_r(RvOpcode::FSGNJ_D, FA0, lhs_reg, lhs_reg);
                    }
                    if rhs_reg != FA0 + 1 {
                        self.emit_fp_r(RvOpcode::FSGNJ_D, FA0 + 1, rhs_reg, rhs_reg);
                    }
                    self.emit(RvInstruction {
                        opcode: RvOpcode::CALL,
                        rd: Some(RA),
                        rs1: None,
                        rs2: None,
                        rs3: None,
                        imm: 0,
                        symbol: Some("fmod".to_string()),
                        is_fp: false,
                        comment: Some("FRem via fmod".to_string()),
                    });
                    if rd != FA0 {
                        self.emit_fp_r(RvOpcode::FSGNJ_D, rd, FA0, FA0);
                    }
                }
                self.has_calls = true;
            }
        }
    }

    // ===================================================================
    // GEP (GetElementPtr) lowering
    // ===================================================================

    /// Lower a GetElementPtr instruction.
    fn select_gep(&mut self, result: Value, base: Value, indices: &[Value], result_type: &IrType) {
        let rd = self.vmap.alloc_gpr(result);
        let base_reg = self.src_reg(base);

        // Start with base address
        if rd != base_reg {
            self.emit_i(RvOpcode::ADDI, rd, base_reg, 0); // MV
        }

        // Apply each index: address += index * element_size
        let elem_size = result_type.size_bytes(&self.target) as i64;
        for idx in indices {
            let idx_reg = self.src_reg(*idx);
            if elem_size == 1 {
                self.emit_r(RvOpcode::ADD, rd, rd, idx_reg);
            } else if elem_size > 0 && (elem_size as u64).is_power_of_two() {
                let shift = (elem_size as u64).trailing_zeros() as i64;
                self.emit_i(RvOpcode::SLLI, T0, idx_reg, shift);
                self.emit_r(RvOpcode::ADD, rd, rd, T0);
            } else {
                // General case: multiply by element size
                self.emit_immediate(T0, elem_size);
                self.emit_r(RvOpcode::MUL, T0, idx_reg, T0);
                self.emit_r(RvOpcode::ADD, rd, rd, T0);
            }
        }
    }

    // ===================================================================
    // Main instruction selection dispatcher
    // ===================================================================

    /// Select RISC-V 64 instructions for a single IR instruction.
    ///
    /// Dispatches on the IR instruction variant and generates the
    /// corresponding RV64 machine instruction sequence.
    pub fn select_instruction(&mut self, ir_inst: &Instruction) -> Vec<RvInstruction> {
        let start_idx = self.instructions.len();

        match ir_inst {
            Instruction::Alloca {
                result,
                ty,
                alignment,
                span: _,
            } => {
                // Allocate stack space for the variable
                let size = ty.size_bytes(&self.target) as i64;
                let align = match alignment {
                    Some(a) if *a > 0 => *a as i64,
                    _ => ty.align_bytes(&self.target) as i64,
                };
                // Align the offset
                self.alloca_offset = (self.alloca_offset + align - 1) & !(align - 1);
                self.alloca_offset += size;
                let offset = -self.alloca_offset;

                // Store the stack offset for this alloca
                self.vmap.set_stack_offset(*result, offset);

                // Also give it a GPR that holds the address
                let rd = self.vmap.alloc_gpr(*result);
                // Compute address: rd = FP + offset
                if fits_i12(offset) {
                    self.emit_i(RvOpcode::ADDI, rd, FP, offset);
                } else {
                    self.emit_immediate(T0, offset);
                    self.emit_r(RvOpcode::ADD, rd, FP, T0);
                }
            }

            Instruction::Load {
                result,
                ptr,
                ty,
                volatile: _,
                span: _,
            } => {
                let ptr_reg = self.src_reg(*ptr);
                let rd = self.dest_reg(*result, ty);
                self.emit_load(rd, ptr_reg, 0, ty);
            }

            Instruction::Store {
                value,
                ptr,
                volatile: _,
                span: _,
            } => {
                let val_reg = self.src_reg(*value);
                let ptr_reg = self.src_reg(*ptr);
                // Determine store type from the value (default to I64)
                // The IR Store doesn't carry a type directly; infer from value mapping
                let ty = IrType::I64; // Default; real type tracking done by IR
                self.emit_store(val_reg, ptr_reg, 0, &ty);
            }

            Instruction::BinOp {
                result,
                op,
                lhs,
                rhs,
                ty,
                span: _,
            } => {
                let lhs_reg = self.src_reg(*lhs);
                let rhs_reg = self.src_reg(*rhs);
                let rd = self.dest_reg(*result, ty);
                self.select_binop(rd, lhs_reg, rhs_reg, op, ty);
            }

            Instruction::ICmp {
                result,
                op,
                lhs,
                rhs,
                span: _,
            } => {
                let lhs_reg = self.src_reg(*lhs);
                let rhs_reg = self.src_reg(*rhs);
                let rd = self.vmap.alloc_gpr(*result);
                self.emit_comparison(rd, lhs_reg, rhs_reg, op);
            }

            Instruction::FCmp {
                result,
                op,
                lhs,
                rhs,
                span: _,
            } => {
                let lhs_reg = self.src_reg(*lhs);
                let rhs_reg = self.src_reg(*rhs);
                let rd = self.vmap.alloc_gpr(*result);
                // Determine if double based on FP register class of operands.
                // FPRs have IDs >= 32. We default to double (F64) for safety since
                // the IR FCmp does not carry the operand type directly.
                let is_double_op = true; // conservative default
                self.emit_fcmp(rd, lhs_reg, rhs_reg, op, is_double_op);
            }

            Instruction::Branch { target, span: _ } => {
                let label = self.block_label(target.index());
                self.emit_branch(None, &label);
            }

            Instruction::CondBranch {
                condition,
                then_block,
                else_block,
                span: _,
            } => {
                let cond_reg = self.src_reg(*condition);
                let then_label = self.block_label(then_block.index());
                let else_label = self.block_label(else_block.index());

                // BNE cond_reg, ZERO, then_label (branch if true)
                self.emit_b(RvOpcode::BNE, cond_reg, ZERO, &then_label);
                // Fall-through or jump to else
                self.emit_branch(None, &else_label);
            }

            Instruction::Switch {
                value,
                default,
                cases,
                span: _,
            } => {
                let val_reg = self.src_reg(*value);
                let default_label = self.block_label(default.index());
                let case_entries: Vec<(i64, String)> = cases
                    .iter()
                    .map(|(val, blk)| (*val, self.block_label(blk.index())))
                    .collect();
                self.emit_switch(val_reg, &default_label, &case_entries);
            }

            Instruction::Call {
                result,
                callee,
                args,
                return_type,
                span: _,
            } => {
                self.has_calls = true;

                // Prepare argument registers
                let mut arg_regs: Vec<(u8, bool)> = Vec::new();
                for arg in args.iter() {
                    let reg = self.src_reg(*arg);
                    let is_fp_arg = is_fpr(reg);
                    arg_regs.push((reg, is_fp_arg));
                }

                // Determine callee symbol name from the Value index
                let callee_name = format!("__ir_callee_{}", callee.index());

                // result is a Value; allocate a register for it
                let result_reg = if *result != Value::UNDEF {
                    Some(if Self::uses_fpr(return_type) {
                        self.vmap.alloc_fpr(*result)
                    } else {
                        self.vmap.alloc_gpr(*result)
                    })
                } else {
                    None
                };
                let result_is_fp = Self::uses_fpr(return_type);

                self.emit_call(&callee_name, &arg_regs, result_reg, result_is_fp);
            }

            Instruction::Return { value, span: _ } => {
                if let Some(val) = value {
                    let val_reg = self.src_reg(*val);
                    // Move return value to a0 or fa0
                    if is_fpr(val_reg) {
                        if val_reg != FA0 {
                            self.emit_fp_r(RvOpcode::FSGNJ_D, FA0, val_reg, val_reg);
                        }
                    } else if val_reg != A0 {
                        self.emit_i(RvOpcode::ADDI, A0, val_reg, 0); // MV
                    }
                }
                // Epilogue will be emitted at function level
            }

            Instruction::Phi {
                result,
                ty,
                incoming: _,
                span: _,
            } => {
                // Phi nodes should be eliminated before instruction selection.
                // If encountered, just allocate a register for the result.
                let _rd = self.dest_reg(*result, ty);
            }

            Instruction::GetElementPtr {
                result,
                base,
                indices,
                result_type,
                in_bounds: _,
                span: _,
            } => {
                self.select_gep(*result, *base, indices, result_type);
            }

            Instruction::BitCast {
                result,
                value,
                to_type,
                span: _,
            } => {
                let src = self.src_reg(*value);
                let rd = self.dest_reg(*result, to_type);
                // BitCast: reinterpret bits, no conversion
                if rd != src {
                    if Self::uses_fpr(to_type) && is_gpr(src) {
                        self.emit_fp_unary(RvOpcode::FMV_D_X, rd, src);
                    } else if !Self::uses_fpr(to_type) && is_fpr(src) {
                        self.emit_fp_unary(RvOpcode::FMV_X_D, rd, src);
                    } else {
                        self.emit_i(RvOpcode::ADDI, rd, src, 0); // MV
                    }
                }
            }

            Instruction::Trunc {
                result,
                value,
                to_type,
                span: _,
            } => {
                let src = self.src_reg(*value);
                let rd = self.vmap.alloc_gpr(*result);
                // Truncation: mask to target width
                match to_type {
                    IrType::I1 => self.emit_i(RvOpcode::ANDI, rd, src, 1),
                    IrType::I8 => self.emit_i(RvOpcode::ANDI, rd, src, 0xFF),
                    IrType::I16 => {
                        self.emit_i(RvOpcode::SLLI, rd, src, 48);
                        self.emit_i(RvOpcode::SRLI, rd, rd, 48);
                    }
                    IrType::I32 => {
                        self.emit_i(RvOpcode::ADDIW, rd, src, 0);
                    }
                    _ => {
                        if rd != src {
                            self.emit_i(RvOpcode::ADDI, rd, src, 0);
                        }
                    }
                }
            }

            Instruction::ZExt {
                result,
                value,
                to_type,
                span: _,
            } => {
                let src = self.src_reg(*value);
                let rd = self.vmap.alloc_gpr(*result);
                self.emit_conversion(rd, src, &IrType::I32, to_type);
            }

            Instruction::SExt {
                result,
                value,
                to_type,
                span: _,
            } => {
                let src = self.src_reg(*value);
                let rd = self.vmap.alloc_gpr(*result);
                // Sign extension: use SLLI+SRAI pattern
                // Infer source width from current register width (heuristic)
                // Default: sign-extend from 32 to 64
                match to_type {
                    IrType::I64 | IrType::Ptr => {
                        // ADDIW sign-extends 32→64
                        self.emit_i(RvOpcode::ADDIW, rd, src, 0);
                    }
                    IrType::I32 => {
                        // Sign-extend from smaller type
                        self.emit_i(RvOpcode::SLLI, rd, src, 48);
                        self.emit_i(RvOpcode::SRAI, rd, rd, 48);
                    }
                    _ => {
                        if rd != src {
                            self.emit_i(RvOpcode::ADDI, rd, src, 0);
                        }
                    }
                }
            }

            Instruction::IntToPtr {
                result,
                value,
                span: _,
            } => {
                let src = self.src_reg(*value);
                let rd = self.vmap.alloc_gpr(*result);
                // IntToPtr: no-op on RV64 (both 64 bits)
                if rd != src {
                    self.emit_i(RvOpcode::ADDI, rd, src, 0);
                }
            }

            Instruction::PtrToInt {
                result,
                value,
                to_type,
                span: _,
            } => {
                let src = self.src_reg(*value);
                let rd = self.vmap.alloc_gpr(*result);
                // PtrToInt: no-op on RV64, possibly truncate
                if rd != src {
                    self.emit_i(RvOpcode::ADDI, rd, src, 0);
                }
                // If target is smaller than 64 bits, truncate
                match to_type {
                    IrType::I32 => {
                        self.emit_i(RvOpcode::ADDIW, rd, rd, 0);
                    }
                    IrType::I16 => {
                        self.emit_i(RvOpcode::SLLI, rd, rd, 48);
                        self.emit_i(RvOpcode::SRLI, rd, rd, 48);
                    }
                    IrType::I8 => {
                        self.emit_i(RvOpcode::ANDI, rd, rd, 0xFF);
                    }
                    _ => {}
                }
            }

            Instruction::InlineAsm {
                result: _,
                template,
                constraints,
                operands,
                clobbers,
                has_side_effects: _,
                is_volatile: _,
                goto_targets,
                span: _,
            } => {
                let operand_regs: Vec<u8> = operands.iter().map(|op| self.src_reg(*op)).collect();
                let target_labels: Vec<String> = goto_targets
                    .iter()
                    .map(|blk| self.block_label(blk.index()))
                    .collect();
                self.emit_inline_asm(
                    template,
                    constraints,
                    &operand_regs,
                    clobbers,
                    &target_labels,
                );
            }
        }

        // Return the newly emitted instructions
        self.instructions[start_idx..].to_vec()
    }

    // ===================================================================
    // Function-level instruction selection
    // ===================================================================

    /// Select RISC-V 64 instructions for an entire IR function.
    ///
    /// This is the main entry point for instruction selection. It:
    /// 1. Scans all blocks to build the block label map
    /// 2. Classifies function parameters per LP64D ABI
    /// 3. Selects instructions for each basic block
    /// 4. Computes the stack frame layout
    /// 5. Emits prologue and epilogue
    pub fn select_function(&mut self, func: &IrFunction, _abi: &RiscV64Abi) -> Vec<RvInstruction> {
        self.reset();

        // Phase 1: Build block label map
        for (i, block) in func.blocks().iter().enumerate() {
            let label = if let Some(ref lbl) = block.label {
                lbl.clone()
            } else {
                format!(".LBB_{}_{}", func.name, i)
            };
            self.block_labels.insert(block.index, label);
        }

        // Phase 2: Set up parameter registers per LP64D ABI
        let mut abi_state = RiscV64Abi::new();
        for param in func.params.iter() {
            let loc = abi_state.classify_arg(&param_ir_to_ctype(&param.ty));
            match loc {
                ArgLocation::Register(reg) => {
                    self.vmap.set_reg(param.value, reg as u8);
                }
                ArgLocation::RegisterPair(r1, _r2) => {
                    self.vmap.set_reg(param.value, r1 as u8);
                }
                ArgLocation::Stack(offset) => {
                    self.vmap.set_stack_offset(param.value, offset as i64);
                    // Load from stack into a temp register
                    let rd = self.vmap.alloc_gpr(param.value);
                    self.emit_i(RvOpcode::LD, rd, FP, offset as i64);
                }
            }
        }

        // Phase 3: Select instructions for each basic block
        for block in func.blocks().iter() {
            let label = self
                .block_labels
                .get(&block.index)
                .cloned()
                .unwrap_or_else(|| format!(".LBB_{}", block.index));
            self.emit_block_label(&label);

            for inst in &block.instructions {
                self.select_instruction(inst);
            }
        }

        // Phase 4: Compute frame layout
        // Callee-saved registers (excluding FP which is always saved)
        let callee_saved_count =
            self.used_callee_saved_gprs.len() + self.used_callee_saved_fprs.len();

        // Frame size = locals + callee_saved_regs * 8 + RA + FP + alignment
        let locals_size = self.alloca_offset.unsigned_abs() as usize;
        let frame_layout = RiscV64Abi::compute_frame_layout(
            &func
                .params
                .iter()
                .map(|p| param_ir_to_ctype(&p.ty))
                .collect::<Vec<_>>(),
            locals_size,
            callee_saved_count,
        );
        self.frame_size = frame_layout.total_size as i64;

        // Ensure frame size is aligned to STACK_ALIGNMENT (16 bytes)
        self.frame_size =
            (self.frame_size + (STACK_ALIGNMENT as i64) - 1) & !((STACK_ALIGNMENT as i64) - 1);

        // Phase 5: Generate prologue and epilogue
        // Insert prologue at the beginning
        let body_instructions = std::mem::take(&mut self.instructions);
        self.emit_prologue();
        let mut result = std::mem::take(&mut self.instructions);
        result.extend(body_instructions);

        // Add epilogue at the end (before any existing RET)
        self.instructions = result;
        self.emit_epilogue();

        std::mem::take(&mut self.instructions)
    }
}

// ===========================================================================
// Helper: convert IrType to CType for ABI classification
// ===========================================================================

/// Map an IR type to a C type for ABI calling convention classification.
fn param_ir_to_ctype(ty: &IrType) -> CType {
    match ty {
        IrType::Void => CType::Void,
        IrType::I1 | IrType::I8 => CType::Char,
        IrType::I16 => CType::Short,
        IrType::I32 => CType::Int,
        IrType::I64 => CType::Long,
        IrType::I128 => CType::LongLong,
        IrType::F32 => CType::Float,
        IrType::F64 => CType::Double,
        IrType::F80 => CType::LongDouble,
        IrType::Ptr => CType::Pointer(Box::new(CType::Void), TypeQualifiers::default()),
        IrType::Array(_, _) => CType::Pointer(Box::new(CType::Void), TypeQualifiers::default()),
        IrType::Struct(_) => CType::Struct {
            name: None,
            fields: Vec::new(),
            packed: false,
            aligned: None,
        },
        IrType::Function(_, _) => CType::Pointer(Box::new(CType::Void), TypeQualifiers::default()),
    }
}
