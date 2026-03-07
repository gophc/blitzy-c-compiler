//! # x86-64 Instruction Selection and Emission
//!
//! Converts BCC IR instructions into x86-64 machine instructions.  This module
//! is the **primary validation target** — it is validated first in the backend
//! order: x86-64 → i686 → AArch64 → RISC-V 64.
//!
//! ## Key Design Points
//!
//! - **Variable-length encoding**: x86-64 instructions range 1–15 bytes; REX
//!   prefix handling is critical for registers R8–R15 and XMM8–XMM15.
//! - **System V AMD64 ABI**: 16 GPRs (RAX–R15), 16 SSE registers (XMM0–XMM15).
//!   Integer arithmetic uses GPRs; floating-point uses SSE/SSE2 (**not** x87).
//! - **Complex addressing**: `[base + index*scale + disp]` with scale ∈ {1,2,4,8}.
//! - **Implicit register constraints**: DIV/IDIV use RDX:RAX; shifts use CL.
//! - **Stack alignment**: 16-byte at call sites; 128-byte red zone for leaf functions.
//! - **CMOV optimisation**: conditional moves avoid branch misprediction penalty.
//! - **LEA optimisation**: used for multi-operand address computation and fast arithmetic.
//!
//! ## Zero-Dependency Mandate
//!
//! Only `std` and `crate::` references are used.  No external crates.

use crate::backend::traits::{
    self as traits, ArchCodegen, MachineBasicBlock, MachineFunction, MachineInstruction,
    MachineOperand, RegisterInfo, RelocationTypeInfo,
};
use crate::backend::x86_64::abi::{
    self, RetLocation, X86_64Abi, INTEGER_ARG_REGS, RED_ZONE_SIZE, SSE_ARG_REGS,
};
use crate::backend::x86_64::registers::{
    self, CALLEE_SAVED_GPRS, RAX, RBP, RCX, RDX, RSP, XMM0, XMM1,
};
use crate::common::diagnostics::{DiagnosticEngine, Span};
use crate::common::fx_hash::FxHashMap;
use crate::common::target::Target;
use crate::ir::function::IrFunction;
use crate::ir::instructions::{BinOp, FCmpOp, ICmpOp, Instruction, Value};
use crate::ir::types::IrType;

// ===========================================================================
// X86Opcode — x86-64 machine opcode enumeration
// ===========================================================================

/// x86-64 machine instruction opcodes.
///
/// Each variant maps to one or more physical x86-64 instructions.  The
/// assembler module (`encoder.rs`) converts these to binary encodings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum X86Opcode {
    // -- Data movement --
    /// `MOV dst, src` — register-to-register, immediate-to-register, or memory.
    Mov,
    /// `MOVZX dst, src` — zero-extend move (8/16-bit → 32/64-bit).
    MovZX,
    /// `MOVSX dst, src` — sign-extend move (8/16/32-bit → 32/64-bit).
    MovSX,
    /// `LEA dst, [addr]` — load effective address (no memory access).
    Lea,
    /// `PUSH src` — push onto stack, decrement RSP.
    Push,
    /// `POP dst` — pop from stack, increment RSP.
    Pop,
    /// `XCHG a, b` — exchange two operands atomically.
    Xchg,

    // -- Integer arithmetic --
    /// `ADD dst, src`
    Add,
    /// `SUB dst, src`
    Sub,
    /// `IMUL dst, src` — signed multiply (two-/three-operand form).
    Imul,
    /// `IDIV src` — signed divide RDX:RAX by src.  Quotient→RAX, remainder→RDX.
    Idiv,
    /// `DIV src` — unsigned divide RDX:RAX by src.
    Div,
    /// `NEG dst` — two's complement negate.
    Neg,
    /// `INC dst`
    Inc,
    /// `DEC dst`
    Dec,

    // -- Bitwise --
    /// `AND dst, src`
    And,
    /// `OR dst, src`
    Or,
    /// `XOR dst, src`
    Xor,
    /// `NOT dst`
    Not,
    /// `SHL dst, CL/imm8`
    Shl,
    /// `SHR dst, CL/imm8` — logical shift right.
    Shr,
    /// `SAR dst, CL/imm8` — arithmetic shift right.
    Sar,
    /// `ROL dst, CL/imm8`
    Rol,
    /// `ROR dst, CL/imm8`
    Ror,

    // -- Comparison --
    /// `CMP a, b` — sets EFLAGS; result discarded.
    Cmp,
    /// `TEST a, b` — bitwise AND; sets EFLAGS; result discarded.
    Test,

    // -- Conditional move (CMOV) — all 16 conditions --
    Cmovo,
    Cmovno,
    Cmovb,
    Cmovae,
    Cmove,
    Cmovne,
    Cmovbe,
    Cmova,
    Cmovs,
    Cmovns,
    Cmovp,
    Cmovnp,
    Cmovl,
    Cmovge,
    Cmovle,
    Cmovg,

    // -- SETcc — set byte on condition — all 16 conditions --
    Seto,
    Setno,
    Setb,
    Setae,
    Sete,
    Setne,
    Setbe,
    Seta,
    Sets,
    Setns,
    Setp,
    Setnp,
    Setl,
    Setge,
    Setle,
    Setg,

    // -- Control flow / branches — all 16 conditions + JMP --
    Jmp,
    Jo,
    Jno,
    Jb,
    Jae,
    Je,
    Jne,
    Jbe,
    Ja,
    Js,
    Jns,
    Jp,
    Jnp,
    Jl,
    Jge,
    Jle,
    Jg,
    /// `CALL target`
    Call,
    /// `RET`
    Ret,

    /// `LoadInd dst, ptr` — Load 64-bit from memory at address in ptr register.
    /// Encoded as `MOV r64, [ptr]`.
    LoadInd,
    /// `LoadInd32 dst, ptr` — Load 32-bit from `[ptr]`, zero-extends to 64-bit.
    LoadInd32,
    /// `LoadInd16 dst, ptr` — Load 16-bit from `[ptr]`, zero-extends to 64-bit.
    LoadInd16,
    /// `LoadInd8 dst, ptr` — Load 8-bit from `[ptr]`, zero-extends to 64-bit.
    LoadInd8,

    /// `StoreInd ptr, src` — Store 64-bit src to memory at address in ptr register.
    /// Encoded as `MOV [ptr], src`.
    StoreInd,
    /// `StoreInd32 ptr, src` — Store low 32-bit of src to `[ptr]`.
    StoreInd32,
    /// `StoreInd16 ptr, src` — Store low 16-bit of src to `[ptr]`.
    StoreInd16,
    /// `StoreInd8 ptr, src` — Store low 8-bit of src to `[ptr]`.
    StoreInd8,
    /// `NOP`
    Nop,

    // -- SSE / SSE2 floating-point --
    /// `MOVSD xmm, xmm/m64` — move scalar double.
    Movsd,
    /// `MOVSS xmm, xmm/m32` — move scalar single.
    Movss,
    Addsd,
    Addss,
    Subsd,
    Subss,
    Mulsd,
    Mulss,
    Divsd,
    Divss,
    /// `UCOMISD xmm, xmm/m64` — unordered compare, set EFLAGS.
    Ucomisd,
    /// `UCOMISS xmm, xmm/m32` — unordered compare, set EFLAGS.
    Ucomiss,
    Cvtsi2sd,
    Cvtsi2ss,
    Cvtsd2si,
    Cvtss2si,
    Cvtsd2ss,
    Cvtss2sd,

    // -- Stack frame --
    Enter,
    Leave,

    // -- Special --
    /// `CDQ` — sign-extend EAX → EDX:EAX (for 32-bit IDIV).
    Cdq,
    /// `CQO` — sign-extend RAX → RDX:RAX (for 64-bit IDIV).
    Cqo,
    /// `ENDBR64` — CET indirect-branch tracking marker.
    Endbr64,
    /// `PAUSE` — spin-loop hint (used in retpoline).
    Pause,
    /// `LFENCE` — load fence (used in retpoline).
    Lfence,
    /// Inline assembly marker — template is carried in the operands.
    InlineAsm,
}

impl X86Opcode {
    /// Convert to the `u32` opcode value used by [`MachineInstruction`].
    #[inline]
    pub fn as_u32(self) -> u32 {
        self as u32
    }

    /// Try to construct from a raw `u32` opcode value.
    pub fn from_u32(val: u32) -> Option<Self> {
        // Safe table-driven lookup (avoids transmute size mismatch).
        const TABLE: &[X86Opcode] = &[
            X86Opcode::Mov,
            X86Opcode::MovZX,
            X86Opcode::MovSX,
            X86Opcode::Lea,
            X86Opcode::Push,
            X86Opcode::Pop,
            X86Opcode::Xchg,
            X86Opcode::Add,
            X86Opcode::Sub,
            X86Opcode::Imul,
            X86Opcode::Idiv,
            X86Opcode::Div,
            X86Opcode::Neg,
            X86Opcode::Inc,
            X86Opcode::Dec,
            X86Opcode::And,
            X86Opcode::Or,
            X86Opcode::Xor,
            X86Opcode::Not,
            X86Opcode::Shl,
            X86Opcode::Shr,
            X86Opcode::Sar,
            X86Opcode::Rol,
            X86Opcode::Ror,
            X86Opcode::Cmp,
            X86Opcode::Test,
            X86Opcode::Cmovo,
            X86Opcode::Cmovno,
            X86Opcode::Cmovb,
            X86Opcode::Cmovae,
            X86Opcode::Cmove,
            X86Opcode::Cmovne,
            X86Opcode::Cmovbe,
            X86Opcode::Cmova,
            X86Opcode::Cmovs,
            X86Opcode::Cmovns,
            X86Opcode::Cmovp,
            X86Opcode::Cmovnp,
            X86Opcode::Cmovl,
            X86Opcode::Cmovge,
            X86Opcode::Cmovle,
            X86Opcode::Cmovg,
            X86Opcode::Seto,
            X86Opcode::Setno,
            X86Opcode::Setb,
            X86Opcode::Setae,
            X86Opcode::Sete,
            X86Opcode::Setne,
            X86Opcode::Setbe,
            X86Opcode::Seta,
            X86Opcode::Sets,
            X86Opcode::Setns,
            X86Opcode::Setp,
            X86Opcode::Setnp,
            X86Opcode::Setl,
            X86Opcode::Setge,
            X86Opcode::Setle,
            X86Opcode::Setg,
            X86Opcode::Jmp,
            X86Opcode::Jo,
            X86Opcode::Jno,
            X86Opcode::Jb,
            X86Opcode::Jae,
            X86Opcode::Je,
            X86Opcode::Jne,
            X86Opcode::Jbe,
            X86Opcode::Ja,
            X86Opcode::Js,
            X86Opcode::Jns,
            X86Opcode::Jp,
            X86Opcode::Jnp,
            X86Opcode::Jl,
            X86Opcode::Jge,
            X86Opcode::Jle,
            X86Opcode::Jg,
            X86Opcode::Call,
            X86Opcode::Ret,
            X86Opcode::LoadInd,
            X86Opcode::LoadInd32,
            X86Opcode::LoadInd16,
            X86Opcode::LoadInd8,
            X86Opcode::StoreInd,
            X86Opcode::StoreInd32,
            X86Opcode::StoreInd16,
            X86Opcode::StoreInd8,
            X86Opcode::Nop,
            X86Opcode::Movsd,
            X86Opcode::Movss,
            X86Opcode::Addsd,
            X86Opcode::Addss,
            X86Opcode::Subsd,
            X86Opcode::Subss,
            X86Opcode::Mulsd,
            X86Opcode::Mulss,
            X86Opcode::Divsd,
            X86Opcode::Divss,
            X86Opcode::Ucomisd,
            X86Opcode::Ucomiss,
            X86Opcode::Cvtsi2sd,
            X86Opcode::Cvtsi2ss,
            X86Opcode::Cvtsd2si,
            X86Opcode::Cvtss2si,
            X86Opcode::Cvtsd2ss,
            X86Opcode::Cvtss2sd,
            X86Opcode::Enter,
            X86Opcode::Leave,
            X86Opcode::Cdq,
            X86Opcode::Cqo,
            X86Opcode::Endbr64,
            X86Opcode::Pause,
            X86Opcode::Lfence,
            X86Opcode::InlineAsm,
        ];
        TABLE.get(val as usize).copied()
    }
}

// ===========================================================================
// X86Operand — x86-64 addressing mode representation
// ===========================================================================

/// x86-64 operand addressing modes.
///
/// These are higher-level than the raw `MachineOperand` and carry
/// x86-specific addressing mode information.  The assembler converts
/// `X86Operand` values into encoded instruction bytes.
#[derive(Debug, Clone)]
pub enum X86Operand {
    /// Physical register.
    Reg(u16),
    /// Immediate constant.
    Imm(i64),
    /// Memory: `[base + index*scale + disp]`.
    Mem {
        base: Option<u16>,
        index: Option<u16>,
        scale: u8,
        disp: i64,
        size: u8,
    },
    /// RIP-relative: `[rip + symbol + offset]` for PIC.
    RipRelative { symbol: String, offset: i64 },
    /// Branch label.
    Label(String),
    /// Global symbol reference.
    Symbol(String),
}

// ===========================================================================
// FrameLayout — stack frame layout computation result
// ===========================================================================

/// Describes the stack frame layout for a single x86-64 function.
///
/// The frame pointer (RBP) points to the saved RBP on the stack.
/// Local allocas live at negative offsets below RBP.  The stack grows
/// downward (toward lower addresses).
///
/// ```text
/// ┌──────────────────────┐  ← caller's RSP (before CALL)
/// │    return address     │  ← pushed by CALL
/// ├──────────────────────┤
/// │    saved RBP          │  ← RBP points here after PUSH RBP; MOV RBP,RSP
/// ├──────────────────────┤
/// │   callee-saved regs   │
/// ├──────────────────────┤
/// │   local allocas       │  ← negative offsets from RBP
/// ├──────────────────────┤
/// │   spill area          │
/// ├──────────────────────┤
/// │   outgoing args       │  ← aligned to 16 bytes at CALL
/// └──────────────────────┘  ← RSP (after prologue)
/// ```
pub struct FrameLayout {
    /// Total stack frame size in bytes (the SUB RSP, N amount).
    pub total_size: usize,
    /// Map from IR `Value` (alloca result) to RBP-relative offset (negative).
    pub alloca_offsets: FxHashMap<Value, i32>,
    /// Byte offset from RBP where spill slots begin (negative).
    pub spill_area_offset: i32,
    /// Whether the function contains any CALL instructions.
    pub has_calls: bool,
    /// Callee-saved registers that are used and must be saved/restored.
    pub callee_saved: Vec<u16>,
}

// ===========================================================================
// X86_64CodeGen — main codegen struct
// ===========================================================================

/// x86-64 instruction selector and code generator.
///
/// Converts IR functions into [`MachineFunction`]s containing x86-64 machine
/// instructions.  Handles all 20 IR instruction variants, System V AMD64 ABI
/// compliance, and complex x86-64 addressing modes.
pub struct X86_64CodeGen {
    /// Target architecture (always `Target::X86_64`).
    target: Target,
    /// ABI helper for argument/return classification.
    abi: X86_64Abi,
    /// Next virtual register number for the current function.
    next_vreg: u32,
    /// Map from IR `Value` to the `MachineOperand` holding its result.
    value_map: FxHashMap<Value, MachineOperand>,
    /// Map from IR `Value` to its `IrType` — used for ABI classification
    /// during call instruction selection so FP arguments are routed to
    /// SSE registers (XMM0-7) rather than integer registers.
    value_types: FxHashMap<Value, IrType>,
    /// Map from IR block index to machine block index.
    block_map: FxHashMap<usize, usize>,
    /// Current frame layout (populated during `compute_frame_layout`).
    frame: Option<FrameLayout>,
    /// Cache of compile-time integer constants resolved from the IR
    /// constant-sentinel pattern (`BinOp(Add, result, UNDEF)`).
    /// Populated by a pre-scan pass during `lower_function`.
    constant_cache: FxHashMap<Value, i64>,
    /// Cache of compile-time float constants resolved from the IR
    /// constant-sentinel pattern, mapping Value to the `.Lconst.f.*`
    /// global name for RIP-relative SSE loads.
    float_constant_cache: FxHashMap<Value, String>,
    /// Accumulator mapping every allocated vreg to its IR Value.
    /// Unlike `value_map` (which stores only the *latest* operand per Value
    /// and loses earlier vregs when a Value is redefined in a later block,
    /// e.g. after phi-elimination), this map preserves every single
    /// vreg→Value association created during codegen so that
    /// `apply_allocation_result` can map *all* vregs to physical registers.
    vreg_ir_map: FxHashMap<u32, Value>,
    /// Map from IR callee `Value` to function name for direct call
    /// resolution. Populated from `IrModule::func_ref_map` during lowering.
    func_ref_names: FxHashMap<Value, String>,
    /// Map from IR `Value` (representing a global variable address) to the
    /// global variable name.  Used to emit RIP-relative addressing for
    /// global variable loads/stores on x86-64.
    global_var_refs: FxHashMap<Value, String>,
}

impl X86_64CodeGen {
    /// Create a new x86-64 code generator for the given target.
    ///
    /// # Panics
    ///
    /// Debug-asserts that `target` is `Target::X86_64`.
    pub fn new(target: Target) -> Self {
        debug_assert!(
            target == Target::X86_64,
            "X86_64CodeGen requires Target::X86_64"
        );
        let abi = X86_64Abi::new(target);
        X86_64CodeGen {
            target,
            abi,
            next_vreg: 0,
            value_map: FxHashMap::default(),
            value_types: FxHashMap::default(),
            block_map: FxHashMap::default(),
            frame: None,
            constant_cache: FxHashMap::default(),
            float_constant_cache: FxHashMap::default(),
            vreg_ir_map: FxHashMap::default(),
            func_ref_names: FxHashMap::default(),
            global_var_refs: FxHashMap::default(),
        }
    }

    /// Set module-level function reference map for direct call resolution.
    pub fn set_func_ref_names(&mut self, map: FxHashMap<Value, String>) {
        self.func_ref_names = map;
    }

    /// Set module-level global variable reference map for RIP-relative
    /// addressing during instruction selection.
    pub fn set_global_var_refs(&mut self, map: FxHashMap<Value, String>) {
        self.global_var_refs = map;
    }

    /// Pre-populate the constant cache by scanning the given IR function
    /// for `BinOp(Add, result, UNDEF)` sentinel instructions and extracting
    /// constant integer values from the corresponding module globals.
    ///
    /// The IR lowering phase (`emit_int_const`) pairs each compile-time
    /// integer constant with a global named `.Lconst.i.{G}`.  We collect
    /// all sentinel `Value`s in instruction order and all integer-constant
    /// globals sorted by their index suffix, then pair them 1:1.  Float
    /// constants (`.Lconst.f.*`) are handled similarly.
    pub fn populate_constant_cache(
        &mut self,
        func: &IrFunction,
        globals: &[crate::ir::module::GlobalVariable],
    ) {
        self.constant_cache.clear();
        self.float_constant_cache.clear();

        // ---------------------------------------------------------------
        // Strategy: use the DIRECT Value→constant mappings recorded during
        // IR lowering on `IrFunction`.  This avoids the fragile positional
        // matching between orphaned Values and global `.Lconst.*` entries
        // that breaks when optimisation passes remove some sentinels.
        // ---------------------------------------------------------------

        // 1. Populate integer constants from the authoritative map on
        //    the function (recorded at emit_int_const time).
        for (&val, &imm) in &func.constant_values {
            self.constant_cache.insert(val, imm);
        }

        // 2. Populate float constants from the authoritative map.
        for (&val, (name, _fval)) in &func.float_constant_values {
            self.float_constant_cache.insert(val, name.clone());
        }

        // 3. Fallback: if the direct maps are empty (e.g. functions
        //    built without the new IR lowering), fall back to the
        //    legacy positional-matching approach.
        if func.constant_values.is_empty() && func.float_constant_values.is_empty() {
            self.populate_constant_cache_legacy(func, globals);
        }
    }

    /// Legacy constant cache population using positional matching.
    ///
    /// Used only when `IrFunction::constant_values` is empty (for
    /// backwards compatibility with functions not built by the new
    /// IR lowering that records direct mappings).
    fn populate_constant_cache_legacy(
        &mut self,
        func: &IrFunction,
        globals: &[crate::ir::module::GlobalVariable],
    ) {
        // Collect all constant Values: both surviving sentinels
        // and orphaned (sentinel-removed) references.
        let mut defined = std::collections::HashSet::new();
        for param in &func.params {
            if param.value != Value::UNDEF {
                defined.insert(param.value.index());
            }
        }
        for block in func.blocks() {
            for inst in block.instructions() {
                if let Some(r) = inst.result() {
                    if r != Value::UNDEF {
                        defined.insert(r.index());
                    }
                }
            }
        }

        let mut referenced = std::collections::HashSet::new();
        for block in func.blocks() {
            for inst in block.instructions() {
                for op in inst.operands() {
                    if op != Value::UNDEF {
                        referenced.insert(op.index());
                    }
                }
            }
        }

        let func_ref_indices: crate::common::fx_hash::FxHashSet<u32> =
            self.func_ref_names.keys().map(|v| v.index()).collect();
        let global_ref_indices: crate::common::fx_hash::FxHashSet<u32> =
            self.global_var_refs.keys().map(|v| v.index()).collect();

        // Collect orphaned values (referenced but not defined).
        let orphans: std::collections::HashSet<u32> = referenced
            .difference(&defined)
            .copied()
            .filter(|idx| !func_ref_indices.contains(idx) && !global_ref_indices.contains(idx))
            .collect();

        // Collect surviving integer sentinel values.
        let mut int_sentinels: Vec<u32> = Vec::new();
        let mut float_sentinels: Vec<u32> = Vec::new();
        for block in func.blocks() {
            for inst in block.instructions() {
                if let Instruction::BinOp {
                    result, op, lhs, rhs, ty, ..
                } = inst
                {
                    let is_sentinel = (*op == BinOp::Add || *op == BinOp::FAdd)
                        && *rhs == Value::UNDEF
                        && *lhs == *result;
                    if is_sentinel {
                        if *op == BinOp::FAdd
                            || matches!(ty, IrType::F32 | IrType::F64 | IrType::F80)
                        {
                            float_sentinels.push(result.index());
                        } else {
                            int_sentinels.push(result.index());
                        }
                    }
                }
            }
        }

        // Integer orphans: orphaned values that are NOT float sentinels.
        let float_sentinel_set: std::collections::HashSet<u32> =
            float_sentinels.iter().copied().collect();
        let int_sentinel_set: std::collections::HashSet<u32> =
            int_sentinels.iter().copied().collect();

        // All integer constant values = orphans (non-float) + surviving int sentinels.
        let mut all_int_const_vals: Vec<u32> = orphans
            .iter()
            .copied()
            .filter(|v| !float_sentinel_set.contains(v))
            .collect();
        all_int_const_vals.extend(int_sentinels.iter());
        all_int_const_vals.sort();
        all_int_const_vals.dedup();

        // All float constant values = float orphans + surviving float sentinels.
        let mut all_float_const_vals: Vec<u32> = orphans
            .iter()
            .copied()
            .filter(|v| !int_sentinel_set.contains(v) && !all_int_const_vals.contains(v))
            .collect();
        all_float_const_vals.extend(float_sentinels.iter());
        all_float_const_vals.sort();
        all_float_const_vals.dedup();

        // Collect integer constants from globals, sorted by creation index.
        let mut int_consts: Vec<(u32, i64)> = Vec::new();
        let mut float_consts: Vec<(u32, String)> = Vec::new();
        for gv in globals {
            if let Some(crate::ir::module::Constant::Integer(v)) = gv.initializer.as_ref() {
                if let Some(idx_str) = gv.name.strip_prefix(".Lconst.i.") {
                    if let Ok(idx) = idx_str.parse::<u32>() {
                        int_consts.push((idx, *v as i64));
                    }
                }
            }
            if let Some(crate::ir::module::Constant::Float(_)) = gv.initializer.as_ref() {
                if let Some(idx_str) = gv.name.strip_prefix(".Lconst.f.") {
                    if let Ok(idx) = idx_str.parse::<u32>() {
                        float_consts.push((idx, gv.name.clone()));
                    }
                }
            }
        }
        int_consts.sort_by_key(|(idx, _)| *idx);
        float_consts.sort_by_key(|(idx, _)| *idx);

        // Pair ALL integer constant Values with sorted globals.
        for (val_idx, (_gidx, gval)) in all_int_const_vals.iter().zip(int_consts.iter()) {
            self.constant_cache.insert(Value(*val_idx), *gval);
        }

        // Pair ALL float constant Values with sorted float globals.
        for (val_idx, (_gidx, gname)) in all_float_const_vals.iter().zip(float_consts.iter()) {
            self.float_constant_cache
                .insert(Value(*val_idx), gname.clone());
        }
    }

    // -- Virtual register allocation ---

    /// Allocate a fresh virtual register number.
    #[inline]
    fn alloc_vreg(&mut self) -> u32 {
        let v = self.next_vreg;
        self.next_vreg += 1;
        v
    }

    /// Allocate a virtual register and return it as a `MachineOperand`.
    #[inline]
    fn new_vreg(&mut self) -> MachineOperand {
        MachineOperand::VirtualRegister(self.alloc_vreg())
    }

    /// Record the machine operand that holds an IR value's result.
    ///
    /// Also accumulates every vreg→Value association in `vreg_ir_map`
    /// so that values defined more than once (e.g. phi-eliminated copies)
    /// have ALL of their vregs mapped for register allocation.
    #[inline]
    fn set_value(&mut self, val: Value, op: MachineOperand) {
        if let MachineOperand::VirtualRegister(vreg) = &op {
            self.vreg_ir_map.insert(*vreg, val);
        }
        self.value_map.insert(val, op);
    }

    /// Retrieve the machine operand for an IR value.
    ///
    /// Returns a `VirtualRegister(u32::MAX)` sentinel if not found
    /// (should not happen for well-formed IR).
    fn get_value(&self, val: Value) -> MachineOperand {
        if let Some(op) = self.value_map.get(&val) {
            return op.clone();
        }
        // Check the constant cache — the value may be a compile-time
        // integer constant whose sentinel instruction was eliminated
        // by optimisation passes.
        if let Some(&imm) = self.constant_cache.get(&val) {
            return MachineOperand::Immediate(imm);
        }
        // Check float constant cache — the value may be a floating-point
        // constant whose sentinel was eliminated.  Return a GlobalSymbol
        // so the encoder emits a RIP-relative SSE load.
        if let Some(name) = self.float_constant_cache.get(&val) {
            return MachineOperand::GlobalSymbol(name.clone());
        }
        // Check global variable reference map — the value may represent
        // the address of a global variable.  Return a GlobalSymbol so
        // the load/store handlers can emit RIP-relative addressing.
        if let Some(name) = self.global_var_refs.get(&val) {
            return MachineOperand::GlobalSymbol(name.clone());
        }
        MachineOperand::Immediate(0)
    }

    /// Resolve a constant-sentinel IR value to its immediate integer value.
    ///
    /// The IR lowering phase represents compile-time integer constants as
    /// `BinOp(Add, result, Value::UNDEF)` sentinels paired with a global
    /// variable holding `Constant::Integer(n)`.  The global is named
    /// `.Lconst.i.{G}` where `G` is the global index at creation time.
    ///
    /// We search the `IrFunction`'s parent module (passed via globals
    /// snapshot) for the corresponding constant.  Matching is done by
    /// scanning global names for the `.Lconst.i.` prefix and correlating
    /// via the module-level globals list passed to `lower_function`.
    fn resolve_constant_value(&self, _result: Value, _func: &IrFunction) -> Option<i64> {
        // The constant value is retrieved from the constant_cache
        // populated during the pre-scan phase of `lower_function`.
        self.constant_cache.get(&_result).copied()
    }

    // -----------------------------------------------------------------------
    // Helper: create a MachineInstruction from opcode + operands
    // -----------------------------------------------------------------------

    /// Shorthand: build an instruction with an opcode, optional result, and operands.
    fn mk_inst(
        opcode: X86Opcode,
        result: Option<MachineOperand>,
        operands: &[MachineOperand],
    ) -> MachineInstruction {
        let mut inst = MachineInstruction::new(opcode.as_u32());
        for op in operands {
            inst.operands.push(op.clone());
        }
        inst.result = result;
        inst
    }

    /// Build a terminator instruction.
    fn mk_term(opcode: X86Opcode, operands: &[MachineOperand]) -> MachineInstruction {
        let mut inst = MachineInstruction::new(opcode.as_u32());
        for op in operands {
            inst.operands.push(op.clone());
        }
        inst.is_terminator = true;
        inst.is_branch = matches!(
            opcode,
            X86Opcode::Jmp
                | X86Opcode::Je
                | X86Opcode::Jne
                | X86Opcode::Jl
                | X86Opcode::Jle
                | X86Opcode::Jg
                | X86Opcode::Jge
                | X86Opcode::Jb
                | X86Opcode::Jbe
                | X86Opcode::Ja
                | X86Opcode::Jae
                | X86Opcode::Js
                | X86Opcode::Jns
        );
        inst
    }

    /// Determine operand size in bytes from an IR type.
    #[allow(dead_code)]
    fn operand_size(ty: &IrType, target: &Target) -> u8 {
        ty.size_bytes(target) as u8
    }

    /// Choose the appropriate MOV opcode for a given IR type: GPR or SSE.
    fn mov_opcode(ty: &IrType) -> X86Opcode {
        if ty.is_float() {
            match ty {
                IrType::F64 | IrType::F80 => X86Opcode::Movsd,
                IrType::F32 => X86Opcode::Movss,
                _ => X86Opcode::Movsd,
            }
        } else {
            X86Opcode::Mov
        }
    }

    // ===================================================================
    // compute_frame_layout
    // ===================================================================

    /// Compute the stack frame layout for a function.
    ///
    /// Scans all entry-block allocas, assigns RBP-relative offsets, and
    /// determines whether the function is a leaf (no calls) or not.
    pub fn compute_frame_layout(&mut self, func: &IrFunction) -> FrameLayout {
        let target = &self.target;
        let mut alloca_offsets: FxHashMap<Value, i32> = FxHashMap::default();
        let mut current_offset: i64 = 0; // bytes below RBP (grows negative)
        let mut has_calls = false;

        // Scan all blocks for CALL instructions.
        for block in func.blocks() {
            for inst in block.instructions() {
                if let Instruction::Call { .. } = inst {
                    has_calls = true;
                }
            }
        }

        // Scan entry block for alloca instructions and assign offsets.
        let entry = func.entry_block();
        for inst in entry.instructions() {
            if let Instruction::Alloca {
                result,
                ty,
                alignment,
                ..
            } = inst
            {
                let size = ty.size_bytes(target).max(1) as i64;
                let align = alignment.unwrap_or_else(|| ty.align_bytes(target)) as i64;
                // Align the offset downward (stack grows down).
                current_offset -= size;
                if align > 1 {
                    current_offset &= !(align - 1);
                }
                alloca_offsets.insert(*result, current_offset as i32);
            }
        }

        // Determine callee-saved registers that may be needed.
        // Initially we record all of them; after register allocation, this
        // list is refined to only those actually used.
        let callee_saved: Vec<u16> = if has_calls {
            CALLEE_SAVED_GPRS.to_vec()
        } else {
            Vec::new()
        };

        // Spill area starts after local allocas.
        let spill_area_offset = current_offset as i32;

        // Total frame size (rounded up to 16-byte alignment).
        let raw_size = (-current_offset) as usize;
        let total_size = if has_calls || raw_size > RED_ZONE_SIZE {
            // Must allocate frame — align to 16 bytes.
            (raw_size + 15) & !15
        } else if raw_size == 0 {
            0
        } else {
            // Leaf function with small frame can use red zone.
            // No SUB RSP needed, but track the logical size.
            raw_size
        };

        FrameLayout {
            total_size,
            alloca_offsets,
            spill_area_offset,
            has_calls,
            callee_saved,
        }
    }

    // ===================================================================
    // emit_prologue
    // ===================================================================

    /// Generate function prologue machine instructions.
    ///
    /// Standard x86-64 prologue:
    /// ```asm
    /// push rbp
    /// mov rbp, rsp
    /// sub rsp, <frame_size>   ; if frame_size > 0
    /// ; save callee-saved registers used by the function
    /// ```
    pub fn emit_prologue(&self, mf: &MachineFunction) -> Vec<MachineInstruction> {
        let mut prologue = Vec::new();
        let frame_size = mf.frame_size;

        // push rbp
        prologue.push(Self::mk_inst(
            X86Opcode::Push,
            None,
            &[MachineOperand::Register(RBP)],
        ));

        // mov rbp, rsp
        prologue.push(Self::mk_inst(
            X86Opcode::Mov,
            Some(MachineOperand::Register(RBP)),
            &[MachineOperand::Register(RSP)],
        ));

        // sub rsp, <frame_size>  (only if non-zero and not pure red-zone)
        if frame_size > 0 && (!mf.is_leaf || frame_size > RED_ZONE_SIZE) {
            // Stack probe: if frame_size > 4096, emit a probe loop.
            if frame_size > 4096 {
                // Probe loop: touch each page by moving RSP one page at a time.
                // for pages = frame_size / 4096 .. 1:
                //   sub rsp, 4096
                //   test [rsp], rsp   (touch the page)
                // sub rsp, frame_size % 4096
                let full_pages = frame_size / 4096;
                let remainder = frame_size % 4096;

                for _page in 0..full_pages {
                    prologue.push(Self::mk_inst(
                        X86Opcode::Sub,
                        Some(MachineOperand::Register(RSP)),
                        &[
                            MachineOperand::Register(RSP),
                            MachineOperand::Immediate(4096),
                        ],
                    ));
                    // test dword [rsp], 0  — touch the guard page
                    prologue.push(Self::mk_inst(
                        X86Opcode::Test,
                        None,
                        &[
                            MachineOperand::Memory {
                                base: Some(RSP),
                                index: None,
                                scale: 1,
                                displacement: 0,
                            },
                            MachineOperand::Immediate(0),
                        ],
                    ));
                }
                if remainder > 0 {
                    prologue.push(Self::mk_inst(
                        X86Opcode::Sub,
                        Some(MachineOperand::Register(RSP)),
                        &[
                            MachineOperand::Register(RSP),
                            MachineOperand::Immediate(remainder as i64),
                        ],
                    ));
                }
            } else {
                prologue.push(Self::mk_inst(
                    X86Opcode::Sub,
                    Some(MachineOperand::Register(RSP)),
                    &[
                        MachineOperand::Register(RSP),
                        MachineOperand::Immediate(frame_size as i64),
                    ],
                ));
            }
        }

        // Save callee-saved registers.
        for &reg in &mf.callee_saved_regs {
            prologue.push(Self::mk_inst(
                X86Opcode::Push,
                None,
                &[MachineOperand::Register(reg)],
            ));
        }

        prologue
    }

    // ===================================================================
    // emit_epilogue
    // ===================================================================

    /// Generate function epilogue machine instructions.
    ///
    /// Standard x86-64 epilogue:
    /// ```asm
    /// ; restore callee-saved registers (reverse order)
    /// mov rsp, rbp    ; or `leave`
    /// pop rbp
    /// ret
    /// ```
    pub fn emit_epilogue(&self, mf: &MachineFunction) -> Vec<MachineInstruction> {
        let mut epilogue = Vec::new();

        // Restore callee-saved registers in reverse order.
        for &reg in mf.callee_saved_regs.iter().rev() {
            epilogue.push(Self::mk_inst(
                X86Opcode::Pop,
                Some(MachineOperand::Register(reg)),
                &[],
            ));
        }

        // leave (mov rsp, rbp ; pop rbp)
        epilogue.push(Self::mk_inst(X86Opcode::Leave, None, &[]));

        // ret
        let mut ret_inst = Self::mk_inst(X86Opcode::Ret, None, &[]);
        ret_inst.is_terminator = true;
        epilogue.push(ret_inst);

        epilogue
    }

    // ===================================================================
    // lower — top-level entry point for lowering an IR module
    // ===================================================================

    /// Lower an entire IR function to a [`MachineFunction`].
    ///
    /// This is the top-level orchestrator that:
    /// 1. Computes the frame layout.
    /// 2. Maps parameters to their ABI locations.
    /// 3. Lowers each IR basic block via [`select_instruction`].
    /// 4. Emits prologue/epilogue.
    /// 5. Returns the finished `MachineFunction`.
    pub fn lower(
        &mut self,
        func: &IrFunction,
        diag: &mut DiagnosticEngine,
        globals: &[crate::ir::module::GlobalVariable],
    ) -> Result<MachineFunction, String> {
        self.lower_function(func, diag, globals)
    }

    // ===================================================================
    // lower_function
    // ===================================================================

    /// Lower an IR function to machine instructions.
    pub fn lower_function(
        &mut self,
        func: &IrFunction,
        diag: &mut DiagnosticEngine,
        globals: &[crate::ir::module::GlobalVariable],
    ) -> Result<MachineFunction, String> {
        // Reset per-function state.
        self.next_vreg = 0;
        self.value_map = FxHashMap::default();
        self.value_types = FxHashMap::default();
        self.block_map = FxHashMap::default();
        self.frame = None;
        self.constant_cache = FxHashMap::default();
        self.float_constant_cache = FxHashMap::default();

        // If this is only a declaration (no body), return an empty MachineFunction.
        if !func.is_definition {
            return Ok(MachineFunction::new(func.name.clone()));
        }

        // Pre-populate the constant cache from the module globals so that
        // constant-sentinel BinOp instructions can be resolved to immediates.
        self.populate_constant_cache(func, globals);

                // 1. Compute frame layout.
        let frame = self.compute_frame_layout(func);
        self.frame = Some(frame);

        // 2. Map function parameters to ABI locations.
        let param_types: Vec<IrType> = func.params.iter().map(|p| p.ty.clone()).collect();
        let _arg_locations = self.abi.classify_arguments(&param_types);

        // Create virtual registers for each parameter and record in value_map.
        // Also populate value_types so call-site ABI classification can route
        // FP arguments to SSE registers (XMM0–XMM7) rather than GPRs.
        //
        // Additionally, emit MOV instructions from ABI parameter registers
        // (RDI, RSI, RDX, RCX, R8, R9 for integer; XMM0-7 for FP) into
        // the virtual registers, so the register allocator can properly
        // track parameter live ranges.
        let mut param_insts: Vec<MachineInstruction> = Vec::new();
        let mut gpr_idx: usize = 0;
        let mut sse_idx: usize = 0;
        for param in &func.params {
            let vreg = self.new_vreg();
            self.set_value(param.value, vreg.clone());
            self.value_types.insert(param.value, param.ty.clone());

            // Determine whether the parameter is floating-point.
            let is_fp = matches!(param.ty, IrType::F32 | IrType::F64 | IrType::F80);

            if is_fp && sse_idx < SSE_ARG_REGS.len() {
                let abi_reg = SSE_ARG_REGS[sse_idx];
                param_insts.push(Self::mk_inst(
                    X86Opcode::Movsd,
                    Some(vreg),
                    &[MachineOperand::Register(abi_reg)],
                ));
                sse_idx += 1;
            } else if !is_fp && gpr_idx < INTEGER_ARG_REGS.len() {
                let abi_reg = INTEGER_ARG_REGS[gpr_idx];
                param_insts.push(Self::mk_inst(
                    X86Opcode::Mov,
                    Some(vreg),
                    &[MachineOperand::Register(abi_reg)],
                ));
                gpr_idx += 1;
            }
            // Stack parameters would need memory operands — handled later if needed.
        }

        // 3. Create MachineFunction and build block map.
        let mut mf = MachineFunction::new(func.name.clone());

        // Pre-create machine blocks for all IR blocks.
        // blocks[0] is already created by MachineFunction::new.
        self.block_map.insert(0, 0);
        for i in 1..func.block_count() {
            let label = func.blocks[i]
                .label
                .clone()
                .unwrap_or_else(|| format!("bb{}", i));
            let mbb = MachineBasicBlock::with_label(label);
            let idx = mf.add_block(mbb);
            self.block_map.insert(i, idx);
        }

        // 3.5  Pre-populate value_types for all IR instructions that carry a
        //       result type.  This allows `select_call()` to look up the
        //       IrType of each argument value and route F32/F64/F80 arguments
        //       to SSE registers (XMM0–XMM7) instead of integer GPRs.
        for ir_block in func.blocks().iter() {
            for ir_inst in ir_block.instructions() {
                match ir_inst {
                    Instruction::Load { result, ty, .. } => {
                        self.value_types.insert(*result, ty.clone());
                    }
                    Instruction::BinOp { result, ty, .. } => {
                        self.value_types.insert(*result, ty.clone());
                    }
                    Instruction::BitCast {
                        result, to_type, ..
                    }
                    | Instruction::Trunc {
                        result, to_type, ..
                    }
                    | Instruction::ZExt {
                        result, to_type, ..
                    }
                    | Instruction::SExt {
                        result, to_type, ..
                    }
                    | Instruction::PtrToInt {
                        result, to_type, ..
                    } => {
                        self.value_types.insert(*result, to_type.clone());
                    }
                    Instruction::IntToPtr { result, .. } => {
                        // IntToPtr always produces a pointer type.
                        self.value_types.insert(*result, IrType::Ptr);
                    }
                    Instruction::Call {
                        result,
                        return_type,
                        ..
                    } => {
                        self.value_types.insert(*result, return_type.clone());
                    }
                    Instruction::Alloca { result, ty, .. } => {
                        self.value_types.insert(*result, IrType::Ptr);
                        let _ = ty;
                    }
                    Instruction::ICmp { result, .. } => {
                        self.value_types.insert(*result, IrType::I1);
                    }
                    Instruction::FCmp { result, .. } => {
                        self.value_types.insert(*result, IrType::I1);
                    }
                    Instruction::Phi { result, ty, .. } => {
                        self.value_types.insert(*result, ty.clone());
                    }
                    Instruction::GetElementPtr { result, .. } => {
                        self.value_types.insert(*result, IrType::Ptr);
                    }
                    _ => {}
                }
            }
        }

        // 3.6  Inject parameter-loading MOV instructions into the entry block.
        //       These move values from ABI parameter registers (RDI, RSI, ...)
        //       into the virtual registers created for each IR parameter Value.
        for inst in param_insts {
            mf.blocks[0].push_instruction(inst);
        }

        // 4. Lower each IR basic block.
        for (ir_idx, ir_block) in func.blocks().iter().enumerate() {
            let mach_idx = *self.block_map.get(&ir_idx).unwrap_or(&0);
            let mut instructions: Vec<MachineInstruction> = Vec::new();

            for ir_inst in ir_block.instructions() {
                let mut selected = self.select_instruction(ir_inst, func, diag);
                instructions.append(&mut selected);
            }

            // Append instructions to the corresponding machine block.
            for inst in instructions {
                mf.blocks[mach_idx].push_instruction(inst);
            }
        }

        // 5. Finalize frame info.
        if let Some(ref frame) = self.frame {
            mf.frame_size = frame.total_size;
            mf.is_leaf = !frame.has_calls;
            mf.callee_saved_regs = frame.callee_saved.clone();
        }

        // Mark calls.
        if self.frame.as_ref().map_or(false, |f| f.has_calls) {
            mf.mark_has_calls();
        }

        // 6. Build the reverse mapping (vreg → IR Value) so that
        // apply_allocation_result can correctly resolve VirtualRegister
        // operands to physical registers via the register allocator's
        // IR Value-indexed assignment table.
        //
        // Use vreg_ir_map (which accumulates ALL vreg→Value associations,
        // even when a Value is defined in multiple blocks after phi
        // elimination) instead of value_map (which only stores the LAST
        // operand for each Value and loses earlier vregs).
        for (&vreg, &ir_val) in &self.vreg_ir_map {
            mf.vreg_to_ir_value.insert(vreg, ir_val);
        }

        Ok(mf)
    }

    // ===================================================================
    // select_instruction — the core instruction selector
    // ===================================================================

    /// Select x86-64 machine instructions for a single IR instruction.
    ///
    /// Returns a vector of machine instructions because some IR
    /// instructions expand to multiple x86-64 instructions (e.g., IDIV
    /// requires CQO + IDIV, shifts need MOV to CL first).
    pub fn select_instruction(
        &mut self,
        inst: &Instruction,
        func: &IrFunction,
        diag: &mut DiagnosticEngine,
    ) -> Vec<MachineInstruction> {
        match inst {
            // ---------------------------------------------------------------
            // Alloca — no codegen; offsets tracked in FrameLayout
            // ---------------------------------------------------------------
            Instruction::Alloca { result, ty: _, .. } => {
                // Retrieve the RBP-relative offset from the frame layout.
                if let Some(ref frame) = self.frame {
                    if let Some(&offset) = frame.alloca_offsets.get(result) {
                        // LEA vreg, [rbp + offset]
                        let dst = self.new_vreg();
                        self.set_value(*result, dst.clone());
                        let inst = Self::mk_inst(
                            X86Opcode::Lea,
                            Some(dst),
                            &[MachineOperand::Memory {
                                base: Some(RBP),
                                index: None,
                                scale: 1,
                                displacement: offset as i64,
                            }],
                        );
                        return vec![inst];
                    }
                }
                // Fallback: create a virtual register pointing to a stack slot.
                let dst = self.new_vreg();
                self.set_value(*result, dst);
                Vec::new()
            }

            // ---------------------------------------------------------------
            // Load
            // ---------------------------------------------------------------
            Instruction::Load {
                result, ptr, ty, ..
            } => {
                let src = self.get_value(*ptr);
                let dst = self.new_vreg();
                self.set_value(*result, dst.clone());

                // If the pointer is already a Memory or FrameSlot,
                // use a regular Mov with that operand.
                match &src {
                    MachineOperand::Memory { .. } | MachineOperand::FrameSlot(_) => {
                        let opcode = Self::mov_opcode(ty);
                        vec![Self::mk_inst(opcode, Some(dst), &[src])]
                    }
                    MachineOperand::GlobalSymbol(name) => {
                        // Global variable access: emit RIP-relative load.
                        // On x86-64, this becomes `mov dst, [rip + sym]` with
                        // a relocation for the symbol.
                        let mem = MachineOperand::GlobalSymbol(name.clone());
                        let opcode = Self::mov_opcode(ty);
                        vec![Self::mk_inst(opcode, Some(dst), &[mem])]
                    }
                    _ => {
                        // Pointer is in a register (virtual or physical).
                        // Use size-specific LoadInd to ensure correct memory
                        // access width: I8 → byte load, I16 → word, I32 → dword,
                        // I64/Ptr → qword.
                        let opcode = if ty.is_float() {
                            match ty {
                                IrType::F32 => X86Opcode::Movss,
                                _ => X86Opcode::Movsd,
                            }
                        } else {
                            match ty {
                                IrType::I1 | IrType::I8 => X86Opcode::LoadInd8,
                                IrType::I16 => X86Opcode::LoadInd16,
                                IrType::I32 => X86Opcode::LoadInd32,
                                _ => X86Opcode::LoadInd,
                            }
                        };
                        vec![Self::mk_inst(opcode, Some(dst), &[src])]
                    }
                }
            }

            // ---------------------------------------------------------------
            // Store
            // ---------------------------------------------------------------
            Instruction::Store { value, ptr, .. } => {
                let src = self.get_value(*value);
                let dest_ptr = self.get_value(*ptr);

                // If the pointer is already a Memory or FrameSlot,
                // use a regular Mov with that operand as destination.
                match &dest_ptr {
                    MachineOperand::Memory { .. } | MachineOperand::FrameSlot(_) => {
                        vec![Self::mk_inst(X86Opcode::Mov, None, &[dest_ptr, src])]
                    }
                    MachineOperand::GlobalSymbol(name) => {
                        // Global variable store: emit RIP-relative store.
                        // On x86-64, this becomes `mov [rip + sym], src`.
                        // If src is an immediate, materialise it into a
                        // register first so the encoder always gets a
                        // (GlobalSymbol, Register) operand pair.
                        let mem = MachineOperand::GlobalSymbol(name.clone());
                        match &src {
                            MachineOperand::Immediate(_) => {
                                let tmp = self.new_vreg();
                                let mov_imm = Self::mk_inst(
                                    X86Opcode::Mov,
                                    Some(tmp.clone()),
                                    &[src],
                                );
                                let store = Self::mk_inst(
                                    X86Opcode::Mov,
                                    None,
                                    &[mem, tmp],
                                );
                                vec![mov_imm, store]
                            }
                            _ => {
                                vec![Self::mk_inst(X86Opcode::Mov, None, &[mem, src])]
                            }
                        }
                    }
                    _ => {
                        // Pointer is in a register. Use size-specific StoreInd
                        // to match the value type being stored.
                        let store_op = if let Some(vty) = self.value_types.get(value) {
                            if vty.is_float() {
                                match vty {
                                    IrType::F32 => X86Opcode::Movss,
                                    _ => X86Opcode::Movsd,
                                }
                            } else {
                                match vty {
                                    IrType::I1 | IrType::I8 => X86Opcode::StoreInd8,
                                    IrType::I16 => X86Opcode::StoreInd16,
                                    IrType::I32 => X86Opcode::StoreInd32,
                                    _ => X86Opcode::StoreInd,
                                }
                            }
                        } else {
                            X86Opcode::StoreInd
                        };
                        vec![Self::mk_inst(store_op, None, &[dest_ptr, src])]
                    }
                }
            }

            // ---------------------------------------------------------------
            // BinOp — binary arithmetic / logic
            // ---------------------------------------------------------------
            Instruction::BinOp {
                result,
                op,
                lhs,
                rhs,
                ty,
                ..
            } => {
                // Recognise the constant-sentinel pattern emitted by
                // `emit_int_const` / `emit_float_const` in the IR lowering
                // phase.  Integer sentinels use `Add result, UNDEF`;
                // float sentinels use `FAdd result, UNDEF`.
                if *rhs == Value::UNDEF && *lhs == *result {
                    // Check float constant cache first — float constants
                    // must be loaded via RIP-relative movsd/movss from
                    // the .rodata constant pool.
                    if let Some(sym_name) = self.float_constant_cache.get(result).cloned() {
                        let dst = self.new_vreg();
                        self.set_value(*result, dst.clone());
                        let mov_op = if matches!(ty, IrType::F32) {
                            X86Opcode::Movss
                        } else {
                            X86Opcode::Movsd
                        };
                        return vec![Self::mk_inst(
                            mov_op,
                            Some(dst),
                            &[MachineOperand::GlobalSymbol(sym_name)],
                        )];
                    }
                    // Then check integer constant cache.
                    if let Some(imm) = self.resolve_constant_value(*result, func) {
                        let dst = self.new_vreg();
                        self.set_value(*result, dst.clone());
                        return vec![Self::mk_inst(
                            X86Opcode::Mov,
                            Some(dst),
                            &[MachineOperand::Immediate(imm)],
                        )];
                    }
                }
                self.select_binop(*result, *op, *lhs, *rhs, ty)
            }

            // ---------------------------------------------------------------
            // ICmp — integer comparison
            // ---------------------------------------------------------------
            Instruction::ICmp {
                result,
                op,
                lhs,
                rhs,
                ..
            } => self.select_icmp(*result, *op, *lhs, *rhs),

            // ---------------------------------------------------------------
            // FCmp — floating-point comparison
            // ---------------------------------------------------------------
            Instruction::FCmp {
                result,
                op,
                lhs,
                rhs,
                ..
            } => self.select_fcmp(*result, *op, *lhs, *rhs),

            // ---------------------------------------------------------------
            // Branch (unconditional)
            // ---------------------------------------------------------------
            Instruction::Branch { target, .. } => {
                let block_idx = target.index();
                let mach_idx = *self.block_map.get(&block_idx).unwrap_or(&0);
                vec![Self::mk_term(
                    X86Opcode::Jmp,
                    &[MachineOperand::BlockLabel(mach_idx as u32)],
                )]
            }

            // ---------------------------------------------------------------
            // CondBranch
            // ---------------------------------------------------------------
            Instruction::CondBranch {
                condition,
                then_block,
                else_block,
                ..
            } => {
                let cond = self.get_value(*condition);
                let then_idx = *self.block_map.get(&then_block.index()).unwrap_or(&0);
                let else_idx = *self.block_map.get(&else_block.index()).unwrap_or(&0);

                // If the condition is a known constant (from constant folding),
                // emit an unconditional jump instead of TEST + JNE + JMP.
                // This avoids the encoder seeing TEST Imm, Imm which is
                // not a valid x86 instruction pattern.
                if let MachineOperand::Immediate(imm) = &cond {
                    let target = if *imm != 0 { then_idx } else { else_idx };
                    return vec![Self::mk_term(
                        X86Opcode::Jmp,
                        &[MachineOperand::BlockLabel(target as u32)],
                    )];
                }

                let mut out = Vec::new();
                // test cond, cond
                out.push(Self::mk_inst(X86Opcode::Test, None, &[cond.clone(), cond]));
                // jne then_block
                let mut jne = Self::mk_term(
                    X86Opcode::Jne,
                    &[MachineOperand::BlockLabel(then_idx as u32)],
                );
                jne.is_terminator = false; // Not the final terminator yet
                out.push(jne);
                // jmp else_block (fallthrough to else)
                out.push(Self::mk_term(
                    X86Opcode::Jmp,
                    &[MachineOperand::BlockLabel(else_idx as u32)],
                ));
                out
            }

            // ---------------------------------------------------------------
            // Switch — multi-way branch
            // ---------------------------------------------------------------
            Instruction::Switch {
                value,
                default,
                cases,
                ..
            } => {
                let val = self.get_value(*value);
                let def_idx = *self.block_map.get(&default.index()).unwrap_or(&0);

                let mut out = Vec::new();
                // Cascaded comparisons (simple implementation).
                for (case_val, target_block) in cases {
                    let tgt_idx = *self.block_map.get(&target_block.index()).unwrap_or(&0);
                    // cmp val, case_val
                    out.push(Self::mk_inst(
                        X86Opcode::Cmp,
                        None,
                        &[val.clone(), MachineOperand::Immediate(*case_val)],
                    ));
                    // je target_block
                    let mut je = Self::mk_inst(
                        X86Opcode::Je,
                        None,
                        &[MachineOperand::BlockLabel(tgt_idx as u32)],
                    );
                    je.is_branch = true;
                    out.push(je);
                }
                // jmp default
                out.push(Self::mk_term(
                    X86Opcode::Jmp,
                    &[MachineOperand::BlockLabel(def_idx as u32)],
                ));
                out
            }

            // ---------------------------------------------------------------
            // Call
            // ---------------------------------------------------------------
            Instruction::Call {
                result,
                callee,
                args,
                return_type,
                ..
            } => self.select_call(*result, *callee, args, return_type, func),

            // ---------------------------------------------------------------
            // Return
            // ---------------------------------------------------------------
            Instruction::Return { value, .. } => {
                let mut out = Vec::new();

                if let Some(val) = value {
                    let src = self.get_value(*val);
                    // Determine if the return type is FP or integer.
                    let ret_loc = self.abi.classify_return(&func.return_type);
                    match ret_loc {
                        RetLocation::Register(reg) => {
                            let opcode = if registers::is_sse(reg) {
                                X86Opcode::Movsd
                            } else {
                                X86Opcode::Mov
                            };
                            out.push(Self::mk_inst(
                                opcode,
                                Some(MachineOperand::Register(reg)),
                                &[src],
                            ));
                        }
                        RetLocation::RegisterPair(lo, _hi) => {
                            // For 128-bit returns: move low part to lo, high to hi.
                            out.push(Self::mk_inst(
                                X86Opcode::Mov,
                                Some(MachineOperand::Register(lo)),
                                &[src],
                            ));
                        }
                        RetLocation::Indirect => {
                            // Return via hidden pointer (already handled in call setup).
                        }
                        RetLocation::Void => {}
                    }
                }

                let mut ret = Self::mk_inst(X86Opcode::Ret, None, &[]);
                ret.is_terminator = true;
                out.push(ret);
                out
            }

            // ---------------------------------------------------------------
            // Phi — should be eliminated before codegen (Phase 9)
            // ---------------------------------------------------------------
            Instruction::Phi {
                result,
                ty: _,
                incoming: _,
                ..
            } => {
                // Phi nodes should have been eliminated by phi_eliminate.rs.
                // If encountered, emit a diagnostic warning and create a
                // virtual register as a placeholder.
                diag.emit_warning(
                    Span::dummy(),
                    format!("Phi node encountered during x86-64 codegen for %{} — should be eliminated by Phase 9", result.index()),
                );
                let dst = self.new_vreg();
                self.set_value(*result, dst);
                Vec::new()
            }

            // ---------------------------------------------------------------
            // GetElementPtr — address computation
            // ---------------------------------------------------------------
            Instruction::GetElementPtr {
                result,
                base,
                indices,
                result_type,
                ..
            } => {
                let base_op = self.get_value(*base);
                let base_op_saved = base_op.clone();
                let dst = self.new_vreg();
                self.set_value(*result, dst.clone());

                let mut out = Vec::new();
                if indices.is_empty() {
                    // When the base is a global symbol, use LEA to load
                    // the address rather than MOV which loads the value
                    // at that address. GEP computes an address.
                    if matches!(&base_op, MachineOperand::GlobalSymbol(_)) {
                        out.push(Self::mk_inst(X86Opcode::Lea, Some(dst), &[base_op]));
                    } else {
                        out.push(Self::mk_inst(X86Opcode::Mov, Some(dst), &[base_op]));
                    }
                } else {
                    // Compute the element size from the result type.
                    // result_type is the pointed-to element type; its size_bytes
                    // gives us the stride for each index.
                    let elem_size = result_type.size_bytes(&Target::X86_64).max(1) as i64;

                    // Start with the base pointer. For global symbols use
                    // LEA (load effective address) since GEP needs the
                    // address of the global, not its contents.
                    if matches!(&base_op, MachineOperand::GlobalSymbol(_)) {
                        out.push(Self::mk_inst(X86Opcode::Lea, Some(dst.clone()), &[base_op]));
                    } else {
                        out.push(Self::mk_inst(X86Opcode::Mov, Some(dst.clone()), &[base_op]));
                    }

                    for idx_val in indices {
                        // Check constant cache first — compile-time constant
                        // indices can be folded into immediate offsets.
                        let const_idx = self.constant_cache.get(idx_val).copied();
                        if let Some(cidx) = const_idx {
                            let byte_offset = cidx * elem_size;
                            if byte_offset != 0 {
                                out.push(Self::mk_inst(
                                    X86Opcode::Add,
                                    Some(dst.clone()),
                                    &[dst.clone(), MachineOperand::Immediate(byte_offset)],
                                ));
                            }
                            // If offset is 0, no instruction needed.
                            continue;
                        }

                        let idx_op = self.get_value(*idx_val);

                        // Check if index is an immediate — can compute offset at compile time
                        if let MachineOperand::Immediate(imm) = &idx_op {
                            let byte_offset = imm * elem_size;
                            if byte_offset != 0 {
                                out.push(Self::mk_inst(
                                    X86Opcode::Add,
                                    Some(dst.clone()),
                                    &[dst.clone(), MachineOperand::Immediate(byte_offset)],
                                ));
                            }
                            // If offset is 0, no instruction needed.
                        } else if elem_size == 1 {
                            // Element size is 1 byte — no scaling needed.
                            out.push(Self::mk_inst(
                                X86Opcode::Add,
                                Some(dst.clone()),
                                &[dst.clone(), idx_op],
                            ));
                        } else {
                            // Register index with non-unity element size.
                            // Use three-operand IMUL writing to the GEP
                            // result vreg (which IS allocated by the
                            // register allocator) to avoid creating
                            // untracked intermediate virtual registers.
                            //
                            // Save the current base to a temp position,
                            // compute scaled index into dst, then add
                            // the saved base back.
                            //
                            // Pattern:
                            //   IMUL dst, idx_op, elem_size  (3-op)
                            //   ADD  dst, base_saved
                            //
                            // Here base_saved is the base that was
                            // previously MOV'd into dst at the start
                            // of the GEP expansion. We re-fetch it:
                            // since base_op was set at the top, we can
                            // use the original base_op which is still
                            // live in its assigned register.
                            {
                                // Three-operand IMUL: dst = idx * size
                                // This writes into the GEP result vreg.
                                out.push(Self::mk_inst(
                                    X86Opcode::Imul,
                                    Some(dst.clone()),
                                    &[idx_op, MachineOperand::Immediate(elem_size)],
                                ));
                                // ADD dst, base — add the original
                                // base address (still in its register).
                                out.push(Self::mk_inst(
                                    X86Opcode::Add,
                                    Some(dst.clone()),
                                    &[dst.clone(), base_op_saved.clone()],
                                ));
                            }
                        }
                    }
                }
                out
            }

            // ---------------------------------------------------------------
            // BitCast — reinterpretation / phi-copy
            // ---------------------------------------------------------------
            Instruction::BitCast {
                result,
                value,
                to_type,
                ..
            } => {
                let src = self.get_value(*value);
                // If this result already has a VR assigned (e.g. from a
                // phi-copy in a different predecessor block), reuse the
                // same VR so all predecessor paths write to the same
                // virtual register and the register allocator assigns a
                // single physical location.
                let dst = if let Some(existing) = self.value_map.get(result) {
                    existing.clone()
                } else {
                    let vr = self.new_vreg();
                    self.set_value(*result, vr.clone());
                    vr
                };
                // Use MOV for same-class, MOVD/MOVQ for int↔float.
                let opcode = if to_type.is_float() {
                    X86Opcode::Movsd
                } else {
                    X86Opcode::Mov
                };
                vec![Self::mk_inst(opcode, Some(dst), &[src])]
            }

            // ---------------------------------------------------------------
            // Trunc — narrow an integer
            // ---------------------------------------------------------------
            Instruction::Trunc {
                result,
                value,
                to_type,
                ..
            } => {
                let src = self.get_value(*value);
                let dst = self.new_vreg();
                self.set_value(*result, dst.clone());
                // On x86-64, truncation to 32-bit uses a 32-bit MOV which
                // implicitly zero-extends the upper 32 bits.  For smaller
                // widths, use MOVZX from the appropriate sub-register size.
                let opcode = match to_type {
                    IrType::I32 => X86Opcode::Mov,
                    IrType::I16 | IrType::I8 | IrType::I1 => X86Opcode::Mov,
                    _ => X86Opcode::Mov,
                };
                vec![Self::mk_inst(opcode, Some(dst), &[src])]
            }

            // ---------------------------------------------------------------
            // ZExt — zero extend
            // ---------------------------------------------------------------
            Instruction::ZExt {
                result,
                value,
                to_type: _,
                ..
            } => {
                let src = self.get_value(*value);
                let dst = self.new_vreg();
                self.set_value(*result, dst.clone());
                // MOVZX for 8→32, 16→32.  For 32→64, a 32-bit MOV
                // implicitly zero-extends on x86-64.
                let opcode = X86Opcode::MovZX;
                vec![Self::mk_inst(opcode, Some(dst), &[src])]
            }

            // ---------------------------------------------------------------
            // SExt — sign extend
            // ---------------------------------------------------------------
            Instruction::SExt {
                result,
                value,
                to_type: _,
                ..
            } => {
                let src = self.get_value(*value);
                let dst = self.new_vreg();
                self.set_value(*result, dst.clone());
                vec![Self::mk_inst(X86Opcode::MovSX, Some(dst), &[src])]
            }

            // ---------------------------------------------------------------
            // IntToPtr — integer → pointer (on x86-64, pointer = i64)
            // ---------------------------------------------------------------
            Instruction::IntToPtr { result, value, .. } => {
                let src = self.get_value(*value);
                let dst = self.new_vreg();
                self.set_value(*result, dst.clone());
                vec![Self::mk_inst(X86Opcode::Mov, Some(dst), &[src])]
            }

            // ---------------------------------------------------------------
            // PtrToInt — pointer → integer
            // ---------------------------------------------------------------
            Instruction::PtrToInt {
                result,
                value,
                to_type: _,
                ..
            } => {
                let src = self.get_value(*value);
                let dst = self.new_vreg();
                self.set_value(*result, dst.clone());
                vec![Self::mk_inst(X86Opcode::Mov, Some(dst), &[src])]
            }

            // ---------------------------------------------------------------
            // InlineAsm
            // ---------------------------------------------------------------
            Instruction::InlineAsm {
                result,
                template: _,
                constraints: _,
                operands,
                clobbers: _,
                has_side_effects,
                is_volatile,
                goto_targets: _,
                ..
            } => {
                let dst = self.new_vreg();
                self.set_value(*result, dst.clone());

                let mut asm_operands = Vec::new();
                // Bind input operands.
                for op_val in operands {
                    asm_operands.push(self.get_value(*op_val));
                }

                let mut inst = MachineInstruction::new(X86Opcode::InlineAsm.as_u32());
                inst.result = Some(dst);
                for op in &asm_operands {
                    inst.operands.push(op.clone());
                }
                // Mark as having side effects to prevent elimination.
                if *has_side_effects || *is_volatile {
                    inst.is_call = true; // Conservatively treat as call-like.
                }
                vec![inst]
            }
        }
    }

    // ===================================================================
    // Binary operation selection
    // ===================================================================

    fn select_binop(
        &mut self,
        result: Value,
        op: BinOp,
        lhs: Value,
        rhs: Value,
        ty: &IrType,
    ) -> Vec<MachineInstruction> {
        let lhs_op = self.get_value(lhs);
        let rhs_op = self.get_value(rhs);
        let dst = self.new_vreg();
        self.set_value(result, dst.clone());
        let mut out = Vec::new();

        match op {
            // -- Integer arithmetic --
            BinOp::Add => {
                // x86 two-address form: dst = lhs + rhs
                // 1. MOV dst, lhs     (dst now holds the lhs value)
                // 2. ADD dst, rhs     (dst += rhs)
                // NOTE: The encoder normalizes result into operands[0],
                // so we provide only rhs as the explicit operand.
                out.push(Self::mk_inst(X86Opcode::Mov, Some(dst.clone()), &[lhs_op]));
                out.push(Self::mk_inst(X86Opcode::Add, Some(dst.clone()), &[rhs_op]));
            }
            BinOp::Sub => {
                // x86 two-address form: dst = lhs - rhs
                out.push(Self::mk_inst(X86Opcode::Mov, Some(dst.clone()), &[lhs_op]));
                out.push(Self::mk_inst(X86Opcode::Sub, Some(dst.clone()), &[rhs_op]));
            }
            BinOp::Mul => {
                // x86 two-address form: dst = lhs * rhs
                // 1. MOV dst, lhs     (dst now holds the lhs value)
                // 2. IMUL dst, rhs    (dst *= rhs)
                out.push(Self::mk_inst(X86Opcode::Mov, Some(dst.clone()), &[lhs_op]));
                out.push(Self::mk_inst(X86Opcode::Imul, Some(dst), &[rhs_op]));
            }
            BinOp::SDiv => {
                // CQO ; IDIV rhs → quotient in RAX
                // mov RAX, lhs
                out.push(Self::mk_inst(
                    X86Opcode::Mov,
                    Some(MachineOperand::Register(RAX)),
                    &[lhs_op],
                ));
                // cqo (sign-extend RAX into RDX:RAX)
                out.push(Self::mk_inst(X86Opcode::Cqo, None, &[]));
                // idiv rhs
                out.push(Self::mk_inst(X86Opcode::Idiv, None, &[rhs_op]));
                // mov dst, RAX (quotient)
                out.push(Self::mk_inst(
                    X86Opcode::Mov,
                    Some(dst),
                    &[MachineOperand::Register(RAX)],
                ));
            }
            BinOp::UDiv => {
                // xor RDX, RDX ; DIV rhs → quotient in RAX
                // Encoder normalizes result into operands[0], so we only
                // need to supply one copy of RDX in operands.
                out.push(Self::mk_inst(
                    X86Opcode::Xor,
                    Some(MachineOperand::Register(RDX)),
                    &[MachineOperand::Register(RDX)],
                ));
                out.push(Self::mk_inst(
                    X86Opcode::Mov,
                    Some(MachineOperand::Register(RAX)),
                    &[lhs_op],
                ));
                out.push(Self::mk_inst(X86Opcode::Div, None, &[rhs_op]));
                out.push(Self::mk_inst(
                    X86Opcode::Mov,
                    Some(dst),
                    &[MachineOperand::Register(RAX)],
                ));
            }
            BinOp::SRem => {
                // CQO ; IDIV rhs → remainder in RDX
                out.push(Self::mk_inst(
                    X86Opcode::Mov,
                    Some(MachineOperand::Register(RAX)),
                    &[lhs_op],
                ));
                out.push(Self::mk_inst(X86Opcode::Cqo, None, &[]));
                out.push(Self::mk_inst(X86Opcode::Idiv, None, &[rhs_op]));
                out.push(Self::mk_inst(
                    X86Opcode::Mov,
                    Some(dst),
                    &[MachineOperand::Register(RDX)],
                ));
            }
            BinOp::URem => {
                out.push(Self::mk_inst(
                    X86Opcode::Xor,
                    Some(MachineOperand::Register(RDX)),
                    &[MachineOperand::Register(RDX)],
                ));
                out.push(Self::mk_inst(
                    X86Opcode::Mov,
                    Some(MachineOperand::Register(RAX)),
                    &[lhs_op],
                ));
                out.push(Self::mk_inst(X86Opcode::Div, None, &[rhs_op]));
                out.push(Self::mk_inst(
                    X86Opcode::Mov,
                    Some(dst),
                    &[MachineOperand::Register(RDX)],
                ));
            }

            // -- Bitwise --
            // NOTE: The encoder normalizes result into operands[0],
            // so we provide only rhs as the explicit operand.
            BinOp::And => {
                out.push(Self::mk_inst(X86Opcode::Mov, Some(dst.clone()), &[lhs_op]));
                out.push(Self::mk_inst(X86Opcode::And, Some(dst), &[rhs_op]));
            }
            BinOp::Or => {
                out.push(Self::mk_inst(X86Opcode::Mov, Some(dst.clone()), &[lhs_op]));
                out.push(Self::mk_inst(X86Opcode::Or, Some(dst), &[rhs_op]));
            }
            BinOp::Xor => {
                out.push(Self::mk_inst(X86Opcode::Mov, Some(dst.clone()), &[lhs_op]));
                out.push(Self::mk_inst(X86Opcode::Xor, Some(dst), &[rhs_op]));
            }

            // -- Shifts (shift amount must be in CL) --
            BinOp::Shl => {
                out.push(Self::mk_inst(X86Opcode::Mov, Some(dst.clone()), &[lhs_op]));
                out.push(Self::mk_inst(
                    X86Opcode::Mov,
                    Some(MachineOperand::Register(RCX)),
                    &[rhs_op],
                ));
                out.push(Self::mk_inst(
                    X86Opcode::Shl,
                    Some(dst),
                    &[MachineOperand::Register(RCX)],
                ));
            }
            BinOp::AShr => {
                out.push(Self::mk_inst(X86Opcode::Mov, Some(dst.clone()), &[lhs_op]));
                out.push(Self::mk_inst(
                    X86Opcode::Mov,
                    Some(MachineOperand::Register(RCX)),
                    &[rhs_op],
                ));
                out.push(Self::mk_inst(
                    X86Opcode::Sar,
                    Some(dst),
                    &[MachineOperand::Register(RCX)],
                ));
            }
            BinOp::LShr => {
                out.push(Self::mk_inst(X86Opcode::Mov, Some(dst.clone()), &[lhs_op]));
                out.push(Self::mk_inst(
                    X86Opcode::Mov,
                    Some(MachineOperand::Register(RCX)),
                    &[rhs_op],
                ));
                out.push(Self::mk_inst(
                    X86Opcode::Shr,
                    Some(dst),
                    &[MachineOperand::Register(RCX)],
                ));
            }

            // -- Floating-point (SSE / SSE2) --
            BinOp::FAdd => {
                let is_f32 = matches!(ty, IrType::F32);
                let (mov_op, arith_op) = if is_f32 {
                    (X86Opcode::Movss, X86Opcode::Addss)
                } else {
                    (X86Opcode::Movsd, X86Opcode::Addsd)
                };
                out.push(Self::mk_inst(mov_op, Some(dst.clone()), &[lhs_op]));
                out.push(Self::mk_inst(arith_op, Some(dst), &[rhs_op]));
            }
            BinOp::FSub => {
                let is_f32 = matches!(ty, IrType::F32);
                let (mov_op, arith_op) = if is_f32 {
                    (X86Opcode::Movss, X86Opcode::Subss)
                } else {
                    (X86Opcode::Movsd, X86Opcode::Subsd)
                };
                out.push(Self::mk_inst(mov_op, Some(dst.clone()), &[lhs_op]));
                out.push(Self::mk_inst(arith_op, Some(dst), &[rhs_op]));
            }
            BinOp::FMul => {
                let is_f32 = matches!(ty, IrType::F32);
                let (mov_op, arith_op) = if is_f32 {
                    (X86Opcode::Movss, X86Opcode::Mulss)
                } else {
                    (X86Opcode::Movsd, X86Opcode::Mulsd)
                };
                out.push(Self::mk_inst(mov_op, Some(dst.clone()), &[lhs_op]));
                out.push(Self::mk_inst(arith_op, Some(dst), &[rhs_op]));
            }
            BinOp::FDiv => {
                let is_f32 = matches!(ty, IrType::F32);
                let (mov_op, arith_op) = if is_f32 {
                    (X86Opcode::Movss, X86Opcode::Divss)
                } else {
                    (X86Opcode::Movsd, X86Opcode::Divsd)
                };
                out.push(Self::mk_inst(mov_op, Some(dst.clone()), &[lhs_op]));
                out.push(Self::mk_inst(arith_op, Some(dst), &[rhs_op]));
            }
            BinOp::FRem => {
                // IEEE 754 FP remainder — no single x86-64 instruction.
                // Emit a call to a software helper (or use fmod-style sequence).
                // For now, generate a placeholder sequence.
                let is_f32 = matches!(ty, IrType::F32);
                let mov_op = if is_f32 {
                    X86Opcode::Movss
                } else {
                    X86Opcode::Movsd
                };
                // Place lhs in XMM0, rhs in XMM1, call __fmod helper.
                out.push(Self::mk_inst(
                    mov_op,
                    Some(MachineOperand::Register(XMM0)),
                    &[lhs_op],
                ));
                out.push(Self::mk_inst(
                    mov_op,
                    Some(MachineOperand::Register(XMM1)),
                    &[rhs_op],
                ));
                let mut call = Self::mk_inst(
                    X86Opcode::Call,
                    None,
                    &[MachineOperand::GlobalSymbol(
                        if is_f32 { "fmodf" } else { "fmod" }.to_string(),
                    )],
                );
                call.is_call = true;
                out.push(call);
                out.push(Self::mk_inst(
                    mov_op,
                    Some(dst),
                    &[MachineOperand::Register(XMM0)],
                ));
            }
        }

        out
    }

    // ===================================================================
    // Integer comparison selection
    // ===================================================================

    fn select_icmp(
        &mut self,
        result: Value,
        op: ICmpOp,
        lhs: Value,
        rhs: Value,
    ) -> Vec<MachineInstruction> {
        let lhs_op = self.get_value(lhs);
        let rhs_op = self.get_value(rhs);
        let dst = self.new_vreg();
        self.set_value(result, dst.clone());

        let setcc = match op {
            ICmpOp::Eq => X86Opcode::Sete,
            ICmpOp::Ne => X86Opcode::Setne,
            ICmpOp::Slt => X86Opcode::Setl,
            ICmpOp::Sle => X86Opcode::Setle,
            ICmpOp::Sgt => X86Opcode::Setg,
            ICmpOp::Sge => X86Opcode::Setge,
            ICmpOp::Ult => X86Opcode::Setb,
            ICmpOp::Ule => X86Opcode::Setbe,
            ICmpOp::Ugt => X86Opcode::Seta,
            ICmpOp::Uge => X86Opcode::Setae,
        };

        vec![
            // cmp lhs, rhs
            Self::mk_inst(X86Opcode::Cmp, None, &[lhs_op, rhs_op]),
            // setcc dst (1-byte result, zero-extended to full register width later)
            Self::mk_inst(setcc, Some(dst.clone()), &[]),
            // movzx dst, dst (zero-extend byte result to full width)
            Self::mk_inst(X86Opcode::MovZX, Some(dst.clone()), &[dst]),
        ]
    }

    // ===================================================================
    // Floating-point comparison selection
    // ===================================================================

    fn select_fcmp(
        &mut self,
        result: Value,
        op: FCmpOp,
        lhs: Value,
        rhs: Value,
    ) -> Vec<MachineInstruction> {
        let lhs_op = self.get_value(lhs);
        let rhs_op = self.get_value(rhs);
        let dst = self.new_vreg();
        self.set_value(result, dst.clone());

        // x86 UCOMISD/UCOMISS sets CF, ZF, PF per the following table:
        //   a > b  (ordered)   → CF=0, ZF=0, PF=0
        //   a < b  (ordered)   → CF=1, ZF=0, PF=0
        //   a == b (ordered)   → CF=0, ZF=1, PF=0
        //   unordered (NaN)    → CF=1, ZF=1, PF=1
        //
        // For **ordered** comparisons, NaN must produce false.
        // For **unordered** comparisons, NaN must produce true.
        //
        // Simple SETcc alone is WRONG for ordered comparisons because the
        // NaN case (PF=1) may coincidentally satisfy the condition flags.
        //
        // Strategy:
        //   Ordered (Oeq, One, Olt, Ole, Ogt, Oge):
        //     UCOMISD lhs, rhs
        //     SETcc   tmp1          (primary condition)
        //     SETNP   tmp2          (PF=0 → not NaN)
        //     AND     dst, tmp1, tmp2  → result is true only if both hold
        //
        //   Unordered NaN checks (Uno, Ord):
        //     UCOMISD lhs, rhs
        //     SETP    dst   (Uno: PF=1 → NaN)
        //     SETNP   dst   (Ord: PF=0 → both are numbers)
        //
        //   For simplicity the "ordered" cases emit a 5-instruction sequence;
        //   the NaN-only checks emit a 3-instruction sequence.

        let mut out = vec![Self::mk_inst(X86Opcode::Ucomisd, None, &[lhs_op, rhs_op])];

        match op {
            FCmpOp::Uno => {
                // Result = PF (NaN → PF=1)
                out.push(Self::mk_inst(X86Opcode::Setp, Some(dst.clone()), &[]));
                out.push(Self::mk_inst(X86Opcode::MovZX, Some(dst.clone()), &[dst]));
            }
            FCmpOp::Ord => {
                // Result = !PF (both ordered → PF=0)
                out.push(Self::mk_inst(X86Opcode::Setnp, Some(dst.clone()), &[]));
                out.push(Self::mk_inst(X86Opcode::MovZX, Some(dst.clone()), &[dst]));
            }
            _ => {
                // Ordered comparisons: need (condition) AND (not-NaN).
                let primary_cc = match op {
                    FCmpOp::Oeq => X86Opcode::Sete,  // ZF=1
                    FCmpOp::One => X86Opcode::Setne, // ZF=0
                    FCmpOp::Olt => X86Opcode::Setb,  // CF=1
                    FCmpOp::Ole => X86Opcode::Setbe, // CF=1 OR ZF=1
                    FCmpOp::Ogt => X86Opcode::Seta,  // CF=0 AND ZF=0
                    FCmpOp::Oge => X86Opcode::Setae, // CF=0
                    FCmpOp::Uno | FCmpOp::Ord => unreachable!(),
                };
                let tmp_parity = self.new_vreg();
                // SETcc  dst  ← primary condition flag
                out.push(Self::mk_inst(primary_cc, Some(dst.clone()), &[]));
                // SETNP  tmp  ← PF=0 means NOT NaN
                out.push(Self::mk_inst(
                    X86Opcode::Setnp,
                    Some(tmp_parity.clone()),
                    &[],
                ));
                // AND dst, tmp  ← result is true only if (condition AND !NaN)
                out.push(Self::mk_inst(
                    X86Opcode::And,
                    Some(dst.clone()),
                    &[dst.clone(), tmp_parity],
                ));
                // Zero-extend to full register width
                out.push(Self::mk_inst(X86Opcode::MovZX, Some(dst.clone()), &[dst]));
            }
        }

        out
    }

    // ===================================================================
    // Call instruction selection
    // ===================================================================

    fn select_call(
        &mut self,
        result: Value,
        callee: Value,
        args: &[Value],
        return_type: &IrType,
        func: &IrFunction,
    ) -> Vec<MachineInstruction> {
        let mut out = Vec::new();
        let mut gpr_idx: usize = 0;
        let mut sse_idx: usize = 0;
        let mut stack_offset: i32 = 0;

        // 1. Place arguments in registers/stack per System V AMD64 ABI.
        for arg_val in args {
            let arg_op = self.get_value(*arg_val);

            // Classify the argument as FP or integer by consulting the
            // value_types map (populated from IR instruction result types).
            // F32, F64, and F80 arguments are routed to XMM0–XMM7 per the
            // System V AMD64 ABI; all other types use integer GPRs.
            let is_fp = self
                .value_types
                .get(arg_val)
                .map(|ty| matches!(ty, IrType::F32 | IrType::F64 | IrType::F80))
                .unwrap_or(false);

            if is_fp && sse_idx < SSE_ARG_REGS.len() {
                let reg = SSE_ARG_REGS[sse_idx];
                out.push(Self::mk_inst(
                    X86Opcode::Movsd,
                    Some(MachineOperand::Register(reg)),
                    &[arg_op],
                ));
                sse_idx += 1;
            } else if !is_fp && gpr_idx < INTEGER_ARG_REGS.len() {
                let reg = INTEGER_ARG_REGS[gpr_idx];
                out.push(Self::mk_inst(
                    X86Opcode::Mov,
                    Some(MachineOperand::Register(reg)),
                    &[arg_op],
                ));
                gpr_idx += 1;
            } else {
                // Stack argument — push in right-to-left order.
                out.push(Self::mk_inst(X86Opcode::Push, None, &[arg_op]));
                stack_offset += 8;
            }
        }

        // 2. For variadic functions, set AL = number of SSE registers used.
        if func.is_variadic {
            out.push(Self::mk_inst(
                X86Opcode::Mov,
                Some(MachineOperand::Register(RAX)),
                &[MachineOperand::Immediate(sse_idx as i64)],
            ));
        }

        // 3. Emit CALL instruction.
        // Resolve callee — check func_ref_names for direct call target.
        let callee_op = if let Some(fname) = self.func_ref_names.get(&callee) {
            MachineOperand::GlobalSymbol(fname.clone())
        } else {
            self.get_value(callee)
        };
        let mut call_inst = Self::mk_inst(X86Opcode::Call, None, &[callee_op]);
        call_inst.is_call = true;
        out.push(call_inst);

        // 4. Clean up stack arguments (if any were pushed).
        if stack_offset > 0 {
            out.push(Self::mk_inst(
                X86Opcode::Add,
                Some(MachineOperand::Register(RSP)),
                &[
                    MachineOperand::Register(RSP),
                    MachineOperand::Immediate(stack_offset as i64),
                ],
            ));
        }

        // 5. Retrieve return value from RAX (integer) or XMM0 (float).
        let dst = self.new_vreg();
        self.set_value(result, dst.clone());

        let ret_loc = self.abi.classify_return(return_type);
        match ret_loc {
            RetLocation::Register(reg) => {
                let opcode = if registers::is_sse(reg) {
                    X86Opcode::Movsd
                } else {
                    X86Opcode::Mov
                };
                out.push(Self::mk_inst(
                    opcode,
                    Some(dst),
                    &[MachineOperand::Register(reg)],
                ));
            }
            RetLocation::RegisterPair(lo, _hi) => {
                out.push(Self::mk_inst(
                    X86Opcode::Mov,
                    Some(dst),
                    &[MachineOperand::Register(lo)],
                ));
            }
            RetLocation::Indirect | RetLocation::Void => {
                // No register-to-move; result is via pointer or void.
            }
        }

        out
    }
}

// =========================================================================
// ArchCodegen trait implementation for X86_64CodeGen
// =========================================================================

impl ArchCodegen for X86_64CodeGen {
    /// Lower an IR function to a MachineFunction.
    fn lower_function(
        &self,
        func: &IrFunction,
        diag: &mut DiagnosticEngine,
        globals: &[crate::ir::module::GlobalVariable],
        func_ref_map: &FxHashMap<Value, String>,
        global_var_refs: &FxHashMap<Value, String>,
    ) -> Result<MachineFunction, String> {
        let mut codegen = Self::new(self.target);
        codegen.set_func_ref_names(func_ref_map.clone());
        codegen.set_global_var_refs(global_var_refs.clone());
        codegen.lower_function_impl(func, diag, globals)
    }

    /// Emit machine code bytes from a MachineFunction.
    fn emit_assembly(
        &self,
        mf: &MachineFunction,
    ) -> Result<crate::backend::traits::AssembledFunction, String> {
        use crate::backend::x86_64::assembler::encoder::X86_64Encoder;

        let mut encoder = X86_64Encoder::new(0);
        let mut bytes = Vec::new();

        for block in &mf.blocks {
            for inst in &block.instructions {
                let encoded = encoder.encode_instruction(inst);
                bytes.extend_from_slice(&encoded.bytes);
            }
        }
        Ok(crate::backend::traits::AssembledFunction {
            bytes,
            relocations: Vec::new(),
        })
    }

    /// Return the target architecture.
    fn target(&self) -> Target {
        self.target
    }

    /// Return register allocation metadata for x86-64.
    fn register_info(&self) -> RegisterInfo {
        self.abi.x86_64_register_info()
    }

    /// Return supported x86-64 relocation types.
    ///
    /// These match the ELF AMD64 ABI supplement relocation types used by
    /// the built-in assembler and linker for symbol references, PIC/PLT
    /// indirection, and GOT access patterns.
    fn relocation_types(&self) -> &'static [RelocationTypeInfo] {
        static TABLE: &[RelocationTypeInfo] = &[
            // R_X86_64_NONE = 0 — no relocation
            RelocationTypeInfo::new("R_X86_64_NONE", 0, 0, false),
            // R_X86_64_64 = 1 — absolute 64-bit address
            RelocationTypeInfo::new("R_X86_64_64", 1, 8, false),
            // R_X86_64_PC32 = 2 — 32-bit PC-relative (RIP-relative)
            RelocationTypeInfo::new("R_X86_64_PC32", 2, 4, true),
            // R_X86_64_GOT32 = 3 — 32-bit GOT entry offset
            RelocationTypeInfo::new("R_X86_64_GOT32", 3, 4, false),
            // R_X86_64_PLT32 = 4 — 32-bit PLT-relative
            RelocationTypeInfo::new("R_X86_64_PLT32", 4, 4, true),
            // R_X86_64_GLOB_DAT = 6 — GOT entry for dynamic symbol
            RelocationTypeInfo::new("R_X86_64_GLOB_DAT", 6, 8, false),
            // R_X86_64_JUMP_SLOT = 7 — PLT jump slot
            RelocationTypeInfo::new("R_X86_64_JUMP_SLOT", 7, 8, false),
            // R_X86_64_RELATIVE = 8 — base-relative (for PIC data)
            RelocationTypeInfo::new("R_X86_64_RELATIVE", 8, 8, false),
            // R_X86_64_GOTPCREL = 9 — 32-bit PC-relative GOT access
            RelocationTypeInfo::new("R_X86_64_GOTPCREL", 9, 4, true),
            // R_X86_64_32 = 10 — absolute 32-bit (zero-extended)
            RelocationTypeInfo::new("R_X86_64_32", 10, 4, false),
            // R_X86_64_32S = 11 — absolute 32-bit (sign-extended)
            RelocationTypeInfo::new("R_X86_64_32S", 11, 4, false),
            // R_X86_64_16 = 12 — absolute 16-bit
            RelocationTypeInfo::new("R_X86_64_16", 12, 2, false),
            // R_X86_64_PC16 = 13 — 16-bit PC-relative
            RelocationTypeInfo::new("R_X86_64_PC16", 13, 2, true),
            // R_X86_64_PC64 = 24 — 64-bit PC-relative
            RelocationTypeInfo::new("R_X86_64_PC64", 24, 8, true),
            // R_X86_64_GOTPCRELX = 41 — relaxable GOT PC-relative
            RelocationTypeInfo::new("R_X86_64_GOTPCRELX", 41, 4, true),
            // R_X86_64_REX_GOTPCRELX = 42 — relaxable REX GOT PC-relative
            RelocationTypeInfo::new("R_X86_64_REX_GOTPCRELX", 42, 4, true),
        ];
        TABLE
    }

    /// Emit function prologue instructions.
    fn emit_prologue(&self, mf: &MachineFunction) -> Vec<MachineInstruction> {
        X86_64CodeGen::emit_prologue_static(mf)
    }

    /// Emit function epilogue instructions.
    fn emit_epilogue(&self, mf: &MachineFunction) -> Vec<MachineInstruction> {
        X86_64CodeGen::emit_epilogue_static(mf)
    }

    /// The frame pointer register is RBP on x86-64.
    fn frame_pointer_reg(&self) -> u16 {
        RBP
    }

    /// The stack pointer register is RSP on x86-64.
    fn stack_pointer_reg(&self) -> u16 {
        RSP
    }

    /// x86-64 has no explicit return-address register.
    fn return_address_reg(&self) -> Option<u16> {
        None
    }

    /// Classify a single argument for the System V AMD64 ABI.
    fn classify_argument(&self, ty: &IrType) -> traits::ArgLocation {
        let loc = self.abi.classify_argument(ty);
        match loc {
            abi::ArgLocation::Register(r) => traits::ArgLocation::Register(r),
            abi::ArgLocation::RegisterPair(a, b) => traits::ArgLocation::RegisterPair(a, b),
            abi::ArgLocation::Stack(off) => traits::ArgLocation::Stack(off),
            abi::ArgLocation::Indirect(r) => traits::ArgLocation::Register(r),
        }
    }

    /// Classify the return value for the System V AMD64 ABI.
    fn classify_return(&self, ty: &IrType) -> traits::ArgLocation {
        let loc = self.abi.classify_return(ty);
        match loc {
            RetLocation::Register(r) => traits::ArgLocation::Register(r),
            RetLocation::RegisterPair(a, b) => traits::ArgLocation::RegisterPair(a, b),
            RetLocation::Indirect => traits::ArgLocation::Stack(0),
            RetLocation::Void => traits::ArgLocation::Stack(0),
        }
    }
}

// =========================================================================
// Internal bridge methods for ArchCodegen trait
// =========================================================================

impl X86_64CodeGen {
    /// Internal mutable lower_function for the trait.
    fn lower_function_impl(
        &mut self,
        func: &IrFunction,
        diag: &mut DiagnosticEngine,
        globals: &[crate::ir::module::GlobalVariable],
    ) -> Result<MachineFunction, String> {
        self.lower(func, diag, globals)
    }

    /// Static prologue emitter.
    fn emit_prologue_static(mf: &MachineFunction) -> Vec<MachineInstruction> {
        let mut prologue = Vec::new();
        let frame_size = mf.frame_size;

        prologue.push(Self::mk_inst(
            X86Opcode::Push,
            None,
            &[MachineOperand::Register(RBP)],
        ));
        prologue.push(Self::mk_inst(
            X86Opcode::Mov,
            Some(MachineOperand::Register(RBP)),
            &[MachineOperand::Register(RSP)],
        ));

        if frame_size > 0 && (!mf.is_leaf || frame_size > RED_ZONE_SIZE) {
            if frame_size > 4096 {
                let full_pages = frame_size / 4096;
                let remainder = frame_size % 4096;
                for _page in 0..full_pages {
                    prologue.push(Self::mk_inst(
                        X86Opcode::Sub,
                        Some(MachineOperand::Register(RSP)),
                        &[
                            MachineOperand::Register(RSP),
                            MachineOperand::Immediate(4096),
                        ],
                    ));
                    prologue.push(Self::mk_inst(
                        X86Opcode::Test,
                        None,
                        &[
                            MachineOperand::Memory {
                                base: Some(RSP),
                                index: None,
                                scale: 1,
                                displacement: 0,
                            },
                            MachineOperand::Immediate(0),
                        ],
                    ));
                }
                if remainder > 0 {
                    prologue.push(Self::mk_inst(
                        X86Opcode::Sub,
                        Some(MachineOperand::Register(RSP)),
                        &[
                            MachineOperand::Register(RSP),
                            MachineOperand::Immediate(remainder as i64),
                        ],
                    ));
                }
            } else {
                prologue.push(Self::mk_inst(
                    X86Opcode::Sub,
                    Some(MachineOperand::Register(RSP)),
                    &[
                        MachineOperand::Register(RSP),
                        MachineOperand::Immediate(frame_size as i64),
                    ],
                ));
            }
        }

        for &reg in &mf.callee_saved_regs {
            prologue.push(Self::mk_inst(
                X86Opcode::Push,
                None,
                &[MachineOperand::Register(reg)],
            ));
        }

        prologue
    }

    /// Static epilogue emitter.
    fn emit_epilogue_static(mf: &MachineFunction) -> Vec<MachineInstruction> {
        let mut epilogue = Vec::new();

        for &reg in mf.callee_saved_regs.iter().rev() {
            epilogue.push(Self::mk_inst(
                X86Opcode::Pop,
                Some(MachineOperand::Register(reg)),
                &[],
            ));
        }

        epilogue.push(Self::mk_inst(X86Opcode::Leave, None, &[]));

        let mut ret_inst = Self::mk_inst(X86Opcode::Ret, None, &[]);
        ret_inst.is_terminator = true;
        epilogue.push(ret_inst);

        epilogue
    }
}

// =========================================================================
// Display implementations for debugging
// =========================================================================

impl std::fmt::Display for X86Opcode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            X86Opcode::Mov => "mov",
            X86Opcode::MovZX => "movzx",
            X86Opcode::MovSX => "movsx",
            X86Opcode::Lea => "lea",
            X86Opcode::Push => "push",
            X86Opcode::Pop => "pop",
            X86Opcode::Xchg => "xchg",
            X86Opcode::Add => "add",
            X86Opcode::Sub => "sub",
            X86Opcode::Imul => "imul",
            X86Opcode::Idiv => "idiv",
            X86Opcode::Div => "div",
            X86Opcode::Neg => "neg",
            X86Opcode::Inc => "inc",
            X86Opcode::Dec => "dec",
            X86Opcode::And => "and",
            X86Opcode::Or => "or",
            X86Opcode::Xor => "xor",
            X86Opcode::Not => "not",
            X86Opcode::Shl => "shl",
            X86Opcode::Shr => "shr",
            X86Opcode::Sar => "sar",
            X86Opcode::Rol => "rol",
            X86Opcode::Ror => "ror",
            X86Opcode::Cmp => "cmp",
            X86Opcode::Test => "test",
            X86Opcode::Cmovo => "cmovo",
            X86Opcode::Cmovno => "cmovno",
            X86Opcode::Cmovb => "cmovb",
            X86Opcode::Cmovae => "cmovae",
            X86Opcode::Cmove => "cmove",
            X86Opcode::Cmovne => "cmovne",
            X86Opcode::Cmovbe => "cmovbe",
            X86Opcode::Cmova => "cmova",
            X86Opcode::Cmovs => "cmovs",
            X86Opcode::Cmovns => "cmovns",
            X86Opcode::Cmovp => "cmovp",
            X86Opcode::Cmovnp => "cmovnp",
            X86Opcode::Cmovl => "cmovl",
            X86Opcode::Cmovge => "cmovge",
            X86Opcode::Cmovle => "cmovle",
            X86Opcode::Cmovg => "cmovg",
            X86Opcode::Seto => "seto",
            X86Opcode::Setno => "setno",
            X86Opcode::Setb => "setb",
            X86Opcode::Setae => "setae",
            X86Opcode::Sete => "sete",
            X86Opcode::Setne => "setne",
            X86Opcode::Setbe => "setbe",
            X86Opcode::Seta => "seta",
            X86Opcode::Sets => "sets",
            X86Opcode::Setns => "setns",
            X86Opcode::Setp => "setp",
            X86Opcode::Setnp => "setnp",
            X86Opcode::Setl => "setl",
            X86Opcode::Setge => "setge",
            X86Opcode::Setle => "setle",
            X86Opcode::Setg => "setg",
            X86Opcode::Jmp => "jmp",
            X86Opcode::Jo => "jo",
            X86Opcode::Jno => "jno",
            X86Opcode::Jb => "jb",
            X86Opcode::Jae => "jae",
            X86Opcode::Je => "je",
            X86Opcode::Jne => "jne",
            X86Opcode::Jbe => "jbe",
            X86Opcode::Ja => "ja",
            X86Opcode::Js => "js",
            X86Opcode::Jns => "jns",
            X86Opcode::Jp => "jp",
            X86Opcode::Jnp => "jnp",
            X86Opcode::Jl => "jl",
            X86Opcode::Jge => "jge",
            X86Opcode::Jle => "jle",
            X86Opcode::Jg => "jg",
            X86Opcode::Call => "call",
            X86Opcode::Ret => "ret",
            X86Opcode::Nop => "nop",
            X86Opcode::Movsd => "movsd",
            X86Opcode::Movss => "movss",
            X86Opcode::Addsd => "addsd",
            X86Opcode::Addss => "addss",
            X86Opcode::Subsd => "subsd",
            X86Opcode::Subss => "subss",
            X86Opcode::Mulsd => "mulsd",
            X86Opcode::Mulss => "mulss",
            X86Opcode::Divsd => "divsd",
            X86Opcode::Divss => "divss",
            X86Opcode::Ucomisd => "ucomisd",
            X86Opcode::Ucomiss => "ucomiss",
            X86Opcode::Cvtsi2sd => "cvtsi2sd",
            X86Opcode::Cvtsi2ss => "cvtsi2ss",
            X86Opcode::Cvtsd2si => "cvtsd2si",
            X86Opcode::Cvtss2si => "cvtss2si",
            X86Opcode::Cvtsd2ss => "cvtsd2ss",
            X86Opcode::Cvtss2sd => "cvtss2sd",
            X86Opcode::Enter => "enter",
            X86Opcode::Leave => "leave",
            X86Opcode::Cdq => "cdq",
            X86Opcode::Cqo => "cqo",
            X86Opcode::Endbr64 => "endbr64",
            X86Opcode::Pause => "pause",
            X86Opcode::Lfence => "lfence",
            X86Opcode::InlineAsm => "<inline-asm>",
            X86Opcode::LoadInd => "movq_ind_load",
            X86Opcode::LoadInd32 => "movl_ind_load",
            X86Opcode::LoadInd16 => "movw_ind_load",
            X86Opcode::LoadInd8 => "movb_ind_load",
            X86Opcode::StoreInd => "movq_ind_store",
            X86Opcode::StoreInd32 => "movl_ind_store",
            X86Opcode::StoreInd16 => "movw_ind_store",
            X86Opcode::StoreInd8 => "movb_ind_store",
        };
        f.write_str(name)
    }
}

impl std::fmt::Display for X86Operand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            X86Operand::Reg(r) => {
                write!(f, "{}", registers::reg_name(*r, 8))
            }
            X86Operand::Imm(val) => write!(f, "${}", val),
            X86Operand::Mem {
                base,
                index,
                scale,
                disp,
                size: _,
            } => {
                write!(f, "[")?;
                let mut need_plus = false;
                if let Some(b) = base {
                    write!(f, "{}", registers::reg_name(*b, 8))?;
                    need_plus = true;
                }
                if let Some(idx) = index {
                    if need_plus {
                        write!(f, " + ")?;
                    }
                    write!(f, "{}*{}", registers::reg_name(*idx, 8), scale)?;
                    need_plus = true;
                }
                if *disp != 0 {
                    if need_plus && *disp > 0 {
                        write!(f, " + ")?;
                    }
                    write!(f, "{}", disp)?;
                }
                write!(f, "]")
            }
            X86Operand::RipRelative { symbol, offset } => {
                if *offset != 0 {
                    write!(f, "[rip + {} + {}]", symbol, offset)
                } else {
                    write!(f, "[rip + {}]", symbol)
                }
            }
            X86Operand::Label(label) => write!(f, "{}", label),
            X86Operand::Symbol(sym) => write!(f, "{}", sym),
        }
    }
}
