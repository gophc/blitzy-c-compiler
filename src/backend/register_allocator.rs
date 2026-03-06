//! # Linear Scan Register Allocator
//!
//! Implements BCC's register allocation using the **linear scan** algorithm,
//! an efficient O(n log n) approach that processes live intervals in order of
//! increasing start point and maintains an *active* set of currently live
//! intervals.  The allocator is **architecture-agnostic** — it is fully
//! parameterized by [`RegisterInfo`], which describes the physical register
//! sets available on a given target.
//!
//! ## Pipeline Position
//!
//! Register allocation runs after phi-elimination (Phase 9) and before
//! final code emission (Phase 10).  At this point the IR is no longer in
//! SSA form — phi nodes have been replaced by parallel copies — so every
//! virtual register has a single definition and a well-defined live range.
//!
//! ## Algorithm Overview
//!
//! 1. **Live interval computation** ([`compute_live_intervals`]):
//!    Linearizes basic blocks in reverse postorder, assigns sequential
//!    instruction indices, and records the `[def, last_use]` range for
//!    every SSA value.  Phi-node incoming values are extended to the end
//!    of their corresponding predecessor blocks.
//!
//! 2. **Linear scan allocation** ([`allocate_registers`]):
//!    Sorts intervals by start point.  For each interval the algorithm
//!    expires ended intervals, reclaims their physical registers, and
//!    attempts assignment from the correct register class pool (GPR or
//!    FPR).  When the pool is empty, a spill heuristic evicts the
//!    interval whose endpoint is farthest in the future, minimising
//!    overall spill pressure.
//!
//! 3. **Spill code insertion** ([`insert_spill_code`]):
//!    Walks the function IR and, for every spilled value, inserts
//!    `Store` instructions after definitions and `Load` instructions
//!    before uses, allocating stack-frame spill slots via `Alloca`.
//!
//! ## Zero-Dependency
//!
//! This module uses only `std` and internal `crate::` imports.  All hash
//! maps / sets use [`FxHashMap`] / [`FxHashSet`] as mandated by the
//! project's zero-external-dependency policy.

use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::common::target::Target;
#[allow(unused_imports)]
use crate::ir::basic_block::BasicBlock;
use crate::ir::function::IrFunction;
#[allow(unused_imports)]
use crate::ir::instructions::{BlockId, Instruction, Value};
use crate::ir::types::IrType;

// ===========================================================================
// PhysReg — Physical register identifier
// ===========================================================================

/// A physical (machine) register, identified by an architecture-defined index.
///
/// The numeric payload is opaque to the register allocator — the mapping
/// from index to register name (e.g., `0 → RAX` on x86-64) is owned by
/// the architecture backend.  Only equality, hashing, and set-membership
/// operations are used during allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PhysReg(pub u16);

impl std::fmt::Display for PhysReg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "r{}", self.0)
    }
}

// ===========================================================================
// RegClass — Register class (GPR vs FPR)
// ===========================================================================

/// Classification of a register file.
///
/// During live-interval computation each virtual register is assigned a
/// `RegClass` based on its IR type:
/// - Integer types (`I1`–`I128`) and pointers → [`Integer`](RegClass::Integer)
/// - Floating-point types (`F32`, `F64`, `F80`) → [`Float`](RegClass::Float)
///
/// The allocator maintains separate free-lists for each class so that
/// integer values are never allocated to FP registers and vice versa.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegClass {
    /// General-purpose registers — integers and pointers.
    Integer,
    /// Floating-point / SIMD registers.
    Float,
}

// ===========================================================================
// SpillSlot — Stack-frame spill location
// ===========================================================================

/// A stack-frame slot reserved for a spilled virtual register.
///
/// Allocated by [`allocate_registers`] when physical registers are
/// exhausted, and consumed by [`insert_spill_code`] to emit the
/// store/load instructions.
#[derive(Debug, Clone)]
pub struct SpillSlot {
    /// Sequential slot index (0-based, unique per function).
    pub index: u32,
    /// Slot width in bytes (typically 4 or 8).
    pub size: usize,
    /// Byte offset from the frame pointer.  Negative values indicate
    /// locations below the frame pointer (standard for downward-growing
    /// stacks on all supported architectures).
    pub offset: i32,
}

// ===========================================================================
// LiveInterval — Virtual register live range
// ===========================================================================

/// Live range of a single virtual register (SSA value).
///
/// Produced by [`compute_live_intervals`] and consumed (mutated) by
/// [`allocate_registers`].  After allocation exactly one of `assigned`
/// or `spill_slot` is `Some`.
#[derive(Debug, Clone)]
pub struct LiveInterval {
    /// The virtual register this interval tracks.
    pub vreg: Value,
    /// First instruction index where this value appears (def point).
    pub start: u32,
    /// Last instruction index where this value is used.
    pub end: u32,
    /// Register class required by this value.
    pub reg_class: RegClass,
    /// Physical register assigned, or `None` if spilled.
    pub assigned: Option<PhysReg>,
    /// Spill-slot index, or `None` if assigned to a register.
    pub spill_slot: Option<u32>,
    /// IR type of this value — used internally for spill code generation.
    /// Not part of the public API contract but carried for convenience.
    pub(crate) ty: IrType,
}

// ===========================================================================
// RegisterInfo — Architecture register-set descriptor
// ===========================================================================

/// Describes the physical register set of a target architecture for the
/// register allocator's internal use.
///
/// This struct uses [`FxHashSet`] for callee/caller-saved and reserved
/// register sets, enabling O(1) membership tests during allocation.
/// Architecture backends produce a [`crate::backend::traits::RegisterInfo`]
/// (using `Vec<u16>` for ABI-level register descriptions), which is
/// converted to this struct via the [`From`] trait implementation below.
///
/// Register preference order within `allocatable_gpr` / `allocatable_fpr`
/// influences allocation quality — prefer caller-saved registers first
/// to reduce callee-save traffic.
#[derive(Debug, Clone)]
pub struct RegisterInfo {
    /// GPRs available for allocation, in preference order.
    pub allocatable_gpr: Vec<PhysReg>,
    /// FPRs available for allocation, in preference order.
    pub allocatable_fpr: Vec<PhysReg>,
    /// Callee-saved registers — must be saved/restored if used.
    pub callee_saved: FxHashSet<PhysReg>,
    /// Caller-saved registers — clobbered across calls.
    pub caller_saved: FxHashSet<PhysReg>,
    /// Reserved registers that are never allocatable (SP, FP, …).
    pub reserved: FxHashSet<PhysReg>,
}

/// Conversion from the architecture-level [`crate::backend::traits::RegisterInfo`]
/// (which uses `Vec<u16>` for register lists) to the allocator-internal
/// `RegisterInfo` (which uses `PhysReg` wrappers and `FxHashSet` for O(1)
/// membership tests).
///
/// This bridges the two `RegisterInfo` definitions, allowing architecture
/// backends to return their canonical [`crate::backend::traits::RegisterInfo`]
/// from `ArchCodegen::register_info()`, which the code generation driver
/// then converts for the register allocator.
impl From<crate::backend::traits::RegisterInfo> for RegisterInfo {
    fn from(arch_info: crate::backend::traits::RegisterInfo) -> Self {
        RegisterInfo {
            allocatable_gpr: arch_info
                .allocatable_gpr
                .iter()
                .map(|&r| PhysReg(r))
                .collect(),
            allocatable_fpr: arch_info
                .allocatable_fpr
                .iter()
                .map(|&r| PhysReg(r))
                .collect(),
            callee_saved: arch_info.callee_saved.iter().map(|&r| PhysReg(r)).collect(),
            caller_saved: arch_info.caller_saved.iter().map(|&r| PhysReg(r)).collect(),
            reserved: arch_info.reserved.iter().map(|&r| PhysReg(r)).collect(),
        }
    }
}

// ===========================================================================
// AllocationResult — Output of the allocator
// ===========================================================================

/// Complete allocation output for a single function.
///
/// Consumed by the code generator (Phase 10) for machine-code emission
/// and by [`insert_spill_code`] for IR rewriting.
#[derive(Debug)]
pub struct AllocationResult {
    /// Virtual register → physical register mapping.
    /// Spilled values are **not** present in this map.
    pub assignments: FxHashMap<Value, PhysReg>,
    /// Spill-slot definitions (indexed by [`SpillSlot::index`]).
    pub spill_slots: Vec<SpillSlot>,
    /// Total stack-frame size in bytes consumed by spill slots.
    pub frame_size: usize,
    /// Callee-saved registers that the function actually uses and must
    /// therefore be saved in the prologue and restored in the epilogue.
    pub callee_saved_used: Vec<PhysReg>,
    /// Maps spilled virtual register → [`SpillSlot::index`].
    pub(crate) spill_map: FxHashMap<Value, u32>,
    /// Maps virtual register → IR type (for typed spill loads).
    pub(crate) value_types: FxHashMap<Value, IrType>,
}

// ===========================================================================
// Helper — Determine register class from an IR type
// ===========================================================================

/// Maps an [`IrType`] to the register class needed to hold it.
///
/// Floating-point types use [`RegClass::Float`]; everything else
/// (integers, pointers, booleans, aggregates treated as integers)
/// uses [`RegClass::Integer`].
fn reg_class_for_type(ty: &IrType) -> RegClass {
    if ty.is_float() {
        RegClass::Float
    } else {
        RegClass::Integer
    }
}

// ===========================================================================
// Helper — Extract result type from an instruction
// ===========================================================================

/// Returns the IR type produced by an instruction's result value.
///
/// For instructions that have no result (Store, Branch, …) the return
/// value is a sensible default (`I64`) that will never actually be used
/// in allocation because those instructions produce no `Value`.
fn instruction_result_type(inst: &Instruction) -> IrType {
    match inst {
        Instruction::Alloca { .. } => IrType::Ptr,
        Instruction::Load { ty, .. } => ty.clone(),
        Instruction::BinOp { ty, .. } => ty.clone(),
        Instruction::ICmp { .. } | Instruction::FCmp { .. } => IrType::I1,
        Instruction::Call { return_type, .. } => return_type.clone(),
        Instruction::Phi { ty, .. } => ty.clone(),
        Instruction::GetElementPtr { .. } => IrType::Ptr,
        Instruction::BitCast { to_type, .. }
        | Instruction::Trunc { to_type, .. }
        | Instruction::ZExt { to_type, .. }
        | Instruction::SExt { to_type, .. } => to_type.clone(),
        Instruction::IntToPtr { .. } => IrType::Ptr,
        Instruction::PtrToInt { to_type, .. } => to_type.clone(),
        Instruction::InlineAsm { .. } => IrType::I64,
        // Store, Branch, CondBranch, Switch, Return — no result
        _ => IrType::I64,
    }
}

// ===========================================================================
// compute_live_intervals
// ===========================================================================

/// Computes live intervals for every virtual register (SSA value) in the
/// given function.
///
/// # Algorithm
///
/// 1. Linearise basic blocks via [`IrFunction::reverse_postorder`].
/// 2. Assign each instruction a monotonically increasing index.
/// 3. Record def and use points for every [`Value`].
/// 4. Extend intervals for phi-node incoming values to the end of
///    their predecessor blocks (the value must be live at the
///    control-flow edge).
///
/// # Returns
///
/// A vector of [`LiveInterval`]s sorted by ascending start point,
/// ready for consumption by [`allocate_registers`].
pub fn compute_live_intervals(func: &IrFunction) -> Vec<LiveInterval> {
    let rpo = func.reverse_postorder();
    if rpo.is_empty() {
        return Vec::new();
    }

    // ---------------------------------------------------------------
    // Step 1 — Linearise blocks and record block boundaries
    // ---------------------------------------------------------------
    let num_blocks = func.block_count();
    let mut block_start: Vec<u32> = vec![0; num_blocks];
    let mut block_end: Vec<u32> = vec![0; num_blocks];
    let mut idx: u32 = 0;

    for &bi in &rpo {
        if bi < num_blocks {
            block_start[bi] = idx;
            idx += func.blocks[bi].instruction_count() as u32;
            block_end[bi] = idx;
        }
    }

    // ---------------------------------------------------------------
    // Step 2 — Collect def/use points and types for every value
    // ---------------------------------------------------------------
    // We store (start, end, reg_class, ty) per Value.
    let mut starts: FxHashMap<Value, u32> = FxHashMap::default();
    let mut ends: FxHashMap<Value, u32> = FxHashMap::default();
    let mut classes: FxHashMap<Value, RegClass> = FxHashMap::default();
    let mut types: FxHashMap<Value, IrType> = FxHashMap::default();

    // Function parameters are defined at instruction index 0 (function entry).
    for param in &func.params {
        if param.value == Value::UNDEF {
            continue;
        }
        let rc = reg_class_for_type(&param.ty);
        starts.insert(param.value, 0);
        ends.insert(param.value, 0);
        classes.insert(param.value, rc);
        types.insert(param.value, param.ty.clone());
    }

    // Walk every instruction in RPO linear order.
    idx = 0;
    for &bi in &rpo {
        if bi >= num_blocks {
            continue;
        }
        let block = &func.blocks[bi];
        for inst in block.instructions.iter() {
            // --- Record definition ---
            if let Some(result) = inst.result() {
                if result != Value::UNDEF {
                    let ty = instruction_result_type(inst);
                    let rc = reg_class_for_type(&ty);
                    starts.entry(result).or_insert(idx);
                    let end = ends.entry(result).or_insert(idx);
                    if idx > *end {
                        *end = idx;
                    }
                    classes.entry(result).or_insert(rc);
                    types.entry(result).or_insert_with(|| ty);
                }
            }

            // --- Record uses ---
            for op in inst.operands() {
                if op == Value::UNDEF {
                    continue;
                }
                // If this value was never seen as a def (e.g. global reference)
                // we create a conservative entry starting at 0.
                starts.entry(op).or_insert(0);
                let end = ends.entry(op).or_insert(idx);
                if idx > *end {
                    *end = idx;
                }
                classes.entry(op).or_insert(RegClass::Integer);
                types.entry(op).or_insert(IrType::I64);
            }

            idx += 1;
        }
    }

    // ---------------------------------------------------------------
    // Step 3 — Phi-node live-range extension
    // ---------------------------------------------------------------
    // For each phi at block B with incoming (value, pred):
    //   value must be live-out at the end of pred.  Extend its
    //   interval to cover the last instruction of pred.
    for &bi in &rpo {
        if bi >= num_blocks {
            continue;
        }
        let block = &func.blocks[bi];
        for inst in block.instructions.iter() {
            if !Instruction::is_phi(inst) {
                break; // phi nodes always precede non-phi instructions
            }
            if let Instruction::Phi { incoming, .. } = inst {
                for (val, pred_bid) in incoming.iter() {
                    if *val == Value::UNDEF {
                        continue;
                    }
                    let pred_idx = pred_bid.index();
                    if pred_idx < num_blocks {
                        let pred_last = if block_end[pred_idx] > 0 {
                            block_end[pred_idx] - 1
                        } else {
                            0
                        };
                        let end = ends.entry(*val).or_insert(pred_last);
                        if pred_last > *end {
                            *end = pred_last;
                        }
                        starts.entry(*val).or_insert(0);
                        classes.entry(*val).or_insert(RegClass::Integer);
                        types.entry(*val).or_insert(IrType::I64);
                    }
                }
            }
        }
    }

    // ---------------------------------------------------------------
    // Step 4 — Assemble LiveInterval vector
    // ---------------------------------------------------------------
    let mut intervals: Vec<LiveInterval> = Vec::with_capacity(starts.len());
    for (value, &start) in starts.iter() {
        let end = ends.get(value).copied().unwrap_or(start);
        let rc = classes.get(value).copied().unwrap_or(RegClass::Integer);
        let ty = types.get(value).cloned().unwrap_or(IrType::I64);
        intervals.push(LiveInterval {
            vreg: *value,
            start,
            end,
            reg_class: rc,
            assigned: None,
            spill_slot: None,
            ty,
        });
    }

    // Sort by start point — required by the linear scan algorithm.
    intervals.sort_by_key(|iv| (iv.start, iv.vreg.index()));

    intervals
}

// ===========================================================================
// allocate_registers — Linear scan allocation
// ===========================================================================

/// Returns the spill slot size in bytes for a given IR type on the target.
///
/// Derives the size from the type's storage requirements rather than
/// using a fixed 8-byte slot.  This avoids wasting stack space for
/// narrow types (e.g., 4-byte I32 on i686) and ensures sufficient
/// space for wide types (e.g., 16-byte I128).
///
/// The minimum slot size is 1 byte; zero-sized types (Void, Function)
/// are clamped to 1 to avoid degenerate slot allocation.
fn spill_slot_size_for_type(ty: &IrType, target: &Target) -> usize {
    ty.size_bytes(target).max(1)
}

/// Performs linear-scan register allocation over a set of live intervals.
///
/// # Algorithm
///
/// 1. Sort intervals by start point (already done by
///    [`compute_live_intervals`] but re-sorted here for safety).
/// 2. For each interval in order:
///    a. **Expire**: remove active intervals whose end ≤ current start,
///    returning their registers to the free pool.
///    b. **Assign**: pop an available register from the correct class pool.
///    c. **Spill** (if pool empty): compare the current interval's end
///    with the farthest-endpoint active interval of the same class.
///    Spill whichever has the later endpoint (fewer total reloads).
/// 3. Track callee-saved registers that were used.
///
/// # Parameters
///
/// - `intervals`: live intervals (mutated in-place with `assigned` /
///   `spill_slot` fields).
/// - `reg_info`: architecture-supplied register descriptor.
/// - `target`: target architecture — used to compute correct spill slot
///   sizes from value types (e.g., 4 bytes for I32 on i686, 16 for I128).
///
/// # Returns
///
/// An [`AllocationResult`] summarising register assignments, spill slots,
/// stack-frame requirements, and callee-saved usage.
pub fn allocate_registers(
    intervals: &mut [LiveInterval],
    reg_info: &RegisterInfo,
    target: &Target,
) -> AllocationResult {
    // Pre-extract value types before any mutation.
    let value_types: FxHashMap<Value, IrType> = intervals
        .iter()
        .map(|iv| (iv.vreg, iv.ty.clone()))
        .collect();

    // Ensure intervals are sorted by start point.
    intervals.sort_by_key(|iv| (iv.start, iv.vreg.index()));

    // Output accumulators.
    let mut assignments: FxHashMap<Value, PhysReg> = FxHashMap::default();
    let mut spill_slots: Vec<SpillSlot> = Vec::new();
    let mut spill_map: FxHashMap<Value, u32> = FxHashMap::default();
    let mut callee_saved_used: FxHashSet<PhysReg> = FxHashSet::default();
    let mut frame_size: usize = 0;
    let mut next_spill_index: u32 = 0;

    // Free register pools — cloned from RegisterInfo so we can push/pop.
    // Reversed so that the *first* register in the preference list is
    // popped last (i.e. preferred first via LIFO pop from the end).
    let mut free_gpr: Vec<PhysReg> = reg_info.allocatable_gpr.iter().rev().copied().collect();
    let mut free_fpr: Vec<PhysReg> = reg_info.allocatable_fpr.iter().rev().copied().collect();

    // Active list: indices into `intervals`, maintained sorted by end point.
    let mut active: Vec<usize> = Vec::new();


    for i in 0..intervals.len() {
        let cur_start = intervals[i].start;


        // --- Expire old intervals whose endpoints are before cur_start ---
        let mut expired: Vec<usize> = Vec::new();
        for (pos, &ai) in active.iter().enumerate() {
            if intervals[ai].end < cur_start {
                expired.push(pos);
            }
        }
        // Remove in reverse order to keep indices valid.
        for &pos in expired.iter().rev() {
            let ai = active.remove(pos);
            if let Some(reg) = intervals[ai].assigned {
                match intervals[ai].reg_class {
                    RegClass::Integer => free_gpr.push(reg),
                    RegClass::Float => free_fpr.push(reg),
                }
            }
        }

        // --- Attempt allocation ---
        let pool = match intervals[i].reg_class {
            RegClass::Integer => &mut free_gpr,
            RegClass::Float => &mut free_fpr,
        };

        if let Some(reg) = pool.pop() {
            // Register available — assign it.
            intervals[i].assigned = Some(reg);
            assignments.insert(intervals[i].vreg, reg);

            if reg_info.callee_saved.contains(&reg) {
                callee_saved_used.insert(reg);
            }

            // Insert into active list, keeping it sorted by end point.
            let insert_pos = active
                .iter()
                .position(|&ai| intervals[ai].end > intervals[i].end)
                .unwrap_or(active.len());
            active.insert(insert_pos, i);
        } else {
            // --- Spill decision ---
            // Find the active interval of the same class with the farthest end.
            let spill_candidate = active
                .iter()
                .copied()
                .filter(|&ai| intervals[ai].reg_class == intervals[i].reg_class)
                .max_by_key(|&ai| intervals[ai].end);

            if let Some(far_idx) = spill_candidate {
                if intervals[far_idx].end > intervals[i].end {
                    // Spill the farther interval; re-use its register.
                    let reg = intervals[far_idx].assigned.take().unwrap();
                    assignments.remove(&intervals[far_idx].vreg);

                    // Allocate a spill slot for the evicted interval.
                    // Derive slot size from the spilled value's type.
                    let spill_size = spill_slot_size_for_type(&intervals[far_idx].ty, target);
                    let slot_offset = -(frame_size as i32) - (spill_size as i32);
                    let slot = SpillSlot {
                        index: next_spill_index,
                        size: spill_size,
                        offset: slot_offset,
                    };
                    intervals[far_idx].spill_slot = Some(next_spill_index);
                    spill_map.insert(intervals[far_idx].vreg, next_spill_index);
                    spill_slots.push(slot);
                    next_spill_index += 1;
                    frame_size += spill_size;

                    // Remove evicted interval from active.
                    active.retain(|&ai| ai != far_idx);

                    // Assign the freed register to the current interval.
                    intervals[i].assigned = Some(reg);
                    assignments.insert(intervals[i].vreg, reg);

                    if reg_info.callee_saved.contains(&reg) {
                        callee_saved_used.insert(reg);
                    }

                    let insert_pos = active
                        .iter()
                        .position(|&ai| intervals[ai].end > intervals[i].end)
                        .unwrap_or(active.len());
                    active.insert(insert_pos, i);
                } else {
                    // Spill the current interval (shorter or equal — spill self).
                    // Derive slot size from the spilled value's type.
                    let spill_size = spill_slot_size_for_type(&intervals[i].ty, target);
                    let slot_offset = -(frame_size as i32) - (spill_size as i32);
                    let slot = SpillSlot {
                        index: next_spill_index,
                        size: spill_size,
                        offset: slot_offset,
                    };
                    intervals[i].spill_slot = Some(next_spill_index);
                    spill_map.insert(intervals[i].vreg, next_spill_index);
                    spill_slots.push(slot);
                    next_spill_index += 1;
                    frame_size += spill_size;
                }
            } else {
                // No active intervals of the same class — spill current.
                // Derive slot size from the spilled value's type.
                let spill_size = spill_slot_size_for_type(&intervals[i].ty, target);
                let slot_offset = -(frame_size as i32) - (spill_size as i32);
                let slot = SpillSlot {
                    index: next_spill_index,
                    size: spill_size,
                    offset: slot_offset,
                };
                intervals[i].spill_slot = Some(next_spill_index);
                spill_map.insert(intervals[i].vreg, next_spill_index);
                spill_slots.push(slot);
                next_spill_index += 1;
                frame_size += spill_size;
            }
        }
    }

    AllocationResult {
        assignments,
        spill_slots,
        frame_size,
        callee_saved_used: callee_saved_used.into_iter().collect(),
        spill_map,
        value_types,
    }
}

// ===========================================================================
// insert_spill_code
// ===========================================================================

/// Inserts explicit spill stores / reload loads into the function IR.
///
/// For every spilled virtual register identified in `allocation`:
/// - **At each definition site**: a `Store` instruction is inserted
///   immediately after the defining instruction, writing the result to
///   the corresponding stack spill slot.
/// - **At each use site**: a `Load` instruction is inserted immediately
///   before the consuming instruction, reading from the spill slot into
///   a fresh virtual register, and the operand is rewritten to reference
///   the loaded value.
///
/// The entry block receives `Alloca` instructions for each spill slot so
/// that the addresses are available throughout the function.
///
/// # Parameters
///
/// - `func`: the IR function to modify (instructions are rewritten in place).
/// - `allocation`: the allocation result containing spill information.
pub fn insert_spill_code(func: &mut IrFunction, allocation: &AllocationResult) {
    if allocation.spill_map.is_empty() {
        return; // Nothing to spill.
    }

    // Capture a source span from an existing instruction for synthetic
    // instructions.  We reuse an existing span because `Span` is an
    // internal detail of `Instruction` and constructing one from scratch
    // would require importing `crate::common::diagnostics::Span`.
    let default_span = match func
        .blocks
        .iter()
        .flat_map(|b| b.instructions.iter())
        .next()
    {
        Some(inst) => inst.span(),
        None => return, // empty function — nothing to do
    };

    // ------------------------------------------------------------------
    // Step 1 — Create alloca instructions for each spill slot in the
    //          entry block.
    // ------------------------------------------------------------------
    // Maps SpillSlot.index → Value (the alloca result, i.e. the address
    // of the spill slot on the stack).
    let mut slot_addrs: FxHashMap<u32, Value> = FxHashMap::default();
    let mut next_value = func.value_count;

    // We need to know where to insert: after existing allocas but before
    // the first non-alloca instruction.  Find that insertion point.
    let entry_alloca_count = func
        .entry_block()
        .instructions
        .iter()
        .take_while(|inst| Instruction::is_alloca(inst))
        .count();

    // Build alloca instructions for spill slots.
    let mut alloca_insts: Vec<Instruction> = Vec::with_capacity(allocation.spill_slots.len());
    for slot in &allocation.spill_slots {
        let alloca_val = Value(next_value);
        next_value += 1;

        let spill_ty = allocation
            .spill_map
            .iter()
            .find(|(_, &si)| si == slot.index)
            .and_then(|(v, _)| allocation.value_types.get(v))
            .cloned()
            .unwrap_or(IrType::I64);

        let alloca = Instruction::Alloca {
            result: alloca_val,
            ty: spill_ty,
            alignment: Some(8),
            span: default_span,
        };
        slot_addrs.insert(slot.index, alloca_val);
        alloca_insts.push(alloca);
    }

    // Insert allocas into the entry block.
    {
        let entry = func.entry_block_mut();
        for (offset, inst) in alloca_insts.into_iter().enumerate() {
            entry.insert_instruction(entry_alloca_count + offset, inst);
        }
    }

    // ------------------------------------------------------------------
    // Step 2 — Walk every block and insert spill stores / reload loads.
    // ------------------------------------------------------------------
    let num_blocks = func.blocks.len();
    for bi in 0..num_blocks {
        let old_insts = std::mem::take(func.blocks[bi].instructions_mut());
        let mut new_insts: Vec<Instruction> = Vec::with_capacity(old_insts.len());

        for inst in old_insts {
            // --- Insert reload loads BEFORE the instruction for spilled operands ---
            let operand_vals = Instruction::operands(&inst);
            let mut replacements: Vec<(Value, Value)> = Vec::new();

            for op in &operand_vals {
                if *op == Value::UNDEF {
                    continue;
                }
                if let Some(&slot_idx) = allocation.spill_map.get(op) {
                    if let Some(&addr) = slot_addrs.get(&slot_idx) {
                        let load_ty = allocation
                            .value_types
                            .get(op)
                            .cloned()
                            .unwrap_or(IrType::I64);
                        let load_val = Value(next_value);
                        next_value += 1;
                        let load = Instruction::Load {
                            result: load_val,
                            ptr: addr,
                            ty: load_ty,
                            volatile: false,
                            span: default_span,
                        };
                        new_insts.push(load);
                        replacements.push((*op, load_val));
                    }
                }
            }

            // Rewrite operands in the instruction.
            let mut inst = inst;
            if !replacements.is_empty() {
                let operand_refs = Instruction::operands_mut(&mut inst);
                for op_ref in operand_refs {
                    for (from, to) in &replacements {
                        if *op_ref == *from {
                            *op_ref = *to;
                            break;
                        }
                    }
                }
            }

            // Push the (possibly rewritten) instruction.
            new_insts.push(inst.clone());

            // --- Insert spill store AFTER the instruction for spilled results ---
            if let Some(result) = Instruction::result(&inst) {
                if result != Value::UNDEF {
                    if let Some(&slot_idx) = allocation.spill_map.get(&result) {
                        if let Some(&addr) = slot_addrs.get(&slot_idx) {
                            let store = Instruction::Store {
                                value: result,
                                ptr: addr,
                                volatile: false,
                                span: default_span,
                            };
                            new_insts.push(store);
                        }
                    }
                }
            }
        }

        *func.blocks[bi].instructions_mut() = new_insts;
    }

    // Update the function's value counter.
    func.value_count = next_value;
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::function::{FunctionParam, IrFunction};
    use crate::ir::instructions::{BinOp, Instruction, Value};
    use crate::ir::types::IrType;

    /// Build a minimal RegisterInfo with N GPRs and M FPRs.
    fn make_reg_info(n_gpr: u16, n_fpr: u16) -> RegisterInfo {
        let gprs: Vec<PhysReg> = (0..n_gpr).map(PhysReg).collect();
        let fprs: Vec<PhysReg> = (100..100 + n_fpr).map(PhysReg).collect();
        let callee: FxHashSet<PhysReg> = FxHashSet::default();
        let caller: FxHashSet<PhysReg> = FxHashSet::default();
        let reserved: FxHashSet<PhysReg> = FxHashSet::default();
        RegisterInfo {
            allocatable_gpr: gprs,
            allocatable_fpr: fprs,
            callee_saved: callee,
            caller_saved: caller,
            reserved,
        }
    }

    /// Build a trivial function: two params, one add, one return.
    fn make_simple_func() -> IrFunction {
        use crate::common::diagnostics::Span;
        let span = Span {
            start: 0,
            end: 0,
            file_id: 0,
        };
        let params = vec![
            FunctionParam::new("a".into(), IrType::I32, Value(0)),
            FunctionParam::new("b".into(), IrType::I32, Value(1)),
        ];
        let mut func = IrFunction::new("add".into(), params, IrType::I32);
        // Entry block: add two params, return result.
        let add = Instruction::BinOp {
            result: Value(2),
            op: BinOp::Add,
            lhs: Value(0),
            rhs: Value(1),
            ty: IrType::I32,
            span,
        };
        let ret = Instruction::Return {
            value: Some(Value(2)),
            span,
        };
        func.entry_block_mut().push_instruction(add);
        func.entry_block_mut().push_instruction(ret);
        func.value_count = 3;
        func
    }

    // -- compute_live_intervals tests ---------------------------------

    #[test]
    fn live_intervals_simple_function() {
        let func = make_simple_func();
        let intervals = compute_live_intervals(&func);

        // Three values: param a (V0), param b (V1), add result (V2).
        assert_eq!(intervals.len(), 3);

        // All should be integer class.
        for iv in &intervals {
            assert_eq!(iv.reg_class, RegClass::Integer);
        }

        // V0 and V1 are defined at 0 (params), used in the add at index 0.
        // V2 is defined at index 0 (add), used at index 1 (return).
        let v2 = intervals.iter().find(|iv| iv.vreg == Value(2)).unwrap();
        assert!(v2.end >= v2.start);
    }

    #[test]
    fn live_intervals_empty_function() {
        let func = IrFunction::new("empty".into(), vec![], IrType::Void);
        let intervals = compute_live_intervals(&func);
        // Only the entry block exists but it has no instructions so no
        // instruction-defined values (though the block exists).
        // No intervals expected.
        assert!(intervals.is_empty());
    }

    // -- allocate_registers tests -------------------------------------

    #[test]
    fn allocate_all_fit() {
        let func = make_simple_func();
        let mut intervals = compute_live_intervals(&func);
        let reg_info = make_reg_info(8, 4);
        let target = Target::X86_64;
        let result = allocate_registers(&mut intervals, &reg_info, &target);

        // With 8 GPRs and only 3 values, everything should be assigned.
        assert_eq!(result.assignments.len(), 3);
        assert!(result.spill_slots.is_empty());
        assert_eq!(result.frame_size, 0);

        // Each value should have an assigned register.
        for iv in &intervals {
            assert!(iv.assigned.is_some(), "value {:?} not assigned", iv.vreg);
            assert!(iv.spill_slot.is_none());
        }
    }

    #[test]
    fn allocate_spill_needed() {
        // With only 1 GPR, 3 integer values cannot all be assigned.
        let func = make_simple_func();
        let mut intervals = compute_live_intervals(&func);
        let reg_info = make_reg_info(1, 0);
        let target = Target::X86_64;
        let result = allocate_registers(&mut intervals, &reg_info, &target);

        // At least one value must be spilled.
        assert!(!result.spill_slots.is_empty());
        assert!(result.frame_size > 0);
        // Total of assigned + spilled == number of intervals.
        let total = result.assignments.len() + result.spill_map.len();
        assert_eq!(total, intervals.len());
    }

    #[test]
    fn allocate_callee_saved_tracking() {
        let mut callee: FxHashSet<PhysReg> = FxHashSet::default();
        callee.insert(PhysReg(0));
        callee.insert(PhysReg(1));

        let reg_info = RegisterInfo {
            allocatable_gpr: vec![PhysReg(0), PhysReg(1), PhysReg(2)],
            allocatable_fpr: vec![],
            callee_saved: callee,
            caller_saved: FxHashSet::default(),
            reserved: FxHashSet::default(),
        };

        let func = make_simple_func();
        let mut intervals = compute_live_intervals(&func);
        let target = Target::X86_64;
        let result = allocate_registers(&mut intervals, &reg_info, &target);

        // If callee-saved regs were used, they appear in callee_saved_used.
        for reg in &result.callee_saved_used {
            assert!(reg_info.callee_saved.contains(reg));
        }
    }

    // -- reg_class_for_type tests -------------------------------------

    #[test]
    fn reg_class_integer_types() {
        assert_eq!(reg_class_for_type(&IrType::I1), RegClass::Integer);
        assert_eq!(reg_class_for_type(&IrType::I8), RegClass::Integer);
        assert_eq!(reg_class_for_type(&IrType::I32), RegClass::Integer);
        assert_eq!(reg_class_for_type(&IrType::I64), RegClass::Integer);
        assert_eq!(reg_class_for_type(&IrType::Ptr), RegClass::Integer);
    }

    #[test]
    fn reg_class_float_types() {
        assert_eq!(reg_class_for_type(&IrType::F32), RegClass::Float);
        assert_eq!(reg_class_for_type(&IrType::F64), RegClass::Float);
        assert_eq!(reg_class_for_type(&IrType::F80), RegClass::Float);
    }

    // -- insert_spill_code tests --------------------------------------

    #[test]
    fn spill_code_inserts_stores_and_loads() {
        let func_orig = make_simple_func();
        let mut func = func_orig;
        let mut intervals = compute_live_intervals(&func);
        let reg_info = make_reg_info(1, 0);
        let target = Target::X86_64;
        let result = allocate_registers(&mut intervals, &reg_info, &target);

        let inst_count_before: usize = func.blocks.iter().map(|b| b.instruction_count()).sum();
        insert_spill_code(&mut func, &result);
        let inst_count_after: usize = func.blocks.iter().map(|b| b.instruction_count()).sum();

        // Spill code should increase the total instruction count.
        if !result.spill_map.is_empty() {
            assert!(
                inst_count_after > inst_count_before,
                "spill code should insert extra instructions"
            );
        }
    }

    #[test]
    fn no_spill_code_when_no_spills() {
        let mut func = make_simple_func();
        let mut intervals = compute_live_intervals(&func);
        let reg_info = make_reg_info(8, 4);
        let target = Target::X86_64;
        let result = allocate_registers(&mut intervals, &reg_info, &target);

        let inst_count_before: usize = func.blocks.iter().map(|b| b.instruction_count()).sum();
        insert_spill_code(&mut func, &result);
        let inst_count_after: usize = func.blocks.iter().map(|b| b.instruction_count()).sum();

        // No spills ⇒ no extra instructions.
        assert_eq!(inst_count_before, inst_count_after);
    }

    // -- PhysReg display test ----------------------------------------

    #[test]
    fn phys_reg_display() {
        assert_eq!(format!("{}", PhysReg(0)), "r0");
        assert_eq!(format!("{}", PhysReg(15)), "r15");
    }

    // -- SpillSlot fields test ----------------------------------------

    #[test]
    fn spill_slot_fields() {
        let slot = SpillSlot {
            index: 0,
            size: 8,
            offset: -8,
        };
        assert_eq!(slot.index, 0);
        assert_eq!(slot.size, 8);
        assert_eq!(slot.offset, -8);
    }
}
