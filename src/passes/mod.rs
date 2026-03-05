//! # BCC Optimization Passes (Phase 8)
//!
//! This module implements the Phase 8 optimization pipeline for BCC's intermediate
//! representation. The pipeline consists of three core optimization passes orchestrated
//! by a pass manager that iterates until fixpoint.
//!
//! ## Pass Pipeline (Fixed Order)
//!
//! 1. [`constant_folding`] — Evaluate compile-time constant operations, fold conditional
//!    branches with known conditions, propagate constants through use-def chains
//! 2. [`dead_code_elimination`] — Remove instructions with unused results and no side
//!    effects, remove unreachable basic blocks
//! 3. [`simplify_cfg`] — Merge single-predecessor/successor block pairs, eliminate
//!    empty blocks, simplify branch chains, remove trivial phi nodes
//!
//! ## Entry Point
//!
//! The main entry point is [`pass_manager::optimize_module`] (or
//! [`pass_manager::run_optimization_pipeline`] with optimization level), which runs
//! the full pipeline on every function in an IR module.
//!
//! ## Dependencies
//!
//! - Depends on `crate::ir` for IR instruction, basic block, and function types
//! - Depends on `crate::common` for FxHashMap/FxHashSet and diagnostics
//! - Does NOT depend on `crate::frontend` or `crate::backend`
//!
//! ## Architectural Notes
//!
//! - All passes operate on SSA-form IR (produced by mem2reg, Phase 7)
//! - All passes must preserve SSA invariants
//! - The pass order is fixed (not configurable) per project requirements
//! - Only basic optimizations are implemented — advanced optimizations (loop unrolling,
//!   vectorization, inlining) are explicitly out of scope
//! - The pipeline iterates until fixpoint to maximize optimization at minimal complexity

// ---------------------------------------------------------------------------
// Submodule declarations
// ---------------------------------------------------------------------------

/// Pass scheduling and execution framework — orchestrates the fixed-order
/// optimization pipeline (constant folding → dead code elimination → CFG
/// simplification) with fixpoint iteration.
pub mod pass_manager;

/// Constant folding and propagation pass — evaluates compile-time constant
/// operations, folds conditional branches with known conditions, and
/// propagates constants through use-def chains.
pub mod constant_folding;

/// Dead code elimination pass — removes instructions with unused results
/// and no side effects, and removes unreachable basic blocks.
pub mod dead_code_elimination;

/// Control-flow graph simplification pass — merges single-predecessor/successor
/// block pairs, eliminates empty blocks, simplifies branch chains, and removes
/// trivial phi nodes.
pub mod simplify_cfg;

// ---------------------------------------------------------------------------
// Re-exports — main entry points for the optimization pipeline
// ---------------------------------------------------------------------------

/// Re-export of [`pass_manager::optimize_module`] for convenient access.
///
/// Runs the fixed-order optimization pass sequence on every function definition
/// in the module. Returns `true` if any function was modified.
pub use pass_manager::optimize_module;

/// Re-export of [`pass_manager::optimize_function`] for convenient access.
///
/// Runs the fixed-order optimization pass sequence on a single function until
/// fixpoint. Returns `true` if any optimization was applied.
pub use pass_manager::optimize_function;

/// Re-export of [`pass_manager::run_optimization_pipeline`] for convenient access.
///
/// Runs the Phase 8 optimization pipeline on the module according to the
/// specified optimization level (0–3). Returns `true` if any function was modified.
pub use pass_manager::run_optimization_pipeline;
