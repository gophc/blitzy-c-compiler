//! # SSA Construction via Alloca-Then-Promote
//!
//! This module implements SSA (Static Single Assignment) construction
//! using the "alloca-then-promote" approach:
//!
//! 1. **Dominator tree** — Compute the dominator tree for the CFG.
//! 2. **Dominance frontiers** — Determine where phi nodes are needed.
//! 3. **SSA builder** — Insert phi nodes and rename variables.
//! 4. **Phi elimination** — Convert SSA back to register assignments.

pub mod dominator_tree;
