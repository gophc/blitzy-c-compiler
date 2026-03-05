//! # BCC Optimization Passes (Phase 8)
//!
//! This module implements the Phase 8 optimization pipeline for BCC's intermediate
//! representation.

// Individual optimization passes — declared for module compilation.
// Full module documentation and re-exports will be provided by the mod.rs agent.

pub mod constant_folding;
pub mod dead_code_elimination;
pub mod simplify_cfg;
