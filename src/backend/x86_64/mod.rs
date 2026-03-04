//! x86-64 backend — System V AMD64 ABI, variable-length instruction encoding,
//! SSE/SSE2 floating-point, security mitigations (retpoline, CET/IBT, stack probe).
//!
//! This is the PRIMARY validation target in the BCC backend validation order.

pub mod registers;
