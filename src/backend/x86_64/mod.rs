//! # x86-64 Backend — System V AMD64 ABI
//!
//! This is the root module for the x86-64 architecture backend.  It serves
//! two critical purposes:
//!
//! 1. **Module declarations**: Declares all submodules (`registers`, `codegen`,
//!    `abi`, `security`, `assembler`, `linker`).
//! 2. **[`ArchCodegen`] trait implementation**: Provides the [`X86_64Backend`]
//!    struct that implements the architecture abstraction trait defined in
//!    [`crate::backend::traits`].
//!
//! ## Architecture Overview
//!
//! - **ABI**: System V AMD64 — 6 integer argument registers (RDI, RSI, RDX,
//!   RCX, R8, R9), 8 SSE argument registers (XMM0–XMM7), 128-byte red zone,
//!   16-byte stack alignment at call sites.
//! - **Register File**: 16 GPRs (RAX–R15), 16 SSE registers (XMM0–XMM15).
//!   SSE/SSE2 is used for floating-point (**not** x87 FPU on x86-64).
//! - **Instruction Encoding**: Variable-length (1–15 bytes), REX prefix
//!   required for registers R8–R15 and XMM8–XMM15.
//! - **Security Mitigations**: Retpoline (`-mretpoline`), CET/IBT
//!   (`-fcf-protection`), and stack guard-page probing are x86-64 **only**.
//!
//! ## Validation Order
//!
//! The x86-64 backend is the **PRIMARY validation target** — it is validated
//! first in the backend order: x86-64 → i686 → AArch64 → RISC-V 64.
//!
//! ## Zero-Dependency Mandate
//!
//! Only `std` and `crate::` references.  No external crates.

// ===========================================================================
// Submodule declarations — all 6 submodules for the x86-64 backend
// ===========================================================================

pub mod abi;
pub mod assembler;
pub mod codegen;
pub mod linker;
pub mod registers;
pub mod security;

// ===========================================================================
// Re-exports — public API surface for downstream consumers
// ===========================================================================

/// Re-export all 16 GPR register constants (RAX–R15) and 16 SSE register
/// constants (XMM0–XMM15) for convenient access by other modules.
pub use self::registers::{
    R10, R11, R12, R13, R14, R15, R8, R9, RAX, RBP, RBX, RCX, RDI, RDX, RSI, RSP, XMM0, XMM1,
    XMM10, XMM11, XMM12, XMM13, XMM14, XMM15, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7, XMM8, XMM9,
};

/// Re-export the x86-64 ABI implementation for external usage.
pub use self::abi::X86_64Abi;

/// Re-export the x86-64 instruction selection engine.
pub use self::codegen::X86_64CodeGen;

/// Re-export the security mitigation configuration.
pub use self::security::SecurityConfig;

// ===========================================================================
// Internal imports
// ===========================================================================

use crate::backend::traits::{
    ArchCodegen, ArgLocation as TraitsArgLocation, MachineFunction, MachineInstruction,
    MachineOperand, RegisterInfo, RelocationTypeInfo,
};
use crate::common::diagnostics::DiagnosticEngine;
use crate::common::target::Target;
use crate::ir::function::IrFunction;
use crate::ir::types::IrType;

use self::abi::RetLocation;
use self::codegen::X86Opcode;

// ===========================================================================
// Static relocation type descriptors for x86-64 ELF
// ===========================================================================

/// Supported x86-64 ELF relocation types.
///
/// These descriptors are returned by [`X86_64Backend::relocation_types`] and
/// consumed by the common linker infrastructure to dispatch relocation
/// application to the architecture-specific handler.
///
/// | Name                        | ID | Size | PC-Relative | Description                    |
/// |-----------------------------|----|------|-------------|--------------------------------|
/// | `R_X86_64_NONE`             |  0 |   0  | No          | No relocation                  |
/// | `R_X86_64_64`               |  1 |   8  | No          | Direct 64-bit absolute         |
/// | `R_X86_64_PC32`             |  2 |   4  | Yes         | 32-bit PC-relative             |
/// | `R_X86_64_GOT32`            |  3 |   4  | No          | 32-bit GOT entry offset        |
/// | `R_X86_64_PLT32`            |  4 |   4  | Yes         | 32-bit PLT-relative            |
/// | `R_X86_64_GOTPCREL`         |  9 |   4  | Yes         | 32-bit GOT PC-relative         |
/// | `R_X86_64_32`               | 10 |   4  | No          | Direct 32-bit zero-extended    |
/// | `R_X86_64_32S`              | 11 |   4  | No          | Direct 32-bit sign-extended    |
/// | `R_X86_64_GOTPCRELX`        | 41 |   4  | Yes         | Relaxable GOTPCREL             |
/// | `R_X86_64_REX_GOTPCRELX`    | 42 |   4  | Yes         | Relaxable GOTPCREL with REX    |
static X86_64_RELOCATION_TYPES: [RelocationTypeInfo; 10] = [
    RelocationTypeInfo::new("R_X86_64_NONE", 0, 0, false),
    RelocationTypeInfo::new("R_X86_64_64", 1, 8, false),
    RelocationTypeInfo::new("R_X86_64_PC32", 2, 4, true),
    RelocationTypeInfo::new("R_X86_64_GOT32", 3, 4, false),
    RelocationTypeInfo::new("R_X86_64_PLT32", 4, 4, true),
    RelocationTypeInfo::new("R_X86_64_GOTPCREL", 9, 4, true),
    RelocationTypeInfo::new("R_X86_64_32", 10, 4, false),
    RelocationTypeInfo::new("R_X86_64_32S", 11, 4, false),
    RelocationTypeInfo::new("R_X86_64_GOTPCRELX", 41, 4, true),
    RelocationTypeInfo::new("R_X86_64_REX_GOTPCRELX", 42, 4, true),
];

// ===========================================================================
// X86_64Backend — main backend struct
// ===========================================================================

/// The x86-64 architecture backend implementing the [`ArchCodegen`] trait.
///
/// Coordinates instruction selection, register allocation, ABI conformance,
/// security mitigation injection, assembly encoding, and ELF linking for
/// the x86-64 architecture with System V AMD64 calling conventions.
///
/// # Construction
///
/// ```ignore
/// let backend = X86_64Backend::new(
///     Target::X86_64,
///     SecurityConfig::default(),
///     false, // pic
///     false, // shared
///     false, // debug_info
/// );
/// ```
///
/// # Security Mitigations
///
/// The x86-64 backend is the **only** architecture that supports:
/// - **Retpoline** (`-mretpoline`): Indirect call/jump thunk transformation
/// - **CET/IBT** (`-fcf-protection`): `endbr64` insertion at branch targets
/// - **Stack Probe** (always checked): Guard-page probe for frames > 4 KiB
pub struct X86_64Backend {
    /// Target information (pointer width, endianness, predefined macros).
    target: Target,

    /// Security mitigation configuration — retpoline, CET, stack probe settings.
    security_config: SecurityConfig,

    /// Whether PIC code generation is enabled (`-fPIC`).
    pic_enabled: bool,

    /// Whether generating a shared library (`-shared`).
    shared: bool,

    /// Whether debug info generation is enabled (`-g`).
    debug_info: bool,

    /// System V AMD64 ABI implementation for parameter/return classification.
    abi: X86_64Abi,
}

// ===========================================================================
// X86_64Backend — constructor and helper methods
// ===========================================================================

impl X86_64Backend {
    /// Create a new x86-64 backend instance.
    ///
    /// # Arguments
    ///
    /// * `target` — Target configuration (must be `Target::X86_64`).
    /// * `security_config` — Security mitigation settings from CLI flags.
    /// * `pic_enabled` — Whether `-fPIC` was specified.
    /// * `shared` — Whether `-shared` was specified.
    /// * `debug_info` — Whether `-g` was specified.
    ///
    /// # Panics
    ///
    /// Debug-asserts that `target` is `Target::X86_64`.
    pub fn new(
        target: Target,
        security_config: SecurityConfig,
        pic_enabled: bool,
        shared: bool,
        debug_info: bool,
    ) -> Self {
        debug_assert!(
            target == Target::X86_64,
            "X86_64Backend requires Target::X86_64, got {:?}",
            target
        );
        let abi = X86_64Abi::new(target);
        Self {
            target,
            security_config,
            pic_enabled,
            shared,
            debug_info,
            abi,
        }
    }

    /// Get the security configuration.
    #[inline]
    pub fn security_config(&self) -> &SecurityConfig {
        &self.security_config
    }

    /// Whether PIC code generation is enabled.
    #[inline]
    pub fn is_pic(&self) -> bool {
        self.pic_enabled
    }

    /// Whether generating a shared library.
    #[inline]
    pub fn is_shared(&self) -> bool {
        self.shared
    }

    /// Whether debug info is enabled.
    #[inline]
    pub fn has_debug_info(&self) -> bool {
        self.debug_info
    }
}

// ===========================================================================
// Helper: convert ABI-specific ArgLocation to traits::ArgLocation
// ===========================================================================

/// Convert an x86-64 ABI argument location to the trait-level `ArgLocation`.
///
/// The x86-64 ABI module has an `Indirect` variant (hidden pointer in a
/// register) that the trait-level enum does not.  We map it to
/// `ArgLocation::Register` since the pointer is physically passed in a
/// register — the caller allocates storage and passes the address.
fn convert_abi_arg_location(loc: self::abi::ArgLocation) -> TraitsArgLocation {
    match loc {
        self::abi::ArgLocation::Register(r) => TraitsArgLocation::Register(r),
        self::abi::ArgLocation::RegisterPair(r1, r2) => TraitsArgLocation::RegisterPair(r1, r2),
        self::abi::ArgLocation::Stack(off) => TraitsArgLocation::Stack(off),
        self::abi::ArgLocation::Indirect(r) => TraitsArgLocation::Register(r),
    }
}

/// Convert an x86-64 ABI return location to the trait-level `ArgLocation`.
///
/// Mapping:
/// - `RetLocation::Register(r)` → `ArgLocation::Register(r)`
/// - `RetLocation::RegisterPair(r1, r2)` → `ArgLocation::RegisterPair(r1, r2)`
/// - `RetLocation::Indirect` → `ArgLocation::Stack(0)` (hidden pointer return)
/// - `RetLocation::Void` → `ArgLocation::Register(RAX)` (unused placeholder)
fn convert_ret_location(loc: RetLocation) -> TraitsArgLocation {
    match loc {
        RetLocation::Register(r) => TraitsArgLocation::Register(r),
        RetLocation::RegisterPair(r1, r2) => TraitsArgLocation::RegisterPair(r1, r2),
        RetLocation::Indirect => TraitsArgLocation::Stack(0),
        RetLocation::Void => TraitsArgLocation::Register(registers::RAX),
    }
}

// ===========================================================================
// ArchCodegen trait implementation
// ===========================================================================

impl ArchCodegen for X86_64Backend {
    /// Lower an IR function to x86-64 machine instructions.
    ///
    /// This is the main entry point for code generation.  It:
    /// 1. Creates an [`X86_64CodeGen`] bound to the target.
    /// 2. Runs x86-64 instruction selection (which computes the frame
    ///    layout, maps parameters to ABI locations, and emits machine
    ///    instructions for every IR instruction).
    /// 3. Returns the [`MachineFunction`] ready for register allocation,
    ///    prologue/epilogue insertion, and assembly encoding.
    fn lower_function(
        &self,
        func: &IrFunction,
        diag: &mut DiagnosticEngine,
        globals: &[crate::ir::module::GlobalVariable],
    ) -> Result<MachineFunction, String> {
        let mut codegen = X86_64CodeGen::new(self.target);
        codegen.lower(func, diag, globals)
    }

    /// Emit machine code bytes from a fully register-allocated
    /// [`MachineFunction`].
    ///
    /// Uses the built-in x86-64 assembler (`assembler/`) to encode each
    /// [`MachineInstruction`] into bytes, producing relocatable object code.
    /// Security-related bytes (retpoline thunks, CET note section) are also
    /// handled by the assembler when the corresponding flags are active.
    fn emit_assembly(&self, mf: &MachineFunction) -> Result<Vec<u8>, String> {
        let result = assembler::assemble(mf, &self.security_config, self.pic_enabled);
        Ok(result.text)
    }

    /// Return the target architecture.
    #[inline]
    fn target(&self) -> Target {
        self.target
    }

    /// Return register allocation info for x86-64.
    ///
    /// Provides:
    /// - 14 allocatable GPRs (all except RSP, RBP)
    /// - 16 allocatable SSE registers
    /// - Callee-saved set: RBX, R12, R13, R14, R15
    /// - Caller-saved set: RAX, RCX, RDX, RSI, RDI, R8–R11
    /// - Argument GPRs: RDI, RSI, RDX, RCX, R8, R9
    /// - Argument SSE: XMM0–XMM7
    /// - Return GPRs: RAX, RDX
    /// - Return SSE: XMM0, XMM1
    fn register_info(&self) -> RegisterInfo {
        RegisterInfo {
            allocatable_gpr: registers::ALLOCATABLE_GPRS.to_vec(),
            allocatable_fpr: registers::ALLOCATABLE_SSE.to_vec(),
            callee_saved: registers::CALLEE_SAVED_GPRS.to_vec(),
            caller_saved: registers::CALLER_SAVED_GPRS.to_vec(),
            reserved: registers::RESERVED_REGS.to_vec(),
            argument_gpr: registers::ARG_GPRS.to_vec(),
            argument_fpr: registers::ARG_SSE.to_vec(),
            return_gpr: registers::RET_GPRS.to_vec(),
            return_fpr: registers::RET_SSE.to_vec(),
        }
    }

    /// Return supported relocation types for x86-64 ELF.
    ///
    /// Returns a static slice of 10 relocation type descriptors covering
    /// the x86-64 ELF ABI supplement relocations used by BCC's assembler
    /// and linker.
    #[inline]
    fn relocation_types(&self) -> &[RelocationTypeInfo] {
        &X86_64_RELOCATION_TYPES
    }

    /// Emit function prologue for x86-64.
    ///
    /// Standard prologue sequence:
    /// 1. `endbr64` (if CET/IBT enabled via `-fcf-protection`)
    /// 2. `push rbp`
    /// 3. `mov rbp, rsp`
    /// 4. Stack probe loop (if frame > 4096 bytes and stack_probe enabled)
    /// 5. `sub rsp, <frame_size>` (16-byte aligned) — only if no probe loop
    ///    already set RSP
    /// 6. Save callee-saved registers used in the function
    fn emit_prologue(&self, mf: &MachineFunction) -> Vec<MachineInstruction> {
        let mut prologue = Vec::new();

        // CET: endbr64 at function entry
        if self.security_config.cf_protection {
            prologue.push(security::emit_endbr64());
        }

        // push rbp
        let mut push_rbp = MachineInstruction::new(X86Opcode::Push.as_u32());
        push_rbp.operands.push(MachineOperand::Register(RBP));
        prologue.push(push_rbp);

        // mov rbp, rsp
        let mut mov_rbp_rsp = MachineInstruction::new(X86Opcode::Mov.as_u32());
        mov_rbp_rsp.operands.push(MachineOperand::Register(RSP));
        mov_rbp_rsp.result = Some(MachineOperand::Register(RBP));
        prologue.push(mov_rbp_rsp);

        // Stack frame allocation
        let aligned_size = (mf.frame_size + 15) & !15;

        if aligned_size > 0 {
            if self.security_config.stack_probe && mf.frame_size > security::PAGE_SIZE {
                // Large frame: emit probe loop.  The probe loop sets RSP
                // to the final target, so no additional SUB is needed.
                let probe_instrs = security::emit_stack_probe(mf.frame_size);
                prologue.extend(probe_instrs);
            } else {
                // Small frame or stack_probe disabled: simple SUB RSP, N.
                let mut sub_rsp = MachineInstruction::new(X86Opcode::Sub.as_u32());
                sub_rsp.operands.push(MachineOperand::Register(RSP));
                sub_rsp
                    .operands
                    .push(MachineOperand::Immediate(aligned_size as i64));
                sub_rsp.result = Some(MachineOperand::Register(RSP));
                prologue.push(sub_rsp);
            }
        }

        // Save callee-saved registers that the function uses
        for &reg in &mf.callee_saved_regs {
            let mut push_inst = MachineInstruction::new(X86Opcode::Push.as_u32());
            push_inst.operands.push(MachineOperand::Register(reg));
            prologue.push(push_inst);
        }

        prologue
    }

    /// Emit function epilogue for x86-64.
    ///
    /// Standard epilogue sequence:
    /// 1. Restore callee-saved registers (reverse order of prologue saves)
    /// 2. `mov rsp, rbp` (restore stack pointer to frame base)
    /// 3. `pop rbp` (restore caller's frame pointer)
    /// 4. `ret` (return to caller)
    fn emit_epilogue(&self, mf: &MachineFunction) -> Vec<MachineInstruction> {
        let mut epilogue = Vec::new();

        // Restore callee-saved registers in reverse order
        for &reg in mf.callee_saved_regs.iter().rev() {
            let mut pop_inst = MachineInstruction::new(X86Opcode::Pop.as_u32());
            pop_inst.operands.push(MachineOperand::Register(reg));
            pop_inst.result = Some(MachineOperand::Register(reg));
            epilogue.push(pop_inst);
        }

        // mov rsp, rbp — restore stack pointer
        let mut mov_rsp_rbp = MachineInstruction::new(X86Opcode::Mov.as_u32());
        mov_rsp_rbp.operands.push(MachineOperand::Register(RBP));
        mov_rsp_rbp.result = Some(MachineOperand::Register(RSP));
        epilogue.push(mov_rsp_rbp);

        // pop rbp — restore caller's frame pointer
        let mut pop_rbp = MachineInstruction::new(X86Opcode::Pop.as_u32());
        pop_rbp.operands.push(MachineOperand::Register(RBP));
        pop_rbp.result = Some(MachineOperand::Register(RBP));
        epilogue.push(pop_rbp);

        // ret — return to caller
        let mut ret_inst = MachineInstruction::new(X86Opcode::Ret.as_u32());
        ret_inst.is_terminator = true;
        epilogue.push(ret_inst);

        epilogue
    }

    /// Return the frame pointer register (RBP for x86-64).
    #[inline]
    fn frame_pointer_reg(&self) -> u16 {
        registers::RBP
    }

    /// Return the stack pointer register (RSP for x86-64).
    #[inline]
    fn stack_pointer_reg(&self) -> u16 {
        registers::RSP
    }

    /// Return the return address register.
    ///
    /// On x86-64, there is no dedicated link register — the return address
    /// is pushed onto the stack by the `CALL` instruction and lives at
    /// `[RBP+8]`.  Returns `None` to indicate stack-based return address.
    #[inline]
    fn return_address_reg(&self) -> Option<u16> {
        // x86-64 has no link register; the return address is on the stack.
        None
    }

    /// Classify where a function argument of the given IR type should be
    /// placed according to the System V AMD64 ABI.
    ///
    /// Delegates to the x86-64 ABI module for eightbyte classification,
    /// then converts the result to the trait-level [`ArgLocation`](TraitsArgLocation).
    fn classify_argument(&self, ty: &IrType) -> TraitsArgLocation {
        let abi_loc = self.abi.classify_argument(ty);
        convert_abi_arg_location(abi_loc)
    }

    /// Classify where a function return value of the given IR type should
    /// be placed according to the System V AMD64 ABI.
    ///
    /// Small scalars return in RAX (integer) or XMM0 (float).  Structs
    /// ≤16 bytes may return in register pairs.  Larger types are returned
    /// via a hidden pointer argument (indirect return).
    fn classify_return(&self, ty: &IrType) -> TraitsArgLocation {
        let ret_loc = self.abi.classify_return(ty);
        convert_ret_location(ret_loc)
    }
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::target::Target;

    /// Helper to create a default backend for testing.
    fn test_backend() -> X86_64Backend {
        X86_64Backend::new(
            Target::X86_64,
            SecurityConfig::default(),
            false,
            false,
            false,
        )
    }

    #[test]
    fn test_backend_construction() {
        let backend = test_backend();
        assert_eq!(backend.target(), Target::X86_64);
        assert!(!backend.is_pic());
        assert!(!backend.is_shared());
        assert!(!backend.has_debug_info());
    }

    #[test]
    fn test_backend_with_pic() {
        let backend = X86_64Backend::new(
            Target::X86_64,
            SecurityConfig::default(),
            true,
            false,
            false,
        );
        assert!(backend.is_pic());
        assert!(!backend.is_shared());
    }

    #[test]
    fn test_backend_with_shared() {
        let backend =
            X86_64Backend::new(Target::X86_64, SecurityConfig::default(), true, true, false);
        assert!(backend.is_pic());
        assert!(backend.is_shared());
    }

    #[test]
    fn test_backend_with_debug() {
        let backend = X86_64Backend::new(
            Target::X86_64,
            SecurityConfig::default(),
            false,
            false,
            true,
        );
        assert!(backend.has_debug_info());
    }

    #[test]
    fn test_security_config_accessor() {
        let config = SecurityConfig {
            retpoline: true,
            cf_protection: true,
            stack_probe: true,
        };
        let backend = X86_64Backend::new(Target::X86_64, config, false, false, false);
        assert!(backend.security_config().retpoline);
        assert!(backend.security_config().cf_protection);
        assert!(backend.security_config().stack_probe);
    }

    #[test]
    fn test_register_info() {
        let backend = test_backend();
        let info = backend.register_info();

        // 14 allocatable GPRs (16 - RSP - RBP)
        assert_eq!(info.allocatable_gpr.len(), 14);
        // 16 allocatable SSE registers
        assert_eq!(info.allocatable_fpr.len(), 16);
        // 5 callee-saved GPRs: RBX, R12, R13, R14, R15
        assert_eq!(info.callee_saved.len(), 5);
        // 9 caller-saved GPRs: RAX, RCX, RDX, RSI, RDI, R8-R11
        assert_eq!(info.caller_saved.len(), 9);
        // 2 reserved: RSP, RBP
        assert_eq!(info.reserved.len(), 2);
        assert!(info.reserved.contains(&registers::RSP));
        assert!(info.reserved.contains(&registers::RBP));
        // 6 integer argument registers
        assert_eq!(info.argument_gpr.len(), 6);
        assert_eq!(info.argument_gpr[0], registers::RDI);
        // 8 SSE argument registers
        assert_eq!(info.argument_fpr.len(), 8);
        assert_eq!(info.argument_fpr[0], registers::XMM0);
        // 2 integer return registers
        assert_eq!(info.return_gpr.len(), 2);
        assert_eq!(info.return_gpr[0], registers::RAX);
        assert_eq!(info.return_gpr[1], registers::RDX);
        // 2 SSE return registers
        assert_eq!(info.return_fpr.len(), 2);
        assert_eq!(info.return_fpr[0], registers::XMM0);
        assert_eq!(info.return_fpr[1], registers::XMM1);
    }

    #[test]
    fn test_relocation_types() {
        let backend = test_backend();
        let relocs = backend.relocation_types();

        assert_eq!(relocs.len(), 10);
        // Spot-check a few entries
        assert_eq!(relocs[0].name, "R_X86_64_NONE");
        assert_eq!(relocs[0].type_id, 0);
        assert_eq!(relocs[2].name, "R_X86_64_PC32");
        assert_eq!(relocs[2].type_id, 2);
        assert!(relocs[2].is_pc_relative);
        assert_eq!(relocs[4].name, "R_X86_64_PLT32");
        assert_eq!(relocs[4].type_id, 4);
        assert!(relocs[4].is_pc_relative);
    }

    #[test]
    fn test_frame_pointer_reg() {
        let backend = test_backend();
        assert_eq!(backend.frame_pointer_reg(), registers::RBP);
    }

    #[test]
    fn test_stack_pointer_reg() {
        let backend = test_backend();
        assert_eq!(backend.stack_pointer_reg(), registers::RSP);
    }

    #[test]
    fn test_return_address_reg_is_none() {
        let backend = test_backend();
        // x86-64 has no link register — return address is on the stack.
        assert_eq!(backend.return_address_reg(), None);
    }

    #[test]
    fn test_prologue_basic() {
        let backend = test_backend();
        let mf = MachineFunction::new("test_fn".to_string());
        // frame_size=0, no callee-saved regs
        let prologue = backend.emit_prologue(&mf);

        // Expect: push rbp, mov rbp rsp (no sub rsp since frame_size=0)
        assert_eq!(prologue.len(), 2);
        // First instruction: PUSH RBP
        assert_eq!(prologue[0].opcode, X86Opcode::Push.as_u32());
        assert_eq!(
            prologue[0].operands[0],
            MachineOperand::Register(registers::RBP)
        );
        // Second instruction: MOV RBP, RSP
        assert_eq!(prologue[1].opcode, X86Opcode::Mov.as_u32());
    }

    #[test]
    fn test_prologue_with_frame() {
        let backend = test_backend();
        let mut mf = MachineFunction::new("test_fn".to_string());
        mf.frame_size = 32;
        let prologue = backend.emit_prologue(&mf);

        // Expect: push rbp, mov rbp rsp, sub rsp 32
        assert_eq!(prologue.len(), 3);
        assert_eq!(prologue[2].opcode, X86Opcode::Sub.as_u32());
    }

    #[test]
    fn test_prologue_with_cet() {
        let config = SecurityConfig {
            retpoline: false,
            cf_protection: true,
            stack_probe: false,
        };
        let backend = X86_64Backend::new(Target::X86_64, config, false, false, false);
        let mf = MachineFunction::new("test_fn".to_string());
        let prologue = backend.emit_prologue(&mf);

        // Expect: endbr64, push rbp, mov rbp rsp
        assert_eq!(prologue.len(), 3);
        assert_eq!(prologue[0].opcode, X86Opcode::Endbr64.as_u32());
    }

    #[test]
    fn test_prologue_with_stack_probe() {
        let config = SecurityConfig {
            retpoline: false,
            cf_protection: false,
            stack_probe: true,
        };
        let backend = X86_64Backend::new(Target::X86_64, config, false, false, false);
        let mut mf = MachineFunction::new("test_fn".to_string());
        mf.frame_size = 8192; // > 4096, triggers probe

        let prologue = backend.emit_prologue(&mf);
        // Expect: push rbp, mov rbp rsp, [probe instructions...]
        // probe_instrs from security::emit_stack_probe(8192) should be non-empty
        assert!(prologue.len() > 3);
    }

    #[test]
    fn test_prologue_with_callee_saved() {
        let backend = test_backend();
        let mut mf = MachineFunction::new("test_fn".to_string());
        mf.frame_size = 16;
        mf.callee_saved_regs = vec![registers::RBX, registers::R12];

        let prologue = backend.emit_prologue(&mf);
        // Expect: push rbp, mov rbp rsp, sub rsp 16, push rbx, push r12
        assert_eq!(prologue.len(), 5);
        // Last two should be PUSH for callee-saved
        assert_eq!(prologue[3].opcode, X86Opcode::Push.as_u32());
        assert_eq!(
            prologue[3].operands[0],
            MachineOperand::Register(registers::RBX)
        );
        assert_eq!(prologue[4].opcode, X86Opcode::Push.as_u32());
        assert_eq!(
            prologue[4].operands[0],
            MachineOperand::Register(registers::R12)
        );
    }

    #[test]
    fn test_epilogue_basic() {
        let backend = test_backend();
        let mf = MachineFunction::new("test_fn".to_string());
        let epilogue = backend.emit_epilogue(&mf);

        // Expect: mov rsp rbp, pop rbp, ret
        assert_eq!(epilogue.len(), 3);
        assert_eq!(epilogue[0].opcode, X86Opcode::Mov.as_u32());
        assert_eq!(epilogue[1].opcode, X86Opcode::Pop.as_u32());
        assert_eq!(epilogue[2].opcode, X86Opcode::Ret.as_u32());
        assert!(epilogue[2].is_terminator);
    }

    #[test]
    fn test_epilogue_callee_saved_reverse_order() {
        let backend = test_backend();
        let mut mf = MachineFunction::new("test_fn".to_string());
        mf.callee_saved_regs = vec![registers::RBX, registers::R12, registers::R13];

        let epilogue = backend.emit_epilogue(&mf);
        // Expect: pop r13, pop r12, pop rbx, mov rsp rbp, pop rbp, ret
        assert_eq!(epilogue.len(), 6);
        // Callee-saved restores in reverse order: R13, R12, RBX
        assert_eq!(
            epilogue[0].operands[0],
            MachineOperand::Register(registers::R13)
        );
        assert_eq!(
            epilogue[1].operands[0],
            MachineOperand::Register(registers::R12)
        );
        assert_eq!(
            epilogue[2].operands[0],
            MachineOperand::Register(registers::RBX)
        );
    }

    #[test]
    fn test_classify_argument_integer() {
        let backend = test_backend();
        let loc = backend.classify_argument(&IrType::I32);
        // First integer argument goes in RDI
        assert!(matches!(loc, TraitsArgLocation::Register(_)));
    }

    #[test]
    fn test_classify_argument_float() {
        let backend = test_backend();
        let loc = backend.classify_argument(&IrType::F64);
        // First FP argument goes in XMM0
        assert!(matches!(loc, TraitsArgLocation::Register(_)));
    }

    #[test]
    fn test_classify_return_integer() {
        let backend = test_backend();
        let loc = backend.classify_return(&IrType::I64);
        // Integer return in RAX
        assert!(matches!(loc, TraitsArgLocation::Register(_)));
    }

    #[test]
    fn test_classify_return_void() {
        let backend = test_backend();
        let loc = backend.classify_return(&IrType::Void);
        // Void return maps to RAX placeholder
        assert!(matches!(loc, TraitsArgLocation::Register(r) if r == registers::RAX));
    }

    #[test]
    fn test_re_exports_accessible() {
        // Verify that re-exported constants are accessible
        assert_eq!(super::RAX, 0);
        assert_eq!(super::RSP, 4);
        assert_eq!(super::RBP, 5);
        assert_eq!(super::XMM0, 16);
        assert_eq!(super::XMM15, 31);
        assert_eq!(super::R15, 15);
    }
}
