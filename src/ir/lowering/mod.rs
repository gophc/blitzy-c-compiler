//! # IR Lowering — Phase 6
//!
//! This module implements Phase 6 of the BCC compilation pipeline: lowering the
//! semantically-validated, type-annotated AST into BCC's intermediate representation (IR).
//!
//! ## Architecture: Alloca-Then-Promote
//!
//! The lowering follows the mandated "alloca-first" pattern (matching LLVM's approach):
//!
//! 1. **Alloca insertion**: Every local variable is initially placed as an `alloca`
//!    instruction in the function's entry block, regardless of whether it could live
//!    in a register. Parameters are also alloca'd and their incoming values stored.
//!
//! 2. **Body lowering**: The function body is lowered using load/store instructions
//!    to access all local variables through their alloca pointers.
//!
//! 3. **SSA promotion** (Phase 7, NOT this module): The subsequent `mem2reg` pass
//!    promotes eligible allocas (scalar, non-address-taken) to SSA virtual registers.
//!
//! This design simplifies lowering enormously — every variable access is a simple
//! load or store to a known pointer, and SSA construction is deferred to a dedicated pass.
//!
//! ## Submodules
//!
//! - [`expr_lowering`]: Expression lowering (arithmetic, casts, calls, builtins, etc.)
//! - [`stmt_lowering`]: Statement lowering (control flow, loops, switch, goto, labels)
//! - [`decl_lowering`]: Declaration lowering (globals, function skeletons, static locals)
//! - [`asm_lowering`]: Inline assembly lowering (constraints, operands, clobbers, asm goto)
//!
//! ## Pipeline Integration
//!
//! - **Input**: Semantically-validated AST from `crate::frontend::sema`
//! - **Output**: `IrModule` containing `IrFunction`s and `GlobalVariable`s
//! - **Next stage**: `crate::ir::mem2reg` (Phase 7 — SSA construction)
//! - **Does NOT depend on**: `crate::ir::mem2reg`, `crate::passes`, or `crate::backend`
//!
//! ## Recursion Limit
//!
//! A hard 512-depth recursion limit is enforced for deeply nested structures
//! (kernel macros can produce deeply nested ASTs). This is mandated by the
//! project requirements (AAP §0.7.3).

// ============================================================================
// Submodule declarations
// ============================================================================

pub mod asm_lowering;
pub mod decl_lowering;
pub mod expr_lowering;
pub mod stmt_lowering;

// ============================================================================
// Imports — only `std` and `crate::` references (zero-dependency mandate)
// ============================================================================

use std::cell::RefCell;

use crate::common::diagnostics::{DiagnosticEngine, Span};
use crate::common::fx_hash::FxHashMap;
use crate::common::source_map::SourceMap;
use crate::common::string_interner::Interner;
use crate::common::target::Target;
use crate::common::type_builder::TypeBuilder;
use crate::common::types::{alignof_ctype, sizeof_ctype, CType};
use crate::frontend::parser::ast;
use crate::frontend::sema::symbol_table::SymbolTable;
use crate::ir::builder::IrBuilder;
use crate::ir::function::IrFunction;
use crate::ir::instructions::{BlockId, Value};
use crate::ir::module::IrModule;
use crate::ir::types::IrType;

// ============================================================================
// Thread-local interner snapshot for symbol name resolution
// ============================================================================

thread_local! {
    // Snapshot of the string interner content, populated once at the start of
    // `lower_translation_unit` so that all name extraction helpers can resolve
    // Symbol handles to real C identifier strings without threading a borrow of
    // the interner through deeply nested call chains.
    static INTERNER_SNAPSHOT: RefCell<Option<Vec<String>>> = const { RefCell::new(None) };
}

// ============================================================================
// Re-export for external convenience
// ============================================================================

pub use self::expr_lowering::ExprLoweringContext;

// ============================================================================
// Constants
// ============================================================================

/// Default maximum recursion depth for nested constructs during IR lowering.
/// This limit protects against stack overflow on deeply nested kernel macro
/// expansions. Mandated by AAP §0.7.3.
const DEFAULT_MAX_RECURSION_DEPTH: u32 = 512;

// ============================================================================
// LoweringError — errors that can occur during IR lowering
// ============================================================================

/// Errors that can occur during Phase 6 (AST-to-IR lowering).
///
/// These errors represent conditions that prevent the lowering pipeline
/// from producing correct IR for part or all of the translation unit.
/// Non-fatal issues are reported as diagnostics via [`DiagnosticEngine`]
/// instead.
#[derive(Debug)]
pub enum LoweringError {
    /// Recursion depth exceeded during nested construct lowering.
    ///
    /// The `limit` field records the maximum allowed depth (default: 512).
    /// This typically occurs with deeply nested macro expansions in the
    /// Linux kernel source tree.
    RecursionLimitExceeded {
        /// The recursion limit that was exceeded.
        limit: u32,
    },

    /// A C type could not be converted to an IR type.
    ///
    /// This can occur for unsupported type constructs or when the target
    /// architecture does not support a particular type configuration.
    TypeConversionError {
        /// Description of the source C type that failed conversion.
        from: Box<CType>,
        /// The IR type that was expected or partially produced.
        to: Box<IrType>,
        /// Source location of the type reference.
        span: Span,
    },

    /// An AST construct is not yet supported by the lowering pipeline.
    ///
    /// This error is emitted when the lowering encounters a language
    /// feature or GCC extension that has not been implemented.
    UnsupportedConstruct {
        /// Human-readable description of the unsupported construct.
        description: String,
        /// Source location of the unsupported construct.
        span: Span,
    },

    /// An internal compiler error (bug) occurred during lowering.
    ///
    /// This should never happen in correct operation — it indicates a
    /// logic error in the lowering implementation itself.
    InternalError {
        /// Human-readable description of the internal error.
        message: String,
    },
}

impl std::fmt::Display for LoweringError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoweringError::RecursionLimitExceeded { limit } => {
                write!(
                    f,
                    "recursion limit exceeded during IR lowering (limit: {})",
                    limit
                )
            }
            LoweringError::TypeConversionError { from, span, .. } => {
                write!(
                    f,
                    "type conversion error at {}:{}: {:?}",
                    span.file_id, span.start, from
                )
            }
            LoweringError::UnsupportedConstruct { description, span } => {
                write!(
                    f,
                    "unsupported construct at {}:{}: {}",
                    span.file_id, span.start, description
                )
            }
            LoweringError::InternalError { message } => {
                write!(f, "internal compiler error during IR lowering: {}", message)
            }
        }
    }
}

impl std::error::Error for LoweringError {}

// ============================================================================
// SwitchContext — coordination for switch statement lowering
// ============================================================================

/// Context for switch statement lowering — tracks case labels and default.
///
/// Created by [`stmt_lowering`] when a `switch` statement is entered and
/// consumed when the switch body has been fully lowered. Case and default
/// labels register their target blocks here so the switch dispatch
/// instruction can be built after the entire switch body is processed.
pub struct SwitchContext {
    /// The exit block (merge point after the switch body).
    pub exit_block: BlockId,
    /// Collected case labels: (constant value, target block).
    ///
    /// The `i128` value accommodates all C integer constant expression
    /// widths including `unsigned long long`.
    pub cases: Vec<(i128, BlockId)>,
    /// The default block, if a `default:` label was encountered.
    pub default_block: Option<BlockId>,
    /// The switch condition value (the discriminant being switched on).
    pub condition: Value,
}

// ============================================================================
// LoweringContext — central shared context for all lowering submodules
// ============================================================================

/// The central context for AST-to-IR lowering (Phase 6).
///
/// Holds all mutable and immutable state needed during the lowering of a
/// single translation unit. This includes the IR builder, variable mappings,
/// diagnostic engine, target architecture info, and recursion depth tracking.
///
/// The context is created by [`lower_translation_unit`] and threaded through
/// all four submodules ([`decl_lowering`], [`stmt_lowering`], [`expr_lowering`],
/// [`asm_lowering`]) during function body lowering.
///
/// # Alloca-First Pattern
///
/// The `local_vars` map stores alloca pointers (SSA values representing
/// stack addresses) for every local variable in the current function.
/// All variable accesses go through load/store from these pointers,
/// and the subsequent mem2reg pass (Phase 7) promotes eligible allocas
/// to SSA virtual registers.
pub struct LoweringContext<'a> {
    /// The IR builder for constructing instructions and basic blocks.
    ///
    /// Tracks the current insertion point and assigns sequential SSA
    /// value numbers to result-producing instructions.
    pub builder: IrBuilder,

    /// The IR module being constructed (for global variable and string
    /// literal pool access).
    pub module: &'a mut IrModule,

    /// Mapping from variable names to their alloca `Value` (pointer to
    /// the variable's stack storage).
    ///
    /// Populated during function prologue lowering (alloca-first pattern)
    /// and queried during expression lowering for variable references.
    pub local_vars: FxHashMap<String, Value>,

    /// Symbol table from semantic analysis (Phase 5).
    ///
    /// Provides type information, linkage classification, storage class,
    /// definition tracking, and attribute data for all declared symbols.
    /// Read-only during IR lowering.
    pub symbol_table: &'a SymbolTable,

    /// Target architecture information.
    ///
    /// Provides pointer width, type sizes, endianness, and other
    /// architecture-dependent constants needed for correct type lowering
    /// and ABI compliance.
    pub target: &'a Target,

    /// Type builder for constructing complex IR types from C types.
    ///
    /// Provides struct layout computation with packed/aligned attribute
    /// support and flexible array member handling.
    pub type_builder: &'a TypeBuilder,

    /// Source map for span-to-location resolution in diagnostics.
    ///
    /// Used to produce human-readable error messages with filename,
    /// line, and column information.
    pub source_map: &'a SourceMap,

    /// Diagnostic engine for error, warning, and note reporting.
    ///
    /// Non-fatal issues discovered during lowering are emitted here
    /// rather than causing immediate failure.
    pub diagnostics: &'a mut DiagnosticEngine,

    /// Current function being lowered (`None` when lowering globals).
    ///
    /// Set by [`decl_lowering::lower_function_definition`] at the start
    /// of function lowering and cleared when the function is complete.
    pub current_function: Option<IrFunction>,

    /// Current recursion depth for nested construct lowering.
    ///
    /// Incremented by [`enter_recursion`](LoweringContext::enter_recursion)
    /// and decremented by [`exit_recursion`](LoweringContext::exit_recursion).
    pub recursion_depth: u32,

    /// Maximum allowed recursion depth (default: 512).
    ///
    /// Exceeding this limit produces a [`LoweringError::RecursionLimitExceeded`]
    /// error and halts lowering for the current construct.
    pub max_recursion_depth: u32,

    /// Label targets for goto resolution: label name → target [`BlockId`].
    ///
    /// Populated when `label:` statements are encountered during function
    /// body lowering. Used to resolve both backward and forward gotos.
    pub label_targets: FxHashMap<String, BlockId>,

    /// Forward-referenced gotos that need patching: label name →
    /// `Vec<(source_block, instruction_index)>`.
    ///
    /// When a `goto label;` is encountered before the label is defined,
    /// the goto is recorded here. When the label is later encountered,
    /// [`resolve_pending_gotos`](LoweringContext::resolve_pending_gotos)
    /// patches the branch instructions.
    pub pending_gotos: FxHashMap<String, Vec<(BlockId, usize)>>,

    /// String literal deduplication pool: byte content → SSA `Value`.
    ///
    /// Caches the SSA value (global string reference) for each unique
    /// byte sequence so that identical string literals share a single
    /// `.rodata` entry and a single IR value.
    pub string_literal_pool: FxHashMap<Vec<u8>, Value>,

    /// Break target stack for `break` statements in loops and switches.
    ///
    /// Each loop or switch body pushes its exit block onto this stack;
    /// `break` statements branch to the top entry. Popped when the
    /// construct's body lowering is complete.
    pub break_targets: Vec<BlockId>,

    /// Continue target stack for `continue` statements in loops.
    ///
    /// Each loop pushes its increment/condition block onto this stack;
    /// `continue` statements branch to the top entry. Popped when the
    /// loop body lowering is complete.
    pub continue_targets: Vec<BlockId>,

    /// Current switch context for `case`/`default` label lowering.
    ///
    /// Set when entering a `switch` body and cleared when the switch
    /// dispatch instruction is built after the body is fully lowered.
    pub switch_context: Option<SwitchContext>,
}

// ============================================================================
// LoweringContext — construction and lifecycle
// ============================================================================

impl<'a> LoweringContext<'a> {
    /// Create a new lowering context for processing a translation unit.
    ///
    /// The context is initialized with:
    /// - An empty IR builder (no insertion point set)
    /// - Empty variable/label/string maps
    /// - No current function
    /// - Recursion depth 0, max depth 512
    /// - No break/continue targets or switch context
    ///
    /// # Parameters
    ///
    /// - `module`: The IR module being constructed.
    /// - `symbol_table`: The symbol table from semantic analysis.
    /// - `target`: Target architecture information.
    /// - `type_builder`: Type construction utilities.
    /// - `source_map`: Source file tracking for diagnostics.
    /// - `diagnostics`: Error/warning reporting engine.
    pub fn new(
        module: &'a mut IrModule,
        symbol_table: &'a SymbolTable,
        target: &'a Target,
        type_builder: &'a TypeBuilder,
        source_map: &'a SourceMap,
        diagnostics: &'a mut DiagnosticEngine,
    ) -> Self {
        LoweringContext {
            builder: IrBuilder::new(),
            module,
            local_vars: FxHashMap::default(),
            symbol_table,
            target,
            type_builder,
            source_map,
            diagnostics,
            current_function: None,
            recursion_depth: 0,
            max_recursion_depth: DEFAULT_MAX_RECURSION_DEPTH,
            label_targets: FxHashMap::default(),
            pending_gotos: FxHashMap::default(),
            string_literal_pool: FxHashMap::default(),
            break_targets: Vec::new(),
            continue_targets: Vec::new(),
            switch_context: None,
        }
    }
}

// ============================================================================
// LoweringContext — recursion depth management
// ============================================================================

impl<'a> LoweringContext<'a> {
    /// Enter a new recursion level.
    ///
    /// Returns `Err(LoweringError::RecursionLimitExceeded)` if the depth
    /// limit (default: 512) would be exceeded. The caller should propagate
    /// this error to halt lowering of the current construct.
    ///
    /// # Example
    ///
    /// ```ignore
    /// ctx.enter_recursion()?;
    /// // ... lower nested construct ...
    /// ctx.exit_recursion();
    /// ```
    pub fn enter_recursion(&mut self) -> Result<(), LoweringError> {
        self.recursion_depth += 1;
        if self.recursion_depth > self.max_recursion_depth {
            return Err(LoweringError::RecursionLimitExceeded {
                limit: self.max_recursion_depth,
            });
        }
        Ok(())
    }

    /// Exit a recursion level.
    ///
    /// Decrements the recursion depth counter. Uses `saturating_sub` to
    /// prevent underflow if called without a matching `enter_recursion`.
    pub fn exit_recursion(&mut self) {
        self.recursion_depth = self.recursion_depth.saturating_sub(1);
    }
}

// ============================================================================
// LoweringContext — variable management (alloca-first pattern)
// ============================================================================

impl<'a> LoweringContext<'a> {
    /// Look up a local variable's alloca pointer by name.
    ///
    /// Returns `Some(Value)` if the variable has been allocated (i.e., an
    /// alloca instruction exists for it in the entry block), or `None` if
    /// the name is not found in the current function's local variable map.
    ///
    /// The returned `Value` is a pointer to the variable's stack storage.
    /// Use `build_load` to read and `build_store` to write the variable.
    #[inline]
    pub fn get_variable(&self, name: &str) -> Option<Value> {
        self.local_vars.get(name).copied()
    }

    /// Register a new local variable alloca.
    ///
    /// Associates `name` with the alloca `Value` (pointer to stack storage)
    /// in the local variable map. Called during the alloca-first prologue
    /// for each local variable and function parameter.
    ///
    /// If a variable with the same name already exists (e.g., from a
    /// shadowed outer scope), the new alloca replaces it in the map.
    #[inline]
    pub fn set_variable(&mut self, name: String, alloca: Value) {
        self.local_vars.insert(name, alloca);
    }
}

// ============================================================================
// LoweringContext — type conversion
// ============================================================================

impl<'a> LoweringContext<'a> {
    /// Convert a C type to an IR type using this context's target info.
    ///
    /// Delegates to [`IrType::from_ctype`] which handles all C11 type
    /// variants including target-dependent sizes (e.g., `long` is I32 on
    /// i686 but I64 on x86-64), opaque pointers, union-as-byte-array
    /// representation, and qualifier/typedef stripping.
    #[inline]
    pub fn ctype_to_irtype(&self, ctype: &CType) -> IrType {
        IrType::from_ctype(ctype, self.target)
    }
}

// ============================================================================
// LoweringContext — string literal interning
// ============================================================================

impl<'a> LoweringContext<'a> {
    /// Intern a string literal, returning an SSA `Value` referencing it.
    ///
    /// Deduplicates identical byte sequences: if the same bytes have been
    /// interned before, the cached `Value` is returned without creating a
    /// new string pool entry. Otherwise:
    ///
    /// 1. The byte sequence is added to the module's string literal pool
    ///    via [`IrModule::intern_string`].
    /// 2. A fresh SSA `Value` is allocated via the builder to serve as an
    ///    IR-level reference to the global string constant.
    /// 3. The (bytes → Value) mapping is cached for future lookups.
    ///
    /// # Parameters
    ///
    /// - `bytes`: Raw byte content of the string literal, including any
    ///   null terminator. Bytes are stored with PUA-decoded byte-exact
    ///   fidelity.
    ///
    /// # Returns
    ///
    /// An SSA `Value` representing the address of the interned string
    /// constant in the `.rodata` section.
    pub fn intern_string_literal(&mut self, bytes: &[u8]) -> Value {
        // Check the deduplication cache first.
        if let Some(&cached_val) = self.string_literal_pool.get(bytes) {
            return cached_val;
        }

        // Intern in the module's string pool (returns a u32 string ID).
        let _string_id = self.module.intern_string(bytes.to_vec());

        // Allocate a fresh SSA value as the IR-level reference to this
        // global string constant.
        let val = self.builder.fresh_value();

        // Cache for deduplication.
        self.string_literal_pool.insert(bytes.to_vec(), val);

        val
    }
}

// ============================================================================
// LoweringContext — label / goto management
// ============================================================================

impl<'a> LoweringContext<'a> {
    /// Register a label target block for goto resolution.
    ///
    /// Called by [`stmt_lowering`] when a labeled statement (`label:`) is
    /// encountered during function body lowering. The association between
    /// the label name and its target `BlockId` is stored in `label_targets`
    /// for use by subsequent goto statements.
    ///
    /// # Parameters
    ///
    /// - `name`: The label name (e.g., `"error_exit"`).
    /// - `target_block`: The `BlockId` of the basic block that the label
    ///   identifies.
    pub fn register_label(&mut self, name: &str, target_block: BlockId) {
        self.label_targets.insert(name.to_string(), target_block);
    }

    /// Resolve forward-referenced gotos to a label.
    ///
    /// Removes any pending forward-goto entries for the given label name
    /// from `pending_gotos`. The caller ([`stmt_lowering`]) is responsible
    /// for actually patching the branch instructions in the pending blocks
    /// to target the resolved label block.
    ///
    /// This method should be called immediately after [`register_label`]
    /// to handle any gotos that were emitted before the label was defined.
    ///
    /// # Parameters
    ///
    /// - `name`: The label name whose pending gotos should be resolved.
    ///
    /// # Returns
    ///
    /// The list of `(source_block, instruction_index)` pairs for forward
    /// gotos that referenced this label before it was defined. Returns
    /// an empty Vec if there were no pending gotos.
    pub fn resolve_pending_gotos(&mut self, name: &str) -> Vec<(BlockId, usize)> {
        self.pending_gotos.remove(name).unwrap_or_default()
    }
}

// ============================================================================
// LoweringContext — break / continue target management
// ============================================================================

impl<'a> LoweringContext<'a> {
    /// Push a break target (entering a loop or switch body).
    ///
    /// The `target` block is the merge point after the loop/switch body.
    /// `break` statements branch to the topmost break target.
    #[inline]
    pub fn push_break_target(&mut self, target: BlockId) {
        self.break_targets.push(target);
    }

    /// Pop a break target (exiting a loop or switch body).
    ///
    /// Returns the popped `BlockId`, or `None` if the stack was empty
    /// (which would indicate a compiler bug — `break` outside a loop/switch).
    #[inline]
    pub fn pop_break_target(&mut self) -> Option<BlockId> {
        self.break_targets.pop()
    }

    /// Push a continue target (entering a loop body).
    ///
    /// The `target` block is the loop's increment or condition block.
    /// `continue` statements branch to the topmost continue target.
    #[inline]
    pub fn push_continue_target(&mut self, target: BlockId) {
        self.continue_targets.push(target);
    }

    /// Pop a continue target (exiting a loop body).
    ///
    /// Returns the popped `BlockId`, or `None` if the stack was empty.
    #[inline]
    pub fn pop_continue_target(&mut self) -> Option<BlockId> {
        self.continue_targets.pop()
    }
}

// ============================================================================
// Standalone function: lower_translation_unit — PRIMARY ENTRY POINT
// ============================================================================

/// Lower an entire AST translation unit into an IR module.
///
/// This is the **primary entry point** for Phase 6, called by the
/// compilation pipeline driver. It processes all top-level declarations
/// in a two-pass strategy:
///
/// 1. **First pass — Global declarations**: Processes global variable
///    declarations/definitions, external function declarations, and
///    file-scope inline assembly. Typedefs and type-only declarations
///    are skipped (already handled by semantic analysis).
///
/// 2. **Second pass — Function definitions**: Lowers each function
///    definition using the alloca-first pattern. For each function:
///    - Create an `IrFunction` with an entry block
///    - Emit allocas for all parameters and local variables
///    - Lower the function body via [`stmt_lowering`]
///    - Add the completed function to the module
///
/// # Parameters
///
/// - `translation_unit`: The complete, semantically-validated AST.
/// - `symbol_table`: The symbol table from semantic analysis.
/// - `target`: Target architecture for type sizing and ABI.
/// - `type_builder`: Type construction utilities.
/// - `source_map`: Source file tracking for diagnostics.
/// - `diagnostics`: Error/warning reporting engine.
///
/// # Returns
///
/// An [`IrModule`] containing all lowered functions and global variables,
/// or a [`LoweringError`] if a fatal error occurred during lowering.
///
/// # Two-Pass Rationale
///
/// The two-pass strategy ensures that all global symbols (variables and
/// function declarations) are registered in the IR module before any
/// function body is lowered. This allows function bodies to reference
/// globals and call other functions by name without forward-declaration
/// ordering issues.
pub fn lower_translation_unit(
    translation_unit: &ast::TranslationUnit,
    symbol_table: &SymbolTable,
    target: &Target,
    type_builder: &TypeBuilder,
    source_map: &SourceMap,
    diagnostics: &mut DiagnosticEngine,
    interner: &Interner,
) -> Result<IrModule, LoweringError> {
    // Snapshot the interner content into thread-local storage so that all
    // name extraction helpers (extract_declarator_name, etc.) can resolve
    // Symbol handles to real C identifier strings without threading the
    // interner through every call.
    let snapshot: Vec<String> = (0..interner.len())
        .map(|idx| {
            let sym = crate::common::string_interner::Symbol::from_u32(idx as u32);
            interner.resolve(sym).to_string()
        })
        .collect();
    INTERNER_SNAPSHOT.with(|snap| {
        *snap.borrow_mut() = Some(snapshot);
    });

    // The symbol table from semantic analysis (Phase 5) provides type
    // information, linkage classification, storage class, and attribute
    // data for all declared symbols.  It is threaded into the
    // LoweringContext for use during declaration and expression lowering.
    let _ = symbol_table; // Used below via LoweringContext and name_table pre-population.

    // Derive a module name from the source map (first file, or "<unknown>").
    let module_name = source_map
        .get_filename(0)
        .map(|s| s.to_string())
        .unwrap_or_else(|| "<unknown>".to_string());

    let mut module = IrModule::new(module_name);

    // Use the interner snapshot as the name table.  The snapshot maps
    // each Symbol's as_u32() index to its interned string, so
    // resolve_sym(name_table, sym) can resolve any identifier — not just
    // top-level declaration names.  This ensures function parameters,
    // local variables, and all other identifiers are resolvable during
    // expression and statement lowering.
    let name_table: Vec<String> =
        INTERNER_SNAPSHOT.with(|snap| snap.borrow().as_ref().cloned().unwrap_or_default());

    // ====================================================================
    // Pass 1 — Global declarations, function prototypes, file-scope asm
    // ====================================================================
    for ext_decl in &translation_unit.declarations {
        match ext_decl {
            ast::ExternalDeclaration::Declaration(decl) => {
                // Skip typedef declarations — type aliases are already
                // resolved by the semantic analysis phase.
                if matches!(
                    decl.specifiers.storage_class,
                    Some(ast::StorageClass::Typedef)
                ) {
                    continue;
                }

                // Skip declarations with no declarators (tag-only struct/enum
                // definitions, _Static_assert, etc.).
                if decl.declarators.is_empty() {
                    continue;
                }

                // Determine if this declaration contains function-type
                // declarators (function prototypes).
                let has_func_shape = decl
                    .declarators
                    .iter()
                    .any(|id| declarator_has_function_shape(&id.declarator));

                // Function declarations (prototypes without bodies) produce
                // FunctionDeclaration entries in the IR module.
                // Non-function declarations produce GlobalVariable entries.
                if has_func_shape && !declaration_has_initializer(decl) {
                    decl_lowering::lower_function_declaration(
                        decl,
                        &mut module,
                        target,
                        diagnostics,
                        &name_table,
                    );
                } else {
                    decl_lowering::lower_global_variable(
                        decl,
                        &mut module,
                        target,
                        type_builder,
                        diagnostics,
                        &name_table,
                    );
                }
            }

            ast::ExternalDeclaration::AsmStatement(asm_stmt) => {
                // File-scope inline assembly: emit verbatim into the module.
                let template_text = extract_asm_template(asm_stmt);
                if !template_text.is_empty() {
                    module.add_inline_asm(template_text);
                }
            }

            // Function definitions are handled in pass 2; Empty is skipped.
            ast::ExternalDeclaration::FunctionDefinition(_) | ast::ExternalDeclaration::Empty => {}
        }
    }

    // ====================================================================
    // Pass 1.5 — Collect enum constants from the AST
    // ====================================================================
    // Enum constants are file-scope integer constants that must be
    // resolvable during expression lowering (e.g., `return RED;`).
    // We scan all top-level declarations for enum definitions and
    // build a name → value map.
    let enum_constants: FxHashMap<String, i128> = {
        let mut map = FxHashMap::default();
        for ext_decl in &translation_unit.declarations {
            match ext_decl {
                ast::ExternalDeclaration::Declaration(decl) => {
                    collect_enum_constants_from_specifiers(
                        &decl.specifiers.type_specifiers,
                        &name_table,
                        &mut map,
                    );
                }
                ast::ExternalDeclaration::FunctionDefinition(func_def) => {
                    collect_enum_constants_from_specifiers(
                        &func_def.specifiers.type_specifiers,
                        &name_table,
                        &mut map,
                    );
                }
                _ => {}
            }
        }
        map
    };

    // ====================================================================
    // Pass 1.6 — Collect struct/union type definitions from the AST
    // ====================================================================
    // Struct/union definitions at file scope (or embedded in declarations)
    // populate a registry mapping tag name → CType. This allows references
    // like `struct S s;` inside function bodies to resolve the full struct
    // layout, even when the AST specifier only contains the tag name.
    let struct_defs: FxHashMap<String, CType> = {
        let mut map = FxHashMap::default();
        for ext_decl in &translation_unit.declarations {
            match ext_decl {
                ast::ExternalDeclaration::Declaration(decl) => {
                    collect_struct_defs_from_specifiers(
                        &decl.specifiers.type_specifiers,
                        &name_table,
                        &mut map,
                    );
                }
                ast::ExternalDeclaration::FunctionDefinition(func_def) => {
                    collect_struct_defs_from_specifiers(
                        &func_def.specifiers.type_specifiers,
                        &name_table,
                        &mut map,
                    );
                }
                _ => {}
            }
        }
        map
    };

    // ====================================================================
    // Pass 2 — Function definitions (alloca-first pattern)
    // ====================================================================
    for ext_decl in &translation_unit.declarations {
        if let ast::ExternalDeclaration::FunctionDefinition(func_def) = ext_decl {
            decl_lowering::lower_function_definition(
                func_def,
                &mut module,
                target,
                type_builder,
                diagnostics,
                &name_table,
                &enum_constants,
                &struct_defs,
            );

            // Check if a fatal error was emitted during function lowering.
            // We continue lowering other functions to collect more diagnostics,
            // but the final result will reflect the error state.
        }
    }

    // If any fatal errors were emitted during lowering, report the failure
    // but still return the partially-constructed module (the diagnostic
    // engine retains all error details for the pipeline driver to display).
    if diagnostics.has_errors() {
        return Err(LoweringError::InternalError {
            message: format!(
                "IR lowering produced {} error(s)",
                diagnostics.error_count()
            ),
        });
    }

    Ok(module)
}

// ============================================================================
// Standalone function: ctype_to_irtype — C type to IR type conversion
// ============================================================================

/// Convert a C language type to an IR type, respecting target architecture
/// type sizes and alignment rules.
///
/// This is a convenience wrapper around [`IrType::from_ctype`] that
/// provides the canonical C-to-IR type mapping for the lowering pipeline.
///
/// # Conversion Highlights
///
/// - `long` → `I32` on i686 (ILP32), `I64` on x86-64/aarch64/riscv64 (LP64)
/// - `long double` → `F80` on x86 platforms (80-bit extended precision)
/// - `_Atomic(T)` → same representation as `T` (atomicity is in access, not storage)
/// - `_Complex T` → `[2 x T]` (array of two base-type elements)
/// - `union` → `[N x i8]` (byte array sized to the largest member)
/// - All pointers → `Ptr` (opaque pointer, LLVM-style)
/// - Typedefs → resolved to underlying type
/// - Qualifiers → stripped (no effect on IR data layout)
///
/// # Parameters
///
/// - `ctype`: The C type to convert.
/// - `target`: The target architecture for size resolution.
///
/// # Returns
///
/// The corresponding [`IrType`].
pub fn ctype_to_irtype(ctype: &CType, target: &Target) -> IrType {
    IrType::from_ctype(ctype, target)
}

// ============================================================================
// Standalone function: alignment_of — IR-level alignment for a C type
// ============================================================================

/// Get the IR-level alignment (in bytes) for a C type on the given target.
///
/// Wraps [`crate::common::types::alignof_ctype`] for convenient use during
/// IR lowering. The alignment is always a power of two and ≥ 1.
///
/// # Target-Dependent Behavior
///
/// - `long double`: 4-byte alignment on i686, 16-byte on 64-bit targets
/// - `long long`/`double`: 4-byte alignment on i686, 8-byte on 64-bit
/// - Packed structs: alignment 1 regardless of field types
/// - Explicit `__attribute__((aligned(N)))` overrides natural alignment
///
/// # Parameters
///
/// - `ctype`: The C type whose alignment to query.
/// - `target`: The target architecture.
///
/// # Returns
///
/// The required alignment in bytes.
#[inline]
pub fn alignment_of(ctype: &CType, target: &Target) -> usize {
    alignof_ctype(ctype, target)
}

// ============================================================================
// Standalone function: size_of — IR-level size for a C type
// ============================================================================

/// Get the IR-level size (in bytes) for a C type on the given target.
///
/// Wraps [`crate::common::types::sizeof_ctype`] for convenient use during
/// IR lowering.
///
/// # Key Sizes
///
/// - `void`: 1 (GCC extension for pointer arithmetic)
/// - `_Bool`: 1
/// - `long`: 4 on i686, 8 on LP64 targets
/// - `long double`: 12 on i686, 16 on other targets
/// - `_Complex T`: 2 × sizeof(T)
/// - Struct: includes inter-field padding and tail padding
/// - Union: max of all field sizes, padded to alignment
///
/// # Parameters
///
/// - `ctype`: The C type whose size to query.
/// - `target`: The target architecture.
///
/// # Returns
///
/// The size in bytes.
#[inline]
pub fn size_of(ctype: &CType, target: &Target) -> usize {
    sizeof_ctype(ctype, target)
}

// ============================================================================
// Private helpers — declaration classification
// ============================================================================

/// Check if a declarator has a function shape (is a function prototype).
///
/// A declarator has function shape if its direct declarator is the
/// `Function` variant, indicating a parameter list rather than a simple
/// identifier or array shape.
fn declarator_has_function_shape(decl: &ast::Declarator) -> bool {
    match &decl.direct {
        ast::DirectDeclarator::Function { .. } => true,
        ast::DirectDeclarator::Parenthesized(inner) => declarator_has_function_shape(inner),
        _ => false,
    }
}

/// Check if a declaration has any initializer (i.e., is a definition
/// rather than a pure declaration).
fn declaration_has_initializer(decl: &ast::Declaration) -> bool {
    decl.declarators.iter().any(|id| id.initializer.is_some())
}

/// Extract the template string from a file-scope `AsmStatement`.
///
/// For file-scope assembly, the template is the raw assembly text to
/// emit verbatim before any generated code. The template is stored
/// as `Vec<u8>` (raw bytes with PUA-decoded non-UTF-8 content) in the
/// AST `AsmStatement` node.
///
/// Returns the template as a `String` using lossy UTF-8 conversion,
/// which is appropriate for assembly directives that are typically
/// ASCII-only.
fn extract_asm_template(asm_stmt: &ast::AsmStatement) -> String {
    // AsmStatement.template is Vec<u8> (raw bytes, PUA-decoded).
    // For file-scope asm, the template contains the literal assembly
    // text without any operand substitutions.
    String::from_utf8_lossy(&asm_stmt.template).to_string()
}

// Dead functions removed: extract_declarator_name and extract_direct_declarator_name
// were superseded by the versions in decl_lowering.rs that accept a name_table parameter.

/// Scan type specifiers for `enum { ... }` definitions and collect
/// all enumerator name → integer value mappings into `out`.
///
/// This handles both explicit values (`RED = 5`) and implicit auto-
/// incrementing values.  If an enumerator has a constant expression,
/// we attempt simple integer literal evaluation; otherwise we fall
/// back to sequential assignment.
fn collect_enum_constants_from_specifiers(
    specs: &[ast::TypeSpecifier],
    name_table: &[String],
    out: &mut FxHashMap<String, i128>,
) {
    for spec in specs {
        if let ast::TypeSpecifier::Enum(enum_spec) = spec {
            if let Some(ref enumerators) = enum_spec.enumerators {
                let mut next_val: i128 = 0;
                for enumerator in enumerators {
                    // Try to evaluate the explicit value expression.
                    if let Some(ref val_expr) = enumerator.value {
                        if let Some(v) = try_eval_const_int_expr(val_expr) {
                            next_val = v;
                        }
                    }
                    // Resolve the enumerator name from the symbol.
                    let idx = enumerator.name.as_u32() as usize;
                    if idx < name_table.len() {
                        out.insert(name_table[idx].clone(), next_val);
                    }
                    next_val += 1;
                }
            }
        }
    }
}

/// Attempt to evaluate a simple integer constant expression from the AST.
///
/// Handles:
/// - Integer literals
/// - Unary minus on integer literals
/// - Simple binary arithmetic (+, -, *, /) on integer literals
///
/// Returns `None` if the expression is too complex for static evaluation.
/// Collect struct/union type definitions from type specifiers.
///
/// Scans top-level declaration specifiers for `struct` or `union`
/// definitions (those with a tag name AND member list) and registers
/// the full `CType` (with populated fields) in the output map, keyed
/// by tag name. This allows forward references like `struct S s;` in
/// function bodies to resolve the full struct layout.
fn collect_struct_defs_from_specifiers(
    specs: &[ast::TypeSpecifier],
    name_table: &[String],
    out: &mut FxHashMap<String, CType>,
) {
    for spec in specs {
        match spec {
            ast::TypeSpecifier::Struct(s) => {
                if let (Some(ref tag), Some(ref _members)) = (&s.tag, &s.members) {
                    let tag_name_idx = tag.as_u32() as usize;
                    if tag_name_idx < name_table.len() {
                        let tag_name = name_table[tag_name_idx].clone();
                        let fields = decl_lowering::extract_struct_union_fields(s);
                        let ctype = CType::Struct {
                            name: Some(tag_name.clone()),
                            fields,
                            packed: false,
                            aligned: None,
                        };
                        out.insert(tag_name, ctype);
                    }
                }
            }
            ast::TypeSpecifier::Union(u) => {
                if let (Some(ref tag), Some(ref _members)) = (&u.tag, &u.members) {
                    let tag_name_idx = tag.as_u32() as usize;
                    if tag_name_idx < name_table.len() {
                        let tag_name = name_table[tag_name_idx].clone();
                        let fields = decl_lowering::extract_struct_union_fields(u);
                        let ctype = CType::Union {
                            name: Some(tag_name.clone()),
                            fields,
                            packed: false,
                            aligned: None,
                        };
                        out.insert(tag_name, ctype);
                    }
                }
            }
            _ => {}
        }
    }
}

fn try_eval_const_int_expr(expr: &ast::Expression) -> Option<i128> {
    match expr {
        ast::Expression::IntegerLiteral { value, .. } => Some(*value as i128),
        ast::Expression::UnaryOp { op, operand, .. } => {
            match op {
                ast::UnaryOp::Negate => try_eval_const_int_expr(operand).map(|v| -v),
                ast::UnaryOp::Plus => try_eval_const_int_expr(operand),
                ast::UnaryOp::BitwiseNot => try_eval_const_int_expr(operand).map(|v| !v),
                _ => None,
            }
        }
        ast::Expression::Binary { op, left, right, .. } => {
            let l = try_eval_const_int_expr(left)?;
            let r = try_eval_const_int_expr(right)?;
            match op {
                ast::BinaryOp::Add => Some(l + r),
                ast::BinaryOp::Sub => Some(l - r),
                ast::BinaryOp::Mul => Some(l * r),
                ast::BinaryOp::Div => {
                    if r != 0 { Some(l / r) } else { None }
                }
                ast::BinaryOp::Mod => {
                    if r != 0 { Some(l % r) } else { None }
                }
                ast::BinaryOp::ShiftLeft => Some(l << (r as u32)),
                ast::BinaryOp::ShiftRight => Some(l >> (r as u32)),
                ast::BinaryOp::BitwiseAnd => Some(l & r),
                ast::BinaryOp::BitwiseOr => Some(l | r),
                ast::BinaryOp::BitwiseXor => Some(l ^ r),
                _ => None,
            }
        }
        // Parenthesized expression — just unwrap.
        ast::Expression::Parenthesized { inner, .. } => try_eval_const_int_expr(inner),
        _ => None,
    }
}
