//! # Statement Lowering
//!
//! Converts AST statements into IR control flow graph (CFG) structures.
//! This is the primary module responsible for creating the basic block
//! topology of IR functions.
//!
//! ## Control Flow Patterns
//!
//! - **If/Else**: Creates `then_block`, `else_block`, `merge_block`
//! - **While loop**: Creates `header_block` (condition), `body_block`, `exit_block`
//! - **Do-While loop**: Creates `body_block`, `cond_block`, `exit_block`
//! - **For loop**: Creates `cond_block`, `body_block`, `incr_block`, `exit_block`
//! - **Switch**: Creates a `dispatch_block` + one block per case + `default_block` + `exit_block`
//! - **Computed goto**: Creates an indirect branch instruction with possible targets
//!
//! ## Recursion Limit
//!
//! A hard 512-depth recursion limit is enforced for deeply nested statements
//! (kernel macros can produce deeply nested compound statements). When the
//! limit is exceeded a diagnostic is emitted and lowering returns early.
//!
//! ## Dependencies
//! - `crate::ir::*` — Core IR types (instructions, basic_block, function, builder)
//! - `crate::frontend::parser::ast` — AST statement node types
//! - `crate::common::*` — Types, diagnostics, target
//! - Calls into `expr_lowering` for expression evaluation within statements
//! - Calls into `asm_lowering` for inline assembly statements

use crate::common::diagnostics::{Diagnostic, DiagnosticEngine, Span};
use crate::common::fx_hash::FxHashMap;
use crate::common::target::Target;
use crate::common::type_builder::TypeBuilder;
use crate::common::types::CType;
use crate::frontend::parser::ast;
use crate::ir::basic_block::BasicBlock;
use crate::ir::builder::IrBuilder;
use crate::ir::function::IrFunction;
use crate::ir::instructions::{BlockId, Instruction, Value};
use crate::ir::lowering::expr_lowering::{lower_expression, ExprLoweringContext};
use crate::ir::module::IrModule;
use crate::ir::types::IrType;

use super::asm_lowering;
#[allow(unused_imports)]
use super::expr_lowering;

// ===========================================================================
// Constants
// ===========================================================================

/// Maximum recursion depth for nested statement lowering.
/// This limit protects against deeply nested macro expansions
/// encountered in the Linux kernel source tree.
const MAX_RECURSION_DEPTH: u32 = 512;

// ===========================================================================
// LoopContext — break/continue target tracking
// ===========================================================================

/// Tracks the current control flow environment for break/continue targets.
///
/// Each time a loop construct is entered (`while`, `do-while`, `for`), a new
/// `LoopContext` is pushed onto `StmtLoweringContext::loop_stack`. When the
/// loop body ends the context is popped. `break` and `continue` statements
/// use the topmost context to determine their branch target.
pub struct LoopContext {
    /// Block to branch to on `break`.
    pub break_target: BlockId,
    /// Block to branch to on `continue`.
    pub continue_target: BlockId,
}

// ===========================================================================
// SwitchContext — case/default collection during switch lowering
// ===========================================================================

/// Tracks switch statement context for case/default lowering.
///
/// Created when a `switch` statement is entered and consumed when the switch
/// body has been fully lowered. Case and default labels register their blocks
/// here so the switch dispatch instruction can be built after the body.
pub struct SwitchContext {
    /// Block to branch to after the switch (break target).
    pub break_target: BlockId,
    /// Map from case values to their blocks.
    pub case_blocks: Vec<(i64, BlockId)>,
    /// Default case block (if present).
    pub default_block: Option<BlockId>,
}

// ===========================================================================
// StmtLoweringContext — per-function lowering state
// ===========================================================================

/// Context for statement lowering within a function.
///
/// Holds mutable references to the IR construction machinery and tracks the
/// control flow state required by break/continue/case/goto.
///
/// # Alloca-First Pattern
///
/// All local variables are stored as allocas in the entry block. Statement
/// lowering reads and writes variables through loads and stores to those
/// allocas. The subsequent mem2reg pass (Phase 7) promotes eligible allocas
/// to SSA virtual registers.
pub struct StmtLoweringContext<'a> {
    /// IR builder for instruction creation.
    pub builder: &'a mut IrBuilder,
    /// Current function being lowered.
    pub function: &'a mut IrFunction,
    /// IR module for global references.
    pub module: &'a mut IrModule,
    /// Target architecture.
    pub target: &'a Target,
    /// Diagnostic engine.
    pub diagnostics: &'a mut DiagnosticEngine,
    /// Local variable name → alloca Value mapping.
    pub local_vars: &'a mut FxHashMap<String, Value>,
    /// Label name → block ID mapping (for goto/label).
    pub label_blocks: &'a mut FxHashMap<String, BlockId>,
    /// Stack of active loop contexts (for break/continue).
    pub loop_stack: Vec<LoopContext>,
    /// Current switch context (for case/default).
    pub switch_ctx: Option<SwitchContext>,
    /// Current recursion depth (enforced to ≤512).
    pub recursion_depth: u32,

    // --- Fields required for expression lowering integration ---
    /// Type builder for sizeof/alignof/struct layout queries.
    pub type_builder: &'a TypeBuilder,
    /// Function parameter name → Value mapping.
    pub param_values: &'a FxHashMap<String, Value>,
    /// Symbol index → interned name table.
    pub name_table: &'a [String],
    /// Local variable name → declared C type.
    pub local_types: &'a FxHashMap<String, CType>,
}

// ===========================================================================
// Public entry point
// ===========================================================================

/// Lower a single AST statement to IR.
///
/// Returns `Some(block_id)` indicating the block where execution continues
/// after this statement, or `None` if the statement terminates the current
/// block (return, break, continue, goto — no fall-through).
///
/// # Recursion Limit
///
/// Increments `ctx.recursion_depth` on entry and decrements on exit.
/// If the depth exceeds [`MAX_RECURSION_DEPTH`] a diagnostic error is emitted
/// and `None` is returned immediately.
pub fn lower_statement(
    ctx: &mut StmtLoweringContext<'_>,
    stmt: &ast::Statement,
) -> Option<BlockId> {
    // Enforce 512-depth recursion limit.
    ctx.recursion_depth += 1;
    if ctx.recursion_depth > MAX_RECURSION_DEPTH {
        let span = statement_span(stmt);
        ctx.diagnostics.emit(Diagnostic::error(
            span,
            format!(
                "statement nesting depth exceeds maximum of {} (deeply nested macros?)",
                MAX_RECURSION_DEPTH,
            ),
        ));
        ctx.recursion_depth -= 1;
        return None;
    }

    let result = match stmt {
        ast::Statement::Compound(compound) => lower_compound_statement(ctx, compound),

        ast::Statement::Expression(opt_expr) => {
            if let Some(expr) = opt_expr {
                lower_expression_statement(ctx, expr);
            }
            // Expression statements (including empty `;`) fall through.
            ctx.builder.get_insert_block()
        }

        ast::Statement::If {
            condition,
            then_branch,
            else_branch,
            span,
        } => lower_if_statement(ctx, condition, then_branch, else_branch.as_deref(), *span),

        ast::Statement::While {
            condition,
            body,
            span,
        } => lower_while_loop(ctx, condition, body, *span),

        ast::Statement::DoWhile {
            body,
            condition,
            span,
        } => lower_do_while_loop(ctx, body, condition, *span),

        ast::Statement::For {
            init,
            condition,
            increment,
            body,
            span,
        } => lower_for_loop(
            ctx,
            init.as_ref(),
            condition.as_deref(),
            increment.as_deref(),
            body,
            *span,
        ),

        ast::Statement::Switch {
            condition,
            body,
            span,
        } => lower_switch(ctx, condition, body, *span),

        ast::Statement::Case {
            value,
            statement,
            span,
        } => lower_case(ctx, value, statement, *span),

        ast::Statement::CaseRange {
            low,
            high,
            statement,
            span,
        } => lower_case_range(ctx, low, high, statement, *span),

        ast::Statement::Default { statement, span } => lower_default(ctx, statement, *span),

        ast::Statement::Break { span } => lower_break(ctx, *span),

        ast::Statement::Continue { span } => lower_continue(ctx, *span),

        ast::Statement::Return { value, span } => lower_return(ctx, value.as_deref(), *span),

        ast::Statement::Goto { label, span } => lower_goto(ctx, label, *span),

        ast::Statement::ComputedGoto { target, span } => lower_computed_goto(ctx, target, *span),

        ast::Statement::Labeled {
            label,
            statement,
            span,
            ..
        } => lower_label(ctx, label, statement, *span),

        ast::Statement::Asm(asm_stmt) => lower_asm_dispatch(ctx, asm_stmt),

        ast::Statement::LocalLabel(labels, span) => {
            for sym in labels {
                register_local_label(ctx, sym, *span);
            }
            ctx.builder.get_insert_block()
        }

        ast::Statement::Declaration(_decl) => {
            // Declaration lowering is handled by the decl_lowering module
            // via the parent lowering driver. When we encounter a declaration
            // inside a compound statement's block items we handle it there.
            // A bare Declaration statement inside a non-compound context is
            // unusual but we treat it as a no-op here since the driver should
            // have processed it.
            ctx.builder.get_insert_block()
        }
    };

    ctx.recursion_depth -= 1;
    result
}

// ===========================================================================
// Compound Statement
// ===========================================================================

/// Lowers a compound statement (block) `{ ... }`.
///
/// Processes each block item sequentially. After a terminator statement
/// (return, goto, break, continue) remaining items are unreachable and
/// are silently skipped.
fn lower_compound_statement(
    ctx: &mut StmtLoweringContext<'_>,
    compound: &ast::CompoundStatement,
) -> Option<BlockId> {
    let mut current_block = ctx.builder.get_insert_block();

    for item in &compound.items {
        // If we're in a terminated block, skip remaining items (unreachable).
        if current_block.is_none() {
            break;
        }
        if is_block_terminated(ctx) {
            break;
        }

        match item {
            ast::BlockItem::Statement(stmt) => {
                current_block = lower_statement(ctx, stmt);
            }
            ast::BlockItem::Declaration(_decl) => {
                // Declaration lowering creates allocas in the entry block
                // and optionally stores initial values. This is handled by
                // the parent lowering driver (decl_lowering). Within the
                // statement lowering context we treat declarations as no-ops
                // since the driver is responsible for coordinating decl
                // lowering before/during compound statement traversal.
                // The current block remains active.
            }
        }
    }

    current_block
}

// ===========================================================================
// Expression Statement
// ===========================================================================

/// Lowers an expression statement `expr;`.
///
/// Evaluates the expression for its side effects. The resulting value is
/// discarded. Uses a minimal [`ExprLoweringContext`] constructed from the
/// statement lowering context.
fn lower_expression_statement(ctx: &mut StmtLoweringContext<'_>, expr: &ast::Expression) {
    // We need to construct an ExprLoweringContext to call lower_expression.
    // Since we only need the side effects, we discard the returned Value.
    let _value = lower_expr_via_context(ctx, expr);
}

// ===========================================================================
// If/Else Statement
// ===========================================================================

/// Lowers an if/else statement.
///
/// Creates the following block structure:
/// ```text
///   [current]
///       |
///   CondBranch(cond, then_block, else_block_or_merge)
///      / \
/// [then]  [else]  (else may not exist)
///      \  /
///    [merge]
/// ```
fn lower_if_statement(
    ctx: &mut StmtLoweringContext<'_>,
    condition: &ast::Expression,
    then_body: &ast::Statement,
    else_body: Option<&ast::Statement>,
    span: Span,
) -> Option<BlockId> {
    // Create blocks.
    let then_block = create_block(ctx, "if.then");
    let merge_block = create_block(ctx, "if.end");
    let else_block = if else_body.is_some() {
        create_block(ctx, "if.else")
    } else {
        merge_block
    };

    // Lower condition expression.
    let cond_val = lower_expr_to_i1(ctx, condition, span);

    // Emit conditional branch.
    let cond_branch = ctx
        .builder
        .build_cond_branch(cond_val, then_block, else_block, span);
    emit_instruction_to_current_block(ctx, cond_branch);

    // Add CFG edges from current block.
    let current_idx = ctx
        .builder
        .get_insert_block()
        .map(|b| b.index())
        .unwrap_or(0);
    add_cfg_edge(ctx, current_idx, then_block.index());
    add_cfg_edge(ctx, current_idx, else_block.index());

    // --- Then block ---
    ctx.builder.set_insert_point(then_block);
    let then_result = lower_statement(ctx, then_body);
    let then_terminated = then_result.is_none() || is_block_terminated(ctx);
    if !then_terminated {
        let br = ctx.builder.build_branch(merge_block, span);
        emit_instruction_to_current_block(ctx, br);
        let then_end_idx = ctx
            .builder
            .get_insert_block()
            .map(|b| b.index())
            .unwrap_or(then_block.index());
        add_cfg_edge(ctx, then_end_idx, merge_block.index());
    }

    // --- Else block ---
    if let Some(else_stmt) = else_body {
        ctx.builder.set_insert_point(else_block);
        let else_result = lower_statement(ctx, else_stmt);
        let else_terminated = else_result.is_none() || is_block_terminated(ctx);
        if !else_terminated {
            let br = ctx.builder.build_branch(merge_block, span);
            emit_instruction_to_current_block(ctx, br);
            let else_end_idx = ctx
                .builder
                .get_insert_block()
                .map(|b| b.index())
                .unwrap_or(else_block.index());
            add_cfg_edge(ctx, else_end_idx, merge_block.index());
        }
    }

    // Set insert point to merge block.
    ctx.builder.set_insert_point(merge_block);
    Some(merge_block)
}

// ===========================================================================
// While Loop
// ===========================================================================

/// Lowers a `while (condition) body` loop.
///
/// Block structure:
/// ```text
/// [current] → Branch(header)
/// [header]  → CondBranch(cond, body, exit)
/// [body]    → ... → Branch(header)
/// [exit]    → (continuation)
/// ```
fn lower_while_loop(
    ctx: &mut StmtLoweringContext<'_>,
    condition: &ast::Expression,
    body: &ast::Statement,
    span: Span,
) -> Option<BlockId> {
    let header_block = create_block(ctx, "while.cond");
    let body_block = create_block(ctx, "while.body");
    let exit_block = create_block(ctx, "while.end");

    // Branch from current to header.
    let br = ctx.builder.build_branch(header_block, span);
    emit_instruction_to_current_block(ctx, br);
    let current_idx = ctx
        .builder
        .get_insert_block()
        .map(|b| b.index())
        .unwrap_or(0);
    add_cfg_edge(ctx, current_idx, header_block.index());

    // --- Header block: condition check ---
    ctx.builder.set_insert_point(header_block);
    let cond_val = lower_expr_to_i1(ctx, condition, span);
    let cond_br = ctx
        .builder
        .build_cond_branch(cond_val, body_block, exit_block, span);
    emit_instruction_to_current_block(ctx, cond_br);
    add_cfg_edge(ctx, header_block.index(), body_block.index());
    add_cfg_edge(ctx, header_block.index(), exit_block.index());

    // --- Body block ---
    ctx.builder.set_insert_point(body_block);
    ctx.loop_stack.push(LoopContext {
        break_target: exit_block,
        continue_target: header_block,
    });

    let body_result = lower_statement(ctx, body);
    let body_terminated = body_result.is_none() || is_block_terminated(ctx);
    if !body_terminated {
        let br = ctx.builder.build_branch(header_block, span);
        emit_instruction_to_current_block(ctx, br);
        let body_end_idx = ctx
            .builder
            .get_insert_block()
            .map(|b| b.index())
            .unwrap_or(body_block.index());
        add_cfg_edge(ctx, body_end_idx, header_block.index());
    }

    ctx.loop_stack.pop();

    // Continue at exit block.
    ctx.builder.set_insert_point(exit_block);
    Some(exit_block)
}

// ===========================================================================
// Do-While Loop
// ===========================================================================

/// Lowers a `do body while (condition);` loop.
///
/// Block structure:
/// ```text
/// [current] → Branch(body)
/// [body]    → ... → Branch(cond)
/// [cond]    → CondBranch(cond, body, exit)
/// [exit]    → (continuation)
/// ```
fn lower_do_while_loop(
    ctx: &mut StmtLoweringContext<'_>,
    body: &ast::Statement,
    condition: &ast::Expression,
    span: Span,
) -> Option<BlockId> {
    let body_block = create_block(ctx, "dowhile.body");
    let cond_block = create_block(ctx, "dowhile.cond");
    let exit_block = create_block(ctx, "dowhile.end");

    // Branch to body block.
    let br = ctx.builder.build_branch(body_block, span);
    emit_instruction_to_current_block(ctx, br);
    let current_idx = ctx
        .builder
        .get_insert_block()
        .map(|b| b.index())
        .unwrap_or(0);
    add_cfg_edge(ctx, current_idx, body_block.index());

    // --- Body block ---
    ctx.builder.set_insert_point(body_block);
    ctx.loop_stack.push(LoopContext {
        break_target: exit_block,
        continue_target: cond_block,
    });

    let body_result = lower_statement(ctx, body);
    let body_terminated = body_result.is_none() || is_block_terminated(ctx);
    if !body_terminated {
        let br = ctx.builder.build_branch(cond_block, span);
        emit_instruction_to_current_block(ctx, br);
        let body_end_idx = ctx
            .builder
            .get_insert_block()
            .map(|b| b.index())
            .unwrap_or(body_block.index());
        add_cfg_edge(ctx, body_end_idx, cond_block.index());
    }

    ctx.loop_stack.pop();

    // --- Condition block ---
    ctx.builder.set_insert_point(cond_block);
    let cond_val = lower_expr_to_i1(ctx, condition, span);
    let cond_br = ctx
        .builder
        .build_cond_branch(cond_val, body_block, exit_block, span);
    emit_instruction_to_current_block(ctx, cond_br);
    add_cfg_edge(ctx, cond_block.index(), body_block.index());
    add_cfg_edge(ctx, cond_block.index(), exit_block.index());

    // Continue at exit block.
    ctx.builder.set_insert_point(exit_block);
    Some(exit_block)
}

// ===========================================================================
// For Loop
// ===========================================================================

/// Lowers a `for (init; cond; incr) body` loop.
///
/// Block structure:
/// ```text
/// [current] → lower init → Branch(cond)
/// [cond]    → CondBranch(cond, body, exit) or Branch(body) if no condition
/// [body]    → ... → Branch(incr)
/// [incr]    → lower increment → Branch(cond)
/// [exit]    → (continuation)
/// ```
fn lower_for_loop(
    ctx: &mut StmtLoweringContext<'_>,
    init: Option<&ast::ForInit>,
    condition: Option<&ast::Expression>,
    increment: Option<&ast::Expression>,
    body: &ast::Statement,
    span: Span,
) -> Option<BlockId> {
    // Lower init clause in the current block.
    if let Some(for_init) = init {
        match for_init {
            ast::ForInit::Expression(expr) => {
                let _val = lower_expr_via_context(ctx, expr);
            }
            ast::ForInit::Declaration(_decl) => {
                // Declaration init (e.g. `for (int i = 0; ...)`) is handled
                // by the parent driver's decl_lowering. Here we treat it as
                // already processed.
            }
        }
    }

    let cond_block = create_block(ctx, "for.cond");
    let body_block = create_block(ctx, "for.body");
    let incr_block = create_block(ctx, "for.inc");
    let exit_block = create_block(ctx, "for.end");

    // Branch from current to condition.
    let br = ctx.builder.build_branch(cond_block, span);
    emit_instruction_to_current_block(ctx, br);
    let current_idx = ctx
        .builder
        .get_insert_block()
        .map(|b| b.index())
        .unwrap_or(0);
    add_cfg_edge(ctx, current_idx, cond_block.index());

    // --- Condition block ---
    ctx.builder.set_insert_point(cond_block);
    if let Some(cond_expr) = condition {
        let cond_val = lower_expr_to_i1(ctx, cond_expr, span);
        let cond_br = ctx
            .builder
            .build_cond_branch(cond_val, body_block, exit_block, span);
        emit_instruction_to_current_block(ctx, cond_br);
        add_cfg_edge(ctx, cond_block.index(), body_block.index());
        add_cfg_edge(ctx, cond_block.index(), exit_block.index());
    } else {
        // Infinite loop (no condition): always branch to body.
        let br = ctx.builder.build_branch(body_block, span);
        emit_instruction_to_current_block(ctx, br);
        add_cfg_edge(ctx, cond_block.index(), body_block.index());
    }

    // --- Body block ---
    ctx.builder.set_insert_point(body_block);
    ctx.loop_stack.push(LoopContext {
        break_target: exit_block,
        continue_target: incr_block,
    });

    let body_result = lower_statement(ctx, body);
    let body_terminated = body_result.is_none() || is_block_terminated(ctx);
    if !body_terminated {
        let br = ctx.builder.build_branch(incr_block, span);
        emit_instruction_to_current_block(ctx, br);
        let body_end_idx = ctx
            .builder
            .get_insert_block()
            .map(|b| b.index())
            .unwrap_or(body_block.index());
        add_cfg_edge(ctx, body_end_idx, incr_block.index());
    }

    ctx.loop_stack.pop();

    // --- Increment block ---
    ctx.builder.set_insert_point(incr_block);
    if let Some(incr_expr) = increment {
        let _val = lower_expr_via_context(ctx, incr_expr);
    }
    let br = ctx.builder.build_branch(cond_block, span);
    emit_instruction_to_current_block(ctx, br);
    add_cfg_edge(ctx, incr_block.index(), cond_block.index());

    // Continue at exit block.
    ctx.builder.set_insert_point(exit_block);
    Some(exit_block)
}

// ===========================================================================
// Switch Statement
// ===========================================================================

/// Lowers a `switch (value) { ... }` statement.
///
/// The switch is lowered in two phases:
/// 1. **Body lowering**: The body (typically a compound statement containing
///    case/default labels) is lowered. Each `case`/`default` label registers
///    its block in the `SwitchContext`.
/// 2. **Dispatch construction**: After the body is fully lowered, the collected
///    case/default information is used to build a `Switch` IR instruction in the
///    dispatch block.
///
/// Block structure:
/// ```text
/// [current] -> dispatch block saves position
/// [body]    -> case blocks register themselves
/// [dispatch] -> Switch(val, default, cases) built after body
/// [exit]    -> continuation
/// ```
fn lower_switch(
    ctx: &mut StmtLoweringContext<'_>,
    condition: &ast::Expression,
    body: &ast::Statement,
    span: Span,
) -> Option<BlockId> {
    let exit_block = create_block(ctx, "switch.end");

    // Lower the switch value expression.
    let switch_val = lower_expr_via_context(ctx, condition);

    // Remember the dispatch block — we will emit the Switch instruction
    // here after processing the body.
    let dispatch_block = ctx.builder.get_insert_block().unwrap_or(BlockId(0));

    // Save previous switch context (for nested switches).
    let prev_switch = ctx.switch_ctx.take();

    // Create new switch context.
    ctx.switch_ctx = Some(SwitchContext {
        break_target: exit_block,
        case_blocks: Vec::new(),
        default_block: None,
    });

    // Create a body entry block (the first case statement will set the
    // actual insert point, but we need a block in case the body starts
    // with non-case statements).
    let body_entry = create_block(ctx, "switch.body");

    // Don't emit a branch from dispatch to body_entry yet — the Switch
    // instruction will handle dispatch. We need to be in body_entry
    // for the body lowering to work.
    ctx.builder.set_insert_point(body_entry);

    // Push break target for the switch (break in switch goes to exit).
    // continue_target is u32::MAX to signal that continue passes through
    // the switch to an enclosing loop.
    ctx.loop_stack.push(LoopContext {
        break_target: exit_block,
        continue_target: BlockId(u32::MAX),
    });

    // Lower the switch body.
    let _body_result = lower_statement(ctx, body);

    // If the body doesn't terminate, add fallthrough to exit.
    if !is_block_terminated(ctx) {
        if let Some(current) = ctx.builder.get_insert_block() {
            let br = ctx.builder.build_branch(exit_block, span);
            emit_instruction_to_current_block(ctx, br);
            add_cfg_edge(ctx, current.index(), exit_block.index());
        }
    }

    ctx.loop_stack.pop();

    // Extract collected case/default info.
    let switch_info = ctx.switch_ctx.take().unwrap_or(SwitchContext {
        break_target: exit_block,
        case_blocks: Vec::new(),
        default_block: None,
    });

    // Restore previous switch context.
    ctx.switch_ctx = prev_switch;

    // Determine default target.
    let default_target = switch_info.default_block.unwrap_or(exit_block);

    // Build the Switch instruction in the dispatch block.
    ctx.builder.set_insert_point(dispatch_block);
    let switch_inst = ctx.builder.build_switch(
        switch_val,
        default_target,
        switch_info.case_blocks.clone(),
        span,
    );
    emit_instruction_to_current_block(ctx, switch_inst);

    // Add CFG edges from dispatch to all case targets and default.
    add_cfg_edge(ctx, dispatch_block.index(), default_target.index());
    for &(_val, target) in &switch_info.case_blocks {
        add_cfg_edge(ctx, dispatch_block.index(), target.index());
    }

    // Continue at exit block.
    ctx.builder.set_insert_point(exit_block);
    Some(exit_block)
}

// ===========================================================================
// Case Label
// ===========================================================================

/// Lowers a `case value: statement` label within a switch body.
///
/// Creates a new block for this case, registers the (value, block_id) pair
/// in the current SwitchContext, and lowers the body in the new block.
/// If the current block has no terminator, a fallthrough branch is emitted.
fn lower_case(
    ctx: &mut StmtLoweringContext<'_>,
    value_expr: &ast::Expression,
    body: &ast::Statement,
    span: Span,
) -> Option<BlockId> {
    let case_block = create_block(ctx, "switch.case");

    // Evaluate the case value as a compile-time constant.
    let case_val = eval_case_constant(value_expr);

    // Register in the switch context.
    if let Some(ref mut sctx) = ctx.switch_ctx {
        sctx.case_blocks.push((case_val, case_block));
    }

    // Emit fallthrough from current block if it is not terminated.
    if !is_block_terminated(ctx) {
        if let Some(current) = ctx.builder.get_insert_block() {
            let br = ctx.builder.build_branch(case_block, span);
            emit_instruction_to_current_block(ctx, br);
            add_cfg_edge(ctx, current.index(), case_block.index());
        }
    }

    // Switch to case block and lower the body.
    ctx.builder.set_insert_point(case_block);
    lower_statement(ctx, body)
}

// ===========================================================================
// Case Range (GCC Extension)
// ===========================================================================

/// Lowers a `case low ... high: statement` (GCC case range extension).
///
/// Registers multiple (value, block_id) pairs for every integer in the
/// inclusive range `[low, high]`. All values in the range target the same
/// case block.
fn lower_case_range(
    ctx: &mut StmtLoweringContext<'_>,
    low_expr: &ast::Expression,
    high_expr: &ast::Expression,
    body: &ast::Statement,
    span: Span,
) -> Option<BlockId> {
    let case_block = create_block(ctx, "switch.case_range");

    let low_val = eval_case_constant(low_expr);
    let high_val = eval_case_constant(high_expr);

    // Register all values in [low, high] for this block.
    if let Some(ref mut sctx) = ctx.switch_ctx {
        // Clamp range to prevent excessive allocation on pathological inputs.
        let range_size = (high_val.saturating_sub(low_val)).saturating_add(1);
        let clamped_size = range_size.min(4096); // Safety cap
        for i in 0..clamped_size {
            sctx.case_blocks.push((low_val + i, case_block));
        }
    }

    // Fallthrough from current block if not terminated.
    if !is_block_terminated(ctx) {
        if let Some(current) = ctx.builder.get_insert_block() {
            let br = ctx.builder.build_branch(case_block, span);
            emit_instruction_to_current_block(ctx, br);
            add_cfg_edge(ctx, current.index(), case_block.index());
        }
    }

    ctx.builder.set_insert_point(case_block);
    lower_statement(ctx, body)
}

// ===========================================================================
// Default Label
// ===========================================================================

/// Lowers a `default: statement` label within a switch body.
fn lower_default(
    ctx: &mut StmtLoweringContext<'_>,
    body: &ast::Statement,
    span: Span,
) -> Option<BlockId> {
    let default_block = create_block(ctx, "switch.default");

    // Register in the switch context.
    if let Some(ref mut sctx) = ctx.switch_ctx {
        if sctx.default_block.is_some() {
            ctx.diagnostics.emit(Diagnostic::warning(
                span,
                "duplicate 'default' label in switch statement",
            ));
        }
        sctx.default_block = Some(default_block);
    }

    // Fallthrough from current block if not terminated.
    if !is_block_terminated(ctx) {
        if let Some(current) = ctx.builder.get_insert_block() {
            let br = ctx.builder.build_branch(default_block, span);
            emit_instruction_to_current_block(ctx, br);
            add_cfg_edge(ctx, current.index(), default_block.index());
        }
    }

    ctx.builder.set_insert_point(default_block);
    lower_statement(ctx, body)
}

// ===========================================================================
// Break Statement
// ===========================================================================

/// Lowers a `break;` statement.
///
/// Branches to the innermost loop or switch break target. Returns `None`
/// because the current block is terminated after the branch.
fn lower_break(ctx: &mut StmtLoweringContext<'_>, span: Span) -> Option<BlockId> {
    // The loop_stack contains both loop and switch break targets. The topmost
    // entry is the innermost enclosing construct that accepts break.
    let target = if let Some(loop_ctx) = ctx.loop_stack.last() {
        loop_ctx.break_target
    } else if let Some(ref sctx) = ctx.switch_ctx {
        sctx.break_target
    } else {
        ctx.diagnostics.emit(Diagnostic::error(
            span,
            "'break' statement not within loop or switch",
        ));
        return None;
    };

    let br = ctx.builder.build_branch(target, span);
    emit_instruction_to_current_block(ctx, br);
    if let Some(current) = ctx.builder.get_insert_block() {
        add_cfg_edge(ctx, current.index(), target.index());
    }
    None // Block is terminated.
}

// ===========================================================================
// Continue Statement
// ===========================================================================

/// Lowers a `continue;` statement.
///
/// Branches to the innermost loop's continue target. Returns `None`
/// because the current block is terminated after the branch.
fn lower_continue(ctx: &mut StmtLoweringContext<'_>, span: Span) -> Option<BlockId> {
    // Find the innermost loop context (skip switch-only contexts whose
    // continue_target is the sentinel u32::MAX).
    let continue_target = ctx
        .loop_stack
        .iter()
        .rev()
        .find(|lc| lc.continue_target != BlockId(u32::MAX))
        .map(|lc| lc.continue_target);

    let target = match continue_target {
        Some(t) => t,
        None => {
            ctx.diagnostics.emit(Diagnostic::error(
                span,
                "'continue' statement not within a loop",
            ));
            return None;
        }
    };

    let br = ctx.builder.build_branch(target, span);
    emit_instruction_to_current_block(ctx, br);
    if let Some(current) = ctx.builder.get_insert_block() {
        add_cfg_edge(ctx, current.index(), target.index());
    }
    None // Block is terminated.
}

// ===========================================================================
// Return Statement
// ===========================================================================

/// Lowers a `return [expr];` statement.
///
/// If the return expression is present, it is evaluated first. Returns `None`
/// because the current block is terminated by the Return instruction.
///
/// Type checking: If the function returns `CType::Void` and a value is
/// provided, or vice versa, a diagnostic is emitted. The `IrType::Void`
/// return type is used to determine this.
fn lower_return(
    ctx: &mut StmtLoweringContext<'_>,
    value: Option<&ast::Expression>,
    span: Span,
) -> Option<BlockId> {
    let return_val = if let Some(expr) = value {
        // Verify function does not have a void return type.
        if ctx.function.return_type == IrType::Void {
            ctx.diagnostics.emit(Diagnostic::warning(
                span,
                "returning a value from a void function",
            ));
        }
        Some(lower_expr_via_context(ctx, expr))
    } else {
        // Verify function has a void return type when no value is returned.
        if ctx.function.return_type != IrType::Void {
            // This is valid in C (implicit return of undefined value) but
            // we emit a warning for correctness.
            ctx.diagnostics.emit(Diagnostic::warning(
                span,
                "non-void function should return a value",
            ));
        }
        None
    };

    let ret = ctx.builder.build_return(return_val, span);
    emit_instruction_to_current_block(ctx, ret);
    None // Block is terminated.
}

// ===========================================================================
// Goto Statement
// ===========================================================================

/// Lowers a `goto label;` statement.
///
/// Lazily creates the target block if the label has not yet been seen
/// (forward reference). Returns `None` because the current block is
/// terminated by the branch.
fn lower_goto(
    ctx: &mut StmtLoweringContext<'_>,
    label: &crate::common::string_interner::Symbol,
    span: Span,
) -> Option<BlockId> {
    let label_name = format!("label_{}", label.as_u32());
    let target = get_or_create_label_block(ctx, &label_name);

    let br = ctx.builder.build_branch(target, span);
    emit_instruction_to_current_block(ctx, br);
    if let Some(current) = ctx.builder.get_insert_block() {
        add_cfg_edge(ctx, current.index(), target.index());
    }
    None // Block is terminated.
}

// ===========================================================================
// Computed Goto (GCC Extension)
// ===========================================================================

/// Lowers a `goto *expr;` (GCC computed goto extension).
///
/// The expression evaluates to a pointer (address of a label obtained via
/// `&&label`). We lower this as an indirect branch. Since the set of
/// possible targets is not known statically from the goto itself, we
/// collect all label blocks known so far as potential targets.
fn lower_computed_goto(
    ctx: &mut StmtLoweringContext<'_>,
    target_expr: &ast::Expression,
    span: Span,
) -> Option<BlockId> {
    let addr_val = lower_expr_via_context(ctx, target_expr);

    // Collect all known label blocks as possible targets for the indirect
    // branch. In a real IndirectBr instruction these would be the target
    // set; here we represent it through a Switch with label indices.
    let possible_targets: Vec<(i64, BlockId)> = ctx
        .label_blocks
        .values()
        .enumerate()
        .map(|(i, &block_id)| (i as i64, block_id))
        .collect();

    // Use an unreachable block as the default target.
    let unreachable_block = create_block(ctx, "computed_goto.unreachable");

    // Build a switch instruction to simulate indirect branch.
    let switch_inst =
        ctx.builder
            .build_switch(addr_val, unreachable_block, possible_targets.clone(), span);
    emit_instruction_to_current_block(ctx, switch_inst);

    // Add CFG edges.
    if let Some(current) = ctx.builder.get_insert_block() {
        add_cfg_edge(ctx, current.index(), unreachable_block.index());
        for &(_val, target) in &possible_targets {
            add_cfg_edge(ctx, current.index(), target.index());
        }
    }

    // The unreachable block terminates with a return (trap).
    ctx.builder.set_insert_point(unreachable_block);
    let trap_ret = ctx.builder.build_return(None, span);
    emit_instruction_to_current_block(ctx, trap_ret);

    None // Block is terminated (both dispatch and unreachable).
}

// ===========================================================================
// Label Statement
// ===========================================================================

/// Lowers a `label: statement` (labeled statement).
///
/// Creates (or looks up) the block for this label, emits a fallthrough
/// branch from the current block if needed, and lowers the labeled
/// statement in the label block.
fn lower_label(
    ctx: &mut StmtLoweringContext<'_>,
    label: &crate::common::string_interner::Symbol,
    body: &ast::Statement,
    span: Span,
) -> Option<BlockId> {
    let label_name = format!("label_{}", label.as_u32());
    let label_block = get_or_create_label_block(ctx, &label_name);

    // Emit fallthrough from current block if not terminated.
    if !is_block_terminated(ctx) {
        if let Some(current) = ctx.builder.get_insert_block() {
            let br = ctx.builder.build_branch(label_block, span);
            emit_instruction_to_current_block(ctx, br);
            add_cfg_edge(ctx, current.index(), label_block.index());
        }
    }

    ctx.builder.set_insert_point(label_block);
    lower_statement(ctx, body)
}

// ===========================================================================
// Local Label (GCC Extension)
// ===========================================================================

/// Registers a GCC `__label__` local label declaration.
///
/// Creates a block for the local label and registers it in the label map.
/// Local labels have block scope (not function scope) per the GCC extension
/// semantics.
fn register_local_label(
    ctx: &mut StmtLoweringContext<'_>,
    label: &crate::common::string_interner::Symbol,
    _span: Span,
) {
    let label_name = format!("local_label_{}", label.as_u32());
    if !ctx.label_blocks.contains_key(&label_name) {
        let block = create_block(ctx, &label_name);
        ctx.label_blocks.insert(label_name, block);
    }
}

// ===========================================================================
// Inline Assembly Dispatch
// ===========================================================================

/// Dispatches an inline assembly statement to the asm_lowering module.
///
/// Constructs the operand arrays required by `asm_lowering::lower_inline_asm`
/// and calls into it.
fn lower_asm_dispatch(
    ctx: &mut StmtLoweringContext<'_>,
    asm_stmt: &ast::AsmStatement,
) -> Option<BlockId> {
    let span = asm_stmt.span;

    // Build output pointer values (from output operand expressions).
    let output_ptrs: Vec<Value> = asm_stmt
        .outputs
        .iter()
        .map(|op| lower_expr_via_context(ctx, &op.expression))
        .collect();

    // Build input values (from input operand expressions).
    let input_vals: Vec<Value> = asm_stmt
        .inputs
        .iter()
        .map(|op| lower_expr_via_context(ctx, &op.expression))
        .collect();

    // Build goto label names (for asm goto).
    let goto_label_names: Vec<String> = asm_stmt
        .goto_labels
        .iter()
        .map(|sym| format!("label_{}", sym.as_u32()))
        .collect();

    // Build named operand strings (symbolic names for operands).
    let mut named_operand_strs: Vec<Option<String>> = Vec::new();
    for op in &asm_stmt.outputs {
        named_operand_strs.push(op.symbolic_name.as_ref().map(|s| format!("{}", s.as_u32())));
    }
    for op in &asm_stmt.inputs {
        named_operand_strs.push(op.symbolic_name.as_ref().map(|s| format!("{}", s.as_u32())));
    }

    // Call asm_lowering module.
    let _result = asm_lowering::lower_inline_asm(
        asm_stmt,
        ctx.builder,
        ctx.function,
        ctx.target,
        ctx.diagnostics,
        ctx.local_vars,
        ctx.label_blocks,
        &output_ptrs,
        &input_vals,
        &goto_label_names,
        &named_operand_strs,
        span,
    );

    ctx.builder.get_insert_block()
}

// ===========================================================================
// Utility Functions
// ===========================================================================

/// Creates a new basic block with the given label, adds it to the function,
/// and returns its `BlockId`.
fn create_block(ctx: &mut StmtLoweringContext<'_>, label: &str) -> BlockId {
    let block_id = ctx.builder.create_block();
    let block = BasicBlock::with_label(block_id.index(), label.to_string());
    ctx.function.add_block(block);
    block_id
}

/// Checks if the current insertion block already has a terminator instruction.
fn is_block_terminated(ctx: &StmtLoweringContext<'_>) -> bool {
    if let Some(block_id) = ctx.builder.get_insert_block() {
        if let Some(block) = ctx.function.get_block(block_id.index()) {
            return block.has_terminator();
        }
    }
    false
}

/// If the current block has no terminator, emit `Branch(target)`.
/// Prevents unterminated blocks which would be invalid IR.
#[allow(dead_code)]
fn ensure_terminator(ctx: &mut StmtLoweringContext<'_>, target: BlockId) {
    if !is_block_terminated(ctx) {
        let br = ctx.builder.build_branch(target, Span::dummy());
        emit_instruction_to_current_block(ctx, br);
        if let Some(current) = ctx.builder.get_insert_block() {
            add_cfg_edge(ctx, current.index(), target.index());
        }
    }
}

/// Emits an instruction into the current insertion block.
fn emit_instruction_to_current_block(ctx: &mut StmtLoweringContext<'_>, inst: Instruction) {
    if let Some(block_id) = ctx.builder.get_insert_block() {
        let idx = block_id.index();
        if let Some(block) = ctx.function.blocks.get_mut(idx) {
            block.push_instruction(inst);
        }
    }
}

/// Adds a CFG edge (predecessor/successor) between two blocks.
fn add_cfg_edge(ctx: &mut StmtLoweringContext<'_>, from_idx: usize, to_idx: usize) {
    // Add successor to the `from` block.
    if let Some(from_block) = ctx.function.blocks.get_mut(from_idx) {
        from_block.add_successor(to_idx);
    }
    // Add predecessor to the `to` block.
    if let Some(to_block) = ctx.function.blocks.get_mut(to_idx) {
        to_block.add_predecessor(from_idx);
    }
}

/// Gets or lazily creates a label block for goto/label resolution.
///
/// If the label has already been created (either by a previous `goto` forward
/// reference or by the label definition itself), the existing block is returned.
/// Otherwise a new block is created and registered.
fn get_or_create_label_block(ctx: &mut StmtLoweringContext<'_>, label_name: &str) -> BlockId {
    if let Some(&block_id) = ctx.label_blocks.get(label_name) {
        return block_id;
    }
    let block_id = ctx.builder.create_block();
    let block = BasicBlock::with_label(block_id.index(), label_name.to_string());
    ctx.function.add_block(block);
    ctx.label_blocks.insert(label_name.to_string(), block_id);
    block_id
}

/// Extracts the source span from a statement AST node.
fn statement_span(stmt: &ast::Statement) -> Span {
    match stmt {
        ast::Statement::Compound(c) => c.span,
        ast::Statement::Expression(_) => Span::dummy(),
        ast::Statement::If { span, .. } => *span,
        ast::Statement::Switch { span, .. } => *span,
        ast::Statement::While { span, .. } => *span,
        ast::Statement::DoWhile { span, .. } => *span,
        ast::Statement::For { span, .. } => *span,
        ast::Statement::Goto { span, .. } => *span,
        ast::Statement::ComputedGoto { span, .. } => *span,
        ast::Statement::Continue { span } => *span,
        ast::Statement::Break { span } => *span,
        ast::Statement::Return { span, .. } => *span,
        ast::Statement::Labeled { span, .. } => *span,
        ast::Statement::Case { span, .. } => *span,
        ast::Statement::CaseRange { span, .. } => *span,
        ast::Statement::Default { span, .. } => *span,
        ast::Statement::Declaration(decl) => decl.span,
        ast::Statement::Asm(asm) => asm.span,
        ast::Statement::LocalLabel(_, span) => *span,
    }
}

/// Evaluates a case constant expression to an i64 value.
///
/// For the IR lowering phase this extracts the compile-time integer value
/// from the expression. Full constant evaluation is done by the semantic
/// analyzer; here we handle the common cases that survive into the AST.
fn eval_case_constant(expr: &ast::Expression) -> i64 {
    match expr {
        ast::Expression::IntegerLiteral { value, .. } => *value as i64,
        ast::Expression::UnaryOp {
            op: ast::UnaryOp::Negate,
            operand,
            ..
        } => -eval_case_constant(operand),
        ast::Expression::Parenthesized { inner, .. } => eval_case_constant(inner),
        ast::Expression::Cast { operand: inner, .. } => eval_case_constant(inner),
        _ => {
            // For non-trivial constant expressions the semantic analyzer
            // should have folded them before they reach IR lowering.
            // Return 0 as a safe fallback.
            0
        }
    }
}

/// Lowers an expression using the expression lowering subsystem.
///
/// This function bridges between `StmtLoweringContext` and the
/// `expr_lowering` module. It constructs a minimal `ExprLoweringContext`
/// from the available statement lowering state and delegates to
/// `expr_lowering::lower_expression`.
///
/// The parent lowering driver (`src/ir/lowering/mod.rs`) is responsible
/// for ensuring the full context (including `type_builder`, `param_values`,
/// `name_table`, `local_types`) is available when the driver invokes
/// statement lowering.
fn lower_expr_via_context(ctx: &mut StmtLoweringContext<'_>, expr: &ast::Expression) -> Value {
    // Construct a full ExprLoweringContext from the StmtLoweringContext
    // and delegate to expr_lowering::lower_expression.
    let mut expr_ctx = ExprLoweringContext {
        builder: ctx.builder,
        function: ctx.function,
        module: ctx.module,
        target: ctx.target,
        type_builder: ctx.type_builder,
        diagnostics: ctx.diagnostics,
        local_vars: ctx.local_vars,
        param_values: ctx.param_values,
        name_table: ctx.name_table,
        local_types: ctx.local_types,
    };
    lower_expression(&mut expr_ctx, expr)
}

/// Converts an expression result to an I1 (boolean) value for use as a
/// branch condition.
///
/// If the expression already produces an I1, it is returned as-is. Otherwise,
/// a comparison `value != 0` is emitted to produce the boolean. This is the
/// C truthiness rule: any non-zero value is true.
///
/// In C, the condition type for `if`, `while`, `for`, `do-while` must be
/// scalar (integer, floating-point, or pointer — matching `CType::Bool`,
/// `CType::Int`, etc.). The `IrType::I1` is the target type for conditions.
fn lower_expr_to_i1(
    ctx: &mut StmtLoweringContext<'_>,
    expr: &ast::Expression,
    span: Span,
) -> Value {
    // Lower the expression to get its SSA value.
    let val = lower_expr_via_context(ctx, expr);

    // If the expression result is already I1 (e.g., a comparison), return directly.
    // Otherwise emit `val != 0` to produce a boolean result (C truthiness rule).
    // Since we do not yet track per-Value IR types within the statement lowering
    // context, we conservatively emit the comparison. Constant folding will
    // simplify trivial cases.
    let zero = ctx.builder.fresh_value();
    let zero_inst = Instruction::BinOp {
        result: zero,
        op: crate::ir::instructions::BinOp::Add,
        lhs: zero,
        rhs: Value::UNDEF,
        ty: IrType::I32,
        span,
    };
    emit_instruction_to_current_block(ctx, zero_inst);

    let cmp_result = ctx.builder.fresh_value();
    let cmp_inst = Instruction::ICmp {
        result: cmp_result,
        op: crate::ir::instructions::ICmpOp::Ne,
        lhs: val,
        rhs: zero,
        span,
    };
    emit_instruction_to_current_block(ctx, cmp_inst);
    cmp_result
}
