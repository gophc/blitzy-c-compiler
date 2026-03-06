#![allow(clippy::result_unit_err)]
//! # BCC Parser — Phase 4: Recursive-Descent C11 Parser
//!
//! This module implements a recursive-descent parser consuming tokens from the
//! lexer and producing a complete Abstract Syntax Tree (AST) with comprehensive
//! GCC extension support required for Linux kernel compilation.
//!
//! ## Key Features
//! - Full C11 language support
//! - GCC extensions: statement expressions, typeof, computed gotos, case ranges,
//!   zero-length arrays, `__attribute__`, inline assembly with AT&T syntax
//! - 512-depth recursion limit enforcement
//! - Error recovery with synchronization for multi-error reporting
//! - All AST nodes carry [`Span`] for source location
//!
//! ## Architecture
//! The parser is structured as a main driver ([`Parser`] struct in this file) with
//! 8 submodules handling specific grammar categories:
//! - [`ast`] — AST node definitions (foundational, used by all other submodules)
//! - [`declarations`] — declaration and declarator parsing
//! - [`expressions`] — expression parsing with operator-precedence climbing
//! - [`statements`] — statement and compound-statement parsing
//! - [`types`] — type specifier and qualifier parsing
//! - [`gcc_extensions`] — GCC language extension dispatch
//! - [`attributes`] — `__attribute__((...))` parsing
//! - [`inline_asm`] — inline assembly (`asm`/`__asm__`) parsing
//!
//! ## Dependencies
//! - `crate::common` for diagnostics, string_interner, target
//! - `crate::frontend::lexer` for Token, TokenKind, Lexer
//! - Does **NOT** depend on `ir`, `passes`, or `backend`

// ===========================================================================
// Submodule declarations
// ===========================================================================

pub mod ast;
pub mod attributes;
pub mod declarations;
pub mod expressions;
pub mod gcc_extensions;
pub mod inline_asm;
pub mod statements;
pub mod types;

// Re-export all AST types for convenient access by consumers of the parser module.
pub use ast::*;

// ===========================================================================
// Imports
// ===========================================================================

use crate::common::diagnostics::{Diagnostic, Span};
use crate::common::string_interner::Symbol;
use crate::common::target::Target;
use crate::frontend::lexer::{Lexer, Token, TokenKind};

// ===========================================================================
// Constants
// ===========================================================================

/// Maximum recursion depth for the parser. Enforced to prevent stack overflow
/// on deeply nested kernel macro expansions (e.g., deeply nested struct
/// definitions, complex initializer lists, deeply nested expressions).
const MAX_RECURSION_DEPTH: u32 = 512;

// ===========================================================================
// Parser struct
// ===========================================================================

/// Recursive-descent C11 parser with comprehensive GCC extension support.
///
/// The [`Parser`] struct is the central driver for Phase 4 of the BCC
/// compilation pipeline. It consumes a token stream produced by the lexer
/// and constructs an Abstract Syntax Tree (AST) representing the parsed
/// C translation unit.
///
/// # Token Consumption Model
///
/// The parser maintains a one-token lookahead (`current`) and tracks the
/// previously consumed token (`previous`) for span construction. The
/// [`advance`](Parser::advance) method shifts the window forward by one
/// token.
///
/// # Error Recovery
///
/// On syntax errors, the parser enters "panic mode" (suppressing subsequent
/// errors until recovery), skips tokens to a synchronization point (`;`,
/// `}`, or a declaration-starting keyword), and then resumes normal parsing.
/// This enables multi-error reporting in a single compilation pass.
///
/// # Recursion Limit
///
/// A hard limit of 512 recursion levels is enforced via
/// [`enter_recursion`](Parser::enter_recursion) /
/// [`leave_recursion`](Parser::leave_recursion). This prevents stack
/// overflow when parsing deeply nested constructs produced by kernel macro
/// expansions.
pub struct Parser<'src> {
    /// Lexer producing the token stream.
    ///
    /// The lexer also holds mutable references to the [`Interner`] and
    /// [`DiagnosticEngine`], which the parser accesses through accessor
    /// methods on the lexer.
    pub(crate) lexer: Lexer<'src>,

    /// Current lookahead token.
    ///
    /// This is the token the parser is "looking at" — all `check`, `peek`,
    /// and `match_token` operations inspect this token.
    pub(crate) current: Token,

    /// Previously consumed token.
    ///
    /// After [`advance`](Parser::advance), `previous` holds the token that
    /// was `current` before the advance. This is used for span construction:
    /// the end of an AST node's span is typically `previous.span.end`.
    pub(crate) previous: Token,

    /// Target architecture for size/alignment queries.
    ///
    /// Affects `sizeof`/`_Alignof` resolution, predefined macro awareness,
    /// and architecture-dependent parsing decisions.
    pub(crate) target: Target,

    /// Current recursion depth counter.
    recursion_depth: u32,

    /// Maximum allowed recursion depth (always [`MAX_RECURSION_DEPTH`]).
    max_recursion_depth: u32,

    /// Panic mode flag for error recovery.
    ///
    /// When `true`, the parser suppresses error emission until the next
    /// successful synchronization. This prevents cascading error noise.
    panic_mode: bool,

    /// Set of typedef names known to the parser.
    ///
    /// When parsing, the parser must distinguish typedef names from ordinary
    /// identifiers to correctly parse declarations vs. expressions. This set
    /// tracks all names introduced by `typedef` declarations.
    pub(crate) typedef_names: std::collections::HashSet<u32>,
}

// ===========================================================================
// Implementation
// ===========================================================================

impl<'src> Parser<'src> {
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /// Create a new parser from a pre-constructed lexer and target.
    ///
    /// The lexer must already hold mutable references to the [`Interner`]
    /// and [`DiagnosticEngine`]. The parser accesses them through the lexer's
    /// accessor methods.
    ///
    /// The first token is immediately consumed from the lexer to populate
    /// the `current` lookahead.
    pub fn new(mut lexer: Lexer<'src>, target: Target) -> Self {
        // Prime the lookahead by reading the first token.
        let current = lexer.next_token();

        // Create a dummy "previous" token (before any real tokens).
        let previous = Token::new(TokenKind::Eof, Span::dummy());

        let mut parser = Parser {
            lexer,
            current,
            previous,
            target,
            recursion_depth: 0,
            max_recursion_depth: MAX_RECURSION_DEPTH,
            panic_mode: false,
            typedef_names: std::collections::HashSet::new(),
        };

        // Pre-register GCC builtin type names as typedefs so the parser
        // recognises them as type specifiers in declaration contexts.
        let builtin_type_names = ["__builtin_va_list"];
        for name in &builtin_type_names {
            let sym = parser.intern(name);
            parser.register_typedef(sym);
        }

        parser
    }

    // -----------------------------------------------------------------------
    // Core Token Consumption
    // -----------------------------------------------------------------------

    /// Advance the parser by one token.
    ///
    /// Moves the current token into `previous` and reads the next token
    /// from the lexer into `current`.
    pub fn advance(&mut self) {
        // Move current → previous, fetch next → current.
        self.previous = self.current.clone();
        self.current = self.lexer.next_token();
    }

    /// Expect the current token to match `kind`, consume it, and return it.
    ///
    /// If the current token does not match, emits a diagnostic error
    /// `"expected '<kind>', found '<actual>'"` and returns `Err(())`.
    ///
    /// Uses [`Token::is`] for discriminant-only comparison (ignoring
    /// payloads on data-carrying variants like `Identifier`).
    pub fn expect(&mut self, kind: TokenKind) -> Result<Token, ()> {
        if self.current.is(&kind) {
            self.advance();
            Ok(self.previous.clone())
        } else {
            let span = self.current_span();
            let msg = format!("expected '{}', found '{}'", kind, self.current.kind);
            self.error(span, &msg);
            Err(())
        }
    }

    /// Check whether the current token matches `kind` without consuming it.
    ///
    /// Uses [`Token::is`] for discriminant-only comparison.
    pub fn check(&self, kind: &TokenKind) -> bool {
        self.current.is(kind)
    }

    /// If the current token matches `kind`, consume it and return `true`.
    /// Otherwise return `false` without consuming.
    ///
    /// Uses [`Token::is`] for discriminant-only comparison.
    pub fn match_token(&mut self, kind: &TokenKind) -> bool {
        if self.current.is(kind) {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Consume the current token if it matches `kind`, using a custom error
    /// message on mismatch.
    ///
    /// Behaves identically to [`expect`](Parser::expect) except the error
    /// message is caller-provided rather than auto-generated.
    pub fn consume(&mut self, kind: TokenKind, msg: &str) -> Result<Token, ()> {
        if self.current.is(&kind) {
            self.advance();
            Ok(self.previous.clone())
        } else {
            let span = self.current_span();
            self.error(span, msg);
            Err(())
        }
    }

    // -----------------------------------------------------------------------
    // Lookahead
    // -----------------------------------------------------------------------

    /// Return a reference to the current token's [`TokenKind`].
    ///
    /// This is the cheapest lookahead — no token advancement occurs.
    #[inline]
    pub fn peek(&self) -> &TokenKind {
        &self.current.kind
    }

    /// Peek at the *n*-th future token (0 = the token after `current`).
    ///
    /// This supports multi-token lookahead for disambiguation (e.g.,
    /// distinguishing casts from parenthesized expressions). The returned
    /// token is cloned from the lexer's internal lookahead buffer.
    pub fn peek_nth(&mut self, n: usize) -> Token {
        self.lexer.peek_nth(n).clone()
    }

    // -----------------------------------------------------------------------
    // Span Construction
    // -----------------------------------------------------------------------

    /// Return the source span of the current (not-yet-consumed) token.
    #[inline]
    pub fn current_span(&self) -> Span {
        self.current.span
    }

    /// Return the source span of the most recently consumed token.
    #[inline]
    pub fn previous_span(&self) -> Span {
        self.previous.span
    }

    /// Construct a span from `start` to the end of the most recently
    /// consumed token (`previous`).
    ///
    /// This is the standard pattern for building AST node spans:
    /// ```ignore
    /// let start = self.current_span();
    /// // ... parse tokens ...
    /// let span = self.make_span(start);
    /// ```
    #[inline]
    pub fn make_span(&self, start: Span) -> Span {
        Span::new(start.file_id, start.start, self.previous.span.end)
    }

    // -----------------------------------------------------------------------
    // Recursion Depth Management
    // -----------------------------------------------------------------------

    /// Enter a recursion level. Returns `Err(())` if the depth exceeds
    /// [`MAX_RECURSION_DEPTH`] (512).
    ///
    /// Every recursive parsing function (compound statements, nested
    /// expressions, etc.) must call this at entry and
    /// [`leave_recursion`](Parser::leave_recursion) at exit.
    pub fn enter_recursion(&mut self) -> Result<(), ()> {
        self.recursion_depth += 1;
        if self.recursion_depth > self.max_recursion_depth {
            let span = self.current_span();
            let msg = format!(
                "recursion depth exceeded (limit: {})",
                self.max_recursion_depth
            );
            // Bypass panic_mode — recursion overflow is always reported.
            self.lexer.diagnostics_mut().emit_error(span, msg);
            Err(())
        } else {
            Ok(())
        }
    }

    /// Leave a recursion level, decrementing the depth counter.
    ///
    /// # Panics (debug builds)
    /// Panics if called when the recursion depth is already zero.
    pub fn leave_recursion(&mut self) {
        debug_assert!(
            self.recursion_depth > 0,
            "BCC parser: recursion depth underflow"
        );
        self.recursion_depth = self.recursion_depth.saturating_sub(1);
    }

    // -----------------------------------------------------------------------
    // Error Handling and Recovery
    // -----------------------------------------------------------------------

    /// Report a parse error at the given source span.
    ///
    /// If the parser is already in panic mode (recovering from a previous
    /// error), the error is suppressed to avoid cascading noise. Otherwise,
    /// the diagnostic is emitted and panic mode is entered.
    pub fn error(&mut self, span: Span, msg: &str) {
        if !self.panic_mode {
            self.lexer.diagnostics_mut().emit_error(span, msg);
            self.panic_mode = true;
        }
    }

    /// Emit a warning diagnostic (not affected by panic mode).
    pub fn warn(&mut self, span: Span, msg: &str) {
        let diag = Diagnostic::warning(span, msg);
        self.lexer.diagnostics_mut().emit(diag);
    }

    /// Synchronize the parser after an error by skipping tokens until a
    /// recovery point is found.
    ///
    /// Recovery points include:
    /// - Semicolons (`;`) — consumed, then parsing resumes after them
    /// - Closing braces (`}`) — NOT consumed, so the caller can match them
    /// - Declaration/statement-starting keywords (`int`, `char`, `void`,
    ///   `struct`, `if`, `while`, `for`, `return`, etc.)
    ///
    /// Exits panic mode after synchronization to allow subsequent error
    /// reports.
    pub fn synchronize(&mut self) {
        self.panic_mode = false;

        while !self.current.is_eof() {
            // If the previous token was a semicolon, we already consumed it
            // and are at a valid restart point.
            if self.previous.is(&TokenKind::Semicolon) {
                return;
            }

            match self.current.kind {
                // Semicolons: consume and resume.
                TokenKind::Semicolon => {
                    self.advance();
                    return;
                }
                // Closing brace: don't consume (caller may need it).
                TokenKind::RightBrace => {
                    return;
                }
                // Declaration-starting keywords — resume without consuming.
                TokenKind::Int
                | TokenKind::Char
                | TokenKind::Short
                | TokenKind::Long
                | TokenKind::Float
                | TokenKind::Double
                | TokenKind::Signed
                | TokenKind::Unsigned
                | TokenKind::Void
                | TokenKind::Bool
                | TokenKind::Complex
                | TokenKind::Atomic
                | TokenKind::Struct
                | TokenKind::Union
                | TokenKind::Enum
                | TokenKind::Typedef
                | TokenKind::Extern
                | TokenKind::Static
                | TokenKind::Auto
                | TokenKind::Register
                | TokenKind::ThreadLocal
                | TokenKind::Typeof
                | TokenKind::Attribute
                | TokenKind::Extension
                | TokenKind::Alignas
                | TokenKind::Noreturn
                | TokenKind::Inline
                // Statement-starting keywords
                | TokenKind::If
                | TokenKind::While
                | TokenKind::Do
                | TokenKind::For
                | TokenKind::Return
                | TokenKind::Switch
                | TokenKind::Case
                | TokenKind::Default
                | TokenKind::Break
                | TokenKind::Continue
                | TokenKind::Goto
                | TokenKind::StaticAssert
                | TokenKind::Asm
                | TokenKind::AsmVolatile => {
                    return;
                }
                // Not a sync point — skip this token.
                _ => {
                    self.advance();
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Convenience Accessors
    // -----------------------------------------------------------------------

    /// Check whether any errors have been emitted so far.
    pub fn has_errors(&self) -> bool {
        self.lexer.diagnostics().has_errors()
    }

    /// Resolve an interned [`Symbol`] back to its string representation.
    pub fn resolve_symbol(&self, sym: Symbol) -> &str {
        self.lexer.interner().resolve(sym)
    }

    /// Intern a string, returning its [`Symbol`] handle.
    pub fn intern(&mut self, s: &str) -> Symbol {
        self.lexer.interner_mut().intern(s)
    }

    /// Get the target architecture.
    #[inline]
    pub fn target(&self) -> Target {
        self.target
    }

    /// Check whether an identifier (by Symbol) is a known typedef name.
    pub fn is_typedef_name(&self, sym: Symbol) -> bool {
        self.typedef_names.contains(&sym.as_u32())
    }

    /// Register a symbol as a typedef name.
    pub fn register_typedef(&mut self, sym: Symbol) {
        self.typedef_names.insert(sym.as_u32());
    }

    /// Skip an optional GCC `__asm__("symbol_name")` label on a declaration.
    ///
    /// This construct appears after a declarator to specify the assembly-level
    /// symbol name.  It is used extensively in glibc headers (e.g. to redirect
    /// `fscanf` to `__isoc99_fscanf`).
    pub fn skip_asm_label(&mut self) {
        if self.check(&TokenKind::Asm) {
            self.advance(); // consume `__asm__` / `asm`
            if self.match_token(&TokenKind::LeftParen) {
                let mut depth: u32 = 1;
                while depth > 0 && !self.check(&TokenKind::Eof) {
                    if self.check(&TokenKind::LeftParen) {
                        depth += 1;
                    } else if self.check(&TokenKind::RightParen) {
                        depth -= 1;
                        if depth == 0 {
                            self.advance(); // consume closing ')'
                            break;
                        }
                    }
                    self.advance();
                }
            }
        }
    }

    /// Skip optional trailing `__attribute__((...))` lists that appear after
    /// a declarator/initializer but before `;` or `,`.
    pub fn skip_trailing_attributes(&mut self) {
        while self.check(&TokenKind::Attribute) {
            self.advance(); // consume `__attribute__`
            if self.match_token(&TokenKind::LeftParen) {
                let mut depth: u32 = 1;
                while depth > 0 && !self.check(&TokenKind::Eof) {
                    if self.check(&TokenKind::LeftParen) {
                        depth += 1;
                    } else if self.check(&TokenKind::RightParen) {
                        depth -= 1;
                        if depth == 0 {
                            self.advance(); // consume closing ')'
                            break;
                        }
                    }
                    self.advance();
                }
            }
        }
    }

    /// Check if the current token is an identifier (any identifier).
    pub fn is_at_identifier(&self) -> bool {
        matches!(self.current.kind, TokenKind::Identifier(_))
    }

    /// Extract the [`Symbol`] from the current token if it is an identifier.
    /// Returns `None` if the current token is not an identifier.
    pub fn current_identifier(&self) -> Option<Symbol> {
        match self.current.kind {
            TokenKind::Identifier(sym) => Some(sym),
            _ => None,
        }
    }

    // -----------------------------------------------------------------------
    // Top-Level Parsing Entry Points
    // -----------------------------------------------------------------------

    /// Convenience entry point: parse the entire translation unit.
    ///
    /// Equivalent to calling [`parse_translation_unit`](Parser::parse_translation_unit).
    pub fn parse(&mut self) -> TranslationUnit {
        self.parse_translation_unit()
    }

    /// Parse a complete C translation unit (the top-level grammar rule).
    ///
    /// A translation unit is a sequence of *external declarations*: function
    /// definitions, variable declarations, typedef declarations,
    /// struct/union/enum definitions, `_Static_assert`, and top-level inline
    /// assembly.
    ///
    /// The parser loops until EOF, accumulating declarations. On parse
    /// errors, it calls [`synchronize`](Parser::synchronize) and continues
    /// to enable multi-error reporting.
    pub fn parse_translation_unit(&mut self) -> TranslationUnit {
        let start_span = self.current_span();
        let mut declarations = Vec::new();

        while !self.current.is_eof() {
            // Record the current position so we can detect no-progress loops.
            let pos_before = self.current.span.start;

            match self.parse_external_declaration() {
                Ok(decl) => {
                    declarations.push(decl);
                }
                Err(()) => {
                    // Error already reported; synchronize and continue.
                    self.synchronize();

                    // If synchronize didn't advance past the problematic token,
                    // force-skip one token to guarantee forward progress and
                    // prevent an infinite loop.
                    if !self.current.is_eof() && self.current.span.start == pos_before {
                        self.advance();
                    }
                }
            }
        }

        // Build the span for the entire translation unit.
        let span = if declarations.is_empty() {
            start_span
        } else {
            Span::new(start_span.file_id, start_span.start, self.previous.span.end)
        };

        TranslationUnit { declarations, span }
    }

    /// Parse a single external declaration at file scope.
    ///
    /// External declarations include:
    /// - Function definitions: `int main(void) { ... }`
    /// - Variable/typedef declarations: `int x = 5;`
    /// - Struct/union/enum definitions: `struct foo { int x; };`
    /// - `_Static_assert` declarations
    /// - Top-level inline assembly: `__asm__("...");`
    /// - Empty declarations (stray semicolons)
    ///
    /// The disambiguation between function definitions and declarations is
    /// resolved by checking whether a `{` (function body) follows the first
    /// declarator.
    pub fn parse_external_declaration(&mut self) -> Result<ExternalDeclaration, ()> {
        let start_span = self.current_span();

        // Consume any leading `__extension__` tokens (suppress warnings).
        while self.match_token(&TokenKind::Extension) {}

        // Empty declaration — a stray semicolon at file scope.
        if self.match_token(&TokenKind::Semicolon) {
            return Ok(ExternalDeclaration::Empty);
        }

        // _Static_assert declaration.
        if self.check(&TokenKind::StaticAssert) {
            return self.parse_static_assert_external();
        }

        // Top-level inline assembly statement.
        if self.check(&TokenKind::Asm) || self.check(&TokenKind::AsmVolatile) {
            let asm_stmt = inline_asm::parse_asm_statement(self)?;
            return Ok(ExternalDeclaration::AsmStatement(asm_stmt));
        }

        // Parse declaration specifiers (storage class, type specifiers,
        // qualifiers, function specifiers, alignment, attributes).
        let specifiers = declarations::parse_declaration_specifiers(self)?;

        // Bare specifier declaration with no declarator (e.g., `struct foo;`
        // or `enum bar { A, B };`).
        if self.check(&TokenKind::Semicolon) {
            self.advance();
            let span = self.make_span(start_span);
            return Ok(ExternalDeclaration::Declaration(Box::new(Declaration {
                specifiers,
                declarators: Vec::new(),
                static_assert: None,
                span,
            })));
        }

        // Parse the first declarator.
        let declarator = declarations::parse_declarator(self)?;

        // Detect function definition: specifiers declarator `{` body `}`.
        if self.check(&TokenKind::LeftBrace) {
            let func_def = declarations::parse_function_definition(self, specifiers, declarator)?;
            return Ok(ExternalDeclaration::FunctionDefinition(Box::new(func_def)));
        }

        // Regular declaration — parse optional initializer and additional
        // comma-separated init-declarators.
        self.parse_remaining_declaration(start_span, specifiers, declarator)
    }

    // -----------------------------------------------------------------------
    // Internal Helpers for parse_external_declaration
    // -----------------------------------------------------------------------

    /// Parse a `_Static_assert(expr, "msg");` at file scope.
    ///
    /// Produces an [`ExternalDeclaration::Declaration`] wrapping a
    /// [`Declaration`] with empty specifiers/declarators; the static assert
    /// is encoded as a special declaration form with a zero-declarator list
    /// and the condition/message are stored in the specifiers' span for
    /// downstream semantic handling.
    ///
    /// A dedicated [`StaticAssert`] AST node is constructed and wrapped in
    /// an appropriate declaration container.
    fn parse_static_assert_external(&mut self) -> Result<ExternalDeclaration, ()> {
        let start_span = self.current_span();

        // Consume `_Static_assert`.
        self.advance();

        // Expect `(`.
        self.expect(TokenKind::LeftParen)?;

        // Parse condition expression (constant expression).
        let condition = expressions::parse_constant_expression(self)?;

        // Parse optional `,` followed by string-literal message.
        // C11 requires the message; C23 makes it optional; GCC is lenient.
        let message = if self.match_token(&TokenKind::Comma) {
            self.parse_string_literal_bytes()
        } else {
            None
        };

        // Expect `)` `;`.
        self.expect(TokenKind::RightParen)?;
        self.expect(TokenKind::Semicolon)?;

        let span = self.make_span(start_span);

        // Construct a `StaticAssert` AST node and attach it to the
        // `Declaration` so the semantic analyzer can evaluate the assertion
        // condition as an integer constant expression at compile time.
        let sa = StaticAssert {
            condition: Box::new(condition),
            message,
            span,
        };

        let decl = Declaration {
            specifiers: DeclarationSpecifiers {
                storage_class: None,
                type_specifiers: Vec::new(),
                type_qualifiers: Vec::new(),
                function_specifiers: Vec::new(),
                alignment_specifier: None,
                attributes: Vec::new(),
                span,
            },
            declarators: Vec::new(),
            static_assert: Some(sa),
            span,
        };

        Ok(ExternalDeclaration::Declaration(Box::new(decl)))
    }

    /// Parse a string literal token and return its byte content.
    ///
    /// Returns `None` if the current token is not a string literal (emits
    /// an error in that case).
    fn parse_string_literal_bytes(&mut self) -> Option<Vec<u8>> {
        match &self.current.kind {
            TokenKind::StringLiteral { value, .. } => {
                let bytes = value.as_bytes().to_vec();
                self.advance();
                Some(bytes)
            }
            _ => {
                let span = self.current_span();
                self.error(span, "expected string literal");
                None
            }
        }
    }

    /// After parsing declaration specifiers and the first declarator, parse
    /// the rest of a variable/typedef declaration.
    ///
    /// Handles:
    /// - Optional initializer on the first declarator (`= expr` or `= { ... }`)
    /// - Additional comma-separated init-declarators
    /// - Trailing semicolon
    fn parse_remaining_declaration(
        &mut self,
        start_span: Span,
        specifiers: DeclarationSpecifiers,
        first_declarator: Declarator,
    ) -> Result<ExternalDeclaration, ()> {
        let mut init_declarators = Vec::new();

        // If this is a typedef declaration, register the first declarator's
        // name as a typedef name so the parser can recognize it as a type
        // name in subsequent declarations.
        let is_typedef = matches!(
            specifiers.storage_class,
            Some(crate::frontend::parser::ast::StorageClass::Typedef)
        );
        if is_typedef {
            declarations::register_declarator_name_pub(self, &first_declarator);
        }

        // Skip optional GCC asm label: `__asm__("symbol_name")`
        self.skip_asm_label();

        // Parse optional initializer for the first declarator.
        let first_init = if self.match_token(&TokenKind::Equal) {
            Some(self.parse_initializer()?)
        } else {
            None
        };

        // Skip optional trailing GCC __attribute__ after declarator
        self.skip_trailing_attributes();

        let first_span = self.make_span(start_span);
        init_declarators.push(InitDeclarator {
            declarator: first_declarator,
            initializer: first_init,
            span: first_span,
        });

        // Parse additional comma-separated init-declarators.
        while self.match_token(&TokenKind::Comma) {
            let decl_start = self.current_span();
            let decl = declarations::parse_declarator(self)?;

            // Skip optional GCC asm label
            self.skip_asm_label();

            let init = if self.match_token(&TokenKind::Equal) {
                Some(self.parse_initializer()?)
            } else {
                None
            };

            // Skip optional trailing GCC __attribute__
            self.skip_trailing_attributes();

            let decl_span = self.make_span(decl_start);
            init_declarators.push(InitDeclarator {
                declarator: decl,
                initializer: init,
                span: decl_span,
            });
        }

        // Expect trailing `;`.
        self.expect(TokenKind::Semicolon)?;

        let span = self.make_span(start_span);
        Ok(ExternalDeclaration::Declaration(Box::new(Declaration {
            specifiers,
            declarators: init_declarators,
            static_assert: None,
            span,
        })))
    }

    // -----------------------------------------------------------------------
    // Initializer Parsing (used by parse_external_declaration)
    // -----------------------------------------------------------------------

    /// Parse an initializer: either a simple expression or a brace-enclosed
    /// initializer list.
    ///
    /// Called after consuming the `=` token.
    ///
    /// # Grammar
    /// ```text
    /// initializer:
    ///     assignment-expression
    ///     '{' initializer-list ','? '}'
    /// ```
    fn parse_initializer(&mut self) -> Result<Initializer, ()> {
        if self.check(&TokenKind::LeftBrace) {
            self.parse_brace_initializer()
        } else {
            let expr = expressions::parse_assignment_expression(self)?;
            Ok(Initializer::Expression(Box::new(expr)))
        }
    }

    /// Parse a brace-enclosed initializer list: `{ init, init, ... }`.
    ///
    /// Supports C11 designated initializers (`.field = value`,
    /// `[index] = value`) and GCC array index range designators
    /// (`[low ... high] = value`).
    fn parse_brace_initializer(&mut self) -> Result<Initializer, ()> {
        let start = self.current_span();
        self.expect(TokenKind::LeftBrace)?;

        let mut items = Vec::new();
        let mut trailing_comma = false;

        // Parse initializer list entries until `}`.
        if !self.check(&TokenKind::RightBrace) {
            loop {
                let item_start = self.current_span();
                let mut designators = Vec::new();

                // Parse optional designators: `.field`, `[index]`, `[low...high]`.
                while self.check(&TokenKind::Dot) || self.check(&TokenKind::LeftBracket) {
                    if self.match_token(&TokenKind::Dot) {
                        // Field designator: `.member`.
                        match self.current.kind {
                            TokenKind::Identifier(sym) => {
                                let fspan = self.current_span();
                                self.advance();
                                designators.push(Designator::Field(sym, fspan));
                            }
                            _ => {
                                let span = self.current_span();
                                self.error(span, "expected field name after '.'");
                                return Err(());
                            }
                        }
                    } else if self.match_token(&TokenKind::LeftBracket) {
                        // Array index designator: `[expr]` or `[low ... high]`.
                        let idx = expressions::parse_constant_expression(self)?;

                        if self.match_token(&TokenKind::Ellipsis) {
                            // GCC range designator: `[low ... high]`.
                            let high = expressions::parse_constant_expression(self)?;
                            self.expect(TokenKind::RightBracket)?;
                            let dspan = self.make_span(item_start);
                            designators.push(Designator::IndexRange(
                                Box::new(idx),
                                Box::new(high),
                                dspan,
                            ));
                        } else {
                            self.expect(TokenKind::RightBracket)?;
                            let dspan = self.make_span(item_start);
                            designators.push(Designator::Index(Box::new(idx), dspan));
                        }
                    }
                }

                // If designators are present, expect `=`.
                if !designators.is_empty() {
                    self.expect(TokenKind::Equal)?;
                }

                // Parse the initializer value (recursive for nested braces).
                let init = self.parse_initializer()?;
                let item_span = self.make_span(item_start);

                items.push(DesignatedInitializer {
                    designators,
                    initializer: init,
                    span: item_span,
                });

                // Check for comma separator.
                if !self.match_token(&TokenKind::Comma) {
                    trailing_comma = false;
                    break;
                }

                // Trailing comma before `}` is allowed.
                if self.check(&TokenKind::RightBrace) {
                    trailing_comma = true;
                    break;
                }
            }
        }

        self.expect(TokenKind::RightBrace)?;
        let span = self.make_span(start);

        Ok(Initializer::List {
            designators_and_initializers: items,
            trailing_comma,
            span,
        })
    }
}

// ===========================================================================
// Display
// ===========================================================================

impl<'src> std::fmt::Debug for Parser<'src> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Parser")
            .field("current", &self.current)
            .field("previous", &self.previous)
            .field("target", &self.target)
            .field("recursion_depth", &self.recursion_depth)
            .field("panic_mode", &self.panic_mode)
            .finish()
    }
}
