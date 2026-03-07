//! # BCC Preprocessor — Phases 1–2 of the C11 Compilation Pipeline
//!
//! Implements the C preprocessor with:
//! - Phase 1: Trigraph replacement (`??=` → `#`, etc.) and backslash-newline line splicing
//! - Phase 2: Directive processing (`#include`, `#define`, `#if`, etc.) and macro expansion
//!
//! ## Key Features
//! - Paint-marker recursion protection for self-referential macros (`#define A A` terminates correctly)
//! - 512-depth recursion limit for macro expansion (prevents stack overflow on kernel macros)
//! - PUA encoding for non-UTF-8 source file byte round-tripping (U+E080–U+E0FF)
//! - Full C11 preprocessor compliance with GCC extensions
//! - `__VA_ARGS__` and `__VA_OPT__` variadic macro support
//!
//! ## Usage
//! ```ignore
//! use crate::frontend::preprocessor::Preprocessor;
//! let mut pp = Preprocessor::new(source_map, diagnostics, target, interner);
//! pp.add_include_path("/usr/include");
//! pp.add_define("DEBUG", "1");
//! let tokens = pp.preprocess_file("hello.c");
//! ```
//!
//! ## Dependencies
//! - `crate::common::encoding` — PUA/UTF-8 encoding for source file reading
//! - `crate::common::source_map` — File tracking and `#line` directive handling
//! - `crate::common::diagnostics` — Error/warning reporting (unterminated `#if`, circular includes, etc.)
//! - `crate::common::string_interner` — Macro name interning
//! - `crate::common::target` — Architecture-specific predefined macros
//! - `crate::common::fx_hash` — FxHashMap for macro definition storage

// ---------------------------------------------------------------------------
// Submodule declarations — all 7 preprocessor sub-components
// ---------------------------------------------------------------------------

pub mod directives;
pub mod expression;
pub mod include_handler;
pub mod macro_expander;
pub mod paint_marker;
pub mod predefined;
pub mod token_paster;

// ---------------------------------------------------------------------------
// Imports from crate::common — infrastructure layer
// ---------------------------------------------------------------------------

use crate::common::diagnostics::{Diagnostic, DiagnosticEngine, Span};
use crate::common::encoding::read_source_file;
use crate::common::fx_hash::FxHashMap;
use crate::common::source_map::SourceMap;
use crate::common::string_interner::Interner;
use crate::common::target::Target;

use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// PPTokenKind — preprocessing token classification
// ---------------------------------------------------------------------------

/// Classification of a preprocessing token.
///
/// Preprocessing tokens are coarser than lexer tokens — they do not
/// distinguish keywords from identifiers, and numeric literals are not
/// yet classified by type (int, float, hex, etc.).  The lexer performs
/// that finer classification after preprocessing is complete.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PPTokenKind {
    /// An identifier (or keyword — not distinguished at this stage).
    Identifier,
    /// A preprocessing number: a token that starts with a digit or `.` followed
    /// by a digit.  Not yet classified as integer or floating-point.
    Number,
    /// A string literal, including the surrounding `"` quotes and any
    /// prefix (`u8`, `u`, `U`, `L`).
    StringLiteral,
    /// A character literal, including the surrounding `'` quotes and any
    /// prefix (`L`, `u`, `U`).
    CharLiteral,
    /// An operator or punctuator: `+`, `-`, `->`, `(`, `)`, `{`, `}`, `;`, etc.
    Punctuator,
    /// Whitespace (spaces, tabs) — NOT newlines.
    Whitespace,
    /// A newline character (`\n`).  Significant for directive recognition:
    /// a `#` that is the first non-whitespace token on a line introduces
    /// a preprocessing directive.
    Newline,
    /// A header name in an `#include` directive: either `<path>` or `"path"`.
    HeaderName,
    /// A placemarker token produced by `##` concatenation when one operand is
    /// an empty macro argument.  Concatenating a placemarker with a real token
    /// yields the real token.
    PlacemarkerToken,
    /// End-of-file sentinel.
    EndOfFile,
}

// ---------------------------------------------------------------------------
// PPToken — a single preprocessing token
// ---------------------------------------------------------------------------

/// A preprocessing token — the output of the preprocessor, input to the lexer.
///
/// Carries the token text, source location span, and macro-expansion metadata
/// (whether it was produced by macro expansion and whether it has been painted
/// by the paint-marker system to suppress re-expansion).
#[derive(Debug, Clone)]
pub struct PPToken {
    /// The kind (classification) of this token.
    pub kind: PPTokenKind,
    /// The textual representation of this token exactly as it appears in the
    /// (possibly macro-expanded) source.
    pub text: String,
    /// Source location span for diagnostic reporting.
    pub span: Span,
    /// `true` if this token was produced by macro expansion rather than
    /// appearing directly in the source file.
    pub from_macro: bool,
    /// Paint-marker state.  When `true`, this token must NOT be re-expanded
    /// even if it matches a defined macro name.  Set by the paint-marker
    /// system when a macro name is encountered during its own expansion
    /// (C11 §6.10.3.4 rescanning suppression).
    pub painted: bool,
}

impl PPToken {
    /// Create a new preprocessing token with default expansion metadata.
    ///
    /// `from_macro` and `painted` are both set to `false` — the caller
    /// should override these when constructing tokens during macro expansion.
    #[inline]
    pub fn new(kind: PPTokenKind, text: impl Into<String>, span: Span) -> Self {
        PPToken {
            kind,
            text: text.into(),
            span,
            from_macro: false,
            painted: false,
        }
    }

    /// Create a new preprocessing token produced by macro expansion.
    #[inline]
    pub fn from_expansion(kind: PPTokenKind, text: impl Into<String>, span: Span) -> Self {
        PPToken {
            kind,
            text: text.into(),
            span,
            from_macro: true,
            painted: false,
        }
    }

    /// Create an end-of-file sentinel token.
    #[inline]
    pub fn eof(span: Span) -> Self {
        PPToken {
            kind: PPTokenKind::EndOfFile,
            text: String::new(),
            span,
            from_macro: false,
            painted: false,
        }
    }

    /// Create a placemarker token (used by `##` concatenation with empty args).
    #[inline]
    pub fn placemarker(span: Span) -> Self {
        PPToken {
            kind: PPTokenKind::PlacemarkerToken,
            text: String::new(),
            span,
            from_macro: true,
            painted: false,
        }
    }

    /// Returns `true` if this token is whitespace or newline.
    #[inline]
    pub fn is_whitespace(&self) -> bool {
        matches!(self.kind, PPTokenKind::Whitespace | PPTokenKind::Newline)
    }

    /// Returns `true` if this token is end-of-file.
    #[inline]
    pub fn is_eof(&self) -> bool {
        self.kind == PPTokenKind::EndOfFile
    }
}

// ---------------------------------------------------------------------------
// MacroKind — object-like vs function-like macros
// ---------------------------------------------------------------------------

/// The kind of a macro definition.
///
/// Object-like macros expand to their replacement list without argument
/// substitution.  Function-like macros accept a parenthesised argument list
/// and perform parameter substitution in the replacement.
#[derive(Debug, Clone)]
pub enum MacroKind {
    /// An object-like macro: `#define FOO 42` — expands to replacement
    /// tokens without any argument processing.
    ObjectLike,
    /// A function-like macro: `#define MAX(a, b) ((a) > (b) ? (a) : (b))`.
    FunctionLike {
        /// Parameter names in declaration order.
        params: Vec<String>,
        /// `true` if the macro is variadic (last parameter is `...`).
        /// Variadic arguments are accessible via `__VA_ARGS__`.
        variadic: bool,
    },
}

// ---------------------------------------------------------------------------
// MacroDef — a complete macro definition
// ---------------------------------------------------------------------------

/// A macro definition created by `#define` or predefined by the compiler.
///
/// Stores the macro name, kind (object-like or function-like), replacement
/// token list, and metadata about whether it is a compiler-predefined macro
/// and where it was defined in the source.
#[derive(Debug, Clone)]
pub struct MacroDef {
    /// The macro name (e.g., `"FOO"`, `"__STDC__"`).
    pub name: String,
    /// Whether this is an object-like or function-like macro.
    pub kind: MacroKind,
    /// The replacement token list.  For an empty `#define FOO`, this is empty.
    pub replacement: Vec<PPToken>,
    /// `true` for compiler-predefined macros (`__FILE__`, `__LINE__`,
    /// `__STDC__`, etc.) — these cannot be `#undef`'d.
    pub is_predefined: bool,
    /// Source location where the macro was defined.  [`Span::dummy()`] for
    /// predefined macros.
    pub definition_span: Span,
}

// ---------------------------------------------------------------------------
// ConditionalState — #if/#ifdef/#else/#endif nesting tracker
// ---------------------------------------------------------------------------

/// State for a single level of nested `#if`/`#ifdef`/`#elif`/`#else`/`#endif`
/// conditional compilation.
///
/// The preprocessor maintains a stack of these to handle arbitrary nesting of
/// conditional blocks.
#[derive(Debug, Clone)]
pub struct ConditionalState {
    /// Whether the current branch is active (tokens should be included in output).
    pub active: bool,
    /// Whether *any* branch in this `#if`/`#elif`/`#else` group has been active.
    /// Once a branch has been taken, subsequent `#elif` and `#else` branches
    /// are skipped even if their conditions evaluate to true.
    pub seen_active: bool,
    /// Whether `#else` has been seen for this group.  A second `#else` or
    /// `#elif` after `#else` is an error.
    pub seen_else: bool,
    /// Source span of the opening `#if`/`#ifdef`/`#ifndef` directive, used
    /// for diagnostic reporting of unterminated conditional blocks.
    pub opening_span: Span,
}

impl ConditionalState {
    /// Create a new conditional state for an `#if`/`#ifdef`/`#ifndef` directive.
    ///
    /// `active` indicates whether the condition evaluated to true (non-zero).
    /// `opening_span` is the source location of the directive for diagnostics.
    #[inline]
    pub fn new(active: bool, opening_span: Span) -> Self {
        ConditionalState {
            active,
            seen_active: active,
            seen_else: false,
            opening_span,
        }
    }
}

// ===========================================================================
// Phase 1 — Trigraph Replacement and Line Splicing
// ===========================================================================

/// Replace C trigraph sequences with their single-character equivalents.
///
/// Trigraphs are three-character sequences beginning with `??` that represent
/// characters not available in some legacy character sets.  C11 retains them
/// for backwards compatibility (§5.2.1.1), though they are rarely used in
/// modern code.
///
/// | Trigraph | Replacement |
/// |----------|-------------|
/// | `??=`    | `#`         |
/// | `??/`    | `\`         |
/// | `??'`    | `^`         |
/// | `??(`    | `[`         |
/// | `??)`    | `]`         |
/// | `??!`    | `|`         |
/// | `??<`    | `{`         |
/// | `??>`    | `}`         |
/// | `??-`    | `~`         |
pub fn phase1_trigraphs(input: &str) -> String {
    let bytes = input.as_bytes();
    let len = bytes.len();
    if len < 3 {
        return input.to_string();
    }

    let mut result = String::with_capacity(len);
    let mut i = 0;

    while i < len {
        if i + 2 < len && bytes[i] == b'?' && bytes[i + 1] == b'?' {
            let replacement = match bytes[i + 2] {
                b'=' => Some('#'),
                b'/' => Some('\\'),
                b'\'' => Some('^'),
                b'(' => Some('['),
                b')' => Some(']'),
                b'!' => Some('|'),
                b'<' => Some('{'),
                b'>' => Some('}'),
                b'-' => Some('~'),
                _ => None,
            };
            if let Some(ch) = replacement {
                result.push(ch);
                i += 3;
                continue;
            }
        }
        // Multi-byte UTF-8 characters: inner continuation bytes (0x80–0xBF)
        // never equal '?' (0x3F), so this byte-by-byte scan is safe.
        result.push(bytes[i] as char);
        i += 1;
    }

    result
}

/// Splice lines that are continued with a backslash immediately before a
/// newline, producing a single logical line.
///
/// Handles both Unix (`\n`) and Windows (`\r\n`) line endings.  The
/// backslash and the following newline (and optional carriage return) are
/// removed, concatenating the physical lines into one logical line.
pub fn phase1_line_splice(input: &str) -> String {
    let bytes = input.as_bytes();
    let len = bytes.len();
    let mut result = String::with_capacity(len);
    let mut i = 0;

    while i < len {
        if bytes[i] == b'\\' {
            if i + 1 < len && bytes[i + 1] == b'\n' {
                i += 2;
                continue;
            }
            if i + 2 < len && bytes[i + 1] == b'\r' && bytes[i + 2] == b'\n' {
                i += 3;
                continue;
            }
        }
        result.push(bytes[i] as char);
        i += 1;
    }

    result
}

// ===========================================================================
// Preprocessing Tokenizer — splits input text into PPToken streams
// ===========================================================================

/// Tokenize a string of (Phase-1-processed) source text into preprocessing
/// tokens.
///
/// This tokeniser is intentionally simpler than the full lexer (Phase 3):
/// - It does NOT distinguish C keywords from identifiers.
/// - It does NOT parse numeric literal suffixes in detail.
/// - It does NOT handle adjacent string literal concatenation.
///
/// Its job is solely to split the input into the coarse token categories
/// that the preprocessor needs: identifiers, numbers, strings, character
/// literals, punctuators, whitespace, and newlines.
pub fn tokenize_preprocessing(input: &str, file_id: u32) -> Vec<PPToken> {
    let bytes = input.as_bytes();
    let len = bytes.len();
    let mut tokens = Vec::with_capacity(len / 4);
    let mut pos: usize = 0;

    while pos < len {
        let start = pos;
        let b = bytes[pos];

        // --- Newline ---
        if b == b'\n' {
            tokens.push(PPToken::new(
                PPTokenKind::Newline,
                "\n",
                Span::new(file_id, start as u32, (start + 1) as u32),
            ));
            pos += 1;
            continue;
        }

        // --- Whitespace (space, tab, \r, form-feed — but NOT \n) ---
        if b == b' ' || b == b'\t' || b == b'\r' || b == 0x0C {
            let ws_start = pos;
            while pos < len {
                let c = bytes[pos];
                if c == b' ' || c == b'\t' || c == b'\r' || c == 0x0C {
                    pos += 1;
                } else {
                    break;
                }
            }
            let text = &input[ws_start..pos];
            tokens.push(PPToken::new(
                PPTokenKind::Whitespace,
                text,
                Span::new(file_id, ws_start as u32, pos as u32),
            ));
            continue;
        }

        // --- Line comment: // ... ---
        if b == b'/' && pos + 1 < len && bytes[pos + 1] == b'/' {
            while pos < len && bytes[pos] != b'\n' {
                pos += 1;
            }
            // C11 §5.1.1.2 — each comment is replaced by one space.
            tokens.push(PPToken::new(
                PPTokenKind::Whitespace,
                " ",
                Span::new(file_id, start as u32, pos as u32),
            ));
            continue;
        }

        // --- Block comment: /* ... */ ---
        if b == b'/' && pos + 1 < len && bytes[pos + 1] == b'*' {
            let cmt_start = pos;
            pos += 2;
            loop {
                if pos + 1 >= len {
                    pos = len;
                    break;
                }
                if bytes[pos] == b'*' && bytes[pos + 1] == b'/' {
                    pos += 2;
                    break;
                }
                pos += 1;
            }
            tokens.push(PPToken::new(
                PPTokenKind::Whitespace,
                " ",
                Span::new(file_id, cmt_start as u32, pos as u32),
            ));
            continue;
        }

        // --- String literal: "..." (with optional prefix L, u, U, u8) ---
        if b == b'"' || is_string_prefix(bytes, pos) {
            let (tok, new_pos) = lex_string_literal(input, pos, file_id);
            tokens.push(tok);
            pos = new_pos;
            continue;
        }

        // --- Character literal: '...' (with optional prefix L, u, U) ---
        if b == b'\'' || is_char_prefix(bytes, pos) {
            let (tok, new_pos) = lex_char_literal(input, pos, file_id);
            tokens.push(tok);
            pos = new_pos;
            continue;
        }

        // --- Identifier: [a-zA-Z_][a-zA-Z0-9_]* ---
        if b.is_ascii_alphabetic() || b == b'_' {
            let id_start = pos;
            pos += 1;
            while pos < len {
                let c = bytes[pos];
                if c.is_ascii_alphanumeric() || c == b'_' {
                    pos += 1;
                } else {
                    break;
                }
            }
            let text = &input[id_start..pos];
            tokens.push(PPToken::new(
                PPTokenKind::Identifier,
                text,
                Span::new(file_id, id_start as u32, pos as u32),
            ));
            continue;
        }

        // --- Number: starts with digit, or `.` followed by digit ---
        if b.is_ascii_digit() || (b == b'.' && pos + 1 < len && bytes[pos + 1].is_ascii_digit()) {
            let num_start = pos;
            pos += 1;
            while pos < len {
                let c = bytes[pos];
                let is_number_continuation = c.is_ascii_alphanumeric()
                    || c == b'_'
                    || c == b'.'
                    || ((c == b'+' || c == b'-')
                        && pos > num_start + 1
                        && matches!(bytes[pos - 1], b'e' | b'E' | b'p' | b'P'));
                if is_number_continuation {
                    pos += 1;
                } else {
                    break;
                }
            }
            let text = &input[num_start..pos];
            tokens.push(PPToken::new(
                PPTokenKind::Number,
                text,
                Span::new(file_id, num_start as u32, pos as u32),
            ));
            continue;
        }

        // --- Multi-character punctuators (longest match) ---
        if let Some((punct_text, plen)) = match_punctuator(bytes, pos) {
            tokens.push(PPToken::new(
                PPTokenKind::Punctuator,
                punct_text,
                Span::new(file_id, start as u32, (start + plen) as u32),
            ));
            pos += plen;
            continue;
        }

        // --- Fallback: single character ---
        let ch = get_char_at(input, pos);
        let ch_len = ch.len_utf8();
        pos += ch_len;
        if ch_len > 1 {
            // Multi-byte UTF-8 or PUA — treat as identifier for pass-through.
            tokens.push(PPToken::new(
                PPTokenKind::Identifier,
                &input[start..pos],
                Span::new(file_id, start as u32, pos as u32),
            ));
        } else {
            tokens.push(PPToken::new(
                PPTokenKind::Punctuator,
                &input[start..pos],
                Span::new(file_id, start as u32, pos as u32),
            ));
        }
    }

    // Append EOF sentinel.
    tokens.push(PPToken::eof(Span::new(file_id, pos as u32, pos as u32)));
    tokens
}

// ---------------------------------------------------------------------------
// Tokenizer helper functions
// ---------------------------------------------------------------------------

/// Check if position starts a string literal prefix (`u8"`, `u"`, `U"`, `L"`).
fn is_string_prefix(bytes: &[u8], pos: usize) -> bool {
    let len = bytes.len();
    if pos >= len {
        return false;
    }
    match bytes[pos] {
        b'L' | b'U' => pos + 1 < len && bytes[pos + 1] == b'"',
        b'u' => {
            (pos + 1 < len && bytes[pos + 1] == b'"')
                || (pos + 2 < len && bytes[pos + 1] == b'8' && bytes[pos + 2] == b'"')
        }
        _ => false,
    }
}

/// Check if position starts a character literal prefix (`L'`, `u'`, `U'`).
fn is_char_prefix(bytes: &[u8], pos: usize) -> bool {
    let len = bytes.len();
    if pos >= len {
        return false;
    }
    matches!(bytes[pos], b'L' | b'U' | b'u') && pos + 1 < len && bytes[pos + 1] == b'\''
}

/// Lex a string literal starting at `pos`.
fn lex_string_literal(input: &str, pos: usize, file_id: u32) -> (PPToken, usize) {
    let bytes = input.as_bytes();
    let len = bytes.len();
    let start = pos;
    let mut i = pos;

    // Skip prefix.
    match bytes[i] {
        b'L' | b'U' => i += 1,
        b'u' => {
            i += 1;
            if i < len && bytes[i] == b'8' {
                i += 1;
            }
        }
        _ => {}
    }
    // Skip opening `"`.
    if i < len && bytes[i] == b'"' {
        i += 1;
    }
    // Scan until closing `"`.
    while i < len {
        if bytes[i] == b'\\' && i + 1 < len {
            i += 2;
        } else if bytes[i] == b'"' {
            i += 1;
            break;
        } else if bytes[i] == b'\n' {
            break;
        } else {
            i += 1;
        }
    }
    let text = &input[start..i];
    (
        PPToken::new(
            PPTokenKind::StringLiteral,
            text,
            Span::new(file_id, start as u32, i as u32),
        ),
        i,
    )
}

/// Lex a character literal starting at `pos`.
fn lex_char_literal(input: &str, pos: usize, file_id: u32) -> (PPToken, usize) {
    let bytes = input.as_bytes();
    let len = bytes.len();
    let start = pos;
    let mut i = pos;

    // Skip prefix.
    if matches!(bytes[i], b'L' | b'U' | b'u') {
        i += 1;
    }
    if i < len && bytes[i] == b'\'' {
        i += 1;
    }
    while i < len {
        if bytes[i] == b'\\' && i + 1 < len {
            i += 2;
        } else if bytes[i] == b'\'' {
            i += 1;
            break;
        } else if bytes[i] == b'\n' {
            break;
        } else {
            i += 1;
        }
    }
    let text = &input[start..i];
    (
        PPToken::new(
            PPTokenKind::CharLiteral,
            text,
            Span::new(file_id, start as u32, i as u32),
        ),
        i,
    )
}

/// Match the longest punctuator at the given position.
fn match_punctuator(bytes: &[u8], pos: usize) -> Option<(&'static str, usize)> {
    let len = bytes.len();
    let remaining = len - pos;

    // 3-character punctuators.
    if remaining >= 3 {
        let t = &bytes[pos..pos + 3];
        let m = match t {
            b"<<=" => Some("<<="),
            b">>=" => Some(">>="),
            b"..." => Some("..."),
            _ => None,
        };
        if m.is_some() {
            return m.map(|s| (s, 3));
        }
    }

    // 2-character punctuators.
    if remaining >= 2 {
        let t = &bytes[pos..pos + 2];
        let m = match t {
            b"++" => Some("++"),
            b"--" => Some("--"),
            b"->" => Some("->"),
            b"<<" => Some("<<"),
            b">>" => Some(">>"),
            b"<=" => Some("<="),
            b">=" => Some(">="),
            b"==" => Some("=="),
            b"!=" => Some("!="),
            b"&&" => Some("&&"),
            b"||" => Some("||"),
            b"+=" => Some("+="),
            b"-=" => Some("-="),
            b"*=" => Some("*="),
            b"/=" => Some("/="),
            b"%=" => Some("%="),
            b"&=" => Some("&="),
            b"|=" => Some("|="),
            b"^=" => Some("^="),
            b"##" => Some("##"),
            _ => None,
        };
        if m.is_some() {
            return m.map(|s| (s, 2));
        }
    }

    // 1-character punctuators.
    if remaining >= 1 {
        let text: &'static str = match bytes[pos] {
            b'+' => "+",
            b'-' => "-",
            b'*' => "*",
            b'/' => "/",
            b'%' => "%",
            b'&' => "&",
            b'|' => "|",
            b'^' => "^",
            b'~' => "~",
            b'!' => "!",
            b'<' => "<",
            b'>' => ">",
            b'=' => "=",
            b'(' => "(",
            b')' => ")",
            b'[' => "[",
            b']' => "]",
            b'{' => "{",
            b'}' => "}",
            b';' => ";",
            b':' => ":",
            b',' => ",",
            b'.' => ".",
            b'?' => "?",
            b'#' => "#",
            _ => return None,
        };
        return Some((text, 1));
    }

    None
}

/// Get the Unicode character at a byte position in a string.
#[inline]
fn get_char_at(s: &str, pos: usize) -> char {
    s[pos..].chars().next().unwrap_or('\0')
}

// ===========================================================================
// Preprocessor — the main state machine
// ===========================================================================

/// The main preprocessor state machine.
///
/// Coordinates all seven submodules to transform raw C source text into
/// a fully macro-expanded preprocessing token stream.  The preprocessor
/// is the first stage of the compilation pipeline and is invoked by the
/// CLI driver (`src/main.rs`).
///
/// # Lifetime Parameter
///
/// The `'a` lifetime binds the preprocessor to its borrowed infrastructure:
/// source map, diagnostic engine, and string interner.  All three are owned
/// by the compilation driver and passed by mutable reference.
pub struct Preprocessor<'a> {
    /// Source file registry for tracking loaded files and line/column lookups.
    pub source_map: &'a mut SourceMap,
    /// Diagnostic reporting engine for errors, warnings, and notes.
    pub diagnostics: &'a mut DiagnosticEngine,
    /// Target architecture — determines predefined macros and type sizes.
    pub target: Target,
    /// String interner for deduplicating identifiers and macro names.
    pub interner: &'a mut Interner,
    /// Macro definitions: macro name → [`MacroDef`].
    pub macro_defs: FxHashMap<String, MacroDef>,
    /// User include search paths from `-I` flags (searched first for `"..."` includes).
    pub include_paths: Vec<PathBuf>,
    /// System include paths (searched for `<...>` includes).
    pub system_include_paths: Vec<PathBuf>,
    /// Command-line `-D` defines: `(name, value)` pairs.
    pub cli_defines: Vec<(String, String)>,
    /// Current include nesting depth for depth-limit enforcement.
    pub include_depth: usize,
    /// Maximum include nesting depth (default: 200, matching GCC).
    /// Exceeding this limit produces a clean diagnostic error rather than
    /// a stack overflow, protecting against deeply-chained or circular
    /// `#include` directives.
    pub max_include_depth: usize,
    /// Stack of file paths currently being processed, from outermost to
    /// innermost.  Used for circular `#include` detection — if a file
    /// appears in this stack when an `#include` directive tries to include
    /// it again, the preprocessor emits an error instead of infinitely
    /// recursing.
    include_stack: Vec<PathBuf>,
    /// Maximum recursion depth for macro expansion (default: 512).
    pub max_recursion_depth: usize,
    /// Stack of conditional compilation states for nested `#if`/`#endif` blocks.
    conditional_stack: Vec<ConditionalState>,
}

impl<'a> Preprocessor<'a> {
    /// Create a new preprocessor with the given infrastructure references.
    ///
    /// Initialises the macro definition table with predefined macros for the
    /// target architecture (via [`predefined::register_predefined_macros`]),
    /// sets the recursion depth limit to 512, and prepares an empty include
    /// path list.
    pub fn new(
        source_map: &'a mut SourceMap,
        diagnostics: &'a mut DiagnosticEngine,
        target: Target,
        interner: &'a mut Interner,
    ) -> Self {
        let mut macro_defs = FxHashMap::default();

        // Register compiler-predefined macros (__STDC__, __linux__, arch-specific, etc.).
        predefined::register_predefined_macros(&mut macro_defs, &target);

        Preprocessor {
            source_map,
            diagnostics,
            target,
            interner,
            macro_defs,
            include_paths: Vec::new(),
            system_include_paths: Vec::new(),
            cli_defines: Vec::new(),
            include_depth: 0,
            max_include_depth: 200,
            include_stack: Vec::new(),
            max_recursion_depth: 512,
            conditional_stack: Vec::new(),
        }
    }

    /// Add a user include search path (`-I` flag).
    pub fn add_include_path(&mut self, path: &str) {
        self.include_paths.push(PathBuf::from(path));
    }

    /// Add a system include search path.
    pub fn add_system_include_path(&mut self, path: &str) {
        self.system_include_paths.push(PathBuf::from(path));
    }

    /// Define a macro from a `-D` command-line flag.
    ///
    /// `name` is the macro name, `value` is the replacement text.
    /// An empty `value` defines the macro with an empty replacement list.
    pub fn add_define(&mut self, name: &str, value: &str) {
        // Store for later processing.
        self.cli_defines.push((name.to_string(), value.to_string()));

        // Also immediately register as an object-like macro.
        let replacement = if value.is_empty() {
            Vec::new()
        } else {
            // Tokenise the value string into preprocessing tokens.
            tokenize_preprocessing(value, 0)
                .into_iter()
                .filter(|t| !t.is_eof())
                .collect()
        };

        let _ = self.interner.intern(name);
        self.macro_defs.insert(
            name.to_string(),
            MacroDef {
                name: name.to_string(),
                kind: MacroKind::ObjectLike,
                replacement,
                is_predefined: false,
                definition_span: Span::dummy(),
            },
        );
    }

    /// Remove a macro definition, implementing the `-U` command-line flag.
    ///
    /// This is applied after all `-D` defines during preprocessor
    /// initialization, matching GCC's behaviour where `-U` overrides
    /// earlier `-D` for the same macro name.
    pub fn add_undef(&mut self, name: &str) {
        self.macro_defs.remove(name);
    }

    /// Preprocess a source file, returning the fully macro-expanded token stream.
    ///
    /// This is the main entry point for the preprocessor.  It:
    /// 1. Reads the file with PUA encoding for non-UTF-8 bytes.
    /// 2. Applies Phase 1 (trigraph replacement + line splicing).
    /// 3. Tokenizes into preprocessing tokens.
    /// 4. Processes directives and expands macros (Phase 2).
    /// 5. Returns the expanded token stream (minus whitespace/newlines) for the lexer.
    ///
    /// # Errors
    ///
    /// Returns `Err(())` if the file cannot be read or if a fatal preprocessing
    /// error occurs (e.g., `#error` directive).  Non-fatal errors (warnings,
    /// recoverable syntax issues) are accumulated in the diagnostic engine.
    #[allow(clippy::result_unit_err)]
    pub fn preprocess_file(&mut self, filename: &str) -> Result<Vec<PPToken>, ()> {
        // Step 1: Read source file with PUA encoding.
        let path = Path::new(filename);
        let source = match read_source_file(path) {
            Ok(s) => s,
            Err(e) => {
                self.diagnostics.emit(Diagnostic::error(
                    Span::dummy(),
                    format!("cannot open source file '{}': {}", filename, e),
                ));
                return Err(());
            }
        };

        // Push the initial source file onto the include stack so that
        // circular detection works when an included file tries to
        // re-include the original source.
        let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        self.include_stack.push(canonical);

        // Step 2: Register the file in the source map.
        let file_id = self
            .source_map
            .add_file(filename.to_string(), source.clone());

        // Step 3: Apply Phase 1 transformations.
        let after_trigraphs = phase1_trigraphs(&source);
        let after_splice = phase1_line_splice(&after_trigraphs);

        // Step 4: Tokenize into preprocessing tokens.
        let tokens = tokenize_preprocessing(&after_splice, file_id);

        // Step 5: Process directives and expand macros (Phase 2).
        let expanded = self.process_tokens(&tokens);

        // Pop the initial source file from the include stack.
        self.include_stack.pop();

        // Step 6: Check for unterminated conditional blocks.
        if let Some(cond) = self.conditional_stack.last() {
            self.diagnostics.emit(Diagnostic::error(
                cond.opening_span,
                "unterminated #if/#ifdef/#ifndef — missing #endif",
            ));
        }

        if self.diagnostics.has_errors() {
            return Err(());
        }

        Ok(expanded)
    }

    /// Process a token stream, handling directives and macro expansion.
    ///
    /// This is the main Phase 2 processing loop.  It iterates through the
    /// preprocessing token stream line by line:
    /// - Lines beginning with `#` are dispatched to directive handling.
    /// - Non-directive lines in active conditional blocks are macro-expanded.
    /// - Lines in inactive conditional blocks are skipped (except for
    ///   nesting-tracking directives like `#if`/`#endif`).
    fn process_tokens(&mut self, tokens: &[PPToken]) -> Vec<PPToken> {
        let mut output = Vec::new();
        let mut pos = 0;
        let len = tokens.len();

        while pos < len {
            // End-of-file sentinel — nothing more to process.
            if tokens[pos].kind == PPTokenKind::EndOfFile {
                break;
            }

            // Skip to the first non-whitespace token to check for directive.
            let line_start = pos;

            // Collect whitespace at the start of the line.
            while pos < len && tokens[pos].kind == PPTokenKind::Whitespace {
                pos += 1;
            }

            // Check for directive: `#` as first non-whitespace token.
            if pos < len && tokens[pos].kind == PPTokenKind::Punctuator && tokens[pos].text == "#" {
                let hash_pos = pos;
                pos += 1; // skip `#`

                // Skip whitespace after `#`.
                while pos < len && tokens[pos].kind == PPTokenKind::Whitespace {
                    pos += 1;
                }

                // Collect tokens until end of line (newline or EOF).
                let mut directive_tokens = Vec::new();
                while pos < len
                    && tokens[pos].kind != PPTokenKind::Newline
                    && tokens[pos].kind != PPTokenKind::EndOfFile
                {
                    directive_tokens.push(tokens[pos].clone());
                    pos += 1;
                }

                // Skip the newline.
                if pos < len && tokens[pos].kind == PPTokenKind::Newline {
                    pos += 1;
                }

                // Process the directive.  #include may produce output tokens.
                if let Some(included) =
                    self.process_directive_line(&tokens[hash_pos], &directive_tokens)
                {
                    output.extend(included);
                }
                continue;
            }

            // Not a directive line — check if we are in an active region.
            if !self.is_active() {
                // Skip tokens until end of line.
                while pos < len
                    && tokens[pos].kind != PPTokenKind::Newline
                    && tokens[pos].kind != PPTokenKind::EndOfFile
                {
                    pos += 1;
                }
                if pos < len && tokens[pos].kind == PPTokenKind::Newline {
                    pos += 1;
                }
                continue;
            }

            // Active region — collect and macro-expand the line.
            //
            // C11 §6.10.3.4: After substitution, replacement tokens are
            // rescanned together with subsequent source tokens for further
            // macro names to replace.  We collect all non-directive tokens on
            // this line, then run them through the full MacroExpander which
            // handles multi-level expansion and paint-marker recursion
            // protection correctly.
            let restore_pos = line_start;
            pos = restore_pos;

            // Collect raw tokens for the current line.  In C preprocessing,
            // function-like macro invocations may span multiple source lines
            // (after line-splicing).  We must collect tokens across newlines
            // when a function-like macro argument list is open (unbalanced
            // parentheses).
            let mut line_tokens: Vec<PPToken> = Vec::new();
            let mut paren_depth: i32 = 0;
            let mut in_macro_args = false;
            while pos < len && tokens[pos].kind != PPTokenKind::EndOfFile {
                let tok = &tokens[pos];

                // If we hit a newline and we're NOT inside macro arguments,
                // the line ends here.
                if tok.kind == PPTokenKind::Newline && !in_macro_args {
                    break;
                }

                // Skip newlines that occur inside macro argument lists (treat
                // them as whitespace — C11 §5.1.1.2 translation phases).
                if tok.kind == PPTokenKind::Newline && in_macro_args {
                    pos += 1;
                    continue;
                }

                // Track parenthesis depth for macro argument spanning.
                if tok.kind == PPTokenKind::Punctuator {
                    match tok.text.as_str() {
                        "(" => {
                            paren_depth += 1;
                            if !in_macro_args && paren_depth == 1 {
                                // Check if the preceding non-whitespace token
                                // is a function-like macro name.
                                let last_ident = line_tokens
                                    .iter()
                                    .rev()
                                    .find(|t| t.kind != PPTokenKind::Whitespace);
                                if let Some(prev) = last_ident {
                                    if prev.kind == PPTokenKind::Identifier {
                                        if let Some(md) = self.macro_defs.get(&prev.text) {
                                            if matches!(md.kind, MacroKind::FunctionLike { .. }) {
                                                in_macro_args = true;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        ")" => {
                            paren_depth -= 1;
                            if in_macro_args && paren_depth == 0 {
                                in_macro_args = false;
                            }
                        }
                        _ => {}
                    }
                }

                if tok.kind != PPTokenKind::Whitespace || !line_tokens.is_empty() {
                    line_tokens.push(tok.clone());
                }
                pos += 1;
            }

            // Expand all macros (including multi-level rescanning) using
            // the full MacroExpander.
            if !line_tokens.is_empty() {
                let expanded = {
                    let mut expander = macro_expander::MacroExpander::new(
                        &self.macro_defs,
                        &mut *self.diagnostics,
                        self.max_recursion_depth,
                    );
                    expander.expand_tokens(&line_tokens)
                };
                output.extend(expanded);
            }

            // Emit a newline token so that `-E` output preserves line
            // structure.  The newline is part of the preprocessed output
            // (C11 §5.1.1.2 phase 4 produces a stream of pp-tokens and
            // white-space characters).
            if pos < len && tokens[pos].kind == PPTokenKind::Newline {
                output.push(tokens[pos].clone());
                pos += 1;
            }
        }

        // Always append an EndOfFile sentinel so downstream consumers can rely
        // on the token stream being terminated.
        let eof_span = if let Some(last) = tokens.last() {
            last.span
        } else {
            Span::dummy()
        };
        output.push(PPToken::eof(eof_span));

        output
    }

    /// Returns `true` if the current preprocessing position is in an active
    /// conditional compilation region (i.e., all enclosing `#if` blocks are active).
    fn is_active(&self) -> bool {
        self.conditional_stack.iter().all(|c| c.active)
    }

    /// Process a single preprocessor directive line.
    ///
    /// `hash_token` is the `#` token, `tokens` are the remaining tokens on the
    /// directive line (after `#` and whitespace).
    fn process_directive_line(
        &mut self,
        _hash_token: &PPToken,
        tokens: &[PPToken],
    ) -> Option<Vec<PPToken>> {
        // Null directive: `#` alone on a line — valid C11 no-op.
        if tokens.is_empty() {
            return None;
        }

        let directive = &tokens[0];
        let rest = if tokens.len() > 1 { &tokens[1..] } else { &[] };

        // Strip leading whitespace from rest.
        let rest: Vec<PPToken> = rest
            .iter()
            .skip_while(|t| t.kind == PPTokenKind::Whitespace)
            .cloned()
            .collect();

        let directive_name = directive.text.as_str();

        // Directives that must be processed even in inactive regions
        // (for nesting tracking).
        match directive_name {
            "if" | "ifdef" | "ifndef" => {
                if !self.is_active() {
                    // In inactive region: push a nested inactive conditional.
                    self.conditional_stack
                        .push(ConditionalState::new(false, directive.span));
                    return None;
                }
            }
            "elif" => {
                self.process_elif(directive, &rest);
                return None;
            }
            "else" => {
                self.process_else(directive);
                return None;
            }
            "endif" => {
                self.process_endif(directive);
                return None;
            }
            _ => {
                if !self.is_active() {
                    return None; // Skip all other directives in inactive regions.
                }
            }
        }

        // Active-region directives.
        match directive_name {
            "define" => {
                self.process_define(&rest);
                None
            }
            "undef" => {
                self.process_undef(&rest);
                None
            }
            "include" => {
                // process_include returns Ok(tokens) on success.
                self.process_include(&rest).ok()
            }
            "if" => {
                self.process_if(&rest, directive.span);
                None
            }
            "ifdef" => {
                self.process_ifdef(&rest, directive.span);
                None
            }
            "ifndef" => {
                self.process_ifndef(&rest, directive.span);
                None
            }
            "error" => {
                self.process_error(&rest, directive.span);
                None
            }
            "warning" => {
                self.process_warning(&rest, directive.span);
                None
            }
            "line" => {
                self.process_line_directive(&rest);
                None
            }
            "pragma" => {
                self.process_pragma(&rest);
                None
            }
            _ => {
                // Unknown directive — emit diagnostic.
                self.diagnostics.emit_warning(
                    directive.span,
                    format!(
                        "unknown preprocessing directive '#{}' — ignored",
                        directive_name
                    ),
                );
                None
            }
        }
    }

    // -- Directive handlers ------------------------------------------------

    /// Process `#define` directive.
    fn process_define(&mut self, tokens: &[PPToken]) {
        if tokens.is_empty() {
            self.diagnostics
                .emit_error(Span::dummy(), "expected macro name after #define");
            return;
        }

        let name_token = &tokens[0];
        if name_token.kind != PPTokenKind::Identifier {
            self.diagnostics.emit_error(
                name_token.span,
                format!("macro name '{}' is not an identifier", name_token.text),
            );
            return;
        }

        let name = name_token.text.clone();
        let _ = self.interner.intern(&name);

        let mut idx = 1;

        // Check for function-like macro: `(` immediately follows name (no space).
        let kind;
        if idx < tokens.len()
            && tokens[idx].kind == PPTokenKind::Punctuator
            && tokens[idx].text == "("
        {
            // Function-like macro — parse parameter list.
            idx += 1; // skip `(`
            let mut params = Vec::new();
            let mut variadic = false;

            loop {
                // Skip whitespace.
                while idx < tokens.len() && tokens[idx].kind == PPTokenKind::Whitespace {
                    idx += 1;
                }
                if idx >= tokens.len() {
                    break;
                }

                // Check for `)`.
                if tokens[idx].kind == PPTokenKind::Punctuator && tokens[idx].text == ")" {
                    idx += 1;
                    break;
                }

                // Check for `...` (variadic).
                if tokens[idx].kind == PPTokenKind::Punctuator && tokens[idx].text == "..." {
                    variadic = true;
                    idx += 1;
                    // Expect `)` next.
                    while idx < tokens.len() && tokens[idx].kind == PPTokenKind::Whitespace {
                        idx += 1;
                    }
                    if idx < tokens.len()
                        && tokens[idx].kind == PPTokenKind::Punctuator
                        && tokens[idx].text == ")"
                    {
                        idx += 1;
                    }
                    break;
                }

                // Parameter name.
                if tokens[idx].kind == PPTokenKind::Identifier {
                    params.push(tokens[idx].text.clone());
                    idx += 1;

                    // Skip whitespace.
                    while idx < tokens.len() && tokens[idx].kind == PPTokenKind::Whitespace {
                        idx += 1;
                    }

                    // Check for `,` or `)`.
                    if idx < tokens.len()
                        && tokens[idx].kind == PPTokenKind::Punctuator
                        && tokens[idx].text == ","
                    {
                        idx += 1; // skip comma
                    }
                } else {
                    self.diagnostics.emit_error(
                        tokens[idx].span,
                        "expected parameter name or '...' in macro parameter list",
                    );
                    return;
                }
            }

            kind = MacroKind::FunctionLike { params, variadic };
        } else {
            // Object-like macro — skip whitespace before replacement.
            while idx < tokens.len() && tokens[idx].kind == PPTokenKind::Whitespace {
                idx += 1;
            }
            kind = MacroKind::ObjectLike;
        }

        // Collect replacement tokens (rest of the line).
        let replacement: Vec<PPToken> = tokens[idx..]
            .iter()
            .filter(|t| t.kind != PPTokenKind::EndOfFile)
            .cloned()
            .collect();

        // Trim trailing whitespace from replacement.
        let replacement: Vec<PPToken> = {
            let mut r = replacement;
            while r
                .last()
                .map_or(false, |t| t.kind == PPTokenKind::Whitespace)
            {
                r.pop();
            }
            r
        };

        self.macro_defs.insert(
            name.clone(),
            MacroDef {
                name,
                kind,
                replacement,
                is_predefined: false,
                definition_span: name_token.span,
            },
        );
    }

    /// Process `#undef` directive.
    fn process_undef(&mut self, tokens: &[PPToken]) {
        if tokens.is_empty() {
            self.diagnostics
                .emit_error(Span::dummy(), "expected macro name after #undef");
            return;
        }
        let name = &tokens[0].text;
        if let Some(def) = self.macro_defs.get(name) {
            if def.is_predefined {
                self.diagnostics.emit_warning(
                    tokens[0].span,
                    format!("undefining predefined macro '{}'", name),
                );
            }
        }
        self.macro_defs.remove(name.as_str());
    }

    /// Process `#include` directive.
    fn process_include(&mut self, tokens: &[PPToken]) -> Result<Vec<PPToken>, ()> {
        if tokens.is_empty() {
            self.diagnostics
                .emit_error(Span::dummy(), "expected header name after #include");
            return Err(());
        }

        let directive_span = tokens[0].span;

        // Determine header name and type (system vs user).
        let (header_name, is_system) = match tokens[0].kind {
            PPTokenKind::HeaderName => {
                let text = &tokens[0].text;
                if text.starts_with('<') && text.ends_with('>') {
                    (text[1..text.len() - 1].to_string(), true)
                } else if text.starts_with('"') && text.ends_with('"') {
                    (text[1..text.len() - 1].to_string(), false)
                } else {
                    (text.clone(), false)
                }
            }
            PPTokenKind::StringLiteral => {
                let text = &tokens[0].text;
                let inner = if text.starts_with('"') && text.ends_with('"') && text.len() >= 2 {
                    text[1..text.len() - 1].to_string()
                } else {
                    text.clone()
                };
                (inner, false)
            }
            PPTokenKind::Punctuator if tokens[0].text == "<" => {
                // Collect tokens until `>`.
                let mut header = String::new();
                let mut i = 1;
                while i < tokens.len() {
                    if tokens[i].kind == PPTokenKind::Punctuator && tokens[i].text == ">" {
                        break;
                    }
                    header.push_str(&tokens[i].text);
                    i += 1;
                }
                (header, true)
            }
            _ => {
                self.diagnostics
                    .emit_error(tokens[0].span, "expected header name in #include directive");
                return Err(());
            }
        };

        // Determine the including file for relative path resolution.
        let including_file = self
            .source_map
            .get_filename(directive_span.file_id)
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from("."));

        // Enforce include depth limit before recursing.  This check uses
        // the persistent `include_depth` counter on `self`, which is
        // incremented for every nested `#include` and decremented when the
        // included file finishes processing.
        if self.include_depth >= self.max_include_depth {
            self.diagnostics.emit_error(
                directive_span,
                format!(
                    "#include nested too deeply ({} levels, maximum is {})",
                    self.include_depth, self.max_include_depth
                ),
            );
            return Err(());
        }

        // Build a temporary handler for file resolution and guard checks.
        let handler = include_handler::IncludeHandler::new(
            self.include_paths.clone(),
            self.system_include_paths.clone(),
        );

        // Resolve the header to an absolute path.
        let resolved_path = match handler.resolve_include(&header_name, is_system, &including_file)
        {
            Some(path) => path,
            None => {
                self.diagnostics
                    .emit_error(directive_span, format!("'{}' file not found", header_name));
                return Err(());
            }
        };

        // Check include guards and #pragma once.
        if handler.should_skip_file(&resolved_path, &self.macro_defs) {
            return Ok(Vec::new());
        }

        // Canonicalize the resolved path for reliable circular detection.
        let canonical_path =
            std::fs::canonicalize(&resolved_path).unwrap_or_else(|_| resolved_path.clone());

        // Circular include detection: check if this file is already being
        // processed in the current include chain.  This uses the persistent
        // `include_stack` on `self` (not a per-call handler) so that the
        // stack survives across recursive `process_include` calls.
        if self.include_stack.contains(&canonical_path) {
            // Build a human-readable chain for the diagnostic.
            let chain: Vec<String> = self
                .include_stack
                .iter()
                .map(|p| p.display().to_string())
                .collect();
            let chain_str = chain.join(" → ");
            self.diagnostics.emit_error(
                directive_span,
                format!(
                    "circular #include detected: {} → {}",
                    chain_str,
                    canonical_path.display()
                ),
            );
            return Err(());
        }

        // Read the file contents using PUA-encoded reader.
        let source_content = match crate::common::encoding::read_source_file(&resolved_path) {
            Ok(content) => content,
            Err(e) => {
                self.diagnostics.emit_error(
                    directive_span,
                    format!("cannot read '{}': {}", resolved_path.display(), e),
                );
                return Err(());
            }
        };

        // Register the file in the source map.
        let file_id = self.source_map.add_file(
            resolved_path.to_string_lossy().to_string(),
            source_content.clone(),
        );

        // Apply Phase 1 transforms (trigraphs and line splicing).
        let processed = phase1_line_splice(&phase1_trigraphs(&source_content));

        // Tokenize the included source.
        let included_tokens = tokenize_preprocessing(&processed, file_id);

        // Push file onto the persistent include stack for circular detection
        // and increment depth counter, then recursively preprocess.
        self.include_stack.push(canonical_path.clone());
        self.include_depth += 1;
        let result = self.process_tokens(&included_tokens);
        self.include_depth -= 1;
        self.include_stack.pop();

        // Filter out the EOF token from included output to avoid premature
        // termination of the parent file's token stream.
        let filtered: Vec<PPToken> = result
            .into_iter()
            .filter(|t| t.kind != PPTokenKind::EndOfFile)
            .collect();

        Ok(filtered)
    }

    /// Process `#if` directive.
    fn process_if(&mut self, tokens: &[PPToken], directive_span: Span) {
        // Evaluate the expression — for now, treat non-empty as true.
        let value = self.evaluate_condition(tokens);
        self.conditional_stack
            .push(ConditionalState::new(value, directive_span));
    }

    /// Process `#ifdef` directive.
    fn process_ifdef(&mut self, tokens: &[PPToken], directive_span: Span) {
        let is_defined = if let Some(name_tok) = tokens.first() {
            self.macro_defs.contains_key(&name_tok.text)
        } else {
            self.diagnostics
                .emit_error(directive_span, "expected macro name after #ifdef");
            false
        };
        self.conditional_stack
            .push(ConditionalState::new(is_defined, directive_span));
    }

    /// Process `#ifndef` directive.
    fn process_ifndef(&mut self, tokens: &[PPToken], directive_span: Span) {
        let is_defined = if let Some(name_tok) = tokens.first() {
            self.macro_defs.contains_key(&name_tok.text)
        } else {
            self.diagnostics
                .emit_error(directive_span, "expected macro name after #ifndef");
            false
        };
        self.conditional_stack
            .push(ConditionalState::new(!is_defined, directive_span));
    }

    /// Process `#elif` directive.
    fn process_elif(&mut self, directive_token: &PPToken, tokens: &[PPToken]) {
        if self.conditional_stack.is_empty() {
            self.diagnostics
                .emit_error(directive_token.span, "#elif without matching #if");
            return;
        }
        let top = self.conditional_stack.last_mut().unwrap();
        if top.seen_else {
            self.diagnostics
                .emit_error(directive_token.span, "#elif after #else");
            return;
        }
        if top.seen_active {
            // A previous branch was taken — deactivate.
            top.active = false;
        } else {
            // No branch taken yet — evaluate condition using full evaluator.
            let value = self.evaluate_condition(tokens);
            let top = self.conditional_stack.last_mut().unwrap();
            top.active = value;
            if value {
                top.seen_active = true;
            }
        }
    }

    /// Process `#else` directive.
    fn process_else(&mut self, directive_token: &PPToken) {
        if let Some(top) = self.conditional_stack.last_mut() {
            if top.seen_else {
                self.diagnostics
                    .emit_error(directive_token.span, "duplicate #else");
                return;
            }
            top.seen_else = true;
            top.active = !top.seen_active;
        } else {
            self.diagnostics
                .emit_error(directive_token.span, "#else without matching #if");
        }
    }

    /// Process `#endif` directive.
    fn process_endif(&mut self, directive_token: &PPToken) {
        if self.conditional_stack.pop().is_none() {
            self.diagnostics
                .emit_error(directive_token.span, "#endif without matching #if");
        }
    }

    /// Process `#error` directive.
    fn process_error(&mut self, tokens: &[PPToken], directive_span: Span) {
        let msg: String = tokens
            .iter()
            .map(|t| t.text.as_str())
            .collect::<Vec<_>>()
            .join("");
        self.diagnostics
            .emit_error(directive_span, format!("#error {}", msg.trim()));
    }

    /// Process `#warning` directive.
    fn process_warning(&mut self, tokens: &[PPToken], directive_span: Span) {
        let msg: String = tokens
            .iter()
            .map(|t| t.text.as_str())
            .collect::<Vec<_>>()
            .join("");
        self.diagnostics
            .emit_warning(directive_span, format!("#warning {}", msg.trim()));
    }

    /// Process `#line` directive.
    fn process_line_directive(&mut self, tokens: &[PPToken]) {
        if tokens.is_empty() {
            self.diagnostics
                .emit_error(Span::dummy(), "expected line number after #line");
            return;
        }
        // Parse the line number from the first token.
        if let Ok(line_num) = tokens[0].text.parse::<u32>() {
            // Optionally parse filename.
            let filename = tokens
                .get(1)
                .and_then(|t| {
                    if t.kind == PPTokenKind::Whitespace {
                        tokens.get(2)
                    } else {
                        Some(t)
                    }
                })
                .and_then(|t| {
                    if t.kind == PPTokenKind::StringLiteral {
                        let s = &t.text;
                        if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
                            Some(s[1..s.len() - 1].to_string())
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                });
            let file_id = tokens[0].span.file_id;
            let offset = tokens[0].span.start;
            self.source_map
                .add_line_directive(crate::common::source_map::LineDirective {
                    file_id,
                    directive_offset: offset,
                    new_line: line_num,
                    new_filename: filename,
                });
        } else {
            self.diagnostics.emit_error(
                tokens[0].span,
                format!(
                    "invalid line number '{}' in #line directive",
                    tokens[0].text
                ),
            );
        }
    }

    /// Process `#pragma` directive.
    fn process_pragma(&mut self, tokens: &[PPToken]) {
        if tokens.is_empty() {
            return; // Empty pragma — ignore.
        }
        match tokens[0].text.as_str() {
            "once" => {
                // Mark the current file to prevent re-inclusion.
                // (Tracked by include_handler in the full implementation.)
            }
            "pack" | "GCC" | "clang" => {
                // Recognized but not fully handled at the preprocessor level.
                // The parser/sema will handle pack push/pop semantics.
            }
            _ => {
                // Unknown pragmas are silently ignored per C11 §6.10.6.
            }
        }
    }

    // -- Macro expansion ---------------------------------------------------

    /// Evaluate a preprocessor condition expression.
    ///
    /// This performs the full C11 §6.10.1 evaluation sequence:
    /// 1. Resolve `defined(X)` / `defined X` operators to `1` / `0` tokens
    /// 2. Macro-expand remaining tokens
    /// 3. Replace any remaining identifiers with `0` (handled by evaluator)
    /// 4. Evaluate the resulting integer constant expression
    fn evaluate_condition(&mut self, tokens: &[PPToken]) -> bool {
        // Step 1: Resolve `defined` operators BEFORE macro expansion.
        let with_defined = directives::resolve_defined_operators(tokens, &self.macro_defs);
        // Step 2: Macro-expand remaining tokens.
        let expanded = {
            let mut expander = macro_expander::MacroExpander::new(
                &self.macro_defs,
                &mut *self.diagnostics,
                self.max_recursion_depth,
            );
            expander.expand_tokens(&with_defined)
        };
        // Step 3+4: Evaluate the expression.
        match expression::evaluate_pp_expression(&expanded, &mut *self.diagnostics) {
            Ok(value) => value.is_nonzero(),
            Err(()) => false,
        }
    }

    /// Expand an object-like macro, returning the expanded token list.
    #[allow(dead_code)]
    fn expand_object_macro(&self, macro_def: &MacroDef, invocation_span: Span) -> Vec<PPToken> {
        // For predefined magic macros, generate dynamic values.
        if macro_def.is_predefined {
            match macro_def.name.as_str() {
                "__FILE__" => {
                    let filename = self
                        .source_map
                        .get_filename(invocation_span.file_id)
                        .unwrap_or("<unknown>");
                    return vec![PPToken::from_expansion(
                        PPTokenKind::StringLiteral,
                        format!("\"{}\"", filename),
                        invocation_span,
                    )];
                }
                "__LINE__" => {
                    if let Some(file) = self.source_map.get_file(invocation_span.file_id) {
                        let (line, _) = file.lookup_line_col(invocation_span.start);
                        return vec![PPToken::from_expansion(
                            PPTokenKind::Number,
                            line.to_string(),
                            invocation_span,
                        )];
                    }
                }
                _ => {}
            }
        }

        // Normal object-like expansion: clone replacement tokens.
        macro_def
            .replacement
            .iter()
            .map(|t| {
                let mut tok = t.clone();
                tok.from_macro = true;
                tok.span = invocation_span;
                tok
            })
            .collect()
    }

    /// Expand a function-like macro invocation, collecting arguments from the
    /// token stream starting at `paren_pos` (position of the opening `(`).
    ///
    /// Returns the expanded tokens and the position after the closing `)`.
    #[allow(dead_code)]
    fn expand_function_macro(
        &self,
        macro_def: &MacroDef,
        tokens: &[PPToken],
        name_pos: usize,
        paren_pos: usize,
    ) -> (Vec<PPToken>, usize) {
        let invocation_span = tokens[name_pos].span;
        let (params, variadic) = match &macro_def.kind {
            MacroKind::FunctionLike { params, variadic } => (params.clone(), *variadic),
            MacroKind::ObjectLike => {
                return (vec![tokens[name_pos].clone()], name_pos + 1);
            }
        };

        // Collect arguments.
        let mut args: Vec<Vec<PPToken>> = Vec::new();
        let mut current_arg: Vec<PPToken> = Vec::new();
        let mut depth = 1;
        let mut pos = paren_pos + 1; // skip `(`

        while pos < tokens.len() && depth > 0 {
            let tok = &tokens[pos];
            if tok.kind == PPTokenKind::Punctuator {
                match tok.text.as_str() {
                    "(" => {
                        depth += 1;
                        current_arg.push(tok.clone());
                    }
                    ")" => {
                        depth -= 1;
                        if depth == 0 {
                            args.push(current_arg);
                            current_arg = Vec::new();
                        } else {
                            current_arg.push(tok.clone());
                        }
                    }
                    "," if depth == 1 => {
                        args.push(current_arg);
                        current_arg = Vec::new();
                    }
                    _ => {
                        current_arg.push(tok.clone());
                    }
                }
            } else {
                current_arg.push(tok.clone());
            }
            pos += 1;
        }

        // Perform substitution.
        let mut result = Vec::new();
        for rtok in &macro_def.replacement {
            if rtok.kind == PPTokenKind::Identifier {
                // Check if this is a parameter reference.
                if let Some(param_idx) = params.iter().position(|p| p == &rtok.text) {
                    if let Some(arg_tokens) = args.get(param_idx) {
                        result.extend(arg_tokens.iter().cloned());
                    }
                    continue;
                }
                // Check for __VA_ARGS__.
                if variadic && rtok.text == "__VA_ARGS__" {
                    // Variadic arguments start after the last named parameter.
                    let va_start = params.len();
                    for (i, arg) in args.iter().enumerate().skip(va_start) {
                        if i > va_start {
                            result.push(PPToken::new(
                                PPTokenKind::Punctuator,
                                ",",
                                invocation_span,
                            ));
                        }
                        result.extend(arg.iter().cloned());
                    }
                    continue;
                }
            }
            let mut tok = rtok.clone();
            tok.from_macro = true;
            tok.span = invocation_span;
            result.push(tok);
        }

        (result, pos)
    }
}

// ---------------------------------------------------------------------------
// Helper: simplified condition evaluation
// ---------------------------------------------------------------------------

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Phase 1: Trigraphs -----------------------------------------------

    #[test]
    fn trigraph_all_nine_replacements() {
        assert_eq!(phase1_trigraphs("??="), "#");
        assert_eq!(phase1_trigraphs("??/"), "\\");
        assert_eq!(phase1_trigraphs("??'"), "^");
        assert_eq!(phase1_trigraphs("??("), "[");
        assert_eq!(phase1_trigraphs("??)"), "]");
        assert_eq!(phase1_trigraphs("??!"), "|");
        assert_eq!(phase1_trigraphs("??<"), "{");
        assert_eq!(phase1_trigraphs("??>"), "}");
        assert_eq!(phase1_trigraphs("??-"), "~");
    }

    #[test]
    fn trigraph_in_context() {
        assert_eq!(phase1_trigraphs("a??=b"), "a#b");
        assert_eq!(phase1_trigraphs("x??(y??)z"), "x[y]z");
    }

    #[test]
    fn trigraph_no_match() {
        assert_eq!(phase1_trigraphs("??"), "??");
        assert_eq!(phase1_trigraphs("?x"), "?x");
        assert_eq!(phase1_trigraphs("??@"), "??@");
        assert_eq!(phase1_trigraphs("abc"), "abc");
    }

    #[test]
    fn trigraph_empty_and_short() {
        assert_eq!(phase1_trigraphs(""), "");
        assert_eq!(phase1_trigraphs("a"), "a");
        assert_eq!(phase1_trigraphs("ab"), "ab");
    }

    // -- Phase 1: Line Splicing -------------------------------------------

    #[test]
    fn line_splice_unix() {
        assert_eq!(phase1_line_splice("hello\\\nworld"), "helloworld");
    }

    #[test]
    fn line_splice_windows() {
        assert_eq!(phase1_line_splice("hello\\\r\nworld"), "helloworld");
    }

    #[test]
    fn line_splice_multiple() {
        assert_eq!(phase1_line_splice("a\\\nb\\\nc"), "abc");
    }

    #[test]
    fn line_splice_no_splice() {
        assert_eq!(phase1_line_splice("hello\nworld"), "hello\nworld");
    }

    #[test]
    fn line_splice_backslash_not_before_newline() {
        assert_eq!(phase1_line_splice("path\\to\\file"), "path\\to\\file");
    }

    // -- Tokenizer --------------------------------------------------------

    #[test]
    fn tokenize_simple_identifier() {
        let tokens = tokenize_preprocessing("hello", 0);
        assert_eq!(tokens.len(), 2); // identifier + EOF
        assert_eq!(tokens[0].kind, PPTokenKind::Identifier);
        assert_eq!(tokens[0].text, "hello");
    }

    #[test]
    fn tokenize_number() {
        let tokens = tokenize_preprocessing("42", 0);
        assert_eq!(tokens[0].kind, PPTokenKind::Number);
        assert_eq!(tokens[0].text, "42");
    }

    #[test]
    fn tokenize_hex_number() {
        let tokens = tokenize_preprocessing("0xFF", 0);
        assert_eq!(tokens[0].kind, PPTokenKind::Number);
        assert_eq!(tokens[0].text, "0xFF");
    }

    #[test]
    fn tokenize_string_literal() {
        let tokens = tokenize_preprocessing("\"hello\"", 0);
        assert_eq!(tokens[0].kind, PPTokenKind::StringLiteral);
        assert_eq!(tokens[0].text, "\"hello\"");
    }

    #[test]
    fn tokenize_char_literal() {
        let tokens = tokenize_preprocessing("'a'", 0);
        assert_eq!(tokens[0].kind, PPTokenKind::CharLiteral);
        assert_eq!(tokens[0].text, "'a'");
    }

    #[test]
    fn tokenize_punctuators() {
        let tokens = tokenize_preprocessing("+ - -> == !=", 0);
        let puncts: Vec<&str> = tokens
            .iter()
            .filter(|t| t.kind == PPTokenKind::Punctuator)
            .map(|t| t.text.as_str())
            .collect();
        assert_eq!(puncts, vec!["+", "-", "->", "==", "!="]);
    }

    #[test]
    fn tokenize_directive_hash() {
        let tokens = tokenize_preprocessing("#define FOO 42", 0);
        assert_eq!(tokens[0].kind, PPTokenKind::Punctuator);
        assert_eq!(tokens[0].text, "#");
    }

    #[test]
    fn tokenize_comment_replaced_by_space() {
        let tokens = tokenize_preprocessing("a /* comment */ b", 0);
        let non_ws: Vec<&str> = tokens
            .iter()
            .filter(|t| !t.is_whitespace() && !t.is_eof())
            .map(|t| t.text.as_str())
            .collect();
        assert_eq!(non_ws, vec!["a", "b"]);
    }

    #[test]
    fn tokenize_line_comment() {
        let tokens = tokenize_preprocessing("a // comment\nb", 0);
        let non_ws: Vec<&str> = tokens
            .iter()
            .filter(|t| !t.is_whitespace() && t.kind != PPTokenKind::Newline && !t.is_eof())
            .map(|t| t.text.as_str())
            .collect();
        assert_eq!(non_ws, vec!["a", "b"]);
    }

    // -- PPToken constructors ---------------------------------------------

    #[test]
    fn pptoken_new() {
        let tok = PPToken::new(PPTokenKind::Identifier, "foo", Span::new(0, 0, 3));
        assert_eq!(tok.kind, PPTokenKind::Identifier);
        assert_eq!(tok.text, "foo");
        assert!(!tok.from_macro);
        assert!(!tok.painted);
    }

    #[test]
    fn pptoken_eof() {
        let tok = PPToken::eof(Span::dummy());
        assert!(tok.is_eof());
    }

    #[test]
    fn pptoken_placemarker() {
        let tok = PPToken::placemarker(Span::dummy());
        assert_eq!(tok.kind, PPTokenKind::PlacemarkerToken);
        assert!(tok.from_macro);
    }

    // -- ConditionalState -------------------------------------------------

    #[test]
    fn conditional_state_active() {
        let state = ConditionalState::new(true, Span::dummy());
        assert!(state.active);
        assert!(state.seen_active);
        assert!(!state.seen_else);
    }

    #[test]
    fn conditional_state_inactive() {
        let state = ConditionalState::new(false, Span::dummy());
        assert!(!state.active);
        assert!(!state.seen_active);
        assert!(!state.seen_else);
    }

    // -- MacroDef and MacroKind -------------------------------------------

    #[test]
    fn macro_def_object_like() {
        let def = MacroDef {
            name: "FOO".to_string(),
            kind: MacroKind::ObjectLike,
            replacement: vec![PPToken::new(PPTokenKind::Number, "42", Span::dummy())],
            is_predefined: false,
            definition_span: Span::dummy(),
        };
        assert_eq!(def.name, "FOO");
        assert!(matches!(def.kind, MacroKind::ObjectLike));
    }

    #[test]
    fn macro_def_function_like() {
        let def = MacroDef {
            name: "MAX".to_string(),
            kind: MacroKind::FunctionLike {
                params: vec!["a".to_string(), "b".to_string()],
                variadic: false,
            },
            replacement: Vec::new(),
            is_predefined: false,
            definition_span: Span::dummy(),
        };
        assert!(matches!(def.kind, MacroKind::FunctionLike { .. }));
    }

    // Condition evaluation is now tested via the full expression evaluator
    // in the `expression` submodule.  The previous `evaluate_simple_condition`
    // helper was removed in favour of the complete implementation.
}
