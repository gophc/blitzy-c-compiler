//! Predefined macros module — stub for compilation.
//!
//! This stub provides the `register_predefined_macros` function signature
//! required by `mod.rs`. The full implementation is provided by another agent.

use super::{MacroDef, MacroKind, PPToken, PPTokenKind};
use crate::common::diagnostics::Span;
use crate::common::fx_hash::FxHashMap;
use crate::common::target::Target;

/// Register all compiler-predefined macros for the given target architecture.
///
/// Populates the macro definition table with standard C predefined macros
/// (`__STDC__`, `__STDC_VERSION__`, `__FILE__`, `__LINE__`, etc.) and
/// architecture-specific defines (`__x86_64__`, `__aarch64__`, etc.).
pub fn register_predefined_macros(macro_defs: &mut FxHashMap<String, MacroDef>, target: &Target) {
    // Register standard C macros
    let predefined = [
        ("__STDC__", "1"),
        ("__STDC_VERSION__", "201112L"),
        ("__STDC_HOSTED__", "1"),
    ];

    for (name, value) in &predefined {
        let token = PPToken {
            kind: PPTokenKind::Number,
            text: value.to_string(),
            span: Span::dummy(),
            from_macro: true,
            painted: false,
        };
        macro_defs.insert(
            name.to_string(),
            MacroDef {
                name: name.to_string(),
                kind: MacroKind::ObjectLike,
                replacement: vec![token],
                is_predefined: true,
                definition_span: Span::dummy(),
            },
        );
    }

    // Register architecture-specific macros from the target
    for (name, value) in target.predefined_macros() {
        let token = PPToken {
            kind: PPTokenKind::Number,
            text: value.to_string(),
            span: Span::dummy(),
            from_macro: true,
            painted: false,
        };
        macro_defs.insert(
            name.to_string(),
            MacroDef {
                name: name.to_string(),
                kind: MacroKind::ObjectLike,
                replacement: vec![token],
                is_predefined: true,
                definition_span: Span::dummy(),
            },
        );
    }

    // Register __linux__ and __unix__ (always set for BCC)
    let platform_macros = [("__linux__", "1"), ("__unix__", "1"), ("__ELF__", "1")];
    for (name, value) in &platform_macros {
        let token = PPToken {
            kind: PPTokenKind::Number,
            text: value.to_string(),
            span: Span::dummy(),
            from_macro: true,
            painted: false,
        };
        macro_defs.insert(
            name.to_string(),
            MacroDef {
                name: name.to_string(),
                kind: MacroKind::ObjectLike,
                replacement: vec![token],
                is_predefined: true,
                definition_span: Span::dummy(),
            },
        );
    }
}
