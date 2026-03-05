//! Type specifier and qualifier parsing.
//!
//! This module will be fully implemented by a dedicated agent. These are
//! working implementations sufficient for `mod.rs` compilation.

use super::*;

/// Parse type specifiers in a specifier-qualifier list.
pub fn parse_type_specifiers(_parser: &mut Parser<'_>) -> Result<Vec<TypeSpecifier>, ()> {
    Ok(Vec::new())
}

/// Parse type qualifiers.
pub fn parse_type_qualifiers(_parser: &mut Parser<'_>) -> Result<Vec<TypeQualifier>, ()> {
    Ok(Vec::new())
}

/// Parse a specifier-qualifier-list (for struct members, type names).
pub fn parse_specifier_qualifier_list(
    parser: &mut Parser<'_>,
) -> Result<DeclarationSpecifiers, ()> {
    declarations::parse_declaration_specifiers(parser)
}

/// Check if the current token can start a type specifier.
pub fn is_type_specifier_start(parser: &Parser<'_>) -> bool {
    match parser.peek() {
        TokenKind::Void
        | TokenKind::Char
        | TokenKind::Short
        | TokenKind::Int
        | TokenKind::Long
        | TokenKind::Float
        | TokenKind::Double
        | TokenKind::Signed
        | TokenKind::Unsigned
        | TokenKind::Bool
        | TokenKind::Complex
        | TokenKind::Atomic
        | TokenKind::Struct
        | TokenKind::Union
        | TokenKind::Enum
        | TokenKind::Typeof => true,
        TokenKind::Identifier(sym) => parser.is_typedef_name(*sym),
        _ => false,
    }
}
