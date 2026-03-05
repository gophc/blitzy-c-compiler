//! `__attribute__((...))` parsing for all required GCC attributes.
//!
//! This module will be fully implemented by a dedicated agent. These are
//! working implementations sufficient for `mod.rs` compilation.

use super::*;

/// Parse a `__attribute__((attr-list))` specifier.
///
/// Grammar:
/// ```text
/// attribute-specifier:
///     __attribute__ (( attribute-list ))
///
/// attribute-list:
///     attribute
///     attribute-list , attribute
/// ```
pub fn parse_attribute_specifier(parser: &mut Parser<'_>) -> Result<Attribute, ()> {
    let start_span = parser.current_span();

    // Consume `__attribute__`.
    parser.expect(TokenKind::Attribute)?;

    // Expect `((`
    parser.expect(TokenKind::LeftParen)?;
    parser.expect(TokenKind::LeftParen)?;

    let mut attrs = Vec::new();

    // Parse attribute list until `))`.
    while !parser.check(&TokenKind::RightParen) && !parser.current.is_eof() {
        let attr = parse_single_attribute(parser)?;
        attrs.push(attr);
        if !parser.match_token(&TokenKind::Comma) {
            break;
        }
    }

    // Expect `))`
    parser.expect(TokenKind::RightParen)?;
    parser.expect(TokenKind::RightParen)?;

    let span = parser.make_span(start_span);

    // Return the first attribute (or a placeholder if empty).
    if let Some(attr) = attrs.into_iter().next() {
        Ok(attr)
    } else {
        Ok(Attribute {
            name: parser.intern(""),
            args: Vec::new(),
            span,
        })
    }
}

/// Parse a single attribute within an attribute list.
fn parse_single_attribute(parser: &mut Parser<'_>) -> Result<Attribute, ()> {
    let start_span = parser.current_span();

    // Attribute name (may be a keyword like `const`, `volatile`, etc.).
    let name = match parser.current.kind {
        TokenKind::Identifier(sym) => {
            parser.advance();
            sym
        }
        _ => {
            // Some attributes use keyword names.
            let s = format!("{}", parser.current.kind);
            parser.advance();
            parser.intern(&s)
        }
    };

    // Optional arguments in parentheses.
    let args = if parser.match_token(&TokenKind::LeftParen) {
        let mut arg_list = Vec::new();
        while !parser.check(&TokenKind::RightParen) && !parser.current.is_eof() {
            let arg = parse_attribute_arg(parser)?;
            arg_list.push(arg);
            if !parser.match_token(&TokenKind::Comma) {
                break;
            }
        }
        parser.expect(TokenKind::RightParen)?;
        arg_list
    } else {
        Vec::new()
    };

    let span = parser.make_span(start_span);
    Ok(Attribute { name, args, span })
}

/// Parse a single attribute argument.
fn parse_attribute_arg(parser: &mut Parser<'_>) -> Result<AttributeArg, ()> {
    match &parser.current.kind {
        TokenKind::StringLiteral { value, .. } => {
            let bytes = value.as_bytes().to_vec();
            let span = parser.current_span();
            parser.advance();
            Ok(AttributeArg::String(bytes, span))
        }
        TokenKind::Identifier(sym) => {
            let s = *sym;
            let span = parser.current_span();
            parser.advance();
            // If followed by `(`, it might be a nested attribute call — handle as identifier.
            Ok(AttributeArg::Identifier(s, span))
        }
        _ => {
            let expr = expressions::parse_assignment_expression(parser)?;
            Ok(AttributeArg::Expression(Box::new(expr)))
        }
    }
}
