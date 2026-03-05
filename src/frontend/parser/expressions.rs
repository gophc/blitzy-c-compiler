//! Expression parsing — precedence climbing, operator handling, GCC extensions.
//!
//! This module will be fully implemented by a dedicated agent. These are
//! working implementations sufficient for `mod.rs` compilation and basic
//! expression parsing.

use super::*;
use crate::frontend::lexer::token as token_types;

/// Parse a full expression (comma-separated assignment expressions).
pub fn parse_expression(parser: &mut Parser<'_>) -> Result<Expression, ()> {
    parse_assignment_expression(parser)
}

/// Parse an assignment expression (the basic expression unit in C).
pub fn parse_assignment_expression(parser: &mut Parser<'_>) -> Result<Expression, ()> {
    parse_conditional_expression(parser)
}

/// Parse a constant expression (integer constant expression for #if, array sizes, etc.).
pub fn parse_constant_expression(parser: &mut Parser<'_>) -> Result<Expression, ()> {
    parse_conditional_expression(parser)
}

/// Parse a conditional (ternary) expression.
fn parse_conditional_expression(parser: &mut Parser<'_>) -> Result<Expression, ()> {
    parse_logical_or_expression(parser)
}

/// Parse a logical OR expression.
fn parse_logical_or_expression(parser: &mut Parser<'_>) -> Result<Expression, ()> {
    parse_unary_expression(parser)
}

/// Parse a unary expression (prefix operators, sizeof, etc.).
fn parse_unary_expression(parser: &mut Parser<'_>) -> Result<Expression, ()> {
    parse_primary_expression(parser)
}

/// Convert a token IntegerSuffix to an AST IntegerSuffix.
fn convert_integer_suffix(sfx: token_types::IntegerSuffix) -> IntegerSuffix {
    match sfx {
        token_types::IntegerSuffix::None => IntegerSuffix::None,
        token_types::IntegerSuffix::U => IntegerSuffix::U,
        token_types::IntegerSuffix::L => IntegerSuffix::L,
        token_types::IntegerSuffix::UL => IntegerSuffix::UL,
        token_types::IntegerSuffix::LL => IntegerSuffix::LL,
        token_types::IntegerSuffix::ULL => IntegerSuffix::ULL,
    }
}

/// Convert a token StringPrefix to an AST StringPrefix.
fn convert_string_prefix(pfx: token_types::StringPrefix) -> StringPrefix {
    match pfx {
        token_types::StringPrefix::None => StringPrefix::None,
        token_types::StringPrefix::L => StringPrefix::L,
        token_types::StringPrefix::U8 => StringPrefix::U8,
        token_types::StringPrefix::U16 => StringPrefix::U16,
        token_types::StringPrefix::U32 => StringPrefix::U32,
    }
}

/// Convert a token StringPrefix (used for char literals) to an AST CharPrefix.
fn convert_char_prefix(pfx: token_types::StringPrefix) -> CharPrefix {
    match pfx {
        token_types::StringPrefix::None => CharPrefix::None,
        token_types::StringPrefix::L => CharPrefix::L,
        token_types::StringPrefix::U16 => CharPrefix::U16,
        token_types::StringPrefix::U32 => CharPrefix::U32,
        token_types::StringPrefix::U8 => CharPrefix::None, // u8 not valid for char
    }
}

/// Parse a primary expression (literals, identifiers, parenthesized).
fn parse_primary_expression(parser: &mut Parser<'_>) -> Result<Expression, ()> {
    let span = parser.current_span();

    match &parser.current.kind {
        TokenKind::IntegerLiteral { value, suffix, .. } => {
            let val = *value as u128;
            let sfx = convert_integer_suffix(*suffix);
            parser.advance();
            Ok(Expression::IntegerLiteral {
                value: val,
                suffix: sfx,
                span: parser.previous_span(),
            })
        }
        TokenKind::FloatLiteral { value, suffix, .. } => {
            let val: f64 = value.parse().unwrap_or(0.0);
            let sfx = match suffix {
                token_types::FloatSuffix::None => FloatSuffix::None,
                token_types::FloatSuffix::F => FloatSuffix::F,
                token_types::FloatSuffix::L => FloatSuffix::L,
            };
            parser.advance();
            Ok(Expression::FloatLiteral {
                value: val,
                suffix: sfx,
                span: parser.previous_span(),
            })
        }
        TokenKind::Identifier(sym) => {
            let s = *sym;
            parser.advance();
            Ok(Expression::Identifier {
                name: s,
                span: parser.previous_span(),
            })
        }
        TokenKind::StringLiteral { value, prefix } => {
            let seg = StringSegment {
                value: value.as_bytes().to_vec(),
                span: parser.current_span(),
            };
            let pfx = convert_string_prefix(*prefix);
            parser.advance();
            // Concatenate adjacent string literals.
            let mut segments = vec![seg];
            while let TokenKind::StringLiteral { value, .. } = &parser.current.kind {
                segments.push(StringSegment {
                    value: value.as_bytes().to_vec(),
                    span: parser.current_span(),
                });
                parser.advance();
            }
            Ok(Expression::StringLiteral {
                segments,
                prefix: pfx,
                span: parser.make_span(span),
            })
        }
        TokenKind::CharLiteral { value, prefix } => {
            let val = *value;
            let pfx = convert_char_prefix(*prefix);
            parser.advance();
            Ok(Expression::CharLiteral {
                value: val,
                prefix: pfx,
                span: parser.previous_span(),
            })
        }
        TokenKind::LeftParen => {
            parser.advance();
            let expr = parse_expression(parser)?;
            parser.expect(TokenKind::RightParen)?;
            let end_span = parser.make_span(span);
            Ok(Expression::Parenthesized {
                inner: Box::new(expr),
                span: end_span,
            })
        }
        _ => {
            parser.error(
                span,
                &format!("expected expression, found '{}'", parser.current.kind),
            );
            Err(())
        }
    }
}
