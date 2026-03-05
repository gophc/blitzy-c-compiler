//! GCC extension parsing — statement expressions, typeof, computed gotos, etc.
//!
//! This module will be fully implemented by a dedicated agent. These are
//! working implementations sufficient for `mod.rs` compilation.

use super::*;

/// Check if the current token starts a GCC extension construct.
pub fn is_gcc_extension_start(parser: &Parser<'_>) -> bool {
    matches!(
        parser.peek(),
        TokenKind::Extension | TokenKind::Typeof | TokenKind::Attribute
    )
}

/// Parse a GCC extension expression.
pub fn parse_extension_expression(parser: &mut Parser<'_>) -> Result<Expression, ()> {
    // `__extension__` — consume and parse the following expression.
    if parser.match_token(&TokenKind::Extension) {
        return expressions::parse_assignment_expression(parser);
    }
    let span = parser.current_span();
    parser.error(span, "expected GCC extension expression");
    Err(())
}

/// Parse a GCC extension statement.
pub fn parse_extension_statement(parser: &mut Parser<'_>) -> Result<Statement, ()> {
    // `__extension__` — consume and parse the following statement.
    if parser.match_token(&TokenKind::Extension) {
        return statements::parse_statement(parser);
    }
    let span = parser.current_span();
    parser.error(span, "expected GCC extension statement");
    Err(())
}

/// Parse a GCC extension block (may appear in various contexts).
pub fn parse_extension_block(parser: &mut Parser<'_>) -> Result<Expression, ()> {
    parse_extension_expression(parser)
}

/// Parse a GCC statement expression: `({ ... })`.
pub fn parse_statement_expression(parser: &mut Parser<'_>) -> Result<Expression, ()> {
    let start_span = parser.current_span();
    parser.expect(TokenKind::LeftParen)?;
    let body = statements::parse_compound_statement(parser)?;
    parser.expect(TokenKind::RightParen)?;
    let span = parser.make_span(start_span);
    Ok(Expression::StatementExpression {
        compound: body,
        span,
    })
}

/// Parse a label address expression: `&&label`.
pub fn parse_label_address(parser: &mut Parser<'_>) -> Result<Expression, ()> {
    let start_span = parser.current_span();
    parser.advance(); // consume `&&`
    match parser.current.kind {
        TokenKind::Identifier(sym) => {
            parser.advance();
            let span = parser.make_span(start_span);
            Ok(Expression::AddressOfLabel { label: sym, span })
        }
        _ => {
            let span = parser.current_span();
            parser.error(span, "expected label name after '&&'");
            Err(())
        }
    }
}
