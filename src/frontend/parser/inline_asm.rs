//! Inline assembly (`asm`/`__asm__`) parsing — AT&T syntax, constraints, clobbers.
//!
//! This module will be fully implemented by a dedicated agent. These are
//! working implementations sufficient for `mod.rs` compilation.

use super::*;

/// Parse an inline assembly statement.
///
/// Grammar (simplified):
/// ```text
/// asm-statement:
///     asm ( template : outputs : inputs : clobbers ) ;
///     asm volatile ( template : outputs : inputs : clobbers ) ;
///     asm goto ( template : outputs : inputs : clobbers : labels ) ;
/// ```
pub fn parse_asm_statement(parser: &mut Parser<'_>) -> Result<AsmStatement, ()> {
    let start_span = parser.current_span();

    // Consume `asm` or `__asm__` or `asm volatile`.
    let is_volatile = if parser.match_token(&TokenKind::AsmVolatile) {
        true
    } else {
        parser.expect(TokenKind::Asm)?;
        parser.match_token(&TokenKind::Volatile)
    };

    let is_goto = match parser.peek() {
        TokenKind::Goto => {
            parser.advance();
            true
        }
        _ => false,
    };

    parser.expect(TokenKind::LeftParen)?;

    // Parse template string (raw bytes via PUA decoding).
    let template = parse_asm_string(parser)?;

    // Parse optional output operands, input operands, clobbers, and goto labels.
    let mut outputs = Vec::new();
    let mut inputs = Vec::new();
    let mut clobbers = Vec::new();
    let mut goto_labels = Vec::new();

    if parser.match_token(&TokenKind::Colon) {
        // Output operands.
        outputs = parse_asm_operand_list(parser)?;

        if parser.match_token(&TokenKind::Colon) {
            // Input operands.
            inputs = parse_asm_operand_list(parser)?;

            if parser.match_token(&TokenKind::Colon) {
                // Clobbers.
                clobbers = parse_clobber_list(parser)?;

                if is_goto && parser.match_token(&TokenKind::Colon) {
                    // Goto labels.
                    goto_labels = parse_goto_label_list(parser)?;
                }
            }
        }
    }

    parser.expect(TokenKind::RightParen)?;
    parser.expect(TokenKind::Semicolon)?;

    let span = parser.make_span(start_span);

    Ok(AsmStatement {
        is_volatile,
        is_goto,
        template,
        outputs,
        inputs,
        clobbers,
        goto_labels,
        span,
    })
}

/// Parse an assembly template string (concatenated string literals → raw bytes).
fn parse_asm_string(parser: &mut Parser<'_>) -> Result<Vec<u8>, ()> {
    let mut bytes = Vec::new();

    match &parser.current.kind {
        TokenKind::StringLiteral { value, .. } => {
            bytes.extend_from_slice(value.as_bytes());
            parser.advance();
        }
        _ => {
            let span = parser.current_span();
            parser.error(span, "expected assembly template string");
            return Err(());
        }
    }

    // Concatenate adjacent string literals.
    while let TokenKind::StringLiteral { value, .. } = &parser.current.kind {
        bytes.extend_from_slice(value.as_bytes());
        parser.advance();
    }

    Ok(bytes)
}

/// Parse a list of assembly operands.
fn parse_asm_operand_list(parser: &mut Parser<'_>) -> Result<Vec<AsmOperand>, ()> {
    let mut operands = Vec::new();

    if parser.check(&TokenKind::Colon) || parser.check(&TokenKind::RightParen) {
        return Ok(operands);
    }

    loop {
        let operand = parse_asm_operand(parser)?;
        operands.push(operand);

        if !parser.match_token(&TokenKind::Comma) {
            break;
        }
    }

    Ok(operands)
}

/// Parse a single assembly operand.
fn parse_asm_operand(parser: &mut Parser<'_>) -> Result<AsmOperand, ()> {
    let start_span = parser.current_span();

    // Optional symbolic name: `[name]`.
    let symbolic_name = if parser.match_token(&TokenKind::LeftBracket) {
        let name = match parser.current.kind {
            TokenKind::Identifier(sym) => {
                parser.advance();
                Some(sym)
            }
            _ => {
                let span = parser.current_span();
                parser.error(span, "expected operand name");
                return Err(());
            }
        };
        parser.expect(TokenKind::RightBracket)?;
        name
    } else {
        None
    };

    // Constraint string (raw bytes).
    let constraint = match &parser.current.kind {
        TokenKind::StringLiteral { value, .. } => {
            let val = value.as_bytes().to_vec();
            parser.advance();
            val
        }
        _ => {
            let span = parser.current_span();
            parser.error(span, "expected constraint string");
            return Err(());
        }
    };

    // Expression in parentheses.
    parser.expect(TokenKind::LeftParen)?;
    let expression = expressions::parse_expression(parser)?;
    parser.expect(TokenKind::RightParen)?;

    let span = parser.make_span(start_span);

    Ok(AsmOperand {
        symbolic_name,
        constraint,
        expression: Box::new(expression),
        span,
    })
}

/// Parse a clobber list (comma-separated string literals → AsmClobber).
fn parse_clobber_list(parser: &mut Parser<'_>) -> Result<Vec<AsmClobber>, ()> {
    let mut clobbers = Vec::new();

    if parser.check(&TokenKind::Colon) || parser.check(&TokenKind::RightParen) {
        return Ok(clobbers);
    }

    loop {
        match &parser.current.kind {
            TokenKind::StringLiteral { value, .. } => {
                let register = value.as_bytes().to_vec();
                let span = parser.current_span();
                parser.advance();
                clobbers.push(AsmClobber { register, span });
            }
            _ => break,
        }
        if !parser.match_token(&TokenKind::Comma) {
            break;
        }
    }

    Ok(clobbers)
}

/// Parse a list of goto labels (comma-separated identifiers).
fn parse_goto_label_list(parser: &mut Parser<'_>) -> Result<Vec<Symbol>, ()> {
    let mut labels = Vec::new();

    loop {
        match parser.current.kind {
            TokenKind::Identifier(sym) => {
                labels.push(sym);
                parser.advance();
            }
            _ => break,
        }
        if !parser.match_token(&TokenKind::Comma) {
            break;
        }
    }

    Ok(labels)
}
