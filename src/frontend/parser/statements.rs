//! Statement parsing — control flow, labels, computed gotos.
//!
//! This module will be fully implemented by a dedicated agent. These are
//! working implementations sufficient for `mod.rs` compilation and basic
//! statement parsing.

use super::*;

/// Parse a statement (any statement type).
pub fn parse_statement(parser: &mut Parser<'_>) -> Result<Statement, ()> {
    let span = parser.current_span();

    match parser.peek() {
        TokenKind::LeftBrace => {
            let compound = parse_compound_statement(parser)?;
            Ok(Statement::Compound(compound))
        }
        TokenKind::Return => {
            parser.advance();
            let expr = if !parser.check(&TokenKind::Semicolon) {
                Some(Box::new(expressions::parse_expression(parser)?))
            } else {
                None
            };
            parser.expect(TokenKind::Semicolon)?;
            let _s = parser.make_span(span);
            Ok(Statement::Return {
                value: expr,
                span: _s,
            })
        }
        TokenKind::Semicolon => {
            parser.advance();
            // Empty statement `;` — represented as Expression(None).
            Ok(Statement::Expression(None))
        }
        _ => {
            // Expression statement.
            let expr = expressions::parse_expression(parser)?;
            parser.expect(TokenKind::Semicolon)?;
            Ok(Statement::Expression(Some(Box::new(expr))))
        }
    }
}

/// Parse a compound statement (block): `{ declaration-or-statement* }`.
pub fn parse_compound_statement(parser: &mut Parser<'_>) -> Result<CompoundStatement, ()> {
    let start_span = parser.current_span();
    parser.enter_recursion()?;

    parser.expect(TokenKind::LeftBrace)?;

    let mut items = Vec::new();

    while !parser.check(&TokenKind::RightBrace) && !parser.current.is_eof() {
        // Try parsing as a declaration first, then as a statement.
        if is_declaration_start(parser) {
            match parse_block_declaration(parser) {
                Ok(decl) => items.push(BlockItem::Declaration(Box::new(decl))),
                Err(()) => parser.synchronize(),
            }
        } else {
            match parse_statement(parser) {
                Ok(stmt) => items.push(BlockItem::Statement(stmt)),
                Err(()) => parser.synchronize(),
            }
        }
    }

    parser.expect(TokenKind::RightBrace)?;
    parser.leave_recursion();

    let span = parser.make_span(start_span);
    Ok(CompoundStatement { items, span })
}

/// Check if the current token can start a declaration.
pub fn is_declaration_start(parser: &Parser<'_>) -> bool {
    match parser.peek() {
        // Storage class specifiers
        TokenKind::Typedef
        | TokenKind::Extern
        | TokenKind::Static
        | TokenKind::Auto
        | TokenKind::Register
        | TokenKind::ThreadLocal
        // Type specifiers
        | TokenKind::Void
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
        // Type qualifiers
        | TokenKind::Const
        | TokenKind::Volatile
        | TokenKind::Restrict
        // Function specifiers
        | TokenKind::Inline
        | TokenKind::Noreturn
        // GCC extensions
        | TokenKind::Extension
        | TokenKind::Attribute
        | TokenKind::Typeof
        | TokenKind::Alignas
        // _Static_assert
        | TokenKind::StaticAssert => true,
        // Typedef names
        TokenKind::Identifier(sym) => parser.is_typedef_name(*sym),
        _ => false,
    }
}

/// Parse a declaration within a compound statement (block).
fn parse_block_declaration(parser: &mut Parser<'_>) -> Result<Declaration, ()> {
    let start_span = parser.current_span();

    // _Static_assert
    if parser.check(&TokenKind::StaticAssert) {
        parser.advance();
        parser.expect(TokenKind::LeftParen)?;
        let _cond = expressions::parse_constant_expression(parser)?;
        if parser.match_token(&TokenKind::Comma) {
            parser.parse_string_literal_bytes();
        }
        parser.expect(TokenKind::RightParen)?;
        parser.expect(TokenKind::Semicolon)?;
        let span = parser.make_span(start_span);
        return Ok(Declaration {
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
            span,
        });
    }

    let specifiers = declarations::parse_declaration_specifiers(parser)?;

    // Bare specifier declaration (e.g., `struct foo;`).
    if parser.check(&TokenKind::Semicolon) {
        parser.advance();
        let span = parser.make_span(start_span);
        return Ok(Declaration {
            specifiers,
            declarators: Vec::new(),
            span,
        });
    }

    let mut init_declarators = Vec::new();
    loop {
        let decl_start = parser.current_span();
        let decl = declarations::parse_declarator(parser)?;
        let init = if parser.match_token(&TokenKind::Equal) {
            Some(parser.parse_initializer()?)
        } else {
            None
        };
        let decl_span = parser.make_span(decl_start);
        init_declarators.push(InitDeclarator {
            declarator: decl,
            initializer: init,
            span: decl_span,
        });

        if !parser.match_token(&TokenKind::Comma) {
            break;
        }
    }

    parser.expect(TokenKind::Semicolon)?;
    let span = parser.make_span(start_span);

    // Track typedef names.
    if matches!(specifiers.storage_class, Some(StorageClass::Typedef)) {
        for init_decl in &init_declarators {
            if let Some(sym) = get_declarator_name(&init_decl.declarator) {
                parser.register_typedef(sym);
            }
        }
    }

    Ok(Declaration {
        specifiers,
        declarators: init_declarators,
        span,
    })
}

/// Extract the name (Symbol) from a declarator, if any.
fn get_declarator_name(decl: &Declarator) -> Option<Symbol> {
    get_direct_declarator_name(&decl.direct)
}

/// Extract the name from a direct declarator.
fn get_direct_declarator_name(dd: &DirectDeclarator) -> Option<Symbol> {
    match dd {
        DirectDeclarator::Identifier(sym, _) => Some(*sym),
        DirectDeclarator::Parenthesized(inner) => get_declarator_name(inner),
        DirectDeclarator::Array { base, .. } => get_direct_declarator_name(base),
        DirectDeclarator::Function { base, .. } => get_direct_declarator_name(base),
    }
}
