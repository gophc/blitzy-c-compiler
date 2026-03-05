//! Declaration parsing — variables, functions, typedefs, structs, unions, enums.
//!
//! This module will be fully implemented by a dedicated agent. These are
//! working implementations sufficient for `mod.rs` compilation and basic
//! parsing of simple C declarations.

use super::*;

/// Parse declaration specifiers (storage class, type specifiers, qualifiers,
/// function specifiers, alignment specifier, attributes).
pub fn parse_declaration_specifiers(parser: &mut Parser<'_>) -> Result<DeclarationSpecifiers, ()> {
    let start_span = parser.current_span();

    let mut type_specifiers = Vec::new();
    let mut type_qualifiers = Vec::new();
    let mut storage_class = None;
    let mut function_specifiers = Vec::new();
    let mut attrs = Vec::new();
    let alignment_specifier = None;

    // Parse specifiers in a loop — they can appear in any order.
    loop {
        match parser.peek() {
            // Type specifiers (unit variants — no span payload)
            TokenKind::Void => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Void);
            }
            TokenKind::Char => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Char);
            }
            TokenKind::Short => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Short);
            }
            TokenKind::Int => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Int);
            }
            TokenKind::Long => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Long);
            }
            TokenKind::Float => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Float);
            }
            TokenKind::Double => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Double);
            }
            TokenKind::Signed => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Signed);
            }
            TokenKind::Unsigned => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Unsigned);
            }
            TokenKind::Bool => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Bool);
            }
            TokenKind::Complex => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Complex);
            }
            // Storage class specifiers (unit variants)
            TokenKind::Typedef => {
                parser.advance();
                storage_class = Some(StorageClass::Typedef);
            }
            TokenKind::Extern => {
                parser.advance();
                storage_class = Some(StorageClass::Extern);
            }
            TokenKind::Static => {
                parser.advance();
                storage_class = Some(StorageClass::Static);
            }
            TokenKind::Auto => {
                parser.advance();
                storage_class = Some(StorageClass::Auto);
            }
            TokenKind::Register => {
                parser.advance();
                storage_class = Some(StorageClass::Register);
            }
            TokenKind::ThreadLocal => {
                parser.advance();
                storage_class = Some(StorageClass::ThreadLocal);
            }
            // Type qualifiers (unit variants)
            TokenKind::Const => {
                parser.advance();
                type_qualifiers.push(TypeQualifier::Const);
            }
            TokenKind::Volatile => {
                parser.advance();
                type_qualifiers.push(TypeQualifier::Volatile);
            }
            TokenKind::Restrict => {
                parser.advance();
                type_qualifiers.push(TypeQualifier::Restrict);
            }
            TokenKind::Atomic => {
                parser.advance();
                type_qualifiers.push(TypeQualifier::Atomic);
            }
            // Function specifiers (unit variants)
            TokenKind::Inline => {
                parser.advance();
                function_specifiers.push(FunctionSpecifier::Inline);
            }
            TokenKind::Noreturn => {
                parser.advance();
                function_specifiers.push(FunctionSpecifier::Noreturn);
            }
            // Struct/Union — wrapped in StructOrUnionSpecifier
            TokenKind::Struct | TokenKind::Union => {
                let spec = parse_struct_or_union_specifier(parser)?;
                type_specifiers.push(spec);
            }
            // Enum — wrapped in EnumSpecifier
            TokenKind::Enum => {
                let spec = parse_enum_specifier(parser)?;
                type_specifiers.push(spec);
            }
            // __extension__ — consume and continue
            TokenKind::Extension => {
                parser.advance();
            }
            // __attribute__ — parse and collect
            TokenKind::Attribute => {
                let attr = attributes::parse_attribute_specifier(parser)?;
                attrs.push(attr);
            }
            // Identifier that is a typedef name
            TokenKind::Identifier(sym) => {
                let sym_val = *sym;
                if parser.is_typedef_name(sym_val) && type_specifiers.is_empty() {
                    parser.advance();
                    type_specifiers.push(TypeSpecifier::TypedefName(sym_val));
                } else {
                    break;
                }
            }
            // typeof / __typeof__
            TokenKind::Typeof => {
                let typeof_spec = parse_typeof_specifier(parser)?;
                type_specifiers.push(typeof_spec);
            }
            _ => break,
        }
    }

    let span = parser.make_span(start_span);

    Ok(DeclarationSpecifiers {
        storage_class,
        type_specifiers,
        type_qualifiers,
        function_specifiers,
        alignment_specifier,
        attributes: attrs,
        span,
    })
}

/// Parse a declarator (the part of a declaration that names the entity).
pub fn parse_declarator(parser: &mut Parser<'_>) -> Result<Declarator, ()> {
    let start_span = parser.current_span();

    // Parse optional pointer chain: `*`, `* const`, `* volatile`, etc.
    let pointer = parse_pointer_chain(parser)?;

    // Parse direct declarator.
    let direct = parse_direct_declarator(parser)?;

    // Parse trailing attributes.
    let mut attrs = Vec::new();
    while parser.check(&TokenKind::Attribute) {
        attrs.push(attributes::parse_attribute_specifier(parser)?);
    }

    let span = parser.make_span(start_span);
    Ok(Declarator {
        pointer,
        direct,
        attributes: attrs,
        span,
    })
}

/// Parse a pointer chain: `*`, `**`, `* const *`, etc.
fn parse_pointer_chain(parser: &mut Parser<'_>) -> Result<Option<Pointer>, ()> {
    if !parser.match_token(&TokenKind::Star) {
        return Ok(None);
    }

    let start_span = parser.previous_span();
    let mut qualifiers = Vec::new();

    // Parse qualifiers for this pointer level.
    loop {
        match parser.peek() {
            TokenKind::Const => {
                parser.advance();
                qualifiers.push(TypeQualifier::Const);
            }
            TokenKind::Volatile => {
                parser.advance();
                qualifiers.push(TypeQualifier::Volatile);
            }
            TokenKind::Restrict => {
                parser.advance();
                qualifiers.push(TypeQualifier::Restrict);
            }
            TokenKind::Atomic => {
                parser.advance();
                qualifiers.push(TypeQualifier::Atomic);
            }
            _ => break,
        }
    }

    // Recursively parse inner pointer.
    let inner = parse_pointer_chain(parser)?;
    let span = parser.make_span(start_span);

    Ok(Some(Pointer {
        qualifiers,
        inner: inner.map(Box::new),
        span,
    }))
}

/// Parse a direct declarator: identifier, `(declarator)`, followed by optional
/// array `[...]` or function `(...)` suffixes.
fn parse_direct_declarator(parser: &mut Parser<'_>) -> Result<DirectDeclarator, ()> {
    let start_span = parser.current_span();

    // Base: identifier or parenthesized declarator.
    let mut base = if let Some(sym) = parser.current_identifier() {
        let span = parser.current_span();
        parser.advance();
        DirectDeclarator::Identifier(sym, span)
    } else if parser.check(&TokenKind::LeftParen) {
        // Disambiguate: is this a parenthesized declarator or a function parameter list?
        // Simple heuristic: if the next token after `(` starts a type, it's parameters.
        // Otherwise it's a parenthesized declarator.
        let lookahead = parser.peek_nth(0);
        if lookahead.is(&TokenKind::Star) || lookahead.kind == TokenKind::LeftParen {
            parser.advance(); // consume `(`
            let inner = parse_declarator(parser)?;
            parser.expect(TokenKind::RightParen)?;
            DirectDeclarator::Parenthesized(Box::new(inner))
        } else {
            // Abstract declarator — no name.
            DirectDeclarator::Identifier(parser.intern(""), Span::dummy())
        }
    } else {
        // Abstract declarator — no name. Return a dummy identifier.
        DirectDeclarator::Identifier(parser.intern(""), Span::dummy())
    };

    // Parse suffix: `[...]` (array) or `(...)` (function parameters).
    loop {
        if parser.match_token(&TokenKind::LeftBracket) {
            // Array declarator: `[constant-expression]` or `[]`.
            let size = if parser.check(&TokenKind::RightBracket) {
                None
            } else {
                Some(Box::new(expressions::parse_constant_expression(parser)?))
            };
            parser.expect(TokenKind::RightBracket)?;
            let span = parser.make_span(start_span);
            base = DirectDeclarator::Array {
                base: Box::new(base),
                size,
                qualifiers: Vec::new(),
                is_static: false,
                is_star: false,
                span,
            };
        } else if parser.match_token(&TokenKind::LeftParen) {
            // Function declarator: `(parameter-list)`.
            let params = parse_parameter_list(parser)?;
            let is_variadic = if !params.is_empty() {
                parser.match_token(&TokenKind::Comma) && parser.match_token(&TokenKind::Ellipsis)
            } else {
                parser.match_token(&TokenKind::Ellipsis)
            };
            parser.expect(TokenKind::RightParen)?;
            let span = parser.make_span(start_span);
            base = DirectDeclarator::Function {
                base: Box::new(base),
                params,
                is_variadic,
                span,
            };
        } else {
            break;
        }
    }

    Ok(base)
}

/// Parse a parameter type list for a function declarator.
fn parse_parameter_list(parser: &mut Parser<'_>) -> Result<Vec<ParameterDeclaration>, ()> {
    let mut params = Vec::new();

    // Empty parameter list: `()`.
    if parser.check(&TokenKind::RightParen) {
        return Ok(params);
    }

    // `(void)` — no parameters.
    if parser.check(&TokenKind::Void) {
        let lookahead = parser.peek_nth(0);
        if lookahead.is(&TokenKind::RightParen) {
            parser.advance(); // consume `void`
            return Ok(params);
        }
    }

    loop {
        // Check for `...` (variadic) at the end.
        if parser.check(&TokenKind::Ellipsis) {
            break;
        }

        let param = parse_parameter_declaration(parser)?;
        params.push(param);

        if !parser.match_token(&TokenKind::Comma) {
            break;
        }

        // Stop before `...`.
        if parser.check(&TokenKind::Ellipsis) {
            break;
        }
    }

    Ok(params)
}

/// Parse a single parameter declaration.
fn parse_parameter_declaration(parser: &mut Parser<'_>) -> Result<ParameterDeclaration, ()> {
    let start_span = parser.current_span();
    let specifiers = parse_declaration_specifiers(parser)?;

    // Try to parse a declarator (may be abstract or named).
    let (declarator, abstract_declarator) = if parser.check(&TokenKind::Comma)
        || parser.check(&TokenKind::RightParen)
        || parser.check(&TokenKind::Ellipsis)
    {
        (None, None)
    } else {
        // Try to parse as a named declarator.
        let decl = parse_declarator(parser)?;
        (Some(decl), None)
    };

    let span = parser.make_span(start_span);
    Ok(ParameterDeclaration {
        specifiers,
        declarator,
        abstract_declarator,
        span,
    })
}

/// Parse a struct or union specifier.
fn parse_struct_or_union_specifier(parser: &mut Parser<'_>) -> Result<TypeSpecifier, ()> {
    let start_span = parser.current_span();
    let kind = if parser.check(&TokenKind::Union) {
        parser.advance();
        StructOrUnion::Union
    } else {
        parser.advance(); // consume `struct`
        StructOrUnion::Struct
    };

    // Optional attributes before tag.
    let mut pre_attrs = Vec::new();
    while parser.check(&TokenKind::Attribute) {
        pre_attrs.push(attributes::parse_attribute_specifier(parser)?);
    }

    // Optional tag name.
    let tag = if let Some(sym) = parser.current_identifier() {
        parser.advance();
        Some(sym)
    } else {
        None
    };

    // Optional body `{ member-declaration-list }`.
    let members = if parser.match_token(&TokenKind::LeftBrace) {
        let mut member_list = Vec::new();
        while !parser.check(&TokenKind::RightBrace) && !parser.current.is_eof() {
            match parse_struct_member(parser) {
                Ok(m) => member_list.push(m),
                Err(()) => {
                    parser.synchronize();
                }
            }
        }
        parser.expect(TokenKind::RightBrace)?;
        Some(member_list)
    } else {
        None
    };

    // Optional attributes after body.
    while parser.check(&TokenKind::Attribute) {
        pre_attrs.push(attributes::parse_attribute_specifier(parser)?);
    }

    let span = parser.make_span(start_span);

    let spec = StructOrUnionSpecifier {
        kind,
        tag,
        members,
        attributes: pre_attrs,
        span,
    };

    match kind {
        StructOrUnion::Struct => Ok(TypeSpecifier::Struct(spec)),
        StructOrUnion::Union => Ok(TypeSpecifier::Union(spec)),
    }
}

/// Parse a single struct/union member declaration.
fn parse_struct_member(parser: &mut Parser<'_>) -> Result<StructMember, ()> {
    let start_span = parser.current_span();

    // Handle _Static_assert in structs (C11).
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
        return Ok(StructMember {
            specifiers: SpecifierQualifierList {
                type_specifiers: Vec::new(),
                type_qualifiers: Vec::new(),
                attributes: Vec::new(),
                span,
            },
            declarators: Vec::new(),
            attributes: Vec::new(),
            span,
        });
    }

    let specifiers = parse_specifier_qualifier_list(parser)?;
    let mut declarators = Vec::new();

    // Parse struct declarators (may have bitfield width).
    if !parser.check(&TokenKind::Semicolon) {
        loop {
            let decl_start = parser.current_span();
            let declarator = if parser.check(&TokenKind::Colon) {
                None
            } else {
                Some(parse_declarator(parser)?)
            };

            let bit_width = if parser.match_token(&TokenKind::Colon) {
                Some(Box::new(expressions::parse_constant_expression(parser)?))
            } else {
                None
            };

            let decl_span = parser.make_span(decl_start);
            declarators.push(StructDeclarator {
                declarator,
                bit_width,
                span: decl_span,
            });

            if !parser.match_token(&TokenKind::Comma) {
                break;
            }
        }
    }

    // Optional attributes after declarators.
    let mut member_attrs = Vec::new();
    while parser.check(&TokenKind::Attribute) {
        member_attrs.push(attributes::parse_attribute_specifier(parser)?);
    }

    parser.expect(TokenKind::Semicolon)?;
    let span = parser.make_span(start_span);

    Ok(StructMember {
        specifiers,
        declarators,
        attributes: member_attrs,
        span,
    })
}

/// Parse a specifier-qualifier-list (for struct members, type names).
fn parse_specifier_qualifier_list(parser: &mut Parser<'_>) -> Result<SpecifierQualifierList, ()> {
    let start_span = parser.current_span();
    let mut type_specifiers = Vec::new();
    let mut type_qualifiers = Vec::new();
    let mut attrs = Vec::new();

    loop {
        match parser.peek() {
            TokenKind::Void => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Void);
            }
            TokenKind::Char => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Char);
            }
            TokenKind::Short => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Short);
            }
            TokenKind::Int => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Int);
            }
            TokenKind::Long => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Long);
            }
            TokenKind::Float => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Float);
            }
            TokenKind::Double => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Double);
            }
            TokenKind::Signed => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Signed);
            }
            TokenKind::Unsigned => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Unsigned);
            }
            TokenKind::Bool => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Bool);
            }
            TokenKind::Complex => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Complex);
            }
            TokenKind::Const => {
                parser.advance();
                type_qualifiers.push(TypeQualifier::Const);
            }
            TokenKind::Volatile => {
                parser.advance();
                type_qualifiers.push(TypeQualifier::Volatile);
            }
            TokenKind::Restrict => {
                parser.advance();
                type_qualifiers.push(TypeQualifier::Restrict);
            }
            TokenKind::Atomic => {
                parser.advance();
                type_qualifiers.push(TypeQualifier::Atomic);
            }
            TokenKind::Struct | TokenKind::Union => {
                let spec = parse_struct_or_union_specifier(parser)?;
                type_specifiers.push(spec);
            }
            TokenKind::Enum => {
                let spec = parse_enum_specifier(parser)?;
                type_specifiers.push(spec);
            }
            TokenKind::Attribute => {
                attrs.push(attributes::parse_attribute_specifier(parser)?);
            }
            TokenKind::Extension => {
                parser.advance();
            }
            TokenKind::Typeof => {
                let spec = parse_typeof_specifier(parser)?;
                type_specifiers.push(spec);
            }
            TokenKind::Identifier(sym) => {
                let sym_val = *sym;
                if parser.is_typedef_name(sym_val) && type_specifiers.is_empty() {
                    parser.advance();
                    type_specifiers.push(TypeSpecifier::TypedefName(sym_val));
                } else {
                    break;
                }
            }
            _ => break,
        }
    }

    let span = parser.make_span(start_span);
    Ok(SpecifierQualifierList {
        type_specifiers,
        type_qualifiers,
        attributes: attrs,
        span,
    })
}

/// Parse an enum specifier.
fn parse_enum_specifier(parser: &mut Parser<'_>) -> Result<TypeSpecifier, ()> {
    let start_span = parser.current_span();
    parser.advance(); // consume `enum`

    // Optional attributes.
    let mut attrs = Vec::new();
    while parser.check(&TokenKind::Attribute) {
        attrs.push(attributes::parse_attribute_specifier(parser)?);
    }

    // Optional tag name.
    let tag = if let Some(sym) = parser.current_identifier() {
        parser.advance();
        Some(sym)
    } else {
        None
    };

    // Optional enumerator list `{ ... }`.
    let enumerators = if parser.match_token(&TokenKind::LeftBrace) {
        let mut list = Vec::new();
        while !parser.check(&TokenKind::RightBrace) && !parser.current.is_eof() {
            let enum_start = parser.current_span();
            match parser.current.kind {
                TokenKind::Identifier(sym) => {
                    parser.advance();
                    let value = if parser.match_token(&TokenKind::Equal) {
                        Some(Box::new(expressions::parse_constant_expression(parser)?))
                    } else {
                        None
                    };
                    let espan = parser.make_span(enum_start);
                    list.push(Enumerator {
                        name: sym,
                        value,
                        span: espan,
                    });
                }
                _ => {
                    let span = parser.current_span();
                    parser.error(span, "expected enumerator name");
                    parser.synchronize();
                    break;
                }
            }
            if !parser.match_token(&TokenKind::Comma) {
                break;
            }
        }
        parser.expect(TokenKind::RightBrace)?;
        Some(list)
    } else {
        None
    };

    let span = parser.make_span(start_span);
    Ok(TypeSpecifier::Enum(EnumSpecifier {
        tag,
        enumerators,
        attributes: attrs,
        span,
    }))
}

/// Parse a typeof specifier.
fn parse_typeof_specifier(parser: &mut Parser<'_>) -> Result<TypeSpecifier, ()> {
    parser.advance(); // consume `typeof` / `__typeof__`
    parser.expect(TokenKind::LeftParen)?;

    // Try to parse as expression.
    let expr = expressions::parse_expression(parser)?;
    parser.expect(TokenKind::RightParen)?;

    Ok(TypeSpecifier::Typeof(TypeofArg::Expression(Box::new(expr))))
}

/// Parse a function definition given already-parsed specifiers and declarator.
pub fn parse_function_definition(
    parser: &mut Parser<'_>,
    specifiers: DeclarationSpecifiers,
    declarator: Declarator,
) -> Result<FunctionDefinition, ()> {
    let start_span = specifiers.span;

    // Parse the function body (compound statement).
    let body = statements::parse_compound_statement(parser)?;

    let span = parser.make_span(start_span);
    Ok(FunctionDefinition {
        specifiers,
        declarator,
        old_style_params: Vec::new(),
        body,
        attributes: Vec::new(),
        span,
    })
}
