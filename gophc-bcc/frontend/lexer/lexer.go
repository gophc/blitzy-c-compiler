package lexer

import (
	"fmt"

	"github.com/gophc/blitzy-c-compiler/gophc-bcc/common"
)

type Lexer struct {
	scanner     *Scanner
	interner    *common.Interner
	diagnostics *common.DiagnosticEngine
	fileID      uint32
	lookahead   []Token
}

func NewLexer(source string, fileID uint32, interner *common.Interner, diagnostics *common.DiagnosticEngine) *Lexer {
	return &Lexer{
		scanner:     NewScanner(source),
		interner:    interner,
		diagnostics: diagnostics,
		fileID:      fileID,
		lookahead:   make([]Token, 0),
	}
}

func (l *Lexer) NextToken() Token {
	if len(l.lookahead) > 0 {
		tok := l.lookahead[0]
		l.lookahead = l.lookahead[1:]
		return tok
	}
	return l.lexToken()
}

func (l *Lexer) Peek() *Token {
	if len(l.lookahead) == 0 {
		tok := l.lexToken()
		l.lookahead = append(l.lookahead, tok)
	}
	return &l.lookahead[0]
}

func (l *Lexer) PeekNth(n int) *Token {
	for len(l.lookahead) <= n {
		tok := l.lexToken()
		l.lookahead = append(l.lookahead, tok)
	}
	return &l.lookahead[n]
}

func (l *Lexer) Unget(token Token) {
	l.lookahead = append([]Token{token}, l.lookahead...)
}

func (l *Lexer) TokenizeAll() []Token {
	var tokens []Token
	for {
		tok := l.NextToken()
		tokens = append(tokens, tok)
		if tok.IsEof() {
			break
		}
	}
	return tokens
}

func (l *Lexer) lexToken() Token {
	l.skipWhitespaceAndComments()

	start := l.scanner.Offset()

	if l.scanner.IsEof() {
		return l.makeToken(EofToken(0), start)
	}

	ch := l.scanner.peek()
	if ch == 0 {
		return l.makeToken(EofToken(0), start)
	}

	kind := l.lexChar(ch, start)
	return l.makeToken(kind, start)
}

func (l *Lexer) lexChar(ch rune, start uint32) TokenKind {
	switch {
	case (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch == '_':
		if ch == 'L' || ch == 'u' || ch == 'U' {
			if prefix, isString := detectPrefix(l.scanner); prefix != StringPrefixNone {
				if isString {
					return lexStringLiteral(l.scanner, prefix, l.diagnostics, l.fileID)
				} else {
					return lexCharLiteral(l.scanner, prefix, l.diagnostics, l.fileID)
				}
			}
		}
		return l.lexIdentifierOrKeyword()
	case ch >= '0' && ch <= '9':
		return lexNumber(l.scanner, l.diagnostics, l.fileID)
	case ch == '"':
		return lexStringLiteral(l.scanner, StringPrefixNone, l.diagnostics, l.fileID)
	case ch == '\'':
		return lexCharLiteral(l.scanner, StringPrefixNone, l.diagnostics, l.fileID)
	case ch == '+':
		return l.lexAfterPlus(start)
	case ch == '-':
		return l.lexAfterMinus(start)
	case ch == '*':
		return l.lexAfterStar(start)
	case ch == '/':
		return l.lexAfterSlash(start)
	case ch == '%':
		return l.lexAfterPercent(start)
	case ch == '&':
		return l.lexAfterAmpersand(start)
	case ch == '|':
		return l.lexAfterPipe(start)
	case ch == '^':
		return l.lexAfterCaret(start)
	case ch == '~':
		l.scanner.advance()
		return OperatorToken(TokenTilde)
	case ch == '!':
		return l.lexAfterBang(start)
	case ch == '=':
		return l.lexAfterEqual(start)
	case ch == '<':
		return l.lexAfterLess(start)
	case ch == '>':
		return l.lexAfterGreater(start)
	case ch == '.':
		return l.lexAfterDot(start)
	case ch == ',':
		l.scanner.advance()
		return OperatorToken(TokenComma)
	case ch == ';':
		l.scanner.advance()
		return OperatorToken(TokenSemicolon)
	case ch == ':':
		l.scanner.advance()
		return OperatorToken(TokenColon)
	case ch == '?':
		l.scanner.advance()
		return OperatorToken(TokenQuestion)
	case ch == '(':
		l.scanner.advance()
		return OperatorToken(TokenLeftParen)
	case ch == ')':
		l.scanner.advance()
		return OperatorToken(TokenRightParen)
	case ch == '[':
		l.scanner.advance()
		return OperatorToken(TokenLeftBracket)
	case ch == ']':
		l.scanner.advance()
		return OperatorToken(TokenRightBracket)
	case ch == '{':
		l.scanner.advance()
		return OperatorToken(TokenLeftBrace)
	case ch == '}':
		l.scanner.advance()
		return OperatorToken(TokenRightBrace)
	case ch == '#':
		l.scanner.advance()
		if l.scanner.consumeIf('#') {
			return OperatorToken(TokenHashHash)
		}
		return OperatorToken(TokenHash)
	default:
		return l.lexErrorChar(ch, start)
	}
}

func (l *Lexer) lexAfterPlus(start uint32) TokenKind {
	l.scanner.advance()
	if l.scanner.consumeIf('+') {
		return OperatorToken(TokenPlusPlus)
	} else if l.scanner.consumeIf('=') {
		return OperatorToken(TokenPlusEqual)
	}
	return OperatorToken(TokenPlus)
}

func (l *Lexer) lexAfterMinus(start uint32) TokenKind {
	l.scanner.advance()
	if l.scanner.consumeIf('>') {
		return OperatorToken(TokenArrow)
	} else if l.scanner.consumeIf('-') {
		return OperatorToken(TokenMinusMinus)
	} else if l.scanner.consumeIf('=') {
		return OperatorToken(TokenMinusEqual)
	}
	return OperatorToken(TokenMinus)
}

func (l *Lexer) lexAfterStar(start uint32) TokenKind {
	l.scanner.advance()
	if l.scanner.consumeIf('=') {
		return OperatorToken(TokenStarEqual)
	}
	return OperatorToken(TokenStar)
}

func (l *Lexer) lexAfterSlash(start uint32) TokenKind {
	l.scanner.advance()
	if l.scanner.consumeIf('=') {
		return OperatorToken(TokenSlashEqual)
	}
	return OperatorToken(TokenSlash)
}

func (l *Lexer) lexAfterPercent(start uint32) TokenKind {
	l.scanner.advance()
	if l.scanner.consumeIf('=') {
		return OperatorToken(TokenPercentEqual)
	}
	return OperatorToken(TokenPercent)
}

func (l *Lexer) lexAfterAmpersand(start uint32) TokenKind {
	l.scanner.advance()
	if l.scanner.consumeIf('&') {
		return OperatorToken(TokenAmpAmp)
	} else if l.scanner.consumeIf('=') {
		return OperatorToken(TokenAmpEqual)
	}
	return OperatorToken(TokenAmpersand)
}

func (l *Lexer) lexAfterPipe(start uint32) TokenKind {
	l.scanner.advance()
	if l.scanner.consumeIf('|') {
		return OperatorToken(TokenPipePipe)
	} else if l.scanner.consumeIf('=') {
		return OperatorToken(TokenPipeEqual)
	}
	return OperatorToken(TokenPipe)
}

func (l *Lexer) lexAfterCaret(start uint32) TokenKind {
	l.scanner.advance()
	if l.scanner.consumeIf('=') {
		return OperatorToken(TokenCaretEqual)
	}
	return OperatorToken(TokenCaret)
}

func (l *Lexer) lexAfterBang(start uint32) TokenKind {
	l.scanner.advance()
	if l.scanner.consumeIf('=') {
		return OperatorToken(TokenBangEqual)
	}
	return OperatorToken(TokenBang)
}

func (l *Lexer) lexAfterEqual(start uint32) TokenKind {
	l.scanner.advance()
	if l.scanner.consumeIf('=') {
		return OperatorToken(TokenEqualEqual)
	}
	return OperatorToken(TokenEqual)
}

func (l *Lexer) lexAfterLess(start uint32) TokenKind {
	l.scanner.advance()
	if l.scanner.consumeIf('<') {
		if l.scanner.consumeIf('=') {
			return OperatorToken(TokenLessLessEqual)
		}
		return OperatorToken(TokenLessLess)
	} else if l.scanner.consumeIf('=') {
		return OperatorToken(TokenLessEqual)
	}
	return OperatorToken(TokenLess)
}

func (l *Lexer) lexAfterGreater(start uint32) TokenKind {
	l.scanner.advance()
	if l.scanner.consumeIf('>') {
		if l.scanner.consumeIf('=') {
			return OperatorToken(TokenGreaterGreaterEqual)
		}
		return OperatorToken(TokenGreaterGreater)
	} else if l.scanner.consumeIf('=') {
		return OperatorToken(TokenGreaterEqual)
	}
	return OperatorToken(TokenGreater)
}

func (l *Lexer) lexAfterDot(start uint32) TokenKind {
	if l.scanner.peekNth(1) >= '0' && l.scanner.peekNth(1) <= '9' {
		return l.lexDotFloat(start)
	}

	if l.scanner.peekNth(1) == '.' && l.scanner.peekNth(2) == '.' {
		l.scanner.advance()
		l.scanner.advance()
		l.scanner.advance()
		return OperatorToken(TokenEllipsis)
	}

	l.scanner.advance()
	return OperatorToken(TokenDot)
}

func (l *Lexer) lexDotFloat(start uint32) TokenKind {
	l.scanner.advance()

	for l.scanner.peek() >= '0' && l.scanner.peek() <= '9' {
		l.scanner.advance()
	}

	if l.scanner.peek() == 'e' || l.scanner.peek() == 'E' {
		l.scanner.advance()
		if l.scanner.peek() == '+' || l.scanner.peek() == '-' {
			l.scanner.advance()
		}
		expDigitStart := l.scanner.Offset()
		for l.scanner.peek() >= '0' && l.scanner.peek() <= '9' {
			l.scanner.advance()
		}
		if l.scanner.Offset() == expDigitStart {
			span := common.NewSpan(l.fileID, start, l.scanner.Offset())
			l.diagnostics.EmitError(span, "exponent has no digits in floating constant")
		}
	}

	valueEnd := l.scanner.Offset()
	suffix := parseFloatSuffix(l.scanner)

	value := l.scanner.Slice(int(start), int(valueEnd))

	return FloatLiteralToken{
		Value:  value,
		Suffix: suffix,
		Base:   NumericBaseDecimal,
	}
}

func (l *Lexer) lexErrorChar(ch rune, start uint32) TokenKind {
	l.scanner.advance()
	span := common.NewSpan(l.fileID, start, l.scanner.Offset())

	if common.IsPUAEncoded(ch) {
		if byteVal := common.DecodePUAToByte(ch); byteVal != nil {
			l.diagnostics.EmitError(span, fmt.Sprintf("non-UTF-8 byte 0x%02X outside of string or character literal", *byteVal))
		} else {
			l.diagnostics.EmitError(span, fmt.Sprintf("unexpected character U+%04X", ch))
		}
	} else if ch == 0 {
		l.diagnostics.EmitError(span, "unexpected null character in source")
	} else {
		l.diagnostics.EmitError(span, fmt.Sprintf("unexpected character '%c'", ch))
	}

	return ErrorToken(0)
}

func (l *Lexer) skipWhitespaceAndComments() {
	for {
		ch := l.scanner.peek()
		if ch == 0 {
			return
		}
		if isCWhitespace(ch) {
			l.scanner.advance()
			continue
		}
		if ch == '/' {
			second := l.scanner.peekNth(1)
			if second == '/' {
				l.scanner.advance()
				l.scanner.advance()
				for {
					next := l.scanner.peek()
					if next == 0 || next == '\n' {
						break
					}
					l.scanner.advance()
				}
				continue
			}
			if second == '*' {
				commentStart := l.scanner.Offset()
				l.scanner.advance()
				l.scanner.advance()
				terminated := false
				for {
					advanced := l.scanner.advance()
					if advanced == 0 {
						break
					}
					if advanced == '/' {
						if l.scanner.peek() == '*' {
							l.diagnostics.EmitWarning(common.NewSpan(l.fileID, l.scanner.Offset()-1, l.scanner.Offset()), "'/*' within block comment")
						}
					}
					if advanced == '*' {
						if l.scanner.consumeIf('/') {
							terminated = true
							break
						}
					}
				}
				if !terminated {
					span := common.NewSpan(l.fileID, commentStart, l.scanner.Offset())
					l.diagnostics.EmitError(span, "unterminated block comment")
					return
				}
				continue
			}
			break
		}
		break
	}
}

func isCWhitespace(ch rune) bool {
	return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' || ch == '\x0C' || ch == '\x0B'
}

func (l *Lexer) lexIdentifierOrKeyword() TokenKind {
	start := int(l.scanner.Offset())
	l.scanner.advance()

	l.scanner.skipWhile(func(ch rune) bool {
		return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '_'
	})

	end := int(l.scanner.Offset())
	text := l.scanner.Slice(start, end)

	if kw := LookupKeyword(text); kw != nil {
		return kw
	}

	symbol := l.interner.Intern(text)
	return IdentifierToken{Symbol: symbol}
}

func (l *Lexer) makeToken(kind TokenKind, start uint32) Token {
	return NewToken(kind, common.NewSpan(l.fileID, start, l.scanner.Offset()))
}

func (l *Lexer) CurrentPosition() (uint32, uint32, uint32) {
	pos := l.scanner.Position()
	return pos.Offset, pos.Line, pos.Column
}

func (l *Lexer) Source() string {
	return l.scanner.Source()
}

func (l *Lexer) Diagnostics() *common.DiagnosticEngine {
	return l.diagnostics
}

func (l *Lexer) Interner() *common.Interner {
	return l.interner
}