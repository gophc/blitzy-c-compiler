package lexer

import (
	"fmt"

	"github.com/gophc/blitzy-c-compiler/gophc-bcc/common"
)

func escapeByteToChar(value byte) rune {
	if value < 0x80 {
		return rune(value)
	}
	return common.EncodeByteToPUA(value)
}

func processEscapeSequence(scanner *Scanner, diagnostics *common.DiagnosticEngine, fileID uint32) rune {
	escStart := scanner.Offset()
	ch := scanner.advance()
	if ch == 0 {
		span := common.NewSpan(fileID, escStart, escStart)
		diagnostics.EmitError(span, "unexpected end of file in escape sequence")
		return 0
	}

	switch ch {
	case 'n':
		return '\n'
	case 't':
		return '\t'
	case 'r':
		return '\r'
	case 'a':
		return '\x07'
	case 'b':
		return '\x08'
	case 'f':
		return '\x0C'
	case 'v':
		return '\x0B'
	case '\\':
		return '\\'
	case '"':
		return '"'
	case '\'':
		return '\''
	case '?':
		return '?'
	case '0', '1', '2', '3', '4', '5', '6', '7':
		value := uint32(ch - '0')
		for i := 0; i < 2; i++ {
			next := scanner.peek()
			if next >= '0' && next <= '7' {
				scanner.advance()
				value = value*8 + uint32(next-'0')
			} else {
				break
			}
		}
		if value > 255 {
			span := common.NewSpan(fileID, escStart-1, scanner.Offset())
			diagnostics.EmitWarning(span, "octal escape sequence out of range")
			value &= 0xFF
		}
		return escapeByteToChar(byte(value))
	case 'x':
		hexStart := scanner.Offset()
		var value uint32
		var count uint32
		for {
			d := scanner.peek()
			var digit uint32
			switch {
			case d >= '0' && d <= '9':
				digit = uint32(d - '0')
			case d >= 'a' && d <= 'f':
				digit = uint32(d - 'a' + 10)
			case d >= 'A' && d <= 'F':
				digit = uint32(d - 'A' + 10)
			default:
				goto hexDone
			}
			scanner.advance()
			value = value*16 + digit
			count++
		}
	hexDone:
		if count == 0 {
			span := common.NewSpan(fileID, escStart-1, hexStart)
			diagnostics.EmitError(span, "\\x used with no following hex digits")
			return 0
		}
		if value > 0xFF {
			span := common.NewSpan(fileID, escStart-1, scanner.Offset())
			diagnostics.EmitWarning(span, fmt.Sprintf("hex escape sequence \\x%X out of range for character", value))
			value &= 0xFF
		}
		return escapeByteToChar(byte(value))
	case 'u':
		return processUnicodeEscape(scanner, diagnostics, fileID, 4, escStart)
	case 'U':
		return processUnicodeEscape(scanner, diagnostics, fileID, 8, escStart)
	default:
		span := common.NewSpan(fileID, escStart-1, scanner.Offset())
		diagnostics.EmitWarning(span, fmt.Sprintf("unknown escape sequence '\\%c'", ch))
		return ch
	}
}

func processUnicodeEscape(scanner *Scanner, diagnostics *common.DiagnosticEngine, fileID uint32, numDigits int, escStart uint32) rune {
	var value uint32
	var count int

	for i := 0; i < numDigits; i++ {
		d := scanner.peek()
		var digit uint32
		switch {
		case d >= '0' && d <= '9':
			digit = uint32(d - '0')
		case d >= 'a' && d <= 'f':
			digit = uint32(d - 'a' + 10)
		case d >= 'A' && d <= 'F':
			digit = uint32(d - 'A' + 10)
		default:
			goto unicodeDone
		}
		scanner.advance()
		value = value*16 + digit
		count++
	}
unicodeDone:

	if count != numDigits {
		prefixChar := 'u'
		if numDigits == 8 {
			prefixChar = 'U'
		}
		span := common.NewSpan(fileID, escStart-1, scanner.Offset())
		diagnostics.EmitError(span, fmt.Sprintf("\\%c escape requires exactly %d hex digits, found %d", prefixChar, numDigits, count))
		return 0
	}

	if value >= 0xD800 && value <= 0xDFFF {
		span := common.NewSpan(fileID, escStart-1, scanner.Offset())
		prefixChar := 'u'
		if numDigits == 8 {
			prefixChar = 'U'
		}
		diagnostics.EmitError(span, fmt.Sprintf("\\%c escape sequence value U+%04X is a surrogate code point", prefixChar, value))
		return 0
	}

	return rune(value)
}

func lexStringLiteral(scanner *Scanner, prefix StringPrefix, diagnostics *common.DiagnosticEngine, fileID uint32) TokenKind {
	startOffset := scanner.Offset()

	if scanner.peek() != '"' {
		return nil
	}
	scanner.advance()

	var value []rune

	for {
		ch := scanner.peek()
		switch ch {
		case 0:
			span := common.NewSpan(fileID, startOffset, scanner.Offset())
			diagnostics.EmitError(span, "unterminated string literal")
			goto stringDone
		case '\n':
			span := common.NewSpan(fileID, startOffset, scanner.Offset())
			diagnostics.EmitError(span, "missing terminating '\"' character")
			goto stringDone
		case '"':
			scanner.advance()
			goto stringDone
		case '\\':
			scanner.advance()
			escCh := processEscapeSequence(scanner, diagnostics, fileID)
			if escCh != 0 {
				value = append(value, escCh)
			}
		default:
			scanner.advance()
			value = append(value, ch)
		}
	}
stringDone:

	return StringLiteralToken{
		Value:  string(value),
		Prefix: prefix,
	}
}

func lexCharLiteral(scanner *Scanner, prefix StringPrefix, diagnostics *common.DiagnosticEngine, fileID uint32) TokenKind {
	startOffset := scanner.Offset()

	if scanner.peek() != '\'' {
		return nil
	}
	scanner.advance()

	var chars []uint32

	for {
		ch := scanner.peek()
		switch ch {
		case 0:
			span := common.NewSpan(fileID, startOffset, scanner.Offset())
			diagnostics.EmitError(span, "unterminated character constant")
			goto charDone
		case '\n':
			span := common.NewSpan(fileID, startOffset, scanner.Offset())
			diagnostics.EmitError(span, "missing terminating \"'\" character")
			goto charDone
		case '\'':
			scanner.advance()
			goto charDone
		case '\\':
			scanner.advance()
			escCh := processEscapeSequence(scanner, diagnostics, fileID)
			if escCh != 0 {
				chars = append(chars, uint32(escCh))
			}
		default:
			scanner.advance()
			if common.IsPUAEncoded(ch) {
				if byteVal := common.DecodePUAToByte(ch); byteVal != nil {
					chars = append(chars, uint32(*byteVal))
				} else {
					chars = append(chars, uint32(ch))
				}
			} else {
				chars = append(chars, uint32(ch))
			}
		}
	}
charDone:

	endOffset := scanner.Offset()
	span := common.NewSpan(fileID, startOffset, endOffset)

	if len(chars) == 0 {
		diagnostics.EmitError(span, "empty character constant")
		return CharLiteralToken{Value: 0, Prefix: prefix}
	}

	if len(chars) > 1 {
		var charStr string
		for _, v := range chars {
			if c := rune(v); c >= 0x20 && c < 0x7F {
				charStr += string(c)
			} else {
				charStr += fmt.Sprintf("\\x%02X", v)
			}
		}
		diagnostics.EmitWarning(span, fmt.Sprintf("multi-character character constant '%s'", charStr))
	}

	var value uint32
	for _, v := range chars {
		value = (value << 8) | (v & 0xFF)
	}

	return CharLiteralToken{
		Value:  value,
		Prefix: prefix,
	}
}

func detectPrefix(scanner *Scanner) (StringPrefix, bool) {
	first := scanner.peek()
	if first == 0 {
		return StringPrefixNone, false
	}

	switch first {
	case 'L':
		switch scanner.peekNth(1) {
		case '"':
			scanner.advance()
			return StringPrefixL, true
		case '\'':
			scanner.advance()
			return StringPrefixL, false
		}
	case 'u':
		switch scanner.peekNth(1) {
		case '8':
			if scanner.peekNth(2) == '"' {
				scanner.advance()
				scanner.advance()
				return StringPrefixU8, true
			}
		case '"':
			scanner.advance()
			return StringPrefixU16, true
		case '\'':
			scanner.advance()
			return StringPrefixU16, false
		}
	case 'U':
		switch scanner.peekNth(1) {
		case '"':
			scanner.advance()
			return StringPrefixU32, true
		case '\'':
			scanner.advance()
			return StringPrefixU32, false
		}
	}
	return StringPrefixNone, false
}

func hasAdjacentString(scanner *Scanner) bool {
	n := 0
	for {
		ch := scanner.peekNth(n)
		if ch == ' ' || ch == '\t' || ch == '\n' {
			n++
		} else {
			break
		}
	}

	switch scanner.peekNth(n) {
	case '"':
		return true
	case 'L':
		return scanner.peekNth(n+1) == '"'
	case 'u':
		switch scanner.peekNth(n + 1) {
		case '"':
			return true
		case '8':
			return scanner.peekNth(n+2) == '"'
		}
	case 'U':
		return scanner.peekNth(n+1) == '"'
	}
	return false
}

func skipWhitespace(scanner *Scanner) {
	for {
		ch := scanner.peek()
		if ch == ' ' || ch == '\t' || ch == '\n' {
			scanner.advance()
		} else {
			break
		}
	}
}

func consumeAdjacentPrefix(scanner *Scanner) StringPrefix {
	skipWhitespace(scanner)
	switch scanner.peek() {
	case '"':
		return StringPrefixNone
	case 'L':
		if scanner.peekNth(1) == '"' {
			scanner.advance()
			return StringPrefixL
		}
	case 'u':
		switch scanner.peekNth(1) {
		case '8':
			if scanner.peekNth(2) == '"' {
				scanner.advance()
				scanner.advance()
				return StringPrefixU8
			}
		case '"':
			scanner.advance()
			return StringPrefixU16
		}
	case 'U':
		if scanner.peekNth(1) == '"' {
			scanner.advance()
			return StringPrefixU32
		}
	}
	return StringPrefixNone
}

func mergePrefix(a StringPrefix, b StringPrefix, diagnostics *common.DiagnosticEngine, fileID uint32, span common.Span) StringPrefix {
	switch {
	case a == StringPrefixNone && b == StringPrefixNone:
		return StringPrefixNone
	case a == StringPrefixNone:
		return b
	case b == StringPrefixNone:
		return a
	case a == b:
		return a
	default:
		diagnostics.EmitError(span, fmt.Sprintf("unsupported non-standard concatenation of string literals with incompatible encoding prefixes (%v and %v)", a, b))
		return a
	}
}

func lexStringWithConcatenation(scanner *Scanner, prefix StringPrefix, diagnostics *common.DiagnosticEngine, fileID uint32) TokenKind {
	first := lexStringLiteral(scanner, prefix, diagnostics, fileID)
	firstLiteral, ok := first.(StringLiteralToken)
	if !ok {
		return first
	}

	combinedValue := firstLiteral.Value
	combinedPrefix := firstLiteral.Prefix

	for hasAdjacentString(scanner) {
		concatStart := scanner.Offset()

		nextPrefix := consumeAdjacentPrefix(scanner)

		span := common.NewSpan(fileID, concatStart, scanner.Offset())
		combinedPrefix = mergePrefix(combinedPrefix, nextPrefix, diagnostics, fileID, span)

		next := lexStringLiteral(scanner, nextPrefix, diagnostics, fileID)
		nextLiteral, ok := next.(StringLiteralToken)
		if !ok {
			break
		}
		combinedValue += nextLiteral.Value
	}

	return StringLiteralToken{
		Value:  combinedValue,
		Prefix: combinedPrefix,
	}
}