package lexer

import (
	"fmt"

	"github.com/gophc/blitzy-c-compiler/gophc-bcc/common"
)

func digitValue(ch rune) uint64 {
	switch {
	case ch >= '0' && ch <= '9':
		return uint64(ch - '0')
	case ch >= 'a' && ch <= 'f':
		return uint64(ch - 'a' + 10)
	case ch >= 'A' && ch <= 'F':
		return uint64(ch - 'A' + 10)
	default:
		return 0
	}
}

func parseIntegerValue(digits string, radix uint64) (uint64, bool) {
	var value uint64
	overflowed := false

	for _, ch := range digits {
		d := digitValue(ch)
		if d >= radix {
			continue
		}
		newValue, ok := value*radix+d, true
		if !ok || newValue < value {
			overflowed = true
			value = ^uint64(0)
		} else {
			value = newValue
		}
	}

	return value, overflowed
}

func parseIntegerSuffix(scanner *Scanner, diagnostics *common.DiagnosticEngine, fileID uint32) IntegerSuffix {
	suffixStart := scanner.Offset()

	switch scanner.peek() {
	case 'u', 'U':
		scanner.advance()
		switch scanner.peek() {
		case 'l', 'L':
			scanner.advance()
			l1 := scanner.peek()
			if l1 == scanner.peek() {
				scanner.advance()
				return IntegerSuffixULL
			} else if l1 == 'l' || l1 == 'L' {
				scanner.advance()
				span := common.NewSpan(fileID, suffixStart, scanner.Offset())
				diagnostics.EmitError(span, "invalid integer suffix: mixed-case 'lL' is not allowed; use 'll' or 'LL'")
				return IntegerSuffixULL
			}
			return IntegerSuffixUL
		default:
			return IntegerSuffixU
		}
	case 'l', 'L':
		l1 := scanner.peek()
		scanner.advance()
		switch scanner.peek() {
		case 'l', 'L':
			if l1 == scanner.peek() {
				scanner.advance()
				switch scanner.peek() {
				case 'u', 'U':
					scanner.advance()
					return IntegerSuffixULL
				default:
					return IntegerSuffixLL
				}
			} else {
				scanner.advance()
				hasU := false
				if scanner.peek() == 'u' || scanner.peek() == 'U' {
					scanner.advance()
					hasU = true
				}
				span := common.NewSpan(fileID, suffixStart, scanner.Offset())
				diagnostics.EmitError(span, "invalid integer suffix: mixed-case 'lL' is not allowed; use 'll' or 'LL'")
				if hasU {
					return IntegerSuffixULL
				}
				return IntegerSuffixLL
			}
		case 'u', 'U':
			scanner.advance()
			return IntegerSuffixUL
		default:
			return IntegerSuffixL
		}
	}
	return IntegerSuffixNone
}

func parseFloatSuffix(scanner *Scanner) FloatSuffix {
	switch scanner.peek() {
	case 'f', 'F':
		scanner.advance()
		switch scanner.peek() {
		case 'i', 'j', 'I', 'J':
			scanner.advance()
			return FloatSuffixFI
		default:
			return FloatSuffixF
		}
	case 'l', 'L':
		scanner.advance()
		switch scanner.peek() {
		case 'i', 'j', 'I', 'J':
			scanner.advance()
			return FloatSuffixLI
		default:
			return FloatSuffixL
		}
	case 'i', 'j', 'I', 'J':
		scanner.advance()
		switch scanner.peek() {
		case 'f', 'F':
			scanner.advance()
			return FloatSuffixFI
		case 'l', 'L':
			scanner.advance()
			return FloatSuffixLI
		default:
			return FloatSuffixI
		}
	}
	return FloatSuffixNone
}

func checkTrailingInvalidChars(scanner *Scanner, diagnostics *common.DiagnosticEngine, fileID uint32, literalStart uint32) {
	ch := scanner.peek()
	if ch != 0 && (isAlphaNumeric(ch) || ch == '_') {
		badStart := scanner.Offset()
		for {
			ch = scanner.peek()
			if ch != 0 && (isAlphaNumeric(ch) || ch == '_') {
				scanner.advance()
			} else {
				break
			}
		}
		span := common.NewSpan(fileID, badStart, scanner.Offset())
		diagnostics.EmitError(span, "invalid suffix on numeric literal")
	}
}

func isAlphaNumeric(ch rune) bool {
	return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9')
}

func maybeConvertToImaginary(scanner *Scanner, tok TokenKind) TokenKind {
	switch scanner.peek() {
	case 'i', 'j', 'I', 'J':
		scanner.advance()
		suffix := parseFloatSuffix(scanner)

		valueStr := "0.0"
		if intTok, ok := tok.(IntegerLiteralToken); ok {
			valueStr = intToString(intTok.Value) + ".0"
		}

		return FloatLiteralToken{
			Value:  valueStr,
			Suffix: suffix,
			Base:   NumericBaseDecimal,
		}
	default:
		return tok
	}
}

func intToString(n uint64) string {
	if n == 0 {
		return "0"
	}
	result := ""
	for n > 0 {
		result = string(rune('0'+n%10)) + result
		n /= 10
	}
	return result
}

func lexNumber(scanner *Scanner, diagnostics *common.DiagnosticEngine, fileID uint32) TokenKind {
	startPos := scanner.Position()
	startOffset := startPos.Offset

	if scanner.IsEof() {
		span := common.NewSpan(fileID, startOffset, startOffset)
		diagnostics.EmitError(span, "unexpected end of file in numeric literal")
		return ErrorToken(0)
	}

	firstChar := scanner.peek()
	if firstChar < '0' || firstChar > '9' {
		span := common.NewSpan(fileID, startOffset, startOffset+1)
		diagnostics.EmitError(span, "expected digit in numeric literal")
		return ErrorToken(0)
	}

	if firstChar == '0' {
		secondChar := scanner.peekNth(1)
		scanner.advance()

		switch secondChar {
		case 'x', 'X':
			scanner.advance()
			return lexHexLiteral(scanner, diagnostics, fileID, startOffset)
		case 'b', 'B':
			scanner.advance()
			return lexBinaryLiteral(scanner, diagnostics, fileID, startOffset)
		default:
			return lexAfterLeadingZero(scanner, diagnostics, fileID, startOffset)
		}
	} else {
		return lexDecimalLiteral(scanner, diagnostics, fileID, startOffset)
	}
}

func lexHexLiteral(scanner *Scanner, diagnostics *common.DiagnosticEngine, fileID uint32, startOffset uint32) TokenKind {
	digitsStart := int(scanner.Offset())
	hasIntDigits := false

	for {
		ch := scanner.peek()
		if isHexDigit(ch) {
			scanner.advance()
			hasIntDigits = true
		} else {
			break
		}
	}

	switch scanner.peek() {
	case '.':
		scanner.advance()
		hasFracDigits := false
		for {
			ch := scanner.peek()
			if isHexDigit(ch) {
				scanner.advance()
				hasFracDigits = true
			} else {
				break
			}
		}

		if !hasIntDigits && !hasFracDigits {
			span := common.NewSpan(fileID, startOffset, scanner.Offset())
			diagnostics.EmitError(span, "hexadecimal floating literal requires at least one hex digit")
		}

		return lexHexFloatExponent(scanner, diagnostics, fileID, startOffset)
	case 'p', 'P':
		if !hasIntDigits {
			span := common.NewSpan(fileID, startOffset, scanner.Offset())
			diagnostics.EmitError(span, "hexadecimal floating literal requires at least one hex digit before 'p' exponent")
		}
		return lexHexFloatExponent(scanner, diagnostics, fileID, startOffset)
	}

	if !hasIntDigits {
		span := common.NewSpan(fileID, startOffset, scanner.Offset())
		diagnostics.EmitError(span, "no digits after '0x' in hexadecimal literal")
		return IntegerLiteralToken{
			Value:  0,
			Suffix: IntegerSuffixNone,
			Base:   NumericBaseHexadecimal,
		}
	}

	digitsEnd := int(scanner.Offset())
	digitStr := scanner.Slice(digitsStart, digitsEnd)
	value, overflowed := parseIntegerValue(digitStr, 16)

	if overflowed {
		span := common.NewSpan(fileID, startOffset, uint32(digitsEnd))
		diagnostics.EmitWarning(span, "integer literal is too large for type 'unsigned long long'")
	}

	suffix := parseIntegerSuffix(scanner, diagnostics, fileID)
	intTok := IntegerLiteralToken{
		Value:  value,
		Suffix: suffix,
		Base:   NumericBaseHexadecimal,
	}
	tok := maybeConvertToImaginary(scanner, intTok)
	checkTrailingInvalidChars(scanner, diagnostics, fileID, startOffset)
	return tok
}

func isHexDigit(ch rune) bool {
	return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f') || (ch >= 'A' && ch <= 'F')
}

func lexHexFloatExponent(scanner *Scanner, diagnostics *common.DiagnosticEngine, fileID uint32, startOffset uint32) TokenKind {
	switch scanner.peek() {
	case 'p', 'P':
		scanner.advance()
	default:
		span := common.NewSpan(fileID, startOffset, scanner.Offset())
		diagnostics.EmitError(span, "hexadecimal floating literal requires 'p' or 'P' exponent")
		beforeSuffix := scanner.Offset()
		suffix := parseFloatSuffix(scanner)
		checkTrailingInvalidChars(scanner, diagnostics, fileID, startOffset)
		raw := scanner.Slice(int(startOffset), int(beforeSuffix))
		return FloatLiteralToken{
			Value:  raw,
			Suffix: suffix,
			Base:   NumericBaseHexadecimal,
		}
	}

	scanner.consumeIf('+')
	scanner.consumeIf('-')

	hasExpDigits := false
	for {
		ch := scanner.peek()
		if ch >= '0' && ch <= '9' {
			scanner.advance()
			hasExpDigits = true
		} else {
			break
		}
	}

	if !hasExpDigits {
		span := common.NewSpan(fileID, startOffset, scanner.Offset())
		diagnostics.EmitError(span, "expected decimal digits after exponent in hexadecimal floating literal")
	}

	beforeSuffix := scanner.Offset()
	suffix := parseFloatSuffix(scanner)
	checkTrailingInvalidChars(scanner, diagnostics, fileID, startOffset)
	raw := scanner.Slice(int(startOffset), int(beforeSuffix))

	return FloatLiteralToken{
		Value:  raw,
		Suffix: suffix,
		Base:   NumericBaseHexadecimal,
	}
}

func lexBinaryLiteral(scanner *Scanner, diagnostics *common.DiagnosticEngine, fileID uint32, startOffset uint32) TokenKind {
	digitsStart := int(scanner.Offset())
	hasDigits := false

	for {
		ch := scanner.peek()
		if ch == '0' || ch == '1' {
			scanner.advance()
			hasDigits = true
		} else if ch >= '0' && ch <= '9' {
			badPos := scanner.Offset()
			scanner.advance()
			span := common.NewSpan(fileID, badPos, badPos+1)
			diagnostics.EmitError(span, fmt.Sprintf("invalid digit '%c' in binary literal", ch))
			hasDigits = true
		} else {
			break
		}
	}

	if !hasDigits {
		span := common.NewSpan(fileID, startOffset, scanner.Offset())
		diagnostics.EmitError(span, "no digits after '0b' in binary literal")
		return IntegerLiteralToken{
			Value:  0,
			Suffix: IntegerSuffixNone,
			Base:   NumericBaseBinary,
		}
	}

	digitsEnd := int(scanner.Offset())
	digitStr := scanner.Slice(digitsStart, digitsEnd)
	value, overflowed := parseIntegerValue(digitStr, 2)

	if overflowed {
		span := common.NewSpan(fileID, startOffset, uint32(digitsEnd))
		diagnostics.EmitWarning(span, "integer literal is too large for type 'unsigned long long'")
	}

	suffix := parseIntegerSuffix(scanner, diagnostics, fileID)
	intTok := IntegerLiteralToken{
		Value:  value,
		Suffix: suffix,
		Base:   NumericBaseBinary,
	}
	tok := maybeConvertToImaginary(scanner, intTok)
	checkTrailingInvalidChars(scanner, diagnostics, fileID, startOffset)
	return tok
}

func lexAfterLeadingZero(scanner *Scanner, diagnostics *common.DiagnosticEngine, fileID uint32, startOffset uint32) TokenKind {
	hasNonOctalDigit := false

	for {
		ch := scanner.consumeIfPred(func(c rune) bool { return c >= '0' && c <= '9' })
		if ch == 0 {
			break
		}
		if ch >= '8' {
			hasNonOctalDigit = true
		}
	}

	switch scanner.peek() {
	case '.':
		scanner.advance()
		return lexDecimalFloatAfterDot(scanner, diagnostics, fileID, startOffset)
	case 'e', 'E':
		return lexDecimalExponent(scanner, diagnostics, fileID, startOffset)
	}

	if hasNonOctalDigit {
		span := common.NewSpan(fileID, startOffset, scanner.Offset())
		diagnostics.EmitError(span, "invalid digit in octal literal")
	}

	digitsEnd := int(scanner.Offset())
	allText := scanner.Slice(int(startOffset), digitsEnd)
	octalDigits := allText[1:]

	var value uint64
	var overflowed bool
	if len(octalDigits) == 0 {
		value = 0
		overflowed = false
	} else {
		value, overflowed = parseIntegerValue(octalDigits, 8)
	}

	if overflowed {
		span := common.NewSpan(fileID, startOffset, uint32(digitsEnd))
		diagnostics.EmitWarning(span, "integer literal is too large for type 'unsigned long long'")
	}

	suffix := parseIntegerSuffix(scanner, diagnostics, fileID)
	intTok := IntegerLiteralToken{
		Value:  value,
		Suffix: suffix,
		Base:   NumericBaseOctal,
	}
	tok := maybeConvertToImaginary(scanner, intTok)
	checkTrailingInvalidChars(scanner, diagnostics, fileID, startOffset)
	return tok
}

func lexDecimalLiteral(scanner *Scanner, diagnostics *common.DiagnosticEngine, fileID uint32, startOffset uint32) TokenKind {
	for {
		ch := scanner.peek()
		if ch >= '0' && ch <= '9' {
			scanner.advance()
		} else {
			break
		}
	}

	switch scanner.peek() {
	case '.':
		scanner.advance()
		return lexDecimalFloatAfterDot(scanner, diagnostics, fileID, startOffset)
	case 'e', 'E':
		return lexDecimalExponent(scanner, diagnostics, fileID, startOffset)
	}

	digitsEnd := int(scanner.Offset())
	digitStr := scanner.Slice(int(startOffset), digitsEnd)
	value, overflowed := parseIntegerValue(digitStr, 10)

	if overflowed {
		span := common.NewSpan(fileID, startOffset, uint32(digitsEnd))
		diagnostics.EmitWarning(span, "integer literal is too large for type 'unsigned long long'")
	}

	suffix := parseIntegerSuffix(scanner, diagnostics, fileID)
	intTok := IntegerLiteralToken{
		Value:  value,
		Suffix: suffix,
		Base:   NumericBaseDecimal,
	}
	tok := maybeConvertToImaginary(scanner, intTok)
	checkTrailingInvalidChars(scanner, diagnostics, fileID, startOffset)
	return tok
}

func lexDecimalFloatAfterDot(scanner *Scanner, diagnostics *common.DiagnosticEngine, fileID uint32, startOffset uint32) TokenKind {
	for {
		ch := scanner.peek()
		if ch >= '0' && ch <= '9' {
			scanner.advance()
		} else {
			break
		}
	}

	if scanner.peek() == 'e' || scanner.peek() == 'E' {
		scanner.advance()
		scanner.consumeIf('+')
		scanner.consumeIf('-')

		hasExpDigits := false
		for {
			ch := scanner.peek()
			if ch >= '0' && ch <= '9' {
				scanner.advance()
				hasExpDigits = true
			} else {
				break
			}
		}

		if !hasExpDigits {
			span := common.NewSpan(fileID, startOffset, scanner.Offset())
			diagnostics.EmitError(span, "expected digits after exponent in floating-point literal")
		}
	}

	beforeSuffix := scanner.Offset()
	suffix := parseFloatSuffix(scanner)
	checkTrailingInvalidChars(scanner, diagnostics, fileID, startOffset)
	raw := scanner.Slice(int(startOffset), int(beforeSuffix))

	return FloatLiteralToken{
		Value:  raw,
		Suffix: suffix,
		Base:   NumericBaseDecimal,
	}
}

func lexDecimalExponent(scanner *Scanner, diagnostics *common.DiagnosticEngine, fileID uint32, startOffset uint32) TokenKind {
	scanner.advance()
	scanner.consumeIf('+')
	scanner.consumeIf('-')

	hasExpDigits := false
	for {
		ch := scanner.peek()
		if ch >= '0' && ch <= '9' {
			scanner.advance()
			hasExpDigits = true
		} else {
			break
		}
	}

	if !hasExpDigits {
		span := common.NewSpan(fileID, startOffset, scanner.Offset())
		diagnostics.EmitError(span, "expected digits after exponent in floating-point literal")
	}

	beforeSuffix := scanner.Offset()
	suffix := parseFloatSuffix(scanner)
	checkTrailingInvalidChars(scanner, diagnostics, fileID, startOffset)
	raw := scanner.Slice(int(startOffset), int(beforeSuffix))

	return FloatLiteralToken{
		Value:  raw,
		Suffix: suffix,
		Base:   NumericBaseDecimal,
	}
}
