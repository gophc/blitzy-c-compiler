package lexer

import (
	"github.com/gophc/blitzy-c-compiler/gophc-bcc/common"
)

type NumericBase int

const (
	NumericBaseDecimal NumericBase = iota
	NumericBaseHexadecimal
	NumericBaseOctal
	NumericBaseBinary
)

type IntegerSuffix int

const (
	IntegerSuffixNone IntegerSuffix = iota
	IntegerSuffixU
	IntegerSuffixL
	IntegerSuffixUL
	IntegerSuffixLL
	IntegerSuffixULL
)

type FloatSuffix int

const (
	FloatSuffixNone FloatSuffix = iota
	FloatSuffixF
	FloatSuffixL
	FloatSuffixI
	FloatSuffixFI
	FloatSuffixLI
)

type StringPrefix int

const (
	StringPrefixNone StringPrefix = iota
	StringPrefixL
	StringPrefixU8
	StringPrefixU16
	StringPrefixU32
)

type TokenKind interface{}

type Token struct {
	Kind TokenKind
	Span common.Span
}

func NewToken(kind TokenKind, span common.Span) Token {
	return Token{
		Kind: kind,
		Span: span,
	}
}

func (t *Token) Is(kind TokenKind) bool {
	return isSameTokenKind(t.Kind, kind)
}

func (t *Token) IsEof() bool {
	_, ok := t.Kind.(EofToken)
	return ok
}

func isSameTokenKind(a, b TokenKind) bool {
	switch a.(type) {
	case KeywordToken:
		_, ok := b.(KeywordToken)
		return ok
	case IdentifierToken:
		_, ok := b.(IdentifierToken)
		return ok
	case IntegerLiteralToken:
		_, ok := b.(IntegerLiteralToken)
		return ok
	case FloatLiteralToken:
		_, ok := b.(FloatLiteralToken)
		return ok
	case StringLiteralToken:
		_, ok := b.(StringLiteralToken)
		return ok
	case CharLiteralToken:
		_, ok := b.(CharLiteralToken)
		return ok
	case OperatorToken:
		_, ok := b.(OperatorToken)
		return ok
	case EofToken:
		_, ok := b.(EofToken)
		return ok
	case ErrorToken:
		_, ok := b.(ErrorToken)
		return ok
	}
	return a == b
}

type KeywordToken int

const (
	TokenAuto KeywordToken = iota
	TokenExtern
	TokenRegister
	TokenStatic
	TokenTypedef
	TokenVoid
	TokenChar
	TokenShort
	TokenInt
	TokenLong
	TokenFloat
	TokenDouble
	TokenSigned
	TokenUnsigned
	TokenConst
	TokenVolatile
	TokenRestrict
	TokenInline
	TokenBool
	TokenComplex
	TokenAtomic
	TokenAlignas
	TokenAlignof
	TokenGeneric
	TokenNoreturn
	TokenStaticAssert
	TokenThreadLocal
	TokenIf
	TokenElse
	TokenSwitch
	TokenCase
	TokenDefault
	TokenWhile
	TokenDo
	TokenFor
	TokenBreak
	TokenContinue
	TokenReturn
	TokenGoto
	TokenStruct
	TokenUnion
	TokenEnum
	TokenSizeof
	TokenAttribute
	TokenTypeof
	TokenExtension
	TokenAsm
	TokenAsmVolatile
	TokenLabel
	TokenRealPart
	TokenImagPart
	TokenInt128Keyword
	TokenFloat128Keyword
	TokenFloat16Keyword
	TokenFloat32Keyword
	TokenFloat64Keyword
	TokenAutoType
	TokenBuiltinVaStart
	TokenBuiltinVaEnd
	TokenBuiltinVaArg
	TokenBuiltinVaCopy
	TokenBuiltinOffsetof
	TokenBuiltinTypesCompatibleP
	TokenBuiltinChooseExpr
	TokenBuiltinConstantP
	TokenBuiltinExpect
	TokenBuiltinUnreachable
	TokenBuiltinTrap
	TokenBuiltinClz
	TokenBuiltinClzl
	TokenBuiltinClzll
	TokenBuiltinCtz
	TokenBuiltinCtzl
	TokenBuiltinCtzll
	TokenBuiltinPopcount
	TokenBuiltinPopcountl
	TokenBuiltinPopcountll
	TokenBuiltinBswap16
	TokenBuiltinBswap32
	TokenBuiltinBswap64
	TokenBuiltinFfs
	TokenBuiltinFfsll
	TokenBuiltinFrameAddress
	TokenBuiltinReturnAddress
	TokenBuiltinAssumeAligned
	TokenBuiltinAddOverflow
	TokenBuiltinSubOverflow
	TokenBuiltinMulOverflow
	TokenBuiltinAddOverflowP
	TokenBuiltinSubOverflowP
	TokenBuiltinMulOverflowP
	TokenBuiltinObjectSize
	TokenBuiltinExtractReturnAddr
	TokenBuiltinPrefetch
)

type IdentifierToken struct {
	Symbol common.Symbol
}

type IntegerLiteralToken struct {
	Value   uint64
	Suffix  IntegerSuffix
	Base    NumericBase
}

type FloatLiteralToken struct {
	Value   string
	Suffix  FloatSuffix
	Base    NumericBase
}

type StringLiteralToken struct {
	Value   string
	Prefix  StringPrefix
}

type CharLiteralToken struct {
	Value   uint32
	Prefix  StringPrefix
}

type OperatorToken int

const (
	TokenPlus OperatorToken = iota
	TokenMinus
	TokenStar
	TokenSlash
	TokenPercent
	TokenAmpersand
	TokenPipe
	TokenCaret
	TokenTilde
	TokenLessLess
	TokenGreaterGreater
	TokenEqualEqual
	TokenBangEqual
	TokenLess
	TokenGreater
	TokenLessEqual
	TokenGreaterEqual
	TokenAmpAmp
	TokenPipePipe
	TokenBang
	TokenEqual
	TokenPlusEqual
	TokenMinusEqual
	TokenStarEqual
	TokenSlashEqual
	TokenPercentEqual
	TokenAmpEqual
	TokenPipeEqual
	TokenCaretEqual
	TokenLessLessEqual
	TokenGreaterGreaterEqual
	TokenPlusPlus
	TokenMinusMinus
	TokenDot
	TokenArrow
	TokenQuestion
	TokenColon
	TokenComma
	TokenSemicolon
	TokenEllipsis
	TokenHash
	TokenHashHash
	TokenLeftParen
	TokenRightParen
	TokenLeftBracket
	TokenRightBracket
	TokenLeftBrace
	TokenRightBrace
)

type EofToken int

type ErrorToken int

func LookupKeyword(s string) TokenKind {
	switch s {
	case "auto":
		return TokenAuto
	case "break":
		return TokenBreak
	case "case":
		return TokenCase
	case "char":
		return TokenChar
	case "const":
		return TokenConst
	case "continue":
		return TokenContinue
	case "default":
		return TokenDefault
	case "do":
		return TokenDo
	case "double":
		return TokenDouble
	case "else":
		return TokenElse
	case "enum":
		return TokenEnum
	case "extern":
		return TokenExtern
	case "float":
		return TokenFloat
	case "for":
		return TokenFor
	case "goto":
		return TokenGoto
	case "if":
		return TokenIf
	case "inline":
		return TokenInline
	case "int":
		return TokenInt
	case "long":
		return TokenLong
	case "register":
		return TokenRegister
	case "restrict":
		return TokenRestrict
	case "return":
		return TokenReturn
	case "short":
		return TokenShort
	case "signed":
		return TokenSigned
	case "sizeof":
		return TokenSizeof
	case "static":
		return TokenStatic
	case "struct":
		return TokenStruct
	case "switch":
		return TokenSwitch
	case "typedef":
		return TokenTypedef
	case "union":
		return TokenUnion
	case "unsigned":
		return TokenUnsigned
	case "void":
		return TokenVoid
	case "volatile":
		return TokenVolatile
	case "while":
		return TokenWhile
	case "_Alignas":
		return TokenAlignas
	case "_Alignof", "__alignof__", "__alignof":
		return TokenAlignof
	case "_Atomic":
		return TokenAtomic
	case "_Bool":
		return TokenBool
	case "_Complex", "__complex__", "__complex":
		return TokenComplex
	case "_Generic":
		return TokenGeneric
	case "_Noreturn":
		return TokenNoreturn
	case "_Static_assert":
		return TokenStaticAssert
	case "_Thread_local":
		return TokenThreadLocal
	case "__attribute__":
		return TokenAttribute
	case "__attribute":
		return TokenAttribute
	case "typeof":
		return TokenTypeof
	case "__typeof":
		return TokenTypeof
	case "__typeof__":
		return TokenTypeof
	case "__extension__":
		return TokenExtension
	case "__real__":
		return TokenRealPart
	case "__real":
		return TokenRealPart
	case "__imag__":
		return TokenImagPart
	case "__imag":
		return TokenImagPart
	case "asm":
		return TokenAsm
	case "__asm__":
		return TokenAsm
	case "__asm":
		return TokenAsm
	case "__volatile__":
		return TokenVolatile
	case "__inline__":
		return TokenInline
	case "__label__":
		return TokenLabel
	case "__int128":
		return TokenInt128Keyword
	case "__int128_t":
		return TokenInt128Keyword
	case "_Float128":
		return TokenFloat128Keyword
	case "__float128":
		return TokenFloat128Keyword
	case "_Float16":
		return TokenFloat16Keyword
	case "_Float32":
		return TokenFloat32Keyword
	case "_Float32x":
		return TokenFloat32Keyword
	case "_Float64":
		return TokenFloat64Keyword
	case "_Float64x":
		return TokenFloat64Keyword
	case "__auto_type":
		return TokenAutoType
	case "__const":
		return TokenConst
	case "__const__":
		return TokenConst
	case "__restrict":
		return TokenRestrict
	case "__restrict__":
		return TokenRestrict
	case "__volatile":
		return TokenVolatile
	case "__signed__":
		return TokenSigned
	case "__inline":
		return TokenInline
	case "__builtin_va_start":
		return TokenBuiltinVaStart
	case "__builtin_va_end":
		return TokenBuiltinVaEnd
	case "__builtin_va_arg":
		return TokenBuiltinVaArg
	case "__builtin_va_copy":
		return TokenBuiltinVaCopy
	case "__builtin_offsetof":
		return TokenBuiltinOffsetof
	case "__builtin_types_compatible_p":
		return TokenBuiltinTypesCompatibleP
	case "__builtin_choose_expr":
		return TokenBuiltinChooseExpr
	case "__builtin_constant_p":
		return TokenBuiltinConstantP
	case "__builtin_expect":
		return TokenBuiltinExpect
	case "__builtin_unreachable":
		return TokenBuiltinUnreachable
	case "__builtin_trap":
		return TokenBuiltinTrap
	case "__builtin_clz":
		return TokenBuiltinClz
	case "__builtin_clzl":
		return TokenBuiltinClzl
	case "__builtin_clzll":
		return TokenBuiltinClzll
	case "__builtin_ctz":
		return TokenBuiltinCtz
	case "__builtin_ctzl":
		return TokenBuiltinCtzl
	case "__builtin_ctzll":
		return TokenBuiltinCtzll
	case "__builtin_popcount":
		return TokenBuiltinPopcount
	case "__builtin_popcountl":
		return TokenBuiltinPopcountl
	case "__builtin_popcountll":
		return TokenBuiltinPopcountll
	case "__builtin_bswap16":
		return TokenBuiltinBswap16
	case "__builtin_bswap32":
		return TokenBuiltinBswap32
	case "__builtin_bswap64":
		return TokenBuiltinBswap64
	case "__builtin_ffs":
		return TokenBuiltinFfs
	case "__builtin_ffsll":
		return TokenBuiltinFfsll
	case "__builtin_frame_address":
		return TokenBuiltinFrameAddress
	case "__builtin_return_address":
		return TokenBuiltinReturnAddress
	case "__builtin_assume_aligned":
		return TokenBuiltinAssumeAligned
	case "__builtin_add_overflow":
		return TokenBuiltinAddOverflow
	case "__builtin_sub_overflow":
		return TokenBuiltinSubOverflow
	case "__builtin_mul_overflow":
		return TokenBuiltinMulOverflow
	case "__builtin_add_overflow_p":
		return TokenBuiltinAddOverflowP
	case "__builtin_sub_overflow_p":
		return TokenBuiltinSubOverflowP
	case "__builtin_mul_overflow_p":
		return TokenBuiltinMulOverflowP
	case "__builtin_object_size":
		return TokenBuiltinObjectSize
	case "__builtin_extract_return_addr":
		return TokenBuiltinExtractReturnAddr
	case "__builtin_prefetch":
		return TokenBuiltinPrefetch
	default:
		return nil
	}
}