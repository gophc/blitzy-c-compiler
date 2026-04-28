# Frontend Lexer 定义 (Rust BCC 对比 Go 实现)

## 1. mod.rs (Rust) - Lexer 结构体

### 类型定义

```rust
pub struct Lexer<'src> {
    scanner: Scanner<'src>,
    interner: &'src mut Interner,
    diagnostics: &'src mut DiagnosticEngine,
    file_id: u32,
    lookahead: Vec<Token>,
}
```

### 函数签名

```rust
// 构造
pub fn new(source: &'src str, file_id: u32, interner: &'src mut Interner, diagnostics: &'src mut DiagnosticEngine) -> Self

// 公共 API - token 流接口
pub fn next_token(&mut self) -> Token
pub fn peek(&mut self) -> &Token
pub fn peek_nth(&mut self, n: usize) -> &Token
pub fn unget(&mut self, token: Token)
pub fn tokenize_all(&mut self) -> Vec<Token>

// 内部 lexing - 主调度
fn lex_token(&mut self) -> Token

// 操作符消歧义
fn lex_after_plus(&mut self, start: u32) -> Token
fn lex_after_minus(&mut self, start: u32) -> Token
fn lex_after_star(&mut self, start: u32) -> Token
fn lex_after_slash(&mut self, start: u32) -> Token
fn lex_after_percent(&mut self, start: u32) -> Token
fn lex_after_ampersand(&mut self, start: u32) -> Token
fn lex_after_pipe(&mut self, start: u32) -> Token
fn lex_after_caret(&mut self, start: u32) -> Token
fn lex_after_bang(&mut self, start: u32) -> Token
fn lex_after_equal(&mut self, start: u32) -> Token
fn lex_after_less(&mut self, start: u32) -> Token
fn lex_after_greater(&mut self, start: u32) -> Token
fn lex_after_dot(&mut self, start: u32) -> Token

// 错误字符处理
fn lex_error_char(&mut self, ch: char, start: u32) -> Token

// 空白和注释跳过
fn skip_whitespace_and_comments(&mut self)

// 标识符和关键字
fn lex_identifier_or_keyword(&mut self) -> TokenKind

// 点开头浮点数
fn lex_dot_float(&mut self, start: u32) -> TokenKind

// Token 构造辅助
fn make_token(&self, kind: TokenKind, start: u32) -> Token

// 访问器方法
pub fn current_position(&self) -> (u32, u32, u32)
pub fn source(&self) -> &'src str
pub fn diagnostics(&self) -> &DiagnosticEngine
pub fn diagnostics_mut(&mut self) -> &mut DiagnosticEngine
pub fn interner(&self) -> &Interner
pub fn interner_mut(&mut self) -> &mut Interner
```

---

## 2. token.rs (Rust) - Token 类型

### 类型定义

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NumericBase {
    Decimal,
    Hexadecimal,
    Octal,
    Binary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntegerSuffix {
    None,
    U,
    L,
    UL,
    LL,
    ULL,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FloatSuffix {
    None,
    F,
    L,
    I,
    FI,
    LI,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StringPrefix {
    None,
    L,
    U8,
    U16,
    U32,
}

pub enum TokenKind { ... } // 大型枚举，包含所有关键字和操作符

pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}
```

### 函数签名

```rust
impl Token {
    pub fn new(kind: TokenKind, span: Span) -> Self
    pub fn is(&self, kind: &TokenKind) -> bool
    pub fn is_eof(&self) -> bool
}

impl TokenKind {
    pub fn is_keyword(&self) -> bool
    pub fn is_literal(&self) -> bool
    pub fn is_operator(&self) -> bool
    pub fn is_assignment_operator(&self) -> bool
    pub fn is_comparison_operator(&self) -> bool
    pub fn is_unary_operator(&self) -> bool
    pub fn keyword_str(&self) -> Option<&'static str>
    pub fn punctuator_str(&self) -> Option<&'static str>
}

pub fn lookup_keyword(s: &str) -> Option<TokenKind>
```

---

## 3. scanner.rs (Rust) - Scanner 类���

### 类型定义

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Position {
    pub offset: u32,
    pub line: u32,
    pub column: u32,
}

pub struct Scanner<'src> {
    source: &'src str,
    chars: Peekable<CharIndices<'src>>,
    offset: usize,
    line: u32,
    column: u32,
    lookahead: Vec<(char, Position)>,
}
```

### 函数签名

```rust
impl Position {
    pub fn new(offset: u32, line: u32, column: u32) -> Self
}

impl<'src> Scanner<'src> {
    pub fn new(source: &'src str) -> Self
    pub fn advance(&mut self) -> Option<char>
    pub fn peek(&mut self) -> Option<char>
    pub fn peek_nth(&mut self, n: usize) -> Option<char>
    pub fn unget(&mut self, ch: char, pos: Position)
    pub fn position(&self) -> Position
    pub fn offset(&self) -> u32
    pub fn line(&self) -> u32
    pub fn column(&self) -> u32
    pub fn is_eof(&mut self) -> bool
    pub fn source(&self) -> &'src str
    pub fn slice(&self, start: usize, end: usize) -> &'src str
    pub fn skip_while(&mut self, predicate: impl Fn(char) -> bool)
    pub fn consume_if(&mut self, ch: char) -> bool
    pub fn consume_if_pred(&mut self, pred: impl Fn(char) -> bool) -> Option<char>
    pub fn is_pua_char(ch: char) -> bool
}
```

---

## 4. number_literal.rs (Rust)

### 函数签名

```rust
fn digit_value(ch: char) -> u64
fn parse_integer_value(digits: &str, radix: u32) -> (u64, bool)
fn parse_integer_suffix(scanner: &mut Scanner, diagnostics: &mut DiagnosticEngine, file_id: u32) -> IntegerSuffix
fn parse_float_suffix(scanner: &mut Scanner) -> FloatSuffix
fn check_trailing_invalid_chars(scanner: &mut Scanner, diagnostics: &mut DiagnosticEngine, file_id: u32, literal_start: u32)
fn maybe_convert_to_imaginary(scanner: &mut Scanner, token: TokenKind) -> TokenKind
pub fn lex_number(scanner: &mut Scanner, diagnostics: &mut DiagnosticEngine, file_id: u32) -> TokenKind

fn lex_hex_literal(scanner: &mut Scanner, diagnostics: &mut DiagnosticEngine, file_id: u32, start_offset: u32) -> TokenKind
fn lex_hex_float_exponent(scanner: &mut Scanner, diagnostics: &mut DiagnosticEngine, file_id: u32, start_offset: u32) -> TokenKind
fn lex_binary_literal(scanner: &mut Scanner, diagnostics: &mut DiagnosticEngine, file_id: u32, start_offset: u32) -> TokenKind
fn lex_after_leading_zero(scanner: &mut Scanner, diagnostics: &mut DiagnosticEngine, file_id: u32, start_offset: u32) -> TokenKind
fn lex_decimal_literal(scanner: &mut Scanner, diagnostics: &mut DiagnosticEngine, file_id: u32, start_offset: u32) -> TokenKind
fn lex_decimal_float_after_dot(scanner: &mut Scanner, diagnostics: &mut DiagnosticEngine, file_id: u32, start_offset: u32) -> TokenKind
fn lex_decimal_exponent(scanner: &mut Scanner, diagnostics: &mut DiagnosticEngine, file_id: u32, start_offset: u32) -> TokenKind
```

---

## 5. string_literal.rs (Rust)

### 函数签名

```rust
fn escape_byte_to_char(value: u8) -> char
fn process_escape_sequence(scanner: &mut Scanner, diagnostics: &mut DiagnosticEngine, file_id: u32) -> Option<char>
fn process_unicode_escape(scanner: &mut Scanner, diagnostics: &mut DiagnosticEngine, file_id: u32, num_digits: usize, esc_start: u32) -> Option<char>
pub fn lex_string_literal(scanner: &mut Scanner, prefix: StringPrefix, diagnostics: &mut DiagnosticEngine, file_id: u32) -> TokenKind
pub fn lex_char_literal(scanner: &mut Scanner, prefix: StringPrefix, diagnostics: &mut DiagnosticEngine, file_id: u32) -> TokenKind
pub fn detect_prefix(scanner: &mut Scanner) -> Option<(StringPrefix, bool)>
pub fn has_adjacent_string(scanner: &mut Scanner) -> bool
fn skip_whitespace(scanner: &mut Scanner)
fn merge_prefix(a: StringPrefix, b: StringPrefix, diagnostics: &mut DiagnosticEngine, _file_id: u32, span: Span) -> StringPrefix
fn consume_adjacent_prefix(scanner: &mut Scanner) -> Option<StringPrefix>
pub fn lex_string_with_concatenation(scanner: &mut Scanner, prefix: StringPrefix, diagnostics: &mut DiagnosticEngine, file_id: u32) -> TokenKind
```

---

# Go 实现状态

## 已完成的补充 ✅

### lexer.go 已补全
- `DiagnosticsMut() *common.DiagnosticEngine`
- `InternerMut() *common.Interner`

### token.go 已补全
- `IsKeyword(kind TokenKind) bool`
- `IsLiteral(kind TokenKind) bool`
- `IsOperator(kind TokenKind) bool`
- `IsAssignmentOperator(kind TokenKind) bool`
- `IsComparisonOperator(kind TokenKind) bool`
- `IsUnaryOperator(kind TokenKind) bool`
- `KeywordStr(kind TokenKind) string`
- `PunctuatorStr(kind TokenKind) string`

---

## 待检查/可能需要修复的问题

### scanner.go
- `fillLookahead` 和 `nextLookaheadPosition` 的实现可能需要验证

### string_literal.go  
- `LexStringWithConcatenation` 需要验证实现正确性