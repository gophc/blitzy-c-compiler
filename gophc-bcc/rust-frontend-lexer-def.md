# BCC Rust 前端词法分析器定义

本文档按文件整理 `src/frontend/lexer/` 下所有类型、函数和方法签名（私有），按定义出现顺序，不包括测试。

---

## 1. src/frontend/lexer/mod.rs

### Lexer<'src> 类型定义

```rust
pub struct Lexer<'src> {
    source: &'src str,
    file_id: u32,
    position: u32,
    line: u32,
    column: u32,
    current_token: Option<Token>,
    pushed_back: Vec<Token>,
    interner: &'src mut Interner,
    diagnostics: &'src mut DiagnosticEngine,
}
```

### Lexer<'src> 方法

```rust
impl<'src> Lexer<'src> {
    pub fn new(source: &'src str, file_id: u32, interner: &'src mut Interner, diagnostics: &'src mut DiagnosticEngine) -> Self
    pub fn next_token(&mut self) -> Token
    pub fn peek(&mut self) -> &Token
    pub fn peek_nth(&mut self, n: usize) -> &Token
    pub fn unget(&mut self, token: Token)
    pub fn tokenize_all(&mut self) -> Vec<Token>
    fn lex_token(&mut self) -> Token
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
    fn lex_error_char(&mut self, ch: char, start: u32) -> Token
    fn skip_whitespace_and_comments(&mut self)
    fn lex_identifier_or_keyword(&mut self) -> TokenKind
    fn lex_dot_float(&mut self, start: u32) -> TokenKind
    fn make_token(&self, kind: TokenKind, start: u32) -> Token
    pub fn current_position(&self) -> (u32, u32, u32)
    pub fn source(&self) -> &'src str
    pub fn diagnostics(&self) -> &DiagnosticEngine
    pub fn diagnostics_mut(&mut self) -> &mut DiagnosticEngine
    fn interner(&self) -> &Interner
    fn interner_mut(&mut self) -> &mut Interner
    fn try_consume_ident_char(&mut self) -> bool
}
```

### 私有辅助函数

```rust
fn is_c_whitespace(ch: char) -> bool
```

---

## 2. src/frontend/lexer/token.rs

### NumericBase 类型定义

```rust
pub enum NumericBase {
    Decimal,
    Octal,
    Hex,
    Binary,
}
```

### IntegerSuffix 类型定义

```rust
pub enum IntegerSuffix {
    None,
    U,
    L,
    UL,
    LL,
    ULL,
}
```

### FloatSuffix 类型定义

```rust
pub enum FloatSuffix {
    None,
    F,
    L,
    I,
    FI,
    LI,
}
```

### StringPrefix 类型定义

```rust
pub enum StringPrefix {
    None,
    L,
    U8,
    U16,
    U32,
}
```

### TokenKind 类型定义

```rust
pub enum TokenKind {
    Keyword,
    Identifier,
    Integer,
    Float,
    String,
    Char,
    Punctuator,
    Eof,
    Invalid,
}
```

### Token 类型定义

```rust
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}
```

### TokenKind 方法

```rust
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

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
}
```

### Token 方法

```rust
impl Token {
    pub fn new(kind: TokenKind, span: Span) -> Self
    pub fn is(&self, kind: &TokenKind) -> bool
    pub fn is_eof(&self) -> bool
}
```

### 公共函数

```rust
pub fn lookup_keyword(s: &str) -> Option<TokenKind>
```

---

## 3. src/frontend/lexer/string_literal.rs

### 私有辅助函数

```rust
fn escape_byte_to_char(value: u8) -> char
fn process_escape_sequence(scanner: &mut Scanner, diagnostics: &mut DiagnosticEngine, file_id: u32) -> Option<char>
fn process_unicode_escape(scanner: &mut Scanner, diagnostics: &mut DiagnosticEngine, file_id: u32, num_digits: usize, esc_start: u32) -> Option<char>
fn skip_whitespace(scanner: &mut Scanner)
fn merge_prefix(a: StringPrefix, b: StringPrefix, diagnostics: &mut DiagnosticEngine, _file_id: u32, span: Span) -> StringPrefix
fn consume_adjacent_prefix(scanner: &mut Scanner) -> Option<StringPrefix>
```

### 公共函数

```rust
pub fn lex_string_literal(scanner: &mut Scanner, prefix: StringPrefix, diagnostics: &mut DiagnosticEngine, file_id: u32) -> TokenKind
pub fn lex_char_literal(scanner: &mut Scanner, prefix: StringPrefix, diagnostics: &mut DiagnosticEngine, file_id: u32) -> TokenKind
pub fn detect_prefix(scanner: &mut Scanner) -> Option<(StringPrefix, bool)>
pub fn has_adjacent_string(scanner: &mut Scanner) -> bool
pub fn lex_string_with_concatenation(scanner: &mut Scanner, prefix: StringPrefix, diagnostics: &mut DiagnosticEngine, file_id: u32) -> TokenKind
```

---

## 4. src/frontend/lexer/scanner.rs

### Position 类型定义

```rust
pub struct Position {
    offset: u32,
    line: u32,
    column: u32,
}
```

### Position 方法

```rust
impl Position {
    pub fn new(offset: u32, line: u32, column: u32) -> Self
}
```

### Scanner<'src> 类型定义

```rust
pub struct Scanner<'src> {
    source: &'src str,
    len: usize,
    pos: usize,
    line: u32,
    column: u32,
    lookahead: Vec<(char, Position)>,
}
```

### Scanner 方法

```rust
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
    fn fill_lookahead(&mut self, target_len: usize)
    fn next_lookahead_position(&self) -> Position
    fn source_byte_len(&self, ch: char, at_offset: usize) -> usize
}
```

---

## 5. src/frontend/lexer/number_literal.rs

### 私有辅助函数

```rust
fn digit_value(ch: char) -> u64
fn parse_integer_value(digits: &str, radix: u32) -> (u64, bool)
fn parse_integer_suffix(scanner: &mut Scanner, diagnostics: &mut DiagnosticEngine, file_id: u32) -> IntegerSuffix
fn parse_float_suffix(scanner: &mut Scanner) -> FloatSuffix
fn check_trailing_invalid_chars(scanner: &mut Scanner, diagnostics: &mut DiagnosticEngine, file_id: u32, _literal_start: u32)
fn maybe_convert_to_imaginary(scanner: &mut Scanner, token: TokenKind) -> TokenKind
fn lex_hex_literal(scanner: &mut Scanner, diagnostics: &mut DiagnosticEngine, file_id: u32, start_offset: u32) -> TokenKind
fn lex_hex_float_exponent(scanner: &mut Scanner, diagnostics: &mut DiagnosticEngine, file_id: u32, start_offset: u32) -> TokenKind
fn lex_binary_literal(scanner: &mut Scanner, diagnostics: &mut DiagnosticEngine, file_id: u32, start_offset: u32) -> TokenKind
fn lex_after_leading_zero(scanner: &mut Scanner, diagnostics: &mut DiagnosticEngine, file_id: u32, start_offset: u32) -> TokenKind
fn lex_decimal_literal(scanner: &mut Scanner, diagnostics: &mut DiagnosticEngine, file_id: u32, start_offset: u32) -> TokenKind
fn lex_decimal_float_after_dot(scanner: &mut Scanner, diagnostics: &mut DiagnosticEngine, file_id: u32, start_offset: u32) -> TokenKind
fn lex_decimal_exponent(scanner: &mut Scanner, diagnostics: &mut DiagnosticEngine, file_id: u32, start_offset: u32) -> TokenKind
```

### 公共函数

```rust
pub fn lex_number(scanner: &mut Scanner, diagnostics: &mut DiagnosticEngine, file_id: u32) -> TokenKind
```