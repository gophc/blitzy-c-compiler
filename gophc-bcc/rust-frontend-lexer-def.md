# Rust 前端词法分析器定义

本文档按文件整理 `src/frontend/lexer/` 下所有类型、函数和方法签名（私有），按定义出现顺序，不包括测试。

---

## mod.rs

### 类型

- `Lexer<'src>` — struct (第114行)

### Lexer<'src> impl 方法

1. `pub fn new(source: &'src str, file_id: u32, interner: &'src mut Interner, diagnostics: &'src mut DiagnosticEngine) -> Self`
2. `pub fn next_token(&mut self) -> Token`
3. `pub fn peek(&mut self) -> &Token`
4. `pub fn peek_nth(&mut self, n: usize) -> &Token`
5. `pub fn unget(&mut self, token: Token)`
6. `pub fn tokenize_all(&mut self) -> Vec<Token>`
7. `fn lex_token(&mut self) -> Token` +1 (私有)
8. `fn lex_after_plus(&mut self, start: u32) -> Token` +2 (私有)
9. `fn lex_after_minus(&mut self, start: u32) -> Token` +3 (私有)
10. `fn lex_after_star(&mut self, start: u32) -> Token` +4 (私有)
11. `fn lex_after_slash(&mut self, start: u32) -> Token` +5 (私有)
12. `fn lex_after_percent(&mut self, start: u32) -> Token` +6 (私有)
13. `fn lex_after_ampersand(&mut self, start: u32) -> Token` +7 (私有)
14. `fn lex_after_pipe(&mut self, start: u32) -> Token` +8 (私有)
15. `fn lex_after_caret(&mut self, start: u32) -> Token` +9 (私有)
16. `fn lex_after_bang(&mut self, start: u32) -> Token` +10 (私有)
17. `fn lex_after_equal(&mut self, start: u32) -> Token` +11 (私有)
18. `fn lex_after_less(&mut self, start: u32) -> Token` +12 (私有)
19. `fn lex_after_greater(&mut self, start: u32) -> Token` +13 (私有)
20. `fn lex_after_dot(&mut self, start: u32) -> Token` +14 (私有)
21. `fn lex_error_char(&mut self, ch: char, start: u32) -> Token` +15 (私有)
22. `fn skip_whitespace_and_comments(&mut self)` +16 (私有)
23. `fn lex_identifier_or_keyword(&mut self) -> TokenKind` +17 (私有)
24. `fn lex_dot_float(&mut self, start: u32) -> TokenKind` +18 (私有)
25. `fn make_token(&self, kind: TokenKind, start: u32) -> Token` +19 (私有)
26. `pub fn current_position(&self) -> (u32, u32, u32)`
27. `pub fn source(&self) -> &'src str`
28. `pub fn diagnostics(&self) -> &DiagnosticEngine`
29. `pub fn diagnostics_mut(&mut self) -> &mut DiagnosticEngine`
30. `fn interner(&self) -> &Interner` +20 (私有)
31. `fn interner_mut(&mut self) -> &mut Interner` +21 (私有)
32. `fn try_consume_ident_char(&mut self) -> bool` +22 (私有)

### 模块级函数

- `fn is_c_whitespace(ch: char) -> bool` (私有)

---

## token.rs

### 类型

1. `pub enum NumericBase` (第51行)
2. `pub enum IntegerSuffix` (第72行)
3. `pub enum FloatSuffix` (第94行)
4. `pub enum StringPrefix` (第122行)
5. `pub enum TokenKind` (第153行)
6. `pub struct Token` (第1098行)

### TokenKind impl 方法

1. `pub fn is_keyword(&self) -> bool`
2. `pub fn is_literal(&self) -> bool`
3. `pub fn is_operator(&self) -> bool`
4. `pub fn is_assignment_operator(&self) -> bool`
5. `pub fn is_comparison_operator(&self) -> bool`
6. `pub fn is_unary_operator(&self) -> bool`
7. `pub fn keyword_str(&self) -> Option<&'static str>`
8. `pub fn punctuator_str(&self) -> Option<&'static str>`
9. `fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result` +1 (Display 实现)

### Token impl 方法

1. `pub fn new(kind: TokenKind, span: Span) -> Self`
2. `pub fn is(&self, kind: &TokenKind) -> bool`
3. `pub fn is_eof(&self) -> bool`

### 模块级函数

- `pub fn lookup_keyword(s: &str) -> Option<TokenKind>`

---

## string_literal.rs

### 内部帮助函数

- `fn escape_byte_to_char(value: u8) -> char` (私有)

### 模块级函数

1. `fn process_escape_sequence(scanner: &mut Scanner, diagnostics: &mut DiagnosticEngine, file_id: u32) -> Option<char>` +1 (私有)
2. `fn process_unicode_escape(scanner: &mut Scanner, diagnostics: &mut DiagnosticEngine, file_id: u32, num_digits: usize, esc_start: u32) -> Option<char>` +2 (私有)
3. `pub fn lex_string_literal(scanner: &mut Scanner, prefix: StringPrefix, diagnostics: &mut DiagnosticEngine, file_id: u32) -> TokenKind`
4. `pub fn lex_char_literal(scanner: &mut Scanner, prefix: StringPrefix, diagnostics: &mut DiagnosticEngine, file_id: u32) -> TokenKind`
5. `pub fn detect_prefix(scanner: &mut Scanner) -> Option<(StringPrefix, bool)>`
6. `pub fn has_adjacent_string(scanner: &mut Scanner) -> bool`
7. `fn skip_whitespace(scanner: &mut Scanner)` +1 (私有)
8. `fn merge_prefix(a: StringPrefix, b: StringPrefix, diagnostics: &mut DiagnosticEngine, _file_id: u32, span: Span) -> StringPrefix` +2 (私有)
9. `fn consume_adjacent_prefix(scanner: &mut Scanner) -> Option<StringPrefix>` +3 (私有)
10. `pub fn lex_string_with_concatenation(scanner: &mut Scanner, prefix: StringPrefix, diagnostics: &mut DiagnosticEngine, file_id: u32) -> TokenKind`

---

## scanner.rs

### 类型

1. `pub struct Position` (第55行)

### Position impl 方法

- `pub fn new(offset: u32, line: u32, column: u32) -> Self`

### Scanner<'src> 类型

- `pub struct Scanner<'src>` (第103行)

### Scanner impl 方法

1. `pub fn new(source: &'src str) -> Self`
2. `pub fn advance(&mut self) -> Option<char>`
3. `pub fn peek(&mut self) -> Option<char>`
4. `pub fn peek_nth(&mut self, n: usize) -> Option<char>`
5. `pub fn unget(&mut self, ch: char, pos: Position)`
6. `pub fn position(&self) -> Position`
7. `pub fn offset(&self) -> u32`
8. `pub fn line(&self) -> u32`
9. `pub fn column(&self) -> u32`
10. `pub fn is_eof(&mut self) -> bool`
11. `pub fn source(&self) -> &'src str`
12. `pub fn slice(&self, start: usize, end: usize) -> &'src str`
13. `pub fn skip_while(&mut self, predicate: impl Fn(char) -> bool)`
14. `pub fn consume_if(&mut self, ch: char) -> bool`
15. `pub fn consume_if_pred(&mut self, pred: impl Fn(char) -> bool) -> Option<char>`
16. `pub fn is_pua_char(ch: char) -> bool` +1 (静态方法)
17. `fn fill_lookahead(&mut self, target_len: usize)` +2 (私有)
18. `fn next_lookahead_position(&self) -> Position` +3 (私有)
19. `fn source_byte_len(&self, ch: char, at_offset: usize) -> usize` +4 (私有)

---

## number_literal.rs

### 内部帮助函数

1. `fn digit_value(ch: char) -> u64` +1 (私有)
2. `fn parse_integer_value(digits: &str, radix: u32) -> (u64, bool)` +2 (私有)
3. `fn parse_integer_suffix(scanner: &mut Scanner, diagnostics: &mut DiagnosticEngine, file_id: u32) -> IntegerSuffix` +3 (私有)
4. `fn parse_float_suffix(scanner: &mut Scanner) -> FloatSuffix` +4 (私有)
5. `fn check_trailing_invalid_chars(scanner: &mut Scanner, diagnostics: &mut DiagnosticEngine, file_id: u32, _literal_start: u32)` +5 (私有)
6. `fn maybe_convert_to_imaginary(scanner: &mut Scanner, token: TokenKind) -> TokenKind` +6 (私有)
7. `fn lex_hex_literal(scanner: &mut Scanner, diagnostics: &mut DiagnosticEngine, file_id: u32, start_offset: u32) -> TokenKind` +7 (私有)
8. `fn lex_hex_float_exponent(scanner: &mut Scanner, diagnostics: &mut DiagnosticEngine, file_id: u32, start_offset: u32) -> TokenKind` +8 (私有)
9. `fn lex_binary_literal(scanner: &mut Scanner, diagnostics: &mut DiagnosticEngine, file_id: u32, start_offset: u32) -> TokenKind` +9 (私有)
10. `fn lex_after_leading_zero(scanner: &mut Scanner, diagnostics: &mut DiagnosticEngine, file_id: u32, start_offset: u32) -> TokenKind` +10 (私有)
11. `fn lex_decimal_literal(scanner: &mut Scanner, diagnostics: &mut DiagnosticEngine, file_id: u32, start_offset: u32) -> TokenKind` +11 (私有)
12. `fn lex_decimal_float_after_dot(scanner: &mut Scanner, diagnostics: &mut DiagnosticEngine, file_id: u32, start_offset: u32) -> TokenKind` +12 (私有)
13. `fn lex_decimal_exponent(scanner: &mut Scanner, diagnostics: &mut DiagnosticEngine, file_id: u32, start_offset: u32) -> TokenKind` +13 (私有)

### 模块级函数

- `pub fn lex_number(scanner: &mut Scanner, diagnostics: &mut DiagnosticEngine, file_id: u32) -> TokenKind`