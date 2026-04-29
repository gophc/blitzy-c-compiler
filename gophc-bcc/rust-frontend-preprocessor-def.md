# BCC Rust 前端预处理器定义

本文档按文件整理 `src/frontend/preprocessor/` 下所有类型、函数和方法签名（私有），按定义出现顺序，不包括测试。

---

## 1. src/frontend/preprocessor/mod.rs

### PPTokenKind 类型定义

```rust
pub enum PPTokenKind {
    Identifier,
    Number,
    StringLiteral,
    CharLiteral,
    Punctuator,
    Whitespace,
    Newline,
    HeaderName,
    PlacemarkerToken,
    EndOfFile,
}
```

### PPToken 类型定义

```rust
pub struct PPToken {
    pub kind: PPTokenKind,
    pub text: String,
    pub span: Span,
    pub from_macro: bool,
    pub painted: bool,
}
```

### MacroKind 类型定义

```rust
pub enum MacroKind {
    ObjectLike,
    FunctionLike { params: Vec<String>, variadic: bool },
}
```

### MacroDef 类型定义

```rust
pub struct MacroDef {
    pub name: String,
    pub kind: MacroKind,
    pub replacement: Vec<PPToken>,
    pub is_predefined: bool,
    pub definition_span: Span,
}
```

### ConditionalState 类型定义

```rust
pub struct ConditionalState {
    pub active: bool,
    pub seen_active: bool,
    pub seen_else: bool,
    pub opening_span: Span,
}
```

### Preprocessor<'a> 类型定义

```rust
pub struct Preprocessor<'a> {
    pub source_map: &'a mut SourceMap,
    pub diagnostics: &'a mut DiagnosticEngine,
    pub target: Target,
    pub interner: &'a mut Interner,
    pub macro_defs: FxHashMap<String, MacroDef>,
    pub include_paths: Vec<PathBuf>,
    pub system_include_paths: Vec<PathBuf>,
    pub cli_defines: Vec<(String, String)>,
    pub include_depth: usize,
    pub max_include_depth: usize,
    pub max_recursion_depth: usize,
    pub conditional_stack: Vec<ConditionalState>,
    pub counter_value: u64,
    pub macro_push_stack: FxHashMap<String, Vec<Option<MacroDef>>>,
    pub preserve_pragmas: bool,
}
```

### PPToken 方法

```rust
impl PPToken {
    pub fn new(kind: PPTokenKind, text: impl Into<String>, span: Span) -> Self
    pub fn from_expansion(kind: PPTokenKind, text: impl Into<String>, span: Span) -> Self
    pub fn eof(span: Span) -> Self
    pub fn placemarker(span: Span) -> Self
    pub fn is_whitespace(&self) -> bool
    pub fn is_eof(&self) -> bool
}
```

### ConditionalState 方法

```rust
impl ConditionalState {
    pub fn new(active: bool, opening_span: Span) -> Self
}
```

### Preprocessor<'a> 方法

```rust
impl<'a> Preprocessor<'a> {
    pub fn new(source_map: &'a mut SourceMap, diagnostics: &'a mut DiagnosticEngine, target: Target, interner: &'a mut Interner) -> Self
    pub fn add_include_path(&mut self, path: &str)
    pub fn add_system_include_path(&mut self, path: &str)
    pub fn add_define(&mut self, name: &str, value: &str)
    pub fn add_undef(&mut self, name: &str)
    pub fn preprocess_file(&mut self, filename: &str) -> Result<Vec<PPToken>, ()>
    fn process_tokens(&mut self, tokens: &[PPToken]) -> Vec<PPToken>
    fn is_active(&self) -> bool
    fn process_directive_line(&mut self, _hash_token: &PPToken, tokens: &[PPToken]) -> Option<Vec<PPToken>>
    fn process_define(&mut self, tokens: &[PPToken])
    fn process_undef(&mut self, tokens: &[PPToken])
    fn process_include(&mut self, tokens: &[PPToken]) -> Result<Vec<PPToken>, ()>
    fn process_ifdef(&mut self, tokens: &[PPToken], span: Span)
    fn process_ifndef(&mut self, tokens: &[PPToken], span: Span)
    fn process_if(&mut self, tokens: &[PPToken], span: Span)
    fn process_elif(&mut self, hash_token: &PPToken, tokens: &[PPToken])
    fn process_else(&mut self, hash_token: &PPToken)
    fn process_endif(&mut self, hash_token: &PPToken)
}
```

### 公共函数

```rust
pub fn phase1_trigraphs(input: &str) -> String
pub fn phase1_line_splice(input: &str) -> String
pub fn tokenize_preprocessing(input: &str, file_id: u32) -> Vec<PPToken>
```

### 私有函数

```rust
fn is_string_prefix(bytes: &[u8], pos: usize) -> bool
fn is_char_prefix(bytes: &[u8], pos: usize) -> bool
fn lex_string_literal(input: &str, pos: usize, file_id: u32) -> (PPToken, usize)
fn lex_char_literal(input: &str, pos: usize, file_id: u32) -> (PPToken, usize)
fn match_punctuator(bytes: &[u8], pos: usize) -> Option<(&'static str, usize)>
#[inline]
fn get_char_at(s: &str, pos: usize) -> char
```

---

## 2. src/frontend/preprocessor/token_paster.rs

### PasteError 类型定义

```rust
#[derive(Debug)]
pub enum PasteError {
    InvalidToken(String),
}
```

### 公共函数

```rust
pub fn paste_tokens(left: &PPToken, right: &PPToken) -> Result<PPToken, PasteError>
pub fn process_concatenation(tokens: &[PPToken]) -> (Vec<PPToken>, Vec<Diagnostic>)
pub fn stringify_tokens(tokens: &[PPToken]) -> PPToken
```

### 私有函数

```rust
fn classify_concatenated_token(text: &str) -> Option<PPTokenKind>
#[allow(dead_code)]
fn is_valid_preprocessing_token(text: &str) -> bool
#[inline]
fn is_identifier_start(b: u8) -> bool
#[inline]
fn is_identifier_continue(b: u8) -> bool
fn is_valid_pp_number(text: &str) -> bool
fn is_valid_punctuator(text: &str) -> bool
fn is_valid_string_literal(text: &str) -> bool
fn is_valid_char_literal(text: &str) -> bool
fn escape_for_string_literal(text: &str) -> String
fn normalize_whitespace_for_stringify(tokens: &[PPToken]) -> String
#[inline]
fn is_hashhash(token: &PPToken) -> bool
```

---

## 3. src/frontend/preprocessor/predefined.rs

### MagicMacro 类型定义

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MagicMacro {
    File,
    Line,
    Counter,
    Date,
    Time,
}
```

### 公共函数

```rust
pub fn is_magic_macro(name: &str) -> Option<MagicMacro>
pub fn capture_compilation_timestamp() -> (String, String)
pub fn register_predefined_macros(macro_defs: &mut FxHashMap<String, MacroDef>, target: &Target)
```

### 私有函数

```rust
fn unix_timestamp_to_components(total_secs: u64) -> (i32, u32, u32, u32, u32, u64)
fn register_object_macro(macro_defs: &mut FxHashMap<String, MacroDef>, name: &str, value: &str)
fn tokenize_value(value: &str) -> Vec<PPToken>
```

---

## 4. src/frontend/preprocessor/paint_marker.rs

### PaintState 类型定义

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PaintState {
    Unpainted,
    Painted,
}
```

### PaintMarker 类型定义

```rust
pub struct PaintMarker {
    active_expansions: FxHashSet<String>,
}
```

### PaintState 方法

```rust
impl Default for PaintState {
    fn default() -> Self
}

impl fmt::Display for PaintState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
}
```

### PaintMarker 方法

```rust
impl Default for PaintMarker {
    fn default() -> Self
}

impl PaintMarker {
    #[inline]
    pub fn new() -> Self
    #[inline]
    pub fn paint(&mut self, macro_name: &str)
    #[inline]
    pub fn unpaint(&mut self, macro_name: &str)
    #[inline]
    pub fn is_painted(&self, macro_name: &str) -> bool
    #[inline]
    pub fn is_empty(&self) -> bool
    #[inline]
    pub fn active_count(&self) -> usize
    #[inline]
    pub fn check_token_paint(&self, token_text: &str) -> PaintState
}

impl fmt::Debug for PaintMarker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
}
```

---

## 5. src/frontend/preprocessor/macro_expander.rs

### MacroExpander<'a> 类型定义

```rust
pub struct MacroExpander<'a> {
    macro_defs: &'a FxHashMap<String, MacroDef>,
    paint_marker: PaintMarker,
    depth: usize,
    max_depth: usize,
    diagnostics: &'a mut DiagnosticEngine,
    counter: &'a mut u64,
    source_map: Option<&'a SourceMap>,
}
```

### MacroExpander<'a> 方法

```rust
impl<'a> MacroExpander<'a> {
    pub fn new(macro_defs: &'a FxHashMap<String, MacroDef>, diagnostics: &'a mut DiagnosticEngine, max_depth: usize, counter: &'a mut u64) -> Self
    pub fn new_with_source_map(macro_defs: &'a FxHashMap<String, MacroDef>, diagnostics: &'a mut DiagnosticEngine, max_depth: usize, counter: &'a mut u64, source_map: &'a SourceMap) -> Self
    pub fn expand_tokens(&mut self, tokens: &[PPToken]) -> Vec<PPToken>
    fn prepare_object_replacement(&mut self, macro_def: &MacroDef, invocation_span: Span) -> Vec<PPToken>
    fn perform_function_substitution(&mut self, macro_def: &MacroDef, args: Vec<Vec<PPToken>>, invocation_span: Span) -> Vec<PPToken>
    fn validate_arg_count(&mut self, macro_name: &str, is_predefined: bool, params: &[String], variadic: bool, args: &[Vec<PPToken>], span: Span) -> bool
    fn substitute_params(&mut self, replacement: &[PPToken], params: &[String], unexpanded_args: &[Vec<PPToken>], expanded_args: &[Vec<PPToken>], variadic: bool, invocation_span: Span) -> Vec<PPToken>
    fn process_paste(&mut self, tokens: &[PPToken], invocation_span: Span) -> Vec<PPToken>
    fn collect_arguments(&mut self, tokens: &[PPToken], start: usize) -> Result<(Vec<Vec<PPToken>>, usize), ()>
    fn find_param_or_va_index(&self, name: &str, params: &[String], variadic: bool) -> Option<usize>
    fn get_argument_tokens(&self, param_idx: usize, args: &[Vec<PPToken>], params: &[String], variadic: bool) -> Vec<PPToken>
    fn get_va_args_tokens(&self, args: &[Vec<PPToken>], params: &[String]) -> Vec<PPToken>
}
```

### 常量

```rust
const VA_ARGS_INDEX: usize = usize::MAX
```

### 私有函数

```rust
fn find_lparen(tokens: &[PPToken], start: usize) -> Option<usize>
fn find_named_param_index(name: &str, params: &[String]) -> Option<usize>
#[inline]
fn is_hashhash_token(tok: &PPToken) -> bool
fn followed_by_hashhash(tokens: &[PPToken], pos: usize) -> bool
fn preceded_by_hashhash(tokens: &[PPToken], pos: usize) -> bool
fn is_arg_empty(arg: &[PPToken]) -> bool
fn collect_va_opt_content(tokens: &[PPToken], start: usize) -> (Vec<PPToken>, usize)
```

---

## 6. src/frontend/preprocessor/include_handler.rs

### IncludeError 类型定义

```rust
#[derive(Debug)]
pub enum IncludeError {
    Circular(PathBuf),
    TooDeep(usize),
    NotFound(String),
    IoError(std::io::Error),
}
```

### IncludeHandler 类型定义

```rust
pub struct IncludeHandler {
    user_paths: Vec<PathBuf>,
    system_paths: Vec<PathBuf>,
    include_stack: Vec<PathBuf>,
    guarded_files: FxHashMap<PathBuf, String>,
    pragma_once_files: FxHashSet<PathBuf>,
    max_include_depth: usize,
}
```

### IncludeError 方法

```rust
impl fmt::Display for IncludeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
}

impl std::error::Error for IncludeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)>
}

impl From<std::io::Error> for IncludeError {
    fn from(err: std::io::Error) -> Self
}

impl IncludeError {
    pub fn to_diagnostic(&self, span: Span, source_map: Option<&SourceMap>) -> Diagnostic
    pub fn as_warning(&self, span: Span) -> Diagnostic
}
```

### IncludeHandler 方法

```rust
impl IncludeHandler {
    pub fn new(user_paths: Vec<PathBuf>, system_paths: Vec<PathBuf>) -> Self
    pub fn add_user_path(&mut self, path: PathBuf)
    pub fn add_system_path(&mut self, path: PathBuf)
    pub fn resolve_include(&self, header: &str, is_system: bool, including_file: &Path) -> Option<PathBuf>
    pub fn resolve_include_next(&self, header: &str, including_file: &Path) -> Option<PathBuf>
    fn search_paths(&self, paths: &[PathBuf], header: &str) -> Option<PathBuf>
    pub fn push_include(&mut self, path: &Path) -> Result<(), IncludeError>
    pub fn pop_include(&mut self)
    pub fn register_guard(&mut self, path: &Path, guard_macro: String)
    pub fn should_skip_guarded(&self, path: &Path, defined_macros: &FxHashMap<String, MacroDef>) -> bool
    pub fn mark_pragma_once(&mut self, path: &Path)
    pub fn is_pragma_once(&self, path: &Path) -> bool
    pub fn should_skip_file(&self, path: &Path, defined_macros: &FxHashMap<String, MacroDef>) -> bool
    pub fn read_include_file(&self, path: &Path) -> Result<String, IncludeError>
    pub fn read_and_register(&self, path: &Path, source_map: &mut SourceMap) -> Result<(u32, String), IncludeError>
    pub fn get_filename_from_source_map<'a>(&self, source_map: &'a SourceMap, file_id: u32) -> Option<&'a str>
    #[inline]
    pub fn depth(&self) -> usize
    #[inline]
    pub fn include_stack(&self) -> &[PathBuf]
}
```

### 私有函数

```rust
fn canonicalize_path(path: &Path) -> PathBuf
```

### 公共函数

```rust
pub fn detect_include_guard(tokens: &[PPToken]) -> Option<String>
```

---

## 7. src/frontend/preprocessor/expression.rs

### ShiftDir 类型定义（私有）

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShiftDir {
    Left,
    Right,
}
```

### PPValue 类型定义

```rust
#[derive(Debug, Clone, Copy)]
pub enum PPValue {
    Signed(i64),
    Unsigned(u64),
}
```

### ExprParser<'a> 类型定义（私有）

```rust
struct ExprParser<'a> {
    tokens: &'a [&'a PPToken],
    pos: usize,
    diagnostics: &'a mut DiagnosticEngine,
}
```

### PPValue 方法

```rust
impl PPValue {
    #[inline]
    pub fn is_nonzero(&self) -> bool
    #[inline]
    pub fn to_i64(&self) -> i64
    #[inline]
    pub fn to_u64(&self) -> u64
    #[inline]
    pub fn is_unsigned(&self) -> bool
}
```

### 公共函数

```rust
#[allow(clippy::result_unit_err)]
pub fn evaluate_pp_expression(tokens: &[PPToken], diagnostics: &mut DiagnosticEngine) -> Result<PPValue, ()>
```

### ExprParser<'a> 方法

```rust
impl<'a> ExprParser<'a> {
    fn new(tokens: &'a [&'a PPToken], diagnostics: &'a mut DiagnosticEngine) -> Self
    #[inline]
    fn at_end(&self) -> bool
    #[inline]
    fn peek_token(&self) -> &PPToken
    fn advance(&mut self) -> &PPToken
    #[inline]
    fn is_punct(&self, text: &str) -> bool
    fn eat_punct(&mut self, text: &str) -> bool
    fn current_span(&self) -> Span
    fn span_from(&self, start_span: Span) -> Span
    fn parse_ternary(&mut self) -> Result<PPValue, ()>
    fn parse_logical_or(&mut self) -> Result<PPValue, ()>
    fn parse_logical_and(&mut self) -> Result<PPValue, ()>
    fn parse_bitwise_or(&mut self) -> Result<PPValue, ()>
    fn parse_bitwise_xor(&mut self) -> Result<PPValue, ()>
    fn parse_bitwise_and(&mut self) -> Result<PPValue, ()>
    fn is_single_punct(&self, ch: &str) -> bool
    fn parse_equality(&mut self) -> Result<PPValue, ()>
    fn parse_relational(&mut self) -> Result<PPValue, ()>
    fn parse_shift(&mut self) -> Result<PPValue, ()>
    fn apply_shift(&mut self, left: PPValue, right: PPValue, dir: ShiftDir, span: Span) -> Result<PPValue, ()>
    fn parse_additive(&mut self) -> Result<PPValue, ()>
    fn parse_multiplicative(&mut self) -> Result<PPValue, ()>
    fn apply_division(&mut self, left: PPValue, right: PPValue, is_modulo: bool, span: Span) -> Result<PPValue, ()>
    fn parse_unary(&mut self) -> Result<PPValue, ()>
    fn parse_primary(&mut self) -> Result<PPValue, ()>
    fn parse_defined_operator(&mut self, _kw_span: Span) -> Result<PPValue, ()>
    fn parse_has_attribute_operator(&mut self, _kw_span: Span) -> Result<PPValue, ()>
    fn parse_has_builtin_operator(&mut self, _kw_span: Span) -> Result<PPValue, ()>
    fn parse_has_include_operator(&mut self, _kw_span: Span) -> Result<PPValue, ()>
    fn parse_has_feature_operator(&mut self, _kw_span: Span) -> Result<PPValue, ()>
}
```

### 私有函数

```rust
fn is_supported_attribute(name: &str) -> bool
fn is_supported_builtin(name: &str) -> bool
fn parse_integer_literal(text: &str, span: Span, diagnostics: &mut DiagnosticEngine) -> Result<PPValue, ()>
fn strip_integer_suffix(text: &str) -> (&str, bool)
fn parse_char_constant(text: &str, span: Span, diagnostics: &mut DiagnosticEngine) -> Result<PPValue, ()>
fn strip_char_prefix_and_quotes(text: &str) -> &str
fn parse_escape_sequence(input: &str, span: Span, diagnostics: &mut DiagnosticEngine) -> Result<Vec<u8>, ()>
#[inline]
fn is_hex_digit(b: u8) -> bool
#[inline]
fn hex_digit_value(b: u8) -> u8
fn apply_binary_bitwise<F>(left: PPValue, right: PPValue, op: F) -> PPValue
where
    F: Fn(u64, u64) -> u64,
```

---

## 8. src/frontend/preprocessor/directives.rs

### DirectiveResult 类型定义

```rust
pub enum DirectiveResult {
    Tokens(Vec<PPToken>),
    None,
    Fatal,
}
```

### 公共函数

```rust
pub fn process_directive(pp: &mut Preprocessor, directive_token: &PPToken, tokens: &[PPToken]) -> Result<DirectiveResult, ()>
pub fn resolve_defined_operators(tokens: &[PPToken], macro_defs: &FxHashMap<String, MacroDef>) -> Vec<PPToken>
pub fn verify_no_unterminated_conditionals(pp: &mut Preprocessor)
```

### 私有函数

```rust
fn is_preprocessing_active(pp: &Preprocessor) -> bool
fn skip_whitespace(tokens: &[PPToken]) -> &[PPToken]
fn tokens_to_message(tokens: &[PPToken]) -> String
fn trim_trailing_whitespace(mut tokens: Vec<PPToken>) -> Vec<PPToken>
fn warn_extra_tokens(diagnostics: &mut DiagnosticEngine, tokens: &[PPToken])
fn process_define(pp: &mut Preprocessor, tokens: &[PPToken], directive_span: Span) -> Result<(), ()>
fn parse_function_like_params<'a>(pp: &mut Preprocessor, tokens: &'a [PPToken], directive_span: Span) -> Result<(MacroKind, &'a [PPToken]), ()>
fn macro_definitions_equivalent(existing: &MacroDef, new_kind: &MacroKind, new_replacement: &[PPToken]) -> bool
fn process_undef(pp: &mut Preprocessor, tokens: &[PPToken], directive_span: Span) -> Result<(), ()>
fn process_include(pp: &mut Preprocessor, tokens: &[PPToken], directive_span: Span) -> Result<Vec<PPToken>, ()>
fn parse_include_header(pp: &mut Preprocessor, tokens: &[PPToken], directive_span: Span) -> Result<(String, bool), ()>
fn detect_and_register_guard(handler: &mut IncludeHandler, path: &Path, macro_defs: &FxHashMap<String, MacroDef>)
fn process_if(pp: &mut Preprocessor, tokens: &[PPToken], directive_span: Span) -> Result<(), ()>
fn process_ifdef(pp: &mut Preprocessor, tokens: &[PPToken], directive_span: Span) -> Result<(), ()>
fn process_ifndef(pp: &mut Preprocessor, tokens: &[PPToken], directive_span: Span) -> Result<(), ()>
fn process_elif(pp: &mut Preprocessor, tokens: &[PPToken], directive_span: Span) -> Result<DirectiveResult, ()>
fn process_else(pp: &mut Preprocessor, tokens: &[PPToken], directive_span: Span) -> Result<DirectiveResult, ()>
fn process_endif(pp: &mut Preprocessor, tokens: &[PPToken], directive_span: Span) -> Result<DirectiveResult, ()>
fn process_pragma(pp: &mut Preprocessor, tokens: &[PPToken], directive_span: Span) -> Result<(), ()>
fn process_pragma_pack(_pp: &mut Preprocessor, tokens: &[PPToken], _directive_span: Span) -> Result<(), ()>
fn extract_pragma_macro_name(tokens: &[PPToken]) -> Option<String>
fn process_pragma_push_macro(pp: &mut Preprocessor, tokens: &[PPToken]) -> Result<(), ()>
fn process_pragma_pop_macro(pp: &mut Preprocessor, tokens: &[PPToken]) -> Result<(), ()>
fn process_error(pp: &mut Preprocessor, tokens: &[PPToken], directive_span: Span)
fn process_warning(pp: &mut Preprocessor, tokens: &[PPToken], directive_span: Span)
fn process_line(pp: &mut Preprocessor, tokens: &[PPToken], directive_span: Span) -> Result<(), ()>
```