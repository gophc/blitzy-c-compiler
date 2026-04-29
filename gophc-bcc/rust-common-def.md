# BCC Rust Common 模块定义

本文档按文件整理 `src/common/` 下所有类型、函数和方法签名（私有），按定义出现顺序，不包括测试。

---

## 1. src/common/types.rs

### 类型定义

```rust
pub struct TypeQualifiers {
    pub is_const: bool,
    pub is_volatile: bool,
    pub is_restrict: bool,
    pub is_atomic: bool,
}

pub struct StructField {
    pub name: Option<String>,
    pub ty: CType,
    pub bit_width: Option<u32>,
}

pub enum CType {
    Void,
    Bool,
    Char,
    SChar,
    UChar,
    Short,
    UShort,
    Int,
    UInt,
    Long,
    ULong,
    LongLong,
    ULongLong,
    Int128,
    UInt128,
    Float,
    Double,
    LongDouble,
    Complex(Box<CType>),
    Pointer(Box<CType>, TypeQualifiers),
    Array(Box<CType>, Option<usize>),
    Function { return_type: Box<CType>, params: Vec<CType>, variadic: bool },
    Struct { name: Option<String>, fields: Vec<StructField>, packed: bool, aligned: Option<usize> },
    Union { name: Option<String>, fields: Vec<StructField>, packed: bool, aligned: Option<usize> },
    Enum { name: Option<String>, underlying_type: Box<CType> },
    Atomic(Box<CType>),
    Typedef { name: String, underlying: Box<CType> },
    Qualified(Box<CType>, TypeQualifiers),
}

pub enum MachineType {
    I8,
    I16,
    I32,
    I64,
    I128,
    F32,
    F64,
    F80,
    Ptr,
    Void,
    Integer,
    SSE,
    X87,
    Memory,
}
```

### TypeQualifiers 方法

```rust
impl TypeQualifiers {
    pub fn is_empty(&self) -> bool
    pub fn merge(self, other: TypeQualifiers) -> TypeQualifiers
}
```

### 公共函数

```rust
pub fn set_tag_type_defs(defs: HashMap<String, CType>)
pub fn clear_tag_type_defs()
pub fn sizeof_ctype(ty: &CType, target: &Target) -> usize
pub fn alignof_ctype(ty: &CType, target: &Target) -> usize
pub fn sizeof_ctype_resolved(ty: &CType, target: &Target, tag_types: &HashMap<String, CType>) -> usize
pub fn alignof_ctype_resolved(ty: &CType, target: &Target, tag_types: &HashMap<String, CType>) -> usize
pub fn is_void(ty: &CType) -> bool
pub fn is_integer(ty: &CType) -> bool
pub fn is_unsigned(ty: &CType) -> bool
pub fn is_signed(ty: &CType) -> bool
pub fn is_floating(ty: &CType) -> bool
pub fn is_arithmetic(ty: &CType) -> bool
pub fn is_scalar(ty: &CType) -> bool
pub fn is_pointer(ty: &CType) -> bool
pub fn is_array(ty: &CType) -> bool
pub fn is_function(ty: &CType) -> bool
pub fn is_struct_or_union(ty: &CType) -> bool
pub fn is_complete(ty: &CType) -> bool
pub fn is_compatible(a: &CType, b: &CType) -> bool
pub fn resolve_and_strip(ty: &CType) -> &CType
pub fn unqualified(ty: &CType) -> &CType
pub fn resolve_typedef(ty: &CType) -> &CType
pub fn integer_promotion(ty: &CType) -> CType
pub fn integer_rank(ty: &CType) -> u8
```

### 私有函数

```rust
fn resolve_tag_ref(tag: &str) -> Option<CType>
fn compute_struct_size(fields: &[StructField], packed: bool, aligned: Option<usize>, target: &Target) -> usize
fn compute_union_size(fields: &[StructField], packed: bool, aligned: Option<usize>, target: &Target) -> usize
fn align_up(value: usize, align: usize) -> usize
fn compute_struct_or_union_align(fields: &[StructField], packed: bool, aligned: Option<usize>, target: &Target) -> usize
fn is_compatible_inner(a: &CType, b: &CType) -> bool
fn compute_struct_size_resolved(fields: &[StructField], packed: bool, aligned: Option<usize>, target: &Target, tag_types: &HashMap<String, CType>) -> usize
fn compute_union_size_resolved(fields: &[StructField], packed: bool, aligned: Option<usize>, target: &Target, tag_types: &HashMap<String, CType>) -> usize
fn compute_struct_or_union_align_resolved(fields: &[StructField], packed: bool, aligned: Option<usize>, target: &Target, tag_types: &HashMap<String, CType>) -> usize
```

---

## 2. src/common/type_builder.rs

### 类型定义

```rust
pub struct FieldLayout {
    pub offset: usize,
    pub size: usize,
    pub alignment: usize,
    pub bitfield_info: Option<(usize, usize)>,
}

pub struct StructLayout {
    pub fields: Vec<FieldLayout>,
    pub size: usize,
    pub alignment: usize,
    pub has_flexible_array: bool,
}

pub struct TypeBuilder {
    target: Target,
}
```

### TypeBuilder 方法

```rust
impl TypeBuilder {
    pub fn new(target: Target) -> Self
    pub fn target(&self) -> &Target
    pub fn pointer_to(&self, pointee: CType) -> CType
    pub fn array_of(&self, element: CType, size: Option<usize>) -> CType
    pub fn function_type(&self, return_type: CType, params: Vec<CType>, variadic: bool) -> CType
    pub fn const_qualified(&self, ty: CType) -> CType
    pub fn volatile_qualified(&self, ty: CType) -> CType
    pub fn atomic_type(&self, ty: CType) -> CType
    pub fn compute_struct_layout(&self, fields: &[CType], packed: bool, explicit_align: Option<usize>) -> StructLayout
    pub fn compute_struct_layout_with_fields(&self, fields: &[StructField], packed: bool, explicit_align: Option<usize>) -> StructLayout
    pub fn compute_union_layout(&self, fields: &[CType], packed: bool, explicit_align: Option<usize>) -> StructLayout
    pub fn sizeof_type(&self, ty: &CType) -> usize
    pub fn alignof_type(&self, ty: &CType) -> usize
}
```

### 私有函数

```rust
fn align_up(value: usize, align: usize) -> usize
fn compute_aggregate_alignment(max_natural_field_align: usize, packed: bool, explicit_align: Option<usize>) -> usize
fn resolve_and_strip(ty: &CType) -> &CType
fn integer_bit_width(ty: &CType) -> u32
fn types_equal_for_conversion(a: &CType, b: &CType) -> bool
fn is_unsigned_type(ty: &CType) -> bool
fn to_unsigned(ty: &CType) -> CType
fn is_integer_type_inner(ty: &CType) -> bool
```

### 公共函数

```rust
pub fn is_integer_type(ty: &CType) -> bool
pub fn is_arithmetic_type(ty: &CType) -> bool
pub fn is_scalar_type(ty: &CType) -> bool
pub fn is_complete_type(ty: &CType) -> bool
pub fn integer_rank(ty: &CType) -> u8
pub fn usual_arithmetic_conversion(lhs: &CType, rhs: &CType) -> CType
pub fn integer_promote(ty: &CType) -> CType
```

---

## 3. src/common/temp_files.rs

### 类型定义

```rust
pub struct TempFile {
    path: PathBuf,
    delete_on_drop: bool,
}

pub struct TempDir {
    path: PathBuf,
    delete_on_drop: bool,
}
```

### TempFile 方法

```rust
impl TempFile {
    pub fn new(suffix: &str) -> io::Result<Self>
    pub fn new_in(dir: &Path, suffix: &str) -> io::Result<Self>
    pub fn path(&self) -> &Path
    pub fn keep(mut self) -> PathBuf
    pub fn into_path(self) -> PathBuf
}

impl Drop for TempFile {
    fn drop(&mut self)
}
```

### TempDir 方法

```rust
impl TempDir {
    pub fn new() -> io::Result<Self>
    pub fn new_in(parent: &Path) -> io::Result<Self>
    pub fn path(&self) -> &Path
    pub fn create_file(&self, name: &str) -> io::Result<TempFile>
    pub fn keep(mut self) -> PathBuf
}

impl Drop for TempDir {
    fn drop(&mut self)
}
```

### 公共函数

```rust
fn unique_name(prefix: &str, suffix: &str) -> String
pub fn create_temp_object_file() -> io::Result<TempFile>
pub fn create_temp_assembly_file() -> io::Result<TempFile>
pub fn create_temp_preprocessed_file() -> io::Result<TempFile>
```

---

## 4. src/common/target.rs

### 类型定义

```rust
pub enum Endianness {
    Little,
    Big,
}

pub enum Target {
    X86_64,
    I686,
    AArch64,
    RiscV64,
}
```

### Endianness 方法

```rust
impl fmt::Display for Endianness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
}
```

### Target 方法

```rust
impl Target {
    pub fn from_str(s: &str) -> Option<Target>
    pub fn pointer_width(&self) -> usize
    pub fn pointer_align(&self) -> usize
    pub fn is_64bit(&self) -> bool
    pub fn endianness(&self) -> Endianness
    pub fn long_size(&self) -> usize
    pub fn long_double_size(&self) -> usize
    pub fn long_double_align(&self) -> usize
    pub fn elf_machine(&self) -> u16
    pub fn elf_class(&self) -> u8
    pub fn elf_data(&self) -> u8
    pub fn elf_flags(&self) -> u32
    pub fn predefined_macros(&self) -> Vec<(&'static str, &'static str)>
    pub fn max_align(&self) -> usize
    pub fn page_size(&self) -> usize
    pub fn dynamic_linker(&self) -> &'static str
    pub fn default_entry_point(&self) -> &'static str
    pub fn system_include_paths(&self) -> Vec<&'static str>
    pub fn system_library_paths(&self) -> Vec<&'static str>
}

impl fmt::Display for Target {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
}
```

---

## 5. src/common/string_interner.rs

### 类型定义

```rust
pub struct Symbol(u32)
```

### Symbol 方法

```rust
impl Symbol {
    pub fn as_u32(&self) -> u32
    pub fn from_u32(raw: u32) -> Self
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
}
```

### 类型定义

```rust
pub struct Interner {
    map: FxHashMap<String, Symbol>,
    strings: Vec<String>,
}
```

### Interner 方法

```rust
impl Interner {
    pub fn new() -> Self
    pub fn intern(&mut self, s: &str) -> Symbol
    pub fn resolve(&self, sym: Symbol) -> &str
    pub fn get(&self, s: &str) -> Option<Symbol>
    pub fn len(&self) -> usize
    pub fn is_empty(&self) -> bool
}

impl Default for Interner {
    fn default() -> Self
}

impl Index<Symbol> for Interner {
    type Output = str
    fn index(&self, sym: Symbol) -> &str
}

impl fmt::Debug for Interner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
}
```

---

## 6. src/common/source_map.rs

### 类型定义

```rust
pub struct SourceLocation {
    pub file_id: u32,
    pub filename: String,
    pub line: u32,
    pub column: u32,
}

pub struct LineDirective {
    pub file_id: u32,
    pub directive_offset: u32,
    pub new_line: u32,
    pub new_filename: Option<String>,
}

pub struct SourceFile {
    pub id: u32,
    pub filename: String,
    pub content: String,
    line_offsets: Vec<u32>,
}

pub struct SourceMap {
    files: Vec<SourceFile>,
    line_directives: Vec<Vec<LineDirective>>,
}
```

### SourceLocation 方法

```rust
impl fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
}
```

### SourceFile 方法

```rust
impl SourceFile {
    fn new(id: u32, filename: String, content: String) -> Self
    fn compute_line_offsets(content: &str) -> Vec<u32>
    pub fn lookup_line_col(&self, byte_offset: u32) -> (u32, u32)
    pub fn get_line_content(&self, line: u32) -> &str
    pub fn line_count(&self) -> usize
}
```

### SourceMap 方法

```rust
impl SourceMap {
    pub fn new() -> Self
    pub fn add_file(&mut self, filename: String, content: String) -> u32
    pub fn get_file(&self, file_id: u32) -> Option<&SourceFile>
    pub fn lookup_location(&self, file_id: u32, byte_offset: u32) -> Option<SourceLocation>
    pub fn get_filename(&self, file_id: u32) -> Option<&str>
    pub fn get_line_directive_filenames(&self, file_id: u32) -> Vec<String>
    pub fn add_line_directive(&mut self, directive: LineDirective)
    pub fn resolve_location(&self, file_id: u32, byte_offset: u32) -> SourceLocation
    pub fn format_span(&self, file_id: u32, start: u32, _end: u32) -> String
}

impl Default for SourceMap {
    fn default() -> Self
}
```

---

## 7. src/common/long_double.rs

### 类型定义

```rust
pub struct LongDouble {
    sign: bool,
    exponent: u16,
    significand: u64,
}
```

### LongDouble 常量

```rust
impl LongDouble {
    pub const ZERO: LongDouble
    pub const NEG_ZERO: LongDouble
    pub const ONE: LongDouble
    pub const INFINITY: LongDouble
    pub const NEG_INFINITY: LongDouble
    pub const NAN: LongDouble
}
```

### LongDouble 方法

```rust
impl LongDouble {
    pub fn is_zero(&self) -> bool
    pub fn is_infinity(&self) -> bool
    pub fn is_nan(&self) -> bool
    pub fn is_negative(&self) -> bool
    pub fn is_denormal(&self) -> bool
    fn is_normal(&self) -> bool
    fn true_exponent(&self) -> i32
    fn abs_greater_than(&self, other: &LongDouble) -> bool
    fn normalize_round(sign: bool, exp: i32, sig: u128) -> LongDouble
    fn round_and_pack(sign: bool, exp: i32, sig_hi: u64, sig_lo: u64) -> LongDouble
    pub fn add(self, other: LongDouble) -> LongDouble
    pub fn sub(self, other: LongDouble) -> LongDouble
    pub fn mul(self, other: LongDouble) -> LongDouble
    pub fn div(self, other: LongDouble) -> LongDouble
    pub fn neg(self) -> LongDouble
    pub fn total_cmp(&self, other: &LongDouble) -> Ordering
    pub fn from_f64(val: f64) -> LongDouble
    pub fn to_f64(&self) -> f64
    pub fn from_i64(val: i64) -> LongDouble
    pub fn from_u64(val: u64) -> LongDouble
    pub fn to_i64(&self) -> i64
    pub fn to_u64(&self) -> u64
    pub fn to_bytes(&self) -> [u8; 10]
    pub fn from_bytes(bytes: &[u8; 10]) -> LongDouble
}

impl PartialEq for LongDouble {
    fn eq(&self, other: &Self) -> bool
}

impl PartialOrd for LongDouble {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering>
}

impl fmt::Display for LongDouble {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
}

impl fmt::Debug for LongDouble {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
}

impl Default for LongDouble {
    fn default() -> Self
}

impl std::ops::Add for LongDouble {
    type Output = LongDouble
    fn add(self, rhs: LongDouble) -> LongDouble
}

impl std::ops::Sub for LongDouble {
    type Output = LongDouble
    fn sub(self, rhs: LongDouble) -> LongDouble
}

impl std::ops::Mul for LongDouble {
    type Output = LongDouble
    fn mul(self, rhs: LongDouble) -> LongDouble
}

impl std::ops::Div for LongDouble {
    type Output = LongDouble
    fn div(self, rhs: LongDouble) -> LongDouble
}

impl std::ops::Neg for LongDouble {
    type Output = LongDouble
    fn neg(self) -> LongDouble
}
```

---

## 8. src/common/fx_hash.rs

### 类型定义

```rust
pub struct FxHasher {
    hash: usize,
}
```

### 类型别名

```rust
pub type FxHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>
pub type FxHashSet<K> = HashSet<K, BuildHasherDefault<FxHasher>>
```

### FxHasher 方法

```rust
impl FxHasher {
    pub fn new() -> Self
    fn add_to_hash(&mut self, word: usize)
}

impl Default for FxHasher {
    fn default() -> Self
}

impl Clone for FxHasher {
    fn clone(&self) -> Self
}

impl Hasher for FxHasher {
    fn finish(&self) -> u64
    fn write(&mut self, bytes: &[u8])
    fn write_u8(&mut self, i: u8)
    fn write_u16(&mut self, i: u16)
    fn write_u32(&mut self, i: u32)
    fn write_u64(&mut self, i: u64)
    fn write_usize(&mut self, i: usize)
}
```

### 公共函数

```rust
pub fn fx_hash_map<K, V>() -> FxHashMap<K, V>
pub fn fx_hash_map_with_capacity<K, V>(capacity: usize) -> FxHashMap<K, V>
pub fn fx_hash_set<K>() -> FxHashSet<K>
pub fn fx_hash_set_with_capacity<K>(capacity: usize) -> FxHashSet<K>
```

---

## 9. src/common/encoding.rs

### 公共函数

```rust
pub fn encode_byte_to_pua(byte: u8) -> char
pub fn decode_pua_to_byte(ch: char) -> Option<u8>
pub fn is_pua_encoded(ch: char) -> bool
pub fn read_source_file(path: &Path) -> io::Result<String>
pub fn encode_bytes_to_string(bytes: &[u8]) -> String
pub fn decode_string_to_bytes(s: &str) -> Vec<u8>
pub fn extract_string_bytes(s: &str) -> Vec<u8>
```

---

## 10. src/common/diagnostics.rs

### 类型定义

```rust
pub enum Severity {
    Error,
    Warning,
    Note,
}

pub struct Span {
    pub file_id: u32,
    pub start: u32,
    pub end: u32,
}

pub struct SubDiagnostic {
    pub span: Span,
    pub message: String,
}

pub struct FixSuggestion {
    pub span: Span,
    pub replacement: String,
    pub message: String,
}

pub struct Diagnostic {
    pub severity: Severity,
    pub span: Span,
    pub message: String,
    pub notes: Vec<SubDiagnostic>,
    pub fix_suggestion: Option<FixSuggestion>,
}

pub struct DiagnosticEngine {
    diagnostics: Vec<Diagnostic>,
    error_count: usize,
    warning_count: usize,
    suppress_depth: usize,
}
```

### Severity 方法

```rust
impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
}
```

### Span 方法

```rust
impl Span {
    pub fn new(file_id: u32, start: u32, end: u32) -> Self
    pub fn dummy() -> Self
    pub fn merge(self, other: Span) -> Span
    pub fn is_dummy(&self) -> bool
}
```

### Diagnostic 方法

```rust
impl Diagnostic {
    pub fn error(span: Span, message: impl Into<String>) -> Self
    pub fn warning(span: Span, message: impl Into<String>) -> Self
    pub fn note(span: Span, message: impl Into<String>) -> Self
    pub fn with_note(mut self, span: Span, message: impl Into<String>) -> Self
    pub fn with_fix(mut self, span: Span, replacement: impl Into<String>, message: impl Into<String>) -> Self
}
```

### DiagnosticEngine 方法

```rust
impl DiagnosticEngine {
    pub fn new() -> Self
    pub fn begin_suppress(&mut self)
    pub fn end_suppress(&mut self)
    pub fn emit(&mut self, diag: Diagnostic)
    pub fn emit_error(&mut self, span: Span, msg: impl Into<String>)
    pub fn emit_warning(&mut self, span: Span, msg: impl Into<String>)
    pub fn emit_note(&mut self, span: Span, msg: impl Into<String>)
    pub fn error_count(&self) -> usize
    pub fn warning_count(&self) -> usize
    pub fn has_errors(&self) -> bool
    pub fn diagnostics(&self) -> &[Diagnostic]
    pub fn clear(&mut self)
    pub fn print_all(&self, source_map: &SourceMap)
    fn print_diagnostic(&self, diag: &Diagnostic, source_map: &SourceMap)
    pub fn span_filename<'a>(&self, span: &Span, source_map: &'a SourceMap) -> Option<&'a str>
}

impl Default for DiagnosticEngine {
    fn default() -> Self
}

impl fmt::Debug for DiagnosticEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
}
```

---

## 11. src/common/mod.rs

模块入口文件，仅包含模块声明，无类型或函数定义。

---

## 十二、src/common/long_double.rs (续)

### LongDouble 内部方法

```rust
impl LongDouble {
    // 内部辅助方法
    fn true_exponent(&self) -> i32
    fn abs_greater_than(&self, other: &LongDouble) -> bool
    fn normalize_round(sign: bool, exp: i32, sig: u128) -> LongDouble
    fn round_and_pack(sign: bool, exp: i32, sig_hi: u64, sig_lo: u64) -> LongDouble
}
```

---

## 十三、src/common/type_builder.rs (续)

### 类型谓词函数

```rust
pub fn is_integer_type(ty: &CType) -> bool
pub fn is_arithmetic_type(ty: &CType) -> bool
pub fn is_scalar_type(ty: &CType) -> bool
pub fn is_complete_type(ty: &CType) -> bool
pub fn integer_rank(ty: &CType) -> u8
pub fn usual_arithmetic_conversion(lhs: &CType, rhs: &CType) -> CType
pub fn integer_promote(ty: &CType) -> CType

// 内部辅助函数
fn integer_bit_width(ty: &CType) -> u32
fn types_equal_for_conversion(a: &CType, b: &CType) -> bool
fn is_unsigned_type(ty: &CType) -> bool
fn to_unsigned(ty: &CType) -> CType
fn is_integer_type_inner(ty: &CType) -> bool
```