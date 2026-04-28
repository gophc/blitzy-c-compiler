# BCC Rust Common 模块类型和函数定义

> **注意**: 以下函数在 Rust 中有重载(相同函数名,不同参数),Go 实现时需要使用不同的函数名

---

## 重载函数列表 (Rust → Go 映射)

| Rust 函数 | Go 函数 | 说明 |
|----------|--------|------|
| `sizeof_ctype(ty, target)` | `SizeofCType(ty, target)` | 基本版本 |
| `sizeof_ctype_resolved(ty, target, tag_types)` | `SizeofCTypeResolved(ty, target, tagTypes)` | 解析标签类型版本 |
| `alignof_ctype(ty, target)` | `AlignofCType(ty, target)` | 基本版本 |
| `alignof_ctype_resolved(ty, target, tag_types)` | `AlignofCTypeResolved(ty, target, tagTypes)` | 解析标签类型版本 |
| `is_integer_type(ty)` (type_builder.rs) | `IsIntegerType(ty)` | 类型谓词 |
| `is_arithmetic_type(ty)` (type_builder.rs) | `IsArithmeticType(ty)` | 类型谓词 |
| `is_scalar_type(ty)` (type_builder.rs) | `IsScalarType(ty)` | 类型谓词 |
| `is_complete_type(ty)` (type_builder.rs) | `IsCompleteType(ty)` | 类型谓词 |
| `integer_rank(ty)` (types.rs) | `IntegerRank(ty)` | 整数等级 |
| `integer_rank(ty)` (type_builder.rs) | `IntegerRank(ty)` | 同一函数 |
| `integer_promotion(ty)` (types.rs) | `IntegerPromotion(ty)` | 整数提升 |
| `integer_promote(ty)` (type_builder.rs) | `IntegerPromotion(ty)` | 同一函数 |
| `compute_struct_layout(fields, ...)` (type_builder.rs) | `ComputeStructLayout(fieldTypes, ...)` | []CType 版本 |
| `compute_struct_layout_with_fields(fields, ...)` (type_builder.rs) | `ComputeStructLayoutWithFields(fields, ...)` | []StructField 版本(含位域) |

---

## types.rs

### 类型

```rust
// 类型限定符
pub struct TypeQualifiers {
    pub is_const: bool,
    pub is_volatile: bool,
    pub is_restrict: bool,
    pub is_atomic: bool,
}

// 结构体/联合体字段
pub struct StructField {
    pub name: Option<String>,
    pub ty: CType,
    pub bit_width: Option<u32>,
}

// C 语言类型
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

// 机器类型
pub enum MachineType {
    I8, I16, I32, I64, I128,
    F32, F64, F80,
    Ptr, Void,
    Integer, SSE, X87, Memory,
}
```

### 函数

```rust
// 标签类型定义
pub fn set_tag_type_defs(defs: HashMap<String, CType>)
pub fn clear_tag_type_defs()

// 大小和对齐
pub fn sizeof_ctype(ty: &CType, target: &Target) -> usize
pub fn alignof_ctype(ty: &CType, target: &Target) -> usize
pub fn sizeof_ctype_resolved(ty: &CType, target: &Target, tag_types: &HashMap<String, CType>) -> usize
pub fn alignof_ctype_resolved(ty: &CType, target: &Target, tag_types: &HashMap<String, CType>) -> usize

// 类型谓词
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

// 类型兼容性
pub fn is_compatible(a: &CType, b: &CType) -> bool
pub fn resolve_and_strip(ty: &CType) -> &CType
pub fn unqualified(ty: &CType) -> &CType
pub fn resolve_typedef(ty: &CType) -> CType
pub fn integer_promotion(ty: &CType) -> CType
pub fn integer_rank(ty: &CType) -> u8
```

---

## type_builder.rs

### 类型

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

### 函数

```rust
// TypeBuilder 方法
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

// 自由函数
pub fn is_integer_type(ty: &CType) -> bool
pub fn is_arithmetic_type(ty: &CType) -> bool
pub fn is_scalar_type(ty: &CType) -> bool
pub fn is_complete_type(ty: &CType) -> bool
pub fn integer_rank(ty: &CType) -> u8
pub fn usual_arithmetic_conversion(lhs: &CType, rhs: &CType) -> CType
pub fn integer_promote(ty: &CType) -> CType
```

---

## temp_files.rs

### 类型

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

### 函数

```rust
impl TempFile {
    pub fn new(suffix: &str) -> io::Result<Self>
    pub fn new_in(dir: &Path, suffix: &str) -> io::Result<Self>
    pub fn path(&self) -> &Path
    pub fn keep(mut self) -> PathBuf
    pub fn into_path(self) -> PathBuf
}

impl TempDir {
    pub fn new() -> io::Result<Self>
    pub fn new_in(parent: &Path) -> io::Result<Self>
    pub fn path(&self) -> &Path
    pub fn create_file(&self, name: &str) -> io::Result<TempFile>
    pub fn keep(mut self) -> PathBuf
}

pub fn create_temp_object_file() -> io::Result<TempFile>
pub fn create_temp_assembly_file() -> io::Result<TempFile>
pub fn create_temp_preprocessed_file() -> io::Result<TempFile>
```

---

## target.rs

### 类型

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

### 函数

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
```

---

## string_interner.rs

### 类型

```rust
pub struct Symbol(u32)

pub struct Interner {
    map: FxHashMap<String, Symbol>,
    strings: Vec<String>,
}
```

### 函数

```rust
impl Symbol {
    pub fn as_u32(&self) -> u32
    pub fn from_u32(raw: u32) -> Self
}

impl Interner {
    pub fn new() -> Self
    pub fn intern(&mut self, s: &str) -> Symbol
    pub fn resolve(&self, sym: Symbol) -> &str
    pub fn get(&self, s: &str) -> Option<Symbol>
    pub fn len(&self) -> usize
    pub fn is_empty(&self) -> bool
}

impl Index<Symbol> for Interner {
    type Output = str
    fn index(&self, sym: Symbol) -> &str
}
```

---

## source_map.rs

### 类型

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

### 函数

```rust
impl SourceFile {
    fn new(id: u32, filename: String, content: String) -> Self
    fn compute_line_offsets(content: &str) -> Vec<u32>
    fn lookup_line_col(&self, byte_offset: u32) -> (u32, u32)
    fn get_line_content(&self, line: u32) -> &str
    fn line_count(&self) -> usize
}

impl SourceMap {
    pub fn new() -> Self
    pub fn add_file(&mut self, filename: String, content: String) -> u32
    pub fn get_file(&self, file_id: u32) -> Option<&SourceFile>
    pub fn lookup_location(&self, file_id: u32, byte_offset: u32) -> Option<SourceLocation>
    pub fn get_filename(&self, file_id: u32) -> Option<&str>
    pub fn get_line_directive_filenames(&self, file_id: u32) -> Vec<String>
    pub fn add_line_directive(&mut self, directive: LineDirective)
    pub fn resolve_location(&self, file_id: u32, byte_offset: u32) -> SourceLocation
    pub fn format_span(&self, file_id: u32, start: u32, end: u32) -> String
}
```

---

## fx_hash.rs

### 类型

```rust
pub struct FxHasher {
    hash: usize,
}

pub type FxHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>
pub type FxHashSet<K> = HashSet<K, BuildHasherDefault<FxHasher>>
```

### 函数

```rust
impl FxHasher {
    pub fn new() -> Self
}

pub fn fx_hash_map<K, V>() -> FxHashMap<K, V>
pub fn fx_hash_map_with_capacity<K, V>(capacity: usize) -> FxHashMap<K, V>
pub fn fx_hash_set<K>() -> FxHashSet<K>
pub fn fx_hash_set_with_capacity<K>(capacity: usize) -> FxHashSet<K>
```

---

## long_double.rs

### 类型

```rust
pub struct LongDouble {
    sign: bool,
    exponent: u16,
    significand: u64,
}
```

### 常量

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

### 函数

```rust
impl LongDouble {
    pub fn is_zero(&self) -> bool
    pub fn is_infinity(&self) -> bool
    pub fn is_nan(&self) -> bool
    pub fn is_negative(&self) -> bool
    pub fn is_denormal(&self) -> bool
    
    pub fn add(self, other: LongDouble) -> LongDouble
    pub fn sub(self, other: LongDouble) -> LongDouble
    pub fn mul(self, other: LongDouble) -> LongDouble
    pub fn div(self, other: LongDouble) -> LongDouble
    pub fn neg(self) -> LongDouble
    
    pub fn from_f64(val: f64) -> LongDouble
    pub fn to_f64(&self) -> f64
    pub fn from_i64(val: i64) -> LongDouble
    pub fn to_i64(&self) -> i64
    pub fn from_u64(val: u64) -> LongDouble
    pub fn to_u64(&self) -> u64
    
    pub fn to_bytes(&self) -> [u8; 10]
    pub fn from_bytes(bytes: &[u8; 10]) -> LongDouble
    
    pub fn total_cmp(&self, other: &LongDouble) -> Ordering
}
```

---

## encoding.rs

### 函数

```rust
const MAX_SOURCE_FILE_SIZE: u64 = 256 * 1024 * 1024
const PUA_BASE: u32 = 0xE000
const PUA_LOW: u32 = 0xE080
const PUA_HIGH: u32 = 0xE0FF

pub fn encode_byte_to_pua(byte: u8) -> char
pub fn decode_pua_to_byte(ch: char) -> Option<u8>
pub fn is_pua_encoded(ch: char) -> bool
pub fn read_source_file(path: &Path) -> io::Result<String>
pub fn encode_bytes_to_string(bytes: &[u8]) -> String
pub fn decode_string_to_bytes(s: &str) -> Vec<u8>
pub fn extract_string_bytes(s: &str) -> Vec<u8>
```

---

## diagnostics.rs

### 类型

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

### 函数

```rust
impl Span {
    pub fn new(file_id: u32, start: u32, end: u32) -> Self
    pub fn dummy() -> Self
    pub fn merge(self, other: Span) -> Span
    pub fn is_dummy(&self) -> bool
}

impl Diagnostic {
    pub fn error(span: Span, message: impl Into<String>) -> Self
    pub fn warning(span: Span, message: impl Into<String>) -> Self
    pub fn note(span: Span, message: impl Into<String>) -> Self
    pub fn with_note(mut self, span: Span, message: impl Into<String>) -> Self
    pub fn with_fix(mut self, span: Span, replacement: impl Into<String>, message: impl Into<String>) -> Self
}

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
    pub fn span_filename<'a>(&self, span: &Span, source_map: &'a SourceMap) -> Option<&'a str>
}
```