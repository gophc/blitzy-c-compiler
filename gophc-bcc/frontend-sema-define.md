# BCC Frontend Sema 模块的函数和类型签名

本文件记录了 Rust 实现 `src/frontend/sema` 中的所有公开函数签名和类型签名，用于与 Go 实现 `gophc-bcc/frontend/sema` 对比。

## 1. mod.rs (SemanticAnalyzer)

### 类型

```rust
pub struct SemanticAnalyzer<'a> {
    diagnostics: &'a mut DiagnosticEngine,
    type_builder: &'a TypeBuilder,
    target: Target,
    interner: &'a Interner,
    scopes: ScopeStack,
    symbols: SymbolTable,
    recursion_depth: u32,
    max_recursion_depth: u32,
    current_function_return_type: Option<CType>,
    in_loop: bool,
    in_switch: bool,
    switch_case_values: FxHashMap<i128, Span>,
    switch_has_default: bool,
    enum_constant_values: FxHashMap<String, i128>,
    tag_types_by_name: std::collections::HashMap<String, CType>,
}

enum OffsetofStep {
    Field(String),
    Index(usize),
}
```

### 函数

```rust
impl<'a> SemanticAnalyzer<'a> {
    pub fn new(
        diagnostics: &'a mut DiagnosticEngine,
        type_builder: &'a TypeBuilder,
        target: Target,
        interner: &'a Interner,
    ) -> Self
    
    pub fn analyze(&mut self, translation_unit: &mut TranslationUnit) -> Result<(), ()>
    
    pub fn analyze_external_declaration(
        &mut self,
        decl: &mut ExternalDeclaration,
    ) -> Result<(), ()>
    
    pub fn analyze_function_definition(&mut self, func: &mut FunctionDefinition) -> Result<(), ()>
    
    pub fn analyze_declaration(&mut self, decl: &mut Declaration) -> Result<(), ()>
    
    pub fn analyze_expression(&mut self, expr: &mut Expression) -> Result<CType, ()>
    
    fn analyze_expression_inner(&mut self, expr: &mut Expression) -> Result<CType, ()>
    
    fn is_const_qualified(&self, ty: &CType) -> bool
    
    fn count_initializer_elements(...) -> usize
    
    fn try_eval_index_expr(...) -> Option<i128>
    
    fn complete_array_type(ty: &CType, count: usize) -> CType
    
    fn register_builtin_typedef(&mut self, name: &str, ty: CType)
}
```

### 导出

```rust
pub use attribute_handler::{
    AttributeContext, AttributeHandler, SymbolVisibility, ValidatedAttribute,
};
pub use builtin_eval::BuiltinEvaluator;
pub use constant_eval::{ConstValue, ConstantEvaluator};
pub use initializer::{AnalyzedInit, InitializerAnalyzer};
pub use scope::{LabelEntry, ScopeKind, ScopeStack, TagEntry, TagKind};
pub use symbol_table::{Linkage, StorageClass, SymbolEntry, SymbolId, SymbolKind, SymbolTable};
pub use type_checker::TypeChecker;
```

---

## 2. type_checker.rs

### 类型

```rust
pub struct TypeChecker<'a> {
    diagnostics: &'a mut DiagnosticEngine,
    type_builder: &'a TypeBuilder,
    target: Target,
}

const EMPTY_QUALS: TypeQualifiers = TypeQualifiers { is_const: false, is_volatile: false, is_restrict: false, is_atomic: false };
const CONST_QUALS: TypeQualifiers = TypeQualifiers { is_const: true, is_volatile: false, is_restrict: false, is_atomic: false };
```

### 函数

```rust
impl<'a> TypeChecker<'a> {
    pub fn new(
        diagnostics: &'a mut DiagnosticEngine,
        type_builder: &'a TypeBuilder,
        target: Target,
    ) -> Self

    fn size_t_type(&self) -> CType
    fn ptrdiff_t_type(&self) -> CType
    fn strip_type(ty: &CType) -> CType
    fn decay_type(ty: &CType) -> CType
    fn get_qualifiers(ty: &CType) -> TypeQualifiers
    fn integer_literal_type(&self, value: u128, suffix: &IntegerSuffix, is_hex_or_octal: bool) -> CType
    fn float_literal_type(&self, suffix: &FloatSuffix) -> CType
    fn string_literal_type(&self, prefix: &StringPrefix) -> CType
    
    pub fn integer_promotion(&self, ty: &CType) -> CType
    pub fn usual_arithmetic_conversion(&self, lhs: &CType, rhs: &CType) -> CType
    pub fn are_types_compatible(&self, a: &CType, b: &CType) -> bool
    pub fn is_implicitly_convertible(&self, from: &CType, to: &CType) -> bool
    
    pub fn check_expression_type(&mut self, expr: &Expression) -> Result<CType, ()>
    pub fn check_binary_op(&mut self, op: &BinaryOp, lhs_ty: &CType, rhs_ty: &CType, span: Span) -> Result<CType, ()>
    pub fn check_unary_op(&mut self, op: &UnaryOp, operand_ty: &CType, span: Span) -> Result<CType, ()>
    pub fn check_function_call(&mut self, callee_type: &CType, args: &[CType], span: Span) -> Result<CType, ()>
    pub fn check_member_access(&mut self, struct_type: &CType, member_name: &str, is_arrow: bool, span: Span) -> Result<CType, ()>
    fn find_member_in_fields(fields: &[StructField], member_name: &str) -> Option<CType>
    fn assign_op_to_binary_op(op: &AssignOp) -> BinaryOp
    fn builtin_result_type(&mut self, builtin: &BuiltinKind, _args: &[Expression], _span: Span) -> Result<CType, ()>
    fn check_pointer_arithmetic(...) -> Result<CType, ()>
    fn check_array_subscript(...) -> Result<CType, ()>
    fn check_conditional(...) -> Result<CType, ()>
    fn check_assignment(...) -> Result<(), ()>
    fn default_argument_promotion(ty: &CType) -> CType
}
```

---

## 3. symbol_table.rs

### 类型

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SymbolId(pub u32);

impl SymbolId {
    #[inline]
    pub fn as_u32(&self) -> u32
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolKind {
    Variable,
    Function,
    TypedefName,
    EnumConstant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Linkage {
    External,
    Internal,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageClass {
    Auto,
    Register,
    Static,
    Extern,
    Typedef,
    ThreadLocal,
}

#[derive(Debug, Clone)]
pub struct SymbolEntry {
    pub name: Symbol,
    pub ty: CType,
    pub kind: SymbolKind,
    pub linkage: Linkage,
    pub storage_class: StorageClass,
    pub is_defined: bool,
    pub is_tentative: bool,
    pub span: Span,
    pub attributes: Vec<ValidatedAttribute>,
    pub is_weak: bool,
    pub visibility: Option<SymbolVisibility>,
    pub section: Option<String>,
    pub is_used: bool,
    pub scope_depth: u32,
}

pub struct SymbolTable {
    symbols: Vec<SymbolEntry>,
    name_to_ids: FxHashMap<Symbol, Vec<SymbolId>>,
    current_scope_depth: u32,
}
```

### 函数

```rust
impl SymbolTable {
    pub fn new() -> Self
    
    pub fn declare(&mut self, mut entry: SymbolEntry, diagnostics: &mut DiagnosticEngine) -> Result<SymbolId, ()>
    pub fn define(&mut self, id: SymbolId)
    pub fn lookup(&self, name: Symbol) -> Option<&SymbolEntry>
    pub fn lookup_in_current_scope(&self, name: Symbol) -> Option<&SymbolEntry>
    pub fn lookup_mut(&mut self, name: Symbol) -> Option<&mut SymbolEntry>
    pub fn get(&self, id: SymbolId) -> &SymbolEntry
    pub fn get_mut(&mut self, id: SymbolId) -> &mut SymbolEntry
    pub fn enter_scope(&mut self)
    pub fn leave_scope(&mut self)
    pub fn resolve_linkage(&self, name: Symbol, storage: StorageClass, scope_depth: u32) -> Linkage
    pub fn check_linkage_compatibility(&self, existing: &SymbolEntry, new_entry: &SymbolEntry, diagnostics: &mut DiagnosticEngine) -> Result<(), ()>
    pub fn finalize_tentative_definitions(&mut self, _diagnostics: &mut DiagnosticEngine)
    pub fn mark_weak(&mut self, id: SymbolId)
    pub fn mark_used(&mut self, name: Symbol)
    pub fn check_unused_symbols(&self, diagnostics: &mut DiagnosticEngine, interner: &Interner)
    pub fn declare_enum_constant(&mut self, name: Symbol, _value: i128, ty: CType, span: Span) -> Result<SymbolId, ()>
    pub fn declare_function(&mut self, name: Symbol, ty: CType, storage: StorageClass, span: Span, diagnostics: &mut DiagnosticEngine) -> Result<SymbolId, ()>
    
    fn find_in_current_scope(&self, name: Symbol) -> Option<SymbolId>
    fn types_are_compatible(&self, a: &CType, b: &CType) -> bool
    fn merge_attributes(entry: &mut SymbolEntry, new_attrs: &[ValidatedAttribute])
}

impl Default for SymbolTable {
    fn default() -> Self { Self::new() }
}
```

---

## 4. scope.rs

### 类型

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeKind {
    Global,
    File,
    Function,
    Block,
    FunctionPrototype,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TagKind {
    Struct,
    Union,
    Enum,
}

#[derive(Debug, Clone)]
pub struct TagEntry {
    pub kind: TagKind,
    pub ty: CType,
    pub is_complete: bool,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct LabelEntry {
    pub defined_at: Option<Span>,
    pub referenced_at: Vec<Span>,
    pub is_local: bool,
}

pub struct Scope {
    pub kind: ScopeKind,
    pub depth: u32,
    ordinary: FxHashMap<Symbol, SymbolId>,
    tags: FxHashMap<Symbol, TagEntry>,
    labels: FxHashMap<Symbol, LabelEntry>,
    typedefs: FxHashMap<Symbol, ()>,
    pub is_loop_or_switch: bool,
}

impl Scope {
    fn new(kind: ScopeKind, depth: u32) -> Self
    #[inline]
    pub fn mark_loop_or_switch(&mut self)
}

pub struct ScopeStack {
    scopes: Vec<Scope>,
    depth: u32,
}

impl Default for ScopeStack {
    fn default() -> Self { ... }
}

impl ScopeStack {
    pub fn new() -> Self
    pub fn push_scope(&mut self, kind: ScopeKind)
    pub fn pop_scope(&mut self, diagnostics: &mut DiagnosticEngine) -> Scope
    pub fn current_scope(&self) -> &Scope
    pub fn current_scope_mut(&mut self) -> &mut Scope
    pub fn current_depth(&self) -> u32
    pub fn current_kind(&self) -> ScopeKind
    pub fn scope_depth(&self) -> usize
    pub fn debug_dump(&self, interner: &Interner)
    pub fn lookup_ordinary(&self, name: Symbol) -> Option<SymbolId>
    pub fn lookup_ordinary_in_current_scope(&self, name: Symbol) -> Option<SymbolId>
    pub fn declare_ordinary(&mut self, name: Symbol, id: SymbolId) -> Option<SymbolId>
    pub fn lookup_tag(&self, name: Symbol) -> Option<&TagEntry>
    pub fn lookup_tag_by_str(&self, name: &str, interner: &Interner) -> Option<&TagEntry>
    pub fn lookup_tag_in_current_scope(&self, name: Symbol) -> Option<&TagEntry>
    pub fn declare_tag(&mut self, name: Symbol, entry: TagEntry) -> Option<TagEntry>
    pub fn complete_tag(&mut self, name: Symbol, ty: CType)
    pub fn all_tags(&self) -> Vec<(Symbol, TagEntry)>
    pub fn all_ordinary_symbols(&self) -> Vec<(Symbol, SymbolId)>
    pub fn declare_label(&mut self, name: Symbol, span: Span, is_local: bool, diagnostics: &mut DiagnosticEngine)
    pub fn define_label(&mut self, name: Symbol, span: Span, diagnostics: &mut DiagnosticEngine)
    pub fn reference_label(&mut self, name: Symbol, span: Span)
    pub fn validate_labels(&self, diagnostics: &mut DiagnosticEngine)
    pub fn is_file_scope(&self) -> bool
    pub fn is_function_scope(&self) -> bool
    pub fn find_enclosing_function(&self) -> Option<&Scope>
    pub fn find_enclosing_loop_or_switch(&self) -> bool
    pub fn is_typedef_name(&self, name: Symbol) -> bool
    pub fn register_typedef(&mut self, name: Symbol)
    
    fn find_function_scope_index(&self) -> Option<usize>
    fn find_label_scope_index(&self, name: Symbol) -> usize
    fn validate_labels_in_scope(&self, scope: &Scope, diagnostics: &mut DiagnosticEngine)
    fn validate_local_labels_in_scope(&self, scope: &Scope, diagnostics: &mut DiagnosticEngine)
}
```

---

## 5. builtin_eval.rs

### 类型

```rust
#[derive(Debug, Clone)]
pub enum BuiltinResult {
    ConstantInt(i128),
    ConstantBool(bool),
    ConstantFloat(f64),
    ResolvedType(CType),
    RuntimeCall { builtin: BuiltinKind, result_type: CType },
    NoValue,
}

pub struct BuiltinEvaluator<'a> {
    diagnostics: &'a mut DiagnosticEngine,
    type_builder: &'a TypeBuilder,
    target: Target,
}
```

### 函数

```rust
impl<'a> BuiltinEvaluator<'a> {
    pub fn new(
        diagnostics: &'a mut DiagnosticEngine,
        type_builder: &'a TypeBuilder,
        target: Target,
    ) -> Self

    pub fn evaluate_builtin(
        &mut self,
        builtin: &BuiltinKind,
        args: &[Expression],
        span: Span,
    ) -> Result<BuiltinResult, ()>
    
    pub fn is_builtin_name(name: &str) -> bool
    pub fn get_builtin_return_type(&self, builtin: &BuiltinKind) -> CType
    pub fn builtin_return_type_by_name(name: &str, target: Target) -> CType
    
    // 私有评估方法
    fn eval_constant_p(&mut self, args: &[Expression], span: Span) -> Result<BuiltinResult, ()>
    fn eval_types_compatible_p(&mut self, args: &[Expression], span: Span) -> Result<BuiltinResult, ()>
    fn eval_choose_expr(&mut self, args: &[Expression], span: Span) -> Result<BuiltinResult, ()>
    fn eval_offsetof(&mut self, args: &[Expression], span: Span) -> Result<BuiltinResult, ()>
    fn eval_single_arg_int(&mut self, name: &str, args: &[Expression], result_type: CType, span: Span) -> Result<BuiltinResult, ()>
    fn eval_expect(&mut self, args: &[Expression], span: Span) -> Result<BuiltinResult, ()>
    fn eval_unreachable(&mut self, args: &[Expression], span: Span) -> Result<BuiltinResult, ()>
    fn eval_prefetch(&mut self, args: &[Expression], span: Span) -> Result<BuiltinResult, ()>
    fn eval_va_start(&mut self, args: &[Expression], span: Span) -> Result<BuiltinResult, ()>
    fn eval_va_end(&mut self, args: &[Expression], span: Span) -> Result<BuiltinResult, ()>
    fn eval_va_arg(&mut self, args: &[Expression], span: Span) -> Result<BuiltinResult, ()>
    fn eval_va_copy(&mut self, args: &[Expression], span: Span) -> Result<BuiltinResult, ()>
    fn eval_frame_address(&mut self, args: &[Expression], span: Span) -> Result<BuiltinResult, ()>
    fn eval_return_address(&mut self, args: &[Expression], span: Span) -> Result<BuiltinResult, ()>
    fn eval_trap(&mut self, args: &[Expression], span: Span) -> Result<BuiltinResult, ()>
    fn eval_assume_aligned(&mut self, args: &[Expression], span: Span) -> Result<BuiltinResult, ()>
    fn eval_overflow_arith(&mut self, name: &str, args: &[Expression], span: Span) -> Result<BuiltinResult, ()>
    fn validate_arg_count(&mut self, builtin_name: &str, args: &[Expression], expected: usize, span: Span) -> Result<(), ()>
    fn validate_arg_type(&mut self, builtin_name: &str, arg_type: &CType, expected: &CType, arg_index: usize, span: Span) -> Result<(), ()>
    fn size_t_type(&self) -> CType
    fn is_compile_time_constant(expr: &Expression) -> bool
    fn try_evaluate_integer_constant(expr: &Expression) -> Option<i128>
}
```

---

## 6. constant_eval.rs

### 类型

```rust
#[derive(Debug, Clone)]
pub enum ConstValue {
    SignedInt(i128),
    UnsignedInt(u128),
    Float(f64),
    Address { symbol: Symbol, offset: i64 },
    StringLiteral(Vec<u8>),
}

impl ConstValue {
    pub fn to_i128(&self) -> Option<i128>
    pub fn to_u128(&self) -> Option<u128>
    pub fn to_bool(&self) -> bool
    pub fn is_zero(&self) -> bool
    pub fn negate(&self) -> ConstValue
    fn is_signed(&self) -> bool
    fn is_unsigned_val(&self) -> bool
}

pub struct ConstantEvaluator<'a> {
    diagnostics: &'a mut DiagnosticEngine,
    target: Target,
    enum_values: FxHashMap<Symbol, i128>,
    tag_types: FxHashMap<Symbol, CType>,
    variable_types: FxHashMap<Symbol, CType>,
    typedef_types: FxHashMap<Symbol, CType>,
    name_table: Vec<String>,
    tag_types_by_name: std::collections::HashMap<String, CType>,
}
```

### 函数

```rust
impl<'a> ConstantEvaluator<'a> {
    pub fn new(diagnostics: &'a mut DiagnosticEngine, target: Target) -> Self
    
    pub fn register_enum_value(&mut self, name: Symbol, value: i128)
    pub fn register_tag_type(&mut self, tag: Symbol, ty: CType)
    pub fn set_tag_name_index(&mut self, map: std::collections::HashMap<String, CType>)
    pub fn take_tag_name_index(&mut self) -> std::collections::HashMap<String, CType>
    pub fn register_variable_type(&mut self, name: Symbol, ty: CType)
    pub fn register_typedef_type(&mut self, name: Symbol, ty: CType)
    pub fn set_name_table(&mut self, names: Vec<String>)
    
    pub fn evaluate_integer_constant(&mut self, expr: &Expression, span: Span) -> Result<i128, ()>
    pub fn evaluate_constant_expr(&mut self, expr: &Expression) -> Result<ConstValue, ()>
    pub fn is_constant_expression(&self, expr: &Expression) -> bool
    pub fn validate_array_size(&mut self, expr: &Expression, span: Span) -> Result<usize, ()>
    pub fn validate_bitfield_width(&mut self, expr: &Expression, base_type: &CType, span: Span) -> Result<u32, ()>
    pub fn evaluate_static_assert(&mut self, expr: &Expression, message: Option<&str>, span: Span) -> Result<(), ()>
    
    fn evaluate_integer_literal(&self, value: u128, suffix: &IntegerSuffix, _span: Span) -> ConstValue
    fn evaluate_binary_op(&mut self, op: BinaryOp, lhs: &Expression, rhs: &Expression, span: Span) -> Result<ConstValue, ()>
    fn evaluate_shift(&mut self, lv: i128, rv: i128, result_unsigned: bool, is_left: bool, span: Span) -> Result<ConstValue, ()>
    fn evaluate_unary_op(&mut self, op: UnaryOp, operand: &Expression, span: Span) -> Result<ConstValue, ()>
    fn evaluate_ternary(&mut self, cond: &Expression, then_expr: Option<&Expression>, else_expr: &Expression, _span: Span) -> Result<ConstValue, ()>
    fn evaluate_comma_exprs(&mut self, exprs: &[Expression], span: Span) -> Result<ConstValue, ()>
    fn evaluate_offsetof(&mut self, _type_arg: &Expression, _member_arg: &Expression, _span: Span) -> Result<ConstValue, ()>
    fn evaluate_sizeof_type(&mut self, type_name: &TypeName, span: Span) -> Result<ConstValue, ()>
    fn evaluate_sizeof_expr(&mut self, operand: &Expression, span: Span) -> Result<ConstValue, ()>
    fn evaluate_alignof(&mut self, type_name: &TypeName, span: Span) -> Result<ConstValue, ()>
    fn evaluate_cast_expr(&mut self, type_name: &TypeName, operand: &Expression, span: Span) -> Result<ConstValue, ()>
    fn apply_cast(&mut self, target_type: &CType, val: &ConstValue, span: Span) -> Result<ConstValue, ()>
    fn truncate_to_width(&self, value: i128, bit_width: usize, is_unsigned_target: bool) -> ConstValue
    fn evaluate_builtin_call(&mut self, builtin: &BuiltinKind, args: &[Expression], span: Span) -> Result<ConstValue, ()>
    fn evaluate_function_call_as_constant(&mut self, callee: &Expression, args: &[Expression], span: Span) -> Result<ConstValue, ()>
    fn usual_arithmetic_conversions(&mut self, lv: &ConstValue, rv: &ConstValue) -> (i128, i128, bool)
    fn emit_diag(&mut self, severity: Severity, span: Span, msg: &str)
}
```

---

## 7. initializer.rs

### 类型

```rust
#[derive(Debug, Clone)]
pub enum AnalyzedInit {
    Expression { expr: Expression, target_type: CType },
    Struct { fields: Vec<(usize, AnalyzedInit)>, struct_type: CType },
    Union { field_index: usize, initializer: Box<AnalyzedInit>, union_type: CType },
    Array { elements: Vec<(usize, AnalyzedInit)>, array_type: CType, array_size: usize },
    Zero { target_type: CType },
}

pub struct InitializerAnalyzer<'a> {
    diagnostics: &'a mut DiagnosticEngine,
    type_builder: &'a TypeBuilder,
    target: Target,
    interner: &'a Interner,
}
```

### 函数

```rust
impl<'a> InitializerAnalyzer<'a> {
    pub fn new(
        diagnostics: &'a mut DiagnosticEngine,
        type_builder: &'a TypeBuilder,
        target: Target,
        interner: &'a Interner,
    ) -> Self

    pub fn analyze_initializer(&mut self, init: &Initializer, target_type: &CType, span: Span) -> Result<AnalyzedInit, ()>
    pub fn is_constant_initializer(&self, init: &AnalyzedInit) -> bool
    
    fn analyze_simple_init(&mut self, expr: &Expression, target_type: &CType, _span: Span) -> Result<AnalyzedInit, ()>
    fn analyze_init_list(&mut self, items: &[DesignatedInitializer], target_type: &CType, span: Span) -> Result<AnalyzedInit, ()>
    fn analyze_initializer_value(&mut self, init: &Initializer, target_type: &CType, span: Span) -> Result<AnalyzedInit, ()>
    fn analyze_struct_init_cursor(&mut self, items: &[DesignatedInitializer], pos: &mut usize, fields: &[StructField], struct_type: CType, span: Span) -> Result<AnalyzedInit, ()>
    fn emit_field_override_warning(&mut self, field_inits: &[Option<AnalyzedInit>], field_idx: usize, fields: &[StructField], span: Span)
    fn analyze_union_init_cursor(&mut self, items: &[DesignatedInitializer], pos: &mut usize, fields: &[StructField], union_type: CType, span: Span) -> Result<AnalyzedInit, ()>
    fn analyze_array_init_cursor(&mut self, items: &[DesignatedInitializer], pos: &mut usize, elem_type: &CType, known_size: Option<usize>, array_type: CType, span: Span) -> Result<AnalyzedInit, ()>
    fn update_max(max: &mut Option<usize>, idx: usize)
    fn analyze_one_element(&mut self, items: &[DesignatedInitializer], pos: &mut usize, target_type: &CType, span: Span) -> Result<AnalyzedInit, ()>
    fn analyze_with_nested_desig(&mut self, designators: &[Designator], init: &Initializer, target_type: &CType, span: Span) -> Result<AnalyzedInit, ()>
    fn analyze_anonymous_field_init(&mut self, path: &[usize], remaining_desig: &[Designator], init: &Initializer, outer_fields: &[StructField], span: Span) -> Result<AnalyzedInit, ()>
    fn analyze_string_init(&mut self, segments: &[StringSegment], prefix: StringPrefix, str_span: Span, elem_type: &CType, known_size: Option<usize>, array_type: &CType) -> Result<AnalyzedInit, ()>
    fn resolve_type(ty: &CType) -> CType
    fn is_char_element_type(ty: &CType, prefix: StringPrefix) -> bool
    fn is_aggregate_type(ty: &CType) -> bool
    fn find_field_index(fields: &[StructField], name: Symbol) -> Option<usize>
    fn find_in_anonymous(fields: &[StructField], name: Symbol) -> Option<(Vec<usize>, &StructField)>
    fn char_width_for_prefix(prefix: StringPrefix) -> usize
    fn wchar_width(&self) -> usize
    fn try_eval_integer(expr: &Expression) -> Option<i128>
}
```

---

## 8. attribute_handler.rs

### 类型

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum ValidatedAttribute {
    Aligned(u64),
    Packed,
    Section(String),
    Used,
    Unused,
    Weak,
    Constructor(Option<i32>),
    Destructor(Option<i32>),
    Visibility(SymbolVisibility),
    Deprecated(Option<String>),
    NoReturn,
    NoInline,
    AlwaysInline,
    Cold,
    Hot,
    Format { archetype: FormatArchetype, string_index: u32, first_to_check: u32 },
    FormatArg(u32),
    Malloc,
    Pure,
    Const,
    WarnUnusedResult,
    Fallthrough,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SymbolVisibility {
    Default,
    Hidden,
    Protected,
    Internal,
}

impl Default for SymbolVisibility {
    fn default() -> Self { SymbolVisibility::Default }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FormatArchetype {
    Printf,
    Scanf,
    Strftime,
    Strfmon,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttributeContext {
    Function,
    Variable,
    Type,
    Field,
    Statement,
    Label,
}

pub struct AttributeHandler<'a> {
    diagnostics: &'a mut DiagnosticEngine,
    target: Target,
    interner: &'a Interner,
}
```

### 函数

```rust
impl<'a> AttributeHandler<'a> {
    pub fn new(
        diagnostics: &'a mut DiagnosticEngine,
        target: Target,
        interner: &'a Interner,
    ) -> Self

    pub fn validate_attributes(&mut self, attrs: &[Attribute], context: AttributeContext, _span: Span) -> Vec<ValidatedAttribute>
    fn is_applicable(&self, attr: &ValidatedAttribute, context: AttributeContext) -> bool
    fn attribute_kind_str(&self, attr: &ValidatedAttribute) -> &'static str
    fn validate_aligned(&mut self, args: &[AttributeArg], span: Span) -> Option<ValidatedAttribute>
    fn validate_packed(&mut self, args: &[AttributeArg], span: Span) -> Option<ValidatedAttribute>
    fn validate_section(&mut self, args: &[AttributeArg], span: Span) -> Option<ValidatedAttribute>
    fn validate_used(&mut self, _span: Span) -> Option<ValidatedAttribute>
    fn validate_unused(&mut self, _span: Span) -> Option<ValidatedAttribute>
    fn validate_weak(&mut self, _span: Span) -> Option<ValidatedAttribute>
    fn validate_constructor(&mut self, args: &[AttributeArg], span: Span) -> Option<ValidatedAttribute>
    fn validate_destructor(&mut self, args: &[AttributeArg], span: Span) -> Option<ValidatedAttribute>
    fn validate_ctor_dtor(&mut self, args: &[AttributeArg], span: Span, is_ctor: bool) -> Option<ValidatedAttribute>
    fn validate_visibility(&mut self, args: &[AttributeArg], span: Span) -> Option<ValidatedAttribute>
    fn validate_deprecated(&mut self, args: &[AttributeArg], span: Span) -> Option<ValidatedAttribute>
    fn validate_noreturn(&mut self, _span: Span) -> Option<ValidatedAttribute>
    fn validate_noinline(&mut self, _span: Span) -> Option<ValidatedAttribute>
    fn validate_always_inline(&mut self, _span: Span) -> Option<ValidatedAttribute>
    fn validate_cold(&mut self, _span: Span) -> Option<ValidatedAttribute>
    fn validate_hot(&mut self, _span: Span) -> Option<ValidatedAttribute>
    fn validate_format(&mut self, args: &[AttributeArg], span: Span) -> Option<ValidatedAttribute>
    fn validate_format_arg(&mut self, args: &[AttributeArg], span: Span) -> Option<ValidatedAttribute>
    fn validate_malloc(&mut self, _span: Span) -> Option<ValidatedAttribute>
    fn validate_pure(&mut self, _span: Span) -> Option<ValidatedAttribute>
    fn validate_const(&mut self, _span: Span) -> Option<ValidatedAttribute>
    fn validate_warn_unused_result(&mut self, _span: Span) -> Option<ValidatedAttribute>
    fn validate_fallthrough(&mut self, _span: Span) -> Option<ValidatedAttribute>
    fn resolve_conflicts(&mut self, attrs: &mut Vec<ValidatedAttribute>)
    fn keep_last_of<F>(&self, attrs: &mut Vec<ValidatedAttribute>, pred: F) where F: Fn(&ValidatedAttribute) -> bool
    pub fn propagate_to_type(&self, attrs: &[ValidatedAttribute], ty: &mut CType)
    pub fn propagate_to_symbol(&self, attrs: &[ValidatedAttribute], symbol_attrs: &mut Vec<ValidatedAttribute>)
    fn extract_integer_arg(&mut self, arg: &AttributeArg, span: Span) -> Option<u64>
    fn extract_string_arg(&mut self, arg: &AttributeArg, span: Span) -> Option<String>
    fn eval_const_integer(&mut self, expr: &Expression, span: Span) -> Option<u64>
    fn is_const_char_ptr(ty: &CType) -> bool
    fn is_pointer_type(ty: &CType) -> bool
    fn is_function_type(ty: &CType) -> bool
    fn make_const_qualified(ty: CType) -> CType
}

// Free-Standing Utilities
pub fn has_attribute_by_symbol(attrs: &[Attribute], name: Symbol) -> bool
pub fn extract_visibility(attrs: &[ValidatedAttribute]) -> SymbolVisibility
pub fn extract_section(attrs: &[ValidatedAttribute]) -> Option<&str>
pub fn has_used_attribute(attrs: &[ValidatedAttribute]) -> bool
pub fn has_weak_attribute(attrs: &[ValidatedAttribute]) -> bool
pub fn has_noreturn_attribute(attrs: &[ValidatedAttribute]) -> bool
```