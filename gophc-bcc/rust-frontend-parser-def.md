# BCC Rust 前端语法分析器定义

本文档按文件整理 `src/frontend/parser/` 下所有类型、函数和方法签名（私有），按定义出现顺序，不包括测试。

---

## 1. src/frontend/parser/mod.rs

### Parser<'src> 类型定义

```rust
pub struct Parser<'src> {
    pub(crate) lexer: Lexer<'src>,
    pub(crate) current: Token,
    pub(crate) previous: Token,
    pub(crate) target: Target,
    recursion_depth: u32,
    max_recursion_depth: u32,
    panic_mode: bool,
    pub(crate) typedef_names: crate::common::fx_hash::FxHashSet<u32>,
    typedef_shadow_stack: Vec<Vec<u32>>,
}
```

### Parser<'src> 方法

```rust
impl<'src> Parser<'src> {
    pub fn new(mut lexer: Lexer<'src>, target: Target) -> Self
    pub fn advance(&mut self)
    pub fn expect(&mut self, kind: TokenKind) -> Result<Token, ()>
    pub fn check(&self, kind: &TokenKind) -> bool
    pub fn match_token(&mut self, kind: &TokenKind) -> bool
    pub fn consume(&mut self, kind: TokenKind, msg: &str) -> Result<Token, ()>
    #[inline]
    pub fn peek(&self) -> &TokenKind
    pub fn is_type_specifier_start(&self) -> bool
    pub fn peek_nth(&mut self, n: usize) -> Token
    #[inline]
    pub fn current_span(&self) -> Span
    #[inline]
    pub fn previous_span(&self) -> Span
    #[inline]
    pub fn make_span(&self, start: Span) -> Span
    pub fn enter_recursion(&mut self) -> Result<(), ()>
    pub fn is_at_recursion_limit(&self) -> bool
    pub fn leave_recursion(&mut self)
    pub fn error(&mut self, span: Span, msg: &str)
    pub fn warn(&mut self, span: Span, msg: &str)
    pub fn synchronize(&mut self)
    pub fn has_errors(&self) -> bool
    pub fn error_count(&self) -> usize
    pub fn too_many_errors(&self) -> bool
    pub fn resolve_symbol(&self, sym: Symbol) -> &str
    pub fn intern(&mut self, s: &str) -> Symbol
    #[inline]
    pub fn target(&self) -> Target
    pub fn is_typedef_name(&self, sym: Symbol) -> bool
    pub fn register_typedef(&mut self, sym: Symbol)
    pub fn shadow_typedef(&mut self, sym: Symbol)
    pub fn push_typedef_scope(&mut self)
    pub fn pop_typedef_scope(&mut self)
    pub fn skip_asm_label(&mut self) -> Option<String>
    pub fn skip_trailing_attributes(&mut self)
    pub fn skip_c23_attribute(&mut self)
    pub fn is_at_identifier(&self) -> bool
    pub fn current_identifier(&self) -> Option<Symbol>
    pub fn parse(&mut self) -> TranslationUnit
    pub fn parse_translation_unit(&mut self) -> TranslationUnit
    pub fn parse_external_declaration(&mut self) -> Result<ExternalDeclaration, ()>
    fn parse_static_assert_external(&mut self) -> Result<ExternalDeclaration, ()>
    fn parse_string_literal_bytes(&mut self) -> Option<Vec<u8>>
    fn parse_remaining_declaration(&mut self, start_span: Span, specifiers: DeclarationSpecifiers, first_declarator: Declarator) -> Result<ExternalDeclaration, ()>
    fn parse_initializer(&mut self) -> Result<Initializer, ()>
    fn parse_brace_initializer(&mut self) -> Result<Initializer, ()>
}

impl<'src> fmt::Debug for Parser<'src> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
}
```

---

## 2. src/frontend/parser/ast.rs

### TranslationUnit 类型定义

```rust
pub struct TranslationUnit {
    pub declarations: Vec<ExternalDeclaration>,
    pub span: Span,
}
```

### ExternalDeclaration 类型定义

```rust
pub enum ExternalDeclaration {
    FunctionDefinition(Box<FunctionDefinition>),
    Declaration(Box<Declaration>),
    AsmStatement(AsmStatement),
    Empty,
}
```

### Declaration 类型定义

```rust
pub struct Declaration {
    pub specifiers: DeclarationSpecifiers,
    pub declarators: Vec<InitDeclarator>,
    pub static_assert: Option<StaticAssert>,
    pub span: Span,
}
```

### InitDeclarator 类型定义

```rust
pub struct InitDeclarator {
    pub declarator: Declarator,
    pub initializer: Option<Initializer>,
    pub asm_register: Option<String>,
    pub span: Span,
}
```

### DeclarationSpecifiers 类型定义

```rust
pub struct DeclarationSpecifiers {
    pub storage_class: Option<StorageClass>,
    pub type_specifiers: Vec<TypeSpecifier>,
    pub type_qualifiers: Vec<TypeQualifier>,
    pub function_specifiers: Vec<FunctionSpecifier>,
    pub alignment_specifier: Option<AlignmentSpecifier>,
    pub attributes: Vec<Attribute>,
    pub span: Span,
}
```

### StorageClass 类型定义

```rust
pub enum StorageClass {
    Auto,
    Register,
    Static,
    Extern,
    Typedef,
    ThreadLocal,
}
```

### FunctionSpecifier 类型定义

```rust
pub enum FunctionSpecifier {
    Inline,
    Noreturn,
}
```

### AlignmentSpecifier 类型定义

```rust
pub struct AlignmentSpecifier {
    pub arg: AlignasArg,
    pub span: Span,
}
```

### AlignasArg 类型定义

```rust
pub enum AlignasArg {
    Type(TypeName),
    Expression(Box<Expression>),
}
```

### TypeSpecifier 类型定义

```rust
pub enum TypeSpecifier {
    Void,
    Char,
    Short,
    Int,
    Long,
    Float,
    Double,
    Signed,
    Unsigned,
    Bool,
    Complex,
    Int128,
    Float128,
    Float16,
    Float32,
    Float64,
    Struct(StructOrUnionSpecifier),
    Union(StructOrUnionSpecifier),
    Enum(EnumSpecifier),
    TypedefName(Symbol),
    Atomic(Box<TypeName>),
    Typeof(TypeofArg),
    AutoType,
}
```

### TypeofArg 类型定义

```rust
pub enum TypeofArg {
    Expression(Box<Expression>),
    TypeName(Box<TypeName>),
}
```

### StructOrUnionSpecifier 类型定义

```rust
pub struct StructOrUnionSpecifier {
    pub kind: StructOrUnion,
    pub tag: Option<Symbol>,
    pub members: Option<Vec<StructMember>>,
    pub attributes: Vec<Attribute>,
    pub span: Span,
}
```

### StructOrUnion 类型定义

```rust
pub enum StructOrUnion {
    Struct,
    Union,
}
```

### StructMember 类型定义

```rust
pub struct StructMember {
    pub specifiers: SpecifierQualifierList,
    pub declarators: Vec<StructDeclarator>,
    pub attributes: Vec<Attribute>,
    pub span: Span,
}
```

### StructDeclarator 类型定义

```rust
pub struct StructDeclarator {
    pub declarator: Option<Declarator>,
    pub bit_width: Option<Box<Expression>>,
    pub span: Span,
}
```

### EnumSpecifier 类型定义

```rust
pub struct EnumSpecifier {
    pub tag: Option<Symbol>,
    pub enumerators: Option<Vec<Enumerator>>,
    pub attributes: Vec<Attribute>,
    pub span: Span,
}
```

### Enumerator 类型定义

```rust
pub struct Enumerator {
    pub name: Symbol,
    pub value: Option<Box<Expression>>,
    pub span: Span,
}
```

### SpecifierQualifierList 类型定义

```rust
pub struct SpecifierQualifierList {
    pub type_specifiers: Vec<TypeSpecifier>,
    pub type_qualifiers: Vec<TypeQualifier>,
    pub attributes: Vec<Attribute>,
    pub span: Span,
}
```

### TypeQualifier 类型定义

```rust
pub enum TypeQualifier {
    Const,
    Volatile,
    Restrict,
    Atomic,
}
```

### TypeName 类型定义

```rust
pub struct TypeName {
    pub specifier_qualifiers: SpecifierQualifierList,
    pub abstract_declarator: Option<AbstractDeclarator>,
    pub span: Span,
}
```

### AbstractDeclarator 类型定义

```rust
pub struct AbstractDeclarator {
    pub pointer: Option<Pointer>,
    pub direct: Option<DirectAbstractDeclarator>,
    pub span: Span,
}
```

### DirectAbstractDeclarator 类型定义

```rust
pub enum DirectAbstractDeclarator {
    Parenthesized(Box<AbstractDeclarator>),
    Array {
        base: Option<Box<DirectAbstractDeclarator>>,
        size: Option<Box<Expression>>,
        qualifiers: Vec<TypeQualifier>,
        is_static: bool,
    },
    Function {
        base: Option<Box<DirectAbstractDeclarator>>,
        params: Vec<ParameterDeclaration>,
        is_variadic: bool,
    },
}
```

### Declarator 类型定义

```rust
pub struct Declarator {
    pub pointer: Option<Pointer>,
    pub direct: DirectDeclarator,
    pub attributes: Vec<Attribute>,
    pub span: Span,
}
```

### Declarator 方法

```rust
impl Declarator {
    pub fn has_function_suffix(&self) -> bool
}
```

### Pointer 类型定义

```rust
pub struct Pointer {
    pub qualifiers: Vec<TypeQualifier>,
    pub inner: Option<Box<Pointer>>,
    pub span: Span,
}
```

### DirectDeclarator 类型定义

```rust
pub enum DirectDeclarator {
    Identifier(Symbol, Span),
    Parenthesized(Box<Declarator>),
    Array {
        base: Box<DirectDeclarator>,
        size: Option<Box<Expression>>,
        qualifiers: Vec<TypeQualifier>,
        is_static: bool,
        is_star: bool,
        span: Span,
    },
    Function {
        base: Box<DirectDeclarator>,
        params: Vec<ParameterDeclaration>,
        is_variadic: bool,
        span: Span,
    },
}
```

### ParameterDeclaration 类型定义

```rust
pub struct ParameterDeclaration {
    pub specifiers: DeclarationSpecifiers,
    pub declarator: Option<Declarator>,
    pub abstract_declarator: Option<AbstractDeclarator>,
    pub span: Span,
}
```

### FunctionDefinition 类型定义

```rust
pub struct FunctionDefinition {
    pub specifiers: DeclarationSpecifiers,
    pub declarator: Declarator,
    pub old_style_params: Vec<Declaration>,
    pub body: CompoundStatement,
    pub attributes: Vec<Attribute>,
    pub span: Span,
}
```

### Statement 类型定义

```rust
pub enum Statement {
    Compound(CompoundStatement),
    Expression(Option<Box<Expression>>),
    If {
        condition: Box<Expression>,
        then_branch: Box<Statement>,
        else_branch: Option<Box<Statement>>,
        span: Span,
    },
    Switch {
        condition: Box<Expression>,
        body: Box<Statement>,
        span: Span,
    },
    While {
        condition: Box<Expression>,
        body: Box<Statement>,
        span: Span,
    },
    DoWhile {
        body: Box<Statement>,
        condition: Box<Expression>,
        span: Span,
    },
    For {
        init: Option<ForInit>,
        condition: Option<Box<Expression>>,
        increment: Option<Box<Expression>>,
        body: Box<Statement>,
        span: Span,
    },
    Goto { label: Symbol, span: Span },
    ComputedGoto { target: Box<Expression>, span: Span },
    Continue { span: Span },
    Break { span: Span },
    Return {
        value: Option<Box<Expression>>,
        span: Span,
    },
    Labeled {
        label: Symbol,
        attributes: Vec<Attribute>,
        statement: Box<Statement>,
        span: Span,
    },
    Case {
        value: Box<Expression>,
        statement: Box<Statement>,
        span: Span,
    },
    CaseRange {
        low: Box<Expression>,
        high: Box<Expression>,
        statement: Box<Statement>,
        span: Span,
    },
    Default {
        statement: Box<Statement>,
        span: Span,
    },
    Declaration(Box<Declaration>),
    Asm(AsmStatement),
    LocalLabel(Vec<Symbol>, Span),
}
```

### CompoundStatement 类型定义

```rust
pub struct CompoundStatement {
    pub items: Vec<BlockItem>,
    pub span: Span,
}
```

### BlockItem 类型定义

```rust
pub enum BlockItem {
    Declaration(Box<Declaration>),
    Statement(Statement),
}
```

### ForInit 类型定义

```rust
pub enum ForInit {
    Declaration(Box<Declaration>),
    Expression(Box<Expression>),
}
```

### Expression 类型定义

```rust
pub enum Expression {
    IntegerLiteral {
        value: u128,
        suffix: IntegerSuffix,
        is_hex_or_octal: bool,
        span: Span,
    },
    FloatLiteral {
        value: f64,
        suffix: FloatSuffix,
        span: Span,
    },
    StringLiteral {
        segments: Vec<StringSegment>,
        prefix: StringPrefix,
        span: Span,
    },
    CharLiteral {
        value: u32,
        prefix: CharPrefix,
        span: Span,
    },
    Identifier { name: Symbol, span: Span },
    Parenthesized { inner: Box<Expression>, span: Span },
    ArraySubscript {
        base: Box<Expression>,
        index: Box<Expression>,
        span: Span,
    },
    FunctionCall {
        callee: Box<Expression>,
        args: Vec<Expression>,
        span: Span,
    },
    MemberAccess {
        object: Box<Expression>,
        member: Symbol,
        span: Span,
    },
    PointerMemberAccess {
        object: Box<Expression>,
        member: Symbol,
        span: Span,
    },
    PostIncrement {
        operand: Box<Expression>,
        span: Span,
    },
    PostDecrement {
        operand: Box<Expression>,
        span: Span,
    },
    PreIncrement {
        operand: Box<Expression>,
        span: Span,
    },
    PreDecrement {
        operand: Box<Expression>,
        span: Span,
    },
    UnaryOp {
        op: UnaryOp,
        operand: Box<Expression>,
        span: Span,
    },
    SizeofExpr {
        operand: Box<Expression>,
        span: Span,
    },
    SizeofType {
        type_name: Box<TypeName>,
        span: Span,
    },
    AlignofType {
        type_name: Box<TypeName>,
        span: Span,
    },
    AlignofExpr { expr: Box<Expression>, span: Span },
    Cast {
        type_name: Box<TypeName>,
        operand: Box<Expression>,
        span: Span,
    },
    Binary {
        op: BinaryOp,
        left: Box<Expression>,
        right: Box<Expression>,
        span: Span,
    },
    Conditional {
        condition: Box<Expression>,
        then_expr: Option<Box<Expression>>,
        else_expr: Box<Expression>,
        span: Span,
    },
    Assignment {
        op: AssignOp,
        target: Box<Expression>,
        value: Box<Expression>,
        span: Span,
    },
    Comma { exprs: Vec<Expression>, span: Span },
    CompoundLiteral {
        type_name: Box<TypeName>,
        initializer: Initializer,
        span: Span,
    },
    StatementExpression {
        compound: CompoundStatement,
        span: Span,
    },
    BuiltinCall {
        builtin: BuiltinKind,
        args: Vec<Expression>,
        span: Span,
    },
    Generic {
        controlling: Box<Expression>,
        associations: Vec<GenericAssociation>,
        span: Span,
    },
    AddressOfLabel { label: Symbol, span: Span },
}
```

### UnaryOp 类型定义

```rust
pub enum UnaryOp {
    AddressOf,
    Deref,
    Plus,
    Negate,
    BitwiseNot,
    LogicalNot,
    RealPart,
    ImagPart,
}
```

### BinaryOp 类型定义

```rust
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    ShiftLeft,
    ShiftRight,
    LogicalAnd,
    LogicalOr,
    Equal,
    NotEqual,
    Less,
    Greater,
    LessEqual,
    GreaterEqual,
}
```

### AssignOp 类型定义

```rust
pub enum AssignOp {
    Assign,
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
    ModAssign,
    AndAssign,
    OrAssign,
    XorAssign,
    ShlAssign,
    ShrAssign,
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

### CharPrefix 类型定义

```rust
pub enum CharPrefix {
    None,
    L,
    U16,
    U32,
}
```

### StringSegment 类型定义

```rust
pub struct StringSegment {
    pub value: Vec<u8>,
    pub span: Span,
}
```

### GenericAssociation 类型定义

```rust
pub struct GenericAssociation {
    pub type_name: Option<TypeName>,
    pub expression: Box<Expression>,
    pub span: Span,
}
```

### BuiltinKind 类型定义

```rust
pub enum BuiltinKind {
    Expect,
    Unreachable,
    ConstantP,
    Offsetof,
    TypesCompatibleP,
    ChooseExpr,
    Clz,
    ClzL,
    ClzLL,
    Ctz,
    CtzL,
    CtzLL,
    Popcount,
    PopcountL,
    PopcountLL,
    Bswap16,
    Bswap32,
    Bswap64,
    Ffs,
    Ffsll,
    VaStart,
    VaEnd,
    VaArg,
    VaCopy,
    FrameAddress,
    ReturnAddress,
    Trap,
    AssumeAligned,
    AddOverflow,
    SubOverflow,
    MulOverflow,
    PrefetchData,
    ObjectSize,
    ExtractReturnAddr,
    Abort,
    Exit,
    Memcpy,
    Memset,
    Memcmp,
    Strlen,
    Strcmp,
    Strncmp,
    Abs,
    Labs,
    Llabs,
    Alloca,
    Inf,
    Inff,
    Infl,
    Nan,
    Nanf,
    Nanl,
    HugeVal,
    HugeValf,
    HugeVall,
    Signbit,
    Isnan,
    Isinf,
    Isfinite,
    IsinfSign,
    Copysign,
    Copysignf,
    Copysignl,
    Fabs,
    Fabsf,
    Fabsl,
    ClassifyType,
    BuiltinConstantP,
    SyncValCompareAndSwap,
    SyncBoolCompareAndSwap,
    SyncFetchAndAdd,
    SyncFetchAndSub,
    SyncFetchAndAnd,
    SyncFetchAndOr,
    SyncFetchAndXor,
    SyncLockTestAndSet,
    SyncLockRelease,
    SyncSynchronize,
    AtomicLoadN,
    AtomicStoreN,
    AtomicExchangeN,
    AtomicCompareExchangeN,
    AtomicFetchAdd,
    AtomicFetchSub,
    AtomicFetchAnd,
    AtomicFetchOr,
    AtomicFetchXor,
    MulOverflowP,
    AddOverflowP,
    SubOverflowP,
}
```

### Initializer 类型定义

```rust
pub enum Initializer {
    Expression(Box<Expression>),
    List {
        designators_and_initializers: Vec<DesignatedInitializer>,
        trailing_comma: bool,
        span: Span,
    },
}
```

### DesignatedInitializer 类型定义

```rust
pub struct DesignatedInitializer {
    pub designators: Vec<Designator>,
    pub initializer: Initializer,
    pub span: Span,
}
```

### Designator 类型定义

```rust
pub enum Designator {
    Field(Symbol, Span),
    Index(Box<Expression>, Span),
    IndexRange(Box<Expression>, Box<Expression>, Span),
}
```

### Attribute 类型定义

```rust
pub struct Attribute {
    pub name: Symbol,
    pub args: Vec<AttributeArg>,
    pub span: Span,
}
```

### AttributeArg 类型定义

```rust
pub enum AttributeArg {
    Identifier(Symbol, Span),
    Expression(Box<Expression>),
    String(Vec<u8>, Span),
    Type(TypeName),
}
```

### AsmStatement 类型定义

```rust
pub struct AsmStatement {
    pub is_volatile: bool,
    pub is_goto: bool,
    pub template: Vec<u8>,
    pub outputs: Vec<AsmOperand>,
    pub inputs: Vec<AsmOperand>,
    pub clobbers: Vec<AsmClobber>,
    pub goto_labels: Vec<Symbol>,
    pub span: Span,
}
```

### AsmOperand 类型定义

```rust
pub struct AsmOperand {
    pub symbolic_name: Option<Symbol>,
    pub constraint: Vec<u8>,
    pub expression: Box<Expression>,
    pub span: Span,
}
```

### AsmClobber 类型定义

```rust
pub struct AsmClobber {
    pub register: Vec<u8>,
    pub span: Span,
}
```

### StaticAssert 类��定义

```rust
pub struct StaticAssert {
    pub condition: Box<Expression>,
    pub message: Option<Vec<u8>>,
    pub span: Span,
}
```

### Expression 方法

```rust
impl Expression {
    pub fn span(&self) -> Span
}
```

### Statement 方法

```rust
impl Statement {
    pub fn span(&self) -> Span
}
```

### TranslationUnit 方法

```rust
impl TranslationUnit {
    pub fn new() -> Self
}

impl Default for TranslationUnit {
    fn default() -> Self
}
```

### DeclarationSpecifiers 方法

```rust
impl DeclarationSpecifiers {
    pub fn new() -> Self
}
```

---

## 3. src/frontend/parser/types.rs

### TypeSpecifierList 类型定义

```rust
pub struct TypeSpecifierList {
    pub specifiers: Vec<TypeSpecifier>,
    pub span: Span,
}
```

### TypeQualifiers 类型定义

```rust
pub struct TypeQualifiers {
    pub is_const: bool,
    pub is_volatile: bool,
    pub is_restrict: bool,
    pub is_atomic: bool,
    pub span: Span,
}
```

### TypeQualifiers 方法

```rust
impl TypeQualifiers {
    pub fn is_empty(&self) -> bool
    pub fn to_qualifier_list(&self) -> Vec<TypeQualifier>
}

impl Default for TypeQualifiers {
    fn default() -> Self
}
```

### SpecifierFlags 类型定义（私有）

```rust
struct SpecifierFlags {
    has_void: bool,
    has_char: bool,
    has_short: bool,
    has_int: bool,
    long_count: u8,
    has_float: bool,
    has_double: bool,
    has_signed: bool,
    has_unsigned: bool,
    has_bool: bool,
    has_complex: bool,
    has_int128: bool,
    has_struct_union_enum: bool,
    has_typedef_name: bool,
    has_atomic: bool,
    has_typeof: bool,
}
```

### SpecifierFlags 方法

```rust
impl SpecifierFlags {
    fn new() -> Self
    fn has_any_primary(&self) -> bool
}
```

### 公共函数

```rust
pub fn parse_type_specifiers(parser: &mut Parser<'_>) -> Result<TypeSpecifierList, ()>
pub fn parse_type_qualifiers(parser: &mut Parser<'_>) -> Result<TypeQualifiers, ()>
pub fn parse_specifier_qualifier_list(parser: &mut Parser<'_>) -> Result<SpecifierQualifierList, ()>
pub fn is_type_specifier_start(token: &TokenKind, parser: &Parser<'_>) -> bool
pub fn parse_alignas(parser: &mut Parser<'_>) -> Result<AlignmentSpecifier, ()>
```

### 私有函数

```rust
fn parse_typeof(parser: &mut Parser<'_>) -> Result<TypeSpecifier, ()>
fn parse_atomic_type_specifier(parser: &mut Parser<'_>) -> Result<TypeSpecifier, ()>
fn parse_type_name_inner(parser: &mut Parser<'_>) -> Result<TypeName, ()>
fn parse_abstract_declarator_opt(parser: &mut Parser<'_>) -> Result<Option<AbstractDeclarator>, ()>
fn parse_pointer_chain_for_abstract(parser: &mut Parser<'_>) -> Result<Option<Pointer>, ()>
fn parse_direct_abstract_declarator_opt(parser: &mut Parser<'_>) -> Result<Option<DirectAbstractDeclarator>, ()>
fn parse_struct_or_union(parser: &mut Parser<'_>) -> Result<TypeSpecifier, ()>
fn parse_struct_member(parser: &mut Parser<'_>) -> Result<StructMember, ()>
fn parse_enum(parser: &mut Parser<'_>) -> Result<TypeSpecifier, ()>
fn check_add_void(flags: &SpecifierFlags) -> Result<(), &'static str>
fn check_add_char(flags: &SpecifierFlags) -> Result<(), &'static str>
fn check_add_short(flags: &SpecifierFlags) -> Result<(), &'static str>
fn check_add_int(flags: &SpecifierFlags) -> Result<(), &'static str>
fn check_add_long(flags: &SpecifierFlags) -> Result<(), &'static str>
fn check_add_float(flags: &SpecifierFlags) -> Result<(), &'static str>
fn check_add_double(flags: &SpecifierFlags) -> Result<(), &'static str>
fn check_add_signed(flags: &SpecifierFlags) -> Result<(), &'static str>
fn check_add_unsigned(flags: &SpecifierFlags) -> Result<(), &'static str>
fn check_add_bool(flags: &SpecifierFlags) -> Result<(), &'static str>
fn check_add_complex(flags: &SpecifierFlags) -> Result<(), &'static str>
```

---

## 4. src/frontend/parser/declarations.rs

### 公共函数

```rust
pub fn parse_declaration(parser: &mut Parser<'_>) -> Result<Declaration, ()>
pub fn parse_declaration_specifiers(parser: &mut Parser<'_>) -> Result<DeclarationSpecifiers, ()>
pub fn parse_declarator(parser: &mut Parser<'_>) -> Result<Declarator, ()>
pub fn parse_function_definition(parser: &mut Parser<'_>, specifiers: DeclarationSpecifiers, declarator: Declarator) -> Result<FunctionDefinition, ()>
pub fn parse_abstract_declarator(parser: &mut Parser<'_>) -> Result<AbstractDeclarator, ()>
pub fn parse_type_name(parser: &mut Parser<'_>) -> Result<TypeName, ()>
```

### 私有函数

```rust
fn validate_storage_class(current: Option<StorageClass>, has_thread_local: bool, new_specifier: &str) -> Result<(), String>
fn parse_pointer_chain(parser: &mut Parser<'_>) -> Result<Option<Pointer>, ()>
fn parse_direct_declarator(parser: &mut Parser<'_>) -> Result<DirectDeclarator, ()>
fn parse_array_suffix(parser: &mut Parser<'_>, base: DirectDeclarator, start_span: Span) -> Result<DirectDeclarator, ()>
fn parse_function_suffix(parser: &mut Parser<'_>, base: DirectDeclarator, start_span: Span) -> Result<DirectDeclarator, ()>
fn parse_parameter_type_list(parser: &mut Parser<'_>) -> Result<(Vec<ParameterDeclaration>, bool), ()>
fn parse_parameter_declaration(parser: &mut Parser<'_>) -> Result<ParameterDeclaration, ()>
fn is_declarator_start(parser: &mut Parser<'_>) -> bool
fn parse_struct_or_union_specifier(parser: &mut Parser<'_>) -> Result<TypeSpecifier, ()>
fn parse_struct_member(parser: &mut Parser<'_>) -> Result<StructMember, ()>
fn parse_enum_specifier(parser: &mut Parser<'_>) -> Result<TypeSpecifier, ()>
fn parse_typeof_specifier(parser: &mut Parser<'_>) -> Result<TypeSpecifier, ()>
fn parse_alignas_specifier(parser: &mut Parser<'_>) -> Result<AlignmentSpecifier, ()>
fn parse_static_assert_declaration(parser: &mut Parser<'_>) -> Result<Declaration, ()>
fn parse_initializer(parser: &mut Parser<'_>) -> Result<Initializer, ()>
fn parse_brace_initializer(parser: &mut Parser<'_>) -> Result<Initializer, ()>
fn parse_direct_abstract_declarator_opt(parser: &mut Parser<'_>) -> Result<Option<DirectAbstractDeclarator>, ()>
fn parse_abstract_array_declarator(parser: &mut Parser<'_>) -> Result<DirectAbstractDeclarator, ()>
fn parse_abstract_suffix_chain(parser: &mut Parser<'_>, _base: DirectAbstractDeclarator) -> Result<DirectAbstractDeclarator, ()>
fn parse_specifier_qualifier_list_for_type_name(parser: &mut Parser<'_>) -> Result<SpecifierQualifierList, ()>
```

---

## 5. src/frontend/parser/expressions.rs

### Associativity 类型定义（私有）

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Associativity {
    Left,
    Right,
}
```

### 公共函数

```rust
pub fn parse_expression(parser: &mut Parser<'_>) -> Result<Expression, ()>
pub fn parse_assignment_expression(parser: &mut Parser<'_>) -> Result<Expression, ()>
pub fn parse_constant_expression(parser: &mut Parser<'_>) -> Result<Expression, ()>
```

### 私有函数

```rust
fn parse_conditional_expression(parser: &mut Parser<'_>) -> Result<Expression, ()>
fn parse_binary_expression(parser: &mut Parser<'_>, min_prec: u8) -> Result<Expression, ()>
fn get_binary_op_info(token: &TokenKind) -> Option<(u8, Associativity, BinaryOp)>
fn get_assign_op(token: &TokenKind) -> Option<AssignOp>
fn parse_cast_expression(parser: &mut Parser<'_>) -> Result<Expression, ()>
fn parse_unary_expression(parser: &mut Parser<'_>) -> Result<Expression, ()>
fn parse_sizeof_expression(parser: &mut Parser<'_>) -> Result<Expression, ()>
fn parse_compound_literal_body(parser: &mut Parser<'_>, type_name: crate::frontend::parser::ast::TypeName, start: Span) -> Result<Expression, ()>
fn parse_alignof_expression(parser: &mut Parser<'_>) -> Result<Expression, ()>
fn parse_postfix_expression(parser: &mut Parser<'_>) -> Result<Expression, ()>
fn parse_postfix_tail(parser: &mut Parser<'_>, mut expr: Expression) -> Result<Expression, ()>
fn parse_argument_list(parser: &mut Parser<'_>) -> Result<Vec<Expression>, ()>
fn expect_identifier(parser: &mut Parser<'_>) -> Result<Symbol, ()>
fn parse_primary_expression(parser: &mut Parser<'_>) -> Result<Expression, ()>
fn parse_generic_selection(parser: &mut Parser<'_>) -> Result<Expression, ()>
fn parse_builtin_offsetof(parser: &mut Parser<'_>) -> Result<Expression, ()>
fn parse_builtin_types_compatible(parser: &mut Parser<'_>) -> Result<Expression, ()>
fn parse_builtin_choose_expr(parser: &mut Parser<'_>) -> Result<Expression, ()>
fn parse_builtin_va_arg(parser: &mut Parser<'_>) -> Result<Expression, ()>
fn parse_builtin_simple(parser: &mut Parser<'_>, kind: BuiltinKind) -> Result<Expression, ()>
fn parse_type_name(parser: &mut Parser<'_>) -> Result<TypeName, ()>
fn parse_abstract_declarator_opt(parser: &mut Parser<'_>) -> Result<Option<AbstractDeclarator>, ()>
fn parse_pointer_chain(parser: &mut Parser<'_>) -> Result<Option<Pointer>, ()>
fn parse_direct_abstract_declarator_opt(parser: &mut Parser<'_>) -> Result<Option<DirectAbstractDeclarator>, ()>
fn convert_integer_suffix(sfx: token_types::IntegerSuffix) -> IntegerSuffix
fn convert_float_suffix(sfx: token_types::FloatSuffix) -> FloatSuffix
fn convert_string_prefix(pfx: token_types::StringPrefix) -> StringPrefix
fn convert_char_prefix(pfx: token_types::StringPrefix) -> CharPrefix
#[inline]
fn _get_expression_span_or_dummy(expr: &Expression) -> Span
fn parse_hex_float(s: &str) -> f64
```

---

## 6. src/frontend/parser/statements.rs

### 公共函数

```rust
pub fn parse_statement(parser: &mut Parser<'_>) -> Result<Statement, ()>
pub fn parse_compound_statement(parser: &mut Parser<'_>) -> Result<CompoundStatement, ()>
pub fn is_declaration_start(parser: &Parser<'_>) -> bool
```

### 私有函数

```rust
fn parse_if_statement(parser: &mut Parser<'_>) -> Result<Statement, ()>
fn parse_switch_statement(parser: &mut Parser<'_>) -> Result<Statement, ()>
fn parse_while_statement(parser: &mut Parser<'_>) -> Result<Statement, ()>
fn parse_do_while_statement(parser: &mut Parser<'_>) -> Result<Statement, ()>
fn parse_for_statement(parser: &mut Parser<'_>) -> Result<Statement, ()>
fn parse_goto_statement(parser: &mut Parser<'_>) -> Result<Statement, ()>
fn parse_break_statement(parser: &mut Parser<'_>) -> Result<Statement, ()>
fn parse_continue_statement(parser: &mut Parser<'_>) -> Result<Statement, ()>
fn parse_return_statement(parser: &mut Parser<'_>) -> Result<Statement, ()>
fn parse_labeled_statement(parser: &mut Parser<'_>, label_sym: Symbol, start_span: Span) -> Result<Statement, ()>
fn parse_case_label(parser: &mut Parser<'_>) -> Result<Statement, ()>
fn parse_default_label(parser: &mut Parser<'_>) -> Result<Statement, ()>
fn parse_expression_statement(parser: &mut Parser<'_>) -> Result<Statement, ()>
fn parse_declaration_as_statement(parser: &mut Parser<'_>) -> Result<Statement, ()>
fn parse_block_declaration(parser: &mut Parser<'_>) -> Result<Declaration, ()>
fn parse_static_assert_declaration(parser: &mut Parser<'_>, start_span: Span) -> Result<Declaration, ()>
#[inline]
fn parse_condition_expression(parser: &mut Parser<'_>) -> Result<Expression, ()>
#[inline]
fn parse_assignment_expr(parser: &mut Parser<'_>) -> Result<Expression, ()>
#[inline]
fn merge_spans(start: Span, end: Span) -> Span
fn error_recovery_span() -> Span
fn get_declarator_name(declarator: &Declarator) -> Option<Symbol>
fn register_declarator_name(parser: &mut Parser<'_>, declarator: &Declarator)
fn shadow_function_parameter_typedefs(parser: &mut Parser<'_>, declarator: &Declarator)
fn skip_asm_label(parser: &mut Parser<'_>) -> Option<String>
fn skip_trailing_attributes(parser: &mut Parser<'_>)
```

---

## 7. src/frontend/parser/attributes.rs

### 公共函数

```rust
pub fn parse_attribute_specifier(parser: &mut Parser<'_>) -> Result<Vec<Attribute>, ()>
```

### 私有函数

```rust
fn parse_single_attribute(parser: &mut Parser<'_>) -> Result<Attribute, ()>
fn parse_attribute_args(parser: &mut Parser<'_>, canonical_name: &str, has_args: bool, start_span: Span) -> Result<Vec<AttributeArg>, ()>
fn extract_attribute_name(parser: &mut Parser<'_>) -> Option<Symbol>
fn strip_underscores(name: &str) -> &str
fn parse_aligned_args(parser: &mut Parser<'_>) -> Result<Vec<AttributeArg>, ()>
fn parse_string_arg_attribute(parser: &mut Parser<'_>) -> Result<Vec<AttributeArg>, ()>
fn parse_visibility_args(parser: &mut Parser<'_>) -> Result<Vec<AttributeArg>, ()>
fn parse_deprecated_args(parser: &mut Parser<'_>) -> Result<Vec<AttributeArg>, ()>
fn parse_priority_args(parser: &mut Parser<'_>) -> Result<Vec<AttributeArg>, ()>
fn parse_format_args(parser: &mut Parser<'_>) -> Result<Vec<AttributeArg>, ()>
fn parse_format_arg_args(parser: &mut Parser<'_>) -> Result<Vec<AttributeArg>, ()>
fn parse_generic_parenthesized_args(parser: &mut Parser<'_>) -> Result<Vec<AttributeArg>, ()>
fn parse_single_generic_arg(parser: &mut Parser<'_>) -> Result<AttributeArg, ()>
fn parse_string_literal(parser: &mut Parser<'_>) -> Result<(Vec<u8>, Span), ()>
fn skip_to_comma_or_close(parser: &mut Parser<'_>)
fn skip_balanced_parens_consume(parser: &mut Parser<'_>)
fn skip_to_right_paren(parser: &mut Parser<'_>)
```

---

## 8. src/frontend/parser/inline_asm.rs

### 公共函数

```rust
pub fn parse_asm_statement(parser: &mut Parser<'_>) -> Result<AsmStatement, ()>
```

### 私有函数

```rust
fn parse_qualifiers(parser: &mut Parser<'_>, is_volatile: &mut bool, is_goto: &mut bool)
fn parse_template_string(parser: &mut Parser<'_>) -> Result<Vec<u8>, ()>
fn extract_string_literal(parser: &Parser<'_>) -> Option<Vec<u8>>
fn is_string_literal(parser: &Parser<'_>) -> bool
fn parse_operand_list(parser: &mut Parser<'_>) -> Result<Vec<AsmOperand>, ()>
fn parse_single_operand(parser: &mut Parser<'_>) -> Result<AsmOperand, ()>
fn parse_optional_symbolic_name(parser: &mut Parser<'_>) -> Result<Option<Symbol>, ()>
fn parse_clobber_list(parser: &mut Parser<'_>) -> Result<Vec<AsmClobber>, ()>
fn parse_single_clobber(parser: &mut Parser<'_>) -> Result<AsmClobber, ()>
fn parse_goto_label_list(parser: &mut Parser<'_>) -> Result<Vec<Symbol>, ()>
fn parse_single_goto_label(parser: &mut Parser<'_>, seen: &mut Vec<u32>) -> Result<Symbol, ()>
#[inline]
fn is_at_section_boundary(parser: &Parser<'_>) -> bool
fn recover_to_asm_end(parser: &mut Parser<'_>)
```

---

## 9. src/frontend/parser/gcc_extensions.rs

### 公共函数

```rust
pub fn is_gcc_extension_start(parser: &Parser<'_>) -> bool
pub fn parse_extension_expression(parser: &mut Parser<'_>) -> Result<Expression, ()>
pub fn parse_extension_statement(parser: &mut Parser<'_>) -> Result<Statement, ()>
pub fn parse_extension_block(parser: &mut Parser<'_>) -> Result<Statement, ()>
pub fn parse_statement_expression(parser: &mut Parser<'_>) -> Result<Expression, ()>
pub fn parse_label_address(parser: &mut Parser<'_>) -> Result<Expression, ()>
pub fn parse_computed_goto(parser: &mut Parser<'_>) -> Result<Statement, ()>
pub fn parse_case_range(parser: &mut Parser<'_>) -> Result<Statement, ()>
pub fn parse_typeof_specifier(parser: &mut Parser<'_>) -> Result<TypeSpecifier, ()>
```

### 私有函数

```rust
fn parse_local_label_decl(parser: &mut Parser<'_>) -> Result<Statement, ()>
fn get_expression_span(expr: &Expression) -> Span
fn is_declaration_start_after_extension(parser: &Parser<'_>) -> bool
```