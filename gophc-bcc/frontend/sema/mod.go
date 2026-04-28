package sema

import (
	"fmt"

	"github.com/gophc/blitzy-c-compiler/gophc-bcc/common"
)

type SemanticAnalyzer struct {
	diagnostics *common.DiagnosticEngine
	typeBuilder *common.TypeBuilder
	target      common.Target
	interner    *common.Interner
	scopes      *ScopeStack
	symbols     *SymbolTable
}

func NewSemanticAnalyzer(diagnostics *common.DiagnosticEngine, typeBuilder *common.TypeBuilder, target common.Target, interner *common.Interner) *SemanticAnalyzer {
	return &SemanticAnalyzer{
		diagnostics: diagnostics,
		typeBuilder: typeBuilder,
		target:      target,
		interner:    interner,
		scopes:      NewScopeStack(),
		symbols:     NewSymbolTable(),
	}
}

func (s *SemanticAnalyzer) Analyze(translationUnit *TranslationUnit) error {
	for _, decl := range translationUnit.Declarations {
		s.analyzeExternalDeclaration(decl)
	}

	if s.diagnostics.HasErrors() {
		return fmt.Errorf("semantic analysis failed")
	}
	return nil
}

func (s *SemanticAnalyzer) analyzeExternalDeclaration(decl ExternalDeclaration) error {
	return nil
}

type TranslationUnit struct {
	Declarations []ExternalDeclaration
}

type ExternalDeclaration any

type FunctionDefinition struct {
	Specifiers DeclarationSpecifiers
	Declarator Declarator
	Body       *CompoundStatement
	Span       common.Span
}

type DeclarationSpecifiers struct {
	TypeSpecifiers []any
	Attributes     []any
}

type Declaration struct {
	Specifiers  DeclarationSpecifiers
	Declarators []InitDeclarator
	Span        common.Span
}

type Declarator struct {
	Pointer    *Pointer
	DirectDecl any
	Attributes []any
}

type Pointer struct {
	Inner *Pointer
	Quals common.TypeQualifiers
}

type DirectDeclarator struct {
	Kind       any
	Params     []any
	OldStyle   []any
	Attributes []any
}

type Param struct {
	Attrs []any
	Ty    common.CType
	Name  *string
	Decl  any
}

type CompoundStatement struct {
	Statements []any
	Span       common.Span
}

type Statement any

type InitDeclarator struct {
	Declarator  any
	Initializer *any
	Span        common.Span
}

type Initializer any

type AsmStatement struct {
	Expression any
	Span       common.Span
}

type EmptyDeclaration struct{}

type TypeSpecifier any
