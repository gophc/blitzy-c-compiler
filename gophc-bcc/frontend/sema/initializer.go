package sema

import (
	"github.com/gophc/blitzy-c-compiler/gophc-bcc/common"
)

type AnalyzedInit interface {
	isAnalyzedInit()
}

type AnalyzedInitExpression struct {
	Expr       any
	TargetType common.CType
}

func (*AnalyzedInitExpression) isAnalyzedInit() {}

type AnalyzedInitStruct struct {
	Fields []struct {
		Index int
		Init  AnalyzedInit
	}
	StructType common.CType
}

func (*AnalyzedInitStruct) isAnalyzedInit() {}

type AnalyzedInitUnion struct {
	FieldIndex int
	Init       AnalyzedInit
	UnionType  common.CType
}

func (*AnalyzedInitUnion) isAnalyzedInit() {}

type AnalyzedInitArray struct {
	Elements []struct {
		Index int
		Init  AnalyzedInit
	}
	ArrayType common.CType
	ArraySize int
}

func (*AnalyzedInitArray) isAnalyzedInit() {}

type AnalyzedInitZero struct {
	TargetType common.CType
}

func (*AnalyzedInitZero) isAnalyzedInit() {}

type InitializerAnalyzer struct {
	diagnostics *common.DiagnosticEngine
	typeBuilder *common.TypeBuilder
	target      common.Target
	interner    *common.Interner
}

func NewInitializerAnalyzer(diagnostics *common.DiagnosticEngine, typeBuilder *common.TypeBuilder, target common.Target, interner *common.Interner) *InitializerAnalyzer {
	return &InitializerAnalyzer{
		diagnostics: diagnostics,
		typeBuilder: typeBuilder,
		target:      target,
		interner:    interner,
	}
}

func (ia *InitializerAnalyzer) AnalyzeInitializer(init any, targetType common.CType, span common.Span) (AnalyzedInit, error) {
	resolved := ia.resolveType(targetType)
	switch init := init.(type) {
	case InitExpression:
		return ia.analyzeSimpleInit(init.Expr, resolved, span)
	case InitList:
		return ia.analyzeInitList(init.DesignatorsAndInitializers, resolved, span)
	}
	return nil, nil
}

func (ia *InitializerAnalyzer) IsConstantInitializer(init AnalyzedInit) bool {
	switch v := init.(type) {
	case *AnalyzedInitExpression:
		return ia.isConstantExpression(v.Expr)
	case *AnalyzedInitStruct:
		for _, f := range v.Fields {
			if !ia.IsConstantInitializer(f.Init) {
				return false
			}
		}
		return true
	case *AnalyzedInitUnion:
		return ia.IsConstantInitializer(v.Init)
	case *AnalyzedInitArray:
		for _, e := range v.Elements {
			if !ia.IsConstantInitializer(e.Init) {
				return false
			}
		}
		return true
	case *AnalyzedInitZero:
		return true
	}
	return false
}

func (ia *InitializerAnalyzer) analyzeSimpleInit(expr any, targetType common.CType, span common.Span) (AnalyzedInit, error) {
	return &AnalyzedInitExpression{Expr: expr, TargetType: targetType}, nil
}

func (ia *InitializerAnalyzer) analyzeInitList(items []any, targetType common.CType, span common.Span) (AnalyzedInit, error) {
	switch ty := targetType.(type) {
	case *common.CTypeStruct:
		return ia.analyzeStructInit(items, ty.Fields, targetType, span)
	case *common.CTypeUnion:
		return ia.analyzeUnionInit(items, ty.Fields, targetType, span)
	case *common.CTypeArray:
		return ia.analyzeArrayInit(items, ty.Elem, ty.Size, targetType, span)
	}
	if len(items) == 0 {
		return &AnalyzedInitZero{TargetType: targetType}, nil
	}
	return nil, nil
}

func (ia *InitializerAnalyzer) analyzeStructInit(items []any, fields []common.StructField, structType common.CType, span common.Span) (AnalyzedInit, error) {
	numFields := len(fields)
	fieldInits := make([]AnalyzedInit, numFields)
	currentField := 0

	for i, item := range items {
		if i >= numFields {
			break
		}
		fieldTy := fields[currentField].Ty
		resolvedFieldTy := ia.resolveType(fieldTy)
		init, _ := ia.AnalyzeInitializer(item, resolvedFieldTy, span)
		fieldInits[currentField] = init
		currentField++
	}

	resultFields := make([]struct {
		Index int
		Init  AnalyzedInit
	}, 0)
	for i, init := range fieldInits {
		if init == nil {
			init = &AnalyzedInitZero{TargetType: fields[i].Ty}
		}
		resultFields = append(resultFields, struct {
			Index int
			Init  AnalyzedInit
		}{i, init})
	}

	return &AnalyzedInitStruct{Fields: resultFields, StructType: structType}, nil
}

func (ia *InitializerAnalyzer) analyzeUnionInit(items []any, fields []common.StructField, unionType common.CType, span common.Span) (AnalyzedInit, error) {
	if len(items) == 0 || len(fields) == 0 {
		return &AnalyzedInitZero{TargetType: unionType}, nil
	}

	fieldIndex := 0
	fieldTy := fields[fieldIndex].Ty
	resolved := ia.resolveType(fieldTy)
	init, _ := ia.AnalyzeInitializer(items[0], resolved, span)

	return &AnalyzedInitUnion{FieldIndex: fieldIndex, Init: init, UnionType: unionType}, nil
}

func (ia *InitializerAnalyzer) analyzeArrayInit(items []any, elemType common.CType, knownSize *int, arrayType common.CType, span common.Span) (AnalyzedInit, error) {
	resolvedElem := ia.resolveType(elemType)
	elementInits := make([]struct {
		Index int
		Init  AnalyzedInit
	}, 0)

	for i, item := range items {
		init, _ := ia.AnalyzeInitializer(item, resolvedElem, span)
		elementInits = append(elementInits, struct {
			Index int
			Init  AnalyzedInit
		}{i, init})
	}

	finalSize := len(items)
	if knownSize != nil {
		finalSize = *knownSize
	}

	resultElements := make([]struct {
		Index int
		Init  AnalyzedInit
	}, 0)
	for i := 0; i < finalSize; i++ {
		found := false
		for _, e := range elementInits {
			if e.Index == i {
				resultElements = append(resultElements, e)
				found = true
				break
			}
		}
		if !found {
			resultElements = append(resultElements, struct {
				Index int
				Init  AnalyzedInit
			}{i, &AnalyzedInitZero{TargetType: elemType}})
		}
	}

	return &AnalyzedInitArray{Elements: resultElements, ArrayType: arrayType, ArraySize: finalSize}, nil
}

func (ia *InitializerAnalyzer) resolveType(ty common.CType) common.CType {
	return common.ResolveAndStrip(ty)
}

func (ia *InitializerAnalyzer) isCharElementType(ty common.CType) bool {
	switch ty.(type) {
	case common.CTypeBase:
		return ty == common.CTypeChar || ty == common.CTypeSChar || ty == common.CTypeUChar
	}
	return false
}

func (ia *InitializerAnalyzer) isAggregateType(ty common.CType) bool {
	switch ty.(type) {
	case *common.CTypeStruct, *common.CTypeUnion, *common.CTypeArray:
		return true
	}
	return false
}

func (ia *InitializerAnalyzer) findFieldIndex(fields []common.StructField, name *string) int {
	for i, field := range fields {
		if field.Name != nil && name != nil && *field.Name == *name {
			return i
		}
	}
	return -1
}

func (ia *InitializerAnalyzer) isConstantExpression(expr any) bool {
	return false
}

type InitExpression struct {
	Expr any
}

type InitList struct {
	DesignatorsAndInitializers []any
}
