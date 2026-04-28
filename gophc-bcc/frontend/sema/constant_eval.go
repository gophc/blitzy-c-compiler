package sema

import (
	"errors"

	"github.com/gophc/blitzy-c-compiler/gophc-bcc/common"
)

type ConstValue interface {
	toI128() (int64, bool)
	toU128() (uint64, bool)
	toBool() bool
}

type ConstValueSigned struct {
	Value int64
}

func (v *ConstValueSigned) toI128() (int64, bool) {
	return v.Value, true
}

func (v *ConstValueSigned) toU128() (uint64, bool) {
	if v.Value < 0 {
		return 0, false
	}
	return uint64(v.Value), true
}

func (v *ConstValueSigned) toBool() bool {
	return v.Value != 0
}

type ConstValueUnsigned struct {
	Value uint64
}

func (v *ConstValueUnsigned) toI128() (int64, bool) {
	if v.Value > 0x7FFFFFFFFFFFFFFF {
		return 0, false
	}
	return int64(v.Value), true
}

func (v *ConstValueUnsigned) toU128() (uint64, bool) {
	return v.Value, true
}

func (v *ConstValueUnsigned) toBool() bool {
	return v.Value != 0
}

type ConstValueFloat struct {
	Value float64
}

func (v *ConstValueFloat) toI128() (int64, bool) {
	return int64(v.Value), true
}

func (v *ConstValueFloat) toU128() (uint64, bool) {
	if v.Value < 0 {
		return 0, false
	}
	return uint64(v.Value), true
}

func (v *ConstValueFloat) toBool() bool {
	return v.Value != 0 && v.Value == v.Value
}

type ConstValueAddress struct {
	Symbol common.Symbol
	Offset int64
}

func (v *ConstValueAddress) toI128() (int64, bool) {
	return 0, false
}

func (v *ConstValueAddress) toU128() (uint64, bool) {
	return 0, false
}

func (v *ConstValueAddress) toBool() bool {
	return true
}

type ConstEvaluator struct {
	diagnostics   *common.DiagnosticEngine
	target        common.Target
	enumValues    map[common.Symbol]int64
	tagTypes      map[common.Symbol]common.CType
	variableTypes map[common.Symbol]common.CType
	typedefTypes  map[common.Symbol]common.CType
}

func NewConstEvaluator(diagnostics *common.DiagnosticEngine, target common.Target) *ConstEvaluator {
	return &ConstEvaluator{
		diagnostics:   diagnostics,
		target:        target,
		enumValues:    make(map[common.Symbol]int64),
		tagTypes:      make(map[common.Symbol]common.CType),
		variableTypes: make(map[common.Symbol]common.CType),
		typedefTypes:  make(map[common.Symbol]common.CType),
	}
}

func (ce *ConstEvaluator) RegisterEnumValue(name common.Symbol, value int64) {
	ce.enumValues[name] = value
}

func (ce *ConstEvaluator) RegisterTagType(tag common.Symbol, ty common.CType) {
	ce.tagTypes[tag] = ty
}

func (ce *ConstEvaluator) RegisterVariableType(name common.Symbol, ty common.CType) {
	ce.variableTypes[name] = ty
}

func (ce *ConstEvaluator) RegisterTypedefType(name common.Symbol, ty common.CType) {
	ce.typedefTypes[name] = ty
}

func (ce *ConstEvaluator) EvaluateIntegerConstant(expr any, span common.Span) (int64, error) {
	value, err := ce.EvaluateConstantExpr(expr)
	if err != nil {
		return 0, err
	}
	if v, ok := value.(*ConstValueSigned); ok {
		return v.Value, nil
	}
	if v, ok := value.(*ConstValueUnsigned); ok {
		return int64(v.Value), nil
	}
	if v, ok := value.(*ConstValueFloat); ok {
		return int64(v.Value), nil
	}
	return 0, errors.New("not an integer constant expression")
}

func (ce *ConstEvaluator) EvaluateConstantExpr(expr any) (ConstValue, error) {
	switch expr.(type) {
	case nil:
		return &ConstValueSigned{Value: 0}, nil
	}
	return nil, errors.New("not a compile-time constant")
}

func (ce *ConstEvaluator) IsConstantExpression(expr any) bool {
	return false
}

func (ce *ConstEvaluator) ValidateArraySize(expr any, span common.Span) (int, error) {
	value, err := ce.EvaluateIntegerConstant(expr, span)
	if err != nil {
		return 0, err
	}
	if value < 0 {
		return 0, errors.New("array size must be positive")
	}
	if value == 0 {
		ce.diagnostics.EmitWarning(span, "zero-length array is a GCC extension")
		return 0, nil
	}
	return int(value), nil
}

func (ce *ConstEvaluator) ValidateBitfieldWidth(expr any, baseType common.CType, span common.Span) (uint32, error) {
	value, err := ce.EvaluateIntegerConstant(expr, span)
	if err != nil {
		return 0, err
	}
	if value < 0 {
		return 0, errors.New("bitfield width must be non-negative")
	}
	maxWidth := common.SizeofCType(baseType, &ce.target) * 8
	if value > int64(maxWidth) {
		return 0, errors.New("bitfield width exceeds the width of type")
	}
	return uint32(value), nil
}

func (ce *ConstEvaluator) EvaluateStaticAssert(expr any, message *string, span common.Span) error {
	value, err := ce.EvaluateIntegerConstant(expr, span)
	if err != nil {
		return err
	}
	if value == 0 {
		return errors.New("static assertion failed")
	}
	return nil
}
