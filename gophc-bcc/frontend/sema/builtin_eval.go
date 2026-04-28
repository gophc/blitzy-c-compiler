package sema

import (
	"math"

	"github.com/gophc/blitzy-c-compiler/gophc-bcc/common"
)

type BuiltinResult interface {
	isBuiltinResult()
}

type BuiltinResultConstantInt struct {
	Value int64
}

func (*BuiltinResultConstantInt) isBuiltinResult() {}

type BuiltinResultConstantBool struct {
	Value bool
}

func (*BuiltinResultConstantBool) isBuiltinResult() {}

type BuiltinResultConstantFloat struct {
	Value float64
}

func (*BuiltinResultConstantFloat) isBuiltinResult() {}

type BuiltinResultResolvedType struct {
	Ty common.CType
}

func (*BuiltinResultResolvedType) isBuiltinResult() {}

type BuiltinResultRuntimeCall struct {
	Builtin    string
	ResultType common.CType
}

func (*BuiltinResultRuntimeCall) isBuiltinResult() {}

type BuiltinResultNoValue struct{}

func (*BuiltinResultNoValue) isBuiltinResult() {}

type BuiltinEvaluator struct {
	diagnostics *common.DiagnosticEngine
	typeBuilder *common.TypeBuilder
	target      common.Target
}

func NewBuiltinEvaluator(diagnostics *common.DiagnosticEngine, typeBuilder *common.TypeBuilder, target common.Target) *BuiltinEvaluator {
	return &BuiltinEvaluator{
		diagnostics: diagnostics,
		typeBuilder: typeBuilder,
		target:      target,
	}
}

func (be *BuiltinEvaluator) EvaluateBuiltin(builtin string, args []any, span common.Span) (BuiltinResult, error) {
	switch builtin {
	case "__builtin_constant_p":
		return be.evalConstantP(args, span)
	case "__builtin_types_compatible_p":
		return be.evalTypesCompatibleP(args, span)
	case "__builtin_choose_expr":
		return be.evalChooseExpr(args, span)
	case "__builtin_offsetof":
		return be.evalOffsetof(args, span)
	case "__builtin_expect", "__builtin_unreachable", "__builtin_prefetch",
		"__builtin_va_start", "__builtin_va_end", "__builtin_va_arg", "__builtin_va_copy",
		"__builtin_frame_address", "__builtin_return_address", "__builtin_trap",
		"__builtin_assume_aligned", "__builtin_add_overflow", "__builtin_sub_overflow",
		"__builtin_mul_overflow", "__builtin_object_size", "__builtin_extract_return_addr":
		return &BuiltinResultRuntimeCall{Builtin: builtin, ResultType: common.CTypeVoid}, nil
	case "__builtin_clz", "__builtin_clzl", "__builtin_clzll",
		"__builtin_ctz", "__builtin_ctzl", "__builtin_ctzll",
		"__builtin_popcount", "__builtin_popcountl", "__builtin_popcountll",
		"__builtin_ffs", "__builtin_ffsll":
		return &BuiltinResultRuntimeCall{Builtin: builtin, ResultType: common.CTypeInt}, nil
	case "__builtin_bswap16":
		return &BuiltinResultRuntimeCall{Builtin: builtin, ResultType: common.CTypeUShort}, nil
	case "__builtin_bswap32":
		return &BuiltinResultRuntimeCall{Builtin: builtin, ResultType: common.CTypeUInt}, nil
	case "__builtin_bswap64":
		return &BuiltinResultRuntimeCall{Builtin: builtin, ResultType: common.CTypeULongLong}, nil
	case "__builtin_inf", "__builtin_huge_val":
		return &BuiltinResultConstantFloat{Value: math.Inf(1)}, nil
	case "__builtin_nan":
		return &BuiltinResultConstantFloat{Value: math.NaN()}, nil
	}
	return &BuiltinResultRuntimeCall{Builtin: builtin, ResultType: common.CTypeInt}, nil
}

func (be *BuiltinEvaluator) GetBuiltinReturnType(builtin string) common.CType {
	switch builtin {
	case "__builtin_constant_p", "__builtin_types_compatible_p":
		return common.CTypeInt
	case "__builtin_offsetof":
		if be.target.Is64Bit() {
			return common.CTypeULong
		}
		return common.CTypeUInt
	case "__builtin_clz", "__builtin_clzl", "__builtin_clzll",
		"__builtin_ctz", "__builtin_ctzl", "__builtin_ctzll",
		"__builtin_popcount", "__builtin_popcountl", "__builtin_popcountll",
		"__builtin_ffs", "__builtin_ffsll":
		return common.CTypeInt
	case "__builtin_bswap16":
		return common.CTypeUShort
	case "__builtin_bswap32":
		return common.CTypeUInt
	case "__builtin_bswap64":
		return common.CTypeULongLong
	case "__builtin_expect":
		return common.CTypeLong
	case "__builtin_unreachable", "__builtin_prefetch",
		"__builtin_trap", "__builtin_va_start", "__builtin_va_end",
		"__builtin_va_copy":
		return common.CTypeVoid
	case "__builtin_frame_address", "__builtin_return_address",
		"__builtin_assume_aligned":
		return &common.CTypePointer{Pointee: common.CTypeVoid, Quals: common.TypeQualifiers{}}
	case "__builtin_add_overflow", "__builtin_sub_overflow",
		"__builtin_mul_overflow":
		return common.CTypeBool
	case "__builtin_object_size":
		if be.target.Is64Bit() {
			return common.CTypeULong
		}
		return common.CTypeUInt
	case "__builtin_inf", "__builtin_huge_val":
		return common.CTypeDouble
	case "__builtin_inff", "__builtin_huge_valf":
		return common.CTypeFloat
	case "__builtin_infl", "__builtin_huge_vall":
		return common.CTypeLongDouble
	case "__builtin_nan", "__builtin_nanf", "__builtin_nanl":
		return common.CTypeDouble
	}
	return common.CTypeInt
}

func (be *BuiltinEvaluator) IsBuiltinName(name string) bool {
	names := map[string]bool{
		"__builtin_constant_p":         true,
		"__builtin_types_compatible_p": true,
		"__builtin_choose_expr":        true,
		"__builtin_offsetof":           true,
		"__builtin_clz":                true,
		"__builtin_clzl":               true,
		"__builtin_clzll":              true,
		"__builtin_ctz":                true,
		"__builtin_ctzl":               true,
		"__builtin_ctzll":              true,
		"__builtin_popcount":           true,
		"__builtin_popcountl":          true,
		"__builtin_popcountll":         true,
		"__builtin_ffs":                true,
		"__builtin_ffsll":              true,
		"__builtin_expect":             true,
		"__builtin_unreachable":        true,
		"__builtin_prefetch":           true,
		"__builtin_va_start":           true,
		"__builtin_va_end":             true,
		"__builtin_va_arg":             true,
		"__builtin_va_copy":            true,
		"__builtin_frame_address":      true,
		"__builtin_return_address":     true,
		"__builtin_trap":               true,
		"__builtin_assume_aligned":     true,
	}
	return names[name]
}

func (be *BuiltinEvaluator) evalConstantP(args []any, span common.Span) (BuiltinResult, error) {
	if len(args) != 1 {
		be.diagnostics.EmitError(span, "__builtin_constant_p requires 1 argument")
		return nil, nil
	}
	return &BuiltinResultConstantInt{Value: 0}, nil
}

func (be *BuiltinEvaluator) evalTypesCompatibleP(args []any, span common.Span) (BuiltinResult, error) {
	if len(args) != 2 {
		be.diagnostics.EmitError(span, "__builtin_types_compatible_p requires 2 arguments")
		return nil, nil
	}
	return &BuiltinResultConstantInt{Value: 0}, nil
}

func (be *BuiltinEvaluator) evalChooseExpr(args []any, span common.Span) (BuiltinResult, error) {
	if len(args) != 3 {
		be.diagnostics.EmitError(span, "__builtin_choose_expr requires 3 arguments")
		return nil, nil
	}
	return &BuiltinResultConstantBool{Value: false}, nil
}

func (be *BuiltinEvaluator) evalOffsetof(args []any, span common.Span) (BuiltinResult, error) {
	if len(args) != 2 {
		be.diagnostics.EmitError(span, "__builtin_offsetof requires 2 arguments")
		return nil, nil
	}
	return &BuiltinResultConstantInt{Value: 0}, nil
}

func (be *BuiltinEvaluator) validateArgCount(name string, args []any, expected int, span common.Span) error {
	if len(args) != expected {
		be.diagnostics.EmitError(span, name+" requires 1 argument(s)")
		return nil
	}
	return nil
}
