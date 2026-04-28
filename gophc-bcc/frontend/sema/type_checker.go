package sema

import (
	"errors"

	"github.com/gophc/blitzy-c-compiler/gophc-bcc/common"
)

type TypeChecker struct {
	diagnostics *common.DiagnosticEngine
	typeBuilder *common.TypeBuilder
	target      common.Target
}

func NewTypeChecker(diagnostics *common.DiagnosticEngine, typeBuilder *common.TypeBuilder, target common.Target) *TypeChecker {
	return &TypeChecker{
		diagnostics: diagnostics,
		typeBuilder: typeBuilder,
		target:      target,
	}
}

func (tc *TypeChecker) SizeTType() common.CType {
	return common.CTypeULong
}

func (tc *TypeChecker) PtrdiffTType() common.CType {
	return common.CTypeLong
}

func (tc *TypeChecker) StripType(ty common.CType) common.CType {
	return common.ResolveAndStrip(ty)
}

func (tc *TypeChecker) DecayType(ty common.CType) common.CType {
	resolved := common.ResolveTypedef(ty)
	switch v := resolved.(type) {
	case *common.CTypeArray:
		return &common.CTypePointer{Pointee: v.Elem, Quals: common.TypeQualifiers{}}
	case *common.CTypeFunction:
		return &common.CTypePointer{Pointee: resolved, Quals: common.TypeQualifiers{}}
	case *common.CTypeQualified:
		return tc.DecayType(v.Inner)
	}
	return common.Unqualified(resolved)
}

func (tc *TypeChecker) GetQualifiers(ty common.CType) common.TypeQualifiers {
	resolved := common.ResolveTypedef(ty)
	switch v := resolved.(type) {
	case *common.CTypeQualified:
		return v.Quals
	case *common.CTypePointer:
		return v.Quals
	}
	return common.TypeQualifiers{}
}

func (tc *TypeChecker) IntegerLiteralType(value uint64, suffix int, isHexOrOctal bool) common.CType {
	longSz := tc.target.LongSize()
	switch suffix {
	case 0:
		if value <= 0x7FFFFFFF {
			return common.CTypeInt
		} else if isHexOrOctal && value <= 0xFFFFFFFF {
			return common.CTypeUInt
		} else if longSz == 8 && value <= 0x7FFFFFFFFFFFFFFF {
			return common.CTypeLong
		} else if longSz == 8 && isHexOrOctal && value <= 0xFFFFFFFFFFFFFFFF {
			return common.CTypeULong
		} else if longSz != 8 && value <= 0x7FFFFFFF {
			return common.CTypeLong
		} else if isHexOrOctal && value <= 0xFFFFFFFF {
			return common.CTypeULong
		} else if value <= 0x7FFFFFFFFFFFFFFF {
			return common.CTypeLongLong
		} else if isHexOrOctal && value <= 0xFFFFFFFFFFFFFFFF {
			return common.CTypeULongLong
		}
		return common.CTypeULongLong
	case 1:
		if value <= 0xFFFFFFFF {
			return common.CTypeUInt
		} else if longSz == 8 && value <= 0xFFFFFFFFFFFFFFFF {
			return common.CTypeULong
		}
		return common.CTypeULongLong
	case 2:
		if longSz == 8 {
			if value <= 0x7FFFFFFFFFFFFFFF {
				return common.CTypeLong
			} else if isHexOrOctal && value <= 0xFFFFFFFFFFFFFFFF {
				return common.CTypeULong
			}
			return common.CTypeLongLong
		} else if value <= 0x7FFFFFFF {
			return common.CTypeLong
		} else if isHexOrOctal && value <= 0xFFFFFFFF {
			return common.CTypeULongLong
		}
		return common.CTypeLongLong
	case 3:
		return common.CTypeULong
	case 4:
		if value <= 0x7FFFFFFFFFFFFFFF {
			return common.CTypeLongLong
		}
		return common.CTypeULongLong
	case 5:
		return common.CTypeULongLong
	}
	return common.CTypeInt
}

func (tc *TypeChecker) FloatLiteralType(suffix int) common.CType {
	switch suffix {
	case 0:
		return common.CTypeDouble
	case 1:
		return common.CTypeFloat
	case 2:
		return common.CTypeLongDouble
	}
	return common.CTypeDouble
}

func (tc *TypeChecker) StringLiteralType(prefix int) common.CType {
	charType := common.CTypeChar
	switch prefix {
	case 0, 4:
		charType = common.CTypeChar
	case 1:
		charType = common.CTypeInt
	case 2:
		charType = common.CTypeUShort
	case 3:
		charType = common.CTypeUInt
	}
	return &common.CTypePointer{
		Pointee: charType,
		Quals:   common.TypeQualifiers{IsConst: true},
	}
}

func (tc *TypeChecker) IntegerPromotion(ty common.CType) common.CType {
	return common.IntegerPromotion(ty)
}

func (tc *TypeChecker) UsualArithmeticConversion(lhs, rhs common.CType) common.CType {
	return common.UsualArithmeticConversion(lhs, rhs)
}

func (tc *TypeChecker) CheckArraySubscript(baseTy common.CType, indexTy common.CType, span common.Span) (common.CType, error) {
	pointee := common.ResolveAndStrip(baseTy)
	if pointee != nil {
		return pointee, nil
	}
	stripped := common.ResolveAndStrip(baseTy)
	isPrimitive := false
	switch stripped {
	case common.CTypeInt, common.CTypeUInt, common.CTypeLong, common.CTypeULong, common.CTypeChar, common.CTypeVoid:
		isPrimitive = true
	}
	if isPrimitive {
		return common.CTypeInt, nil
	}
	return nil, errors.New("subscripted value is not an array or pointer")
}

func (tc *TypeChecker) CheckConditional(thenTy common.CType, elseTy common.CType, span common.Span) (common.CType, error) {
	if common.IsScalar(common.ResolveAndStrip(thenTy)) && common.IsScalar(common.ResolveAndStrip(elseTy)) {
		return tc.UsualArithmeticConversion(thenTy, elseTy), nil
	}
	return nil, errors.New("operands to conditional operator must have scalar types")
}

func (tc *TypeChecker) CheckAssignment(targetTy common.CType, valueTy common.CType, span common.Span) error {
	if !tc.IsImplicitlyConvertible(valueTy, targetTy) {
		return errors.New("incompatible types in assignment")
	}
	return nil
}

func (tc *TypeChecker) DefaultArgumentPromotion(ty common.CType) common.CType {
	resolved := common.ResolveTypedef(ty)
	switch v := resolved.(type) {
	case common.CTypeBase:
		switch v {
		case common.CTypeFloat:
			return common.CTypeDouble
		case common.CTypeChar, common.CTypeSChar, common.CTypeUInt8,
			common.CTypeShort, common.CTypeUShort:
			return common.CTypeInt
		}
	}
	return ty
}

func (tc *TypeChecker) AreTypesCompatible(a, b common.CType) bool {
	return common.IsCompatible(a, b)
}

func (tc *TypeChecker) IsImplicitlyConvertible(from, to common.CType) bool {
	fromStripped := tc.StripType(from)
	toStripped := tc.StripType(to)

	if tc.AreTypesCompatible(fromStripped, toStripped) {
		return true
	}

	if common.IsArithmetic(fromStripped) && common.IsArithmetic(toStripped) {
		return true
	}

	if common.IsPointer(fromStripped) && common.IsPointer(toStripped) {
		return true
	}

	if common.IsInteger(fromStripped) && common.IsInteger(toStripped) {
		return true
	}

	return false
}

func (tc *TypeChecker) CheckBinaryOp(op int, lhsTy, rhsTy common.CType, span common.Span) (common.CType, error) {
	lhs := tc.StripType(lhsTy)
	rhs := tc.StripType(rhsTy)

	switch op {
	case 0:
		if common.IsPointer(lhsTy) && common.IsInteger(rhsTy) {
			return tc.CheckPointerArithmetic(lhsTy, rhsTy, op, span)
		}
		if common.IsInteger(lhsTy) && common.IsPointer(rhsTy) {
			return tc.CheckPointerArithmetic(rhsTy, lhsTy, op, span)
		}
		if common.IsArithmetic(lhs) && common.IsArithmetic(rhs) {
			return tc.UsualArithmeticConversion(lhsTy, rhsTy), nil
		}
		return nil, errors.New("invalid operands to binary '+'")

	case 1:
		if common.IsPointer(lhsTy) && common.IsInteger(rhsTy) {
			return tc.CheckPointerArithmetic(lhsTy, rhsTy, op, span)
		}
		if common.IsPointer(lhsTy) && common.IsPointer(rhsTy) {
			return tc.PtrdiffTType(), nil
		}
		if common.IsArithmetic(lhs) && common.IsArithmetic(rhs) {
			return tc.UsualArithmeticConversion(lhsTy, rhsTy), nil
		}
		return nil, errors.New("invalid operands to binary '-'")

	case 2, 3:
		if common.IsArithmetic(lhs) && common.IsArithmetic(rhs) {
			return tc.UsualArithmeticConversion(lhsTy, rhsTy), nil
		}
		return nil, errors.New("invalid operands to binary '*'")

	case 4:
		if common.IsInteger(lhs) && common.IsInteger(rhs) {
			return tc.UsualArithmeticConversion(lhsTy, rhsTy), nil
		}
		return nil, errors.New("invalid operands to binary '%'")

	case 5, 6:
		if common.IsInteger(lhs) && common.IsInteger(rhs) {
			return tc.IntegerPromotion(lhsTy), nil
		}
		return nil, errors.New("invalid operands to shift operator")

	case 7, 8, 9, 10, 11, 12:
		if common.IsArithmetic(lhs) && common.IsArithmetic(rhs) {
			return common.CTypeInt, nil
		}
		if common.IsPointer(lhs) && common.IsPointer(rhs) {
			return common.CTypeInt, nil
		}
		return nil, errors.New("invalid operands to comparison operator")

	case 13, 14, 15:
		if common.IsInteger(lhs) && common.IsInteger(rhs) {
			return tc.UsualArithmeticConversion(lhsTy, rhsTy), nil
		}
		return nil, errors.New("invalid operands to bitwise operator")

	case 16, 17:
		if common.IsScalar(lhs) && common.IsScalar(rhs) {
			return common.CTypeInt, nil
		}
		return nil, errors.New("invalid operands to logical operator")
	}
	return nil, nil
}

func (tc *TypeChecker) CheckPointerArithmetic(ptrTy, intTy common.CType, op int, span common.Span) (common.CType, error) {
	if op == 0 {
		return ptrTy, nil
	}
	return tc.PtrdiffTType(), nil
}

func (tc *TypeChecker) CheckUnaryOp(op int, operandTy common.CType, span common.Span) (common.CType, error) {
	resolved := tc.StripType(operandTy)

	switch op {
	case 0:
		return &common.CTypePointer{Pointee: common.ResolveTypedef(operandTy), Quals: common.TypeQualifiers{}}, nil
	case 1:
		if ptr, ok := resolved.(*common.CTypePointer); ok {
			return ptr.Pointee, nil
		}
		return nil, errors.New("indirection requires pointer operand")
	case 2, 3:
		if common.IsArithmetic(resolved) {
			return tc.IntegerPromotion(operandTy), nil
		}
		return nil, errors.New("unary operator requires arithmetic operand")
	case 4:
		if common.IsInteger(resolved) {
			return tc.IntegerPromotion(operandTy), nil
		}
		return nil, errors.New("bitwise '~' requires integer operand")
	case 5:
		if common.IsScalar(resolved) {
			return common.CTypeInt, nil
		}
		return nil, errors.New("logical '!' requires scalar operand")
	}
	return nil, nil
}

func (tc *TypeChecker) CheckFunctionCall(calleeType common.CType, args []common.CType, span common.Span) (common.CType, error) {
	resolved := tc.StripType(calleeType)

	var funcType common.CType
	if ptr, ok := resolved.(*common.CTypePointer); ok {
		funcType = tc.StripType(ptr.Pointee)
	} else {
		funcType = resolved
	}

	switch ft := funcType.(type) {
	case *common.CTypeFunction:
		if !ft.Variadic && len(args) != len(ft.Params) {
			return nil, errors.New("wrong number of arguments")
		}
		if ft.Variadic && len(args) < len(ft.Params) {
			return nil, errors.New("too few arguments")
		}
		return ft.ReturnType, nil
	}
	return nil, errors.New("called object is not a function or function pointer")
}

func (tc *TypeChecker) CheckMemberAccess(structType common.CType, memberName string, isArrow bool, span common.Span) (common.CType, error) {
	resolved := tc.StripType(structType)

	var actualType common.CType
	if isArrow {
		if ptr, ok := resolved.(*common.CTypePointer); ok {
			actualType = tc.StripType(ptr.Pointee)
		} else {
			return nil, errors.New("member reference through '->' requires pointer type")
		}
	} else {
		actualType = resolved
	}

	var fields []common.StructField
	switch st := actualType.(type) {
	case *common.CTypeStruct:
		fields = st.Fields
	case *common.CTypeUnion:
		fields = st.Fields
	default:
		opStr := "."
		if isArrow {
			opStr = "->"
		}
		return nil, errors.New("member reference base type is not a struct or union (operator '" + opStr + "')")
	}

	for _, field := range fields {
		if field.Name != nil && *field.Name == memberName {
			return field.Ty, nil
		}
	}

	return nil, errors.New("no member named '" + memberName + "' in struct/union")
}
