package common

type TypeQualifiers struct {
	IsConst    bool
	IsVolatile bool
	IsRestrict bool
	IsAtomic   bool
}

func (q TypeQualifiers) IsEmpty() bool {
	return !q.IsConst && !q.IsVolatile && !q.IsRestrict && !q.IsAtomic
}

func (q TypeQualifiers) Merge(other TypeQualifiers) TypeQualifiers {
	return TypeQualifiers{
		IsConst:    q.IsConst || other.IsConst,
		IsVolatile: q.IsVolatile || other.IsVolatile,
		IsRestrict: q.IsRestrict || other.IsRestrict,
		IsAtomic:   q.IsAtomic || other.IsAtomic,
	}
}

type StructField struct {
	Name     *string
	Ty       CType
	BitWidth *uint32
}

type CType interface {
	isCType()
}

type CTypeBase int

const (
	CTypeVoid CTypeBase = iota
	CTypeBool
	CTypeChar
	CTypeSChar
	CTypeUChar
	CTypeShort
	CTypeUShort
	CTypeInt
	CTypeUInt
	CTypeLong
	CTypeULong
	CTypeLongLong
	CTypeULongLong
	CTypeInt128
	CTypeUInt128
	CTypeFloat
	CTypeDouble
	CTypeLongDouble
	CTypeEnum_
)

func (CTypeBase) isCType() {}

type CTypeComplex struct {
	Base CType
}

func (*CTypeComplex) isCType() {}

type CTypePointer struct {
	Pointee CType
	Quals   TypeQualifiers
}

func (*CTypePointer) isCType() {}

type CTypeArray struct {
	Elem CType
	Size *int
}

func (*CTypeArray) isCType() {}

type CTypeFunction struct {
	ReturnType CType
	Params     []CType
	Variadic   bool
}

func (*CTypeFunction) isCType() {}

type CTypeStruct struct {
	Name    *string
	Fields  []StructField
	Packed  bool
	Aligned *int
}

func (*CTypeStruct) isCType() {}

type CTypeUnion struct {
	Name    *string
	Fields  []StructField
	Packed  bool
	Aligned *int
}

func (*CTypeUnion) isCType() {}

type CTypeEnum struct {
	Name           *string
	UnderlyingType CType
}

func (*CTypeEnum) isCType() {}

type CTypeAtomic struct {
	Inner CType
}

func (*CTypeAtomic) isCType() {}

type CTypTypedef struct {
	Name       string
	Underlying CType
}

func (*CTypTypedef) isCType() {}

type CTypeQualified struct {
	Inner CType
	Quals TypeQualifiers
}

func (*CTypeQualified) isCType() {}

type MachineType int

const (
	MachineI8 MachineType = iota
	MachineI16
	MachineI32
	MachineI64
	MachineI128
	MachineF32
	MachineF64
	MachineF80
	MachinePtr
	MachineVoid
	MachineInteger
	MachineSSE
	MachineX87
	MachineMemory
)

func SizeofCType(ty CType, target *Target) int {
	switch v := ty.(type) {
	case CTypeBase:
		switch v {
		case CTypeVoid:
			return 1
		case CTypeBool:
			return 1
		case CTypeChar, CTypeSChar, CTypeUChar:
			return 1
		case CTypeShort, CTypeUShort:
			return 2
		case CTypeInt, CTypeUInt:
			return 4
		case CTypeLong, CTypeULong:
			return target.LongSize()
		case CTypeLongLong, CTypeULongLong:
			return 8
		case CTypeInt128, CTypeUInt128:
			return 16
		case CTypeFloat:
			return 4
		case CTypeDouble:
			return 8
		case CTypeLongDouble:
			return target.LongDoubleSize()
		default:
			panic("unhandled default case")
		}
	case *CTypeComplex:
		return 2 * SizeofCType(v.Base, target)
	case *CTypePointer:
		return target.PointerWidth()
	case *CTypeArray:
		if v.Size != nil {
			return SizeofCType(v.Elem, target) * *v.Size
		}
		return 0
	case *CTypeFunction:
		return 1
	case *CTypeStruct:
		return computeStructSize(v.Fields, v.Packed, v.Aligned, target)
	case *CTypeUnion:
		return computeUnionSize(v.Fields, v.Packed, v.Aligned, target)
	case *CTypeEnum:
		return SizeofCType(v.UnderlyingType, target)
	case *CTypeAtomic:
		return SizeofCType(v.Inner, target)
	case *CTypTypedef:
		return SizeofCType(v.Underlying, target)
	case *CTypeQualified:
		return SizeofCType(v.Inner, target)
	}
	return 0
}

func computeStructSize(fields []StructField, packed bool, aligned *int, target *Target) int {
	if len(fields) == 0 {
		return 0
	}

	offset := 0
	maxAlign := 1

	for _, field := range fields {
		fieldSize := SizeofCType(field.Ty, target)
		fieldAlign := 1
		if !packed {
			fieldAlign = AlignofCType(field.Ty, target)
		}

		offset = alignUp(offset, fieldAlign)
		offset += fieldSize

		if fieldAlign > maxAlign {
			maxAlign = fieldAlign
		}
	}

	structAlign := maxAlign
	if aligned != nil {
		if *aligned > structAlign {
			structAlign = *aligned
		}
	} else if packed {
		structAlign = 1
	}

	return alignUp(offset, structAlign)
}

func computeUnionSize(fields []StructField, packed bool, aligned *int, target *Target) int {
	if len(fields) == 0 {
		return 0
	}

	maxSize := 0
	maxAlign := 1

	for _, field := range fields {
		fieldSize := SizeofCType(field.Ty, target)
		fieldAlign := 1
		if !packed {
			fieldAlign = AlignofCType(field.Ty, target)
		}

		if fieldSize > maxSize {
			maxSize = fieldSize
		}
		if fieldAlign > maxAlign {
			maxAlign = fieldAlign
		}
	}

	unionAlign := maxAlign
	if aligned != nil {
		if *aligned > unionAlign {
			unionAlign = *aligned
		}
	} else if packed {
		unionAlign = 1
	}

	return alignUp(maxSize, unionAlign)
}

func AlignofCType(ty CType, target *Target) int {
	switch v := ty.(type) {
	case CTypeBase:
		switch v {
		case CTypeVoid:
			return 1
		case CTypeBool:
			return 1
		case CTypeChar, CTypeSChar, CTypeUChar:
			return 1
		case CTypeShort, CTypeUShort:
			return 2
		case CTypeInt, CTypeUInt:
			return 4
		case CTypeLong, CTypeULong:
			return target.LongSize()
		case CTypeLongLong, CTypeULongLong:
			if *target == TargetI686 {
				return 4
			}
			return 8
		case CTypeInt128, CTypeUInt128:
			return 16
		case CTypeFloat:
			return 4
		case CTypeDouble:
			if *target == TargetI686 {
				return 4
			}
			return 8
		case CTypeLongDouble:
			return target.LongDoubleAlign()
		}
	case *CTypeComplex:
		return AlignofCType(v.Base, target)
	case *CTypePointer:
		return target.PointerWidth()
	case *CTypeArray:
		return AlignofCType(v.Elem, target)
	case *CTypeFunction:
		return 1
	case *CTypeStruct:
		return computeStructOrUnionAlign(v.Fields, v.Packed, v.Aligned, target)
	case *CTypeUnion:
		return computeStructOrUnionAlign(v.Fields, v.Packed, v.Aligned, target)
	case *CTypeEnum:
		return AlignofCType(v.UnderlyingType, target)
	case *CTypeAtomic:
		return AlignofCType(v.Inner, target)
	case *CTypTypedef:
		return AlignofCType(v.Underlying, target)
	case *CTypeQualified:
		return AlignofCType(v.Inner, target)
	}
	return 1
}

func SizeofCTypeResolved(ty CType, target *Target, tagTypes map[string]CType) int {
	switch v := ty.(type) {
	case *CTypeStruct:
		if v.Name != nil && len(v.Fields) == 0 {
			if resolved, ok := tagTypes[*v.Name]; ok {
				return SizeofCTypeResolved(resolved, target, tagTypes)
			}
			return 0
		}
		return computeStructSizeResolved(v.Fields, v.Packed, v.Aligned, target, tagTypes)
	case *CTypeUnion:
		if v.Name != nil && len(v.Fields) == 0 {
			if resolved, ok := tagTypes[*v.Name]; ok {
				return SizeofCTypeResolved(resolved, target, tagTypes)
			}
			return 0
		}
		return computeUnionSizeResolved(v.Fields, v.Packed, v.Aligned, target, tagTypes)
	case *CTypeArray:
		if v.Size != nil {
			return SizeofCTypeResolved(v.Elem, target, tagTypes) * *v.Size
		}
		return 0
	case *CTypeComplex:
		return 2 * SizeofCTypeResolved(v.Base, target, tagTypes)
	case *CTypeAtomic:
		return SizeofCTypeResolved(v.Inner, target, tagTypes)
	case *CTypTypedef:
		return SizeofCTypeResolved(v.Underlying, target, tagTypes)
	case *CTypeQualified:
		return SizeofCTypeResolved(v.Inner, target, tagTypes)
	case *CTypeEnum:
		return SizeofCTypeResolved(v.UnderlyingType, target, tagTypes)
	default:
		return SizeofCType(ty, target)
	}
}

func AlignofCTypeResolved(ty CType, target *Target, tagTypes map[string]CType) int {
	switch v := ty.(type) {
	case *CTypeStruct:
		if v.Name != nil && len(v.Fields) == 0 {
			if resolved, ok := tagTypes[*v.Name]; ok {
				return AlignofCTypeResolved(resolved, target, tagTypes)
			}
			return 1
		}
		return computeStructOrUnionAlignResolved(v.Fields, v.Packed, v.Aligned, target, tagTypes)
	case *CTypeUnion:
		if v.Name != nil && len(v.Fields) == 0 {
			if resolved, ok := tagTypes[*v.Name]; ok {
				return AlignofCTypeResolved(resolved, target, tagTypes)
			}
			return 1
		}
		return computeStructOrUnionAlignResolved(v.Fields, v.Packed, v.Aligned, target, tagTypes)
	case *CTypeArray:
		return AlignofCTypeResolved(v.Elem, target, tagTypes)
	case *CTypeComplex:
		return AlignofCTypeResolved(v.Base, target, tagTypes)
	case *CTypeAtomic:
		return AlignofCTypeResolved(v.Inner, target, tagTypes)
	case *CTypTypedef:
		return AlignofCTypeResolved(v.Underlying, target, tagTypes)
	case *CTypeQualified:
		return AlignofCTypeResolved(v.Inner, target, tagTypes)
	case *CTypeEnum:
		return AlignofCTypeResolved(v.UnderlyingType, target, tagTypes)
	default:
		return AlignofCType(ty, target)
	}
}

func computeStructSizeResolved(fields []StructField, packed bool, aligned *int, target *Target, tagTypes map[string]CType) int {
	if len(fields) == 0 {
		return 0
	}

	offset := 0
	maxAlign := 1

	for _, field := range fields {
		fieldSize := SizeofCTypeResolved(field.Ty, target, tagTypes)
		fieldAlign := 1
		if !packed {
			fieldAlign = AlignofCTypeResolved(field.Ty, target, tagTypes)
		}

		offset = alignUp(offset, fieldAlign)
		offset += fieldSize

		if fieldAlign > maxAlign {
			maxAlign = fieldAlign
		}
	}

	structAlign := maxAlign
	if aligned != nil {
		if *aligned > structAlign {
			structAlign = *aligned
		}
	} else if packed {
		structAlign = 1
	}

	return alignUp(offset, structAlign)
}

func computeUnionSizeResolved(fields []StructField, packed bool, aligned *int, target *Target, tagTypes map[string]CType) int {
	if len(fields) == 0 {
		return 0
	}

	maxSize := 0
	maxAlign := 1

	for _, field := range fields {
		fieldSize := SizeofCTypeResolved(field.Ty, target, tagTypes)
		fieldAlign := 1
		if !packed {
			fieldAlign = AlignofCTypeResolved(field.Ty, target, tagTypes)
		}

		if fieldSize > maxSize {
			maxSize = fieldSize
		}
		if fieldAlign > maxAlign {
			maxAlign = fieldAlign
		}
	}

	unionAlign := maxAlign
	if aligned != nil {
		if *aligned > unionAlign {
			unionAlign = *aligned
		}
	} else if packed {
		unionAlign = 1
	}

	return alignUp(maxSize, unionAlign)
}

func computeStructOrUnionAlignResolved(fields []StructField, packed bool, aligned *int, target *Target, tagTypes map[string]CType) int {
	natural := 1
	if !packed {
		for _, field := range fields {
			fa := AlignofCTypeResolved(field.Ty, target, tagTypes)
			if fa > natural {
				natural = fa
			}
		}
	}

	if aligned != nil {
		if *aligned > natural {
			return *aligned
		}
	}
	return natural
}

func computeStructOrUnionAlign(fields []StructField, packed bool, aligned *int, target *Target) int {
	natural := 1
	if !packed {
		for _, field := range fields {
			fa := AlignofCType(field.Ty, target)
			if fa > natural {
				natural = fa
			}
		}
	}

	if aligned != nil {
		if *aligned > natural {
			return *aligned
		}
	}
	return natural
}

func alignUp(value, align int) int {
	if align == 0 {
		return value
	}
	mask := align - 1
	return (value + mask) & ^mask
}

func IsVoid(ty CType) bool {
	return ResolveAndStrip(ty) == CTypeVoid
}

func IsInteger(ty CType) bool {
	stripped := ResolveAndStrip(ty)
	_, ok := stripped.(CTypeBase)
	if !ok {
		return false
	}
	switch stripped {
	case CTypeBool, CTypeChar, CTypeSChar, CTypeUChar,
		CTypeShort, CTypeUShort, CTypeInt, CTypeUInt,
		CTypeLong, CTypeULong, CTypeLongLong, CTypeULongLong,
		CTypeInt128, CTypeUInt128:
		if en, ok := stripped.(*CTypeEnum); ok {
			return IsInteger(en.UnderlyingType)
		}
		return true
	case CTypeEnum_:
		return true
	}
	return false
}

func IsUnsigned(ty CType) bool {
	stripped := ResolveAndStrip(ty)
	if en, ok := stripped.(*CTypeEnum); ok {
		return IsUnsigned(en.UnderlyingType)
	}
	switch stripped {
	case CTypeBool, CTypeUChar, CTypeUShort, CTypeUInt, CTypeULong, CTypeULongLong, CTypeUInt128:
		return true
	}
	return false
}

func IsSigned(ty CType) bool {
	stripped := ResolveAndStrip(ty)
	if en, ok := stripped.(*CTypeEnum); ok {
		return IsSigned(en.UnderlyingType)
	}
	switch stripped {
	case CTypeChar, CTypeSChar, CTypeShort, CTypeInt, CTypeLong, CTypeLongLong, CTypeInt128:
		return true
	}
	return false
}

func IsFloating(ty CType) bool {
	stripped := ResolveAndStrip(ty)
	switch stripped {
	case CTypeFloat, CTypeDouble, CTypeLongDouble:
		return true
	}
	return false
}

func IsArithmetic(ty CType) bool {
	return IsInteger(ty) || IsFloating(ty)
}

func IsScalar(ty CType) bool {
	return IsArithmetic(ty) || IsPointer(ty)
}

func IsPointer(ty CType) bool {
	_, ok := ResolveAndStrip(ty).(*CTypePointer)
	return ok
}

func IsArray(ty CType) bool {
	_, ok := ResolveAndStrip(ty).(*CTypeArray)
	return ok
}

func IsFunction(ty CType) bool {
	_, ok := ResolveAndStrip(ty).(*CTypeFunction)
	return ok
}

func IsStructOrUnion(ty CType) bool {
	stripped := ResolveAndStrip(ty)
	_, structOK := stripped.(*CTypeStruct)
	_, unionOK := stripped.(*CTypeUnion)
	return structOK || unionOK
}

func ResolveAndStrip(ty CType) CType {
	switch v := ty.(type) {
	case *CTypeQualified:
		return ResolveAndStrip(v.Inner)
	case *CTypTypedef:
		return ResolveAndStrip(v.Underlying)
	case *CTypeAtomic:
		return ResolveAndStrip(v.Inner)
	}
	return ty
}

func Unqualified(ty CType) CType {
	if q, ok := ty.(*CTypeQualified); ok {
		return q.Inner
	}
	return ty
}

func ResolveTypedef(ty CType) CType {
	if t, ok := ty.(*CTypTypedef); ok {
		return ResolveTypedef(t.Underlying)
	}
	return ty
}

func IntegerPromotion(ty CType) CType {
	resolved := ResolveAndStrip(ty)
	switch resolved {
	case CTypeBool, CTypeChar, CTypeSChar, CTypeShort:
		return CTypeInt
	case CTypeUChar, CTypeUShort:
		return CTypeInt
	case CTypeEnum_:
		return IntegerPromotion(resolved.(*CTypeEnum).UnderlyingType)
	}
	return ty
}

func IntegerRank(ty CType) uint8 {
	resolved := ResolveAndStrip(ty)
	switch resolved {
	case CTypeBool:
		return 1
	case CTypeChar, CTypeSChar, CTypeUChar:
		return 2
	case CTypeShort, CTypeUShort:
		return 3
	case CTypeInt, CTypeUInt:
		return 4
	case CTypeLong, CTypeULong:
		return 5
	case CTypeLongLong, CTypeULongLong:
		return 6
	case CTypeInt128, CTypeUInt128:
		return 7
	case CTypeEnum_:
		if e, ok := resolved.(*CTypeEnum); ok && e != nil {
			return IntegerRank(e.UnderlyingType)
		}
	}
	return 0
}

func IsCompatible(a, b CType) bool {
	a = ResolveAndStrip(a)
	b = ResolveAndStrip(b)
	return isCompatibleInner(a, b)
}

func isCompatibleInner(a, b CType) bool {
	switch ta := a.(type) {
	case CTypeBase:
		switch tb := b.(type) {
		case CTypeBase:
			return ta == tb
		}
	case *CTypeComplex:
		if tb, ok := b.(*CTypeComplex); ok {
			return isCompatibleInner(ta.Base, tb.Base)
		}
	case *CTypePointer:
		if tb, ok := b.(*CTypePointer); ok {
			return isCompatibleInner(ta.Pointee, tb.Pointee)
		}
	case *CTypeArray:
		if tb, ok := b.(*CTypeArray); ok {
			if !isCompatibleInner(ta.Elem, tb.Elem) {
				return false
			}
			if ta.Size != nil && tb.Size != nil {
				return *ta.Size == *tb.Size
			}
			return true
		}
	case *CTypeFunction:
		if tb, ok := b.(*CTypeFunction); ok {
			if ta.Variadic != tb.Variadic {
				return false
			}
			if !isCompatibleInner(ta.ReturnType, tb.ReturnType) {
				return false
			}
			if len(ta.Params) != len(tb.Params) {
				return false
			}
			for i := range ta.Params {
				if !isCompatibleInner(ta.Params[i], tb.Params[i]) {
					return false
				}
			}
			return true
		}
	case *CTypeStruct:
		if tb, ok := b.(*CTypeStruct); ok {
			if ta.Name != nil && tb.Name != nil {
				return *ta.Name == *tb.Name
			}
			return false
		}
	case *CTypeUnion:
		if tb, ok := b.(*CTypeUnion); ok {
			if ta.Name != nil && tb.Name != nil {
				return *ta.Name == *tb.Name
			}
			return false
		}
	case *CTypeEnum:
		if tb, ok := b.(*CTypeEnum); ok {
			if ta.Name != nil && tb.Name != nil {
				return *ta.Name == *tb.Name
			}
			return false
		}
	case *CTypeAtomic:
		if tb, ok := b.(*CTypeAtomic); ok {
			return isCompatibleInner(ta.Inner, tb.Inner)
		}
	}
	return false
}
