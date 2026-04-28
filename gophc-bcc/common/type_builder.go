package common

type FieldLayout struct {
	Offset       int
	Size         int
	Alignment    int
	BitfieldInfo *BitfieldInfo
}

type BitfieldInfo struct {
	BitOffset int
	BitWidth  int
}

type StructLayout struct {
	Fields            []FieldLayout
	Size              int
	Alignment         int
	HasFlexibleArray bool
}

type TypeBuilder struct {
	target Target
}

func NewTypeBuilder(target Target) *TypeBuilder {
	return &TypeBuilder{target: target}
}

func (tb *TypeBuilder) Target() *Target {
	return &tb.target
}

func (tb *TypeBuilder) PointerTo(pointee CType) CType {
	return &CTypePointer{
		Pointee: pointee,
		Quals:  TypeQualifiers{},
	}
}

func (tb *TypeBuilder) ArrayOf(element CType, size *int) CType {
	return &CTypeArray{
		Elem: element,
		Size: size,
	}
}

func (tb *TypeBuilder) FunctionType(returnType CType, params []CType, variadic bool) CType {
	return &CTypeFunction{
		ReturnType: returnType,
		Params:     params,
		Variadic:   variadic,
	}
}

func (tb *TypeBuilder) ConstQualified(ty CType) CType {
	switch v := ty.(type) {
	case *CTypeQualified:
		v.Quals.IsConst = true
		return v
	default:
		return &CTypeQualified{
			Inner: ty,
			Quals: TypeQualifiers{IsConst: true},
		}
	}
}

func (tb *TypeBuilder) VolatileQualified(ty CType) CType {
	switch v := ty.(type) {
	case *CTypeQualified:
		v.Quals.IsVolatile = true
		return v
	default:
		return &CTypeQualified{
			Inner: ty,
			Quals: TypeQualifiers{IsVolatile: true},
		}
	}
}

func (tb *TypeBuilder) AtomicType(ty CType) CType {
	return &CTypeAtomic{Inner: ty}
}

func (tb *TypeBuilder) ComputeStructLayout(fieldTypes []CType, packed bool, explicitAlign *int) *StructLayout {
	fields := make([]FieldLayout, len(fieldTypes))
	offset := 0
	maxAlign := 1
	hasFlexibleArray := false

	for i, fieldTy := range fieldTypes {
		isLast := i == len(fieldTypes)-1
		isFlex := isLast && isArrayWithNilSize(fieldTy)
		if isFlex {
			hasFlexibleArray = true
		}

		fieldSize := 0
		if !isFlex {
			fieldSize = SizeofCType(fieldTy, &tb.target)
		}

		fieldAlign := 1
		if !packed {
			fieldAlign = AlignofCType(fieldTy, &tb.target)
		}

		offset = alignUp(offset, fieldAlign)
		fields[i] = FieldLayout{
			Offset:    offset,
			Size:     fieldSize,
			Alignment: fieldAlign,
		}

		offset += fieldSize
		if fieldAlign > maxAlign {
			maxAlign = fieldAlign
		}
	}

	structAlign := maxAlign
	if explicitAlign != nil {
		if *explicitAlign > structAlign {
			structAlign = *explicitAlign
		}
	} else if packed {
		structAlign = 1
	}

	totalSize := alignUp(offset, structAlign)
	return &StructLayout{
		Fields: fields,
		Size: totalSize,
		Alignment: structAlign,
		HasFlexibleArray: hasFlexibleArray,
	}
}

func (tb *TypeBuilder) ComputeUnionLayout(fieldTypes []CType, packed bool, explicitAlign *int) *StructLayout {
	fields := make([]FieldLayout, len(fieldTypes))
	maxSize := 0
	maxAlign := 1
	hasFlexibleArray := false

	for i, fieldTy := range fieldTypes {
		fieldSize := SizeofCType(fieldTy, &tb.target)
		fieldAlign := 1
		if !packed {
			fieldAlign = AlignofCType(fieldTy, &tb.target)
		}

		fields[i] = FieldLayout{
			Offset:    0,
			Size:     fieldSize,
			Alignment: fieldAlign,
		}

		if fieldSize > maxSize {
			maxSize = fieldSize
		}
		if fieldAlign > maxAlign {
			maxAlign = fieldAlign
		}
		_ = hasFlexibleArray
		_ = i
	}

	unionAlign := maxAlign
	if explicitAlign != nil {
		if *explicitAlign > unionAlign {
			unionAlign = *explicitAlign
		}
	} else if packed {
		unionAlign = 1
	}

	totalSize := alignUp(maxSize, unionAlign)
	return &StructLayout{
		Fields: fields,
		Size: totalSize,
		Alignment: unionAlign,
		HasFlexibleArray: false,
	}
}

func (tb *TypeBuilder) SizeofType(ty CType) int {
	return SizeofCType(ty, &tb.target)
}

func (tb *TypeBuilder) AlignofType(ty CType) int {
	return AlignofCType(ty, &tb.target)
}

func isArrayWithNilSize(ty CType) bool {
	if a, ok := ty.(*CTypeArray); ok {
		return a.Size == nil
	}
	return false
}

func IsIntegerType(ty CType) bool {
	resolved := ResolveAndStrip(ty)
	switch resolved {
	case CTypeBool, CTypeChar, CTypeSChar, CTypeUChar,
		CTypeShort, CTypeUShort, CTypeInt, CTypeUInt,
		CTypeLong, CTypeULong, CTypeLongLong, CTypeULongLong:
		return true
	}
	if e, ok := resolved.(*CTypeEnum); ok && e != nil {
		return true
	}
	return false
}

func IsArithmeticType(ty CType) bool {
	if IsIntegerType(ty) {
		return true
	}
	resolved := ResolveAndStrip(ty)
	switch resolved {
	case CTypeFloat, CTypeDouble, CTypeLongDouble:
		return true
	}
	return false
}

func IsScalarType(ty CType) bool {
	if IsIntegerType(ty) {
		return true
	}
	resolved := ResolveAndStrip(ty)
	switch resolved {
	case CTypeFloat, CTypeDouble, CTypeLongDouble:
		return true
	}
	if p, ok := resolved.(*CTypePointer); ok && p != nil {
		return true
	}
	return false
}

func IsCompleteType(ty CType) bool {
	resolved := ResolveAndStrip(ty)
	switch resolved {
	case CTypeVoid:
		return false
	}
	if a, ok := resolved.(*CTypeArray); ok && a != nil {
		return a.Size != nil
	}
	return true
}

func UsualArithmeticConversion(lhs, rhs CType) CType {
	left := ResolveAndStrip(lhs)
	right := ResolveAndStrip(rhs)

	if left == CTypeLongDouble || right == CTypeLongDouble {
		return CTypeLongDouble
	}
	if left == CTypeDouble || right == CTypeDouble {
		return CTypeDouble
	}
	if left == CTypeFloat || right == CTypeFloat {
		return CTypeFloat
	}

	promotedL := IntegerPromotion(left)
	promotedR := IntegerPromotion(right)

	lUnsigned := IsUnsigned(promotedL)
	rUnsigned := IsUnsigned(promotedR)
	lRank := IntegerRank(promotedL)
	rRank := IntegerRank(promotedR)

	if lUnsigned == rUnsigned {
		if lRank >= rRank {
			return promotedL
		}
		return promotedR
	}

	var signedTy, unsignedTy CType
	var signedRank, unsignedRank uint8
	if lUnsigned {
		signedTy = promotedR
		signedRank = rRank
		unsignedTy = promotedL
		unsignedRank = lRank
	} else {
		signedTy = promotedL
		signedRank = lRank
		unsignedTy = promotedR
		unsignedRank = rRank
	}

	if unsignedRank >= signedRank {
		return unsignedTy
	}

	if signedRank > unsignedRank {
		return signedTy
	}

	return toUnsigned(signedTy)
}

func toUnsigned(ty CType) CType {
	switch ty {
	case CTypeChar:
		return CTypeUChar
	case CTypeShort:
		return CTypeUShort
	case CTypeInt:
		return CTypeUInt
	case CTypeLong:
		return CTypeULong
	case CTypeLongLong:
		return CTypeULongLong
	}
	return ty
}