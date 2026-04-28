package sema

import (
	"github.com/gophc/blitzy-c-compiler/gophc-bcc/common"
)

type ValidatedAttribute interface {
	isValidatedAttribute()
}

type ValidatedAttributeAligned struct {
	Value uint64
}

func (ValidatedAttributeAligned) isValidatedAttribute() {}

type ValidatedAttributePacked struct{}

func (ValidatedAttributePacked) isValidatedAttribute() {}

type ValidatedAttributeSection struct {
	Name string
}

func (ValidatedAttributeSection) isValidatedAttribute() {}

type ValidatedAttributeUsed struct{}

func (ValidatedAttributeUsed) isValidatedAttribute() {}

type ValidatedAttributeUnused struct{}

func (ValidatedAttributeUnused) isValidatedAttribute() {}

type ValidatedAttributeWeak struct{}

func (ValidatedAttributeWeak) isValidatedAttribute() {}

type ValidatedAttributeConstructor struct {
	Priority *int32
}

func (ValidatedAttributeConstructor) isValidatedAttribute() {}

type ValidatedAttributeDestructor struct {
	Priority *int32
}

func (ValidatedAttributeDestructor) isValidatedAttribute() {}

type ValidatedAttributeVisibility struct {
	Visibility SymbolVisibility
}

func (ValidatedAttributeVisibility) isValidatedAttribute() {}

type ValidatedAttributeDeprecated struct {
	Message *string
}

func (ValidatedAttributeDeprecated) isValidatedAttribute() {}

type ValidatedAttributeNoReturn struct{}

func (ValidatedAttributeNoReturn) isValidatedAttribute() {}

type ValidatedAttributeNoInline struct{}

func (ValidatedAttributeNoInline) isValidatedAttribute() {}

type ValidatedAttributeAlwaysInline struct{}

func (ValidatedAttributeAlwaysInline) isValidatedAttribute() {}

type ValidatedAttributeCold struct{}

func (ValidatedAttributeCold) isValidatedAttribute() {}

type ValidatedAttributeHot struct{}

func (ValidatedAttributeHot) isValidatedAttribute() {}

type ValidatedAttributeFormat struct {
	Archetype    FormatArchetype
	StringIndex  uint32
	FirstToCheck uint32
}

func (ValidatedAttributeFormat) isValidatedAttribute() {}

type ValidatedAttributeFormatArg struct {
	Index uint32
}

func (ValidatedAttributeFormatArg) isValidatedAttribute() {}

type ValidatedAttributeMalloc struct{}

func (ValidatedAttributeMalloc) isValidatedAttribute() {}

type ValidatedAttributePure struct{}

func (ValidatedAttributePure) isValidatedAttribute() {}

type ValidatedAttributeConst struct{}

func (ValidatedAttributeConst) isValidatedAttribute() {}

type ValidatedAttributeWarnUnusedResult struct{}

func (ValidatedAttributeWarnUnusedResult) isValidatedAttribute() {}

type ValidatedAttributeFallthrough struct{}

func (ValidatedAttributeFallthrough) isValidatedAttribute() {}

type SymbolVisibility int

const (
	SymbolVisibilityDefault SymbolVisibility = iota
	SymbolVisibilityHidden
	SymbolVisibilityProtected
	SymbolVisibilityInternal
)

type FormatArchetype int

const (
	FormatArchetypePrintf FormatArchetype = iota
	FormatArchetypeScanf
	FormatArchetypeStrftime
	FormatArchetypeStrfmon
)

type AttributeHandler struct {
	diagnostics *common.DiagnosticEngine
}

func NewAttributeHandler(diagnostics *common.DiagnosticEngine) *AttributeHandler {
	return &AttributeHandler{diagnostics: diagnostics}
}

func (ah *AttributeHandler) validateAligned(args []AttributeArg, span common.Span) ValidatedAttribute {
	arg := args[0]
	ah.validateIntArg(arg, span)
	return &ValidatedAttributeAligned{Value: 0}
}

func (ah *AttributeHandler) validatePacked(span common.Span) ValidatedAttribute {
	return &ValidatedAttributePacked{}
}

func (ah *AttributeHandler) validateSection(args []AttributeArg, span common.Span) ValidatedAttribute {
	arg := args[0]
	ah.validateStringArg(arg, span)
	name := ""
	return &ValidatedAttributeSection{Name: name}
}

func (ah *AttributeHandler) validateUsed(span common.Span) ValidatedAttribute {
	return &ValidatedAttributeUsed{}
}

func (ah *AttributeHandler) validateUnused(span common.Span) ValidatedAttribute {
	return &ValidatedAttributeUnused{}
}

func (ah *AttributeHandler) validateWeak(span common.Span) ValidatedAttribute {
	return &ValidatedAttributeWeak{}
}

func (ah *AttributeHandler) validateConstructor(args []AttributeArg, span common.Span) ValidatedAttribute {
	return ah.validateCtorDtor(args, span, true)
}

func (ah *AttributeHandler) validateDestructor(args []AttributeArg, span common.Span) ValidatedAttribute {
	return ah.validateCtorDtor(args, span, false)
}

func (ah *AttributeHandler) validateCtorDtor(args []AttributeArg, span common.Span, isCtor bool) ValidatedAttribute {
	if len(args) > 1 {
		ah.diagnostics.EmitError(span, "attribute takes at most 1 argument")
		return nil
	}
	var priority *int32
	if len(args) > 0 {
		ah.validateIntArg(args[0], span)
	}
	if isCtor {
		return &ValidatedAttributeConstructor{Priority: priority}
	}
	return &ValidatedAttributeDestructor{Priority: priority}
}

func (ah *AttributeHandler) validateVisibility(args []AttributeArg, span common.Span) ValidatedAttribute {
	if len(args) > 1 {
		ah.diagnostics.EmitError(span, "attribute takes at most 1 argument")
		return nil
	}
	visibility := SymbolVisibilityDefault
	if len(args) > 0 {
		ah.validateStringArg(args[0], span)
	}
	return &ValidatedAttributeVisibility{Visibility: visibility}
}

func (ah *AttributeHandler) validateDeprecated(args []AttributeArg, span common.Span) ValidatedAttribute {
	var message *string
	if len(args) > 0 {
		ah.validateStringArg(args[0], span)
	}
	return &ValidatedAttributeDeprecated{Message: message}
}

func (ah *AttributeHandler) validateNoReturn(span common.Span) ValidatedAttribute {
	return &ValidatedAttributeNoReturn{}
}

func (ah *AttributeHandler) validateNoInline(span common.Span) ValidatedAttribute {
	return &ValidatedAttributeNoInline{}
}

func (ah *AttributeHandler) validateAlwaysInline(span common.Span) ValidatedAttribute {
	return &ValidatedAttributeAlwaysInline{}
}

func (ah *AttributeHandler) validateCold(span common.Span) ValidatedAttribute {
	return &ValidatedAttributeCold{}
}

func (ah *AttributeHandler) validateHot(span common.Span) ValidatedAttribute {
	return &ValidatedAttributeHot{}
}

func (ah *AttributeHandler) validateFormat(args []AttributeArg, span common.Span) ValidatedAttribute {
	if len(args) < 3 {
		ah.diagnostics.EmitError(span, "attribute requires 3 arguments")
		return nil
	}
	if len(args) > 3 {
		ah.diagnostics.EmitWarning(span, "attribute takes 3 arguments")
	}
	ah.validateIntArg(args[0], span)
	ah.validateIntArg(args[1], span)
	ah.validateIntArg(args[2], span)
	return &ValidatedAttributeFormat{
		Archetype:    FormatArchetypePrintf,
		StringIndex:  0,
		FirstToCheck: 0,
	}
}

func (ah *AttributeHandler) validateFormatArg(args []AttributeArg, span common.Span) ValidatedAttribute {
	if len(args) != 1 {
		ah.diagnostics.EmitError(span, "attribute requires 1 argument")
		return nil
	}
	ah.validateIntArg(args[0], span)
	return &ValidatedAttributeFormatArg{Index: 0}
}

func (ah *AttributeHandler) validateMalloc(span common.Span) ValidatedAttribute {
	return &ValidatedAttributeMalloc{}
}

func (ah *AttributeHandler) validatePure(span common.Span) ValidatedAttribute {
	return &ValidatedAttributePure{}
}

func (ah *AttributeHandler) validateConst(span common.Span) ValidatedAttribute {
	return &ValidatedAttributeConst{}
}

func (ah *AttributeHandler) validateWarnUnusedResult(span common.Span) ValidatedAttribute {
	return &ValidatedAttributeWarnUnusedResult{}
}

func (ah *AttributeHandler) validateFallthrough(span common.Span) ValidatedAttribute {
	return &ValidatedAttributeFallthrough{}
}

func (ah *AttributeHandler) validateIntArg(arg AttributeArg, span common.Span) {}

func (ah *AttributeHandler) validateStringArg(arg AttributeArg, span common.Span) {}

type AttributeArg interface {
	isAttributeArg()
}

type AttributeArgInt struct {
	Value int64
}

func (AttributeArgInt) isAttributeArg() {}

type AttributeArgString struct {
	Value string
}

func (AttributeArgString) isAttributeArg() {}

type AttributeArgSelector struct {
	Attribute string
	Name      string
}

func (AttributeArgSelector) isAttributeArg() {}

type AttributeArgType struct {
	Type common.CType
}

func (AttributeArgType) isAttributeArg() {}

func ExtractVisibility(attrs []ValidatedAttribute) SymbolVisibility {
	for i := len(attrs) - 1; i >= 0; i-- {
		if v, ok := attrs[i].(ValidatedAttributeVisibility); ok {
			return v.Visibility
		}
	}
	return SymbolVisibilityDefault
}

func ExtractSection(attrs []ValidatedAttribute) *string {
	for i := len(attrs) - 1; i >= 0; i-- {
		if s, ok := attrs[i].(ValidatedAttributeSection); ok {
			return &s.Name
		}
	}
	return nil
}

func ExtractWeak(attrs []ValidatedAttribute) bool {
	for i := len(attrs) - 1; i >= 0; i-- {
		if _, ok := attrs[i].(ValidatedAttributeWeak); ok {
			return true
		}
	}
	return false
}

func ExtractUsed(attrs []ValidatedAttribute) bool {
	for i := len(attrs) - 1; i >= 0; i-- {
		if _, ok := attrs[i].(ValidatedAttributeUsed); ok {
			return true
		}
	}
	return false
}

func ExtractAligned(attrs []ValidatedAttribute) (uint64, bool) {
	for i := len(attrs) - 1; i >= 0; i-- {
		if a, ok := attrs[i].(ValidatedAttributeAligned); ok {
			return a.Value, true
		}
	}
	return 0, false
}

func ExtractPacked(attrs []ValidatedAttribute) bool {
	for i := len(attrs) - 1; i >= 0; i-- {
		if _, ok := attrs[i].(ValidatedAttributePacked); ok {
			return true
		}
	}
	return false
}

func ExtractConstructor(attrs []ValidatedAttribute) *int32 {
	for i := len(attrs) - 1; i >= 0; i-- {
		if c, ok := attrs[i].(ValidatedAttributeConstructor); ok {
			return c.Priority
		}
	}
	return nil
}

func ExtractDestructor(attrs []ValidatedAttribute) *int32 {
	for i := len(attrs) - 1; i >= 0; i-- {
		if d, ok := attrs[i].(ValidatedAttributeDestructor); ok {
			return d.Priority
		}
	}
	return nil
}

func ExtractNoReturn(attrs []ValidatedAttribute) bool {
	for i := len(attrs) - 1; i >= 0; i-- {
		if _, ok := attrs[i].(ValidatedAttributeNoReturn); ok {
			return true
		}
	}
	return false
}

func ExtractMalloc(attrs []ValidatedAttribute) bool {
	for i := len(attrs) - 1; i >= 0; i-- {
		if _, ok := attrs[i].(ValidatedAttributeMalloc); ok {
			return true
		}
	}
	return false
}

func ExtractPure(attrs []ValidatedAttribute) bool {
	for i := len(attrs) - 1; i >= 0; i-- {
		if _, ok := attrs[i].(ValidatedAttributePure); ok {
			return true
		}
	}
	return false
}

func ExtractConst(attrs []ValidatedAttribute) bool {
	for i := len(attrs) - 1; i >= 0; i-- {
		if _, ok := attrs[i].(ValidatedAttributeConst); ok {
			return true
		}
	}
	return false
}

func ExtractAlwaysInline(attrs []ValidatedAttribute) bool {
	for i := len(attrs) - 1; i >= 0; i-- {
		if _, ok := attrs[i].(ValidatedAttributeAlwaysInline); ok {
			return true
		}
	}
	return false
}

func ExtractCold(attrs []ValidatedAttribute) bool {
	for i := len(attrs) - 1; i >= 0; i-- {
		if _, ok := attrs[i].(ValidatedAttributeCold); ok {
			return true
		}
	}
	return false
}

func ExtractHot(attrs []ValidatedAttribute) bool {
	for i := len(attrs) - 1; i >= 0; i-- {
		if _, ok := attrs[i].(ValidatedAttributeHot); ok {
			return true
		}
	}
	return false
}

func ExtractDeprecated(attrs []ValidatedAttribute) **string {
	for i := len(attrs) - 1; i >= 0; i-- {
		if d, ok := attrs[i].(ValidatedAttributeDeprecated); ok {
			return &d.Message
		}
	}
	return nil
}

func ExtractFormat(attrs []ValidatedAttribute) (FormatArchetype, uint32, uint32) {
	for i := len(attrs) - 1; i >= 0; i-- {
		if f, ok := attrs[i].(ValidatedAttributeFormat); ok {
			return f.Archetype, f.StringIndex, f.FirstToCheck
		}
	}
	return FormatArchetypePrintf, 0, 0
}

func ExtractFormatArg(attrs []ValidatedAttribute) (uint32, bool) {
	for i := len(attrs) - 1; i >= 0; i-- {
		if f, ok := attrs[i].(ValidatedAttributeFormatArg); ok {
			return f.Index, true
		}
	}
	return 0, false
}

func ExtractWarnUnusedResult(attrs []ValidatedAttribute) bool {
	for i := len(attrs) - 1; i >= 0; i-- {
		if _, ok := attrs[i].(ValidatedAttributeWarnUnusedResult); ok {
			return true
		}
	}
	return false
}

func ExtractNoInline(attrs []ValidatedAttribute) bool {
	for i := len(attrs) - 1; i >= 0; i-- {
		if _, ok := attrs[i].(ValidatedAttributeNoInline); ok {
			return true
		}
	}
	return false
}
