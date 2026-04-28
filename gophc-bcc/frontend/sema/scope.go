package sema

import (
	"github.com/gophc/blitzy-c-compiler/gophc-bcc/common"
)

type ScopeKind int

const (
	ScopeKindGlobal ScopeKind = iota
	ScopeKindFile
	ScopeKindFunction
	ScopeKindBlock
	ScopeKindFunctionPrototype
)

type TagKind int

const (
	TagKindStruct TagKind = iota
	TagKindUnion
	TagKindEnum
)

type TagEntry struct {
	Kind       TagKind
	Ty         common.CType
	IsComplete bool
	Span       common.Span
}

type LabelEntry struct {
	DefinedAt    *common.Span
	DeclaredAt   *common.Span
	ReferencedAt []common.Span
	IsLocal      bool
}

type Scope struct {
	Kind           ScopeKind
	Depth          uint32
	Ordinary       map[common.Symbol]SymbolId
	Tags           map[common.Symbol]TagEntry
	Labels         map[common.Symbol]LabelEntry
	Typedefs       map[common.Symbol]struct{}
	IsLoopOrSwitch bool
}

func newScope(kind ScopeKind, depth uint32) *Scope {
	return &Scope{
		Kind:     kind,
		Depth:    depth,
		Ordinary: make(map[common.Symbol]SymbolId),
		Tags:     make(map[common.Symbol]TagEntry),
		Labels:   make(map[common.Symbol]LabelEntry),
		Typedefs: make(map[common.Symbol]struct{}),
	}
}

func (sc *Scope) MarkLoopOrSwitch() {
	sc.IsLoopOrSwitch = true
}

type ScopeStack struct {
	scopes []*Scope
	depth  uint32
}

func NewScopeStack() *ScopeStack {
	stack := &ScopeStack{
		scopes: make([]*Scope, 0, 32),
		depth:  0,
	}
	stack.scopes = append(stack.scopes, newScope(ScopeKindGlobal, 0))
	return stack
}

func (ss *ScopeStack) PushScope(kind ScopeKind) {
	ss.depth++
	ss.scopes = append(ss.scopes, newScope(kind, ss.depth))
}

func (ss *ScopeStack) PopScope(diagnostics *common.DiagnosticEngine) *Scope {
	if len(ss.scopes) <= 1 {
		panic("cannot pop the last (Global) scope")
	}

	scope := ss.scopes[len(ss.scopes)-1]
	ss.scopes = ss.scopes[:len(ss.scopes)-1]

	if ss.depth > 0 {
		ss.depth--
	}

	if scope.Kind == ScopeKindFunction {
		ss.validateLabelsInScope(scope, diagnostics)
	}

	if scope.Kind == ScopeKindBlock && len(scope.Labels) > 0 {
		ss.validateLocalLabelsInScope(scope, diagnostics)
	}

	return scope
}

func (ss *ScopeStack) CurrentScope() *Scope {
	return ss.scopes[len(ss.scopes)-1]
}

func (ss *ScopeStack) CurrentScopeMut() *Scope {
	return ss.scopes[len(ss.scopes)-1]
}

func (ss *ScopeStack) CurrentDepth() uint32 {
	return ss.depth
}

func (ss *ScopeStack) CurrentKind() ScopeKind {
	return ss.CurrentScope().Kind
}

func (ss *ScopeStack) IsFileScope() bool {
	kind := ss.CurrentKind()
	return kind == ScopeKindFile || kind == ScopeKindGlobal
}

func (ss *ScopeStack) IsFunctionScope() bool {
	for i := len(ss.scopes) - 1; i >= 0; i-- {
		if ss.scopes[i].Kind == ScopeKindFunction {
			return true
		}
	}
	return false
}

func (ss *ScopeStack) FindEnclosingFunction() *Scope {
	for i := len(ss.scopes) - 1; i >= 0; i-- {
		if ss.scopes[i].Kind == ScopeKindFunction {
			return ss.scopes[i]
		}
	}
	return nil
}

func (ss *ScopeStack) FindLabelScopeIndex(name common.Symbol) int {
	for i := len(ss.scopes) - 1; i >= 0; i-- {
		scope := ss.scopes[i]
		if scope.Kind == ScopeKindBlock {
			if entry, ok := scope.Labels[name]; ok && entry.IsLocal {
				return i
			}
		}
		if scope.Kind == ScopeKindFunction {
			return i
		}
	}
	return 0
}

func (ss *ScopeStack) FindEnclosingLoopOrSwitch() bool {
	for i := len(ss.scopes) - 1; i >= 0; i-- {
		if ss.scopes[i].IsLoopOrSwitch {
			return true
		}
		if ss.scopes[i].Kind == ScopeKindFunction {
			return false
		}
	}
	return false
}

func (ss *ScopeStack) LookupOrdinary(name common.Symbol) SymbolId {
	for i := len(ss.scopes) - 1; i >= 0; i-- {
		if id, ok := ss.scopes[i].Ordinary[name]; ok {
			return id
		}
	}
	return 0
}

func (ss *ScopeStack) LookupOrdinaryInCurrentScope(name common.Symbol) SymbolId {
	scope := ss.CurrentScope()
	if id, ok := scope.Ordinary[name]; ok {
		return id
	}
	return 0
}

func (ss *ScopeStack) DeclareOrdinary(name common.Symbol, id SymbolId) (SymbolId, bool) {
	scope := ss.CurrentScopeMut()
	delete(scope.Typedefs, name)
	if existing, ok := scope.Ordinary[name]; ok {
		scope.Ordinary[name] = id
		return existing, true
	}
	scope.Ordinary[name] = id
	return 0, false
}

func (ss *ScopeStack) LookupTag(name common.Symbol) *TagEntry {
	for i := len(ss.scopes) - 1; i >= 0; i-- {
		if entry, ok := ss.scopes[i].Tags[name]; ok {
			return &entry
		}
	}
	return nil
}

func (ss *ScopeStack) LookupTagInCurrentScope(name common.Symbol) *TagEntry {
	scope := ss.CurrentScope()
	if entry, ok := scope.Tags[name]; ok {
		return &entry
	}
	return nil
}

func (ss *ScopeStack) DeclareTag(name common.Symbol, entry TagEntry) (TagEntry, bool) {
	scope := ss.CurrentScopeMut()
	if existing, ok := scope.Tags[name]; ok {
		scope.Tags[name] = entry
		return existing, true
	}
	scope.Tags[name] = entry
	return TagEntry{}, false
}

func (ss *ScopeStack) LookupTagByStr(name string, interner *common.Interner) *TagEntry {
	for i := len(ss.scopes) - 1; i >= 0; i-- {
		for sym, entry := range ss.scopes[i].Tags {
			if interner.Resolve(sym) == name {
				return &entry
			}
		}
	}
	return nil
}

func (ss *ScopeStack) CompleteTag(name common.Symbol, ty common.CType) {
	for i := len(ss.scopes) - 1; i >= 0; i-- {
		if entry, ok := ss.scopes[i].Tags[name]; ok {
			entry.Ty = ty
			entry.IsComplete = true
			ss.scopes[i].Tags[name] = entry
			return
		}
	}
}

func (ss *ScopeStack) AllTags() []struct {
	Name  common.Symbol
	Entry TagEntry
} {
	result := make([]struct {
		Name  common.Symbol
		Entry TagEntry
	}, 0)
	for _, scope := range ss.scopes {
		for name, entry := range scope.Tags {
			result = append(result, struct {
				Name  common.Symbol
				Entry TagEntry
			}{name, entry})
		}
	}
	return result
}

func (ss *ScopeStack) AllOrdinarySymbols() []struct {
	Name common.Symbol
	Id   SymbolId
} {
	result := make([]struct {
		Name common.Symbol
		Id   SymbolId
	}, 0)
	for _, scope := range ss.scopes {
		for name, id := range scope.Ordinary {
			result = append(result, struct {
				Name common.Symbol
				Id   SymbolId
			}{name, id})
		}
	}
	return result
}

func (ss *ScopeStack) DeclareLabel(name common.Symbol, span common.Span, isLocal bool, diagnostics *common.DiagnosticEngine) {
	targetIdx := len(ss.scopes) - 1
	if !isLocal {
		for i := len(ss.scopes) - 1; i >= 0; i-- {
			if ss.scopes[i].Kind == ScopeKindFunction {
				targetIdx = i
				break
			}
		}
	}

	targetScope := ss.scopes[targetIdx]

	if _, exists := targetScope.Labels[name]; exists {
		diagnostics.EmitError(span, "redefinition of label")
		return
	}

	targetScope.Labels[name] = LabelEntry{
		DefinedAt:    nil,
		ReferencedAt: make([]common.Span, 0),
		IsLocal:      isLocal,
	}
}

func (ss *ScopeStack) DefineLabel(name common.Symbol, span common.Span, diagnostics *common.DiagnosticEngine) {
	targetIdx := ss.findLabelScopeIndex(name)
	targetScope := ss.scopes[targetIdx]

	if entry, exists := targetScope.Labels[name]; exists {
		if entry.DefinedAt != nil {
			diagnostics.EmitError(span, "redefinition of label")
			return
		}
		entry.DefinedAt = &span
		targetScope.Labels[name] = entry
	} else {
		targetScope.Labels[name] = LabelEntry{
			DefinedAt:    &span,
			ReferencedAt: make([]common.Span, 0),
			IsLocal:      false,
		}
	}
}

func (ss *ScopeStack) ReferenceLabel(name common.Symbol, span common.Span) {
	targetIdx := ss.findLabelScopeIndex(name)
	targetScope := ss.scopes[targetIdx]

	if entry, exists := targetScope.Labels[name]; exists {
		entry.ReferencedAt = append(entry.ReferencedAt, span)
		targetScope.Labels[name] = entry
	} else {
		targetScope.Labels[name] = LabelEntry{
			DefinedAt:    nil,
			ReferencedAt: []common.Span{span},
			IsLocal:      false,
		}
	}
}

func (ss *ScopeStack) ValidateLabels(diagnostics *common.DiagnosticEngine) {
	for _, scope := range ss.scopes {
		if scope.Kind == ScopeKindFunction || len(scope.Labels) > 0 {
			ss.validateLabelsInScope(scope, diagnostics)
		}
	}
}

func (ss *ScopeStack) IsTypedefName(name common.Symbol) bool {
	for i := len(ss.scopes) - 1; i >= 0; i-- {
		if _, ok := ss.scopes[i].Typedefs[name]; ok {
			return true
		}
		if _, ok := ss.scopes[i].Ordinary[name]; ok {
			return false
		}
	}
	return false
}

func (ss *ScopeStack) RegisterTypedef(name common.Symbol) {
	scope := ss.CurrentScopeMut()
	scope.Typedefs[name] = struct{}{}
}

func (ss *ScopeStack) findFunctionScopeIndex() int {
	for i := len(ss.scopes) - 1; i >= 0; i-- {
		if ss.scopes[i].Kind == ScopeKindFunction {
			return i
		}
	}
	return 0
}

func (ss *ScopeStack) findLabelScopeIndex(name common.Symbol) int {
	for i := len(ss.scopes) - 1; i >= 0; i-- {
		scope := ss.scopes[i]
		if scope.Kind == ScopeKindBlock {
			if entry, ok := scope.Labels[name]; ok && entry.IsLocal {
				return i
			}
		}
		if scope.Kind == ScopeKindFunction {
			return i
		}
	}
	return 0
}

func (ss *ScopeStack) validateLabelsInScope(scope *Scope, diagnostics *common.DiagnosticEngine) {
	for _, entry := range scope.Labels {
		if entry.DefinedAt == nil && len(entry.ReferencedAt) == 0 {
			if entry.IsLocal {
				diagnostics.EmitWarning(*entry.DefinedAt, "local label declared but never used")
			}
			continue
		}

		if entry.DefinedAt == nil {
			for _, refSpan := range entry.ReferencedAt {
				diagnostics.EmitError(refSpan, "use of undeclared label")
			}
		}

		if entry.DefinedAt != nil && len(entry.ReferencedAt) == 0 {
			diagnostics.EmitWarning(*entry.DefinedAt, "label defined but not used")
		}
	}
}

func (ss *ScopeStack) validateLocalLabelsInScope(scope *Scope, diagnostics *common.DiagnosticEngine) {
	for _, entry := range scope.Labels {
		if !entry.IsLocal {
			continue
		}

		if entry.DefinedAt == nil && len(entry.ReferencedAt) > 0 {
			for _, refSpan := range entry.ReferencedAt {
				diagnostics.EmitError(refSpan, "use of undeclared local label")
			}
		}

		if entry.DefinedAt != nil && len(entry.ReferencedAt) == 0 {
			diagnostics.EmitWarning(*entry.DefinedAt, "local label defined but not used")
		}
	}
}
