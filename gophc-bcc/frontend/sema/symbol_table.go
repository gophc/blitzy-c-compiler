package sema

import (
	"fmt"

	"github.com/gophc/blitzy-c-compiler/gophc-bcc/common"
)

type SymbolId uint32

func (id SymbolId) AsUint32() uint32 {
	return uint32(id)
}

type SymbolKind int

const (
	SymbolKindVariable SymbolKind = iota
	SymbolKindFunction
	SymbolKindTypedefName
	SymbolKindEnumConstant
)

type Linkage int

const (
	LinkageExternal Linkage = iota
	LinkageInternal
	LinkageNone
)

type StorageClass int

const (
	StorageClassAuto StorageClass = iota
	StorageClassRegister
	StorageClassStatic
	StorageClassExtern
	StorageClassTypedef
)

type SymbolEntry struct {
	Name         common.Symbol
	Ty           common.CType
	Kind         SymbolKind
	Linkage      Linkage
	StorageClass StorageClass
	IsDefined    bool
	IsTentative  bool
	Span         common.Span
	Attributes   []ValidatedAttribute
	IsWeak       bool
	Visibility   *SymbolVisibility
	Section      *string
	IsUsed       bool
	ScopeDepth   uint32
}

type SymbolTable struct {
	symbols           []SymbolEntry
	nameToIds         map[common.Symbol][]SymbolId
	currentScopeDepth uint32
}

func NewSymbolTable() *SymbolTable {
	return &SymbolTable{
		symbols:           make([]SymbolEntry, 0),
		nameToIds:         make(map[common.Symbol][]SymbolId),
		currentScopeDepth: 0,
	}
}

func (st *SymbolTable) Declare(entry SymbolEntry, diagnostics *common.DiagnosticEngine) (SymbolId, error) {
	entry.ScopeDepth = st.currentScopeDepth

	if existingIdPtr := st.findInCurrentScope(entry.Name); existingIdPtr != nil {
		existingId := *existingIdPtr
		existing := &st.symbols[existingId.AsUint32()]

		if existing.Kind == SymbolKindTypedefName && entry.Kind == SymbolKindTypedefName {
			if st.typesAreCompatible(existing.Ty, entry.Ty) {
				return existingId, nil
			}
			diagnostics.EmitError(entry.Span, "conflicting types for typedef")
			return 0, fmt.Errorf("conflicting types")
		}

		if existing.Kind == SymbolKindEnumConstant || entry.Kind == SymbolKindEnumConstant {
			diagnostics.EmitError(entry.Span, "redefinition of enumerator")
			return 0, fmt.Errorf("redefinition")
		}

		if !st.typesAreCompatible(existing.Ty, entry.Ty) {
			diagnostics.EmitError(entry.Span, "conflicting types for symbol")
			return 0, fmt.Errorf("conflicting types")
		}

		if existing.StorageClass == StorageClassExtern && entry.StorageClass == StorageClassExtern && !existing.IsDefined && !entry.IsDefined {
			st.mergeAttributes(existing, entry.Attributes)
			return existingId, nil
		}

		if !existing.IsDefined && entry.IsDefined {
			existing.IsDefined = true
			existing.IsTentative = false
			existing.Span = entry.Span
			st.mergeAttributes(existing, entry.Attributes)
			if entry.IsWeak {
				existing.IsWeak = true
			}
			return existingId, nil
		}

		if existing.IsTentative && entry.StorageClass == StorageClassExtern {
			st.mergeAttributes(existing, entry.Attributes)
			return existingId, nil
		}

		if existing.IsTentative && entry.IsTentative {
			existing.Span = entry.Span
			st.mergeAttributes(existing, entry.Attributes)
			return existingId, nil
		}

		if !existing.IsDefined && !entry.IsDefined {
			st.mergeAttributes(existing, entry.Attributes)
			return existingId, nil
		}

		if existing.IsDefined && entry.IsDefined {
			diagnostics.EmitError(entry.Span, "redefinition of symbol")
			return 0, fmt.Errorf("redefinition")
		}

		if existing.IsDefined && !entry.IsDefined {
			st.mergeAttributes(existing, entry.Attributes)
			return existingId, nil
		}
	}

	id := SymbolId(len(st.symbols))
	st.symbols = append(st.symbols, entry)
	st.nameToIds[entry.Name] = append(st.nameToIds[entry.Name], id)
	return id, nil
}

func (st *SymbolTable) Define(id SymbolId) {
	entry := &st.symbols[id.AsUint32()]
	entry.IsDefined = true
	entry.IsTentative = false
}

func (st *SymbolTable) Lookup(name common.Symbol) *SymbolEntry {
	if ids, ok := st.nameToIds[name]; ok && len(ids) > 0 {
		last := ids[len(ids)-1]
		return &st.symbols[last.AsUint32()]
	}
	return nil
}

func (st *SymbolTable) LookupInCurrentScope(name common.Symbol) *SymbolEntry {
	if ids, ok := st.nameToIds[name]; ok {
		for i := len(ids) - 1; i >= 0; i-- {
			entry := &st.symbols[ids[i].AsUint32()]
			if entry.ScopeDepth == st.currentScopeDepth {
				return entry
			}
		}
	}
	return nil
}

func (st *SymbolTable) Get(id SymbolId) *SymbolEntry {
	return &st.symbols[id.AsUint32()]
}

func (st *SymbolTable) LookupMut(name common.Symbol) *SymbolEntry {
	if ids, ok := st.nameToIds[name]; ok && len(ids) > 0 {
		last := ids[len(ids)-1]
		return &st.symbols[last.AsUint32()]
	}
	return nil
}

func (st *SymbolTable) EnterScope() {
	st.currentScopeDepth++
}

func (st *SymbolTable) LeaveScope() {
	depth := st.currentScopeDepth

	for name, ids := range st.nameToIds {
		var remaining []SymbolId
		for _, id := range ids {
			if st.symbols[id.AsUint32()].ScopeDepth != depth {
				remaining = append(remaining, id)
			}
		}
		if len(remaining) == 0 {
			delete(st.nameToIds, name)
		} else {
			st.nameToIds[name] = remaining
		}
	}

	if st.currentScopeDepth > 0 {
		st.currentScopeDepth--
	}
}

func (st *SymbolTable) ResolveLinkage(name common.Symbol, storage StorageClass, scopeDepth uint32) Linkage {
	if storage == StorageClassTypedef || storage == StorageClassRegister {
		return LinkageNone
	}

	isFileScope := scopeDepth == 0

	if isFileScope {
		switch storage {
		case StorageClassStatic:
			return LinkageInternal
		case StorageClassExtern:
			if prior := st.Lookup(name); prior != nil && prior.Linkage == LinkageInternal {
				return LinkageInternal
			}
			return LinkageExternal
		default:
			return LinkageExternal
		}
	} else {
		switch storage {
		case StorageClassExtern:
			if prior := st.Lookup(name); prior != nil && (prior.Linkage == LinkageInternal || prior.Linkage == LinkageExternal) {
				return prior.Linkage
			}
			return LinkageExternal
		case StorageClassStatic:
			return LinkageNone
		default:
			return LinkageNone
		}
	}
}

func (st *SymbolTable) DeclareFunction(name common.Symbol, ty common.CType, storage StorageClass, span common.Span, diagnostics *common.DiagnosticEngine) (SymbolId, error) {
	linkage := st.ResolveLinkage(name, storage, st.currentScopeDepth)

	entry := SymbolEntry{
		Name:         name,
		Ty:           ty,
		Kind:         SymbolKindFunction,
		Linkage:      linkage,
		StorageClass: storage,
		IsDefined:    false,
		IsTentative:  false,
		Span:         span,
	}

	return st.Declare(entry, diagnostics)
}

func (st *SymbolTable) DeclareEnumConstant(name common.Symbol, value int64, ty common.CType, span common.Span) (SymbolId, error) {
	if existingId := st.findInCurrentScope(name); existingId != nil {
		return 0, fmt.Errorf("redefinition")
	}

	entry := SymbolEntry{
		Name:         name,
		Ty:           ty,
		Kind:         SymbolKindEnumConstant,
		Linkage:      LinkageNone,
		StorageClass: StorageClassAuto,
		IsDefined:    true,
		IsTentative:  false,
		Span:         span,
	}

	id := SymbolId(len(st.symbols))
	st.symbols = append(st.symbols, entry)
	st.nameToIds[name] = append(st.nameToIds[name], id)
	return id, nil
}

func (st *SymbolTable) MarkWeak(id SymbolId) {
	st.symbols[id.AsUint32()].IsWeak = true
}

func (st *SymbolTable) MarkUsed(name common.Symbol) {
	if ids, ok := st.nameToIds[name]; ok && len(ids) > 0 {
		last := ids[len(ids)-1]
		st.symbols[last.AsUint32()].IsUsed = true
	}
}

func (st *SymbolTable) CheckLinkageCompatibility(existing *SymbolEntry, newEntry *SymbolEntry, diagnostics *common.DiagnosticEngine) error {
	if existing.Linkage == LinkageInternal && newEntry.Linkage == LinkageExternal {
		diagnostics.EmitError(newEntry.Span, "non-static declaration follows static declaration")
		return fmt.Errorf("linkage conflict")
	}
	if existing.Linkage == LinkageExternal && newEntry.Linkage == LinkageExternal {
		if !st.typesAreCompatible(existing.Ty, newEntry.Ty) {
			diagnostics.EmitError(newEntry.Span, "conflicting types for symbol with external linkage")
			return fmt.Errorf("linkage conflict")
		}
	}
	return nil
}

func (st *SymbolTable) CheckUnusedSymbols(diagnostics *common.DiagnosticEngine, interner *common.Interner) {
	for i := range st.symbols {
		entry := &st.symbols[i]
		if entry.IsUsed {
			continue
		}
		hasUnusedAttr := false
		for _, attr := range entry.Attributes {
			switch attr.(type) {
			case ValidatedAttributeUnused, ValidatedAttributeUsed:
				hasUnusedAttr = true
			}
		}
		if hasUnusedAttr {
			continue
		}
		if entry.StorageClass == StorageClassExtern && !entry.IsDefined {
			continue
		}
		if entry.Kind == SymbolKindTypedefName {
			continue
		}
		if entry.Kind == SymbolKindEnumConstant {
			continue
		}
		symName := interner.Resolve(entry.Name)
		if symName == "main" || symName == "_start" {
			continue
		}
		var msg string
		switch entry.Kind {
		case SymbolKindVariable:
			msg = "unused variable"
		case SymbolKindFunction:
			msg = "unused function"
		default:
			continue
		}
		diagnostics.EmitWarning(entry.Span, msg)
	}
}

func (st *SymbolTable) findInCurrentScope(name common.Symbol) *SymbolId {
	if ids, ok := st.nameToIds[name]; ok {
		for i := len(ids) - 1; i >= 0; i-- {
			entry := st.symbols[ids[i].AsUint32()]
			if entry.ScopeDepth == st.currentScopeDepth {
				return &ids[i]
			}
		}
	}
	return nil
}

func (st *SymbolTable) typesAreCompatible(a, b common.CType) bool {
	if a == b {
		return true
	}
	return false
}

func (st *SymbolTable) mergeAttributes(entry *SymbolEntry, newAttrs []ValidatedAttribute) {
	for _, attr := range newAttrs {
		attr := attr
		switch v := attr.(type) {
		case ValidatedAttributeWeak:
			entry.IsWeak = true
		case ValidatedAttributeVisibility:
			entry.Visibility = &v.Visibility
		case ValidatedAttributeSection:
			s := string(v.Name)
			entry.Section = &s
		case ValidatedAttributeUsed:
			entry.IsUsed = true
		}
		found := false
		for _, existing := range entry.Attributes {
			if existing == attr {
				found = true
				break
			}
		}
		if !found {
			entry.Attributes = append(entry.Attributes, attr)
		}
	}
}
