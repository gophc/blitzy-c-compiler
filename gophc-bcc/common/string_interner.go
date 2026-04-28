package common

import (
	"strconv"
	"sync"
)

type Symbol uint32

func (s Symbol) AsUInt32() uint32 {
	return uint32(s)
}

func SymbolFromUInt32(raw uint32) Symbol {
	return Symbol(raw)
}

func (s Symbol) String() string {
	return strconv.Itoa(int(s))
}

type SymbolString string

type Interner struct {
	strToSym map[string]Symbol
	symToStr []string
	mu       sync.RWMutex
}

func NewInterner() *Interner {
	return &Interner{
		strToSym: make(map[string]Symbol),
		symToStr: make([]string, 0),
	}
}

func (in *Interner) Intern(s string) Symbol {
	in.mu.Lock()
	defer in.mu.Unlock()

	if sym, ok := in.strToSym[s]; ok {
		return sym
	}

	idx := len(in.symToStr)
	sym := Symbol(idx)
	in.symToStr = append(in.symToStr, s)
	in.strToSym[s] = sym
	return sym
}

func (in *Interner) Resolve(sym Symbol) string {
	in.mu.RLock()
	defer in.mu.RUnlock()

	idx := int(sym)
	if idx >= 0 && idx < len(in.symToStr) {
		return in.symToStr[idx]
	}
	return ""
}

func (in *Interner) Get(s string) (Symbol, bool) {
	in.mu.RLock()
	defer in.mu.RUnlock()

	sym, ok := in.strToSym[s]
	return sym, ok
}

func (in *Interner) Len() int {
	in.mu.RLock()
	defer in.mu.RUnlock()

	return len(in.symToStr)
}

func (in *Interner) IsEmpty() bool {
	return in.Len() == 0
}

type InternerSymbols struct {
	interner *Interner
	sym      Symbol
}

func (s SymbolString) String() string {
	return string(s)
}
