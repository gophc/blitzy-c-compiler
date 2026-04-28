package common

import (
	"hash"
	"hash/fnv"
)

type FxHasher struct {
	h hash.Hash64
}

func NewFxHasher() *FxHasher {
	h := fnv.New64a()
	return &FxHasher{h: h}
}

func (fh *FxHasher) Write(p []byte) (int, error) {
	return fh.h.Write(p)
}

func (fh *FxHasher) WriteByte(b byte) error {
	_, err := fh.h.Write([]byte{b})
	return err
}

func (fh *FxHasher) WriteUint64(i uint64) {
	buf := make([]byte, 8)
	for j := 7; j >= 0; j-- {
		buf[j] = byte(i & 0xFF)
		i >>= 8
	}
	fh.h.Write(buf)
}

func (fh *FxHasher) Sum() []byte {
	return fh.h.Sum([]byte{})
}

type FxHashMap map[interface{}]interface{}

func NewFxHashMap() FxHashMap {
	return make(FxHashMap)
}

func (m FxHashMap) Put(key, value interface{}) {
	m[key] = value
}

func (m FxHashMap) Get(key interface{}) (interface{}, bool) {
	val, ok := m[key]
	return val, ok
}

func (m FxHashMap) Len() int {
	return len(m)
}

func (m FxHashMap) Delete(key interface{}) {
	delete(m, key)
}

type FxHashSet map[interface{}]struct{}

func NewFxHashSet() FxHashSet {
	return make(FxHashSet)
}

func (s FxHashSet) Add(key interface{}) {
	s[key] = struct{}{}
}

func (s FxHashSet) Contains(key interface{}) bool {
	_, ok := s[key]
	return ok
}

func (s FxHashSet) Len() int {
	return len(s)
}

func (s FxHashSet) Delete(key interface{}) {
	delete(s, key)
}
