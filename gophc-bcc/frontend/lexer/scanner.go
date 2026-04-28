package lexer

import (
	"github.com/gophc/blitzy-c-compiler/gophc-bcc/common"
)

type Position struct {
	Offset uint32
	Line   uint32
	Column uint32
}

func NewPosition(offset uint32, line uint32, column uint32) Position {
	return Position{
		Offset: offset,
		Line:   line,
		Column: column,
	}
}

type Scanner struct {
	source    string
	offset    int
	line      uint32
	column    uint32
	lookahead [][2]interface{}
}

func NewScanner(source string) *Scanner {
	return &Scanner{
		source:    source,
		offset:    0,
		line:      1,
		column:    1,
		lookahead: make([][2]interface{}, 0),
	}
}

func (s *Scanner) advance() rune {
	ch, _ := s.doAdvance()
	if ch == 0 {
		return 0
	}

	if ch == '\n' {
		s.line++
		s.column = 1
	} else {
		s.column++
	}
	return ch
}

func (s *Scanner) doAdvance() (rune, Position) {
	if len(s.lookahead) > 0 {
		item := s.lookahead[0]
		s.lookahead = s.lookahead[1:]
		ch := item[0].(rune)
		pos := item[1].(Position)

		if ch == '\n' {
			s.offset = int(pos.Offset) + 1
		} else if pos.Offset > 0 {
			s.offset = int(pos.Offset) + len(string(ch))
		}

		if ch == '\n' {
			s.line = pos.Line + 1
			s.column = 1
		} else {
			s.line = pos.Line
			s.column = pos.Column + uint32(len(string(ch)))
		}
		return ch, pos
	}

	if s.offset >= len(s.source) {
		return 0, Position{}
	}

	pos := Position{
		Offset: uint32(s.offset),
		Line:   s.line,
		Column: s.column,
	}

	ch := rune(s.source[s.offset])
	s.offset++

	if ch == '\r' {
		if s.offset < len(s.source) && rune(s.source[s.offset]) == '\n' {
			s.offset++
		}
		s.line++
		s.column = 1
		return '\n', pos
	}

	if ch == '\n' {
		s.line++
		s.column = 1
		return ch, pos
	}

	s.column++
	return ch, pos
}

func (s *Scanner) peek() rune {
	if len(s.lookahead) > 0 {
		return s.lookahead[0][0].(rune)
	}

	if s.offset >= len(s.source) {
		return 0
	}

	ch := rune(s.source[s.offset])
	if ch == '\r' {
		return '\n'
	}
	return ch
}

func (s *Scanner) peekNth(n int) rune {
	for len(s.lookahead) <= n {
		s.fillLookahead()
	}

	if n < len(s.lookahead) {
		return s.lookahead[n][0].(rune)
	}
	return 0
}

func (s *Scanner) fillLookahead() {
	pos := s.nextLookaheadPosition()
	ch, _ := s.doAdvance()
	if ch != 0 {
		s.lookahead = append(s.lookahead, [2]interface{}{ch, pos})
	}
}

func (s *Scanner) nextLookaheadPosition() Position {
	if len(s.lookahead) > 0 {
		last := s.lookahead[len(s.lookahead)-1]
		ch := last[0].(rune)
		pos := last[1].(Position)
		if ch == '\n' {
			return Position{
				Offset: pos.Offset + 1,
				Line:   pos.Line + 1,
				Column: 1,
			}
		}
		return Position{
			Offset: pos.Offset + uint32(len(string(ch))),
			Line:   pos.Line,
			Column: pos.Column + uint32(len(string(ch))),
		}
	}
	return Position{
		Offset: uint32(s.offset),
		Line:   s.line,
		Column: s.column,
	}
}

func (s *Scanner) unget(ch rune, pos Position) {
	s.lookahead = append([][2]interface{}{{ch, pos}}, s.lookahead...)
	s.offset = int(pos.Offset)
	s.line = pos.Line
	s.column = pos.Column
}

func (s *Scanner) Position() Position {
	return Position{
		Offset: uint32(s.offset),
		Line:   s.line,
		Column: s.column,
	}
}

func (s *Scanner) Offset() uint32 {
	return uint32(s.offset)
}

func (s *Scanner) Line() uint32 {
	return s.line
}

func (s *Scanner) Column() uint32 {
	return s.column
}

func (s *Scanner) IsEof() bool {
	return s.peek() == 0
}

func (s *Scanner) Source() string {
	return s.source
}

func (s *Scanner) Slice(start int, end int) string {
	if start < 0 || end > len(s.source) || start > end {
		return ""
	}
	return s.source[start:end]
}

func (s *Scanner) skipWhile(pred func(rune) bool) {
	for pred(s.peek()) {
		s.advance()
	}
}

func (s *Scanner) consumeIf(ch rune) bool {
	if s.peek() == ch {
		s.advance()
		return true
	}
	return false
}

func (s *Scanner) consumeIfPred(pred func(rune) bool) rune {
	ch := s.peek()
	if pred(ch) {
		s.advance()
		return ch
	}
	return 0
}

func (s *Scanner) IsPuaChar(ch rune) bool {
	return common.IsPUAEncoded(ch)
}
