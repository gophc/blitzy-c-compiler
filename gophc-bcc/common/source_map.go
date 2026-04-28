package common

type SourceLocation struct {
	FileID   uint32
	Filename string
	Line     uint32
	Column   uint32
}

func (loc SourceLocation) String() string {
	return loc.Filename + ":" + uintToString(loc.Line) + ":" + uintToString(loc.Column)
}

type LineDirective struct {
	FileID          uint32
	DirectiveOffset uint32
	NewLine         uint32
	NewFilename     *string
}

type SourceFile struct {
	ID          uint32
	Filename    string
	Content     string
	LineOffsets []uint32
}

func NewSourceFile(id uint32, filename string, content string) *SourceFile {
	lineOffsets := computeLineOffsets(content)
	return &SourceFile{
		ID:          id,
		Filename:    filename,
		Content:     content,
		LineOffsets: lineOffsets,
	}
}

func (sf *SourceFile) LookupLineCol(byteOffset uint32) (uint32, uint32) {
	lineIndex := binarySearchLE(sf.LineOffsets, byteOffset)
	lineStart := sf.LineOffsets[lineIndex]
	lineNumber := uint32(lineIndex + 1)
	column := byteOffset - lineStart + 1

	return lineNumber, column
}

func (sf *SourceFile) GetLineContent(line uint32) string {
	if line == 0 || int(line) > len(sf.LineOffsets) {
		return ""
	}

	idx := int(line - 1)
	start := int(sf.LineOffsets[idx])

	var end int
	if idx+1 < len(sf.LineOffsets) {
		nextStart := int(sf.LineOffsets[idx+1])
		if nextStart > 0 && nextStart-1 < len(sf.Content) && sf.Content[nextStart-1] == '\n' {
			end = nextStart - 1
		} else {
			end = nextStart
		}
	} else {
		end = len(sf.Content)
	}

	if start > len(sf.Content) {
		start = len(sf.Content)
	}
	if end > len(sf.Content) {
		end = len(sf.Content)
	}
	if end > start {
		result := sf.Content[start:end]
		if len(result) > 0 && result[len(result)-1] == '\r' {
			result = result[:len(result)-1]
		}
		return result
	}
	return ""
}

func (sf *SourceFile) LineCount() int {
	return len(sf.LineOffsets)
}

type SourceMap struct {
	files          []*SourceFile
	lineDirectives map[uint32][]LineDirective
}

func NewSourceMap() *SourceMap {
	return &SourceMap{
		files:          make([]*SourceFile, 0),
		lineDirectives: make(map[uint32][]LineDirective),
	}
}

func (sm *SourceMap) AddFile(filename string, content string) uint32 {
	id := uint32(len(sm.files))
	file := NewSourceFile(id, filename, content)
	sm.files = append(sm.files, file)
	return id
}

func (sm *SourceMap) GetFile(fileID uint32) *SourceFile {
	if int(fileID) < len(sm.files) {
		return sm.files[fileID]
	}
	return nil
}

func (sm *SourceMap) LookupLocation(fileID uint32, byteOffset uint32) *SourceLocation {
	file := sm.GetFile(fileID)
	if file == nil {
		return nil
	}
	line, column := file.LookupLineCol(byteOffset)
	return &SourceLocation{
		FileID:   fileID,
		Filename: file.Filename,
		Line:     line,
		Column:   column,
	}
}

func (sm *SourceMap) GetFilename(fileID uint32) string {
	file := sm.GetFile(fileID)
	if file != nil {
		return file.Filename
	}
	return ""
}

func (sm *SourceMap) GetLineDirectiveFilenames(fileID uint32) []string {
	directives, ok := sm.lineDirectives[fileID]
	if !ok {
		return nil
	}

	names := make([]string, 0)
	seen := make(map[string]bool)
	for _, d := range directives {
		if d.NewFilename != nil && !seen[*d.NewFilename] {
			seen[*d.NewFilename] = true
			names = append(names, *d.NewFilename)
		}
	}
	return names
}

func (sm *SourceMap) AddLineDirective(directive LineDirective) {
	fileID := directive.FileID
	if _, ok := sm.lineDirectives[fileID]; !ok {
		sm.lineDirectives[fileID] = make([]LineDirective, 0)
	}
	sm.lineDirectives[fileID] = append(sm.lineDirectives[fileID], directive)
}

func (sm *SourceMap) ResolveLocation(fileID uint32, byteOffset uint32) *SourceLocation {
	file := sm.GetFile(fileID)
	if file == nil {
		return &SourceLocation{
			FileID:   fileID,
			Filename: "<unknown>",
			Line:     0,
			Column:   0,
		}
	}

	physLine, column := file.LookupLineCol(byteOffset)
	resolvedLine := physLine
	resolvedFilename := file.Filename

	directives, ok := sm.lineDirectives[fileID]
	if ok {
		idx := binarySearchLedirective(directives, byteOffset)
		if idx >= 0 {
			dir := directives[idx]
			dirPhysLine, _ := file.LookupLineCol(dir.DirectiveOffset)
			linesSince := physLine - dirPhysLine
			resolvedLine = dir.NewLine + linesSince
			if dir.NewFilename != nil {
				resolvedFilename = *dir.NewFilename
			}
		}
	}

	return &SourceLocation{
		FileID:   fileID,
		Filename: resolvedFilename,
		Line:     resolvedLine,
		Column:   column,
	}
}

func (sm *SourceMap) FormatSpan(fileID uint32, start uint32, end uint32) string {
	loc := sm.ResolveLocation(fileID, start)
	return loc.Filename + ":" + uintToString(loc.Line) + ":" + uintToString(loc.Column)
}

func computeLineOffsets(content string) []uint32 {
	offsets := []uint32{0}
	for i, c := range content {
		if c == '\n' {
			offsets = append(offsets, uint32(i+1))
		}
	}
	return offsets
}

func binarySearchLE(arr []uint32, target uint32) int {
	low, high := 0, len(arr)-1
	for low < high {
		mid := (low + high + 1) / 2
		if arr[mid] <= target {
			low = mid
		} else {
			high = mid - 1
		}
	}
	return low
}

func binarySearchLedirective(arr []LineDirective, target uint32) int {
	low, high := 0, len(arr)-1
	result := -1
	for low <= high {
		mid := (low + high) / 2
		if arr[mid].DirectiveOffset <= target {
			result = mid
			low = mid + 1
		} else {
			high = mid - 1
		}
	}
	return result
}

func uintToString(u uint32) string {
	if u == 0 {
		return "0"
	}
	result := ""
	for u > 0 {
		result = string(rune('0'+u%10)) + result
		u /= 10
	}
	return result
}
