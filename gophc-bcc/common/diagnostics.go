package common

type Severity int

const (
	SeverityError Severity = iota
	SeverityWarning
	SeverityNote
)

func (s Severity) String() string {
	switch s {
	case SeverityError:
		return "error"
	case SeverityWarning:
		return "warning"
	case SeverityNote:
		return "note"
	default:
		return ""
	}
}

type Span struct {
	FileID uint32
	Start uint32
	End   uint32
}

func NewSpan(fileID uint32, start uint32, end uint32) Span {
	return Span{
		FileID: fileID,
		Start: start,
		End:   end,
	}
}

func DummySpan() Span {
	return Span{}
}

func (s Span) Merge(other Span) Span {
	if s.FileID != other.FileID {
		return s
	}
	return Span{
		FileID: s.FileID,
		Start: minUint32(s.Start, other.Start),
		End:   maxUint32(s.End, other.End),
	}
}

func (s Span) IsDummy() bool {
	return s.FileID == 0 && s.Start == 0 && s.End == 0
}

type SubDiagnostic struct {
	Span    Span
	Message string
}

type FixSuggestion struct {
	Span        Span
	Replacement string
	Message    string
}

type Diagnostic struct {
	Severity      Severity
	Span         Span
	Message      string
	Notes        []SubDiagnostic
	FixSuggestion *FixSuggestion
}

func NewDiagnostic(severity Severity, span Span, message string) *Diagnostic {
	return &Diagnostic{
		Severity: severity,
		Span:    span,
		Message: message,
		Notes:  make([]SubDiagnostic, 0),
	}
}

func Error(span Span, message string) *Diagnostic {
	return NewDiagnostic(SeverityError, span, message)
}

func Warning(span Span, message string) *Diagnostic {
	return NewDiagnostic(SeverityWarning, span, message)
}

func Note(span Span, message string) *Diagnostic {
	return NewDiagnostic(SeverityNote, span, message)
}

func (d *Diagnostic) WithNote(span Span, message string) *Diagnostic {
	d.Notes = append(d.Notes, SubDiagnostic{
		Span:    span,
		Message: message,
	})
	return d
}

func (d *Diagnostic) WithFix(span Span, replacement string, message string) *Diagnostic {
	d.FixSuggestion = &FixSuggestion{
		Span:        span,
		Replacement: replacement,
		Message:    message,
	}
	return d
}

type DiagnosticEngine struct {
	diagnostics  []*Diagnostic
	errorCount   int
	warningCount int
	suppressDepth int
}

func NewDiagnosticEngine() *DiagnosticEngine {
	return &DiagnosticEngine{
		diagnostics: make([]*Diagnostic, 0),
	}
}

func (de *DiagnosticEngine) BeginSuppress() {
	de.suppressDepth++
}

func (de *DiagnosticEngine) EndSuppress() {
	if de.suppressDepth > 0 {
		de.suppressDepth--
	}
}

func (de *DiagnosticEngine) Emit(diag *Diagnostic) {
	if de.suppressDepth > 0 {
		return
	}
	switch diag.Severity {
	case SeverityError:
		de.errorCount++
	case SeverityWarning:
		de.warningCount++
	}
	de.diagnostics = append(de.diagnostics, diag)
}

func (de *DiagnosticEngine) EmitError(span Span, msg string) {
	de.Emit(Error(span, msg))
}

func (de *DiagnosticEngine) EmitWarning(span Span, msg string) {
	de.Emit(Warning(span, msg))
}

func (de *DiagnosticEngine) EmitNote(span Span, msg string) {
	de.Emit(Note(span, msg))
}

func (de *DiagnosticEngine) ErrorCount() int {
	return de.errorCount
}

func (de *DiagnosticEngine) WarningCount() int {
	return de.warningCount
}

func (de *DiagnosticEngine) HasErrors() bool {
	return de.errorCount > 0
}

func (de *DiagnosticEngine) Diagnostics() []*Diagnostic {
	return de.diagnostics
}

func (de *DiagnosticEngine) Clear() {
	de.diagnostics = make([]*Diagnostic, 0)
	de.errorCount = 0
	de.warningCount = 0
}

func (de *DiagnosticEngine) PrintAll(sourceMap *SourceMap) {
	if len(de.diagnostics) == 0 {
		return
	}

	for _, diag := range de.diagnostics {
		de.printDiagnostic(diag, sourceMap)
	}

	stderr := getStderr()
	stderr.Write([]string{
		stringUint(de.errorCount) + " error(s), " + stringUint(de.warningCount) + " warning(s) generated.",
	})
}

func (de *DiagnosticEngine) printDiagnostic(diag *Diagnostic, sourceMap *SourceMap) {
	stderr := getStderr()

	if diag.Severity == SeverityError || diag.Severity == SeverityWarning || diag.Severity == SeverityNote {
		if diag.Span.IsDummy() {
			stderr.Write([]string{
				"<unknown>: " + diag.Severity.String() + ": " + diag.Message,
			})
		} else {
			loc := sourceMap.ResolveLocation(diag.Span.FileID, diag.Span.Start)
			stderr.Write([]string{
				loc.Filename + ":" + uintToString(loc.Line) + ":" + uintToString(loc.Column) + ": " + diag.Severity.String() + ": " + diag.Message,
			})
		}
	}

	for _, sub := range diag.Notes {
		if sub.Span.IsDummy() {
			stderr.Write([]string{"  note: " + sub.Message})
		} else {
			locationStr := sourceMap.FormatSpan(sub.Span.FileID, sub.Span.Start, sub.Span.End)
			stderr.Write([]string{locationStr + ": note: " + sub.Message})
		}
	}

	if diag.FixSuggestion != nil {
		fix := diag.FixSuggestion
		if fix.Span.IsDummy() {
			stderr.Write([]string{"  fix: " + fix.Message + " (replace with '" + fix.Replacement + "')"})
		} else {
			locationStr := sourceMap.FormatSpan(fix.Span.FileID, fix.Span.Start, fix.Span.End)
			stderr.Write([]string{locationStr + ": fix: " + fix.Message + " (replace with '" + fix.Replacement + "')"})
		}
	}
}

func minUint32(a, b uint32) uint32 {
	if a < b {
		return a
	}
	return b
}

func maxUint32(a, b uint32) uint32 {
	if a > b {
		return a
	}
	return b
}

type writer interface {
	Write([]string) (int, error)
}

type stderrWriter struct{}

func (s stderrWriter) Write(lines []string) (int, error) {
	total := 0
	for _, line := range lines {
		println(line)
		total += len(line)
	}
	return total, nil
}

func getStderr() writer {
	return stderrWriter{}
}

func stringUint(i int) string {
	if i == 0 {
		return "0"
	}
	result := ""
	for i > 0 {
		result = string(rune('0'+i%10)) + result
		i /= 10
	}
	return result
}