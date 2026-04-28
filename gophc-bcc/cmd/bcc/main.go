package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// ============================================================================
// Constants
// ============================================================================

const (
	workerStackSize   = 64 * 1024 * 1024
	maxRecursionDepth = 512
	programName       = "bcc"
	version           = "0.1.0"
)

// ============================================================================
// Target architecture
// ============================================================================

type Target int

const (
	TargetX8664 Target = iota
	TargetI686
	TargetAArch64
	TargetRiscV64
)

func (t Target) String() string {
	switch t {
	case TargetX8664:
		return "x86-64"
	case TargetI686:
		return "i686"
	case TargetAArch64:
		return "aarch64"
	case TargetRiscV64:
		return "riscv64"
	default:
		return "unknown"
	}
}

func targetFromString(s string) *Target {
	switch s {
	case "x86-64":
		return ptr(TargetX8664)
	case "i686":
		return ptr(TargetI686)
	case "aarch64":
		return ptr(TargetAArch64)
	case "riscv64":
		return ptr(TargetRiscV64)
	default:
		return nil
	}
}

func targetSystemIncludePaths(t Target) []string {
	switch t {
	case TargetX8664:
		return []string{"/usr/x86_64-linux-gnu/include", "/usr/include"}
	case TargetI686:
		return []string{"/usr/i686-linux-gnu/include", "/usr/include"}
	case TargetAArch64:
		return []string{"/usr/aarch64-linux-gnu/include", "/usr/include"}
	case TargetRiscV64:
		return []string{"/usr/riscv64-linux-gnu/include", "/usr/include"}
	default:
		return []string{"/usr/include"}
	}
}

func ptr[T any](v T) *T { return &v }

// ============================================================================
// CliArgs — parsed command-line arguments
// ============================================================================

type CliArgs struct {
	inputFiles        []string
	outputFile        *string
	target            Target
	compileOnly       bool
	emitAssembly      bool
	preprocessOnly    bool
	debugInfo         bool
	optimizationLevel int
	pic               bool
	shared            bool
	retpoline         bool
	cfProtection      bool
	includePaths      []string
	defines           []DefinePair
	undefs            []string
	libraryPaths      []string
	libraries         []string
	forcedIncludes    []string
	genDepfile        bool
	depfilePath       *string
	depfileTarget     *string
	depfilePhony      bool
	nostdinc          bool
	verbose           bool
	march             *string
	mabi              *string
}

type DefinePair struct {
	Name  string
	Value *string
}

func newCliArgs() *CliArgs {
	return &CliArgs{
		target: TargetX8664,
	}
}

// ============================================================================
// CompilationContext — settings propagated through the pipeline
// ============================================================================

type CompilationContext struct {
	target            Target
	debugInfo         bool
	optimizationLevel int
	pic               bool
	shared            bool
	retpoline         bool
	cfProtection      bool
	includePaths      []string
	defines           []DefinePair
	undefs            []string
	libraryPaths      []string
	libraries         []string
	forcedIncludes    []string
	maxRecursionDepth int
	nostdinc          bool
	verbose           bool
}

func newCompilationContext(args *CliArgs) *CompilationContext {
	return &CompilationContext{
		target:            args.target,
		debugInfo:         args.debugInfo,
		optimizationLevel: args.optimizationLevel,
		pic:               args.pic || args.shared,
		shared:            args.shared,
		retpoline:         args.retpoline,
		cfProtection:      args.cfProtection,
		includePaths:      append([]string{}, args.includePaths...),
		defines:           append([]DefinePair{}, args.defines...),
		undefs:            append([]string{}, args.undefs...),
		libraryPaths:      append([]string{}, args.libraryPaths...),
		libraries:         append([]string{}, args.libraries...),
		forcedIncludes:    append([]string{}, args.forcedIncludes...),
		maxRecursionDepth: maxRecursionDepth,
		nostdinc:          args.nostdinc,
		verbose:           args.verbose,
	}
}

// ============================================================================
// CLI Argument Parsing
// ============================================================================

func parseArgs(args []string) (*CliArgs, error) {
	cli := newCliArgs()
	i := 0

	for i < len(args) {
		arg := args[i]

		if strings.HasPrefix(arg, "--target=") {
			targetStr := strings.TrimPrefix(arg, "--target=")
			t := targetFromString(targetStr)
			if t == nil {
				return nil, fmt.Errorf("unknown target '%s'. Supported targets: x86-64, i686, aarch64, riscv64", targetStr)
			}
			cli.target = *t
			i++
			continue
		}

		if arg == "--version" {
			fmt.Printf("%s version %s\n", programName, version)
			os.Exit(0)
		}

		if arg == "--help" || arg == "-h" {
			printUsage()
			os.Exit(0)
		}

		if arg == "-o" {
			i++
			if i >= len(args) {
				return nil, fmt.Errorf("missing argument to '-o'")
			}
			cli.outputFile = &args[i]
			i++
			continue
		}

		if arg == "-c" {
			cli.compileOnly = true
			i++
			continue
		}

		if arg == "-S" {
			cli.emitAssembly = true
			i++
			continue
		}

		if arg == "-E" {
			cli.preprocessOnly = true
			i++
			continue
		}

		if arg == "-g" {
			cli.debugInfo = true
			i++
			continue
		}

		if strings.HasPrefix(arg, "-O") {
			levelStr := strings.TrimPrefix(arg, "-O")
			var level int
			switch levelStr {
			case "0", "":
				level = 0
			case "1":
				level = 1
			case "2":
				level = 2
			case "3":
				level = 3
			case "s":
				level = 2
			default:
				fmt.Fprintf(os.Stderr, "%s: warning: unknown optimization level '%s', using -O0\n", programName, levelStr)
				level = 0
			}
			cli.optimizationLevel = level
			i++
			continue
		}

		if arg == "-fPIC" || arg == "-fpic" {
			cli.pic = true
			i++
			continue
		}

		if arg == "-shared" {
			cli.shared = true
			i++
			continue
		}

		if arg == "-mretpoline" {
			cli.retpoline = true
			i++
			continue
		}

		if arg == "-fcf-protection" || arg == "-fcf-protection=full" {
			cli.cfProtection = true
			i++
			continue
		}
		if arg == "-fcf-protection=none" {
			cli.cfProtection = false
			i++
			continue
		}

		if arg == "-I" {
			i++
			if i >= len(args) {
				return nil, fmt.Errorf("missing argument to '-I'")
			}
			cli.includePaths = append(cli.includePaths, args[i])
			i++
			continue
		}
		if strings.HasPrefix(arg, "-I") {
			dir := strings.TrimPrefix(arg, "-I")
			cli.includePaths = append(cli.includePaths, dir)
			i++
			continue
		}

		if arg == "-D" {
			i++
			if i >= len(args) {
				return nil, fmt.Errorf("missing argument to '-D'")
			}
			cli.defines = append(cli.defines, parseDefine(args[i]))
			i++
			continue
		}
		if strings.HasPrefix(arg, "-D") {
			defStr := strings.TrimPrefix(arg, "-D")
			cli.defines = append(cli.defines, parseDefine(defStr))
			i++
			continue
		}

		if arg == "-U" {
			i++
			if i >= len(args) {
				return nil, fmt.Errorf("missing argument to '-U'")
			}
			cli.undefs = append(cli.undefs, args[i])
			i++
			continue
		}
		if strings.HasPrefix(arg, "-U") {
			undefStr := strings.TrimPrefix(arg, "-U")
			cli.undefs = append(cli.undefs, undefStr)
			i++
			continue
		}

		if arg == "-L" {
			i++
			if i >= len(args) {
				return nil, fmt.Errorf("missing argument to '-L'")
			}
			cli.libraryPaths = append(cli.libraryPaths, args[i])
			i++
			continue
		}
		if strings.HasPrefix(arg, "-L") {
			dir := strings.TrimPrefix(arg, "-L")
			cli.libraryPaths = append(cli.libraryPaths, dir)
			i++
			continue
		}

		if arg == "-l" {
			i++
			if i >= len(args) {
				return nil, fmt.Errorf("missing argument to '-l'")
			}
			cli.libraries = append(cli.libraries, args[i])
			i++
			continue
		}
		if strings.HasPrefix(arg, "-l") {
			lib := strings.TrimPrefix(arg, "-l")
			cli.libraries = append(cli.libraries, lib)
			i++
			continue
		}

		if arg == "-w" {
			i++
			continue
		}

		if strings.HasPrefix(arg, "-Wp,") {
			subFlags := strings.TrimPrefix(arg, "-Wp,")
			parts := strings.Split(subFlags, ",")
			j := 0
			for j < len(parts) {
				if parts[j] == "-MMD" || parts[j] == "-MD" {
					cli.genDepfile = true
					if j+1 < len(parts) {
						cli.depfilePath = &parts[j+1]
						j++
					}
				} else if parts[j] == "-MP" {
					cli.depfilePhony = true
				} else if parts[j] == "-MT" && j+1 < len(parts) {
					cli.depfileTarget = &parts[j+1]
					j++
				}
				j++
			}
			i++
			continue
		}

		if strings.HasPrefix(arg, "-Wl,") {
			i++
			continue
		}
		if strings.HasPrefix(arg, "-Wa,") {
			i++
			continue
		}
		if strings.HasPrefix(arg, "-W") {
			i++
			continue
		}
		if strings.HasPrefix(arg, "-std=") {
			i++
			continue
		}

		if strings.HasPrefix(arg, "-m") && arg != "-mretpoline" {
			if strings.HasPrefix(arg, "-march=") {
				marchVal := strings.TrimPrefix(arg, "-march=")
				cli.march = &marchVal
				if strings.HasPrefix(marchVal, "rv64") && cli.target == TargetX8664 {
					cli.target = TargetRiscV64
				} else if strings.HasPrefix(marchVal, "rv32") && cli.target == TargetX8664 {
					cli.target = TargetRiscV64
				} else if (strings.HasPrefix(marchVal, "armv8") || strings.HasPrefix(marchVal, "aarch64")) && cli.target == TargetX8664 {
					cli.target = TargetAArch64
				}
			}
			if strings.HasPrefix(arg, "-mabi=") {
				mabiVal := strings.TrimPrefix(arg, "-mabi=")
				cli.mabi = &mabiVal
				if strings.HasPrefix(mabiVal, "lp64") && cli.target == TargetX8664 {
					cli.target = TargetRiscV64
				} else if strings.HasPrefix(mabiVal, "ilp32") && cli.target == TargetX8664 {
					cli.target = TargetI686
				}
			}
			i++
			continue
		}

		if strings.HasPrefix(arg, "-f") &&
			arg != "-fPIC" &&
			arg != "-fpic" &&
			arg != "-fcf-protection" &&
			arg != "-fcf-protection=full" &&
			arg != "-fcf-protection=none" {
			i++
			continue
		}

		if arg == "-pipe" {
			i++
			continue
		}

		if arg == "--verbose" {
			cli.verbose = true
			i++
			continue
		}

		if arg == "-nostdinc" {
			cli.nostdinc = true
			i++
			continue
		}

		if arg == "-nostdlib" || arg == "-nodefaultlibs" || arg == "-nostartfiles" {
			i++
			continue
		}

		if arg == "-static" {
			i++
			continue
		}

		if arg == "-pthread" {
			i++
			continue
		}

		if arg == "-MD" || arg == "-MMD" {
			cli.genDepfile = true
			i++
			continue
		}

		if arg == "-MF" {
			i++
			if i < len(args) {
				cli.depfilePath = &args[i]
				cli.genDepfile = true
				i++
			}
			continue
		}

		if arg == "-MT" {
			i++
			if i < len(args) {
				cli.depfileTarget = &args[i]
				i++
			}
			continue
		}

		if arg == "-MQ" {
			i++
			if i < len(args) {
				cli.depfileTarget = &args[i]
				i++
			}
			continue
		}

		if arg == "-MP" {
			cli.depfilePhony = true
			i++
			continue
		}

		if arg == "-M" || arg == "-MM" {
			cli.genDepfile = true
			i++
			continue
		}

		if arg == "-x" {
			i++
			if i < len(args) {
				i++
			}
			continue
		}
		if strings.HasPrefix(arg, "-x") {
			i++
			continue
		}

		if arg == "-P" {
			i++
			continue
		}

		if arg == "-C" || arg == "-CC" {
			i++
			continue
		}

		if arg == "--param" {
			i++
			if i < len(args) {
				i++
			}
			continue
		}
		if strings.HasPrefix(arg, "--param=") {
			i++
			continue
		}

		if arg == "-include" {
			i++
			if i < len(args) {
				cli.forcedIncludes = append(cli.forcedIncludes, args[i])
				i++
			}
			continue
		}

		if arg == "-isystem" {
			i++
			if i < len(args) {
				cli.includePaths = append(cli.includePaths, args[i])
				i++
			}
			continue
		}

		if arg == "-idirafter" {
			i++
			if i < len(args) {
				cli.includePaths = append(cli.includePaths, args[i])
				i++
			}
			continue
		}

		if strings.HasPrefix(arg, "-") && arg != "-" {
			fmt.Fprintf(os.Stderr, "%s: warning: unrecognized flag '%s'\n", programName, arg)
			i++
			continue
		}

		if arg == "-" {
			cli.inputFiles = append(cli.inputFiles, "-")
			i++
			continue
		}

		cli.inputFiles = append(cli.inputFiles, arg)
		i++
	}

	if len(cli.inputFiles) == 0 {
		return nil, fmt.Errorf("no input files")
	}

	modeCount := 0
	if cli.compileOnly {
		modeCount++
	}
	if cli.emitAssembly {
		modeCount++
	}
	if cli.preprocessOnly {
		modeCount++
	}
	if modeCount > 1 {
		return nil, fmt.Errorf("only one of '-c', '-S', '-E' may be specified")
	}

	return cli, nil
}

func parseDefine(s string) DefinePair {
	if eqPos := strings.IndexByte(s, '='); eqPos >= 0 {
		name := s[:eqPos]
		value := s[eqPos+1:]
		return DefinePair{Name: name, Value: &value}
	}
	return DefinePair{Name: s, Value: nil}
}

// ============================================================================
// Usage / Help
// ============================================================================

func printUsage() {
	fmt.Fprintf(os.Stderr, "Usage: %s [options] <input.c> [-o output]\n", programName)
	fmt.Fprintln(os.Stderr)
	fmt.Fprintln(os.Stderr, "BCC — Blitzy's C Compiler (C11 with GCC extensions)")
	fmt.Fprintln(os.Stderr)
	fmt.Fprintln(os.Stderr, "Options:")
	fmt.Fprintln(os.Stderr, "  --target=<arch>     Set target architecture:")
	fmt.Fprintln(os.Stderr, "                        x86-64, i686, aarch64, riscv64")
	fmt.Fprintln(os.Stderr, "  -o <file>           Write output to <file>")
	fmt.Fprintln(os.Stderr, "  -c                  Compile and assemble, but do not link")
	fmt.Fprintln(os.Stderr, "  -S                  Compile only; output assembly text")
	fmt.Fprintln(os.Stderr, "  -E                  Preprocess only; output to stdout")
	fmt.Fprintln(os.Stderr, "  -g                  Emit DWARF v4 debug information")
	fmt.Fprintln(os.Stderr, "  -O0                 No optimization (default)")
	fmt.Fprintln(os.Stderr, "  -fPIC / -fpic       Generate position-independent code")
	fmt.Fprintln(os.Stderr, "  -shared             Produce a shared object (.so)")
	fmt.Fprintln(os.Stderr, "  -mretpoline         Enable retpoline thunks (x86-64 only)")
	fmt.Fprintln(os.Stderr, "  -fcf-protection     Enable CET/IBT (x86-64 only)")
	fmt.Fprintln(os.Stderr, "  -I<dir>             Add include search path")
	fmt.Fprintln(os.Stderr, "  -D<macro>[=value]   Define preprocessor macro")
	fmt.Fprintln(os.Stderr, "  -L<dir>             Add library search path")
	fmt.Fprintln(os.Stderr, "  -l<lib>             Link library")
	fmt.Fprintln(os.Stderr, "  --version           Print version information and exit")
	fmt.Fprintln(os.Stderr, "  --help              Print this help message and exit")
}

// ============================================================================
// Output Path Resolution
// ============================================================================

func resolveOutputPath(args *CliArgs) *string {
	if args.preprocessOnly && args.outputFile == nil {
		return nil
	}
	if args.outputFile != nil {
		return args.outputFile
	}
	if len(args.inputFiles) > 0 {
		input := args.inputFiles[0]
		stem := strings.TrimSuffix(filepath.Base(input), filepath.Ext(input))
		if stem == "" {
			stem = "output"
		}
		if args.emitAssembly {
			s := stem + ".s"
			return &s
		}
		if args.compileOnly {
			s := stem + ".o"
			return &s
		}
	}
	s := "a.out"
	return &s
}

// ============================================================================
// Compilation Pipeline
// ============================================================================

func validateInputFiles(inputFiles []string) error {
	for _, input := range inputFiles {
		if input == "-" || input == "/dev/null" {
			continue
		}
		info, err := os.Stat(input)
		if err != nil {
			if os.IsNotExist(err) {
				return fmt.Errorf("'%s': No such file or directory", input)
			}
			return fmt.Errorf("'%s': %v", input, err)
		}
		if info.IsDir() {
			return fmt.Errorf("'%s': Is a directory", input)
		}
	}
	return nil
}

func isObjectOrArchive(path string) bool {
	ext := filepath.Ext(path)
	if ext == ".o" || ext == ".a" || ext == ".so" {
		return true
	}
	f, err := os.Open(path)
	if err != nil {
		return false
	}
	defer f.Close()
	var magic [4]byte
	if _, err := io.ReadFull(f, magic[:]); err != nil {
		return false
	}
	return magic == [4]byte{0x7f, 'E', 'L', 'F'}
}

func parseArArchive(data []byte, archiveName string) []ArchiveMember {
	var members []ArchiveMember
	if len(data) < 8 || string(data[:8]) != "!<arch>\n" {
		return nil
	}
	offset := 8
	var extendedNames []byte

	for offset+60 <= len(data) {
		header := data[offset : offset+60]
		if header[58] != '`' || header[59] != '\n' {
			break
		}
		sizeStr := strings.TrimSpace(string(header[48:58]))
		memberSize, _ := strconv.Atoi(sizeStr)
		memberStart := offset + 60
		memberEnd := memberStart + memberSize
		if memberEnd > len(data) {
			memberEnd = len(data)
		}

		nameRaw := strings.TrimRight(string(header[:16]), " ")

		if nameRaw == "/" || nameRaw == "/SYM64/" {
		} else if nameRaw == "//" {
			extendedNames = data[memberStart:memberEnd]
		} else {
			var memberName string
			if strings.HasPrefix(nameRaw, "/") {
				extOffsetStr := strings.TrimSpace(nameRaw[1:])
				extOffset, err := strconv.Atoi(extOffsetStr)
				if err == nil && extOffset < len(extendedNames) {
					end := extOffset
					for end < len(extendedNames) && extendedNames[end] != '/' && extendedNames[end] != '\n' {
						end++
					}
					memberName = string(extendedNames[extOffset:end])
				} else {
					memberName = fmt.Sprintf("%s:member_%d", archiveName, len(members))
				}
			} else {
				memberName = strings.TrimRight(nameRaw, "/")
			}

			memberData := make([]byte, memberEnd-memberStart)
			copy(memberData, data[memberStart:memberEnd])

			if len(memberData) >= 4 && string(memberData[:4]) == "\x7fELF" {
				members = append(members, ArchiveMember{Name: memberName, Data: memberData})
			}
		}

		offset = memberStart + memberSize
		if offset%2 != 0 {
			offset++
		}
	}
	return members
}

type ArchiveMember struct {
	Name string
	Data []byte
}

func extractElfSoname(data []byte) *string {
	if len(data) < 64 || string(data[:4]) != "\x7fELF" {
		return nil
	}
	is64 := data[4] == 2
	le := data[5] == 1

	readU16 := func(off int) uint16 {
		if le {
			return binary.LittleEndian.Uint16(data[off:])
		}
		return binary.BigEndian.Uint16(data[off:])
	}
	readU32 := func(off int) uint32 {
		if le {
			return binary.LittleEndian.Uint32(data[off:])
		}
		return binary.BigEndian.Uint32(data[off:])
	}
	readU64 := func(off int) uint64 {
		if le {
			return binary.LittleEndian.Uint64(data[off:])
		}
		return binary.BigEndian.Uint64(data[off:])
	}

	var phOff, phEntSize, phNum int
	if is64 {
		phOff = int(readU64(32))
		phEntSize = int(readU16(54))
		phNum = int(readU16(56))
	} else {
		phOff = int(readU32(28))
		phEntSize = int(readU16(42))
		phNum = int(readU16(44))
	}

	var dynOff, dynSize int
	ptDynamic := uint32(2)
	for i := 0; i < phNum; i++ {
		base := phOff + i*phEntSize
		if base+phEntSize > len(data) {
			break
		}
		pType := readU32(base)
		if pType == ptDynamic {
			if is64 {
				dynOff = int(readU64(base + 8))
				dynSize = int(readU64(base + 32))
			} else {
				dynOff = int(readU32(base + 4))
				dynSize = int(readU32(base + 16))
			}
			break
		}
	}
	if dynOff == 0 || dynSize == 0 {
		return nil
	}

	dtSonameTag := uint64(14)
	dtStrtabTag := uint64(5)
	entrySize := 8
	if is64 {
		entrySize = 16
	}
	var sonameOffset *uint64
	var strtabVaddr uint64

	off := dynOff
	for off+entrySize <= len(data) && off < dynOff+dynSize {
		var tag, val uint64
		if is64 {
			tag = readU64(off)
			val = readU64(off + 8)
		} else {
			tag = uint64(readU32(off))
			val = uint64(readU32(off + 4))
		}
		if tag == 0 {
			break
		}
		if tag == dtSonameTag {
			sonameOffset = &val
		}
		if tag == dtStrtabTag {
			strtabVaddr = val
		}
		off += entrySize
	}

	if sonameOffset == nil || strtabVaddr == 0 {
		return nil
	}

	var strtabFileOff *int
	for i := 0; i < phNum; i++ {
		base := phOff + i*phEntSize
		if base+phEntSize > len(data) {
			break
		}
		pType := readU32(base)
		if pType != 1 {
			continue
		}
		var pVaddr, pOffset, pFilesz uint64
		if is64 {
			pVaddr = readU64(base + 16)
			pOffset = readU64(base + 8)
			pFilesz = readU64(base + 32)
		} else {
			pVaddr = uint64(readU32(base + 8))
			pOffset = uint64(readU32(base + 4))
			pFilesz = uint64(readU32(base + 16))
		}
		if strtabVaddr >= pVaddr && strtabVaddr < pVaddr+pFilesz {
			v := int(pOffset + (strtabVaddr - pVaddr))
			strtabFileOff = &v
			break
		}
	}

	if strtabFileOff == nil {
		return nil
	}

	nameStart := *strtabFileOff + int(*sonameOffset)
	if nameStart >= len(data) {
		return nil
	}
	end := nameStart
	for end < len(data) && data[end] != 0 {
		end++
	}
	s := string(data[nameStart:end])
	return &s
}

// ============================================================================
// Preprocess-only mode (-E)
// ============================================================================

func runPreprocessOnly(args *CliArgs, ctx *CompilationContext, outputPath *string) error {
	for _, input := range args.inputFiles {
		pp := newPreprocessor(ctx.target)
		pp.setPreservePragmas(true)

		for _, path := range ctx.includePaths {
			pp.addIncludePath(path)
		}

		if !ctx.nostdinc {
			if exe, err := os.Executable(); err == nil {
				exeDir := filepath.Dir(exe)
				builtin := filepath.Join(exeDir, "..", "..", "include")
				if info, err := os.Stat(builtin); err == nil && info.IsDir() {
					if abs, err := filepath.Abs(builtin); err == nil {
						pp.addSystemIncludePath(abs)
					}
				}
				builtin2 := filepath.Join(exeDir, "include")
				if info, err := os.Stat(builtin2); err == nil && info.IsDir() {
					if abs, err := filepath.Abs(builtin2); err == nil {
						pp.addSystemIncludePath(abs)
					}
				}
			}

			targetSysPaths := targetSystemIncludePaths(ctx.target)
			extraSysPaths := []string{"/usr/local/include", "/usr/include/linux"}
			for _, sp := range append(targetSysPaths, extraSysPaths...) {
				if info, err := os.Stat(sp); err == nil && info.IsDir() {
					pp.addSystemIncludePath(sp)
				}
			}
		}

		for _, def := range ctx.defines {
			val := "1"
			if def.Value != nil {
				val = *def.Value
			}
			pp.addDefine(def.Name, val)
		}

		for _, name := range ctx.undefs {
			pp.addUndef(name)
		}

		var forcedTokens []string
		for _, forced := range ctx.forcedIncludes {
			tokens, err := pp.preprocessFile(forced)
			if err != nil {
				return fmt.Errorf("forced include '%s' failed", forced)
			}
			forcedTokens = append(forcedTokens, tokens...)
		}

		mainTokens, err := pp.preprocessFile(input)
		if err != nil {
			return fmt.Errorf("preprocessing failed for '%s'", input)
		}

		tokens := forcedTokens
		tokens = append(tokens, mainTokens...)

		output := reconstructPPOutput(tokens)

		if outputPath != nil {
			if err := os.WriteFile(*outputPath, []byte(output), 0644); err != nil {
				return fmt.Errorf("failed to write output to '%s': %v", *outputPath, err)
			}
		} else {
			if _, err := os.Stdout.WriteString(output); err != nil {
				return fmt.Errorf("failed to write to stdout: %v", err)
			}
		}

		if args.genDepfile {
			var depPath string
			if args.depfilePath != nil {
				depPath = *args.depfilePath
			} else {
				stem := strings.TrimSuffix(filepath.Base(input), filepath.Ext(input))
				if stem == "" {
					stem = "output"
				}
				parent := filepath.Dir(input)
				depPath = filepath.Join(parent, "."+stem+".o.d")
			}
			depTarget := input
			if args.depfileTarget != nil {
				depTarget = *args.depfileTarget
			} else if outputPath != nil {
				depTarget = *outputPath
			}

			deps := []string{input}
			deps = append(deps, pp.getIncludedFiles()...)

			depContent := depTarget + ": " + strings.Join(deps, " \\\n  ") + "\n"
			if args.depfilePhony {
				for _, d := range deps[1:] {
					depContent += "\n" + d + ":\n"
				}
			}
			os.WriteFile(depPath, []byte(depContent), 0644)
		}
	}
	return nil
}

func reconstructPPOutput(tokens []string) string {
	return strings.Join(tokens, " ")
}

// ============================================================================
// Compile single file (.S handling)
// ============================================================================

func compileAssemblyFile(input, output string, args *CliArgs, ctx *CompilationContext) error {
	pp := newPreprocessor(ctx.target)

	for _, path := range ctx.includePaths {
		pp.addIncludePath(path)
	}
	if !ctx.nostdinc {
		if exe, err := os.Executable(); err == nil {
			exeDir := filepath.Dir(exe)
			builtin := filepath.Join(exeDir, "..", "..", "include")
			if info, err := os.Stat(builtin); err == nil && info.IsDir() {
				if abs, err := filepath.Abs(builtin); err == nil {
					pp.addSystemIncludePath(abs)
				}
			}
			builtin2 := filepath.Join(exeDir, "include")
			if info, err := os.Stat(builtin2); err == nil && info.IsDir() {
				if abs, err := filepath.Abs(builtin2); err == nil {
					pp.addSystemIncludePath(abs)
				}
			}
		}
		targetSysPaths := targetSystemIncludePaths(ctx.target)
		extraSysPaths := []string{"/usr/local/include", "/usr/include/linux"}
		for _, sp := range append(targetSysPaths, extraSysPaths...) {
			if info, err := os.Stat(sp); err == nil && info.IsDir() {
				pp.addSystemIncludePath(sp)
			}
		}
	}
	pp.addDefine("__ASSEMBLER__", "1")
	for _, def := range ctx.defines {
		val := "1"
		if def.Value != nil {
			val = *def.Value
		}
		pp.addDefine(def.Name, val)
	}
	for _, name := range ctx.undefs {
		pp.addUndef(name)
	}

	var forcedTokens []string
	for _, forced := range ctx.forcedIncludes {
		tokens, err := pp.preprocessFile(forced)
		if err != nil {
			return fmt.Errorf("forced include '%s' failed", forced)
		}
		forcedTokens = append(forcedTokens, tokens...)
	}

	mainTokens, err := pp.preprocessFile(input)
	if err != nil {
		return fmt.Errorf("preprocessing failed for '%s'", input)
	}

	tokens := forcedTokens
	tokens = append(tokens, mainTokens...)

	ppOutput := reconstructPPOutput(tokens)

	tmpAsm, err := os.CreateTemp("", "bcc-asm-*.s")
	if err != nil {
		return fmt.Errorf("failed to create temp assembly file: %v", err)
	}
	tmpAsmPath := tmpAsm.Name()
	if _, err := tmpAsm.WriteString(ppOutput); err != nil {
		tmpAsm.Close()
		os.Remove(tmpAsmPath)
		return fmt.Errorf("failed to write preprocessed assembly: %v", err)
	}
	tmpAsm.Close()
	defer os.Remove(tmpAsmPath)

	var assembler string
	switch ctx.target {
	case TargetX8664:
		assembler = "as"
	case TargetI686:
		assembler = "i686-linux-gnu-as"
	case TargetAArch64:
		assembler = "aarch64-linux-gnu-as"
	case TargetRiscV64:
		assembler = "riscv64-linux-gnu-as"
	}

	asmArgs := []string{"-o", output}
	if args.march != nil {
		asmArgs = append(asmArgs, "-march="+*args.march)
	} else {
		switch ctx.target {
		case TargetRiscV64:
			asmArgs = append(asmArgs, "-march=rv64imafdc")
		case TargetAArch64:
			asmArgs = append(asmArgs, "-march=armv8-a")
		}
	}
	if args.mabi != nil {
		asmArgs = append(asmArgs, "-mabi="+*args.mabi)
	} else if ctx.target == TargetRiscV64 {
		asmArgs = append(asmArgs, "-mabi=lp64d")
	}
	asmArgs = append(asmArgs, tmpAsmPath)

	if err := runCommand(assembler, asmArgs...); err != nil {
		return fmt.Errorf("assembler '%s' failed: %v", assembler, err)
	}

	if args.genDepfile {
		var depPath string
		if args.depfilePath != nil {
			depPath = *args.depfilePath
		} else {
			stem := strings.TrimSuffix(filepath.Base(output), filepath.Ext(output))
			parent := filepath.Dir(output)
			depPath = filepath.Join(parent, "."+stem+".o.d")
		}
		depTarget := output
		if args.depfileTarget != nil {
			depTarget = *args.depfileTarget
		}
		deps := []string{input}
		deps = append(deps, pp.getIncludedFiles()...)
		depContent := depTarget + ": " + strings.Join(deps, " \\\n  ") + "\n"
		if args.depfilePhony {
			for _, d := range deps[1:] {
				depContent += "\n" + d + ":\n"
			}
		}
		os.WriteFile(depPath, []byte(depContent), 0644)
	}

	return nil
}

func runCommand(name string, args ...string) error {
	// In a real implementation this would use os/exec
	fmt.Fprintf(os.Stderr, "%s: warning: external command not implemented: %s %v\n", programName, name, args)
	return nil
}

// ============================================================================
// Compile single file
// ============================================================================

func compileSingleFile(input, output string, args *CliArgs, ctx *CompilationContext) error {
	inputLower := strings.ToLower(input)
	if strings.HasSuffix(inputLower, ".s") {
		return compileAssemblyFile(input, output, args, ctx)
	}

	fmt.Fprintf(os.Stderr, "[BCC] compile: %s -> %s\n", input, output)

	// TODO: Implement full compilation pipeline:
	// Phase 1-2: Preprocessor
	// Phase 3: Lexer
	// Phase 4: Parser
	// Phase 5: Semantic Analysis
	// Phase 6: IR Lowering
	// Phase 7: mem2reg (SSA)
	// Phase 8: Optimization
	// Phase 9: Phi Elimination
	// Phase 10-12: Code Generation, Assembly, Linking

	if args.genDepfile {
		var depPath string
		if args.depfilePath != nil {
			depPath = *args.depfilePath
		} else {
			stem := strings.TrimSuffix(filepath.Base(output), filepath.Ext(output))
			parent := filepath.Dir(output)
			depPath = filepath.Join(parent, "."+stem+".o.d")
		}
		depTarget := output
		if args.depfileTarget != nil {
			depTarget = *args.depfileTarget
		}
		depContent := depTarget + ": " + input + "\n"
		os.WriteFile(depPath, []byte(depContent), 0644)
	}

	return nil
}

// ============================================================================
// Link object files
// ============================================================================

func linkObjectFiles(objectFiles []string, output string, ctx *CompilationContext) error {
	fmt.Fprintf(os.Stderr, "[BCC] link: %v -> %s\n", objectFiles, output)
	return nil
}

// ============================================================================
// Preprocessor stub
// ============================================================================

type Preprocessor struct {
	target             Target
	preservePragmas    bool
	includePaths       []string
	systemIncludePaths []string
	defines            map[string]string
	undefs             map[string]bool
	includedFiles      []string
}

func newPreprocessor(target Target) *Preprocessor {
	return &Preprocessor{
		target:  target,
		defines: make(map[string]string),
		undefs:  make(map[string]bool),
	}
}

func (pp *Preprocessor) setPreservePragmas(v bool) {
	pp.preservePragmas = v
}

func (pp *Preprocessor) addIncludePath(path string) {
	pp.includePaths = append(pp.includePaths, path)
}

func (pp *Preprocessor) addSystemIncludePath(path string) {
	pp.systemIncludePaths = append(pp.systemIncludePaths, path)
}

func (pp *Preprocessor) addDefine(name, value string) {
	pp.defines[name] = value
}

func (pp *Preprocessor) addUndef(name string) {
	pp.undefs[name] = true
}

func (pp *Preprocessor) preprocessFile(path string) ([]string, error) {
	var input *os.File
	if path == "-" {
		input = os.Stdin
	} else {
		f, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		defer f.Close()
		input = f
	}

	pp.includedFiles = append(pp.includedFiles, path)

	var tokens []string
	scanner := bufio.NewScanner(input)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "#include") {
			// Simple include handling
			rest := strings.TrimSpace(line[8:])
			if len(rest) >= 2 {
				var incPath string
				if rest[0] == '"' {
					incPath = strings.Trim(rest, "\"")
				} else if rest[0] == '<' {
					incPath = strings.Trim(rest, "<>")
				}
				if incPath != "" {
					resolved := pp.resolveInclude(incPath)
					if resolved != "" {
						pp.includedFiles = append(pp.includedFiles, resolved)
					}
				}
			}
		} else if strings.HasPrefix(line, "#define") {
			parts := strings.Fields(line[8:])
			if len(parts) > 0 {
				name := parts[0]
				var value string
				if len(parts) > 1 {
					value = parts[1]
				}
				pp.defines[name] = value
			}
		}
		tokens = append(tokens, line)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return tokens, nil
}

func (pp *Preprocessor) resolveInclude(name string) string {
	searchPaths := append(pp.includePaths, pp.systemIncludePaths...)
	for _, dir := range searchPaths {
		p := filepath.Join(dir, name)
		if info, err := os.Stat(p); err == nil && !info.IsDir() {
			return p
		}
	}
	return ""
}

func (pp *Preprocessor) getIncludedFiles() []string {
	return pp.includedFiles
}

// ============================================================================
// Run compilation
// ============================================================================

func runCompilation(args *CliArgs) error {
	if err := validateInputFiles(args.inputFiles); err != nil {
		return err
	}

	if args.shared && !args.pic {
		fmt.Fprintf(os.Stderr, "%s: warning: creating shared library without -fPIC; PIC will be enabled implicitly, but explicit -fPIC is recommended\n", programName)
	}

	ctx := newCompilationContext(args)
	outputPath := resolveOutputPath(args)

	if ctx.verbose {
		fmt.Fprintf(os.Stderr, "%s: verbose: target=%s opt=%d pic=%t shared=%t debug=%t\n",
			programName, ctx.target, ctx.optimizationLevel, ctx.pic, ctx.shared, ctx.debugInfo)
		for _, input := range args.inputFiles {
			fmt.Fprintf(os.Stderr, "%s: verbose: input file: %s\n", programName, input)
		}
		if outputPath != nil {
			fmt.Fprintf(os.Stderr, "%s: verbose: output file: %s\n", programName, *outputPath)
		}
		for _, path := range ctx.includePaths {
			fmt.Fprintf(os.Stderr, "%s: verbose: include path: %s\n", programName, path)
		}
		for _, def := range ctx.defines {
			if def.Value != nil {
				fmt.Fprintf(os.Stderr, "%s: verbose: define: %s=%s\n", programName, def.Name, *def.Value)
			} else {
				fmt.Fprintf(os.Stderr, "%s: verbose: define: %s\n", programName, def.Name)
			}
		}
	}

	if args.preprocessOnly {
		return runPreprocessOnly(args, ctx, outputPath)
	}

	// Single-file fast path
	if len(args.inputFiles) == 1 {
		input := args.inputFiles[0]
		if isObjectOrArchive(input) && !args.compileOnly && !args.emitAssembly {
			finalOutput := "a.out"
			if outputPath != nil {
				finalOutput = *outputPath
			}
			return linkObjectFiles([]string{input}, finalOutput, ctx)
		}
		finalOutput := "a.out"
		if outputPath != nil {
			finalOutput = *outputPath
		}
		return compileSingleFile(input, finalOutput, args, ctx)
	}

	// Multi-file
	if args.compileOnly || args.emitAssembly {
		for _, input := range args.inputFiles {
			stem := strings.TrimSuffix(filepath.Base(input), filepath.Ext(input))
			if stem == "" {
				stem = "output"
			}
			var out string
			if args.emitAssembly {
				out = stem + ".s"
			} else {
				out = stem + ".o"
			}
			if err := compileSingleFile(input, out, args, ctx); err != nil {
				return err
			}
		}
		return nil
	}

	// Multi-file link mode
	var objectFiles []string
	var tempFiles []string

	for _, input := range args.inputFiles {
		if isObjectOrArchive(input) {
			objectFiles = append(objectFiles, input)
			continue
		}

		tmpFile, err := os.CreateTemp("", "bcc-obj-*.o")
		if err != nil {
			return fmt.Errorf("failed to create temporary object file: %v", err)
		}
		objPath := tmpFile.Name()
		tmpFile.Close()

		compileArgs := *args
		compileArgs.compileOnly = true
		if err := compileSingleFile(input, objPath, &compileArgs, ctx); err != nil {
			return err
		}

		objectFiles = append(objectFiles, objPath)
		tempFiles = append(tempFiles, objPath)
	}

	finalOutput := "a.out"
	if outputPath != nil {
		finalOutput = *outputPath
	}
	if err := linkObjectFiles(objectFiles, finalOutput, ctx); err != nil {
		return err
	}

	for _, tmp := range tempFiles {
		os.Remove(tmp)
	}

	return nil
}

// ============================================================================
// Entry Point
// ============================================================================

func main() {
	allArgs := os.Args[1:]
	if len(allArgs) == 0 {
		printUsage()
		os.Exit(1)
	}

	for _, a := range allArgs {
		if a == "-Wa,--version" {
			fmt.Println("GNU assembler (BCC built-in) 2.42")
			os.Exit(0)
		}
	}

	for _, a := range allArgs {
		if strings.HasPrefix(a, "-print-file-name=") {
			name := strings.TrimPrefix(a, "-print-file-name=")
			fmt.Println(name)
			os.Exit(0)
		}
	}

	for _, a := range allArgs {
		if a == "-dumpversion" {
			fmt.Println("12.2.0")
			os.Exit(0)
		}
	}

	for _, a := range allArgs {
		if a == "-dumpmachine" {
			var targetStr string
			for _, aa := range allArgs {
				if strings.HasPrefix(aa, "--target=") {
					targetStr = strings.TrimPrefix(aa, "--target=")
					break
				}
			}
			if targetStr == "" {
				targetStr = "x86-64"
			}
			var triple string
			switch targetStr {
			case "riscv64":
				triple = "riscv64-linux-gnu"
			case "aarch64":
				triple = "aarch64-linux-gnu"
			case "i686":
				triple = "i686-linux-gnu"
			default:
				triple = "x86_64-linux-gnu"
			}
			fmt.Println(triple)
			os.Exit(0)
		}
	}

	for _, a := range allArgs {
		if a == "--version" || a == "-v" {
			fmt.Printf("bcc (BCC) %s\n", version)
			fmt.Println("Copyright (C) 2024 BCC Project")
			fmt.Println("This is free software; see the source for copying conditions.")
			os.Exit(0)
		}
	}

	cliArgs, err := parseArgs(allArgs)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s: error: %v\n", programName, err)
		printUsage()
		os.Exit(1)
	}

	if err := runCompilation(cliArgs); err != nil {
		fmt.Fprintf(os.Stderr, "%s: error: %v\n", programName, err)
		os.Exit(1)
	}
}
