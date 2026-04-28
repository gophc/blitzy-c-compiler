package common

type Endianness int

const (
	EndianLittle Endianness = iota
	EndianBig
)

func (e Endianness) String() string {
	switch e {
	case EndianLittle:
		return "little-endian"
	case EndianBig:
		return "big-endian"
	default:
		return "unknown"
	}
}

type Target int

const (
	TargetX86_64 Target = iota
	TargetI686
	TargetAArch64
	TargetRiscV64
)

func (t Target) String() string {
	switch t {
	case TargetX86_64:
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

func TargetFromString(s string) *Target {
	lower := toLowerASCII(s)
	switch lower {
	case "x86-64", "x86_64", "amd64", "x86-64-linux-gnu", "x86_64-linux-gnu":
		t := TargetX86_64
		return &t
	case "i686", "i386", "i486", "i586", "i686-linux-gnu", "i386-linux-gnu":
		t := TargetI686
		return &t
	case "aarch64", "arm64", "aarch64-linux-gnu", "arm64-linux-gnu":
		t := TargetAArch64
		return &t
	case "riscv64", "riscv64gc", "riscv64-linux-gnu", "riscv64gc-linux-gnu":
		t := TargetRiscV64
		return &t
	default:
		return nil
	}
}

func toLowerASCII(s string) string {
	b := []byte(s)
	for i := range b {
		if b[i] >= 'A' && b[i] <= 'Z' {
			b[i] = b[i] + 32
		}
	}
	return string(b)
}

func (t Target) PointerWidth() int {
	switch t {
	case TargetI686:
		return 4
	case TargetX86_64, TargetAArch64, TargetRiscV64:
		return 8
	default:
		return 8
	}
}

func (t Target) PointerAlign() int {
	return t.PointerWidth()
}

func (t Target) Is64Bit() bool {
	return t.PointerWidth() == 8
}

func (t Target) Endianness() Endianness {
	return EndianLittle
}

func (t Target) LongSize() int {
	switch t {
	case TargetI686:
		return 4
	case TargetX86_64, TargetAArch64, TargetRiscV64:
		return 8
	default:
		return 8
	}
}

func (t Target) LongDoubleSize() int {
	switch t {
	case TargetI686:
		return 12
	case TargetX86_64, TargetAArch64, TargetRiscV64:
		return 16
	default:
		return 16
	}
}

func (t Target) LongDoubleAlign() int {
	switch t {
	case TargetI686:
		return 4
	case TargetX86_64, TargetAArch64, TargetRiscV64:
		return 16
	default:
		return 16
	}
}

func (t Target) ElfMachine() uint16 {
	switch t {
	case TargetX86_64:
		return 62
	case TargetI686:
		return 3
	case TargetAArch64:
		return 183
	case TargetRiscV64:
		return 243
	default:
		return 0
	}
}

func (t Target) ElfClass() uint8 {
	switch t {
	case TargetI686:
		return 1
	case TargetX86_64, TargetAArch64, TargetRiscV64:
		return 2
	default:
		return 2
	}
}

func (t Target) ElfData() uint8 {
	return 1
}

func (t Target) ElfFlags() uint32 {
	switch t {
	case TargetRiscV64:
		return 0x0001
	default:
		return 0
	}
}

type PredefMacro struct {
	Name        string
	Replacement string
}

func (t Target) PredefinedMacros() []PredefMacro {
	macros := []PredefMacro{
		{"__STDC__", "1"},
		{"__STDC_VERSION__", "201112L"},
		{"__STDC_HOSTED__", "1"},
		{"__linux__", "1"},
		{"__linux", "1"},
		{"linux", "1"},
		{"__unix__", "1"},
		{"__unix", "1"},
		{"unix", "1"},
		{"__ELF__", "1"},
	}

	switch t {
	case TargetX86_64:
		macros = append(macros, []PredefMacro{
			{"__x86_64__", "1"},
			{"__x86_64", "1"},
			{"__amd64__", "1"},
			{"__amd64", "1"},
			{"__LP64__", "1"},
			{"_LP64", "1"},
			{"__SSE__", "1"},
			{"__SSE2__", "1"},
		}...)
	case TargetI686:
		macros = append(macros, []PredefMacro{
			{"__i386__", "1"},
			{"__i386", "1"},
			{"i386", "1"},
			{"__i686__", "1"},
			{"__ILP32__", "1"},
		}...)
	case TargetAArch64:
		macros = append(macros, []PredefMacro{
			{"__aarch64__", "1"},
			{"__ARM_64BIT_STATE", "1"},
			{"__ARM_ARCH", "8"},
			{"__ARM_ARCH_ISA_A64", "1"},
			{"__LP64__", "1"},
			{"_LP64", "1"},
		}...)
	case TargetRiscV64:
		macros = append(macros, []PredefMacro{
			{"__riscv", "1"},
			{"__riscv_xlen", "64"},
			{"__riscv_flen", "64"},
			{"__riscv_float_abi_double", "1"},
			{"__riscv_mul", "1"},
			{"__riscv_div", "1"},
			{"__riscv_atomic", "1"},
			{"__riscv_compressed", "1"},
			{"__riscv_cmodel_medany", "1"},
			{"__LP64__", "1"},
			{"_LP64", "1"},
		}...)
	}

	return macros
}

func (t Target) MaxAlign() int {
	return 16
}

func (t Target) PageSize() int {
	return 4096
}

func (t Target) DynamicLinker() string {
	switch t {
	case TargetX86_64:
		return "/lib64/ld-linux-x86-64.so.2"
	case TargetI686:
		return "/lib/ld-linux.so.2"
	case TargetAArch64:
		return "/lib/ld-linux-aarch64.so.1"
	case TargetRiscV64:
		return "/lib/ld-linux-riscv64-lp64d.so.1"
	default:
		return ""
	}
}

func (t Target) DefaultEntryPoint() string {
	return "_start"
}

func (t Target) SystemIncludePaths() []string {
	switch t {
	case TargetX86_64:
		return []string{
			"/usr/lib/gcc/x86_64-linux-gnu/13/include",
			"/usr/lib/gcc/x86_64-linux-gnu/14/include",
			"/usr/include/x86_64-linux-gnu",
			"/usr/include",
		}
	case TargetI686:
		return []string{
			"/usr/lib/gcc/x86_64-linux-gnu/13/include",
			"/usr/lib/gcc/x86_64-linux-gnu/14/include",
			"/usr/include/i386-linux-gnu",
			"/usr/include/x86_64-linux-gnu",
			"/usr/include",
		}
	case TargetAArch64:
		return []string{
			"/usr/lib/gcc-cross/aarch64-linux-gnu/13/include",
			"/usr/lib/gcc-cross/aarch64-linux-gnu/14/include",
			"/usr/lib/gcc/x86_64-linux-gnu/13/include",
			"/usr/include/aarch64-linux-gnu",
			"/usr/include/x86_64-linux-gnu",
			"/usr/include",
		}
	case TargetRiscV64:
		return []string{
			"/usr/lib/gcc-cross/riscv64-linux-gnu/13/include",
			"/usr/lib/gcc-cross/riscv64-linux-gnu/14/include",
			"/usr/lib/gcc/x86_64-linux-gnu/13/include",
			"/usr/include/riscv64-linux-gnu",
			"/usr/include/x86_64-linux-gnu",
			"/usr/include",
		}
	default:
		return nil
	}
}

func (t Target) SystemLibraryPaths() []string {
	switch t {
	case TargetX86_64:
		return []string{
			"/usr/lib/x86_64-linux-gnu",
			"/lib/x86_64-linux-gnu",
			"/usr/lib64",
			"/lib64",
		}
	case TargetI686:
		return []string{
			"/usr/lib/i386-linux-gnu",
			"/lib/i386-linux-gnu",
			"/usr/lib32",
			"/lib32",
		}
	case TargetAArch64:
		return []string{
			"/usr/lib/aarch64-linux-gnu",
			"/lib/aarch64-linux-gnu",
			"/usr/lib",
			"/lib",
		}
	case TargetRiscV64:
		return []string{
			"/usr/lib/riscv64-linux-gnu",
			"/lib/riscv64-linux-gnu",
			"/usr/lib",
			"/lib",
		}
	default:
		return nil
	}
}
