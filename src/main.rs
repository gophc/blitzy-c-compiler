//! # BCC CLI Entry Point and Compilation Driver
//!
//! This is the binary crate entry point for BCC (Blitzy's C Compiler). It
//! orchestrates the entire compilation pipeline from command-line argument
//! parsing through final ELF output.
//!
//! ## Critical Architecture Decisions
//!
//! - **64 MiB Worker Thread**: All compilation work runs on a dedicated worker
//!   thread spawned with `std::thread::Builder::new().stack_size(64 * 1024 * 1024)`.
//!   The main thread only spawns this worker and waits for its result.
//! - **512-Recursion-Depth Limit**: Configured in [`CompilationContext`] and
//!   propagated to the parser and macro expander.
//! - **Zero-Dependency Mandate**: No external crates. All argument parsing is
//!   hand-rolled using `std::env::args()`.
//! - **GCC-Compatible CLI**: Flag syntax is compatible with `make CC=./bcc`
//!   for Linux kernel builds.
//!
//! ## Pipeline Stages
//!
//! ```text
//! Phase 1-2: Preprocessor  (trigraphs, line splicing, macro expansion)
//! Phase 3:   Lexer          (tokenization)
//! Phase 4:   Parser         (AST construction)
//! Phase 5:   Sema           (type checking, scope, builtins)
//! Phase 6:   IR Lowering    (AST → IR with allocas)
//! Phase 7:   mem2reg        (SSA construction)
//! Phase 8:   Optimization   (constant folding, DCE, CFG simplify)
//! Phase 9:   Phi Elimination(SSA → register-friendly form)
//! Phase 10:  Code Generation(architecture-specific)
//! Phase 11:  Assembly       (built-in assembler → .o)
//! Phase 12:  Linking        (built-in linker → ELF executable/shared object)
//! ```

// ============================================================================
// Standard library imports — the ONLY allowed external dependency
// ============================================================================

use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process;
use std::thread;

// ============================================================================
// BCC library crate imports (via `bcc::` prefix)
// ============================================================================

use bcc::common::diagnostics::DiagnosticEngine;
use bcc::common::source_map::SourceMap;
use bcc::common::string_interner::Interner;
use bcc::common::target::Target;
use bcc::common::temp_files::{create_temp_object_file, TempDir, TempFile};
use bcc::common::type_builder::TypeBuilder;

use bcc::backend::generation::{generate_code, CodegenContext};
use bcc::frontend::lexer::Lexer;
use bcc::frontend::parser::Parser;
use bcc::frontend::preprocessor::Preprocessor;
use bcc::frontend::sema::SemanticAnalyzer;
use bcc::ir::lowering::lower_translation_unit;
use bcc::ir::mem2reg::{eliminate_phi_nodes, run_mem2reg};
use bcc::passes::run_optimization_pipeline;

// ============================================================================
// Constants
// ============================================================================

/// Worker thread stack size: 64 MiB. Required for deeply nested kernel macro
/// expansions and complex AST structures. Mandated by AAP §0.7.3.
const WORKER_STACK_SIZE: usize = 64 * 1024 * 1024;

/// Maximum recursion depth for the parser and macro expander.
/// Prevents stack overflow on deeply nested constructs. Mandated by AAP §0.7.3.
const MAX_RECURSION_DEPTH: usize = 512;

/// Program name used in diagnostic messages (GCC-compatible format).
const PROGRAM_NAME: &str = "bcc";

/// Version string for `--version` output.
const VERSION: &str = "0.1.0";

// ============================================================================
// CliArgs — parsed command-line arguments
// ============================================================================

/// Holds all parsed command-line arguments for BCC.
///
/// This struct captures every GCC-compatible flag that BCC supports. The
/// [`parse_args`] function constructs this from `std::env::args()`.
///
/// # Supported Flags
///
/// | Flag                | Field              | Description                          |
/// |---------------------|--------------------|--------------------------------------|
/// | `--target=<arch>`   | `target`           | Target architecture                  |
/// | `-o <file>`         | `output_file`      | Output file path                     |
/// | `-c`                | `compile_only`     | Produce .o only (no linking)         |
/// | `-S`                | `emit_assembly`    | Emit assembly text                   |
/// | `-E`                | `preprocess_only`  | Preprocess only                      |
/// | `-g`                | `debug_info`       | Emit DWARF v4 debug info             |
/// | `-O0`               | `optimization_level` | Optimization level (only 0 supported) |
/// | `-fPIC` / `-fpic`   | `pic`              | Position-independent code            |
/// | `-shared`           | `shared`           | Produce shared object (.so)          |
/// | `-mretpoline`       | `retpoline`        | Retpoline thunks (x86-64 only)       |
/// | `-fcf-protection`   | `cf_protection`    | CET/IBT (x86-64 only)               |
/// | `-I<dir>`           | `include_paths`    | Include search path                  |
/// | `-D<macro>[=val]`   | `defines`          | Preprocessor define                  |
/// | `-L<dir>`           | `library_paths`    | Library search path                  |
/// | `-l<lib>`           | `libraries`        | Link library                         |
#[derive(Debug, Clone)]
struct CliArgs {
    /// Input .c source file paths.
    input_files: Vec<String>,
    /// Output file path (`-o`). None means default naming.
    output_file: Option<String>,
    /// Target architecture. Defaults to X86_64.
    target: Target,
    /// `-c`: compile to .o without linking.
    compile_only: bool,
    /// `-S`: emit assembly text output.
    emit_assembly: bool,
    /// `-E`: preprocess only, output to stdout or -o file.
    preprocess_only: bool,
    /// `-g`: emit DWARF v4 debug information.
    debug_info: bool,
    /// Optimization level (only 0 is supported).
    optimization_level: u8,
    /// `-fPIC` / `-fpic`: generate position-independent code.
    pic: bool,
    /// `-shared`: produce a shared object (ET_DYN).
    shared: bool,
    /// `-mretpoline`: enable retpoline thunks (x86-64 only).
    retpoline: bool,
    /// `-fcf-protection`: enable CET/IBT (x86-64 only).
    cf_protection: bool,
    /// `-I<dir>`: include search paths (multiple allowed).
    include_paths: Vec<String>,
    /// `-D<macro>[=value]`: preprocessor defines (multiple allowed).
    defines: Vec<(String, Option<String>)>,
    /// `-U<macro>`: preprocessor undefines (multiple allowed).
    /// Applied after all `-D` defines during preprocessor initialization.
    undefs: Vec<String>,
    /// `-L<dir>`: library search paths (multiple allowed).
    library_paths: Vec<String>,
    /// `-l<lib>`: libraries to link (multiple allowed).
    libraries: Vec<String>,
}

impl Default for CliArgs {
    fn default() -> Self {
        CliArgs {
            input_files: Vec::new(),
            output_file: None,
            target: Target::X86_64,
            compile_only: false,
            emit_assembly: false,
            preprocess_only: false,
            debug_info: false,
            optimization_level: 0,
            pic: false,
            shared: false,
            retpoline: false,
            cf_protection: false,
            include_paths: Vec::new(),
            defines: Vec::new(),
            undefs: Vec::new(),
            library_paths: Vec::new(),
            libraries: Vec::new(),
        }
    }
}

// ============================================================================
// CompilationContext — settings propagated through the pipeline
// ============================================================================

/// Compilation settings propagated through the entire pipeline.
///
/// Constructed from [`CliArgs`] by [`CompilationContext::from_cli_args`].
/// Every pipeline stage receives a reference to this context to access
/// target information, flags, and limits.
///
/// Some fields (library_paths, libraries, max_recursion_depth) are populated
/// but consumed later during the linker phase; suppress dead_code warnings.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct CompilationContext {
    /// Target architecture.
    target: Target,
    /// Whether to emit DWARF v4 debug info (`-g`).
    debug_info: bool,
    /// Optimization level (0 = no optimization).
    optimization_level: u8,
    /// Whether to generate position-independent code (`-fPIC`).
    pic: bool,
    /// Whether to produce a shared library (`-shared`).
    shared: bool,
    /// Whether retpoline thunks are enabled (`-mretpoline`, x86-64 only).
    retpoline: bool,
    /// Whether CET/IBT is enabled (`-fcf-protection`, x86-64 only).
    cf_protection: bool,
    /// Include search paths from `-I` flags.
    include_paths: Vec<String>,
    /// Preprocessor defines from `-D` flags.
    defines: Vec<(String, Option<String>)>,
    /// Preprocessor undefines from `-U` flags.
    /// Applied after defines during preprocessor initialization.
    undefs: Vec<String>,
    /// Library search paths from `-L` flags.
    library_paths: Vec<String>,
    /// Libraries to link from `-l` flags.
    libraries: Vec<String>,
    /// Maximum recursion depth for parser and macro expander.
    /// Fixed at 512 per AAP §0.7.3.
    max_recursion_depth: usize,
}

impl CompilationContext {
    /// Construct a [`CompilationContext`] from parsed CLI arguments.
    ///
    /// Sets the recursion depth limit to 512 as mandated by AAP §0.7.3.
    fn from_cli_args(args: &CliArgs) -> Self {
        CompilationContext {
            target: args.target,
            debug_info: args.debug_info,
            optimization_level: args.optimization_level,
            pic: args.pic || args.shared, // -shared implies PIC
            shared: args.shared,
            retpoline: args.retpoline,
            cf_protection: args.cf_protection,
            include_paths: args.include_paths.clone(),
            defines: args.defines.clone(),
            undefs: args.undefs.clone(),
            library_paths: args.library_paths.clone(),
            libraries: args.libraries.clone(),
            max_recursion_depth: MAX_RECURSION_DEPTH,
        }
    }
}

// ============================================================================
// CLI Argument Parsing
// ============================================================================

/// Parse command-line arguments into a [`CliArgs`] struct.
///
/// Hand-rolled argument parser that supports GCC-compatible flag syntax.
/// Flags can be combined as `-Idir` (attached) or `-I dir` (separate).
///
/// # Errors
///
/// Returns `Err(message)` if:
/// - An unknown `--target=` value is provided
/// - `-o` is missing its argument
/// - Mutually exclusive flags (`-c`, `-S`, `-E`) are combined
/// - No input files are provided
fn parse_args(args: &[String]) -> Result<CliArgs, String> {
    let mut cli = CliArgs::default();
    let mut i = 0;

    while i < args.len() {
        let arg = &args[i];

        // --target=<arch>
        if let Some(target_str) = arg.strip_prefix("--target=") {
            match Target::from_str(target_str) {
                Some(t) => cli.target = t,
                None => {
                    return Err(format!(
                        "unknown target '{}'. Supported targets: x86-64, i686, aarch64, riscv64",
                        target_str
                    ));
                }
            }
            i += 1;
            continue;
        }

        // --version
        if arg == "--version" {
            println!("{} version {}", PROGRAM_NAME, VERSION);
            process::exit(0);
        }

        // --help
        if arg == "--help" || arg == "-h" {
            print_usage();
            process::exit(0);
        }

        // -o <output>
        if arg == "-o" {
            i += 1;
            if i >= args.len() {
                return Err("missing argument to '-o'".to_string());
            }
            cli.output_file = Some(args[i].clone());
            i += 1;
            continue;
        }

        // -c (compile only)
        if arg == "-c" {
            cli.compile_only = true;
            i += 1;
            continue;
        }

        // -S (emit assembly)
        if arg == "-S" {
            cli.emit_assembly = true;
            i += 1;
            continue;
        }

        // -E (preprocess only)
        if arg == "-E" {
            cli.preprocess_only = true;
            i += 1;
            continue;
        }

        // -g (debug info)
        if arg == "-g" {
            cli.debug_info = true;
            i += 1;
            continue;
        }

        // -O<level>
        if let Some(level_str) = arg.strip_prefix("-O") {
            let level = match level_str {
                "0" | "" => 0,
                "1" => 1,
                "2" => 2,
                "3" => 3,
                "s" => 2, // -Os maps to -O2
                _ => {
                    // GCC silently treats unknown -O levels as -O2
                    eprintln!(
                        "{}: warning: unknown optimization level '{}', using -O0",
                        PROGRAM_NAME, level_str
                    );
                    0
                }
            };
            cli.optimization_level = level;
            i += 1;
            continue;
        }

        // -fPIC / -fpic
        if arg == "-fPIC" || arg == "-fpic" {
            cli.pic = true;
            i += 1;
            continue;
        }

        // -shared
        if arg == "-shared" {
            cli.shared = true;
            i += 1;
            continue;
        }

        // -mretpoline
        if arg == "-mretpoline" {
            cli.retpoline = true;
            i += 1;
            continue;
        }

        // -fcf-protection
        if arg == "-fcf-protection" || arg == "-fcf-protection=full" {
            cli.cf_protection = true;
            i += 1;
            continue;
        }
        if arg == "-fcf-protection=none" {
            cli.cf_protection = false;
            i += 1;
            continue;
        }

        // -I<dir> or -I <dir>
        if arg == "-I" {
            i += 1;
            if i >= args.len() {
                return Err("missing argument to '-I'".to_string());
            }
            cli.include_paths.push(args[i].clone());
            i += 1;
            continue;
        }
        if let Some(dir) = arg.strip_prefix("-I") {
            cli.include_paths.push(dir.to_string());
            i += 1;
            continue;
        }

        // -D<macro>[=value] or -D <macro>[=value]
        if arg == "-D" {
            i += 1;
            if i >= args.len() {
                return Err("missing argument to '-D'".to_string());
            }
            let define_str = &args[i];
            cli.defines.push(parse_define(define_str));
            i += 1;
            continue;
        }
        if let Some(define_str) = arg.strip_prefix("-D") {
            cli.defines.push(parse_define(define_str));
            i += 1;
            continue;
        }

        // -U<macro> (undefine — store and propagate to preprocessor)
        if arg == "-U" {
            i += 1;
            if i >= args.len() {
                return Err("missing argument to '-U'".to_string());
            }
            cli.undefs.push(args[i].clone());
            i += 1;
            continue;
        }
        if let Some(undef_str) = arg.strip_prefix("-U") {
            cli.undefs.push(undef_str.to_string());
            i += 1;
            continue;
        }

        // -L<dir> or -L <dir>
        if arg == "-L" {
            i += 1;
            if i >= args.len() {
                return Err("missing argument to '-L'".to_string());
            }
            cli.library_paths.push(args[i].clone());
            i += 1;
            continue;
        }
        if let Some(dir) = arg.strip_prefix("-L") {
            cli.library_paths.push(dir.to_string());
            i += 1;
            continue;
        }

        // -l<lib> or -l <lib>
        if arg == "-l" {
            i += 1;
            if i >= args.len() {
                return Err("missing argument to '-l'".to_string());
            }
            cli.libraries.push(args[i].clone());
            i += 1;
            continue;
        }
        if let Some(lib) = arg.strip_prefix("-l") {
            cli.libraries.push(lib.to_string());
            i += 1;
            continue;
        }

        // -w (suppress all warnings — GCC compat)
        if arg == "-w" {
            i += 1;
            continue;
        }

        // -Wall, -Wextra, -Werror, -W<anything> — accepted silently for GCC compat
        if arg.starts_with("-W") {
            i += 1;
            continue;
        }

        // -std=<standard> — accepted silently (we always target C11)
        if arg.starts_with("-std=") {
            i += 1;
            continue;
        }

        // -m32, -m64, -march=, -mtune= — accepted silently for GCC compat
        if arg.starts_with("-m") && arg != "-mretpoline" {
            i += 1;
            continue;
        }

        // -f<flag> — accept known flags, warn on unknown
        if arg.starts_with("-f")
            && arg != "-fPIC"
            && arg != "-fpic"
            && arg != "-fcf-protection"
            && arg != "-fcf-protection=full"
            && arg != "-fcf-protection=none"
        {
            // Silently accept common GCC flags for kernel build compatibility
            i += 1;
            continue;
        }

        // -pipe — accepted silently (GCC compat)
        if arg == "-pipe" {
            i += 1;
            continue;
        }

        // -nostdinc, -nostdlib, -nodefaultlibs — accepted silently
        if arg == "-nostdinc"
            || arg == "-nostdlib"
            || arg == "-nodefaultlibs"
            || arg == "-nostartfiles"
        {
            i += 1;
            continue;
        }

        // -static
        if arg == "-static" {
            i += 1;
            continue;
        }

        // -pthread
        if arg == "-pthread" {
            i += 1;
            continue;
        }

        // Unknown flags starting with '-' — warn but continue (GCC compat)
        if arg.starts_with('-') && arg != "-" {
            eprintln!("{}: warning: unrecognized flag '{}'", PROGRAM_NAME, arg);
            i += 1;
            continue;
        }

        // Anything else is an input file
        cli.input_files.push(arg.clone());
        i += 1;
    }

    // Validate: at least one input file required
    if cli.input_files.is_empty() {
        return Err("no input files".to_string());
    }

    // Validate: mutual exclusivity of -c, -S, -E
    let mode_count = cli.compile_only as u8 + cli.emit_assembly as u8 + cli.preprocess_only as u8;
    if mode_count > 1 {
        return Err("only one of '-c', '-S', '-E' may be specified".to_string());
    }

    Ok(cli)
}

/// Parse a `-D` define string into a `(name, optional_value)` pair.
///
/// Handles both `-DFOO` (define as empty) and `-DFOO=bar` (define with value).
fn parse_define(s: &str) -> (String, Option<String>) {
    if let Some(eq_pos) = s.find('=') {
        let name = s[..eq_pos].to_string();
        let value = s[eq_pos + 1..].to_string();
        (name, Some(value))
    } else {
        (s.to_string(), None)
    }
}

// ============================================================================
// Usage / Help
// ============================================================================

/// Print usage information to stderr (GCC-compatible format).
///
/// Shows all supported flags with brief descriptions. Called when `--help`
/// is passed or when argument parsing fails.
fn print_usage() {
    eprintln!("Usage: {} [options] <input.c> [-o output]", PROGRAM_NAME);
    eprintln!();
    eprintln!("BCC — Blitzy's C Compiler (C11 with GCC extensions)");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --target=<arch>     Set target architecture:");
    eprintln!("                        x86-64, i686, aarch64, riscv64");
    eprintln!("  -o <file>           Write output to <file>");
    eprintln!("  -c                  Compile and assemble, but do not link");
    eprintln!("  -S                  Compile only; output assembly text");
    eprintln!("  -E                  Preprocess only; output to stdout");
    eprintln!("  -g                  Emit DWARF v4 debug information");
    eprintln!("  -O0                 No optimization (default)");
    eprintln!("  -fPIC / -fpic       Generate position-independent code");
    eprintln!("  -shared             Produce a shared object (.so)");
    eprintln!("  -mretpoline         Enable retpoline thunks (x86-64 only)");
    eprintln!("  -fcf-protection     Enable CET/IBT (x86-64 only)");
    eprintln!("  -I<dir>             Add include search path");
    eprintln!("  -D<macro>[=value]   Define preprocessor macro");
    eprintln!("  -L<dir>             Add library search path");
    eprintln!("  -l<lib>             Link library");
    eprintln!("  --version           Print version information and exit");
    eprintln!("  --help              Print this help message and exit");
}

// ============================================================================
// Output Path Resolution
// ============================================================================

/// Determine the output file path based on CLI args and compilation mode.
///
/// - If `-o <file>` is specified, use that path.
/// - If `-E`, output goes to stdout (returns `None`).
/// - If `-S`, change extension to `.s`.
/// - If `-c`, change extension to `.o`.
/// - Otherwise, default to `a.out`.
fn resolve_output_path(args: &CliArgs) -> Option<String> {
    if args.preprocess_only && args.output_file.is_none() {
        // -E without -o: output to stdout
        return None;
    }

    if let Some(ref out) = args.output_file {
        return Some(out.clone());
    }

    // Derive from first input file
    if let Some(ref input) = args.input_files.first() {
        let input_path = Path::new(input);
        let stem = input_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output");

        if args.emit_assembly {
            return Some(format!("{}.s", stem));
        }
        if args.compile_only {
            return Some(format!("{}.o", stem));
        }
    }

    // Default: a.out
    Some("a.out".to_string())
}

// ============================================================================
// Compilation Pipeline
// ============================================================================

/// Validate that all input files exist and are readable.
///
/// Uses `std::fs::read_to_string` to verify the file is accessible
/// before starting the compilation pipeline. This provides early,
/// clear error messages rather than cryptic failures deep in the
/// preprocessor.
///
/// # Returns
///
/// `Ok(())` if all files are readable, `Err(message)` for the first
/// unreadable file.
fn validate_input_files(input_files: &[String]) -> Result<(), String> {
    for input in input_files {
        let path = Path::new(input);
        if !path.exists() {
            return Err(format!("'{}': No such file or directory", input));
        }
        // Verify the file is actually readable by attempting to read it.
        // This catches permission errors early. We discard the content since
        // the preprocessor will re-read it with its own PUA-aware encoding.
        let _content: io::Result<String> = fs::read_to_string(path);
        if let Err(ref e) = _content {
            return Err(format!("'{}': {}", input, e));
        }
    }
    Ok(())
}

/// Run the full compilation pipeline for all input files.
///
/// This is the core function that executes on the 64 MiB worker thread.
/// It processes each input file through the complete compilation pipeline
/// (or subset thereof, depending on `-E`, `-S`, `-c` flags) and produces
/// the final output.
///
/// # Pipeline Modes
///
/// - **`-E` (preprocess only)**: Preprocessor → output to stdout or file
/// - **`-S` (assembly output)**: Full pipeline through codegen → assembly text
/// - **`-c` (compile only)**: Full pipeline through assembler → .o file
/// - **Default (link)**: Full pipeline through linker → ELF executable or .so
///
/// # Multi-File Compilation
///
/// When multiple input files are provided without `-E` or `-S`:
/// 1. Each file is compiled to a temporary .o file
/// 2. All .o files are linked together to produce the final output
///
/// # Returns
///
/// `Ok(())` on success, `Err(message)` on fatal error.
fn run_compilation(args: CliArgs) -> Result<(), String> {
    // Validate all input files exist and are readable before starting the pipeline.
    validate_input_files(&args.input_files)?;

    // Warn when -shared is used without explicit -fPIC — the compiler will
    // implicitly enable PIC, but the user should be aware that creating a
    // shared library without -fPIC may produce incorrect code with text
    // relocations in some scenarios.
    if args.shared && !args.pic {
        eprintln!(
            "{}: warning: creating shared library without -fPIC; \
             PIC will be enabled implicitly, but explicit -fPIC is recommended",
            PROGRAM_NAME
        );
    }

    let ctx = CompilationContext::from_cli_args(&args);
    let output_path = resolve_output_path(&args);

    // -E mode: preprocess only
    if args.preprocess_only {
        return run_preprocess_only(&args, &ctx, output_path.as_deref());
    }

    // Single-file fast path
    if args.input_files.len() == 1 {
        return compile_single_file(
            &args.input_files[0],
            output_path.as_deref().unwrap_or("a.out"),
            &args,
            &ctx,
        );
    }

    // Multi-file: compile each to .o, then link
    if args.compile_only || args.emit_assembly {
        // -c or -S with multiple files: process each independently
        for input in &args.input_files {
            let out = if let Some(ref explicit) = args.output_file {
                // With -o and multiple files, only the last file gets -o
                // (GCC behavior with -c -o applies to each file individually)
                if args.input_files.len() == 1 {
                    explicit.clone()
                } else {
                    let stem = Path::new(input)
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("output");
                    if args.emit_assembly {
                        format!("{}.s", stem)
                    } else {
                        format!("{}.o", stem)
                    }
                }
            } else {
                let stem = Path::new(input)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("output");
                if args.emit_assembly {
                    format!("{}.s", stem)
                } else {
                    format!("{}.o", stem)
                }
            };
            compile_single_file(input, &out, &args, &ctx)?;
        }
        return Ok(());
    }

    // Multi-file link mode: compile each to temp .o, then link all
    let _temp_dir =
        TempDir::new().map_err(|e| format!("failed to create temporary directory: {}", e))?;

    let mut object_files: Vec<PathBuf> = Vec::new();

    for input in &args.input_files {
        let temp_obj: TempFile = create_temp_object_file()
            .map_err(|e| format!("failed to create temporary object file: {}", e))?;
        let obj_path = temp_obj.path().to_path_buf();

        // Compile to temporary .o
        let mut compile_args = args.clone();
        compile_args.compile_only = true;
        compile_single_file(
            input,
            obj_path.to_str().unwrap_or("temp.o"),
            &compile_args,
            &ctx,
        )?;

        object_files.push(obj_path);
        // Keep the temp file alive (don't drop it yet)
        temp_obj.keep();
    }

    // Link all object files together
    let final_output = output_path.as_deref().unwrap_or("a.out");
    link_object_files(&object_files, final_output, &ctx)?;

    // Clean up temporary .o files
    for obj_path in &object_files {
        let _ = fs::remove_file(obj_path);
    }

    Ok(())
}

/// Preprocess-only mode (`-E`): run preprocessor and output result.
fn run_preprocess_only(
    args: &CliArgs,
    ctx: &CompilationContext,
    output_path: Option<&str>,
) -> Result<(), String> {
    for input in &args.input_files {
        let mut source_map = SourceMap::new();
        let mut diagnostics = DiagnosticEngine::new();
        let mut interner = Interner::new();

        let mut pp =
            Preprocessor::new(&mut source_map, &mut diagnostics, ctx.target, &mut interner);

        // Add include paths from CLI
        for path in &ctx.include_paths {
            pp.add_include_path(path);
        }

        // Add compiler-builtin include path (for stdarg.h, stddef.h, etc.).
        // Locate the `include/` directory relative to the bcc binary.
        if let Ok(exe) = std::env::current_exe() {
            if let Some(exe_dir) = exe.parent() {
                let builtin = exe_dir.join("../../include");
                if builtin.is_dir() {
                    if let Some(s) = builtin
                        .canonicalize()
                        .ok()
                        .and_then(|p| p.to_str().map(|s| s.to_string()))
                    {
                        pp.add_system_include_path(&s);
                    }
                }
                // Also try directly next to executable.
                let builtin2 = exe_dir.join("include");
                if builtin2.is_dir() {
                    if let Some(s) = builtin2
                        .canonicalize()
                        .ok()
                        .and_then(|p| p.to_str().map(|s| s.to_string()))
                    {
                        pp.add_system_include_path(&s);
                    }
                }
            }
        }

        // Add default system include paths so that `#include <stdio.h>` etc. work.
        let system_paths = [
            "/usr/include",
            "/usr/local/include",
            "/usr/include/x86_64-linux-gnu",
            "/usr/include/linux",
        ];
        for sp in &system_paths {
            if std::path::Path::new(sp).is_dir() {
                pp.add_system_include_path(sp);
            }
        }

        // Add defines from CLI
        for (name, value) in &ctx.defines {
            let val = value.as_deref().unwrap_or("1");
            pp.add_define(name, val);
        }

        // Apply undefs from CLI (after defines, matching GCC behaviour)
        for name in &ctx.undefs {
            pp.add_undef(name);
        }

        // Run preprocessor
        let tokens = pp.preprocess_file(input).map_err(|_| {
            diagnostics.print_all(&source_map);
            format!("preprocessing failed for '{}'", input)
        })?;

        // Reconstruct preprocessed output from tokens
        let mut output = String::new();
        for token in &tokens {
            if !token.text.is_empty() {
                output.push_str(&token.text);
                output.push(' ');
            }
        }

        // Write output
        if let Some(path) = output_path {
            fs::write(path, &output)
                .map_err(|e| format!("failed to write output to '{}': {}", path, e))?;
        } else {
            // Write to stdout
            let stdout = io::stdout();
            let mut handle = stdout.lock();
            handle
                .write_all(output.as_bytes())
                .map_err(|e| format!("failed to write to stdout: {}", e))?;
        }

        // Print any accumulated diagnostics
        if diagnostics.has_errors() {
            diagnostics.print_all(&source_map);
            return Err(format!("preprocessing failed for '{}'", input));
        }
    }

    Ok(())
}

/// Compile a single source file through the full pipeline.
///
/// Runs all applicable phases (preprocessing through code generation/linking)
/// based on the compilation flags.
fn compile_single_file(
    input: &str,
    output: &str,
    args: &CliArgs,
    ctx: &CompilationContext,
) -> Result<(), String> {
    let mut source_map = SourceMap::new();
    let mut diagnostics = DiagnosticEngine::new();
    let mut interner = Interner::new();
    let type_builder = TypeBuilder::new(ctx.target);

    // ========================================================================
    // Phase 1-2: Preprocessing
    // ========================================================================

    let mut pp = Preprocessor::new(&mut source_map, &mut diagnostics, ctx.target, &mut interner);

    // Configure preprocessor with CLI options
    for path in &ctx.include_paths {
        pp.add_include_path(path);
    }
    // Compiler-builtin include path (for stdarg.h, stddef.h, etc.).
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            let builtin = exe_dir.join("../../include");
            if builtin.is_dir() {
                if let Some(s) = builtin
                    .canonicalize()
                    .ok()
                    .and_then(|p| p.to_str().map(|s| s.to_string()))
                {
                    pp.add_system_include_path(&s);
                }
            }
            let builtin2 = exe_dir.join("include");
            if builtin2.is_dir() {
                if let Some(s) = builtin2
                    .canonicalize()
                    .ok()
                    .and_then(|p| p.to_str().map(|s| s.to_string()))
                {
                    pp.add_system_include_path(&s);
                }
            }
        }
    }
    // Default system include paths.
    let system_paths = [
        "/usr/include",
        "/usr/local/include",
        "/usr/include/x86_64-linux-gnu",
        "/usr/include/linux",
    ];
    for sp in &system_paths {
        if std::path::Path::new(sp).is_dir() {
            pp.add_system_include_path(sp);
        }
    }
    for (name, value) in &ctx.defines {
        let val = value.as_deref().unwrap_or("1");
        pp.add_define(name, val);
    }

    // Apply undefs from CLI (after defines, matching GCC behaviour)
    for name in &ctx.undefs {
        pp.add_undef(name);
    }

    let pp_tokens = pp.preprocess_file(input).map_err(|_| {
        diagnostics.print_all(&source_map);
        format!("preprocessing failed for '{}'", input)
    })?;

    // Check for preprocessing errors
    if diagnostics.has_errors() {
        diagnostics.print_all(&source_map);
        return Err(format!("preprocessing failed for '{}'", input));
    }

    // ========================================================================
    // Phase 3: Lexing
    // ========================================================================

    // Reconstruct source text from preprocessor tokens for the lexer.
    // The lexer expects a contiguous source string; we build one from the
    // preprocessed token stream, using the first file's content registered
    // in the source map if available, or reconstructing from tokens.
    let pp_text: String = pp_tokens
        .iter()
        .map(|t| t.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    // Register the preprocessed text as a virtual file in the source map
    // so the lexer can produce spans that the diagnostic engine can resolve.
    let pp_file_id = source_map.add_file(format!("<preprocessed:{}>", input), pp_text.clone());

    let lexer = Lexer::new(&pp_text, pp_file_id, &mut interner, &mut diagnostics);

    // Check for lexer errors (emitted during tokenization via next_token calls
    // from the parser, so we don't tokenize_all here — the parser drives the lexer)

    // ========================================================================
    // Phase 4: Parsing
    // ========================================================================

    let mut parser = Parser::new(lexer, ctx.target);
    let mut translation_unit = parser.parse();

    // Check for parse errors
    if diagnostics.has_errors() {
        diagnostics.print_all(&source_map);
        return Err(format!("parsing failed for '{}'", input));
    }

    // ========================================================================
    // Phase 5: Semantic Analysis
    // ========================================================================

    // The SemanticAnalyzer borrows `diagnostics` mutably, so we scope its
    // lifetime in a block to release the borrow before checking diagnostics.
    let sema_ok = {
        let mut sema =
            SemanticAnalyzer::new(&mut diagnostics, &type_builder, ctx.target, &interner);

        let analyze_result = sema.analyze(&mut translation_unit);
        let finalize_result = sema.finalize();

        analyze_result.is_ok() && finalize_result.is_ok()
    };
    // `sema` is now dropped — mutable borrow of `diagnostics` is released.

    if !sema_ok || diagnostics.has_errors() {
        diagnostics.print_all(&source_map);
        return Err(format!("semantic analysis failed for '{}'", input));
    }

    // Create a fresh symbol table for lowering. The type-annotated AST from
    // semantic analysis carries resolved types in its nodes. The lowering
    // pass uses the symbol table primarily for linkage and storage class
    // information, which is also encoded in the AST attributes.
    let symbol_table = bcc::frontend::sema::SymbolTable::new();

    // ========================================================================
    // Phase 6: IR Lowering (AST → IR with allocas)
    // ========================================================================

    let mut ir_module = lower_translation_unit(
        &translation_unit,
        &symbol_table,
        &ctx.target,
        &type_builder,
        &source_map,
        &mut diagnostics,
        &interner,
    )
    .map_err(|e| {
        diagnostics.print_all(&source_map);
        format!("IR lowering failed for '{}': {}", input, e)
    })?;

    if diagnostics.has_errors() {
        diagnostics.print_all(&source_map);
        return Err(format!("IR lowering failed for '{}'", input));
    }

    // Phase 7: SSA Construction (mem2reg — alloca promotion)
    // ========================================================================

    run_mem2reg(&mut ir_module);

    // Phase 8: Optimization (constant folding, DCE, CFG simplification)
    // ========================================================================

    let _optimized = run_optimization_pipeline(&mut ir_module, ctx.optimization_level);

    // Phase 9: Phi Elimination (SSA → register-friendly form)
    // ========================================================================

    for func in ir_module.functions.iter_mut() {
        if func.is_definition {
            eliminate_phi_nodes(func);
        }
    }

    // Phase 10-12: Code Generation, Assembly, Linking
    // ========================================================================

    let codegen_ctx = CodegenContext {
        target: ctx.target,
        debug_info: ctx.debug_info,
        optimization_level: ctx.optimization_level,
        pic: ctx.pic,
        shared: ctx.shared,
        retpoline: ctx.retpoline,
        cf_protection: ctx.cf_protection,
        output_path: output.to_string(),
        compile_only: args.compile_only,
        emit_assembly: args.emit_assembly,
    };

    let output_bytes = generate_code(&ir_module, &codegen_ctx, &mut diagnostics, &source_map)
        .map_err(|e| {
            diagnostics.print_all(&source_map);
            format!("code generation failed for '{}': {}", input, e)
        })?;

    if diagnostics.has_errors() {
        diagnostics.print_all(&source_map);
        return Err(format!("code generation failed for '{}'", input));
    }

    // Write the output bytes to the destination file
    fs::write(output, &output_bytes)
        .map_err(|e| format!("failed to write output to '{}': {}", output, e))?;

    // For ELF executables, set the executable permission bit
    if !args.compile_only && !args.emit_assembly {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = fs::Permissions::from_mode(0o755);
            let _ = fs::set_permissions(output, perms);
        }
    }

    // Print any accumulated warnings (non-fatal diagnostics)
    if diagnostics.warning_count() > 0 {
        diagnostics.print_all(&source_map);
    }

    Ok(())
}

/// Link multiple object files into a final executable or shared library.
///
/// Invokes the built-in linker infrastructure (`bcc::backend::linker_common`)
/// to perform symbol resolution, section merging, relocation application,
/// and ELF output generation.
///
/// # Pipeline
///
/// 1. Read each `.o` file from disk.
/// 2. Parse each ELF object into a [`LinkerInput`] via the ELF parser in
///    `generation.rs`.
/// 3. Construct a [`LinkerConfig`] from the [`CompilationContext`].
/// 4. Create the architecture-specific relocation handler.
/// 5. Call `link()` from `linker_common` to produce a [`LinkerOutput`].
/// 6. Write the output bytes to the destination file with executable
///    permissions.
///
/// # Errors
///
/// Returns `Err(message)` if any object file cannot be read, the linker
/// reports unresolved symbols, or the output cannot be written.
fn link_object_files(
    object_files: &[PathBuf],
    output: &str,
    ctx: &CompilationContext,
) -> Result<(), String> {
    use bcc::backend::generation::{create_relocation_handler, parse_elf_object_to_linker_input};
    use bcc::backend::linker_common::{link, LinkerConfig, OutputType};

    // Determine output type based on compilation flags.
    let output_type = if ctx.shared {
        OutputType::SharedLibrary
    } else {
        OutputType::Executable
    };

    let mut config = LinkerConfig::new(ctx.target, output_type);
    config.output_path = output.to_string();
    config.pic = ctx.pic;
    config.library_paths = ctx.library_paths.clone();
    config.libraries = ctx.libraries.clone();
    config.emit_debug = ctx.debug_info;

    // Parse each object file into a LinkerInput.
    let mut inputs = Vec::with_capacity(object_files.len());
    for (idx, obj_path) in object_files.iter().enumerate() {
        let data = fs::read(obj_path)
            .map_err(|e| format!("failed to read object file '{}': {}", obj_path.display(), e))?;
        let filename = obj_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("input.o");
        let linker_input = parse_elf_object_to_linker_input(idx as u32, filename, &data);
        inputs.push(linker_input);
    }

    // Create the architecture-specific relocation handler.
    let handler = create_relocation_handler(ctx.target);

    // Run the built-in linker.
    let mut diagnostics = DiagnosticEngine::new();
    let linker_output = link(&config, inputs, handler.as_ref(), &mut diagnostics)
        .map_err(|e| format!("linking failed: {}", e))?;

    if diagnostics.has_errors() {
        let source_map = SourceMap::new();
        diagnostics.print_all(&source_map);
        return Err("linking failed due to errors".to_string());
    }

    // Write the linked output to disk.
    fs::write(output, &linker_output.elf_data)
        .map_err(|e| format!("failed to write output to '{}': {}", output, e))?;

    // Set executable permission bit on Unix.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = fs::Permissions::from_mode(0o755);
        let _ = fs::set_permissions(output, perms);
    }

    Ok(())
}

// ============================================================================
// Entry Point
// ============================================================================

/// BCC entry point.
///
/// 1. Parses command-line arguments
/// 2. On error, prints usage to stderr and exits with code 1
/// 3. Spawns a 64 MiB worker thread for all compilation work
/// 4. Joins the worker thread and propagates its exit code
///
/// Exit codes:
/// - 0: success
/// - 1: compilation error or internal error
fn main() {
    // Collect args (skip program name)
    let all_args: Vec<String> = env::args().collect();
    let args_slice = if all_args.len() > 1 {
        &all_args[1..]
    } else {
        // No arguments — print usage and exit
        print_usage();
        process::exit(1);
    };

    // Parse command-line arguments
    let cli_args = match parse_args(args_slice) {
        Ok(args) => args,
        Err(msg) => {
            eprintln!("{}: error: {}", PROGRAM_NAME, msg);
            print_usage();
            process::exit(1);
        }
    };

    // Spawn the worker thread with 64 MiB stack for compilation work.
    // The main thread only spawns this worker and waits for its result.
    // This is mandated by AAP §0.7.3 to handle deeply nested kernel
    // macro expansions and complex AST structures without stack overflow.
    let builder = thread::Builder::new()
        .name("bcc-worker".to_string())
        .stack_size(WORKER_STACK_SIZE);

    let handle = builder
        .spawn(move || run_compilation(cli_args))
        .expect("failed to spawn worker thread");

    // Join the worker thread and propagate its exit code
    match handle.join() {
        Ok(Ok(())) => {
            // Successful compilation
            process::exit(0);
        }
        Ok(Err(msg)) => {
            // Compilation error — message already printed by the pipeline
            eprintln!("{}: error: {}", PROGRAM_NAME, msg);
            process::exit(1);
        }
        Err(_panic_payload) => {
            // Worker thread panicked — internal compiler error
            eprintln!(
                "{}: internal compiler error: worker thread panicked",
                PROGRAM_NAME
            );
            eprintln!(
                "This is a bug in BCC. Please report it with the source file \
                 that triggered the error."
            );
            process::exit(1);
        }
    }
}
