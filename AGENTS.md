# BCC — Blitzy's C Compiler

## Architecture

Zero-dependency C11 compiler in Rust producing Linux ELF for x86-64, i686, AArch64, RISC-V 64. All components (assembler, linker, DWARF emitter, FxHash, long-double math) are hand-rolled — `[dependencies]` in `Cargo.toml` is empty.

Module dependency order (enforced in `src/lib.rs`): `common` → `frontend` → `ir` → `passes` → `backend`.

## Build & Test

- Rust 1.93.0+, Linux host only. Windows/macOS will NOT work.
- Build: `cargo build --release` (release profile: `opt-level=3`, `lto="thin"`, `codegen-units=1`)
- Binary at `target/release/bcc`
- Test: `cargo test --release` (tests invoke the built binary as a subprocess — build it first)
- Single test: `cargo test --release <test_name>` (e.g. `checkpoint1_hello_world`)
- CI order: `cargo fmt --all -- --check` → `cargo clippy -- -D warnings` → `cargo build --release` → `cargo test --release`
- `RUST_MIN_STACK=67108864` is required at runtime (set in `.cargo/config.toml`); compilations run on a 64 MiB worker thread

## Code Style

- `rustfmt`: edition 2021, max_width=100, tab_spaces=4, use_field_init_shorthand=true
- Clippy config in `clippy.toml`: type-complexity-threshold=500, too-many-arguments-threshold=10, cognitive-complexity-threshold=100
- `lib.rs` has crate-level `#![allow(clippy::too_many_arguments, clippy::type_complexity, clippy::large_enum_variant)]` — do not remove these

## Testing

- Integration tests in `tests/` invoke `bcc` binary as a subprocess via `tests/common/mod.rs` helpers (`bcc_path()`, `compile()`, `compile_and_run()`, `fixture_path()`)
- Test fixtures: `tests/fixtures/`
- Checkpoint tests: `checkpoint1_hello_world.rs` through `checkpoint7_stretch.rs`
- Regression tests: `regression_bugs.rs`, `regression_chibicc.rs`, `regression_fuzz.rs`, `regression_regehr.rs`, `regression_sqlite.rs`
- Cross-architecture binary execution requires QEMU user-mode (`qemu-aarch64`, `qemu-riscv64`, `qemu-i386`)
- Backend validation order (fixed): x86-64 → i686 → AArch64 → RISC-V 64
- Checkpoints 1–6 are sequential hard gates; failure at any gate blocks forward progress
- A binary compiled without `-g` must have ZERO `.debug_*` sections

## CLI Quirks

- GCC-compatible flag parser in `src/main.rs` — silently accepts many GCC flags (`-Wall`, `-Werror`, `-std=`, `-m*`, `-f*`, `-pipe`, `-nostdlib`, etc.)
- `--target=x86-64|i686|aarch64|riscv64` (note: `x86-64` with dash, not `x86_64`)
- Security mitigations (`-mretpoline`, `-fcf-protection`) are x86-64 only
- `-shared` implies `-fPIC` (warns if omitted)
- Static libc linking is **not yet supported** — libc-dependent programs fail at link stage; freestanding programs link successfully

## Important Constraints

- **Parser and macro expander recursion limit: 512** (hardcoded in `src/main.rs` as `MAX_RECURSION_DEPTH`)
- **DWARF debug info at `-O0` only** — debug info is not supported at other optimization levels
- **Built-in include headers** in `include/` directory (stdarg.h, stddef.h, stdalign.h, etc.)
- Architecture docs in `docs/` — especially `docs/architecture.md` and `docs/validation_checkpoints.md`
