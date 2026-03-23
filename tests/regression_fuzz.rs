//! Regression tests for bugs found via Csmith/YARPGen fuzzing campaigns (Task 7).
//!
//! Each test corresponds to a specific bug class discovered during fuzzing.
//! Tests compile C fixtures with BCC, run the resulting binaries, and verify
//! correct output matches GCC behavior.

mod common;

use common::{assert_exit_success, compile_and_run};

// ===========================================================================
// Bug #1: Empty struct member (bare ';' inside struct body)
// ===========================================================================

#[test]
fn test_fuzz_empty_struct_member_x86_64() {
    let output = compile_and_run(
        "tests/fixtures/fuzz_regressions/empty_struct_member.c",
        "x86-64",
    );
    assert_exit_success(&output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("empty_struct_member OK"),
        "empty_struct_member test failed (x86-64). stdout: {}",
        stdout
    );
}

#[test]
fn test_fuzz_empty_struct_member_aarch64() {
    let output = compile_and_run(
        "tests/fixtures/fuzz_regressions/empty_struct_member.c",
        "aarch64",
    );
    assert_exit_success(&output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("empty_struct_member OK"),
        "empty_struct_member test failed (aarch64). stdout: {}",
        stdout
    );
}

// ===========================================================================
// Bug #2: typeof on array subscript expression
// ===========================================================================

#[test]
fn test_fuzz_typeof_array_subscript_x86_64() {
    let output = compile_and_run(
        "tests/fixtures/fuzz_regressions/typeof_array_subscript.c",
        "x86-64",
    );
    assert_exit_success(&output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("typeof_array_subscript OK"),
        "typeof_array_subscript test failed (x86-64). stdout: {}",
        stdout
    );
}

#[test]
fn test_fuzz_typeof_array_subscript_aarch64() {
    let output = compile_and_run(
        "tests/fixtures/fuzz_regressions/typeof_array_subscript.c",
        "aarch64",
    );
    assert_exit_success(&output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("typeof_array_subscript OK"),
        "typeof_array_subscript test failed (aarch64). stdout: {}",
        stdout
    );
}

// ===========================================================================
// Bug #3: typeof on statement expressions
// ===========================================================================

#[test]
fn test_fuzz_typeof_stmt_expr_x86_64() {
    let output = compile_and_run(
        "tests/fixtures/fuzz_regressions/typeof_stmt_expr.c",
        "x86-64",
    );
    assert_exit_success(&output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("typeof_stmt_expr OK"),
        "typeof_stmt_expr test failed (x86-64). stdout: {}",
        stdout
    );
}

#[test]
fn test_fuzz_typeof_stmt_expr_aarch64() {
    let output = compile_and_run(
        "tests/fixtures/fuzz_regressions/typeof_stmt_expr.c",
        "aarch64",
    );
    assert_exit_success(&output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("typeof_stmt_expr OK"),
        "typeof_stmt_expr test failed (aarch64). stdout: {}",
        stdout
    );
}
