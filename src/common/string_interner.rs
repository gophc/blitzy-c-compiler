//! String interning for identifiers, keywords, and string literals.
//!
//! Provides a [`Symbol`] handle type that wraps a `u32` index, enabling
//! **O(1) equality comparison** of identifiers by comparing integer handles
//! instead of performing full string comparisons. The [`Interner`] struct
//! manages the mapping between strings and their [`Symbol`] handles using
//! an [`FxHashMap`] for deduplication (performance-critical for symbol tables
//! throughout the BCC compiler pipeline).
//!
//! # Architecture
//!
//! The interner maintains two parallel data structures:
//!
//! - `map: FxHashMap<String, Symbol>` — for O(1) average-case lookup during
//!   interning (string → symbol direction).
//! - `strings: Vec<String>` — for O(1) symbol resolution (symbol → string
//!   direction), indexed by `Symbol`'s inner `u32` value.
//!
//! # Thread Safety
//!
//! Thread-safety is **not** required. The interner is used within a single
//! compilation worker thread. Each compilation context owns its own
//! `Interner` instance.
//!
//! # Usage
//!
//! ```ignore
//! use bcc::common::string_interner::{Interner, Symbol};
//!
//! let mut interner = Interner::new();
//! let sym_main = interner.intern("main");
//! let sym_main2 = interner.intern("main"); // returns same Symbol
//! assert_eq!(sym_main, sym_main2);          // O(1) integer comparison
//! assert_eq!(interner.resolve(sym_main), "main");
//! assert_eq!(interner[sym_main], *"main");  // Index syntax
//! ```
//!
//! # Zero-Dependency Compliance
//!
//! This module uses only the Rust standard library (`std`) and the internal
//! [`FxHashMap`] from `crate::common::fx_hash`. No external crates.

use crate::common::fx_hash::FxHashMap;
use std::fmt;
use std::ops::Index;

// ---------------------------------------------------------------------------
// Symbol — zero-cost identifier handle
// ---------------------------------------------------------------------------

/// An interned string handle — a lightweight, copyable `u32` index into an
/// [`Interner`]'s string table.
///
/// `Symbol` values are only meaningful in the context of the [`Interner`]
/// that created them. Comparing symbols from different interners is
/// undefined (but not unsafe — it simply yields incorrect results).
///
/// # Performance
///
/// - **Equality comparison:** O(1) — compares two `u32` values.
/// - **Hashing:** O(1) — hashes a single `u32`.
/// - **Copy:** Free — `Symbol` is `Copy` (4 bytes on stack).
///
/// # Ordering
///
/// `PartialOrd` and `Ord` are derived to allow symbols in sorted
/// collections (e.g., `BTreeMap<Symbol, _>`). The ordering is by
/// insertion order (index), NOT lexicographic.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Symbol(u32);

impl Symbol {
    /// Returns the raw `u32` index of this symbol.
    ///
    /// This index corresponds to the position in the [`Interner`]'s internal
    /// string vector. Useful for serialization, diagnostics, or when a
    /// numeric handle is needed for external data structures.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut interner = Interner::new();
    /// let sym = interner.intern("foo");
    /// assert_eq!(sym.as_u32(), 0); // first interned string gets index 0
    /// ```
    #[inline]
    pub fn as_u32(&self) -> u32 {
        self.0
    }
}

// ---------------------------------------------------------------------------
// Display for Symbol
// ---------------------------------------------------------------------------

/// Displays the symbol as `Symbol(N)` where `N` is the raw index.
///
/// Since a `Symbol` cannot resolve its string without a reference to its
/// owning [`Interner`], this implementation shows the numeric index for
/// diagnostic and debugging purposes. To display the actual string, use
/// [`Interner::resolve`] or the [`Index`] implementation on `Interner`.
impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Symbol({})", self.0)
    }
}

// ---------------------------------------------------------------------------
// Interner — string deduplication and symbol management
// ---------------------------------------------------------------------------

/// A string interner that maps strings to compact [`Symbol`] handles.
///
/// The `Interner` guarantees that:
/// - Each unique string is stored exactly once.
/// - Interning the same string always returns the same [`Symbol`].
/// - Symbol-to-string resolution is O(1) via direct index lookup.
/// - String-to-symbol lookup is O(1) average case via [`FxHashMap`].
///
/// # Capacity
///
/// The interner supports up to `u32::MAX` (4,294,967,295) unique strings.
/// Attempting to intern beyond this limit will panic with a descriptive
/// message. In practice, even the largest C codebases (e.g., the Linux
/// kernel) contain far fewer unique identifiers.
///
/// # Examples
///
/// ```ignore
/// use bcc::common::string_interner::Interner;
///
/// let mut interner = Interner::new();
///
/// // Intern some identifiers
/// let sym_int = interner.intern("int");
/// let sym_main = interner.intern("main");
/// let sym_int2 = interner.intern("int"); // deduplicates
///
/// assert_eq!(sym_int, sym_int2);
/// assert_ne!(sym_int, sym_main);
/// assert_eq!(interner.resolve(sym_int), "int");
/// assert_eq!(interner.len(), 2);
/// ```
pub struct Interner {
    /// String-to-symbol lookup map. Uses `FxHashMap` (Fibonacci hashing) for
    /// performance — this is on the hot path during lexing and parsing where
    /// every identifier token triggers a lookup.
    map: FxHashMap<String, Symbol>,

    /// Symbol-to-string storage. Indexed by `Symbol`'s inner `u32` value.
    /// Strings are stored as owned `String` values for simplicity and to
    /// avoid lifetime complexity. The `Vec` provides O(1) index access.
    strings: Vec<String>,
}

impl Interner {
    /// Creates a new, empty `Interner`.
    ///
    /// Both the lookup map and the string storage vector start empty and
    /// will grow as strings are interned.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let interner = Interner::new();
    /// assert_eq!(interner.len(), 0);
    /// ```
    #[inline]
    pub fn new() -> Self {
        Interner {
            map: FxHashMap::default(),
            strings: Vec::new(),
        }
    }

    /// Interns a string, returning its [`Symbol`] handle.
    ///
    /// If the string has been previously interned, the existing `Symbol` is
    /// returned without allocating. If the string is new, it is stored and
    /// a fresh `Symbol` is created.
    ///
    /// # Arguments
    ///
    /// * `s` — The string slice to intern. An owned copy is made only if the
    ///   string is not already present.
    ///
    /// # Returns
    ///
    /// The [`Symbol`] handle for the interned string — guaranteed to be the
    /// same value for the same input string within this `Interner` instance.
    ///
    /// # Panics
    ///
    /// Panics if the number of unique interned strings exceeds `u32::MAX`
    /// (4,294,967,295). This is astronomically unlikely in practice.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut interner = Interner::new();
    /// let s1 = interner.intern("hello");
    /// let s2 = interner.intern("hello");
    /// assert_eq!(s1, s2); // same Symbol for same string
    /// ```
    pub fn intern(&mut self, s: &str) -> Symbol {
        // Fast path: string already interned — O(1) average-case lookup.
        if let Some(&sym) = self.map.get(s) {
            return sym;
        }

        // Slow path: new string — allocate and register.
        let idx = self.strings.len();
        assert!(
            idx <= u32::MAX as usize,
            "string interner overflow: cannot intern more than {} unique strings",
            u32::MAX
        );
        let sym = Symbol(idx as u32);
        let owned = s.to_owned();
        self.strings.push(owned.clone());
        self.map.insert(owned, sym);
        sym
    }

    /// Resolves a [`Symbol`] back to its original string.
    ///
    /// # Arguments
    ///
    /// * `sym` — A symbol previously returned by [`intern`](Interner::intern)
    ///   on this same `Interner` instance.
    ///
    /// # Returns
    ///
    /// A reference to the interned string. The lifetime is tied to the
    /// `Interner`, so the reference is valid as long as the interner exists.
    ///
    /// # Panics
    ///
    /// Panics (via index bounds check) if `sym` was not produced by this
    /// `Interner` or if the index is out of range. In debug builds, this
    /// produces a clear index-out-of-bounds message.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut interner = Interner::new();
    /// let sym = interner.intern("world");
    /// assert_eq!(interner.resolve(sym), "world");
    /// ```
    #[inline]
    pub fn resolve(&self, sym: Symbol) -> &str {
        debug_assert!(
            (sym.0 as usize) < self.strings.len(),
            "Symbol({}) is out of range for interner with {} entries",
            sym.0,
            self.strings.len()
        );
        &self.strings[sym.0 as usize]
    }

    /// Looks up a string without interning it.
    ///
    /// Returns `Some(symbol)` if the string has been previously interned,
    /// or `None` if it has not. This is useful for checking whether an
    /// identifier exists in the symbol table without side effects.
    ///
    /// # Arguments
    ///
    /// * `s` — The string to look up.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut interner = Interner::new();
    /// assert_eq!(interner.get("foo"), None);
    /// interner.intern("foo");
    /// assert!(interner.get("foo").is_some());
    /// ```
    #[inline]
    pub fn get(&self, s: &str) -> Option<Symbol> {
        self.map.get(s).copied()
    }

    /// Returns the number of unique strings currently interned.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut interner = Interner::new();
    /// assert_eq!(interner.len(), 0);
    /// interner.intern("a");
    /// interner.intern("b");
    /// interner.intern("a"); // duplicate — not counted again
    /// assert_eq!(interner.len(), 2);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.strings.len()
    }

    /// Returns `true` if no strings have been interned.
    ///
    /// This is a convenience method equivalent to `self.len() == 0`.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.strings.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Default trait for Interner
// ---------------------------------------------------------------------------

impl Default for Interner {
    /// Creates a new, empty `Interner`. Equivalent to [`Interner::new()`].
    #[inline]
    fn default() -> Self {
        Interner::new()
    }
}

// ---------------------------------------------------------------------------
// Index<Symbol> for Interner — convenient bracket-access syntax
// ---------------------------------------------------------------------------

/// Allows `interner[symbol]` syntax for resolving a [`Symbol`] to its
/// string. Returns `&str`.
///
/// # Panics
///
/// Panics if the symbol index is out of range (same behavior as
/// [`Interner::resolve`]).
///
/// # Examples
///
/// ```ignore
/// let mut interner = Interner::new();
/// let sym = interner.intern("hello");
/// assert_eq!(&interner[sym], "hello");
/// ```
impl Index<Symbol> for Interner {
    type Output = str;

    #[inline]
    fn index(&self, sym: Symbol) -> &str {
        self.resolve(sym)
    }
}

// ---------------------------------------------------------------------------
// Debug for Interner
// ---------------------------------------------------------------------------

impl fmt::Debug for Interner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Interner")
            .field("count", &self.strings.len())
            .field("strings", &self.strings)
            .finish()
    }
}
