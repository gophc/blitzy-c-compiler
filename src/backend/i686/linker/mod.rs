//! # BCC i686 Built-in ELF Linker
//!
//! Produces ELFCLASS32 binaries (ET_EXEC static executables and ET_DYN shared objects)
//! for the i686 (32-bit x86) target without invoking any external linker.

pub mod relocations;
