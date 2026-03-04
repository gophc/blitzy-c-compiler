//! Software implementation of 80-bit extended precision (x87 format) arithmetic.
//!
//! This module provides the [`LongDouble`] type for `long double` support in the
//! C11 type system and compile-time constant folding of long double expressions.
//! It replaces external math libraries (`num`, `rug`) to comply with the
//! zero-dependency mandate.
//!
//! # x87 Extended Precision Format
//!
//! Unlike IEEE 754 binary64 (`f64`) which has an implicit leading 1 bit,
//! the x87 80-bit format stores the integer part **explicitly** in bit 63
//! of the 64-bit significand:
//!
//! - **1** sign bit
//! - **15** exponent bits (bias = 16383)
//! - **64** significand bits (bit 63 = explicit integer bit, bits 62:0 = fraction)
//!
//! Special value encoding:
//! - Zero: exponent = 0, significand = 0
//! - Denormal: exponent = 0, significand ≠ 0, integer bit = 0
//! - Normal: exponent ∈ [1, 32766], integer bit = 1
//! - Infinity: exponent = 32767, significand = `1 << 63` (integer bit set, fraction = 0)
//! - NaN: exponent = 32767, fraction ≠ 0

use std::cmp::Ordering;
use std::fmt;

/// Software implementation of 80-bit extended precision (x87 format).
///
/// Stored internally as decomposed fields: sign, biased exponent, and significand
/// with explicit integer bit. All arithmetic uses round-to-nearest-even semantics.
#[derive(Clone, Copy)]
pub struct LongDouble {
    /// Sign bit: `false` = positive, `true` = negative.
    sign: bool,
    /// Biased exponent (15 bits). Bias = 16383.
    /// Range: 0 (denormal/zero) to 32767 (infinity/NaN).
    exponent: u16,
    /// Significand with explicit integer bit at bit 63.
    /// For normalized numbers bit 63 is always 1.
    significand: u64,
}

// ============================================================================
// Constants
// ============================================================================

impl LongDouble {
    /// Exponent bias for 80-bit extended precision.
    const EXPONENT_BIAS: u16 = 16383;
    /// Maximum biased exponent value (reserved for infinity and NaN).
    const MAX_EXPONENT: u16 = 32767;
    /// Mask for the explicit integer bit (bit 63).
    const INTEGER_BIT: u64 = 1u64 << 63;
    /// Mask for the fraction portion (bits 62:0).
    const FRACTION_MASK: u64 = (1u64 << 63) - 1;

    /// Positive zero (`+0.0`).
    pub const ZERO: LongDouble = LongDouble {
        sign: false,
        exponent: 0,
        significand: 0,
    };

    /// Negative zero (`-0.0`).
    pub const NEG_ZERO: LongDouble = LongDouble {
        sign: true,
        exponent: 0,
        significand: 0,
    };

    /// Positive one (`1.0`).
    pub const ONE: LongDouble = LongDouble {
        sign: false,
        exponent: 16383,
        significand: 1u64 << 63,
    };

    /// Positive infinity (`+∞`).
    pub const INFINITY: LongDouble = LongDouble {
        sign: false,
        exponent: 32767,
        significand: 1u64 << 63,
    };

    /// Negative infinity (`-∞`).
    pub const NEG_INFINITY: LongDouble = LongDouble {
        sign: true,
        exponent: 32767,
        significand: 1u64 << 63,
    };

    /// Quiet NaN (Not a Number).
    pub const NAN: LongDouble = LongDouble {
        sign: false,
        exponent: 32767,
        significand: (1u64 << 63) | (1u64 << 62),
    };
}

// ============================================================================
// Classification and Predicates
// ============================================================================

impl LongDouble {
    /// Returns `true` if the value is positive or negative zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.exponent == 0 && self.significand == 0
    }

    /// Returns `true` if the value is positive or negative infinity.
    ///
    /// Infinity is encoded as exponent = 32767, integer bit set, fraction = 0.
    #[inline]
    pub fn is_infinity(&self) -> bool {
        self.exponent == Self::MAX_EXPONENT
            && (self.significand & Self::INTEGER_BIT) != 0
            && (self.significand & Self::FRACTION_MASK) == 0
    }

    /// Returns `true` if the value is NaN (Not a Number).
    ///
    /// NaN is encoded as exponent = 32767 with a non-zero fraction.
    #[inline]
    pub fn is_nan(&self) -> bool {
        if self.exponent != Self::MAX_EXPONENT {
            return false;
        }
        // For x87: NaN has integer bit set and non-zero fraction,
        // or integer bit clear with non-zero significand.
        let fraction = self.significand & Self::FRACTION_MASK;
        if (self.significand & Self::INTEGER_BIT) != 0 {
            fraction != 0
        } else {
            // Pseudo-NaN: integer bit clear, any non-zero significand
            self.significand != 0
        }
    }

    /// Returns `true` if the sign bit is set (value is negative).
    ///
    /// Note: NaN and zero can also have the sign bit set.
    #[inline]
    pub fn is_negative(&self) -> bool {
        self.sign
    }

    /// Returns `true` if the value is a denormalized number.
    ///
    /// Denormals have exponent = 0 and a non-zero significand.
    #[inline]
    pub fn is_denormal(&self) -> bool {
        self.exponent == 0 && self.significand != 0
    }

    /// Returns `true` if this is a normal finite number (not zero, denormal,
    /// infinity, or NaN).
    #[inline]
    #[allow(dead_code)]
    fn is_normal(&self) -> bool {
        self.exponent > 0 && self.exponent < Self::MAX_EXPONENT
    }
}

// ============================================================================
// Internal Helpers
// ============================================================================

impl LongDouble {
    /// Returns the unbiased (true) exponent.
    ///
    /// For denormals the true exponent is `1 - BIAS = -16382` (same as biased
    /// exponent 1), matching x87 semantics.
    #[inline]
    fn true_exponent(&self) -> i32 {
        if self.exponent == 0 {
            1 - Self::EXPONENT_BIAS as i32
        } else {
            self.exponent as i32 - Self::EXPONENT_BIAS as i32
        }
    }

    /// Compare absolute magnitude of two `LongDouble` values.
    ///
    /// Returns `true` if `|self| > |other|`.  Used by addition to determine
    /// which operand has the larger magnitude.
    #[inline]
    fn abs_greater_than(&self, other: &LongDouble) -> bool {
        if self.exponent != other.exponent {
            self.exponent > other.exponent
        } else {
            self.significand > other.significand
        }
    }

    /// Normalize a 128-bit significand and round to 64 bits.
    ///
    /// `exp` is the unbiased exponent corresponding to bit 127 of `sig` being
    /// the integer-bit position.  The function shifts `sig` so that its leading
    /// 1 lands at bit 127, extracts the top 64 bits as the result significand,
    /// and uses the bottom 64 bits for round-to-nearest-even.
    fn normalize_round(sign: bool, mut exp: i32, mut sig: u128) -> LongDouble {
        if sig == 0 {
            return if sign { Self::NEG_ZERO } else { Self::ZERO };
        }

        // Shift the leading 1 to bit 127.
        let lz = sig.leading_zeros();
        if lz > 0 {
            sig <<= lz;
            exp -= lz as i32;
        }

        let sig_hi = (sig >> 64) as u64;
        let sig_lo = sig as u64;
        Self::round_and_pack(sign, exp, sig_hi, sig_lo)
    }

    /// Pack a result with rounding.
    ///
    /// `exp` is the unbiased exponent when `sig_hi` has its integer bit at
    /// bit 63.  `sig_lo` carries guard/round/sticky bits for
    /// round-to-nearest-even.
    fn round_and_pack(sign: bool, mut exp: i32, mut sig_hi: u64, sig_lo: u64) -> LongDouble {
        let half: u64 = 1u64 << 63;
        let round_up = if sig_lo > half {
            true
        } else if sig_lo == half {
            // Tie — round to even (round up if LSB of sig_hi is 1).
            (sig_hi & 1) != 0
        } else {
            false
        };

        if round_up {
            sig_hi = sig_hi.wrapping_add(1);
            if sig_hi == 0 {
                // Significand overflowed from 0xFFFF…FFFF to 0.
                sig_hi = Self::INTEGER_BIT;
                exp += 1;
            }
        }

        let biased = exp + Self::EXPONENT_BIAS as i32;

        // Overflow → ±infinity.
        if biased >= Self::MAX_EXPONENT as i32 {
            return if sign { Self::NEG_INFINITY } else { Self::INFINITY };
        }

        // Underflow → denormal or zero.
        if biased <= 0 {
            let shift = (1 - biased) as u32;
            if shift >= 64 {
                return if sign { Self::NEG_ZERO } else { Self::ZERO };
            }
            // Create denormal by shifting significand right.
            sig_hi >>= shift;
            return LongDouble {
                sign,
                exponent: 0,
                significand: sig_hi,
            };
        }

        LongDouble {
            sign,
            exponent: biased as u16,
            significand: sig_hi,
        }
    }
}

// ============================================================================
// Arithmetic Operations
// ============================================================================

impl LongDouble {
    /// Add two `LongDouble` values: `self + other`.
    ///
    /// Handles all IEEE special cases (NaN propagation, infinity arithmetic,
    /// signed zeros) and uses round-to-nearest-even.
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: LongDouble) -> LongDouble {
        // NaN propagation.
        if self.is_nan() {
            return Self::NAN;
        }
        if other.is_nan() {
            return Self::NAN;
        }

        // Infinity cases.
        if self.is_infinity() {
            if other.is_infinity() {
                return if self.sign == other.sign {
                    self
                } else {
                    Self::NAN
                };
            }
            return self;
        }
        if other.is_infinity() {
            return other;
        }

        // Zero cases.
        if self.is_zero() {
            if other.is_zero() {
                // +0 + −0 = +0 in round-to-nearest-even.
                return if self.sign && other.sign {
                    Self::NEG_ZERO
                } else {
                    Self::ZERO
                };
            }
            return other;
        }
        if other.is_zero() {
            return self;
        }

        // Order operands so that `a` has the larger (or equal) magnitude.
        let (a, b) = if self.abs_greater_than(&other) {
            (self, other)
        } else {
            (other, self)
        };

        let a_exp = a.true_exponent();
        let b_exp = b.true_exponent();
        let exp_diff = (a_exp - b_exp) as u32;

        // Widen significands to 128 bits, placing the integer bit at bit 127
        // so that the low 64 bits capture rounding information after alignment.
        let a_wide: u128 = (a.significand as u128) << 64;
        let mut b_wide: u128 = (b.significand as u128) << 64;

        // Align `b` to `a` by shifting right.
        if exp_diff > 0 {
            if exp_diff < 128 {
                let sticky = (b_wide & ((1u128 << exp_diff) - 1)) != 0;
                b_wide >>= exp_diff;
                if sticky {
                    b_wide |= 1;
                }
            } else {
                b_wide = u128::from(b_wide != 0); // preserve sticky
            }
        }

        let result_exp = a_exp; // base exponent for the result

        if a.sign == b.sign {
            // Effective addition — significands may carry.
            let (sum, carry) = a_wide.overflowing_add(b_wide);
            if carry {
                // The 129-bit result has its integer bit at position 128.
                // Shift right by 1 and bump the exponent.
                let sticky = (sum & 1) != 0;
                let shifted = (1u128 << 127) | (sum >> 1);
                let shifted = if sticky { shifted | 1 } else { shifted };
                return Self::normalize_round(a.sign, result_exp + 1, shifted);
            }
            Self::normalize_round(a.sign, result_exp, sum)
        } else {
            // Effective subtraction — `a` ≥ `b` in magnitude so no underflow.
            let diff = a_wide - b_wide;
            if diff == 0 {
                return Self::ZERO; // +0 in round-to-nearest-even
            }
            Self::normalize_round(a.sign, result_exp, diff)
        }
    }

    /// Subtract: `self − other`.
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn sub(self, other: LongDouble) -> LongDouble {
        self.add(other.neg())
    }

    /// Multiply two `LongDouble` values: `self × other`.
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, other: LongDouble) -> LongDouble {
        let result_sign = self.sign != other.sign;

        if self.is_nan() {
            return Self::NAN;
        }
        if other.is_nan() {
            return Self::NAN;
        }

        if self.is_infinity() {
            if other.is_zero() {
                return Self::NAN;
            }
            return if result_sign {
                Self::NEG_INFINITY
            } else {
                Self::INFINITY
            };
        }
        if other.is_infinity() {
            if self.is_zero() {
                return Self::NAN;
            }
            return if result_sign {
                Self::NEG_INFINITY
            } else {
                Self::INFINITY
            };
        }

        if self.is_zero() || other.is_zero() {
            return if result_sign {
                Self::NEG_ZERO
            } else {
                Self::ZERO
            };
        }

        // Result unbiased exponent.
        let exp = self.true_exponent() + other.true_exponent();

        // 64 × 64 → 128 bit product.
        let product = (self.significand as u128) * (other.significand as u128);

        // The product of two significands (each with integer bit at position 63)
        // yields a 128-bit value whose "integer bit squared" position is 126.
        // `normalize_round` interprets bit 127 as the integer-bit position at
        // exponent `exp_param`, so we pass `exp + 1` to account for the
        // one-bit offset:  product / 2^126 × 2^exp = product / 2^127 × 2^(exp+1).
        Self::normalize_round(result_sign, exp + 1, product)
    }

    /// Divide: `self / other`.
    #[allow(clippy::should_implement_trait)]
    pub fn div(self, other: LongDouble) -> LongDouble {
        let result_sign = self.sign != other.sign;

        if self.is_nan() {
            return Self::NAN;
        }
        if other.is_nan() {
            return Self::NAN;
        }

        // inf / inf → NaN; inf / finite → inf.
        if self.is_infinity() {
            if other.is_infinity() {
                return Self::NAN;
            }
            return if result_sign {
                Self::NEG_INFINITY
            } else {
                Self::INFINITY
            };
        }

        // finite / 0 → inf (or NaN for 0/0).
        if other.is_zero() {
            if self.is_zero() {
                return Self::NAN;
            }
            return if result_sign {
                Self::NEG_INFINITY
            } else {
                Self::INFINITY
            };
        }

        // 0 / finite → ±0.
        if self.is_zero() {
            return if result_sign {
                Self::NEG_ZERO
            } else {
                Self::ZERO
            };
        }

        // Both operands are finite and non-zero.
        let exp = self.true_exponent() - other.true_exponent();

        // Divide significands using 128-bit arithmetic.
        // dividend = self.sig << 64  (gives 128 bits of precision in quotient)
        let dividend = (self.significand as u128) << 64;
        let divisor = other.significand as u128;

        let quotient = dividend / divisor;
        let remainder = dividend % divisor;

        // The mathematical value is:
        //   (self.sig / other.sig) × 2^exp
        //   = (quotient / 2^64) × 2^exp
        //   = quotient × 2^(exp − 64)
        //
        // `normalize_round` interprets sig / 2^127 × 2^exp_param, so:
        //   quotient × 2^(exp − 64) = quotient / 2^127 × 2^(exp + 63)
        let mut sig = quotient;
        if remainder != 0 {
            sig |= 1; // sticky bit for rounding
        }

        Self::normalize_round(result_sign, exp + 63, sig)
    }

    /// Negate: flip the sign bit.
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn neg(self) -> LongDouble {
        LongDouble {
            sign: !self.sign,
            exponent: self.exponent,
            significand: self.significand,
        }
    }
}

// ============================================================================
// Comparison
// ============================================================================

impl PartialEq for LongDouble {
    /// IEEE 754 equality: NaN ≠ everything (including itself), −0 == +0.
    fn eq(&self, other: &Self) -> bool {
        if self.is_nan() || other.is_nan() {
            return false;
        }
        // −0 == +0
        if self.is_zero() && other.is_zero() {
            return true;
        }
        self.sign == other.sign
            && self.exponent == other.exponent
            && self.significand == other.significand
    }
}

impl PartialOrd for LongDouble {
    /// IEEE 754 ordering: NaN is unordered, −0 == +0.
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.is_nan() || other.is_nan() {
            return None;
        }
        // Both zeros are equal regardless of sign.
        if self.is_zero() && other.is_zero() {
            return Some(Ordering::Equal);
        }

        // Handle sign differences.
        match (self.sign, other.sign) {
            (false, true) => {
                // self positive, other negative → self > other
                // (both-zero case already handled above)
                return Some(Ordering::Greater);
            }
            (true, false) => {
                return Some(Ordering::Less);
            }
            _ => {}
        }

        // Same sign — compare magnitudes.
        let mag_cmp = if self.exponent != other.exponent {
            self.exponent.cmp(&other.exponent)
        } else {
            self.significand.cmp(&other.significand)
        };

        // For negative numbers the larger magnitude is the *smaller* value.
        Some(if self.sign { mag_cmp.reverse() } else { mag_cmp })
    }
}

impl LongDouble {
    /// Total ordering that is reflexive, antisymmetric, and transitive even
    /// for NaN values.
    ///
    /// Order: −NaN < −∞ < −normals < −denormals < −0 < +0 < +denormals <
    ///        +normals < +∞ < +NaN.
    pub fn total_cmp(&self, other: &LongDouble) -> Ordering {
        // Convert to a sortable integer representation.
        //
        // Order: −NaN < −∞ < −finite < −0 < +0 < +finite < +∞ < +NaN
        //
        // For positive values the magnitude ((exponent << 64) | significand)
        // already sorts correctly.  For negative values we invert: larger
        // magnitudes map to *smaller* sort keys.  We use `−mag − 1` so that
        // −0 (mag = 0 → key = −1) sorts strictly below +0 (mag = 0 → key = 0).
        fn sort_key(v: &LongDouble) -> i128 {
            let mag = ((v.exponent as u128) << 64) | (v.significand as u128);
            if v.sign {
                -(mag as i128) - 1
            } else {
                mag as i128
            }
        }
        sort_key(self).cmp(&sort_key(other))
    }
}

// ============================================================================
// Conversion — f64
// ============================================================================

impl LongDouble {
    /// Create a `LongDouble` from an `f64` value.
    ///
    /// The conversion is exact for all finite `f64` values because the 80-bit
    /// format has strictly more precision and exponent range than binary64.
    pub fn from_f64(val: f64) -> LongDouble {
        let bits = val.to_bits();
        let sign = (bits >> 63) != 0;
        let exp_bits = ((bits >> 52) & 0x7FF) as u16;
        let fraction = bits & ((1u64 << 52) - 1);

        // ±0
        if exp_bits == 0 && fraction == 0 {
            return LongDouble {
                sign,
                exponent: 0,
                significand: 0,
            };
        }

        // Denormal f64 → normalized long double.
        if exp_bits == 0 {
            // value = ±fraction × 2^(−1074)
            let lz = fraction.leading_zeros(); // ≥ 12 since fraction fits in 52 bits
            let significand = fraction << lz; // leading 1 now at bit 63
            // Derive biased exponent:
            //   value = significand × 2^(−1074 − lz)
            //   LD form: 2^(biased − 16383) × significand × 2^(−63)
            //   ⇒ biased = 16446 − 1074 − lz = 15372 − lz
            let biased = 15372i32 - lz as i32;
            if biased < 1 {
                // Extremely small — still denormal in long double format.
                // Shouldn't happen for valid f64 denormals but handle gracefully.
                let shift = (1 - biased) as u32;
                let sig = if shift < 64 {
                    significand >> shift
                } else {
                    0
                };
                return LongDouble {
                    sign,
                    exponent: 0,
                    significand: sig,
                };
            }
            return LongDouble {
                sign,
                exponent: biased as u16,
                significand,
            };
        }

        // ±infinity
        if exp_bits == 0x7FF && fraction == 0 {
            return if sign {
                Self::NEG_INFINITY
            } else {
                Self::INFINITY
            };
        }

        // NaN — preserve quiet bit in the long-double significand.
        if exp_bits == 0x7FF {
            let significand = Self::INTEGER_BIT | (fraction << 11);
            return LongDouble {
                sign,
                exponent: Self::MAX_EXPONENT,
                significand,
            };
        }

        // Normal f64.
        // f64 biased exponent → LD biased exponent:
        //   unbiased = exp_bits − 1023
        //   LD biased = unbiased + 16383 = exp_bits + 15360
        let exponent = exp_bits + 15360;
        // Significand: set explicit integer bit and extend 52 → 63 fraction bits.
        let significand = Self::INTEGER_BIT | (fraction << 11);

        LongDouble {
            sign,
            exponent,
            significand,
        }
    }

    /// Convert to `f64`, rounding to nearest even.
    ///
    /// Precision loss is expected (64-bit significand → 52-bit fraction).
    /// Overflow saturates to `±infinity`; underflow flushes toward `±0`.
    pub fn to_f64(&self) -> f64 {
        if self.is_nan() {
            return f64::NAN;
        }
        if self.is_infinity() {
            return if self.sign {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
        }
        if self.is_zero() {
            return if self.sign { -0.0_f64 } else { 0.0_f64 };
        }

        let true_exp = self.true_exponent();
        let f64_biased = true_exp + 1023;

        // Overflow → ±infinity.
        if f64_biased >= 2047 {
            return if self.sign {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
        }

        // Underflow → f64 denormal or zero.
        if f64_biased <= 0 {
            // Number of positions to shift to form a denormal.
            let shift = (1 - f64_biased) as u32;
            if shift > 63 {
                return if self.sign { -0.0_f64 } else { 0.0_f64 };
            }
            // Shift significand right to create a denormal fraction.
            //   LD value = sig × 2^(true_exp − 63)
            //   f64 denormal value = frac × 2^(−1074)
            //   frac = sig × 2^(true_exp − 63 + 1074) = sig × 2^(true_exp + 1011)
            //   Since true_exp + 1011 < 0 for underflow: frac = sig >> (−true_exp − 1011)
            let sa = (0i32 - true_exp - 1011) as u32;
            if sa >= 64 {
                return if self.sign { -0.0_f64 } else { 0.0_f64 };
            }
            let frac = (self.significand >> sa) & ((1u64 << 52) - 1);
            let bits = ((self.sign as u64) << 63) | frac;
            return f64::from_bits(bits);
        }

        // Normal f64.  Extract 52 fraction bits from 63 fraction bits.
        let fraction_full = self.significand & Self::FRACTION_MASK; // 63 fraction bits
        let frac52 = fraction_full >> 11;
        let round_bits = fraction_full & 0x7FF; // bottom 11 bits
        let mut fraction = frac52;

        // Round to nearest even.
        let tie: u64 = 1u64 << 10; // 0x400
        if round_bits > tie || (round_bits == tie && (fraction & 1) != 0) {
            fraction += 1;
            if fraction >= (1u64 << 52) {
                // Carry into exponent field.
                fraction = 0;
                let new_biased = f64_biased + 1;
                if new_biased >= 2047 {
                    return if self.sign {
                        f64::NEG_INFINITY
                    } else {
                        f64::INFINITY
                    };
                }
                let bits =
                    ((self.sign as u64) << 63) | ((new_biased as u64) << 52) | fraction;
                return f64::from_bits(bits);
            }
        }

        let bits =
            ((self.sign as u64) << 63) | ((f64_biased as u64) << 52) | fraction;
        f64::from_bits(bits)
    }
}

// ============================================================================
// Conversion — Integers
// ============================================================================

impl LongDouble {
    /// Create a `LongDouble` from an `i64`.
    pub fn from_i64(val: i64) -> LongDouble {
        if val == 0 {
            return Self::ZERO;
        }
        let sign = val < 0;
        // Handle i64::MIN carefully (its absolute value doesn't fit in i64).
        let abs_val: u64 = if val == i64::MIN {
            (i64::MAX as u64) + 1
        } else if val < 0 {
            (-val) as u64
        } else {
            val as u64
        };
        let mut result = Self::from_u64(abs_val);
        result.sign = sign;
        result
    }

    /// Create a `LongDouble` from a `u64`.
    pub fn from_u64(val: u64) -> LongDouble {
        if val == 0 {
            return Self::ZERO;
        }
        let lz = val.leading_zeros();
        let significand = val << lz; // leading 1 at bit 63
        // Unbiased exponent: the leading 1 was at bit position (63 − lz).
        // That means the value is significand × 2^(−lz) × ... let's derive:
        //   val = significand >> lz   (undo the shift we applied)
        //   LD value = 2^(biased − 16383) × significand / 2^63
        //            = significand × 2^(biased − 16446)
        //   We need significand × 2^(biased − 16446) = val = significand >> lz
        //                                              = significand × 2^(−lz)
        //   ⇒ biased − 16446 = −lz  ⇒  biased = 16446 − lz
        let biased = 16446u32 - lz;
        LongDouble {
            sign: false,
            exponent: biased as u16,
            significand,
        }
    }

    /// Convert to `i64`, truncating toward zero.
    ///
    /// Values outside the representable range saturate to `i64::MIN` / `i64::MAX`.
    /// NaN returns 0.
    pub fn to_i64(&self) -> i64 {
        if self.is_nan() || self.is_zero() {
            return 0;
        }
        if self.is_infinity() {
            return if self.sign { i64::MIN } else { i64::MAX };
        }

        let true_exp = self.true_exponent();
        if true_exp < 0 {
            return 0; // |value| < 1
        }

        // bit_pos = true_exp is the position of the integer bit (0-indexed).
        // We need the integer part: significand >> (63 − true_exp).
        let shift = 63 - true_exp;
        if shift < 0 {
            // Value is larger than 2^63 — overflow.
            if self.sign {
                // Check for exactly −2^63 = i64::MIN.
                if true_exp == 63
                    && self.significand == Self::INTEGER_BIT
                {
                    return i64::MIN;
                }
                return i64::MIN;
            }
            return i64::MAX;
        }

        let magnitude = self.significand >> (shift as u32);

        if self.sign {
            let max_neg = (i64::MAX as u64) + 1; // 2^63
            if magnitude > max_neg {
                i64::MIN // saturate
            } else {
                // Safe for all magnitude ∈ [0, 2^63]:
                // when magnitude == 2^63, magnitude as i64 is i64::MIN,
                // and 0i64.wrapping_sub(i64::MIN) == i64::MIN (correct).
                0i64.wrapping_sub(magnitude as i64)
            }
        } else if magnitude > i64::MAX as u64 {
            i64::MAX
        } else {
            magnitude as i64
        }
    }

    /// Convert to `u64`, truncating toward zero.
    ///
    /// Negative values and NaN return 0. Overflow saturates to `u64::MAX`.
    pub fn to_u64(&self) -> u64 {
        if self.is_nan() || self.is_zero() {
            return 0;
        }
        if self.sign {
            return 0; // negative → 0 for unsigned
        }
        if self.is_infinity() {
            return u64::MAX;
        }

        let true_exp = self.true_exponent();
        if true_exp < 0 {
            return 0; // |value| < 1
        }

        let shift = 63 - true_exp;
        if shift < 0 {
            // Value ≥ 2^64 — overflow to u64::MAX.
            return u64::MAX;
        }

        self.significand >> (shift as u32)
    }
}

// ============================================================================
// Raw Byte Serialization (x87 10-byte format, little-endian)
// ============================================================================

impl LongDouble {
    /// Serialize to the 10-byte little-endian x87 memory format.
    ///
    /// Layout:
    /// - Bytes 0–7: significand (little-endian `u64`)
    /// - Bytes 8–9: `sign_bit | exponent` (little-endian `u16`, sign is MSB)
    ///
    /// Used for emitting `long double` constants into `.rodata`.
    pub fn to_bytes(&self) -> [u8; 10] {
        let mut out = [0u8; 10];
        let sig_bytes = self.significand.to_le_bytes();
        out[..8].copy_from_slice(&sig_bytes);
        let exp_sign: u16 = self.exponent | ((self.sign as u16) << 15);
        let es_bytes = exp_sign.to_le_bytes();
        out[8] = es_bytes[0];
        out[9] = es_bytes[1];
        out
    }

    /// Deserialize from the 10-byte little-endian x87 memory format.
    pub fn from_bytes(bytes: &[u8; 10]) -> LongDouble {
        let mut sig_buf = [0u8; 8];
        sig_buf.copy_from_slice(&bytes[..8]);
        let significand = u64::from_le_bytes(sig_buf);
        let mut es_buf = [0u8; 2];
        es_buf.copy_from_slice(&bytes[8..10]);
        let exp_sign = u16::from_le_bytes(es_buf);
        let sign = (exp_sign >> 15) != 0;
        let exponent = exp_sign & 0x7FFF;
        LongDouble {
            sign,
            exponent,
            significand,
        }
    }
}

// ============================================================================
// Display / Debug
// ============================================================================

impl fmt::Display for LongDouble {
    /// Displays the value by converting to `f64` first (acceptable precision
    /// loss for diagnostic / display purposes).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_nan() {
            return write!(f, "NaN");
        }
        if self.is_infinity() {
            return if self.sign {
                write!(f, "-inf")
            } else {
                write!(f, "inf")
            };
        }
        // Delegate to f64 formatting for finite values.
        let val = self.to_f64();
        fmt::Display::fmt(&val, f)
    }
}

impl fmt::Debug for LongDouble {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LongDouble {{ sign: {}, exp: {}, sig: {:#018x} }}",
            self.sign, self.exponent, self.significand
        )
    }
}

// ============================================================================
// Standard trait impls required by the rest of the compiler
// ============================================================================

impl Default for LongDouble {
    #[inline]
    fn default() -> Self {
        Self::ZERO
    }
}

// ============================================================================
// std::ops trait implementations (delegates to named methods)
// ============================================================================

impl std::ops::Add for LongDouble {
    type Output = LongDouble;
    #[inline]
    fn add(self, rhs: LongDouble) -> LongDouble {
        LongDouble::add(self, rhs)
    }
}

impl std::ops::Sub for LongDouble {
    type Output = LongDouble;
    #[inline]
    fn sub(self, rhs: LongDouble) -> LongDouble {
        LongDouble::sub(self, rhs)
    }
}

impl std::ops::Mul for LongDouble {
    type Output = LongDouble;
    #[inline]
    fn mul(self, rhs: LongDouble) -> LongDouble {
        LongDouble::mul(self, rhs)
    }
}

impl std::ops::Div for LongDouble {
    type Output = LongDouble;
    #[inline]
    fn div(self, rhs: LongDouble) -> LongDouble {
        LongDouble::div(self, rhs)
    }
}

impl std::ops::Neg for LongDouble {
    type Output = LongDouble;
    #[inline]
    fn neg(self) -> LongDouble {
        LongDouble::neg(self)
    }
}
