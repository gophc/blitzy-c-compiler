package common

import (
	"math"
	"math/big"
	"strconv"
)

type LongDouble struct {
	sign     bool
	exponent uint16
	mantissa uint64
}

const (
	EXP_BIAS uint16 = 16383
	EXP_MAX  uint16 = 32767
	INT_BIT  uint64 = 1 << 63
	FRAC_MSK uint64 = (1 << 63) - 1
)

var (
	Zero    = LongDouble{}
	NegZero = LongDouble{sign: true}
	One     = LongDouble{exponent: EXP_BIAS, mantissa: INT_BIT}
	Inf     = LongDouble{exponent: EXP_MAX, mantissa: INT_BIT}
	NegInf  = LongDouble{sign: true, exponent: EXP_MAX, mantissa: INT_BIT}
	NaN     = LongDouble{exponent: EXP_MAX, mantissa: INT_BIT | (1 << 62)}
)

func (ld LongDouble) IsZero() bool {
	return ld.exponent == 0 && ld.mantissa == 0
}

func (ld LongDouble) TrueExponent() int {
	return int(ld.exponent) - int(EXP_BIAS)
}

func (ld LongDouble) IsInf() bool {
	return ld.exponent == EXP_MAX && (ld.mantissa&INT_BIT) != 0
}

func (ld LongDouble) IsNaN() bool {
	if ld.exponent != EXP_MAX {
		return false
	}
	if (ld.mantissa & INT_BIT) != 0 {
		return (ld.mantissa & FRAC_MSK) != 0
	}
	return ld.mantissa != 0
}

func (ld LongDouble) IsNegative() bool {
	return ld.sign
}

func (ld LongDouble) IsDenormal() bool {
	return ld.exponent == 0 && ld.mantissa != 0
}

func (ld LongDouble) Neg() LongDouble {
	if ld.IsZero() {
		return NegZero
	}
	return LongDouble{
		sign:     !ld.sign,
		exponent: ld.exponent,
		mantissa: ld.mantissa,
	}
}

func (ld LongDouble) Float64() float64 {
	if ld.IsNaN() {
		return math.NaN()
	}
	if ld.IsInf() {
		if ld.sign {
			return math.Inf(-1)
		}
		return math.Inf(1)
	}
	if ld.IsZero() {
		if ld.sign {
			return math.Copysign(0, -1)
		}
		return 0.0
	}

	trueExp := ld.TrueExponent()
	f64Biased := trueExp + 1023

	if f64Biased >= 2047 {
		if ld.sign {
			return math.Inf(-1)
		}
		return math.Inf(1)
	}

	if f64Biased <= 0 {
		shift := 1 - f64Biased
		if shift >= 64 {
			if ld.sign {
				return math.Copysign(0, -1)
			}
			return 0.0
		}
		sa := -trueExp - 1011
		if sa >= 64 {
			if ld.sign {
				return math.Copysign(0, -1)
			}
			return 0.0
		}
		frac := (ld.mantissa >> uint(sa)) & ((1 << 52) - 1)
		bits := uint64(0)
		if ld.sign {
			bits |= 1 << 63
		}
		bits |= uint64(f64Biased&0x7FF) << 52
		bits |= frac
		return math.Float64frombits(bits)
	}

	fracFull := ld.mantissa & FRAC_MSK
	frac52 := fracFull >> 11
	roundBits := fracFull & 0x7FF
	mutFrac := frac52

	tie := uint64(1) << 10
	if roundBits > tie || (roundBits == tie && (mutFrac&1) != 0) {
		mutFrac++
		if mutFrac >= (1 << 52) {
			newBiased := f64Biased + 1
			if newBiased >= 2047 {
				if ld.sign {
					return math.Inf(-1)
				}
				return math.Inf(1)
			}
			bits := uint64(0)
			if ld.sign {
				bits |= 1 << 63
			}
			bits |= uint64(newBiased&0x7FF) << 52
			return math.Float64frombits(bits)
		}
	}

	bits := uint64(0)
	if ld.sign {
		bits |= 1 << 63
	}
	bits |= uint64(f64Biased&0x7FF) << 52
	bits |= mutFrac
	return math.Float64frombits(bits)
}

func Float64ToLongDouble(f float64) LongDouble {
	bits := math.Float64bits(f)
	sign := (bits >> 63) != 0
	expBits := uint16((bits >> 52) & 0x7FF)
	frac := bits & ((1 << 52) - 1)

	if expBits == 0 && frac == 0 {
		if sign {
			return NegZero
		}
		return Zero
	}

	if expBits == 0 {
		lz := leadingZeros64(frac)
		significand := frac << uint(lz)
		biased := int(15372) - int(lz)
		if biased < 1 {
			shift := uint(1 - biased)
			if shift < 64 {
				significand >>= shift
			} else {
				significand = 0
			}
			return LongDouble{
				sign:     sign,
				exponent: 0,
				mantissa: significand,
			}
		}
		return LongDouble{
			sign:     sign,
			exponent: uint16(biased),
			mantissa: significand,
		}
	}

	if expBits == 0x7FF && frac == 0 {
		if sign {
			return NegInf
		}
		return Inf
	}

	if expBits == 0x7FF {
		significand := INT_BIT | (frac << 11)
		return LongDouble{
			sign:     sign,
			exponent: EXP_MAX,
			mantissa: significand,
		}
	}

	exponent := expBits + 15360
	significand := INT_BIT | (frac << 11)

	return LongDouble{
		sign:     sign,
		exponent: exponent,
		mantissa: significand,
	}
}

func (ld LongDouble) Int64() int64 {
	if ld.IsNaN() || ld.IsZero() {
		return 0
	}
	if ld.IsInf() {
		if ld.sign {
			return math.MinInt64
		}
		return math.MaxInt64
	}

	trueExp := ld.TrueExponent()
	if trueExp < 0 {
		return 0
	}

	shift := 63 - trueExp
	if shift < 0 {
		if ld.sign {
			if trueExp == 63 && ld.mantissa == INT_BIT {
				return math.MinInt64
			}
			return math.MinInt64
		}
		return math.MaxInt64
	}

	magnitude := ld.mantissa >> uint(shift)

	if ld.sign {
		maxNeg := uint64(1<<63) + 1
		if magnitude > maxNeg {
			return math.MinInt64
		}
		if magnitude == 1<<63 {
			return math.MinInt64
		}
		return -int64(magnitude)
	}
	if magnitude > math.MaxInt64 {
		return math.MaxInt64
	}
	return int64(magnitude)
}

func Int64ToLongDouble(val int64) LongDouble {
	if val == 0 {
		return Zero
	}
	sign := val < 0
	var absVal uint64
	if val == math.MinInt64 {
		absVal = uint64(math.MaxInt64) + 1
	} else if val < 0 {
		absVal = uint64(-val)
	} else {
		absVal = uint64(val)
	}
	result := UInt64ToLongDouble(absVal)
	result.sign = sign
	return result
}

func (ld LongDouble) UInt64() uint64 {
	if ld.IsNaN() || ld.IsZero() || ld.sign {
		return 0
	}
	if ld.IsInf() {
		return math.MaxUint64
	}

	trueExp := ld.TrueExponent()
	if trueExp < 0 {
		return 0
	}

	shift := 63 - trueExp
	if shift < 0 {
		return math.MaxUint64
	}

	return ld.mantissa >> uint(shift)
}

func UInt64ToLongDouble(val uint64) LongDouble {
	if val == 0 {
		return Zero
	}
	lz := leadingZeros64(val)
	significand := val << uint(lz)
	biased := uint32(16446) - uint32(lz)
	return LongDouble{
		sign:     false,
		exponent: uint16(biased),
		mantissa: significand,
	}
}

func (ld LongDouble) Bytes() [10]byte {
	var out [10]byte
	out[0] = byte(ld.mantissa)
	out[1] = byte(ld.mantissa >> 8)
	out[2] = byte(ld.mantissa >> 16)
	out[3] = byte(ld.mantissa >> 24)
	out[4] = byte(ld.mantissa >> 32)
	out[5] = byte(ld.mantissa >> 40)
	out[6] = byte(ld.mantissa >> 48)
	out[7] = byte(ld.mantissa >> 56)
	expSign := ld.exponent
	if ld.sign {
		expSign |= 0x8000
	}
	out[8] = byte(expSign)
	out[9] = byte(expSign >> 8)
	return out
}

func FromBytes(bytes []byte) LongDouble {
	if len(bytes) < 10 {
		return Zero
	}
	mantissa := uint64(bytes[0]) |
		uint64(bytes[1])<<8 |
		uint64(bytes[2])<<16 |
		uint64(bytes[3])<<24 |
		uint64(bytes[4])<<32 |
		uint64(bytes[5])<<40 |
		uint64(bytes[6])<<48 |
		uint64(bytes[7])<<56
	expSign := uint16(bytes[8]) | uint16(bytes[9])<<8
	sign := (expSign & 0x8000) != 0
	exponent := expSign & 0x7FFF
	return LongDouble{
		sign:     sign,
		exponent: exponent,
		mantissa: mantissa,
	}
}

func leadingZeros64(x uint64) int {
	if x == 0 {
		return 64
	}
	n := 0
	if x>>32 == 0 {
		n += 32
		x <<= 32
	}
	if x>>48 == 0 {
		n += 16
		x <<= 16
	}
	if x>>56 == 0 {
		n += 8
		x <<= 8
	}
	if x>>60 == 0 {
		n += 4
		x <<= 4
	}
	if x>>62 == 0 {
		n += 2
		x <<= 2
	}
	if x>>63 == 0 {
		n++
	}
	return n
}

func (ld LongDouble) String() string {
	if ld.IsNaN() {
		return "NaN"
	}
	if ld.IsInf() {
		if ld.sign {
			return "-inf"
		}
		return "inf"
	}
	return strconv.FormatFloat(ld.Float64(), 'f', -1, 64)
}

func (ld LongDouble) TotalCmp(other LongDouble) int {
	less := func(v LongDouble) *big.Int {
		mag := new(big.Int).SetUint64(uint64(v.exponent))
		mag.Lsh(mag, 64)
		mag.Or(mag, new(big.Int).SetUint64(v.mantissa))
		if v.sign {
			return mag.Neg(mag).Sub(mag, big.NewInt(1))
		}
		return mag
	}
	thisKey := less(ld)
	otherKey := less(other)
	return thisKey.Cmp(otherKey)
}
