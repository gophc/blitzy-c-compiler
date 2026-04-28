package common

type LongDouble struct {
	sign bool
	exp uint16
	mant uint64
}

const (
	EXP_BIAS uint16 = 16383
	EXP_MAX  uint16 = 32767
	INT_BIT  uint64 = 1 << 63
	FRAC_MSK uint64 = (1 << 63) - 1
)

var (
	Zero = LongDouble{}
	One  = LongDouble{exp: EXP_BIAS, mant: INT_BIT}
	Inf  = LongDouble{exp: EXP_MAX, mant: INT_BIT}
	NaN  = LongDouble{exp: EXP_MAX, mant: INT_BIT | (1 << 62)}
)

func (ld LongDouble) IsZero() bool {
	return ld.exp == 0 && ld.mant == 0
}

func (ld LongDouble) IsInf() bool {
	return ld.exp == EXP_MAX && (ld.mant&INT_BIT) != 0
}

func (ld LongDouble) IsNaN() bool {
	if ld.exp != EXP_MAX {
		return false
	}
	return (ld.mant & FRAC_MSK) != 0
}

func (ld LongDouble) Float64() float64 {
	return 0.0
}

func Float64ToLongDouble(f float64) LongDouble {
	return LongDouble{}
}

func (ld LongDouble) String() string {
	return "LongDouble"
}