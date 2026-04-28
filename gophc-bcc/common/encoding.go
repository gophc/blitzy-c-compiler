package common

const (
	PUABase uint32 = 0xE000
	PUALow  uint32 = 0xE080
	PUAHigh uint32 = 0xE0FF
)

func EncodeByteToPUA(b byte) rune {
	if b < 0x80 {
		panic("Only bytes 0x80-0xFF need PUA encoding, got 0x" + stringByteHex(b))
	}
	return rune(PUABase + uint32(b))
}

func DecodePUAToByte(ch rune) *uint8 {
	cp := uint32(ch)
	if cp >= PUALow && cp <= PUAHigh {
		bb := uint8(cp - PUABase)
		return &bb
	}
	return nil
}

func IsPUAEncoded(ch rune) bool {
	cp := uint32(ch)
	return cp >= PUALow && cp <= PUAHigh
}

func EncodeBytesToString(bytes []byte) string {
	result := make([]rune, 0, len(bytes))

	for i := 0; i < len(bytes); i++ {
		b := bytes[i]
		if b < 0x80 || isValidUTF8(bytes, i) {
			result = append(result, rune(b))
		} else {
			result = append(result, EncodeByteToPUA(b))
		}
	}

	return string(result)
}

func DecodeStringToBytes(s string) []byte {
	result := make([]byte, 0, len(s))

	runes := []rune(s)
	for _, r := range runes {
		decoded := DecodePUAToByte(r)
		if decoded != nil {
			result = append(result, *decoded)
		} else {
			utf8Bytes := []byte(string(r))
			result = append(result, utf8Bytes...)
		}
	}

	return result
}

func ExtractStringBytes(s string) []byte {
	return DecodeStringToBytes(s)
}

func isValidUTF8(bytes []byte, pos int) bool {
	if pos >= len(bytes) {
		return false
	}

	b := bytes[pos]
	if b < 0x80 {
		return true
	}

	count := 0
	needs := 0

	if b&0x20 == 0 {
		count = 1
		needs = 0xC0
	} else if b&0x10 == 0 {
		count = 2
		needs = 0xE0
	} else if b&0x08 == 0 {
		count = 3
		needs = 0xF0
	} else {
		return false
	}

	for i := 1; i <= count; i++ {
		if pos+i >= len(bytes) {
			return false
		}
		if bytes[pos+i]&0xC0 != 0x80 {
			return false
		}
	}

	_ = needs
	return true
}

func stringByteHex(b byte) string {
	hexDigits := "0123456789ABCDEF"
	result := make([]byte, 2)
	result[0] = byte(hexDigits[(b>>4)&0x0F])
	result[1] = byte(hexDigits[b&0x0F])
	return string(result)
}
