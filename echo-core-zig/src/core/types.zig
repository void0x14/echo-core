const std = @import("std");

pub const fp16_t = u16;

pub const CACHE_LINE_SIZE: usize = 64;

pub const EC_ALIGN: usize = 64;

inline fn fp16ToFp32(h: fp16_t) f32 {
    const bits: u32 = @intCast(h);
    const sign = (bits >> 15) & 0x1;
    const exp = (bits >> 10) & 0x1f;
    const mantissa = bits & 0x3ff;
    if (exp == 0) {
        if (mantissa == 0) {
            return if (sign == 1) -0.0 else 0.0;
        }
        const frac = @as(f32, mantissa) / 1024.0;
        return if (sign == 1) -frac else frac;
    }
    if (exp == 31) {
        const inf_or_nan = if (mantissa == 0) std.math.inf(f32) else std.math.nan;
        return if (sign == 1) -inf_or_nan else inf_or_nan;
    }
    const significand = @as(f32, mantissa | 0x400) / 1024.0;
    const e = @as(i32, exp) - 15;
    const result = significand * std.math.pow(f32, 2.0, @floatFromInt(e));
    return if (sign == 1) -result else result;
}

inline fn fp32ToFp16(f: f32) fp16_t {
    if (f == 0) return 0;
    const bits: u32 = @bitCast(f);
    const sign: u1 = @truncate(bits >> 31);
    const raw_exp: u8 = @truncate((bits >> 23) & 0xff);
    const exp: i32 = @as(i32, raw_exp) - 127;
    const mantissa: u23 = @truncate(bits & 0x7fffff);
    if (exp < -24) return @truncate(sign << 15);
    if (exp < -14) {
        const shift: u6 = @intCast(-exp - 14);
        const frac = (0x4000 | (mantissa >> shift)) >> shift;
        return @truncate((sign << 15) | frac);
    }
    if (exp > 15) return @truncate((sign << 15) | 0x7c00);
    const ieee754_mantissa = (mantissa | 0x800000) >> 13;
    const biased_exp: u5 = @truncate(@as(u16, exp + 15));
    return @truncate((sign << 15) | (@as(u16, biased_exp) << 10) | ieee754_mantissa);
}

inline fn fp16ToFp32Row(src: [*]const fp16_t, dst: [*]f32, n: usize) void {
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        dst[i + 0] = fp16ToFp32(src[i + 0]);
        dst[i + 1] = fp16ToFp32(src[i + 1]);
        dst[i + 2] = fp16ToFp32(src[i + 2]);
        dst[i + 3] = fp16ToFp32(src[i + 3]);
        dst[i + 4] = fp16ToFp32(src[i + 4]);
        dst[i + 5] = fp16ToFp32(src[i + 5]);
        dst[i + 6] = fp16ToFp32(src[i + 6]);
        dst[i + 7] = fp16ToFp32(src[i + 7]);
    }
    while (i < n) : (i += 1) {
        dst[i] = fp16ToFp32(src[i]);
    }
}

inline fn fp32ToFp16Row(src: [*]const f32, dst: [*]fp16_t, n: usize) void {
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        dst[i + 0] = fp32ToFp16(src[i + 0]);
        dst[i + 1] = fp32ToFp16(src[i + 1]);
        dst[i + 2] = fp32ToFp16(src[i + 2]);
        dst[i + 3] = fp32ToFp16(src[i + 3]);
        dst[i + 4] = fp32ToFp16(src[i + 4]);
        dst[i + 5] = fp32ToFp16(src[i + 5]);
        dst[i + 6] = fp32ToFp16(src[i + 6]);
        dst[i + 7] = fp32ToFp16(src[i + 7]);
    }
    while (i < n) : (i += 1) {
        dst[i] = fp32ToFp16(src[i]);
    }
}

pub inline fn fp16_to_fp32(h: fp16_t) f32 {
    return fp16ToFp32(h);
}

pub inline fn fp32_to_fp16(f: f32) fp16_t {
    return fp32ToFp16(f);
}

pub inline fn fp16_to_fp32_row(src: [*]const fp16_t, dst: [*]f32, n: usize) void {
    return fp16ToFp32Row(src, dst, n);
}

pub inline fn fp32_to_fp16_row(src: [*]const f32, dst: [*]fp16_t, n: usize) void {
    return fp32ToFp16Row(src, dst, n);
}

test "fp16 conversion roundtrip" {
    const test_values = [_]f32{ 0.0, 1.0, -1.0, 3.14159, -100.0, 0.5 };
    for (test_values) |f| {
        const h = fp32_to_fp16(f);
        const back = fp16_to_fp32(h);
        const diff = @abs(f - back);
        const tolerance = @abs(f) * 0.01 + 0.0001;
        try std.testing.expect(diff < tolerance);
    }
}
