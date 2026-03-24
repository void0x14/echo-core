const std = @import("std");

pub const fp16_t = u16;

pub const CACHE_LINE_SIZE: usize = 64;

pub const EC_ALIGN: usize = 64;

inline fn fp16ToFp32(h: fp16_t) f32 {
    return @as(f32, @floatCast(@as(f16, @bitCast(h))));
}

inline fn fp32ToFp16(f: f32) fp16_t {
    return @as(fp16_t, @bitCast(@as(f16, @floatCast(f))));
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

test "fp16 subnormal matches IEEE half" {
    const smallest_subnormal: fp16_t = 0x0001;
    const expected = @as(f32, @floatCast(@as(f16, @bitCast(smallest_subnormal))));
    try std.testing.expectEqual(expected, fp16_to_fp32(smallest_subnormal));
}
