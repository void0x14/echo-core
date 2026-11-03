const std = @import("std");
const types = @import("../core/types.zig");

const V8f = @Vector(8, f32);
const V8u16 = @Vector(8, u16);

inline fn loadF32x8(ptr: [*]const f32) V8f {
    var buf: [8]f32 = undefined;
    @memcpy(@as([*]u8, @ptrCast(&buf))[0..32], @as([*]const u8, @ptrCast(ptr))[0..32]);
    return @bitCast(buf);
}

inline fn storeF32x8(ptr: [*]f32, val: V8f) void {
    var buf: [8]f32 = undefined;
    buf = @bitCast(val);
    @memcpy(@as([*]u8, @ptrCast(ptr))[0..32], @as([*]const u8, @ptrCast(&buf))[0..32]);
}

inline fn loadF16x8asF32(ptr: [*]const types.fp16_t) V8f {
    var buf16: [8]types.fp16_t = undefined;
    @memcpy(@as([*]u8, @ptrCast(&buf16))[0..16], @as([*]const u8, @ptrCast(ptr))[0..16]);
    const v16: V8u16 = @bitCast(buf16);
    const vf16: @Vector(8, f16) = @bitCast(v16);
    return @floatCast(vf16);
}

pub fn rmsNormAvx2(input: [*]const f32, output: [*]f32, weight: [*]const types.fp16_t, n: usize) void {
    var i: usize = 0;
    var sum_sq: f32 = 0;

    while (i + 8 <= n) : (i += 8) {
        const x = loadF32x8(input + i);
        sum_sq += @reduce(.Add, x * x);
    }
    while (i < n) : (i += 1) {
        sum_sq += input[i] * input[i];
    }

    const inv_rms = 1.0 / @sqrt(sum_sq / @as(f32, @floatFromInt(n)) + 1e-6);
    const inv_rms_v: V8f = @splat(inv_rms);

    i = 0;
    while (i + 8 <= n) : (i += 8) {
        const x = loadF32x8(input + i);
        const w = loadF16x8asF32(weight + i);
        storeF32x8(output + i, x * inv_rms_v * w);
    }
    while (i < n) : (i += 1) {
        output[i] = input[i] * inv_rms * types.fp16_to_fp32(weight[i]);
    }
}
