const std = @import("std");

const V8f = @Vector(8, f32);

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

pub fn softmaxAvx2(values: []f32) void {
    const n = values.len;
    var i: usize = 0;

    var max_val: f32 = -std.math.inf(f32);
    while (i + 8 <= n) : (i += 8) {
        const v = loadF32x8(values.ptr + i);
        const cmax = @reduce(.Max, v);
        if (cmax > max_val) max_val = cmax;
    }
    while (i < n) : (i += 1) {
        if (values[i] > max_val) max_val = values[i];
    }

    var sum: f32 = 0;
    i = 0;
    while (i + 8 <= n) : (i += 8) {
        const v = loadF32x8(values.ptr + i);
        const shifted = v - @as(V8f, @splat(max_val));
        const ex = @exp(shifted);
        sum += @reduce(.Add, ex);
        storeF32x8(values.ptr + i, ex);
    }
    while (i < n) : (i += 1) {
        values[i] = @exp(values[i] - max_val);
        sum += values[i];
    }

    const inv_sum = 1.0 / sum;
    i = 0;
    while (i + 8 <= n) : (i += 8) {
        const v = loadF32x8(values.ptr + i);
        storeF32x8(values.ptr + i, v * @as(V8f, @splat(inv_sum)));
    }
    while (i < n) : (i += 1) {
        values[i] *= inv_sum;
    }
}

test "softmax basic" {
    const original: [8]f32 = .{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var buf: [8]f32 = original;
    softmaxAvx2(&buf);

    var expected: [8]f32 = undefined;
    var max_val: f32 = -std.math.inf(f32);
    for (&original) |v| { if (v > max_val) max_val = v; }
    var sum: f32 = 0;
    for (&expected, 0..) |_, i| {
        expected[i] = @exp(original[i] - max_val);
        sum += expected[i];
    }
    for (&expected, 0..) |_, i| expected[i] /= sum;

    for (&expected, 0..) |e, i| {
        try std.testing.expectApproxEqAbs(e, buf[i], 0.001);
    }
}

test "softmax single element" {
    var buf: [1]f32 = .{42.0};
    softmaxAvx2(&buf);
    try std.testing.expectApproxEqAbs(1.0, buf[0], 0.001);
}

test "softmax all equal" {
    var buf: [16]f32 = @splat(3.0);
    softmaxAvx2(&buf);
    const expected = 1.0 / 16.0;
    for (&buf) |v| try std.testing.expectApproxEqAbs(expected, v, 0.001);
}
