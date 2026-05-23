const std = @import("std");
const types = @import("../core/types.zig");
const quant = @import("quant.zig");

pub fn quantizeXToQ8K(x: [*]const f32, q8: [*]quant.block_q8_K, K: usize) void {
    var b: usize = 0;
    while (b < K / 256) : (b += 1) {
        const bx = x + b * 256;
        var amax: f32 = 0;
        var ii: usize = 0;
        while (ii < 256) : (ii += 8) {
            const a = amax_v8(@as(*align(1) const [8]f32, @ptrCast(bx + ii)).*);
            if (a > amax) amax = a;
        }
        if (amax < 1e-30) { q8[b].d = 0; @memset(&q8[b].qs, 0); @memset(&q8[b].bsums, 0); continue; }
        q8[b].d = amax / 127.0;
        const isc_v = @as(@Vector(8, f32), @splat(127.0 / amax));
        ii = 0;
        while (ii < 256) : (ii += 8) {
            const xv: @Vector(8, f32) = @as(*align(1) const [8]f32, @ptrCast(bx + ii)).*;
            const sc: [8]f32 = xv * isc_v;
            for (0..8) |j_| {
                const v: i32 = @intFromFloat(@round(sc[j_]));
                q8[b].qs[ii + j_] = @intCast(@max(-128, @min(127, v)));
            }
        }
        var j: usize = 0;
        while (j < 16) : (j += 4) {
            const vi0: @Vector(16, i8) = @as(*align(1) const [16]i8, @ptrCast(&q8[b].qs[j * 16])).*;
            const vi1: @Vector(16, i8) = @as(*align(1) const [16]i8, @ptrCast(&q8[b].qs[(j + 1) * 16])).*;
            const vi2: @Vector(16, i8) = @as(*align(1) const [16]i8, @ptrCast(&q8[b].qs[(j + 2) * 16])).*;
            const vi3: @Vector(16, i8) = @as(*align(1) const [16]i8, @ptrCast(&q8[b].qs[(j + 3) * 16])).*;
            q8[b].bsums[j] = @intCast(@reduce(.Add, @as(@Vector(16, i32), @intCast(vi0))));
            q8[b].bsums[j + 1] = @intCast(@reduce(.Add, @as(@Vector(16, i32), @intCast(vi1))));
            q8[b].bsums[j + 2] = @intCast(@reduce(.Add, @as(@Vector(16, i32), @intCast(vi2))));
            q8[b].bsums[j + 3] = @intCast(@reduce(.Add, @as(@Vector(16, i32), @intCast(vi3))));
        }
    }
}

inline fn amax_v8(a: [8]f32) f32 {
    const x: @Vector(8, f32) = a;
    return @reduce(.Max, @select(f32, x < @as(@Vector(8, f32), @splat(0)), -x, x));
}

fn dotQ4K(q4: *align(1) const quant.block_q4_K, q8: *const quant.block_q8_K) f32 {
    const d0 = types.fp16_to_fp32(q4.d);
    const dmin0 = types.fp16_to_fp32(q4.dmin);
    const df = q8.d * d0;
    const dmf = -q8.d * dmin0;

    var sumi: i32 = 0;
    var mcorr: f32 = 0;

    var j: u32 = 0;
    while (j < 4) : (j += 1) {
        const si = j * 2;
        const mn0: u8 = if (si < 4) q4.scales[si + 4] & 63 else (q4.scales[si + 4] >> 4) | ((q4.scales[si] >> 6) << 4);
        const mn1: u8 = if (si + 1 < 4) q4.scales[si + 1 + 4] & 63 else (q4.scales[si + 1 + 4] >> 4) | ((q4.scales[si + 1] >> 6) << 4);

        const q4raw: @Vector(32, u8) = @as(*align(1) const [32]u8, @ptrCast(&q4.qs[j * 32])).*;
        const q4l: @Vector(32, i32) = @intCast(q4raw & @as(@Vector(32, u8), @splat(0x0F)));
        const q4h: @Vector(32, i32) = @intCast((q4raw >> @as(@Vector(32, u8), @splat(4))) & @as(@Vector(32, u8), @splat(0x0F)));
        const q8l_v: @Vector(32, i8) = @as(*align(1) const [32]i8, @ptrCast(&q8.qs[j * 64])).*;
        const q8h_v: @Vector(32, i8) = @as(*align(1) const [32]i8, @ptrCast(&q8.qs[j * 64 + 32])).*;
        const q8l: @Vector(32, i32) = @intCast(q8l_v);
        const q8h: @Vector(32, i32) = @intCast(q8h_v);

        sumi += @reduce(.Add, q4l * q8l);
        sumi += @reduce(.Add, q4h * q8h);

        mcorr += dmf * (@as(f32, @floatFromInt(mn0)) * @as(f32, @floatFromInt(@reduce(.Add, q8l))) +
                          @as(f32, @floatFromInt(mn1)) * @as(f32, @floatFromInt(@reduce(.Add, q8h))));
    }

    return df * @as(f32, @floatFromInt(sumi)) + mcorr;
}

pub fn matvecQ4K_int(blocks: [*]const u8, x: [*]const f32, y: [*]f32, M: u32, K: u32) void {
    const bpr = K / 256;
    var q8b: [64]quant.block_q8_K = undefined;
    quantizeXToQ8K(x, @ptrCast(&q8b), @intCast(K));
    var m: u32 = 0;
    while (m < M) : (m += 1) {
        var s: f32 = 0;
        const rp = blocks + @as(usize, m) * @as(usize, bpr) * 144;
        for (0..bpr) |b| s += dotQ4K(@ptrCast(rp + b * 144), &q8b[b]);
        y[m] += s;
    }
}

pub fn matvecQ4K_intPre(blocks: [*]const u8, q8_pre: [*]const quant.block_q8_K, y: [*]f32, M: u32, K: u32) void {
    const bpr = K / 256;
    var m: u32 = 0;
    while (m < M) : (m += 1) {
        var s: f32 = 0;
        const rp = blocks + @as(usize, m) * @as(usize, bpr) * 144;
        for (0..bpr) |b| s += dotQ4K(@ptrCast(rp + b * 144), &q8_pre[b]);
        y[m] += s;
    }
}

test "Q4 int dot" {
    var q4 = std.mem.zeroes(quant.block_q4_K);
    q4.d = types.fp32_to_fp16(2); q4.dmin = types.fp32_to_fp16(1);
    q4.scales[0] = 1; q4.scales[4] = 2; q4.scales[8] = 0x11; q4.scales[9] = 0x22;
    var prng = std.Random.DefaultPrng.init(123);
    for (0..q4.qs.len) |i| q4.qs[i] = prng.random().int(u8);
    var x: [256]f32 = undefined;
    for (0..256) |i| x[i] = @as(f32, @floatFromInt(@as(i32, @intCast(prng.random().int(u8) & 15)) - 8)) * 0.1;
    var yo: [1]f32 = .{0};
    matvecQ4K_int(@as([*]const u8, @ptrCast(&q4)), &x, &yo, 1, 256);
    var exp: f32 = 0; const df_ = types.fp16_to_fp32(q4.d); const dmf_ = types.fp16_to_fp32(q4.dmin);
    for (0..4) |blk| { const js = blk * 2;
        const sc0: u8 = if (js < 4) q4.scales[js] & 63 else (q4.scales[js + 4] & 0x0F) | ((q4.scales[js - 4] >> 6) << 4);
        const mn0: u8 = if (js < 4) q4.scales[js + 4] & 63 else (q4.scales[js + 4] >> 4) | ((q4.scales[js] >> 6) << 4);
        const sc1: u8 = if (js + 1 < 4) q4.scales[js + 1] & 63 else (q4.scales[js + 1 + 4] & 0x0F) | ((q4.scales[js + 1 - 4] >> 6) << 4);
        const mn1: u8 = if (js + 1 < 4) q4.scales[js + 1 + 4] & 63 else (q4.scales[js + 1 + 4] >> 4) | ((q4.scales[js + 1] >> 6) << 4);
        const rs0 = df_ * @as(f32, @floatFromInt(sc0)); const rm0 = dmf_ * @as(f32, @floatFromInt(mn0));
        const rs1 = df_ * @as(f32, @floatFromInt(sc1)); const rm1 = dmf_ * @as(f32, @floatFromInt(mn1));
        const qo = blk * 32; const wo = blk * 64;
        for (0..16) |k| {
            exp += (rs0 * @as(f32, @floatFromInt(q4.qs[qo + k] & 0x0F)) - rm0) * x[wo + k];
            exp += (rs1 * @as(f32, @floatFromInt(q4.qs[qo + 16 + k] & 0x0F)) - rm1) * x[wo + 16 + k];
            exp += (rs0 * @as(f32, @floatFromInt(q4.qs[qo + k] >> 4)) - rm0) * x[wo + 32 + k];
            exp += (rs1 * @as(f32, @floatFromInt(q4.qs[qo + 16 + k] >> 4)) - rm1) * x[wo + 48 + k];
        }
    }
    try std.testing.expectApproxEqAbs(exp, yo[0], 50);
}
