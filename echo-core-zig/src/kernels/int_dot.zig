const std = @import("std");
const types = @import("../core/types.zig");
const quant = @import("quant.zig");

fn quantQ8(x: [*]const f32, q8: [*]quant.block_q8_K, K: usize) void {
    var b: usize = 0; while (b < K / 256) : (b += 1) {
        const bx = x + b * 256;
        var amax: f32 = 0;
        for (0..256) |i| { const ax = @abs(bx[i]); if (ax > amax) amax = ax; }
        if (amax == 0) { q8[b].d = 0; @memset(&q8[b].qs, 0); @memset(&q8[b].bsums, 0); continue; }
        q8[b].d = amax / 127.0;
        const is = 127.0 / amax;
        for (0..256) |i| { const v: i32 = @intFromFloat(@round(bx[i] * is)); q8[b].qs[i] = @intCast(@max(-128, @min(127, v))); }
        for (0..16) |j| { var s: i32 = 0; for (0..16) |ii| s += q8[b].qs[j * 16 + ii]; q8[b].bsums[j] = @intCast(s); }
    }
}

fn dotQ4(q4: *align(1) const quant.block_q4_K, q8: *const quant.block_q8_K) f32 {
    const d = types.fp16_to_fp32(q4.d);
    const dm = types.fp16_to_fp32(q4.dmin);
    const qd = q8.d;
    var r: f32 = 0;
    var j: usize = 0;
    while (j < 4) : (j += 1) {
        const si = j * 2;
        const sc0: u8 = if (si < 4) q4.scales[si] & 63 else (q4.scales[si + 4] & 0x0F) | ((q4.scales[si - 4] >> 6) << 4);
        const mn0: u8 = if (si < 4) q4.scales[si + 4] & 63 else (q4.scales[si + 4] >> 4) | ((q4.scales[si] >> 6) << 4);
        const sc1: u8 = if (si + 1 < 4) q4.scales[si + 1] & 63 else (q4.scales[si + 1 + 4] & 0x0F) | ((q4.scales[si + 1 - 4] >> 6) << 4);
        const mn1: u8 = if (si + 1 < 4) q4.scales[si + 1 + 4] & 63 else (q4.scales[si + 1 + 4] >> 4) | ((q4.scales[si + 1] >> 6) << 4);
        const qo = j * 32;
        const wo = j * 64;
        const M4: @Vector(32, u8) = @splat(0x0F);
        const S4: @Vector(32, u8) = @splat(4);
        const qb = @as(*const @Vector(32, u8), @ptrCast(@alignCast(&q4.qs[qo]))).*;
        const ql: @Vector(32, i32) = @intCast(qb & M4);
        const qh: @Vector(32, i32) = @intCast((qb >> S4) & M4);
        const q8l: @Vector(32, i32) = @intCast(@as(*const @Vector(32, i8), @ptrCast(@alignCast(&q8.qs[wo]))).*);
        const q8h: @Vector(32, i32) = @intCast(@as(*const @Vector(32, i8), @ptrCast(@alignCast(&q8.qs[wo + 32]))).*);
        const dl: i32 = @reduce(.Add, ql * q8l);
        const dh: i32 = @reduce(.Add, qh * q8h);
        const sl: i32 = @reduce(.Add, q8l);
        const sh: i32 = @reduce(.Add, q8h);
        r += qd * (d * @as(f32, @floatFromInt(sc0)) * @as(f32, @floatFromInt(dl)) - dm * @as(f32, @floatFromInt(mn0)) * @as(f32, @floatFromInt(sl)));
        r += qd * (d * @as(f32, @floatFromInt(sc1)) * @as(f32, @floatFromInt(dh)) - dm * @as(f32, @floatFromInt(mn1)) * @as(f32, @floatFromInt(sh)));
    }
    return r;
}

pub fn matvecQ4K_int(blocks: [*]const u8, x: [*]const f32, y: [*]f32, M: u32, K: u32) void {
    const bpr = K / 256;
    var q8b: [64]quant.block_q8_K = undefined;
    quantQ8(x, @ptrCast(&q8b), K);
    var m: u32 = 0;
    while (m < M) : (m += 1) {
        var s: f32 = 0;
        const rp = blocks + @as(usize, m) * @as(usize, bpr) * 144;
        for (0..bpr) |b| { const q4: *align(1) const quant.block_q4_K = @ptrCast(rp + b * 144); s += dotQ4(q4, &q8b[b]); }
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
    var exp: f32 = 0; const df = types.fp16_to_fp32(q4.d); const dmf = types.fp16_to_fp32(q4.dmin);
    for (0..4) |blk| { const js = blk * 2;
        const sc0: u8 = if (js < 4) q4.scales[js] & 63 else (q4.scales[js + 4] & 0x0F) | ((q4.scales[js - 4] >> 6) << 4);
        const mn0: u8 = if (js < 4) q4.scales[js + 4] & 63 else (q4.scales[js + 4] >> 4) | ((q4.scales[js] >> 6) << 4);
        const sc1: u8 = if (js + 1 < 4) q4.scales[js + 1] & 63 else (q4.scales[js + 1 + 4] & 0x0F) | ((q4.scales[js + 1 - 4] >> 6) << 4);
        const mn1: u8 = if (js + 1 < 4) q4.scales[js + 1 + 4] & 63 else (q4.scales[js + 1 + 4] >> 4) | ((q4.scales[js + 1] >> 6) << 4);
        const rs0 = df * @as(f32, @floatFromInt(sc0)); const rm0 = dmf * @as(f32, @floatFromInt(mn0));
        const rs1 = df * @as(f32, @floatFromInt(sc1)); const rm1 = dmf * @as(f32, @floatFromInt(mn1));
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
