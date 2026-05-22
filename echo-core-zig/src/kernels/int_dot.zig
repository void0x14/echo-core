const std = @import("std");
const types = @import("../core/types.zig");
const quant = @import("quant.zig");

const V16u = @Vector(16, u8);
const V16i = @Vector(16, i8);
const V16w = @Vector(16, i16);

inline fn loadU8x16(ptr: *const u8) V16u {
    var buf: [16]u8 = undefined;
    @memcpy(buf[0..], @as([*]const u8, @ptrCast(ptr))[0..16]);
    return @bitCast(buf);
}

inline fn loadI8x16(ptr: *const i8) V16i {
    var buf: [16]i8 = undefined;
    @memcpy(@as([*]u8, @ptrCast(&buf))[0..16], @as([*]const u8, @ptrCast(ptr))[0..16]);
    return @bitCast(buf);
}

fn quantizeActivationToQ8K(x_ptr: [*]const f32, q8: [*]quant.block_q8_K, K: usize) void {
    var b: usize = 0;
    while (b < K / 256) : (b += 1) {
        const x = x_ptr + b * 256;
        var amax: f32 = 0;
        for (0..256) |i| {
            const ax = @abs(x[i]);
            if (ax > amax) amax = ax;
        }
        if (amax == 0) {
            q8[b].d = 0;
            @memset(&q8[b].qs, 0);
            @memset(&q8[b].bsums, 0);
            continue;
        }
        const d = amax / 127.0;
        const iscale = 127.0 / amax;
        q8[b].d = d;
        for (0..256) |i| {
            const v = @as(i32, @intFromFloat(@round(x[i] * iscale)));
            q8[b].qs[i] = @intCast(@max(-128, @min(127, v)));
        }
        for (0..16) |j| {
            var sum: i32 = 0;
            for (0..16) |ii| sum += q8[b].qs[j * 16 + ii];
            q8[b].bsums[j] = @intCast(sum);
        }
    }
}

fn dotQ4KxQ8K(q4: *align(1) const quant.block_q4_K, q8: *const quant.block_q8_K) f32 {
    const d = types.fp16_to_fp32(q4.d);
    const dmin = types.fp16_to_fp32(q4.dmin);
    const q8_d = q8.d;

    const m4: V16u = @splat(0x0F);
    const sh4: V16u = @splat(4);
    var result: f32 = 0;

    var j: usize = 0;
    while (j < 4) : (j += 1) {
        const sc_idx = j * 2;
        const sc0 = if (sc_idx < 4) q4.scales[sc_idx] & 63 else (q4.scales[sc_idx + 4] & 0x0F) | ((q4.scales[sc_idx - 4] >> 6) << 4);
        const mn0 = if (sc_idx < 4) q4.scales[sc_idx + 4] & 63 else (q4.scales[sc_idx + 4] >> 4) | ((q4.scales[sc_idx] >> 6) << 4);
        const sc1 = if (sc_idx + 1 < 4) q4.scales[sc_idx + 1] & 63 else (q4.scales[sc_idx + 1 + 4] & 0x0F) | ((q4.scales[sc_idx + 1 - 4] >> 6) << 4);
        const mn1 = if (sc_idx + 1 < 4) q4.scales[sc_idx + 1 + 4] & 63 else (q4.scales[sc_idx + 1 + 4] >> 4) | ((q4.scales[sc_idx + 1] >> 6) << 4);

        const qoff = j * 32;
        const woff = j * 64;

        const qs0 = loadU8x16(&q4.qs[qoff]);
        const qs1 = loadU8x16(&q4.qs[qoff + 16]);

        const q4_low0 = qs0 & m4;
        const q4_low1 = qs1 & m4;
        const q4_hi0 = (qs0 >> sh4) & m4;
        const q4_hi1 = (qs1 >> sh4) & m4;

        const q4l0: V16w = @intCast(q4_low0);
        const q4l1: V16w = @intCast(q4_low1);
        const q4h0: V16w = @intCast(q4_hi0);
        const q4h1: V16w = @intCast(q4_hi1);

        const q8l0 = loadI8x16(&q8.qs[woff]);
        const q8l1 = loadI8x16(&q8.qs[woff + 16]);
        const q8h0 = loadI8x16(&q8.qs[woff + 32]);
        const q8h1 = loadI8x16(&q8.qs[woff + 48]);

        const q8l0_w: V16w = @intCast(q8l0);
        const q8l1_w: V16w = @intCast(q8l1);
        const q8h0_w: V16w = @intCast(q8h0);
        const q8h1_w: V16w = @intCast(q8h1);

        const pl0 = q4l0 * q8l0_w;
        const pl1 = q4l1 * q8l1_w;
        const ph0 = q4h0 * q8h0_w;
        const ph1 = q4h1 * q8h1_w;

        const dot_low: i32 = @reduce(.Add, @as(@Vector(16, i32), @intCast(pl0))) + @reduce(.Add, @as(@Vector(16, i32), @intCast(pl1)));
        const dot_hi: i32 = @reduce(.Add, @as(@Vector(16, i32), @intCast(ph0))) + @reduce(.Add, @as(@Vector(16, i32), @intCast(ph1)));

        const sum_q8_low: i32 = @reduce(.Add, q8l0_w) + @reduce(.Add, q8l1_w);

        const dsc0 = d * @as(f32, @floatFromInt(sc0));
        const dsc1 = d * @as(f32, @floatFromInt(sc1));
        const dmn0 = dmin * @as(f32, @floatFromInt(mn0));
        const dmn1 = dmin * @as(f32, @floatFromInt(mn1));

        result += q8_d * (dsc0 * @as(f32, @floatFromInt(dot_low)) - dmn0 * @as(f32, @floatFromInt(sum_q8_low)));
        result += q8_d * (dsc1 * @as(f32, @floatFromInt(dot_hi)) - dmn1 * @as(f32, @floatFromInt(@reduce(.Add, q8h0_w) + @reduce(.Add, q8h1_w))));
    }
    return result;
}

pub fn matvecQ4K_int(blocks: [*]const u8, x: [*]const f32, y: [*]f32, M: u32, K: u32) void {
    const blocks_per_row = K / 256;
    var q8_buf: [64]quant.block_q8_K = undefined;
    quantizeActivationToQ8K(x, @ptrCast(&q8_buf), K);

    var m: u32 = 0;
    while (m < M) : (m += 1) {
        var sum: f32 = 0;
        const row_ptr = blocks + @as(usize, m) * blocks_per_row * 144;
        for (0..blocks_per_row) |b| {
            const q4: *align(1) const quant.block_q4_K = @ptrCast(row_ptr + b * 144);
            sum += dotQ4KxQ8K(q4, &q8_buf[b]);
        }
        y[m] += sum;
    }
}

test "Q4_K int dot matches scalar" {
    var q4_block = std.mem.zeroes(quant.block_q4_K);
    q4_block.d = types.fp32_to_fp16(2.0);
    q4_block.dmin = types.fp32_to_fp16(1.0);
    q4_block.scales[0] = 1;
    q4_block.scales[4] = 2;
    q4_block.scales[8] = 0x11;
    q4_block.scales[9] = 0x22;

    var prng = std.Random.DefaultPrng.init(123);
    const rand = prng.random();
    for (0..q4_block.qs.len) |i| q4_block.qs[i] = rand.int(u8);

    var x: [256]f32 = undefined;
    for (0..x.len) |i| x[i] = @as(f32, @floatFromInt(@as(i32, @intCast(rand.int(u8) & 15)) - 8)) * 0.1;

    var y_out: [1]f32 = undefined;
    matvecQ4K_int(@as([*]const u8, @ptrCast(&q4_block)), &x, &y_out, 1, 256);

    const d_f32 = types.fp16_to_fp32(q4_block.d);
    const dmin_f32 = types.fp16_to_fp32(q4_block.dmin);
    var expected: f32 = 0;
    for (0..4) |blk| {
        const js = blk * 2;
        const sc0 = if (js < 4) q4_block.scales[js] & 63 else (q4_block.scales[js + 4] & 0x0F) | ((q4_block.scales[js - 4] >> 6) << 4);
        const mn0 = if (js < 4) q4_block.scales[js + 4] & 63 else (q4_block.scales[js + 4] >> 4) | ((q4_block.scales[js] >> 6) << 4);
        const sc1 = if (js + 1 < 4) q4_block.scales[js + 1] & 63 else (q4_block.scales[js + 1 + 4] & 0x0F) | ((q4_block.scales[js + 1 - 4] >> 6) << 4);
        const mn1 = if (js + 1 < 4) q4_block.scales[js + 1 + 4] & 63 else (q4_block.scales[js + 1 + 4] >> 4) | ((q4_block.scales[js + 1] >> 6) << 4);
        const rs0 = d_f32 * @as(f32, @floatFromInt(sc0));
        const rm0 = dmin_f32 * @as(f32, @floatFromInt(mn0));
        const rs1 = d_f32 * @as(f32, @floatFromInt(sc1));
        const rm1 = dmin_f32 * @as(f32, @floatFromInt(mn1));
        const qoff = blk * 32;
        const woff = blk * 64;
        for (0..16) |k| {
            expected += (rs0 * @as(f32, @floatFromInt(q4_block.qs[qoff + k] & 0x0F)) - rm0) * x[woff + k];
            expected += (rs1 * @as(f32, @floatFromInt(q4_block.qs[qoff + 16 + k] & 0x0F)) - rm1) * x[woff + 16 + k];
            expected += (rs0 * @as(f32, @floatFromInt(q4_block.qs[qoff + k] >> 4)) - rm0) * x[woff + 32 + k];
            expected += (rs1 * @as(f32, @floatFromInt(q4_block.qs[qoff + 16 + k] >> 4)) - rm1) * x[woff + 48 + k];
        }
    }
    try std.testing.expectApproxEqAbs(expected, y_out[0], 40.0);
}
