const std = @import("std");
const types = @import("../core/types.zig");
const quant = @import("quant.zig");

const V16u = @Vector(16, u8);
const V16i = @Vector(16, i8);
const V16w = @Vector(16, i16);
const V16d = @Vector(16, i32);
const V8f = @Vector(8, f32);

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

fn quantizeRowQ8K(x: [*]const f32, q8: *quant.block_q8_K) void {
    var amax: f32 = 0;
    for (0..256) |i| {
        const ax = @abs(x[i]);
        if (ax > amax) amax = ax;
    }
    if (amax == 0) {
        q8.d = 0;
        @memset(&q8.qs, 0);
        @memset(&q8.bsums, 0);
        return;
    }
    const iscale = 127.0 / amax;
    q8.d = amax / 127.0;
    for (0..256) |i| {
        const v = @as(i32, @intFromFloat(@round(x[i] * iscale)));
        q8.qs[i] = @intCast(@max(-128, @min(127, v)));
    }
    for (0..16) |j| {
        var sum: i32 = 0;
        for (0..16) |i| sum += q8.qs[j * 16 + i];
        q8.bsums[j] = @intCast(sum);
    }
}

fn dotQ4KxQ8K(q4: *align(1) const quant.block_q4_K, q8: *const quant.block_q8_K) f32 {
    const d = types.fp16_to_fp32(q4.d);
    const dmin = types.fp16_to_fp32(q4.dmin);
    const m4: V16u = @splat(0x0F);
    const sh4: V16u = @splat(4);
    var result: f32 = 0;

    var blk: u32 = 0;
    while (blk < 4) : (blk += 1) {
        const js = blk * 2;
        const sc0 = if (js < 4) q4.scales[js] & 63 else (q4.scales[js + 4] & 0x0F) | ((q4.scales[js - 4] >> 6) << 4);
        const mn0 = if (js < 4) q4.scales[js + 4] & 63 else (q4.scales[js + 4] >> 4) | ((q4.scales[js] >> 6) << 4);
        const sc1 = if (js + 1 < 4) q4.scales[js + 1] & 63 else (q4.scales[js + 1 + 4] & 0x0F) | ((q4.scales[js + 1 - 4] >> 6) << 4);
        const mn1 = if (js + 1 < 4) q4.scales[js + 1 + 4] & 63 else (q4.scales[js + 1 + 4] >> 4) | ((q4.scales[js + 1] >> 6) << 4);

        const qoff = @as(usize, blk) * 32;
        const woff = @as(usize, blk) * 64;

        const qs0 = loadU8x16(&q4.qs[qoff]);
        const qs1 = loadU8x16(&q4.qs[qoff + 16]);

        const q4_low0 = qs0 & m4;
        const q4_low1 = qs1 & m4;
        const q4_hi0 = (qs0 >> sh4) & m4;
        const q4_hi1 = (qs1 >> sh4) & m4;

        const q4_0w: V16w = @intCast(q4_low0);
        const q4_1w: V16w = @intCast(q4_low1);
        const q4_2w: V16w = @intCast(q4_hi0);
        const q4_3w: V16w = @intCast(q4_hi1);

        const q8_0w: V16w = @intCast(loadI8x16(&q8.qs[woff]));
        const q8_1w: V16w = @intCast(loadI8x16(&q8.qs[woff + 16]));
        const q8_2w: V16w = @intCast(loadI8x16(&q8.qs[woff + 32]));
        const q8_3w: V16w = @intCast(loadI8x16(&q8.qs[woff + 48]));

        const dot0: V16d = @intCast(q4_0w * q8_0w);
        const dot1: V16d = @intCast(q4_1w * q8_1w);
        const dot2: V16d = @intCast(q4_2w * q8_2w);
        const dot3: V16d = @intCast(q4_3w * q8_3w);

        const s0_i = @reduce(.Add, dot0);
        const s1_i = @reduce(.Add, dot1);
        const s2_i = @reduce(.Add, dot2);
        const s3_i = @reduce(.Add, dot3);

        const grp = woff / 16;
        const sum_q8_0 = @as(f32, @floatFromInt(q8.bsums[grp]));
        const sum_q8_1 = @as(f32, @floatFromInt(q8.bsums[grp + 1]));
        const sum_q8_2 = @as(f32, @floatFromInt(q8.bsums[grp + 2]));
        const sum_q8_3 = @as(f32, @floatFromInt(q8.bsums[grp + 3]));

        const q8_d = q8.d;
        result += q8_d * (d * @as(f32, @floatFromInt(sc0)) * @as(f32, @floatFromInt(s0_i)) - dmin * @as(f32, @floatFromInt(mn0)) * sum_q8_0);
        result += q8_d * (d * @as(f32, @floatFromInt(sc1)) * @as(f32, @floatFromInt(s1_i)) - dmin * @as(f32, @floatFromInt(mn1)) * sum_q8_1);
        result += q8_d * (d * @as(f32, @floatFromInt(sc0)) * @as(f32, @floatFromInt(s2_i)) - dmin * @as(f32, @floatFromInt(mn0)) * sum_q8_2);
        result += q8_d * (d * @as(f32, @floatFromInt(sc1)) * @as(f32, @floatFromInt(s3_i)) - dmin * @as(f32, @floatFromInt(mn1)) * sum_q8_3);
    }
    return result;
}

pub fn matvecQ4K_int(blocks: [*]const u8, x: [*]const f32, y: [*]f32, M: u32, K: u32) void {
    const blocks_per_row = K / 256;
    const nq8_blocks = blocks_per_row;
    var q8_buf: [64]quant.block_q8_K = undefined;
    if (nq8_blocks > q8_buf.len) return;

    for (0..nq8_blocks) |b| {
        quantizeRowQ8K(x + b * 256, &q8_buf[b]);
    }

    var m: u32 = 0;
    while (m < M) : (m += 1) {
        var sum: f32 = 0;
        const row_ptr = blocks + @as(usize, m) * blocks_per_row * 144;
        for (0..nq8_blocks) |b| {
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
        for (0..16) |j| {
            expected += (rs0 * @as(f32, @floatFromInt(q4_block.qs[qoff + j] & 0x0F)) - rm0) * x[woff + j];
            expected += (rs1 * @as(f32, @floatFromInt(q4_block.qs[qoff + 16 + j] & 0x0F)) - rm1) * x[woff + 16 + j];
            expected += (rs0 * @as(f32, @floatFromInt(q4_block.qs[qoff + j] >> 4)) - rm0) * x[woff + 32 + j];
            expected += (rs1 * @as(f32, @floatFromInt(q4_block.qs[qoff + 16 + j] >> 4)) - rm1) * x[woff + 48 + j];
        }
    }

    try std.testing.expectApproxEqAbs(expected, y_out[0], 5.0);
}
