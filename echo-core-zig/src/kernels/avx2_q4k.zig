const std = @import("std");
const types = @import("../core/types.zig");
const quant = @import("quant.zig");

const V16 = @Vector(16, u8);
const V16f = @Vector(16, f32);
const V16i = @Vector(16, i32);

inline fn loadU8x16(ptr: *const u8) V16 {
    var buf: [16]u8 = undefined;
    @memcpy(buf[0..], @as([*]const u8, @ptrCast(ptr))[0..16]);
    return @bitCast(buf);
}

inline fn loadF32x16(ptr: [*]const f32) V16f {
    var buf: [16]f32 = undefined;
    const src_b: [*]const u8 = @ptrCast(ptr);
    const dst_b: [*]u8 = @ptrCast(&buf);
    @memcpy(dst_b[0..64], src_b[0..64]);
    return @bitCast(buf);
}

fn dotQ4Block(blocks: [*]const u8, x: [*]const f32) f32 {
    const d = types.fp16_to_fp32(std.mem.readInt(u16, blocks[0..2], .little));
    const dmin = types.fp16_to_fp32(std.mem.readInt(u16, blocks[2..4], .little));
    const scales = blocks[4..16];
    const qs = blocks[16..144];

    const m4: V16 = @splat(0x0F);
    const sh4: V16 = @splat(4);
    var result: f32 = 0;

    var blk: u32 = 0;
    while (blk < 4) : (blk += 1) {
        const js = blk * 2;

        const sc0: u8 = if (js < 4) scales[js] & 63 else (scales[js + 4] & 0x0F) | ((scales[js - 4] >> 6) << 4);
        const mn0: u8 = if (js < 4) scales[js + 4] & 63 else (scales[js + 4] >> 4) | ((scales[js] >> 6) << 4);
        const sc1: u8 = if (js + 1 < 4) scales[js + 1] & 63 else (scales[js + 1 + 4] & 0x0F) | ((scales[js + 1 - 4] >> 6) << 4);
        const mn1: u8 = if (js + 1 < 4) scales[js + 1 + 4] & 63 else (scales[js + 1 + 4] >> 4) | ((scales[js + 1] >> 6) << 4);

        const rs0 = d * @as(f32, @floatFromInt(sc0));
        const rm0 = dmin * @as(f32, @floatFromInt(mn0));
        const rs1 = d * @as(f32, @floatFromInt(sc1));
        const rm1 = dmin * @as(f32, @floatFromInt(mn1));

        const qoff = @as(usize, blk) * 32;
        const woff = @as(usize, blk) * 64;

        const qs0 = loadU8x16(&qs[qoff]);
        const qs1 = loadU8x16(&qs[qoff + 16]);

        const q_low0 = qs0 & m4;
        const q_low1 = qs1 & m4;
        const q_hi0 = (qs0 >> sh4) & m4;
        const q_hi1 = (qs1 >> sh4) & m4;

        const q0_f: V16f = @floatFromInt(@as(V16i, @intCast(q_low0)));
        const q1_f: V16f = @floatFromInt(@as(V16i, @intCast(q_low1)));
        const q2_f: V16f = @floatFromInt(@as(V16i, @intCast(q_hi0)));
        const q3_f: V16f = @floatFromInt(@as(V16i, @intCast(q_hi1)));

        const x0 = loadF32x16(x + woff);
        const x1 = loadF32x16(x + woff + 16);
        const x2 = loadF32x16(x + woff + 32);
        const x3 = loadF32x16(x + woff + 48);

        const w0 = q0_f * @as(V16f, @splat(rs0)) - @as(V16f, @splat(rm0));
        const w1 = q1_f * @as(V16f, @splat(rs1)) - @as(V16f, @splat(rm1));
        const w2 = q2_f * @as(V16f, @splat(rs0)) - @as(V16f, @splat(rm0));
        const w3 = q3_f * @as(V16f, @splat(rs1)) - @as(V16f, @splat(rm1));

        result += @reduce(.Add, w0 * x0);
        result += @reduce(.Add, w1 * x1);
        result += @reduce(.Add, w2 * x2);
        result += @reduce(.Add, w3 * x3);
    }

    return result;
}

pub fn matvecQ4K_avx2(blocks: [*]const u8, x: [*]const f32, y: [*]f32, M: u32, K: u32) void {
    const blocks_per_row = K / 256;
    const block_stride = 144;

    var m: u32 = 0;
    while (m < M) : (m += 1) {
        var sum: f32 = 0;
        const row_ptr = blocks + @as(usize, m) * blocks_per_row * block_stride;

        var b: u32 = 0;
        while (b < blocks_per_row) : (b += 1) {
            const bp = row_ptr + b * block_stride;
            const x_blk = x + b * 256;
            sum += dotQ4Block(bp, x_blk);
        }
        y[m] += sum;
    }
}

test "Q4_K AVX2 dot matches scalar for zero block" {
    var block = std.mem.zeroes(quant.block_q4_K);
    block.d = types.fp32_to_fp16(63.0);
    block.dmin = types.fp32_to_fp16(63.0);

    for (0..4) |j| {
        block.scales[j] = @intCast(j + 1);
        block.scales[j + 4] = @intCast(j);
    }
    block.scales[0] |= 1 << 6;
    block.scales[1] |= 1 << 6;
    block.scales[2] |= 1 << 6;
    block.scales[3] |= 1 << 6;
    block.scales[8] = 0x10 | 0x02;
    block.scales[9] = 0x20 | 0x03;
    block.scales[10] = 0x30 | 0x04;
    block.scales[11] = 0x40 | 0x05;
    @memset(&block.qs, 0x55);

    var x: [256]f32 = undefined;
    for (0..x.len) |i| x[i] = @as(f32, @floatFromInt((i % 11) + 1)) * 0.01;

    const avx2_result = dotQ4Block(std.mem.asBytes(&block).ptr, &x);

    const d_f32 = types.fp16_to_fp32(block.d);
    const dmin_f32 = types.fp16_to_fp32(block.dmin);
    var expected: f32 = 0;
    for (0..4) |blk| {
        const js = blk * 2;
        const sc0: u8 = if (js < 4) block.scales[js] & 63 else (block.scales[js + 4] & 0x0F) | ((block.scales[js - 4] >> 6) << 4);
        const mn0: u8 = if (js < 4) block.scales[js + 4] & 63 else (block.scales[js + 4] >> 4) | ((block.scales[js] >> 6) << 4);
        const sc1: u8 = if (js + 1 < 4) block.scales[js + 1] & 63 else (block.scales[js + 1 + 4] & 0x0F) | ((block.scales[js + 1 - 4] >> 6) << 4);
        const mn1: u8 = if (js + 1 < 4) block.scales[js + 1 + 4] & 63 else (block.scales[js + 1 + 4] >> 4) | ((block.scales[js + 1] >> 6) << 4);
        const rs0 = d_f32 * @as(f32, @floatFromInt(sc0));
        const rm0 = dmin_f32 * @as(f32, @floatFromInt(mn0));
        const rs1 = d_f32 * @as(f32, @floatFromInt(sc1));
        const rm1 = dmin_f32 * @as(f32, @floatFromInt(mn1));
        const qoff = blk * 32;
        const woff = blk * 64;

        for (0..16) |j| {
            expected += (rs0 * @as(f32, @floatFromInt(block.qs[qoff + j] & 0x0F)) - rm0) * x[woff + j];
            expected += (rs1 * @as(f32, @floatFromInt(block.qs[qoff + 16 + j] & 0x0F)) - rm1) * x[woff + 16 + j];
            expected += (rs0 * @as(f32, @floatFromInt(block.qs[qoff + j] >> 4)) - rm0) * x[woff + 32 + j];
            expected += (rs1 * @as(f32, @floatFromInt(block.qs[qoff + 16 + j] >> 4)) - rm1) * x[woff + 48 + j];
        }
    }

    try std.testing.expectApproxEqAbs(expected, avx2_result, 0.01);
}

test "Q4_K AVX2 dot matches scalar for random block" {
    var block = std.mem.zeroes(quant.block_q4_K);
    block.d = types.fp32_to_fp16(2.0);
    block.dmin = types.fp32_to_fp16(1.0);

    var prng = std.Random.DefaultPrng.init(99);
    const rand = prng.random();
    for (0..block.qs.len) |i| block.qs[i] = rand.int(u8);
    for (0..block.scales.len) |i| block.scales[i] = rand.int(u8);

    var x: [256]f32 = undefined;
    for (0..x.len) |i| x[i] = @as(f32, @floatFromInt(@as(i32, @intCast(rand.int(u8) & 15)) - 8)) * 0.1;

    const avx2_result = dotQ4Block(std.mem.asBytes(&block).ptr, &x);

    const d_f32 = types.fp16_to_fp32(block.d);
    const dmin_f32 = types.fp16_to_fp32(block.dmin);
    var expected: f32 = 0;
    for (0..4) |blk| {
        const js = blk * 2;
        const sc0: u8 = if (js < 4) block.scales[js] & 63 else (block.scales[js + 4] & 0x0F) | ((block.scales[js - 4] >> 6) << 4);
        const mn0: u8 = if (js < 4) block.scales[js + 4] & 63 else (block.scales[js + 4] >> 4) | ((block.scales[js] >> 6) << 4);
        const sc1: u8 = if (js + 1 < 4) block.scales[js + 1] & 63 else (block.scales[js + 1 + 4] & 0x0F) | ((block.scales[js + 1 - 4] >> 6) << 4);
        const mn1: u8 = if (js + 1 < 4) block.scales[js + 1 + 4] & 63 else (block.scales[js + 1 + 4] >> 4) | ((block.scales[js + 1] >> 6) << 4);
        const rs0 = d_f32 * @as(f32, @floatFromInt(sc0));
        const rm0 = dmin_f32 * @as(f32, @floatFromInt(mn0));
        const rs1 = d_f32 * @as(f32, @floatFromInt(sc1));
        const rm1 = dmin_f32 * @as(f32, @floatFromInt(mn1));
        const qoff = blk * 32;
        const woff = blk * 64;

        for (0..16) |j| {
            expected += (rs0 * @as(f32, @floatFromInt(block.qs[qoff + j] & 0x0F)) - rm0) * x[woff + j];
            expected += (rs1 * @as(f32, @floatFromInt(block.qs[qoff + 16 + j] & 0x0F)) - rm1) * x[woff + 16 + j];
            expected += (rs0 * @as(f32, @floatFromInt(block.qs[qoff + j] >> 4)) - rm0) * x[woff + 32 + j];
            expected += (rs1 * @as(f32, @floatFromInt(block.qs[qoff + 16 + j] >> 4)) - rm1) * x[woff + 48 + j];
        }
    }

    try std.testing.expectApproxEqAbs(expected, avx2_result, 0.1);
}
