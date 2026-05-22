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

fn dotQ5Block(blocks: [*]const u8, x: [*]const f32) f32 {
    const d = types.fp16_to_fp32(std.mem.readInt(u16, blocks[0..2], .little));
    const dmin = types.fp16_to_fp32(std.mem.readInt(u16, blocks[2..4], .little));
    const scales = blocks[4..16];
    const qh = blocks[16..48];
    const qs = blocks[48..176];

    const m4: V16 = @splat(0x0F);
    const sh4: V16 = @splat(4);
    const s1: V16 = @splat(1);
    var result: f32 = 0;
    var ql_off: usize = 0;

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

        const woff = blk * 64;
        const shift1: u3 = @intCast(blk * 2);
        const shift2: u3 = @intCast(blk * 2 + 1);

        // Process l = 0..15
        {
            const qs16 = loadU8x16(&qs[ql_off]);
            const qh16 = loadU8x16(&qh[0]);

            const low = qs16 & m4;
            const high = (qs16 >> sh4) & m4;

            const hi1 = ((qh16 >> @as(@Vector(16, u3), @splat(shift1))) & s1) << sh4;
            const hi2 = ((qh16 >> @as(@Vector(16, u3), @splat(shift2))) & s1) << sh4;

            const q1: V16 = low | hi1;
            const q2: V16 = high | hi2;

            const q1f: V16f = @floatFromInt(@as(V16i, @intCast(q1)));
            const q2f: V16f = @floatFromInt(@as(V16i, @intCast(q2)));

            const x0 = loadF32x16(x + woff);
            const x1 = loadF32x16(x + woff + 32);

            const w1 = q1f * @as(V16f, @splat(rs0)) - @as(V16f, @splat(rm0));
            const w2 = q2f * @as(V16f, @splat(rs1)) - @as(V16f, @splat(rm1));

            result += @reduce(.Add, w1 * x0);
            result += @reduce(.Add, w2 * x1);
        }

        // Process l = 16..31
        {
            const qs16 = loadU8x16(&qs[ql_off + 16]);
            const qh16 = loadU8x16(&qh[16]);

            const low = qs16 & m4;
            const high = (qs16 >> sh4) & m4;

            const hi1 = ((qh16 >> @as(@Vector(16, u3), @splat(shift1))) & s1) << sh4;
            const hi2 = ((qh16 >> @as(@Vector(16, u3), @splat(shift2))) & s1) << sh4;

            const q1: V16 = low | hi1;
            const q2: V16 = high | hi2;

            const q1f: V16f = @floatFromInt(@as(V16i, @intCast(q1)));
            const q2f: V16f = @floatFromInt(@as(V16i, @intCast(q2)));

            const x0 = loadF32x16(x + woff + 16);
            const x1 = loadF32x16(x + woff + 48);

            const w1 = q1f * @as(V16f, @splat(rs0)) - @as(V16f, @splat(rm0));
            const w2 = q2f * @as(V16f, @splat(rs1)) - @as(V16f, @splat(rm1));

            result += @reduce(.Add, w1 * x0);
            result += @reduce(.Add, w2 * x1);
        }

        ql_off += 32;
    }

    return result;
}

pub fn matvecQ5K_avx2(blocks: [*]const u8, x: [*]const f32, y: [*]f32, M: u32, K: u32) void {
    const blocks_per_row = K / 256;
    const block_stride = 176;

    var m: u32 = 0;
    while (m < M) : (m += 1) {
        var sum: f32 = 0;
        const row_ptr = blocks + @as(usize, m) * blocks_per_row * block_stride;

        var b: u32 = 0;
        while (b < blocks_per_row) : (b += 1) {
            const bp = row_ptr + b * block_stride;
            const x_blk = x + b * 256;
            sum += dotQ5Block(bp, x_blk);
        }
        y[m] += sum;
    }
}

test "Q5_K AVX2 dot matches scalar for zero block" {
    var block = std.mem.zeroes(quant.block_q5_K);
    block.d = types.fp32_to_fp16(2.0);
    block.dmin = types.fp32_to_fp16(1.0);
    block.scales[0] = 1;
    block.scales[1] = 2;
    block.scales[2] = 3;
    block.scales[3] = 4;
    block.scales[4] = 5;
    block.scales[5] = 6;
    block.scales[6] = 7;
    block.scales[7] = 8;
    block.qs[0] = 0x32;
    block.qh[0] = 0xE4;

    var x: [256]f32 = undefined;
    for (0..x.len) |i| x[i] = @as(f32, @floatFromInt((i % 5) + 1)) * 0.01;

    const avx2_result = dotQ5Block(std.mem.asBytes(&block).ptr, &x);

    const d_f32 = types.fp16_to_fp32(block.d);
    const dmin_f32 = types.fp16_to_fp32(block.dmin);
    var expected: f32 = 0;
    var ql_off2: usize = 0;
    var mask1: u8 = 1;
    var mask2: u8 = 2;
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
        const woff = blk * 64;

        for (0..32) |l| {
            const q = block.qs[ql_off2 + l];
            const hi1: u8 = if ((block.qh[l] & mask1) != 0) 16 else 0;
            const hi2: u8 = if ((block.qh[l] & mask2) != 0) 16 else 0;
            expected += (rs0 * @as(f32, @floatFromInt((q & 0x0F) + hi1)) - rm0) * x[woff + l];
            expected += (rs1 * @as(f32, @floatFromInt((q >> 4) + hi2)) - rm1) * x[woff + 32 + l];
        }

        ql_off2 += 32;
        mask1 <<= 2;
        mask2 <<= 2;
    }

    try std.testing.expectApproxEqAbs(expected, avx2_result, 0.01);
}

test "Q5_K AVX2 dot matches scalar for random block" {
    var block = std.mem.zeroes(quant.block_q5_K);
    block.d = types.fp32_to_fp16(1.5);
    block.dmin = types.fp32_to_fp16(0.5);

    var prng = std.Random.DefaultPrng.init(77);
    const rand = prng.random();
    for (0..block.qs.len) |i| block.qs[i] = rand.int(u8);
    for (0..block.qh.len) |i| block.qh[i] = rand.int(u8);
    for (0..block.scales.len) |i| block.scales[i] = rand.int(u8);

    var x: [256]f32 = undefined;
    for (0..x.len) |i| x[i] = @as(f32, @floatFromInt(@as(i32, @intCast(rand.int(u8) & 31)) - 16)) * 0.05;

    const avx2_result = dotQ5Block(std.mem.asBytes(&block).ptr, &x);

    const d_f32 = types.fp16_to_fp32(block.d);
    const dmin_f32 = types.fp16_to_fp32(block.dmin);
    var expected: f32 = 0;
    var ql_off2: usize = 0;
    var mask1: u8 = 1;
    var mask2: u8 = 2;
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
        const woff = blk * 64;

        for (0..32) |l| {
            const q = block.qs[ql_off2 + l];
            const hi1: u8 = if ((block.qh[l] & mask1) != 0) 16 else 0;
            const hi2: u8 = if ((block.qh[l] & mask2) != 0) 16 else 0;
            expected += (rs0 * @as(f32, @floatFromInt((q & 0x0F) + hi1)) - rm0) * x[woff + l];
            expected += (rs1 * @as(f32, @floatFromInt((q >> 4) + hi2)) - rm1) * x[woff + 32 + l];
        }

        ql_off2 += 32;
        mask1 <<= 2;
        mask2 <<= 2;
    }

    try std.testing.expectApproxEqAbs(expected, avx2_result, 0.1);
}
