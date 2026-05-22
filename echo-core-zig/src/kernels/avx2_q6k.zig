const std = @import("std");
const types = @import("../core/types.zig");
const quant = @import("quant.zig");

inline fn loadU8x16(ptr: *const u8) @Vector(16, u8) {
    var buf: [16]u8 = undefined;
    @memcpy(buf[0..], @as([*]const u8, @ptrCast(ptr))[0..16]);
    return @bitCast(buf);
}

inline fn loadF32x16(ptr: [*]const f32) @Vector(16, f32) {
    var buf: [16]f32 = undefined;
    const src_b: [*]const u8 = @ptrCast(ptr);
    const dst_b: [*]u8 = @ptrCast(&buf);
    @memcpy(dst_b[0..64], src_b[0..64]);
    return @bitCast(buf);
}

fn dotQ6Block(block: *align(1) const quant.block_q6_K, x: [*]const f32) f32 {
    const d = types.fp16_to_fp32(block.d);
    const m4: @Vector(16, u8) = @splat(0x0F);
    const m2: @Vector(16, u8) = @splat(0x03);
    const sh2: @Vector(16, u8) = @splat(2);
    const sh4: @Vector(16, u8) = @splat(4);
    const sh6: @Vector(16, u8) = @splat(6);
    const s32: @Vector(16, f32) = @splat(32.0);
    const zeroi: @Vector(16, i32) = @splat(0);
    var result: f32 = 0;

    var n: usize = 0;
    while (n < 256) : (n += 128) {
        const ql_base: usize = n / 2;
        const qh_base: usize = n / 4;
        const sc_base: usize = n / 16;

        var l: usize = 0;
        while (l < 32) : (l += 16) {
            const is = l / 16;

            const ql_lo = loadU8x16(&block.ql[ql_base + l]);
            const ql_hi = loadU8x16(&block.ql[ql_base + 32 + l]);
            const qh = loadU8x16(&block.qh[qh_base + l]);

            const q1 = (ql_lo & m4) | ((qh & m2) << sh4);
            const q2 = (ql_hi & m4) | (((qh >> sh2) & m2) << sh4);
            const q3 = ((ql_lo >> sh4) & m4) | (((qh >> sh4) & m2) << sh4);
            const q4 = ((ql_hi >> sh4) & m4) | (((qh >> sh6) & m2) << sh4);

            const s0 = @as(f32, @floatFromInt(block.scales[sc_base + is + 0]));
            const s1 = @as(f32, @floatFromInt(block.scales[sc_base + is + 2]));
            const s2 = @as(f32, @floatFromInt(block.scales[sc_base + is + 4]));
            const s3 = @as(f32, @floatFromInt(block.scales[sc_base + is + 6]));

            const x0 = loadF32x16(x + n + l);
            const x1 = loadF32x16(x + n + l + 32);
            const x2 = loadF32x16(x + n + l + 64);
            const x3 = loadF32x16(x + n + l + 96);

            const q1_f: @Vector(16, f32) = @floatFromInt(@as(@Vector(16, i32), @intCast(q1)) + zeroi);
            const q2_f: @Vector(16, f32) = @floatFromInt(@as(@Vector(16, i32), @intCast(q2)) + zeroi);
            const q3_f: @Vector(16, f32) = @floatFromInt(@as(@Vector(16, i32), @intCast(q3)) + zeroi);
            const q4_f: @Vector(16, f32) = @floatFromInt(@as(@Vector(16, i32), @intCast(q4)) + zeroi);

            const w0 = (q1_f - s32) * @as(@Vector(16, f32), @splat(d * s0));
            const w1 = (q2_f - s32) * @as(@Vector(16, f32), @splat(d * s1));
            const w2 = (q3_f - s32) * @as(@Vector(16, f32), @splat(d * s2));
            const w3 = (q4_f - s32) * @as(@Vector(16, f32), @splat(d * s3));

            const dot0 = @reduce(.Add, w0 * x0);
            const dot1 = @reduce(.Add, w1 * x1);
            const dot2 = @reduce(.Add, w2 * x2);
            const dot3 = @reduce(.Add, w3 * x3);

            result += dot0 + dot1 + dot2 + dot3;
        }
    }
    return result;
}

pub fn matvecQ6K_avx2(blocks: [*]const u8, x: [*]const f32, y: [*]f32, M: u32, K: u32) void {
    const blocks_per_row = K / 256;
    const block_stride = 210;

    var m: u32 = 0;
    while (m < M) : (m += 1) {
        var sum: f32 = 0;
        const row_ptr = blocks + @as(usize, m) * blocks_per_row * block_stride;

        var b: u32 = 0;
        while (b < blocks_per_row) : (b += 1) {
            const block: *align(1) const quant.block_q6_K = @ptrCast(row_ptr + b * block_stride);
            const x_blk = x + b * 256;
            sum += dotQ6Block(block, x_blk);
        }
        y[m] += sum;
    }
}

test "Q6_K AVX2 dot matches scalar dequant for zero block" {
    var block = std.mem.zeroes(quant.block_q6_K);
    block.d = types.fp32_to_fp16(1.0);
    block.scales[0] = 1;
    block.scales[2] = -1;
    block.scales[4] = 2;
    block.scales[6] = -2;
    block.scales[8] = 1;
    block.scales[10] = -1;
    block.scales[12] = 2;
    block.scales[14] = -2;
    block.ql[0] = 0x10;
    block.qh[0] = 0xE4;

    var x: [256]f32 = undefined;
    for (0..x.len) |i| x[i] = @as(f32, @floatFromInt((i % 5) + 1)) * 0.01;

    const avx2_result = dotQ6Block(&block, &x);

    var expected: f32 = 0;
    const d = types.fp16_to_fp32(block.d);
    var bn: usize = 0;
    while (bn < 256) : (bn += 128) {
        var l: usize = 0;
        while (l < 32) : (l += 1) {
            const is = l / 16;
            const qh_b = block.qh[bn / 4 + l];
            const q1: i8 = @intCast((block.ql[bn / 2 + l] & 0x0F) | (((qh_b >> 0) & 0x03) << 4));
            const q2: i8 = @intCast((block.ql[bn / 2 + 32 + l] & 0x0F) | (((qh_b >> 2) & 0x03) << 4));
            const q3: i8 = @intCast((block.ql[bn / 2 + l] >> 4) | (((qh_b >> 4) & 0x03) << 4));
            const q4: i8 = @intCast((block.ql[bn / 2 + 32 + l] >> 4) | (((qh_b >> 6) & 0x03) << 4));
            const s0 = block.scales[bn / 16 + is + 0];
            const s1 = block.scales[bn / 16 + is + 2];
            const s2 = block.scales[bn / 16 + is + 4];
            const s3 = block.scales[bn / 16 + is + 6];
            expected += d * @as(f32, @floatFromInt(s0)) * @as(f32, @floatFromInt(q1 - 32)) * x[bn + l + 0];
            expected += d * @as(f32, @floatFromInt(s1)) * @as(f32, @floatFromInt(q2 - 32)) * x[bn + l + 32];
            expected += d * @as(f32, @floatFromInt(s2)) * @as(f32, @floatFromInt(q3 - 32)) * x[bn + l + 64];
            expected += d * @as(f32, @floatFromInt(s3)) * @as(f32, @floatFromInt(q4 - 32)) * x[bn + l + 96];
        }
    }

    try std.testing.expectApproxEqAbs(expected, avx2_result, 0.01);
}

test "Q6_K AVX2 dot matches scalar dequant random block" {
    var block = std.mem.zeroes(quant.block_q6_K);
    block.d = types.fp32_to_fp16(2.5);

    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();
    for (0..block.ql.len) |i| block.ql[i] = rand.int(u8);
    for (0..block.qh.len) |i| block.qh[i] = rand.int(u8);
    for (0..block.scales.len) |i| block.scales[i] = rand.int(i8);

    var x: [256]f32 = undefined;
    for (0..x.len) |i| x[i] = @as(f32, @floatFromInt(@as(i32, @intCast(rand.int(u8) & 63)) - 32)) * 0.5;

    const avx2_result = dotQ6Block(&block, &x);

    var expected: f32 = 0;
    const d = types.fp16_to_fp32(block.d);
    var bn: usize = 0;
    while (bn < 256) : (bn += 128) {
        var l: usize = 0;
        while (l < 32) : (l += 1) {
            const is = l / 16;
            const qh_b = block.qh[bn / 4 + l];
            const q1: i8 = @intCast((block.ql[bn / 2 + l] & 0x0F) | (((qh_b >> 0) & 0x03) << 4));
            const q2: i8 = @intCast((block.ql[bn / 2 + 32 + l] & 0x0F) | (((qh_b >> 2) & 0x03) << 4));
            const q3: i8 = @intCast((block.ql[bn / 2 + l] >> 4) | (((qh_b >> 4) & 0x03) << 4));
            const q4: i8 = @intCast((block.ql[bn / 2 + 32 + l] >> 4) | (((qh_b >> 6) & 0x03) << 4));
            const s0 = block.scales[bn / 16 + is + 0];
            const s1 = block.scales[bn / 16 + is + 2];
            const s2 = block.scales[bn / 16 + is + 4];
            const s3 = block.scales[bn / 16 + is + 6];
            expected += d * @as(f32, @floatFromInt(s0)) * @as(f32, @floatFromInt(q1 - 32)) * x[bn + l + 0];
            expected += d * @as(f32, @floatFromInt(s1)) * @as(f32, @floatFromInt(q2 - 32)) * x[bn + l + 32];
            expected += d * @as(f32, @floatFromInt(s2)) * @as(f32, @floatFromInt(q3 - 32)) * x[bn + l + 64];
            expected += d * @as(f32, @floatFromInt(s3)) * @as(f32, @floatFromInt(q4 - 32)) * x[bn + l + 96];
        }
    }

    try std.testing.expectApproxEqAbs(expected, avx2_result, 0.1);
}
