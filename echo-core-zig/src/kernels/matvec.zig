const std = @import("std");
const types = @import("../core/types.zig");
const config = @import("../core/config.zig");
const quant = @import("quant.zig");
const gguf = @import("../gguf/reader.zig");

pub fn matvecFp16Fp32(
    comptime TILE_K: u32,
    comptime TILE_M: u32,
    W: [*]const u8,
    x: [*]const f32,
    y: [*]f32,
    M: u32,
    K: u32,
) void {
    _ = TILE_K;
    _ = TILE_M;
    const W_fp16: [*]const types.fp16_t = @ptrCast(@alignCast(W));
    var m: u32 = 0;
    while (m < M) : (m += 1) {
        const W_row = W_fp16 + @as(usize, m) * K;
        var acc: f32 = 0;

        var k: u32 = 0;
        while (k < K) : (k += 1) {
            acc += types.fp16_to_fp32(W_row[k]) * x[k];
        }

        y[m] += acc;
    }
}

pub fn matvecF32Fp32(
    comptime TILE_K: u32,
    comptime TILE_M: u32,
    W: [*]const u8,
    x: [*]const f32,
    y: [*]f32,
    M: u32,
    K: u32,
) void {
    _ = TILE_K;
    _ = TILE_M;
    const W_f32: [*]const f32 = @ptrCast(@alignCast(W));
    var m: u32 = 0;
    while (m < M) : (m += 1) {
        const W_row = W_f32 + @as(usize, m) * K;
        var acc: f32 = 0;

        var k: u32 = 0;
        while (k < K) : (k += 1) {
            acc += W_row[k] * x[k];
        }

        y[m] += acc;
    }
}

pub fn matvecDispatch(
    W: [*]const u8,
    x: [*]const f32,
    y: [*]f32,
    M: u32,
    K: u32,
    config_: anytype,
) void {
    _ = config_;
    matvecFp16Fp32(config.Intel13500H_Tiles.TILE_K, config.Intel13500H_Tiles.TILE_M, W, x, y, M, K);
}

pub fn matvecDispatchQuant(
    comptime TILE_K: u32,
    comptime TILE_M: u32,
    W: [*]const u8,
    x: [*]const f32,
    y: [*]f32,
    M: u32,
    K: u32,
    dtype: gguf.GGMLType,
) void {
    switch (dtype) {
        .f16 => matvecFp16Fp32(TILE_K, TILE_M, W, x, y, M, K),
        .f32 => matvecF32Fp32(TILE_K, TILE_M, W, x, y, M, K),
        .q8_0 => matvecQ80(W, x, y, M, K),
        .q6_k => {
            // Q6_K not yet optimized - treat as fp16 for now
            std.debug.print("WARN: Q6_K not optimized, treating as FP16 (may produce incorrect results)\n", .{});
            matvecFp16Fp32(TILE_K, TILE_M, W, x, y, M, K);
        },
        .q4_k => matvecQ4K(W, x, y, M, K),
        .q2_k => matvecQ2K(W, x, y, M, K),
        .iq2_xs, .iq4_xs => {
            // IQ types not yet optimized - treat as fp16 for now
            std.debug.print("WARN: {s} not optimized, treating as FP16 (may produce incorrect results)\n", .{@tagName(dtype)});
            matvecFp16Fp32(TILE_K, TILE_M, W, x, y, M, K);
        },
        else => {
            std.debug.print("WARN: unsupported dtype {s}, falling back to FP16 (may produce incorrect results)\n", .{@tagName(dtype)});
            matvecFp16Fp32(TILE_K, TILE_M, W, x, y, M, K);
        },
    }
}

pub fn matvecQ80(
    blocks: [*]const u8,
    x: [*]const f32,
    y: [*]f32,
    M: u32,
    K: u32,
) void {
    const blocks_per_row = K / 32;
    const block_stride = 34;

    var m: u32 = 0;
    while (m < M) : (m += 1) {
        var sum: f32 = 0;
        const row_ptr = blocks + @as(usize, m) * blocks_per_row * block_stride;

        var b: u32 = 0;
        while (b < blocks_per_row) : (b += 1) {
            const bp = row_ptr + b * block_stride;
            const d = types.fp16_to_fp32(std.mem.readInt(u16, bp[0..2], .little));
            const qs = @as([*]const i8, @ptrCast(bp + 2));
            const x_blk = x + b * 32;

            var block_sum: f32 = 0;
            var j: u32 = 0;
            while (j < 32) : (j += 1) {
                block_sum += @as(f32, @floatFromInt(qs[j])) * x_blk[j];
            }
            sum += d * block_sum;
        }
        y[m] += sum;
    }
}

pub fn matvecQ4K(
    blocks: [*]const u8,
    x: [*]const f32,
    y: [*]f32,
    M: u32,
    K: u32,
) void {
    const blocks_per_row = K / 256;
    const block_stride = 144;

    var m: u32 = 0;
    while (m < M) : (m += 1) {
        var sum: f32 = 0;
        const row_ptr = blocks + @as(usize, m) * blocks_per_row * block_stride;

        var b: u32 = 0;
        while (b < blocks_per_row) : (b += 1) {
            const bp = row_ptr + b * block_stride;
            const d = types.fp16_to_fp32(std.mem.readInt(u16, bp[0..2], .little));
            const dmin = types.fp16_to_fp32(std.mem.readInt(u16, bp[2..4], .little));
            const scales = bp[4..];
            const qs = bp[16..];
            const x_blk = x + b * 256;

            var blk: u32 = 0;
            while (blk < 4) : (blk += 1) {
                const js = blk * 2;
                const sc0: u8 = if (js < 4)
                    scales[js] & 63
                else
                    (scales[js + 4] & 0x0F) | ((scales[js - 4] >> 6) << 4);
                const mn0: u8 = if (js < 4)
                    scales[js + 4] & 63
                else
                    (scales[js + 4] >> 4) | ((scales[js] >> 6) << 4);
                const sc1: u8 = if (js + 1 < 4)
                    scales[js + 1] & 63
                else
                    (scales[js + 1 + 4] & 0x0F) | ((scales[js + 1 - 4] >> 6) << 4);
                const mn1: u8 = if (js + 1 < 4)
                    scales[js + 1 + 4] & 63
                else
                    (scales[js + 1 + 4] >> 4) | ((scales[js + 1] >> 6) << 4);

                const rs0 = d * @as(f32, @floatFromInt(sc0));
                const rm0 = dmin * @as(f32, @floatFromInt(mn0));
                const rs1 = d * @as(f32, @floatFromInt(sc1));
                const rm1 = dmin * @as(f32, @floatFromInt(mn1));

                const qoff = @as(usize, blk) * 32;
                const woff = @as(usize, blk) * 64;

                var j: u32 = 0;
                while (j < 16) : (j += 1) {
                    sum += (rs0 * @as(f32, @floatFromInt(qs[qoff + j] & 0x0F)) - rm0) * x_blk[woff + j];
                    sum += (rs1 * @as(f32, @floatFromInt(qs[qoff + 16 + j] & 0x0F)) - rm1) * x_blk[woff + 16 + j];
                    sum += (rs0 * @as(f32, @floatFromInt(qs[qoff + j] >> 4)) - rm0) * x_blk[woff + 32 + j];
                    sum += (rs1 * @as(f32, @floatFromInt(qs[qoff + 16 + j] >> 4)) - rm1) * x_blk[woff + 48 + j];
                }
            }
        }
        y[m] += sum;
    }
}

pub fn matvecQ2K(
    blocks: [*]const u8,
    x: [*]const f32,
    y: [*]f32,
    M: u32,
    K: u32,
) void {
    const blocks_per_row = K / 256;
    const block_stride = 84;

    var m: u32 = 0;
    while (m < M) : (m += 1) {
        var sum: f32 = 0;
        const row_ptr = blocks + @as(usize, m) * blocks_per_row * block_stride;

        var b: u32 = 0;
        while (b < blocks_per_row) : (b += 1) {
            const bp = row_ptr + b * block_stride;
            const d_all = types.fp16_to_fp32(std.mem.readInt(u16, bp[0..2], .little));
            const m_all = types.fp16_to_fp32(std.mem.readInt(u16, bp[2..4], .little));
            const scales = bp[4..];
            const qs = bp[20..];
            const x_blk = x + b * 256;

            var j: u32 = 0;
            while (j < 256) : (j += 1) {
                const sb: u32 = j / 8;
                const s = sb / 2;
                const scale = if (sb % 2 == 0)
                    d_all * @as(f32, @floatFromInt(scales[s] & 0x0F))
                else
                    d_all * @as(f32, @floatFromInt(scales[s + 8] & 0x0F));
                const min_val = if (sb % 2 == 0)
                    m_all * @as(f32, @floatFromInt(scales[s] >> 4))
                else
                    m_all * @as(f32, @floatFromInt(scales[s + 8] >> 4));
                const byte_idx: usize = j / 4;
                const bit_off: u3 = @intCast((j % 4) * 2);
                const q = (qs[byte_idx] >> bit_off) & 0x03;
                sum += (scale * @as(f32, @floatFromInt(q)) - min_val) * x_blk[j];
            }
        }
        y[m] += sum;
    }
}

test "matvec basic" {
    var W: [4 * 8]types.fp16_t = undefined;
    var x: [8]f32 = undefined;
    var y: [4]f32 = undefined;

    for (0..4) |m| {
        for (0..8) |k| {
            W[m * 8 + k] = types.fp32_to_fp16(@floatFromInt(m + 1));
        }
    }

    for (0..8) |k| {
        x[k] = @floatFromInt(k + 1);
    }

    @memset(&y, 0);

    const W_bytes: [*]const u8 = @ptrCast(&W);
    matvecFp16Fp32(8, 4, W_bytes, &x, &y, 4, 8);

    try std.testing.expectEqual(y[0], 36.0);
    try std.testing.expectEqual(y[1], 72.0);
    try std.testing.expectEqual(y[2], 108.0);
    try std.testing.expectEqual(y[3], 144.0);
}

test "matvec f32 basic" {
    var W = [_]f32{
        1, 1, 1, 1,
        2, 2, 2, 2,
    };
    var x = [_]f32{ 1, 2, 3, 4 };
    var y = [_]f32{ 0, 0 };

    const W_bytes: [*]const u8 = @ptrCast(&W);
    matvecF32Fp32(4, 2, W_bytes, &x, &y, 2, 4);

    try std.testing.expectEqual(@as(f32, 10), y[0]);
    try std.testing.expectEqual(@as(f32, 20), y[1]);
}

test "matvecQ4K matches canonical scale packing" {
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

    var y = [_]f32{0};
    matvecQ4K(std.mem.asBytes(&block).ptr, &x, &y, 1, 256);

    const d_f32 = types.fp16_to_fp32(block.d);
    const dmin_f32 = types.fp16_to_fp32(block.dmin);
    var expected: f32 = 0;
    for (0..4) |blk| {
        const js = blk * 2;
        const sc0: u8 = if (js < 4)
            block.scales[js] & 63
        else
            (block.scales[js + 4] & 0x0F) | ((block.scales[js - 4] >> 6) << 4);
        const mn0: u8 = if (js < 4)
            block.scales[js + 4] & 63
        else
            (block.scales[js + 4] >> 4) | ((block.scales[js] >> 6) << 4);
        const sc1: u8 = if (js + 1 < 4)
            block.scales[js + 1] & 63
        else
            (block.scales[js + 1 + 4] & 0x0F) | ((block.scales[js + 1 - 4] >> 6) << 4);
        const mn1: u8 = if (js + 1 < 4)
            block.scales[js + 1 + 4] & 63
        else
            (block.scales[js + 1 + 4] >> 4) | ((block.scales[js + 1] >> 6) << 4);

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

    try std.testing.expectApproxEqAbs(expected, y[0], 0.01);
}
