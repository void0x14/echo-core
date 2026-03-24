const std = @import("std");
const types = @import("../core/types.zig");
const config = @import("../core/config.zig");

pub fn matvecFp16Fp32(
    comptime TILE_K: u32,
    comptime TILE_M: u32,
    W: [*]const types.fp16_t,
    x: [*]const f32,
    y: [*]f32,
    M: u32,
    K: u32,
) void {
    _ = TILE_K;
    _ = TILE_M;
    var m: u32 = 0;
    while (m < M) : (m += 1) {
        const W_row = W + @as(usize, m) * K;
        var acc: f32 = 0;

        var k: u32 = 0;
        while (k < K) : (k += 1) {
            acc += types.fp16_to_fp32(W_row[k]) * x[k];
        }

        y[m] += acc;
    }
}

pub fn matvecDispatch(
    W: [*]const types.fp16_t,
    x: [*]const f32,
    y: [*]f32,
    M: u32,
    K: u32,
    config_: anytype,
) void {
    _ = config_;
    matvecFp16Fp32(config.Intel13500H_Tiles.TILE_K, config.Intel13500H_Tiles.TILE_M, W, x, y, M, K);
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
            const d = types.fp16_to_fp32(@as(u16, @bitCast(@as(u32, bp[0]) | (@as(u32, bp[1]) << 8))));
            const qs = @as([*]const i8, @ptrCast(bp + 2));
            const x_blk = x + b * 32;

            var block_sum: f32 = 0;
            var j: u32 = 0;
            while (j < 32) : (j += 1) {
                block_sum += @as(f32, qs[j]) * x_blk[j];
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
            const d = types.fp16_to_fp32(@as(u16, @bitCast(@as(u32, bp[0]) | (@as(u32, bp[1]) << 8))));
            const dmin = types.fp16_to_fp32(@as(u16, @bitCast(@as(u32, bp[2]) | (@as(u32, bp[3]) << 8))));
            const scales = bp[4..];
            const qs = bp[16..];
            const x_blk = x + b * 256;

            var blk: u32 = 0;
            while (blk < 4) : (blk += 1) {
                const js = blk * 2;
                const sc0 = scales[js] & 63;
                const mn0 = scales[js + 4] & 63;
                const sc1 = scales[js + 1] & 63;
                const mn1 = scales[js + 1 + 4] & 63;

                const rs0 = d * @as(f32, sc0);
                const rm0 = dmin * @as(f32, mn0);
                const rs1 = d * @as(f32, sc1);
                const rm1 = dmin * @as(f32, mn1);

                const qoff = @as(usize, blk) * 32;
                const woff = @as(usize, blk) * 64;

                var j: u32 = 0;
                while (j < 16) : (j += 1) {
                    sum += (rs0 * @as(f32, qs[qoff + j] & 0x0F) - rm0) * x_blk[woff + j];
                    sum += (rs1 * @as(f32, qs[qoff + 16 + j] & 0x0F) - rm1) * x_blk[woff + 16 + j];
                    sum += (rs0 * @as(f32, qs[qoff + j] >> 4) - rm0) * x_blk[woff + 32 + j];
                    sum += (rs1 * @as(f32, qs[qoff + 16 + j] >> 4) - rm1) * x_blk[woff + 48 + j];
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
            const d_all = types.fp16_to_fp32(@as(u16, @bitCast(@as(u32, bp[0]) | (@as(u32, bp[1]) << 8))));
            const m_all = types.fp16_to_fp32(@as(u16, @bitCast(@as(u32, bp[2]) | (@as(u32, bp[3]) << 8))));
            const scales = bp[4..];
            const qs = bp[20..];
            const x_blk = x + b * 256;

            var j: u32 = 0;
            while (j < 256) : (j += 1) {
                const sb: u32 = j / 8;
                const s = sb / 2;
                const scale = if (sb % 2 == 0) d_all * @as(f32, scales[s] & 0x0F) else d_all * @as(f32, scales[s + 8] & 0x0F);
                const min_val = if (sb % 2 == 0) m_all * @as(f32, scales[s] >> 4) else m_all * @as(f32, scales[s + 8] >> 4);
                const byte_idx: usize = j / 4;
                const bit_off: u3 = @intCast((j % 4) * 2);
                const q = (qs[byte_idx] >> bit_off) & 0x03;
                sum += (scale * @as(f32, q) - min_val) * x_blk[j];
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

    matvecFp16Fp32(8, 4, &W, &x, &y, 4, 8);

    try std.testing.expectEqual(y[0], 36.0);
    try std.testing.expectEqual(y[1], 72.0);
    try std.testing.expectEqual(y[2], 108.0);
    try std.testing.expectEqual(y[3], 144.0);
}
