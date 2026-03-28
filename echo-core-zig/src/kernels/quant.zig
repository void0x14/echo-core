const std = @import("std");
const types = @import("../core/types.zig");
const iq2_tables = @import("iq2_tables.zig");

pub const block_q8_0 = extern struct {
    d: u16,
    qs: [32]i8,
};

pub const block_q4_K = extern struct {
    d: u16,
    dmin: u16,
    scales: [12]u8,
    qs: [128]u8,
};

pub const block_q6_K = extern struct {
    ql: [128]u8,
    qh: [64]u8,
    scales: [16]i8,
    d: u16,
};

pub const block_q5_K = extern struct {
    d: u16,
    dmin: u16,
    scales: [12]u8,
    qh: [32]u8,
    qs: [128]u8,
};

pub const block_iq2_xs = extern struct {
    d: u16,
    qs: [32]u16,
    scales: [8]u8,
};

pub const block_iq4_xs = extern struct {
    d: u16,
    scales_h: u16,
    scales_l: [4]u8,
    qs: [128]u8,
};

const kvalues_iq4nl = [16]i8{ -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

fn getScaleMinK4(j: usize, q: [12]u8) struct { sc: u8, m: u8 } {
    return if (j < 4)
        .{ .sc = q[j] & 63, .m = q[j + 4] & 63 }
    else
        .{
            .sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4),
            .m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        };
}

pub fn quantizePerTokenSymmetric(
    input: [*]const f32,
    output: [*]i8,
    scales: [*]f32,
    num_tokens: u32,
    num_elements: u32,
) void {
    var t: u32 = 0;
    while (t < num_tokens) : (t += 1) {
        const row = input + @as(usize, t) * num_elements;
        const out_row = output + @as(usize, t) * num_elements;

        var max_abs: f32 = 0;
        var i: u32 = 0;
        while (i < num_elements) : (i += 1) {
            const abs_val = if (row[i] < 0) -row[i] else row[i];
            if (abs_val > max_abs) max_abs = abs_val;
        }

        const scale = if (max_abs > 0) max_abs / 127.0 else 1.0;
        scales[t] = scale;

        i = 0;
        while (i < num_elements) : (i += 1) {
            var v = @round(row[i] / scale);
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            out_row[i] = @intFromFloat(v);
        }
    }
}

pub fn fusedDequantDotInt8(
    query: [*]const f32,
    key_cache: [*]const i8,
    scales: [*]const f32,
    scores: [*]f32,
    dim: u32,
    seq_len: u32,
) void {
    var pos: u32 = 0;
    while (pos < seq_len) : (pos += 1) {
        const key = key_cache + @as(usize, pos) * dim;
        const scale = scales[pos];

        var score: f32 = 0;
        var k: u32 = 0;
        while (k < dim) : (k += 1) {
            score += @as(f32, @floatFromInt(key[k])) * scale * query[k];
        }

        scores[pos] = score;
    }
}

pub fn fusedDequantDotQ80(
    blocks: [*]const block_q8_0,
    n_blocks: u32,
    query_fp32: [*]const f32,
) f32 {
    var acc: f32 = 0;

    var b: u32 = 0;
    while (b < n_blocks) : (b += 1) {
        const scale = types.fp16_to_fp32(blocks[b].d);

        var i: u32 = 0;
        while (i < 32) : (i += 1) {
            acc += scale * @as(f32, @floatFromInt(blocks[b].qs[i])) * query_fp32[@as(usize, b) * 32 + i];
        }
    }

    return acc;
}

pub fn fusedDequantDotQ4K(
    blocks: [*]const block_q4_K,
    n_blocks: u32,
    query_fp32: [*]const f32,
) f32 {
    var acc: f32 = 0;

    var b: u32 = 0;
    while (b < n_blocks) : (b += 1) {
        const d_f32 = types.fp16_to_fp32(blocks[b].d);
        const dmin_f32 = types.fp16_to_fp32(blocks[b].dmin);
        const scales = blocks[b].scales;
        const qs = blocks[b].qs;

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

            const rs0 = d_f32 * @as(f32, @floatFromInt(sc0));
            const rm0 = dmin_f32 * @as(f32, @floatFromInt(mn0));
            const rs1 = d_f32 * @as(f32, @floatFromInt(sc1));
            const rm1 = dmin_f32 * @as(f32, @floatFromInt(mn1));

            const qoff = @as(usize, blk) * 32;
            const woff = @as(usize, blk) * 64;

            var j: u32 = 0;
            while (j < 16) : (j += 1) {
                const w0 = rs0 * @as(f32, @floatFromInt(qs[qoff + j] & 0x0F)) - rm0;
                acc += w0 * query_fp32[b * 256 + woff + j];
                const w1 = rs1 * @as(f32, @floatFromInt(qs[qoff + 16 + j] & 0x0F)) - rm1;
                acc += w1 * query_fp32[b * 256 + woff + 16 + j];
                const w2 = rs0 * @as(f32, @floatFromInt(qs[qoff + j] >> 4)) - rm0;
                acc += w2 * query_fp32[b * 256 + woff + 32 + j];
                const w3 = rs1 * @as(f32, @floatFromInt(qs[qoff + 16 + j] >> 4)) - rm1;
                acc += w3 * query_fp32[b * 256 + woff + 48 + j];
            }
        }
    }

    return acc;
}

pub fn dequantizeQ80ToFp16(src: [*]const u8, dst: [*]types.fp16_t, n_weights: usize) void {
    const n_blocks = n_weights / 32;
    const blocks_bytes: []align(@alignOf(block_q8_0)) const u8 = @alignCast(src[0 .. n_blocks * @sizeOf(block_q8_0)]);
    const blocks_slice = std.mem.bytesAsSlice(block_q8_0, blocks_bytes);
    var b: usize = 0;
    while (b < n_blocks) : (b += 1) {
        const block = blocks_slice[b];
        const d = types.fp16_to_fp32(block.d);
        var j: u32 = 0;
        while (j < 32) : (j += 1) {
            const val = d * @as(f32, @floatFromInt(block.qs[j]));
            dst[b * 32 + j] = types.fp32_to_fp16(val);
        }
    }
}

pub fn dequantizeQ4KToFp16(src: [*]const u8, dst: [*]types.fp16_t, n_weights: usize) void {
    const n_blocks = n_weights / 256;
    const blocks_bytes: []align(@alignOf(block_q4_K)) const u8 = @alignCast(src[0 .. n_blocks * @sizeOf(block_q4_K)]);
    const blocks_slice = std.mem.bytesAsSlice(block_q4_K, blocks_bytes);
    var b: usize = 0;
    while (b < n_blocks) : (b += 1) {
        const block = blocks_slice[b];
        const d_f32 = types.fp16_to_fp32(block.d);
        const dmin_f32 = types.fp16_to_fp32(block.dmin);

        var blk: u32 = 0;
        while (blk < 4) : (blk += 1) {
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

            const qoff = @as(usize, blk) * 32;
            const woff = @as(usize, blk) * 64;

            var j: u32 = 0;
            while (j < 16) : (j += 1) {
                dst[b * 256 + woff + j] = types.fp32_to_fp16(rs0 * @as(f32, @floatFromInt(block.qs[qoff + j] & 0x0F)) - rm0);
                dst[b * 256 + woff + 16 + j] = types.fp32_to_fp16(rs1 * @as(f32, @floatFromInt(block.qs[qoff + 16 + j] & 0x0F)) - rm1);
                dst[b * 256 + woff + 32 + j] = types.fp32_to_fp16(rs0 * @as(f32, @floatFromInt(block.qs[qoff + j] >> 4)) - rm0);
                dst[b * 256 + woff + 48 + j] = types.fp32_to_fp16(rs1 * @as(f32, @floatFromInt(block.qs[qoff + 16 + j] >> 4)) - rm1);
            }
        }
    }
}

pub fn dequantizeQ2KToFp16(src: [*]const u8, dst: [*]types.fp16_t, n_weights: usize) void {
    const n_blocks = n_weights / 256;
    var b: usize = 0;
    while (b < n_blocks) : (b += 1) {
        const block = src + b * 84;
        const d_all = types.fp16_to_fp32(std.mem.readInt(u16, block[0..2], .little));
        const m_all = types.fp16_to_fp32(std.mem.readInt(u16, block[2..4], .little));
        const scales = block[4..20];
        const qs = block[20..84];

        var j: usize = 0;
        while (j < 256) : (j += 1) {
            const sb = j / 8;
            const s = sb / 2;
            const scale = if (sb % 2 == 0)
                d_all * @as(f32, @floatFromInt(scales[s] & 0x0F))
            else
                d_all * @as(f32, @floatFromInt(scales[s + 8] & 0x0F));
            const min_val = if (sb % 2 == 0)
                m_all * @as(f32, @floatFromInt(scales[s] >> 4))
            else
                m_all * @as(f32, @floatFromInt(scales[s + 8] >> 4));
            const byte_idx = j / 4;
            const bit_off: u3 = @intCast((j % 4) * 2);
            const q = (qs[byte_idx] >> bit_off) & 0x03;
            dst[b * 256 + j] = types.fp32_to_fp16(scale * @as(f32, @floatFromInt(q)) - min_val);
        }
    }
}

pub fn dequantizeQ6KToFp16(src: [*]const u8, dst: [*]types.fp16_t, n_weights: usize) void {
    const n_blocks = n_weights / 256;
    const blocks_bytes: []align(@alignOf(block_q6_K)) const u8 = @alignCast(src[0 .. n_blocks * @sizeOf(block_q6_K)]);
    const blocks_slice = std.mem.bytesAsSlice(block_q6_K, blocks_bytes);

    var b: usize = 0;
    while (b < n_blocks) : (b += 1) {
        const block = blocks_slice[b];
        const d = types.fp16_to_fp32(block.d);

        var n: usize = 0;
        while (n < 256) : (n += 128) {
            var l: usize = 0;
            while (l < 32) : (l += 1) {
                const is = l / 16;
                const qh = block.qh[n / 4 + l];
                const q1: i8 = @intCast((block.ql[n / 2 + l] & 0x0F) | (((qh >> 0) & 0x03) << 4));
                const q2: i8 = @intCast((block.ql[n / 2 + 32 + l] & 0x0F) | (((qh >> 2) & 0x03) << 4));
                const q3: i8 = @intCast((block.ql[n / 2 + l] >> 4) | (((qh >> 4) & 0x03) << 4));
                const q4: i8 = @intCast((block.ql[n / 2 + 32 + l] >> 4) | (((qh >> 6) & 0x03) << 4));

                dst[b * 256 + n + l + 0] = types.fp32_to_fp16(d * @as(f32, @floatFromInt(block.scales[n / 16 + is + 0])) * @as(f32, @floatFromInt(q1 - 32)));
                dst[b * 256 + n + l + 32] = types.fp32_to_fp16(d * @as(f32, @floatFromInt(block.scales[n / 16 + is + 2])) * @as(f32, @floatFromInt(q2 - 32)));
                dst[b * 256 + n + l + 64] = types.fp32_to_fp16(d * @as(f32, @floatFromInt(block.scales[n / 16 + is + 4])) * @as(f32, @floatFromInt(q3 - 32)));
                dst[b * 256 + n + l + 96] = types.fp32_to_fp16(d * @as(f32, @floatFromInt(block.scales[n / 16 + is + 6])) * @as(f32, @floatFromInt(q4 - 32)));
            }
        }
    }
}

pub fn dequantizeQ5KToFp16(src: [*]const u8, dst: [*]types.fp16_t, n_weights: usize) void {
    const n_blocks = n_weights / 256;
    const blocks_bytes: []align(@alignOf(block_q5_K)) const u8 = @alignCast(src[0 .. n_blocks * @sizeOf(block_q5_K)]);
    const blocks_slice = std.mem.bytesAsSlice(block_q5_K, blocks_bytes);

    for (blocks_slice, 0..) |block, b| {
        const d = types.fp16_to_fp32(block.d);
        const min_val = types.fp16_to_fp32(block.dmin);

        var is: usize = 0;
        var mask1: u8 = 1;
        var mask2: u8 = 2;
        var ql_off: usize = 0;
        var out_off: usize = 0;
        while (out_off < 256) : (out_off += 64) {
            const sm0 = getScaleMinK4(is + 0, block.scales);
            const d1 = d * @as(f32, @floatFromInt(sm0.sc));
            const m1 = min_val * @as(f32, @floatFromInt(sm0.m));
            const sm1 = getScaleMinK4(is + 1, block.scales);
            const d2 = d * @as(f32, @floatFromInt(sm1.sc));
            const m2 = min_val * @as(f32, @floatFromInt(sm1.m));

            for (0..32) |l| {
                const q = block.qs[ql_off + l];
                const hi1: u8 = if ((block.qh[l] & mask1) != 0) 16 else 0;
                const hi2: u8 = if ((block.qh[l] & mask2) != 0) 16 else 0;
                dst[b * 256 + out_off + l] = types.fp32_to_fp16(d1 * @as(f32, @floatFromInt((q & 0x0F) + hi1)) - m1);
                dst[b * 256 + out_off + 32 + l] = types.fp32_to_fp16(d2 * @as(f32, @floatFromInt((q >> 4) + hi2)) - m2);
            }

            ql_off += 32;
            is += 2;
            mask1 <<= 2;
            mask2 <<= 2;
        }
    }
}

pub fn dequantizeIQ4XSToFp16(src: [*]const u8, dst: [*]types.fp16_t, n_weights: usize) void {
    const n_blocks = n_weights / 256;
    const blocks_bytes: []align(@alignOf(block_iq4_xs)) const u8 = @alignCast(src[0 .. n_blocks * @sizeOf(block_iq4_xs)]);
    const blocks_slice = std.mem.bytesAsSlice(block_iq4_xs, blocks_bytes);

    for (blocks_slice, 0..) |block, b| {
        const d = types.fp16_to_fp32(block.d);
        var qs_off: usize = 0;
        var out_off: usize = 0;

        for (0..8) |ib| {
            const ls_low = (block.scales_l[ib / 2] >> @as(u3, @intCast(4 * (ib % 2)))) & 0x0F;
            const ls_high = ((block.scales_h >> @as(u4, @intCast(2 * ib))) & 0x03) << 4;
            const ls: i32 = @as(i32, ls_low | ls_high);
            const dl = d * @as(f32, @floatFromInt(ls - 32));

            for (0..16) |j| {
                const q = block.qs[qs_off + j];
                dst[b * 256 + out_off + j] = types.fp32_to_fp16(dl * @as(f32, @floatFromInt(kvalues_iq4nl[q & 0x0F])));
                dst[b * 256 + out_off + 16 + j] = types.fp32_to_fp16(dl * @as(f32, @floatFromInt(kvalues_iq4nl[q >> 4])));
            }

            qs_off += 16;
            out_off += 32;
        }
    }
}

pub fn dequantizeIQ2XSToFp16(src: [*]const u8, dst: [*]types.fp16_t, n_weights: usize) void {
    const n_blocks = n_weights / 256;
    const blocks_bytes: []align(@alignOf(block_iq2_xs)) const u8 = @alignCast(src[0 .. n_blocks * @sizeOf(block_iq2_xs)]);
    const blocks_slice = std.mem.bytesAsSlice(block_iq2_xs, blocks_bytes);

    var b: usize = 0;
    while (b < n_blocks) : (b += 1) {
        const block = blocks_slice[b];
        const d = types.fp16_to_fp32(block.d);

        var ib32: usize = 0;
        while (ib32 < 8) : (ib32 += 1) {
            const db0 = d * (0.5 + @as(f32, @floatFromInt(block.scales[ib32] & 0x0F))) * 0.25;
            const db1 = d * (0.5 + @as(f32, @floatFromInt(block.scales[ib32] >> 4))) * 0.25;

            var l: usize = 0;
            while (l < 4) : (l += 1) {
                const q = block.qs[4 * ib32 + l];
                const grid = iq2_tables.iq2xs_grid[q & 0x01FF];
                const signs = iq2_tables.ksigns_iq2xs[q >> 9];
                const scale = if (l < 2) db0 else db1;

                var j: usize = 0;
                while (j < 8) : (j += 1) {
                    const sign: f32 = if ((signs & iq2_tables.kmask_iq2xs[j]) != 0) -1.0 else 1.0;
                    dst[b * 256 + ib32 * 32 + l * 8 + j] = types.fp32_to_fp16(scale * @as(f32, @floatFromInt(grid[j])) * sign);
                }
            }
        }
    }
}

test "block sizes match C++" {
    try std.testing.expectEqual(@sizeOf(block_q8_0), 34);
    try std.testing.expectEqual(@sizeOf(block_q4_K), 144);
    try std.testing.expectEqual(@as(usize, 176), @sizeOf(block_q5_K));
    try std.testing.expectEqual(@as(usize, 210), @sizeOf(block_q6_K));
    try std.testing.expectEqual(@as(usize, 74), @sizeOf(block_iq2_xs));
    try std.testing.expectEqual(@as(usize, 136), @sizeOf(block_iq4_xs));
}

test "Q6_K dequant basic shape" {
    var raw: [210]u8 = [_]u8{0} ** 210;
    var dst: [256]types.fp16_t = undefined;
    dequantizeQ6KToFp16(&raw, &dst, dst.len);

    const first = types.fp16_to_fp32(dst[0]);
    const second = types.fp16_to_fp32(dst[32]);
    try std.testing.expectApproxEqAbs(@as(f32, 0), first, 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0), second, 0.001);
}

test "IQ2_XS dequant basic shape" {
    var raw: [74]u8 = [_]u8{0} ** 74;
    var dst: [256]types.fp16_t = undefined;
    dequantizeIQ2XSToFp16(&raw, &dst, dst.len);

    try std.testing.expectApproxEqAbs(@as(f32, 0), types.fp16_to_fp32(dst[0]), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0), types.fp16_to_fp32(dst[255]), 0.001);
}

test "Q5_K dequant basic shape" {
    var raw: [176]u8 = [_]u8{0} ** 176;
    var dst: [256]types.fp16_t = undefined;
    dequantizeQ5KToFp16(&raw, &dst, dst.len);

    try std.testing.expectApproxEqAbs(@as(f32, 0), types.fp16_to_fp32(dst[0]), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0), types.fp16_to_fp32(dst[255]), 0.001);
}

test "IQ4_XS dequant basic shape" {
    var raw: [136]u8 = [_]u8{0} ** 136;
    var dst: [256]types.fp16_t = undefined;
    dequantizeIQ4XSToFp16(&raw, &dst, dst.len);

    try std.testing.expectApproxEqAbs(@as(f32, 0), types.fp16_to_fp32(dst[0]), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0), types.fp16_to_fp32(dst[255]), 0.001);
}

test "Q4_K dequant matches canonical scale packing" {
    var block = std.mem.zeroes(block_q4_K);
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

    var query: [256]f32 = undefined;
    for (0..query.len) |i| query[i] = @as(f32, @floatFromInt((i % 11) + 1)) * 0.01;

    const blocks = [_]block_q4_K{block};
    const fused = fusedDequantDotQ4K(&blocks, 1, &query);

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
            expected += (rs0 * @as(f32, @floatFromInt(block.qs[qoff + j] & 0x0F)) - rm0) * query[woff + j];
            expected += (rs1 * @as(f32, @floatFromInt(block.qs[qoff + 16 + j] & 0x0F)) - rm1) * query[woff + 16 + j];
            expected += (rs0 * @as(f32, @floatFromInt(block.qs[qoff + j] >> 4)) - rm0) * query[woff + 32 + j];
            expected += (rs1 * @as(f32, @floatFromInt(block.qs[qoff + 16 + j] >> 4)) - rm1) * query[woff + 48 + j];
        }
    }

    try std.testing.expectApproxEqAbs(expected, fused, 0.01);
}
