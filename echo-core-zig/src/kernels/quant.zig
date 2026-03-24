const std = @import("std");
const types = @import("../core/types.zig");

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

test "block sizes match C++" {
    try std.testing.expectEqual(@sizeOf(block_q8_0), 34);
    try std.testing.expectEqual(@sizeOf(block_q4_K), 144);
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
