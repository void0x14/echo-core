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
            out_row[i] = @intCast(v);
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
            score += @as(f32, @intCast(key[k])) * scale * query[k];
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
            acc += scale * @as(f32, blocks[b].qs[i]) * query_fp32[@as(usize, b) * 32 + i];
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

            const sc0 = scales[js] & 63;
            const mn0 = scales[js + 4] & 63;
            const sc1 = scales[js + 1] & 63;
            const mn1 = scales[js + 1 + 4] & 63;

            const rs0 = d_f32 * @as(f32, sc0);
            const rm0 = dmin_f32 * @as(f32, mn0);
            const rs1 = d_f32 * @as(f32, sc1);
            const rm1 = dmin_f32 * @as(f32, mn1);

            const qoff = @as(usize, blk) * 32;
            const woff = @as(usize, blk) * 64;

            var j: u32 = 0;
            while (j < 16) : (j += 1) {
                const w0 = rs0 * @as(f32, qs[qoff + j] & 0x0F) - rm0;
                acc += w0 * query_fp32[b * 256 + woff + j];
                const w1 = rs1 * @as(f32, qs[qoff + 16 + j] & 0x0F) - rm1;
                acc += w1 * query_fp32[b * 256 + woff + 16 + j];
                const w2 = rs0 * @as(f32, qs[qoff + j] >> 4) - rm0;
                acc += w2 * query_fp32[b * 256 + woff + 32 + j];
                const w3 = rs1 * @as(f32, qs[qoff + 16 + j] >> 4) - rm1;
                acc += w3 * query_fp32[b * 256 + woff + 48 + j];
            }
        }
    }

    return acc;
}

pub fn dequantizeQ80ToFp16(src: [*]const u8, dst: [*]types.fp16_t, n_weights: usize) void {
    const n_blocks = n_weights / 32;
    var b: usize = 0;
    while (b < n_blocks) : (b += 1) {
        const block = @as([*]const block_q8_0, @ptrCast(src))[b];
        const d = types.fp16_to_fp32(block.d);
        var j: u32 = 0;
        while (j < 32) : (j += 1) {
            const val = d * @as(f32, block.qs[j]);
            dst[b * 32 + j] = types.fp32_to_fp16(val);
        }
    }
}

pub fn dequantizeQ4KToFp16(src: [*]const u8, dst: [*]types.fp16_t, n_weights: usize) void {
    const n_blocks = n_weights / 256;
    var b: usize = 0;
    while (b < n_blocks) : (b += 1) {
        const block = @as([*]const block_q4_K, @ptrCast(src))[b];
        const d_f32 = types.fp16_to_fp32(block.d);
        const dmin_f32 = types.fp16_to_fp32(block.dmin);

        var blk: u32 = 0;
        while (blk < 4) : (blk += 1) {
            const js = blk * 2;
            const sc0 = block.scales[js] & 63;
            const mn0 = block.scales[js + 4] & 63;
            const sc1 = block.scales[js + 1] & 63;
            const mn1 = block.scales[js + 1 + 4] & 63;

            const rs0 = d_f32 * @as(f32, sc0);
            const rm0 = dmin_f32 * @as(f32, mn0);
            const rs1 = d_f32 * @as(f32, sc1);
            const rm1 = dmin_f32 * @as(f32, mn1);

            const qoff = @as(usize, blk) * 32;
            const woff = @as(usize, blk) * 64;

            var j: u32 = 0;
            while (j < 16) : (j += 1) {
                dst[b * 256 + woff + j] = types.fp32_to_fp16(rs0 * @as(f32, block.qs[qoff + j] & 0x0F) - rm0);
                dst[b * 256 + woff + 16 + j] = types.fp32_to_fp16(rs1 * @as(f32, block.qs[qoff + 16 + j] & 0x0F) - rm1);
                dst[b * 256 + woff + 32 + j] = types.fp32_to_fp16(rs0 * @as(f32, block.qs[qoff + j] >> 4) - rm0);
                dst[b * 256 + woff + 48 + j] = types.fp32_to_fp16(rs1 * @as(f32, block.qs[qoff + 16 + j] >> 4) - rm1);
            }
        }
    }
}

test "block sizes match C++" {
    try std.testing.expectEqual(@sizeOf(block_q8_0), 34);
    try std.testing.expectEqual(@sizeOf(block_q4_K), 144);
}
