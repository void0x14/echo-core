const std = @import("std");
const types = @import("../core/types.zig");
const config = @import("core_config");
const quant = @import("quant.zig");
const avx2_q6k = @import("avx2_q6k.zig");
const avx2_q4k = @import("avx2_q4k.zig");
const avx2_q5k = @import("avx2_q5k.zig");
const int_dot = @import("int_dot.zig");
const gguf = @import("../gguf/reader.zig");


pub fn matvecFp16Vec(
    W: [*]const u8,
    x: [*]const f32,
    y: [*]f32,
    M: u32,
    K: u32,
) void {
    const W_fp16: [*]const types.fp16_t = @ptrCast(@alignCast(W));
    var m: u32 = 0;
    while (m < M) : (m += 1) {
        const W_row = W_fp16 + @as(usize, m) * K;
        var acc0: @Vector(8, f32) = @splat(0);
        var acc1: @Vector(8, f32) = @splat(0);
        var acc2: @Vector(8, f32) = @splat(0);
        var acc3: @Vector(8, f32) = @splat(0);
        var k: u32 = 0;
        while (k + 32 <= K) : (k += 32) {
            @prefetch(@as([*]const u8, @ptrCast(W_row + k + 256)), .{ .rw = .read, .locality = 2, .cache = .data });
            inline for (0..4) |q| {
                const wh: @Vector(8, f16) = @as(*align(1) const [8]f16, @ptrCast(W_row + k + q * 8)).*;
                const wf: @Vector(8, f32) = @floatCast(wh);
                const xv: @Vector(8, f32) = @as(*align(1) const [8]f32, @ptrCast(x + k + q * 8)).*;
                switch (q) {
                    0 => acc0 = @mulAdd(@Vector(8, f32), wf, xv, acc0),
                    1 => acc1 = @mulAdd(@Vector(8, f32), wf, xv, acc1),
                    2 => acc2 = @mulAdd(@Vector(8, f32), wf, xv, acc2),
                    3 => acc3 = @mulAdd(@Vector(8, f32), wf, xv, acc3),
                    else => unreachable,
                }
            }
        }
        var acc: f32 = @reduce(.Add, acc0 + acc1 + acc2 + acc3);
        while (k < K) : (k += 1) {
            acc += types.fp16_to_fp32(W_row[k]) * x[k];
        }
        y[m] += acc;
    }
}

pub fn matvecF32Vec(
    W: [*]const u8,
    x: [*]const f32,
    y: [*]f32,
    M: u32,
    K: u32,
) void {
    const W_f32: [*]const f32 = @ptrCast(@alignCast(W));
    var m: u32 = 0;
    while (m < M) : (m += 1) {
        const W_row = W_f32 + @as(usize, m) * K;
        var acc0: @Vector(8, f32) = @splat(0);
        var acc1: @Vector(8, f32) = @splat(0);
        var acc2: @Vector(8, f32) = @splat(0);
        var acc3: @Vector(8, f32) = @splat(0);
        var k: u32 = 0;
        while (k + 32 <= K) : (k += 32) {
            inline for (0..4) |q| {
            @prefetch(@as([*]const u8, @ptrCast(W_row + k + 256)), .{ .rw = .read, .locality = 2, .cache = .data });
                const wf: @Vector(8, f32) = @as(*align(1) const [8]f32, @ptrCast(W_row + k + q * 8)).*;
                const xv: @Vector(8, f32) = @as(*align(1) const [8]f32, @ptrCast(x + k + q * 8)).*;
                switch (q) {
                    0 => acc0 = @mulAdd(@Vector(8, f32), wf, xv, acc0),
                    1 => acc1 = @mulAdd(@Vector(8, f32), wf, xv, acc1),
                    2 => acc2 = @mulAdd(@Vector(8, f32), wf, xv, acc2),
                    3 => acc3 = @mulAdd(@Vector(8, f32), wf, xv, acc3),
                    else => unreachable,
                }
            }
        }
        var acc: f32 = @reduce(.Add, acc0 + acc1 + acc2 + acc3);
        while (k < K) : (k += 1) {
            acc += W_row[k] * x[k];
        }
        y[m] += acc;
    }
}
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
        .f16 => matvecFp16Vec(W, x, y, M, K),
        .f32 => matvecF32Vec(W, x, y, M, K),
        .q8_0 => if (K % 32 == 0 and K <= 16384) matvecQ80Int(W, x, y, M, K) else matvecQ80(W, x, y, M, K),
        .q4_k => if (K >= 256) int_dot.matvecQ4K_int(W, x, y, M, K) else matvecGenericDequant(W, x, y, M, K, dtype),
        .q5_k => if (K >= 256) avx2_q5k.matvecQ5K_avx2(W, x, y, M, K) else matvecGenericDequant(W, x, y, M, K, dtype),
        .q2_k => if (K >= 256) matvecQ2K(W, x, y, M, K) else matvecGenericDequant(W, x, y, M, K, dtype),
        .q6_k => if (K >= 256) avx2_q6k.matvecQ6K_avx2(W, x, y, M, K) else matvecGenericDequant(W, x, y, M, K, dtype),
        .q3_k => {
            std.debug.print("WARN: Q3_K matvec not optimized, using generic dequant path\n", .{});
            matvecGenericDequant(W, x, y, M, K, dtype);
        },
        .iq2_xs, .iq4_xs => {
            std.debug.print("WARN: {s} matvec not optimized, using generic dequant path\n", .{@tagName(dtype)});
            matvecGenericDequant(W, x, y, M, K, dtype);
        },
        else => {
            std.debug.print("WARN: unsupported dtype {s}, falling back to FP16 (may produce incorrect results)\n", .{@tagName(dtype)});
            matvecFp16Fp32(TILE_K, TILE_M, W, x, y, M, K);
        },
    }
}

pub fn quantizeXToQ80(x: [*]const f32, q8: [*]quant.block_q8_0, K: usize) void {
    const blocks_per_row = K / 32;
    var b: usize = 0;
    while (b < blocks_per_row) : (b += 1) {
        const xb = x + b * 32;
        var amax: f32 = 0;
        for (0..32) |j| {
            const a = @abs(xb[j]);
            if (a > amax) amax = a;
        }
        if (amax < 1e-30) {
            q8[b].d = 0;
            @memset(&q8[b].qs, 0);
            continue;
        }
        const d = amax / 127.0;
        q8[b].d = types.fp32_to_fp16(d);
        const inv = 127.0 / amax;
        for (0..32) |j| {
            const v: i32 = @intFromFloat(@round(xb[j] * inv));
            q8[b].qs[j] = @intCast(@max(-127, @min(127, v)));
        }
    }
}

pub fn matvecQ80IntPre(
    blocks: [*]const u8,
    q8_x: [*]const quant.block_q8_0,
    y: [*]f32,
    M: u32,
    K: u32,
) void {
    const blocks_per_row = K / 32;
    const block_stride: usize = 34;

    var m: u32 = 0;
    while (m < M) : (m += 1) {
        const row_ptr = blocks + @as(usize, m) * blocks_per_row * block_stride;
        var s0: f32 = 0;
        var s1: f32 = 0;
        var s2: f32 = 0;
        var s3: f32 = 0;
        var b: usize = 0;
        while (b + 4 <= blocks_per_row) : (b += 4) {
            inline for (0..4) |q| {
                const bp = row_ptr + (b + q) * block_stride;
                @prefetch(bp + 16 * block_stride, .{ .rw = .read, .locality = 2, .cache = .data });
                const d_w = types.fp16_to_fp32(std.mem.readInt(u16, bp[0..2], .little));
                const d_x = types.fp16_to_fp32(q8_x[b + q].d);
                const wv: @Vector(32, i8) = @as(*align(1) const [32]i8, @ptrCast(bp + 2)).*;
                const xv: @Vector(32, i8) = q8_x[b + q].qs;
                const prod: @Vector(32, i32) = @as(@Vector(32, i32), @intCast(wv)) * @as(@Vector(32, i32), @intCast(xv));
                const term = (d_w * d_x) * @as(f32, @floatFromInt(@reduce(.Add, prod)));
                switch (q) {
                    0 => s0 += term,
                    1 => s1 += term,
                    2 => s2 += term,
                    3 => s3 += term,
                    else => unreachable,
                }
            }
        }
        while (b < blocks_per_row) : (b += 1) {
            const bp = row_ptr + b * block_stride;
            const d_w = types.fp16_to_fp32(std.mem.readInt(u16, bp[0..2], .little));
            const d_x = types.fp16_to_fp32(q8_x[b].d);
            const wv: @Vector(32, i8) = @as(*align(1) const [32]i8, @ptrCast(bp + 2)).*;
            const xv: @Vector(32, i8) = q8_x[b].qs;
            const prod: @Vector(32, i32) = @as(@Vector(32, i32), @intCast(wv)) * @as(@Vector(32, i32), @intCast(xv));
            s0 += (d_w * d_x) * @as(f32, @floatFromInt(@reduce(.Add, prod)));
        }
        y[m] += (s0 + s1) + (s2 + s3);
    }
}

fn matvecQ80Int(
    blocks: [*]const u8,
    x: [*]const f32,
    y: [*]f32,
    M: u32,
    K: u32,
) void {
    const blocks_per_row = K / 32;
    var q8_buf: [512]quant.block_q8_0 = undefined;
    if (blocks_per_row <= 512) {
        quantizeXToQ80(x, &q8_buf, K);
        matvecQ80IntPre(blocks, &q8_buf, y, M, K);
    } else {
        matvecQ80(blocks, x, y, M, K);
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

fn matvecGenericDequant(
    blocks: [*]const u8,
    x: [*]const f32,
    y: [*]f32,
    M: u32,
    K: u32,
    dtype: gguf.GGMLType,
) void {
    var fp16_buf: [8192]types.fp16_t = undefined;
    const batch_size = @min(@as(usize, M), fp16_buf.len / @as(usize, K));
    if (batch_size == 0) return;
    var m: u32 = 0;
    while (m < M) {
        const batch = @min(@as(usize, M - m), batch_size);
        for (0..batch) |i| {
            const row_fp16 = fp16_buf[i * K .. (i + 1) * K];
            const row_ptr = blocks + @as(usize, m + i) * quant.rowQuantizedBytes(K, dtype);
            quant.dequantizeRow(row_ptr, @ptrCast(row_fp16.ptr), K, dtype);
            var sum: f32 = 0;
            for (0..K) |j| {
                sum += types.fp16_to_fp32(row_fp16[j]) * x[j];
            }
            y[m + i] += sum;
        }
        m += @intCast(batch);
    }
}

pub fn matvecQ5K(
    blocks: [*]const u8,
    x: [*]const f32,
    y: [*]f32,
    M: u32,
    K: u32,
) void {
    const blocks_per_row = K / 256;
    const block_stride = 176;

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
            const qh = bp[16..];
            const qs = bp[48..];
            const x_blk = x + b * 256;

            var mask1: u8 = 1;
            var mask2: u8 = 2;
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

                for (0..32) |l| {
                    const q = qs[ql_off + l];
                    const hi1: u8 = if ((qh[l] & mask1) != 0) 16 else 0;
                    const hi2: u8 = if ((qh[l] & mask2) != 0) 16 else 0;
                    sum += (rs0 * @as(f32, @floatFromInt((q & 0x0F) + hi1)) - rm0) * x_blk[woff + l];
                    sum += (rs1 * @as(f32, @floatFromInt((q >> 4) + hi2)) - rm1) * x_blk[woff + 32 + l];
                }

                ql_off += 32;
                mask1 <<= 2;
                mask2 <<= 2;
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
 fn matmatQ80Int(blocks: [*]const u8, X: [*]const f32, Y: [*]f32, M: u32, K: u32, T: u32, ys: usize, yo: usize) void {
    const blocks_per_row: usize = @intCast(K / 32);
    const block_stride: usize = 34;
    const Ks: usize = @intCast(K);
    const Ts: usize = @intCast(T);
    const alloc = std.heap.smp_allocator;
    const xq = alloc.alloc(i8, @max(Ts * Ks, 1)) catch {
        for (0..Ts) |t| matvecQ80Int(blocks, X + t * Ks, Y + t * ys + yo, M, K);
        return;
    };
    defer alloc.free(xq);
    const xs = alloc.alloc(f32, @max(Ts * blocks_per_row, 1)) catch {
        for (0..Ts) |t| matvecQ80Int(blocks, X + t * Ks, Y + t * ys + yo, M, K);
        return;
    };
    defer alloc.free(xs);
    for (0..Ts) |t| {
        const x = X + t * Ks;
        var b: usize = 0;
        while (b < blocks_per_row) : (b += 1) {
            const xb = x + b * 32;
            var amax: f32 = 0;
            for (0..32) |j| { const a = @abs(xb[j]); if (a > amax) amax = a; }
            if (amax < 1e-30) { xs[t * blocks_per_row + b] = 0; for (0..32) |j| xq[t * Ks + b * 32 + j] = 0; continue; }
            xs[t * blocks_per_row + b] = amax / 127.0;
            const inv = 127.0 / amax;
            for (0..32) |j| { const v: i32 = @intFromFloat(@round(xb[j] * inv)); xq[t * Ks + b * 32 + j] = @intCast(@max(-127, @min(127, v))); }
        }
    }
    var m: u32 = 0;
    while (m < M) : (m += 1) {
        const row_ptr = blocks + @as(usize, m) * blocks_per_row * block_stride;
        var tb: usize = 0;
        while (tb < Ts) : (tb += 4) {
            const tn = @min(@as(usize, 4), Ts - tb);
            var sums = [_]f32{ 0, 0, 0, 0 };
            var b: usize = 0;
            while (b < blocks_per_row) : (b += 1) {
                const bp = row_ptr + b * block_stride;
                @prefetch(bp + 16 * block_stride, .{ .rw = .read, .locality = 2, .cache = .data });
                const d = types.fp16_to_fp32(std.mem.readInt(u16, bp[0..2], .little));
                const wv: @Vector(32, i8) = @as(*align(1) const [32]i8, @ptrCast(bp + 2)).*;
                const w32: @Vector(32, i32) = @intCast(wv);
                inline for (0..4) |tt| {
                    if (tt < tn) {
                        const t = tb + tt;
                        const xv: @Vector(32, i8) = xq[t * Ks + b * 32 ..][0..32].*;
                        const prod: @Vector(32, i32) = w32 * @as(@Vector(32, i32), @intCast(xv));
                        sums[tt] += d * xs[t * blocks_per_row + b] * @as(f32, @floatFromInt(@reduce(.Add, prod)));
                    }
                }
            }
 inline for (0..4) |tt| { if (tt < tn) Y[(tb + tt) * ys + yo + m] = sums[tt]; }
        }
    }
}

 fn matmatFloat(comptime is_f16: bool, W: [*]const u8, X: [*]const f32, Y: [*]f32, M: u32, K: u32, T: u32, ys: usize, yo: usize) void {
    const Ks: usize = @intCast(K);
    const Ts: usize = @intCast(T);
    var m: u32 = 0;
    while (m < M) : (m += 1) {
        var tb: usize = 0;
        while (tb < Ts) : (tb += 8) {
            const tn = @min(@as(usize, 8), Ts - tb);
            var acc = [_]@Vector(8, f32){ @splat(0), @splat(0), @splat(0), @splat(0), @splat(0), @splat(0), @splat(0), @splat(0) };
            var k: usize = 0;
            while (k + 8 <= Ks) : (k += 8) {
                var wf: @Vector(8, f32) = undefined;
                if (is_f16) {
                    const wh: @Vector(8, f16) = @as(*align(1) const [8]f16, @ptrCast(@as([*]const types.fp16_t, @ptrCast(@alignCast(W))) + @as(usize, m) * Ks + k)).*;
                    wf = @floatCast(wh);
                } else {
                    wf = @as(*align(1) const [8]f32, @ptrCast(@as([*]const f32, @ptrCast(@alignCast(W))) + @as(usize, m) * Ks + k)).*;
                }
                inline for (0..8) |tt| {
                    if (tt < tn) {
                        const xv: @Vector(8, f32) = @as(*align(1) const [8]f32, @ptrCast(X + (tb + tt) * Ks + k)).*;
                        acc[tt] = @mulAdd(@Vector(8, f32), wf, xv, acc[tt]);
                    }
                }
            }
            inline for (0..8) |tt| {
                if (tt < tn) {
                    var s = @reduce(.Add, acc[tt]);
                    var kk = k;
                    while (kk < Ks) : (kk += 1) {
                        const wv = if (is_f16) types.fp16_to_fp32(@as([*]const types.fp16_t, @ptrCast(@alignCast(W)))[@as(usize, m) * Ks + kk]) else @as([*]const f32, @ptrCast(@alignCast(W)))[@as(usize, m) * Ks + kk];
                        s += wv * X[(tb + tt) * Ks + kk];
                    }
 Y[(tb + tt) * ys + yo + m] = s;
                }
            }
        }
    }
}

pub fn matmatDispatchQuant(W: [*]const u8, X: [*]const f32, Y: [*]f32, M: u32, K: u32, T: u32, dtype: gguf.GGMLType) void {
    matmatDispatchQuantStrided(W, X, Y, M, K, T, dtype, @intCast(M), 0);
}

pub fn matmatDispatchQuantStrided(W: [*]const u8, X: [*]const f32, Y: [*]f32, M: u32, K: u32, T: u32, dtype: gguf.GGMLType, ys: usize, yo: usize) void {
    if (T == 0) return;
    if (T == 1) { @memset(Y[yo .. yo + @as(usize, @intCast(M))], 0); matvecDispatchQuant(0, 0, W, X, Y + yo, M, K, dtype); return; }
    switch (dtype) {
        .f16 => matmatFloat(true, W, X, Y, M, K, T, ys, yo),
        .f32 => matmatFloat(false, W, X, Y, M, K, T, ys, yo),
        .q8_0 => if (K % 32 == 0) matmatQ80Float(W, X, Y, M, K, T, ys, yo) else { for (0..T) |t| { @memset(Y[t * ys + yo ..][0..@intCast(M)], 0); matvecDispatchQuant(0, 0, W, X + t * @as(usize, K), Y + t * ys + yo, M, K, dtype); } },
        .q4_k => if (K >= 256) int_dot.matmatQ4K_int(W, X, Y, M, K, T, ys, yo) else { for (0..T) |t| { @memset(Y[t * ys + yo ..][0..@intCast(M)], 0); matvecDispatchQuant(0, 0, W, X + t * @as(usize, K), Y + t * ys + yo, M, K, dtype); } },
        .q5_k => if (K >= 256) avx2_q5k.matmatQ5K_avx2(W, X, Y, M, K, T, ys, yo) else { for (0..T) |t| { @memset(Y[t * ys + yo ..][0..@intCast(M)], 0); matvecDispatchQuant(0, 0, W, X + t * @as(usize, K), Y + t * ys + yo, M, K, dtype); } },
        .q6_k => if (K >= 256) avx2_q6k.matmatQ6K_avx2(W, X, Y, M, K, T, ys, yo) else { for (0..T) |t| { @memset(Y[t * ys + yo ..][0..@intCast(M)], 0); matvecDispatchQuant(0, 0, W, X + t * @as(usize, K), Y + t * ys + yo, M, K, dtype); } },
        else => { for (0..T) |t| { @memset(Y[t * ys + yo ..][0..@intCast(M)], 0); matvecDispatchQuant(0, 0, W, X + t * @as(usize, K), Y + t * ys + yo, M, K, dtype); } },
    }
}
fn matmatQ80Float(blocks: [*]const u8, X: [*]const f32, Y: [*]f32, M: u32, K: u32, T: u32, ys: usize, yo: usize) void {
    const blocks_per_row: usize = @intCast(K / 32);
    const block_stride: usize = 34;
    const Ks: usize = @intCast(K);
    const Ts: usize = @intCast(T);
    var m: u32 = 0;
    while (m < M) : (m += 1) {
        const row_ptr = blocks + @as(usize, m) * blocks_per_row * block_stride;
        var tb: usize = 0;
        while (tb + 8 <= Ts) : (tb += 8) {
            var acc = [_]@Vector(8, f32){ @splat(0), @splat(0), @splat(0), @splat(0), @splat(0), @splat(0), @splat(0), @splat(0) };
            var b: usize = 0;
            while (b < blocks_per_row) : (b += 1) {
                const bp = row_ptr + b * block_stride;
                @prefetch(bp + 16 * block_stride, .{ .rw = .read, .locality = 2, .cache = .data });
                const d: @Vector(8, f32) = @splat(types.fp16_to_fp32(std.mem.readInt(u16, bp[0..2], .little)));
                inline for (0..4) |q| {
                    const wv8: @Vector(8, i8) = @as(*align(1) const [8]i8, @ptrCast(bp + 2 + q * 8)).*;
                    const wf = @as(@Vector(8, f32), @floatFromInt(@as(@Vector(8, i32), @intCast(wv8)))) * d;
                    inline for (0..8) |tt| {
                        const xv: @Vector(8, f32) = @as(*align(1) const [8]f32, @ptrCast(X + (tb + tt) * Ks + b * 32 + q * 8)).*;
                        acc[tt] = @mulAdd(@Vector(8, f32), wf, xv, acc[tt]);
                    }
                }
            }
            inline for (0..8) |tt| Y[(tb + tt) * ys + yo + m] = @reduce(.Add, acc[tt]);
        }
        while (tb < Ts) : (tb += 1) {
            var a0: @Vector(8, f32) = @splat(0);
            var a1: @Vector(8, f32) = @splat(0);
            var b: usize = 0;
            while (b < blocks_per_row) : (b += 1) {
                const bp = row_ptr + b * block_stride;
                const d: @Vector(8, f32) = @splat(types.fp16_to_fp32(std.mem.readInt(u16, bp[0..2], .little)));
                inline for (0..4) |q| {
                    const wv8: @Vector(8, i8) = @as(*align(1) const [8]i8, @ptrCast(bp + 2 + q * 8)).*;
                    const wf = @as(@Vector(8, f32), @floatFromInt(@as(@Vector(8, i32), @intCast(wv8)))) * d;
                    const xv: @Vector(8, f32) = @as(*align(1) const [8]f32, @ptrCast(X + tb * Ks + b * 32 + q * 8)).*;
                    if (q % 2 == 0) { a0 = @mulAdd(@Vector(8, f32), wf, xv, a0); } else { a1 = @mulAdd(@Vector(8, f32), wf, xv, a1); }
                }
            }
            Y[tb * ys + yo + m] = @reduce(.Add, a0 + a1);
        }
    }
}
