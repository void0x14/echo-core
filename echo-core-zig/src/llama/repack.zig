//! Port: ggml-cpu/repack.cpp + arch/x86/repack.cpp, Q4_K için AVX2 8x8 yolu.
//! Aritmetik gemm/gemv ile aynıdır (i16 sarmalı toplam, sb başına FMA, aynı FP sırası).
const std = @import("std");
const avx = @import("avx.zig");
const q = @import("quants.zig");
const Params = @import("pool.zig").Params;
const F8 = avx.F8;
const B32 = avx.B32;
const W16 = avx.W16;
const I8 = avx.I8;
const U16V = @Vector(16, u16);

pub const BlockQ4Kx8 = extern struct { d: [8]u16, dmin: [8]u16, scales: [96]u8, qs: [1024]u8 };
pub const BlockQ8Kx4 = extern struct { d: [4]f32, qs: [1024]i8, bsums: [64]i16 };

comptime {
    std.debug.assert(@sizeOf(BlockQ4Kx8) == 1152);
    std.debug.assert(@sizeOf(BlockQ8Kx4) == 1168);
}

/// make_block_q4_Kx8(in, 8)
fn makeBlock(in: *const [8]q.BlockQ4K, out: *BlockQ4Kx8) void {
    for (0..8) |i| {
        out.d[i] = in[i].d;
        out.dmin[i] = in[i].dmin;
    }
    for (0..128) |i| {
        const src_id = i % 8;
        const src_offset = (i / 8) * 8;
        @memcpy(out.qs[i * 8..][0..8], in[src_id].qs[src_offset..][0..8]);
    }
    var s: [8]u8 = undefined;
    var m: [8]u8 = undefined;
    for (0..2) |half| {
        for (0..4) |i| {
            for (0..8) |j| {
                if (half == 0) {
                    s[j] = in[j].scales[i] & 63;
                    m[j] = in[j].scales[i + 4] & 63;
                } else {
                    s[j] = ((in[j].scales[i] & 192) >> 2) | (in[j].scales[i + 8] & 15);
                    m[j] = ((in[j].scales[i + 4] & 192) >> 2) | ((in[j].scales[i + 8] & 240) >> 4);
                }
            }
            const o = i * 12 + half * 48;
            for (0..4) |c| {
                out.scales[o + c] = (s[c] & 63) +% ((s[c + 4] & 48) << 2);
                out.scales[o + 4 + c] = (m[c] & 63) +% ((m[c + 4] & 48) << 2);
                out.scales[o + 8 + c] = (s[c + 4] & 15) +% ((m[c + 4] & 15) << 4);
            }
        }
    }
}

/// repack_q4_K_to_q4_K_8_bl
pub fn repackQ4K(src: [*]const q.BlockQ4K, dst: [*]BlockQ4Kx8, nrow: usize, nblocks: usize) void {
    var tmp: [8]q.BlockQ4K = undefined;
    var d = dst;
    var s = src;
    var b: usize = 0;
    while (b < nrow) : (b += 8) {
        for (0..nblocks) |x| {
            for (0..8) |i| tmp[i] = s[x + i * nblocks];
            makeBlock(&tmp, &d[0]);
            d += 1;
        }
        s += 8 * nblocks;
    }
}

/// ggml_quantize_mat_q8_K_4x8 (AVX2): 4 satır, satır başına 8 bayt iç içe.
fn quantizeMat4x8(x: [*]const f32, y: [*]BlockQ8Kx4, k: usize) void {
    const nb = k / 256;
    for (0..nb) |i| {
        var iscale: [4]F8 = undefined;
        for (0..4) |r| {
            const row = x + r * k + i * 256;
            var amax: f32 = 0;
            var max: f32 = 0;
            for (0..256) |j| {
                const a = @abs(row[j]);
                if (amax < a) {
                    amax = a;
                    max = row[j];
                }
            }
            const is: f32 = if (amax != 0) -127.0 / max else 0;
            y[i].d[r] = if (amax != 0) 1.0 / is else 0;
            iscale[r] = avx.splat(is);
        }
        for (0..32) |j| {
            const v0 = avx.loadF(x + 0 * k + i * 256 + j * 8) * iscale[0];
            const v1 = avx.loadF(x + 1 * k + i * 256 + j * 8) * iscale[1];
            const v2 = avx.loadF(x + 2 * k + i * 256 + j * 8) * iscale[2];
            const v3 = avx.loadF(x + 3 * k + i * 256 + j * 8) * iscale[3];
            const p0 = avx.packs32(avx.cvtps2dq(v0), avx.cvtps2dq(v1));
            const p1 = avx.packs32(avx.cvtps2dq(v2), avx.cvtps2dq(v3));
            const p: I8 = @bitCast(avx.packs16(p0, p1));
            const r = @shuffle(i32, p, undefined, [8]i32{ 0, 4, 1, 5, 2, 6, 3, 7 });
            @as(*align(1) [8]i32, @ptrCast(&y[i].qs[32 * j])).* = r;
        }
        // bsums: sb başına [r0 g0..g3, r1 g0..g3, ...]
        for (0..4) |sb| {
            for (0..4) |g| {
                for (0..4) |r| {
                    var sum: i32 = 0;
                    for (0..2) |c| {
                        const o = sb * 256 + g * 64 + c * 32 + r * 8;
                        for (0..8) |e| sum += y[i].qs[o + e];
                    }
                    y[i].bsums[sb * 16 + r * 4 + g] = @intCast(sum);
                }
            }
        }
    }
}

inline fn srli16(x: B32, comptime n: u4) B32 {
    const v: U16V = @bitCast(x);
    return @bitCast(v >> @splat(n));
}

/// utmp -> [8 skala, 8 min] u8 (scalesK4 ile aynı açma)
inline fn unpackScales(p: *const [12]u8) @Vector(16, u8) {
    const kmask1: u32 = 0x3f3f3f3f;
    const kmask2: u32 = 0x0f0f0f0f;
    const kmask3: u32 = 0x03030303;
    var utmp: [4]u32 = undefined;
    @memcpy(@as([*]u8, @ptrCast(&utmp))[0..12], p);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
    return @bitCast(@as(@Vector(4, u32), utmp));
}

/// i32 şerit başına [c, c] i16: madd(iacc, scale) = (l0+l1)*scale_c
inline fn scaleLanes(ms: @Vector(16, u8)) W16 {
    const low: @Vector(8, u8) = @shuffle(u8, ms, undefined, [8]i32{ 0, 1, 2, 3, 4, 5, 6, 7 });
    const w: @Vector(8, i16) = @intCast(low);
    const d = @shuffle(i16, w, undefined, [16]i32{ 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7 });
    return d;
}
/// i32 şerit başına [min_s0_c, min_s1_c] i16
inline fn minLanes(m0: @Vector(16, u8), m1: @Vector(16, u8)) W16 {
    const a: @Vector(8, u8) = @shuffle(u8, m0, undefined, [8]i32{ 8, 9, 10, 11, 12, 13, 14, 15 });
    const b: @Vector(8, u8) = @shuffle(u8, m1, undefined, [8]i32{ 8, 9, 10, 11, 12, 13, 14, 15 });
    const wa: @Vector(8, i16) = @intCast(a);
    const wb: @Vector(8, i16) = @intCast(b);
    return @shuffle(i16, wa, wb, [16]i32{ 0, ~@as(i32, 0), 1, ~@as(i32, 1), 2, ~@as(i32, 2), 3, ~@as(i32, 3), 4, ~@as(i32, 4), 5, ~@as(i32, 5), 6, ~@as(i32, 6), 7, ~@as(i32, 7) });
}

/// 8 sütunu sıraya sokar: h=0 her sütunun ilk 4 baytı, h=1 ikinci 4 baytı (0x88 / 0xdd vshufps).
inline fn colLanes(a0145: B32, a2367: B32, comptime h: usize) B32 {
    const x: I8 = @bitCast(a0145);
    const y: I8 = @bitCast(a2367);
    const r = if (h == 0)
        @shuffle(i32, x, y, [8]i32{ 0, 2, ~@as(i32, 0), ~@as(i32, 2), 4, 6, ~@as(i32, 4), ~@as(i32, 6) })
    else
        @shuffle(i32, x, y, [8]i32{ 1, 3, ~@as(i32, 1), ~@as(i32, 3), 5, 7, ~@as(i32, 5), ~@as(i32, 7) });
    return @bitCast(r);
}

inline fn rowDup(p: *const anyopaque) B32 {
    const v: i32 = @as(*align(1) const i32, @ptrCast(p)).*;
    return @bitCast(@as(I8, @splat(v)));
}

/// ggml_gemm_q4_K_8x8_q8_K: R adet Q8_Kx4 bloğu (R*4 satır), 8 sütun.
fn gemmCols(
    comptime R: usize,
    nb: usize,
    b_ptr: [*]const BlockQ4Kx8,
    a_ptrs: *const [R][*]const BlockQ8Kx4,
    s: [*]f32,
    bs: usize,
) void {
    @setEvalBranchQuota(100000);
    const m4b: B32 = @splat(0x0F);
    var acc: [R * 4]F8 = undefined;
    var acc_min: [R * 4]F8 = undefined;
    inline for (0..R * 4) |i| {
        acc[i] = @splat(0);
        acc_min[i] = @splat(0);
    }
    for (0..nb) |b| {
        const col_scale = avx.loadPh(&b_ptr[b].d);
        const col_dmin = avx.loadPh(&b_ptr[b].dmin);
        inline for (0..4) |sb| {
            const ms0 = unpackScales(b_ptr[b].scales[24 * sb..][0..12]);
            const ms1 = unpackScales(b_ptr[b].scales[12 + 24 * sb..][0..12]);
            const scales_0 = scaleLanes(ms0);
            const scales_1 = scaleLanes(ms1);
            const mins_01 = minLanes(ms0, ms1);
            var iacc0: [R * 4]W16 = undefined;
            var iacc1: [R * 4]W16 = undefined;
            inline for (0..R * 4) |i| {
                iacc0[i] = @splat(0);
                iacc1[i] = @splat(0);
            }
            const qs: [*]const u8 = &b_ptr[b].qs;
            inline for (0..4) |g| {
                const raw0123 = avx.loadB(qs + sb * 256 + g * 64);
                const raw4567 = avx.loadB(qs + sb * 256 + g * 64 + 32);
                const r0: @Vector(4, i64) = @bitCast(raw0123);
                const r1: @Vector(4, i64) = @bitCast(raw4567);
                // [B0 B1 | B4 B5], [B2 B3 | B6 B7]
                const raw0145: B32 = @bitCast(@shuffle(i64, r0, r1, [4]i32{ 0, 1, ~@as(i32, 0), ~@as(i32, 1) }));
                const raw2367: B32 = @bitCast(@shuffle(i64, r0, r1, [4]i32{ 2, 3, ~@as(i32, 2), ~@as(i32, 3) }));
                const lo0145 = raw0145 & m4b;
                const lo2367 = raw2367 & m4b;
                const hi0145 = srli16(raw0145, 4) & m4b;
                const hi2367 = srli16(raw2367, 4) & m4b;
                inline for (0..2) |h| {
                    const lo = colLanes(lo0145, lo2367, h);
                    const hi = colLanes(hi0145, hi2367, h);
                    inline for (0..R) |rp| {
                        const a = &a_ptrs[rp][b];
                        inline for (0..4) |r| {
                            const o0 = 256 * sb + 32 * g + 8 * r + 4 * h;
                            iacc0[rp * 4 + r] +%= avx.maddubs(lo, rowDup(&a.qs[o0]));
                            iacc1[rp * 4 + r] +%= avx.maddubs(hi, rowDup(&a.qs[o0 + 128]));
                        }
                    }
                }
            }
            inline for (0..R) |rp| {
                const a = &a_ptrs[rp][b];
                const bsums = avx.loadW(&a.bsums[16 * sb]);
                const hs = avx.hadd16(avx.low128(bsums), avx.high128(bsums));
                const hs4: @Vector(4, i32) = @bitCast(hs);
                inline for (0..4) |r| {
                    const iacc = avx.madd(iacc0[rp * 4 + r], scales_0) +% avx.madd(iacc1[rp * 4 + r], scales_1);
                    const scale_r = avx.splat(a.d[r]);
                    const fi: F8 = @floatFromInt(iacc);
                    acc[rp * 4 + r] = @mulAdd(F8, fi, col_scale * scale_r, acc[rp * 4 + r]);
                    const bs_r: W16 = @bitCast(@as(I8, @splat(hs4[r])));
                    const fm: F8 = @floatFromInt(avx.madd(bs_r, mins_01));
                    acc_min[rp * 4 + r] = @mulAdd(F8, fm, col_dmin * scale_r, acc_min[rp * 4 + r]);
                }
            }
        }
    }
    inline for (0..R * 4) |i| avx.storeF(s + i * bs, acc[i] - acc_min[i]);
}

/// ggml_gemv_q4_K_8x8_q8_K: tek satır Q8_K, 8 sütun.
inline fn gemvCols(nb: usize, b_ptr: [*]const BlockQ4Kx8, a_ptr: [*]const q.BlockQ8K, s: [*]f32) void {
    const m4b: B32 = @splat(0x0F);
    var acc_row: F8 = @splat(0);
    var acc_min: F8 = @splat(0);
    for (0..nb) |b| {
        const row_scale = avx.splat(a_ptr[b].d);
        const col_scale = avx.loadPh(&b_ptr[b].d);
        const col_dmin = avx.loadPh(&b_ptr[b].dmin);
        var iacc_b: I8 = @splat(0);
        var iacc_min_b: I8 = @splat(0);
        const q8sums = avx.loadW(&a_ptr[b].bsums);
        const q8s4: @Vector(4, i32) = @bitCast(avx.hadd16(avx.low128(q8sums), avx.high128(q8sums)));
        const qs: [*]const u8 = &b_ptr[b].qs;
        const q8: [*]const i8 = &a_ptr[b].qs;
        inline for (0..4) |sb| {
            const ms0 = unpackScales(b_ptr[b].scales[24 * sb..][0..12]);
            const ms1 = unpackScales(b_ptr[b].scales[12 + 24 * sb..][0..12]);
            var iacc0: W16 = @splat(0);
            var iacc1: W16 = @splat(0);
            inline for (0..4) |g| {
                const raw0123 = avx.loadB(qs + sb * 256 + g * 64);
                const raw4567 = avx.loadB(qs + sb * 256 + g * 64 + 32);
                const r0: @Vector(4, i64) = @bitCast(raw0123);
                const r1: @Vector(4, i64) = @bitCast(raw4567);
                const raw0145: B32 = @bitCast(@shuffle(i64, r0, r1, [4]i32{ 0, 1, ~@as(i32, 0), ~@as(i32, 1) }));
                const raw2367: B32 = @bitCast(@shuffle(i64, r0, r1, [4]i32{ 2, 3, ~@as(i32, 2), ~@as(i32, 3) }));
                const lo0145 = raw0145 & m4b;
                const lo2367 = raw2367 & m4b;
                const hi0145 = srli16(raw0145, 4) & m4b;
                const hi2367 = srli16(raw2367, 4) & m4b;
                inline for (0..2) |h| {
                    const e = sb * 64 + g * 8 + h * 4;
                    iacc0 +%= avx.maddubs(colLanes(lo0145, lo2367, h), rowDup(q8 + e));
                    iacc1 +%= avx.maddubs(colLanes(hi0145, hi2367, h), rowDup(q8 + e + 32));
                }
            }
            iacc_b +%= avx.madd(iacc0, scaleLanes(ms0)) +% avx.madd(iacc1, scaleLanes(ms1));
            const bs_sb: W16 = @bitCast(@as(I8, @splat(q8s4[sb])));
            iacc_min_b +%= avx.madd(bs_sb, minLanes(ms0, ms1));
        }
        const fi: F8 = @floatFromInt(iacc_b);
        const fm: F8 = @floatFromInt(iacc_min_b);
        acc_row = @mulAdd(F8, fi, col_scale * row_scale, acc_row);
        acc_min = @mulAdd(F8, fm, col_dmin * row_scale, acc_min);
    }
    avx.storeF(s, acc_row - acc_min);
}

fn gemm(n: usize, s: [*]f32, bs: usize, vx: [*]const BlockQ4Kx8, vy: [*]const BlockQ8Kx4, nr: usize, nc: usize) void {
    const nb = n / 256;
    const anr = nr - nr % 16;
    var y: usize = 0;
    while (y < anr / 4) : (y += 4) {
        const a_ptrs = [4][*]const BlockQ8Kx4{ vy + y * nb, vy + (y + 1) * nb, vy + (y + 2) * nb, vy + (y + 3) * nb };
        var x: usize = 0;
        while (x < nc / 8) : (x += 1) gemmCols(4, nb, vx + x * nb, &a_ptrs, s + y * 4 * bs + x * 8, bs);
    }
    while (y < nr / 4) : (y += 1) {
        const a_ptrs = [1][*]const BlockQ8Kx4{vy + y * nb};
        var x: usize = 0;
        while (x < nc / 8) : (x += 1) gemmCols(1, nb, vx + x * nb, &a_ptrs, s + y * 4 * bs + x * 8, bs);
    }
}

/// tensor_traits::forward_mul_mat
pub fn forwardMulMat(p: *const Params, a: anytype, x: [*]const f32, n: usize, y: [*]f32) void {
    const ne10 = a.k;
    const nb = ne10 / 256;
    const nbw1 = nb * 292;
    const wdata = p.wdata;
    const full = n - n % 4;
    var r: usize = p.ith * 4;
    while (r < full) : (r += p.nth * 4) {
        quantizeMat4x8(x + r * ne10, @ptrCast(@alignCast(wdata + r * nbw1)), ne10);
    }
    r = full + p.ith;
    while (r < n) : (r += p.nth) {
        q.quantizeRowQ8K(x + r * ne10, @ptrCast(@alignCast(wdata + r * nbw1)), ne10);
    }

    const nr0 = a.m;
    const nth_scaled = p.nth * 4;
    const chunk_size0 = (nr0 + nth_scaled - 1) / nth_scaled;
    var nchunk0 = (nr0 + chunk_size0 - 1) / chunk_size0;
    const min_chunk: usize = 8;
    if (nchunk0 > 0 and (nr0 / nchunk0) < min_chunk and nr0 >= min_chunk) nchunk0 = (nr0 + min_chunk - 1) / min_chunk;
    var dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    if (p.nth == 1 or (nchunk0 < p.nth and (nr0 + p.nth - 1) / p.nth >= min_chunk)) {
        nchunk0 = p.nth;
        dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    }
    nchunk0 = @min(nchunk0, (nr0 + min_chunk - 1) / min_chunk);

    if (p.ith == 0) p.pool.chunkSet(@intCast(p.nth));
    p.barrier();

    const vx: [*]const BlockQ4Kx8 = @ptrCast(@alignCast(a.data));
    var current: i64 = p.ith;
    while (current < nchunk0) {
        const c: usize = @intCast(current);
        var start = dr0 * c;
        var end = @min(start + dr0, nr0);
        if (start % 8 != 0) start = start + 8 - start % 8;
        if (end % 8 != 0) end = end + 8 - end % 8;
        end = @min(end, nr0);
        if (start < end) {
            const ncols = end - start;
            const b0 = vx + start / 8 * nb;
            if (n > 3) gemm(ne10, y + start, a.m, b0, @ptrCast(@alignCast(wdata)), full, ncols);
            var iter = full;
            while (iter < n) : (iter += 1) {
                const row: [*]const q.BlockQ8K = @ptrCast(@alignCast(wdata + iter * nbw1));
                var cx: usize = 0;
                while (cx < ncols / 8) : (cx += 1) gemvCols(nb, b0 + cx * nb, row, y + iter * a.m + start + cx * 8);
            }
        }
        current = p.pool.chunkAdd(1);
    }
}
