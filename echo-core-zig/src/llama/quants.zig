//! Port: ggml/src/ggml-common.h blokları, ggml-cpu/arch/x86/quants.c (AVX2), ggml-quants.c (ref).
const std = @import("std");
const avx = @import("avx.zig");
const F8 = avx.F8;
const B32 = avx.B32;
const W16 = avx.W16;
const I8 = avx.I8;
const fp16 = avx.fp16;
const U16V = @Vector(16, u16);

pub const BlockQ80 = extern struct { d: u16, qs: [32]i8 };
pub const BlockQ4K = extern struct { d: u16, dmin: u16, scales: [12]u8, qs: [128]u8 };
pub const BlockQ5K = extern struct { d: u16, dmin: u16, scales: [12]u8, qh: [32]u8, qs: [128]u8 };
pub const BlockQ6K = extern struct { ql: [128]u8, qh: [64]u8, scales: [16]i8, d: u16 };
pub const BlockQ8K = extern struct { d: f32, qs: [256]i8, bsums: [16]i16 };

comptime {
    std.debug.assert(@sizeOf(BlockQ80) == 34);
    std.debug.assert(@sizeOf(BlockQ4K) == 144);
    std.debug.assert(@sizeOf(BlockQ5K) == 176);
    std.debug.assert(@sizeOf(BlockQ6K) == 210);
    std.debug.assert(@sizeOf(BlockQ8K) == 292);
}

inline fn srli16(x: B32, comptime n: u4) B32 {
    const v: U16V = @bitCast(x);
    return @bitCast(v >> @splat(n));
}
inline fn slli16(x: B32, comptime n: u4) B32 {
    const v: U16V = @bitCast(x);
    return @bitCast(v << @splat(n));
}
inline fn set1(v: i8) B32 {
    return @splat(v);
}

/// get_scale_shuffle_k4
const shuf_k4: [8]B32 = blk: {
    var t: [8]B32 = undefined;
    for (0..8) |i| {
        var a: [32]i8 = undefined;
        for (0..16) |j| {
            a[2 * j] = @intCast(2 * i);
            a[2 * j + 1] = @intCast(2 * i + 1);
        }
        t[i] = a;
    }
    break :blk t;
};
/// get_scale_shuffle: satır i = {2*i x8, 2*i+1 x8}
const shuf_q6: [8]avx.B16 = blk: {
    var t: [8]avx.B16 = undefined;
    for (0..8) |i| {
        var a: [16]i8 = undefined;
        for (0..8) |j| {
            a[j] = @intCast(2 * i);
            a[8 + j] = @intCast(2 * i + 1);
        }
        t[i] = a;
    }
    break :blk t;
};

/// mul_sum_i8_pairs_float
inline fn mulSumI8Pairs(x: B32, y: B32) F8 {
    const ax = avx.sign8(x, x);
    const sy = avx.sign8(y, x);
    const dot = avx.maddubs(ax, sy);
    const s = avx.madd(@splat(1), dot);
    return @floatFromInt(s);
}

pub inline fn dotQ80Q80(n: usize, x: [*]const BlockQ80, y: [*]const BlockQ80) f32 {
    const nb = n / 32;
    var acc: F8 = @splat(0);
    var ib: usize = 0;
    while (ib < nb) : (ib += 1) {
        const d = avx.splat(fp16(x[ib].d) * fp16(y[ib].d));
        const qx = avx.loadB(&x[ib].qs);
        const qy = avx.loadB(&y[ib].qs);
        acc = @mulAdd(F8, d, mulSumI8Pairs(qx, qy), acc);
    }
    return avx.hsum(acc);
}

inline fn scalesK4(scales: *const [12]u8) W16 {
    const kmask1: u32 = 0x3f3f3f3f;
    const kmask2: u32 = 0x0f0f0f0f;
    const kmask3: u32 = 0x03030303;
    var utmp: [4]u32 = undefined;
    @memcpy(@as([*]u8, @ptrCast(&utmp))[0..12], scales);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
    const b: @Vector(16, u8) = @bitCast(@as(@Vector(4, u32), utmp));
    return @intCast(b);
}

inline fn dup128(x: avx.W8) W16 {
    return @shuffle(i16, x, undefined, [16]i32{ 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7 });
}

pub inline fn dotQ4KQ8K(n: usize, x: [*]const BlockQ4K, y: [*]const BlockQ8K) f32 {
    const nb = n / 256;
    const m4 = set1(0xF);
    var acc: F8 = @splat(0);
    var acc_m: avx.F4 = @splat(0);
    var i: usize = 0;
    while (i < nb) : (i += 1) {
        const dx = fp16(x[i].d);
        const d = y[i].d * dx;
        const dmin = -y[i].d * fp16(x[i].dmin);
        const mins_and_scales = scalesK4(&x[i].scales);
        const q8sums = avx.loadW(&y[i].bsums);
        const q8s = avx.hadd16(avx.low128(q8sums), avx.high128(q8sums));
        const prod = avx.madd128(avx.high128(mins_and_scales), q8s);
        const fprod: avx.F4 = @floatFromInt(prod);
        acc_m = @mulAdd(avx.F4, @splat(dmin), fprod, acc_m);
        const scales = dup128(avx.low128(mins_and_scales));
        var sumi: I8 = @splat(0);
        const q4: [*]const u8 = &x[i].qs;
        const q8: [*]const i8 = &y[i].qs;
        inline for (0..4) |j| {
            const scale_l: W16 = @bitCast(avx.shuffle8(@bitCast(scales), shuf_k4[2 * j]));
            const scale_h: W16 = @bitCast(avx.shuffle8(@bitCast(scales), shuf_k4[2 * j + 1]));
            const q4bits = avx.loadB(q4 + j * 32);
            const q4l = q4bits & m4;
            const q4h = srli16(q4bits, 4) & m4;
            const p16l = avx.madd(scale_l, avx.maddubs(q4l, avx.loadB(q8 + j * 64)));
            const p16h = avx.madd(scale_h, avx.maddubs(q4h, avx.loadB(q8 + j * 64 + 32)));
            sumi +%= p16l +% p16h;
        }
        const fs: F8 = @floatFromInt(sumi);
        acc = @mulAdd(F8, avx.splat(d), fs, acc);
    }
    return avx.hsum(acc) + @reduce(.Add, acc_m);
}

pub inline fn dotQ5KQ8K(n: usize, x: [*]const BlockQ5K, y: [*]const BlockQ8K) f32 {
    const nb = n / 256;
    const m4 = set1(0xF);
    var acc: F8 = @splat(0);
    var summs: f32 = 0;
    var i: usize = 0;
    while (i < nb) : (i += 1) {
        const q5: [*]const u8 = &x[i].qs;
        const q8: [*]const i8 = &y[i].qs;
        const d = y[i].d * fp16(x[i].d);
        const dmin = -y[i].d * fp16(x[i].dmin);
        const mins_and_scales = scalesK4(&x[i].scales);
        const q8sums = avx.loadW(&y[i].bsums);
        const q8s = avx.hadd16(avx.low128(q8sums), avx.high128(q8sums));
        const prod = avx.madd128(avx.high128(mins_and_scales), q8s);
        summs = @mulAdd(f32, dmin, @floatFromInt(@reduce(.Add, prod)), summs);
        const scales = dup128(avx.low128(mins_and_scales));
        const hbits = avx.loadB(&x[i].qh);
        var sumi: I8 = @splat(0);
        inline for (0..4) |j| {
            const scale_0: W16 = @bitCast(avx.shuffle8(@bitCast(scales), shuf_k4[2 * j]));
            const scale_1: W16 = @bitCast(avx.shuffle8(@bitCast(scales), shuf_k4[2 * j + 1]));
            const q5bits = avx.loadB(q5 + j * 32);
            const hm0 = comptime @as(B32, @splat(@as(i8, @bitCast(@as(u8, 1) << @intCast(2 * j)))));
            const hm1 = comptime @as(B32, @splat(@as(i8, @bitCast(@as(u8, 1) << @intCast(2 * j + 1)))));
            const q5l_0 = q5bits & m4;
            const q5h_0 = slli16(srli16(hbits & hm0, 2 * j), 4);
            const q5_0 = q5l_0 +% q5h_0;
            const q5l_1 = srli16(q5bits, 4) & m4;
            const q5h_1 = slli16(srli16(hbits & hm1, 2 * j + 1), 4);
            const q5_1 = q5l_1 +% q5h_1;
            const p16_0 = avx.madd(scale_0, avx.maddubs(q5_0, avx.loadB(q8 + j * 64)));
            const p16_1 = avx.madd(scale_1, avx.maddubs(q5_1, avx.loadB(q8 + j * 64 + 32)));
            sumi +%= p16_0 +% p16_1;
        }
        const fs: F8 = @floatFromInt(sumi);
        acc = @mulAdd(F8, avx.splat(d), fs, acc);
    }
    return avx.hsum(acc) + summs;
}

pub inline fn dotQ6KQ8K(n: usize, x: [*]const BlockQ6K, y: [*]const BlockQ8K) f32 {
    const nb = n / 256;
    const m3 = set1(3);
    const m15 = set1(15);
    var acc: F8 = @splat(0);
    var i: usize = 0;
    while (i < nb) : (i += 1) {
        const d = y[i].d * fp16(x[i].d);
        const q4: [*]const u8 = &x[i].ql;
        const qh: [*]const u8 = &x[i].qh;
        const q8: [*]const i8 = &y[i].qs;
        const q8sums = avx.loadW(&y[i].bsums);
        const scales: avx.B16 = @as(*align(1) const [16]i8, @ptrCast(&x[i].scales)).*;
        const scales_16: W16 = @intCast(scales);
        const q8sclsub: I8 = avx.madd(q8sums, scales_16) << @splat(5);
        var sumi: I8 = @splat(0);
        inline for (0..2) |j| {
            const q4bits1 = avx.loadB(q4 + j * 64);
            const q4bits2 = avx.loadB(q4 + j * 64 + 32);
            const q4bitsH = avx.loadB(qh + j * 32);
            const q4h_0 = slli16(q4bitsH & m3, 4);
            const q4h_1 = slli16(q4bitsH & set1(12), 2);
            const q4h_2 = q4bitsH & set1(48);
            const q4h_3 = srli16(q4bitsH & set1(-64), 2);
            const q4_0 = (q4bits1 & m15) | q4h_0;
            const q4_1 = (q4bits2 & m15) | q4h_1;
            const q4_2 = (srli16(q4bits1, 4) & m15) | q4h_2;
            const q4_3 = (srli16(q4bits2, 4) & m15) | q4h_3;
            const p16_0 = avx.maddubs(q4_0, avx.loadB(q8 + j * 128));
            const p16_1 = avx.maddubs(q4_1, avx.loadB(q8 + j * 128 + 32));
            const p16_2 = avx.maddubs(q4_2, avx.loadB(q8 + j * 128 + 64));
            const p16_3 = avx.maddubs(q4_3, avx.loadB(q8 + j * 128 + 96));
            const scale_0: W16 = @intCast(avx.shuffle8_128(scales, shuf_q6[j * 4 + 0]));
            const scale_1: W16 = @intCast(avx.shuffle8_128(scales, shuf_q6[j * 4 + 1]));
            const scale_2: W16 = @intCast(avx.shuffle8_128(scales, shuf_q6[j * 4 + 2]));
            const scale_3: W16 = @intCast(avx.shuffle8_128(scales, shuf_q6[j * 4 + 3]));
            sumi +%= avx.madd(scale_0, p16_0) +% avx.madd(scale_1, p16_1);
            sumi +%= avx.madd(scale_2, p16_2) +% avx.madd(scale_3, p16_3);
        }
        sumi -%= q8sclsub;
        const fs: F8 = @floatFromInt(sumi);
        acc = @mulAdd(F8, avx.splat(d), fs, acc);
    }
    return avx.hsum(acc);
}

/// quantize_row_q8_0 (AVX2)
pub fn quantizeRowQ80(x_in: [*]const f32, y: [*]BlockQ80, k: usize) void {
    quantizeRowQ(127.0, x_in, y, k);
}

/// qmax=63: H30 R8 yolu için 7 bit aktivasyon (a+64 u8'e sığar, maddubs doymaz).
pub fn quantizeRowQ(comptime qmax: f32, x_in: [*]const f32, y: [*]BlockQ80, k: usize) void {
    var x = x_in;
    const nb = k / 32;
    for (0..nb) |i| {
        var v0 = avx.loadF(x);
        var v1 = avx.loadF(x + 8);
        var v2 = avx.loadF(x + 16);
        var v3 = avx.loadF(x + 24);
        x += 32;
        const a = avx.maxps(avx.maxps(@abs(v0), @abs(v1)), avx.maxps(@abs(v2), @abs(v3)));
        var m4 = avx.maxps128(avx.high128(a), avx.low128(a));
        m4 = avx.maxps128(m4, @shuffle(f32, m4, undefined, [4]i32{ 2, 3, 2, 3 }));
        m4 = avx.maxps128(m4, @shuffle(f32, m4, undefined, [4]i32{ 1, 1, 3, 3 }));
        const max_scalar = m4[0];
        const d = max_scalar / qmax;
        y[i].d = avx.fp32to16(d);
        const mul = avx.splat(if (max_scalar != 0.0) qmax / max_scalar else 0.0);
        v0 *= mul;
        v1 *= mul;
        v2 *= mul;
        v3 *= mul;
        // ggml önce _mm256_round_ps(NEAREST) yapar; cvtps2dq zaten aynı yuvarlamayı kullanır.
        const c0 = avx.cvtps2dq(v0);
        const c1 = avx.cvtps2dq(v1);
        const c2 = avx.cvtps2dq(v2);
        const c3 = avx.cvtps2dq(v3);
        const p0 = avx.packs32(c0, c1);
        const p1 = avx.packs32(c2, c3);
        const p = avx.packs16(p0, p1);
        const r = @shuffle(i32, @as(I8, @bitCast(p)), undefined, [8]i32{ 0, 4, 1, 5, 2, 6, 3, 7 });
        @as(*align(1) [8]i32, @ptrCast(&y[i].qs)).* = r;
    }
}

inline fn nearestInt(fval: f32) i32 {
    const val = fval + 12582912.0;
    const i: i32 = @bitCast(val);
    return (i & 0x007fffff) - 0x00400000;
}

/// quantize_row_q8_K_ref
pub fn quantizeRowQ8K(x_in: [*]const f32, y: [*]BlockQ8K, k: usize) void {
    var x = x_in;
    const nb = k / 256;
    for (0..nb) |i| {
        var max: f32 = 0;
        var amax: f32 = 0;
        for (0..256) |j| {
            const ax = @abs(x[j]);
            if (ax > amax) {
                amax = ax;
                max = x[j];
            }
        }
        if (amax == 0) {
            y[i].d = 0;
            @memset(&y[i].qs, 0);
            x += 256;
            continue;
        }
        const iscale = -127.0 / max;
        for (0..256) |j| {
            const v = nearestInt(iscale * x[j]);
            y[i].qs[j] = @intCast(@max(-128, @min(127, v)));
        }
        for (0..16) |j| {
            var sum: i32 = 0;
            for (0..16) |ii| sum += y[i].qs[j * 16 + ii];
            y[i].bsums[j] = @intCast(sum);
        }
        y[i].d = 1.0 / iscale;
        x += 256;
    }
}

pub fn dequantizeQ80(x: [*]const BlockQ80, y: [*]f32, k: usize) void {
    const nb = k / 32;
    for (0..nb) |i| {
        const d = fp16(x[i].d);
        for (0..32) |j| y[i * 32 + j] = @as(f32, @floatFromInt(x[i].qs[j])) * d;
    }
}

inline fn scaleMinK4(j: usize, q: *const [12]u8, d: *u8, m: *u8) void {
    if (j < 4) {
        d.* = q[j] & 63;
        m.* = q[j + 4] & 63;
    } else {
        d.* = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m.* = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

pub fn dequantizeQ4K(x: [*]const BlockQ4K, y_in: [*]f32, k: usize) void {
    var y = y_in;
    for (0..k / 256) |i| {
        var q: [*]const u8 = &x[i].qs;
        const d = fp16(x[i].d);
        const min = fp16(x[i].dmin);
        var is: usize = 0;
        var j: usize = 0;
        while (j < 256) : (j += 64) {
            var sc: u8 = 0;
            var m: u8 = 0;
            scaleMinK4(is, &x[i].scales, &sc, &m);
            const d1 = d * @as(f32, @floatFromInt(sc));
            const m1 = min * @as(f32, @floatFromInt(m));
            scaleMinK4(is + 1, &x[i].scales, &sc, &m);
            const d2 = d * @as(f32, @floatFromInt(sc));
            const m2 = min * @as(f32, @floatFromInt(m));
            for (0..32) |l| y[l] = d1 * @as(f32, @floatFromInt(q[l] & 0xF)) - m1;
            for (0..32) |l| y[32 + l] = d2 * @as(f32, @floatFromInt(q[l] >> 4)) - m2;
            y += 64;
            q += 32;
            is += 2;
        }
    }
}

pub fn dequantizeQ5K(x: [*]const BlockQ5K, y_in: [*]f32, k: usize) void {
    var y = y_in;
    for (0..k / 256) |i| {
        var ql: [*]const u8 = &x[i].qs;
        const qh: [*]const u8 = &x[i].qh;
        const d = fp16(x[i].d);
        const min = fp16(x[i].dmin);
        var is: usize = 0;
        var u1_: u8 = 1;
        var u2_: u8 = 2;
        var j: usize = 0;
        while (j < 256) : (j += 64) {
            var sc: u8 = 0;
            var m: u8 = 0;
            scaleMinK4(is, &x[i].scales, &sc, &m);
            const d1 = d * @as(f32, @floatFromInt(sc));
            const m1 = min * @as(f32, @floatFromInt(m));
            scaleMinK4(is + 1, &x[i].scales, &sc, &m);
            const d2 = d * @as(f32, @floatFromInt(sc));
            const m2 = min * @as(f32, @floatFromInt(m));
            for (0..32) |l| {
                const hi: u8 = if (qh[l] & u1_ != 0) 16 else 0;
                y[l] = d1 * @as(f32, @floatFromInt((ql[l] & 0xF) + hi)) - m1;
            }
            for (0..32) |l| {
                const hi: u8 = if (qh[l] & u2_ != 0) 16 else 0;
                y[32 + l] = d2 * @as(f32, @floatFromInt((ql[l] >> 4) + hi)) - m2;
            }
            y += 64;
            ql += 32;
            is += 2;
            u1_ <<= 2;
            u2_ <<= 2;
        }
    }
}

pub fn dequantizeQ6K(x: [*]const BlockQ6K, y_in: [*]f32, k: usize) void {
    var y = y_in;
    for (0..k / 256) |i| {
        const d = fp16(x[i].d);
        var ql: [*]const u8 = &x[i].ql;
        var qh: [*]const u8 = &x[i].qh;
        var sc: [*]const i8 = &x[i].scales;
        var n: usize = 0;
        while (n < 256) : (n += 128) {
            for (0..32) |l| {
                const is = l / 16;
                const q1: i32 = @as(i32, (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const q2: i32 = @as(i32, (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const q3: i32 = @as(i32, (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const q4: i32 = @as(i32, (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l] = d * @as(f32, @floatFromInt(sc[is])) * @as(f32, @floatFromInt(q1));
                y[l + 32] = d * @as(f32, @floatFromInt(sc[is + 2])) * @as(f32, @floatFromInt(q2));
                y[l + 64] = d * @as(f32, @floatFromInt(sc[is + 4])) * @as(f32, @floatFromInt(q3));
                y[l + 96] = d * @as(f32, @floatFromInt(sc[is + 6])) * @as(f32, @floatFromInt(q4));
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}
