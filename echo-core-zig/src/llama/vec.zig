//! Port: ggml-cpu/vec.h + vec.cpp (AVX2 dalları; GGML_F32_STEP = GGML_F16_STEP = 32).
const std = @import("std");
const avx = @import("avx.zig");
const F8 = avx.F8;
const I8 = avx.I8;
const loadF = avx.loadF;
const storeF = avx.storeF;
const loadPh = avx.loadPh;
const storePh = avx.storePh;
const splat = avx.splat;

pub inline fn dotF32(n: usize, x: [*]const f32, y: [*]const f32) f32 {
    const np = n & ~@as(usize, 31);
    var s0: F8 = @splat(0);
    var s1: F8 = @splat(0);
    var s2: F8 = @splat(0);
    var s3: F8 = @splat(0);
    var i: usize = 0;
    while (i < np) : (i += 32) {
        s0 = @mulAdd(F8, loadF(x + i), loadF(y + i), s0);
        s1 = @mulAdd(F8, loadF(x + i + 8), loadF(y + i + 8), s1);
        s2 = @mulAdd(F8, loadF(x + i + 16), loadF(y + i + 16), s2);
        s3 = @mulAdd(F8, loadF(x + i + 24), loadF(y + i + 24), s3);
    }
    var sum = avx.hsumHadd((s0 + s2) + (s1 + s3));
    while (i < n) : (i += 1) sum += x[i] * y[i];
    return sum;
}

pub inline fn dotF16(n: usize, x: [*]const u16, y: [*]const u16) f32 {
    const np = n & ~@as(usize, 31);
    var s0: F8 = @splat(0);
    var s1: F8 = @splat(0);
    var s2: F8 = @splat(0);
    var s3: F8 = @splat(0);
    var i: usize = 0;
    while (i < np) : (i += 32) {
        s0 = @mulAdd(F8, loadPh(x + i), loadPh(y + i), s0);
        s1 = @mulAdd(F8, loadPh(x + i + 8), loadPh(y + i + 8), s1);
        s2 = @mulAdd(F8, loadPh(x + i + 16), loadPh(y + i + 16), s2);
        s3 = @mulAdd(F8, loadPh(x + i + 24), loadPh(y + i + 24), s3);
    }
    var sum: f64 = avx.hsumHadd((s0 + s2) + (s1 + s3));
    while (i < n) : (i += 1) sum += avx.fp16(x[i]) * avx.fp16(y[i]);
    return @floatCast(sum);
}

pub inline fn madF32(n: usize, y: [*]f32, x: [*]const f32, v: f32) void {
    const np = n & ~@as(usize, 31);
    const vx = splat(v);
    var i: usize = 0;
    while (i < np) : (i += 8) storeF(y + i, @mulAdd(F8, loadF(x + i), vx, loadF(y + i)));
    while (i < n) : (i += 1) y[i] += x[i] * v;
}

pub inline fn madF16(n: usize, y: [*]u16, x: [*]const u16, v: f32) void {
    const np = n & ~@as(usize, 31);
    const vx = splat(v);
    var i: usize = 0;
    while (i < np) : (i += 8) storePh(y + i, @mulAdd(F8, loadPh(x + i), vx, loadPh(y + i)));
    while (i < n) : (i += 1) y[i] = avx.fp32to16(avx.fp16(y[i]) + avx.fp16(x[i]) * v);
}

pub inline fn scaleF32(n: usize, y: [*]f32, v: f32) void {
    const np = n & ~@as(usize, 31);
    const vx = splat(v);
    var i: usize = 0;
    while (i < np) : (i += 8) storeF(y + i, loadF(y + i) * vx);
    while (i < n) : (i += 1) y[i] *= v;
}

pub inline fn scaleF16(n: usize, y: [*]u16, v: f32) void {
    const np = n & ~@as(usize, 31);
    const vx = splat(v);
    var i: usize = 0;
    while (i < np) : (i += 8) storePh(y + i, loadPh(y + i) * vx);
    while (i < n) : (i += 1) y[i] = avx.fp32to16(avx.fp16(y[i]) * v);
}

pub inline fn addF32(n: usize, z: [*]f32, x: [*]const f32, y: [*]const f32) void {
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) storeF(z + i, loadF(x + i) + loadF(y + i));
    while (i < n) : (i += 1) z[i] = x[i] + y[i];
}

pub inline fn mulF32(n: usize, z: [*]f32, x: [*]const f32, y: [*]const f32) void {
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) storeF(z + i, loadF(x + i) * loadF(y + i));
    while (i < n) : (i += 1) z[i] = x[i] * y[i];
}

pub inline fn maxF32(n: usize, x: [*]const f32) f32 {
    var m: f32 = -std.math.inf(f32);
    for (0..n) |i| m = if (m > x[i]) m else x[i];
    return m;
}

/// ggml_v_expf (AVX2)
pub inline fn vExpf(x: F8) F8 {
    const r = splat(0x1.8p23);
    const z = @mulAdd(F8, x, splat(0x1.715476p+0), r);
    const n = z - r;
    const b = @mulAdd(F8, -n, splat(0x1.7f7d1cp-20), @mulAdd(F8, -n, splat(0x1.62e4p-1), x));
    const e: I8 = @as(I8, @bitCast(z)) << @splat(23);
    const k: F8 = @bitCast(e +% @as(I8, @bitCast(splat(1))));
    const an = @abs(n);
    const c = an > splat(126);
    const u = b * b;
    const j = @mulAdd(
        F8,
        @mulAdd(F8, @mulAdd(F8, splat(0x1.0e4020p-7), b, splat(0x1.573e2ep-5)), u, @mulAdd(F8, splat(0x1.555e66p-3), b, splat(0x1.fffdb6p-2))),
        u,
        splat(0x1.ffffecp-1) * b,
    );
    if (!@reduce(.Or, c)) return @mulAdd(F8, j, k, k);
    const le0 = n <= splat(0);
    const g: I8 = @select(i32, le0, @as(I8, @splat(@bitCast(@as(u32, 0x82000000)))), @as(I8, @splat(0)));
    const s1: F8 = @bitCast(g +% @as(I8, @splat(0x7f000000)));
    const s2: F8 = @bitCast(e -% g);
    const d = an > splat(192);
    return @select(f32, d, s1 * s1, @select(f32, c, @mulAdd(F8, s2, j, s2) * s1, @mulAdd(F8, k, j, k)));
}

pub inline fn vSilu(x: F8) F8 {
    return x / (splat(1) + vExpf(splat(0) - x));
}

pub inline fn siluF32(x: f32) f32 {
    return x / (1.0 + avx.expf(-x));
}

pub inline fn silu(n: usize, y: [*]f32, x: [*]const f32) void {
    var i: usize = 0;
    while (i + 7 < n) : (i += 8) storeF(y + i, vSilu(loadF(x + i)));
    while (i < n) : (i += 1) y[i] = siluF32(x[i]);
}

pub inline fn swiglu(n: usize, y: [*]f32, x: [*]const f32, g: [*]const f32) void {
    var i: usize = 0;
    while (i + 7 < n) : (i += 8) storeF(y + i, vSilu(loadF(x + i)) * loadF(g + i));
    while (i < n) : (i += 1) y[i] = siluF32(x[i]) * g[i];
}

/// ggml_vec_soft_max_f32
pub inline fn softMax(n: usize, y: [*]f32, x: [*]const f32, max: f32) f64 {
    var i: usize = 0;
    var sum: f64 = 0;
    const m = splat(max);
    while (i + 7 < n) : (i += 8) {
        const val = vExpf(loadF(x + i) - m);
        storeF(y + i, val);
        sum += avx.hsum(val);
    }
    while (i < n) : (i += 1) {
        const val = avx.expf(x[i] - max);
        sum += val;
        y[i] = val;
    }
    return sum;
}

pub inline fn fp16to32Row(x: [*]const u16, y: [*]f32, n: usize) void {
    var i: usize = 0;
    while (i + 7 < n) : (i += 8) storeF(y + i, loadPh(x + i));
    while (i < n) : (i += 1) y[i] = avx.fp16(x[i]);
}

pub inline fn fp32to16Row(x: [*]const f32, y: [*]u16, n: usize) void {
    var i: usize = 0;
    while (i + 7 < n) : (i += 8) storePh(y + i, loadF(x + i));
    while (i < n) : (i += 1) y[i] = avx.fp32to16(x[i]);
}

/// simd-gemm.h: C[M x N] += A[M x K] * B[K x N]  (GEMM_RM=6, GEMM_RN=2)
inline fn gemmUkernel(comptime RM: usize, comptime RN: usize, C: [*]f32, A: [*]const f32, B: [*]const f32, K: usize, N: usize) void {
    var acc: [RM][RN]F8 = undefined;
    inline for (0..RM) |i| inline for (0..RN) |r| {
        acc[i][r] = loadF(C + i * N + r * 8);
    };
    var kk: usize = 0;
    while (kk < K) : (kk += 1) {
        var bv: [RN]F8 = undefined;
        inline for (0..RN) |r| bv[r] = loadF(B + kk * N + r * 8);
        inline for (0..RM) |i| {
            const p = splat(A[i * K + kk]);
            inline for (0..RN) |r| acc[i][r] = @mulAdd(F8, bv[r], p, acc[i][r]);
        }
    }
    inline for (0..RM) |i| inline for (0..RN) |r| {
        storeF(C + i * N + r * 8, acc[i][r]);
    };
}

pub fn simdGemm(C_in: [*]f32, A_in: [*]const f32, B: [*]const f32, M: usize, K: usize, N: usize) void {
    var C = C_in;
    var A = A_in;
    var ii: usize = 0;
    while (ii + 6 <= M) : (ii += 6) {
        var jj: usize = 0;
        while (jj + 16 <= N) : (jj += 16) gemmUkernel(6, 2, C + jj, A, B + jj, K, N);
        while (jj + 8 <= N) : (jj += 8) gemmUkernel(6, 1, C + jj, A, B + jj, K, N);
        while (jj < N) : (jj += 1) {
            for (0..6) |i| {
                var a = C[i * N + jj];
                for (0..K) |kk| a += A[i * K + kk] * B[kk * N + jj];
                C[i * N + jj] = a;
            }
        }
        A += 6 * K;
        C += 6 * N;
    }
    while (ii < M) : (ii += 1) {
        var jj: usize = 0;
        while (jj + 16 <= N) : (jj += 16) gemmUkernel(1, 2, C + jj, A, B + jj, K, N);
        while (jj + 8 <= N) : (jj += 8) gemmUkernel(1, 1, C + jj, A, B + jj, K, N);
        while (jj < N) : (jj += 1) {
            var a = C[jj];
            for (0..K) |kk| a += A[kk] * B[kk * N + jj];
            C[jj] = a;
        }
        A += K;
        C += N;
    }
}

test "simdGemm: naif çarpımla aynı" {
    var prng = std.Random.DefaultPrng.init(1);
    const r = prng.random();
    const sizes = [_][3]usize{ .{ 43, 256, 64 }, .{ 43, 64, 256 }, .{ 7, 256, 64 }, .{ 1, 64, 256 }, .{ 13, 5, 19 } };
    for (sizes) |sz| {
        const M = sz[0];
        const K = sz[1];
        const N = sz[2];
        const a = try std.testing.allocator.alloc(f32, M * K);
        defer std.testing.allocator.free(a);
        const b = try std.testing.allocator.alloc(f32, K * N);
        defer std.testing.allocator.free(b);
        const c = try std.testing.allocator.alloc(f32, M * N);
        defer std.testing.allocator.free(c);
        for (a) |*v| v.* = r.float(f32) - 0.5;
        for (b) |*v| v.* = r.float(f32) - 0.5;
        @memset(c, 0);
        simdGemm(c.ptr, a.ptr, b.ptr, M, K, N);
        for (0..M) |i| for (0..N) |j| {
            var s: f32 = 0;
            for (0..K) |k| s += a[i * K + k] * b[k * N + j];
            try std.testing.expectApproxEqAbs(s, c[i * N + j], 1e-3);
        };
    }
}

test "vExpf: -inf ve büyük negatif şeritler sıfıra gider, diğerleri expf ile uyumlu" {
    try avx.initLibm();
    const xs = [8]f32{ -std.math.inf(f32), 0.0, -1.0, 1.0, -200.0, -90.0, 5.0, -0.25 };
    const v = vExpf(xs);
    for (xs, 0..) |x, i| {
        const e = std.math.exp(x);
        try std.testing.expectApproxEqRel(e + 1.0, v[i] + 1.0, 1e-5);
    }
}
