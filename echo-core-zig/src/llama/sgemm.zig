//! Port: ggml-cpu/llamafile/sgemm.cpp (AVX2, VECTOR_REGISTERS == 16).
//! tinyBLAS<8, __m256, __m256, TA, TB, float> ve tinyBLAS_Q0_AVX<block_q8_0, block_q8_0, float>.
const avx = @import("avx.zig");
const q = @import("quants.zig");
const Params = @import("pool.zig").Params;
const F8 = avx.F8;
const F4 = avx.F4;

inline fn blockSize(comptime M: usize, m: usize) usize {
    const nb = (m + M - 1) / M;
    return if (m % nb == 0) m / nb else m / nb + 1;
}
inline fn blocPos(ib: usize, ibn: usize, size: usize) usize {
    return if (ib < ibn) ib * size else ibn * size + (ib - ibn) * (size - 1);
}

pub fn TinyBlas(comptime T: type) type {
    return struct {
        const Self = @This();
        p: *const Params,
        k: usize,
        a: [*]const T,
        lda: usize,
        b: [*]const T,
        ldb: usize,
        c: [*]f32,
        ldc: usize,

        inline fn load(p: [*]const T) F8 {
            return if (T == f32) avx.loadF(p) else avx.loadPh(p);
        }

        pub fn matmul(self: *const Self, m: usize, n: usize) bool {
            if (self.k % 8 != 0) return false;
            const bm: usize = if (m % 16 == 0 and m / 16 >= self.p.nth) 4 else if (m % 8 == 0) 2 else if (m % 4 == 0) 1 else return false;
            switch (blockSize(3, n)) {
                3 => self.gemm(4, 3, bm, m, n, 24),
                2 => self.gemm(4, 2, bm, m, n, 24),
                1 => self.gemm(4, 1, bm, m, n, 24),
                else => unreachable,
            }
            return true;
        }

        inline fn gemmBloc(self: *const Self, comptime RM: usize, comptime RN: usize, ii: usize, jj: usize) void {
            var cv: [RN][RM]F8 = undefined;
            inline for (0..RN) |j| inline for (0..RM) |i| {
                cv[j][i] = @splat(0);
            };
            var l: usize = 0;
            while (l < self.k) : (l += 8) {
                if (RM <= RN) {
                    var av: [RM]F8 = undefined;
                    inline for (0..RM) |i| av[i] = load(self.a + self.lda * (ii + i) + l);
                    inline for (0..RN) |j| {
                        const bv = load(self.b + self.ldb * (jj + j) + l);
                        inline for (0..RM) |i| cv[j][i] = @mulAdd(F8, av[i], bv, cv[j][i]);
                    }
                } else {
                    var bv: [RN]F8 = undefined;
                    inline for (0..RN) |j| bv[j] = load(self.b + self.ldb * (jj + j) + l);
                    inline for (0..RM) |i| {
                        const av = load(self.a + self.lda * (ii + i) + l);
                        inline for (0..RN) |j| cv[j][i] = @mulAdd(F8, av, bv[j], cv[j][i]);
                    }
                }
            }
            inline for (0..RN) |j| inline for (0..RM) |i| {
                self.c[self.ldc * (jj + j) + (ii + i)] = avx.hsum(cv[j][i]);
            };
        }

        fn gemm(self: *const Self, comptime RM: usize, comptime RN: usize, bm: usize, m: usize, n: usize, bn: usize) void {
            const ytiles = m / (RM * bm);
            const xtiles = (n + RN - 1) / RN;
            const jj_rn = xtiles - (xtiles * RN - n);
            const nb_bn: usize = if (xtiles < bn) 1 else (xtiles + bn / 2) / bn;
            const size_bn = if (xtiles % nb_bn == 0) xtiles / nb_bn else xtiles / nb_bn + 1;
            const jj_bn = nb_bn - (nb_bn * size_bn - xtiles);
            const nb_job = ytiles * nb_bn;
            const pool = self.p.pool;
            if (self.p.ith == 0) pool.chunkSet(@intCast(self.p.nth));
            pool.barrier();
            var job: i64 = self.p.ith;
            while (job < nb_job) {
                const uj: usize = @intCast(job);
                const ii = (uj % ytiles) * RM * bm;
                const jb = uj / ytiles;
                const jr0 = blocPos(jb, jj_bn, size_bn);
                const jrn = blocPos(jb + 1, jj_bn, size_bn);
                const jj0 = blocPos(jr0, jj_rn, RN);
                const jj2 = blocPos(jrn, jj_rn, RN);
                const jj1 = @min(jj2, jj_rn * RN);
                var bi: usize = 0;
                while (bi < bm * RM) : (bi += RM) {
                    var jj = jj0;
                    while (jj < jj1) : (jj += RN) self.gemmBloc(RM, RN, ii + bi, jj);
                    if (RN > 1) {
                        while (jj < jj2) : (jj += RN - 1) self.gemmBloc(RM, RN - 1, ii + bi, jj);
                    }
                }
                job = pool.chunkAdd(1);
            }
            pool.barrier();
        }
    };
}

pub const TinyBlasQ0 = struct {
    k: usize,
    a: [*]const q.BlockQ80,
    lda: usize,
    b: [*]const q.BlockQ80,
    ldb: usize,
    c: [*]f32,
    ldc: usize,
    ith: usize,
    nth: usize,

    pub fn matmul(self: *const TinyBlasQ0, m: usize, n: usize) void {
        self.mnpack(0, m, 0, n);
    }

    fn mnpack(self: *const TinyBlasQ0, m0: usize, m: usize, n0: usize, n: usize) void {
        var mc: usize = undefined;
        var nc: usize = undefined;
        const key: usize = (@as(usize, @min(m - m0, 4)) << 4) | @as(usize, @min(n - n0, 4));
        switch (key) {
            0x44, 0x43, 0x42 => {
                mc = 4;
                nc = 2;
                self.gemm4xN(2, m0, m, n0, n);
            },
            0x34, 0x24 => {
                mc = 2;
                nc = 4;
                self.gemmMx4(2, m0, m, n0, n);
            },
            0x33, 0x32 => {
                mc = 3;
                nc = 2;
                self.gemm(3, 2, m0, m, n0, n);
            },
            0x23 => {
                mc = 2;
                nc = 3;
                self.gemm(2, 3, m0, m, n0, n);
            },
            0x41 => {
                mc = 4;
                nc = 1;
                self.gemm4xN(1, m0, m, n0, n);
            },
            0x22 => {
                mc = 2;
                nc = 2;
                self.gemm(2, 2, m0, m, n0, n);
            },
            0x14 => {
                mc = 1;
                nc = 4;
                self.gemmMx4(1, m0, m, n0, n);
            },
            0x31 => {
                mc = 3;
                nc = 1;
                self.gemm(3, 1, m0, m, n0, n);
            },
            0x13 => {
                mc = 1;
                nc = 3;
                self.gemm(1, 3, m0, m, n0, n);
            },
            0x21 => {
                mc = 2;
                nc = 1;
                self.gemm(2, 1, m0, m, n0, n);
            },
            0x12 => {
                mc = 1;
                nc = 2;
                self.gemm(1, 2, m0, m, n0, n);
            },
            0x11 => {
                mc = 1;
                nc = 1;
                self.gemm(1, 1, m0, m, n0, n);
            },
            else => return,
        }
        const mp = m0 + (m - m0) / mc * mc;
        const np = n0 + (n - n0) / nc * nc;
        self.mnpack(mp, m, n0, np);
        self.mnpack(m0, m, np, n);
    }

    inline fn updot(u: avx.B32, s: avx.B32) F8 {
        const res = avx.madd(@splat(1), avx.maddubs(u, s));
        return @floatFromInt(res);
    }
    /// _mm_cvtph_ps(_mm_set_epi64x(0, delta))
    inline fn delta4(a: [*]const q.BlockQ80, stride: usize, base: usize, l: usize) F4 {
        const h = @Vector(4, f16){
            @bitCast(a[stride * (base + 0) + l].d),
            @bitCast(a[stride * (base + 1) + l].d),
            @bitCast(a[stride * (base + 2) + l].d),
            @bitCast(a[stride * (base + 3) + l].d),
        };
        return @floatCast(h);
    }
    inline fn bcast(v: F4, comptime i: i32) F8 {
        return @shuffle(f32, v, undefined, [8]i32{ i, i, i, i, i, i, i, i });
    }

    noinline fn gemm4xN(self: *const TinyBlasQ0, comptime RN: usize, m0: usize, m: usize, n0: usize, n: usize) void {
        const ytiles = (m - m0) / 4;
        const xtiles = (n - n0) / RN;
        const tiles = xtiles * ytiles;
        const duty = (tiles + self.nth - 1) / self.nth;
        const start = duty * self.ith;
        const end = @min(start + duty, tiles);
        var job = start;
        while (job < end) : (job += 1) {
            const ii = m0 + job / xtiles * 4;
            const jj = n0 + job % xtiles * RN;
            var cv: [RN][4]F8 = undefined;
            inline for (0..RN) |j| inline for (0..4) |i| {
                cv[j][i] = @splat(0);
            };
            const a = self.a;
            const b = self.b;
            for (0..self.k) |l| {
                const da = delta4(a, self.lda, ii, l);
                const avec0 = avx.loadB(&a[self.lda * (ii + 0) + l].qs);
                const avec1 = avx.loadB(&a[self.lda * (ii + 1) + l].qs);
                const avec2 = avx.loadB(&a[self.lda * (ii + 2) + l].qs);
                const avec3 = avx.loadB(&a[self.lda * (ii + 3) + l].qs);
                const a0 = avx.sign8(avec0, avec0);
                const a1 = avx.sign8(avec1, avec1);
                const a2 = avx.sign8(avec2, avec2);
                const a3 = avx.sign8(avec3, avec3);
                inline for (0..RN) |j| {
                    const bl = &b[self.ldb * (jj + j) + l];
                    const db: F4 = @splat(avx.fp16(bl.d));
                    const dvec = da * db;
                    const bv = avx.loadB(&bl.qs);
                    cv[j][0] = @mulAdd(F8, bcast(dvec, 0), updot(a0, avx.sign8(bv, avec0)), cv[j][0]);
                    cv[j][1] = @mulAdd(F8, bcast(dvec, 1), updot(a1, avx.sign8(bv, avec1)), cv[j][1]);
                    cv[j][2] = @mulAdd(F8, bcast(dvec, 2), updot(a2, avx.sign8(bv, avec2)), cv[j][2]);
                    cv[j][3] = @mulAdd(F8, bcast(dvec, 3), updot(a3, avx.sign8(bv, avec3)), cv[j][3]);
                }
            }
            inline for (0..RN) |j| inline for (0..4) |i| {
                self.c[self.ldc * (jj + j) + (ii + i)] = avx.hsum(cv[j][i]);
            };
        }
    }

    noinline fn gemmMx4(self: *const TinyBlasQ0, comptime RM: usize, m0: usize, m: usize, n0: usize, n: usize) void {
        const ytiles = (m - m0) / RM;
        const xtiles = (n - n0) / 4;
        const tiles = xtiles * ytiles;
        const duty = (tiles + self.nth - 1) / self.nth;
        const start = duty * self.ith;
        const end = @min(start + duty, tiles);
        var job = start;
        while (job < end) : (job += 1) {
            const ii = m0 + job / xtiles * RM;
            const jj = n0 + job % xtiles * 4;
            var cv: [4][RM]F8 = undefined;
            inline for (0..4) |j| inline for (0..RM) |i| {
                cv[j][i] = @splat(0);
            };
            const a = self.a;
            const b = self.b;
            for (0..self.k) |l| {
                const db = delta4(b, self.ldb, jj, l);
                const bvec0 = avx.loadB(&b[self.ldb * (jj + 0) + l].qs);
                const bvec1 = avx.loadB(&b[self.ldb * (jj + 1) + l].qs);
                const bvec2 = avx.loadB(&b[self.ldb * (jj + 2) + l].qs);
                const bvec3 = avx.loadB(&b[self.ldb * (jj + 3) + l].qs);
                inline for (0..RM) |i| {
                    const al = &a[self.lda * (ii + i) + l];
                    const da: F4 = @splat(avx.fp16(al.d));
                    const dvec = da * db;
                    const av = avx.loadB(&al.qs);
                    const u = avx.sign8(av, av);
                    cv[0][i] = @mulAdd(F8, bcast(dvec, 0), updot(u, avx.sign8(bvec0, av)), cv[0][i]);
                    cv[1][i] = @mulAdd(F8, bcast(dvec, 1), updot(u, avx.sign8(bvec1, av)), cv[1][i]);
                    cv[2][i] = @mulAdd(F8, bcast(dvec, 2), updot(u, avx.sign8(bvec2, av)), cv[2][i]);
                    cv[3][i] = @mulAdd(F8, bcast(dvec, 3), updot(u, avx.sign8(bvec3, av)), cv[3][i]);
                }
            }
            inline for (0..4) |j| inline for (0..RM) |i| {
                self.c[self.ldc * (jj + j) + (ii + i)] = avx.hsum(cv[j][i]);
            };
        }
    }

    noinline fn gemm(self: *const TinyBlasQ0, comptime RM: usize, comptime RN: usize, m0: usize, m: usize, n0: usize, n: usize) void {
        const ytiles = (m - m0) / RM;
        const xtiles = (n - n0) / RN;
        const tiles = xtiles * ytiles;
        const duty = (tiles + self.nth - 1) / self.nth;
        const start = duty * self.ith;
        const end = @min(start + duty, tiles);
        var job = start;
        while (job < end) : (job += 1) {
            const ii = m0 + job / xtiles * RM;
            const jj = n0 + job % xtiles * RN;
            var cv: [RN][RM]F8 = undefined;
            inline for (0..RN) |j| inline for (0..RM) |i| {
                cv[j][i] = @splat(0);
            };
            for (0..self.k) |l| {
                inline for (0..RN) |j| inline for (0..RM) |i| {
                    const al = &self.a[self.lda * (ii + i) + l];
                    const bl = &self.b[self.ldb * (jj + j) + l];
                    const av = avx.loadB(&al.qs);
                    const t = updot(avx.sign8(av, av), avx.sign8(avx.loadB(&bl.qs), av));
                    cv[j][i] = @mulAdd(F8, avx.splat(avx.fp16(al.d) * avx.fp16(bl.d)), t, cv[j][i]);
                };
            }
            inline for (0..RN) |j| inline for (0..RM) |i| {
                self.c[self.ldc * (jj + j) + (ii + i)] = avx.hsum(cv[j][i]);
            };
        }
    }
};
