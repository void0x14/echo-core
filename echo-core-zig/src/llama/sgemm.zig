//! Port: ggml-cpu/llamafile/sgemm.cpp (AVX2, VECTOR_REGISTERS == 16).
//! tinyBLAS<8, __m256, __m256, TA, TB, float> ve tinyBLAS_Q0_AVX<block_q8_0, block_q8_0, float>.
const std = @import("std");
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

/// tinyBLAS_Q0_AVX::gemm4xN<2> iç döngüsü (GCC -O3 çıktısıyla aynı komut biçimleri ve aynı FP sırası).
/// LLVM |a| değerlerini CSE edip 4 akümülatörü stack'e spill ediyordu; burada |a| her (i, j) için
/// bellekten okunur, 8 akümülatör register'da kalır. Bir bloğun boyu 34 bayt: d (u16) + qs (32 bayt).
const gemm4x2_asm = blk: {
    var t: []const u8 =
        \\ vmovups 0x00(%[cv]), %%ymm8
        \\ vmovups 0x20(%[cv]), %%ymm9
        \\ vmovups 0x40(%[cv]), %%ymm10
        \\ vmovups 0x60(%[cv]), %%ymm11
        \\ vmovups 0x80(%[cv]), %%ymm12
        \\ vmovups 0xa0(%[cv]), %%ymm13
        \\ vmovups 0xc0(%[cv]), %%ymm14
        \\ vmovups 0xe0(%[cv]), %%ymm15
        \\ vmovdqu (%[ones]), %%ymm7
        \\ .p2align 4
        \\ 1:
        \\ movzwl (%[a3],%[i]), %%eax
        \\ shl $48, %%rax
        \\ movzwl (%[a2],%[i]), %%ecx
        \\ shl $32, %%rcx
        \\ or %%rcx, %%rax
        \\ movzwl (%[a1],%[i]), %%ecx
        \\ shl $16, %%ecx
        \\ or %%rcx, %%rax
        \\ movzwl (%[a0],%[i]), %%ecx
        \\ or %%rcx, %%rax
        \\ vmovq %%rax, %%xmm2
        \\ vcvtph2ps %%xmm2, %%xmm2
        \\
    ;
    const imm = [4][]const u8{ "$0x00", "$0x55", "$0xaa", "$0xff" };
    for (0..2) |j| {
        const bn = if (j == 0) "b0" else "b1";
        t = t ++ "movzwl (%[" ++ bn ++ "],%[i]), %%eax\n" ++
            "vbroadcastss (%[tab],%%rax,4), %%xmm1\n" ++
            "vmulps %%xmm2, %%xmm1, %%xmm1\n" ++
            "vperm2f128 $0, %%ymm1, %%ymm1, %%ymm1\n" ++
            "vmovdqu 2(%[" ++ bn ++ "],%[i]), %%ymm0\n";
        for (0..4) |i| {
            const an = "a" ++ [1]u8{'0' + i};
            const acc = "%%ymm" ++ std.fmt.comptimePrint("{d}", .{8 + j * 4 + i});
            t = t ++ "vmovdqu 2(%[" ++ an ++ "],%[i]), %%ymm3\n" ++
                "vpabsb %%ymm3, %%ymm4\n" ++
                "vpsignb %%ymm3, %%ymm0, %%ymm3\n" ++
                "vpmaddubsw %%ymm3, %%ymm4, %%ymm3\n" ++
                "vpmaddwd %%ymm3, %%ymm7, %%ymm3\n" ++
                "vcvtdq2ps %%ymm3, %%ymm3\n" ++
                "vpermilps " ++ imm[i] ++ ", %%ymm1, %%ymm4\n" ++
                "vfmadd231ps %%ymm3, %%ymm4, " ++ acc ++ "\n";
        }
    }
    t = t ++
        \\ add $34, %[i]
        \\ cmp %[i], %[end]
        \\ jne 1b
        \\ vmovups %%ymm8, 0x00(%[cv])
        \\ vmovups %%ymm9, 0x20(%[cv])
        \\ vmovups %%ymm10, 0x40(%[cv])
        \\ vmovups %%ymm11, 0x60(%[cv])
        \\ vmovups %%ymm12, 0x80(%[cv])
        \\ vmovups %%ymm13, 0xa0(%[cv])
        \\ vmovups %%ymm14, 0xc0(%[cv])
        \\ vmovups %%ymm15, 0xe0(%[cv])
    ;
    break :blk t;
};

const ones16: [16]i16 = @splat(1);

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

    /// updot(_mm256_sign_epi8(a, a), _mm256_sign_epi8(b, a)). sign(a,a) == abs(a); a bellekten
    /// her seferinde okunur: LLVM aksi halde a'ları tutmak için akümülatörleri stack'e spill ediyor.
    inline fn updot(ap: *const [32]i8, bv: avx.B32) F8 {
        const av = avx.loadB(ap);
        const res = avx.madd(@splat(1), avx.maddubs(@bitCast(@abs(av)), avx.sign8(bv, av)));
        return @floatFromInt(res);
    }
    /// _mm_cvtph_ps(_mm_set_epi64x(0, delta))
    inline fn delta4(a: [*]const q.BlockQ80, stride: usize, base: usize, l: usize) F4 {
        const d: u64 = @as(u64, a[stride * (base + 3) + l].d) << 48 |
            @as(u64, a[stride * (base + 2) + l].d) << 32 |
            @as(u64, a[stride * (base + 1) + l].d) << 16 |
            a[stride * (base + 0) + l].d;
        const w: @Vector(2, u64) = .{ d, 0 };
        return avx.fp16x4(@bitCast(w));
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
            if (RN == 2) {
                var i_off: usize = 0;
                const i_end: usize = self.k * @sizeOf(q.BlockQ80);
                asm volatile (gemm4x2_asm
                    : [i] "+r" (i_off),
                    : [a0] "r" (a + self.lda * (ii + 0)),
                      [a1] "r" (a + self.lda * (ii + 1)),
                      [a2] "r" (a + self.lda * (ii + 2)),
                      [a3] "r" (a + self.lda * (ii + 3)),
                      [b0] "r" (b + self.ldb * (jj + 0)),
                      [b1] "r" (b + self.ldb * (jj + 1)),
                      [tab] "r" (&avx.fp16_table),
                      [ones] "r" (&ones16),
                      [end] "r" (i_end),
                      [cv] "r" (&cv),
                    : .{ .rax = true, .rcx = true, .ymm0 = true, .ymm1 = true, .ymm2 = true, .ymm3 = true, .ymm4 = true, .ymm7 = true, .ymm8 = true, .ymm9 = true, .ymm10 = true, .ymm11 = true, .ymm12 = true, .ymm13 = true, .ymm14 = true, .ymm15 = true, .memory = true, .cc = true });
            } else for (0..self.k) |l| {
                const da = delta4(a, self.lda, ii, l);
                const a0 = &a[self.lda * (ii + 0) + l].qs;
                const a1 = &a[self.lda * (ii + 1) + l].qs;
                const a2 = &a[self.lda * (ii + 2) + l].qs;
                const a3 = &a[self.lda * (ii + 3) + l].qs;
                inline for (0..RN) |j| {
                    const bl = &b[self.ldb * (jj + j) + l];
                    const db: F4 = @splat(avx.fp16_table[bl.d]);
                    const dvec = da * db;
                    const bv = avx.loadB(&bl.qs);
                    cv[j][0] = @mulAdd(F8, bcast(dvec, 0), updot(a0, bv), cv[j][0]);
                    cv[j][1] = @mulAdd(F8, bcast(dvec, 1), updot(a1, bv), cv[j][1]);
                    cv[j][2] = @mulAdd(F8, bcast(dvec, 2), updot(a2, bv), cv[j][2]);
                    cv[j][3] = @mulAdd(F8, bcast(dvec, 3), updot(a3, bv), cv[j][3]);
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
                const b0 = avx.loadB(&b[self.ldb * (jj + 0) + l].qs);
                const b1 = avx.loadB(&b[self.ldb * (jj + 1) + l].qs);
                const b2 = avx.loadB(&b[self.ldb * (jj + 2) + l].qs);
                const b3 = avx.loadB(&b[self.ldb * (jj + 3) + l].qs);
                inline for (0..RM) |i| {
                    const al = &a[self.lda * (ii + i) + l];
                    const da: F4 = @splat(avx.fp16_table[al.d]);
                    const dvec = da * db;
                    cv[0][i] = @mulAdd(F8, bcast(dvec, 0), updot(&al.qs, b0), cv[0][i]);
                    cv[1][i] = @mulAdd(F8, bcast(dvec, 1), updot(&al.qs, b1), cv[1][i]);
                    cv[2][i] = @mulAdd(F8, bcast(dvec, 2), updot(&al.qs, b2), cv[2][i]);
                    cv[3][i] = @mulAdd(F8, bcast(dvec, 3), updot(&al.qs, b3), cv[3][i]);
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
                    const t = updot(&al.qs, avx.loadB(&bl.qs));
                    cv[j][i] = @mulAdd(F8, avx.splat(avx.fp16_table[al.d] * avx.fp16_table[bl.d]), t, cv[j][i]);
                };
            }
            inline for (0..RN) |j| inline for (0..RM) |i| {
                self.c[self.ldc * (jj + j) + (ii + i)] = avx.hsum(cv[j][i]);
            };
        }
    }
};
