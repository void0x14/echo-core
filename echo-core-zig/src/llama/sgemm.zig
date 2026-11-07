//! Port: ggml-cpu/llamafile/sgemm.cpp (AVX2, VECTOR_REGISTERS == 16).
//! tinyBLAS<8, __m256, __m256, TA, TB, float> ve tinyBLAS_Q0_AVX<block_q8_0, block_q8_0, float>.
const std = @import("std");
const avx = @import("avx.zig");
const q = @import("quants.zig");
const Params = @import("pool.zig").Params;
const Pool = @import("pool.zig").Pool;
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

/// tinyBLAS_Q0_AVX::gemm4xN<2> iç döngüsü: aynı aritmetik ve FP sırası, 16 register'a sığan plan.
/// LLVM |a| değerlerini CSE edip akümülatörleri stack'e spill ediyordu. Burada 8 akümülatör
/// (ymm8-15) register'da; i dış döngü olduğu için a ve |a| satır başına bir kez hesaplanır,
/// b0, b1 ve delta çarpımları register'da kalır, 1 sabiti bellekten okunur.
/// Bir bloğun boyu 34 bayt: d (u16) + qs (32 bayt).
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
        \\ vmovq %%rax, %%xmm4
        \\ vcvtph2ps %%xmm4, %%xmm4
        \\ movzwl (%[b0],%[i]), %%eax
        \\ vbroadcastss (%[tab],%%rax,4), %%xmm2
        \\ movzwl (%[b1],%[i]), %%eax
        \\ vbroadcastss (%[tab],%%rax,4), %%xmm3
        \\ vmulps %%xmm4, %%xmm2, %%xmm2
        \\ vmulps %%xmm4, %%xmm3, %%xmm3
        \\ vperm2f128 $0, %%ymm2, %%ymm2, %%ymm2
        \\ vperm2f128 $0, %%ymm3, %%ymm3, %%ymm3
        \\ vmovdqu 2(%[b0],%[i]), %%ymm0
        \\ vmovdqu 2(%[b1],%[i]), %%ymm1
        \\
    ;
    const imm = [4][]const u8{ "$0x00", "$0x55", "$0xaa", "$0xff" };
    for (0..4) |i| {
        const an = "a" ++ [1]u8{'0' + i};
        t = t ++ "vmovdqu 2(%[" ++ an ++ "],%[i]), %%ymm4\n" ++
            "vpabsb %%ymm4, %%ymm5\n";
        for (0..2) |j| {
            const bv = if (j == 0) "%%ymm0" else "%%ymm1";
            const dv = if (j == 0) "%%ymm2" else "%%ymm3";
            const acc = "%%ymm" ++ std.fmt.comptimePrint("{d}", .{8 + j * 4 + i});
            t = t ++ "vpsignb %%ymm4, " ++ bv ++ ", %%ymm6\n" ++
                "vpmaddubsw %%ymm6, %%ymm5, %%ymm6\n" ++
                "vpmaddwd (%[ones]), %%ymm6, %%ymm6\n" ++
                "vcvtdq2ps %%ymm6, %%ymm6\n" ++
                "vpermilps " ++ imm[i] ++ ", " ++ dv ++ ", %%ymm7\n" ++
                "vfmadd231ps %%ymm6, %%ymm7, " ++ acc ++ "\n";
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
    pool: *Pool,
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
        if (RN == 2) {
            // mnpack'te RN=2 yalnız ilk çağrıdır, sayaç ona ait: satır bloğu dinamik alınır, blok L2'de kalırken tüm x geçer.
            const rb = 4;
            const nblk = (ytiles + rb - 1) / rb;
            var cur: i64 = @intCast(self.ith);
            while (cur < nblk) {
                const b0: usize = @as(usize, @intCast(cur)) * rb;
                for (0..xtiles) |xt| {
                    for (b0..@min(b0 + rb, ytiles)) |yt| self.tile4xN(RN, m0 + yt * 4, n0 + xt * RN);
                }
                cur = self.pool.chunkAdd(1);
            }
            return;
        }
        const tiles = xtiles * ytiles;
        const duty = (tiles + self.nth - 1) / self.nth;
        const start = duty * self.ith;
        const end = @min(start + duty, tiles);
        var job = start;
        while (job < end) : (job += 1) self.tile4xN(RN, m0 + job / xtiles * 4, n0 + job % xtiles * RN);
    }

    inline fn tile4xN(self: *const TinyBlasQ0, comptime RN: usize, ii: usize, jj: usize) void {
        {
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
                    : .{ .rax = true, .rcx = true, .ymm0 = true, .ymm1 = true, .ymm2 = true, .ymm3 = true, .ymm4 = true, .ymm5 = true, .ymm6 = true, .ymm7 = true, .ymm8 = true, .ymm9 = true, .ymm10 = true, .ymm11 = true, .ymm12 = true, .ymm13 = true, .ymm14 = true, .ymm15 = true, .memory = true, .cc = true });
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

/// H20: Q8_0_R8 — 8 satır iç içe. ymm k'nın r. 32-bit şeridi = satır r'nin 4k..4k+3 baytları.
/// H24: ölçekler f16, bloklar çiftli. H30: aktivasyon 7 bit, u8'e kaydırılmış (a + 64); çift başına yalnız
/// maddubs(a_u8, w) + vpaddw. Düzeltme 64·Σw her i16 şeridi için önceden hesaplanır (mod 2^16).
/// Çift (2p, 2p+1) = [16B ölçek][16B ölçek][32B düz.][32B düz.][256B quant][256B quant] → blok 304 bayt.
pub const R8_BLOCK = 304;
const R8_NC = 4;
const R8_ASM = true;

inline fn r8Scale(gbase: [*]const u8, l: usize) [*]const u8 {
    return gbase + (l / 2) * 2 * R8_BLOCK + (l % 2) * 16;
}
inline fn r8Corr(gbase: [*]const u8, l: usize) [*]const u8 {
    return gbase + (l / 2) * 2 * R8_BLOCK + 32 + (l % 2) * 32;
}
inline fn r8Quant(gbase: [*]const u8, l: usize) [*]const u8 {
    return gbase + (l / 2) * 2 * R8_BLOCK + 96 + (l % 2) * 256;
}

pub fn repackQ80x8(src: [*]const q.BlockQ80, dst: [*]u8, m: usize, nb: usize) void {
    std.debug.assert(nb % 2 == 0);
    for (0..m / 8) |g| {
        const gbase = dst + g * nb * R8_BLOCK;
        for (0..nb) |l| {
            const sc: [*]u16 = @ptrCast(@alignCast(@constCast(r8Scale(gbase, l))));
            const cr: [*]i16 = @ptrCast(@alignCast(@constCast(r8Corr(gbase, l))));
            const qd: [*]u8 = @constCast(r8Quant(gbase, l));
            for (0..8) |r| {
                const blk = &src[(g * 8 + r) * nb + l];
                sc[r] = blk.d;
                const qs = std.mem.asBytes(&blk.qs);
                for (0..8) |k| @memcpy(qd[32 * k + 4 * r ..][0..4], qs[4 * k ..][0..4]);
                // şerit 2r: baytlar 4k, 4k+1; şerit 2r+1: 4k+2, 4k+3 (tüm k)
                var c0: i32 = 0;
                var c1: i32 = 0;
                for (0..8) |k| {
                    c0 += @as(i32, blk.qs[4 * k]) + blk.qs[4 * k + 1];
                    c1 += @as(i32, blk.qs[4 * k + 2]) + blk.qs[4 * k + 3];
                }
                cr[2 * r] = @truncate(c0 * R8_AOFF);
                cr[2 * r + 1] = @truncate(c1 * R8_AOFF);
            }
        }
    }
}

/// H30: R8 yolu aktivasyonu 7 bit nicemlenir (|a| ≤ 63) ve u8'e (a + 64) kaydırılır.
pub const R8_AOFF = 64;
pub fn quantR8(x: [*]const f32, b: [*]q.BlockQ80, k: usize) void {
    q.quantizeRowQ(63.0, x, b, k);
    for (b[0 .. k / 32]) |*blk| {
        const v: avx.B32 = @bitCast(blk.qs);
        blk.qs = @bitCast(v +% @as(avx.B32, @splat(R8_AOFF)));
    }
}

/// H34: R8 4 sütunluk çekirdek, elle asm (derleyici 8 ağırlık + 4 akümülatörü yığına döküyordu).
/// H36: yazılım boru hattı — bloğun sonu (subw/madd/cvt/ölçek/fma) bir sonraki bloğun iç döngüsüne
/// serpiştirilir; iki i16 toplam seti dönüşümlü (A: ymm12-15, B: ymm2,3,4,7).
/// ymm0 w, ymm1 av, ymm5 satır ölçeği, ymm6 geçici, ymm8-11 akümülatör. Döngü blok çifti başına (608 bayt).
const r8_bn = [4][]const u8{ "%[b0]", "%[b1]", "%[b2]", "%[b3]" };
const r8_SA = [4]u8{ 12, 13, 14, 15 };
const r8_SB = [4]u8{ 2, 3, 4, 7 };

fn r8Inner(comptime set: [4]u8, comptime h: usize, comptime k: usize) []const u8 {
    var t: []const u8 = std.fmt.comptimePrint("vmovdqa {d}(%[w]), %%ymm0\n", .{96 + h * 256 + 32 * k});
    for (0..4) |c| {
        t = t ++ std.fmt.comptimePrint("vpbroadcastd {d}({s}), %%ymm1\n", .{ h * 34 + 2 + 4 * k, r8_bn[c] });
        t = t ++ if (k == 0)
            std.fmt.comptimePrint("vpmaddubsw %%ymm0, %%ymm1, %%ymm{d}\n", .{set[c]})
        else
            std.fmt.comptimePrint("vpmaddubsw %%ymm0, %%ymm1, %%ymm1\nvpaddw %%ymm1, %%ymm{0d}, %%ymm{0d}\n", .{set[c]});
    }
    return t;
}

/// Blok sonu, parça p: -1 satır ölçeği, 0..3 sütun c'nin i16→f32 kısmı, 4..7 sütun c'nin ölçek+fma kısmı.
fn r8End(comptime set: [4]u8, comptime h: i32, comptime p: i32, comptime wo: i32, comptime bo: i32) []const u8 {
    if (p < 0) return std.fmt.comptimePrint("vcvtph2ps {d}(%[w]), %%ymm5\n", .{wo + h * 16});
    if (p < 4) {
        const r = set[@intCast(p)];
        return std.fmt.comptimePrint("vpsubw {1d}(%[w]), %%ymm{0d}, %%ymm{0d}\nvpmaddwd (%[ones]), %%ymm{0d}, %%ymm{0d}\nvcvtdq2ps %%ymm{0d}, %%ymm{0d}\n", .{ r, wo + 32 + h * 32 });
    }
    const c: usize = @intCast(p - 4);
    return std.fmt.comptimePrint("movzwl {1d}({2s}), %%eax\nvbroadcastss (%[tab],%%rax,4), %%ymm6\nvmulps %%ymm5, %%ymm6, %%ymm6\nvfmadd231ps %%ymm6, %%ymm{0d}, %%ymm{3d}\n", .{ set[c], bo + h * 34, r8_bn[c], 8 + c });
}

fn r8Block(comptime si: [4]u8, comptime hi: usize, comptime se: [4]u8, comptime he: i32, comptime wo: i32, comptime bo: i32) []const u8 {
    var t: []const u8 = r8End(se, he, -1, wo, bo);
    for (0..8) |k| t = t ++ r8Inner(si, hi, k) ++ r8End(se, he, @intCast(k), wo, bo);
    return t;
}

const r8x4_asm = blk: {
    @setEvalBranchQuota(400000);
    var t: []const u8 =
        \\ vxorps %%ymm8, %%ymm8, %%ymm8
        \\ vxorps %%ymm9, %%ymm9, %%ymm9
        \\ vxorps %%ymm10, %%ymm10, %%ymm10
        \\ vxorps %%ymm11, %%ymm11, %%ymm11
        \\
    ;
    for (0..8) |k| t = t ++ r8Inner(r8_SA, 0, k);
    t = t ++ "dec %[cnt]\njz 2f\n.p2align 4\n1:\n";
    t = t ++ r8Block(r8_SB, 1, r8_SA, 0, 0, 0);
    t = t ++ "add $68, %[b0]\nadd $68, %[b1]\nadd $68, %[b2]\nadd $68, %[b3]\nadd $608, %[w]\n";
    t = t ++ r8Block(r8_SA, 0, r8_SB, 1, -608, -68);
    t = t ++ "dec %[cnt]\njnz 1b\n2:\n";
    t = t ++ r8Block(r8_SB, 1, r8_SA, 0, 0, 0);
    t = t ++ r8End(r8_SB, 1, -1, 0, 0);
    for (0..8) |p| t = t ++ r8End(r8_SB, 1, p, 0, 0);
    t = t ++
        \\ vmovups %%ymm8, 0x00(%[cv])
        \\ vmovups %%ymm9, 0x20(%[cv])
        \\ vmovups %%ymm10, 0x40(%[cv])
        \\ vmovups %%ymm11, 0x60(%[cv])
        \\
    ;
    break :blk t;
};

pub const GemmR8 = struct {
    pool: *Pool,
    nb: usize,
    a: [*]const u8,
    b: [*]const q.BlockQ80,
    ldb: usize,
    c: [*]f32,
    ldc: usize,
    ith: usize,

    pub const GB = 4; // 32 satır / iş parçası

    pub fn chunks(m: usize) usize {
        return (m / 8 + GB - 1) / GB;
    }

    pub fn matmul(self: *const GemmR8, m: usize, n: usize) void {
        const nblk = chunks(m);
        var cur: i64 = @intCast(self.ith);
        while (cur < nblk) {
            self.chunk(@intCast(cur), m, n);
            cur = self.pool.chunkAdd(1);
        }
    }

    pub fn chunk(self: *const GemmR8, cb: usize, m: usize, n: usize) void {
        const groups = m / 8;
        const g0 = cb * GB;
        const g1 = @min(g0 + GB, groups);
        for (g0..g1) |g| {
            var j: usize = 0;
            if (R8_ASM) while (j + 4 <= n) : (j += 4) self.tileAsm(g, j);
            while (j + R8_NC <= n) : (j += R8_NC) self.tile(R8_NC, g, j);
            while (j < n) : (j += 1) self.tile(1, g, j);
        }
    }

    fn tileAsm(self: *const GemmR8, g: usize, j0: usize) void {
        var cv: [4]F8 align(32) = undefined;
        var w = self.a + g * self.nb * R8_BLOCK;
        var b0: [*]const q.BlockQ80 = self.b + self.ldb * j0;
        var b1 = b0 + self.ldb;
        var b2 = b1 + self.ldb;
        var b3 = b2 + self.ldb;
        var cnt = self.nb / 2;
        asm volatile (r8x4_asm
            : [w] "+r" (w),
              [b0] "+r" (b0),
              [b1] "+r" (b1),
              [b2] "+r" (b2),
              [b3] "+r" (b3),
              [cnt] "+r" (cnt),
            : [tab] "r" (&avx.fp16_table),
              [ones] "r" (&ones16),
              [cv] "r" (&cv),
            : .{ .rax = true, .ymm0 = true, .ymm1 = true, .ymm2 = true, .ymm3 = true, .ymm7 = true, .ymm4 = true, .ymm5 = true, .ymm6 = true, .ymm8 = true, .ymm9 = true, .ymm10 = true, .ymm11 = true, .ymm12 = true, .ymm13 = true, .ymm14 = true, .ymm15 = true, .memory = true, .cc = true });
        inline for (0..4) |c| avx.storeF(self.c + self.ldc * (j0 + c) + g * 8, cv[c]);
    }

    inline fn tile(self: *const GemmR8, comptime NC: usize, g: usize, j0: usize) void {
        var acc: [NC]F8 = @splat(@splat(0));
        const ones: avx.W16 = @splat(1);
        const gbase = self.a + g * self.nb * R8_BLOCK;
        for (0..self.nb) |l| {
            const base = r8Quant(gbase, l);
            const sa: F8 = @floatCast(@as(@Vector(8, f16), @as(*align(16) const [8]f16, @ptrCast(@alignCast(r8Scale(gbase, l)))).*));
            const corr: avx.W16 = @as(*align(32) const avx.W16, @ptrCast(@alignCast(r8Corr(gbase, l)))).*;
            var s16: [NC]avx.W16 = undefined;
            inline for (0..8) |k| {
                const w = avx.loadB(base + 32 * k);
                inline for (0..NC) |c| {
                    const blk = &self.b[self.ldb * (j0 + c) + l];
                    const a4: i32 = @as(*align(1) const i32, @ptrCast(&blk.qs[4 * k])).*;
                    const av: avx.B32 = @bitCast(@as(avx.I8, @splat(a4)));
                    const p = avx.maddubs(av, w);
                    s16[c] = if (k == 0) p else s16[c] +% p;
                }
            }
            var isum: [NC]avx.I8 = undefined;
            inline for (0..NC) |c| isum[c] = avx.madd(ones, s16[c] -% corr);
            inline for (0..NC) |c| {
                const db: F8 = @splat(avx.fp16_table[self.b[self.ldb * (j0 + c) + l].d]);
                acc[c] = @mulAdd(F8, @floatFromInt(isum[c]), sa * db, acc[c]);
            }
        }
        inline for (0..NC) |c| avx.storeF(self.c + self.ldc * (j0 + c) + g * 8, acc[c]);
    }
};
