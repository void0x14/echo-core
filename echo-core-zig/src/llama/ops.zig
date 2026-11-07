//! Port: ggml-cpu/ggml-cpu.c (mul_mat), ggml-cpu/ops.cpp (rms_norm, flash_attn_ext, gated_delta_net,
//! ssm_conv, rope, concat, ...). Thread bölme ve chunk mantığı ggml ile aynıdır.
const std = @import("std");
const avx = @import("avx.zig");
const vec = @import("vec.zig");
const q = @import("quants.zig");
const sgemm = @import("sgemm.zig");
const repack = @import("repack.zig");
const Params = @import("pool.zig").Params;

pub const Type = enum {
    f32,
    f16,
    q8_0,
    q4_k,
    q5_k,
    q6_k,
    /// ggml_repack_get_optimal_repack_type: Q4_K ve ne[1] % 8 == 0 ise block_q4_Kx8
    q4_k_8x8,
};

pub const Mat = struct {
    data: [*]const u8,
    type: Type,
    /// ne[0]: satır başına eleman
    k: usize,
    /// ne[1]: satır sayısı
    m: usize,
    /// nb[1]: satır adımı (bayt)
    nb1: usize,
    /// H20: Q8_0_R8 kopya (batch yolu); tek token yolu `data` kullanır.
    rep: ?[*]const u8 = null,
};

pub fn rowSize(t: Type, k: usize) usize {
    return switch (t) {
        .f32 => k * 4,
        .f16 => k * 2,
        .q8_0 => k / 32 * 34,
        .q4_k, .q4_k_8x8 => k / 256 * 144,
        .q5_k => k / 256 * 176,
        .q6_k => k / 256 * 210,
    };
}

/// type_traits_cpu[t].vec_dot_type satır boyutu
fn dotRowSize(t: Type, k: usize) usize {
    return switch (t) {
        .f32 => k * 4,
        .f16 => k * 2,
        .q8_0 => k / 32 * 34,
        .q4_k, .q4_k_8x8, .q5_k, .q6_k => k / 256 * 292,
    };
}

inline fn vecDot(comptime T: Type, n: usize, s: *f32, x: [*]const u8, y: [*]const u8) void {
    s.* = switch (T) {
        .f32 => vec.dotF32(n, @ptrCast(@alignCast(x)), @ptrCast(@alignCast(y))),
        .f16 => vec.dotF16(n, @ptrCast(@alignCast(x)), @ptrCast(@alignCast(y))),
        .q8_0 => q.dotQ80Q80(n, @ptrCast(@alignCast(x)), @ptrCast(@alignCast(y))),
        .q4_k => q.dotQ4KQ8K(n, @ptrCast(@alignCast(x)), @ptrCast(@alignCast(y))),
        .q5_k => q.dotQ5KQ8K(n, @ptrCast(@alignCast(x)), @ptrCast(@alignCast(y))),
        .q6_k => q.dotQ6KQ8K(n, @ptrCast(@alignCast(x)), @ptrCast(@alignCast(y))),
        .q4_k_8x8 => @compileError("repack yolu vec_dot kullanmaz"),
    };
}

/// ggml_compute_forward_mul_mat_one_chunk
fn mulMatOneChunk(
    comptime T: Type,
    a: Mat,
    wdata: [*]const u8,
    row_size: usize,
    y: [*]f32,
    ir0_start: usize,
    ir0_end: usize,
    ir1_start: usize,
    ir1_end: usize,
) void {
    if (ir0_start >= ir0_end or ir1_start >= ir1_end) return;
    const blck = 16;
    var tmp: [32]f32 = undefined;
    var iir1 = ir1_start;
    while (iir1 < ir1_end) : (iir1 += blck) {
        var iir0 = ir0_start;
        while (iir0 < ir0_end) : (iir0 += blck) {
            var ir1 = iir1;
            while (ir1 < iir1 + blck and ir1 < ir1_end) : (ir1 += 1) {
                const src1_col = wdata + ir1 * row_size;
                const dst_col = y + ir1 * a.m;
                var ir0 = iir0;
                while (ir0 < iir0 + blck and ir0 < ir0_end) : (ir0 += 1) {
                    vecDot(T, a.k, &tmp[ir0 - iir0], a.data + ir0 * a.nb1, src1_col);
                }
                const n = @min(iir0 + blck, ir0_end) - iir0;
                @memcpy(dst_col[iir0..][0..n], tmp[0..n]);
            }
        }
    }
}

fn mulMatChunks(comptime T: Type, p: *const Params, a: Mat, wdata: [*]const u8, row_size: usize, n: usize, y: [*]f32) void {
    const nr0 = a.m;
    const nr1 = n;
    const chunk_size: usize = if (nr0 == 1 or nr1 == 1) 64 else 16;
    var nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
    var nchunk1 = (nr1 + chunk_size - 1) / chunk_size;
    if (nchunk0 * nchunk1 < p.nth * 4) {
        nchunk0 = if (nr0 > nr1) p.nth else 1;
        nchunk1 = if (nr0 > nr1) 1 else p.nth;
    }
    const dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    const dr1 = (nr1 + nchunk1 - 1) / nchunk1;
    const total = nchunk0 * nchunk1;
    var current: i64 = p.ith;
    while (current < total) {
        const c: usize = @intCast(current);
        const ith0 = c % nchunk0;
        const ith1 = c / nchunk0;
        const ir0_start = dr0 * ith0;
        const ir0_end = @min(ir0_start + dr0, nr0);
        const ir1_start = dr1 * ith1;
        const ir1_end = @min(ir1_start + dr1, nr1);
        mulMatOneChunk(T, a, wdata, row_size, y, ir0_start, ir0_end, ir1_start, ir1_end);
        if (p.nth >= total) break;
        current = p.pool.chunkAdd(1);
    }
}

/// ggml_compute_forward_mul_mat: y = a * x; x satırları `a.k` uzunlukta ve bitişik.
pub fn mulMat(p: *const Params, a: Mat, x: [*]const f32, n: usize, y: [*]f32) void {
    switch (a.type) {
        .f32 => {
            if (n >= 2) {
                const tb = sgemm.TinyBlas(f32){
                    .p = p,
                    .k = a.k,
                    .a = @ptrCast(@alignCast(a.data)),
                    .lda = a.nb1 / 4,
                    .b = x,
                    .ldb = a.k,
                    .c = y,
                    .ldc = a.m,
                };
                if (tb.matmul(a.m, n)) return;
            }
            if (p.ith == 0) p.pool.chunkSet(@intCast(p.nth));
            p.barrier();
            mulMatChunks(.f32, p, a, @ptrCast(x), a.k * 4, n, y);
            return;
        },
        .q4_k_8x8 => return repack.forwardMulMat(p, a, x, n, y),
        else => {},
    }

    const row_size = dotRowSize(a.type, a.k);
    const wdata = p.wdata;
    const bs: usize = switch (a.type) {
        .f16 => 1,
        .q8_0 => 32,
        else => 256,
    };
    const start = (p.ith * a.k / bs) / p.nth;
    const end = ((p.ith + 1) * a.k / bs) / p.nth;
    if (end > start) {
        for (0..n) |row| {
            const src = x + row * a.k + start * bs;
            const dst = wdata + row * row_size;
            const cnt = (end - start) * bs;
            switch (a.type) {
                .f16 => vec.fp32to16Row(src, @ptrCast(@alignCast(dst + start * 2)), cnt),
                .q8_0 => q.quantizeRowQ80(src, @ptrCast(@alignCast(dst + start * 34)), cnt),
                else => q.quantizeRowQ8K(src, @ptrCast(@alignCast(dst + start * 292)), cnt),
            }
        }
    }
    if (p.ith == 0) p.pool.chunkSet(@intCast(p.nth));
    p.barrier();

    switch (a.type) {
        .f16 => {
            if (n >= 2) {
                const tb = sgemm.TinyBlas(u16){
                    .p = p,
                    .k = a.k,
                    .a = @ptrCast(@alignCast(a.data)),
                    .lda = a.nb1 / 2,
                    .b = @ptrCast(@alignCast(wdata)),
                    .ldb = a.k,
                    .c = y,
                    .ldc = a.m,
                };
                if (tb.matmul(a.m, n)) return;
            }
            mulMatChunks(.f16, p, a, wdata, row_size, n, y);
        },
        .q8_0 => {
            if (a.rep != null) {
                const g = sgemm.GemmR8{ .pool = p.pool, .nb = a.k / 32, .a = a.rep.?, .b = @ptrCast(@alignCast(wdata)), .ldb = row_size / 34, .c = y, .ldc = a.m, .ith = p.ith };
                g.matmul(a.m, n);
                return;
            }
            if (n >= 2) {
                const tb = sgemm.TinyBlasQ0{
                    .pool = p.pool,
                    .k = a.k / 32,
                    .a = @ptrCast(@alignCast(a.data)),
                    .lda = a.nb1 / 34,
                    .b = @ptrCast(@alignCast(wdata)),
                    .ldb = row_size / 34,
                    .c = y,
                    .ldc = a.m,
                    .ith = p.ith,
                    .nth = p.nth,
                };
                tb.matmul(a.m, n);
                return;
            }
            mulMatChunks(.q8_0, p, a, wdata, row_size, n, y);
        },
        .q4_k => mulMatChunks(.q4_k, p, a, wdata, row_size, n, y),
        .q5_k => mulMatChunks(.q5_k, p, a, wdata, row_size, n, y),
        .q6_k => mulMatChunks(.q6_k, p, a, wdata, row_size, n, y),
        .f32, .q4_k_8x8 => unreachable,
    }
}

/// H22: aynı girdiyi kullanan Q8_0_R8 matrisleri tek quantize + tek iş havuzu + tek bariyerle.
/// Koşul sağlanmazsa sırayla mulMat (her biri sonunda bariyer).
pub fn mulMatMulti(p: *const Params, mats: []const Mat, x: [*]const f32, n: usize, ys: []const [*]f32) void {
    const k = mats[0].k;
    var ok = true;
    for (mats) |a| ok = ok and a.rep != null and a.k == k and a.type == .q8_0;
    if (!ok) {
        for (mats, ys) |a, y| {
            mulMat(p, a, x, n, y);
            p.barrier();
        }
        return;
    }
    const row_size = dotRowSize(.q8_0, k);
    const wdata = p.wdata;
    const start = (p.ith * k / 32) / p.nth;
    const end = ((p.ith + 1) * k / 32) / p.nth;
    if (end > start) {
        for (0..n) |row| q.quantizeRowQ80(x + row * k + start * 32, @ptrCast(@alignCast(wdata + row * row_size + start * 34)), (end - start) * 32);
    }
    if (p.ith == 0) p.pool.chunkSet(@intCast(p.nth));
    p.barrier();
    var total: usize = 0;
    for (mats) |a| total += sgemm.GemmR8.chunks(a.m);
    var cur: i64 = @intCast(p.ith);
    while (cur < total) {
        var c: usize = @intCast(cur);
        for (mats, ys) |a, y| {
            const nc = sgemm.GemmR8.chunks(a.m);
            if (c < nc) {
                const g = sgemm.GemmR8{ .pool = p.pool, .nb = k / 32, .a = a.rep.?, .b = @ptrCast(@alignCast(wdata)), .ldb = row_size / 34, .c = y, .ldc = a.m, .ith = p.ith };
                g.chunk(c, a.m, n);
                break;
            }
            c -= nc;
        }
        cur = p.pool.chunkAdd(1);
    }
    p.barrier();
}

/// rms_norm + mul (fused). x satırı `x_stride` adımla, y satırı `y_stride` adımla.
/// `nrow` satır, satır uzunluğu `nc`. Satırlar round-robin bölünür (ggml: i01 = ith; += nth).
/// `w` null değilse y = x*scale*w, `post` sonradan çarpan (gdn_l2_norm'daki scale node'u).
pub fn rmsNorm(
    p: *const Params,
    x_in: [*]const f32,
    x_row: usize,
    x_group: usize,
    y_in: [*]f32,
    y_row: usize,
    y_group: usize,
    nc: usize,
    n_rows: usize,
    n_groups: usize,
    eps: f32,
    w: ?[*]const f32,
    post: f32,
) void {
    for (0..n_groups) |g| {
        var r: usize = p.ith;
        while (r < n_rows) : (r += p.nth) {
            const x = x_in + g * x_group + r * x_row;
            const y = y_in + g * y_group + r * y_row;
            // ggml_float (double) toplam; SIMD ile aynı kesinlik
            var sum: f64 = 0;
            var i: usize = 0;
            var acc0: avx.D4 = @splat(0);
            var acc1: avx.D4 = @splat(0);
            while (i + 8 <= nc) : (i += 8) {
                const v = avx.loadF(x + i);
                const sq = v * v;
                const lo: avx.D4 = @floatCast(avx.low128(sq));
                const hi: avx.D4 = @floatCast(avx.high128(sq));
                acc0 += lo;
                acc1 += hi;
            }
            sum = @reduce(.Add, acc0 + acc1);
            while (i < nc) : (i += 1) sum += x[i] * x[i];
            const mean: f32 = @floatCast(sum / @as(f64, @floatFromInt(nc)));
            const scale = 1.0 / @sqrt(mean + eps);
            const vs = avx.splat(scale);
            i = 0;
            if (w) |wp| {
                while (i + 8 <= nc) : (i += 8) avx.storeF(y + i, avx.loadF(x + i) * vs * avx.loadF(wp + i));
                while (i < nc) : (i += 1) y[i] = x[i] * scale * wp[i];
            } else {
                const vp = avx.splat(post);
                while (i + 8 <= nc) : (i += 8) avx.storeF(y + i, avx.loadF(x + i) * vs * vp);
                while (i < nc) : (i += 1) y[i] = x[i] * scale * post;
            }
        }
    }
}

/// add/mul: satır başına thread bölme (binary-ops.cpp get_thread_range)
pub fn add(p: *const Params, nc: usize, nr: usize, z: [*]f32, x: [*]const f32, y: [*]const f32) void {
    const r = p.range(nr);
    for (r[0]..r[1]) |i| vec.addF32(nc, z + i * nc, x + i * nc, y + i * nc);
}
pub fn mul(p: *const Params, nc: usize, nr: usize, z: [*]f32, x: [*]const f32, y: [*]const f32) void {
    const r = p.range(nr);
    for (r[0]..r[1]) |i| vec.mulF32(nc, z + i * nc, x + i * nc, y + i * nc);
}
pub fn silu(p: *const Params, nc: usize, nr: usize, y: [*]f32, x: [*]const f32) void {
    const r = p.range(nr);
    for (r[0]..r[1]) |i| vec.silu(nc, y + i * nc, x + i * nc);
}
pub fn swiglu(p: *const Params, nc: usize, nr: usize, y: [*]f32, x: [*]const f32, g: [*]const f32) void {
    const r = p.range(nr);
    for (r[0]..r[1]) |i| vec.swiglu(nc, y + i * nc, x + i * nc, g + i * nc);
}

pub inline fn sigmoidF(x: f32) f32 {
    return 1.0 / (1.0 + avx.expf(-x));
}
inline fn softplusF(x: f32) f32 {
    return if (x > 20.0) x else avx.logf(1.0 + avx.expf(x));
}

/// alpha -> ADD(ssm_dt) -> SOFTPLUS -> MUL(ssm_a), tek geçişte (satır = token)
pub fn gateFromAlpha(p: *const Params, nc: usize, nr: usize, g: [*]f32, alpha: [*]const f32, dt: [*]const f32, a: [*]const f32) void {
    const r = p.range(nr);
    for (r[0]..r[1]) |t| {
        for (0..nc) |h| g[t * nc + h] = softplusF(alpha[t * nc + h] + dt[h]) * a[h];
    }
}
pub fn sigmoidInPlace(p: *const Params, nc: usize, nr: usize, x: [*]f32) void {
    const r = p.range(nr);
    for (r[0]..r[1]) |t| {
        for (0..nc) |h| x[t * nc + h] = sigmoidF(x[t * nc + h]);
    }
}
/// attn_gated = attn * sigmoid(gate); gate Qcur_full'un 2*head_dim adımlı ikinci yarısı.
pub fn gateAttn(p: *const Params, n_t: usize, heads: usize, dim: usize, attn: [*]f32, qfull: [*]const f32) void {
    const r = p.range(n_t);
    for (r[0]..r[1]) |t| {
        for (0..heads) |h| {
            const a = attn + (t * heads + h) * dim;
            const g = qfull + t * heads * dim * 2 + h * dim * 2 + dim;
            for (0..dim) |d| a[d] *= sigmoidF(g[d]);
        }
    }
}

const CONV_ROWS = 64;

/// CONCAT(conv_states, qkv^T) dim=0 + CPY(conv_state_last -> cache_r); satır bazında.
pub fn convConcat(p: *const Params, rows: usize, kc: usize, n_t: usize, state: [*]f32, qkv: [*]const f32, out: [*]f32, trace: ?[*]f32) void {
    const r = p.range(rows);
    const ncs = kc + n_t;
    // qkv satırları `rows` adımlı: blok içinde t dış döngüde kalır, her t aynı önbellek satırlarını okur
    var b0 = r[0];
    while (b0 < r[1]) : (b0 += CONV_ROWS) {
        const b1 = @min(b0 + CONV_ROWS, r[1]);
        for (b0..b1) |row| {
            for (0..kc) |c| out[row * ncs + c] = state[row * kc + c];
        }
        for (0..n_t) |t| {
            for (b0..b1) |row| out[row * ncs + kc + t] = qkv[t * rows + row];
        }
        for (b0..b1) |row| {
            for (0..kc) |c| state[row * kc + c] = out[row * ncs + n_t + c];
        }
        // trace: [t][row][c] = t token islendikten sonraki pencere (t+1..t+kc)
        if (trace) |tr| {
            for (0..n_t) |t| {
                for (b0..b1) |row| {
                    for (0..kc) |c| tr[(t * rows + row) * kc + c] = out[row * ncs + t + c];
                }
            }
        }
    }
}

/// ggml_compute_forward_ssm_conv_f32
pub fn ssmConv(p: *const Params, nr: usize, nc: usize, ncs: usize, n_t: usize, src: [*]const f32, c: [*]const f32, dst: [*]f32) void {
    const rg = p.range(nr);
    var b0 = rg[0];
    while (b0 < rg[1]) : (b0 += CONV_ROWS) {
        const ir = @min(CONV_ROWS, rg[1] - b0);
        for (0..n_t) |t| {
            const s = src + b0 * ncs + t;
            const cc = c + b0 * nc;
            const x = dst + b0 + t * nr;
            for (0..ir) |row| {
                var sumf: f32 = 0;
                for (0..nc) |w| sumf += s[w + row * ncs] * cc[w + row * nc];
                x[row] = sumf;
            }
        }
    }
}

/// ggml_compute_forward_gated_delta_net_f32 (K=1, GDA, n_seqs=1).
/// Durum yerinde güncellenir: ggml'deki GET_ROWS + memcpy + CPY üçlüsü aynı sonucu verir.
pub fn gatedDeltaNet(
    p: *const Params,
    s_v: usize,
    h_n: usize,
    n_t: usize,
    q_in: [*]const f32,
    k_in: [*]const f32,
    v_in: [*]const f32,
    v_stride: usize,
    g: [*]const f32,
    beta: [*]const f32,
    state: [*]f32,
    out: [*]f32,
    trace: ?[*]f32,
) void {
    const nr = h_n;
    const nth_scaled = p.nth * 4;
    const chunk_size = (nr + nth_scaled - 1) / nth_scaled;
    var nchunk = (nr + chunk_size - 1) / chunk_size;
    if (p.nth == 1 or nchunk < p.nth) nchunk = p.nth;
    if (p.ith == 0) p.pool.chunkSet(@intCast(p.nth));
    p.barrier();
    const dr = (nr + nchunk - 1) / nchunk;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(s_v)));

    var current: i64 = p.ith;
    while (current < nchunk) {
        const ir0 = dr * @as(usize, @intCast(current));
        const ir1 = @min(ir0 + dr, nr);
        var iv1 = ir0;
        while (iv1 < ir1) : (iv1 += 1) {
            const s_out = state + iv1 * s_v * s_v;
            var attn_data = out + iv1 * s_v;
            for (0..n_t) |t| {
                const q_d = q_in + t * h_n * s_v + iv1 * s_v;
                const k_d = k_in + t * h_n * s_v + iv1 * s_v;
                const v_d = v_in + t * v_stride + iv1 * s_v;
                const beta_val = beta[t * h_n + iv1];
                // satır başına tek geçiş: satır L1'de kalır, 64 KB durum üç kez dolaşılmaz
                if (trace) |tr| {
                    const dst = tr + (t * h_n + iv1) * s_v * s_v;
                    @memcpy(dst[0 .. s_v * s_v], s_out[0 .. s_v * s_v]);
                }
                const e = avx.expf(g[t * h_n + iv1]);
                for (0..s_v) |j| {
                    const row = s_out + j * s_v;
                    vec.scaleF32(s_v, row, e);
                    const sum = vec.dotF32(s_v, row, k_d);
                    vec.madF32(s_v, row, k_d, (v_d[j] - sum) * beta_val);
                    attn_data[j] = vec.dotF32(s_v, row, q_d) * scale;
                }
                attn_data += s_v * h_n;
            }
        }
        current = p.pool.chunkAdd(1);
    }
}

/// ggml_compute_forward_rope_flt + ggml_mrope_cache_init + rotate_pairs (yerinde).
/// Metin token'larında p_t = p_h = p_w, bu yüzden sektörler aynı açıyı verir; yalnız
/// döndürülen n_dims için cos/sin hesaplanır (ggml ne0 boyunca hesaplayıp atıyor).
pub fn rope(p: *const Params, x: [*]f32, ne0: usize, ne1: usize, n_t: usize, base_pos: usize, n_dims: usize, theta_scale: f32) void {
    const nr = ne1 * n_t;
    const rg = p.range(nr);
    var cache: [128]f32 = undefined;
    var last_t: usize = std.math.maxInt(usize);
    for (rg[0]..rg[1]) |ir| {
        const t = ir / ne1;
        const h = ir % ne1;
        if (last_t != t) {
            var theta: f32 = @floatFromInt(base_pos + t);
            var d: usize = 0;
            while (d < n_dims) : (d += 2) {
                cache[d] = avx.cosf(theta);
                cache[d + 1] = avx.sinf(theta);
                theta *= theta_scale;
            }
            last_t = t;
        }
        const row = x + (t * ne1 + h) * ne0;
        const n_offset = n_dims / 2;
        var d: usize = 0;
        while (d < n_dims) : (d += 2) {
            const ic = d / 2;
            const c = cache[d];
            const s = cache[d + 1];
            const x0 = row[ic];
            const x1 = row[ic + n_offset];
            // gcc -ffp-contract=fast: fmsub(x0, c, x1*s), fmadd(x0, s, x1*c)
            row[ic] = @mulAdd(f32, x0, c, -(x1 * s));
            row[ic + n_offset] = @mulAdd(f32, x0, s, x1 * c);
        }
    }
}

/// SET_ROWS: f32 -> f16 KV cache
pub fn setRowsF16(p: *const Params, nc: usize, n_t: usize, src: [*]const f32, dst: [*]u16) void {
    const r = p.range(n_t);
    for (r[0]..r[1]) |t| vec.fp32to16Row(src + t * nc, dst + t * nc, nc);
}

pub const FlashArgs = struct {
    q: [*]const f32, // {dk, h_q, n_t}
    k: [*]const u16, // {dk*h_kv, n_cells}
    v: [*]const u16,
    dst: [*]f32, // {dv, h_q, n_t}
    dk: usize,
    h_q: usize,
    h_kv: usize,
    n_t: usize,
    n_kv: usize,
    first_cell: usize,
    scale: f32,

    inline fn qRow(a: *const FlashArgs, token: usize, head: usize) [*]const f32 {
        return a.q + (token * a.h_q + head) * a.dk;
    }
    inline fn kRow(a: *const FlashArgs, cell: usize, kh: usize) [*]const u16 {
        return a.k + (cell * a.h_kv + kh) * a.dk;
    }
    inline fn vRow(a: *const FlashArgs, cell: usize, kh: usize) [*]const u16 {
        return a.v + (cell * a.h_kv + kh) * a.dk;
    }
    /// set_input_kq_mask: nedensel; hücre i, token t için görünürse true
    inline fn visible(a: *const FlashArgs, token: usize, cell: usize) bool {
        return cell <= a.first_cell + token;
    }
};

const CACHE_LINE_F32 = 16;
const Q_TILE = 64;
const KV_TILE = 64;

fn flashOneChunk(p: *const Params, a: *const FlashArgs, ir0: usize, ir1: usize, ic_start: usize, ic_end: usize, partials: ?[*]f32, partial_stride: usize) void {
    const dk = a.dk;
    const dv = a.dk;
    const rk2 = a.h_q / a.h_kv;
    const vkq32 = p.fdata + p.ith * (dk + 2 * dv + CACHE_LINE_F32);
    const vkq16: [*]u16 = @ptrCast(@alignCast(vkq32 + dv));
    const q_q: [*]u16 = @ptrCast(@alignCast(vkq32 + 2 * dv));

    var ir = ir0;
    while (ir < ir1) : (ir += 1) {
        const iq2 = ir / a.n_t;
        const iq1 = ir - iq2 * a.n_t;
        var s_sum: f32 = 0;
        var m_max: f32 = -std.math.inf(f32);
        @memset(vkq16[0..dv], 0);
        const ik2 = iq2 / rk2;
        vec.fp32to16Row(a.qRow(iq1, iq2), q_q, dk);

        var ic = ic_start;
        while (ic < ic_end) : (ic += 1) {
            if (!a.visible(iq1, ic)) continue;
            var s = vec.dotF16(dk, a.kRow(ic, ik2), q_q);
            s *= a.scale;
            const m_old = m_max;
            var ms: f32 = 1.0;
            var vs: f32 = 1.0;
            if (s > m_max) {
                m_max = s;
                ms = avx.expf(m_old - m_max);
                vec.scaleF16(dv, vkq16, ms);
            } else {
                vs = avx.expf(s - m_max);
            }
            vec.madF16(dv, vkq16, a.vRow(ic, ik2), vs);
            s_sum = @mulAdd(f32, s_sum, ms, vs);
        }
        vec.fp16to32Row(vkq16, vkq32, dv);

        if (partials) |pt| {
            const partial = pt + ir * partial_stride;
            partial[0] = m_max;
            partial[1] = s_sum;
            @memcpy((partial + 2)[0..dv], vkq32[0..dv]);
        } else {
            const s_inv: f32 = if (s_sum == 0.0) 0.0 else 1.0 / s_sum;
            vec.scaleF32(dv, vkq32, s_inv);
            @memcpy((a.dst + (iq1 * a.h_q + iq2) * dv)[0..dv], vkq32[0..dv]);
        }
    }
}

fn flashTiled(p: *const Params, a: *const FlashArgs, ir0: usize, ir1: usize) void {
    const dk = a.dk;
    const dv = a.dk;
    const rk2 = a.h_q / a.h_kv;
    const per_thread = Q_TILE * dk + 2 * Q_TILE * KV_TILE + Q_TILE * dv + KV_TILE * dv + KV_TILE * dk + CACHE_LINE_F32;
    const base = p.fdata + p.ith * per_thread;
    const q_f32 = base;
    const kq = base + Q_TILE * dk;
    const mask32 = kq + Q_TILE * KV_TILE;
    const vkq32 = mask32 + Q_TILE * KV_TILE;
    const v32 = vkq32 + Q_TILE * dv;
    const k_f32 = v32 + KV_TILE * dv;
    const inf = std.math.inf(f32);

    var ir = ir0;
    while (ir < ir1) {
        const iq2 = ir / a.n_t;
        const iq1 = ir - iq2 * a.n_t;
        // ggml padded satırları da hesaplar; sonuçları kullanılmadığı için yalnız tile_rows işlenir.
        const tile_rows: usize = @min(Q_TILE, @min(ir1 - ir, a.n_t - iq1));
        var s_sum: [Q_TILE]f32 = undefined;
        var m_max: [Q_TILE]f32 = undefined;
        for (0..tile_rows) |i| {
            s_sum[i] = 0;
            m_max[i] = -inf;
        }
        @memset(vkq32[0 .. tile_rows * dv], 0);
        const ik2 = iq2 / rk2;
        for (0..tile_rows) |tq| @memcpy((q_f32 + tq * dk)[0..dk], a.qRow(iq1 + tq, iq2)[0..dk]);
        @memset(k_f32[0 .. dk * KV_TILE], 0);
        @memset(v32[0 .. KV_TILE * dv], 0);

        var ic: usize = 0;
        while (ic < a.n_kv) : (ic += KV_TILE) {
            const kv_tile: usize = @min(KV_TILE, a.n_kv - ic);
            var can_skip = true;
            for (0..tile_rows) |tq| {
                for (0..kv_tile) |tk| {
                    const vis = a.visible(iq1 + tq, ic + tk);
                    mask32[tq * KV_TILE + tk] = if (vis) 0 else -inf;
                    if (vis) can_skip = false;
                }
                for (kv_tile..KV_TILE) |tk| mask32[tq * KV_TILE + tk] = -inf;
            }
            if (can_skip) continue;

            for (0..kv_tile) |tk| {
                const kr = a.kRow(ic + tk, ik2);
                for (0..dk) |d| k_f32[d * KV_TILE + tk] = avx.fp16(kr[d]);
            }
            @memset(kq[0 .. tile_rows * KV_TILE], 0);
            vec.simdGemm(kq, q_f32, k_f32, tile_rows, dk, KV_TILE);
            vec.scaleF32(tile_rows * KV_TILE, kq, a.scale);
            if (kv_tile < KV_TILE) {
                for (0..tile_rows) |tq| {
                    for (kv_tile..KV_TILE) |tk| kq[tq * KV_TILE + tk] = -inf;
                }
            }
            vec.addF32(tile_rows * KV_TILE, kq, kq, mask32);

            var skip: [Q_TILE]bool = undefined;
            for (0..tile_rows) |tq| {
                const kq_row = kq + tq * KV_TILE;
                const tile_max = vec.maxF32(KV_TILE, kq_row);
                skip[tq] = false;
                if (tile_max == -inf) {
                    skip[tq] = true;
                    continue;
                }
                const m_old = m_max[tq];
                const m_new = @max(m_old, tile_max);
                if (m_new > m_old) {
                    const ms = avx.expf(m_old - m_new);
                    vec.scaleF32(dv, vkq32 + tq * dv, ms);
                    s_sum[tq] *= ms;
                }
                m_max[tq] = m_new;
                s_sum[tq] = @floatCast(@as(f64, s_sum[tq]) + vec.softMax(KV_TILE, kq_row, kq_row, m_new));
            }
            for (0..kv_tile) |tk| vec.fp16to32Row(a.vRow(ic + tk, ik2), v32 + tk * dv, dv);
            for (0..tile_rows) |tq| {
                if (skip[tq]) @memset((kq + tq * KV_TILE)[0..KV_TILE], 0);
            }
            vec.simdGemm(vkq32, kq, v32, tile_rows, KV_TILE, dv);
        }

        for (0..tile_rows) |tq| {
            const s_inv: f32 = if (s_sum[tq] == 0.0) 0.0 else 1.0 / s_sum[tq];
            vec.scaleF32(dv, vkq32 + tq * dv, s_inv);
            @memcpy((a.dst + ((iq1 + tq) * a.h_q + iq2) * dv)[0..dv], (vkq32 + tq * dv)[0..dv]);
        }
        ir += tile_rows;
    }
}

/// ggml_compute_forward_flash_attn_ext_f16
pub fn flashAttn(p: *const Params, a: *const FlashArgs) void {
    const dv = a.dk;
    const use_split = a.n_t == 1 and a.n_kv >= 512;
    if (use_split) {
        const per_thread = a.dk + 2 * dv + CACHE_LINE_F32;
        const chunk_size = (a.n_kv + p.nth - 1) / p.nth;
        const partial_size = 2 + dv;
        const partials_base = p.fdata + p.nth * per_thread;
        const ic_start = p.ith * chunk_size;
        const ic_end = @min(ic_start + chunk_size, a.n_kv);
        const partial_stride = p.nth * partial_size;
        const chunk_partials = partials_base + p.ith * partial_size;
        if (ic_start < a.n_kv) {
            for (0..a.h_q) |h| flashOneChunk(p, a, h, h + 1, ic_start, ic_end, chunk_partials, partial_stride);
        } else {
            for (0..a.h_q) |h| {
                const qp = chunk_partials + h * partial_stride;
                qp[0] = -std.math.inf(f32);
                qp[1] = 0.0;
            }
        }
        p.barrier();
        // ggml_flash_attn_ext_reduce_partials
        const vkq_final = p.fdata + p.ith * per_thread;
        var h: usize = p.ith;
        while (h < a.h_q) : (h += p.nth) {
            var m_final = -std.math.inf(f32);
            var s_final: f32 = 0.0;
            @memset(vkq_final[0..dv], 0);
            for (0..p.nth) |c| {
                if (c * chunk_size >= a.n_kv) continue;
                const partial = partials_base + (h * p.nth + c) * partial_size;
                const m_chunk = partial[0];
                const s_chunk = partial[1];
                if (s_chunk == 0.0) continue;
                const m_new = @max(m_final, m_chunk);
                const scale_old = avx.expf(m_final - m_new);
                const scale_new = avx.expf(m_chunk - m_new);
                for (0..dv) |d| vkq_final[d] = @mulAdd(f32, vkq_final[d], scale_old, partial[2 + d] * scale_new);
                s_final = @mulAdd(f32, s_final, scale_old, s_chunk * scale_new);
                m_final = m_new;
            }
            if (s_final != 0.0) vec.scaleF32(dv, vkq_final, 1.0 / s_final);
            @memcpy((a.dst + h * dv)[0..dv], vkq_final[0..dv]);
        }
        return;
    }

    const nr = a.n_t * a.h_q;
    const nth_scaled = p.nth * 4;
    const chunk_size = (nr + nth_scaled - 1) / nth_scaled;
    var nchunk = (nr + chunk_size - 1) / chunk_size;
    if (p.nth == 1 or nchunk < p.nth) nchunk = p.nth;
    if (p.ith == 0) p.pool.chunkSet(@intCast(p.nth));
    p.barrier();
    const dr = (nr + nchunk - 1) / nchunk;
    const use_tiled = a.n_t >= Q_TILE and dv % 8 == 0;
    var current: i64 = p.ith;
    while (current < nchunk) {
        const ir0 = dr * @as(usize, @intCast(current));
        const ir1 = @min(ir0 + dr, nr);
        if (use_tiled) {
            flashTiled(p, a, ir0, ir1);
        } else {
            flashOneChunk(p, a, ir0, ir1, 0, a.n_kv, null, 0);
        }
        current = p.pool.chunkAdd(1);
    }
}

pub fn flashScratchFloats(dk: usize, h_q: usize, nth: usize) usize {
    const tiled = Q_TILE * dk + 2 * Q_TILE * KV_TILE + Q_TILE * dk + KV_TILE * dk + KV_TILE * dk + CACHE_LINE_F32;
    const scalar = dk + 2 * dk + CACHE_LINE_F32;
    return nth * @max(tiled, scalar) + h_q * nth * (2 + dk);
}

test "flashTiled, skaler flash yolu ile aynı sonucu verir" {
    try avx.initLibm();
    const alloc = std.testing.allocator;
    const Pool = @import("pool.zig").Pool;
    const pool = try Pool.init(alloc, 1);
    defer pool.deinit();
    const dk = 256;
    const h_q = 8;
    const h_kv = 2;
    const n_t = 128;
    const n_kv = 256;
    const fdata = try alloc.alloc(f32, flashScratchFloats(dk, h_q, 1));
    defer alloc.free(fdata);
    const qb = try alloc.alloc(f32, n_t * h_q * dk);
    defer alloc.free(qb);
    const kb = try alloc.alloc(u16, n_kv * h_kv * dk);
    defer alloc.free(kb);
    const vb = try alloc.alloc(u16, n_kv * h_kv * dk);
    defer alloc.free(vb);
    const d1 = try alloc.alloc(f32, n_t * h_q * dk);
    defer alloc.free(d1);
    const d2 = try alloc.alloc(f32, n_t * h_q * dk);
    defer alloc.free(d2);
    var prng = std.Random.DefaultPrng.init(5);
    const r = prng.random();
    for (qb) |*v| v.* = r.float(f32) - 0.5;
    for (kb) |*v| v.* = avx.fp32to16(r.float(f32) - 0.5);
    for (vb) |*v| v.* = avx.fp32to16(r.float(f32) - 0.5);
    var w: [8]u8 = undefined;
    const p = Params{ .ith = 0, .nth = 1, .pool = pool, .wdata = &w, .fdata = fdata.ptr };
    var a = FlashArgs{ .q = qb.ptr, .k = kb.ptr, .v = vb.ptr, .dst = d1.ptr, .dk = dk, .h_q = h_q, .h_kv = h_kv, .n_t = n_t, .n_kv = n_kv, .first_cell = 0, .scale = 1.0 / 16.0 };
    const nr = n_t * h_q;
    flashTiled(&p, &a, 0, nr);
    a.dst = d2.ptr;
    flashOneChunk(&p, &a, 0, nr, 0, n_kv, null, 0);
    for (d1, d2, 0..) |x, y, i| {
        if (@abs(x - y) > 1e-2) {
            std.debug.print("fark @ tok={d} head={d} d={d}: tiled={d} skaler={d}\n", .{ i / (h_q * dk), i / dk % h_q, i % dk, x, y });
            return error.TestFailed;
        }
    }
}
