//! LM head için küme indeksi: tüm satırlar yerine gizli vektöre en yakın kümelerin satırları okunur.
//! Yaklaşıktır; tam yol `mulMat(output)` olarak kalır.
const std = @import("std");
const q = @import("quants.zig");
const vec = @import("vec.zig");
const ops = @import("ops.zig");
const sgemm = @import("sgemm.zig");
const F8 = @Vector(8, f32);
const Alloc = std.mem.Allocator;
/// Varsayılan LM head yolu: IVF (yaklaşık, yalnız top-1). 4096 küme, nprobe 512.
pub const default_l: usize = 4096;
pub const default_nprobe: usize = 512;

fn parallel(nth: usize, ctx: anytype, comptime f: fn (@TypeOf(ctx), usize) void) !void {
    var th: [64]std.Thread = undefined;
    for (1..nth) |i| th[i] = try std.Thread.spawn(.{}, f, .{ ctx, i });
    f(ctx, 0);
    for (1..nth) |i| th[i].join();
}

fn normalize(x: []f32) void {
    const n2 = vec.dotF32(x.len, x.ptr, x.ptr);
    if (n2 <= 0) return;
    const s = 1.0 / @sqrt(n2);
    for (x) |*v| v.* *= s;
}

fn nearest(c: [*]const f32, l: usize, k: usize, x: [*]const f32) u32 {
    var best: f32 = -std.math.inf(f32);
    var bi: u32 = 0;
    for (0..l) |i| {
        const s = vec.dotF32(k, c + i * k, x);
        if (s > best) {
            best = s;
            bi = @intCast(i);
        }
    }
    return bi;
}

pub const Index = struct {
    l: usize,
    k: usize,
    centroids: []align(64) f32,
    assign: []u32,

    pub fn deinit(self: *Index, a: Alloc) void {
        a.free(self.centroids);
        a.free(self.assign);
    }
};

const Train = struct {
    w: ops.Mat,
    s: []align(64) f32,
    ids: []const u32,
    c: []align(64) f32,
    l: usize,
    nth: usize,
    sums: [][]f32,
    counts: [][]u32,
    assign: []u32,

    fn fill(t: *Train, ith: usize) void {
        const k = t.w.k;
        var i = ith;
        while (i < t.ids.len) : (i += t.nth) {
            const y = t.s[i * k..][0..k];
            q.dequantizeQ80(@ptrCast(@alignCast(t.w.data + t.ids[i] * t.w.nb1)), y.ptr, k);
            normalize(y);
        }
    }
    fn step(t: *Train, ith: usize) void {
        const k = t.w.k;
        @memset(t.sums[ith], 0);
        @memset(t.counts[ith], 0);
        var i = ith;
        while (i < t.ids.len) : (i += t.nth) {
            const a = nearest(t.c.ptr, t.l, k, t.s.ptr + i * k);
            t.counts[ith][a] += 1;
            vec.madF32(k, t.sums[ith].ptr + a * k, t.s.ptr + i * k, 1.0);
        }
    }
    fn all(t: *Train, ith: usize) void {
        const k = t.w.k;
        var tmp: [4096]f32 = undefined;
        var i = ith;
        while (i < t.w.m) : (i += t.nth) {
            q.dequantizeQ80(@ptrCast(@alignCast(t.w.data + i * t.w.nb1)), &tmp, k);
            normalize(tmp[0..k]);
            t.assign[i] = nearest(t.c.ptr, t.l, k, &tmp);
        }
    }
};

/// Küresel k-means: örnek üzerinde eğitim, sonra tüm satırların ataması.
pub fn build(w: ops.Mat, l: usize, nth: usize, a: Alloc) !Index {
    const k = w.k;
    const n_s = @min(w.m, @max(l * 32, 60000));
    const ids = try a.alloc(u32, w.m);
    defer a.free(ids);
    for (ids, 0..) |*v, i| v.* = @intCast(i);
    var prng = std.Random.DefaultPrng.init(1);
    prng.random().shuffle(u32, ids);
    const s = try a.alignedAlloc(f32, .@"64", n_s * k);
    defer a.free(s);
    const c = try a.alignedAlloc(f32, .@"64", l * k);
    errdefer a.free(c);
    const sums = try a.alloc([]f32, nth);
    defer a.free(sums);
    const counts = try a.alloc([]u32, nth);
    defer a.free(counts);
    for (sums, counts) |*x, *y| {
        x.* = try a.alloc(f32, l * k);
        y.* = try a.alloc(u32, l);
    }
    defer for (sums, counts) |x, y| {
        a.free(x);
        a.free(y);
    };
    const assign = try a.alloc(u32, w.m);
    errdefer a.free(assign);
    var t = Train{ .w = w, .s = s, .ids = ids[0..n_s], .c = c, .l = l, .nth = nth, .sums = sums, .counts = counts, .assign = assign };
    try parallel(nth, &t, Train.fill);
    @memcpy(c, s[0 .. l * k]);
    for (0..6) |_| {
        try parallel(nth, &t, Train.step);
        for (0..l) |i| {
            const y = c[i * k..][0..k];
            var n: u32 = 0;
            for (counts) |ct| n += ct[i];
            if (n == 0) continue;
            @memset(y, 0);
            for (sums) |sm| vec.madF32(k, y.ptr, sm.ptr + i * k, 1.0);
            normalize(y);
        }
    }
    try parallel(nth, &t, Train.all);
    return .{ .l = l, .k = k, .centroids = c, .assign = assign };
}


/// Çalışma zamanı indeksi: Q8 merkezler ve kümeye göre sıralı Q8 satırlar.
pub const Runtime = struct {
    l: usize,
    k: usize,
    nb1: usize,
    cq: []align(64) u8,
    start: []u32,
    order: []u32,
    rows: []align(64) u8,
    /// H40: kümeler 8'lik R8 gruplarına paketli; g8[c] kümenin ilk grubu, pos8 satır konumu (dolguda maxInt).
    g8: []u32,
    pos8: []u32,
    r8: []align(64) u8,

    pub fn deinit(self: *Runtime, a: Alloc) void {
        a.free(self.g8);
        a.free(self.pos8);
        a.free(self.r8);
        a.free(self.cq);
        a.free(self.start);
        a.free(self.order);
        a.free(self.rows);
        a.destroy(self);
    }
};

fn makeRuntime(w: ops.Mat, l: usize, cq: []align(64) u8, assign: []const u32, a: Alloc) !*Runtime {
    const start = try a.alloc(u32, l + 1);
    @memset(start, 0);
    for (assign) |c| start[c + 1] += 1;
    for (0..l) |i| start[i + 1] += start[i];
    const pos = try a.dupe(u32, start[0..l]);
    defer a.free(pos);
    const order = try a.alloc(u32, w.m);
    const rows = try a.alignedAlloc(u8, .@"64", w.m * w.nb1);
    for (assign, 0..) |c, i| {
        const j = pos[c];
        pos[c] += 1;
        order[j] = @intCast(i);
        @memcpy(rows[j * w.nb1..][0..w.nb1], (w.data + i * w.nb1)[0..w.nb1]);
    }
    const nb = w.k / 32;
    const g8 = try a.alloc(u32, l + 1);
    g8[0] = 0;
    for (0..l) |c| g8[c + 1] = g8[c] + (start[c + 1] - start[c] + 7) / 8;
    const pos8 = try a.alloc(u32, g8[l] * 8);
    @memset(pos8, std.math.maxInt(u32));
    const r8 = try a.alignedAlloc(u8, .@"64", g8[l] * nb * sgemm.R8_BLOCK);
    const tmp = try a.alloc(q.BlockQ80, 8 * nb);
    defer a.free(tmp);
    for (0..l) |c| {
        var gi: usize = g8[c];
        var r0: usize = start[c];
        while (r0 < start[c + 1]) : ({
            r0 += 8;
            gi += 1;
        }) {
            @memset(std.mem.sliceAsBytes(tmp), 0);
            for (0..@min(8, start[c + 1] - r0)) |i| {
                @memcpy(std.mem.sliceAsBytes(tmp[i * nb ..][0..nb]), rows[(r0 + i) * w.nb1 ..][0..w.nb1]);
                pos8[gi * 8 + i] = @intCast(r0 + i);
            }
            sgemm.repackQ80x8(tmp.ptr, r8.ptr + gi * nb * sgemm.R8_BLOCK, 8, nb);
        }
    }
    const r = try a.create(Runtime);
    r.* = .{ .l = l, .k = w.k, .nb1 = w.nb1, .cq = cq, .start = start, .order = order, .rows = rows, .g8 = g8, .pos8 = pos8, .r8 = r8 };
    return r;
}

/// `<model>.ivf<L>` varsa okur, yoksa eğitip yazar.
pub fn open(model_path: []const u8, w: ops.Mat, l: usize, nth: usize, a: Alloc, io: std.Io) !*Runtime {
    if (w.type != .q8_0) return error.IvfYalnizQ80;
    const path = try std.fmt.allocPrint(a, "{s}.ivf{d}", .{ model_path, l });
    defer a.free(path);
    const cq = try a.alignedAlloc(u8, .@"64", l * w.nb1);
    errdefer a.free(cq);
    const assign = try a.alloc(u32, w.m);
    defer a.free(assign);
    const size = cq.len + assign.len * 4;
    if (std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(size + 1))) |data| {
        defer a.free(data);
        if (data.len != size) return error.IvfDosyaBoyutu;
        @memcpy(cq, data[0..cq.len]);
        @memcpy(std.mem.sliceAsBytes(assign), data[cq.len..]);
    } else |_| {
        var idx = try build(w, l, nth, a);
        defer idx.deinit(a);
        for (0..l) |c| q.quantizeRowQ80(idx.centroids.ptr + c * w.k, @ptrCast(@alignCast(cq.ptr + c * w.nb1)), w.k);
        @memcpy(assign, idx.assign);
        const f = try std.Io.Dir.cwd().createFile(io, path, .{});
        defer f.close(io);
        var buf: [65536]u8 = undefined;
        var fw = f.writer(io, &buf);
        try fw.interface.writeAll(cq);
        try fw.interface.writeAll(std.mem.sliceAsBytes(assign));
        try fw.interface.flush();
    }
    return makeRuntime(w, l, cq, assign, a);
}

pub const Best = extern struct { value: f32 = -std.math.inf(f32), token: u32 = 0, pad: [14]u32 = @splat(0) };
/// H40: R8 ön elemesinin en iyi 4 adayı (satır konumu); reduce dotQ80Q80 ile kesin skorlar.
pub const Best4 = extern struct {
    v: [4]f32 = @splat(-std.math.inf(f32)),
    pos: [4]u32 = @splat(std.math.maxInt(u32)),
    pad: [8]u32 = @splat(0),

    inline fn push(b: *Best4, v: f32, pos: u32) void {
        if (!(v > b.v[3])) return;
        var i: usize = 3;
        while (i > 0 and v > b.v[i - 1]) : (i -= 1) {
            b.v[i] = b.v[i - 1];
            b.pos[i] = b.pos[i - 1];
        }
        b.v[i] = v;
        b.pos[i] = pos;
    }
};

/// En yüksek skorlu `nprobe` kümenin kimlikleri, artan sırada (quickselect + sort).
fn topSel(sc: []const f32, ord: []u32, nprobe: usize) []u32 {
    for (ord, 0..) |*o, i| o.* = @intCast(i);
    var lo: usize = 0;
    var hi: usize = ord.len - 1;
    const want = nprobe - 1;
    while (lo < hi) {
        const pv = sc[ord[(lo + hi) / 2]];
        var i = lo;
        var j = hi;
        while (i <= j) {
            while (sc[ord[i]] > pv) i += 1;
            while (sc[ord[j]] < pv) j -= 1;
            if (i <= j) {
                std.mem.swap(u32, &ord[i], &ord[j]);
                i += 1;
                if (j == 0) break;
                j -= 1;
            }
        }
        if (want <= j) {
            hi = j;
        } else if (want >= i) {
            lo = i;
        } else break;
    }
    const sel = ord[0..nprobe];
    std.sort.pdq(u32, sel, {}, std.sort.asc(u32));
    return sel;
}

/// H14: n satırın IVF araması tek geçişte. Her satır yalnız kendi seçtiği kümelerde skorlanır
/// (Search ile aynı sonuç); satırlar kümelerin birleşimi üzerinden bir kez okunur.
pub const Batch = struct {
    pub const MAXR = 128;
    const MW = MAXR / 64; // küme başına maske kelimesi
    const PieceC = struct { first: u32, last: u32, cl: u32 };
    ix: *const Runtime,
    nprobe: usize,
    nth: usize,
    n: usize = 0,
    scores: []f32,
    ord: []u32,
    mask: []u64,
    pieces: []PieceC,
    n_pieces: usize = 0,
    best: []Best4,
    tokens: [MAXR]u32 = @splat(0),

    pub fn init(ix: *const Runtime, nprobe: usize, nth: usize, a: Alloc) !*Batch {
        const s = try a.create(Batch);
        s.* = .{
            .ix = ix,
            .nprobe = @min(nprobe, ix.l),
            .nth = nth,
            .scores = try a.alloc(f32, MAXR * ix.l),
            .ord = try a.alloc(u32, nth * ix.l),
            .mask = try a.alloc(u64, ix.l * MW),
            .pieces = try a.alloc(PieceC, ix.g8[ix.l] / PIECE_G + ix.l + 1),
            .best = try a.alloc(Best4, nth * MAXR),
        };
        return s;
    }
    pub fn deinit(s: *Batch, a: Alloc) void {
        a.free(s.scores);
        a.free(s.ord);
        a.free(s.mask);
        a.free(s.pieces);
        a.free(s.best);
        a.destroy(s);
    }

    /// x: n satır Q8_0, satır adımı ix.k/32 blok.
    pub fn scoreRange(s: *Batch, x: [*]const q.BlockQ80, n: usize, ith: usize, nth: usize) void {
        const ix = s.ix;
        const dr = (ix.l + nth - 1) / nth;
        const c0 = @min(dr * ith, ix.l);
        const c1 = @min(c0 + dr, ix.l);
        const rb = ix.k / 32;
        for (c0..c1) |c| {
            const cr: [*]const q.BlockQ80 = @ptrCast(@alignCast(ix.cq.ptr + c * ix.nb1));
            for (0..n) |r| s.scores[r * ix.l + c] = q.dotQ80Q80(ix.k, cr, x + r * rb);
        }
    }

    /// Tek thread, scoreRange'den önce.
    pub fn begin(s: *Batch, n: usize) void {
        s.n = n;
        @memset(s.mask, 0);
    }

    /// H41: satır başı küme seçimi thread'lere dağıtılır (her thread kendi ord'u, maske atomik OR).
    pub fn select(s: *Batch, ith: usize, nth: usize) void {
        const l = s.ix.l;
        const ord = s.ord[ith * l ..][0..l];
        var r = ith;
        while (r < s.n) : (r += nth) {
            for (topSel(s.scores[r * l ..][0..l], ord, s.nprobe)) |c| _ = @atomicRmw(u64, &s.mask[c * MW + r / 64], .Or, @as(u64, 1) << @intCast(r % 64), .monotonic);
        }
    }

    /// Tek thread, select'ten sonra: seçilen kümeleri parçalara böler.
    pub fn plan(s: *Batch) void {
        var np: usize = 0;
        for (0..s.ix.l) |c| {
            var any: u64 = 0;
            for (s.mask[c * MW ..][0..MW]) |w| any |= w;
            if (any == 0) continue;
            var a = s.ix.g8[c];
            const e = s.ix.g8[c + 1];
            while (a < e) {
                const b = @min(a + PIECE_G, e);
                s.pieces[np] = .{ .first = a, .last = b, .cl = @intCast(c) };
                np += 1;
                a = b;
            }
        }
        s.n_pieces = np;
    }

    /// xr: n satır quantR8 aktivasyonu. Parça = kümenin R8 grupları; maskedeki satırlar 4'erli çekirdekle.
    pub fn scan(s: *const Batch, xr: [*]const q.BlockQ80, piece: usize, b: []Best4) void {
        const ix = s.ix;
        const p = s.pieces[piece];
        const rb = ix.k / 32;
        var cols: [MAXR + 3]u32 = undefined;
        var nc: usize = 0;
        for (s.mask[p.cl * MW ..][0..MW], 0..) |w, wi| {
            var bits = w;
            while (bits != 0) : (bits &= bits - 1) {
                cols[nc] = @intCast(wi * 64 + @ctz(bits));
                nc += 1;
            }
        }
        const n4 = (nc + 3) / 4 * 4;
        for (nc..n4) |i| cols[i] = cols[nc - 1];
        var cv: [4]F8 align(32) = undefined;
        for (p.first..p.last) |g| {
            const w = ix.r8.ptr + g * rb * sgemm.R8_BLOCK;
            var j: usize = 0;
            while (j < n4) : (j += 4) {
                sgemm.r8x4(w, .{ xr + cols[j] * rb, xr + cols[j + 1] * rb, xr + cols[j + 2] * rb, xr + cols[j + 3] * rb }, rb, &cv);
                for (j..@min(j + 4, nc)) |ci| {
                    const r = cols[ci];
                    const sc: [8]f32 = cv[ci - j];
                    for (sc, ix.pos8[g * 8 ..][0..8]) |v, ps| {
                        if (ps != std.math.maxInt(u32)) b[r].push(v, ps);
                    }
                }
            }
        }
    }

    /// Adayları 8 bit x (xq) ile kesin skorlar: eski tam taramayla aynı top-1 (aday listede olduğu sürece).
    pub fn reduce(s: *Batch, xq: [*]const q.BlockQ80) void {
        const ix = s.ix;
        const rb = ix.k / 32;
        for (0..s.n) |r| {
            var bv: f32 = -std.math.inf(f32);
            var bt: u32 = 0;
            for (0..s.nth) |th| {
                const c = s.best[th * MAXR + r];
                for (c.pos) |ps| {
                    if (ps == std.math.maxInt(u32)) continue;
                    const v = q.dotQ80Q80(ix.k, @ptrCast(@alignCast(ix.rows.ptr + ps * ix.nb1)), xq + r * rb);
                    const t = ix.order[ps];
                    if (v > bv or (v == bv and t < bt)) {
                        bv = v;
                        bt = t;
                    }
                }
            }
            s.tokens[r] = bt;
        }
    }
};
const Piece = struct { first: u32, last: u32 };
const PIECE_ROWS = 256;
const PIECE_G = 32;

/// Tek arama durumu. Evreler: scoreRange (paralel) -> plan (tek) -> scan (paralel) -> reduce (tek).
pub const Search = struct {
    ix: *const Runtime,
    nprobe: usize,
    scores: []f32,
    ord: []u32,
    pieces: []Piece,
    n_pieces: usize = 0,
    best: []Best,
    token: u32 = 0,
    value: f32 = 0,

    pub fn init(ix: *const Runtime, nprobe: usize, nth: usize, a: Alloc) !*Search {
        const s = try a.create(Search);
        s.* = .{
            .ix = ix,
            .nprobe = @min(nprobe, ix.l),
            .scores = try a.alloc(f32, ix.l),
            .ord = try a.alloc(u32, ix.l),
            .pieces = try a.alloc(Piece, ix.order.len / PIECE_ROWS + ix.l + 1),
            .best = try a.alloc(Best, nth),
        };
        return s;
    }
    pub fn deinit(s: *Search, a: Alloc) void {
        a.free(s.scores);
        a.free(s.ord);
        a.free(s.pieces);
        a.free(s.best);
        a.destroy(s);
    }

    pub fn scoreRange(s: *Search, x: [*]const q.BlockQ80, ith: usize, nth: usize) void {
        const ix = s.ix;
        const dr = (ix.l + nth - 1) / nth;
        const c0 = @min(dr * ith, ix.l);
        const c1 = @min(c0 + dr, ix.l);
        for (c0..c1) |c| s.scores[c] = q.dotQ80Q80(ix.k, @ptrCast(@alignCast(ix.cq.ptr + c * ix.nb1)), x);
    }

    /// En yüksek skorlu nprobe kümeyi seçer (quickselect), bellek sırasına dizer, parçalara böler.
    pub fn plan(s: *Search) void {
        const sel = topSel(s.scores, s.ord, s.nprobe);
        var n: usize = 0;
        for (sel) |c| {
            var a = s.ix.start[c];
            const e = s.ix.start[c + 1];
            while (a < e) {
                const b = @min(a + PIECE_ROWS, e);
                s.pieces[n] = .{ .first = a, .last = b };
                n += 1;
                a = b;
            }
        }
        s.n_pieces = n;
    }

    pub fn scan(s: *const Search, x: [*]const q.BlockQ80, piece: usize, b: *Best) void {
        const ix = s.ix;
        const p = s.pieces[piece];
        for (p.first..p.last) |r| {
            const v = q.dotQ80Q80(ix.k, @ptrCast(@alignCast(ix.rows.ptr + r * ix.nb1)), x);
            const t = ix.order[r];
            if (v > b.value or (v == b.value and t < b.token)) {
                b.value = v;
                b.token = t;
            }
        }
    }

    pub fn reduce(s: *Search) void {
        var b = s.best[0];
        for (s.best[1..]) |o| {
            if (o.value > b.value or (o.value == b.value and o.token < b.token)) b = o;
        }
        s.token = b.token;
        s.value = b.value;
    }
};

fn mapFile(path: []const u8, io: std.Io) ![]align(std.heap.page_size_min) const u8 {
    const f = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer f.close(io);
    const n: usize = @intCast((try f.stat(io)).size);
    return std.posix.mmap(null, n, .{ .READ = true }, .{ .TYPE = .PRIVATE }, f.handle, 0);
}
fn npyData(m: []const u8) []const u8 {
    const h = @as(usize, m[8]) | (@as(usize, m[9]) << 8);
    return m[10 + h..];
}

/// Üretim arama kodunun gerçek gizli vektörlerdeki top-1 geri çağırması (x: n*k f32, t: n i64).
pub fn eval(model_path: []const u8, w: ops.Mat, x_path: []const u8, t_path: []const u8, l: usize, a: Alloc, io: std.Io) !void {
    const xb = npyData(try mapFile(x_path, io));
    const tb = npyData(try mapFile(t_path, io));
    const k = w.k;
    const n = xb.len / (k * 4);
    const ix = try open(model_path, w, l, 6, a, io);
    defer ix.deinit(a);
    const xf = try a.alignedAlloc(f32, .@"64", k);
    defer a.free(xf);
    const xq = try a.alignedAlloc(q.BlockQ80, .@"64", k / 32);
    defer a.free(xq);
    for ([_]usize{ 32, 64, 128, 256, 512 }) |np| {
        const s = try Search.init(ix, np, 1, a);
        defer s.deinit(a);
        var hit: usize = 0;
        var rows: u64 = 0;
        const TEXT_TOKENS = 400; // shortcut: girdi 20 metin × 400 token varsayar; farklı girdide metin sınırı dosyadan okunmalı
        const per = try a.alloc(u32, n / TEXT_TOKENS + 1);
        defer a.free(per);
        @memset(per, 0);
        const miss_i = try a.alloc(u32, n);
        defer a.free(miss_i);
        const miss_v = try a.alloc(f32, n);
        defer a.free(miss_v);
        var nm: usize = 0;
        const t0 = std.Io.Timestamp.now(io, .awake);
        for (0..n) |i| {
            @memcpy(std.mem.sliceAsBytes(xf), xb[i * k * 4..][0 .. k * 4]);
            q.quantizeRowQ80(xf.ptr, xq.ptr, k);
            s.scoreRange(xq.ptr, 0, 1);
            s.plan();
            var b = Best{};
            for (0..s.n_pieces) |pi| {
                s.scan(xq.ptr, pi, &b);
                rows += s.pieces[pi].last - s.pieces[pi].first;
            }
            s.best[0] = b;
            s.reduce();
            if (s.token == @as(u32, @intCast(std.mem.readInt(i64, tb[i * 8..][0..8], .little)))) {
                hit += 1;
                per[i / TEXT_TOKENS] += 1;
            } else {
                miss_i[nm] = @intCast(i);
                miss_v[nm] = s.value;
                nm += 1;
            }
        }
        const ms = @as(f64, @floatFromInt(t0.untilNow(io, .awake).toNanoseconds())) / 1e6 / @as(f64, @floatFromInt(n));
        std.debug.print("L={d} nprobe={d}: top-1 %{d:.2}, satır %{d:.2}, t=1 {d:.3} ms/arama\n", .{ l, np, @as(f64, @floatFromInt(hit)) * 100 / @as(f64, @floatFromInt(n)), @as(f64, @floatFromInt(rows)) * 100 / @as(f64, @floatFromInt(n * w.m)), ms });
        // Kuyruk: metin başına doğruluk ve kaçan tokende doğru token ile bulunan arasındaki logit farkı.
        const gaps = miss_v[0..nm];
        for (gaps, miss_i[0..nm]) |*g, i| {
            @memcpy(std.mem.sliceAsBytes(xf), xb[i * k * 4..][0 .. k * 4]);
            q.quantizeRowQ80(xf.ptr, xq.ptr, k);
            const want: u32 = @intCast(std.mem.readInt(i64, tb[i * 8..][0..8], .little));
            const r = std.mem.indexOfScalar(u32, ix.order, want).?;
            g.* = q.dotQ80Q80(ix.k, @ptrCast(@alignCast(ix.rows.ptr + r * ix.nb1)), xq.ptr) - g.*;
        }
        std.mem.sort(f32, gaps, {}, std.sort.asc(f32));
        const texts = per[0 .. n / TEXT_TOKENS];
        std.mem.sort(u32, texts, {}, std.sort.asc(u32));
        var close: usize = 0;
        for (gaps) |g| {
            if (g < 0.5) close += 1;
        }
        if (nm > 0) std.debug.print("  metin top-1 (%): en kötü {d:.2}, medyan {d:.2}; kaçan {d}: fark medyan {d:.3}, p95 {d:.3}, <0,5 olan %{d:.1}\n", .{ @as(f64, @floatFromInt(texts[0])) * 100 / TEXT_TOKENS, @as(f64, @floatFromInt(texts[texts.len / 2])) * 100 / TEXT_TOKENS, nm, gaps[nm / 2], gaps[nm * 95 / 100], @as(f64, @floatFromInt(close)) * 100 / @as(f64, @floatFromInt(nm)) });
    }
}
