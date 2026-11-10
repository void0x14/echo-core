//! H02 (REST, arXiv 2311.08252): harici depodan (datastore) taslak çekme.
//! Depo = kullanıcının kendi .zig ve .md dosyaları, model tokenizer'ıyla tokenlanıp tek diziye eklenir.
//! Eşleme: depodaki tokenların suffix array'i; sorgu = bağlamın son n tokenı, n büyükten küçüğe denenir,
//! ilk eşleşen (en uzun) uzunlukta tüm eşleşmelerin devamları token token oylanır → tek zincir taslak.
//! Doğrulamayı çağıran yapar (decodeVerify); bu modül yalnız aday üretir.
const std = @import("std");
const Bpe = @import("../tokenizer/bpe.zig").Bpe;

const Alloc = std.mem.Allocator;
const SEP: u32 = std.math.maxInt(u32); // dosya ayracı: eşleşme ve zincir buradan öteye geçmez
const SORT_DEPTH: usize = 24; // sıralama yalnız ilk 24 tokena bakar; sorgu en çok MAXN token

pub const MAXN: usize = 16;

pub const Store = struct {
    a: Alloc,
    ds: []u32,
    sa: []u32,
    n_files: usize,

    pub fn deinit(s: *Store) void {
        s.a.free(s.ds);
        s.a.free(s.sa);
    }

    pub fn lessSuffix(ds: []const u32, x: u32, y: u32) bool {
        const lx = ds[x..];
        const ly = ds[y..];
        const n = @min(@min(lx.len, ly.len), SORT_DEPTH);
        for (0..n) |i| if (lx[i] != ly[i]) return lx[i] < ly[i];
        return lx.len < ly.len;
    }

    fn skip(path: []const u8) bool {
        var it = std.mem.splitScalar(u8, path, std.fs.path.sep);
        while (it.next()) |c| {
            if (std.mem.eql(u8, c, ".git") or std.mem.eql(u8, c, "zig-cache") or std.mem.eql(u8, c, ".zig-cache") or
                std.mem.eql(u8, c, "zig-out") or std.mem.eql(u8, c, ".claude") or std.mem.eql(u8, c, "node_modules")) return true;
        }
        return false;
    }

    /// `root` altındaki .zig ve .md dosyalarından depo kurar.
    pub fn build(a: Alloc, io: std.Io, root: []const u8, bpe: *const Bpe) !Store {
        var dir = try std.Io.Dir.cwd().openDir(io, root, .{ .iterate = true });
        defer dir.close(io);
        var paths = std.array_list.Managed([]u8).init(a);
        defer {
            for (paths.items) |p| a.free(p);
            paths.deinit();
        }
        var w = try dir.walk(a);
        defer w.deinit();
        while (try w.next(io)) |e| {
            if (e.kind != .file or skip(e.path)) continue;
            if (!std.mem.endsWith(u8, e.path, ".zig") and !std.mem.endsWith(u8, e.path, ".md")) continue;
            try paths.append(try a.dupe(u8, e.path));
        }
        std.mem.sort([]u8, paths.items, {}, struct {
            fn lt(_: void, x: []u8, y: []u8) bool {
                return std.mem.lessThan(u8, x, y);
            }
        }.lt);

        var toks = std.array_list.Managed(u32).init(a);
        errdefer toks.deinit();
        var n_files: usize = 0;
        for (paths.items) |p| {
            const data = dir.readFileAlloc(io, p, a, .limited(16 << 20)) catch continue;
            defer a.free(data);
            if (!std.unicode.utf8ValidateSlice(data)) continue;
            try bpe.encode(data, &toks, false);
            try toks.append(SEP);
            n_files += 1;
        }
        if (toks.items.len == 0) return error.DepoBos;
        const ds = try toks.toOwnedSlice();
        errdefer a.free(ds);
        const sa = try a.alloc(u32, ds.len);
        errdefer a.free(sa);
        for (sa, 0..) |*x, i| x.* = @intCast(i);
        std.mem.sort(u32, sa, @as([]const u32, ds), struct {
            fn lt(d: []const u32, x: u32, y: u32) bool {
                return lessSuffix(d, x, y);
            }
        }.lt);
        return .{ .a = a, .ds = ds, .sa = sa, .n_files = n_files };
    }

    /// pat ile başlayan ilk suffix'in SA sırası (lower bound) ve pat'tan büyük ilk sıra (upper bound).
    fn range(s: *const Store, pat: []const u32) struct { lo: usize, hi: usize } {
        var lo: usize = 0;
        var hi: usize = s.sa.len;
        while (lo < hi) { // ilk suffix >= pat (pat önek karşılaştırması)
            const mid = (lo + hi) / 2;
            if (cmp(s.ds[s.sa[mid]..], pat) < 0) lo = mid + 1 else hi = mid;
        }
        const first = lo;
        hi = s.sa.len;
        while (lo < hi) { // ilk suffix > pat (önek olarak)
            const mid = (lo + hi) / 2;
            if (cmp(s.ds[s.sa[mid]..], pat) <= 0) lo = mid + 1 else hi = mid;
        }
        return .{ .lo = first, .hi = lo };
    }

    /// suffix'in ilk pat.len tokenını pat ile karşılaştırır (-1, 0, 1); kısa suffix küçük sayılır.
    fn cmp(suf: []const u32, pat: []const u32) i32 {
        const n = @min(suf.len, pat.len);
        for (0..n) |i| if (suf[i] != pat[i]) return if (suf[i] < pat[i]) -1 else 1;
        return if (suf.len < pat.len) -1 else 0;
    }

    pub const Hit = struct { n: usize, cands: usize };

    /// Bağlamın (ctx) son tokenlarından en uzun eşleşmeyi (en az min_n) bulur; devam zincirini out'a yazar.
    /// Dönüş: yazılan token sayısı; hit.n eşleşen uzunluk, hit.cands aday sayısı.
    pub fn chain(s: *const Store, ctx: []const u32, min_n: usize, out: []u32, hit: *Hit) usize {
        hit.* = .{ .n = 0, .cands = 0 };
        var n: usize = @min(ctx.len, MAXN);
        while (n >= @max(min_n, 1)) : (n -= 1) {
            const pat = ctx[ctx.len - n ..];
            const r = s.range(pat);
            if (r.hi > r.lo) {
                hit.* = .{ .n = n, .cands = r.hi - r.lo };
                return s.vote(r.lo, r.hi, n, out);
            }
            if (n == 0) break;
        }
        return 0;
    }

    /// Eşleşen aralıktaki adayların devamları derinlik derinlik oylanır (REST'in trie'sinin tek dalı:
    /// her derinlikte en sık sonraki token, yalnız o dala uyan adaylar kalır).
    fn vote(s: *const Store, lo: usize, hi: usize, n: usize, out: []u32) usize {
        const cap = 64;
        var alive: [cap]u32 = undefined;
        var na: usize = 0;
        var i = lo;
        while (i < hi and na < cap) : (i += 1) {
            alive[na] = s.sa[i] + @as(u32, @intCast(n));
            na += 1;
        }
        var len: usize = 0;
        while (len < out.len and na > 0) {
            // en sık sonraki token (aday sayısı küçük: O(na^2) yerine ilk görülen sayılır)
            var best: u32 = SEP;
            var best_c: usize = 0;
            for (alive[0..na], 0..) |p, x| {
                if (p >= s.ds.len) continue;
                const t = s.ds[p];
                if (t == SEP) continue;
                var c: usize = 0;
                for (alive[x..na]) |q| {
                    if (q < s.ds.len and s.ds[q] == t) c += 1;
                }
                if (c > best_c) {
                    best_c = c;
                    best = t;
                }
            }
            if (best_c == 0) break;
            out[len] = best;
            len += 1;
            var k: usize = 0;
            for (alive[0..na]) |p| {
                if (p < s.ds.len and s.ds[p] == best) {
                    alive[k] = p + 1;
                    k += 1;
                }
            }
            na = k;
        }
        return len;
    }
};
