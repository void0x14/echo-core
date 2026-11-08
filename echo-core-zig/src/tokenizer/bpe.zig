//! Byte-level BPE (GPT-2 / Qwen). Qwen resmi tokenizer.json ile birebir: ön-bölme `qwen35` regex'inin
//! elle yazılmış hali + merge sırası + özel token ayırma. Doğrulama: benchmarks/tokenizer (6902 paragraf).
const std = @import("std");
const gguf = @import("../gguf/reader.zig");
const nfc = @import("nfc.zig");

const Allocator = std.mem.Allocator;
const ArrayList = std.array_list.Managed;

/// Unicode bayrakları (1=L 2=M 4=N 8=White_Space), iki kademeli tablo; benchmarks/tokenizer/hazirla.py üretir.
const ucat = @embedFile("ucat.bin");
const FL: u8 = 1;
const FM: u8 = 2;
const FN: u8 = 4;
const FS: u8 = 8;

fn bayrak(cp: u32) u8 {
    if (cp >= 0x110000) return 0;
    // düzen: u32 blok sayısı | u16[0x1100] blok indeksi | 256 baytlık bloklar
    const blk: usize = std.mem.readInt(u16, ucat[4 + (cp >> 8) * 2 ..][0..2], .little);
    return ucat[4 + 0x2200 + blk * 256 + (cp & 255)];
}

pub const Special = struct { text: []const u8, id: u32 };

const Merge = struct { rank: u32, out: u32 };

pub const Bpe = struct {
    allocator: Allocator,
    /// id -> ham bayt (özel tokenlar: kendi metni)
    pieces: [][]u8,
    types: []u8,
    by_bytes: std.StringHashMap(u32),
    pairs: std.AutoHashMap(u64, Merge),
    byte_id: [256]u32,
    /// metinden ayrılan tokenlar (kontrol + kullanıcı tanımlı), uzundan kısaya
    specials: []Special,

    pub fn initFromReader(reader: *const gguf.Reader, allocator: Allocator) !Bpe {
        const toks = reader.tokens.items;
        const n = toks.len;
        var self = Bpe{
            .allocator = allocator,
            .pieces = try allocator.alloc([]u8, n),
            .types = try allocator.alloc(u8, n),
            .by_bytes = std.StringHashMap(u32).init(allocator),
            .pairs = std.AutoHashMap(u64, Merge).init(allocator),
            .byte_id = @splat(0),
            .specials = &.{},
        };
        @memset(self.types, 1);
        if (reader.metadata.get("tokenizer.ggml.token_type")) |v| switch (v) {
            .ints => |t| for (t, 0..) |x, i| {
                self.types[i] = @intCast(x);
            },
            else => {},
        };
        const u2b = unicodeToByte();
        var sp = ArrayList(Special).init(allocator);
        defer sp.deinit();
        for (toks, 0..) |t, i| {
            if (self.types[i] == 1) {
                self.pieces[i] = try decodeGpt2(allocator, t, &u2b);
                try self.by_bytes.put(self.pieces[i], @intCast(i));
            } else {
                self.pieces[i] = try allocator.dupe(u8, t);
                if (self.types[i] == 3 or self.types[i] == 4) try sp.append(.{ .text = self.pieces[i], .id = @intCast(i) });
            }
        }
        std.mem.sort(Special, sp.items, {}, struct {
            fn f(_: void, a: Special, b: Special) bool {
                return a.text.len > b.text.len;
            }
        }.f);
        self.specials = try sp.toOwnedSlice();
        for (0..256) |b| self.byte_id[b] = self.by_bytes.get(&.{@intCast(b)}) orelse return error.MissingByteToken;

        const merges = if (reader.metadata.get("tokenizer.ggml.merges")) |v| switch (v) {
            .strings => |s| s,
            else => return error.MissingMerges,
        } else return error.MissingMerges;
        try self.pairs.ensureTotalCapacity(@intCast(merges.len));
        var buf = ArrayList(u8).init(allocator);
        defer buf.deinit();
        for (merges, 0..) |m, r| {
            const sp_i = std.mem.indexOfScalar(u8, m, ' ') orelse continue;
            buf.clearRetainingCapacity();
            try appendGpt2(&buf, m[0..sp_i], &u2b);
            const la = buf.items.len;
            try appendGpt2(&buf, m[sp_i + 1 ..], &u2b);
            const a = self.by_bytes.get(buf.items[0..la]) orelse continue;
            const b = self.by_bytes.get(buf.items[la..]) orelse continue;
            const o = self.by_bytes.get(buf.items) orelse continue;
            const gop = self.pairs.getOrPutAssumeCapacity(@as(u64, a) << 32 | b);
            if (!gop.found_existing) gop.value_ptr.* = .{ .rank = @intCast(r), .out = o };
        }
        return self;
    }

    pub fn deinit(self: *Bpe) void {
        for (self.pieces) |p| self.allocator.free(p);
        self.allocator.free(self.pieces);
        self.allocator.free(self.types);
        self.allocator.free(self.specials);
        self.by_bytes.deinit();
        self.pairs.deinit();
    }

    /// Özel token metnini id'ye çevirir (ör. "<|im_end|>").
    pub fn specialId(self: *const Bpe, text: []const u8) ?u32 {
        for (self.specials) |s| if (std.mem.eql(u8, s.text, text)) return s.id;
        return null;
    }

    pub fn isControl(self: *const Bpe, id: u32) bool {
        return id < self.types.len and self.types[id] == 3;
    }

    /// Token'ın ham baytları; kontrol tokenları için boş (sohbet çıktısına basılmaz).
    pub fn piece(self: *const Bpe, id: u32) []const u8 {
        if (id >= self.pieces.len or self.types[id] == 3) return "";
        return self.pieces[id];
    }

    /// parse_special: metindeki <|im_start|> vb. tek token olarak alınır (HF / llama.cpp varsayılanı).
    /// Kullanıcı tanımlı tokenlar (<think>, <tool_call>) her zaman ayrılır, resmi tokenizer gibi.
    pub fn encode(self: *const Bpe, text: []const u8, out: *ArrayList(u32), parse_special: bool) !void {
        var bas: usize = 0;
        var p: usize = 0;
        while (std.mem.indexOfScalarPos(u8, text, p, '<')) |q| {
            p = q + 1;
            for (self.specials) |s| {
                if (!parse_special and self.types[s.id] == 3) continue;
                if (std.mem.startsWith(u8, text[q..], s.text)) {
                    try self.encodeRaw(text[bas..q], out);
                    try out.append(s.id);
                    bas = q + s.text.len;
                    p = bas;
                    break;
                }
            }
        }
        try self.encodeRaw(text[bas..], out);
    }

    /// Özel token dışındaki parça: NFC (resmi normalizer) → ön-bölme → BPE.
    fn encodeRaw(self: *const Bpe, text0: []const u8, out: *ArrayList(u32)) !void {
        if (text0.len == 0) return;
        var nb = ArrayList(u8).init(self.allocator);
        defer nb.deinit();
        const text = (try nfc.normalize(text0, &nb)) orelse text0;
        var it = PreTok{ .t = text };
        var syms = ArrayList(u32).init(self.allocator);
        defer syms.deinit();
        while (it.next()) |w| {
            if (w.len == 1) {
                try out.append(self.byte_id[w[0]]);
            } else if (w.len <= 64) {
                try self.mergeScan(w, &syms);
                try out.appendSlice(syms.items);
            } else {
                try self.mergeHeap(w, out);
            }
        }
    }

    inline fn pair(self: *const Bpe, a: u32, b: u32) ?Merge {
        return self.pairs.get(@as(u64, a) << 32 | b);
    }

    /// Kısa kelime: her turda en düşük sıralı komşu çifti bul, birleştir (O(n²), n ≤ 64).
    fn mergeScan(self: *const Bpe, w: []const u8, v: *ArrayList(u32)) !void {
        v.clearRetainingCapacity();
        for (w) |c| try v.append(self.byte_id[c]);
        while (v.items.len > 1) {
            var best: u32 = std.math.maxInt(u32);
            var bi: usize = 0;
            var bo: u32 = 0;
            for (0..v.items.len - 1) |i| if (self.pair(v.items[i], v.items[i + 1])) |m| if (m.rank < best) {
                best = m.rank;
                bi = i;
                bo = m.out;
            };
            if (best == std.math.maxInt(u32)) break;
            v.items[bi] = bo;
            _ = v.orderedRemove(bi + 1);
        }
    }

    const Sym = struct { id: i64, prev: i64, next: i64 };
    const HE = struct { r: u32, pos: u32, a: u32, b: u32 };
    fn heLess(x: HE, y: HE) bool {
        return if (x.r != y.r) x.r < y.r else x.pos < y.pos;
    }

    /// Uzun kelime (ör. boşluksuz base64): bağlı liste + öncelik kuyruğu, O(n log n).
    fn mergeHeap(self: *const Bpe, w: []const u8, out: *ArrayList(u32)) !void {
        const a = self.allocator;
        const s = try a.alloc(Sym, w.len);
        defer a.free(s);
        var h = ArrayList(HE).init(a);
        defer h.deinit();
        for (w, 0..) |c, i| s[i] = .{ .id = self.byte_id[c], .prev = @as(i64, @intCast(i)) - 1, .next = if (i + 1 < w.len) @intCast(i + 1) else -1 };
        const Q = struct {
            fn push(hh: *ArrayList(HE), e: HE) !void {
                try hh.append(e);
                var i = hh.items.len - 1;
                while (i > 0) {
                    const p = (i - 1) / 2;
                    if (!heLess(hh.items[i], hh.items[p])) break;
                    std.mem.swap(HE, &hh.items[i], &hh.items[p]);
                    i = p;
                }
            }
            fn pop(hh: *ArrayList(HE)) HE {
                const top = hh.items[0];
                const last = hh.pop().?;
                if (hh.items.len > 0) {
                    hh.items[0] = last;
                    var i: usize = 0;
                    while (true) {
                        const l = 2 * i + 1;
                        if (l >= hh.items.len) break;
                        var c = l;
                        if (l + 1 < hh.items.len and heLess(hh.items[l + 1], hh.items[l])) c = l + 1;
                        if (!heLess(hh.items[c], hh.items[i])) break;
                        std.mem.swap(HE, &hh.items[i], &hh.items[c]);
                        i = c;
                    }
                }
                return top;
            }
        };
        const add = struct {
            fn f(bpe: *const Bpe, ss: []Sym, hh: *ArrayList(HE), l: usize, r: usize) !void {
                if (bpe.pair(@intCast(ss[l].id), @intCast(ss[r].id))) |m| try Q.push(hh, .{ .r = m.rank, .pos = @intCast(l), .a = @intCast(ss[l].id), .b = @intCast(ss[r].id) });
            }
        }.f;
        for (0..w.len - 1) |i| try add(self, s, &h, i, i + 1);
        while (h.items.len > 0) {
            const t = Q.pop(&h);
            const l = t.pos;
            if (s[l].id != t.a) continue;
            if (s[l].next < 0) continue;
            const r: usize = @intCast(s[l].next);
            if (s[r].id != t.b) continue;
            s[l].id = self.pair(t.a, t.b).?.out;
            s[r].id = -1;
            s[l].next = s[r].next;
            if (s[r].next >= 0) s[@intCast(s[r].next)].prev = @intCast(l);
            if (s[l].prev >= 0) try add(self, s, &h, @intCast(s[l].prev), l);
            if (s[l].next >= 0) try add(self, s, &h, l, @intCast(s[l].next));
        }
        var i: i64 = 0;
        while (i >= 0) : (i = s[@intCast(i)].next) try out.append(@intCast(s[@intCast(i)].id));
    }
};

/// qwen35 ön-bölme: (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
const PreTok = struct {
    t: []const u8,
    i: usize = 0,

    const Cp = struct { c: u32, f: u8, len: u8 };
    fn at(self: *const PreTok, i: usize) ?Cp {
        if (i >= self.t.len) return null;
        const b = self.t[i];
        const l: u8 = std.unicode.utf8ByteSequenceLength(b) catch 1;
        if (l == 1 or i + l > self.t.len) return .{ .c = if (b < 0x80) b else 0xFFFD, .f = if (b < 0x80) bayrak(b) else 0, .len = 1 };
        const c = std.unicode.utf8Decode(self.t[i..][0..l]) catch return .{ .c = 0xFFFD, .f = 0, .len = 1 };
        return .{ .c = c, .f = bayrak(c), .len = l };
    }
    fn isP(c: Cp) bool {
        return c.f & (FS | FL | FM | FN) == 0;
    }
    fn isNl(c: Cp) bool {
        return c.c == '\r' or c.c == '\n';
    }
    fn low(c: u32) u32 {
        return if (c >= 'A' and c <= 'Z') c + 32 else c;
    }

    fn next(self: *PreTok) ?[]const u8 {
        const k = self.i;
        const c0 = self.at(k) orelse return null;
        const e = self.match(k, c0);
        self.i = e;
        return self.t[k..e];
    }

    fn runWhile(self: *const PreTok, start: usize, comptime pred: fn (Cp) bool) usize {
        var j = start;
        while (self.at(j)) |c| : (j += c.len) if (!pred(c)) break;
        return j;
    }
    fn isLM(c: Cp) bool {
        return c.f & (FL | FM) != 0;
    }

    fn match(self: *const PreTok, k: usize, c0: Cp) usize {
        const k1 = k + c0.len;
        if (c0.c == '\'') if (self.at(k1)) |c1| {
            const a = low(c1.c);
            if (a == 's' or a == 't' or a == 'm' or a == 'd') return k1 + 1;
            if (self.at(k1 + 1)) |c2| {
                const b = low(c2.c);
                if ((a == 'r' and b == 'e') or (a == 'v' and b == 'e') or (a == 'l' and b == 'l')) return k1 + 2;
            }
        };
        if (isLM(c0)) return self.runWhile(k1, isLM);
        if (!isNl(c0) and c0.f & (FL | FN) == 0) if (self.at(k1)) |c1| if (isLM(c1)) return self.runWhile(k1, isLM);
        if (c0.f & FN != 0) return k1;
        var s = k;
        if (c0.c == ' ') if (self.at(k1)) |c1| if (isP(c1)) {
            s = k1;
        };
        if (self.at(s)) |cs| if (isP(cs)) {
            const e = self.runWhile(s + cs.len, isP);
            return self.runWhile(e, isNl);
        };
        // boşluk dizisi
        var j = k;
        var nl: ?usize = null;
        var prev = k;
        while (self.at(j)) |c| {
            if (c.f & FS == 0) break;
            if (isNl(c)) nl = j + c.len;
            prev = j;
            j += c.len;
        }
        if (j == k) return k1; // sınıfsız karakter
        if (nl) |x| return x;
        if (j >= self.t.len) return j;
        if (prev > k) return prev; // \s+(?!\S): son boşluk bir sonraki kelimeye kalır
        return j;
    }
};

fn unicodeToByte() [324]u8 {
    var m: [324]u8 = @splat(0);
    var n: u16 = 0;
    for (0..256) |b| {
        const d = (b >= 33 and b <= 126) or (b >= 161 and b <= 172) or (b >= 174 and b <= 255);
        if (d) m[b] = @intCast(b) else {
            m[256 + n] = @intCast(b);
            n += 1;
        }
    }
    return m;
}

fn appendGpt2(out: *ArrayList(u8), s: []const u8, u2b: *const [324]u8) !void {
    var it = std.unicode.Utf8View.initUnchecked(s).iterator();
    while (it.nextCodepoint()) |c| {
        if (c >= u2b.len) return error.BadGpt2Token;
        try out.append(u2b[c]);
    }
}

fn decodeGpt2(a: Allocator, s: []const u8, u2b: *const [324]u8) ![]u8 {
    var o = ArrayList(u8).init(a);
    errdefer o.deinit();
    try appendGpt2(&o, s, u2b);
    return o.toOwnedSlice();
}

/// Akış için: tamamlanmamış UTF-8 dizisini bir sonraki tokena kadar bekletir.
pub const Utf8Stream = struct {
    buf: [8]u8 = undefined,
    n: usize = 0,

    /// Basılabilecek kısmı `out`a yazar.
    pub fn feed(self: *Utf8Stream, bytes: []const u8, out: *ArrayList(u8)) !void {
        for (bytes) |b| {
            if (self.n == 0 and b < 0x80) {
                try out.append(b);
                continue;
            }
            self.buf[self.n] = b;
            self.n += 1;
            const need: usize = std.unicode.utf8ByteSequenceLength(self.buf[0]) catch 1;
            if (self.n >= need or self.n == self.buf.len) {
                try out.appendSlice(self.buf[0..self.n]);
                self.n = 0;
            }
        }
    }
};

test "PreTok qwen35 basit" {
    var it = PreTok{ .t = "Hello world's  test\n\n123 ğüş!!" };
    const beklenen = [_][]const u8{ "Hello", " world", "'s", " ", " test", "\n\n", "1", "2", "3", " ğüş", "!!" };
    for (beklenen) |b| try std.testing.expectEqualStrings(b, it.next().?);
    try std.testing.expect(it.next() == null);
}
