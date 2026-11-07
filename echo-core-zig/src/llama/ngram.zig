//! H01 taslağı: token yerine normalize edilmiş yüzey baytları üzerinde n-gram arar.
//! Prompt ile üretim aynı metni farklı tokenlara bölse bile pencere eşleşir.
const std = @import("std");

const Alloc = std.mem.Allocator;
pub const W: usize = 12;
fn mix(h: u64, v: u32) u64 {
    var x = h ^ (@as(u64, v) +% 0x9e3779b97f4a7c15 +% (h << 6) +% (h >> 2));
    x ^= x >> 33;
    x *%= 0xff51afd7ed558ccd;
    x ^= x >> 33;
    return x;
}

const Entry = struct { tok: u32, len: u8, tail: [W]u8, hits: u16 = 1 };

fn hashBytes(b: []const u8) u64 {
    var h: u64 = 0xcbf29ce484222325;
    for (b) |c| {
        h ^= c;
        h *%= 0x100000001b3;
    }
    return h;
}

fn tailOf(piece: []const u8) struct { len: u8, tail: [W]u8 } {
    var r: [W]u8 = undefined;
    const n = @min(piece.len, W);
    @memcpy(r[0..n], piece[piece.len - n ..]);
    return .{ .len = @intCast(n), .tail = r };
}

fn pushWin(win: *[W]u8, wl: *usize, bytes: []const u8) void {
    if (bytes.len >= W) {
        @memcpy(win[0..W], bytes[bytes.len - W ..]);
        wl.* = W;
        return;
    }
    const keep = @min(wl.*, W - bytes.len);
    if (keep < wl.*) std.mem.copyForwards(u8, win[0..keep], win[wl.* - keep ..][0..keep]);
    @memcpy(win[keep..][0..bytes.len], bytes);
    wl.* = keep + bytes.len;
}

pub const Drafter = struct {
    allocator: Alloc,
    map: std.AutoHashMap(u64, Entry),
    win: [W]u8 = undefined,
    win_len: usize = 0,

    pub fn init(allocator: Alloc) !Drafter {
        return .{ .allocator = allocator, .map = std.AutoHashMap(u64, Entry).init(allocator) };
    }

    pub fn deinit(self: *Drafter) void {
        self.map.deinit();
    }

    pub fn reset(self: *Drafter) void {
        self.map.clearRetainingCapacity();
        self.win_len = 0;
    }

    /// Token ve onun normalize edilmiş yüzey parçasını geçmişe ekler.
    pub fn add(self: *Drafter, tok: u32, piece: []const u8) !void {
        if (self.win_len >= W) {
            const t = tailOf(piece);
            const k = hashBytes(self.win[0..W]);
            const hits: u16 = if (self.map.get(k)) |e| @min(e.hits + 1, 4096) else 1;
            try self.map.put(k, .{ .tok = tok, .len = t.len, .tail = t.tail, .hits = hits });
        }
        pushWin(&self.win, &self.win_len, piece);
    }

    /// Son W bayttan zincir kurar. `out` doldurulur, uzunluk döner.
    pub fn chain(self: *const Drafter, out: []u32, min_hits: u16) usize {
        var win = self.win;
        var wl = self.win_len;
        var n: usize = 0;
        while (n < out.len) {
            if (wl < W) break;
            const e = self.map.get(hashBytes(win[0..W])) orelse break;
            if (n == 0 and e.hits < min_hits) break;
            out[n] = e.tok;
            n += 1;
            pushWin(&win, &wl, e.tail[0..e.len]);
        }
        return n;
    }
};

test "yuzey penceresi token sinirini yok sayar" {
    var d = try Drafter.init(std.testing.allocator);
    defer d.deinit();
    // prompt: "The" + " " + "quick" + " " + "brown" + " " + "fox"
    for ([_]struct { t: u32, s: []const u8 }{
        .{ .t = 1, .s = "The" },   .{ .t = 2, .s = " " },
        .{ .t = 3, .s = "quick" }, .{ .t = 4, .s = " " },
        .{ .t = 5, .s = "brown" }, .{ .t = 6, .s = " " },
        .{ .t = 7, .s = "fox" },
    }) |x| try d.add(x.t, x.s);
    // üretim farklı tokenlarla aynı metni yazsın: " quick" tek parça
    for ([_]struct { t: u32, s: []const u8 }{
        .{ .t = 11, .s = " quick" }, .{ .t = 12, .s = " brown" },
    }) |x| try d.add(x.t, x.s);
    var out: [4]u32 = undefined;
    const n = d.chain(&out, 1);
    try std.testing.expect(n >= 1);
    try std.testing.expectEqual(@as(u32, 6), out[0]); // bosluk
}

/// Token düzeyi taslak: akış modelin kendi ürettiği tokenlar olduğunda birebir hizalı.
pub const TokenDrafter = struct {
    allocator: Alloc,
    n: usize,
    map: std.AutoHashMap(u64, u32),
    hist: std.ArrayList(u32),

    pub fn init(allocator: Alloc, n: usize) !TokenDrafter {
        return .{ .allocator = allocator, .n = n, .map = std.AutoHashMap(u64, u32).init(allocator), .hist = .empty };
    }
    pub fn deinit(self: *TokenDrafter) void {
        self.map.deinit();
        self.hist.deinit(self.allocator);
    }
    pub fn reset(self: *TokenDrafter) void {
        self.map.clearRetainingCapacity();
        self.hist.clearRetainingCapacity();
    }
    fn key(self: *const TokenDrafter, start: usize) u64 {
        var h: u64 = 0xcbf29ce484222325;
        for (self.hist.items[start..][0..self.n]) |t| h = mix(h, t);
        return h;
    }
    pub fn add(self: *TokenDrafter, tok: u32) !void {
        try self.hist.append(self.allocator, tok);
        const len = self.hist.items.len;
        if (len > self.n) try self.map.put(self.key(len - self.n - 1), tok);
    }
    pub fn chain(self: *const TokenDrafter, out: []u32) usize {
        if (out.len == 0 or self.hist.items.len < self.n) return 0;
        var win: [16]u32 = undefined;
        const n = @min(self.n, win.len);
        @memcpy(win[0..n], self.hist.items[self.hist.items.len - n ..][0..n]);
        var len: usize = 0;
        while (len < out.len) {
            var h: u64 = 0xcbf29ce484222325;
            for (win[0..n]) |t| h = mix(h, t);
            const next = self.map.get(h) orelse break;
            out[len] = next;
            len += 1;
            if (n > 1) std.mem.copyForwards(u32, win[0 .. n - 1], win[1..n]);
            win[n - 1] = next;
        }
        return len;
    }
};
