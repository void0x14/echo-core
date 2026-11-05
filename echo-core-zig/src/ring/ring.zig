//! Dolaşan blok ring'i: dönüşüm + CRC32C + commit kopyası + yeniden üretim.
//! Deney A'nın tek uygulaması; blok ölçümü ve uçtan uca decode ölçümü bunu kullanır.
//! Hata yalnızca yazılım ring slot'larında enjekte edilir; fiziksel CPU sinyali değildir.

const std = @import("std");

pub const BLK = 4096;
const KEY: u64 = 0x9E3779B97F4A7C15;

pub const Mode = enum { direct, checksum, recon };

pub const Config = struct {
    mode: Mode,
    /// Slot başına hata olasılığı (%70 bit çevirme, %30 burst bozulma).
    p: f64 = 0,
    /// Sabit blok oranı: her denemede aynı bit tersine döner (retry ile kurtarılamaz).
    q: f64 = 0,
    seed: u64 = 0,
    slots: usize = 64,
    retry: u32 = 3,
};

pub const Stats = struct {
    delivered: u64 = 0,
    detected: u64 = 0,
    retries: u64 = 0,
    recovered: u64 = 0,
    lost: u64 = 0,
};

fn splitmix(s: *u64) u64 {
    s.* +%= KEY;
    var z = s.*;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

fn unitF(x: u64) f64 {
    return @as(f64, @floatFromInt(x >> 11)) * (1.0 / 9007199254740992.0);
}

/// Dönüşüm: seq'e bağlı XOR keystream. Kendi tersidir: xform(xform(x)) == x.
fn xform(dst: []u8, src: []const u8, seq: u64) void {
    var st: u64 = 0x5851F42D4C957F2D ^ (seq *% 0xA24BAED4963EE407);
    var i: usize = 0;
    while (i < BLK) : (i += 8) {
        const a = std.mem.readInt(u64, src[i..][0..8], .little);
        std.mem.writeInt(u64, dst[i..][0..8], a ^ splitmix(&st), .little);
    }
}

/// Donanım CRC32C adımı (SSE4.2 crc32q).
fn crcStep(c: u64, v: u64) u64 {
    var r = c;
    asm volatile ("crc32q %[v], %[r]" : [r] "+r" (r) : [v] "r" (v));
    return r;
}

/// CRC32C; başlangıç değeri seq: yanlış sıradaki blok da CRC'de yakalanır.
fn crcBlk(p: []const u8, seq: u64) u32 {
    var c: u64 = seq & 0xFFFFFFFF;
    var i: usize = 0;
    while (i < BLK) : (i += 8) c = crcStep(c, std.mem.readInt(u64, p[i..][0..8], .little));
    return @truncate(c);
}

pub const Ring = struct {
    cfg: Config,
    slots: []u8,
    commit: []u8,
    commit_state: []u8,
    rng_s: u64,

    pub fn init(allocator: std.mem.Allocator, cfg: Config) !Ring {
        const slots = try allocator.alloc(u8, cfg.slots * BLK);
        errdefer allocator.free(slots);
        const commit = try allocator.alloc(u8, cfg.slots * BLK);
        errdefer allocator.free(commit);
        const commit_state = try allocator.alloc(u8, cfg.slots);
        @memset(commit_state, 0);
        return .{
            .cfg = cfg,
            .slots = slots,
            .commit = commit,
            .commit_state = commit_state,
            .rng_s = (cfg.seed *% KEY) | 1,
        };
    }

    pub fn deinit(self: *Ring, allocator: std.mem.Allocator) void {
        allocator.free(self.slots);
        allocator.free(self.commit);
        allocator.free(self.commit_state);
    }

    /// Bir bloğu ring'den geçirir. true = teslim edildi (out'a yazıldı), false = kayıp.
    pub fn passBlock(self: *Ring, seq: u64, x: []const u8, out: []u8, st: *Stats) bool {
        const k: usize = @intCast(seq % self.cfg.slots);
        const slot = self.slots[k * BLK ..][0..BLK];
        switch (self.cfg.mode) {
            .direct => {
                _ = self.put(k, seq, x, false);
                @memcpy(out, slot);
                st.delivered += 1;
                return true;
            },
            .checksum => {
                const crc = self.put(k, seq, x, true);
                if (crcBlk(slot, seq) != crc) {
                    st.detected += 1;
                    st.lost += 1;
                    return false;
                }
                xform(out, slot, seq);
                st.delivered += 1;
                return true;
            },
            .recon => {
                // Kaynağı commit'e kadar sakla; hata olursa commit kopyasından yeniden üret.
                const keep = self.commit[k * BLK ..][0..BLK];
                @memcpy(keep, x);
                self.commit_state[k] = 1;
                var a: u32 = 0;
                var crc = self.put(k, seq, keep, true);
                while (crcBlk(slot, seq) != crc) {
                    st.detected += 1;
                    if (a == self.cfg.retry) {
                        st.lost += 1;
                        self.commit_state[k] = 0;
                        return false;
                    }
                    a += 1;
                    st.retries += 1;
                    crc = self.put(k, seq, keep, true);
                }
                if (a > 0) st.recovered += 1;
                xform(out, slot, seq);
                self.commit_state[k] = 0;
                st.delivered += 1;
                return true;
            },
        }
    }

    /// Commit kaydı kapanmamış slot sayısı (0 olmalı).
    pub fn uncleaned(self: *const Ring) u32 {
        var n: u32 = 0;
        for (self.commit_state) |s| n += @intFromBool(s != 0);
        return n;
    }

    fn rnd(self: *Ring) u64 {
        self.rng_s ^= self.rng_s >> 12;
        self.rng_s ^= self.rng_s << 25;
        self.rng_s ^= self.rng_s >> 27;
        return self.rng_s *% 2685821657736338717;
    }

    fn isStuck(self: *const Ring, seq: u64) bool {
        if (self.cfg.q <= 0) return false;
        var s = self.cfg.seed ^ 0x51CC ^ seq;
        return unitF(splitmix(&s)) < self.cfg.q;
    }

    fn stuckBit(self: *const Ring, seq: u64) usize {
        var s = self.cfg.seed ^ 0xB17 ^ seq;
        return @intCast(splitmix(&s) % (BLK * 8));
    }

    /// Slot'a yaz ve ring içi hata enjekte et. xf=true: T(x) yazılır, niyet edilen CRC döner.
    fn put(self: *Ring, k: usize, seq: u64, x: []const u8, xf: bool) u32 {
        const d = self.slots[k * BLK ..][0..BLK];
        var crc: u32 = 0;
        if (xf) {
            xform(d, x, seq);
            crc = crcBlk(d, seq);
        } else {
            @memcpy(d, x);
        }

        if (self.cfg.p > 0 and unitF(self.rnd()) < self.cfg.p) {
            if (unitF(self.rnd()) < 0.7) {
                var n = 1 + self.rnd() % 3;
                while (n > 0) : (n -= 1) {
                    const b: usize = @intCast(self.rnd() % (BLK * 8));
                    d[b >> 3] ^= @as(u8, 1) << @as(u3, @intCast(b & 7));
                }
            } else {
                const len: usize = 8 * @as(usize, @intCast(1 + self.rnd() % 8));
                const off: usize = @intCast(self.rnd() % @as(u64, BLK - len));
                for (d[off..][0..len]) |*byte| byte.* = @truncate(self.rnd());
            }
        }

        if (self.isStuck(seq)) {
            const b = self.stuckBit(seq);
            d[b >> 3] ^= @as(u8, 1) << @as(u3, @intCast(b & 7));
        }
        return crc;
    }
};
