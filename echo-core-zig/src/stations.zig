//! Deney C: istasyonlu (retime/recheck) ve istasyonsuz tek geçiş boru hattı.
//! Yazmalar: W2 (S1→S2 kopya), W3 (S2→S3 kopya), W4 (S3 ters dönüşüm → çıktı).
//! Kaynak tarafı (S0/S1, tutulan kaynak) güvenilir varsayılır; hata yalnızca W2..W4 yazmalarında enjekte edilir.
//! single  : hiçbir istasyonda kontrol yok; yanlış teslim sessizdir.
//! station : her yazmadan sonra CRC kontrolü; tutmazsa üst istasyonun doğrulanmış buffer'ından yeniden yaz (en fazla R).

const std = @import("std");

pub const BLK = 4096;
const KEY: u64 = 0x9E3779B97F4A7C15;

pub const Mode = enum { single, station };

pub const Config = struct {
    mode: Mode,
    /// Yazma başına hata olasılığı (%70 bit çevirme, %30 burst).
    p: f64 = 0,
    /// Sabit blok oranı: her yazmada aynı bit tersine döner (yeniden yazma ile kurtarılamaz).
    q: f64 = 0,
    seed: u64 = 0,
    retry: u32 = 3,
};

pub const Stats = struct {
    delivered: u64 = 0,
    detected: u64 = 0,
    retries: u64 = 0,
    lost: u64 = 0,
    crc_checks: u64 = 0,
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

/// Dönüşüm: seq'e bağlı XOR keystream. Kendi tersidir.
fn xform(dst: []u8, src: []const u8, seq: u64) void {
    var st: u64 = 0x5851F42D4C957F2D ^ (seq *% 0xA24BAED4963EE407);
    var i: usize = 0;
    while (i < BLK) : (i += 8) {
        const a = std.mem.readInt(u64, src[i..][0..8], .little);
        std.mem.writeInt(u64, dst[i..][0..8], a ^ splitmix(&st), .little);
    }
}

fn crcStep(c: u64, v: u64) u64 {
    var r = c;
    asm volatile ("crc32q %[v], %[r]" : [r] "+r" (r) : [v] "r" (v));
    return r;
}

/// CRC32C; başlangıç değeri seq: yanlış sıradaki blok da CRC'de yakalanır.
pub fn crcBlk(p: []const u8, seq: u64) u32 {
    var c: u64 = seq & 0xFFFFFFFF;
    var i: usize = 0;
    while (i < BLK) : (i += 8) c = crcStep(c, std.mem.readInt(u64, p[i..][0..8], .little));
    return @truncate(c);
}

pub const Pipe = struct {
    cfg: Config,
    b1: []u8,
    b2: []u8,
    b3: []u8,
    rng_s: u64,

    pub fn init(allocator: std.mem.Allocator, cfg: Config) !Pipe {
        const b1 = try allocator.alloc(u8, BLK);
        errdefer allocator.free(b1);
        const b2 = try allocator.alloc(u8, BLK);
        errdefer allocator.free(b2);
        const b3 = try allocator.alloc(u8, BLK);
        return .{ .cfg = cfg, .b1 = b1, .b2 = b2, .b3 = b3, .rng_s = (cfg.seed *% KEY) | 1 };
    }

    pub fn deinit(self: *Pipe, allocator: std.mem.Allocator) void {
        allocator.free(self.b1);
        allocator.free(self.b2);
        allocator.free(self.b3);
    }

    fn rnd(self: *Pipe) u64 {
        self.rng_s ^= self.rng_s >> 12;
        self.rng_s ^= self.rng_s << 25;
        self.rng_s ^= self.rng_s >> 27;
        return self.rng_s *% 2685821657736338717;
    }

    fn isStuck(self: *const Pipe, seq: u64) bool {
        if (self.cfg.q <= 0) return false;
        var s = self.cfg.seed ^ 0x51CC ^ seq;
        return unitF(splitmix(&s)) < self.cfg.q;
    }

    fn stuckBit(self: *const Pipe, seq: u64) usize {
        var s = self.cfg.seed ^ 0xB17 ^ seq;
        return @intCast(splitmix(&s) % (BLK * 8));
    }

    /// Bir yazma: dst = (xf ? T⁻¹(src) : src); sonra yazma-başına hata enjeksiyonu.
    fn write(self: *Pipe, dst: []u8, src: []const u8, seq: u64, xf: bool) void {
        if (xf) xform(dst, src, seq) else @memcpy(dst, src);

        if (self.cfg.p > 0 and unitF(self.rnd()) < self.cfg.p) {
            if (unitF(self.rnd()) < 0.7) {
                var n = 1 + self.rnd() % 3;
                while (n > 0) : (n -= 1) {
                    const b: usize = @intCast(self.rnd() % (BLK * 8));
                    dst[b >> 3] ^= @as(u8, 1) << @as(u3, @intCast(b & 7));
                }
            } else {
                const len: usize = 8 * @as(usize, @intCast(1 + self.rnd() % 8));
                const off: usize = @intCast(self.rnd() % @as(u64, BLK - len));
                for (dst[off..][0..len]) |*byte| byte.* = @truncate(self.rnd());
            }
        }

        if (self.isStuck(seq)) {
            const b = self.stuckBit(seq);
            dst[b >> 3] ^= @as(u8, 1) << @as(u3, @intCast(b & 7));
        }
    }

    /// İstasyon: dst'yi yaz, CRC kontrol et; tutmazsa üst buffer'dan yeniden yaz (en fazla retry).
    fn stage(self: *Pipe, dst: []u8, src: []const u8, seq: u64, xf: bool, want: u32, st: *Stats) bool {
        self.write(dst, src, seq, xf);
        var attempts: u32 = 0;
        while (true) {
            st.crc_checks += 1;
            if (crcBlk(dst, seq) == want) return true;
            st.detected += 1;
            if (attempts == self.cfg.retry) {
                st.lost += 1;
                return false;
            }
            attempts += 1;
            st.retries += 1;
            self.write(dst, src, seq, xf);
        }
    }

    /// Bir blok. Dönüş: true = teslim edildi (out'a yazıldı), false = kayıp (yalnızca station modunda).
    pub fn pass(self: *Pipe, seq: u64, x: []const u8, out: []u8, st: *Stats) bool {
        // S1 (kaynak tarafı, güvenilir): T(x) tutulur.
        xform(self.b1, x, seq);
        switch (self.cfg.mode) {
            .single => {
                self.write(self.b2, self.b1, seq, false);
                self.write(self.b3, self.b2, seq, false);
                self.write(out, self.b3, seq, true);
                st.delivered += 1;
                return true;
            },
            .station => {
                const crc_t = crcBlk(self.b1, seq);   // W2/W3 beklenen değeri
                const crc_x = crcBlk(x, seq);          // W4 beklenen değeri (özgün girdi)
                if (!self.stage(self.b2, self.b1, seq, false, crc_t, st)) return false;
                if (!self.stage(self.b3, self.b2, seq, false, crc_t, st)) return false;
                if (!self.stage(out, self.b3, seq, true, crc_x, st)) return false;
                st.delivered += 1;
                return true;
            },
        }
    }
};
