//! Deney D: sezai mod — keşif katmanı ile üretim (throughput) katmanının ayrımı.
//! İş birimi: pencere başına W = 2048 blok; her blok T(x) + CRC32C + T⁻¹ (dönüşüm, doğrulama, teslim).
//! Girdi: GGUF'taki gerçek tensör baytları. Altı işçi thread, fiziksel çekirdek 0..5'e pinli;
//! pencere başına k işçi aktif (1..6). Yük fazları (her biri faz_saniye): hog thread'leri
//! [], [1,4], [1,2,4,5], [0..5] çekirdeklerinde işçilerle yarışır.
//! Koşullar:
//!   static   : k = 6 sabit (naif yol)
//!   explore  : her pencerede döngüsel k (sürekli keşif)
//!   fixed    : üretim k = 6 sabit; pencerelerin %12.5'i rastgele k (sabit küçük keşif bütçesi)
//!   adaptive : ödül-güdümlü denetleyici (EMA ödülü, değişim tespiti, p99 ve doğrulama korumaları)
//! Ödül = pencerenin doğru teslim ettiği bayt / pencere süresi.
//! Çalıştırma: zig-out/bin/deney-d <model.gguf> <static|explore|fixed|adaptive> <tohum> <faz_saniye>
//! Çıktı: stderr'de tek satır JSON.

const std = @import("std");
const gguf = @import("gguf");

const BLK = 4096;
const W = 2048;
const WORKERS = 6;
const WORKER_CPU = [WORKERS]usize{ 0, 3, 1, 4, 2, 5 };
const HOG_SETS = [_][]const usize{ &.{}, &.{ 1, 4 }, &.{ 1, 2, 4, 5 }, &.{ 0, 1, 2, 3, 4, 5 } };
const PHASES = HOG_SETS.len;
const MAX_WINDOWS = 400_000;
const KEY: u64 = 0x9E3779B97F4A7C15;

const Kind = enum { static, explore, fixed, adaptive, adaptive1 };

const Shared = struct {
    weights: []const u8,
    chunks: u64,
    base_seq: std.atomic.Value(u64),
    k_active: std.atomic.Value(u32),
    gen: std.atomic.Value(u64),
    done: [WORKERS]std.atomic.Value(u64),
    out: []u8,
    lat: []u32,
    stop: std.atomic.Value(bool),
};

fn nowNs() u64 {
    var ts: std.os.linux.timespec = undefined;
    _ = std.os.linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn pinTo(cpu: usize) void {
    var set = std.mem.zeroes(std.os.linux.cpu_set_t);
    set[cpu / @bitSizeOf(usize)] |= @as(usize, 1) << @as(u6, @intCast(cpu % @bitSizeOf(usize)));
    std.os.linux.sched_setaffinity(0, &set) catch @panic("sched_setaffinity başarısız");
}

fn splitmix(s: *u64) u64 {
    s.* +%= KEY;
    var z = s.*;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

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

fn crcBlk(p: []const u8, seq: u64) u32 {
    var c: u64 = seq & 0xFFFFFFFF;
    var i: usize = 0;
    while (i < BLK) : (i += 8) c = crcStep(c, std.mem.readInt(u64, p[i..][0..8], .little));
    return @truncate(c);
}

fn worker(sh: *Shared, idx: usize) void {
    pinTo(WORKER_CPU[idx]);
    var tmp: [BLK]u8 align(64) = undefined;
    var seen: u64 = 0;
    while (true) {
        var g = sh.gen.load(.acquire);
        while (g == seen) {
            if (sh.stop.load(.acquire)) return;
            std.Thread.yield() catch {};
            g = sh.gen.load(.acquire);
        }
        seen = g;
        const k: usize = sh.k_active.load(.acquire);
        if (idx < k) {
            const base = sh.base_seq.load(.acquire);
            var csum: u32 = 0;
            var j: usize = idx;
            while (j < W) : (j += k) {
                const seq = base + j;
                const x = sh.weights[@intCast((seq % sh.chunks) * BLK) ..][0..BLK];
                const t0 = nowNs();
                xform(tmp[0..], x, seq);
                csum +%= crcBlk(tmp[0..], seq);
                xform(sh.out[j * BLK ..][0..BLK], tmp[0..], seq);
                sh.lat[j] = @intCast(@min(nowNs() - t0, std.math.maxInt(u32)));
            }
            std.mem.doNotOptimizeAway(csum);
        }
        sh.done[idx].store(g, .release);
    }
}

fn hog(stop: *std.atomic.Value(bool), cpu: usize) void {
    pinTo(cpu);
    var x: u64 = 1;
    while (!stop.load(.acquire)) x = x *% 6364136223846793005 +% 1;
    std.mem.doNotOptimizeAway(x);
}

fn unitF(x: u64) f64 {
    return @as(f64, @floatFromInt(x >> 11)) * (1.0 / 9007199254740992.0);
}

const Policy = struct {
    kind: Kind,
    rng_s: u64,
    idx: u64 = 0,
    ema: [WORKERS + 1]f64 = @splat(0),
    seen: [WORKERS + 1]bool = @splat(false),
    best: usize = WORKERS,
    eps: f64 = 0.1,
    drops: u32 = 0,
    force_k1: bool = false,
    p99_min: f64 = std.math.inf(f64),
    // adaptive (v2) durumu
    peak: f64 = 0,
    drop_run: u32 = 0,
    rolling: [64]f64 = @splat(0),
    rolling_n: usize = 0,
    p99_hi_run: u32 = 0,

    const Choice = struct { k: usize, explored: bool };

    fn rnd(self: *Policy) u64 {
        self.rng_s ^= self.rng_s >> 12;
        self.rng_s ^= self.rng_s << 25;
        self.rng_s ^= self.rng_s >> 27;
        return self.rng_s *% 2685821657736338717;
    }

    fn choose(self: *Policy) Choice {
        switch (self.kind) {
            .static => return .{ .k = WORKERS, .explored = false },
            .explore => return .{ .k = @intCast(self.idx % WORKERS + 1), .explored = true },
            .fixed => {
                if (unitF(self.rnd()) < 0.125) return .{ .k = @intCast(self.rnd() % WORKERS + 1), .explored = true };
                return .{ .k = WORKERS, .explored = false };
            },
            .adaptive, .adaptive1 => {
                if (self.force_k1) {
                    self.force_k1 = false;
                    return .{ .k = 1, .explored = true };
                }
                var k: usize = 1;
                while (k <= WORKERS) : (k += 1) {
                    if (!self.seen[k]) return .{ .k = k, .explored = true };
                }
                if (unitF(self.rnd()) < self.eps) return .{ .k = @intCast(self.rnd() % WORKERS + 1), .explored = true };
                return .{ .k = self.best, .explored = false };
            },
        }
    }

    /// Revize edilmiş denetleyici (v2): yumuşatılmış ödül, tepe oranıyla değişim tespiti, medyan tabanlı p99 koruması.
    fn updateV2(self: *Policy, k: usize, reward: f64, p99: f64, wrong: u32) void {
        self.idx += 1;
        if (wrong > 0) {
            self.force_k1 = true;
            self.eps = @max(self.eps, 0.5);
        }

        self.ema[k] = if (self.seen[k]) 0.1 * reward + 0.9 * self.ema[k] else reward;
        self.seen[k] = true;

        var bi: usize = self.best;
        var bv: f64 = -1;
        var j: usize = 1;
        while (j <= WORKERS) : (j += 1) {
            if (self.seen[j] and self.ema[j] > bv) {
                bv = self.ema[j];
                bi = j;
            }
        }
        if (bi != self.best) {
            self.best = bi;
            self.peak = self.ema[bi];
            self.drop_run = 0;
        } else if (self.ema[bi] > self.peak) {
            self.peak = self.ema[bi];
        }

        // Değişim: en iyi kolun EMA'sı tepe değerinin %75'inin altında üç pencere üst üste → yeniden keşif.
        if (k == self.best and self.ema[self.best] < 0.75 * self.peak) {
            self.drop_run += 1;
        } else {
            self.drop_run = 0;
        }
        if (self.drop_run >= 3) {
            self.eps = @max(self.eps, 0.15);
            self.drop_run = 0;
            self.peak = self.ema[self.best];
            var q: usize = 1;
            while (q <= WORKERS) : (q += 1) {
                if (q != self.best) self.seen[q] = false;
            }
        }
        self.eps = @max(0.03, self.eps * 0.97);

        // p99 koruması: son 64 pencere medyanının 2 katının üstünde üç pencere üst üste → keşfi artır.
        self.rolling[self.rolling_n % 64] = p99;
        self.rolling_n += 1;
        if (self.rolling_n >= 16) {
            const n = @min(self.rolling_n, 64);
            var tmp: [64]f64 = undefined;
            @memcpy(tmp[0..n], self.rolling[0..n]);
            std.mem.sort(f64, tmp[0..n], {}, std.sort.asc(f64));
            const med = tmp[n / 2];
            self.p99_hi_run = if (p99 > 2.0 * med) self.p99_hi_run + 1 else 0;
            if (self.p99_hi_run >= 3) self.eps = @max(self.eps, 0.1);
        }
    }

    fn update(self: *Policy, k: usize, reward: f64, p99: f64, wrong: u32) void {
        if (self.kind == .adaptive) return self.updateV2(k, reward, p99, wrong);
        self.idx += 1;
        if (self.kind != .adaptive1) return;

        if (wrong > 0) {
            self.force_k1 = true;
            self.eps = @max(self.eps, 0.5);
        }

        const had = self.seen[k];
        const prev = self.ema[k];
        self.ema[k] = if (had) 0.2 * reward + 0.8 * prev else reward;
        self.seen[k] = true;

        // Değişim tespiti: en iyi kolun ödülü EMA'nın %20 altına iki pencere üst üste düşerse yeniden keşfet.
        if (had and k == self.best and reward < 0.8 * prev) {
            self.drops += 1;
        } else {
            self.drops = 0;
        }
        if (self.drops >= 2) {
            self.eps = 0.3;
            self.drops = 0;
            var j: usize = 1;
            while (j <= WORKERS) : (j += 1) {
                if (j != self.best) self.seen[j] = false;
            }
        } else {
            self.eps = @max(0.05, self.eps * 0.98);
        }

        // p99 koruması: gecikme sıçradıysa keşfi artır.
        if (p99 > 3.0 * self.p99_min) self.eps = @max(self.eps, 0.2);
        self.p99_min = @min(self.p99_min, p99);

        var bi: usize = self.best;
        var bv: f64 = -1;
        var j: usize = 1;
        while (j <= WORKERS) : (j += 1) {
            if (self.seen[j] and self.ema[j] > bv) {
                bv = self.ema[j];
                bi = j;
            }
        }
        self.best = bi;
    }
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;

    var args_it = try std.process.Args.Iterator.initAllocator(init.minimal.args, gpa);
    defer args_it.deinit();
    var args = std.array_list.Managed([]const u8).init(gpa);
    defer args.deinit();
    while (args_it.next()) |arg| try args.append(arg);

    if (args.items.len != 5) {
        std.debug.print("kullanım: {s} <model.gguf> <static|explore|fixed|adaptive> <tohum> <faz_saniye>\n", .{args.items[0]});
        return error.MissingArgument;
    }
    const kind = std.meta.stringToEnum(Kind, args.items[2]) orelse return error.UnknownCondition;
    const seed = try std.fmt.parseInt(u64, args.items[3], 10);
    const phase_s = try std.fmt.parseInt(u64, args.items[4], 10);

    var reader = try gguf.Reader.openWithAllocator(args.items[1], gpa);
    defer reader.deinit();
    var name: ?[]const u8 = null;
    var best_size: u64 = 0;
    var it = reader.tensors.iterator();
    while (it.next()) |e| {
        if (std.mem.endsWith(u8, e.key_ptr.*, "ffn_down.weight") and e.value_ptr.size > best_size) {
            name = e.key_ptr.*;
            best_size = e.value_ptr.size;
        }
    }
    const weights = try reader.loadTensor(name orelse return error.TensorNotFound);
    const chunks: u64 = weights.len / BLK;

    var sh = Shared{
        .weights = weights,
        .chunks = chunks,
        .base_seq = std.atomic.Value(u64).init(0),
        .k_active = std.atomic.Value(u32).init(WORKERS),
        .gen = std.atomic.Value(u64).init(0),
        .done = undefined,
        .out = try gpa.alloc(u8, W * BLK),
        .lat = try gpa.alloc(u32, W),
        .stop = std.atomic.Value(bool).init(false),
    };
    for (&sh.done) |*d| d.* = std.atomic.Value(u64).init(0);

    var threads: [WORKERS]std.Thread = undefined;
    for (0..WORKERS) |i| threads[i] = try std.Thread.spawn(.{}, worker, .{ &sh, i });

    var pol = Policy{ .kind = kind, .rng_s = (seed *% KEY) | 1 };
    const rec_phase = try gpa.alloc(u8, MAX_WINDOWS);
    const rec_k = try gpa.alloc(u8, MAX_WINDOWS);
    const rec_expl = try gpa.alloc(bool, MAX_WINDOWS);
    const rec_wall = try gpa.alloc(u64, MAX_WINDOWS);
    const rec_correct = try gpa.alloc(u32, MAX_WINDOWS);
    const rec_wrong = try gpa.alloc(u32, MAX_WINDOWS);
    const rec_p99 = try gpa.alloc(u32, MAX_WINDOWS);
    var nwin: usize = 0;

    var hog_stop = std.atomic.Value(bool).init(false);
    var hog_threads: [8]std.Thread = undefined;
    var n_hogs: usize = 0;
    var cur_phase: usize = PHASES;
    const phase_ns: u64 = phase_s * std.time.ns_per_s;
    const start = nowNs();
    var g: u64 = 0;
    var seq_base: u64 = 0;
    const lat_tmp = try gpa.alloc(u32, W);

    while (true) {
        const el = nowNs() - start;
        if (el >= phase_ns * PHASES) break;
        const ph: usize = @intCast(el / phase_ns);
        if (ph != cur_phase) {
            if (n_hogs > 0) {
                hog_stop.store(true, .release);
                for (hog_threads[0..n_hogs]) |t| t.join();
                n_hogs = 0;
                hog_stop.store(false, .release);
            }
            for (HOG_SETS[ph]) |cpu| {
                hog_threads[n_hogs] = try std.Thread.spawn(.{}, hog, .{ &hog_stop, cpu });
                n_hogs += 1;
            }
            cur_phase = ph;
        }

        const choice = pol.choose();
        sh.k_active.store(@intCast(choice.k), .release);
        sh.base_seq.store(seq_base, .release);
        g += 1;
        const t0 = nowNs();
        sh.gen.store(g, .release);
        for (0..WORKERS) |i| {
            while (sh.done[i].load(.acquire) != g) {}
        }
        const wall = nowNs() - t0;

        // Oracle süre dışı: pencerenin her doğru teslimi özgün girdiyle karşılaştırılır.
        var correct: u32 = 0;
        var wrong: u32 = 0;
        for (0..W) |j| {
            const x = weights[@intCast(((seq_base + j) % chunks) * BLK) ..][0..BLK];
            if (std.mem.eql(u8, x, sh.out[j * BLK ..][0..BLK])) correct += 1 else wrong += 1;
        }
        @memcpy(lat_tmp, sh.lat);
        std.mem.sort(u32, lat_tmp, {}, std.sort.asc(u32));
        const p99: f64 = @floatFromInt(lat_tmp[(W - 1) * 99 / 100]);

        const wall_s = @as(f64, @floatFromInt(wall)) * 1e-9;
        const reward = @as(f64, @floatFromInt(correct)) * BLK / wall_s;
        pol.update(choice.k, reward, p99, wrong);

        if (nwin >= MAX_WINDOWS) return error.TooManyWindows;
        rec_phase[nwin] = @intCast(ph);
        rec_k[nwin] = @intCast(choice.k);
        rec_expl[nwin] = choice.explored;
        rec_wall[nwin] = wall;
        rec_correct[nwin] = correct;
        rec_wrong[nwin] = wrong;
        rec_p99[nwin] = @intFromFloat(p99);
        nwin += 1;
        seq_base += W;
    }

    if (n_hogs > 0) {
        hog_stop.store(true, .release);
        for (hog_threads[0..n_hogs]) |t| t.join();
    }
    sh.stop.store(true, .release);
    for (threads) |t| t.join();

    // Faz başına özet.
    var tot_bytes: u64 = 0;
    var tot_wall: u64 = 0;
    var tot_wrong: u64 = 0;
    std.debug.print("{{\"cond\":\"{s}\",\"seed\":{d},\"faz_s\":{d},\"pencere\":{d},\"fazlar\":[", .{ args.items[2], seed, phase_s, nwin });
    for (0..PHASES) |ph| {
        var bytes: u64 = 0;
        var wall_sum: u64 = 0;
        var wrong_sum: u64 = 0;
        var explored: u64 = 0;
        var wins: u64 = 0;
        var kh: [WORKERS + 1]u64 = @splat(0);
        var p99s = std.array_list.Managed(u32).init(gpa);
        defer p99s.deinit();
        for (0..nwin) |w| {
            if (rec_phase[w] != ph) continue;
            wins += 1;
            bytes += @as(u64, rec_correct[w]) * BLK;
            wall_sum += rec_wall[w];
            wrong_sum += rec_wrong[w];
            if (rec_expl[w]) explored += 1;
            kh[rec_k[w]] += 1;
            try p99s.append(rec_p99[w]);
        }
        std.mem.sort(u32, p99s.items, {}, std.sort.asc(u32));
        const p99_med: f64 = if (p99s.items.len > 0) @floatFromInt(p99s.items[p99s.items.len / 2]) else 0;
        const thr = @as(f64, @floatFromInt(bytes)) / (@as(f64, @floatFromInt(wall_sum)) * 1e-9) / 1e6;
        std.debug.print(
            "{s}{{\"faz\":{d},\"yuk\":{d},\"thr_mb_s\":{d:.1},\"p99_us_med\":{d:.1},\"pencere\":{d},\"kesif\":{d}," ++
                "\"yanlis\":{d},\"k\":[{d},{d},{d},{d},{d},{d}]}}",
            .{
                if (ph == 0) "" else ",",
                ph,
                HOG_SETS[ph].len,
                thr,
                p99_med / 1000.0,
                wins,
                explored,
                wrong_sum,
                kh[1],
                kh[2],
                kh[3],
                kh[4],
                kh[5],
                kh[6],
            },
        );
        tot_bytes += bytes;
        tot_wall += wall_sum;
        tot_wrong += wrong_sum;
    }
    const tot_thr = @as(f64, @floatFromInt(tot_bytes)) / (@as(f64, @floatFromInt(tot_wall)) * 1e-9) / 1e6;
    std.debug.print("],\"toplam_thr_mb_s\":{d:.1},\"toplam_yanlis\":{d}}}\n", .{ tot_thr, tot_wrong });
}
