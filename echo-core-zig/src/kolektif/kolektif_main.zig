//! Deney K (v2). Ön kayıt koşusu kaynağı: kolektif_main_v1_prereg.zig.txt (sha256: benchmarks/kolektif-bin-v1-sha256.txt).
//! Varsayılan seçenekler (se=1;pad=0;pp=64) v1 ile aynı davranışı verir.
//!
//! Kollar: owned, free, cmd, cmd_mask, cmd_batch (v1); ring (SPSC dairesel dolaşım), cmd_energy (paket
//! enerjisi geri beslemesi). Ek probe'lar: probe-hot (paylaşılan satır çekişmesi), probe-bw (akış bant genişliği).
//!
//! Kullanım:
//!   kolektif run <arm> <işçiCPU,...> <ws_kb> <sn> <none|hog2|stall> <tohum> <ccx_csv_12> [seçenekler]
//!     seçenekler: "se=N;pad=0|1;pp=N;et=N;ec=N;es=F"
//!       se: durum yazma periyodu (tile); pad: 64B hizalı metadata ve kc (yanlış paylaşım ablasyonu)
//!       pp: pasif işçi yoklama periyodu; et/ec/es: power PMU tipi/config/ölçek (cmd_energy için)
//!   kolektif probe-chase <ws_kb> <cpu> [warm_cpu]
//!   kolektif probe-crc
//!   kolektif probe-hot <cpu,...> <0|1 paylaşılan> <iter>
//!   kolektif probe-bw <cpu,...> <ws_kb> <geçiş>
//! Çıktı: stderr'de tek satır JSON.

const std = @import("std");

const TILE = 4096;
const ROWS = 16;
const COLS = 256;
const MAX_W = 8;
const NONE_W: u8 = 0xFF;
const SAMPLE_CAP = 1 << 19;
const COMMANDER_CPU: usize = 0;
const TICK_NS: u64 = 200_000;
const ACTIVE_AGE_NS: u64 = 2_000_000;
const STALL_PERIOD_NS: u64 = 100_000_000;
const STALL_NS: u64 = 20_000_000;
const PROBE_PERIOD: u32 = 64;
const ENERGY_WINDOW_NS: u64 = 5_000_000;
const KEY: u64 = 0x9E3779B97F4A7C15;

// İşçi durum satırı (64 bayt = 8×u64). Her işçi yalnızca kendi satırını yazar.
const S_TILES = 0;
const S_LAST = 1;
const S_EMA = 2;
const S_STALL = 3;
const S_FAILS = 4;
const S_MISM = 5;
const S_CPUBAD = 6;
// Komutan panosu: [0]=aktif maske, [1]=toplu çekme, [2]=yoklama periyodu
const B_MASK = 0;
const B_BATCH = 1;
const B_PROBE = 2;
// Kontrol: [0]=go, [1]=stop, [2]=stall sayısı, [3]=emir değişimi, [4]=telemetri okuma,
// [5]=telemetri ns, [6]=enerji pencere sayısı, [7]=enerji kontrolünün son k değeri
const C_GO = 0;
const C_STOP = 1;
const C_STALLS = 2;
const C_ORDERS = 3;
const C_TREADS = 4;
const C_TNS = 5;
const C_EWIN = 6;
const C_KFIN = 7;
// Hareket türü: 0 aynı işçi, 1 aynı CCX farklı işçi, 2 farklı CCX, 3 ilk dokunuş
const K_SAME = 0;
const K_CCX = 1;
const K_CROSS = 2;
const K_FIRST = 3;

// cmd_mask / cmd_batch / ring / cmd_energy: ÖN KAYIT SONRASI kollar.
const Arm = enum { owned, free, cmd, cmd_mask, cmd_batch, ring, cmd_energy };
const Load = enum { none, hog2, stall };

fn hasOrders(arm: Arm) bool {
    return arm == .cmd or arm == .cmd_mask or arm == .cmd_batch;
}

const Opts = struct {
    status_every: u64 = 1,
    padded: bool = false,
    pad_level: u8 = 0,
    probe_period: u32 = PROBE_PERIOD,
    energy_type: u64 = 0,
    energy_config: u64 = 0,
    energy_scale: f64 = 0,
    nocmd: bool = false,
};

fn parseOpts(s: []const u8) !Opts {
    var o = Opts{};
    var it = std.mem.splitScalar(u8, s, ';');
    while (it.next()) |kv| {
        if (kv.len == 0) continue;
        const eq = std.mem.indexOfScalar(u8, kv, '=') orelse return error.BadOpt;
        const k = kv[0..eq];
        const v = kv[eq + 1 ..];
        if (std.mem.eql(u8, k, "se")) {
            o.status_every = try std.fmt.parseInt(u64, v, 10);
        } else if (std.mem.eql(u8, k, "pad")) {
            o.pad_level = try std.fmt.parseInt(u8, v, 10);
            o.padded = o.pad_level != 0;
        } else if (std.mem.eql(u8, k, "pp")) {
            o.probe_period = try std.fmt.parseInt(u32, v, 10);
        } else if (std.mem.eql(u8, k, "et")) {
            o.energy_type = try std.fmt.parseInt(u64, v, 10);
        } else if (std.mem.eql(u8, k, "ec")) {
            o.energy_config = try std.fmt.parseInt(u64, v, 0);
        } else if (std.mem.eql(u8, k, "nc")) {
            o.nocmd = (try std.fmt.parseInt(u8, v, 10)) != 0;
        } else if (std.mem.eql(u8, k, "es")) {
            o.energy_scale = try std.fmt.parseFloat(f64, v);
        } else {
            return error.BadOpt;
        }
    }
    return o;
}

fn nowNs() u64 {
    var ts: std.os.linux.timespec = undefined;
    _ = std.os.linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn sleepNs(ns: u64) void {
    var ts = std.os.linux.timespec{
        .sec = @intCast(ns / 1_000_000_000),
        .nsec = @intCast(ns % 1_000_000_000),
    };
    _ = std.os.linux.nanosleep(&ts, null);
}

fn pinTo(cpu: usize) void {
    var set = std.mem.zeroes(std.os.linux.cpu_set_t);
    set[cpu / @bitSizeOf(usize)] |= @as(usize, 1) << @as(u6, @intCast(cpu % @bitSizeOf(usize)));
    std.os.linux.sched_setaffinity(0, &set) catch @panic("sched_setaffinity başarısız");
}

fn currentCpu() usize {
    var c: usize = 0;
    _ = std.os.linux.getcpu(&c, null);
    return c;
}

fn splitmix(s: *u64) u64 {
    s.* +%= KEY;
    var z = s.*;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

fn inc(p: *u64, d: u64) void {
    // Tek yazıcılı satır: okuma-artır-yaz yeterli (aynı satırı yalnızca sahibi yazar).
    @atomicStore(u64, p, @atomicLoad(u64, p, .monotonic) + d, .release);
}

fn crcStep(c: u64, v: u64) u64 {
    var r = c;
    asm volatile ("crc32q %[v], %[r]" : [r] "+r" (r) : [v] "r" (v));
    return r;
}

fn crcBlk(p: []const u8, seq: u64) u32 {
    var c: u64 = seq & 0xFFFFFFFF;
    var i: usize = 0;
    while (i < TILE) : (i += 8) c = crcStep(c, std.mem.readInt(u64, p[i..][0..8], .little));
    return @truncate(c);
}

/// Vektörleştirilmiş (16 lane): hesap kısa kalır, böylece veri hareketinin maliyeti görünür.
fn computeTile(tiles: []const i8, x: *const [COLS]i8, t: usize, out: *[ROWS]i32) void {
    const V = 16;
    const base = t * TILE;
    for (0..ROWS) |r| {
        var acc: @Vector(V, i32) = @splat(0);
        var c: usize = 0;
        while (c < COLS) : (c += V) {
            const a: @Vector(V, i8) = tiles[base + r * COLS + c ..][0..V].*;
            const b: @Vector(V, i8) = x[c..][0..V].*;
            acc += @as(@Vector(V, i32), a) * @as(@Vector(V, i32), b);
        }
        out[r] = @reduce(.Add, acc);
    }
}

// ---- Vyukov sınırlı MPMC kuyruğu (paylaşılan havuz) ----
// Ön kayıt sonrası düzeltme: itme hiçbir zaman başarısız olmaz; kapasite ≥ 2·nt.
const Cell = struct { seq: usize, val: u32 };

const Queue = struct {
    cells: []Cell,
    mask: usize,
    pos: []usize, // pos[0]=enq, pos[8]=deq (ayrı önbellek satırları)
};

fn qPush(q: *Queue, v: u32) void {
    var pos = @atomicLoad(usize, &q.pos[0], .monotonic);
    while (true) {
        const c = &q.cells[pos & q.mask];
        const seq = @atomicLoad(usize, &c.seq, .acquire);
        const dif: isize = @bitCast(seq -% pos);
        if (dif == 0) {
            if (@cmpxchgWeak(usize, &q.pos[0], pos, pos +% 1, .monotonic, .monotonic) == null) {
                c.val = v;
                @atomicStore(usize, &c.seq, pos +% 1, .release);
                return;
            }
            pos = @atomicLoad(usize, &q.pos[0], .monotonic);
        } else if (dif < 0) {
            std.Thread.yield() catch {};
            pos = @atomicLoad(usize, &q.pos[0], .monotonic);
        } else {
            pos = @atomicLoad(usize, &q.pos[0], .monotonic);
        }
    }
}

fn qPop(q: *Queue) ?u32 {
    var pos = @atomicLoad(usize, &q.pos[8], .monotonic);
    while (true) {
        const c = &q.cells[pos & q.mask];
        const seq = @atomicLoad(usize, &c.seq, .acquire);
        const dif: isize = @bitCast(seq -% (pos +% 1));
        if (dif == 0) {
            if (@cmpxchgWeak(usize, &q.pos[8], pos, pos +% 1, .monotonic, .monotonic) == null) {
                const v = c.val;
                @atomicStore(usize, &c.seq, pos +% q.mask +% 1, .release);
                return v;
            }
            pos = @atomicLoad(usize, &q.pos[8], .monotonic);
        } else if (dif < 0) {
            // Üretici yolda mı? (enq > pos ise bu konumda bir öğe yazılmış olmalı)
            if (@atomicLoad(usize, &q.pos[0], .monotonic) > pos) {
                std.Thread.yield() catch {};
                pos = @atomicLoad(usize, &q.pos[8], .monotonic);
            } else {
                return null;
            }
        } else {
            pos = @atomicLoad(usize, &q.pos[8], .monotonic);
        }
    }
}

// ---- SPSC dairesel bağlantılar (ring kolu): her işçinin tek üreticisi (w-1) ve tek tüketicisi (w) var ----
const Ring = struct {
    buf: []u32, // işçi başına cap öğe
    idx: []u64, // işçi başına 16 u64: [0]=tail (üretici yazar), [8]=head (tüketici yazar)
    cap: usize,
};

fn ringPush(r: *Ring, dst: usize, v: u32) void {
    const tail = @atomicLoad(u64, &r.idx[dst * 16], .monotonic);
    while (true) {
        const head = @atomicLoad(u64, &r.idx[dst * 16 + 8], .acquire);
        if (tail -% head < r.cap) break;
        std.Thread.yield() catch {};
    }
    r.buf[dst * r.cap + @as(usize, @intCast(tail % r.cap))] = v;
    @atomicStore(u64, &r.idx[dst * 16], tail +% 1, .release);
}

fn ringPop(r: *Ring, w: usize) ?u32 {
    const head = @atomicLoad(u64, &r.idx[w * 16 + 8], .monotonic);
    const tail = @atomicLoad(u64, &r.idx[w * 16], .acquire);
    if (head == tail) return null;
    const v = r.buf[w * r.cap + @as(usize, @intCast(head % r.cap))];
    @atomicStore(u64, &r.idx[w * 16 + 8], head +% 1, .release);
    return v;
}

/// İşçi yerel sayaçları (paylaşılan satırlara yazmadan önce birikir).
const Local = struct {
    n: u64 = 0,
    kinds: [4]u64 = .{ 0, 0, 0, 0 },
};

const Shared = struct {
    arm: Arm,
    nw: usize,
    cpus: []const usize,
    nt: usize,
    tiles: []const i8,
    x: *const [COLS]i8,
    ref: []const i32,
    ccx_of_cpu: []const u8,
    q: Queue,
    ring: Ring,
    st: []u64, // MAX_W×8: işçi telemetri satırları
    board: []u64, // 8: komutan panosu
    ctl: []u64, // 8: kontrol
    yw: [][]i64, // işçiye özel kısmi toplamlar (satır başına)
    owned: [][]u32, // owned kolu: işçi başına sabit tile listesi
    last_w: []u8, // tile metadata; indeks t·ms8
    last_ccx: []u8,
    count: []u32, // indeks t·ms32
    seen: []u8,
    smp_ns: [][]u32,
    smp_kind: [][]u8,
    smp_n: []usize,
    kc: []u64, // işçi×kcs tür sayaçları
    ms8: usize,
    ms32: usize,
    kcs: usize,
    sns: usize,
    sts: usize,
    padded: bool,
    status_every: u64,
    probe_period: u32,
    energy_scale: f64,
};

fn processTile(sh: *Shared, w: usize, t: usize, lc: *Local) void {
    const st = sh.st[w * sh.sts ..][0..8];
    const cpu_now = currentCpu();
    if (cpu_now != sh.cpus[w]) inc(&st[S_CPUBAD], 1);
    const ccx_now: u8 = sh.ccx_of_cpu[cpu_now];
    const m8 = t * sh.ms8;
    const m32 = t * sh.ms32;

    // Tile sahipliği bu noktada tekil: kuyruk/halka erişimi (acquire/release) bunu garanti eder.
    const lw = sh.last_w[m8];
    const kind: u8 = if (lw == NONE_W)
        K_FIRST
    else if (lw == w)
        K_SAME
    else if (sh.last_ccx[m8] == ccx_now)
        K_CCX
    else
        K_CROSS;

    const t0 = nowNs();
    var out: [ROWS]i32 = undefined;
    computeTile(sh.tiles, sh.x, t, &out);
    var bad: u64 = 0;
    const yw = sh.yw[w];
    for (0..ROWS) |r| {
        if (out[r] != sh.ref[t * ROWS + r]) bad += 1;
        yw[t * ROWS + r] += out[r];
    }
    const dt = nowNs() - t0;

    if (bad != 0) inc(&st[S_MISM], bad);
    sh.last_w[m8] = @intCast(w);
    sh.last_ccx[m8] = ccx_now;
    sh.count[m32] += 1;
    sh.seen[m8] |= @as(u8, 1) << @intCast(w);
    if (sh.padded) {
        lc.kinds[kind] += 1;
    } else {
        sh.kc[w * sh.kcs + kind] += 1;
    }
    const n = sh.smp_n[w * sh.sns];
    if (n < SAMPLE_CAP) {
        sh.smp_ns[w][n] = @intCast(@min(dt, std.math.maxInt(u32)));
        sh.smp_kind[w][n] = kind;
        sh.smp_n[w * sh.sns] = n + 1;
    }

    lc.n += 1;
    if (sh.status_every <= 1 or lc.n % sh.status_every == 0) {
        const now = nowNs();
        const ema_old = @atomicLoad(u64, &st[S_EMA], .monotonic);
        @atomicStore(u64, &st[S_EMA], if (lc.n == 1) dt else (ema_old * 9 + dt) / 10, .release);
        @atomicStore(u64, &st[S_TILES], lc.n, .release);
        @atomicStore(u64, &st[S_LAST], now, .release);
    }
}

fn worker(sh: *Shared, w: usize) void {
    pinTo(sh.cpus[w]);
    const st = sh.st[w * sh.sts ..][0..8];
    while (@atomicLoad(u64, &sh.ctl[C_GO], .acquire) == 0) std.Thread.yield() catch {};
    var lc = Local{};
    var poll: u32 = 0;
    var own_i: usize = 0;
    while (@atomicLoad(u64, &sh.ctl[C_STOP], .acquire) == 0) {
        const su = @atomicLoad(u64, &st[S_STALL], .acquire);
        if (su != 0 and nowNs() < su) {
            while (nowNs() < su) std.Thread.yield() catch {};
            continue;
        }
        switch (sh.arm) {
            .owned => {
                const lst = sh.owned[w];
                processTile(sh, w, lst[own_i], &lc);
                own_i = (own_i + 1) % lst.len;
            },
            .ring => {
                const t = ringPop(&sh.ring, w) orelse {
                    inc(&st[S_FAILS], 1);
                    std.Thread.yield() catch {};
                    continue;
                };
                processTile(sh, w, t, &lc);
                ringPush(&sh.ring, (w + 1) % sh.nw, t);
            },
            .free, .cmd, .cmd_mask, .cmd_batch, .cmd_energy => {
                var batch: u32 = 1;
                if (sh.arm != .free) {
                    const mask = @atomicLoad(u64, &sh.board[B_MASK], .acquire);
                    const me: u6 = @intCast(w);
                    if ((mask >> me) & 1 == 0) {
                        // Pasif işçi: yalnızca periyodik yoklama.
                        poll +%= 1;
                        if (poll % sh.probe_period != 0) {
                            std.Thread.yield() catch {};
                            continue;
                        }
                    } else {
                        batch = @intCast(@atomicLoad(u64, &sh.board[B_BATCH], .acquire));
                    }
                }
                var i: u32 = 0;
                while (i < batch) : (i += 1) {
                    const t = qPop(&sh.q) orelse {
                        inc(&st[S_FAILS], 1);
                        break;
                    };
                    processTile(sh, w, t, &lc);
                    qPush(&sh.q, t);
                }
            },
        }
    }
    // Son boşaltma: yerel sayaçlar paylaşılan satırlara yazılır.
    @atomicStore(u64, &st[S_TILES], lc.n, .release);
    if (sh.padded) {
        for (0..4) |k| sh.kc[w * sh.kcs + k] = lc.kinds[k];
    }
}

fn energyRaw(fd: std.os.linux.fd_t) u64 {
    var v: u64 = 0;
    _ = std.os.linux.read(fd, @ptrCast(&v), 8);
    return v;
}

fn commander(sh: *Shared, energy_fd: std.os.linux.fd_t) void {
    pinTo(COMMANDER_CPU);
    var next = nowNs() + TICK_NS;
    var cur_mask: u64 = std.math.maxInt(u64);
    var cur_batch: u64 = std.math.maxInt(u64);
    // Enerji geri beslemesi: k (aktif işçi sayısı) üzerinde tepe tırmanma (J/tile'ı azaltmak için).
    var k: usize = sh.nw;
    var dir: isize = -1;
    var prev_metric: f64 = 1e30;
    var e_prev: u64 = if (sh.arm == .cmd_energy) energyRaw(energy_fd) else 0;
    var t_prev: u64 = 0;
    var next_win: u64 = nowNs() + ENERGY_WINDOW_NS;
    while (@atomicLoad(u64, &sh.ctl[C_STOP], .acquire) == 0) {
        while (nowNs() < next) {}
        next += TICK_NS;
        // Yeşil bulut: tüm işçi durum satırlarını oku.
        const t0 = nowNs();
        var mask: u64 = 0;
        var i: usize = 0;
        while (i < sh.nw) : (i += 1) {
            const last = @atomicLoad(u64, &sh.st[i * sh.sts + S_LAST], .acquire);
            if (t0 -| last < ACTIVE_AGE_NS) mask |= @as(u64, 1) << @intCast(i);
        }
        const t1 = nowNs();
        inc(&sh.ctl[C_TREADS], 1);
        inc(&sh.ctl[C_TNS], t1 - t0);

        if (sh.arm == .cmd_energy) {
            if (t0 >= next_win) {
                next_win = t0 + ENERGY_WINDOW_NS;
                const e_now = energyRaw(energy_fd);
                var tot: u64 = 0;
                var j: usize = 0;
                while (j < sh.nw) : (j += 1) tot +%= @atomicLoad(u64, &sh.st[j * sh.sts + S_TILES], .acquire);
                const d_e = @as(f64, @floatFromInt(e_now -% e_prev)) * sh.energy_scale;
                const d_t = tot -% t_prev;
                if (d_t > 0) {
                    const metric = d_e / @as(f64, @floatFromInt(d_t));
                    if (metric > prev_metric) dir = -dir; // kötüleşti: yön değiştir
                    prev_metric = metric;
                    var nk: isize = @as(isize, @intCast(k)) + dir;
                    if (nk < 2) {
                        nk = 2;
                        dir = 1;
                    }
                    if (nk > @as(isize, @intCast(sh.nw))) {
                        nk = @intCast(sh.nw);
                        dir = -1;
                    }
                    k = @intCast(nk);
                }
                e_prev = e_now;
                t_prev = tot;
                inc(&sh.ctl[C_EWIN], 1);
            }
            mask = (@as(u64, 1) << @intCast(k)) - 1;
            const batch: u64 = 1;
            if (mask != cur_mask or batch != cur_batch) {
                @atomicStore(u64, &sh.board[B_MASK], mask, .release);
                @atomicStore(u64, &sh.board[B_BATCH], batch, .release);
                cur_mask = mask;
                cur_batch = batch;
                inc(&sh.ctl[C_ORDERS], 1);
            }
        } else if (hasOrders(sh.arm)) {
            // Mikro tarif: cmd kolları emir yayınlar (v1 ile aynı kurallar).
            if (mask == 0) mask = (@as(u64, 1) << @intCast(sh.nw)) - 1;
            var batch: u64 = if (@popCount(mask) >= 4) 2 else 1;
            if (sh.arm == .cmd_mask) batch = 1;
            if (sh.arm == .cmd_batch) {
                mask = (@as(u64, 1) << @intCast(sh.nw)) - 1;
                batch = 2;
            }
            if (mask != cur_mask or batch != cur_batch) {
                @atomicStore(u64, &sh.board[B_MASK], mask, .release);
                @atomicStore(u64, &sh.board[B_BATCH], batch, .release);
                cur_mask = mask;
                cur_batch = batch;
                inc(&sh.ctl[C_ORDERS], 1);
            }
        }
    }
    sh.ctl[C_KFIN] = k;
}

fn hog(stop: *u64, cpu: usize) void {
    pinTo(cpu);
    var x: u64 = 1;
    while (@atomicLoad(u64, stop, .acquire) == 0) x = x *% KEY +% 1;
    std.mem.doNotOptimizeAway(x);
}

fn disturber(sh: *Shared) void {
    var k: usize = 0;
    while (@atomicLoad(u64, &sh.ctl[C_STOP], .acquire) == 0) {
        sleepNs(STALL_PERIOD_NS);
        if (@atomicLoad(u64, &sh.ctl[C_STOP], .acquire) != 0) break;
        const w = k % sh.nw;
        k += 1;
        @atomicStore(u64, &sh.st[w * sh.sts + S_STALL], nowNs() + STALL_NS, .release);
        inc(&sh.ctl[C_STALLS], 1);
    }
}

fn pct(tmp: []u32, p: usize) u32 {
    if (tmp.len == 0) return 0;
    std.mem.sort(u32, tmp, {}, std.sort.asc(u32));
    return tmp[(tmp.len - 1) * p / 100];
}

fn parseCpus(gpa: std.mem.Allocator, s: []const u8) ![]usize {
    var list = std.array_list.Managed(usize).init(gpa);
    var it = std.mem.splitScalar(u8, s, ',');
    while (it.next()) |part| try list.append(try std.fmt.parseInt(usize, part, 10));
    return list.toOwnedSlice();
}

fn runBench(
    arm: Arm,
    cpus: []const usize,
    ws_kb: usize,
    sec: f64,
    load: Load,
    seed: u64,
    ccx_of_cpu: []const u8,
    opts: Opts,
) !void {
    const pa = std.heap.page_allocator;
    const nw = cpus.len;
    if (nw == 0 or nw > MAX_W) return error.BadWorkers;
    if (cpus[0] == COMMANDER_CPU) return error.CommanderCpuUsed;
    const nt = ws_kb * 1024 / TILE;
    if (nt < nw) return error.WorkingSetTooSmall;

    var s = seed ^ KEY;
    const tiles = try pa.alloc(i8, nt * TILE);
    for (tiles) |*b| b.* = @as(i8, @intCast(@as(i64, @intCast(splitmix(&s) % 255)) - 127));
    var x: [COLS]i8 = undefined;
    for (&x) |*b| b.* = @as(i8, @intCast(@as(i64, @intCast(splitmix(&s) % 255)) - 127));
    const ref = try pa.alloc(i32, nt * ROWS);
    for (0..nt) |t| {
        var out: [ROWS]i32 = undefined;
        computeTile(tiles, &x, t, &out);
        @memcpy(ref[t * ROWS ..][0..ROWS], out[0..]);
    }

    // Kuyruk ve halka kapasitesi ≥ 2·nt (ön kayıt sonrası düzeltme).
    var cap: usize = 2;
    while (cap < 2 * nt) cap *= 2;
    const cells = try pa.alloc(Cell, cap);
    for (cells, 0..) |*c, i| c.* = .{ .seq = i, .val = 0 };
    const pos = try pa.alloc(usize, 16);
    @memset(pos, 0);
    var q = Queue{ .cells = cells, .mask = cap - 1, .pos = pos };
    if (arm != .owned and arm != .ring) {
        for (0..nt) |t| qPush(&q, @intCast(t));
    }
    const ring_buf = try pa.alloc(u32, nw * cap);
    const ring_idx = try pa.alloc(u64, nw * 16);
    @memset(ring_idx, 0);
    var ring = Ring{ .buf = ring_buf, .idx = ring_idx, .cap = cap };
    if (arm == .ring) {
        for (0..nt) |t| ringPush(&ring, t % nw, @intCast(t));
    }

    // Metadata düzeni: pad=1 ise her tile'ın kendi 64B satırı vardır (yanlış paylaşım ablasyonu).
    const ms8: usize = switch (opts.pad_level) { 0 => 1, 1 => 64, else => 128 };
    const ms32: usize = switch (opts.pad_level) { 0 => 1, 1 => 16, else => 32 };
    const kcs: usize = if (opts.padded) 16 else 4;
    const sns: usize = switch (opts.pad_level) { 0 => 1, 1 => 8, else => 16 }; // smp_n: işçi başına 64 B (pad=1) veya 128 B (pad=2)
    const sts: usize = if (opts.pad_level >= 2) 16 else 8; // durum satırları: işçi başına 64 B veya 128 B

    const st = try pa.alloc(u64, MAX_W * 16);
    @memset(st, 0);
    const board = try pa.alloc(u64, 8);
    @memset(board, 0);
    board[B_MASK] = (@as(u64, 1) << @intCast(nw)) - 1;
    board[B_BATCH] = 1;
    board[B_PROBE] = opts.probe_period;
    const ctl = try pa.alloc(u64, 8);
    @memset(ctl, 0);

    const yw = try pa.alloc([]i64, nw);
    const owned = try pa.alloc([]u32, nw);
    const smp_ns = try pa.alloc([]u32, nw);
    const smp_kind = try pa.alloc([]u8, nw);
    const smp_n = try pa.alloc(usize, nw * sns);
    @memset(smp_n, 0);
    const kc = try pa.alloc(u64, nw * kcs);
    @memset(kc, 0);
    for (0..nw) |w| {
        yw[w] = try pa.alloc(i64, nt * ROWS);
        @memset(yw[w], 0);
        smp_ns[w] = try pa.alloc(u32, SAMPLE_CAP);
        smp_kind[w] = try pa.alloc(u8, SAMPLE_CAP);
        var cnt: usize = 0;
        for (0..nt) |t| if (t % nw == w) {
            cnt += 1;
        };
        owned[w] = try pa.alloc(u32, cnt);
        var j: usize = 0;
        for (0..nt) |t| if (t % nw == w) {
            owned[w][j] = @intCast(t);
            j += 1;
        };
    }
    const last_w = try pa.alloc(u8, nt * ms8);
    @memset(last_w, NONE_W);
    const last_ccx = try pa.alloc(u8, nt * ms8);
    @memset(last_ccx, 0);
    const count = try pa.alloc(u32, nt * ms32);
    @memset(count, 0);
    const seen = try pa.alloc(u8, nt * ms8);
    @memset(seen, 0);

    // Enerji sayacı (yalnızca cmd_energy): power PMU, sistem geneli paket enerjisi.
    var energy_fd: std.os.linux.fd_t = -1;
    if (arm == .cmd_energy) {
        if (opts.energy_scale <= 0) return error.EnergyOptsMissing;
        var attr = std.mem.zeroes(std.os.linux.perf_event_attr);
        attr.type = @enumFromInt(@as(u32, @intCast(opts.energy_type)));
        attr.config = opts.energy_config;
        attr.size = @sizeOf(std.os.linux.perf_event_attr);
        const rc = std.os.linux.perf_event_open(&attr, -1, 0, -1, 0);
        if (@as(isize, @bitCast(rc)) < 0) return error.PerfOpenFailed;
        energy_fd = @intCast(rc);
    }

    var sh = Shared{
        .arm = arm,
        .nw = nw,
        .cpus = cpus,
        .nt = nt,
        .tiles = tiles,
        .x = &x,
        .ref = ref,
        .ccx_of_cpu = ccx_of_cpu,
        .q = q,
        .ring = ring,
        .st = st,
        .board = board,
        .ctl = ctl,
        .yw = yw,
        .owned = owned,
        .last_w = last_w,
        .last_ccx = last_ccx,
        .count = count,
        .seen = seen,
        .smp_ns = smp_ns,
        .smp_kind = smp_kind,
        .smp_n = smp_n,
        .kc = kc,
        .ms8 = ms8,
        .ms32 = ms32,
        .kcs = kcs,
        .sns = sns,
        .sts = sts,
        .padded = opts.padded,
        .status_every = opts.status_every,
        .probe_period = opts.probe_period,
        .energy_scale = opts.energy_scale,
    };

    var threads: [MAX_W]std.Thread = undefined;
    for (0..nw) |w| threads[w] = try std.Thread.spawn(.{}, worker, .{ &sh, w });
    const cmd_t: ?std.Thread = if (opts.nocmd) null else try std.Thread.spawn(.{}, commander, .{ &sh, energy_fd });
    var hogs: [2]std.Thread = undefined;
    var n_hogs: usize = 0;
    var hog_stop: u64 = 0;
    if (load == .hog2) {
        const hog_cpus = [2]usize{ cpus[0], cpus[nw - 1] };
        for (hog_cpus) |c| {
            hogs[n_hogs] = try std.Thread.spawn(.{}, hog, .{ &hog_stop, c });
            n_hogs += 1;
        }
    }
    var dist: ?std.Thread = null;
    if (load == .stall) dist = try std.Thread.spawn(.{}, disturber, .{&sh});

    const e_start: u64 = if (energy_fd >= 0) energyRaw(energy_fd) else 0;
    const t_start = nowNs();
    @atomicStore(u64, &ctl[C_GO], 1, .release);
    sleepNs(@intFromFloat(sec * 1e9));
    @atomicStore(u64, &ctl[C_STOP], 1, .release);
    for (threads[0..nw]) |t| t.join();
    if (cmd_t) |ct| ct.join();
    @atomicStore(u64, &hog_stop, 1, .release);
    for (hogs[0..n_hogs]) |t| t.join();
    if (dist) |d| d.join();
    const elapsed_ns = nowNs() - t_start;
    const e_end: u64 = if (energy_fd >= 0) energyRaw(energy_fd) else 0;
    const e_j: f64 = if (energy_fd >= 0) @as(f64, @floatFromInt(e_end -% e_start)) * opts.energy_scale else 0;

    // Tile korunumu: işçiler durduktan sonra kuyruk / halkalar boşaltılır. drained == nt, dup == 0 olmalı.
    var drained: u64 = 0;
    var dup: u64 = 0;
    if (arm == .owned) {
        drained = nt; // sahiplik listeleri tile'ları bölüşür
    } else {
        const seen_q = try pa.alloc(u8, nt);
        @memset(seen_q, 0);
        if (arm == .ring) {
            for (0..nw) |w| {
                while (ringPop(&ring, w)) |t| {
                    drained += 1;
                    if (seen_q[t] != 0) dup += 1;
                    seen_q[t] = 1;
                }
            }
        } else {
            while (qPop(&q)) |t| {
                drained += 1;
                if (seen_q[t] != 0) dup += 1;
                seen_q[t] = 1;
            }
        }
    }

    // Son kontrol (süre dışı): Σ_w y_w == Σ_t sayım_t · ref_t, satır satır.
    var y_mism: u64 = 0;
    for (0..nt * ROWS) |i| {
        var y: i64 = 0;
        for (0..nw) |w| y += yw[w][i];
        const exp = @as(i64, count[(i / ROWS) * ms32]) * @as(i64, ref[i]);
        if (y != exp) y_mism += 1;
    }

    var tiles_total: u64 = 0;
    var dist_sum: u64 = 0;
    for (0..nt) |t| {
        tiles_total += count[t * ms32];
        dist_sum += @popCount(seen[t * ms8]);
    }
    var mism: u64 = 0;
    var fails: u64 = 0;
    var cpubad: u64 = 0;
    var tiles_w: [MAX_W]u64 = undefined;
    @memset(&tiles_w, 0);
    for (0..nw) |w| {
        mism += st[w * sh.sts + S_MISM];
        fails += st[w * sh.sts + S_FAILS];
        cpubad += st[w * sh.sts + S_CPUBAD];
        tiles_w[w] = st[w * sh.sts + S_TILES];
    }
    var kind_n: [4]u64 = .{ 0, 0, 0, 0 };
    for (0..nw) |w| for (0..4) |k| {
        kind_n[k] += kc[w * kcs + k];
    };
    // Tür başına medyan/p99 (örnek tavanı içinde).
    var p50: [4]u32 = .{ 0, 0, 0, 0 };
    var p99: [4]u32 = .{ 0, 0, 0, 0 };
    var kk: usize = 0;
    while (kk < 4) : (kk += 1) {
        var total: usize = 0;
        for (0..nw) |w| {
            for (smp_kind[w][0..smp_n[w * sns]]) |kd| {
                if (kd == kk) total += 1;
            }
        }
        const tmp = try pa.alloc(u32, total);
        defer pa.free(tmp);
        var j: usize = 0;
        for (0..nw) |w| {
            for (smp_kind[w][0..smp_n[w * sns]], smp_ns[w][0..smp_n[w * sns]]) |kd, v| {
                if (kd == kk) {
                    tmp[j] = v;
                    j += 1;
                }
            }
        }
        if (total > 0) {
            p50[kk] = pct(tmp, 50);
            p99[kk] = pct(tmp, 99);
        }
    }

    const tiles_per_s = @as(f64, @floatFromInt(tiles_total)) / (@as(f64, @floatFromInt(elapsed_ns)) * 1e-9);
    const tread = ctl[C_TREADS];
    const tns_per_read: f64 = if (tread > 0) @as(f64, @floatFromInt(ctl[C_TNS])) / @as(f64, @floatFromInt(tread)) else 0;
    // Zig biçim çağrısı en fazla 32 argüman alır: çıktı iki parçada, aynı stderr satırına yazılır.
    std.debug.print(
        "{{\"arm\":\"{s}\",\"load\":\"{s}\",\"ws_kb\":{d},\"nt\":{d},\"nw\":{d}," ++
            "\"elapsed_s\":{d:.4},\"tiles\":{d},\"tiles_per_s\":{d:.1},\"passes\":{d:.2}," ++
            "\"mism\":{d},\"y_mism\":{d},\"cpu_bad\":{d},\"fails\":{d}," ++
            "\"stalls\":{d},\"orders\":{d},\"tele_reads\":{d},\"tele_ns_per_read\":{d:.1}," ++
            "\"dist_mean\":{d:.3},\"hop_same\":{d},\"hop_ccx\":{d},\"hop_cross\":{d},\"first\":{d}," ++
            "\"drained\":{d},\"dup\":{d},",
        .{
            @tagName(arm),                                  @tagName(load),                ws_kb,            nt,
            nw,                                             @as(f64, @floatFromInt(elapsed_ns)) * 1e-9, tiles_total,
            tiles_per_s,                                    @as(f64, @floatFromInt(tiles_total)) / @as(f64, @floatFromInt(nt)),
            mism,                                           y_mism,                        cpubad,           fails,
            ctl[C_STALLS],                                  ctl[C_ORDERS],                 tread,            tns_per_read,
            @as(f64, @floatFromInt(dist_sum)) / @as(f64, @floatFromInt(nt)),
            kind_n[K_SAME],                                 kind_n[K_CCX],                 kind_n[K_CROSS],  kind_n[K_FIRST],
            drained,                                        dup,
        },
    );
    std.debug.print(
        "\"p50_same_ns\":{d},\"p99_same_ns\":{d},\"p50_ccx_ns\":{d},\"p99_ccx_ns\":{d}," ++
            "\"p50_cross_ns\":{d},\"p99_cross_ns\":{d},\"p50_first_ns\":{d},\"p99_first_ns\":{d}," ++
            "\"tiles_w0\":{d},\"tiles_w1\":{d},\"tiles_w2\":{d},\"tiles_w3\":{d}," ++
            "\"tiles_w4\":{d},\"tiles_w5\":{d},\"tiles_w6\":{d},\"tiles_w7\":{d}," ++
            "\"ewin\":{d},\"k_final\":{d},\"e_j\":{d:.4}}}\n",
        .{
            p50[K_SAME],   p99[K_SAME],   p50[K_CCX],   p99[K_CCX],
            p50[K_CROSS],  p99[K_CROSS],  p50[K_FIRST], p99[K_FIRST],
            tiles_w[0],    tiles_w[1],    tiles_w[2],   tiles_w[3],
            tiles_w[4],    tiles_w[5],    tiles_w[6],   tiles_w[7],
            ctl[C_EWIN],   ctl[C_KFIN],   e_j,
        },
    );
}

fn probeChase(ws_kb: usize, cpu: usize, warm: ?usize) !void {
    const pa = std.heap.page_allocator;
    const lines = ws_kb * 1024 / 64;
    const perm = try pa.alloc(u32, lines);
    for (perm, 0..) |*p, i| p.* = @intCast(i);
    var s: u64 = KEY ^ ws_kb;
    var i: usize = lines - 1;
    while (i > 0) : (i -= 1) {
        const j: usize = @intCast(splitmix(&s) % i);
        const tmp = perm[i];
        perm[i] = perm[j];
        perm[j] = tmp;
    }
    // Sattolo: tek döngü. Her satır (64 bayt) sonraki satırın indeksini tutar.
    const buf = try pa.alloc(u64, lines * 8);
    for (0..lines) |k| buf[@as(usize, perm[k]) * 8] = perm[(k + 1) % lines];
    pinTo(cpu);
    var dt: u64 = 0;
    var steps: usize = 0;
    if (warm) |wc| {
        // Her tur: veri önce `wc` çekirdeğinin önbelleğine çekilir; ölçüm bu CPU'da TEK tam geçiştir.
        for (0..8) |_| {
            const th = try std.Thread.spawn(.{}, warmWalk, .{ buf, @as(u64, perm[0]), lines, wc });
            th.join();
            var idx: u64 = perm[0];
            const t0 = nowNs();
            for (0..lines) |_| idx = buf[@as(usize, @intCast(idx)) * 8];
            dt += nowNs() - t0;
            steps += lines;
            std.mem.doNotOptimizeAway(idx);
        }
    } else {
        steps = @max(4_000_000, lines * 4);
        var idx: u64 = perm[0];
        for (0..steps / 4) |_| idx = buf[@as(usize, @intCast(idx)) * 8];
        const t0 = nowNs();
        for (0..steps) |_| idx = buf[@as(usize, @intCast(idx)) * 8];
        dt = nowNs() - t0;
        std.mem.doNotOptimizeAway(idx);
    }
    std.debug.print("{{\"probe\":\"chase\",\"ws_kb\":{d},\"cpu\":{d},\"warm\":{d},\"steps\":{d},\"ns_per_access\":{d:.3}}}\n", .{
        ws_kb, cpu, if (warm) |wc| @as(i64, @intCast(wc)) else @as(i64, -1), steps,
        @as(f64, @floatFromInt(dt)) / @as(f64, @floatFromInt(steps)),
    });
}

fn warmWalk(buf: []u64, start: u64, steps: usize, cpu: usize) void {
    pinTo(cpu);
    var idx = start;
    for (0..steps) |_| idx = buf[@as(usize, @intCast(idx)) * 8];
    std.mem.doNotOptimizeAway(idx);
}

fn probeCrc() void {
    pinTo(1);
    var buf: [TILE]u8 align(64) = undefined;
    var out: [TILE]u8 align(64) = undefined;
    var s: u64 = KEY;
    for (&buf) |*b| b.* = @truncate(splitmix(&s));
    const n: u64 = 200_000;
    var acc: u64 = 0;
    const t0 = nowNs();
    for (0..n) |k| acc +%= crcBlk(&buf, k);
    const t_crc = nowNs() - t0;
    const t1 = nowNs();
    for (0..n) |_| {
        @memcpy(&out, &buf);
        std.mem.doNotOptimizeAway(&out);
    }
    const t_copy = nowNs() - t1;
    std.mem.doNotOptimizeAway(acc);
    std.mem.doNotOptimizeAway(out[0]);
    std.debug.print("{{\"probe\":\"crc\",\"n\":{d},\"crc_ns_per_4k\":{d:.1},\"copy_ns_per_4k\":{d:.1}}}\n", .{
        n,
        @as(f64, @floatFromInt(t_crc)) / @as(f64, @floatFromInt(n)),
        @as(f64, @floatFromInt(t_copy)) / @as(f64, @floatFromInt(n)),
    });
}

/// Paylaşılan satır çekişmesi: her işçi aynı (paylaşılan) ya da kendi (özel) sayacına atomik toplar.
fn hotWorker(cell: *u64, iters: u64, cpu: usize, go: *const u64, out_ns: *u64) void {
    pinTo(cpu);
    while (@atomicLoad(u64, go, .acquire) == 0) std.Thread.yield() catch {};
    const t0 = nowNs();
    var i: u64 = 0;
    while (i < iters) : (i += 1) _ = @atomicRmw(u64, cell, .Add, 1, .monotonic);
    out_ns.* = nowNs() - t0;
}

fn probeHot(gpa: std.mem.Allocator, cpus: []const usize, shared: bool, iters: u64) !void {
    const pa = std.heap.page_allocator;
    const nw = cpus.len;
    const cells = try pa.alloc(u64, nw * 8);
    @memset(cells, 0);
    const outs = try pa.alloc(u64, nw);
    @memset(outs, 0);
    var go: u64 = 0;
    var threads: [MAX_W]std.Thread = undefined;
    for (0..nw) |w| {
        const cell: *u64 = if (shared) &cells[0] else &cells[w * 8];
        threads[w] = try std.Thread.spawn(.{}, hotWorker, .{ cell, iters, cpus[w], &go, &outs[w] });
    }
    @atomicStore(u64, &go, 1, .release);
    for (threads[0..nw]) |t| t.join();
    var sum: f64 = 0;
    for (outs) |o| sum += @as(f64, @floatFromInt(o)) / @as(f64, @floatFromInt(iters));
    _ = gpa;
    std.debug.print("{{\"probe\":\"hot\",\"shared\":{d},\"nw\":{d},\"iters\":{d},\"ns_per_op_mean\":{d:.2}}}\n", .{
        @as(u8, if (shared) 1 else 0), nw, iters, sum / @as(f64, @floatFromInt(nw)),
    });
}

/// Akış okuma bant genişliği: her işçi tampondaki kendi dilimini okur; GB/s toplamı.
fn bwWorker(slice: []const u64, go: *const u64, cpu: usize, passes: usize, out_ns: *u64) void {
    pinTo(cpu);
    while (@atomicLoad(u64, go, .acquire) == 0) std.Thread.yield() catch {};
    const t0 = nowNs();
    var acc: u64 = 0;
    for (0..passes) |_| {
        for (slice) |x| acc +%= x;
    }
    out_ns.* = nowNs() - t0;
    std.mem.doNotOptimizeAway(acc);
}

fn warmRead(buf: []const u64, cpu: usize) void {
    pinTo(cpu);
    var acc: u64 = 0;
    for (buf) |x| acc +%= x;
    std.mem.doNotOptimizeAway(acc);
}

/// warm verilirse: her tur önce veri `warm` çekirdeğinin (başka CCX olabilir) önbelleğine çekilir,
/// sonra okuyucu işçiler kendi dilimlerini okur; ölçüm yalnızca okuma geçişidir. Tur ortancası raporlanır.
fn probeBw(gpa: std.mem.Allocator, cpus: []const usize, ws_kb: usize, passes: usize, warm: ?usize) !void {
    _ = gpa;
    const pa = std.heap.page_allocator;
    const nw = cpus.len;
    const words = ws_kb * 1024 / 8;
    const buf = try pa.alloc(u64, words);
    for (buf, 0..) |*b, i| b.* = @as(u64, i) *% KEY;
    const outs = try pa.alloc(u64, nw);
    const cycles: usize = if (warm != null) 6 else 1;
    const per = words / nw;
    const gbs = try pa.alloc(f64, cycles);
    for (0..cycles) |c| {
        if (warm) |wc| {
            const th = try std.Thread.spawn(.{}, warmRead, .{ buf, wc });
            th.join();
        }
        @memset(outs, 0);
        var go: u64 = 0;
        var threads: [MAX_W]std.Thread = undefined;
        for (0..nw) |w| {
            const slice = buf[w * per ..][0..per];
            threads[w] = try std.Thread.spawn(.{}, bwWorker, .{ slice, &go, cpus[w], passes, &outs[w] });
        }
        @atomicStore(u64, &go, 1, .release);
        for (threads[0..nw]) |t| t.join();
        var max_ns: u64 = 0;
        for (outs) |o| max_ns = @max(max_ns, o);
        const used_passes: usize = if (warm != null) 1 else passes;
        gbs[c] = @as(f64, @floatFromInt(per * nw * 8 * used_passes)) / @as(f64, @floatFromInt(max_ns));
    }
    std.mem.sort(f64, gbs, {}, std.sort.asc(f64));
    std.debug.print("{{\"probe\":\"bw\",\"ws_kb\":{d},\"nw\":{d},\"passes\":{d},\"warm\":{d},\"cycles\":{d},\"gb_per_s\":{d:.2}}}\n", .{
        ws_kb, nw, passes, if (warm) |wc| @as(i64, @intCast(wc)) else @as(i64, -1), cycles, gbs[cycles / 2],
    });
}

/// Yalnız hesap: bellek yok, metadata yok. Çapraz-CCX cezası hesap kaynaklı mı, bellek sistemi mi?
/// use_getcpu=1 ise her yinelemede getcpu syscall'ı yapılır (owned yolunun tile başına maliyeti).
fn compWorker(cpu: usize, sec_ns: u64, use_getcpu: bool, go: *const u64, out: *u64) void {
    pinTo(cpu);
    var tile: [TILE]i8 = undefined;
    var s: u64 = KEY ^ cpu;
    for (&tile) |*b| b.* = @as(i8, @intCast(@as(i64, @intCast(splitmix(&s) % 255)) - 127));
    var x: [COLS]i8 = undefined;
    for (&x) |*b| b.* = @as(i8, @intCast(@as(i64, @intCast(splitmix(&s) % 255)) - 127));
    while (@atomicLoad(u64, go, .acquire) == 0) std.Thread.yield() catch {};
    const end = nowNs() + sec_ns;
    var o: [ROWS]i32 = undefined;
    var n: u64 = 0;
    var sink: i32 = 0;
    while (nowNs() < end) : (n += 1) {
        if (use_getcpu) sink +%= @intCast(currentCpu());
        computeTile(tile[0..], &x, 0, &o);
        sink +%= o[0];
    }
    std.mem.doNotOptimizeAway(sink);
    out.* = n;
}

fn probeComp(cpus: []const usize, sec: f64, use_getcpu: bool) !void {
    const pa = std.heap.page_allocator;
    const outs = try pa.alloc(u64, cpus.len);
    @memset(outs, 0);
    var go: u64 = 0;
    var threads: [MAX_W]std.Thread = undefined;
    const sec_ns: u64 = @intFromFloat(sec * 1e9);
    for (cpus, 0..) |c, w| threads[w] = try std.Thread.spawn(.{}, compWorker, .{ c, sec_ns, use_getcpu, &go, &outs[w] });
    @atomicStore(u64, &go, 1, .release);
    for (threads[0..cpus.len]) |t| t.join();
    var total: u64 = 0;
    for (outs) |o| total += o;
    std.debug.print("{{\"probe\":\"comp\",\"nw\":{d},\"getcpu\":{d},\"iter_per_s_total\":{d:.0}}}\n", .{
        cpus.len, @as(u8, if (use_getcpu) 1 else 0), @as(f64, @floatFromInt(total)) / sec,
    });
}

/// Yalnız çekirdek: her işçinin kendi tile tamponu (ws_kb), metadata/örnekleme/birikim yok.
/// Owned yolunun en küçük karşılığı; çapraz cezanın veri yolundan mı bookkeeping'den mi geldiğini ayırır.
var K_MLW: []u8 = &.{};
var K_MLC: []u8 = &.{};
var K_MC: []u32 = &.{};
var K_MSEEN: []u8 = &.{};

fn kernWorker(cpu: usize, ws_kb: usize, sec_ns: u64, flags: u16, go: *const u64, out: *u64, w: usize, nw: usize, shared: []const i8, refs: []const i32) void {
    pinTo(cpu);
    const nt = ws_kb * 1024 / TILE;
    const pa = std.heap.page_allocator;
    const tiles: []i8 = if ((flags & 64) != 0) @constCast(shared) else (pa.alloc(i8, nt * TILE) catch @panic("oom"));
    const samp = pa.alloc(u32, 1 << 19) catch @panic("oom");
    const meta = pa.alloc(u32, nt * 16) catch @panic("oom");
    const stline = pa.alloc(u64, 8) catch @panic("oom");
    var s: u64 = KEY ^ cpu ^ ws_kb;
    if ((flags & 64) == 0) {
        for (tiles) |*b| b.* = @as(i8, @intCast(@as(i64, @intCast(splitmix(&s) % 255)) - 127));
    }
    var x: [COLS]i8 = undefined;
    for (&x) |*b| b.* = @as(i8, @intCast(@as(i64, @intCast(splitmix(&s) % 255)) - 127));
    const ywv = pa.alloc(i64, nt * ROWS) catch @panic("oom");

    @memset(ywv, 0);
    var o: [ROWS]i32 = undefined;
    var sink: i32 = 0;
    for (0..nt) |t| computeTile(tiles, &x, if (flags & 64 != 0) w + nw * t else t, &o);
    while (@atomicLoad(u64, go, .acquire) == 0) std.Thread.yield() catch {};
    const end = nowNs() + sec_ns;
    var n: u64 = 0;
    var t: usize = 0;
    var sn: usize = 0;
    while (nowNs() < end) {
        for (0..256) |_| {
            var c0: usize = 0;
            if (flags & 1 != 0) c0 = currentCpu();
            const t0 = if (flags & 2 != 0) nowNs() else 0;
            const gt: usize = if (flags & 64 != 0) w + nw * t else t;
            computeTile(tiles, &x, gt, &o);
            sink +%= o[0];
            if (flags & 128 != 0) {
                for (0..ROWS) |r| ywv[t * ROWS + r] += o[r];
            }
            if (flags & 512 != 0) {
                const mlw = K_MLW;
                const mlc = K_MLC;
                const mc = K_MC;
                const mseen = K_MSEEN;
                const ms8: usize = if (flags & 1024 != 0) 128 else 64;
                const ms32: usize = ms8 / 4;
                const lw = mlw[gt * ms8];
                const lc = mlc[gt * ms8];
                const kd: u8 = if (lw == 0xFF) 3 else if (lw == w) 0 else if (lc == @as(u8, @intCast(w & 1))) 1 else 2;
                sink +%= kd;
                mlw[gt * ms8] = @intCast(w);
                mlc[gt * ms8] = @intCast(w & 1);
                mc[gt * ms32] += 1;
                mseen[gt * ms8] |= @as(u8, 1) << @intCast(w & 7);
            }
            if (flags & 256 != 0) {
                for (0..ROWS) |r| {
                    if (o[r] != refs[gt * ROWS + r]) sink +%= 1;
                }
            }
            if (flags & 2 != 0) sink +%= @bitCast(@as(u32, @truncate(nowNs() - t0)));
            if (flags & 4 != 0) {
                samp[sn] = @truncate(c0 + 1);
                sn = (sn + 1) % samp.len;
            }
            if (flags & 8 != 0) {
                meta[t * 16] +%= 1;
            }
            if (flags & 16 != 0) {
                stline[0] = n;
                @atomicStore(u64, &stline[1], n, .release);
            }
            t += 1;
            if (t == nt) t = 0;
        }
        n += 256;
    }
    std.mem.doNotOptimizeAway(sink);
    out.* = n;
}

fn probeKern(cpus: []const usize, ws_kb: usize, sec: f64, flags: u16) !void {
    const pa = std.heap.page_allocator;
    const nt_one = ws_kb * 1024 / TILE;
    const shared = try pa.alloc(i8, nt_one * TILE * cpus.len);
    const refs = try pa.alloc(i32, nt_one * ROWS * cpus.len);
    @memset(refs, 0);
    const ntot = nt_one * cpus.len;
    K_MLW = try pa.alloc(u8, ntot * 128);
    @memset(K_MLW, 0xFF);
    K_MLC = try pa.alloc(u8, ntot * 128);
    @memset(K_MLC, 0);
    K_MC = try pa.alloc(u32, ntot * 32);
    @memset(K_MC, 0);
    K_MSEEN = try pa.alloc(u8, ntot * 128);
    @memset(K_MSEEN, 0);
    var sh_seed: u64 = KEY ^ ws_kb;
    for (shared) |*b| b.* = @as(i8, @intCast(@as(i64, @intCast(splitmix(&sh_seed) % 255)) - 127));
    const outs = try pa.alloc(u64, cpus.len);
    @memset(outs, 0);
    var go: u64 = 0;
    var threads: [MAX_W]std.Thread = undefined;
    const sec_ns: u64 = @intFromFloat(sec * 1e9);
    for (cpus, 0..) |c, w| threads[w] = try std.Thread.spawn(.{}, kernWorker, .{ c, ws_kb, sec_ns, flags, &go, &outs[w], w, cpus.len, shared, refs });
    // İşçiler ısınana kadar bekle: go bayrağı ısınma sonrası ayarlanır (kernWorker içinde beklenir).
    sleepNs(300 * std.time.ns_per_ms);
    @atomicStore(u64, &go, 1, .release);
    for (threads[0..cpus.len]) |t| t.join();
    var total: u64 = 0;
    for (outs) |o| total += o;
    std.debug.print("{{\"probe\":\"kern\",\"nw\":{d},\"ws_kb\":{d},\"flags\":{d},\"tiles_per_s_total\":{d:.0}}}\n", .{
        cpus.len, ws_kb, flags, @as(f64, @floatFromInt(total)) / sec,
    });
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    var args_it = try std.process.Args.Iterator.initAllocator(init.minimal.args, gpa);
    defer args_it.deinit();
    var args = std.array_list.Managed([]const u8).init(gpa);
    defer args.deinit();
    while (args_it.next()) |arg| try args.append(arg);

    if (args.items.len < 2) {
        std.debug.print("kullanım: kolektif run <arm> <cpus> <ws_kb> <sn> <load> <tohum> <ccx_csv> [seçenekler] | probe-chase <ws_kb> <cpu> [warm] | probe-crc | probe-hot <cpus> <0|1> <iter> | probe-bw <cpus> <ws_kb> <geçiş>\n", .{});
        return error.MissingArgument;
    }
    const mode = args.items[1];
    if (std.mem.eql(u8, mode, "run")) {
        if (args.items.len != 9 and args.items.len != 10) return error.BadArgs;
        const arm: Arm = std.meta.stringToEnum(Arm, args.items[2]) orelse return error.BadArm;
        const cpus = try parseCpus(gpa, args.items[3]);
        const ws_kb = try std.fmt.parseInt(usize, args.items[4], 10);
        const sec = try std.fmt.parseFloat(f64, args.items[5]);
        const load: Load = std.meta.stringToEnum(Load, args.items[6]) orelse return error.BadLoad;
        const seed = try std.fmt.parseInt(u64, args.items[7], 10);
        const ccx_list = try parseCpus(gpa, args.items[8]);
        const ccx = try gpa.alloc(u8, ccx_list.len);
        for (ccx_list, 0..) |v, i| ccx[i] = @intCast(v);
        const opts = if (args.items.len == 10) try parseOpts(args.items[9]) else Opts{};
        try runBench(arm, cpus, ws_kb, sec, load, seed, ccx, opts);
    } else if (std.mem.eql(u8, mode, "probe-chase")) {
        if (args.items.len != 4 and args.items.len != 5) return error.BadArgs;
        const warm: ?usize = if (args.items.len == 5) try std.fmt.parseInt(usize, args.items[4], 10) else null;
        try probeChase(try std.fmt.parseInt(usize, args.items[2], 10), try std.fmt.parseInt(usize, args.items[3], 10), warm);
    } else if (std.mem.eql(u8, mode, "probe-crc")) {
        probeCrc();
    } else if (std.mem.eql(u8, mode, "probe-hot")) {
        if (args.items.len != 5) return error.BadArgs;
        const cpus = try parseCpus(gpa, args.items[2]);
        const shared = (try std.fmt.parseInt(u8, args.items[3], 10)) != 0;
        try probeHot(gpa, cpus, shared, try std.fmt.parseInt(u64, args.items[4], 10));
    } else if (std.mem.eql(u8, mode, "probe-comp")) {
        if (args.items.len != 5) return error.BadArgs;
        const cpus = try parseCpus(gpa, args.items[2]);
        try probeComp(cpus, try std.fmt.parseFloat(f64, args.items[3]), (try std.fmt.parseInt(u8, args.items[4], 10)) != 0);
    } else if (std.mem.eql(u8, mode, "probe-kern")) {
        if (args.items.len != 5 and args.items.len != 6) return error.BadArgs;
        const cpus = try parseCpus(gpa, args.items[2]);
        try probeKern(cpus, try std.fmt.parseInt(usize, args.items[3], 10), try std.fmt.parseFloat(f64, args.items[4]), if (args.items.len > 5) try std.fmt.parseInt(u16, args.items[5], 10) else 0);
    } else if (std.mem.eql(u8, mode, "probe-bw")) {
        if (args.items.len != 5 and args.items.len != 6) return error.BadArgs;
        const cpus = try parseCpus(gpa, args.items[2]);
        const warm: ?usize = if (args.items.len == 6) try std.fmt.parseInt(usize, args.items[5], 10) else null;
        try probeBw(gpa, cpus, try std.fmt.parseInt(usize, args.items[3], 10), try std.fmt.parseInt(usize, args.items[4], 10), warm);
    } else {
        return error.UnknownMode;
    }
}
