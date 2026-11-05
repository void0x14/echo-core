//! Deney S (Satürn): kolektif dolaşım tarifinin Ryzen 5 3600 üzerindeki gözlemlenebilir karşılıkları.
//! Bu ikili yazılımla fiziksel devre kurmaz: yalnızca komut akışı üretir ve donanımın kendi
//! koherens, önbellek, fabric ve güç davranışını PMU, MSR ve zamanlama ile ölçer.
//!
//! Alt komutlar (çıktı: stderr'de tek satır JSON):
//!   c2c <a> <b> <n>                         tek satır, tek yön çekirdekler arası gecikme
//!   dalga <komutan> <izleyiciler> <n>       emir dalgası: bir yazma, birçok izleyici
//!   satir <a> <b> <n> <l1|dirty|clean|dram>  tek transfer: satır b'de iken a'nın yükü
//!   akis <cpular> <kb> <sn> <r|w>           sayaç kalibrasyonu için ardışık okuma/yazma
//!   merdiven <ad>                           bit yolu merdiveni (cycle/iş)
//!   pencere <tür> <N>                       ROB/PRF/LDQ/STQ pencere probu (Wong yöntemi)
//!   halka <sahipli|serbest|park> <cpular> <satır> <sn> <tohum>
//!   firtina <sakin|sabit|gezen|bulut|bulut_j|rol|rol_k|rol_ayna> <cpular> <satır> <sn> <tohum> <düşüş_cpu>
//!   patlama <yok|kardes|ccx|capraz|bozuk|bozuk_ccx|kendi> <satır> <k> <sn> <tohum>   keşif: talepsiz teslim
//!   dusus <satır> <yazan_cpu> <düşen_cpu> <w|r> <tekrar>   keşif: kirli/temiz düşüş kalibrasyonu
//!   rezervuar <ortak|ozel|yok> <okuyucular> <gecikmeler> <T_tick> <adım> <G> <K> <tohum> <çıktı>
//!   sezai <ortak|ozel> <okuyucular> <gecikme> <T_tick> <G> <K> <tohum> <dosya>   canlı; yönetici tools/run_sezai.py
//!   kuplaj <avx|bos> <komutanlar> <okuyucular> <T_tick> <adım> <tohum> <alu|zincir> <çıktı>
//!   gezgin <cpular> <tau_ms> <sn>

const std = @import("std");
const linux = std.os.linux;

const KEY: u64 = 0x9E3779B97F4A7C15;
const LINE = 64;
const HUGE = 2 << 20;
const Line = extern struct { w: [8]u64 align(64) };
const Pad = extern struct { v: u64 align(64) };

// ---- düşük seviye ----

inline fn tsc() u64 {
    var lo: u32 = undefined;
    var hi: u32 = undefined;
    asm volatile ("lfence\nrdtsc\nlfence"
        : [lo] "={eax}" (lo),
          [hi] "={edx}" (hi),
    );
    return (@as(u64, hi) << 32) | lo;
}

/// ecx=0 MPERF, ecx=1 APERF. Kullanıcı seviyesinden okunur (Zen 2 `rdpru`).
inline fn rdpru(comptime which: u32) u64 {
    var lo: u32 = undefined;
    var hi: u32 = undefined;
    asm volatile (".byte 0x0f,0x01,0xfd"
        : [lo] "={eax}" (lo),
          [hi] "={edx}" (hi),
        : [c] "{ecx}" (which),
    );
    return (@as(u64, hi) << 32) | lo;
}

fn nowNs() u64 {
    var ts: linux.timespec = undefined;
    _ = linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn pinTo(cpu: usize) void {
    var set = std.mem.zeroes(linux.cpu_set_t);
    set[cpu / @bitSizeOf(usize)] |= @as(usize, 1) << @as(u6, @intCast(cpu % @bitSizeOf(usize)));
    linux.sched_setaffinity(0, &set) catch @panic("sched_setaffinity");
}

fn splitmix(s: *u64) u64 {
    s.* +%= KEY;
    var z = s.*;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

inline fn xs(s: *u64) u64 {
    var x = s.*;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    s.* = x;
    return x *% 2685821657736338717;
}

inline fn range(r: u64, n: u64) u64 {
    return @intCast((@as(u128, r) * n) >> 64);
}

inline fn clflushopt(p: *const anyopaque) void {
    asm volatile ("clflushopt (%[p])"
        : // çıktı yok
        : [p] "r" (p),
    );
}

fn tscMhz() f64 {
    const t0 = tsc();
    const n0 = nowNs();
    while (nowNs() - n0 < 20_000_000) {}
    return @as(f64, @floatFromInt(tsc() - t0)) * 1000.0 / @as(f64, @floatFromInt(nowNs() - n0));
}

fn allocLines(n: usize) []Line {
    const bytes = n * LINE;
    const raw = std.heap.page_allocator.alignedAlloc(u8, .fromByteUnits(HUGE), bytes) catch @panic("oom");
    const p: [*]Line = @ptrCast(@alignCast(raw.ptr));
    return p[0..n];
}

fn pct(xsv: []u32, p: usize) u32 {
    std.mem.sort(u32, xsv, {}, std.sort.asc(u32));
    return xsv[(xsv.len - 1) * p / 100];
}

fn parseCpus(gpa: std.mem.Allocator, s: []const u8) ![]usize {
    var out = std.array_list.Managed(usize).init(gpa);
    var it = std.mem.splitScalar(u8, s, ',');
    while (it.next()) |p| try out.append(try std.fmt.parseInt(usize, p, 10));
    return out.toOwnedSlice();
}

fn readInt(path: [*:0]const u8) i64 {
    const rc = linux.open(path, .{ .ACCMODE = .RDONLY }, 0);
    if (@as(isize, @bitCast(rc)) < 0) return -1;
    const fd: i32 = @intCast(rc);
    defer _ = linux.close(fd);
    var b: [64]u8 = undefined;
    const n = linux.read(fd, &b, b.len);
    if (@as(isize, @bitCast(n)) <= 0) return -1;
    const t = std.mem.trim(u8, b[0..n], " \n");
    return std.fmt.parseInt(i64, t, 10) catch -1;
}

fn dupZ(gpa: std.mem.Allocator, x: []const u8) ![:0]u8 {
    const out = try gpa.allocSentinel(u8, x.len, 0);
    @memcpy(out, x);
    return out;
}

fn writeAll(fd: i32, data: []const u8) void {
    var off: usize = 0;
    while (off < data.len) {
        const n = linux.write(fd, data[off..].ptr, data.len - off);
        if (@as(isize, @bitCast(n)) <= 0) @panic("write");
        off += n;
    }
}

// ---- donanım sayaçları: DF, L3, paket/çekirdek enerjisi, resctrl ----

const Ev = struct { name: []const u8, typ: u32, cfg: u64, cpu: i32 };

const Pmu = struct {
    const MAX = 16;
    evs: [MAX]Ev = undefined,
    fds: [MAX]i32 = undefined,
    n: usize = 0,
    msr: [6]i32 = @splat(-1),

    fn add(self: *Pmu, name: []const u8, typ: i64, cfg: u64, cpu: i32) void {
        if (typ < 0) return;
        var attr = std.mem.zeroes(linux.perf_event_attr);
        attr.type = @enumFromInt(@as(u32, @intCast(typ)));
        attr.config = cfg;
        attr.size = @sizeOf(linux.perf_event_attr);
        const rc = linux.perf_event_open(&attr, -1, cpu, -1, 0);
        if (@as(isize, @bitCast(rc)) < 0) return;
        self.evs[self.n] = .{ .name = name, .typ = @intCast(typ), .cfg = cfg, .cpu = cpu };
        self.fds[self.n] = @intCast(rc);
        self.n += 1;
    }

    fn open() Pmu {
        var p = Pmu{};
        const df = readInt("/sys/bus/event_source/devices/amd_df/type");
        const l3 = readInt("/sys/bus/event_source/devices/amd_l3/type");
        const pw = readInt("/sys/bus/event_source/devices/power/type");
        // DRAM kanal sayaçları: umask 0x38 ve 0x07, `stream` ile kalibre edildi (rapor 3).
        p.add("df_c0_38", df, 0x3807, 0);
        p.add("df_c1_38", df, 0x3847, 0);
        p.add("df_c0_07", df, 0x0707, 0);
        p.add("df_c1_07", df, 0x0747, 0);
        p.add("l3req_x0", l3, 0xff04, 0);
        p.add("l3miss_x0", l3, 0x0106, 0);
        p.add("sdp_x0", l3, 0x3f9a, 0);
        p.add("lat16_x0", l3, 0x0090, 0);
        p.add("l3req_x1", l3, 0xff04, 3);
        p.add("l3miss_x1", l3, 0x0106, 3);
        p.add("sdp_x1", l3, 0x3f9a, 3);
        p.add("lat16_x1", l3, 0x0090, 3);
        p.add("epkg_raw", pw, 2, 0);
        const paths = [6][*:0]const u8{ "/dev/cpu/0/msr", "/dev/cpu/1/msr", "/dev/cpu/2/msr", "/dev/cpu/3/msr", "/dev/cpu/4/msr", "/dev/cpu/5/msr" };
        for (paths, 0..) |path, i| {
            const rc = linux.open(path, .{ .ACCMODE = .RDONLY }, 0);
            if (@as(isize, @bitCast(rc)) >= 0) p.msr[i] = @intCast(rc);
        }
        return p;
    }

    const Snap = struct { v: [MAX]u64 = @splat(0), core: [6]u64 = @splat(0), occ: [2]i64 = @splat(-1), mbm: [2]i64 = @splat(-1) };

    fn snap(self: *const Pmu) Snap {
        var s = Snap{};
        for (0..self.n) |i| _ = linux.read(self.fds[i], @ptrCast(&s.v[i]), 8);
        for (self.msr, 0..) |fd, i| {
            if (fd >= 0) _ = linux.pread(fd, @ptrCast(&s.core[i]), 8, 0xC001029A);
        }
        s.occ[0] = readInt("/sys/fs/resctrl/mon_groups/saturn/mon_data/mon_L3_00/llc_occupancy");
        s.occ[1] = readInt("/sys/fs/resctrl/mon_groups/saturn/mon_data/mon_L3_01/llc_occupancy");
        s.mbm[0] = readInt("/sys/fs/resctrl/mon_groups/saturn/mon_data/mon_L3_00/mbm_total_bytes");
        s.mbm[1] = readInt("/sys/fs/resctrl/mon_groups/saturn/mon_data/mon_L3_01/mbm_total_bytes");
        return s;
    }

    fn print(self: *const Pmu, tag: []const u8, a: Snap, b: Snap) void {
        std.debug.print("\"{s}\":{{", .{tag});
        for (0..self.n) |i| std.debug.print("\"{s}\":{d},", .{ self.evs[i].name, b.v[i] -% a.v[i] });
        std.debug.print("\"ecore\":[", .{});
        for (0..6) |i| std.debug.print("{d}{s}", .{ @as(u32, @truncate(b.core[i])) -% @as(u32, @truncate(a.core[i])), if (i < 5) "," else "" });
        std.debug.print("],\"occ\":[{d},{d}],\"mbm\":[{d},{d}]}}", .{ b.occ[0], b.occ[1], b.mbm[0] - a.mbm[0], b.mbm[1] - a.mbm[1] });
    }
};

// ---- c2c: tek satır gecikmesi ----

fn c2cEcho(cell: *u64, cpu: usize, n: usize) void {
    pinTo(cpu);
    for (0..n) |i| {
        const odd: u64 = i * 2 + 1;
        while (@atomicLoad(u64, cell, .acquire) != odd) {}
        @atomicStore(u64, cell, odd + 1, .release);
    }
}

fn cmdC2c(a: usize, b: usize, n: usize) !void {
    const mhz = tscMhz();
    const l = allocLines(1);
    @memset(std.mem.sliceAsBytes(l), 0);
    const cell = &l[0].w[0];
    const lat = try std.heap.page_allocator.alloc(u32, n);
    const th = try std.Thread.spawn(.{}, c2cEcho, .{ cell, b, n });
    pinTo(a);
    for (0..n) |i| {
        const t0 = tsc();
        @atomicStore(u64, cell, i * 2 + 1, .release);
        while (@atomicLoad(u64, cell, .acquire) != i * 2 + 2) {}
        lat[i] = @intCast(@min(tsc() - t0, std.math.maxInt(u32)));
    }
    th.join();
    const p1 = pct(lat, 1);
    const p50 = pct(lat, 50);
    const p99 = pct(lat, 99);
    // Gidiş-dönüş / 2; tsc() kendi başına yaklaşık sabit bir ek getirir, ham değer korunur.
    std.debug.print("{{\"cmd\":\"c2c\",\"a\":{d},\"b\":{d},\"n\":{d},\"tsc_mhz\":{d:.1},\"rt_p1\":{d},\"rt_p50\":{d},\"rt_p99\":{d},\"oneway_ns_p50\":{d:.2}}}\n", .{
        a, b, n, mhz, p1, p50, p99, @as(f64, @floatFromInt(p50)) / 2.0 * 1000.0 / mhz,
    });
}

// ---- dalga: komutanın tek yazması, izleyicilere varış ----

/// Emir tek kelimedir: (tsc düşük 48 bit << 16) | seq. İki ayrı depo iki geçersizleme üretirdi.
fn dalgaWatch(word: *u64, cpu: usize, n: usize, out: []i64, ready: *u64) void {
    pinTo(cpu);
    @memset(out, -1);
    _ = @atomicRmw(u64, ready, .Add, 1, .acq_rel);
    var last: u64 = 0;
    while (last & 0xFFFF != n) {
        const w = @atomicLoad(u64, word, .acquire);
        if (w != last) {
            const t2 = tsc();
            last = w;
            const d = (t2 & 0xFFFF_FFFF_FFFF) -% (w >> 16);
            out[(w & 0xFFFF) - 1] = @intCast(d & 0xFFFF_FFFF_FFFF);
        }
    }
}

fn cmdDalga(gpa: std.mem.Allocator, cmd: usize, ws: []const usize, n: usize) !void {
    if (n > 65535) return error.TooMany;
    const mhz = tscMhz();
    const l = allocLines(2);
    @memset(std.mem.sliceAsBytes(l), 0);
    const word = &l[0].w[0];
    var ready: u64 = 0;
    const outs = try gpa.alloc([]i64, ws.len);
    const ths = try gpa.alloc(std.Thread, ws.len);
    for (ws, 0..) |c, i| {
        outs[i] = try gpa.alloc(i64, n);
        ths[i] = try std.Thread.spawn(.{}, dalgaWatch, .{ word, c, n, outs[i], &ready });
    }
    pinTo(cmd);
    while (@atomicLoad(u64, &ready, .acquire) != ws.len) {}
    var rng: u64 = 0x1234567 | 1;
    for (1..n + 1) |s| {
        const wait = 180_000 + range(xs(&rng), 360_000); // 50–150 µs
        const t = tsc();
        while (tsc() - t < wait) {}
        @atomicStore(u64, word, ((tsc() & 0xFFFF_FFFF_FFFF) << 16) | s, .release);
    }
    for (ths) |t| t.join();
    std.debug.print("{{\"cmd\":\"dalga\",\"komutan\":{d},\"n\":{d},\"tsc_mhz\":{d:.1},\"w\":[", .{ cmd, n, mhz });
    for (ws, 0..) |c, i| {
        var kept: usize = 0;
        const tmp = try gpa.alloc(u32, n);
        for (outs[i]) |d| {
            if (d < 0) continue;
            tmp[kept] = @intCast(@min(d, std.math.maxInt(u32)));
            kept += 1;
        }
        std.debug.print("{{\"cpu\":{d},\"seen\":{d},\"p1\":{d},\"p50\":{d},\"p99\":{d}}}{s}", .{ c, kept, pct(tmp[0..kept], 1), pct(tmp[0..kept], 50), pct(tmp[0..kept], 99), if (i + 1 < ws.len) "," else "" });
    }
    std.debug.print("],\"ilk\":[", .{});
    for (ws, 0..) |_, i| {
        var first: u64 = 0;
        for (0..n) |k| {
            if (outs[i][k] < 0) continue;
            var best = true;
            for (ws, 0..) |_, j| {
                if (j != i and outs[j][k] >= 0 and outs[j][k] < outs[i][k]) best = false;
            }
            if (best) first += 1;
        }
        std.debug.print("{d}{s}", .{ first, if (i + 1 < ws.len) "," else "" });
    }
    std.debug.print("]}}\n", .{});
}

// ---- satır: tek transferin maliyeti; satırın nerede olduğunu protokol belirler ----

const SatirMod = enum { l1, dirty, clean, dram };

const SatirSync = extern struct { req: u64 align(64), ack: u64 align(64) };

fn satirOwner(sy: *SatirSync, data: []Line, cpu: usize, n: usize, mod: SatirMod) void {
    pinTo(cpu);
    var rng: u64 = 0xABCDEF1 | 1;
    var sink: u64 = 0;
    for (0..n) |i| {
        while (@atomicLoad(u64, &sy.req, .acquire) != i + 1) {}
        const line = &data[range(xs(&rng), data.len)];
        switch (mod) {
            .dirty => @as(*volatile u64, @ptrCast(&line.w[0])).* = i,
            .clean => sink +%= @as(*volatile u64, @ptrCast(&line.w[0])).*,
            else => {},
        }
        @atomicStore(u64, &sy.ack, i + 1, .release);
    }
    std.mem.doNotOptimizeAway(sink);
}

fn cmdSatir(a: usize, b: usize, n: usize, mod: SatirMod) !void {
    const pa = std.heap.page_allocator;
    const ls = allocLines(2);
    @memset(std.mem.sliceAsBytes(ls), 0);
    const sy: *SatirSync = @ptrCast(@alignCast(&ls[0]));
    // Sahip tarafın rastgele sırası ile aynı tohum: ölçüm aynı satıra bakar.
    const data = allocLines(1 << 14);
    @memset(std.mem.sliceAsBytes(data), 0);
    const lat = try pa.alloc(u32, n);
    const th = try std.Thread.spawn(.{}, satirOwner, .{ sy, data, b, n, mod });
    pinTo(a);
    var rng: u64 = 0xABCDEF1 | 1;
    for (0..n) |i| {
        const line = &data[range(xs(&rng), data.len)];
        // Ölçüm çekirdeğinin L2/L3 kopyasını süpür: uzak kopya tek kaynak olsun.
        if (mod == .clean or mod == .dram) {
            clflushopt(line);
            asm volatile ("mfence");
        }
        @atomicStore(u64, &sy.req, i + 1, .release);
        while (@atomicLoad(u64, &sy.ack, .acquire) != i + 1) {}
        if (mod == .l1) _ = @as(*volatile u64, @ptrCast(&line.w[0])).*;
        const t0 = tsc();
        _ = @as(*volatile u64, @ptrCast(&line.w[0])).*;
        const t1 = tsc();
        lat[i] = @intCast(t1 - t0);
    }
    th.join();
    std.debug.print("{{\"cmd\":\"satir\",\"a\":{d},\"b\":{d},\"mod\":\"{s}\",\"n\":{d},\"p1\":{d},\"p50\":{d},\"p90\":{d},\"p99\":{d}}}\n", .{
        a, b, @tagName(mod), n, pct(lat, 1), pct(lat, 50), pct(lat, 90), pct(lat, 99),
    });
}

// ---- akış: DF sayaç kalibrasyonu ----

fn streamWorker(buf: []u8, cpu: usize, ns: u64, write: bool, out: *u64) void {
    pinTo(cpu);
    const end = nowNs() + ns;
    var total: u64 = 0;
    var sink: u64 = 0;
    while (nowNs() < end) {
        var i: usize = 0;
        if (write) {
            while (i < buf.len) : (i += LINE) @as(*volatile u64, @ptrCast(@alignCast(&buf[i]))).* = i;
        } else {
            while (i < buf.len) : (i += LINE) sink +%= @as(*volatile u64, @ptrCast(@alignCast(&buf[i]))).*;
        }
        total += buf.len;
    }
    std.mem.doNotOptimizeAway(sink);
    out.* = total;
}

fn cmdAkis(gpa: std.mem.Allocator, cpus: []const usize, kb: usize, sec: f64, write: bool) !void {
    const lines = kb * 1024 / LINE;
    const l = allocLines(lines * cpus.len);
    @memset(std.mem.sliceAsBytes(l), 1);
    const pmu = Pmu.open();
    const ths = try gpa.alloc(std.Thread, cpus.len);
    const outs = try gpa.alloc(u64, cpus.len);
    const a = pmu.snap();
    const t0 = nowNs();
    for (cpus, 0..) |c, i| {
        const seg = std.mem.sliceAsBytes(l[i * lines .. (i + 1) * lines]);
        ths[i] = try std.Thread.spawn(.{}, streamWorker, .{ seg, c, @as(u64, @intFromFloat(sec * 1e9)), write, &outs[i] });
    }
    for (ths) |t| t.join();
    const dt = nowNs() - t0;
    const b = pmu.snap();
    var tot: u64 = 0;
    for (outs) |o| tot += o;
    std.debug.print("{{\"cmd\":\"akis\",\"write\":{},\"kb\":{d},\"bytes\":{d},\"sec\":{d:.3},\"gbps\":{d:.2},", .{ write, kb, tot, @as(f64, @floatFromInt(dt)) * 1e-9, @as(f64, @floatFromInt(tot)) / @as(f64, @floatFromInt(dt)) });
    pmu.print("pmu", a, b);
    std.debug.print("}}\n", .{});
}

// ---- merdiven: bit yolları ----

fn chaseBuild(lines: usize, seed: u64) []Line {
    const l = allocLines(lines);
    const perm = std.heap.page_allocator.alloc(u32, lines) catch @panic("oom");
    defer std.heap.page_allocator.free(perm);
    for (perm, 0..) |*p, i| p.* = @intCast(i);
    var s = seed;
    var i: usize = lines - 1;
    while (i > 0) : (i -= 1) {
        const j = range(splitmix(&s), i);
        std.mem.swap(u32, &perm[i], &perm[j]);
    }
    for (0..lines) |k| l[perm[k]].w[0] = @intFromPtr(&l[perm[(k + 1) % lines]]);
    return l;
}

fn chaseLoop(start: u64, n: u64) u64 {
    var p = start;
    var c = n;
    asm volatile (
        \\1:
        \\mov (%[p]), %[p]
        \\dec %[c]
        \\jnz 1b
        : [p] "+r" (p),
          [c] "+r" (c),
    );
    return p;
}

const BEST_REPS = 7;

fn ladderOne(name: []const u8, ops: u64, f: anytype, args: anytype) void {
    pinTo(1);
    var best_c: u64 = std.math.maxInt(u64);
    var best_t: u64 = 0;
    for (0..BEST_REPS) |_| {
        const a0 = rdpru(1);
        const t0 = tsc();
        _ = @call(.auto, f, args);
        const c = rdpru(1) - a0;
        const t = tsc() - t0;
        if (c < best_c) {
            best_c = c;
            best_t = t;
        }
    }
    std.debug.print("{{\"cmd\":\"merdiven\",\"ad\":\"{s}\",\"ops\":{d},\"cycle_op\":{d:.3},\"tick_op\":{d:.3}}}\n", .{
        name, ops, @as(f64, @floatFromInt(best_c)) / @as(f64, @floatFromInt(ops)), @as(f64, @floatFromInt(best_t)) / @as(f64, @floatFromInt(ops)),
    });
}

fn opReg(n: u64) u64 {
    var a: u64 = 0;
    var c = n;
    asm volatile (
        \\1:
        \\add $1, %[a]
        \\dec %[c]
        \\jnz 1b
        : [a] "+r" (a),
          [c] "+r" (c),
    );
    return a;
}

/// rsp tabanlı depo→yük: Zen 2 bellek yeniden adlandırması adayı (kırmızı bölge).
fn opRename(n: u64) u64 {
    var a: u64 = 0;
    var c = n;
    asm volatile (
        \\1:
        \\mov %[a], -8(%%rsp)
        \\mov -8(%%rsp), %[a]
        \\add $1, %[a]
        \\dec %[c]
        \\jnz 1b
        : [a] "+r" (a),
          [c] "+r" (c),
    );
    return a;
}

/// Dizinli adres: yeniden adlandırma dışı, klasik store-to-load forwarding.
fn opStlf(n: u64, buf: *Line) u64 {
    var a: u64 = 0;
    var c = n;
    const i: u64 = 1;
    asm volatile (
        \\1:
        \\mov %[a], (%[b],%[i],8)
        \\mov (%[b],%[i],8), %[a]
        \\add $1, %[a]
        \\dec %[c]
        \\jnz 1b
        : [a] "+r" (a),
          [c] "+r" (c),
        : [b] "r" (buf),
          [i] "r" (i),
    );
    return a;
}

/// Dar depo + geniş yük: iletim başarısız olur (SOG 2.12), L1'den beklenir.
fn opStlfFail(n: u64, buf: *Line) u64 {
    var a: u64 = 0;
    var c = n;
    asm volatile (
        \\1:
        \\mov %[a:k], (%[b])
        \\mov (%[b]), %[a]
        \\add $1, %[a]
        \\dec %[c]
        \\jnz 1b
        : [a] "+r" (a),
          [c] "+r" (c),
        : [b] "r" (buf),
    );
    return a;
}

/// 256 bit depo→yük döngüsü.
fn opYmm(n: u64, buf: *Line) u64 {
    var c = n;
    asm volatile (
        \\vpxor %%ymm0, %%ymm0, %%ymm0
        \\1:
        \\vmovdqa %%ymm0, (%[b])
        \\vmovdqa (%[b]), %%ymm0
        \\vpaddq %%ymm0, %%ymm0, %%ymm0
        \\dec %[c]
        \\jnz 1b
        \\vzeroupper
        : [c] "+r" (c),
        : [b] "r" (buf),
    );
    return c;
}

fn cmdMerdiven(name: []const u8) void {
    const big: u64 = 200_000_000;
    const buf = allocLines(4);
    @memset(std.mem.sliceAsBytes(buf), 0);
    if (std.mem.eql(u8, name, "reg")) return ladderOne(name, big, opReg, .{big});
    if (std.mem.eql(u8, name, "rename")) return ladderOne(name, big, opRename, .{big});
    if (std.mem.eql(u8, name, "stlf")) return ladderOne(name, big, opStlf, .{ big, &buf[0] });
    if (std.mem.eql(u8, name, "stlf_fail")) return ladderOne(name, big / 4, opStlfFail, .{ big / 4, &buf[0] });
    if (std.mem.eql(u8, name, "ymm")) return ladderOne(name, big, opYmm, .{ big, &buf[0] });
    const sizes = [_]struct { n: []const u8, kb: usize, ops: u64 }{
        .{ .n = "l1", .kb = 16, .ops = big },
        .{ .n = "l2", .kb = 256, .ops = big / 2 },
        .{ .n = "l3", .kb = 8192, .ops = big / 8 },
        .{ .n = "dram", .kb = 512 * 1024, .ops = big / 40 },
    };
    for (sizes) |s| {
        if (std.mem.eql(u8, name, s.n)) {
            const l = chaseBuild(s.kb * 1024 / LINE, KEY);
            pinTo(1);
            _ = chaseLoop(@intFromPtr(&l[0]), s.kb * 1024 / LINE * 2);
            return ladderOne(name, s.ops, chaseLoop, .{ @as(u64, @intFromPtr(&l[0])), s.ops });
        }
    }
    @panic("merdiven adı");
}

// ---- pencere: dolgu sayısına göre iki kaçırma örtüşüyor mu (Wong) ----

const Filler = enum { nop, int, fp, load, store };

fn fillText(comptime f: Filler) []const u8 {
    return switch (f) {
        .nop => "nop\n",
        .int => "add $1, %%r8\n",
        .fp => "vaddps %%ymm1, %%ymm1, %%ymm0\n",
        .load => "mov (%[b]), %%r9\n",
        .store => "mov %%r9, 64(%[b])\n",
    };
}

fn windowLoop(comptime f: Filler, comptime N: usize, a0: u64, b0: u64, n: u64, buf: *Line) u64 {
    const rep = std.fmt.comptimePrint(".rept {d}\n", .{N}) ++ comptime fillText(f) ++ ".endr\n";
    var a = a0;
    var b = b0;
    var c = n;
    asm volatile ("1:\nmov (%[p]), %[p]\n" ++ rep ++ "mov (%[q]), %[q]\n" ++ rep ++ "dec %[c]\njnz 1b\n"
        : [p] "+r" (a),
          [q] "+r" (b),
          [c] "+r" (c),
        : [b] "r" (buf),
        : .{ .r8 = true, .r9 = true, .ymm0 = true });
    return a ^ b;
}

const WIN_NS = blk: {
    var out: [40]usize = undefined;
    for (&out, 0..) |*v, i| v.* = 8 * (i + 1);
    break :blk out;
};

fn cmdPencere(fname: []const u8, n_req: usize) void {
    const lines: usize = 256 * 1024 * 1024 / LINE;
    const l = chaseBuild(lines, KEY);
    const buf = allocLines(4);
    @memset(std.mem.sliceAsBytes(buf), 0);
    pinTo(1);
    const iters: u64 = 400_000;
    const a0: u64 = @intFromPtr(&l[0]);
    const b0: u64 = @intFromPtr(&l[lines / 2]);
    _ = chaseLoop(a0, 2_000_000);
    const f = std.meta.stringToEnum(Filler, fname) orelse @panic("dolgu");
    inline for (WIN_NS) |N| {
        if (N == n_req) {
            var best: u64 = std.math.maxInt(u64);
            for (0..3) |_| {
                const t0 = rdpru(1);
                switch (f) {
                    inline else => |ff| _ = windowLoop(ff, N, a0, b0, iters, &buf[0]),
                }
                best = @min(best, rdpru(1) - t0);
            }
            std.debug.print("{{\"cmd\":\"pencere\",\"dolgu\":\"{s}\",\"N\":{d},\"cycle_iter\":{d:.2}}}\n", .{ fname, N, @as(f64, @floatFromInt(best)) / @as(f64, @floatFromInt(iters)) });
            return;
        }
    }
    @panic("N");
}

// ---- ortak: sahipsiz satır dönüşümü ----
// Bir ziyaret: kelime × g_c (mod 2^64), CAS ile. g_c tek olduğundan dönüşüm bijektif ve değişmeli:
// sıra ne olursa olsun, her çekirdeğin ziyaret dizisi tohumundan yeniden üretilip tersi uygulanarak
// özgün yük bit bit geri kurulur. Sıcak yolda kuyruk, kilit, sahip tablosu veya sağlama yoktur.

inline fn visit(p: *u64, g: u64, fails: *u64) void {
    var old = @atomicLoad(u64, p, .monotonic);
    while (@cmpxchgWeak(u64, p, old, old *% g, .acq_rel, .monotonic)) |seen| {
        old = seen;
        fails.* += 1;
    }
}

fn modInv(g: u64) u64 {
    var x = g;
    for (0..6) |_| x = x *% (2 -% g *% x);
    return x;
}

fn payloadInit(lines: []Line, seed: u64) void {
    var s = seed ^ 0x5A7;
    for (lines) |*l| {
        for (&l.w) |*w| w.* = splitmix(&s);
    }
}

fn payloadCheck(lines: []const Line, seed: u64) u64 {
    var s = seed ^ 0x5A7;
    var bad: u64 = 0;
    for (lines) |*l| {
        for (l.w) |w| {
            if (w != splitmix(&s)) bad += 1;
        }
    }
    return bad;
}

fn coreG(seed: u64, cpu: usize) u64 {
    var s = seed ^ (@as(u64, cpu) *% 0xD1342543DE82EF95);
    return splitmix(&s) | 1;
}

fn coreRng(seed: u64, cpu: usize) u64 {
    var s = seed +% (@as(u64, cpu) << 40) +% 0xA5A5;
    return splitmix(&s) | 1;
}

inline fn latBin(d: u64) usize {
    return @min(15, 64 - @clz(d | 1));
}

fn sleepSec(sec: f64) void {
    const ns: u64 = @intFromFloat(sec * 1e9);
    var ts = linux.timespec{ .sec = @intCast(ns / 1_000_000_000), .nsec = @intCast(ns % 1_000_000_000) };
    _ = linux.nanosleep(&ts, null);
}

// ---- halka: sahipsiz dolaşım ve tam geri kurma ----

const HalkaKol = enum { sahipli, serbest, park };

const HalkaCtx = struct {
    lines: []Line,
    lo: usize,
    n: usize,
    cpu: usize,
    seed: u64,
    go: *u64,
    stop: *u64,
    ops: u64 = 0,
    fails: u64 = 0,
    hist: [16]u64 = @splat(0),
};

fn halkaWorker(c: *HalkaCtx) void {
    pinTo(c.cpu);
    const g = coreG(c.seed, c.cpu);
    var s = coreRng(c.seed, c.cpu);
    while (@atomicLoad(u64, c.go, .acquire) == 0) {}
    var n: u64 = 0;
    var fails: u64 = 0;
    var hist: [16]u64 = @splat(0);
    while (true) : (n += 1) {
        if (n & 255 == 0 and @atomicLoad(u64, c.stop, .monotonic) != 0) break;
        const r = xs(&s);
        const p = &c.lines[c.lo + range(r, c.n)].w[r & 7];
        if (n & 63 == 0) {
            const t0 = tsc();
            visit(p, g, &fails);
            hist[latBin(tsc() - t0)] += 1;
        } else {
            visit(p, g, &fails);
        }
    }
    c.ops = n;
    c.fails = fails;
    c.hist = hist;
}

fn halkaReverse(c: *const HalkaCtx, writers: []u8, bit: u8) void {
    const gi = modInv(coreG(c.seed, c.cpu));
    var s = coreRng(c.seed, c.cpu);
    for (0..c.ops) |_| {
        const r = xs(&s);
        const i = c.lo + range(r, c.n);
        const p = &c.lines[i].w[r & 7];
        p.* = p.* *% gi;
        writers[i] |= bit;
    }
}

fn cmdHalka(gpa: std.mem.Allocator, kol: HalkaKol, cpus: []const usize, m: usize, sec: f64, seed: u64) !void {
    const lines = allocLines(m);
    payloadInit(lines, seed);
    const ctl = allocLines(2);
    @memset(std.mem.sliceAsBytes(ctl), 0);
    const go = &ctl[0].w[0];
    const stop = &ctl[1].w[0];
    const cs = try gpa.alloc(HalkaCtx, cpus.len);
    const ths = try gpa.alloc(std.Thread, cpus.len);
    const part = m / @max(cpus.len, 1);
    pinTo(0);
    const pmu = Pmu.open();
    if (kol != .park) {
        for (cpus, 0..) |cpu, i| {
            cs[i] = .{
                .lines = lines,
                .lo = if (kol == .sahipli) i * part else 0,
                .n = if (kol == .sahipli) part else m,
                .cpu = cpu,
                .seed = seed,
                .go = go,
                .stop = stop,
            };
            ths[i] = try std.Thread.spawn(.{}, halkaWorker, .{&cs[i]});
        }
        @atomicStore(u64, go, 1, .release);
    }
    sleepSec(0.3);
    const a = pmu.snap();
    const t0 = nowNs();
    sleepSec(sec);
    const b = pmu.snap();
    const dt = nowNs() - t0;
    @atomicStore(u64, stop, 1, .release);
    var ops: u64 = 0;
    var fails: u64 = 0;
    const writers = try gpa.alloc(u8, m);
    @memset(writers, 0);
    if (kol != .park) {
        for (ths) |t| t.join();
        for (cs, 0..) |*c, i| {
            ops += c.ops;
            fails += c.fails;
            halkaReverse(c, writers, @as(u8, 1) << @as(u3, @intCast(i)));
        }
    }
    const bad = payloadCheck(lines, seed);
    var wsum: u64 = 0;
    for (writers) |w| wsum += @popCount(w);
    std.debug.print("{{\"cmd\":\"halka\",\"kol\":\"{s}\",\"m\":{d},\"sec\":{d:.3},\"ops\":{d},\"fails\":{d},\"yanlis_kelime\":{d},\"kelime\":{d},\"yazar_ort\":{d:.3},", .{ @tagName(kol), m, @as(f64, @floatFromInt(dt)) * 1e-9, ops, fails, bad, m * 8, @as(f64, @floatFromInt(wsum)) / @as(f64, @floatFromInt(m)) });
    std.debug.print("\"cekirdek\":[", .{});
    if (kol != .park) {
        for (cs, 0..) |*c, i| {
            std.debug.print("{{\"cpu\":{d},\"ops\":{d},\"fails\":{d},\"hist\":[", .{ c.cpu, c.ops, c.fails });
            for (c.hist, 0..) |h, k| std.debug.print("{d}{s}", .{ h, if (k < 15) "," else "" });
            std.debug.print("]}}{s}", .{if (i + 1 < cs.len) "," else ""});
        }
    }
    std.debug.print("],", .{});
    pmu.print("pmu", a, b);
    std.debug.print("}}\n", .{});
}

// ---- fırtına: göz, duvar, mor bölge; bulut; IF→cIOD düşüşü ----

// bulut_j: ÖN KAYIT SONRASI keşif. Bulut + sektör içi rastgele merkez (16 sabit nokta kısıtını kaldırır).
// rol, rol_k: ÖN KAYIT EKİ. Çizimin coğrafyası: ordu dağınık, mor merkez çevresinde; emir üç ayrı satırda.
const FirtinaKol = enum { sakin, sabit, gezen, bulut, bulut_j, rol, rol_k, rol_ayna };
const SECTORS = 16;
const CMD_LOG = 1 << 16;

const FirtinaShared = struct {
    lines: []Line,
    m: usize,
    re: usize,
    table: []const i32,
    seed: u64,
    kol: FirtinaKol,
    go: *u64,
    stop: *u64,
    center: *u64,
    orders: [3]*u64,
    status: []Line, // çekirdek başına bir satır: sektör başına ziyaret (bulutun topladığı takım taklavat)
};

const FirtinaCtx = struct {
    sh: *const FirtinaShared,
    idx: usize,
    cpu: usize,
    iters: u64 = 0,
    ops: u64 = 0,
    fails: u64 = 0,
    log_n: usize = 0,
    log_it: []u64,
    log_c: []u64,
    hist: [3][16]u64 = @splat(@splat(0)),
};

fn rolde(kol: FirtinaKol) bool {
    return kol == .rol or kol == .rol_k or kol == .rol_ayna;
}

/// Ordunun emir satırı; orduda değilse null (mor bölge).
fn orduSira(kol: FirtinaKol, cpu: usize) ?usize {
    if (!rolde(kol)) return null;
    if (kol == .rol_ayna) return if (cpu == 4) 0 else if (cpu == 5) 1 else null;
    if (cpu == 1) return 0;
    if (cpu == (if (kol == .rol) @as(usize, 2) else 3)) return 1;
    return null;
}

/// Ziyaret ofseti: ordu halkaya tekdüze dağılır ama göze yazmaz; diğerleri rüzgâr profilini izler.
inline fn visitOff(sh: *const FirtinaShared, ordu: bool, r: u64) i64 {
    if (!ordu) return sh.table[(r >> 8) & 4095];
    const o = @as(i64, @intCast(range(r, sh.m))) - @as(i64, @intCast(sh.m / 2));
    return if (@abs(o) < sh.re) @intCast(sh.re) else o;
}

inline fn ringIdx(c: u64, off: i64, m: usize) usize {
    const sm: i64 = @intCast(m);
    return @intCast(@mod(@as(i64, @intCast(c)) + off, sm));
}

fn firtinaWorker(c: *FirtinaCtx) void {
    const sh = c.sh;
    pinTo(c.cpu);
    const g = coreG(sh.seed, c.cpu);
    var s = coreRng(sh.seed, c.cpu);
    const sira = orduSira(sh.kol, c.cpu);
    const ordu = sira != null;
    const emir: *u64 = if (rolde(sh.kol)) sh.orders[sira orelse 2] else sh.center;
    var cen: u64 = @atomicLoad(u64, emir, .acquire);
    while (@atomicLoad(u64, sh.go, .acquire) == 0) {}
    var it: u64 = 0;
    var ops: u64 = 0;
    var fails: u64 = 0;
    var hist: [3][16]u64 = @splat(@splat(0));
    var sect: [SECTORS]u32 = @splat(0);
    const st: *[16]u32 = @ptrCast(@alignCast(&sh.status[c.idx]));
    const eye_w: i64 = @intCast(2 * sh.re);
    while (true) : (it += 1) {
        if (it & 1023 == 0) {
            if (@atomicLoad(u64, sh.stop, .monotonic) != 0) break;
            const nc = @atomicLoad(u64, emir, .monotonic);
            if (nc != cen and c.log_n < CMD_LOG) {
                cen = nc;
                c.log_it[c.log_n] = it;
                c.log_c[c.log_n] = nc;
                c.log_n += 1;
            }
            if (it & 4095 == 0 and sh.kol != .sakin) {
                for (sect, 0..) |v, i| @atomicStore(u32, &st[i], v, .release);
            }
        }
        const r = xs(&s);
        if (r & 15 == 0) {
            // Göz: yalnız okunur. Fırtına sürerken dingin kalıyor mu?
            const off = @as(i64, @intCast(range(r, @intCast(eye_w)))) - @as(i64, @intCast(sh.re));
            const p: *volatile u64 = @ptrCast(&sh.lines[ringIdx(cen, off, sh.m)].w[(r >> 4) & 7]);
            const t0 = tsc();
            _ = p.*;
            hist[0][latBin(tsc() - t0)] += 1;
            continue;
        }
        if (sh.kol == .sakin) continue;
        const off = visitOff(sh, ordu, r);
        const idx = ringIdx(cen, off, sh.m);
        const p = &sh.lines[idx].w[(r >> 4) & 7];
        if (it & 31 == 1) {
            const zone: usize = if (@abs(off) < 8 * sh.re) 1 else 2;
            const t0 = tsc();
            visit(p, g, &fails);
            hist[zone][latBin(tsc() - t0)] += 1;
        } else {
            visit(p, g, &fails);
        }
        sect[idx * SECTORS / sh.m] += 1;
        ops += 1;
    }
    c.iters = it;
    c.ops = ops;
    c.fails = fails;
    c.hist = hist;
}

/// Aynı rastgele akış ve merkez günlüğüyle ziyaretleri tersine uygular; satır başına yazma sayısını da verir.
fn firtinaReverse(c: *const FirtinaCtx, cen0: u64, visits: []u32) void {
    const sh = c.sh;
    const gi = modInv(coreG(sh.seed, c.cpu));
    var s = coreRng(sh.seed, c.cpu);
    var cen = cen0;
    var li: usize = 0;
    const ordu = orduSira(sh.kol, c.cpu) != null;
    for (0..c.iters) |it| {
        while (li < c.log_n and c.log_it[li] == it) : (li += 1) cen = c.log_c[li];
        const r = xs(&s);
        if (r & 15 == 0) continue;
        if (sh.kol == .sakin) continue;
        const off = visitOff(sh, ordu, r);
        const idx = ringIdx(cen, off, sh.m);
        const p = &sh.lines[idx].w[(r >> 4) & 7];
        p.* = p.* *% gi;
        visits[idx] += 1;
    }
}

fn flushAll(lines: []Line, cpu: usize, out: *u64) void {
    pinTo(cpu);
    const t0 = tsc();
    for (lines) |*l| clflushopt(l);
    asm volatile ("sfence");
    out.* = tsc() - t0;
}

fn spitAll(lines: []const Line, cpu: usize, out: *u64) void {
    pinTo(cpu);
    var sink: u64 = 0;
    var s: u64 = 0x77 | 1;
    const t0 = tsc();
    for (0..lines.len) |_| sink +%= @as(*const volatile u64, @ptrCast(&lines[range(xs(&s), lines.len)].w[0])).*;
    out.* = tsc() - t0;
    std.mem.doNotOptimizeAway(sink);
}

fn cmdFirtina(gpa: std.mem.Allocator, kol: FirtinaKol, cpus: []const usize, m: usize, sec: f64, seed: u64, dusus_cpu: usize) !void {
    const lines = allocLines(m);
    payloadInit(lines, seed);
    const ctl = allocLines(6);
    @memset(std.mem.sliceAsBytes(ctl), 0);
    const status = allocLines(cpus.len);
    @memset(std.mem.sliceAsBytes(status), 0);

    // Rüzgâr profili: gözde sıfır, duvara yaklaştıkça artar (r=8·re), dışarı doğru ters kare ile söner.
    const re = @max(m / 512, 4);
    const rw = 8 * re;
    const half = m / 2;
    const table = try gpa.alloc(i32, 4096);
    var tot: f64 = 0;
    for (re..half) |d| {
        const x = @as(f64, @floatFromInt(d));
        const r = @as(f64, @floatFromInt(rw));
        tot += if (d < rw) x / r else (r / x) * (r / x);
    }
    var cum: f64 = 0;
    var next: usize = 0;
    var sgn: i32 = 1;
    for (re..half) |d| {
        const x = @as(f64, @floatFromInt(d));
        const r = @as(f64, @floatFromInt(rw));
        cum += if (d < rw) x / r else (r / x) * (r / x);
        while (next < 4096 and (@as(f64, @floatFromInt(next)) + 0.5) / 4096.0 <= cum / tot) : (next += 1) {
            table[next] = sgn * @as(i32, @intCast(d));
            sgn = -sgn;
        }
    }
    while (next < 4096) : (next += 1) table[next] = @intCast(re);

    const cen0: u64 = m / 2;
    const center = &ctl[2].w[0];
    @atomicStore(u64, center, cen0, .release);
    const orders = [3]*u64{ &ctl[3].w[0], &ctl[4].w[0], &ctl[5].w[0] };
    for (orders) |o| @atomicStore(u64, o, cen0, .release);
    const sh: *FirtinaShared = @ptrCast(@alignCast((allocLines(4)).ptr));
    sh.* = .{ .lines = lines, .m = m, .re = re, .table = table, .seed = seed, .kol = kol, .go = &ctl[0].w[0], .stop = &ctl[1].w[0], .center = center, .orders = orders, .status = status };
    const cs = try gpa.alloc(FirtinaCtx, cpus.len);
    const ths = try gpa.alloc(std.Thread, cpus.len);
    for (cpus, 0..) |cpu, i| {
        cs[i] = .{ .sh = sh, .idx = i, .cpu = cpu, .log_it = try gpa.alloc(u64, CMD_LOG), .log_c = try gpa.alloc(u64, CMD_LOG) };
        ths[i] = try std.Thread.spawn(.{}, firtinaWorker, .{&cs[i]});
    }
    pinTo(if (kol == .rol_ayna) 3 else 0);
    const pmu = Pmu.open();
    @atomicStore(u64, sh.go, 1, .release);
    sleepSec(0.2);
    const a = pmu.snap();
    const t0 = nowNs();
    const dur: u64 = @intFromFloat(sec * 1e9);

    // Komutan + bulut: ordu durumunu toplar, kapasite (M ziyaret) dolunca kararır ve yeni hücre kurar.
    var rains: u64 = 0;
    var rng: u64 = seed | 1;
    var seen_total: u64 = 0;
    var cloud: u64 = 0;
    var sec_tot: [SECTORS]u64 = @splat(0);
    while (nowNs() - t0 < dur) {
        if (kol == .bulut or kol == .gezen or kol == .bulut_j or rolde(kol)) {
            var total: u64 = 0;
            for (&sec_tot) |*v| v.* = 0;
            for (0..cpus.len) |i| {
                const st: *const [16]u32 = @ptrCast(@alignCast(&status[i]));
                for (0..SECTORS) |k| {
                    const v = @atomicLoad(u32, &st[k], .acquire);
                    sec_tot[k] += v;
                    total += v;
                }
            }
            cloud += total -% seen_total;
            seen_total = total;
            if (cloud >= m) {
                cloud = 0;
                rains += 1;
                var nc: u64 = range(xs(&rng), m);
                if (kol == .bulut or kol == .bulut_j or rolde(kol)) {
                    var lo: usize = 0;
                    for (sec_tot, 0..) |v, k| {
                        if (v < sec_tot[lo]) lo = k;
                    }
                    nc = if (kol == .bulut) (2 * lo + 1) * m / (2 * SECTORS) else lo * m / SECTORS + range(xs(&rng), m / SECTORS);
                }
                if (rolde(kol)) {
                    for (orders) |o| @atomicStore(u64, o, nc, .release);
                } else {
                    @atomicStore(u64, center, nc, .release);
                }
            }
            const tw = tsc();
            while (tsc() - tw < 36000) {} // 10 µs: atlının turu
        } else {
            sleepSec(0.01);
        }
    }
    const b = pmu.snap();
    const dt = nowNs() - t0;
    @atomicStore(u64, sh.stop, 1, .release);
    for (ths) |t| t.join();

    // Düşüş: fırtına biter, kirli satırlar IF üzerinden cIOD'deki bellek denetleyicisine iner.
    var t_flush: u64 = 0;
    var t_flush2: u64 = 0;
    var t_spit: u64 = 0;
    const s0 = pmu.snap();
    (try std.Thread.spawn(.{}, flushAll, .{ lines, dusus_cpu, &t_flush })).join();
    const s1 = pmu.snap();
    (try std.Thread.spawn(.{}, flushAll, .{ lines, dusus_cpu, &t_flush2 })).join();
    const s2 = pmu.snap();
    (try std.Thread.spawn(.{}, spitAll, .{ lines, dusus_cpu, &t_spit })).join();
    const s3 = pmu.snap();

    const visits = try gpa.alloc(u32, m);
    @memset(visits, 0);
    var ops: u64 = 0;
    var fails: u64 = 0;
    for (cs) |*c| {
        ops += c.ops;
        fails += c.fails;
        firtinaReverse(c, cen0, visits);
    }
    const bad = payloadCheck(lines, seed);
    var vsum: f64 = 0;
    var vsq: f64 = 0;
    var zero: u64 = 0;
    for (visits) |v| {
        const x = @as(f64, @floatFromInt(v));
        vsum += x;
        vsq += x * x;
        if (v == 0) zero += 1;
    }
    const n_f = @as(f64, @floatFromInt(m));
    const mean = vsum / n_f;
    const cv = if (mean > 0) @sqrt(@max(vsq / n_f - mean * mean, 0)) / mean else 0;

    std.debug.print("{{\"cmd\":\"firtina\",\"kol\":\"{s}\",\"m\":{d},\"re\":{d},\"sec\":{d:.3},\"ops\":{d},\"fails\":{d},\"yanlis_kelime\":{d},\"yagmur\":{d},\"cv\":{d:.3},\"dokunulmamis\":{d},\"t_dusus\":{d},\"t_dusus2\":{d},\"t_tukur\":{d},\"dusus_cpu\":{d},\"hist\":[", .{ @tagName(kol), m, re, @as(f64, @floatFromInt(dt)) * 1e-9, ops, fails, bad, rains, cv, zero, t_flush, t_flush2, t_spit, dusus_cpu });
    for (cs, 0..) |*c, i| {
        std.debug.print("{{\"cpu\":{d},\"iters\":{d},\"z\":[", .{ c.cpu, c.iters });
        for (0..3) |z| {
            std.debug.print("[", .{});
            for (c.hist[z], 0..) |h, k| std.debug.print("{d}{s}", .{ h, if (k < 15) "," else "" });
            std.debug.print("]{s}", .{if (z < 2) "," else ""});
        }
        std.debug.print("]}}{s}", .{if (i + 1 < cs.len) "," else ""});
    }
    std.debug.print("],", .{});
    pmu.print("pmu_firtina", a, b);
    std.debug.print(",", .{});
    pmu.print("pmu_dusus", s0, s1);
    std.debug.print(",", .{});
    pmu.print("pmu_dusus2", s1, s2);
    std.debug.print(",", .{});
    pmu.print("pmu_tukur", s2, s3);
    std.debug.print("}}\n", .{});
}

// ---- düşüş kalibrasyonu (ÖN KAYIT SONRASI keşif): F-4'ün paydası kirli satır sayısı olmalıydı ----

fn touchAll(lines: []Line, cpu: usize, write: bool) void {
    pinTo(cpu);
    var sink: u64 = 0;
    for (lines, 0..) |*l, i| {
        if (write) {
            @as(*volatile u64, @ptrCast(&l.w[0])).* = i;
        } else {
            sink +%= @as(*volatile u64, @ptrCast(&l.w[0])).*;
        }
    }
    std.mem.doNotOptimizeAway(sink);
}

fn cmdDusus(m: usize, yaz_cpu: usize, dus_cpu: usize, write: bool, reps: usize) !void {
    const pa = std.heap.page_allocator;
    const lines = allocLines(m);
    @memset(std.mem.sliceAsBytes(lines), 1);
    const pmu = Pmu.open();
    pinTo(0);
    const net = try pa.alloc(i64, reps);
    const tt = try pa.alloc(u32, reps);
    for (0..reps) |r| {
        (try std.Thread.spawn(.{}, touchAll, .{ lines, yaz_cpu, write })).join();
        var t1: u64 = 0;
        var t2: u64 = 0;
        const s0 = pmu.snap();
        (try std.Thread.spawn(.{}, flushAll, .{ lines, dus_cpu, &t1 })).join();
        const s1 = pmu.snap();
        (try std.Thread.spawn(.{}, flushAll, .{ lines, dus_cpu, &t2 })).join();
        const s2 = pmu.snap();
        const d1 = (s1.v[0] -% s0.v[0]) + (s1.v[1] -% s0.v[1]);
        const d2 = (s2.v[0] -% s1.v[0]) + (s2.v[1] -% s1.v[1]);
        net[r] = @as(i64, @intCast(d1)) - @as(i64, @intCast(d2));
        tt[r] = @intCast(@min(t1, std.math.maxInt(u32)));
    }
    std.mem.sort(i64, net, {}, std.sort.asc(i64));
    std.debug.print("{{\"cmd\":\"dusus\",\"m\":{d},\"yaz_cpu\":{d},\"dus_cpu\":{d},\"yaz\":{},\"tekrar\":{d},\"net_p50\":{d},\"net_p10\":{d},\"net_p90\":{d},\"net_satir\":{d:.3},\"t_p50\":{d}}}\n", .{
        m, yaz_cpu, dus_cpu, write, reps, net[reps / 2], net[reps / 10], net[reps * 9 / 10], @as(f64, @floatFromInt(net[reps / 2])) / @as(f64, @floatFromInt(m)), pct(tt, 50),
    });
}

// ---- patlama (ÖN KAYIT EKİ): okuyucu istemeden veri getirilir ----
// Okuyucu zinciri izler, yolu bilmez. Yardımcı yolu bilir ve önündeki k satırı okur. Bu, kullanıcının
// "ters karadelik" sözünün en yakın ölçülebilir karşılığı olarak BENİM formalizasyonumdur.

const PatKol = enum { yok, kardes, ccx, capraz, bozuk, bozuk_ccx, kendi };

const PatShared = struct {
    lines: []Line,
    perm: []const u32,
    k: u64,
    go: *u64,
    stop: *u64,
    pos: *u64,
};

fn patYardimci(sh: *const PatShared, cpu: usize, kay: u64, out: *u64) void {
    pinTo(cpu);
    const m: u64 = sh.perm.len;
    var h: u64 = 0;
    var n: u64 = 0;
    var sink: u64 = 0;
    while (@atomicLoad(u64, sh.stop, .monotonic) == 0) {
        const p = @atomicLoad(u64, sh.pos, .monotonic);
        if (h < p + 2) h = p + 2;
        if (h >= p + sh.k) {
            asm volatile ("pause");
            continue;
        }
        sink +%= @as(*const volatile u64, @ptrCast(&sh.lines[sh.perm[(h + kay) % m]].w[1])).*;
        h += 1;
        n += 1;
    }
    std.mem.doNotOptimizeAway(sink);
    out.* = n;
}

fn patOkuyucu(sh: *const PatShared, kendi: bool, steps: *u64, last: *u64, hist: *[16]u64) void {
    pinTo(1);
    var idx: u64 = sh.perm[0];
    while (@atomicLoad(u64, sh.go, .acquire) == 0) {}
    var s: u64 = 0;
    while (true) : (s += 1) {
        if (s & 15 == 0) {
            @atomicStore(u64, sh.pos, s, .monotonic);
            if (@atomicLoad(u64, sh.stop, .monotonic) != 0) break;
        }
        if (kendi) @prefetch(&sh.lines[sh.perm[(s + sh.k) % sh.perm.len]], .{ .rw = .read, .locality = 3, .cache = .data });
        const p: *const volatile u64 = @ptrCast(&sh.lines[idx].w[0]);
        if (s & 1023 == 512) {
            const t0 = tsc();
            idx = p.*;
            hist[latBin(tsc() - t0)] += 1;
        } else {
            idx = p.*;
        }
    }
    steps.* = s;
    last.* = idx;
}

fn cmdPatlama(gpa: std.mem.Allocator, kol: PatKol, m: usize, k: u64, sec: f64, seed: u64) !void {
    const lines = allocLines(m);
    @memset(std.mem.sliceAsBytes(lines), 0);
    const perm = try gpa.alloc(u32, m);
    for (perm, 0..) |*p, i| p.* = @intCast(i);
    var r = seed | 1;
    var i: usize = m - 1;
    while (i > 0) : (i -= 1) std.mem.swap(u32, &perm[i], &perm[range(xs(&r), i)]); // Sattolo: tek döngü
    for (0..m) |j| lines[perm[j]].w[0] = perm[(j + 1) % m];
    const ctl = allocLines(4);
    @memset(std.mem.sliceAsBytes(ctl), 0);
    const sh: *PatShared = @ptrCast(@alignCast((allocLines(2)).ptr));
    sh.* = .{ .lines = lines, .perm = perm, .k = k, .go = &ctl[0].w[0], .stop = &ctl[1].w[0], .pos = &ctl[2].w[0] };
    var steps: u64 = 0;
    var last: u64 = 0;
    var yuk: u64 = 0;
    var hist: [16]u64 = @splat(0);
    const yr: ?usize = switch (kol) {
        .yok, .kendi => null,
        .kardes, .bozuk => 7,
        .ccx, .bozuk_ccx => 2,
        .capraz => 3,
    };
    pinTo(0);
    const pmu = Pmu.open();
    const ok = try std.Thread.spawn(.{}, patOkuyucu, .{ sh, kol == .kendi, &steps, &last, &hist });
    const yt: ?std.Thread = if (yr) |c| try std.Thread.spawn(.{}, patYardimci, .{ sh, c, if (kol == .bozuk or kol == .bozuk_ccx) m / 2 else 0, &yuk }) else null;
    sleepSec(0.05);
    @atomicStore(u64, sh.go, 1, .release);
    const a = pmu.snap();
    const t0 = nowNs();
    sleepSec(sec);
    const b = pmu.snap();
    @atomicStore(u64, sh.stop, 1, .release);
    ok.join();
    const dt = nowNs() - t0;
    if (yt) |t| t.join();
    // Zincir kapalı döngü: s adımdan sonra okuyucu perm[s mod m]'de olmalı.
    const yanlis: u8 = if (last == perm[steps % m]) 0 else 1;
    std.debug.print("{{\"cmd\":\"patlama\",\"kol\":\"{s}\",\"m\":{d},\"k\":{d},\"sec\":{d:.3},\"adim\":{d},\"ns_adim\":{d:.3},\"yardimci_yuk\":{d},\"yanlis\":{d},\"hist\":[", .{ @tagName(kol), m, k, @as(f64, @floatFromInt(dt)) * 1e-9, steps, @as(f64, @floatFromInt(dt)) / @as(f64, @floatFromInt(steps)), yuk, yanlis });
    for (hist, 0..) |h, n| std.debug.print("{d}{s}", .{ h, if (n < 15) "," else "" });
    std.debug.print("],", .{});
    pmu.print("pmu", a, b);
    std.debug.print("}}\n", .{});
}

// ---- rezervuar: yazılım kanalı olmadan kolektif durum ----
// Komutan u(t) bitini yalnız hangi satırlara yazacağına çevirir. Okuyucular değerleri kullanmaz, u'yu
// hiç görmez; yalnız kendi rastgele yüklerinin süresini ölçer. u(t−k) okunabiliyorsa bilgi
// koherens/önbellek durumunda taşınmıştır.

const RezKol = enum { ortak, ozel, yok };
const LAT_BINS = 4;
const GROUPS = 3;

const RezShared = struct {
    a: []Line, // A: u=0, B: u=1, N: hiç yazılmaz
    b: []Line,
    n: []Line,
    t0: *u64,
    t_tick: u64,
    steps: usize,
};

const RezCtx = struct {
    sh: *const RezShared,
    cpu: usize,
    delay: u64,
    seed: u64,
    hist: []u32,
    samples: u64 = 0,
};

fn rezReader(c: *RezCtx) void {
    const sh = c.sh;
    pinTo(c.cpu);
    var s = coreRng(c.seed, c.cpu);
    var t0: u64 = 0;
    while (t0 == 0) t0 = @atomicLoad(u64, sh.t0, .acquire);
    const gs = [GROUPS][]Line{ sh.a, sh.b, sh.n };
    var samples: u64 = 0;
    while (true) {
        const r = xs(&s);
        const gi = (r & 0xFF) * 3 >> 8;
        const g = gs[gi];
        const p: *const volatile u64 = @ptrCast(&g[range(r, g.len)].w[0]);
        const ta = tsc();
        _ = p.*;
        const tb = tsc();
        if (tb < t0) continue;
        const step = (tb - t0) / sh.t_tick;
        if (step >= sh.steps) break;
        const d = tb - ta;
        // Kovalar S0'dan: yerel, aynı CCX, CCX dışı/DRAM, yavaş.
        const k: usize = if (d <= 108) 0 else if (d <= 216) 1 else if (d <= 576) 2 else 3;
        c.hist[step * GROUPS * LAT_BINS + gi * LAT_BINS + k] += 1;
        samples += 1;
        if (c.delay != 0) {
            const tw = tsc();
            while (tsc() - tw < c.delay) {}
        }
    }
    c.samples = samples;
}

fn cmdRezervuar(gpa: std.mem.Allocator, kol: RezKol, cpus: []const usize, delays: []const u64, t_tick: u64, steps: usize, g: usize, k: usize, seed: u64, out: [*:0]const u8) !void {
    const mem = allocLines(g * 5 + 8);
    @memset(std.mem.sliceAsBytes(mem), 0);
    const ctl = &mem[g * 5];
    const shm: *RezShared = @ptrCast(@alignCast(&mem[g * 5 + 2]));
    const a = mem[0..g];
    const b = mem[g .. 2 * g];
    const n = mem[2 * g .. 3 * g];
    const pa = mem[3 * g .. 4 * g];
    const pb = mem[4 * g .. 5 * g];
    shm.* = .{ .a = a, .b = b, .n = n, .t0 = &ctl.w[0], .t_tick = t_tick, .steps = steps };
    const us = try gpa.alloc(u8, steps);
    const cs = try gpa.alloc(RezCtx, cpus.len);
    const ths = try gpa.alloc(std.Thread, cpus.len);
    for (cpus, 0..) |cpu, i| {
        const h = std.mem.bytesAsSlice(u32, std.heap.page_allocator.alignedAlloc(u8, .fromByteUnits(HUGE), steps * GROUPS * LAT_BINS * 4) catch @panic("oom"));
        @memset(h, 0);
        cs[i] = .{ .sh = shm, .cpu = cpu, .delay = delays[i], .seed = seed, .hist = @alignCast(h) };
        ths[i] = try std.Thread.spawn(.{}, rezReader, .{&cs[i]});
    }
    pinTo(0);
    sleepSec(0.05);
    var s = seed *% KEY | 1;
    var late: u64 = 0;
    const t0 = tsc() + 3_600_000;
    @atomicStore(u64, shm.t0, t0, .release);
    for (0..steps) |t| {
        const when = t0 + t * t_tick;
        while (tsc() < when) {}
        const r = xs(&s);
        const u: u8 = @intCast(r >> 63);
        us[t] = u;
        const set = switch (kol) {
            .ortak => if (u == 0) a else b,
            .ozel => if (u == 0) pa else pb,
            .yok => a[0..0],
        };
        if (set.len != 0) {
            for (0..k) |_| {
                const i = range(xs(&s), g);
                @as(*volatile u64, @ptrCast(&set[i].w[0])).* = t;
            }
        }
        if (tsc() > when + t_tick) late += 1;
    }
    for (ths) |t| t.join();
    const rc = linux.open(out, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, 0o644);
    if (@as(isize, @bitCast(rc)) < 0) return error.OutOpen;
    const fd: i32 = @intCast(rc);
    writeAll(fd, us);
    for (cs) |*c| writeAll(fd, std.mem.sliceAsBytes(c.hist));
    _ = linux.close(fd);
    std.debug.print("{{\"cmd\":\"rezervuar\",\"kol\":\"{s}\",\"t_tick\":{d},\"steps\":{d},\"g\":{d},\"k\":{d},\"nr\":{d},\"gec\":{d},\"ornek\":[", .{ @tagName(kol), t_tick, steps, g, k, cpus.len, late });
    for (cs, 0..) |*c, i| std.debug.print("{d}{s}", .{ c.samples, if (i + 1 < cs.len) "," else "" });
    std.debug.print("]}}\n", .{});
}

// ---- sezai: canlı rezervuar. Sayımlar paylaşılan dosyaya düşer; okuma, ödül ve emir Python'dadır. ----
// Düzen: 0. satır t0, adım, geç (komutan yazar); 1. satır maske, dur (yönetici yazar, nadir).
// Okuyucular u görmez; u halkası yalnız yöneticinin etiketidir.

const SEZ_RING: usize = 1 << 16;
const SEZ_HUCRE = GROUPS * LAT_BINS;
const SEZ_US_OFF = 4096;
const SEZ_H_OFF = SEZ_US_OFF + SEZ_RING;
const SezKol = enum { ortak, ozel };

const SezShared = struct {
    a: []Line,
    b: []Line,
    n: []Line,
    ctl: [*]u64,
    t_tick: u64,
};

const SezCtx = struct {
    sh: *const SezShared,
    cpu: usize,
    bit: u64,
    delay: u64,
    seed: u64,
    hist: [*]u32,
    samples: u64 = 0,
};

fn sezReader(c: *SezCtx) void {
    const sh = c.sh;
    pinTo(c.cpu);
    var s = coreRng(c.seed, c.cpu);
    var t0: u64 = 0;
    while (t0 == 0) t0 = @atomicLoad(u64, &sh.ctl[0], .acquire);
    const gs = [GROUPS][]Line{ sh.a, sh.b, sh.n };
    var samples: u64 = 0;
    while (@atomicLoad(u64, &sh.ctl[9], .monotonic) == 0) {
        if (@atomicLoad(u64, &sh.ctl[8], .monotonic) & c.bit == 0) {
            sleepSec(0.001);
            continue;
        }
        const r = xs(&s);
        const gi = (r & 0xFF) * 3 >> 8;
        const g = gs[gi];
        const p: *const volatile u64 = @ptrCast(&g[range(r, g.len)].w[0]);
        const ta = tsc();
        _ = p.*;
        const tb = tsc();
        if (tb < t0) continue;
        const slot = ((tb - t0) / sh.t_tick) & (SEZ_RING - 1);
        const d = tb - ta;
        const k: usize = if (d <= 108) 0 else if (d <= 216) 1 else if (d <= 576) 2 else 3;
        c.hist[slot * SEZ_HUCRE + gi * LAT_BINS + k] += 1;
        samples += 1;
        if (c.delay != 0) {
            const tw = tsc();
            while (tsc() - tw < c.delay) {}
        }
    }
    c.samples = samples;
}

fn cmdSezai(gpa: std.mem.Allocator, kol: SezKol, cpus: []const usize, delay: u64, t_tick: u64, g: usize, k: usize, seed: u64, path: [*:0]const u8) !void {
    const rc = linux.open(path, .{ .ACCMODE = .RDWR }, 0);
    if (@as(isize, @bitCast(rc)) < 0) return error.OutOpen;
    const fd: i32 = @intCast(rc);
    const uzun = SEZ_H_OFF + cpus.len * SEZ_RING * SEZ_HUCRE * 4;
    const mp = linux.mmap(null, uzun, .{ .READ = true, .WRITE = true }, .{ .TYPE = .SHARED }, fd, 0);
    if (@as(isize, @bitCast(mp)) < 0 and @as(isize, @bitCast(mp)) > -4096) return error.MapFail;
    const taban: [*]u8 = @ptrFromInt(mp);
    const ctl: [*]u64 = @ptrCast(@alignCast(taban));
    const us = taban + SEZ_US_OFF;

    const mem = allocLines(g * 5 + 8);
    @memset(std.mem.sliceAsBytes(mem), 0);
    const shm: *SezShared = @ptrCast(@alignCast(&mem[g * 5 + 2]));
    const a = mem[0..g];
    const b = mem[g .. 2 * g];
    const n = mem[2 * g .. 3 * g];
    const pa = mem[3 * g .. 4 * g];
    const pb = mem[4 * g .. 5 * g];
    shm.* = .{ .a = a, .b = b, .n = n, .ctl = ctl, .t_tick = t_tick };
    const cs = try gpa.alloc(SezCtx, cpus.len);
    const ths = try gpa.alloc(std.Thread, cpus.len);
    for (cpus, 0..) |cpu, i| {
        const h: [*]u32 = @ptrCast(@alignCast(taban + SEZ_H_OFF + i * SEZ_RING * SEZ_HUCRE * 4));
        cs[i] = .{ .sh = shm, .cpu = cpu, .bit = @as(u64, 1) << @intCast(i), .delay = delay, .seed = seed, .hist = h };
        ths[i] = try std.Thread.spawn(.{}, sezReader, .{&cs[i]});
    }
    pinTo(0);
    sleepSec(0.05);
    var s = seed *% KEY | 1;
    var late: u64 = 0;
    const t0 = tsc() + 3_600_000;
    @atomicStore(u64, &ctl[0], t0, .release);
    var t: u64 = 0;
    while (@atomicLoad(u64, &ctl[9], .monotonic) == 0) : (t += 1) {
        const when = t0 + t * t_tick;
        while (tsc() < when) {}
        const r = xs(&s);
        const u: u8 = @intCast(r >> 63);
        us[t & (SEZ_RING - 1)] = u;
        const set = switch (kol) {
            .ortak => if (u == 0) a else b,
            .ozel => if (u == 0) pa else pb,
        };
        for (0..k) |_| {
            const i = range(xs(&s), g);
            @as(*volatile u64, @ptrCast(&set[i].w[0])).* = t;
        }
        // Halkada t. satır dolu sayılır: yönetici t'yi t+1 yayımlanınca okur.
        @atomicStore(u64, &ctl[1], t + 1, .release);
        if (tsc() > when + t_tick) late += 1;
        ctl[3] = late;
    }
    for (ths) |th| th.join();
    std.debug.print("{{\"cmd\":\"sezai\",\"kol\":\"{s}\",\"t_tick\":{d},\"adim\":{d},\"g\":{d},\"k\":{d},\"gec\":{d},\"ornek\":[", .{ @tagName(kol), t_tick, t, g, k, late });
    for (cs, 0..) |*c, i| std.debug.print("{d}{s}", .{ c.samples, if (i + 1 < cs.len) "," else "" });
    std.debug.print("]}}\n", .{});
}

// ---- kuplaj: paylaşılan veri yok; yalnız güç/saat/ısı üzerinden çekirdekler arası etki ----

const KupKol = enum { avx, bos };
const OkurTuru = enum { alu, zincir };

const KupShared = struct { t0: *u64, t_tick: u64, steps: usize, tur: OkurTuru };
const KupCtx = struct { sh: *const KupShared, cpu: usize, f: []f64 };

/// Tek komutanlı akışla aynı u dizisini tohumdan üretir ve aynı adım sınırlarında yanar/söner.
fn kupDriver(sh: *const KupShared, cpu: usize, seed: u64, kol: KupKol) void {
    pinTo(cpu);
    var s = seed *% KEY | 1;
    var t0: u64 = 0;
    while (t0 == 0) t0 = @atomicLoad(u64, sh.t0, .acquire);
    for (0..sh.steps) |t| {
        const when = t0 + (t + 1) * sh.t_tick;
        const u = xs(&s) >> 63;
        if (u == 1 and kol == .avx) {
            while (tsc() < when) avxBurst();
        } else {
            while (tsc() < when) asm volatile ("pause");
        }
    }
    asm volatile ("vzeroupper");
}

fn aluBlock() void {
    var x: u64 = 1;
    asm volatile (".rept 1024\nadd $1, %[x]\n.endr\n"
        : [x] "+r" (x),
    );
    std.mem.doNotOptimizeAway(x);
}

fn kupReader(c: *KupCtx) void {
    const sh = c.sh;
    pinTo(c.cpu);
    var ch: []Line = &.{};
    var p: u64 = 0;
    if (sh.tur == .zincir) {
        ch = chaseBuild(8 * 1024 * 1024 / LINE, KEY ^ c.cpu);
        p = chaseLoop(@intFromPtr(&ch[0]), ch.len * 2);
    }
    var t0: u64 = 0;
    while (t0 == 0) t0 = @atomicLoad(u64, sh.t0, .acquire);
    while (true) {
        const ta = tsc();
        const pa = rdpru(1);
        if (sh.tur == .alu) aluBlock() else p = chaseLoop(p, 64);
        const pb = rdpru(1);
        const tb = tsc();
        if (tb < t0) continue;
        const step = (tb - t0) / sh.t_tick;
        if (step >= sh.steps) break;
        const f = c.f[step * 4 ..][0..4];
        const dt = @as(f64, @floatFromInt(tb - ta));
        f[0] += dt;
        f[1] += @floatFromInt(pb - pa);
        f[2] += 1;
        f[3] = if (f[2] == 1) dt else @min(f[3], dt);
    }
}

const FMA_K: [4]f64 = @splat(0.5);
const FMA_Z: [4]f64 = @splat(0.25);

inline fn avxBurst() void {
    asm volatile (
        \\vmovupd %[a], %%ymm8
        \\vmovupd %[b], %%ymm9
        \\vxorpd %%ymm0, %%ymm0, %%ymm0
        \\vxorpd %%ymm1, %%ymm1, %%ymm1
        \\vxorpd %%ymm2, %%ymm2, %%ymm2
        \\vxorpd %%ymm3, %%ymm3, %%ymm3
        \\.rept 32
        \\vfmadd231pd %%ymm8, %%ymm9, %%ymm0
        \\vfmadd231pd %%ymm8, %%ymm9, %%ymm1
        \\vfmadd231pd %%ymm8, %%ymm9, %%ymm2
        \\vfmadd231pd %%ymm8, %%ymm9, %%ymm3
        \\.endr
        : // çıktı yok
        : [a] "m" (FMA_K),
          [b] "m" (FMA_Z),
        : .{ .ymm0 = true, .ymm1 = true, .ymm2 = true, .ymm3 = true, .ymm8 = true, .ymm9 = true });
}

fn cmdKuplaj(gpa: std.mem.Allocator, kol: KupKol, cmds: []const usize, cpus: []const usize, t_tick: u64, steps: usize, seed: u64, tur: OkurTuru, out: [*:0]const u8) !void {
    const ctl = allocLines(3);
    @memset(std.mem.sliceAsBytes(ctl), 0);
    // Yığında değil kendi satırında: komutanın yığın yazmaları okuyuculara yanlış paylaşım olarak sızıyordu.
    const sh: *KupShared = @ptrCast(@alignCast(&ctl[1]));
    sh.* = .{ .t0 = &ctl[0].w[0], .t_tick = t_tick, .steps = steps, .tur = tur };
    const us = try gpa.alloc(u8, steps);
    const cs = try gpa.alloc(KupCtx, cpus.len);
    const ths = try gpa.alloc(std.Thread, cpus.len);
    for (cpus, 0..) |cpu, i| {
        const f = try gpa.alloc(f64, steps * 4);
        @memset(f, 0);
        cs[i] = .{ .sh = sh, .cpu = cpu, .f = f };
        ths[i] = try std.Thread.spawn(.{}, kupReader, .{&cs[i]});
    }
    const cmd = cmds[0];
    const drv = try gpa.alloc(std.Thread, cmds.len - 1);
    for (cmds[1..], 0..) |c2, i| drv[i] = try std.Thread.spawn(.{}, kupDriver, .{ sh, c2, seed, kol });
    pinTo(cmd);
    sleepSec(if (tur == .zincir) 0.6 else 0.05);
    var s = seed *% KEY | 1;
    const t0 = tsc() + 36_000_000;
    @atomicStore(u64, sh.t0, t0, .release);
    for (0..steps) |t| {
        const when = t0 + (t + 1) * t_tick;
        const u: u8 = @intCast(xs(&s) >> 63);
        us[t] = u;
        if (u == 1 and kol == .avx) {
            while (tsc() < when) avxBurst();
        } else {
            while (tsc() < when) asm volatile ("pause");
        }
    }
    asm volatile ("vzeroupper");
    for (ths) |t| t.join();
    for (drv) |t| t.join();
    const rc = linux.open(out, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, 0o644);
    if (@as(isize, @bitCast(rc)) < 0) return error.OutOpen;
    const fd: i32 = @intCast(rc);
    writeAll(fd, us);
    for (cs) |*c| writeAll(fd, std.mem.sliceAsBytes(c.f));
    _ = linux.close(fd);
    std.debug.print("{{\"cmd\":\"kuplaj\",\"kol\":\"{s}\",\"komutan\":{d},\"nk\":{d},\"t_tick\":{d},\"steps\":{d},\"nr\":{d},\"tur\":\"{s}\"}}\n", .{ @tagName(kol), cmd, cmds.len, t_tick, steps, cpus.len, @tagName(tur) });
}

// ---- gezgin: aynı hesap sabit çekirdekte mi, dolaşarak mı (activity migration) ----

fn gezginWork(buf: []Line) u64 {
    var sum: u64 = 0;
    var i: usize = 0;
    while (i < buf.len) : (i += 1) sum +%= @as(*const volatile u64, @ptrCast(&buf[i].w[0])).*;
    for (0..256) |_| avxBurst();
    return sum;
}

fn cmdGezgin(cpus: []const usize, tau_ms: f64, sec: f64) !void {
    const buf = allocLines(128 * 1024 / LINE);
    @memset(std.mem.sliceAsBytes(buf), 1);
    const pmu = Pmu.open();
    pinTo(cpus[0]);
    const tau: u64 = @intFromFloat(tau_ms * 3_600_000.0);
    const a = pmu.snap();
    const t0 = tsc();
    const end = t0 + @as(u64, @intFromFloat(sec * 3.6e9));
    var next = t0 + tau;
    var hops: u64 = 0;
    var ops: u64 = 0;
    var sink: u64 = 0;
    var ap: u64 = 0;
    var mp: u64 = 0;
    var ap0 = rdpru(1);
    var mp0 = rdpru(0);
    var pos: usize = 0;
    var tpeak: i64 = 0;
    var tsum: i64 = 0;
    var tn: i64 = 0;
    var tnext = t0;
    while (true) {
        sink +%= gezginWork(buf);
        ops += 1;
        const t = tsc();
        if (t >= tnext) {
            const v = readInt("/sys/class/hwmon/hwmon2/temp1_input");
            tpeak = @max(tpeak, v);
            tsum += v;
            tn += 1;
            tnext = t + 360_000_000;
        }
        if (t >= end) break;
        if (tau != 0 and t >= next) {
            ap +%= rdpru(1) -% ap0;
            mp +%= rdpru(0) -% mp0;
            pos = (pos + 1) % cpus.len;
            pinTo(cpus[pos]);
            ap0 = rdpru(1);
            mp0 = rdpru(0);
            hops += 1;
            next = t + tau;
        }
    }
    ap +%= rdpru(1) -% ap0;
    mp +%= rdpru(0) -% mp0;
    asm volatile ("vzeroupper");
    const b = pmu.snap();
    std.mem.doNotOptimizeAway(sink);
    const real = @as(f64, @floatFromInt(tsc() - t0)) / 3.6e9;
    std.debug.print("{{\"cmd\":\"gezgin\",\"tau_ms\":{d},\"ncpu\":{d},\"sec\":{d:.3},\"ops\":{d},\"ops_s\":{d:.1},\"hops\":{d},\"ghz\":{d:.4},\"t_tepe\":{d},\"t_ort\":{d},", .{ tau_ms, cpus.len, real, ops, @as(f64, @floatFromInt(ops)) / real, hops, 3.6 * @as(f64, @floatFromInt(ap)) / @as(f64, @floatFromInt(mp)), tpeak, @divTrunc(tsum, @max(tn, 1)) });
    pmu.print("pmu", a, b);
    std.debug.print("}}\n", .{});
}

// ---- ana ----

pub fn main(init: std.process.Init) !void {
    const gpa = std.heap.page_allocator;
    var it = try std.process.Args.Iterator.initAllocator(init.minimal.args, gpa);
    defer it.deinit();
    var av = std.array_list.Managed([]const u8).init(gpa);
    while (it.next()) |a| try av.append(a);
    const a = av.items;
    if (a.len < 2) return error.MissingArgument;
    const c = a[1];
    const pI = std.fmt.parseInt;
    if (std.mem.eql(u8, c, "c2c")) {
        try cmdC2c(try pI(usize, a[2], 10), try pI(usize, a[3], 10), try pI(usize, a[4], 10));
    } else if (std.mem.eql(u8, c, "dalga")) {
        try cmdDalga(gpa, try pI(usize, a[2], 10), try parseCpus(gpa, a[3]), try pI(usize, a[4], 10));
    } else if (std.mem.eql(u8, c, "satir")) {
        try cmdSatir(try pI(usize, a[2], 10), try pI(usize, a[3], 10), try pI(usize, a[4], 10), std.meta.stringToEnum(SatirMod, a[5]) orelse return error.BadMod);
    } else if (std.mem.eql(u8, c, "akis")) {
        try cmdAkis(gpa, try parseCpus(gpa, a[2]), try pI(usize, a[3], 10), try std.fmt.parseFloat(f64, a[4]), a[5][0] == 'w');
    } else if (std.mem.eql(u8, c, "merdiven")) {
        cmdMerdiven(a[2]);
    } else if (std.mem.eql(u8, c, "pencere")) {
        cmdPencere(a[2], try pI(usize, a[3], 10));
    } else if (std.mem.eql(u8, c, "halka")) {
        try cmdHalka(gpa, std.meta.stringToEnum(HalkaKol, a[2]) orelse return error.BadArm, try parseCpus(gpa, a[3]), try pI(usize, a[4], 10), try std.fmt.parseFloat(f64, a[5]), try pI(u64, a[6], 10));
    } else if (std.mem.eql(u8, c, "firtina")) {
        try cmdFirtina(gpa, std.meta.stringToEnum(FirtinaKol, a[2]) orelse return error.BadArm, try parseCpus(gpa, a[3]), try pI(usize, a[4], 10), try std.fmt.parseFloat(f64, a[5]), try pI(u64, a[6], 10), try pI(usize, a[7], 10));
    } else if (std.mem.eql(u8, c, "patlama")) {
        try cmdPatlama(gpa, std.meta.stringToEnum(PatKol, a[2]) orelse return error.BadArm, try pI(usize, a[3], 10), try pI(u64, a[4], 10), try std.fmt.parseFloat(f64, a[5]), try pI(u64, a[6], 10));
    } else if (std.mem.eql(u8, c, "dusus")) {
        try cmdDusus(try pI(usize, a[2], 10), try pI(usize, a[3], 10), try pI(usize, a[4], 10), a[5][0] == 'w', try pI(usize, a[6], 10));
    } else if (std.mem.eql(u8, c, "rezervuar")) {
        const cpus = try parseCpus(gpa, a[3]);
        const ds = try parseCpus(gpa, a[4]);
        const dl = try gpa.alloc(u64, ds.len);
        for (ds, 0..) |d, i| dl[i] = d;
        const out = try dupZ(gpa, a[10]);
        try cmdRezervuar(gpa, std.meta.stringToEnum(RezKol, a[2]) orelse return error.BadArm, cpus, dl, try pI(u64, a[5], 10), try pI(usize, a[6], 10), try pI(usize, a[7], 10), try pI(usize, a[8], 10), try pI(u64, a[9], 10), out);
    } else if (std.mem.eql(u8, c, "sezai")) {
        const out = try dupZ(gpa, a[9]);
        try cmdSezai(gpa, std.meta.stringToEnum(SezKol, a[2]) orelse return error.BadArm, try parseCpus(gpa, a[3]), try pI(u64, a[4], 10), try pI(u64, a[5], 10), try pI(usize, a[6], 10), try pI(usize, a[7], 10), try pI(u64, a[8], 10), out);
    } else if (std.mem.eql(u8, c, "kuplaj")) {
        const out = try dupZ(gpa, a[9]);
        try cmdKuplaj(gpa, std.meta.stringToEnum(KupKol, a[2]) orelse return error.BadArm, try parseCpus(gpa, a[3]), try parseCpus(gpa, a[4]), try pI(u64, a[5], 10), try pI(usize, a[6], 10), try pI(u64, a[7], 10), std.meta.stringToEnum(OkurTuru, a[8]) orelse return error.BadArm, out);
    } else if (std.mem.eql(u8, c, "gezgin")) {
        try cmdGezgin(try parseCpus(gpa, a[2]), try std.fmt.parseFloat(f64, a[3]), try std.fmt.parseFloat(f64, a[4]));
    } else {
        return error.UnknownCommand;
    }
}
