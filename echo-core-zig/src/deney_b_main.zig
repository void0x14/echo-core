//! Deney B: iş parçacığı yerleşimi (aynı çekirdek / aynı CCX / farklı CCX / serbest) ring gecikmesini değiştirir mi?
//! Tek üretici, tek tüketici, SPSC ring (64 slot, 4096 B blok). Her blok doğrulanır.
//! İki ölçüm: ping-pong (aynı anda tek blok uçuşta; saf el sıkışma gecikmesi) ve pipeline (ring dolu; throughput).
//! Yerleşim sysfs'ten gelen CPU çiftleriyle belirlenir (driver: tools/run_deney_b.py).
//! Çalıştırma: zig-out/bin/deney-b <single|pair|free> <cpuA> <cpuB>
//! Çıktı: stderr'de tek satır JSON.

const std = @import("std");

const BLK = 4096;
const SLOTS = 64;
const N_PP: u64 = 100_000;
const N_PIPE: u64 = 200_000;
const KEY: u64 = 0x9E3779B97F4A7C15;

const Slot = struct {
    /// 0 = boş; seq+1 = `seq` numaralı blok yayınlandı (release/acquire ile).
    seq: std.atomic.Value(u64),
    t_pub: u64,
    data: [BLK]u8 align(64),
};

const Shared = struct {
    slots: []Slot,
    /// Tüketicinin işlediği blok sayısı (flow kontrolü).
    consumed: std.atomic.Value(u64),
    n: u64,
    pingpong: bool,
    lat: []u64,
};

const ProdCtx = struct { sh: *Shared, pin: ?usize, mask: u64 = 0 };
const ConsCtx = struct { sh: *Shared, pin: ?usize, mask: u64 = 0, mism: u64 = 0 };

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

fn currentCpu() usize {
    var c: usize = 0;
    _ = std.os.linux.getcpu(&c, null);
    return c;
}

fn fillBlock(slot: *Slot, seq: u64) void {
    const w: *[BLK / 8]u64 = @ptrCast(@alignCast(&slot.data));
    for (w, 0..) |*x, i| x.* = seq *% KEY +% @as(u64, @intCast(i));
}

fn verifyBlock(slot: *Slot, seq: u64) bool {
    const w: *[BLK / 8]u64 = @ptrCast(@alignCast(&slot.data));
    for (w, 0..) |x, i| {
        if (x != seq *% KEY +% @as(u64, @intCast(i))) return false;
    }
    return true;
}

fn producer(ctx: *ProdCtx) void {
    if (ctx.pin) |c| pinTo(c);
    const sh = ctx.sh;
    var seq: u64 = 0;
    while (seq < sh.n) : (seq += 1) {
        // Ping-pong: önceki blok tüketilmeden yenisini yayınlama. Pipeline: ring dolmadan yayınla.
        if (sh.pingpong) {
            while (sh.consumed.load(.acquire) < seq) {}
        } else {
            while (sh.consumed.load(.acquire) + SLOTS <= seq) {}
        }
        const slot = &sh.slots[@intCast(seq % SLOTS)];
        fillBlock(slot, seq);
        slot.t_pub = nowNs();
        slot.seq.store(seq + 1, .release);
        if (seq % 256 == 0) ctx.mask |= @as(u64, 1) << @as(u6, @intCast(currentCpu()));
    }
}

fn consumer(ctx: *ConsCtx) void {
    if (ctx.pin) |c| pinTo(c);
    const sh = ctx.sh;
    var seq: u64 = 0;
    while (seq < sh.n) : (seq += 1) {
        const slot = &sh.slots[@intCast(seq % SLOTS)];
        while (slot.seq.load(.acquire) != seq + 1) {}
        // Gecikme: yayın → blok tam okunup doğrulanana kadar (veri transferi dahil; doğrulama maliyeti tüm koşullarda aynı).
        const ok = verifyBlock(slot, seq);
        sh.lat[@intCast(seq)] = nowNs() - slot.t_pub;
        if (!ok) ctx.mism += 1;
        sh.consumed.store(seq + 1, .release);
        if (seq % 256 == 0) ctx.mask |= @as(u64, 1) << @as(u6, @intCast(currentCpu()));
    }
}

fn runPair(gpa: std.mem.Allocator, pingpong: bool, n: u64, pin_p: ?usize, pin_c: ?usize, out: *Result) !void {
    const slots = try gpa.alloc(Slot, SLOTS);
    defer gpa.free(slots);
    for (slots) |*s| s.seq = std.atomic.Value(u64).init(0);
    const lat = try gpa.alloc(u64, @intCast(n));
    defer gpa.free(lat);
    var sh = Shared{
        .slots = slots,
        .consumed = std.atomic.Value(u64).init(0),
        .n = n,
        .pingpong = pingpong,
        .lat = lat,
    };
    var pc = ProdCtx{ .sh = &sh, .pin = pin_p };
    var cc = ConsCtx{ .sh = &sh, .pin = pin_c };

    const t0 = nowNs();
    const tp = try std.Thread.spawn(.{}, producer, .{&pc});
    const tc = try std.Thread.spawn(.{}, consumer, .{&cc});
    tp.join();
    tc.join();
    const elapsed = nowNs() - t0;

    out.lat_p50 = percentile(lat, 50);
    out.lat_p99 = percentile(lat, 99);
    out.elapsed_ns = elapsed;
    out.mism = cc.mism;
    out.mask_p = pc.mask;
    out.mask_c = cc.mask;
}

fn percentile(lat: []u64, pct: u64) u64 {
    const tmp = std.heap.page_allocator.dupe(u64, lat) catch @panic("oom");
    defer std.heap.page_allocator.free(tmp);
    std.mem.sort(u64, tmp, {}, std.sort.asc(u64));
    return tmp[@intCast((tmp.len - 1) * pct / 100)];
}

const Result = struct {
    lat_p50: u64 = 0,
    lat_p99: u64 = 0,
    elapsed_ns: u64 = 0,
    mism: u64 = 0,
    mask_p: u64 = 0,
    mask_c: u64 = 0,
};

/// Tek thread: üretici ve tüketici aynı thread'de sırayla çalışır (iş parçacığı geçişi yok).
fn runSingle(gpa: std.mem.Allocator, n: u64, pin: ?usize, out: *Result) !void {
    if (pin) |c| pinTo(c);
    const slots = try gpa.alloc(Slot, 1);
    defer gpa.free(slots);
    const lat = try gpa.alloc(u64, @intCast(n));
    defer gpa.free(lat);
    const slot = &slots[0];
    var mism: u64 = 0;
    var mask: u64 = 0;
    const t0 = nowNs();
    var seq: u64 = 0;
    while (seq < n) : (seq += 1) {
        const s0 = nowNs();
        fillBlock(slot, seq);
        if (!verifyBlock(slot, seq)) mism += 1;
        lat[@intCast(seq)] = nowNs() - s0;
        if (seq % 256 == 0) mask |= @as(u64, 1) << @as(u6, @intCast(currentCpu()));
    }
    out.elapsed_ns = nowNs() - t0;
    out.lat_p50 = percentile(lat, 50);
    out.lat_p99 = percentile(lat, 99);
    out.mism = mism;
    out.mask_p = mask;
    out.mask_c = mask;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    var args_it = try std.process.Args.Iterator.initAllocator(init.minimal.args, gpa);
    defer args_it.deinit();
    var args = std.array_list.Managed([]const u8).init(gpa);
    defer args.deinit();
    while (args_it.next()) |arg| try args.append(arg);

    if (args.items.len < 2) {
        std.debug.print("kullanım: {s} <single <cpu> | pair <cpuA> <cpuB> | free>\n", .{args.items[0]});
        return error.MissingArgument;
    }
    const mode = args.items[1];

    var pp = Result{};
    var pipe = Result{};
    if (std.mem.eql(u8, mode, "single")) {
        const cpu = try std.fmt.parseInt(usize, args.items[2], 10);
        try runSingle(gpa, N_PP, cpu, &pp);
        try runSingle(gpa, N_PIPE, cpu, &pipe);
    } else if (std.mem.eql(u8, mode, "pair")) {
        const a = try std.fmt.parseInt(usize, args.items[2], 10);
        const b = try std.fmt.parseInt(usize, args.items[3], 10);
        try runPair(gpa, true, N_PP, a, b, &pp);
        try runPair(gpa, false, N_PIPE, a, b, &pipe);
    } else if (std.mem.eql(u8, mode, "free")) {
        try runPair(gpa, true, N_PP, null, null, &pp);
        try runPair(gpa, false, N_PIPE, null, null, &pipe);
    } else {
        return error.UnknownMode;
    }

    const pipe_s = @as(f64, @floatFromInt(pipe.elapsed_ns)) * 1e-9;
    const mb_s = @as(f64, @floatFromInt(N_PIPE)) * BLK / pipe_s / 1e6;
    std.debug.print(
        "{{\"mode\":\"{s}\",\"pp_p50_ns\":{d},\"pp_p99_ns\":{d},\"pp_mism\":{d}," ++
            "\"pipe_mb_s\":{d:.1},\"pipe_p50_ns\":{d},\"pipe_p99_ns\":{d},\"pipe_mism\":{d}," ++
            "\"mask_prod\":{d},\"mask_cons\":{d},\"n_pp\":{d},\"n_pipe\":{d}}}\n",
        .{
            mode,            pp.lat_p50,     pp.lat_p99,  pp.mism,
            mb_s,            pipe.lat_p50,   pipe.lat_p99, pipe.mism,
            pipe.mask_p,     pipe.mask_c,    N_PP,        N_PIPE,
        },
    );
}
