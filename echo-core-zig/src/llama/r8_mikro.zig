//! R8 çekirdek mikro ölçümü: tek thread, n=8, matris boyu ortamdan (R8_M satır, k=1024).
//! zig test -OReleaseFast src/llama/r8_mikro.zig  →  perf stat ile döngü/FP0 sayılır.
const std = @import("std");
const avx = @import("avx.zig");
const q = @import("quants.zig");
const sgemm = @import("sgemm.zig");

test "r8 mikro" {
    try avx.initLibm();
    const a = std.heap.page_allocator;
    const m: usize = if (std.c.getenv("R8_M")) |s| try std.fmt.parseInt(usize, std.mem.span(s), 10) else 128;
    const reps: usize = if (std.c.getenv("R8_REP")) |s| try std.fmt.parseInt(usize, std.mem.span(s), 10) else 1000;
    const nb = 32;
    const n: usize = if (std.c.getenv("R8_N")) |s| try std.fmt.parseInt(usize, std.mem.span(s), 10) else 8;
    var rng = std.Random.DefaultPrng.init(1);
    const r = rng.random();
    const src = try a.alloc(q.BlockQ80, m * nb);
    for (src) |*b| {
        b.d = @bitCast(@as(f16, 0.01));
        for (&b.qs) |*x| x.* = r.intRangeAtMost(i8, -127, 127);
    }
    const rep = try a.alignedAlloc(u8, .@"64", m * nb * sgemm.R8_BLOCK);
    sgemm.repackQ80x8(src.ptr, rep.ptr, m, nb);
    const act = try a.alloc(q.BlockQ80, n * nb);
    for (act) |*b| {
        b.d = @bitCast(@as(f16, 0.02));
        for (&b.qs) |*x| x.* = r.intRangeAtMost(i8, -127, 127);
    }
    for (act) |*b| for (&b.qs) |*x| { x.* = @divTrunc(x.*, 2) +% 64; };
    const c = try a.alloc(f32, n * m);
    const g = sgemm.GemmR8{ .pool = undefined, .nb = nb, .a = rep.ptr, .b = act.ptr, .ldb = nb, .c = c.ptr, .ldc = m, .ith = 0 };
    for (0..reps) |_| {
        for (0..sgemm.GemmR8.chunks(m)) |cb| g.chunk(cb, m, n);
    }
    const ops: f64 = @floatFromInt(m / 8 * nb * 8 * n * 2 * reps); // zorunlu FP0 op
    var s: f64 = 0;
    for (c) |x| s += x;
    std.debug.print("m={d} ops={d:.0} sum={d:.3}\n", .{ m, ops, s });
}
