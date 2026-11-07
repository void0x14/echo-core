//! R8 çekirdek mikro ölçümü: tek thread, n=8, matris boyu ortamdan (R8_M satır, k=1024).
//! zig test -lc -OReleaseFast src/llama/r8_mikro.zig  →  perf stat ile döngü sayılır. R8_N sütun sayısı.
//! Ağırlıklar ±20: i16 şerit toplamı taşmaz, maxerr kayan nokta referansına göre çekirdek hatasını gösterir.
const std = @import("std");
const avx = @import("avx.zig");
const q = @import("quants.zig");
const sgemm = @import("sgemm.zig");

test "r8 mikro" {
    try avx.initLibm();
    const a = std.heap.page_allocator;
    const m: usize = if (std.c.getenv("R8_M")) |s| try std.fmt.parseInt(usize, std.mem.span(s), 10) else 128;
    const reps: usize = if (std.c.getenv("R8_REP")) |s| try std.fmt.parseInt(usize, std.mem.span(s), 10) else 1000;
    const nb: usize = if (std.c.getenv("R8_NB")) |s| try std.fmt.parseInt(usize, std.mem.span(s), 10) else 32;
    const n: usize = if (std.c.getenv("R8_N")) |s| try std.fmt.parseInt(usize, std.mem.span(s), 10) else 8;
    var rng = std.Random.DefaultPrng.init(1);
    const r = rng.random();
    const src = try a.alloc(q.BlockQ80, m * nb);
    for (src) |*b| {
        b.d = @bitCast(@as(f16, 0.01));
        for (&b.qs) |*x| x.* = r.intRangeAtMost(i8, -20, 20);
    }
    const rep = try a.alignedAlloc(u8, .@"64", m * nb * sgemm.R8_BLOCK);
    sgemm.repackQ80x8(src.ptr, rep.ptr, m, nb);
    const act = try a.alloc(q.BlockQ80, n * nb);
    const ref_act = try a.alloc(q.BlockQ80, n * nb);
    const xf = try a.alloc(f32, n * nb * 32);
    for (xf) |*v| v.* = r.floatNorm(f32);
    for (0..n) |j| {
        sgemm.quantR8(xf.ptr + j * nb * 32, act.ptr + j * nb, nb * 32);
        q.quantizeRowQ(63.0, xf.ptr + j * nb * 32, ref_act.ptr + j * nb, nb * 32);
    }
    const c = try a.alloc(f32, n * m);
    const g = sgemm.GemmR8{ .pool = undefined, .nb = nb, .a = rep.ptr, .b = act.ptr, .ldb = nb, .c = c.ptr, .ldc = m, .ith = 0 };
    for (0..reps) |_| {
        for (0..sgemm.GemmR8.chunks(m)) |cb| g.chunk(cb, m, n);
    }
    // H56: R16 çekirdeği R8 ile bit-aynı olmalı
    if (m % 16 == 0) {
        const rep16 = try a.alignedAlloc(u8, .@"64", m * nb * sgemm.R8_BLOCK);
        sgemm.repackQ80x16(src.ptr, rep16.ptr, m, nb);
        const c16 = try a.alloc(f32, n * m);
        var g16 = g;
        g16.a = rep16.ptr;
        g16.c = c16.ptr;
        g16.r16 = true;
        for (0..sgemm.GemmR8.chunks(m)) |cb| g16.chunk(cb, m, n);
        var nd: usize = 0;
        for (c, c16) |x, y| nd += @intFromBool(@as(u32, @bitCast(x)) != @as(u32, @bitCast(y)));
        std.debug.print("r16 bit farkı: {d}/{d}\n", .{ nd, c.len });
        try std.testing.expectEqual(0, nd);
        if (std.c.getenv("R16_T") != null) for (0..reps) |_| {
            for (0..sgemm.GemmR8.chunks(m)) |cb| g16.chunk(cb, m, n);
        };
    }
    var maxe: f32 = 0;
    var maxv: f32 = 0;
    for (0..n) |j| for (0..m) |row| {
        var ref: f32 = 0;
        for (0..nb) |l| {
            const wb = src[row * nb + l];
            const ab = ref_act[j * nb + l];
            var si: i32 = 0;
            for (wb.qs, ab.qs) |x, y| si += @as(i32, x) * y;
            ref += avx.fp16_table[wb.d] * avx.fp16_table[ab.d] * @as(f32, @floatFromInt(si));
        }
        maxe = @max(maxe, @abs(ref - c[j * m + row]));
        maxv = @max(maxv, @abs(ref));
    };
    std.debug.print("maxerr={d:.5} maxref={d:.3}\n", .{ maxe, maxv });
    const ops: f64 = @floatFromInt(m / 8 * nb * 8 * n * 2 * reps); // zorunlu FP0 op
    var s: f64 = 0;
    for (c) |x| s += x;
    std.debug.print("m={d} ops={d:.0} sum={d:.3}\n", .{ m, ops, s });
}
