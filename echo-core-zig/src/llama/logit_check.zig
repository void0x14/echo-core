//! Tam yolun logit'leri referansla bit bit aynı mı: echo-core-zig --logit-check [--save] [-t N]
//! H20 (Q8_0_R8) toplama sırasını değiştirir: parite için ECHO_NO_R8=1 ile çalıştır.
//! Referans llama.cpp ile sıfır farklı doğrulanmış kod yolundan alındı.
const std = @import("std");
const qwen35 = @import("qwen35.zig");

const REF = "benchmarks/logit-ref.txt";
const NAMES = [_][]const u8{ "Qwen3.5-0.8B-Q8_0", "Qwen3.5-0.8B-UD-Q8_K_XL", "unc-0.8b-q5km" };
const SEQS = [_][]const usize{ &.{ 5, 1, 128, 1, 300, 1 }, &.{ 512, 1, 1 }, &.{ 7, 300, 300, 1, 1 } };

pub fn run(save: bool, n_threads: u32, a: std.mem.Allocator, io: std.Io) !void {
    var text: std.Io.Writer.Allocating = .init(a);
    defer text.deinit();
    for (NAMES) |name| {
        const path = try std.fmt.allocPrint(a, "../ArastirmaModelleri/{s}.gguf", .{name});
        defer a.free(path);
        const model = try qwen35.Model.load(path, a);
        defer model.deinit();
        for (SEQS, 0..) |steps, si| {
            var n: usize = 0;
            for (steps) |s| n += s;
            var prng = std.Random.DefaultPrng.init(1234);
            const toks = try a.alloc(u32, n);
            defer a.free(toks);
            for (toks) |*t| t.* = 1000 + prng.random().uintLessThan(u32, 47000);
            const ctx = try qwen35.Context.init(model, n + 1, n_threads, a);
            defer ctx.deinit();
            var h = std.crypto.hash.sha2.Sha256.init(.{});
            var off: usize = 0;
            for (steps) |s| {
                try ctx.decode(toks[off..][0..s]);
                off += s;
                h.update(std.mem.sliceAsBytes(ctx.logits));
            }
            const d = h.finalResult();
            try text.writer.print("{s}|{d} {x}\n", .{ name, si, &d });
        }
    }
    const got = text.written();
    if (save) {
        const f = try std.Io.Dir.cwd().createFile(io, REF, .{});
        defer f.close(io);
        var buf: [4096]u8 = undefined;
        var w = f.writer(io, &buf);
        try w.interface.writeAll(got);
        try w.interface.flush();
        std.debug.print("yazıldı: {s}\n", .{REF});
        return;
    }
    const ref = try std.Io.Dir.cwd().readFileAlloc(io, REF, a, .limited(1 << 16));
    defer a.free(ref);
    var ok: usize = 0;
    var total: usize = 0;
    var gi = std.mem.splitScalar(u8, got, '\n');
    var ri = std.mem.splitScalar(u8, ref, '\n');
    while (gi.next()) |g| {
        const r = ri.next() orelse "";
        if (g.len == 0) continue;
        total += 1;
        if (std.mem.eql(u8, g, r)) ok += 1 else std.debug.print("FARK {s}\n", .{g});
    }
    std.debug.print("{d}/{d} eşit\n", .{ ok, total });
    if (ok != total) return error.LogitFarki;
}
