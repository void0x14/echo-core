//! `--tok-check model korpus.txt ref.txt`: korpusu "\n\n" ile paragraflara böler, her paragrafı encode eder,
//! ref satırıyla (resmi tokenizer id'leri; benchmarks/tokenizer/hazirla.py) karşılaştırır, hızı yazar.
const std = @import("std");
const gguf = @import("../gguf/reader.zig");
const Bpe = @import("bpe.zig").Bpe;

const ArrayList = std.array_list.Managed;

pub fn run(model_path: []const u8, korpus_path: []const u8, ref_path: []const u8, a: std.mem.Allocator, io: std.Io) !void {
    var reader = try gguf.Reader.openWithAllocator(model_path, a);
    defer reader.deinit();
    var bpe = try Bpe.initFromReader(&reader, a);
    defer bpe.deinit();
    const korpus = try std.Io.Dir.cwd().readFileAlloc(io, korpus_path, a, .unlimited);
    defer a.free(korpus);
    const ref = try std.Io.Dir.cwd().readFileAlloc(io, ref_path, a, .unlimited);
    defer a.free(ref);

    var out = ArrayList(u32).init(a);
    defer out.deinit();
    var tam: usize = 0;
    var top: usize = 0;
    var rl = std.mem.splitScalar(u8, ref, '\n');
    var it = std.mem.splitSequence(u8, korpus, "\n\n");
    var ilk_fark = true;
    while (it.next()) |par| {
        const line = rl.next() orelse "";
        out.clearRetainingCapacity();
        try bpe.encode(par, &out, true);
        var ok = true;
        var ids = std.mem.tokenizeScalar(u8, line, ' ');
        var j: usize = 0;
        while (ids.next()) |s| : (j += 1) {
            const v = try std.fmt.parseInt(u32, s, 10);
            if (j >= out.items.len or out.items[j] != v) {
                ok = false;
                break;
            }
        }
        if (j != out.items.len) ok = false;
        if (ok) tam += 1 else if (ilk_fark) {
            ilk_fark = false;
            std.debug.print("ilk fark: paragraf {d}, token {d}: {s}\n", .{ top, j, par[0..@min(par.len, 80)] });
        }
        top += 1;
    }
    // hız: tüm korpus tek parça, 5 tekrar en iyisi
    var best: u64 = std.math.maxInt(u64);
    for (0..5) |_| {
        out.clearRetainingCapacity();
        const t0 = std.Io.Timestamp.now(io, .awake);
        try bpe.encode(korpus, &out, true);
        best = @min(best, @as(u64, @intCast(t0.untilNow(io, .awake).toNanoseconds())));
    }
    const mbs = @as(f64, @floatFromInt(korpus.len)) / @as(f64, @floatFromInt(best)) * 1e3;
    std.debug.print("{s}: {d}/{d} paragraf ayni, {d} token, {d:.1} MB/s\n", .{ korpus_path, tam, top, out.items.len, mbs });
}
