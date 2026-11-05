//! Port: llama.cpp tools/llama-bench/llama-bench.cpp (test_prompt, test_gen, warmup, tekrar).
//! Aynı ölçüm tanımı: her test için taze context, warmup, tekrar başına llama_memory_clear.
const std = @import("std");
const qwen35 = @import("qwen35.zig");

const Test = struct { n_prompt: usize, n_gen: usize };

fn testPrompt(ctx: *qwen35.Context, rng: std.Random, n_vocab: usize, n_prompt: usize, tokens: []u32) !void {
    const n_batch = tokens.len;
    var n_processed: usize = 0;
    while (n_processed < n_prompt) {
        const n = @min(n_prompt - n_processed, n_batch);
        for (tokens[0..n]) |*t| t.* = @intCast(rng.uintLessThan(usize, n_vocab));
        try ctx.decode(tokens[0..n]);
        n_processed += n;
    }
}

fn testGen(ctx: *qwen35.Context, rng: std.Random, n_vocab: usize, n_gen: usize) !void {
    var token: [1]u32 = .{@intCast(rng.uintLessThan(usize, n_vocab))};
    for (0..n_gen) |_| {
        try ctx.decode(&token);
        token[0] = @intCast(rng.uintLessThan(usize, n_vocab));
    }
}

pub const Options = struct {
    n_threads: u32 = 6,
    reps: usize = 5,
    prompts: []const usize = &.{ 128, 512 },
    gens: []const usize = &.{ 32, 128 },
};

/// Her test için bir JSON satırı yazar: llama-bench -o json ile aynı alan adları.
pub fn run(model_path: []const u8, opt: Options, allocator: std.mem.Allocator, io: std.Io, out: *std.Io.Writer) !void {
    const model = try qwen35.Model.load(model_path, allocator);
    defer model.deinit();

    var tests: std.ArrayList(Test) = .empty;
    defer tests.deinit(allocator);
    for (opt.prompts) |p| if (p > 0) try tests.append(allocator, .{ .n_prompt = p, .n_gen = 0 });
    for (opt.gens) |n| if (n > 0) try tests.append(allocator, .{ .n_prompt = 0, .n_gen = n });

    var prng = std.Random.DefaultPrng.init(1234);
    const rng = prng.random();
    const tokens = try allocator.alloc(u32, 2048);
    defer allocator.free(tokens);

    for (tests.items) |t| {
        const ctx = try qwen35.Context.init(model, t.n_prompt + t.n_gen, opt.n_threads, allocator);
        defer ctx.deinit();

        // warmup
        ctx.clear();
        if (t.n_prompt > 0) try testPrompt(ctx, rng, model.n_vocab, t.n_prompt, tokens);
        if (t.n_gen > 0) try testGen(ctx, rng, model.n_vocab, 1);

        const samples = try allocator.alloc(f64, opt.reps);
        defer allocator.free(samples);
        for (samples) |*s| {
            ctx.clear();
            const t0 = std.Io.Timestamp.now(io, .awake);
            if (t.n_prompt > 0) try testPrompt(ctx, rng, model.n_vocab, t.n_prompt, tokens);
            if (t.n_gen > 0) try testGen(ctx, rng, model.n_vocab, t.n_gen);
            const ns: f64 = @floatFromInt(t0.untilNow(io, .awake).toNanoseconds());
            s.* = @as(f64, @floatFromInt(t.n_prompt + t.n_gen)) * 1e9 / ns;
        }
        var sum: f64 = 0;
        for (samples) |s| sum += s;
        try out.print("{{\"n_prompt\":{d},\"n_gen\":{d},\"n_threads\":{d},\"avg_ts\":{d:.3},\"samples_ts\":[", .{ t.n_prompt, t.n_gen, opt.n_threads, sum / @as(f64, @floatFromInt(samples.len)) });
        for (samples, 0..) |s, i| {
            if (i > 0) try out.writeAll(",");
            try out.print("{d:.3}", .{s});
        }
        try out.writeAll("]}\n");
        try out.flush();
    }
}

/// Doğrulama: verilen token dizileri için logit dökümü (llama.cpp referansıyla karşılaştırılır).
/// `steps`: ardışık batch boyutları; her batch sonunda son token'ın logit'leri yazılır.
pub fn dumpLogits(model_path: []const u8, tokens: []const u32, steps: []const usize, n_threads: u32, n_ctx: usize, allocator: std.mem.Allocator, out: *std.Io.Writer) !void {
    const model = try qwen35.Model.load(model_path, allocator);
    defer model.deinit();
    const ctx = try qwen35.Context.init(model, n_ctx, n_threads, allocator);
    defer ctx.deinit();
    var off: usize = 0;
    for (steps) |n| {
        try ctx.decode(tokens[off..][0..n]);
        off += n;
        try out.writeAll(std.mem.sliceAsBytes(ctx.logits));
    }
    try out.flush();
}
