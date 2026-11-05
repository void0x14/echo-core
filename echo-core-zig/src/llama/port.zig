//! qwen35 modelleri için InferencePort karşılığı: main.zig her iki portu aynı arayüzle sürer.
const std = @import("std");
const config = @import("core_config");
const gguf = @import("../gguf/reader.zig");
const tokenizer = @import("../tokenizer/tokenizer.zig");
const qwen35 = @import("qwen35.zig");
const ivf = @import("ivf.zig");

const ArrayList = std.array_list.Managed;

pub const Port = struct {
    allocator: std.mem.Allocator,
    model: *qwen35.Model,
    ctx: ?*qwen35.Context = null,
    tokenizer_: tokenizer.SimpleTokenizer,
    cfg: config.ModelConfig,
    n_threads: u32,
    /// Varsayılan LM head: IVF (yaklaşık, yalnız top-1). null: tam yol.
    lm_ix: ?*ivf.Runtime = null,

    pub const BenchResult = struct {
        text: []u8,
        prompt_tokens: usize,
        generated_tokens: u32,
        prefill_ms: i64,
        decode_ms: i64,
    };

    /// llama_context_default n_ctx yerine: konuşma için makul taban, istek daha büyükse ona göre.
    const MIN_CTX = 4096;

    pub fn init(model_path: []const u8, allocator: std.mem.Allocator) !Port {
        var reader = try gguf.Reader.openWithAllocator(model_path, allocator);
        defer reader.deinit();
        const cpus = std.Thread.getCpuCount() catch 2;
        const n_threads: u32 = @intCast(@max(cpus / 2, 1)); // llama.cpp varsayılanı: fiziksel çekirdek
        const model = try qwen35.Model.load(model_path, allocator);
        errdefer model.deinit();
        // Varsayılan LM head IVF; yalnız Q8_0 çıktıda desteklenir, aksi halde tam yol.
        const lm_ix = ivf.open(model_path, model.output, ivf.default_l, n_threads, allocator, std.Io.Threaded.global_single_threaded.io()) catch null;
        errdefer if (lm_ix) |ix| ix.deinit(allocator);
        return .{
            .allocator = allocator,
            .model = model,
            .lm_ix = lm_ix,
            .tokenizer_ = try tokenizer.SimpleTokenizer.initFromReader(&reader, allocator),
            .cfg = reader.config,
            .n_threads = n_threads,
        };
    }

    pub fn deinit(self: *Port, allocator: std.mem.Allocator) void {
        _ = allocator;
        if (self.ctx) |c| {
            if (c.lm) |s| s.deinit(self.allocator);
            c.deinit();
        }
        if (self.lm_ix) |ix| ix.deinit(self.allocator);
        self.model.deinit();
        self.tokenizer_.deinit();
    }

    /// Yeterli boş hücresi olan context döndürür; büyütmek geçmişi siler, bu yüzden yalnız boşken büyür.
    fn context(self: *Port, need: usize) !*qwen35.Context {
        if (self.ctx) |c| {
            if (c.head + need <= c.n_ctx) return c;
            if (c.head != 0) return error.ContextFull;
            if (c.lm) |s| s.deinit(self.allocator);
            c.deinit();
            self.ctx = null;
        }
        const c = try qwen35.Context.init(self.model, @max(need, MIN_CTX), self.n_threads, self.allocator);
        errdefer c.deinit();
        if (self.lm_ix) |ix| c.lm = try ivf.Search.init(ix, ivf.default_nprobe, self.n_threads, self.allocator);
        self.ctx = c;
        return c;
    }

    fn greedy(logits: []const f32) u32 {
        var best: usize = 0;
        for (logits, 0..) |v, i| {
            if (v > logits[best]) best = i;
        }
        return @intCast(best);
    }

    fn run(self: *Port, ids: []const u32, max_tokens: u32, all_ids: *ArrayList(u32), io: ?std.Io, prefill_ms: *i64, decode_ms: *i64) !u32 {
        const ctx = try self.context(ids.len + max_tokens);
        const t0 = if (io) |i| std.Io.Timestamp.now(i, .awake) else undefined;
        try ctx.decode(ids);
        if (io) |i| prefill_ms.* = t0.untilNow(i, .awake).toMilliseconds();
        const t1 = if (io) |i| std.Io.Timestamp.now(i, .awake) else undefined;
        var generated: u32 = 0;
        while (generated < max_tokens) : (generated += 1) {
            const next: u32 = if (ctx.lm) |s| s.token else greedy(ctx.logits);
            if (next == self.tokenizer_.eos()) break;
            try all_ids.append(next);
            try ctx.decode(&.{next});
        }
        if (io) |i| decode_ms.* = t1.untilNow(i, .awake).toMilliseconds();
        return generated;
    }

    pub fn generate(self: *Port, prompt: []const u8, max_tokens: u32) ![]u8 {
        var ids = try self.tokenizer_.encode(prompt);
        defer ids.deinit();
        if (ids.items.len == 0) return error.EmptyInput;
        var all_ids = ArrayList(u32).init(self.allocator);
        defer all_ids.deinit();
        try all_ids.appendSlice(ids.items);
        var unused: i64 = 0;
        _ = try self.run(ids.items, max_tokens, &all_ids, null, &unused, &unused);
        return self.tokenizer_.decode(all_ids.items);
    }

    pub fn benchmark(self: *Port, prompt: []const u8, max_tokens: u32, io: std.Io) !BenchResult {
        self.reset();
        var ids = try self.tokenizer_.encode(prompt);
        defer ids.deinit();
        if (ids.items.len == 0) return error.EmptyInput;
        var all_ids = ArrayList(u32).init(self.allocator);
        defer all_ids.deinit();
        try all_ids.appendSlice(ids.items);
        var prefill_ms: i64 = 0;
        var decode_ms: i64 = 0;
        const generated = try self.run(ids.items, max_tokens, &all_ids, io, &prefill_ms, &decode_ms);
        return .{
            .text = try self.tokenizer_.decode(all_ids.items),
            .prompt_tokens = ids.items.len,
            .generated_tokens = generated,
            .prefill_ms = prefill_ms,
            .decode_ms = decode_ms,
        };
    }

    pub fn freeGenerated(self: *Port, text: []u8) void {
        self.tokenizer_.allocator.free(text);
    }

    pub fn reset(self: *Port) void {
        if (self.ctx) |c| c.clear();
    }

    pub fn getConfig(self: *const Port) config.ModelConfig {
        return self.cfg;
    }
};
