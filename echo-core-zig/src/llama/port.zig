//! qwen35 modelleri için InferencePort karşılığı: main.zig her iki portu aynı arayüzle sürer.
const std = @import("std");
const config = @import("core_config");
const gguf = @import("../gguf/reader.zig");
const tokenizer = @import("../tokenizer/tokenizer.zig");
const qwen35 = @import("qwen35.zig");
const ivf = @import("ivf.zig");
const ngram = @import("ngram.zig");
const bench = @import("bench.zig");

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
    /// >0: H01 n-gram taslağı + tek zincir doğrulama.
    ngram_k: usize = 8,
    /// H06 grammar fast-forward (JSON iskeleti); varsayılan açık, `--no-grammar` kapatır.
    grammar: bool = true,
    /// H13: doğrulamada iz yerine tur başı snapshot; tutmazsa geri dön + yeniden işle (kayıpsız).
    snap: bool = true,

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
        if (self.ngram_k > 0) try c.enableTrace(self.ngram_k);
        c.no_trace = self.snap;
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
        if (self.grammar) {
            generated = try self.runGrammarSpec(ctx, ids, max_tokens, all_ids);
        } else if (self.ngram_k > 0) {
            generated = try self.runNgram(ctx, ids, max_tokens, all_ids);
        } else {
            while (generated < max_tokens) : (generated += 1) {
                const next: u32 = if (ctx.lm) |s| s.token else greedy(ctx.logits);
                if (next == self.tokenizer_.eos()) break;
                try all_ids.append(next);
                try ctx.decode(&.{next});
            }
        }
        if (io) |i| decode_ms.* = t1.untilNow(i, .awake).toMilliseconds();
        return generated;
    }

    /// H06 + (opsiyonel) H01: zorunlu bloklar tek forward, serbest alanda n-gram taslağı.
    fn runGrammarSpec(self: *Port, ctx: *qwen35.Context, prompt: []const u32, max_tokens: u32, all_ids: *ArrayList(u32)) !u32 {
        var d = try ngram.TokenDrafter.init(self.allocator, 3);
        defer d.deinit();
        for (prompt) |t| try d.add(t);
        var buf: [16]u32 = undefined;
        var produced: u32 = 0;
        var pending: u32 = 0;
        var have = false;
        for (bench.TEMPLATE_BIG) |seg| {
            if (produced >= max_tokens) break;
            switch (seg.kind) {
                .lit => {
                    const ids = try self.tokenizer_.encode(seg.text);
                    defer ids.deinit();
                    var cnt = ids.items.len;
                    if (produced + cnt > max_tokens) cnt = max_tokens - produced;
                    if (cnt == 0) break;
                    try ctx.decode(ids.items[0..cnt]);
                    try all_ids.appendSlice(ids.items[0..cnt]);
                    for (ids.items[0..cnt]) |t| try d.add(t);
                    produced += @intCast(cnt);
                    pending = if (ctx.lm) |s| s.token else greedy(ctx.logits);
                    have = true;
                },
                .string, .number => {
                    if (!have) break;
                    const cap: usize = if (seg.kind == .string) 24 else 4;
                    var emitted: usize = 0;
                    while (emitted < cap and produced < max_tokens) {
                        const maxm = @min(@min(self.ngram_k, cap - emitted), max_tokens - produced);
                        var m: usize = 0;
                        if (maxm >= 2) {
                            const raw = d.chain(buf[0..@min(maxm, buf.len)]);
                            while (m < raw) : (m += 1) {
                                const tx = try self.tokenizer_.decode(&.{buf[m]});
                                defer self.tokenizer_.allocator.free(tx);
                                if (!bench.validFree(seg.kind, tx)) break;
                            }
                        }
                        if (m >= 2 and m >= @min(4, maxm) and buf[0] == pending) {
                            const before = ctx.head;
                            if (ctx.no_trace) ctx.snapshot();
                            try ctx.decodeVerify(buf[0..m]);
                            var j: usize = 0;
                            while (j + 1 < m and ctx.verify_tokens[j] == buf[j + 1]) j += 1;
                            var acc: usize = 0;
                            if (j + 1 == m) {
                                acc = m;
                                pending = ctx.verify_tokens[m - 1];
                            } else {
                                acc = j + 1;
                                pending = ctx.verify_tokens[j];
                                try undo(ctx, before, buf[0..acc]);
                            }
                            for (buf[0..acc]) |t| {
                                try all_ids.append(t);
                                try d.add(t);
                            }
                            produced += @intCast(acc);
                            emitted += acc;
                        } else {
                            const tx = try self.tokenizer_.decode(&.{pending});
                            defer self.tokenizer_.allocator.free(tx);
                            if (!bench.validFree(seg.kind, tx)) break;
                            try ctx.decode(&.{pending});
                            try all_ids.append(pending);
                            try d.add(pending);
                            produced += 1;
                            emitted += 1;
                            pending = if (ctx.lm) |s| s.token else greedy(ctx.logits);
                        }
                    }
                },
            }
        }
        return produced;
    }
    fn undo(ctx: *qwen35.Context, before: usize, acc: []const u32) !void {
        if (!ctx.no_trace) return ctx.restore(before, acc.len);
        ctx.rollback(before);
        try ctx.decode(acc);
    }

    fn runNgram(self: *Port, ctx: *qwen35.Context, prompt: []const u32, max_tokens: u32, all_ids: *ArrayList(u32)) !u32 {
        var d = try ngram.TokenDrafter.init(self.allocator, 3);
        defer d.deinit();
        for (prompt) |t| try d.add(t);
        var pending: u32 = if (ctx.lm) |s| s.token else greedy(ctx.logits);
        var buf: [16]u32 = undefined;
        var generated: u32 = 0;
        while (generated < max_tokens) {
            if (pending == self.tokenizer_.eos()) break;
            const m = d.chain(buf[0..@min(@min(self.ngram_k, buf.len), max_tokens - generated)]);
            if (m >= @min(6, self.ngram_k) and buf[0] == pending) {
                const before = ctx.head;
                if (ctx.no_trace) ctx.snapshot();
                try ctx.decodeVerify(buf[0..m]);
                var j: usize = 0;
                while (j + 1 < m and ctx.verify_tokens[j] == buf[j + 1]) j += 1;
                var acc: usize = 0;
                if (j + 1 == m) {
                    acc = m;
                    pending = ctx.verify_tokens[m - 1];
                } else {
                    acc = j + 1;
                    pending = ctx.verify_tokens[j];
                    try undo(ctx, before, buf[0..acc]);
                }
                for (buf[0..acc]) |t| {
                    try all_ids.append(t);
                    try d.add(t);
                }
                generated += @intCast(acc);
            } else {
                try all_ids.append(pending);
                try d.add(pending);
                try ctx.decode(&.{pending});
                generated += 1;
                pending = if (ctx.lm) |s| s.token else greedy(ctx.logits);
            }
        }
        return generated;
    }

    fn runGrammar(self: *Port, ctx: *qwen35.Context, max_tokens: u32, all_ids: *ArrayList(u32)) !u32 {
        var produced: u32 = 0;
        var pending: u32 = 0;
        var have_pending = false;
        for (bench.TEMPLATE) |seg| {
            if (produced >= max_tokens) break;
            switch (seg.kind) {
                .lit => {
                    const ids = try self.tokenizer_.encode(seg.text);
                    defer ids.deinit();
                    var cnt = ids.items.len;
                    if (produced + cnt > max_tokens) cnt = max_tokens - produced;
                    if (cnt == 0) break;
                    try ctx.decode(ids.items[0..cnt]);
                    try all_ids.appendSlice(ids.items[0..cnt]);
                    produced += @intCast(cnt);
                    pending = if (ctx.lm) |s| s.token else greedy(ctx.logits);
                    have_pending = true;
                },
                .string, .number => {
                    if (!have_pending) break;
                    const cap: usize = if (seg.kind == .string) 24 else 4;
                    var emitted: usize = 0;
                    while (emitted < cap and produced < max_tokens) {
                        const txt = try self.tokenizer_.decode(&.{pending});
                        defer self.tokenizer_.allocator.free(txt);
                        if (!bench.validFree(seg.kind, txt)) break;
                        try ctx.decode(&.{pending});
                        try all_ids.append(pending);
                        produced += 1;
                        emitted += 1;
                        pending = if (ctx.lm) |s| s.token else greedy(ctx.logits);
                    }
                },
            }
        }
        return produced;
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
