//! Port: llama.cpp tools/llama-bench/llama-bench.cpp (test_prompt, test_gen, warmup, tekrar).
//! Aynı ölçüm tanımı: her test için taze context, warmup, tekrar başına llama_memory_clear.
const std = @import("std");
const qwen35 = @import("qwen35.zig");
const ivf = @import("ivf.zig");
const ngram = @import("ngram.zig");
const tokenizer = @import("../tokenizer/tokenizer.zig");

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

fn testGen(ctx: *qwen35.Context, rng: std.Random, n_vocab: usize, n_gen: usize, greedy: bool) !void {
    var token: [1]u32 = .{@intCast(rng.uintLessThan(usize, n_vocab))};
    for (0..n_gen) |_| {
        try ctx.decode(&token);
        if (!greedy) {
            token[0] = @intCast(rng.uintLessThan(usize, n_vocab));
        } else if (ctx.lm) |s| {
            token[0] = s.token;
        } else {
            token[0] = @intCast(std.mem.indexOfMax(f32, ctx.logits));
        }
    }
}
/// Toplu TG ölçümü: her decode çağrısına batch token verir.
/// Gerçek üretim değil; ağırlık okumasını batch'e yaymanın tavanını ölçer.
fn testGenBatch(ctx: *qwen35.Context, rng: std.Random, n_vocab: usize, n_gen: usize, batch: usize, tokens: []u32) !void {
    var done: usize = 0;
    while (done < n_gen) {
        const n = @min(@max(batch, 1), n_gen - done);
        for (tokens[0..n]) |*t| t.* = @intCast(rng.uintLessThan(usize, n_vocab));
        try ctx.decode(tokens[0..n]);
        done += n;
    }
}

pub const Options = struct {
    n_threads: u32 = 6,
    reps: usize = 5,
    prompts: []const usize = &.{ 128, 512 },
    gens: []const usize = &.{ 32, 128 },
    /// Varsayılan: IVF LM head (yaklaşık, ayrı etiketlenir). 0: tam LM head.
    lm_clusters: usize = ivf.default_l,
    lm_nprobe: usize = ivf.default_nprobe,
    greedy: bool = false,
    /// >1: TG'yi batch'li ölç (pipeline/batch tavanı); gerçek üretim değil.
    gen_batch: usize = 1,
    /// >0: H01 n-gram taslağı + doğrulama, k = taslak uzunluğu.
    ngram_k: usize = 128,
    /// true: H06 grammar fast-forward (JSON iskeleti zorunlu blok).
    grammar: bool = true,
    /// H11 (P6): taslak zinciri tek forward'da ödenir, model farklı derse fark sayılır, geri alma yok.
    p6: bool = false,
    /// H13: iz yerine tur başı snapshot; tutmazsa geri dön + yeniden işle (kayıpsız).
    snap: bool = true,
    /// H17: doğrulama turuna girmek için en kısa taslak (H01: 6).
    min_draft: usize = 6,
    /// H18 (P6 "önceden markete gitmişsindir"): taslak hafızası tekrarlar arasında silinmez.
    memory: bool = true, // H29: varsayılan; --memory 0 kapatır
};


pub const Stats = struct {
    rounds: usize = 0,
    drafted: usize = 0,
    accepted: usize = 0,
    /// H11: modelin kendi seçimiyle tutmayan taslak pozisyonu ("kafaya yazılan +10 TL").
    miss: usize = 0,
    /// Taslaksız tek token adımı.
    single: usize = 0,
    forced_blocks: usize = 0,
    forced_tokens: usize = 0,
};

fn nextToken(ctx: *qwen35.Context) u32 {
    if (ctx.lm) |s| return s.token;
    return @intCast(std.mem.indexOfMax(f32, ctx.logits));
}

/// H01: n-gram taslağı + tek zincir doğrulama. Token akışı modelin kendi ürettiği
/// tokenlar olduğu için pencere birebir hizalı.
/// Tutmayan turda durumu geri alır: H13 (no_trace) snapshot'a dönüp kabul edileni yeniden işler, H01 iz kullanır.
fn undo(ctx: *qwen35.Context, before: usize, acc: []const u32) !void {
    if (!ctx.no_trace) return ctx.restore(before, acc.len);
    ctx.rollback(before);
    try ctx.decode(acc);
}

/// H11: doğrulama çıktısını sayar, hepsini alır; sonraki token modelin son tahmini.
fn payAll(ctx: *qwen35.Context, buf: []const u32, st: *Stats) u32 {
    const m = buf.len;
    for (0..m - 1) |i| {
        if (ctx.verify_tokens[i] != buf[i + 1]) st.miss += 1;
    }
    st.accepted += m;
    return ctx.verify_tokens[m - 1];
}

fn testGenNgram(ctx: *qwen35.Context, rng: std.Random, n_vocab: usize, n_gen: usize, k: usize, p6: bool, min_draft: usize, mem: ?*ngram.TokenDrafter, st: *Stats, a: std.mem.Allocator) !usize {
    var local = try ngram.TokenDrafter.init(a, 3);
    defer local.deinit();
    const d = mem orelse &local;
    try d.restart();
    var buf: [128]u32 = undefined;
    var pending: u32 = @intCast(rng.uintLessThan(usize, n_vocab));
    try ctx.decode(&.{pending});
    try d.add(pending);
    var produced: usize = 1;
    pending = nextToken(ctx);
    var lim: usize = @min(k, buf.len); // H25: tam kabulde ×2, rette ½ (en az 8)
    while (produced < n_gen) {
        const m = d.chain(buf[0..@min(lim, n_gen - produced)]); // istenen uzunluğu aşma
        if (m >= @max(@min(min_draft, k), 1) and buf[0] == pending) {
            st.rounds += 1;
            st.drafted += m;
            const before = ctx.head;
            if (ctx.no_trace and !p6) ctx.snapshot();
            try ctx.decodeVerify(buf[0..m]);
            if (p6) {
                pending = payAll(ctx, buf[0..m], st);
                for (buf[0..m]) |x| try d.add(x);
                produced += m;
                continue;
            }
            var j: usize = 0;
            while (j + 1 < m and ctx.verify_tokens[j] == buf[j + 1]) j += 1;
            var acc: usize = 0;
            if (j + 1 == m) {
                acc = m;
                pending = ctx.verify_tokens[m - 1];
                if (m == lim) lim = @min(lim * 2, k, buf.len);
            } else {
                acc = j + 1;
                pending = ctx.verify_tokens[j];
                try undo(ctx, before, buf[0..acc]);
                lim = @max(lim / 2, @min(8, k));
            }
            st.accepted += acc;
            for (buf[0..acc]) |x| try d.add(x);
            produced += acc;
        } else {
            st.single += 1;
            try ctx.decode(&.{pending});
            try d.add(pending);
            produced += 1;
            pending = nextToken(ctx);
        }
    }
    return produced;
}

pub const SegKind = enum { lit, string, number };
pub const Seg = struct { kind: SegKind, text: []const u8 };
pub const TEMPLATE = [_]Seg{
    .{ .kind = .lit, .text = "{\"name\": \"" },
    .{ .kind = .string, .text = "" },
    .{ .kind = .lit, .text = "\", \"age\": " },
    .{ .kind = .number, .text = "" },
    .{ .kind = .lit, .text = ", \"city\": \"" },
    .{ .kind = .string, .text = "" },
    .{ .kind = .lit, .text = "\"}" },
};
pub const TEMPLATE_BIG = [_]Seg{
    .{ .kind = .lit, .text = "{\"name\": \"" },
    .{ .kind = .string, .text = "" },
    .{ .kind = .lit, .text = "\", \"age\": " },
    .{ .kind = .number, .text = "" },
    .{ .kind = .lit, .text = ", \"city\": \"" },
    .{ .kind = .string, .text = "" },
    .{ .kind = .lit, .text = "\", \"country\": \"" },
    .{ .kind = .string, .text = "" },
    .{ .kind = .lit, .text = "\", \"email\": \"" },
    .{ .kind = .string, .text = "" },
    .{ .kind = .lit, .text = "\", \"phone\": \"" },
    .{ .kind = .string, .text = "" },
    .{ .kind = .lit, .text = "\", \"job\": \"" },
    .{ .kind = .string, .text = "" },
    .{ .kind = .lit, .text = "\", \"company\": \"" },
    .{ .kind = .string, .text = "" },
    .{ .kind = .lit, .text = "\", \"street\": \"" },
    .{ .kind = .string, .text = "" },
    .{ .kind = .lit, .text = "\", \"zip\": " },
    .{ .kind = .number, .text = "" },
    .{ .kind = .lit, .text = ", \"notes\": \"" },
    .{ .kind = .string, .text = "" },
    .{ .kind = .lit, .text = "\"}" },
};

pub fn validFree(kind: SegKind, s: []const u8) bool {
    if (s.len == 0) return false;
    return switch (kind) {
        .lit => true,
        .string => blk: {
            for (s) |c| if (c == '"' or c == '\\' or c == '\n' or c == '\r') break :blk false;
            break :blk true;
        },
        .number => blk: {
            for (s) |c| {
                const ok = std.ascii.isDigit(c) or c == '.' or c == '-' or c == '+' or c == 'e' or c == 'E' or c == ' ';
                if (!ok) break :blk false;
            }
            break :blk true;
        },
    };
}

/// H06: zorunlu literal blokları tek forward'da ekler, serbest alanlarda token maskeler.
fn testGenGrammar(ctx: *qwen35.Context, tk: *const tokenizer.SimpleTokenizer, n_gen: usize, st: *Stats) !usize {
    var produced: usize = 0;
    var pending: u32 = 0;
    var have_pending = false;
    for (TEMPLATE) |seg| {
        if (produced >= n_gen) break;
        switch (seg.kind) {
            .lit => {
                const ids = try tk.encode(seg.text);
                defer ids.deinit();
                var cnt = ids.items.len;
                if (produced + cnt > n_gen) cnt = n_gen - produced;
                if (cnt == 0) break;
                try ctx.decode(ids.items[0..cnt]);
                produced += cnt;
                st.forced_blocks += 1;
                st.forced_tokens += cnt;
                pending = nextToken(ctx);
                have_pending = true;
            },
            .string, .number => {
                if (!have_pending) break;
                const cap: usize = if (seg.kind == .string) 24 else 4;
                var emitted: usize = 0;
                while (emitted < cap and produced < n_gen) {
                    const txt = try tk.decode(&.{pending});
                    defer tk.allocator.free(txt);
                    if (!validFree(seg.kind, txt)) break;
                    try ctx.decode(&.{pending});
                    produced += 1;
                    emitted += 1;
                    pending = nextToken(ctx);
                }
            },
        }
    }
    // şablon bittiyse n_gen'e kadar normal greedy (bench eşitliği)
    while (produced < n_gen) {
        if (!have_pending) break;
        try ctx.decode(&.{pending});
        produced += 1;
        pending = nextToken(ctx);
    }
    return produced;
}

fn runGen(ctx: *qwen35.Context, opt: Options, rng: std.Random, n_vocab: usize, n_gen: usize, tk: ?*const tokenizer.SimpleTokenizer, tokens: []u32, mem: ?*ngram.TokenDrafter, st: *Stats, a: std.mem.Allocator) !usize {
    if (opt.grammar) return testGenGrammarSpec(ctx, tk orelse return error.NoTokenizer, n_gen, st, a, opt.ngram_k, opt.p6);
    if (opt.ngram_k > 0) return testGenNgram(ctx, rng, n_vocab, n_gen, opt.ngram_k, opt.p6, opt.min_draft, mem, st, a);
    if (opt.gen_batch > 1) {
        try testGenBatch(ctx, rng, n_vocab, n_gen, opt.gen_batch, tokens);
        return n_gen;
    }
    try testGen(ctx, rng, n_vocab, n_gen, opt.greedy);
    return n_gen;
}

/// H06 + (opsiyonel) H01: zorunlu bloklar tek forward, serbest alanlarda n-gram taslağı.
fn testGenGrammarSpec(ctx: *qwen35.Context, tk: *const tokenizer.SimpleTokenizer, n_gen: usize, st: *Stats, a: std.mem.Allocator, ngram_k: usize, p6: bool) !usize {
    var d = try ngram.TokenDrafter.init(a, 3);
    defer d.deinit();
    var buf: [128]u32 = undefined;
    var produced: usize = 0;
    var pending: u32 = 0;
    var have = false;
    for (TEMPLATE_BIG) |seg| {
        if (produced >= n_gen) break;
        switch (seg.kind) {
            .lit => {
                const ids = try tk.encode(seg.text);
                defer ids.deinit();
                var cnt = ids.items.len;
                if (produced + cnt > n_gen) cnt = n_gen - produced;
                if (cnt == 0) break;
                try ctx.decode(ids.items[0..cnt]);
                for (ids.items[0..cnt]) |t| try d.add(t);
                produced += cnt;
                st.forced_blocks += 1;
                st.forced_tokens += cnt;
                pending = nextToken(ctx);
                have = true;
            },
            .string, .number => {
                if (!have) break;
                const cap: usize = if (seg.kind == .string) 24 else 4;
                var emitted: usize = 0;
                while (emitted < cap and produced < n_gen) {
                    const maxm = @min(@min(ngram_k, cap - emitted), n_gen - produced);
                    var m: usize = 0;
                    if (maxm >= 2) {
                        const raw = d.chain(buf[0..@min(maxm, buf.len)]);
                        while (m < raw) : (m += 1) {
                            const tx = try tk.decode(&.{buf[m]});
                            defer tk.allocator.free(tx);
                            if (!validFree(seg.kind, tx)) break;
                        }
                    }
                    if (m >= 2 and m >= @min(4, maxm) and buf[0] == pending) {
                        st.rounds += 1;
                        st.drafted += m;
                        const before = ctx.head;
                        if (ctx.no_trace and !p6) ctx.snapshot();
                        try ctx.decodeVerify(buf[0..m]);
                        if (p6) {
                            pending = payAll(ctx, buf[0..m], st);
                            for (buf[0..m]) |t| try d.add(t);
                            produced += m;
                            emitted += m;
                            continue;
                        }
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
                        st.accepted += acc;
                        for (buf[0..acc]) |t| try d.add(t);
                        produced += acc;
                        emitted += acc;
                    } else {
                        const tx = try tk.decode(&.{pending});
                        defer tk.allocator.free(tx);
                        if (!validFree(seg.kind, tx)) break;
                        try ctx.decode(&.{pending});
                        try d.add(pending);
                        produced += 1;
                        emitted += 1;
                        pending = nextToken(ctx);
                    }
                }
            },
        }
    }
    return produced;
}
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

    const lm_ix = if (opt.lm_clusters > 0) try ivf.open(model_path, model.output, opt.lm_clusters, opt.n_threads, allocator, io) else null;
    defer if (lm_ix) |ix| ix.deinit(allocator);

    var tk_store: tokenizer.SimpleTokenizer = undefined;
    var tk_ptr: ?*const tokenizer.SimpleTokenizer = null;
    if (opt.grammar) {
        var reader = try @import("../gguf/reader.zig").Reader.openWithAllocator(model_path, allocator);
        defer reader.deinit();
        tk_store = try tokenizer.SimpleTokenizer.initFromReader(&reader, allocator);
        tk_ptr = &tk_store;
    }
    defer if (opt.grammar) tk_store.deinit();

    for (tests.items) |t| {
        const ctx = try qwen35.Context.init(model, t.n_prompt + t.n_gen, opt.n_threads, allocator);
        defer ctx.deinit();
        if (lm_ix) |ix| ctx.lm = try ivf.Search.init(ix, opt.lm_nprobe, opt.n_threads, allocator);
        defer if (ctx.lm) |s| s.deinit(allocator);
        if (opt.ngram_k > 0) {
            ctx.enableTrace(@max(opt.ngram_k, 2)) catch |e| {
                std.debug.print("trace kurulamadi: {}\n", .{e});
                return e;
            };
        }
        ctx.no_trace = opt.p6 or opt.snap;
        var st = Stats{};
        var mem_store = try ngram.TokenDrafter.init(allocator, 3);
        defer mem_store.deinit();
        const mem: ?*ngram.TokenDrafter = if (opt.memory) &mem_store else null;

        // warmup
        ctx.clear();
        if (t.n_prompt > 0) try testPrompt(ctx, rng, model.n_vocab, t.n_prompt, tokens);
        if (t.n_gen > 0) _ = try runGen(ctx, opt, rng, model.n_vocab, @min(t.n_gen, 8), tk_ptr, tokens, mem, &st, allocator);

        const samples = try allocator.alloc(f64, opt.reps);
        defer allocator.free(samples);
        for (samples) |*s| {
            // İçerik bağımlı modlarda tekrarlar aynı metni ölçsün (deterministik).
            if (opt.greedy or opt.ngram_k > 0 or opt.grammar) prng = std.Random.DefaultPrng.init(1234);
            const rng_r = prng.random();
            ctx.clear();
            const t0 = std.Io.Timestamp.now(io, .awake);
            if (t.n_prompt > 0) try testPrompt(ctx, rng_r, model.n_vocab, t.n_prompt, tokens);
            var produced: usize = 0;
            if (t.n_gen > 0) produced = try runGen(ctx, opt, rng_r, model.n_vocab, t.n_gen, tk_ptr, tokens, mem, &st, allocator);
            const ns: f64 = @floatFromInt(t0.untilNow(io, .awake).toNanoseconds());
            s.* = @as(f64, @floatFromInt(t.n_prompt + produced)) * 1e9 / ns;
        }
        var sum: f64 = 0;
        for (samples) |s| sum += s;
        const alpha: f64 = if (st.drafted > 0) @as(f64, @floatFromInt(st.accepted)) / @as(f64, @floatFromInt(st.drafted)) else 0;
        const per_round: f64 = if (st.rounds > 0) @as(f64, @floatFromInt(st.accepted)) / @as(f64, @floatFromInt(st.rounds)) else 0;
        try out.print("{{\"n_prompt\":{d},\"n_gen\":{d},\"gen_batch\":{d},\"n_threads\":{d},\"lm_clusters\":{d},\"lm_nprobe\":{d},\"greedy\":{},\"ngram_k\":{d},\"grammar\":{},\"alpha\":{d:.3},\"accept_per_round\":{d:.2},\"drafted\":{d},\"accepted\":{d},\"forced_tokens\":{d},\"forced_blocks\":{d},\"p6\":{},\"miss\":{d},\"single\":{d},\"avg_ts\":{d:.3},\"samples_ts\":[", .{ t.n_prompt, t.n_gen, opt.gen_batch, opt.n_threads, opt.lm_clusters, if (opt.lm_clusters > 0) opt.lm_nprobe else 0, opt.greedy, opt.ngram_k, opt.grammar, alpha, per_round, st.drafted, st.accepted, st.forced_tokens, st.forced_blocks, opt.p6, st.miss, st.single, sum / @as(f64, @floatFromInt(samples.len)) });
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
