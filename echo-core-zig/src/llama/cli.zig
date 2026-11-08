//! echo-core sohbet / tamamlama arayüzü (llama-cli benzeri).
//! Metin stdout'a akar; günlük ve istatistik stderr'e gider (pipe'lanabilir).
const std = @import("std");
const gguf = @import("../gguf/reader.zig");
const Bpe = @import("../tokenizer/bpe.zig").Bpe;
const Utf8Stream = @import("../tokenizer/bpe.zig").Utf8Stream;
const qwen35 = @import("qwen35.zig");
const ivf = @import("ivf.zig");
const ngram = @import("ngram.zig");
const sampler = @import("sampler.zig");

const ArrayList = std.array_list.Managed;
extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
const Alloc = std.mem.Allocator;

pub const yardim =
    \\Kullanım: echo-core -m MODEL.gguf [seçenekler]
    \\
    \\  -m, --model YOL        model dosyası (ilk .gguf argümanı da olur)
    \\  -p, --prompt METİN     tek soru sor, cevabı yaz, çık
    \\  -f, --file YOL         soruyu dosyadan oku
    \\                         (-p/-f yoksa ve stdin terminal değilse stdin okunur;
    \\                          terminalse sohbet modu açılır)
    \\  -i, --interactive      -p verilse bile sohbet moduna geç
    \\  -s, --sys METİN        sistem mesajı
    \\  -n, --n-predict N      en çok N token üret (-1: durma tokenına kadar) [-1]
    \\  -c, --ctx-size N       context uzunluğu [4096]
    \\  -t, --threads N        iş parçacığı [fiziksel çekirdek]
    \\      --temp X           sıcaklık; 0 = greedy [0.7]
    \\      --top-k N          [20]   (0 = kapalı)
    \\      --top-p X          [0.8]
    \\      --min-p X          [0.0]
    \\      --repeat-penalty X [1.0]
    \\      --presence-penalty X [0.0]
    \\      --seed N           [rastgele]
    \\      --stop METİN       bu metin gelince dur (birden çok verilebilir)
    \\      --think            düşünme modu (<think> ... </think>)
    \\      --raw              chat şablonu yok, düz metin tamamlama
    \\      --fast             hız profili: 8 katman atlanır, yaklaşık LM head, greedy
    \\                         (TG ~1000+ tok/s, kalite düşer)
    \\      --exact            llama.cpp ile aynı sayısal sonuç (R8 çekirdeği kapalı):
    \\                         prompt işleme ~2x yavaş, üretim ~%5 hızlı
    \\      --ngram            greedy'de n-gram taslak doğrulama (tekrarlı metinde hızlı,
    \\                         doğal metinde yavaş; --fast ile otomatik açık)
    \\      --prefill-fast     prompt işleme hız profili (prefill 2000 ayarı: FFN budama +
    \\                         12 katman atlama; kalite çok düşer)
    \\      --print-tokens     prompt'un token id'lerini yaz ve çık (şablon doğrulama)
    \\      --no-stats         sondaki hız satırını yazma
    \\      --server           OpenAI uyumlu HTTP sunucu (/v1/chat/completions, /v1/completions)
    \\      --host ADRES       [127.0.0.1]
    \\      --port N           [8080]
    \\  -h, --help             bu yardım
    \\
    \\Sohbet komutları: /reset  /sys METİN  /quit   (satır sonu \ ile çok satırlı giriş)
    \\Ctrl+C: üretimi keser; boş satırda Ctrl+C veya Ctrl+D: çıkış.
    \\
;

pub const Opts = struct {
    model: ?[]const u8 = null,
    prompt: ?[]const u8 = null,
    file: ?[]const u8 = null,
    sys: ?[]const u8 = null,
    interactive: bool = false,
    n_predict: i64 = -1,
    n_ctx: usize = 4096,
    threads: ?u32 = null,
    sp: sampler.Params = .{},
    seed_set: bool = false,
    stops: ArrayList([]const u8),
    think: bool = false,
    raw: bool = false,
    fast: bool = false,
    exact: bool = false,
    prefill_fast: bool = false,
    print_tokens: bool = false,
    /// greedy'de n-gram taslak doğrulama; tam LM head ile doğal metinde yavaşlatır (ölçüm: 37 → 23 tok/s), --fast açar
    ngram: bool = false,
    stats: bool = true,
    server: bool = false,
    host: []const u8 = "127.0.0.1",
    port: u16 = 8080,
};

fn need(args: []const []const u8, i: *usize) ![]const u8 {
    i.* += 1;
    if (i.* >= args.len) {
        std.debug.print("hata: {s} bir değer bekliyor\n", .{args[i.* - 1]});
        return error.InvalidArgs;
    }
    return args[i.*];
}

pub fn parse(args: []const []const u8, a: Alloc) !Opts {
    var o = Opts{ .stops = ArrayList([]const u8).init(a) };
    errdefer o.stops.deinit();
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const x = args[i];
        const eq = std.mem.eql;
        if (eq(u8, x, "-m") or eq(u8, x, "--model")) o.model = try need(args, &i)
        else if (eq(u8, x, "-p") or eq(u8, x, "--prompt")) o.prompt = try need(args, &i)
        else if (eq(u8, x, "-f") or eq(u8, x, "--file")) o.file = try need(args, &i)
        else if (eq(u8, x, "-s") or eq(u8, x, "--sys") or eq(u8, x, "--system")) o.sys = try need(args, &i)
        else if (eq(u8, x, "-i") or eq(u8, x, "--interactive") or eq(u8, x, "-cnv")) o.interactive = true
        else if (eq(u8, x, "-n") or eq(u8, x, "--n-predict") or eq(u8, x, "--max-tokens")) o.n_predict = try std.fmt.parseInt(i64, try need(args, &i), 10)
        else if (eq(u8, x, "-c") or eq(u8, x, "--ctx-size")) o.n_ctx = try std.fmt.parseInt(usize, try need(args, &i), 10)
        else if (eq(u8, x, "-t") or eq(u8, x, "--threads")) o.threads = try std.fmt.parseInt(u32, try need(args, &i), 10)
        else if (eq(u8, x, "--temp")) o.sp.temp = try std.fmt.parseFloat(f32, try need(args, &i))
        else if (eq(u8, x, "--top-k")) o.sp.top_k = try std.fmt.parseInt(u32, try need(args, &i), 10)
        else if (eq(u8, x, "--top-p")) o.sp.top_p = try std.fmt.parseFloat(f32, try need(args, &i))
        else if (eq(u8, x, "--min-p")) o.sp.min_p = try std.fmt.parseFloat(f32, try need(args, &i))
        else if (eq(u8, x, "--repeat-penalty")) o.sp.repeat_penalty = try std.fmt.parseFloat(f32, try need(args, &i))
        else if (eq(u8, x, "--presence-penalty")) o.sp.presence_penalty = try std.fmt.parseFloat(f32, try need(args, &i))
        else if (eq(u8, x, "--seed")) {
            o.sp.seed = try std.fmt.parseInt(u64, try need(args, &i), 10);
            o.seed_set = true;
        } else if (eq(u8, x, "--stop")) try o.stops.append(try need(args, &i))
        else if (eq(u8, x, "--think")) o.think = true
        else if (eq(u8, x, "--raw")) o.raw = true
        else if (eq(u8, x, "--fast")) o.fast = true
        else if (eq(u8, x, "--exact")) o.exact = true
        else if (eq(u8, x, "--prefill-fast")) o.prefill_fast = true
        else if (eq(u8, x, "--print-tokens")) o.print_tokens = true
        else if (eq(u8, x, "--no-ngram")) o.ngram = false
        else if (eq(u8, x, "--ngram")) o.ngram = true
        else if (eq(u8, x, "--no-stats")) o.stats = false
        else if (eq(u8, x, "--server") or eq(u8, x, "serve")) o.server = true
        else if (eq(u8, x, "--host")) o.host = try need(args, &i)
        else if (eq(u8, x, "--port")) o.port = try std.fmt.parseInt(u16, try need(args, &i), 10)
        else if (eq(u8, x, "-h") or eq(u8, x, "--help")) {
            std.debug.print("{s}", .{yardim});
            std.process.exit(0);
        } else if (o.model == null and std.mem.endsWith(u8, x, ".gguf")) o.model = x
        else {
            std.debug.print("hata: bilinmeyen seçenek '{s}' (yardım: --help)\n", .{x});
            return error.InvalidArgs;
        }
    }
    if (o.model == null) {
        std.debug.print("hata: model verilmedi (-m MODEL.gguf)\n\n{s}", .{yardim});
        return error.InvalidArgs;
    }
    if (o.n_ctx < 64) return error.InvalidArgs;
    return o;
}

// ---- Ctrl+C ----
pub var kesildi = std.atomic.Value(bool).init(false);
pub var uretiyor = std.atomic.Value(bool).init(false);
fn sigint(_: std.posix.SIG) callconv(.c) void {
    if (uretiyor.load(.acquire)) kesildi.store(true, .release) else std.process.exit(130);
}

/// Metin parçası alıcısı (sunucu SSE için); null ise `out`a yazılır.
pub const Sink = struct { ctx: *anyopaque, f: *const fn (*anyopaque, []const u8) anyerror!void };

pub const Session = struct {
    a: Alloc,
    io: std.Io,
    o: *const Opts,
    model: *qwen35.Model,
    ctx: *qwen35.Context,
    lm_ix: ?*ivf.Runtime,
    bpe: Bpe,
    smp: sampler.Sampler,
    /// context'teki tüm tokenlar (ceza + n-gram için)
    hist: ArrayList(u32),
    stop_ids: [4]u32 = @splat(std.math.maxInt(u32)),
    /// önceki asistan cevabı <|im_end|> ile kapatılmadı
    acik_cevap: bool = false,
    sys: ?[]const u8,
    out: *std.Io.Writer,
    /// tur başına: en çok token (-1 sınırsız) ve stop metinleri (sunucu istek başına değiştirir)
    limit: i64,
    stops: []const []const u8,
    sink: ?Sink = null,

    pub fn tokenize(s: *Session, text: []const u8, special: bool, dst: *ArrayList(u32)) !void {
        try s.bpe.encode(text, dst, special);
    }

    /// ChatML (Qwen3.5 şablonu): kullanıcı mesajını ve asistan girişini token dizisine çevirir.
    fn turn(s: *Session, user: []const u8, dst: *ArrayList(u32)) !void {
        if (s.o.raw) return s.tokenize(user, false, dst);
        if (s.acik_cevap) try s.tokenize("<|im_end|>\n", true, dst);
        if (s.hist.items.len == 0) if (s.sys) |sy| {
            try s.tokenize("<|im_start|>system\n", true, dst);
            try s.tokenize(sy, false, dst);
            try s.tokenize("<|im_end|>\n", true, dst);
        };
        try s.tokenize("<|im_start|>user\n", true, dst);
        try s.tokenize(user, false, dst); // kullanıcı metnindeki <|im_start|> özel token sayılmaz
        try s.tokenize("<|im_end|>\n<|im_start|>assistant\n", true, dst);
        try s.tokenize(if (s.o.think) "<think>\n" else "<think>\n\n</think>\n\n", true, dst);
    }

    fn feed(s: *Session, ids: []const u32) !void {
        if (s.ctx.head + ids.len > s.ctx.n_ctx) return error.ContextFull;
        try s.ctx.decode(ids);
        try s.hist.appendSlice(ids);
    }

    fn next(s: *Session) u32 {
        if (s.ctx.lm) |l| return l.token;
        return s.smp.sample(s.ctx.logits, s.hist.items);
    }

    fn isStop(s: *const Session, t: u32) bool {
        for (s.stop_ids) |x| if (x == t) return true;
        return s.bpe.isControl(t);
    }

    pub const Sonuc = struct { n_prompt: usize, n_gen: usize, t_prompt: u64, t_gen: u64, neden: []const u8, fed: bool = false };

    /// Bir tur: prompt'u işler, cevabı akıtır.
    fn respond(s: *Session, user: []const u8) !Sonuc {
        var ids = ArrayList(u32).init(s.a);
        defer ids.deinit();
        try s.turn(user, &ids);
        const r = try s.runIds(ids.items);
        // asistan turu KV'de hiç kapanmaz (durma tokenı yazılmaz); bir sonraki tur <|im_end|> ile kapatır
        if (r.fed) s.acik_cevap = !s.o.raw;
        return r;
    }

    /// Hazır token dizisini işler ve cevabı akıtır (sohbet ve sunucu ortak).
    pub fn runIds(s: *Session, ids_in: []const u32) !Sonuc {
        const ids = struct { items: []const u32 }{ .items = ids_in };
        var r = Sonuc{ .n_prompt = ids.items.len, .n_gen = 0, .t_prompt = 0, .t_gen = 0, .neden = "durma tokenı" };
        if (s.ctx.head + ids.items.len + 1 > s.ctx.n_ctx) {
            r.neden = "context dolu";
            return r;
        }
        kesildi.store(false, .release);
        uretiyor.store(true, .release);
        defer uretiyor.store(false, .release);

        const t0 = std.Io.Timestamp.now(s.io, .awake);
        try s.feed(ids.items);
        r.t_prompt = @intCast(t0.untilNow(s.io, .awake).toNanoseconds());
        r.fed = true;

        const t1 = std.Io.Timestamp.now(s.io, .awake);
        try s.generate(&r);
        r.t_gen = @intCast(t1.untilNow(s.io, .awake).toNanoseconds());
        return r;
    }

    fn generate(s: *Session, r: *Sonuc) !void {
        var yaz = Yazici{ .s = s };
        defer yaz.deinit();
        const greedy = s.ctx.lm != null or s.smp.p.temp <= 0;
        const limit: usize = if (s.limit < 0) std.math.maxInt(usize) else @intCast(s.limit);
        var d: ?ngram.TokenDrafter = null;
        defer if (d) |*x| x.deinit();
        if (greedy and s.o.ngram) {
            d = try ngram.TokenDrafter.init(s.a, 3);
            for (s.hist.items) |t| try d.?.add(t);
        }
        var pending = s.next();
        var buf: [64]u32 = undefined;
        while (r.n_gen < limit) {
            if (kesildi.load(.acquire)) {
                r.neden = "Ctrl+C";
                break;
            }
            if (s.isStop(pending)) {
                break;
            }
            if (s.ctx.head + 1 >= s.ctx.n_ctx) {
                r.neden = "context dolu";
                break;
            }
            // greedy: n-gram taslağı tek forward'da doğrulanır (kayıpsız; kabul edilen tokenlar aynı)
            if (d) |*dr| {
                const room = @min(buf.len, limit - r.n_gen, s.ctx.n_ctx - s.ctx.head - 1);
                const m = dr.chain(buf[0..room]);
                if (m >= 4 and buf[0] == pending and s.ctx.trace_k >= m) {
                    const before = s.ctx.head;
                    s.ctx.snapshot();
                    try s.ctx.decodeVerify(buf[0..m]);
                    var j: usize = 0;
                    while (j + 1 < m and s.ctx.verify_tokens[j] == buf[j + 1]) j += 1;
                    var acc = j + 1;
                    // kabul edilenlerden durma tokenına kadar olanı al
                    var stop_at: ?usize = null;
                    for (buf[0..acc], 0..) |t, q| if (s.isStop(t)) {
                        stop_at = q;
                        break;
                    };
                    if (stop_at) |q| acc = q;
                    const nxt = s.ctx.verify_tokens[acc -| 1];
                    if (acc < m or stop_at != null) {
                        s.ctx.rollback(before);
                        if (acc > 0) try s.ctx.decode(buf[0..acc]);
                    }
                    for (buf[0..acc]) |t| {
                        try s.hist.append(t);
                        try dr.add(t);
                        if (try yaz.add(t)) {
                            r.neden = "stop metni";
                            r.n_gen += 1;
                            return;
                        }
                    }
                    r.n_gen += acc;
                    if (stop_at) |q| {
                        pending = buf[q];
                        continue;
                    }
                    pending = if (acc == m) s.ctx.verify_tokens[m - 1] else nxt;
                    continue;
                }
            }
            if (try yaz.add(pending)) {
                r.neden = "stop metni";
                r.n_gen += 1;
                try s.feed(&.{pending});
                return;
            }
            try s.feed(&.{pending});
            if (d) |*dr| try dr.add(pending);
            r.n_gen += 1;
            pending = s.next();
        }
        if (r.n_gen >= limit) r.neden = "-n sınırı";
    }
};

/// Token baytlarını UTF-8 güvenli akıtır; --stop metinlerini yakalar (eşleşme olasılığı olan kuyruğu bekletir).
const Yazici = struct {
    s: *Session,
    u: Utf8Stream = .{},
    bekleyen: ArrayList(u8) = undefined,
    init_ok: bool = false,

    fn deinit(y: *Yazici) void {
        if (!y.init_ok) return;
        y.y_flush(y.bekleyen.items.len) catch {};
        y.bekleyen.deinit();
    }
    fn y_flush(y: *Yazici, n: usize) !void {
        if (y.s.sink) |k| {
            if (n > 0) try k.f(k.ctx, y.bekleyen.items[0..n]);
        } else {
            try y.s.out.writeAll(y.bekleyen.items[0..n]);
            y.s.out.flush() catch {};
        }
        std.mem.copyForwards(u8, y.bekleyen.items, y.bekleyen.items[n..]);
        y.bekleyen.shrinkRetainingCapacity(y.bekleyen.items.len - n);
    }
    /// true: stop metni bulundu, üretim durmalı.
    fn add(y: *Yazici, t: u32) !bool {
        if (!y.init_ok) {
            y.bekleyen = ArrayList(u8).init(y.s.a);
            y.init_ok = true;
        }
        try y.u.feed(y.s.bpe.piece(t), &y.bekleyen);
        var maxlen: usize = 0;
        for (y.s.stops) |st| {
            if (std.mem.indexOf(u8, y.bekleyen.items, st)) |at| {
                y.bekleyen.shrinkRetainingCapacity(at);
                try y.y_flush(at);
                return true;
            }
            maxlen = @max(maxlen, st.len);
        }
        // stop metninin başlangıcı olabilecek kuyruğu tut, gerisini bas
        var keep: usize = @min(y.bekleyen.items.len, maxlen -| 1);
        while (keep < y.bekleyen.items.len and keep > 0 and (y.bekleyen.items[y.bekleyen.items.len - keep] & 0xC0) == 0x80) keep += 1;
        try y.y_flush(y.bekleyen.items.len - keep);
        return false;
    }
};

pub fn run(args: []const []const u8, a: Alloc, io: std.Io) !void {
    var o = parse(args, a) catch std.process.exit(2);
    defer o.stops.deinit();
    const path = o.model.?;
    if (!qwen35.supported) {
        std.debug.print("hata: bu CPU desteklenmiyor (AVX2 + FMA gerekli)\n", .{});
        std.process.exit(1);
    }
    if (!qwen35.Model.isQwen35(path, a)) {
        std.debug.print("hata: '{s}' açılamadı ya da desteklenmeyen mimari (şu an yalnız qwen35)\n", .{path});
        std.process.exit(1);
    }
    if (!o.seed_set) o.sp.seed = @truncate(@as(u96, @bitCast(std.Io.Timestamp.now(io, .real).toNanoseconds())));

    var reader = try gguf.Reader.openWithAllocator(path, a);
    var bpe = try Bpe.initFromReader(&reader, a);
    reader.deinit();
    defer bpe.deinit();

    if (o.exact) _ = setenv("ECHO_NO_R8", "1", 1); // Model.load okur
    if (o.prefill_fast) {
        // README "Prefill 2000 ayarı"; aktivasyon dosyası yoksa ağırlık normuyla budanır
        _ = setenv("ECHO_FFN_KEEP", "1920", 0);
        _ = setenv("ECHO_SKIP", "20,7,18,11,12,8,5,13,10,9,6,17", 0);
        const act = "benchmarks/ffn-act-qwen35-0.8b-q8.bin";
        if (std.Io.Dir.cwd().access(io, act, .{})) |_| {
            _ = setenv("ECHO_FFN_ACT", act, 0);
        } else |_| std.debug.print("uyarı: {s} yok (echo-core-zig dizininden çalıştırın)\n", .{act});
    }
    const model = try qwen35.Model.load(path, a);
    defer model.deinit();
    // varsayılan kayıpsız: katman atlama yalnız --fast ile (ECHO_SKIP elle verilmişse ona uyulur)
    if (!o.fast and std.c.getenv("ECHO_SKIP") == null) model.skip = 0;
    const cpus = std.Thread.getCpuCount() catch 2;
    const n_threads: u32 = o.threads orelse @intCast(@max(cpus / 2, 1));
    const lm_ix: ?*ivf.Runtime = if (o.fast) ivf.open(path, model.output, ivf.default_l, n_threads, a, io) catch null else null;
    defer if (lm_ix) |ix| ix.deinit(a);

    const ctx = try qwen35.Context.init(model, o.n_ctx, n_threads, a);
    defer ctx.deinit();
    if (lm_ix) |ix| ctx.lm = try ivf.Search.init(ix, ivf.default_nprobe, n_threads, a);
    defer if (ctx.lm) |l| l.deinit(a);
    if (o.fast) {
        o.sp.temp = 0;
        o.ngram = true;
    }
    if (o.ngram and (o.fast or o.sp.temp <= 0)) {
        try ctx.enableTrace(64);
        ctx.no_trace = true; // geri alma snapshot ile (H13)
    }

    var obuf: [4096]u8 = undefined;
    var ow = std.Io.File.stdout().writer(io, &obuf);
    var s = Session{
        .a = a,
        .io = io,
        .o = &o,
        .model = model,
        .ctx = ctx,
        .lm_ix = lm_ix,
        .bpe = bpe,
        .smp = sampler.Sampler.init(o.sp),
        .hist = ArrayList(u32).init(a),
        .sys = o.sys,
        .out = &ow.interface,
        .limit = o.n_predict,
        .stops = o.stops.items,
    };
    defer s.hist.deinit();
    s.stop_ids[0] = bpe.specialId("<|im_end|>") orelse std.math.maxInt(u32);
    s.stop_ids[1] = bpe.specialId("<|endoftext|>") orelse std.math.maxInt(u32);

    var act = std.posix.Sigaction{ .handler = .{ .handler = sigint }, .mask = std.posix.sigemptyset(), .flags = 0 };
    std.posix.sigaction(.INT, &act, null);

    if (o.server) return @import("server.zig").serve(&s, o.host, o.port, std.fs.path.basename(path));

    // tek seferlik soru
    var tek: ?[]u8 = null;
    defer if (tek) |t| a.free(t);
    if (o.file) |f| tek = try std.Io.Dir.cwd().readFileAlloc(io, f, a, .unlimited);
    const stdin_tty = std.Io.File.stdin().isTty(io) catch true;
    if (o.prompt == null and tek == null and !stdin_tty and !o.interactive) {
        var ib: [4096]u8 = undefined;
        var ir = std.Io.File.stdin().readerStreaming(io, &ib);
        tek = try ir.interface.allocRemaining(a, .unlimited);
    }
    const soru: ?[]const u8 = o.prompt orelse tek;
    if (o.print_tokens) {
        var ids = ArrayList(u32).init(a);
        defer ids.deinit();
        try s.turn(soru orelse "", &ids);
        for (ids.items, 0..) |t, i| std.debug.print("{s}{d}", .{ if (i == 0) "" else ",", t });
        std.debug.print("\n", .{});
        return;
    }
    if (soru) |q| {
        try cevapla(&s, q);
        if (!o.interactive) return;
    }
    try sohbet(&s);
}

fn cevapla(s: *Session, q: []const u8) !void {
    const r = s.respond(q) catch |e| switch (e) {
        error.ContextFull => {
            std.debug.print("\n[context doldu ({d} token). /reset ile temizle ya da -c ile büyüt]\n", .{s.ctx.n_ctx});
            return;
        },
        else => return e,
    };
    try s.out.writeAll("\n");
    try s.out.flush();
    if (std.mem.eql(u8, r.neden, "context dolu"))
        std.debug.print("[context doldu ({d}/{d} token). /reset ile temizle ya da -c ile büyüt]\n", .{ s.ctx.head, s.ctx.n_ctx });
    if (s.o.stats) {
        const tp = @as(f64, @floatFromInt(r.n_prompt)) / (@as(f64, @floatFromInt(@max(r.t_prompt, 1))) / 1e9);
        const tg = @as(f64, @floatFromInt(r.n_gen)) / (@as(f64, @floatFromInt(@max(r.t_gen, 1))) / 1e9);
        std.debug.print("\x1b[2m[prompt {d} tok, {d:.0} tok/s | üretim {d} tok, {d:.1} tok/s | bitiş: {s} | ctx {d}/{d}]\x1b[0m\n", .{ r.n_prompt, tp, r.n_gen, tg, r.neden, s.ctx.head, s.ctx.n_ctx });
    }
}

fn sohbet(s: *Session) !void {
    std.debug.print("echo-core sohbet — /reset, /sys METİN, /quit (Ctrl+D çıkış)\n", .{});
    var ib: [1 << 16]u8 = undefined;
    var ir = std.Io.File.stdin().readerStreaming(s.io, &ib);
    var msg = ArrayList(u8).init(s.a);
    defer msg.deinit();
    while (true) {
        std.debug.print("{s}", .{if (msg.items.len == 0) "\x1b[1m> \x1b[0m" else ". "});
        const line = ir.interface.takeDelimiter('\n') catch |e| switch (e) {
            error.StreamTooLong => {
                std.debug.print("[satır çok uzun, -f ile dosyadan verin]\n", .{});
                continue;
            },
            else => return e,
        } orelse break;
        const l = std.mem.trimEnd(u8, line, "\r");
        if (std.mem.endsWith(u8, l, "\\")) {
            try msg.appendSlice(l[0 .. l.len - 1]);
            try msg.append('\n');
            continue;
        }
        try msg.appendSlice(l);
        defer msg.clearRetainingCapacity();
        const t = std.mem.trim(u8, msg.items, " \t");
        if (t.len == 0) continue;
        if (std.mem.eql(u8, t, "/quit") or std.mem.eql(u8, t, "/exit")) break;
        if (std.mem.eql(u8, t, "/reset")) {
            s.ctx.clear();
            s.hist.clearRetainingCapacity();
            s.acik_cevap = false;
            std.debug.print("[geçmiş silindi]\n", .{});
            continue;
        }
        if (std.mem.startsWith(u8, t, "/sys")) {
            s.sys = try s.a.dupe(u8, std.mem.trim(u8, t[4..], " "));
            s.ctx.clear();
            s.hist.clearRetainingCapacity();
            s.acik_cevap = false;
            std.debug.print("[sistem mesajı ayarlandı, geçmiş silindi]\n", .{});
            continue;
        }
        try cevapla(s, t);
    }
}
