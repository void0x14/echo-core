//! echo-core sohbet / tamamlama arayüzü (llama-cli benzeri).
//! Metin stdout'a akar; günlük ve istatistik stderr'e gider (pipe'lanabilir).
const std = @import("std");
const gguf = @import("../gguf/reader.zig");
const Bpe = @import("../tokenizer/bpe.zig").Bpe;
const Utf8Stream = @import("../tokenizer/bpe.zig").Utf8Stream;
const qwen35 = @import("qwen35.zig");
const ivf = @import("ivf.zig");
const ngram = @import("ngram.zig");
const rest = @import("rest.zig");
const esp_mod = @import("esp.zig");
const gorev = @import("gorev.zig");
const sampler = @import("sampler.zig");

const ArrayList = std.array_list.Managed;
extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;
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
    \\  -ç N                   N çekirdek: işlem 0..N-1 çekirdeklerine bağlanır, N iş parçacığı
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
    \\      --ngram            greedy'de n-gram taslak doğrulama (tekrarlı metinde hızlı,
    \\                         doğal metinde yavaş; varsayılan açık)
    \\      --rest DİZİN       H02: DİZİN altındaki .zig/.md dosyalarından retrieval taslağı (kayıpsız doğrulama)
    \\      --esp              H03 ESP: eğitimsiz çoklu token tahmini (maske tokenlar), n-gram yerine; greedy, kayıpsız
    \\                         (ECHO_ESP_K maske sayısı [1], ECHO_ESP_L100 maske kayma oranı x100 [10])
    \\      --skip LİSTE       atlanacak katmanlar, ör. 20,7,18 (ECHO_SKIP)
    \\      --ffn-keep N       FFN'de tutulacak nöron sayısı (ECHO_FFN_KEEP)
    \\      --ffn-act DOSYA    FFN budama için aktivasyon dosyası (ECHO_FFN_ACT)
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
    /// tek çalışma düzeni: tam model (atlama yok, FFN tam) + yaklaşık LM head + n-gram, greedy
    fast: bool = true,
    print_tokens: bool = false,
    /// greedy'de n-gram taslak doğrulama; tam LM head ile doğal metinde yavaşlatır (ölçüm: 37 → 23 tok/s), --fast açar
    ngram: bool = false,
    /// --no-ngram: n-gram taslağı hiç kullanılmaz (--fast bile açmaz)
    no_ng: bool = false,
    /// H02: depo kökü (--rest DİZİN); .zig ve .md dosyalarından retrieval taslağı
    rest: ?[]const u8 = null,
    /// H03: ESP taslağı (n-gram yerine)
    esp: bool = false,
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
        else if (eq(u8, x, "-ç")) {
            const n = try std.fmt.parseInt(u32, try need(args, &i), 10);
            if (n == 0 or n > 1024) return error.InvalidArgs;
            var mask: [16]u64 = @splat(0);
            for (0..n) |c| mask[c / 64] |= @as(u64, 1) << @intCast(c % 64);
            if (sched_setaffinity(0, @sizeOf(@TypeOf(mask)), &mask) != 0) std.debug.print("uyarı: çekirdek bağlama başarısız\n", .{});
            o.threads = n;
        }
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
        else if (eq(u8, x, "--skip")) _ = setenv("ECHO_SKIP", try a.dupeSentinel(u8, try need(args, &i), 0), 1)
        else if (eq(u8, x, "--ffn-keep")) _ = setenv("ECHO_FFN_KEEP", try a.dupeSentinel(u8, try need(args, &i), 0), 1)
        else if (eq(u8, x, "--ffn-act")) _ = setenv("ECHO_FFN_ACT", try a.dupeSentinel(u8, try need(args, &i), 0), 1)
        else if (eq(u8, x, "--print-tokens")) o.print_tokens = true
        else if (eq(u8, x, "--no-ngram")) {
            o.ngram = false;
            o.no_ng = true;
        } else if (eq(u8, x, "--rest")) o.rest = try need(args, &i)
        else if (eq(u8, x, "--ngram")) o.ngram = true
        else if (eq(u8, x, "--esp")) o.esp = true
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

/// Motorla oturum.
pub const Session = SessionT(qwen35);

const TaslakYol = struct { sonraki: usize = 0, bekle: usize = 1, boy: usize = 8, deneme: usize = 0, kabul: usize = 0, onerilen: usize = 0 };

pub fn SessionT(comptime E: type) type {
return struct {
    const Self = @This();
    a: Alloc,
    io: std.Io,
    o: *const Opts,
    model: *E.Model,
    ctx: *E.Context,
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
    /// H10: istek başına araç çağrısı iskeleti (sunucu doldurur)
    iskelet: ?*gorev.Iskelet = null,
    /// sunucu: önek kaydı (null = kapalı). kayit_ids: kaydın kapsadığı tokenlar.
    kayit: ?E.Context.Kayit = null,
    kayit_ids: ArrayList(u32) = undefined,
    /// runIds'te bu kadar token işlenince durum kaydedilir (0 = kaydetme)
    kayit_noktasi: usize = 0,
    /// sunucu: prompt bu noktalarda (runIds'e göre) ayrı parçalarla işlenir; önek yeniden kullanımıyla
    /// sıfırdan işleme aynı parçalamayı görür → aynı sayısal sonuç
    bolum: []const usize = &.{},
    /// taslakçı (--fast ayarlı model): ana ctx ile aynı tokenları işler
    tc: ?*E.Context = null,
    taslak_k: usize = 0,
    dkayit: E.Context.Kayit = .{},
    dsnap: E.Context.Kayit = .{},
    /// taslak yolları (0 n-gram, 1 model): tutmayan yol geri çekilir; oturum boyunca öğrenilir
    // model taslakçısı (sohbet setinde %1 kabul) temkinli başlar: ilk denemeden sonra 64 token bekler
    gc: [3]TaslakYol = .{ .{}, .{ .bekle = 64 }, .{} },
    /// H02 deposu (null = kapalı)
    rest: ?*const rest.Store = null,

    pub fn temizle(s: *Self) void {
        s.ctx.clear();
        if (s.tc) |tc| tc.clear();
        s.hist.clearRetainingCapacity();
    }
    pub fn kayittanYukle(s: *Self, k: *const E.Context.Kayit) void {
        s.ctx.yukle(k);
        if (s.tc) |tc| tc.yukle(&s.dkayit);
        s.hist.shrinkRetainingCapacity(k.head);
    }

    /// Taslakçı pending'den başlayıp out.len-1 token önerir; işlediği tokenlar out[0..m-1].
    fn taslakla(s: *Self, tc: *E.Context, pending: u32, out: []u32) !usize {
        if (comptime E != qwen35) return 0;
        try tc.kaydet(&s.dsnap);
        tc.head_atla = false;
        defer tc.head_atla = true;
        out[0] = pending;
        var m: usize = 1;
        while (m < out.len) {
            try tc.decode(out[m - 1 .. m]);
            out[m] = tc.lm.?.token;
            m += 1;
            if (s.isStop(out[m - 1])) break;
        }
        return m;
    }
    /// Kabulden sonra taslakçıyı ana ctx ile aynı noktaya getirir (ana ctx buf[0..acc] işledi).
    fn esitle(s: *Self, tc: *E.Context, buf: []const u32, acc: usize, model_taslak: bool) !void {
        const m = buf.len;
        if (!model_taslak) {
            if (acc > 0) try tc.decode(buf[0..acc]);
        } else if (acc + 1 == m) {
            // taslakçı tam bu kadarını işledi
        } else if (acc == m) {
            try tc.decode(buf[m - 1 .. m]);
        } else {
            tc.yukle(&s.dsnap);
            if (acc > 0) try tc.decode(buf[0..acc]);
        }
    }

    pub fn tokenize(s: *Self, text: []const u8, special: bool, dst: *ArrayList(u32)) !void {
        try s.bpe.encode(text, dst, special);
    }

    /// ChatML (Qwen3.5 şablonu): kullanıcı mesajını ve asistan girişini token dizisine çevirir.
    fn turn(s: *Self, user: []const u8, dst: *ArrayList(u32)) !void {
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

    fn feed(s: *Self, ids: []const u32) !void {
        if (s.ctx.head + ids.len > s.ctx.n_ctx) return error.ContextFull;
        try s.ctx.decode(ids);
        if (s.tc) |tc| try tc.decode(ids);
        try s.hist.appendSlice(ids);
    }

    fn next(s: *Self) u32 {
        if (s.ctx.lm) |l| return l.token;
        return s.smp.sample(s.ctx.logits, s.hist.items);
    }

    fn isStop(s: *const Self, t: u32) bool {
        for (s.stop_ids) |x| if (x == t) return true;
        return s.bpe.isControl(t);
    }

    pub const Sonuc = struct { n_prompt: usize, n_gen: usize, t_prompt: u64, t_gen: u64, neden: []const u8, fed: bool = false, ff_tok: usize = 0, ff_blok: usize = 0 };

    /// Bir tur: prompt'u işler, cevabı akıtır.
    fn respond(s: *Self, user: []const u8) !Sonuc {
        var ids = ArrayList(u32).init(s.a);
        defer ids.deinit();
        try s.turn(user, &ids);
        const r = try s.runIds(ids.items);
        // asistan turu KV'de hiç kapanmaz (durma tokenı yazılmaz); bir sonraki tur <|im_end|> ile kapatır
        if (r.fed) s.acik_cevap = !s.o.raw;
        return r;
    }

    /// Hazır token dizisini işler ve cevabı akıtır (sohbet ve sunucu ortak).
    pub fn runIds(s: *Self, ids_in: []const u32) !Sonuc {
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
        const kn = s.kayit_noktasi;
        var bas: usize = 0;
        for (s.bolum) |b| if (b > bas and b < ids.items.len) {
            if (kn > bas and kn < b) break; // bolum sıralı; kn araya düşerse aşağıdaki yol işler
            try s.feed(ids.items[bas..b]);
            bas = b;
            if (s.kayit != null and b == kn) {
                try s.ctx.kaydet(&s.kayit.?);
                if (s.tc) |tc| try tc.kaydet(&s.dkayit);
                s.kayit_ids.clearRetainingCapacity();
                try s.kayit_ids.appendSlice(s.hist.items);
            }
        };
        if (s.kayit != null and kn > bas and kn < ids.items.len) {
            try s.feed(ids.items[bas..kn]);
            try s.ctx.kaydet(&s.kayit.?);
            if (s.tc) |tc| try tc.kaydet(&s.dkayit);
            s.kayit_ids.clearRetainingCapacity();
            try s.kayit_ids.appendSlice(s.hist.items);
            bas = kn;
        }
        try s.feed(ids.items[bas..]);
        r.t_prompt = @intCast(t0.untilNow(s.io, .awake).toNanoseconds());
        r.fed = true;

        const t1 = std.Io.Timestamp.now(s.io, .awake);
        try s.generate(&r);
        r.t_gen = @intCast(t1.untilNow(s.io, .awake).toNanoseconds());
        return r;
    }

    fn generate(s: *Self, r: *Sonuc) !void {
        var yaz = Yazici{ .s = .{ .a = s.a, .out = s.out, .sink = s.sink, .bpe = &s.bpe, .stops = s.stops } };
        defer yaz.deinit();
        const greedy = s.ctx.lm != null or s.smp.p.temp <= 0;
        const limit: usize = if (s.limit < 0) std.math.maxInt(usize) else @intCast(s.limit);
        var d: ?ngram.TokenDrafter = null;
        defer if (d) |*x| x.deinit();
        if (greedy and (s.o.ngram or s.rest != null)) {
            d = try ngram.TokenDrafter.init(s.a, envNum("ECHO_NG_N", 2));
            for (s.hist.items) |t| try d.?.add(t);
        }
        var esp: ?esp_mod.Esp = null;
        defer if (esp) |*x| x.deinit();
        if (comptime E == qwen35) if (greedy and s.o.esp and s.tc == null and s.ctx.trace_k > 0) {
            esp = try esp_mod.Esp.init(s.a, s.ctx, @intCast(envNum("ECHO_ESP_K", 1)), @as(f32, @floatFromInt(envNum("ECHO_ESP_L100", 10))) / 100.0);
            esp.?.seed(s.ctx, s.hist.items);
        };
        defer if (esp) |x| if (std.c.getenv("ECHO_ESP_IST") != null) std.debug.print("ESP tur={d} draftli_tur={d} onerilen={d} kabul={d} uretilen={d}\n", .{ x.rounds, x.rounds_draft, x.drafted, x.accepted, x.tokens });
        var pending = s.next();
        var buf: [256]u32 = undefined;
        const ng_max = @min(envNum("ECHO_NG_MAX", 128), buf.len);
        const ng_min = envNum("ECHO_NG_MIN", 3);
        // tutmayan taslak yolu geri çekilir: reddedilince bekleme 2 katına (en çok 64 token), tutunca sıfır
        // zincir boyu kabule uyar: 8'den başlar, tamamı tutarsa 2 katı (ECHO_NG_MAX'a kadar), tutmazsa kabulün 2 katı
        const gc = &s.gc;
        for (gc) |*g| {
            g.sonraki = 0; // n_gen her turda sıfırdan sayar; bekleme ve boy oturum boyunca kalır
            g.deneme = 0;
            g.kabul = 0;
            g.onerilen = 0;
        }
        var t_taslak: u64 = 0;
        var t_dog: u64 = 0; // doğrulama + geri sarma + yeniden işleme (ana ctx)
        var t_es: u64 = 0; // taslakçı eşitleme
        var t_tek: u64 = 0; // tek token: ana ctx
        var t_tek_tc: u64 = 0; // tek token: taslakçı
        var dv_ms: [9]u64 = @splat(0); // decodeVerify süresi, taslak boyuna göre (8 = 8 ve üstü)
        var dv_n: [9]u64 = @splat(0);
        var t_rest: u64 = 0; // H02 sorgu süresi
        var rest_sorgu: u64 = 0;
        var rest_eslesme: u64 = 0;
        defer if (s.rest != null and std.c.getenv("ECHO_REST_DEBUG") != null) std.debug.print("[H02 sorgu={d} eşleşme={d} sorgu_ms={d}]\n", .{ rest_sorgu, rest_eslesme, t_rest / 1_000_000 });
        const rest_min = envNum("ECHO_REST_MIN", 3);
        var t_rd: u64 = 0; // ret sonrası yeniden işleme
        const t_bas = std.Io.Timestamp.now(s.io, .awake);
        // ECHO_TASLAK_IST=dosya: yalnız sayılar eklenir (içerik yazılmaz)
        defer if (std.c.getenv("ECHO_TASLAK_IST")) |yol| istYaz(s.io, std.mem.span(yol), .{ gc[0].deneme, gc[0].kabul, gc[0].onerilen, gc[1].deneme, gc[1].kabul, gc[1].onerilen, gc[2].deneme, gc[2].kabul, gc[2].onerilen, (t_taslak + t_rest) / 1_000_000, r.n_gen, @as(u64, @intCast(t_bas.untilNow(s.io, .awake).toNanoseconds())) / 1_000_000, t_dog / 1_000_000, t_es / 1_000_000, t_tek / 1_000_000, t_tek_tc / 1_000_000, t_rd / 1_000_000, dv_n[2], dv_ms[2] / 1_000_000, dv_n[3], dv_ms[3] / 1_000_000, dv_n[4], dv_ms[4] / 1_000_000, dv_n[5], dv_ms[5] / 1_000_000, dv_n[6], dv_ms[6] / 1_000_000, dv_n[7], dv_ms[7] / 1_000_000, dv_n[8], dv_ms[8] / 1_000_000 });
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
            // H03 ESP: pending + draft + maske tek forward; kabul edilen tokenlar greedy ile birebir
            if (comptime E == qwen35) if (esp) |*ep| {
                const room = @min(limit - r.n_gen, s.ctx.n_ctx - s.ctx.head - 1);
                const td = std.Io.Timestamp.now(s.io, .awake);
                if (try ep.begin(s.ctx, pending, room -| 1)) {
                    var acc = ep.matched(s.ctx);
                    var stop_at: ?usize = null;
                    for (ep.buf[0..acc], 0..) |t, q| if (s.isStop(t)) {
                        stop_at = q;
                        break;
                    };
                    if (stop_at) |q| acc = q;
                    const nxt = s.ctx.verify_tokens[acc -| 1];
                    try ep.finish(s.ctx, acc);
                    t_dog += @intCast(td.untilNow(s.io, .awake).toNanoseconds());
                    for (ep.buf[0..acc]) |t| {
                        try s.hist.append(t);
                        if (try yaz.add(t)) {
                            r.neden = "stop metni";
                            r.n_gen += 1;
                            return;
                        }
                    }
                    r.n_gen += acc;
                    pending = if (stop_at) |q| ep.buf[q] else nxt;
                    continue;
                }
            };
            // greedy: n-gram taslağı tek forward'da doğrulanır (kayıpsız; kabul edilen tokenlar aynı)
            // H10: şemadan kesin olan iskelet metni model çağrılmadan tek blok olarak yazılır
            if (s.iskelet) |k| {
                var blok = ArrayList(u8).init(s.a);
                defer blok.deinit();
                if (k.ilk_zorla and r.n_gen == 0 and !k.aktif) {
                    k.ilk_zorla = false;
                    k.gozle("<tool_call>");
                    try blok.appendSlice("<tool_call>");
                }
                if (k.aktif) {
                    var pt: []const u8 = s.bpe.piece(pending);
                    while (true) switch (try k.bak(pt)) {
                        .zorla => |t| {
                            try blok.appendSlice(t);
                            k.zorlandi(t);
                            pt = ""; // zorlanan metinden sonra modelin eski tokenı geçersiz
                        },
                        else => break,
                    };
                }
                if (blok.items.len > 0) {
                    var fid = ArrayList(u32).init(s.a);
                    defer fid.deinit();
                    try s.tokenize(blok.items, false, &fid);
                    const n = @min(fid.items.len, limit - r.n_gen, s.ctx.n_ctx - s.ctx.head - 1);
                    for (fid.items[0..n], 0..) |t, i| if (try yaz.add(t)) {
                        r.neden = "stop metni";
                        r.n_gen += i + 1;
                        try s.feed(fid.items[0 .. i + 1]);
                        return;
                    };
                    try s.feed(fid.items[0..n]);
                    if (d) |*dr| for (fid.items[0..n]) |t| try dr.add(t);
                    r.n_gen += n;
                    r.ff_tok += n;
                    r.ff_blok += 1;
                    pending = s.next();
                    continue;
                }
            }
            // H10: <tool_call> içinde n-gram kapalı (iskelet her tokenı görmeli)
            const ng_ok = if (s.iskelet) |k| !k.aktif else true;
            if (if (ng_ok and d != null) &d.? else null) |dr| {
                const room = @min(ng_max, limit - r.n_gen, s.ctx.n_ctx - s.ctx.head - 1);
                var m: usize = if (s.o.ngram and r.n_gen >= gc[0].sonraki) dr.chain(buf[0..@min(room, gc[0].boy)]) else 0;
                var model_taslak = false;
                var yol: usize = 0;
                if (!(m >= ng_min and buf[0] == pending and s.ctx.trace_k >= m)) {
                    m = 0;
                    // H02: depodan en uzun suffix eşleşmesinin devamı (buf[0] = pending)
                    if (s.rest) |st| if (room >= 2 and r.n_gen >= gc[2].sonraki) {
                        var ctxt: [rest.MAXN]u32 = undefined;
                        const hn: usize = @min(s.hist.items.len, rest.MAXN - 1);
                        @memcpy(ctxt[0..hn], s.hist.items[s.hist.items.len - hn ..]);
                        ctxt[hn] = pending;
                        var hit: rest.Store.Hit = undefined;
                        const t_r = std.Io.Timestamp.now(s.io, .awake);
                        const nn = st.chain(ctxt[0 .. hn + 1], rest_min, buf[1..@min(room, gc[2].boy + 1, s.ctx.trace_k)], &hit);
                        t_rest += @intCast(t_r.untilNow(s.io, .awake).toNanoseconds());
                        rest_sorgu += 1;
                        if (hit.n > 0) rest_eslesme += 1;
                        if (nn + 1 >= ng_min) {
                            buf[0] = pending;
                            m = nn + 1;
                            yol = 2;
                        }
                    };
                    if (m == 0) if (s.tc) |tc| if (room >= 2 and r.n_gen >= gc[1].sonraki) {
                        const tt = std.Io.Timestamp.now(s.io, .awake);
                        m = try s.taslakla(tc, pending, buf[0..@min(room, s.taslak_k + 1, s.ctx.trace_k)]);
                        t_taslak += @intCast(tt.untilNow(s.io, .awake).toNanoseconds());
                        model_taslak = true;
                        yol = 1;
                        if (m < 2) {
                            try s.esitle(tc, buf[0..m], 0, true);
                            m = 0;
                        }
                    };
                }
                if (m > 0) {
                    const td = std.Io.Timestamp.now(s.io, .awake);
                    const before = s.ctx.head;
                    s.ctx.snapshot();
                    const tv = std.Io.Timestamp.now(s.io, .awake);
                    try s.ctx.decodeVerify(buf[0..m]);
                    dv_ms[@min(m, 8)] += @intCast(tv.untilNow(s.io, .awake).toNanoseconds());
                    dv_n[@min(m, 8)] += 1;
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
                        const trd = std.Io.Timestamp.now(s.io, .awake);
                        s.ctx.rollback(before);
                        if (acc > 0) try s.ctx.decode(buf[0..acc]);
                        t_rd += @intCast(trd.untilNow(s.io, .awake).toNanoseconds());
                    }
                    t_dog += @intCast(td.untilNow(s.io, .awake).toNanoseconds());
                    const te = std.Io.Timestamp.now(s.io, .awake);
                    if (s.tc) |tc| try s.esitle(tc, buf[0..m], acc, model_taslak);
                    t_es += @intCast(te.untilNow(s.io, .awake).toNanoseconds());
                    const g = &gc[yol];
                    g.deneme += 1;
                    g.kabul += acc -| 1;
                    g.onerilen += m - 1;
                    if (acc <= 1) {
                        g.sonraki = r.n_gen + acc + g.bekle;
                        g.bekle = @min(g.bekle * 2, 512);
                    } else g.bekle = 1;
                    g.boy = if (acc == m) @min(g.boy * 2, ng_max) else std.math.clamp(acc * 2, ng_min, ng_max);
                    for (buf[0..acc]) |t| {
                        try s.hist.append(t);
                        try dr.add(t);
                        if (s.iskelet) |k| k.gozle(s.bpe.piece(t));
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
            {
                if (s.ctx.head + 1 > s.ctx.n_ctx) return error.ContextFull;
                const t1 = std.Io.Timestamp.now(s.io, .awake);
                try s.ctx.decode(&.{pending});
                t_tek += @intCast(t1.untilNow(s.io, .awake).toNanoseconds());
                const t2 = std.Io.Timestamp.now(s.io, .awake);
                if (s.tc) |tc| try tc.decode(&.{pending});
                t_tek_tc += @intCast(t2.untilNow(s.io, .awake).toNanoseconds());
                try s.hist.append(pending);
            }
            if (d) |*dr| try dr.add(pending);
            if (s.iskelet) |k| k.gozle(s.bpe.piece(pending));
            r.n_gen += 1;
            pending = s.next();
        }
        if (r.n_gen >= limit) r.neden = "-n sınırı";
    }
};
}

/// Token baytlarını UTF-8 güvenli akıtır; --stop metinlerini yakalar (eşleşme olasılığı olan kuyruğu bekletir).
const Yazici = struct {
    s: struct { a: Alloc, out: *std.Io.Writer, sink: ?Sink, bpe: *const Bpe, stops: []const []const u8 },
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
    // model verilmezse programın yanındaki depodaki model kullanılır (zig-out/bin → ../../../ArastirmaModelleri)
    const path = o.model orelse blk: {
        var buf: [4096]u8 = undefined;
        const n = std.process.executableDirPath(io, &buf) catch 0;
        const yol = try std.fmt.allocPrint(a, "{s}/../../../ArastirmaModelleri/Qwen3.5-0.8B-Q8_0.gguf", .{buf[0..n]});
        std.Io.Dir.cwd().access(io, yol, .{}) catch {
            std.debug.print("hata: model bulunamadı ({s}); -m MODEL.gguf ile verin\n\n{s}", .{ yol, yardim });
            std.process.exit(2);
        };
        break :blk yol;
    };
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
    // yerleşik şablon Qwen3.5 ChatML; modelin kendi şablonu farklıysa uyar
    if (!o.raw) if (reader.metadata.get("tokenizer.chat_template")) |v| if (v == .string) {
        if (std.mem.indexOf(u8, v.string, "<|im_start|>") == null or std.mem.indexOf(u8, v.string, "<think>") == null)
            std.debug.print("uyarı: modelin sohbet şablonu Qwen3.5 ChatML değil; yerleşik şablon kullanılıyor (--raw ile kapatılır)\n", .{});
    };
    reader.deinit();
    defer bpe.deinit();

    return runWith(qwen35, &o, path, bpe, a, io);
}

fn runWith(comptime E: type, o_: *Opts, path: []const u8, bpe: Bpe, a: Alloc, io: std.Io) !void {
    const o = o_;
    if (o.fast) {
        // tek düzen (10 Kas, "D"): katman atlama ve FFN budama yok; 18 atlama "0 0 0", FFN 1920 döngü üretiyordu
        _ = setenv("ECHO_NG_N", "2", 0);
        _ = setenv("ECHO_NG_MIN", "3", 0);
        _ = setenv("ECHO_NG_MAX", "128", 0);
    }
    // --fast: ayar (atlama + FFN budama) taslakçı modelde; çıktıyı tam model doğrular (ECHO_TASLAK=0 eski tek model)
    const taslak_k = envNum("ECHO_TASLAK", 0);
    const taslak_ac = E == qwen35 and o.fast and taslak_k > 0;
    const dmodel: ?*E.Model = if (taslak_ac) try E.Model.load(path, a) else null;
    defer if (dmodel) |m| m.deinit();
    if (taslak_ac) {
        _ = unsetenv("ECHO_FFN_KEEP");
        _ = unsetenv("ECHO_FFN_ACT");
        _ = setenv("ECHO_SKIP", "", 1);
    }
    const model = try E.Model.load(path, a);
    defer model.deinit();
    // varsayılan kayıpsız: katman atlama yalnız --fast ile (ECHO_SKIP elle verilmişse ona uyulur)
    if (@hasField(E.Model, "skip")) if (std.c.getenv("ECHO_SKIP") == null) {
        model.skip = 0;
    };
    const cpus = std.Thread.getCpuCount() catch 2;
    const n_threads: u32 = o.threads orelse if (envNum("ECHO_THREADS", 0) > 0) @intCast(envNum("ECHO_THREADS", 0)) else @intCast(@max(cpus / 2, 1));
    const lm_ix: ?*ivf.Runtime = if (E != qwen35) null else if (o.fast) ivf.open(path, model.output, ivf.default_l, n_threads, a, io) catch null else null;
    defer if (lm_ix) |ix| ix.deinit(a);

    const ctx = try E.Context.init(model, o.n_ctx, n_threads, a);
    defer ctx.deinit();
    if (E == qwen35) if (lm_ix) |ix| {
        const np: u32 = if (std.c.getenv("ECHO_NPROBE")) |v| std.fmt.parseInt(u32, std.mem.span(v), 10) catch ivf.default_nprobe else ivf.default_nprobe;
        ctx.lm = try ivf.Search.init(ix, np, n_threads, a);
    };
    defer if (ctx.lm) |l| l.deinit(a);
    const dctx: ?*E.Context = if (dmodel) |m| try E.Context.init(m, o.n_ctx, n_threads, a) else null;
    defer if (dctx) |c| c.deinit();
    if (E == qwen35) if (dctx) |c| {
        c.lm = try ivf.Search.init(lm_ix orelse return error.IvfYok, if (ctx.lm) |l| l.nprobe else ivf.default_nprobe, n_threads, a);
        c.head_atla = true; // yalnız taslakla() içinde açılır
    };
    defer if (dctx) |c| if (c.lm) |l| l.deinit(a);
    if (o.fast) {
        o.sp.temp = 0;
        o.ngram = !o.no_ng and !o.esp and std.c.getenv("ECHO_NO_NGRAM") == null; // ECHO_NO_NGRAM: taslaksız ölçüm tabanı
    }
    var rest_store: ?rest.Store = null;
    defer if (rest_store) |*st| st.deinit();
    if (o.rest) |dizin| {
        const tb = std.Io.Timestamp.now(io, .awake);
        rest_store = try rest.Store.build(a, io, dizin, &bpe);
        std.debug.print("[H02 depo: {d} dosya, {d} token, kuruluş {d} ms]\n", .{ rest_store.?.n_files, rest_store.?.ds.len, @divTrunc(tb.untilNow(io, .awake).toNanoseconds(), 1_000_000) });
    }
    if (o.esp) {
        if (E != qwen35) return error.InvalidArgs;
        try ctx.enableTrace(16);
        ctx.no_trace = true; // geri alma snapshot ile
    } else if ((o.ngram or o.rest != null) and (o.fast or o.sp.temp <= 0)) {
        try ctx.enableTrace(@intCast(@min(envNum("ECHO_NG_MAX", 128), 256)));
        ctx.no_trace = true; // geri alma snapshot ile (H13)
    }

    var obuf: [4096]u8 = undefined;
    var ow = std.Io.File.stdout().writer(io, &obuf);
    var s = SessionT(E){
        .a = a,
        .io = io,
        .o = o,
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
        .tc = dctx,
        .taslak_k = taslak_k,
        .rest = if (rest_store) |*st| st else null,
    };
    defer s.hist.deinit();
    defer s.dkayit.deinit(a);
    defer s.dsnap.deinit(a);
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

fn cevapla(s: anytype, q: []const u8) !void {
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

fn sohbet(s: anytype) !void {
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
            s.temizle();
            s.acik_cevap = false;
            std.debug.print("[geçmiş silindi]\n", .{});
            continue;
        }
        if (std.mem.startsWith(u8, t, "/sys")) {
            s.sys = try s.a.dupe(u8, std.mem.trim(u8, t[4..], " "));
            s.temizle();
            s.acik_cevap = false;
            std.debug.print("[sistem mesajı ayarlandı, geçmiş silindi]\n", .{});
            continue;
        }
        try cevapla(s, t);
    }
}

extern fn sched_setaffinity(pid: c_int, size: usize, mask: *const [16]u64) c_int;
fn envNum(name: [*:0]const u8, def: usize) usize {
    const v = std.c.getenv(name) orelse return def;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch def;
}

fn istYaz(io: std.Io, yol: []const u8, v: anytype) void {
    var b: [640]u8 = undefined;
    const t = std.fmt.bufPrint(&b, "ng_deneme={d} ng_kabul={d} ng_oneri={d} md_deneme={d} md_kabul={d} md_oneri={d} rs_deneme={d} rs_kabul={d} rs_oneri={d} taslak_ms={d} uretim={d} uretim_ms={d} dog_ms={d} esitle_ms={d} tek_ms={d} tek_tc_ms={d} rd_ms={d} n2={d} ms2={d} n3={d} ms3={d} n4={d} ms4={d} n5={d} ms5={d} n6={d} ms6={d} n7={d} ms7={d} n8={d} ms8={d}\n", v) catch return;
    const f = std.Io.Dir.cwd().createFile(io, yol, .{ .truncate = false }) catch return;
    defer f.close(io);
    const n = f.length(io) catch return;
    f.writePositionalAll(io, t, n) catch {};
}
