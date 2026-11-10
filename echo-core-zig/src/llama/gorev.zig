//! H10 — task modu (lossless DEĞİL): araç çağrısı iskeleti şemadan türetilir.
//! Qwen3.5 araç çağrısı biçimi:
//!   <tool_call>\n<function=AD>\n<parameter=P>\nDEĞER\n</parameter>\n</function>\n</tool_call>
//! Çıktı üçe ayrılır:
//!   1) şemadan kesin olan (etiketler, tek adaylı fonksiyon/parametre adı, enum'un kalanı, kapanışlar)
//!      -> model sorulmadan tek blok olarak yazılır (fast-forward; tek forward, prefill hızında)
//!   2) birden çok aday olan seçim (çok araçlı ad, isteğe bağlı parametre) -> model seçer;
//!      ön ek tek adaya inince kalanı bloğa döner
//!   3) serbest değer -> model üretir; değer biterken model "\n" isterse kapanış bloğu yazılır
//! Fikir: llguidance fast-forward (https://github.com/guidance-ai/llguidance) — grameri tek yollu
//! kalan kısımlar model çağrılmadan eklenir. Doğruluk ölçütü dağılım eşdeğerliği değil, çağrının geçerliliği.
//! Sınırlar: değerde çok satırlı metin desteklenmez ("\n" değeri kapatır); en çok 64 parametre.
const std = @import("std");
const Value = std.json.Value;
const Alloc = std.mem.Allocator;
const List = std.array_list.Managed;

pub const Tip = enum { metin, sayi, mantik, diger };
pub const Param = struct {
    ad: []const u8,
    zorunlu: bool,
    tip: Tip,
    /// enum (ya da boolean için true/false); boşsa serbest
    secenek: []const []const u8 = &.{},
};
pub const Arac = struct { ad: []const u8, params: []Param };

pub const Karar = union(enum) {
    /// iskelet burada karar vermiyor: modelin seçtiği token kabul edilir
    model,
    /// bu metin kesin: bloğa eklenir (modelin bekleyen tokenı atılır)
    zorla: []const u8,
    /// çağrı bitti ya da model iskeletten saptı: çağrı sonuna kadar karışma yok
    kapali,
};

const L_FN = "\n<function=";
const L_SON = "</function>\n</tool_call>";
const L_KAPA = "\n</parameter>\n";
const L_PAR = "<parameter=";
const MAX_PARAM = 64;

fn bas(a: []const u8, p: []const u8) bool {
    return std.mem.startsWith(u8, a, p);
}
fn get(v: Value, k: []const u8) ?Value {
    if (v != .object) return null;
    const x = v.object.get(k) orelse return null;
    return if (x == .null) null else x;
}

pub const Iskelet = struct {
    a: Alloc,
    araclar: []Arac,
    /// <tool_call> görüldükten sonra yazılan metin
    g: List(u8),
    aktif: bool = false,
    /// tool_choice required / adlı fonksiyon: ilk token <tool_call> olarak zorlanır
    ilk_zorla: bool = false,
    /// istatistik: zorlanan token ve blok sayısı (cli doldurur)
    zorla_tok: usize = 0,
    zorla_blok: usize = 0,

    /// `tools`: OpenAI tools dizisi (sayılar parse_numbers=false olabilir). `sadece`: tool_choice ile seçilen ad.
    pub fn init(a: Alloc, tools: Value, sadece: ?[]const u8, ilk_zorla: bool) !Iskelet {
        var arr = List(Arac).init(a);
        if (tools == .array) for (tools.array.items) |x| {
            const f = get(x, "function") orelse x;
            const ad = if (get(f, "name")) |n| (if (n == .string) n.string else continue) else continue;
            if (sadece) |s| if (!std.mem.eql(u8, s, ad)) continue;
            var ps = List(Param).init(a);
            if (get(f, "parameters")) |pm| if (get(pm, "properties")) |props| if (props == .object) {
                var it = props.object.iterator();
                while (it.next()) |e| {
                    var p = Param{ .ad = e.key_ptr.*, .zorunlu = false, .tip = .diger };
                    if (get(pm, "required")) |rq| if (rq == .array) for (rq.array.items) |q| {
                        if (q == .string and std.mem.eql(u8, q.string, p.ad)) p.zorunlu = true;
                    };
                    if (get(e.value_ptr.*, "type")) |t| if (t == .string) {
                        const ts = t.string;
                        p.tip = if (std.mem.eql(u8, ts, "string")) .metin else if (std.mem.eql(u8, ts, "integer") or std.mem.eql(u8, ts, "number")) .sayi else if (std.mem.eql(u8, ts, "boolean")) .mantik else .diger;
                    };
                    if (p.tip == .mantik) p.secenek = &.{ "true", "false" };
                    if (p.tip == .metin) if (get(e.value_ptr.*, "enum")) |en| if (en == .array) {
                        var sc = List([]const u8).init(a);
                        for (en.array.items) |q| if (q == .string) try sc.append(q.string);
                        if (sc.items.len == en.array.items.len) p.secenek = sc.items;
                    };
                    try ps.append(p);
                }
            };
            try arr.append(.{ .ad = ad, .params = ps.items });
        };
        return .{ .a = a, .araclar = arr.items, .g = List(u8).init(a), .ilk_zorla = ilk_zorla and arr.items.len > 0 };
    }

    /// Yazılan (kabul edilen) her tokenın metni. <tool_call> görülünce iskelet açılır.
    pub fn gozle(k: *Iskelet, t: []const u8) void {
        if (!k.aktif) {
            if (std.mem.eql(u8, t, "<tool_call>")) {
                k.aktif = true;
                k.g.clearRetainingCapacity();
            }
            return;
        }
        k.g.appendSlice(t) catch {
            k.aktif = false;
        };
    }

    fn kapat(k: *Iskelet) Karar {
        k.aktif = false;
        return .kapali;
    }

    /// Şimdiki metinden sonra kesin olan devam. `pend`: modelin seçtiği (henüz yazılmamış) tokenın metni.
    pub fn bak(k: *Iskelet, pend: []const u8) !Karar {
        if (!k.aktif) return .kapali;
        const g = k.g.items;
        const a = k.a;
        // 1) "\n<function="
        if (g.len < L_FN.len) {
            if (!bas(L_FN, g)) return k.kapat();
            return .{ .zorla = L_FN[g.len..] };
        }
        if (!bas(g, L_FN)) return k.kapat();
        // 2) fonksiyon adı
        const gt = std.mem.indexOfScalarPos(u8, g, L_FN.len, '>') orelse {
            const cur = g[L_FN.len..];
            var n: usize = 0;
            var tek: []const u8 = "";
            for (k.araclar) |t| if (bas(t.ad, cur)) {
                n += 1;
                tek = t.ad;
            };
            if (n == 0) return k.kapat();
            if (n == 1) return .{ .zorla = try std.mem.concat(a, u8, &.{ tek[cur.len..], ">\n" }) };
            return .model;
        };
        const ad = g[L_FN.len..gt];
        var arac: ?*const Arac = null;
        for (k.araclar) |*t| if (std.mem.eql(u8, t.ad, ad)) {
            arac = t;
        };
        const ar = arac orelse return k.kapat();
        if (ar.params.len > MAX_PARAM) return k.kapat();
        if (g.len == gt + 1) return .{ .zorla = "\n" };
        if (g[gt + 1] != '\n') return k.kapat();
        var pos = gt + 2;
        // 3) parametreler
        var yazildi: [MAX_PARAM]bool = @splat(false);
        while (true) {
            const r = g[pos..];
            if (bas(r, L_SON)) { // çağrı tam bitti
                k.aktif = false;
                return .kapali;
            }
            if (bas(r, L_PAR)) {
                const ge = std.mem.indexOfScalarPos(u8, r, L_PAR.len, '>') orelse {
                    const cur = r[L_PAR.len..];
                    var n: usize = 0;
                    var tek: []const u8 = "";
                    for (ar.params, 0..) |p, i| if (!yazildi[i] and bas(p.ad, cur)) {
                        n += 1;
                        tek = p.ad;
                    };
                    if (n == 0) return k.kapat();
                    if (n == 1) return .{ .zorla = try std.mem.concat(a, u8, &.{ tek[cur.len..], ">\n" }) };
                    return .model;
                };
                const pad = r[L_PAR.len..ge];
                var pi: ?usize = null;
                for (ar.params, 0..) |p, i| if (!yazildi[i] and std.mem.eql(u8, p.ad, pad)) {
                    pi = i;
                };
                const p = pi orelse return k.kapat();
                if (r.len == ge + 1) return .{ .zorla = "\n" };
                if (r[ge + 1] != '\n') return k.kapat();
                const v = r[ge + 2 ..];
                if (std.mem.indexOf(u8, v, L_KAPA)) |ci| {
                    yazildi[p] = true;
                    pos += ge + 2 + ci + L_KAPA.len;
                    continue;
                }
                // yarım kapanış ("\n</para") -> kalanı tamamla
                var j: usize = @min(v.len, L_KAPA.len - 1);
                while (j > 0) : (j -= 1) if (bas(L_KAPA, v[v.len - j ..])) return .{ .zorla = L_KAPA[j..] };
                // değer açık
                const sc = ar.params[p].secenek;
                if (sc.len > 0) {
                    var n: usize = 0;
                    var tek: []const u8 = "";
                    for (sc) |c| if (bas(c, v)) {
                        n += 1;
                        tek = c;
                    };
                    if (n == 1) return .{ .zorla = try std.mem.concat(a, u8, &.{ tek[v.len..], L_KAPA }) };
                    // ilk token: model büyük/küçük harf farkıyla yazdıysa ("False", "Fahrenheit") seçeneğe çevrilir
                    const t = std.mem.trimStart(u8, pend, " \n");
                    if (v.len == 0 and t.len > 0) {
                        n = 0;
                        for (sc) |c| if (std.ascii.startsWithIgnoreCase(c, t) or std.ascii.startsWithIgnoreCase(t, c)) {
                            n += 1;
                            tek = c;
                        };
                        if (n == 1) return .{ .zorla = try std.mem.concat(a, u8, &.{ tek, L_KAPA }) };
                    }
                    if (v.len == 0) return .model;
                }
                if (pend.len > 0 and pend[0] == '\n') return .{ .zorla = L_KAPA };
                return .model;
            }
            // parametreler arası: bir sonraki zorunlu parametre, yoksa seçim ya da kapanış
            var zor: ?usize = null;
            for (ar.params, 0..) |p, i| if (!yazildi[i] and p.zorunlu) {
                zor = zor orelse i;
            };
            if (zor) |i| {
                const lit = try std.mem.concat(a, u8, &.{ L_PAR, ar.params[i].ad, ">\n" });
                if (!bas(lit, r)) return k.kapat();
                return .{ .zorla = lit[r.len..] };
            }
            var n: usize = 0;
            var tek: []const u8 = "";
            if (bas(L_SON, r)) {
                n += 1;
                tek = L_SON;
            }
            for (ar.params, 0..) |p, i| if (!yazildi[i]) {
                const lit = try std.mem.concat(a, u8, &.{ L_PAR, p.ad, ">\n" });
                if (bas(lit, r)) {
                    n += 1;
                    tek = lit;
                }
            };
            if (n == 0) return k.kapat();
            if (n == 1) return .{ .zorla = tek[r.len..] };
            return .model;
        }
    }

    /// Zorlanan metni yazılmış sayar.
    pub fn zorlandi(k: *Iskelet, t: []const u8) void {
        k.g.appendSlice(t) catch {
            k.aktif = false;
        };
    }
};
