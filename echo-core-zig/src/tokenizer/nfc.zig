//! Unicode NFC (resmi Qwen tokenizer.json normalizer'ı). Tablolar: benchmarks/tokenizer/nfc_tablo.py.
//! Hızlı yol: UTF-8'de 0xCC ve üstü öncü bayt yoksa (Türkçe/Latin dahil) metin olduğu gibi döner.
const std = @import("std");

const ArrayList = std.array_list.Managed;
const tablo = @embedFile("nfc.bin");

fn u(i: usize) u32 {
    return std.mem.readInt(u32, tablo[i * 4 ..][0..4], .little);
}
const n_qc = u(0);
const n_dec = u(1);
const n_pool = u(2);
const n_ccc = u(3);
const n_comp = u(4);
const o_qc = 5;
const o_dec = o_qc + n_qc * 2;
const o_pool = o_dec + n_dec * 3;
const o_ccc = o_pool + n_pool;
const o_comp = o_ccc + n_ccc * 2;

const SB = 0xAC00;
const LB = 0x1100;
const VB = 0x1161;
const TB = 0x11A7;
const NV = 21;
const NT = 28;
const NS = 11172;

/// kayıt dizisinde ilk alanı cp olan girdiyi ikili arama ile bul (kayıt başına `w` u32)
fn bul(off: u32, n: u32, w: u32, cp: u32) ?u32 {
    var lo: u32 = 0;
    var hi: u32 = n;
    while (lo < hi) {
        const m = (lo + hi) / 2;
        const v = u(off + m * w);
        if (v == cp) return off + m * w;
        if (v < cp) lo = m + 1 else hi = m;
    }
    return null;
}

/// BMP için bit tablosu (derleme anında aralıklardan), üstü ikili arama
const qc_bmp: [65536 / 64]u64 = blk: {
    @setEvalBranchQuota(1 << 20);
    var t: [65536 / 64]u64 = @splat(0);
    for (0..n_qc) |k| {
        var c = u(o_qc + k * 2);
        while (c <= u(o_qc + k * 2 + 1) and c < 65536) : (c += 1) t[c >> 6] |= @as(u64, 1) << @intCast(c & 63);
    }
    break :blk t;
};

fn qc(cp: u32) bool {
    if (cp < 65536) return (qc_bmp[cp >> 6] >> @intCast(cp & 63)) & 1 != 0;
    var lo: u32 = 0;
    var hi: u32 = n_qc;
    while (lo < hi) {
        const m = (lo + hi) / 2;
        if (cp < u(o_qc + m * 2)) hi = m else if (cp > u(o_qc + m * 2 + 1)) lo = m + 1 else return true;
    }
    return false;
}

fn ccc(cp: u32) u32 {
    if (cp < 0x300) return 0;
    return if (bul(o_ccc, n_ccc, 2, cp)) |i| u(i + 1) else 0;
}

fn birlestir(a: u32, b: u32) ?u32 {
    // Hangul LV ve LV+T
    if (a >= LB and a < LB + 19 and b >= VB and b < VB + NV) return SB + ((a - LB) * NV + (b - VB)) * NT;
    if (a >= SB and a < SB + NS and (a - SB) % NT == 0 and b > TB and b < TB + NT) return a + (b - TB);
    var lo: u32 = 0;
    var hi: u32 = n_comp;
    while (lo < hi) {
        const m = (lo + hi) / 2;
        const i = o_comp + m * 3;
        const ka = u(i);
        const kb = u(i + 1);
        if (ka == a and kb == b) return u(i + 2);
        if (ka < a or (ka == a and kb < b)) lo = m + 1 else hi = m;
    }
    return null;
}

/// NFC gerekmiyorsa null; gerekiyorsa `out`a normalize edilmiş metni yazar ve onu döner.
/// Geçersiz UTF-8 olduğu gibi bırakılır.
pub fn normalize(text: []const u8, out: *ArrayList(u8)) !?[]const u8 {
    // yalnız 0xCC+ öncü baytlı karakterlere bak (U+0300 altı hiçbir zaman değişmez)
    var p: usize = 0;
    var gerek = false;
    while (p < text.len) {
        const b = text[p];
        if (b < 0xCC) {
            p += 1;
            continue;
        }
        const l = std.unicode.utf8ByteSequenceLength(b) catch return null;
        if (p + l > text.len) return null;
        const c = std.unicode.utf8Decode(text[p .. p + l]) catch return null;
        if (qc(c)) {
            gerek = true;
            break;
        }
        p += l;
    }
    if (!gerek) return null;
    if (!std.unicode.utf8ValidateSlice(text)) return null;

    // 1) kanonik ayrıştır
    var cps = ArrayList(u32).init(out.allocator);
    defer cps.deinit();
    var it = std.unicode.Utf8View.initUnchecked(text).iterator();
    while (it.nextCodepoint()) |c0| {
        const c: u32 = c0;
        if (c >= SB and c < SB + NS) {
            const s = c - SB;
            try cps.append(LB + s / (NV * NT));
            try cps.append(VB + (s % (NV * NT)) / NT);
            if (s % NT != 0) try cps.append(TB + s % NT);
        } else if (c >= 0xC0 and bul(o_dec, n_dec, 3, c) != null) {
            const i = bul(o_dec, n_dec, 3, c).?;
            for (0..u(i + 2)) |k| try cps.append(u(o_pool + u(i + 1) + @as(u32, @intCast(k))));
        } else try cps.append(c);
    }
    // 2) birleşik işaretleri sınıfa göre sırala (kararlı, araya giren sınıf 0 sınır)
    const v = cps.items;
    var i: usize = 1;
    while (i < v.len) : (i += 1) {
        const k = ccc(v[i]);
        if (k == 0) continue;
        var j = i;
        while (j > 0 and ccc(v[j - 1]) > k) : (j -= 1) std.mem.swap(u32, &v[j], &v[j - 1]);
    }
    // 3) birleştir
    var n: usize = 0;
    var bas: ?usize = null; // son başlangıç karakterinin çıktıdaki yeri
    var son_ccc: i64 = -1; // başlangıçtan sonra en son eklenen karakterin sınıfı
    for (v) |c| {
        const k = ccc(c);
        if (bas) |s| {
            const engel = son_ccc != -1 and (son_ccc == 0 or son_ccc >= k);
            if (!engel) if (birlestir(v[s], c)) |r| {
                v[s] = r;
                continue;
            };
        }
        v[n] = c;
        if (k == 0) {
            bas = n;
            son_ccc = -1;
        } else son_ccc = k;
        n += 1;
    }
    out.clearRetainingCapacity();
    for (v[0..n]) |c| {
        var buf: [4]u8 = undefined;
        const l = try std.unicode.utf8Encode(@intCast(c), &buf);
        try out.appendSlice(buf[0..l]);
    }
    return out.items;
}

test "nfc: ayrışık aksan birleşir, Türkçe hızlı yol, Hangul, sıralama" {
    var o = ArrayList(u8).init(std.testing.allocator);
    defer o.deinit();
    try std.testing.expect((try normalize("şğıçöü Merhaba", &o)) == null);
    try std.testing.expectEqualStrings("é", (try normalize("e\u{301}", &o)).?);
    try std.testing.expectEqualStrings("ş", (try normalize("s\u{327}", &o)).?);
    try std.testing.expectEqualStrings("\u{AC01}", (try normalize("\u{1100}\u{1161}\u{11A8}", &o)).?);
    // q + nokta altı(220) + nokta üstü(230): sıra korunur; ṩ = s + 323 + 307
    try std.testing.expectEqualStrings("\u{1E69}", (try normalize("s\u{307}\u{323}", &o)).?);
    // Angstrom işareti → Å (tekil ayrışma)
    try std.testing.expectEqualStrings("Å", (try normalize("\u{212B}", &o)).?);
}
