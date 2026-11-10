//! H03 ESP (Embedding-Space Probing), arXiv 2603.17942: eğitimsiz çoklu token tahmini.
//! Maske tokenlar prompt embedding ortalamasıdır (Denklem 4), üretimde son token embedding'ine doğru
//! kayar (Denklem 5). Maske satırları doğrulama batch'inin sonuna eklenir; çıktıları sonraki turun draft'ıdır.
//! Doğrulama kayıpsız greedy: draft yalnız modelin kendi top-1'i ile birebir eşleşirse kabul edilir.
//!
//! Tur: batch = [pending, d1..dn, mask x k]. verify[i] = pending/draft i sonrası top-1, verify[m+i] = maske i çıktısı.
//!   Tam kabulde maskeler doğru bağlamdadır (son kabul edilen tokenın hemen sonrası) → verify[m..m+k] sonraki draft.
//!   Reddedilince maske çıktıları bayat → sonraki tur draftsız [pending, mask x k] (daima 1 token kazanır).
const std = @import("std");
const qwen35 = @import("qwen35.zig");

const Alloc = std.mem.Allocator;
pub const MAXK: usize = 8;

pub const Esp = struct {
    a: Alloc,
    k: usize,
    lam: f32,
    tmp: []f32,
    drafts: [MAXK]u32 = @splat(0),
    nd: usize = 0,
    buf: [MAXK + 1]u32 = @splat(0),
    m: usize = 0,
    before: usize = 0,
    split: bool = false,
    // sayaçlar
    rounds: usize = 0,
    rounds_draft: usize = 0,
    drafted: usize = 0,
    accepted: usize = 0, // kabul edilen draft tokenlar (pending hariç)
    tokens: usize = 0, // tur başına üretilen toplam token

    pub fn init(a: Alloc, ctx: *qwen35.Context, k: usize, lam: f32) !Esp {
        if (k == 0 or k > MAXK) return error.InvalidArgs;
        try ctx.espInit();
        return .{ .a = a, .k = k, .lam = lam, .tmp = try a.alloc(f32, ctx.model.n_embd) };
    }
    pub fn deinit(self: *Esp) void {
        self.a.free(self.tmp);
    }

    /// Prompt/geçmiş token embedding ortalaması ile maskeyi kurar; draftları siler.
    pub fn seed(self: *Esp, ctx: *qwen35.Context, toks: []const u32) void {
        ctx.espSeed(toks, self.tmp);
        self.nd = 0;
    }

    /// Bir doğrulama turu: pending + varsa draftlar + k maske, tek forward. false: yer yok (çağıran tek token yoluna düşer).
    pub fn begin(self: *Esp, ctx: *qwen35.Context, pending: u32, limit_extra: usize) !bool {
        const nd = @min(self.nd, limit_extra);
        const m = 1 + nd;
        if (ctx.head + m + self.k + 1 >= ctx.n_ctx or m + self.k > ctx.trace_k) {
            self.nd = 0;
            return false;
        }
        self.buf[0] = pending;
        @memcpy(self.buf[1..m], self.drafts[0..nd]);
        self.m = m;
        self.before = ctx.head;
        self.split = nd > 0;
        if (self.split) ctx.snapshot();
        try ctx.decodeEsp(self.buf[0..m], self.k, self.split);
        return true;
    }

    /// Doğrulamadan geçen önek uzunluğu (pending dahil, en az 1).
    pub fn matched(self: *const Esp, ctx: *const qwen35.Context) usize {
        var j: usize = 0;
        while (j + 1 < self.m and ctx.verify_tokens[j] == self.buf[j + 1]) j += 1;
        return j + 1;
    }

    /// Kabul edilen acc token (buf[0..acc]) için durumu kapatır, sonraki draftı hazırlar, maskeyi günceller.
    pub fn finish(self: *Esp, ctx: *qwen35.Context, acc: usize) !void {
        self.rounds += 1;
        if (self.split) {
            self.rounds_draft += 1;
            self.drafted += self.m - 1;
            self.accepted += acc - 1;
        }
        self.tokens += acc;
        if (acc == self.m) {
            // tam kabul: durum zaten buf[m-1] sonrası; maske çıktıları doğru bağlamda
            @memcpy(self.drafts[0..self.k], ctx.verify_tokens[self.m..][0..self.k]);
            self.nd = self.k;
        } else {
            if (acc == 1 and self.split) {
                ctx.espKeepFirst(self.before);
            } else {
                ctx.rollback(self.before);
                try ctx.decode(self.buf[0..acc]);
            }
            self.nd = 0;
        }
        for (self.buf[0..acc]) |t| ctx.espUpdate(t, self.lam, self.tmp);
    }
};
