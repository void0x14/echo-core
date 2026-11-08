//! Token seçimi: ceza (repeat/presence) → top-k → sıcaklık → softmax → top-p → min-p → çekiliş.
//! temp <= 0: argmax (greedy).
const std = @import("std");

pub const Params = struct {
    temp: f32 = 0.7,
    top_k: u32 = 20,
    top_p: f32 = 0.8,
    min_p: f32 = 0.0,
    repeat_penalty: f32 = 1.0,
    presence_penalty: f32 = 0.0,
    /// ceza için bakılan son token sayısı
    penalty_last_n: usize = 64,
    seed: u64 = 0,
};

/// shortcut: top_k=0 (kapalı) en çok MAX_K adayla çalışır; tam sözlük sıralaması gerekirse büyütülür.
const MAX_K = 1024;

pub const Sampler = struct {
    p: Params,
    rng: std.Random.DefaultPrng,
    cand_id: [MAX_K]u32 = undefined,
    cand_v: [MAX_K]f32 = undefined,

    pub fn init(p: Params) Sampler {
        return .{ .p = p, .rng = std.Random.DefaultPrng.init(p.seed) };
    }

    pub fn argmax(logits: []const f32) u32 {
        var best: usize = 0;
        for (logits, 0..) |v, i| if (v > logits[best]) {
            best = i;
        };
        return @intCast(best);
    }

    /// `logits` cezalar için yerinde değiştirilir. `recent`: son üretilen/okunan tokenlar.
    pub fn sample(self: *Sampler, logits: []f32, recent: []const u32) u32 {
        const p = self.p;
        if (p.repeat_penalty != 1.0 or p.presence_penalty != 0.0) {
            const from = recent.len -| p.penalty_last_n;
            for (recent[from..], from..) |t, i| {
                if (std.mem.indexOfScalar(u32, recent[from..i], t) != null) continue; // her token bir kez
                const v = &logits[t];
                v.* = if (v.* > 0) v.* / p.repeat_penalty else v.* * p.repeat_penalty;
                v.* -= p.presence_penalty;
            }
        }
        if (p.temp <= 0) return argmax(logits);

        // top-k: k elemanlı küçük dizide en büyükleri tut (azalan sıralı)
        const k: usize = if (p.top_k == 0) MAX_K else @min(p.top_k, MAX_K);
        var n: usize = 0;
        for (logits, 0..) |v, i| {
            if (n == k and v <= self.cand_v[k - 1]) continue;
            var j = if (n < k) n else k - 1;
            if (n < k) n += 1;
            while (j > 0 and self.cand_v[j - 1] < v) : (j -= 1) {
                self.cand_v[j] = self.cand_v[j - 1];
                self.cand_id[j] = self.cand_id[j - 1];
            }
            self.cand_v[j] = v;
            self.cand_id[j] = @intCast(i);
        }
        // sıcaklık + softmax
        const mx = self.cand_v[0];
        var sum: f32 = 0;
        for (self.cand_v[0..n]) |*v| {
            v.* = @exp((v.* - mx) / p.temp);
            sum += v.*;
        }
        for (self.cand_v[0..n]) |*v| v.* /= sum;
        // top-p
        if (p.top_p < 1.0) {
            var c: f32 = 0;
            var m: usize = 0;
            while (m < n) : (m += 1) {
                c += self.cand_v[m];
                if (c >= p.top_p) {
                    m += 1;
                    break;
                }
            }
            n = @max(m, 1);
        }
        // min-p
        if (p.min_p > 0) {
            const lim = self.cand_v[0] * p.min_p;
            var m: usize = 1;
            while (m < n and self.cand_v[m] >= lim) m += 1;
            n = m;
        }
        var total: f32 = 0;
        for (self.cand_v[0..n]) |v| total += v;
        var r = self.rng.random().float(f32) * total;
        for (self.cand_v[0..n], self.cand_id[0..n]) |v, id| {
            r -= v;
            if (r <= 0) return id;
        }
        return self.cand_id[n - 1];
    }
};

test "sampler: greedy ve top-k=1 argmax verir, aynı tohum aynı sonuç" {
    var l = [_]f32{ 0.1, 3.0, 2.5, -1.0, 2.9 };
    var s = Sampler.init(.{ .temp = 0 });
    try std.testing.expectEqual(@as(u32, 1), s.sample(&l, &.{}));
    var s1 = Sampler.init(.{ .temp = 1.0, .top_k = 1 });
    try std.testing.expectEqual(@as(u32, 1), s1.sample(&l, &.{}));
    var a = Sampler.init(.{ .temp = 1.0, .top_k = 0, .top_p = 1.0, .seed = 42 });
    var b = Sampler.init(.{ .temp = 1.0, .top_k = 0, .top_p = 1.0, .seed = 42 });
    for (0..20) |_| {
        var l1 = l;
        var l2 = l;
        try std.testing.expectEqual(a.sample(&l1, &.{}), b.sample(&l2, &.{}));
    }
    // repeat penalty: tekrar eden en iyi token cezalanır
    var l3 = [_]f32{ 0.0, 3.0, 2.9 };
    var c = Sampler.init(.{ .temp = 0, .repeat_penalty = 1.5 });
    try std.testing.expectEqual(@as(u32, 2), c.sample(&l3, &.{1}));
}
