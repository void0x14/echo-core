//! qwen35 grafiği GPU'da (Vulkan compute, RX 460 hedefli). Yalnız Q8_0 + F32 tensörler.
//! Tüm durum tek arena'da; komut tamponu bir kez kaydedilir, token başına yalnız parametre yazılır.
//! Graph sırası qwen35.zig ile aynıdır; farkı n_t=1'e göre birleştirilmiş shader'lar.
const std = @import("std");
const vk = @import("vk.zig");
const ops = @import("../ops.zig");
const qwen35 = @import("../qwen35.zig");
const Mat = ops.Mat;

/// .spv dosyaları 4 bayt hizalı olmalı
fn aligned(comptime name: []const u8) []align(4) const u8 {
    const S = struct {
        const data align(4) = @embedFile("shaders/" ++ name ++ ".spv").*;
    };
    return &S.data;
}

const Wm = struct { q: u32, d: u32, rows: u32 };

const LayerGpu = struct {
    attn_norm: u32,
    post_norm: u32,
    ffn_gate: Wm,
    ffn_up: Wm,
    ffn_down: Wm,
    wq: Wm = undefined,
    wk: Wm = undefined,
    wv: Wm = undefined,
    wo: Wm = undefined,
    q_norm: u32 = 0,
    k_norm: u32 = 0,
    wqkv: Wm = undefined,
    wgate: Wm = undefined,
    alpha: Wm = undefined,
    beta: Wm = undefined,
    ssm_out: Wm = undefined,
    conv1d: u32 = 0,
    dt: u32 = 0,
    a: u32 = 0,
    ssm_norm: u32 = 0,
    // durum
    conv_st: u32 = 0,
    gdn_st: u32 = 0,
    kv_k: u32 = 0,
    kv_v: u32 = 0,
};

const Params = extern struct { token: u32, head: u32, n_cells: u32, mode: u32, pos: f32, pad1: [11]u32 = @splat(0), ix: u32, iy: u32, iz: u32 };

pub const Engine = struct {
    allocator: std.mem.Allocator,
    c: vk.Ctx,
    model: *const qwen35.Model,
    arena: vk.Buffer,
    staging: vk.Buffer,
    params_buf: vk.Buffer,
    cmd: vk.Disp,
    cmd_up: vk.Disp,
    cmd_clear: vk.Disp,
    cmd_fetch: vk.Disp,
    cmd_out: vk.Disp,
    params: *volatile Params,
    n_ctx: usize,
    head: usize = 0,
    logits: []f32,
    logits_off: u32 = 0,
    outlist: u32 = 0,
    rnd: u32 = 0,
    state_off: u64 = 0,
    state_size: u64 = 0,
    prof: bool = false,
    qpool: vk.H = 0,
    prof_labels: [PROF_MAX][]const u8 = undefined,
    prof_n: u32 = 0,

    const PROF_MAX = 1024;
    const STAGING: u64 = 64 << 20;

    const Bump = struct {
        off: u64 = 0,
        fn get(self: *Bump, bytes: u64) u32 {
            self.off = (self.off + 63) & ~@as(u64, 63);
            const r = self.off;
            self.off += bytes;
            return @intCast(r / 4);
        }
    };

    fn wmSize(w: Mat) u64 {
        return w.m * w.k + 64 + w.m * (w.k / 32) * 2 + 64;
    }

    fn flush(self: *Engine, n: u64, dst: u64) !void {
        const c = &self.c;
        const cb = self.cmd_up;
        try c.begin(cb);
        const r = vk.BufferCopy{ .src = 0, .dst = dst, .size = n };
        c.f.vkCmdCopyBuffer(cb, self.staging.handle, self.arena.handle, 1, &r);
        try c.end(cb);
        try c.submitWait(cb);
    }

    fn uploadBytes(self: *Engine, src: [*]const u8, n: u64, dst_word: u32) !void {
        var done: u64 = 0;
        const dst: u64 = @as(u64, dst_word) * 4;
        while (done < n) {
            const m = @min(n - done, STAGING);
            @memcpy(self.staging.map.?[0..m], src[done..][0..m]);
            try self.flush(m, dst + done);
            done += m;
        }
    }

    fn uploadMat(self: *Engine, w: Mat, b: *Bump) !Wm {
        if (w.type != .q8_0) return error.GpuYalnizQ80;
        if (w.k % 32 != 0) return error.GpuBoyut;
        const nb = w.k / 32;
        const q = b.get(w.m * w.k);
        const d = b.get(w.m * nb * 2);
        // q: işaretli byte; d: u16
        var r: usize = 0;
        var dst: u64 = @as(u64, q) * 4;
        while (r < w.m) {
            const n = @min(w.m - r, STAGING / w.k);
            const out = self.staging.map.?;
            for (0..n) |i| {
                const row = w.data + (r + i) * w.nb1;
                for (0..nb) |bi| {
                    @memcpy((out + i * w.k + bi * 32)[0..32], (row + bi * 34 + 2)[0..32]);
                }
            }
            try self.flush(n * w.k, dst);
            dst += n * w.k;
            r += n;
        }
        r = 0;
        dst = @as(u64, d) * 4;
        while (r < w.m) {
            const n = @min(w.m - r, STAGING / (nb * 2));
            const out: [*]align(1) u16 = @ptrCast(self.staging.map.?);
            for (0..n) |i| {
                const row = w.data + (r + i) * w.nb1;
                for (0..nb) |bi| out[i * nb + bi] = @as(*align(1) const u16, @ptrCast(row + bi * 34)).*;
            }
            try self.flush(n * nb * 2, dst);
            dst += n * nb * 2;
            r += n;
        }
        return .{ .q = q, .d = d, .rows = @intCast(w.m) };
    }

    fn uploadVec(self: *Engine, p: [*]const f32, n: usize, b: *Bump) !u32 {
        const o = b.get(n * 4);
        try self.uploadBytes(@ptrCast(p), n * 4, o);
        return o;
    }

    pub fn init(model: *const qwen35.Model, n_ctx: usize, prof: bool, allocator: std.mem.Allocator) !*Engine {
        const m = model;
        if (m.n_embd != 1024 or m.d_inner != 2048 or m.d_state != 128 or m.n_group != 16 or m.head_dim != 256 or m.n_head != 8 or m.n_head_kv != 2 or m.n_ff != 3584 or m.d_conv != 4 or m.n_rot != 64) {
            return error.GpuBoyut;
        }
        var total: u64 = 4096;
        total += wmSize(m.tok_embd);
        for (m.layers) |*l| {
            total += wmSize(l.ffn_gate) + wmSize(l.ffn_up) + wmSize(l.ffn_down);
            if (l.recr) {
                total += wmSize(l.wqkv) + wmSize(l.wgate) + wmSize(l.ssm_alpha) + wmSize(l.ssm_beta) + wmSize(l.ssm_out);
                total += 6144 * 4 * 4 + 16 * 8 + 128 * 4 + 6144 * 3 * 4 + 128 * 128 * 16 * 4 + 1024 * 4 + 1024;
            } else {
                total += wmSize(l.wq) + wmSize(l.wk) + wmSize(l.wv) + wmSize(l.wo);
                total += @as(u64, n_ctx) * 512 * 4 * 2 + 256 * 8 + 1024;
            }
            total += 1024 * 4 * 2 + 256;
        }
        total += @as(u64, n_ctx) * 8 * 4 + (1 << 20); // skor + iş tamponları
        total += m.n_vocab * 4 + 4096 + @as(u64, n_ctx + 1) * 8;
        total += 4 << 20;

        const self = try allocator.create(Engine);
        errdefer allocator.destroy(self);
        var c = try vk.Ctx.init();
        const arena = try c.buffer(total, vk.USAGE_STORAGE | vk.USAGE_SRC | vk.USAGE_DST, vk.MEM_LOCAL);
        const staging = try c.buffer(STAGING, vk.USAGE_STORAGE | vk.USAGE_SRC | vk.USAGE_DST, vk.MEM_VISIBLE | vk.MEM_COHERENT | vk.MEM_CACHED);
        const params_buf = try c.buffer(4096, vk.USAGE_STORAGE | vk.USAGE_INDIRECT, vk.MEM_LOCAL | vk.MEM_VISIBLE | vk.MEM_COHERENT);
        c.bind(0, arena);
        c.bind(1, params_buf);
        self.* = .{
            .allocator = allocator,
            .c = c,
            .model = model,
            .arena = arena,
            .staging = staging,
            .params_buf = params_buf,
            .cmd = try c.cmdBuffer(),
            .cmd_up = try c.cmdBuffer(),
            .cmd_clear = try c.cmdBuffer(),
            .cmd_fetch = try c.cmdBuffer(),
            .cmd_out = try c.cmdBuffer(),
            .params = @ptrCast(@alignCast(params_buf.map.?)),
            .n_ctx = n_ctx,
            .logits = try allocator.alloc(f32, m.n_vocab),
            .prof = prof,
        };
        if (prof) self.qpool = try self.c.queryPool(PROF_MAX);
        try self.build();
        self.clear();
        return self;
    }

    pub fn deinit(self: *Engine) void {
        self.c.deinit();
        self.allocator.free(self.logits);
        self.allocator.destroy(self);
    }

    fn build(self: *Engine) !void {
        const m = self.model;
        const a = self.allocator;
        var b = Bump{};
        const c = &self.c;

        // ağırlıklar
        const lm = try self.uploadMat(m.tok_embd, &b);
        const layers = try a.alloc(LayerGpu, m.layers.len);
        defer a.free(layers);
        for (m.layers, layers) |*l, *g| {
            g.* = .{
                .attn_norm = try self.uploadVec(l.attn_norm, 1024, &b),
                .post_norm = try self.uploadVec(l.post_norm, 1024, &b),
                .ffn_gate = try self.uploadMat(l.ffn_gate, &b),
                .ffn_up = try self.uploadMat(l.ffn_up, &b),
                .ffn_down = try self.uploadMat(l.ffn_down, &b),
            };
            if (l.recr) {
                g.wqkv = try self.uploadMat(l.wqkv, &b);
                g.wgate = try self.uploadMat(l.wgate, &b);
                g.alpha = try self.uploadMat(l.ssm_alpha, &b);
                g.beta = try self.uploadMat(l.ssm_beta, &b);
                g.ssm_out = try self.uploadMat(l.ssm_out, &b);
                g.conv1d = try self.uploadVec(l.ssm_conv1d, 6144 * 4, &b);
                g.dt = try self.uploadVec(l.ssm_dt, 16, &b);
                g.a = try self.uploadVec(l.ssm_a, 16, &b);
                g.ssm_norm = try self.uploadVec(l.ssm_norm, 128, &b);
            } else {
                g.wq = try self.uploadMat(l.wq, &b);
                g.wk = try self.uploadMat(l.wk, &b);
                g.wv = try self.uploadMat(l.wv, &b);
                g.wo = try self.uploadMat(l.wo, &b);
                g.q_norm = try self.uploadVec(l.q_norm, 256, &b);
                g.k_norm = try self.uploadVec(l.k_norm, 256, &b);
            }
        }
        const out_norm = try self.uploadVec(m.output_norm, 1024, &b);
        // rope açı çarpanları (ggml: theta *= theta_scale)
        const theta_scale = std.math.pow(f32, m.freq_base, -2.0 / @as(f32, @floatFromInt(m.n_rot)));
        var ts: [32]f32 = undefined;
        {
            var t: f32 = 1.0;
            for (&ts) |*x| {
                x.* = t;
                t *= theta_scale;
            }
        }
        const ts_off = try self.uploadVec(&ts, 32, &b);

        // durum (sıfırlanan bölge)
        self.state_off = (b.off + 63) & ~@as(u64, 63);
        b.off = self.state_off;
        for (m.layers, layers) |*l, *g| {
            if (l.recr) {
                g.conv_st = b.get(6144 * 3 * 4);
                g.gdn_st = b.get(128 * 128 * 16 * 4);
            }
        }
        self.state_size = b.off - self.state_off;
        for (m.layers, layers) |*l, *g| {
            if (!l.recr) {
                g.kv_k = b.get(@as(u64, self.n_ctx) * 512 * 4);
                g.kv_v = b.get(@as(u64, self.n_ctx) * 512 * 4);
            }
        }

        // iş tamponları
        const inp = b.get(1024 * 4);
        const resid = b.get(1024 * 4);
        const attn = b.get(1024 * 4);
        const nq = b.get(1024);
        const nd = b.get(32 * 4);
        const qkv = b.get(6144 * 4);
        const conv_o = b.get(6144 * 4);
        const qb = b.get(2048 * 4);
        const kb = b.get(2048 * 4);
        const alpha = b.get(16 * 4);
        const beta = b.get(16 * 4);
        const gdn_o = b.get(2048 * 4);
        const z = b.get(2048 * 4);
        const q2 = b.get(2048);
        const d2 = b.get(64 * 4);
        const g3 = b.get(3584 * 4);
        const up3 = b.get(3584 * 4);
        const q3 = b.get(3584);
        const d3 = b.get(112 * 4);
        const qfull = b.get(4096 * 4);
        const kk = b.get(512 * 4);
        const qn = b.get(2048 * 4);
        const sc = b.get(@as(u64, self.n_ctx) * 8 * 4);
        self.logits_off = b.get(m.n_vocab * 4);
        // argmax düzeyleri: 248320 -> 3880 -> 61 -> 1
        var lv_n: [8]u32 = undefined;
        var lv_v: [8]u32 = undefined;
        var lv_i: [8]u32 = undefined;
        var n_lv: usize = 0;
        {
            var n: u32 = @intCast(m.n_vocab);
            while (n > 1) {
                n = (n + 63) / 64;
                lv_n[n_lv] = n;
                lv_v[n_lv] = b.get(@as(u64, n) * 4);
                lv_i[n_lv] = b.get(@as(u64, n) * 4);
                n_lv += 1;
            }
        }
        self.outlist = b.get(@as(u64, self.n_ctx + 1) * 4);
        self.rnd = b.get(@as(u64, self.n_ctx + 1) * 4);
        if (b.off > self.arena.size) return error.GpuArenaTasti;
        std.debug.print("gpu arena: {d} MB kullanıldı\n", .{b.off / 1000000});

        // pipeline'lar
        const pm1 = try c.pipeline(aligned("mv1024"));
        const pm2 = try c.pipeline(aligned("mv2048"));
        const pm3 = try c.pipeline(aligned("mv3584"));
        const pembed = try c.pipeline(aligned("embed"));
        const padd = try c.pipeline(aligned("addnorm"));
        const pconv = try c.pipeline(aligned("conv"));
        const pl2 = try c.pipeline(aligned("l2n"));
        const pgdn = try c.pipeline(aligned("gdn"));
        const pgp = try c.pipeline(aligned("gdnpost"));
        const pffn = try c.pipeline(aligned("ffnq"));
        const pqn = try c.pipeline(aligned("qnorm"));
        const pf1 = try c.pipeline(aligned("flash1"));
        const pf2 = try c.pipeline(aligned("flash2"));
        const parg = try c.pipeline(aligned("argmax"));
        const pstep = try c.pipeline(aligned("step"));

        const cb = self.cmd;
        const R = Rec{ .c = c, .cb = cb, .e = self };
        const eps = m.eps;
        const s_f: f32 = @floatFromInt(m.d_state);
        const eps_l2 = eps / s_f;
        const l2 = 1.0 / @sqrt(s_f);
        const ctxs: u32 = @intCast(self.n_ctx);

        try c.beginFlags(cb, 4);
        if (self.prof) c.f.vkCmdResetQueryPool(cb, self.qpool, 0, PROF_MAX);
        R.lab("bas");
        R.go(pembed, .{ lm.q, lm.d, inp }, 1, 1);
        R.bar();
        R.lab("embed");
        // z = inp + 0
        R.go(padd, AddPc{ .a = inp, .b = inp, .bs = 0, .w = layers[0].attn_norm, .z = inp, .nq = nq, .nd = nd, .eps = eps }, 1, 1);
        R.bar();
        R.lab("addnorm");
        for (m.layers, layers, 0..) |*l, *g, il| {
            if (l.recr) {
                R.go(pm1, MvPc{ .q = g.wqkv.q, .d = g.wqkv.d, .x = nq, .xd = nd, .y = qkv }, g.wqkv.rows, 1);
                R.go(pm1, MvPc{ .q = g.alpha.q, .d = g.alpha.d, .x = nq, .xd = nd, .y = alpha }, g.alpha.rows, 1);
                R.go(pm1, MvPc{ .q = g.beta.q, .d = g.beta.d, .x = nq, .xd = nd, .y = beta }, g.beta.rows, 1);
                R.go(pm1, MvPc{ .q = g.wgate.q, .d = g.wgate.d, .x = nq, .xd = nd, .y = z }, g.wgate.rows, 1);
                R.bar();
                R.lab("mv_qkv_ab_wgate");
                R.go(pconv, .{ qkv, g.conv_st, g.conv1d, conv_o }, 96, 1);
                R.bar();
                R.lab("conv");
                R.go(pl2, L2Pc{ .src = conv_o, .dst = qb, .eps = eps_l2, .l2 = l2 }, 16, 1);
                R.go(pl2, L2Pc{ .src = conv_o + 2048, .dst = kb, .eps = eps_l2, .l2 = l2 }, 16, 1);
                R.bar();
                R.lab("l2n");
                R.go(pgdn, GdnPc{ .st = g.gdn_st, .k = kb, .q = qb, .vv = conv_o + 4096, .alpha = alpha, .beta = beta, .dt = g.dt, .a = g.a, .o = gdn_o, .sc = l2 }, 2048, 1);
                R.bar();
                R.lab("gdn");
                R.go(pgp, GpPc{ .g = gdn_o, .w = g.ssm_norm, .z = z, .nq = q2, .nd = d2, .eps = eps }, 1, 1);
                R.bar();
                R.lab("gdnpost");
                R.go(pm2, MvPc{ .q = g.ssm_out.q, .d = g.ssm_out.d, .x = q2, .xd = d2, .y = attn }, g.ssm_out.rows, 1);
                R.bar();
                R.lab("mv_ssm_out");
            } else {
                R.go(pm1, MvPc{ .q = g.wq.q, .d = g.wq.d, .x = nq, .xd = nd, .y = qfull }, g.wq.rows, 1);
                R.go(pm1, MvPc{ .q = g.wk.q, .d = g.wk.d, .x = nq, .xd = nd, .y = kk }, g.wk.rows, 1);
                R.go(pm1, MvPc{ .q = g.wv.q, .d = g.wv.d, .x = nq, .xd = nd, .y = g.kv_v, .y_dyn = 512 }, g.wv.rows, 1);
                R.bar();
                R.lab("mv_wq_wk_wv");
                R.go(pqn, QnPc{ .src = qfull, .stride = 512, .dst = qn, .dstride = 256, .dyn = 0, .w = g.q_norm, .ts = ts_off, .eps = eps }, 8, 1);
                R.go(pqn, QnPc{ .src = kk, .stride = 256, .dst = g.kv_k, .dstride = 256, .dyn = 512, .w = g.k_norm, .ts = ts_off, .eps = eps }, 2, 1);
                R.bar();
                R.lab("qnorm");
                c.f.vkCmdBindPipeline(cb, vk.BIND_COMPUTE, pf1);
                R.push(F1Pc{ .qn = qn, .k = g.kv_k, .s = sc, .stride = ctxs, .scale = 1.0 / 16.0 });
                c.f.vkCmdDispatchIndirect(cb, self.params_buf.handle, @offsetOf(Params, "ix"));
                R.bar();
                R.lab("flash1");
                R.go(pf2, .{ g.kv_v, sc, ctxs, qfull, q2, d2 }, 8, 1);
                R.bar();
                R.lab("flash2");
                R.go(pm2, MvPc{ .q = g.wo.q, .d = g.wo.d, .x = q2, .xd = d2, .y = attn }, g.wo.rows, 1);
                R.bar();
                R.lab("mv_wo");
            }
            // attn_residual + attn_post_norm
            R.go(padd, AddPc{ .a = attn, .b = inp, .bs = 1, .w = g.post_norm, .z = resid, .nq = nq, .nd = nd, .eps = eps }, 1, 1);
            R.bar();
            R.lab("addnorm");
            R.go(pm1, MvPc{ .q = g.ffn_gate.q, .d = g.ffn_gate.d, .x = nq, .xd = nd, .y = g3 }, g.ffn_gate.rows, 1);
            R.go(pm1, MvPc{ .q = g.ffn_up.q, .d = g.ffn_up.d, .x = nq, .xd = nd, .y = up3 }, g.ffn_up.rows, 1);
            R.bar();
            R.lab("mv_ffn_gate_up");
            R.go(pffn, .{ g3, up3, q3, d3 }, 1, 1);
            R.bar();
            R.lab("ffnq");
            R.go(pm3, MvPc{ .q = g.ffn_down.q, .d = g.ffn_down.d, .x = q3, .xd = d3, .y = attn }, g.ffn_down.rows, 1);
            R.bar();
            R.lab("mv_ffn_down");
            // l_out + sonraki katmanın normu
            const next_w = if (il + 1 < layers.len) layers[il + 1].attn_norm else out_norm;
            R.go(padd, AddPc{ .a = attn, .b = resid, .bs = 1, .w = next_w, .z = inp, .nq = nq, .nd = nd, .eps = eps }, 1, 1);
            R.bar();
            R.lab("addnorm");
        }
        R.go(pm1, MvPc{ .q = lm.q, .d = lm.d, .x = nq, .xd = nd, .y = self.logits_off }, lm.rows, 1);
        R.bar();
        R.lab("mv_lm_head");
        {
            var sv = self.logits_off;
            var si: u32 = 0;
            var n: u32 = @intCast(m.n_vocab);
            for (0..n_lv) |i| {
                R.go(parg, .{ sv, si, lv_v[i], lv_i[i], n }, lv_n[i], 1);
                R.bar();
                sv = lv_v[i];
                si = lv_i[i];
                n = lv_n[i];
            }
            R.lab("argmax");
            R.go(pstep, .{ si, self.outlist, self.rnd }, 1, 1);
            c.barrier(cb, vk.PIPE_COMPUTE, vk.PIPE_COMPUTE | vk.PIPE_INDIRECT, vk.ACC_SHADER_WRITE, vk.ACC_SHADER_READ | vk.ACC_INDIRECT);
            R.lab("step");
        }
        try c.end(cb);

        // logit'leri CPU'ya alma (yalnız istenince)
        try c.begin(self.cmd_fetch);
        const rc = vk.BufferCopy{ .src = @as(u64, self.logits_off) * 4, .dst = 0, .size = m.n_vocab * 4 };
        c.f.vkCmdCopyBuffer(self.cmd_fetch, self.arena.handle, self.staging.handle, 1, &rc);
        c.barrier(self.cmd_fetch, vk.PIPE_TRANSFER, vk.PIPE_HOST, vk.ACC_XFER_WRITE, vk.ACC_HOST_READ);
        try c.end(self.cmd_fetch);

        // üretilen tokenleri CPU'ya alma
        try c.begin(self.cmd_out);
        const ro = vk.BufferCopy{ .src = @as(u64, self.outlist) * 4, .dst = 0, .size = @as(u64, self.n_ctx) * 4 };
        c.f.vkCmdCopyBuffer(self.cmd_out, self.arena.handle, self.staging.handle, 1, &ro);
        c.barrier(self.cmd_out, vk.PIPE_TRANSFER, vk.PIPE_HOST, vk.ACC_XFER_WRITE, vk.ACC_HOST_READ);
        try c.end(self.cmd_out);

        // sıfırlama
        try c.begin(self.cmd_clear);
        c.f.vkCmdFillBuffer(self.cmd_clear, self.arena.handle, self.state_off, self.state_size, 0);
        c.barrier(self.cmd_clear, vk.PIPE_TRANSFER, vk.PIPE_COMPUTE, vk.ACC_XFER_WRITE, vk.ACC_SHADER_READ | vk.ACC_SHADER_WRITE);
        try c.end(self.cmd_clear);
    }

    pub fn clear(self: *Engine) void {
        self.head = 0;
        self.c.submitWait(self.cmd_clear) catch @panic("gpu clear");
    }

    pub fn decode1(self: *Engine, token: u32) !void {
        if (self.head + 1 > self.n_ctx) return error.ContextFull;
        if (token >= self.model.n_vocab) return error.InvalidToken;
        const n = self.head + 1;
        self.params.* = .{
            .token = token,
            .head = @intCast(self.head),
            .n_cells = @intCast(n),
            .mode = 0,
            .pos = @floatFromInt(self.head),
            .ix = @intCast((n + 63) / 64),
            .iy = 8,
            .iz = 1,
        };
        try self.c.submitWait(self.cmd);
        self.head = n;
    }

    /// Son decode1'in logit'lerini CPU'ya alır.
    pub fn fetchLogits(self: *Engine) ![]const f32 {
        try self.c.submitWait(self.cmd_fetch);
        const y: [*]const f32 = @ptrCast(@alignCast(self.staging.map.?));
        @memcpy(self.logits, y[0..self.logits.len]);
        return self.logits;
    }

    /// Son decode1'den sonra GPU'nun seçtiği sonraki token (mode 0: argmax).
    pub fn nextToken(self: *Engine) u32 {
        return self.params.token;
    }

    /// `first` tokeninden başlayıp n token üretir; GPU kendi kendine sürer, CPU yalnız gönderim başına bir kez uyanır.
    /// mode 0: greedy (argmax geri beslenir), mode 1: `setRandom` listesi.
    pub fn chain(self: *Engine, first: u32, n: usize, mode: u32, batch: usize) !void {
        if (self.head + n > self.n_ctx) return error.ContextFull;
        const h = self.head;
        self.params.* = .{
            .token = first,
            .head = @intCast(h),
            .n_cells = @intCast(h + 1),
            .mode = mode,
            .pos = @floatFromInt(h),
            .ix = @intCast((h + 1 + 63) / 64),
            .iy = 8,
            .iz = 1,
        };
        var list: [256]vk.Disp = undefined;
        @memset(&list, self.cmd);
        var left = n;
        while (left > 0) {
            const k = @min(left, @min(batch, list.len));
            try self.c.submitListWait(list[0..k]);
            left -= k;
        }
        self.head = h + n;
    }

    pub fn setRandom(self: *Engine, toks: []const u32) !void {
        try self.uploadBytes(@ptrCast(toks.ptr), toks.len * 4, self.rnd);
    }

    pub fn fetchOut(self: *Engine, out: []u32) !void {
        try self.c.submitWait(self.cmd_out);
        const y: [*]const u32 = @ptrCast(@alignCast(self.staging.map.?));
        @memcpy(out, y[0..out.len]);
    }
};

const MvPc = extern struct { q: u32, d: u32, x: u32, xd: u32, y: u32, y_dyn: u32 = 0 };
const AddPc = extern struct { a: u32, b: u32, bs: f32, w: u32, z: u32, nq: u32, nd: u32, eps: f32 };
const L2Pc = extern struct { src: u32, dst: u32, eps: f32, l2: f32 };
const GdnPc = extern struct { st: u32, k: u32, q: u32, vv: u32, alpha: u32, beta: u32, dt: u32, a: u32, o: u32, sc: f32 };
const GpPc = extern struct { g: u32, w: u32, z: u32, nq: u32, nd: u32, eps: f32 };
const QnPc = extern struct { src: u32, stride: u32, dst: u32, dstride: u32, dyn: u32, w: u32, ts: u32, eps: f32 };
const F1Pc = extern struct { qn: u32, k: u32, s: u32, stride: u32, scale: f32 };

const Rec = struct {
    c: *const vk.Ctx,
    cb: vk.Disp,
    e: *Engine,

    /// Şu ana kadarki işi `name` etiketiyle bitirir (yalnız --gpu-prof)
    fn lab(self: Rec, name: []const u8) void {
        if (!self.e.prof) return;
        self.c.f.vkCmdWriteTimestamp(self.cb, 0x800, self.e.qpool, self.e.prof_n);
        self.e.prof_labels[self.e.prof_n] = name;
        self.e.prof_n += 1;
    }

    fn push(self: Rec, pc: anytype) void {
        const T = @TypeOf(pc);
        if (@typeInfo(T).@"struct".is_tuple) {
            var arr: [pc.len]u32 = undefined;
            inline for (pc, 0..) |v, i| arr[i] = v;
            self.c.f.vkCmdPushConstants(self.cb, self.c.layout, vk.STAGE_COMPUTE, 0, @sizeOf(@TypeOf(arr)), &arr);
        } else {
            const copy = pc;
            self.c.f.vkCmdPushConstants(self.cb, self.c.layout, vk.STAGE_COMPUTE, 0, @sizeOf(T), &copy);
        }
    }
    fn go(self: Rec, p: vk.H, pc: anytype, x: u32, y: u32) void {
        self.c.f.vkCmdBindPipeline(self.cb, vk.BIND_COMPUTE, p);
        self.push(pc);
        self.c.f.vkCmdDispatch(self.cb, x, y, 1);
    }
    fn bar(self: Rec) void {
        self.c.barrier(self.cb, vk.PIPE_COMPUTE, vk.PIPE_COMPUTE, vk.ACC_SHADER_WRITE, vk.ACC_SHADER_READ);
    }
};

// -------- ölçüm ve doğrulama --------

/// --gpu-check: aynı token dizisinde GPU ve CPU logit'lerini karşılaştırır.
pub fn check(path: []const u8, allocator: std.mem.Allocator, n_tok: usize) !void {
    const model = try qwen35.Model.load(path, allocator);
    defer model.deinit();
    const cpu = try qwen35.Context.init(model, 1024, 6, allocator);
    defer cpu.deinit();
    const gpu = try Engine.init(model, 1024, false, allocator);
    defer gpu.deinit();
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    var worst: f64 = 0;
    var argmax_same: usize = 0;
    var top5_sum: usize = 0;
    for (0..n_tok) |i| {
        const t: u32 = @intCast(rng.uintLessThan(usize, model.n_vocab));
        const tk = [1]u32{t};
        try cpu.decode(&tk);
        try gpu.decode1(t);
        const a = cpu.logits;
        const b = try gpu.fetchLogits();
        var md: f64 = 0;
        for (a, b) |x, y| md = @max(md, @abs(@as(f64, x) - y));
        worst = @max(worst, md);
        const ta = top5(a);
        const tb = top5(b);
        if (ta[0] == tb[0]) argmax_same += 1;
        if (gpu.nextToken() != tb[0]) std.debug.print("GPU argmax hatası: {d} != {d}\n", .{ gpu.nextToken(), tb[0] });
        var ov: usize = 0;
        for (ta) |x| {
            for (tb) |y| {
                if (x == y) ov += 1;
            }
        }
        top5_sum += ov;
        if (i < 6 or i + 1 == n_tok) std.debug.print("tok {d}: max|fark|={d:.4} argmax cpu={d} gpu={d} top5 kesişim={d}\n", .{ i, md, ta[0], tb[0], ov });
    }
    std.debug.print("TOPLAM {d} token: en kötü max|fark|={d:.4}, argmax aynı {d}/{d}, ort. top5 kesişim {d:.2}\n", .{ n_tok, worst, argmax_same, n_tok, @as(f64, @floatFromInt(top5_sum)) / @as(f64, @floatFromInt(n_tok)) });

    // greedy üretim: aynı başlangıçtan iki dizi, ilk ayrışma noktası
    const gen_n: usize = 200;
    const starts = [_]u32{ 9707, 151644, 1169, 40, 3838, 785, 16 };
    var eq_total: usize = 0;
    for (starts) |st| {
        cpu.clear();
        gpu.clear();
        const out_g = try allocator.alloc(u32, gpu.n_ctx);
        defer allocator.free(out_g);
        try gpu.chain(st, gen_n, 0, 32);
        try gpu.fetchOut(out_g);
        var tok = st;
        var same: usize = 0;
        var div: ?usize = null;
        for (0..gen_n) |i| {
            const tk = [1]u32{tok};
            try cpu.decode(&tk);
            var best: u32 = 0;
            for (cpu.logits, 0..) |v, j| {
                if (v > cpu.logits[best]) best = @intCast(j);
            }
            if (best == out_g[i]) {
                same += 1;
            } else if (div == null) div = i;
            tok = best;
        }
        eq_total += same;
        std.debug.print("başlangıç {d}: ilk ayrışma {?d}, aynı {d}/{d}; ilk 8 GPU:", .{ st, div, same, gen_n });
        for (out_g[0..8]) |t| std.debug.print(" {d}", .{t});
        std.debug.print("\n", .{});
    }
    std.debug.print("greedy eşleşme: {d}/{d}\n", .{ eq_total, gen_n * starts.len });
}

fn top5(x: []const f32) [5]u32 {
    var best: [5]u32 = @splat(0);
    var val: [5]f32 = @splat(-std.math.inf(f32));
    for (x, 0..) |v, i| {
        if (v <= val[4]) continue;
        var j: usize = 4;
        while (j > 0 and v > val[j - 1]) : (j -= 1) {
            val[j] = val[j - 1];
            best[j] = best[j - 1];
        }
        val[j] = v;
        best[j] = @intCast(i);
    }
    return best;
}

/// --gpu-bench: llama-bench döngüsü (taze context, warmup, rastgele token); çıktı aynı JSON alanları.
pub fn bench(path: []const u8, reps: usize, gens: []const usize, allocator: std.mem.Allocator, io: std.Io, out: *std.Io.Writer) !void {
    const model = try qwen35.Model.load(path, allocator);
    defer model.deinit();
    var max_gen: usize = 1;
    for (gens) |g| max_gen = @max(max_gen, g);
    const e = try Engine.init(model, (max_gen + 256) / 256 * 256, false, allocator);
    defer e.deinit();
    var prng = std.Random.DefaultPrng.init(1234);
    const rng = prng.random();
    const rnd = try allocator.alloc(u32, e.n_ctx + 1);
    defer allocator.free(rnd);
    const Mode = struct { name: []const u8, mode: u32, batch: usize };
    const modes = [_]Mode{
        .{ .name = "sync_logit", .mode = 2, .batch = 1 },
        .{ .name = "zincir_rastgele", .mode = 1, .batch = 32 },
        .{ .name = "zincir_greedy", .mode = 0, .batch = 32 },
    };
    for (modes) |md| for (gens) |n| {
        for (rnd) |*t| t.* = @intCast(rng.uintLessThan(usize, model.n_vocab));
        try e.setRandom(rnd);
        e.clear();
        try e.chain(rnd[0], 1, 1, 1);
        const samples = try allocator.alloc(f64, reps);
        defer allocator.free(samples);
        for (samples) |*s| {
            e.clear();
            const t0 = std.Io.Timestamp.now(io, .awake);
            if (md.mode == 2) {
                for (0..n) |i| {
                    try e.decode1(rnd[i]);
                    _ = try e.fetchLogits();
                }
            } else {
                try e.chain(rnd[0], n, md.mode, md.batch);
            }
            const ns: f64 = @floatFromInt(t0.untilNow(io, .awake).toNanoseconds());
            s.* = @as(f64, @floatFromInt(n)) * 1e9 / ns;
        }
        var sum: f64 = 0;
        for (samples) |s| sum += s;
        try out.print("{{\"gpu\":\"{s}\",\"n_prompt\":0,\"n_gen\":{d},\"avg_ts\":{d:.3},\"samples_ts\":[", .{ md.name, n, sum / @as(f64, @floatFromInt(reps)) });
        for (samples, 0..) |s, i| {
            if (i > 0) try out.writeAll(",");
            try out.print("{d:.3}", .{s});
        }
        try out.writeAll("]}\n");
        try out.flush();
    };
}

/// --gpu-prof: bir tokenin GPU zaman dökümü (zaman damgası 40 ns: RX 460 için `vulkaninfo`)
pub fn profile(path: []const u8, ns_per_tick: f64, allocator: std.mem.Allocator, n: usize) !void {
    const model = try qwen35.Model.load(path, allocator);
    defer model.deinit();
    const e = try Engine.init(model, 1024, true, allocator);
    defer e.deinit();
    var prng = std.Random.DefaultPrng.init(7);
    const rng = prng.random();
    var totals: std.StringArrayHashMapUnmanaged(f64) = .empty;
    defer totals.deinit(allocator);
    const qs = try allocator.alloc(u64, e.prof_n);
    defer allocator.free(qs);
    for (0..n) |_| {
        try e.decode1(@intCast(rng.uintLessThan(usize, model.n_vocab)));
        const r = e.c.f.vkGetQueryPoolResults(e.c.dev, e.qpool, 0, e.prof_n, qs.len * 8, qs.ptr, 8, 1 | 2);
        if (r != 0) return error.VulkanFailed;
        for (1..qs.len) |i| {
            const g = try totals.getOrPutValue(allocator, e.prof_labels[i], 0);
            g.value_ptr.* += @as(f64, @floatFromInt(qs[i] - qs[i - 1])) * ns_per_tick / 1e6;
        }
    }
    var tot: f64 = 0;
    for (totals.values()) |v| tot += v;
    var it = totals.iterator();
    while (it.next()) |en| std.debug.print("{s:<18} {d:7.3} ms/tok  %{d:.1}\n", .{ en.key_ptr.*, en.value_ptr.* / @as(f64, @floatFromInt(n)), en.value_ptr.* * 100 / tot });
    std.debug.print("TOPLAM (GPU) {d:.3} ms/tok ({d} etiket)\n", .{ tot / @as(f64, @floatFromInt(n)), e.prof_n });
}
