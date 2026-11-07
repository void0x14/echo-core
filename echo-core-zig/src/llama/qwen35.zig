//! Port: llama.cpp src/models/qwen35.cpp + delta-net-base.cpp graph'ı (CPU, tek sequence).
//! Her `p.barrier()` bir ggml graph node sınırıdır (ggml_graph_compute_thread).
//! n_rs_seq = 0, flash attention ve fused GDN açık: llama-bench varsayılan yolu.
const std = @import("std");
const gguf = @import("../gguf/reader.zig");
const avx = @import("avx.zig");
const vec = @import("vec.zig");
const q = @import("quants.zig");
const ops = @import("ops.zig");
const repack = @import("repack.zig");
const sgemm = @import("sgemm.zig");
const ivf = @import("ivf.zig");
const pool_mod = @import("pool.zig");
const Pool = pool_mod.Pool;
const Params = pool_mod.Params;
const Mat = ops.Mat;

pub const supported = avx.supported;

const Alloc = std.mem.Allocator;

pub const Layer = struct {
    recr: bool,
    attn_norm: [*]const f32,
    post_norm: [*]const f32,
    ffn_gate: Mat,
    ffn_up: Mat,
    ffn_down: Mat,
    // tam attention
    wq: Mat = undefined,
    wk: Mat = undefined,
    wv: Mat = undefined,
    wo: Mat = undefined,
    q_norm: [*]const f32 = undefined,
    k_norm: [*]const f32 = undefined,
    // linear attention (gated delta net)
    wqkv: Mat = undefined,
    wgate: Mat = undefined,
    ssm_alpha: Mat = undefined,
    ssm_beta: Mat = undefined,
    ssm_out: Mat = undefined,
    ssm_conv1d: [*]const f32 = undefined,
    ssm_dt: [*]const f32 = undefined,
    ssm_a: [*]const f32 = undefined,
    ssm_norm: [*]const f32 = undefined,
};

pub const Model = struct {
    allocator: Alloc,
    map: []align(std.heap.page_size_min) const u8,
    repacked: std.ArrayList([]repack.BlockQ4Kx8) = .empty,
    repacked8: std.ArrayList([]align(64) u8) = .empty,
    layers: []Layer,
    tok_embd: Mat,
    output: Mat,
    output_norm: [*]const f32,
    n_vocab: usize,
    n_embd: usize,
    n_ff: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    n_rot: usize,
    freq_base: f32,
    eps: f32,
    d_conv: usize,
    d_inner: usize,
    d_state: usize,
    n_group: usize,
    dt_rank: usize,

    fn metaU(reader: *const gguf.Reader, comptime key: []const u8) !usize {
        const v = reader.metadata.get("qwen35." ++ key) orelse return error.MissingMetadata;
        return switch (v) {
            .uint => |x| @intCast(x),
            .int => |x| @intCast(x),
            else => error.BadMetadata,
        };
    }
    fn metaF(reader: *const gguf.Reader, comptime key: []const u8) !f32 {
        const v = reader.metadata.get("qwen35." ++ key) orelse return error.MissingMetadata;
        return switch (v) {
            .float => |x| @floatCast(x),
            else => error.BadMetadata,
        };
    }

    pub fn isQwen35(path: []const u8, allocator: Alloc) bool {
        var reader = gguf.Reader.openWithAllocator(path, allocator) catch return false;
        defer reader.deinit();
        const v = reader.metadata.get("general.architecture") orelse return false;
        return switch (v) {
            .string => |s| std.mem.eql(u8, s, "qwen35"),
            else => false,
        };
    }

    pub fn load(path: []const u8, allocator: Alloc) !*Model {
        try avx.initLibm();
        var reader = try gguf.Reader.openWithAllocator(path, allocator);
        defer reader.deinit();

        const stat_size = try fileSize(path);
        const map = try std.posix.mmap(null, stat_size, .{ .READ = true }, .{ .TYPE = .PRIVATE, .POPULATE = true }, reader.file.handle, 0);
        errdefer std.posix.munmap(map);

        const self = try allocator.create(Model);
        errdefer allocator.destroy(self);

        const block_count = try metaU(&reader, "block_count");
        const nextn = metaU(&reader, "nextn_predict_layers") catch 0;
        const n_layer = block_count - nextn;
        const interval = metaU(&reader, "full_attention_interval") catch 4;
        const n_embd = try metaU(&reader, "embedding_length");

        const ctx = Loader{ .reader = &reader, .map = map };

        const tok_embd = try ctx.mat("token_embd.weight");
        const output = ctx.mat("output.weight") catch tok_embd;

        self.* = .{
            .allocator = allocator,
            .map = map,
            .layers = try allocator.alloc(Layer, n_layer),
            .tok_embd = tok_embd,
            .output = output,
            .output_norm = try ctx.vec1("output_norm.weight"),
            .n_vocab = tok_embd.m,
            .n_embd = n_embd,
            .n_ff = try metaU(&reader, "feed_forward_length"),
            .n_head = try metaU(&reader, "attention.head_count"),
            .n_head_kv = try metaU(&reader, "attention.head_count_kv"),
            .head_dim = try metaU(&reader, "attention.key_length"),
            .n_rot = try metaU(&reader, "rope.dimension_count"),
            .freq_base = try metaF(&reader, "rope.freq_base"),
            .eps = try metaF(&reader, "attention.layer_norm_rms_epsilon"),
            .d_conv = try metaU(&reader, "ssm.conv_kernel"),
            .d_inner = try metaU(&reader, "ssm.inner_size"),
            .d_state = try metaU(&reader, "ssm.state_size"),
            .n_group = try metaU(&reader, "ssm.group_count"),
            .dt_rank = try metaU(&reader, "ssm.time_step_rank"),
        };
        errdefer allocator.free(self.layers);

        // fused GDN: H_v % H_k == 0; bu port yalnız eşit başlık sayısını destekler.
        if (self.n_group != self.dt_rank) return error.UnsupportedGdnHeads;

        var nb: [96]u8 = undefined;
        for (self.layers, 0..) |*l, il| {
            const recr = (il + 1) % interval != 0;
            l.* = .{
                .recr = recr,
                .attn_norm = try ctx.vec1(try std.fmt.bufPrint(&nb, "blk.{d}.attn_norm.weight", .{il})),
                .post_norm = try ctx.vec1(try std.fmt.bufPrint(&nb, "blk.{d}.post_attention_norm.weight", .{il})),
                .ffn_gate = try ctx.mat(try std.fmt.bufPrint(&nb, "blk.{d}.ffn_gate.weight", .{il})),
                .ffn_up = try ctx.mat(try std.fmt.bufPrint(&nb, "blk.{d}.ffn_up.weight", .{il})),
                .ffn_down = try ctx.mat(try std.fmt.bufPrint(&nb, "blk.{d}.ffn_down.weight", .{il})),
            };
            if (recr) {
                l.wqkv = try ctx.mat(try std.fmt.bufPrint(&nb, "blk.{d}.attn_qkv.weight", .{il}));
                l.wgate = try ctx.mat(try std.fmt.bufPrint(&nb, "blk.{d}.attn_gate.weight", .{il}));
                l.ssm_alpha = try ctx.mat(try std.fmt.bufPrint(&nb, "blk.{d}.ssm_alpha.weight", .{il}));
                l.ssm_beta = try ctx.mat(try std.fmt.bufPrint(&nb, "blk.{d}.ssm_beta.weight", .{il}));
                l.ssm_out = try ctx.mat(try std.fmt.bufPrint(&nb, "blk.{d}.ssm_out.weight", .{il}));
                l.ssm_conv1d = try ctx.vec1(try std.fmt.bufPrint(&nb, "blk.{d}.ssm_conv1d.weight", .{il}));
                l.ssm_dt = try ctx.vec1(try std.fmt.bufPrint(&nb, "blk.{d}.ssm_dt.bias", .{il}));
                l.ssm_a = try ctx.vec1(try std.fmt.bufPrint(&nb, "blk.{d}.ssm_a", .{il}));
                l.ssm_norm = try ctx.vec1(try std.fmt.bufPrint(&nb, "blk.{d}.ssm_norm.weight", .{il}));
            } else {
                l.wq = try ctx.mat(try std.fmt.bufPrint(&nb, "blk.{d}.attn_q.weight", .{il}));
                l.wk = try ctx.mat(try std.fmt.bufPrint(&nb, "blk.{d}.attn_k.weight", .{il}));
                l.wv = try ctx.mat(try std.fmt.bufPrint(&nb, "blk.{d}.attn_v.weight", .{il}));
                l.wo = try ctx.mat(try std.fmt.bufPrint(&nb, "blk.{d}.attn_output.weight", .{il}));
                l.q_norm = try ctx.vec1(try std.fmt.bufPrint(&nb, "blk.{d}.attn_q_norm.weight", .{il}));
                l.k_norm = try ctx.vec1(try std.fmt.bufPrint(&nb, "blk.{d}.attn_k_norm.weight", .{il}));
            }
            try self.repackLayer(l);
        }
        return self;
    }

    /// ggml-cpu repack buffer: MUL_MAT ağırlıkları yüklenirken block_q4_Kx8'e çevrilir.
    /// token_embd (GET_ROWS + bağlı LM head) repack edilmez.
    fn repackLayer(self: *Model, l: *Layer) !void {
        try self.repackMat(&l.ffn_gate);
        try self.repackMat(&l.ffn_up);
        try self.repackMat(&l.ffn_down);
        if (l.recr) {
            for ([_]*Mat{ &l.wqkv, &l.wgate, &l.ssm_alpha, &l.ssm_beta, &l.ssm_out }) |w| try self.repackMat(w);
        } else {
            for ([_]*Mat{ &l.wq, &l.wk, &l.wv, &l.wo }) |w| try self.repackMat(w);
        }
    }

    fn repackMat(self: *Model, w: *Mat) !void {
        if (w.type == .q8_0 and w.m % 8 == 0 and w.k % 64 == 0 and std.c.getenv("ECHO_NO_R8") == null) {
            const nb = w.k / 32;
            const dst = try self.allocator.alignedAlloc(u8, .@"64", w.m / 8 * nb * sgemm.R8_BLOCK);
            errdefer self.allocator.free(dst);
            // H56: 16'ya bölünen matrisler R16 (aynı bayt boyu)
            w.r16 = w.m % 16 == 0 and std.c.getenv("ECHO_NO_R16") == null;
            if (w.r16) sgemm.repackQ80x16(@ptrCast(@alignCast(w.data)), dst.ptr, w.m, nb) else sgemm.repackQ80x8(@ptrCast(@alignCast(w.data)), dst.ptr, w.m, nb);
            try self.repacked8.append(self.allocator, dst);
            w.rep = dst.ptr;
            return;
        }
        if (w.type != .q4_k or w.m % 8 != 0 or w.k % 8 != 0) return;
        const nblocks = w.k / 256;
        const dst = try self.allocator.alloc(repack.BlockQ4Kx8, w.m / 8 * nblocks);
        errdefer self.allocator.free(dst);
        repack.repackQ4K(@ptrCast(@alignCast(w.data)), dst.ptr, w.m, nblocks);
        try self.repacked.append(self.allocator, dst);
        w.data = @ptrCast(dst.ptr);
        w.type = .q4_k_8x8;
    }

    pub fn deinit(self: *Model) void {
        const allocator = self.allocator;
        allocator.free(self.layers);
        for (self.repacked.items) |b| allocator.free(b);
        self.repacked.deinit(allocator);
        for (self.repacked8.items) |b| allocator.free(b);
        self.repacked8.deinit(allocator);
        std.posix.munmap(self.map);
        allocator.destroy(self);
    }

    fn fileSize(path: []const u8) !usize {
        const io = std.Io.Threaded.global_single_threaded.io();
        const f = if (path.len > 0 and path[0] == '/')
            try std.Io.Dir.openFileAbsolute(io, path, .{})
        else
            try std.Io.Dir.cwd().openFile(io, path, .{});
        defer f.close(io);
        return @intCast((try f.stat(io)).size);
    }
};

const Loader = struct {
    reader: *const gguf.Reader,
    map: []align(std.heap.page_size_min) const u8,

    fn info(self: Loader, name: []const u8) !gguf.TensorInfo {
        return self.reader.tensors.get(name) orelse error.TensorNotFound;
    }
    fn ptr(self: Loader, i: gguf.TensorInfo) [*]const u8 {
        return self.map.ptr + self.reader.data_offset + i.offset;
    }
    fn mat(self: Loader, name: []const u8) !Mat {
        const i = try self.info(name);
        const t: ops.Type = switch (i.dtype) {
            .f32 => .f32,
            .f16 => .f16,
            .q8_0 => .q8_0,
            .q4_k => .q4_k,
            .q5_k => .q5_k,
            .q6_k => .q6_k,
            else => return error.UnsupportedTensorType,
        };
        const k: usize = @intCast(i.shape[0]);
        return .{ .data = self.ptr(i), .type = t, .k = k, .m = @intCast(i.shape[1]), .nb1 = ops.rowSize(t, k) };
    }
    /// ggml F32 tensör (norm ağırlıkları, ssm_a, ssm_dt, conv1d)
    fn vec1(self: Loader, name: []const u8) ![*]const f32 {
        const i = try self.info(name);
        if (i.dtype != .f32) return error.UnsupportedTensorType;
        return @ptrCast(@alignCast(self.ptr(i)));
    }
};

fn pad(x: usize, n: usize) usize {
    return (x + n - 1) / n * n;
}

pub const Context = struct {
    allocator: Alloc,
    model: *const Model,
    pool: *Pool,
    n_threads: u32,
    n_ctx: usize,
    n_ubatch: usize,
    head: usize = 0,

    // llama_memory: KV (attention) ve recurrent state (GDN)
    kv_k: [][]u16,
    kv_v: [][]u16,
    rs_conv: [][]f32,
    rs_state: [][]f32,
    /// H39: snapshot tamponu; snapshot() rs_state ile takas eder (kopya yok), GDN t=0'da buradan okur.
    rs_alt: [][]f32 = &.{},
    gdn_alt: bool = false,

    logits: []f32,
    /// null: tam LM head (varsayılan)
    lm: ?*ivf.Search = null,

    wdata: []align(64) u8,
    /// H62: down girdisi (gate/up GEMM'i wdata'yı okurken yazılır)
    wdata2: []align(64) u8,
    /// H63: conv parça sonunda q/k başlığı başına biten kanal sayısı
    l2_cnt: [64]std.atomic.Value(u32) = @splat(.init(0)),
    /// H65: GDN'de başlık başına biten kalem sayısı
    gdn_cnt: [64]std.atomic.Value(u32) = @splat(.init(0)),
    fdata: []align(64) f32,
    theta_scale: f32,

    // graph tamponları
    b_inp: []align(64) f32,
    b_norm: []align(64) f32,
    b_resid: []align(64) f32,
    b_attn: []align(64) f32,
    b_qkv: []align(64) f32,
    b_conv_in: []align(64) f32,
    b_conv_out: []align(64) f32,
    b_q: []align(64) f32,
    b_k: []align(64) f32,
    b_gate: []align(64) f32,
    b_beta: []align(64) f32,
    b_gdn: []align(64) f32,
    b_z: []align(64) f32,
    b_ffn_g: []align(64) f32,
    b_ffn_u: []align(64) f32,
    b_qfull: []align(64) f32,
    b_kk: []align(64) f32,
    b_vv: []align(64) f32,

    // geçerli ubatch
    tokens: [*]const u32 = undefined,
    n_t: usize = 0,
    want_logits: bool = false,
    n_kv: usize = 0,
    /// Spekülatif doğrulama: her pozisyon için top-1 ve geri alma izleri.
    /// want_all true iken graph her satır için LM head çalıştırır (verify_tokens).
    want_all: bool = false,
    verify_tokens: []u32 = &.{},
    trace_k: usize = 0,
    trace_conv: [][]f32 = &.{},
    trace_recr: [][]f32 = &.{},
    verify_len: usize = 0,
    /// H14: doğrulamada LM head'i tüm pozisyonlar için tek geçişte yapar (enableTrace kurar).
    lmb: ?*ivf.Batch = null,
    /// true: decodeVerify iz yazmaz (geri alma yapılmayacaksa; H11/P6).
    no_trace: bool = false,

    fn buf(allocator: Alloc, n: usize) ![]align(64) f32 {
        const b = try allocator.alignedAlloc(f32, .@"64", n);
        @memset(b, 0);
        return b;
    }

    pub fn init(model: *const Model, n_ctx_req: usize, n_threads: u32, allocator: Alloc) !*Context {
        const m = model;
        const n_ctx = pad(@max(n_ctx_req, 1), 256);
        const n_ub: usize = @min(512, n_ctx);
        const self = try allocator.create(Context);
        errdefer allocator.destroy(self);
        const n_layer = m.layers.len;
        const kv_row = m.head_dim * m.n_head_kv;
        const conv_dim = m.d_inner + 2 * m.n_group * m.d_state;
        const s_size = m.d_state * m.d_state * m.dt_rank;

        const kv_k = try allocator.alloc([]u16, n_layer);
        const kv_v = try allocator.alloc([]u16, n_layer);
        const rs_conv = try allocator.alloc([]f32, n_layer);
        const rs_state = try allocator.alloc([]f32, n_layer);
        for (m.layers, 0..) |*l, il| {
            if (l.recr) {
                rs_conv[il] = try allocator.alloc(f32, (m.d_conv - 1) * conv_dim);
                rs_state[il] = try allocator.alloc(f32, s_size);
                kv_k[il] = &.{};
                kv_v[il] = &.{};
            } else {
                // ggml_backend_buffer_clear: tile içindeki boş hücreler de okunur
                kv_k[il] = try allocator.alloc(u16, kv_row * n_ctx);
                kv_v[il] = try allocator.alloc(u16, kv_row * n_ctx);
                @memset(kv_k[il], 0);
                @memset(kv_v[il], 0);
                rs_conv[il] = &.{};
                rs_state[il] = &.{};
            }
        }

        const max_k = @max(m.n_embd, @max(m.n_ff, m.d_inner));
        self.* = .{
            .allocator = allocator,
            .model = model,
            .pool = try Pool.init(allocator, n_threads),
            .n_threads = n_threads,
            .n_ctx = n_ctx,
            .n_ubatch = n_ub,
            .kv_k = kv_k,
            .kv_v = kv_v,
            .rs_conv = rs_conv,
            .rs_state = rs_state,
            .logits = try allocator.alloc(f32, m.n_vocab),
            .wdata = try allocator.alignedAlloc(u8, .@"64", n_ub * max_k * 2), // en büyük satır: F16
            .wdata2 = try allocator.alignedAlloc(u8, .@"64", n_ub * (max_k / 32 * 34 + 64)),
            .fdata = try buf(allocator, ops.flashScratchFloats(m.head_dim, m.n_head, n_threads) + n_threads * (m.d_state + 64)),
            .theta_scale = avx.powf(m.freq_base, -2.0 / @as(f32, @floatFromInt(m.n_rot))),
            .b_inp = try buf(allocator, n_ub * m.n_embd),
            .b_norm = try buf(allocator, n_ub * m.n_embd),
            .b_resid = try buf(allocator, n_ub * m.n_embd),
            .b_attn = try buf(allocator, n_ub * m.n_embd),
            .b_qkv = try buf(allocator, n_ub * conv_dim),
            .b_conv_in = try buf(allocator, (n_ub + m.d_conv - 1) * conv_dim),
            .b_conv_out = try buf(allocator, n_ub * conv_dim),
            .b_q = try buf(allocator, n_ub * m.d_inner),
            .b_k = try buf(allocator, n_ub * m.d_inner),
            .b_gate = try buf(allocator, n_ub * m.dt_rank),
            .b_beta = try buf(allocator, n_ub * m.dt_rank),
            .b_gdn = try buf(allocator, n_ub * m.d_inner),
            .b_z = try buf(allocator, n_ub * m.d_inner),
            .b_ffn_g = try buf(allocator, n_ub * m.n_ff),
            .b_ffn_u = try buf(allocator, n_ub * m.n_ff),
            .b_qfull = try buf(allocator, n_ub * m.head_dim * m.n_head * 2),
            .b_kk = try buf(allocator, n_ub * kv_row),
            .b_vv = try buf(allocator, n_ub * kv_row),
        };
        self.clear();
        return self;
    }

    pub fn deinit(self: *Context) void {
        const a = self.allocator;
        self.pool.deinit();
        for (self.kv_k) |b| if (b.len > 0) a.free(b);
        for (self.kv_v) |b| if (b.len > 0) a.free(b);
        for (self.rs_conv) |b| if (b.len > 0) a.free(b);
        for (self.rs_state) |b| if (b.len > 0) a.free(b);
        for (self.rs_alt) |b| if (b.len > 0) a.free(b);
        if (self.rs_alt.len > 0) a.free(self.rs_alt);
        a.free(self.kv_k);
        a.free(self.kv_v);
        a.free(self.rs_conv);
        a.free(self.rs_state);
        self.freeTrace();
        a.free(self.logits);
        a.free(self.wdata);
        a.free(self.wdata2);
        a.free(self.fdata);
        inline for (.{ "b_inp", "b_norm", "b_resid", "b_attn", "b_qkv", "b_conv_in", "b_conv_out", "b_q", "b_k", "b_gate", "b_beta", "b_gdn", "b_z", "b_ffn_g", "b_ffn_u", "b_qfull", "b_kk", "b_vv" }) |f| {
            a.free(@field(self, f));
        }
        a.destroy(self);
    }

    /// Doğrulama/geri alma için pozisyon başına GDN ve conv durum izlerini hazırlar.
    pub fn enableTrace(self: *Context, k: usize) !void {
        if (k == 0) return error.TraceTooSmall;
        self.freeTrace();
        const a = self.allocator;
        const m = self.model;
        const conv_len = (m.d_conv - 1) * (m.d_inner + 2 * m.n_group * m.d_state);
        const s_size = m.d_state * m.d_state * m.dt_rank;
        const tc = try a.alloc([]f32, m.layers.len);
        errdefer a.free(tc);
        const tr = try a.alloc([]f32, m.layers.len);
        errdefer a.free(tr);
        for (m.layers, 0..) |*l, il| {
            tc[il] = &.{};
            tr[il] = &.{};
            if (!l.recr) continue;
            tc[il] = try a.alloc(f32, k * conv_len);
            tr[il] = try a.alloc(f32, k * s_size);
        }
        self.trace_conv = tc;
        self.trace_recr = tr;
        self.trace_k = k;
        self.verify_tokens = try a.alloc(u32, k);
        if (self.lm) |s| {
            if (k <= ivf.Batch.MAXR) self.lmb = try ivf.Batch.init(s.ix, s.nprobe, self.n_threads, a);
        }
    }

    fn freeTrace(self: *Context) void {
        const a = self.allocator;
        if (self.trace_k == 0) return;
        for (self.trace_conv) |b| if (b.len > 0) a.free(b);
        for (self.trace_recr) |b| if (b.len > 0) a.free(b);
        a.free(self.trace_conv);
        a.free(self.trace_recr);
        a.free(self.verify_tokens);
        if (self.lmb) |b| b.deinit(a);
        self.lmb = null;
        self.trace_conv = &.{};
        self.trace_recr = &.{};
        self.verify_tokens = &.{};
        self.trace_k = 0;
    }

    /// Aday diziyi tek forward'da işler; her pozisyonun top-1'i verify_tokens'a yazılır.
    pub fn decodeVerify(self: *Context, tokens: []const u32) !void {
        if (tokens.len == 0 or self.trace_k < tokens.len) return error.TraceTooSmall;
        self.verify_len = tokens.len;
        self.want_all = true;
        defer self.want_all = false;
        try self.decode(tokens);
    }

    /// Durumu batch başından j token sonrasına geri alır (j < batch uzunluğu).
    /// KV cache kırpılmaz: head geri çekilir, mask ileriyi görmez, üzerine yazılır.
    pub fn restore(self: *Context, batch_start: usize, j: usize) void {
        if (j < self.verify_len) {
        for (self.model.layers, 0..) |*l, il| {
            if (!l.recr) continue;
            const conv_len = self.rs_conv[il].len;
            const s_len = self.rs_state[il].len;
            @memcpy(self.rs_conv[il], self.trace_conv[il][j * conv_len ..][0..conv_len]);
            @memcpy(self.rs_state[il], self.trace_recr[il][j * s_len ..][0..s_len]);
        }
        }
        self.head = batch_start + j;
    }
    /// H13: tur başı durumunu iz yuvası 0'a kopyalar (no_trace ile birlikte kullanılır).
    pub fn snapshot(self: *Context) void {
        if (self.rs_alt.len == 0) {
            const a = self.allocator;
            const alt = a.alloc([]f32, self.rs_state.len) catch @panic("OOM");
            for (alt, self.rs_state) |*d, b| d.* = if (b.len > 0) a.alloc(f32, b.len) catch @panic("OOM") else &.{};
            self.rs_alt = alt;
        }
        for (self.model.layers, 0..) |*l, il| {
            if (!l.recr) continue;
            @memcpy(self.trace_conv[il][0..self.rs_conv[il].len], self.rs_conv[il]);
            std.mem.swap([]f32, &self.rs_state[il], &self.rs_alt[il]);
        }
        self.gdn_alt = true;
    }

    /// H13: snapshot() anına döner; çağıran kabul edilen tokenları yeniden decode eder.
    pub fn rollback(self: *Context, batch_start: usize) void {
        for (self.model.layers, 0..) |*l, il| {
            if (!l.recr) continue;
            @memcpy(self.rs_conv[il], self.trace_conv[il][0..self.rs_conv[il].len]);
            std.mem.swap([]f32, &self.rs_state[il], &self.rs_alt[il]);
        }
        self.gdn_alt = false;
        self.head = batch_start;
    }

    /// llama_memory_clear
    pub fn clear(self: *Context) void {
        self.head = 0;
        for (self.rs_conv) |b| @memset(b, 0);
        for (self.rs_state) |b| @memset(b, 0);
        self.gdn_alt = false;
    }

    /// llama_decode: yalnız son token için logit üretir (llama_batch_get_one).
    pub fn decode(self: *Context, tokens: []const u32) !void {
        if (self.head + tokens.len > self.n_ctx) return error.ContextFull;
        for (tokens) |t| if (t >= self.model.n_vocab) return error.InvalidToken;
        var off: usize = 0;
        while (off < tokens.len) {
            const n = @min(tokens.len - off, self.n_ubatch);
            self.tokens = tokens.ptr + off;
            self.n_t = n;
            self.want_logits = off + n == tokens.len;
            // llama_kv_cache::get_n_kv
            self.n_kv = @min(self.n_ctx, @max(@as(usize, 256), pad(self.head + n, 256)));
            self.pool.run(self, job);
            self.gdn_alt = false;
            self.head += n;
            off += n;
        }
    }

    fn job(ctx: *anyopaque, ith: u32) void {
        const self: *Context = @ptrCast(@alignCast(ctx));
        const p = Params{
            .ith = ith,
            .nth = self.n_threads,
            .pool = self.pool,
            .wdata = self.wdata.ptr,
            .fdata = self.fdata.ptr,
        };
        self.graph(&p);
    }

    fn embedRow(m: *const Model, token: u32, y: [*]f32) void {
        const row = m.tok_embd.data + @as(usize, token) * m.tok_embd.nb1;
        const k = m.n_embd;
        switch (m.tok_embd.type) {
            .f32 => @memcpy(y[0..k], @as([*]const f32, @ptrCast(@alignCast(row)))[0..k]),
            .f16 => vec.fp16to32Row(@ptrCast(@alignCast(row)), y, k),
            .q8_0 => q.dequantizeQ80(@ptrCast(@alignCast(row)), y, k),
            .q4_k => q.dequantizeQ4K(@ptrCast(@alignCast(row)), y, k),
            .q5_k => q.dequantizeQ5K(@ptrCast(@alignCast(row)), y, k),
            .q6_k => q.dequantizeQ6K(@ptrCast(@alignCast(row)), y, k),
            .q4_k_8x8 => unreachable, // token_embd repack edilmez
        }
    }

    /// Yaklaşık LM head: yalnız seçilen kümelerin satırları okunur; `logits` güncellenmez, sonuç `lm.token`.
    fn lmSearch(p: *const Params, s: *ivf.Search, x: [*]const f32, k: usize) void {
        const xq: [*]q.BlockQ80 = @ptrCast(@alignCast(p.wdata));
        if (p.ith == 0) q.quantizeRowQ80(x, xq, k);
        p.barrier();
        s.scoreRange(xq, p.ith, p.nth);
        p.barrier();
        if (p.ith == 0) {
            s.plan();
            p.pool.chunkSet(@intCast(p.nth));
        }
        p.barrier();
        var b = ivf.Best{};
        var cur: i64 = p.ith;
        while (cur < s.n_pieces) {
            s.scan(xq, @intCast(cur), &b);
            cur = p.pool.chunkAdd(1);
        }
        s.best[p.ith] = b;
        p.barrier();
        if (p.ith == 0) s.reduce();
        p.barrier();
    }

    fn graph(self: *Context, p: *const Params) void {
        const m = self.model;
        const n_t = self.n_t;

        // GET_ROWS(token_embd)
        const r = p.range(n_t);
        for (r[0]..r[1]) |t| embedRow(m, self.tokens[t], self.b_inp.ptr + t * m.n_embd);
        p.barrier();
        // H58: matmul girdileri üretici fazda R8 nicemlenir (nicemleme fazı ve bariyeri yok)
        const q0 = inOk(&m.layers[0]);
        ops.addRmsNorm(p, m.n_embd, n_t, self.b_inp.ptr, null, self.b_inp.ptr, self.b_norm.ptr, m.eps, m.layers[0].attn_norm, q0);
        if (q0) inPre(p, &m.layers[0]);
        p.barrier();

        for (m.layers, 0..) |*l, il| {
            if (l.recr) {
                self.layerLinear(p, l, il);
            } else {
                self.layerAttn(p, l, il);
            }
            // attn_residual + attn_post_norm (H57: tek geçiş)
            const ffn_in = [_]ops.Mat{ l.ffn_gate, l.ffn_up };
            const fast = ops.ffnOk(l.ffn_gate, l.ffn_up, l.ffn_down);
            const qf = ops.r8Ok(&ffn_in);
            ops.addRmsNorm(p, m.n_embd, n_t, self.b_resid.ptr, self.b_attn.ptr, self.b_inp.ptr, self.b_norm.ptr, m.eps, l.post_norm, qf);
            if (fast) ops.ffnPre(p, l.ffn_gate) else if (qf) ops.r8Pre(p, &ffn_in);
            p.barrier();
            // build_ffn(LLM_FFN_SILU, LLM_FFN_PAR)
            if (fast) {
                // H62: gate+up parça başına, swiglu + down nicemleme parça sonunda
                ops.ffnGateUp(p, l.ffn_gate, l.ffn_up, l.ffn_down, n_t, self.b_ffn_g.ptr, self.b_ffn_u.ptr, self.wdata2.ptr, self.b_attn.ptr);
            } else {
                ops.mulMatMultiQ(p, &ffn_in, self.b_norm.ptr, n_t, &.{ self.b_ffn_g.ptr, self.b_ffn_u.ptr });
                const qd = ops.r8Ok(&.{l.ffn_down});
                ops.swigluQ(p, m.n_ff, n_t, self.b_ffn_g.ptr, self.b_ffn_g.ptr, self.b_ffn_u.ptr, qd);
                if (qd) ops.r8Pre(p, &.{l.ffn_down});
                p.barrier();
                ops.mulMatMultiQ(p, &.{l.ffn_down}, self.b_ffn_g.ptr, n_t, &.{self.b_attn.ptr});
            }
            // l_out (+ sonraki katmanın attn_norm'u)
            if (il + 1 < m.layers.len) {
                const nl = &m.layers[il + 1];
                const qn = inOk(nl);
                ops.addRmsNorm(p, m.n_embd, n_t, self.b_inp.ptr, self.b_attn.ptr, self.b_resid.ptr, self.b_norm.ptr, m.eps, nl.attn_norm, qn);
                if (qn) inPre(p, nl);
            } else ops.add(p, m.n_embd, n_t, self.b_inp.ptr, self.b_attn.ptr, self.b_resid.ptr);
            p.barrier();
        }

        if (self.want_all and self.trace_k > 0 and self.lmb != null) {
            const b = self.lmb.?;
            ops.rmsNorm(p, self.b_inp.ptr, m.n_embd, 0, self.b_norm.ptr, m.n_embd, 0, m.n_embd, n_t, 1, m.eps, m.output_norm, 1.0);
            p.barrier();
            const xq: [*]q.BlockQ80 = @ptrCast(@alignCast(p.wdata));
            const rb = m.n_embd / 32;
            const rr = p.range(n_t);
            const xr = xq + n_t * rb;
            for (rr[0]..rr[1]) |t| {
                q.quantizeRowQ80(self.b_norm.ptr + t * m.n_embd, xq + t * rb, m.n_embd);
                sgemm.quantR8(self.b_norm.ptr + t * m.n_embd, xr + t * rb, m.n_embd);
            }
            p.barrier();
            if (p.ith == 0) b.begin(n_t);
            b.scoreRange(xr, n_t, p.ith, p.nth);
            p.barrier();
            b.select(p.ith, p.nth);
            p.barrier();
            if (p.ith == 0) {
                b.plan();
                p.pool.chunkSet(@intCast(p.nth));
            }
            p.barrier();
            var best: [ivf.Batch.MAXR]ivf.Best4 = @splat(.{});
            var cur: i64 = p.ith;
            while (cur < b.n_pieces) {
                b.scan(xr, @intCast(cur), best[0..n_t]);
                cur = p.pool.chunkAdd(1);
            }
            @memcpy(b.best[p.ith * ivf.Batch.MAXR ..][0..n_t], best[0..n_t]);
            p.barrier();
            if (p.ith == 0) {
                b.reduce(xq);
                @memcpy(self.verify_tokens[0..n_t], b.tokens[0..n_t]);
            }
            p.barrier();
        } else if (self.want_all and self.trace_k > 0) {
            var vt: usize = 0;
            while (vt < n_t) : (vt += 1) {
                const row = self.b_inp.ptr + vt * m.n_embd;
                ops.rmsNorm(p, row, m.n_embd, 0, self.b_norm.ptr, m.n_embd, 0, m.n_embd, 1, 1, m.eps, m.output_norm, 1.0);
                p.barrier();
                if (self.lm) |s| {
                    lmSearch(p, s, self.b_norm.ptr, m.n_embd);
                    if (p.ith == 0) self.verify_tokens[vt] = s.token;
                } else {
                    ops.mulMat(p, m.output, self.b_norm.ptr, 1, self.logits.ptr);
                    p.barrier();
                    if (p.ith == 0) self.verify_tokens[vt] = @intCast(std.mem.indexOfMax(f32, self.logits));
                }
            }
        } else if (self.want_logits) {
            const last = self.b_inp.ptr + (n_t - 1) * m.n_embd;
            ops.rmsNorm(p, last, m.n_embd, 0, self.b_norm.ptr, m.n_embd, 0, m.n_embd, 1, 1, m.eps, m.output_norm, 1.0);
            p.barrier();
            if (self.lm) |s| {
                lmSearch(p, s, self.b_norm.ptr, m.n_embd);
            } else {
                ops.mulMat(p, m.output, self.b_norm.ptr, 1, self.logits.ptr);
                p.barrier();
            }
        }
        // ggml_graph_compute_thread: son barrier
        p.barrier();
    }

    /// llama_model_qwen35::graph::build_layer_attn_linear
    fn layerLinear(self: *Context, p: *const Params, l: *const Layer, il: usize) void {
        const m = self.model;
        const n_t = self.n_t;
        const conv_dim = m.d_inner + 2 * m.n_group * m.d_state;
        const kc = m.d_conv - 1;
        const s = m.d_state;
        const h_n = m.dt_rank;
        const key_dim = m.n_group * m.d_state;
        const l2 = 1.0 / @sqrt(@as(f32, @floatFromInt(s)));
        const eps_l2 = m.eps / @as(f32, @floatFromInt(s));
        const tracing = self.want_all and !self.no_trace and self.trace_k >= n_t;
        const tr_conv: ?[*]f32 = if (tracing) self.trace_conv[il].ptr else null;
        const tr_recr: ?[*]f32 = if (tracing) self.trace_recr[il].ptr else null;

        // H22: qkv, alpha, beta, z aynı girdiden tek geçişte; ardından bağımsız eleman işlemleri tek fazda.
        const in4 = [_]ops.Mat{ l.wqkv, l.ssm_alpha, l.ssm_beta, l.wgate };
        const out4 = [_][*]f32{ self.b_qkv.ptr, self.b_gate.ptr, self.b_beta.ptr, self.b_z.ptr };
        // convPost: başlık parça sınırına hizalı ve sayaçlar yeterli olmalı
        const fast_in = m.d_conv == 4 and ops.r8Ok(&in4) and s % ops.GemmRows == 0 and 2 * m.n_group <= self.l2_cnt.len;
        // H65: GDN + gated norm tek fazda; GDN aralıkları (set 1) bu GEMM fazında hazırlanır
        const gfast = fast_in and tr_recr == null and s % 32 == 0 and h_n <= self.gdn_cnt.len;
        if (gfast) p.pool.rangeInitS(1, p.ith, h_n * (s / 32));
        if (fast_in) {
            // H61: conv + gate + sigmoid + silu GEMM parça sonunda (satır yerel); ayrı faz ve bariyeri yok
            const Epi = struct {
                c: *Context,
                l: *const Layer,
                il: usize,
                conv_dim: usize,
                h_n: usize,
                tr: ?[*]f32,
                pub fn epi(e: @This(), mi: usize, r0: usize, r1: usize) void {
                    const c = e.c;
                    const nt = c.n_t;
                    switch (mi) {
                        0 => {
                            ops.convRows(e.conv_dim, nt, c.rs_conv[e.il].ptr, c.b_qkv.ptr, e.l.ssm_conv1d, c.b_conv_out.ptr, e.tr, r0, r1);
                            c.convPost(r0, r1);
                        },
                        1 => for (0..nt) |t| for (r0..r1) |h| {
                            c.b_gate[t * e.h_n + h] = ops.softplusF(c.b_gate[t * e.h_n + h] + e.l.ssm_dt[h]) * e.l.ssm_a[h];
                        },
                        2 => for (0..nt) |t| for (r0..r1) |h| {
                            c.b_beta[t * e.h_n + h] = ops.sigmoidF(c.b_beta[t * e.h_n + h]);
                        },
                        else => {
                            const d = c.model.d_inner;
                            for (0..nt) |t| vec.silu(r1 - r0, c.b_z.ptr + t * d + r0, c.b_z.ptr + t * d + r0);
                        },
                    }
                }
            };
            ops.mulMatMultiQE(p, &in4, n_t, &out4, Epi{ .c = self, .l = l, .il = il, .conv_dim = conv_dim, .h_n = h_n, .tr = tr_conv });
        } else {
            ops.mulMatMultiQ(p, &in4, self.b_norm.ptr, n_t, &out4);
            // CONCAT + CPY(conv_state_last)
            if (m.d_conv == 4) {
                // H35: concat + conv + silu tek fazda
                ops.convFused(p, conv_dim, n_t, self.rs_conv[il].ptr, self.b_qkv.ptr, l.ssm_conv1d, self.b_conv_out.ptr, tr_conv);
            } else ops.convConcat(p, conv_dim, kc, n_t, self.rs_conv[il].ptr, self.b_qkv.ptr, self.b_conv_in.ptr, tr_conv);
            ops.gateFromAlpha(p, h_n, n_t, self.b_gate.ptr, self.b_gate.ptr, l.ssm_dt, l.ssm_a);
            ops.sigmoidInPlace(p, h_n, n_t, self.b_beta.ptr);
            ops.silu(p, s, h_n * n_t, self.b_z.ptr, self.b_z.ptr);
            p.barrier();
        }
        if (m.d_conv != 4) {
            ops.ssmConv(p, conv_dim, m.d_conv, kc + n_t, n_t, self.b_conv_in.ptr, l.ssm_conv1d, self.b_conv_out.ptr);
            p.barrier();
            ops.silu(p, conv_dim, n_t, self.b_conv_out.ptr, self.b_conv_out.ptr);
            p.barrier();
        }
        // build_gdn_l2_norm(q), (k)
        // H55: q/k başlık-öncelikli [başlık][token][s]: GDN kalemi ardışık okur (8 KB adım CCX'ler arası gecikmeye takılıyordu)
        // H63: hızlı yolda conv parça sonunda (convPost) yapıldı; v b_conv_in'de başlık-öncelikli
        const vh = if (fast_in) self.b_conv_in.ptr else self.b_qkv.ptr;
        if (!fast_in) {
            ops.rmsNorm(p, self.b_conv_out.ptr, s, conv_dim, self.b_q.ptr, n_t * s, s, s, m.n_group, n_t, eps_l2, null, l2);
            ops.rmsNorm(p, self.b_conv_out.ptr + key_dim, s, conv_dim, self.b_k.ptr, n_t * s, s, s, m.n_group, n_t, eps_l2, null, l2);
            // v da başlık-öncelikli (b_qkv conv'dan sonra boş)
            var vi: usize = p.ith;
            while (vi < n_t * h_n) : (vi += p.nth) {
                const t = vi / h_n;
                const hh = vi % h_n;
                @memcpy((vh + (hh * n_t + t) * s)[0..s], (self.b_conv_out.ptr + t * conv_dim + 2 * key_dim + hh * s)[0..s]);
            }
            p.barrier();
        }
        const qo = ops.r8Ok(&.{l.ssm_out});
        const st_in = if (self.gdn_alt) self.rs_alt[il].ptr else self.rs_state[il].ptr;
        if (gfast) {
            ops.gdnNormQ(p, s, h_n, n_t, self.b_q.ptr, self.b_k.ptr, vh, s, n_t * s, self.b_gate.ptr, self.b_beta.ptr, st_in, self.rs_state[il].ptr, self.b_gdn.ptr, &self.gdn_cnt, self.b_z.ptr, m.eps, l.ssm_norm, qo);
            if (qo) ops.r8Pre(p, &.{l.ssm_out});
            p.barrier();
            return ops.mulMatMultiQ(p, &.{l.ssm_out}, self.b_gdn.ptr, n_t, &.{self.b_attn.ptr});
        }
        // GATED_DELTA_NET (+ CPY new_state: yerinde)
        ops.gatedDeltaNet(p, s, h_n, n_t, self.b_q.ptr, self.b_k.ptr, vh, s, n_t * s, self.b_gate.ptr, self.b_beta.ptr, if (self.gdn_alt) self.rs_alt[il].ptr else self.rs_state[il].ptr, self.rs_state[il].ptr, self.b_gdn.ptr, tr_recr);
        p.barrier();
        // build_norm_gated
        // + mul z + ssm_out girdisi nicemleme (H58: tek faz)
        ops.gatedNormQ(p, s, h_n, n_t, self.b_gdn.ptr, self.b_z.ptr, m.eps, l.ssm_norm, qo);
        if (qo) ops.r8Pre(p, &.{l.ssm_out});
        p.barrier();
        ops.mulMatMultiQ(p, &.{l.ssm_out}, self.b_gdn.ptr, n_t, &.{self.b_attn.ptr});
    }

    /// llama_model_qwen35::graph::build_layer_attn + build_attn_mha (flash attention)
    fn layerAttn(self: *Context, p: *const Params, l: *const Layer, il: usize) void {
        const m = self.model;
        const n_t = self.n_t;
        const dh = m.head_dim;
        const kv_row = dh * m.n_head_kv;

        ops.mulMatMultiQ(p, &.{ l.wq, l.wv, l.wk }, self.b_norm.ptr, n_t, &.{ self.b_qfull.ptr, self.b_vv.ptr, self.b_kk.ptr });
        if (m.n_rot <= 128 and std.c.getenv("ECHO_OLD_ATTN") == null) {
            ops.attnPrep(p, n_t, dh, m.n_head, m.n_head_kv, self.b_qfull.ptr, self.b_gdn.ptr, self.b_kk.ptr, self.b_vv.ptr, self.kv_k[il].ptr + self.head * kv_row, self.kv_v[il].ptr + self.head * kv_row, m.eps, l.q_norm, l.k_norm, self.head, m.n_rot, self.theta_scale);
            p.barrier();
        } else self.attnPrepOld(p, l, il);
        self.attnTail(p, l, il);
    }

    fn attnPrepOld(self: *Context, p: *const Params, l: *const Layer, il: usize) void {
        const m = self.model;
        const n_t = self.n_t;
        const dh = m.head_dim;
        const kv_row = dh * m.n_head_kv;
        // Qcur = view(Qcur_full) -> build_norm -> rope. q tamponu: b_gdn (head_dim*n_head*n_t)
        ops.rmsNorm(p, self.b_qfull.ptr, dh * 2, dh * m.n_head * 2, self.b_gdn.ptr, dh, dh * m.n_head, dh, m.n_head, n_t, m.eps, l.q_norm, 1.0);
        p.barrier();
        ops.rope(p, self.b_gdn.ptr, dh, m.n_head, n_t, self.head, m.n_rot, self.theta_scale);
        p.barrier();
        ops.rmsNorm(p, self.b_kk.ptr, dh, kv_row, self.b_kk.ptr, dh, kv_row, dh, m.n_head_kv, n_t, m.eps, l.k_norm, 1.0);
        p.barrier();
        ops.rope(p, self.b_kk.ptr, dh, m.n_head_kv, n_t, self.head, m.n_rot, self.theta_scale);
        p.barrier();
        // cache_k / cache_v SET_ROWS
        ops.setRowsF16(p, kv_row, n_t, self.b_kk.ptr, self.kv_k[il].ptr + self.head * kv_row);
        p.barrier();
        ops.setRowsF16(p, kv_row, n_t, self.b_vv.ptr, self.kv_v[il].ptr + self.head * kv_row);
        p.barrier();
    }

    fn attnTail(self: *Context, p: *const Params, l: *const Layer, il: usize) void {
        const m = self.model;
        const n_t = self.n_t;
        const dh = m.head_dim;
        // FLASH_ATTN_EXT
        const args = ops.FlashArgs{
            .q = self.b_gdn.ptr,
            .k = self.kv_k[il].ptr,
            .v = self.kv_v[il].ptr,
            .dst = self.b_q.ptr,
            .dk = dh,
            .h_q = m.n_head,
            .h_kv = m.n_head_kv,
            .n_t = n_t,
            .n_kv = self.n_kv,
            .first_cell = self.head,
            .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(dh))),
        };
        ops.flashAttn(p, &args);
        p.barrier();
        // CONT(gate) + SIGMOID + MUL
        const qw = ops.r8Ok(&.{l.wo});
        ops.gateAttn(p, n_t, m.n_head, dh, self.b_q.ptr, self.b_qfull.ptr, qw);
        if (qw) ops.r8Pre(p, &.{l.wo});
        p.barrier();
        ops.mulMatMultiQ(p, &.{l.wo}, self.b_q.ptr, n_t, &.{self.b_attn.ptr});
    }

    /// H63: conv parça sonu [r0, r1): v kanalları b_conv_in'e başlık-öncelikli kopyalanır; q/k başlığının
    /// son parçasını bitiren thread o başlığın l2 normunu tüm tokenlar için yazar (sayaç başlık başına).
    fn convPost(self: *Context, r0: usize, r1: usize) void {
        const m = self.model;
        const n_t = self.n_t;
        const s = m.d_state;
        const kd = m.n_group * s;
        const cd = m.d_inner + 2 * kd;
        const src = self.b_conv_out.ptr;
        if (r0 >= 2 * kd) {
            const cv = r0 - 2 * kd;
            const hh = cv / s;
            const o = cv % s;
            for (0..n_t) |t| @memcpy((self.b_conv_in.ptr + (hh * n_t + t) * s + o)[0 .. r1 - r0], (src + t * cd + r0)[0 .. r1 - r0]);
            return;
        }
        const which = r0 / kd;
        const h = (r0 % kd) / s;
        const cnt = &self.l2_cnt[which * m.n_group + h];
        if (cnt.fetchAdd(@intCast(r1 - r0), .acq_rel) + (r1 - r0) != s) return;
        cnt.store(0, .monotonic);
        const l2 = 1.0 / @sqrt(@as(f32, @floatFromInt(s)));
        const eps_l2 = m.eps / @as(f32, @floatFromInt(s));
        const dst = if (which == 0) self.b_q.ptr else self.b_k.ptr;
        ops.normRows(src + which * kd + h * s, cd, dst + h * n_t * s, s, s, n_t, eps_l2, l2);
    }

    fn inOk(l: *const Layer) bool {
        return if (l.recr) ops.r8Ok(&.{ l.wqkv, l.ssm_alpha, l.ssm_beta, l.wgate }) else ops.r8Ok(&.{ l.wq, l.wv, l.wk });
    }
    fn inPre(p: *const Params, l: *const Layer) void {
        if (l.recr) ops.r8Pre(p, &.{ l.wqkv, l.ssm_alpha, l.ssm_beta, l.wgate }) else ops.r8Pre(p, &.{ l.wq, l.wv, l.wk });
    }
};
