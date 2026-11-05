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

    logits: []f32,

    wdata: []align(64) u8,
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
        a.free(self.kv_k);
        a.free(self.kv_v);
        a.free(self.rs_conv);
        a.free(self.rs_state);
        a.free(self.logits);
        a.free(self.wdata);
        a.free(self.fdata);
        inline for (.{ "b_inp", "b_norm", "b_resid", "b_attn", "b_qkv", "b_conv_in", "b_conv_out", "b_q", "b_k", "b_gate", "b_beta", "b_gdn", "b_z", "b_ffn_g", "b_ffn_u", "b_qfull", "b_kk", "b_vv" }) |f| {
            a.free(@field(self, f));
        }
        a.destroy(self);
    }

    /// llama_memory_clear
    pub fn clear(self: *Context) void {
        self.head = 0;
        for (self.rs_conv) |b| @memset(b, 0);
        for (self.rs_state) |b| @memset(b, 0);
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

    fn graph(self: *Context, p: *const Params) void {
        const m = self.model;
        const n_t = self.n_t;

        // GET_ROWS(token_embd)
        const r = p.range(n_t);
        for (r[0]..r[1]) |t| embedRow(m, self.tokens[t], self.b_inp.ptr + t * m.n_embd);
        p.barrier();

        for (m.layers, 0..) |*l, il| {
            ops.rmsNorm(p, self.b_inp.ptr, m.n_embd, 0, self.b_norm.ptr, m.n_embd, 0, m.n_embd, n_t, 1, m.eps, l.attn_norm, 1.0);
            p.barrier();
            if (l.recr) {
                self.layerLinear(p, l, il);
            } else {
                self.layerAttn(p, l, il);
            }
            // attn_residual
            ops.add(p, m.n_embd, n_t, self.b_resid.ptr, self.b_attn.ptr, self.b_inp.ptr);
            p.barrier();
            // attn_post_norm
            ops.rmsNorm(p, self.b_resid.ptr, m.n_embd, 0, self.b_norm.ptr, m.n_embd, 0, m.n_embd, n_t, 1, m.eps, l.post_norm, 1.0);
            p.barrier();
            // build_ffn(LLM_FFN_SILU, LLM_FFN_PAR)
            ops.mulMat(p, l.ffn_gate, self.b_norm.ptr, n_t, self.b_ffn_g.ptr);
            p.barrier();
            ops.mulMat(p, l.ffn_up, self.b_norm.ptr, n_t, self.b_ffn_u.ptr);
            p.barrier();
            ops.swiglu(p, m.n_ff, n_t, self.b_ffn_g.ptr, self.b_ffn_g.ptr, self.b_ffn_u.ptr);
            p.barrier();
            ops.mulMat(p, l.ffn_down, self.b_ffn_g.ptr, n_t, self.b_attn.ptr);
            p.barrier();
            // l_out
            ops.add(p, m.n_embd, n_t, self.b_inp.ptr, self.b_attn.ptr, self.b_resid.ptr);
            p.barrier();
        }

        if (self.want_logits) {
            const last = self.b_inp.ptr + (n_t - 1) * m.n_embd;
            ops.rmsNorm(p, last, m.n_embd, 0, self.b_norm.ptr, m.n_embd, 0, m.n_embd, 1, 1, m.eps, m.output_norm, 1.0);
            p.barrier();
            ops.mulMat(p, m.output, self.b_norm.ptr, 1, self.logits.ptr);
            p.barrier();
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

        ops.mulMat(p, l.wqkv, self.b_norm.ptr, n_t, self.b_qkv.ptr);
        p.barrier();
        // CONCAT + CPY(conv_state_last)
        ops.convConcat(p, conv_dim, kc, n_t, self.rs_conv[il].ptr, self.b_qkv.ptr, self.b_conv_in.ptr);
        p.barrier();
        ops.ssmConv(p, conv_dim, m.d_conv, kc + n_t, n_t, self.b_conv_in.ptr, l.ssm_conv1d, self.b_conv_out.ptr);
        p.barrier();
        ops.silu(p, conv_dim, n_t, self.b_conv_out.ptr, self.b_conv_out.ptr);
        p.barrier();
        // build_gdn_l2_norm(q), (k)
        ops.rmsNorm(p, self.b_conv_out.ptr, s, conv_dim, self.b_q.ptr, s, m.d_inner, s, m.n_group, n_t, eps_l2, null, l2);
        p.barrier();
        ops.rmsNorm(p, self.b_conv_out.ptr + key_dim, s, conv_dim, self.b_k.ptr, s, m.d_inner, s, m.n_group, n_t, eps_l2, null, l2);
        p.barrier();
        // alpha -> gate
        ops.mulMat(p, l.ssm_alpha, self.b_norm.ptr, n_t, self.b_gate.ptr);
        p.barrier();
        ops.gateFromAlpha(p, h_n, n_t, self.b_gate.ptr, self.b_gate.ptr, l.ssm_dt, l.ssm_a);
        p.barrier();
        // beta
        ops.mulMat(p, l.ssm_beta, self.b_norm.ptr, n_t, self.b_beta.ptr);
        p.barrier();
        ops.sigmoidInPlace(p, h_n, n_t, self.b_beta.ptr);
        p.barrier();
        // GATED_DELTA_NET (+ CPY new_state: yerinde)
        ops.gatedDeltaNet(p, s, h_n, n_t, self.b_q.ptr, self.b_k.ptr, self.b_conv_out.ptr + 2 * key_dim, conv_dim, self.b_gate.ptr, self.b_beta.ptr, self.rs_state[il].ptr, self.b_gdn.ptr);
        p.barrier();
        // build_norm_gated
        ops.rmsNorm(p, self.b_gdn.ptr, s, m.d_inner, self.b_gdn.ptr, s, m.d_inner, s, h_n, n_t, m.eps, l.ssm_norm, 1.0);
        p.barrier();
        ops.mulMat(p, l.wgate, self.b_norm.ptr, n_t, self.b_z.ptr);
        p.barrier();
        ops.silu(p, s, h_n * n_t, self.b_z.ptr, self.b_z.ptr);
        p.barrier();
        ops.mul(p, s, h_n * n_t, self.b_gdn.ptr, self.b_gdn.ptr, self.b_z.ptr);
        p.barrier();
        ops.mulMat(p, l.ssm_out, self.b_gdn.ptr, n_t, self.b_attn.ptr);
        p.barrier();
    }

    /// llama_model_qwen35::graph::build_layer_attn + build_attn_mha (flash attention)
    fn layerAttn(self: *Context, p: *const Params, l: *const Layer, il: usize) void {
        const m = self.model;
        const n_t = self.n_t;
        const dh = m.head_dim;
        const kv_row = dh * m.n_head_kv;

        ops.mulMat(p, l.wq, self.b_norm.ptr, n_t, self.b_qfull.ptr);
        p.barrier();
        // Qcur = view(Qcur_full) -> build_norm -> rope. q tamponu: b_gdn (head_dim*n_head*n_t)
        ops.rmsNorm(p, self.b_qfull.ptr, dh * 2, dh * m.n_head * 2, self.b_gdn.ptr, dh, dh * m.n_head, dh, m.n_head, n_t, m.eps, l.q_norm, 1.0);
        p.barrier();
        ops.rope(p, self.b_gdn.ptr, dh, m.n_head, n_t, self.head, m.n_rot, self.theta_scale);
        p.barrier();
        ops.mulMat(p, l.wv, self.b_norm.ptr, n_t, self.b_vv.ptr);
        p.barrier();
        ops.mulMat(p, l.wk, self.b_norm.ptr, n_t, self.b_kk.ptr);
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
        ops.gateAttn(p, n_t, m.n_head, dh, self.b_q.ptr, self.b_qfull.ptr);
        p.barrier();
        ops.mulMat(p, l.wo, self.b_q.ptr, n_t, self.b_attn.ptr);
        p.barrier();
    }
};
