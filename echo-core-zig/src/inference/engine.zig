const std = @import("std");
const config = @import("core_config");
const memory = @import("../core/memory.zig");
const types = @import("../core/types.zig");
const matvec = @import("../kernels/matvec.zig");
const qwen_linear = @import("../kernels/qwen_linear.zig");
const ssm = @import("../kernels/ssm.zig");
const tokenizer = @import("../tokenizer/tokenizer.zig");
const kv_cache = @import("../kv_cache/cache.zig");
const math = @import("../core/math.zig");
const avx2_softmax = @import("../kernels/avx2_softmax.zig");
const avx2_norm = @import("../kernels/avx2_norm.zig");
const parallel = @import("../kernels/parallel.zig");
const int_dot = @import("../kernels/int_dot.zig");
const quantK = @import("../kernels/quant.zig");
const gguf = @import("../gguf/reader.zig");

const ArrayList = std.array_list.Managed;

/// Helper: Cast byte slice to fp16 slice
fn fp16SliceFromBytes(bytes: []u8) []types.fp16_t {
    const n_elements = bytes.len / @sizeOf(types.fp16_t);
    return @as([*]types.fp16_t, @ptrCast(@alignCast(bytes.ptr)))[0..n_elements];
}

fn rowByteSize(cols: u32, dtype: gguf.GGMLType) usize {
    return switch (dtype) {
        .f16 => cols * @sizeOf(types.fp16_t),
        .f32 => cols * @sizeOf(f32),
        .q8_0 => (cols / 32) * 34,
        .q4_k => (cols / 256) * 144,
        .q5_k => (cols / 256) * 176,
        .q6_k => (cols / 256) * 210,
        .q2_k => (cols / 256) * 84,
        .q3_k => (cols / 256) * 110,
        .iq4_xs => (cols / 256) * 136,
        .iq2_xs => (cols / 256) * 74,
        else => cols * @sizeOf(types.fp16_t),
    };
}

fn loadEmbeddingRowToF32(src: []const u8, dtype: gguf.GGMLType, cols: u32, dst: []f32) void {
    switch (dtype) {
        .f16 => {
            const row = @as([*]const types.fp16_t, @ptrCast(@alignCast(src.ptr)))[0..cols];
            types.fp16_to_fp32_row(row.ptr, dst.ptr, cols);
        },
        .f32 => {
            const row = @as([*]const f32, @ptrCast(@alignCast(src.ptr)))[0..cols];
            @memcpy(dst[0..cols], row);
        },
        .q8_0 => {
            const block_stride: usize = 34;
            const blocks_per_row = cols / 32;
            for (0..blocks_per_row) |block_idx| {
                const bp = src[block_idx * block_stride ..][0..block_stride];
                const d = types.fp16_to_fp32(std.mem.readInt(u16, bp[0..2], .little));
                const qs = @as([*]const i8, @ptrCast(bp.ptr + 2));
                for (0..32) |j| {
                    dst[block_idx * 32 + j] = d * @as(f32, @floatFromInt(qs[j]));
                }
            }
        },
        .q4_k => {
            const blocks_per_row = cols / 256;
            for (0..blocks_per_row) |b| {
                const bp = src[b * 144 ..][0..144];
                const d = types.fp16_to_fp32(std.mem.readInt(u16, bp[0..2], .little));
                const dmin = types.fp16_to_fp32(std.mem.readInt(u16, bp[2..4], .little));
                const scales = bp[4..];
                const qs = bp[16..];
                for (0..4) |blk| {
                    const js = blk * 2;
                    const sc0: u8 = if (js < 4) scales[js] & 63 else (scales[js + 4] & 0x0F) | ((scales[js - 4] >> 6) << 4);
                    const mn0: u8 = if (js < 4) scales[js + 4] & 63 else (scales[js + 4] >> 4) | ((scales[js] >> 6) << 4);
                    const sc1: u8 = if (js + 1 < 4) scales[js + 1] & 63 else (scales[js + 1 + 4] & 0x0F) | ((scales[js + 1 - 4] >> 6) << 4);
                    const mn1: u8 = if (js + 1 < 4) scales[js + 1 + 4] & 63 else (scales[js + 1 + 4] >> 4) | ((scales[js + 1] >> 6) << 4);
                    const rs0 = d * @as(f32, @floatFromInt(sc0));
                    const rm0 = dmin * @as(f32, @floatFromInt(mn0));
                    const rs1 = d * @as(f32, @floatFromInt(sc1));
                    const rm1 = dmin * @as(f32, @floatFromInt(mn1));
                    const qoff = blk * 32;
                    const woff = blk * 64;
                    for (0..16) |j| {
                        dst[b * 256 + woff + j] = rs0 * @as(f32, @floatFromInt(qs[qoff + j] & 0x0F)) - rm0;
                        dst[b * 256 + woff + 16 + j] = rs1 * @as(f32, @floatFromInt(qs[qoff + 16 + j] & 0x0F)) - rm1;
                        dst[b * 256 + woff + 32 + j] = rs0 * @as(f32, @floatFromInt(qs[qoff + j] >> 4)) - rm0;
                        dst[b * 256 + woff + 48 + j] = rs1 * @as(f32, @floatFromInt(qs[qoff + 16 + j] >> 4)) - rm1;
                    }
                }
            }
        },
        .q6_k => {
            const blocks_per_row = cols / 256;
            for (0..blocks_per_row) |b| {
                const bp = src[b * 210 ..][0..210];
                const d = types.fp16_to_fp32(std.mem.readInt(u16, bp[208..210], .little));
                var n: usize = 0;
                while (n < 256) : (n += 128) {
                    var l: usize = 0;
                    while (l < 32) : (l += 1) {
                        const is = l / 16;
                        const qh = bp[128 + n / 4 + l];
                        const q1: i32 = @intCast((bp[n / 2 + l] & 0x0F) | (((qh >> 0) & 0x03) << 4));
                        const q2: i32 = @intCast((bp[n / 2 + 32 + l] & 0x0F) | (((qh >> 2) & 0x03) << 4));
                        const q3: i32 = @intCast((bp[n / 2 + l] >> 4) | (((qh >> 4) & 0x03) << 4));
                        const q4: i32 = @intCast((bp[n / 2 + 32 + l] >> 4) | (((qh >> 6) & 0x03) << 4));
                        const s0: i32 = @as(i8, @bitCast(bp[192 + n / 16 + is + 0]));
                        const s1: i32 = @as(i8, @bitCast(bp[192 + n / 16 + is + 2]));
                        const s2: i32 = @as(i8, @bitCast(bp[192 + n / 16 + is + 4]));
                        const s3: i32 = @as(i8, @bitCast(bp[192 + n / 16 + is + 6]));
                        dst[b * 256 + n + l] = d * @as(f32, @floatFromInt(s0)) * @as(f32, @floatFromInt(q1 - 32));
                        dst[b * 256 + n + l + 32] = d * @as(f32, @floatFromInt(s1)) * @as(f32, @floatFromInt(q2 - 32));
                        dst[b * 256 + n + l + 64] = d * @as(f32, @floatFromInt(s2)) * @as(f32, @floatFromInt(q3 - 32));
                        dst[b * 256 + n + l + 96] = d * @as(f32, @floatFromInt(s3)) * @as(f32, @floatFromInt(q4 - 32));
                    }
                }
            }
        },
        else => unreachable,
    }
}

/// Get dtype for a layer tensor (Q, K, V, O projections, FFN weights)
const layer_slot_stride: u32 = 19;

fn dtypeForTensor(weight_dtypes: []const gguf.GGMLType, layer_idx: u32, tensor_idx: u32) gguf.GGMLType {
    const slot = 1 + layer_idx * layer_slot_stride + tensor_idx;
    return weight_dtypes[slot];
}

fn dtypeForLayerSlot(weight_dtypes: []const gguf.GGMLType, layer_idx: u32, slot_idx: u32) gguf.GGMLType {
    const slot = 1 + layer_idx * layer_slot_stride + slot_idx;
    return weight_dtypes[slot];
}

/// Get dtype for global tensors (token_embedding, final_norm, output_proj)
fn dtypeForGlobal(weight_dtypes: []const gguf.GGMLType, global_idx: u32) gguf.GGMLType {
    // global_idx: 0 = token_embedding, 1 = final_norm, 2 = output_proj
    if (global_idx == 0) return weight_dtypes[0]; // token_embedding
    const n_slots: u32 = @intCast(weight_dtypes.len);
    return weight_dtypes[n_slots - 2 + (global_idx - 1)];
}

fn packedSsmBase(layout: *const memory.WeightLayout, layer_idx: u32) usize {
    const packed_idx = memory.ssmPackedLayerIndex(layout.layer_types, @intCast(layer_idx)) orelse unreachable;
    return layout.ssm_region_offset + packed_idx * layout.ssm_per_layer_size;
}

pub const Engine = struct {
    allocator: std.mem.Allocator,
    config: config.ModelConfig,
    weight_layout: memory.WeightLayout,
    weight_pool: []u8,
    weight_dtypes: []gguf.GGMLType,
    kv_cache: ?kv_cache.KVCache,
    ssm_states: []ssm.SSMState, // Per-layer SSM states (only for SSM layers)
    qwen_states: []qwen_linear.QwenLinearState,
    hidden_state: []f32,
    residual: []f32,
    attn_out: []f32,
    attn_proj: []f32,
    ffn_out: []f32,
    q_proj: []f32,
    k_proj: []f32,
    v_proj: []f32,
    attn_accum: []f32,
    scores: []f32,
    head_q: []f32,
    head_out: []f32,
    ffn_scratch: []f32,
    ffn_gate_buf: []f32,
    ffn_up_buf: []f32,
    logits: []f32,
    // SSM temporary buffers
    ssm_tmp_x: []f32,
    ssm_tmp_z: []f32,
    ssm_tmp_dt: []f32,
    ssm_tmp_B: []f32,
    ssm_tmp_C: []f32,
    qwen_tmp_mixed_qkv: []f32,
    qwen_tmp_conv_out: []f32,
    qwen_tmp_z: []f32,
    qwen_tmp_core: []f32,
    qwen_tmp_alpha: []f32,
    qwen_tmp_beta: []f32,
    x_q8: []u8, // Pre-quantized Q8_K representation of x (K/256 * 292 bytes)
    current_layer_base: usize,
    seq_pos: u32,

    pub fn init(cfg: config.ModelConfig, reader_opt: ?*const gguf.Reader, allocator: std.mem.Allocator) !Engine {
        var layout = try memory.WeightLayout.compute(cfg, reader_opt, allocator);
        errdefer layout.deinit(allocator);

        const kv_dim = cfg.num_kv_heads * cfg.head_dim;
        const q_dim = cfg.num_heads * cfg.head_dim;
        const q_proj_dim = if (cfg.full_attention_interval > 0) q_dim * 2 else q_dim;

        const weight_pool = try allocator.alloc(u8, layout.raw_pool_size);
        errdefer allocator.free(weight_pool);
        @memset(weight_pool, 0);

        // Per-tensor dtype tracking (default to fp16)
        // Slot layout: 1 token_embd + num_layers * (11 attention + 8 ssm) + final_norm + output_proj
        const n_slots = 1 + cfg.num_layers * (11 + 8) + 2;
        const weight_dtypes = try allocator.alloc(gguf.GGMLType, n_slots);
        errdefer allocator.free(weight_dtypes);
        @memset(weight_dtypes, .f16);

        const cache: ?kv_cache.KVCache = null;

        const hidden_state = try allocator.alloc(f32, cfg.hidden_dim);
        errdefer allocator.free(hidden_state);
        const residual = try allocator.alloc(f32, cfg.hidden_dim);
        errdefer allocator.free(residual);
        const attn_out = try allocator.alloc(f32, cfg.hidden_dim);
        errdefer allocator.free(attn_out);
        const attn_proj = try allocator.alloc(f32, cfg.hidden_dim);
        errdefer allocator.free(attn_proj);
        const ffn_out = try allocator.alloc(f32, cfg.hidden_dim);
        errdefer allocator.free(ffn_out);
        const q_proj = try allocator.alloc(f32, q_proj_dim);
        errdefer allocator.free(q_proj);
        const k_proj = try allocator.alloc(f32, kv_dim);
        errdefer allocator.free(k_proj);
        const v_proj = try allocator.alloc(f32, kv_dim);
        errdefer allocator.free(v_proj);
        // attention accumulates into attn_accum before projecting
        const attn_accum = try allocator.alloc(f32, q_dim);
        errdefer allocator.free(attn_accum);
        const scores = try allocator.alloc(f32, if (cfg.max_seq_len > 0) cfg.max_seq_len else 1);
        errdefer allocator.free(scores);
        const head_q = try allocator.alloc(f32, if (cfg.head_dim > 0) cfg.head_dim else 1);
        errdefer allocator.free(head_q);
        const head_out = try allocator.alloc(f32, if (cfg.head_dim > 0) cfg.head_dim else 1);
        errdefer allocator.free(head_out);
        const ffn_scratch = try allocator.alloc(f32, if (cfg.ffn_hidden_dim > 0) cfg.ffn_hidden_dim else 1);
        errdefer allocator.free(ffn_scratch);
        const ffn_gate_buf = try allocator.alloc(f32, if (cfg.ffn_hidden_dim > 0) cfg.ffn_hidden_dim else 1);
        errdefer allocator.free(ffn_gate_buf);
        const ffn_up_buf = try allocator.alloc(f32, if (cfg.ffn_hidden_dim > 0) cfg.ffn_hidden_dim else 1);
        errdefer allocator.free(ffn_up_buf);
        const logits = try allocator.alloc(f32, cfg.vocab_size);
        errdefer allocator.free(logits);

        // Allocate SSM states and temporary buffers (only if model has SSM layers)
        const ssm_states = try allocator.alloc(ssm.SSMState, cfg.num_layers);
        errdefer allocator.free(ssm_states);
        @memset(ssm_states, std.mem.zeroes(ssm.SSMState));

        const qwen_states = try allocator.alloc(qwen_linear.QwenLinearState, cfg.num_layers);
        errdefer allocator.free(qwen_states);
        @memset(qwen_states, std.mem.zeroes(qwen_linear.QwenLinearState));

        const ssm_tmp_x = try allocator.alloc(f32, if (cfg.hidden_dim > 0) cfg.hidden_dim else 1);
        errdefer allocator.free(ssm_tmp_x);
        const ssm_tmp_z = try allocator.alloc(f32, if (cfg.hidden_dim > 0) cfg.hidden_dim else 1);
        errdefer allocator.free(ssm_tmp_z);
        const ssm_tmp_dt = try allocator.alloc(f32, if (cfg.ssm_dt_rank > 0) cfg.ssm_dt_rank else 1);
        errdefer allocator.free(ssm_tmp_dt);
        const ssm_tmp_B = try allocator.alloc(f32, if (cfg.ssm_inner_size > 0) cfg.ssm_inner_size else 1);
        errdefer allocator.free(ssm_tmp_B);
        const ssm_tmp_C = try allocator.alloc(f32, if (cfg.ssm_inner_size > 0) cfg.ssm_inner_size else 1);
        errdefer allocator.free(ssm_tmp_C);

        const qwen_tmp_mixed_qkv = try allocator.alloc(f32, if (cfg.ssm_inner_size > 0) cfg.ssm_inner_size * 3 else 1);
        errdefer allocator.free(qwen_tmp_mixed_qkv);
        const qwen_tmp_conv_out = try allocator.alloc(f32, if (cfg.ssm_inner_size > 0) cfg.ssm_inner_size * 3 else 1);
        errdefer allocator.free(qwen_tmp_conv_out);
        const qwen_tmp_z = try allocator.alloc(f32, if (cfg.ssm_inner_size > 0) cfg.ssm_inner_size else 1);
        errdefer allocator.free(qwen_tmp_z);
        const qwen_tmp_core = try allocator.alloc(f32, if (cfg.ssm_inner_size > 0) cfg.ssm_inner_size else 1);
        errdefer allocator.free(qwen_tmp_core);
        const qwen_tmp_alpha = try allocator.alloc(f32, if (cfg.ssm_dt_rank > 0) cfg.ssm_dt_rank else 1);
        errdefer allocator.free(qwen_tmp_alpha);
        const qwen_tmp_beta = try allocator.alloc(f32, if (cfg.ssm_dt_rank > 0) cfg.ssm_dt_rank else 1);
        errdefer allocator.free(qwen_tmp_beta);

        const max_k: usize = @intCast(@max(cfg.hidden_dim, @max(cfg.ffn_hidden_dim, @max(cfg.num_heads * cfg.head_dim, cfg.num_kv_heads * cfg.head_dim))));
        const x_q8_size = ((max_k + 255) / 256) * 292;
        const x_q8 = try allocator.alloc(u8, x_q8_size);
        errdefer allocator.free(x_q8);

        @memset(hidden_state, 0);
        @memset(residual, 0);
        @memset(attn_out, 0);
        @memset(attn_proj, 0);
        @memset(ffn_out, 0);
        @memset(q_proj, 0);
        @memset(k_proj, 0);
        @memset(v_proj, 0);
        @memset(attn_accum, 0);
        @memset(scores, 0);
        @memset(head_q, 0);
        @memset(head_out, 0);
        @memset(ffn_scratch, 0);
        @memset(ffn_gate_buf, 0);
        @memset(ffn_up_buf, 0);
        @memset(logits, 0);
        @memset(ssm_tmp_x, 0);
        @memset(ssm_tmp_z, 0);
        @memset(ssm_tmp_dt, 0);
        @memset(ssm_tmp_B, 0);
        @memset(ssm_tmp_C, 0);
        @memset(qwen_tmp_mixed_qkv, 0);
        @memset(qwen_tmp_conv_out, 0);
        @memset(qwen_tmp_z, 0);
        @memset(qwen_tmp_core, 0);
        @memset(qwen_tmp_alpha, 0);
        @memset(qwen_tmp_beta, 0);
        @memset(x_q8, 0);

        return .{
            .allocator = allocator,
            .config = cfg,
            .weight_layout = layout,
            .weight_pool = weight_pool,
            .weight_dtypes = weight_dtypes,
            .kv_cache = cache,
            .ssm_states = ssm_states,
            .qwen_states = qwen_states,
            .hidden_state = hidden_state,
            .residual = residual,
            .attn_out = attn_out,
            .attn_proj = attn_proj,
            .ffn_out = ffn_out,
            .q_proj = q_proj,
            .k_proj = k_proj,
            .v_proj = v_proj,
            .attn_accum = attn_accum,
            .scores = scores,
            .head_q = head_q,
            .head_out = head_out,
            .ffn_scratch = ffn_scratch,
            .ffn_gate_buf = ffn_gate_buf,
            .ffn_up_buf = ffn_up_buf,
            .logits = logits,
            .ssm_tmp_x = ssm_tmp_x,
            .ssm_tmp_z = ssm_tmp_z,
            .ssm_tmp_dt = ssm_tmp_dt,
            .ssm_tmp_B = ssm_tmp_B,
            .ssm_tmp_C = ssm_tmp_C,
            .qwen_tmp_mixed_qkv = qwen_tmp_mixed_qkv,
            .qwen_tmp_conv_out = qwen_tmp_conv_out,
            .qwen_tmp_z = qwen_tmp_z,
            .qwen_tmp_core = qwen_tmp_core,
            .qwen_tmp_alpha = qwen_tmp_alpha,
            .qwen_tmp_beta = qwen_tmp_beta,
            .x_q8 = x_q8,
            .current_layer_base = 0,
            .seq_pos = 0,
        };
    }

    pub fn deinit(self: *Engine, allocator: std.mem.Allocator) void {
        self.weight_layout.deinit(allocator);
        allocator.free(self.weight_pool);
        allocator.free(self.weight_dtypes);
        if (self.kv_cache) |*cache| cache.deinit(allocator);
        allocator.free(self.hidden_state);
        allocator.free(self.residual);
        allocator.free(self.attn_out);
        allocator.free(self.attn_proj);
        allocator.free(self.ffn_out);
        allocator.free(self.q_proj);
        allocator.free(self.k_proj);
        allocator.free(self.v_proj);
        allocator.free(self.attn_accum);
        allocator.free(self.scores);
        allocator.free(self.head_q);
        allocator.free(self.head_out);
        allocator.free(self.ffn_scratch);
        allocator.free(self.ffn_gate_buf);
        allocator.free(self.ffn_up_buf);
        allocator.free(self.logits);
        allocator.free(self.ssm_tmp_x);
        allocator.free(self.ssm_tmp_z);
        allocator.free(self.ssm_tmp_dt);
        allocator.free(self.ssm_tmp_B);
        allocator.free(self.ssm_tmp_C);
        allocator.free(self.qwen_tmp_mixed_qkv);
        allocator.free(self.qwen_tmp_conv_out);
        allocator.free(self.qwen_tmp_z);
        allocator.free(self.qwen_tmp_core);
        allocator.free(self.qwen_tmp_alpha);
        allocator.free(self.qwen_tmp_beta);
        allocator.free(self.x_q8);

        // Free SSM states
        for (self.ssm_states) |*state| {
            if (state.conv_state.len > 0 or state.ssm_state.len > 0) {
                state.deinit(allocator);
            }
        }
        allocator.free(self.ssm_states);

        for (self.qwen_states) |*state| {
            if (state.conv_state.len > 0 or state.recurrent_state.len > 0) {
                state.deinit(allocator);
            }
        }
        allocator.free(self.qwen_states);
    }

    pub fn reset(self: *Engine) void {
        if (self.kv_cache) |*cache| cache.reset();
        for (self.ssm_states) |*state| {
            if (state.conv_state.len > 0) {
                state.reset();
            }
        }
        for (self.qwen_states) |*state| {
            if (state.conv_state.len > 0) {
                state.reset();
            }
        }
        self.seq_pos = 0;
    }

    fn ensureKvCache(self: *Engine) !void {
        if (self.kv_cache == null and self.config.max_seq_len > 0 and self.config.num_layers > 0) {
            self.kv_cache = try kv_cache.KVCache.init(self.config, self.allocator);
        }
    }

    pub fn loadWeights(self: *Engine, weights: []const types.fp16_t) !void {
        const weight_bytes = @as([]const u8, @ptrCast(weights));
        if (weight_bytes.len != self.weight_pool.len) return error.InvalidWeights;
        @memcpy(self.weight_pool, weight_bytes);
    }

    /// Process a single token at self.seq_pos, return logits.
    pub fn forwardToken(self: *Engine, token_id: u32) ![]f32 {
        try self.ensureKvCache();

        const hidden = self.config.hidden_dim;
        const vocab = self.config.vocab_size;
        const embedding_dtype = dtypeForGlobal(self.weight_dtypes, 0);
        const embedding_row_bytes = rowByteSize(hidden, embedding_dtype);
        const embed_byte_offset = self.weight_layout.token_embedding_offset + @as(usize, token_id) * embedding_row_bytes;
        const emb_bytes = self.weight_pool[embed_byte_offset..][0..embedding_row_bytes];
        loadEmbeddingRowToF32(emb_bytes, embedding_dtype, hidden, self.hidden_state);

        for (0..self.config.num_layers) |layer_idx| {
            self.layerForward(@intCast(layer_idx), self.hidden_state, self.hidden_state);
        }

        const final_norm_byte_offset = self.weight_layout.final_norm_offset;
        const final_norm_bytes = self.weight_pool[final_norm_byte_offset..][0 .. hidden * @sizeOf(types.fp16_t)];
        const final_norm = fp16SliceFromBytes(final_norm_bytes);
        self.norm(self.hidden_state, self.hidden_state, final_norm);

        @memset(self.logits, 0);
        const output_proj_byte_offset = self.weight_layout.output_proj_offset;
        const output_proj_size = self.weight_layout.ssm_region_offset - self.weight_layout.output_proj_offset;
        const output_proj_bytes = self.weight_pool[output_proj_byte_offset..][0..output_proj_size];
        const output_dtype = dtypeForGlobal(self.weight_dtypes, 2);
        parallel.parallelMatvec(output_proj_bytes.ptr, self.hidden_state.ptr, self.logits.ptr, vocab, hidden, output_dtype);

        self.seq_pos += 1;
        return self.logits;
    }

    /// Legacy: process last token from list (for backward compat with tests).
    pub fn forward(self: *Engine, input_ids: []const u32) ![]f32 {
        if (input_ids.len == 0) return error.EmptyInput;
        return self.forwardToken(input_ids[input_ids.len - 1]);
    }

    fn layerForward(self: *Engine, layer_idx: u32, input: []const f32, output: []f32) void {
        const hidden = self.config.hidden_dim;
        const layer_type = self.weight_layout.layer_types[layer_idx];

        if (layer_type == .ssm) {
            // SSM layer path
            const layer_base = self.weight_layout.layer_offsets[layer_idx];
            self.current_layer_base = layer_base;

            // SSM layers don't have separate attn_norm/ffn_norm in traditional sense
            // They have their own normalization usually
            const layer_norm_byte_offset = layer_base; // SSM norm is at start
            const layer_norm_bytes = self.weight_pool[layer_norm_byte_offset..][0 .. hidden * @sizeOf(types.fp16_t)];
            const layer_norm = fp16SliceFromBytes(layer_norm_bytes);

            @memcpy(self.residual, input);
            self.norm(input, output, layer_norm);
            self.ssmLayerForward(layer_idx, output, self.attn_proj);
            for (0..hidden) |i| output[i] = self.residual[i] + self.attn_proj[i];

            // FFN part of SSM layer (optional, depending on architecture)
            @memcpy(self.residual, output);
            self.ffn(layer_idx, output, self.ffn_out);
            for (0..hidden) |i| output[i] = self.residual[i] + self.ffn_out[i];
        } else if (layer_type == .qwen_linear) {
            const layer_base = self.weight_layout.layer_offsets[layer_idx];
            self.current_layer_base = layer_base;

            const layer_norm_byte_offset = layer_base + self.weight_layout.norm_weight_offset;
            const layer_norm_bytes = self.weight_pool[layer_norm_byte_offset..][0 .. hidden * @sizeOf(types.fp16_t)];
            const layer_norm = fp16SliceFromBytes(layer_norm_bytes);
            const ffn_norm_byte_offset = layer_base + self.weight_layout.ffn_norm_offset;
            const ffn_norm_bytes = self.weight_pool[ffn_norm_byte_offset..][0 .. hidden * @sizeOf(types.fp16_t)];
            const ffn_norm = fp16SliceFromBytes(ffn_norm_bytes);

            @memcpy(self.residual, input);
            self.norm(input, output, layer_norm);
            self.qwenLinearLayerForward(layer_idx, output, self.attn_proj);
            for (0..hidden) |i| output[i] = self.residual[i] + self.attn_proj[i];
            @memcpy(self.residual, output);
            self.norm(output, output, ffn_norm);
            self.ffn(layer_idx, output, self.ffn_out);
            for (0..hidden) |i| output[i] = self.residual[i] + self.ffn_out[i];
        } else {
            const layer_base = self.weight_layout.layer_offsets[layer_idx];
            self.current_layer_base = layer_base;

            const layer_norm_byte_offset = layer_base + self.weight_layout.norm_weight_offset;
            const layer_norm_bytes = self.weight_pool[layer_norm_byte_offset..][0 .. hidden * @sizeOf(types.fp16_t)];
            const layer_norm = fp16SliceFromBytes(layer_norm_bytes);
            const ffn_norm_byte_offset = layer_base + self.weight_layout.ffn_norm_offset;
            const ffn_norm_bytes = self.weight_pool[ffn_norm_byte_offset..][0 .. hidden * @sizeOf(types.fp16_t)];
            const ffn_norm = fp16SliceFromBytes(ffn_norm_bytes);

            @memcpy(self.residual, input);
            self.norm(input, output, layer_norm);
            self.attention(layer_idx, output, self.attn_proj);
            for (0..hidden) |i| output[i] = self.residual[i] + self.attn_proj[i];
            @memcpy(self.residual, output);
            self.norm(output, output, ffn_norm);
            self.ffn(layer_idx, output, self.ffn_out);
            for (0..hidden) |i| output[i] = self.residual[i] + self.ffn_out[i];
        }
    }

    /// Apply RoPE to Q and K projections in-place.
    fn applyRoPE(self: *Engine) void {
        const head_dim = self.config.head_dim;
        const num_heads = self.config.num_heads;
        const num_kv_heads = self.config.num_kv_heads;
        const pos = self.seq_pos;
        const rope_base: f32 = 10000000.0;
        const rotary_dim: usize = 64;

        // Apply to Q heads
        for (0..num_heads) |h| {
            const offset = h * head_dim;
            var i: usize = 0;
            while (i < rotary_dim) : (i += 2) {
                const freq = 1.0 / std.math.pow(f32, rope_base, @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(rotary_dim)));
                const angle = @as(f32, @floatFromInt(pos)) * freq;
                const cos_val = @cos(angle);
                const sin_val = @sin(angle);
                const q0 = self.q_proj[offset + i];
                const q1 = self.q_proj[offset + i + 1];
                self.q_proj[offset + i] = q0 * cos_val - q1 * sin_val;
                self.q_proj[offset + i + 1] = q0 * sin_val + q1 * cos_val;
            }
        }

        // Apply to K heads
        for (0..num_kv_heads) |h| {
            const offset = h * head_dim;
            var i: usize = 0;
            while (i < rotary_dim) : (i += 2) {
                const freq = 1.0 / std.math.pow(f32, rope_base, @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(rotary_dim)));
                const angle = @as(f32, @floatFromInt(pos)) * freq;
                const cos_val = @cos(angle);
                const sin_val = @sin(angle);
                const k0 = self.k_proj[offset + i];
                const k1 = self.k_proj[offset + i + 1];
                self.k_proj[offset + i] = k0 * cos_val - k1 * sin_val;
                self.k_proj[offset + i + 1] = k0 * sin_val + k1 * cos_val;
            }
        }
    }

    /// Apply per-head RMS norm using weight stored at given offset (head_dim elements).
    fn applyHeadNorm(proj: []f32, num_heads_for_proj: u32, head_dim: u32, norm_weight: []const types.fp16_t) void {
        for (0..num_heads_for_proj) |h| {
            const offset = h * head_dim;
            avx2_norm.rmsNormAvx2(proj.ptr + offset, proj.ptr + offset, norm_weight.ptr, head_dim);
        }
    }

    fn attention(self: *Engine, layer_idx: u32, input: []const f32, output: []f32) void {
        const hidden = self.config.hidden_dim;
        const num_heads = self.config.num_heads;
        const num_kv_heads = self.config.num_kv_heads;
        const head_dim = self.config.head_dim;
        const q_dim = num_heads * head_dim;
        const is_qwen35 = self.config.full_attention_interval > 0;
        const q_proj_dim = if (is_qwen35) q_dim * 2 else q_dim;
        const kv_dim = num_kv_heads * head_dim;
        const layer_base = self.weight_layout.layer_offsets[layer_idx];

        const q_proj_offset = layer_base + self.weight_layout.q_proj_offset;
        const k_proj_offset = layer_base + self.weight_layout.k_proj_offset;
        const v_proj_offset = layer_base + self.weight_layout.v_proj_offset;
        const o_proj_offset = layer_base + self.weight_layout.o_proj_offset;

        @memset(self.q_proj, 0);
        @memset(self.k_proj, 0);
        @memset(self.v_proj, 0);

        parallel.parallelMatvec(self.weight_pool[q_proj_offset..].ptr, input.ptr, self.q_proj.ptr, q_proj_dim, hidden, dtypeForTensor(self.weight_dtypes, layer_idx, 1));
        parallel.parallelMatvec(self.weight_pool[k_proj_offset..].ptr, input.ptr, self.k_proj.ptr, kv_dim, hidden, dtypeForTensor(self.weight_dtypes, layer_idx, 2));
        parallel.parallelMatvec(self.weight_pool[v_proj_offset..].ptr, input.ptr, self.v_proj.ptr, kv_dim, hidden, dtypeForTensor(self.weight_dtypes, layer_idx, 3));

        // Apply Q/K head norms if weights are non-zero
        const q_norm_byte_offset = layer_base + self.weight_layout.attn_q_norm_offset;
        const k_norm_byte_offset = layer_base + self.weight_layout.attn_k_norm_offset;
        const q_norm_bytes = self.weight_pool[q_norm_byte_offset..][0 .. head_dim * @sizeOf(types.fp16_t)];
        const k_norm_bytes = self.weight_pool[k_norm_byte_offset..][0 .. head_dim * @sizeOf(types.fp16_t)];
        const q_norm_w = fp16SliceFromBytes(q_norm_bytes);
        const k_norm_w = fp16SliceFromBytes(k_norm_bytes);
        if (!std.mem.allEqual(types.fp16_t, q_norm_w, 0)) {
            if (is_qwen35) {
                for (0..num_heads) |h| {
                    const offset = h * head_dim * 2;
                    avx2_norm.rmsNormAvx2(self.q_proj.ptr + offset, self.q_proj.ptr + offset, q_norm_w.ptr, head_dim);
                }
            } else {
                applyHeadNorm(self.q_proj, num_heads, head_dim, q_norm_w);
            }
        }
        if (!std.mem.allEqual(types.fp16_t, k_norm_w, 0)) {
            applyHeadNorm(self.k_proj[0..kv_dim], num_kv_heads, head_dim, k_norm_w);
        }

        // Apply RoPE
        if (self.config.pos_encoding == .rope) {
            if (is_qwen35) {
                const pos = self.seq_pos;
                const rope_base: f32 = 10000000.0;
                const rotary_dim: usize = 64;
                for (0..num_heads) |h| {
                    const offset = h * head_dim * 2;
                    var i: usize = 0;
                    while (i < rotary_dim) : (i += 2) {
                        const freq = 1.0 / std.math.pow(f32, rope_base, @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(rotary_dim)));
                        const angle = @as(f32, @floatFromInt(pos)) * freq;
                        const cos_val = @cos(angle);
                        const sin_val = @sin(angle);
                        const q0 = self.q_proj[offset + i];
                        const q1 = self.q_proj[offset + i + 1];
                        self.q_proj[offset + i] = q0 * cos_val - q1 * sin_val;
                        self.q_proj[offset + i + 1] = q0 * sin_val + q1 * cos_val;
                    }
                }
                for (0..num_kv_heads) |h| {
                    const offset = h * head_dim;
                    var i: usize = 0;
                    while (i < rotary_dim) : (i += 2) {
                        const freq = 1.0 / std.math.pow(f32, rope_base, @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(rotary_dim)));
                        const angle = @as(f32, @floatFromInt(pos)) * freq;
                        const cos_val = @cos(angle);
                        const sin_val = @sin(angle);
                        const k0 = self.k_proj[offset + i];
                        const k1 = self.k_proj[offset + i + 1];
                        self.k_proj[offset + i] = k0 * cos_val - k1 * sin_val;
                        self.k_proj[offset + i + 1] = k0 * sin_val + k1 * cos_val;
                    }
                }
            } else {
                self.applyRoPE();
            }
        }

        // Append K/V to cache BEFORE attention so current token attends to itself
        if (self.kv_cache) |*cache| cache.append(layer_idx, self.k_proj.ptr, self.v_proj.ptr);

        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

        @memset(self.attn_accum, 0);

        const cache_len = if (self.kv_cache) |cache| cache.seqLen() else 0;
        if (cache_len > 0 and self.kv_cache != null) {
            const cache_layer = self.kv_cache.?.layer(layer_idx);
            for (0..num_heads) |head_index| {
                const h: u32 = @intCast(head_index);
                const kv_head = h / (num_heads / num_kv_heads);
                const q_offset = if (is_qwen35) @as(usize, h) * head_dim * 2 else @as(usize, h) * head_dim;
                @memcpy(self.head_q[0..head_dim], self.q_proj[q_offset..][0..head_dim]);

                if (self.config.use_kv_quantization) {
                    for (0..cache_len) |pos_index| {
                        const pos: u32 = @intCast(pos_index);
                        var score: f32 = 0;
                        const key_head = cache_layer.keys_int8.? + @as(usize, pos) * kv_dim + @as(usize, kv_head) * head_dim;
                        const k_scale = cache_layer.key_scales.?[@as(usize, pos) * num_kv_heads + kv_head];
                        for (0..head_dim) |d| score += self.head_q[d] * @as(f32, @floatFromInt(key_head[d])) * k_scale;
                        self.scores[pos] = score * scale;
                    }
                } else {
                    for (0..cache_len) |pos_index| {
                        const pos: u32 = @intCast(pos_index);
                        var score: f32 = 0;
                        const key_head = cache_layer.keys_fp32.? + @as(usize, pos) * kv_dim + @as(usize, kv_head) * head_dim;
                        for (0..head_dim) |d| score += self.head_q[d] * key_head[d];
                        self.scores[pos] = score * scale;
                    }
                }

                avx2_softmax.softmaxAvx2(self.scores[0..cache_len]);
                @memset(self.head_out[0..head_dim], 0);

                if (self.config.use_kv_quantization) {
                    for (0..cache_len) |pos_index| {
                        const pos: u32 = @intCast(pos_index);
                        const value_head = cache_layer.values_int8.? + @as(usize, pos) * kv_dim + @as(usize, kv_head) * head_dim;
                        const v_scale = cache_layer.val_scales.?[@as(usize, pos) * num_kv_heads + kv_head];
                        for (0..head_dim) |d| {
                            self.head_out[d] += self.scores[pos] * (@as(f32, @floatFromInt(value_head[d])) * v_scale);
                        }
                    }
                } else {
                    for (0..cache_len) |pos_index| {
                        const pos: u32 = @intCast(pos_index);
                        const value_head = cache_layer.values_fp32.? + @as(usize, pos) * kv_dim + @as(usize, kv_head) * head_dim;
                        for (0..head_dim) |d| self.head_out[d] += self.scores[pos] * value_head[d];
                    }
                }

                // Apply sigmoid gating if Qwen35
                if (is_qwen35) {
                    const gate_offset = @as(usize, h) * head_dim * 2 + head_dim;
                    const gate = self.q_proj[gate_offset..][0..head_dim];
                    for (0..head_dim) |d| {
                        const g = gate[d];
                        const sig = 1.0 / (1.0 + @exp(-g));
                        self.head_out[d] *= sig;
                    }
                }

                @memcpy(self.attn_accum[@as(usize, h) * head_dim ..][0..head_dim], self.head_out[0..head_dim]);
            }
        }

        @memset(self.attn_out, 0);
        parallel.parallelMatvec(self.weight_pool[o_proj_offset..].ptr, self.attn_accum.ptr, self.attn_out.ptr, hidden, q_dim, dtypeForTensor(self.weight_dtypes, layer_idx, 4));
        @memcpy(output, self.attn_out);
    }

    /// SSM layer forward pass using Mamba-2 selective scan
    fn ssmLayerForward(self: *Engine, layer_idx: u32, input: []const f32, output: []f32) void {
        const hidden = self.config.hidden_dim;
        const ssm_inner = self.config.ssm_inner_size;
        const ssm_groups = self.config.ssm_num_groups;
        const dt_rank = self.config.ssm_dt_rank;
        const conv_kernel = self.config.ssm_conv_kernel;
        const dt_scale = self.config.ssm_dt_scale;

        const ssm_base = packedSsmBase(&self.weight_layout, layer_idx);

        // Ensure SSM state is initialized for this layer
        if (self.ssm_states[layer_idx].conv_state.len == 0) {
            self.ssm_states[layer_idx] = ssm.SSMState.init(hidden, conv_kernel, ssm_inner, ssm_groups, self.allocator) catch unreachable;
        }

        // Get weight offsets
        const ssm_out_offset = ssm_base + self.weight_layout.ssm_out_offset;
        const ssm_x_offset = ssm_base + self.weight_layout.ssm_x_offset;
        const ssm_dt_offset = ssm_base + self.weight_layout.ssm_dt_offset;
        const ssm_A_offset = ssm_base + self.weight_layout.ssm_A_offset;
        const ssm_B_offset = ssm_base + self.weight_layout.ssm_B_offset;
        const ssm_C_offset = ssm_base + self.weight_layout.ssm_C_offset;
        const ssm_D_offset = ssm_base + self.weight_layout.ssm_D_offset;
        const ssm_conv1d_offset = ssm_base + self.weight_layout.ssm_conv1d_offset;
        const ssm_conv1d_bias_offset = ssm_base + self.weight_layout.ssm_conv1d_bias_offset;

        // Call the SSM kernel
        ssm.ssmForward(
            hidden,
            ssm_inner,
            ssm_groups,
            dt_rank,
            conv_kernel,
            dt_scale,
            input,
            output,
            self.weight_pool.ptr + ssm_out_offset,
            self.weight_pool.ptr + ssm_x_offset,
            self.weight_pool.ptr + ssm_dt_offset,
            self.weight_pool.ptr + ssm_A_offset,
            self.weight_pool.ptr + ssm_B_offset,
            self.weight_pool.ptr + ssm_C_offset,
            self.weight_pool.ptr + ssm_D_offset,
            self.weight_pool.ptr + ssm_conv1d_offset,
            self.weight_pool.ptr + ssm_conv1d_bias_offset,
            dtypeForLayerSlot(self.weight_dtypes, layer_idx, 11), // ssm_out
            dtypeForLayerSlot(self.weight_dtypes, layer_idx, 12), // ssm_x
            dtypeForLayerSlot(self.weight_dtypes, layer_idx, 13), // ssm_dt
            dtypeForLayerSlot(self.weight_dtypes, layer_idx, 15), // ssm_B
            dtypeForLayerSlot(self.weight_dtypes, layer_idx, 16), // ssm_C
            &self.ssm_states[layer_idx],
            self.ssm_tmp_x,
            self.ssm_tmp_z,
            self.ssm_tmp_dt,
            self.ssm_tmp_B,
            self.ssm_tmp_C,
        );
    }

    fn qwenLinearLayerForward(self: *Engine, layer_idx: u32, input: []const f32, output: []f32) void {
        const hidden = self.config.hidden_dim;
        const layer_base = self.weight_layout.layer_offsets[layer_idx];
        const qkv_offset = layer_base + self.weight_layout.q_proj_offset;
        const qkv_end = layer_base + self.weight_layout.o_proj_offset;
        const z_offset = layer_base + self.weight_layout.o_proj_offset;
        const z_end = layer_base + self.weight_layout.ffn_norm_offset;

        const qwen_base = packedSsmBase(&self.weight_layout, layer_idx);
        const out_offset = qwen_base + self.weight_layout.ssm_out_offset;
        const out_end = qwen_base + self.weight_layout.ssm_x_offset;
        const alpha_offset = qwen_base + self.weight_layout.ssm_dt_offset;
        const alpha_end = qwen_base + self.weight_layout.ssm_A_offset;
        const beta_offset = qwen_base + self.weight_layout.ssm_B_offset;
        const beta_end = qwen_base + self.weight_layout.ssm_C_offset;
        const a_offset = qwen_base + self.weight_layout.ssm_A_offset;
        const a_end = qwen_base + self.weight_layout.ssm_B_offset;
        const dt_bias_offset = qwen_base + self.weight_layout.ssm_conv1d_bias_offset;
        const dt_bias_end = qwen_base + self.weight_layout.ssm_per_layer_size;
        const conv_offset = qwen_base + self.weight_layout.ssm_conv1d_offset;
        const conv_end = qwen_base + self.weight_layout.ssm_conv1d_bias_offset;
        const norm_offset = qwen_base + self.weight_layout.ssm_D_offset;
        const norm_end = qwen_base + self.weight_layout.ssm_conv1d_offset;

        const value_dim = self.config.ssm_inner_size;
        const num_v_heads = self.config.ssm_dt_rank;
        const head_v_dim = if (num_v_heads > 0) value_dim / num_v_heads else 0;
        const key_dim = value_dim;
        const num_k_heads = if (head_v_dim > 0) key_dim / head_v_dim else 0;

        if (self.qwen_states[layer_idx].conv_state.len == 0) {
            self.qwen_states[layer_idx] = qwen_linear.QwenLinearState.init(
                key_dim * 2 + value_dim,
                self.config.ssm_conv_kernel,
                num_v_heads,
                if (num_k_heads > 0) key_dim / num_k_heads else 0,
                head_v_dim,
                self.config.hidden_dim,
                self.allocator,
            ) catch unreachable;
        }

        qwen_linear.forward(
            self.config,
            input,
            output,
            &self.qwen_states[layer_idx],
            .{
                .qkv = .{ .bytes = self.weight_pool[qkv_offset..qkv_end], .dtype = dtypeForTensor(self.weight_dtypes, layer_idx, 1) },
                .z = .{ .bytes = self.weight_pool[z_offset..z_end], .dtype = dtypeForTensor(self.weight_dtypes, layer_idx, 4) },
                .alpha = .{ .bytes = self.weight_pool[alpha_offset..alpha_end], .dtype = dtypeForLayerSlot(self.weight_dtypes, layer_idx, 12) },
                .beta = .{ .bytes = self.weight_pool[beta_offset..beta_end], .dtype = dtypeForLayerSlot(self.weight_dtypes, layer_idx, 13) },
                .a_log = .{ .bytes = self.weight_pool[a_offset..a_end], .dtype = dtypeForLayerSlot(self.weight_dtypes, layer_idx, 14) },
                .dt_bias = .{ .bytes = self.weight_pool[dt_bias_offset..dt_bias_end], .dtype = dtypeForLayerSlot(self.weight_dtypes, layer_idx, 15) },
                .conv1d = .{ .bytes = self.weight_pool[conv_offset..conv_end], .dtype = dtypeForLayerSlot(self.weight_dtypes, layer_idx, 16) },
                .norm = .{ .bytes = self.weight_pool[norm_offset..norm_end], .dtype = dtypeForLayerSlot(self.weight_dtypes, layer_idx, 17) },
                .out = .{ .bytes = self.weight_pool[out_offset..out_end], .dtype = dtypeForLayerSlot(self.weight_dtypes, layer_idx, 11) },
            },
            .{
                .mixed_qkv = self.qwen_tmp_mixed_qkv,
                .conv_out = self.qwen_tmp_conv_out,
                .z = self.qwen_tmp_z,
                .core = self.qwen_tmp_core,
                .alpha = self.qwen_tmp_alpha,
                .beta = self.qwen_tmp_beta,
            },
            layer_idx,
        );

        _ = hidden;
    }

    fn ffn(self: *Engine, layer_idx: u32, input: []const f32, output: []f32) void {
        const hidden = self.config.hidden_dim;
        const ffn_h = self.config.ffn_hidden_dim;
        const w1_offset = self.current_layer_base + self.weight_layout.ffn_weight1_offset;
        const w2_offset = self.current_layer_base + self.weight_layout.ffn_weight2_offset;

        switch (self.config.ffn_type) {
            .dense => {
                @memset(self.ffn_scratch[0..ffn_h], 0);
                parallel.parallelMatvec(self.weight_pool[w1_offset..].ptr, input.ptr, self.ffn_scratch.ptr, ffn_h, hidden, dtypeForTensor(self.weight_dtypes, layer_idx, 6));
                for (self.ffn_scratch[0..ffn_h]) |*value| value.* = math.relu(value.*);
                @memset(output[0..hidden], 0);
                parallel.parallelMatvec(self.weight_pool[w2_offset..].ptr, self.ffn_scratch.ptr, output.ptr, hidden, ffn_h, dtypeForTensor(self.weight_dtypes, layer_idx, 7));
            },
            .gated_swi_glu, .gated_gelu => {
                const w3_offset = self.current_layer_base + self.weight_layout.ffn_weight3_offset;
                @memset(self.ffn_gate_buf[0..ffn_h], 0);
                @memset(self.ffn_up_buf[0..ffn_h], 0);
                parallel.parallelMatvec(self.weight_pool[w1_offset..].ptr, input.ptr, self.ffn_gate_buf.ptr, ffn_h, hidden, dtypeForTensor(self.weight_dtypes, layer_idx, 6));
                parallel.parallelMatvec(self.weight_pool[w2_offset..].ptr, input.ptr, self.ffn_up_buf.ptr, ffn_h, hidden, dtypeForTensor(self.weight_dtypes, layer_idx, 7));

                for (0..ffn_h) |i| {
                    self.ffn_gate_buf[i] = switch (self.config.ffn_type) {
                        .gated_swi_glu => math.swish(self.ffn_gate_buf[i]) * self.ffn_up_buf[i],
                        .gated_gelu => math.gelu(self.ffn_gate_buf[i]) * self.ffn_up_buf[i],
                        else => unreachable,
                    };
                }

                @memset(output[0..hidden], 0);
                parallel.parallelMatvec(self.weight_pool[w3_offset..].ptr, self.ffn_gate_buf.ptr, output.ptr, hidden, ffn_h, dtypeForTensor(self.weight_dtypes, layer_idx, 8));
            },
        }
    }

    fn norm(self: *const Engine, input: []const f32, output: []f32, norm_weight: []const types.fp16_t) void {
        const hidden = self.config.hidden_dim;
        switch (self.config.norm_type) {
            .rms_norm => avx2_norm.rmsNormAvx2(input.ptr, output.ptr, norm_weight.ptr, hidden),
            .layer_norm => {
                var mean: f32 = 0;
                for (0..hidden) |i| mean += input[i];
                mean /= @as(f32, @floatFromInt(hidden));

                var variance: f32 = 0;
                for (0..hidden) |i| {
                    const diff = input[i] - mean;
                    variance += diff * diff;
                }
                variance /= @as(f32, @floatFromInt(hidden));
                const inv_std = 1.0 / @sqrt(variance + 1e-6);

                for (0..hidden) |i| output[i] = (input[i] - mean) * inv_std * types.fp16_to_fp32(norm_weight[i]);
            },
        }
    }

    pub fn generate(self: *Engine, tokenizer_: *tokenizer.SimpleTokenizer, prompt: []const u8, max_tokens: u32) ![]u8 {
        var ids = try tokenizer_.encode(prompt);
        defer ids.deinit();

        if (ids.items.len == 0) return error.EmptyInput;

        var all_ids = ArrayList(u32).init(self.allocator);
        defer all_ids.deinit();
        try all_ids.appendSlice(ids.items);

        // Prefill: process all prompt tokens through the KV cache.
        // After this loop, self.logits holds the prediction for the first generated token.
        _ = try self.prefill(ids.items);

        // Decode: sample from prefill logits, then forward each new token.
        var generated: u32 = 0;
        while (generated < max_tokens) : (generated += 1) {
            const next_id = self.sampleGreedy(self.logits);
            if (next_id == tokenizer_.eos()) break;
            try all_ids.append(next_id);
            if (self.kv_cache) |cache| {
                if (cache.seqLen() >= self.config.max_seq_len) break;
            }
            _ = try self.forwardToken(next_id);
        }

        return tokenizer_.decode(all_ids.items);
    }

    fn applyRoPETo(self: *Engine, q: []f32, k: []f32, pos: u32) void {
        const head_dim = self.config.head_dim;
        const num_heads = self.config.num_heads;
        const num_kv_heads = self.config.num_kv_heads;
        const rope_base: f32 = 10000000.0;
        const rotary_dim: usize = 64;
        for (0..num_heads) |h| {
            const offset = h * head_dim;
            var i: usize = 0;
            while (i < rotary_dim) : (i += 2) {
                const freq = 1.0 / std.math.pow(f32, rope_base, @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(rotary_dim)));
                const angle = @as(f32, @floatFromInt(pos)) * freq;
                const cos_val = @cos(angle);
                const sin_val = @sin(angle);
                const q0 = q[offset + i];
                const q1 = q[offset + i + 1];
                q[offset + i] = q0 * cos_val - q1 * sin_val;
                q[offset + i + 1] = q0 * sin_val + q1 * cos_val;
            }
        }
        for (0..num_kv_heads) |h| {
            const offset = h * head_dim;
            var i: usize = 0;
            while (i < rotary_dim) : (i += 2) {
                const freq = 1.0 / std.math.pow(f32, rope_base, @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(rotary_dim)));
                const angle = @as(f32, @floatFromInt(pos)) * freq;
                const cos_val = @cos(angle);
                const sin_val = @sin(angle);
                const k0 = k[offset + i];
                const k1 = k[offset + i + 1];
                k[offset + i] = k0 * cos_val - k1 * sin_val;
                k[offset + i + 1] = k0 * sin_val + k1 * cos_val;
            }
        }
    }

    fn ffnBatched(self: *Engine, layer_idx: u32, X: []f32, out_buf: []f32, T: u32, gate_buf: []f32, up_buf: []f32) void {
        const hidden = self.config.hidden_dim;
        const ffn_h = self.config.ffn_hidden_dim;
        const w1_offset = self.current_layer_base + self.weight_layout.ffn_weight1_offset;
        const w2_offset = self.current_layer_base + self.weight_layout.ffn_weight2_offset;
        switch (self.config.ffn_type) {
            .dense => {
                parallel.parallelMatmat(self.weight_pool[w1_offset..].ptr, X.ptr, gate_buf.ptr, ffn_h, hidden, T, dtypeForTensor(self.weight_dtypes, layer_idx, 6));
                for (gate_buf[0 .. @as(usize, T) * ffn_h]) |*value| value.* = math.relu(value.*);
                parallel.parallelMatmat(self.weight_pool[w2_offset..].ptr, gate_buf.ptr, out_buf.ptr, hidden, ffn_h, T, dtypeForTensor(self.weight_dtypes, layer_idx, 7));
            },
            .gated_swi_glu, .gated_gelu => {
                const w3_offset = self.current_layer_base + self.weight_layout.ffn_weight3_offset;
                parallel.parallelMatmat(self.weight_pool[w1_offset..].ptr, X.ptr, gate_buf.ptr, ffn_h, hidden, T, dtypeForTensor(self.weight_dtypes, layer_idx, 6));
                parallel.parallelMatmat(self.weight_pool[w2_offset..].ptr, X.ptr, up_buf.ptr, ffn_h, hidden, T, dtypeForTensor(self.weight_dtypes, layer_idx, 7));
                for (0 .. @as(usize, T) * ffn_h) |i| {
                    gate_buf[i] = switch (self.config.ffn_type) {
                        .gated_swi_glu => math.swish(gate_buf[i]) * up_buf[i],
                        .gated_gelu => math.gelu(gate_buf[i]) * up_buf[i],
                        else => unreachable,
                    };
                }
                parallel.parallelMatmat(self.weight_pool[w3_offset..].ptr, gate_buf.ptr, out_buf.ptr, hidden, ffn_h, T, dtypeForTensor(self.weight_dtypes, layer_idx, 8));
            },
        }
    }

    const BatchScratch = struct {
        r: []f32,
        mixed: []f32,
        zb: []f32,
        coreb: []f32,
        alphab: []f32,
        betab: []f32,
        qb: []f32,
        kb: []f32,
        vb: []f32,
        ab: []f32,
        ob: []f32,
        gateb: []f32,
        upb: []f32,
    };
    fn layerForwardBatched(self: *Engine, layer_idx: u32, H: []f32, T: u32, sc: *BatchScratch) void {
        const hidden = self.config.hidden_dim;
        const layer_type = self.weight_layout.layer_types[layer_idx];
        const Ts: usize = @intCast(T);

        if (layer_type == .ssm) {
            for (0..Ts) |t| {
                const row = H[t * hidden ..][0..hidden];
                self.layerForward(layer_idx, row, row);
            }
            return;
        }

        const layer_base = self.weight_layout.layer_offsets[layer_idx];
        self.current_layer_base = layer_base;
        const layer_norm_bytes = self.weight_pool[layer_base + self.weight_layout.norm_weight_offset ..][0 .. hidden * @sizeOf(types.fp16_t)];
        const layer_norm = fp16SliceFromBytes(layer_norm_bytes);
        const ffn_norm_bytes = self.weight_pool[layer_base + self.weight_layout.ffn_norm_offset ..][0 .. hidden * @sizeOf(types.fp16_t)];
        const ffn_norm = fp16SliceFromBytes(ffn_norm_bytes);

        @memcpy(sc.r, H);
        for (0..Ts) |t| {
            const row = H[t * hidden ..][0..hidden];
            self.norm(row, row, layer_norm);
        }

        if (layer_type == .qwen_linear) {
            const qkv_offset = layer_base + self.weight_layout.q_proj_offset;
            const qkv_end = layer_base + self.weight_layout.o_proj_offset;
            const z_offset = layer_base + self.weight_layout.o_proj_offset;
            const z_end = layer_base + self.weight_layout.ffn_norm_offset;
            const qwen_base = packedSsmBase(&self.weight_layout, layer_idx);
            const out_offset = qwen_base + self.weight_layout.ssm_out_offset;
            const out_end = qwen_base + self.weight_layout.ssm_x_offset;
            const alpha_offset = qwen_base + self.weight_layout.ssm_dt_offset;
            const alpha_end = qwen_base + self.weight_layout.ssm_A_offset;
            const beta_offset = qwen_base + self.weight_layout.ssm_B_offset;
            const beta_end = qwen_base + self.weight_layout.ssm_C_offset;
            const a_offset = qwen_base + self.weight_layout.ssm_A_offset;
            const a_end = qwen_base + self.weight_layout.ssm_B_offset;
            const dt_bias_offset = qwen_base + self.weight_layout.ssm_conv1d_bias_offset;
            const dt_bias_end = qwen_base + self.weight_layout.ssm_per_layer_size;
            const conv_offset = qwen_base + self.weight_layout.ssm_conv1d_offset;
            const conv_end = qwen_base + self.weight_layout.ssm_conv1d_bias_offset;
            const norm_offset = qwen_base + self.weight_layout.ssm_D_offset;
            const norm_end = qwen_base + self.weight_layout.ssm_conv1d_offset;

            const value_dim = self.config.ssm_inner_size;
            const num_v_heads = self.config.ssm_dt_rank;
            const head_v_dim = if (num_v_heads > 0) value_dim / num_v_heads else 0;
            const conv_dim_b = value_dim * 3;

            if (self.qwen_states[layer_idx].conv_state.len == 0) {
                self.qwen_states[layer_idx] = qwen_linear.QwenLinearState.init(
                    conv_dim_b,
                    self.config.ssm_conv_kernel,
                    num_v_heads,
                    head_v_dim,
                    head_v_dim,
                    self.config.hidden_dim,
                    self.allocator,
                ) catch unreachable;
            }

            const weights = qwen_linear.Weights{
                .qkv = .{ .bytes = self.weight_pool[qkv_offset..qkv_end], .dtype = dtypeForTensor(self.weight_dtypes, layer_idx, 1) },
                .z = .{ .bytes = self.weight_pool[z_offset..z_end], .dtype = dtypeForTensor(self.weight_dtypes, layer_idx, 4) },
                .alpha = .{ .bytes = self.weight_pool[alpha_offset..alpha_end], .dtype = dtypeForLayerSlot(self.weight_dtypes, layer_idx, 12) },
                .beta = .{ .bytes = self.weight_pool[beta_offset..beta_end], .dtype = dtypeForLayerSlot(self.weight_dtypes, layer_idx, 13) },
                .a_log = .{ .bytes = self.weight_pool[a_offset..a_end], .dtype = dtypeForLayerSlot(self.weight_dtypes, layer_idx, 14) },
                .dt_bias = .{ .bytes = self.weight_pool[dt_bias_offset..dt_bias_end], .dtype = dtypeForLayerSlot(self.weight_dtypes, layer_idx, 15) },
                .conv1d = .{ .bytes = self.weight_pool[conv_offset..conv_end], .dtype = dtypeForLayerSlot(self.weight_dtypes, layer_idx, 16) },
                .norm = .{ .bytes = self.weight_pool[norm_offset..norm_end], .dtype = dtypeForLayerSlot(self.weight_dtypes, layer_idx, 17) },
                .out = .{ .bytes = self.weight_pool[out_offset..out_end], .dtype = dtypeForLayerSlot(self.weight_dtypes, layer_idx, 11) },
            };

            parallel.parallelMatmat(weights.qkv.bytes.ptr, H.ptr, sc.mixed.ptr, conv_dim_b, hidden, T, weights.qkv.dtype);
            parallel.parallelMatmat(weights.z.bytes.ptr, H.ptr, sc.zb.ptr, value_dim, hidden, T, weights.z.dtype);
            {
                const n_weights = hidden * num_v_heads;
                const alpha_fp16 = @as([*]types.fp16_t, @ptrCast(@alignCast(self.qwen_states[layer_idx].dequant_scratch.ptr)))[0..n_weights];
                quantK.dequantizeRow(weights.alpha.bytes.ptr, alpha_fp16.ptr, n_weights, weights.alpha.dtype);
                parallel.parallelMatmat(@ptrCast(alpha_fp16.ptr), H.ptr, sc.alphab.ptr, num_v_heads, hidden, T, .f16);
                const beta_fp16 = self.qwen_states[layer_idx].dequant_scratch[n_weights..];
                quantK.dequantizeRow(weights.beta.bytes.ptr, beta_fp16.ptr, n_weights, weights.beta.dtype);
                parallel.parallelMatmat(@ptrCast(beta_fp16.ptr), H.ptr, sc.betab.ptr, num_v_heads, hidden, T, .f16);
            }
            for (0..Ts) |t| {
                qwen_linear.forwardCore(
                    self.config,
                    &self.qwen_states[layer_idx],
                    weights,
                    sc.mixed[t * conv_dim_b ..][0..conv_dim_b],
                    sc.zb[t * value_dim ..][0..value_dim],
                    sc.alphab[t * num_v_heads ..][0..num_v_heads],
                    sc.betab[t * num_v_heads ..][0..num_v_heads],
                    sc.coreb[t * value_dim ..][0..value_dim],
                );
            }
            parallel.parallelMatmat(weights.out.bytes.ptr, sc.coreb.ptr, sc.ob.ptr, hidden, value_dim, T, weights.out.dtype);
        } else {
            const num_heads = self.config.num_heads;
            const num_kv_heads = self.config.num_kv_heads;
            const head_dim = self.config.head_dim;
            const q_dim = num_heads * head_dim;
            const is_qwen35 = self.config.full_attention_interval > 0;
            const q_proj_dim = if (is_qwen35) q_dim * 2 else q_dim;
            const kv_dim = num_kv_heads * head_dim;
            const q_proj_offset = layer_base + self.weight_layout.q_proj_offset;
            const k_proj_offset = layer_base + self.weight_layout.k_proj_offset;
            const v_proj_offset = layer_base + self.weight_layout.v_proj_offset;
            const o_proj_offset = layer_base + self.weight_layout.o_proj_offset;

            parallel.parallelMatmat(self.weight_pool[q_proj_offset..].ptr, H.ptr, sc.qb.ptr, q_proj_dim, hidden, T, dtypeForTensor(self.weight_dtypes, layer_idx, 1));
            parallel.parallelMatmat(self.weight_pool[k_proj_offset..].ptr, H.ptr, sc.kb.ptr, kv_dim, hidden, T, dtypeForTensor(self.weight_dtypes, layer_idx, 2));
            parallel.parallelMatmat(self.weight_pool[v_proj_offset..].ptr, H.ptr, sc.vb.ptr, kv_dim, hidden, T, dtypeForTensor(self.weight_dtypes, layer_idx, 3));

            const q_norm_bytes = self.weight_pool[layer_base + self.weight_layout.attn_q_norm_offset ..][0 .. head_dim * @sizeOf(types.fp16_t)];
            const k_norm_bytes = self.weight_pool[layer_base + self.weight_layout.attn_k_norm_offset ..][0 .. head_dim * @sizeOf(types.fp16_t)];
            const q_norm_w = fp16SliceFromBytes(q_norm_bytes);
            const k_norm_w = fp16SliceFromBytes(k_norm_bytes);
            const do_q_norm = !std.mem.allEqual(types.fp16_t, q_norm_w, 0);
            const do_k_norm = !std.mem.allEqual(types.fp16_t, k_norm_w, 0);
            const base_pos = self.seq_pos;
            const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

            for (0..Ts) |t| {
                const qrow = sc.qb[t * q_proj_dim ..][0..q_proj_dim];
                const krow = sc.kb[t * kv_dim ..][0..kv_dim];
                const vrow = sc.vb[t * kv_dim ..][0..kv_dim];
                if (do_q_norm) {
                    if (is_qwen35) {
                        for (0..num_heads) |h| {
                            const offset = h * head_dim * 2;
                            avx2_norm.rmsNormAvx2(qrow.ptr + offset, qrow.ptr + offset, q_norm_w.ptr, head_dim);
                        }
                    } else {
                        applyHeadNorm(qrow, num_heads, head_dim, q_norm_w);
                    }
                }
                if (do_k_norm) applyHeadNorm(krow, num_kv_heads, head_dim, k_norm_w);
                if (self.config.pos_encoding == .rope) {
                    if (is_qwen35) {
                        const pos = base_pos + @as(u32, @intCast(t));
                        const rope_base: f32 = 10000000.0;
                        const rotary_dim: usize = 64;
                        for (0..num_heads) |h| {
                            const offset = h * head_dim * 2;
                            var i: usize = 0;
                            while (i < rotary_dim) : (i += 2) {
                                const freq = 1.0 / std.math.pow(f32, rope_base, @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(rotary_dim)));
                                const angle = @as(f32, @floatFromInt(pos)) * freq;
                                const cos_val = @cos(angle);
                                const sin_val = @sin(angle);
                                const q0 = qrow[offset + i];
                                const q1 = qrow[offset + i + 1];
                                qrow[offset + i] = q0 * cos_val - q1 * sin_val;
                                qrow[offset + i + 1] = q0 * sin_val + q1 * cos_val;
                            }
                        }
                        for (0..num_kv_heads) |h| {
                            const offset = h * head_dim;
                            var i: usize = 0;
                            while (i < rotary_dim) : (i += 2) {
                                const freq = 1.0 / std.math.pow(f32, rope_base, @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(rotary_dim)));
                                const angle = @as(f32, @floatFromInt(pos)) * freq;
                                const cos_val = @cos(angle);
                                const sin_val = @sin(angle);
                                const k0 = krow[offset + i];
                                const k1 = krow[offset + i + 1];
                                krow[offset + i] = k0 * cos_val - k1 * sin_val;
                                krow[offset + i + 1] = k0 * sin_val + k1 * cos_val;
                            }
                        }
                    } else {
                        self.applyRoPETo(qrow, krow, base_pos + @as(u32, @intCast(t)));
                    }
                }
                if (self.kv_cache) |*cache| cache.append(layer_idx, krow.ptr, vrow.ptr);
                const ab_row = sc.ab[t * q_dim ..][0..q_dim];
                @memset(ab_row, 0);
                if (self.kv_cache) |*cache| {
                    const cache_layer = cache.layer(layer_idx);
                    const cache_len = cache_layer.seq_len;
                    for (0..num_heads) |head_index| {
                        const h: u32 = @intCast(head_index);
                        const kv_head = h / (num_heads / num_kv_heads);
                        const q_offset = if (is_qwen35) @as(usize, h) * head_dim * 2 else @as(usize, h) * head_dim;
                        const head_q = qrow[q_offset..][0..head_dim];
                        if (self.config.use_kv_quantization) {
                            for (0..cache_len) |pos_index| {
                                const pos: u32 = @intCast(pos_index);
                                var score: f32 = 0;
                                const key_head = cache_layer.keys_int8.? + @as(usize, pos) * kv_dim + @as(usize, kv_head) * head_dim;
                                const k_scale = cache_layer.key_scales.?[@as(usize, pos) * num_kv_heads + kv_head];
                                for (0..head_dim) |d| score += head_q[d] * @as(f32, @floatFromInt(key_head[d])) * k_scale;
                                self.scores[pos] = score * scale;
                            }
                        } else {
                            for (0..cache_len) |pos_index| {
                                const pos: u32 = @intCast(pos_index);
                                var score: f32 = 0;
                                const key_head = cache_layer.keys_fp32.? + @as(usize, pos) * kv_dim + @as(usize, kv_head) * head_dim;
                                for (0..head_dim) |d| score += head_q[d] * key_head[d];
                                self.scores[pos] = score * scale;
                            }
                        }
                        avx2_softmax.softmaxAvx2(self.scores[0..cache_len]);
                        const head_out = ab_row[@as(usize, h) * head_dim ..][0..head_dim];
                        if (self.config.use_kv_quantization) {
                            for (0..cache_len) |pos_index| {
                                const pos: u32 = @intCast(pos_index);
                                const value_head = cache_layer.values_int8.? + @as(usize, pos) * kv_dim + @as(usize, kv_head) * head_dim;
                                const v_scale = cache_layer.val_scales.?[@as(usize, pos) * num_kv_heads + kv_head];
                                for (0..head_dim) |d| {
                                    head_out[d] += self.scores[pos] * (@as(f32, @floatFromInt(value_head[d])) * v_scale);
                                }
                            }
                        } else {
                            for (0..cache_len) |pos_index| {
                                const pos: u32 = @intCast(pos_index);
                                const value_head = cache_layer.values_fp32.? + @as(usize, pos) * kv_dim + @as(usize, kv_head) * head_dim;
                                for (0..head_dim) |d| head_out[d] += self.scores[pos] * value_head[d];
                            }
                        }
                        if (is_qwen35) {
                            const gate_offset = @as(usize, h) * head_dim * 2 + head_dim;
                            const gate = qrow[gate_offset..][0..head_dim];
                            for (0..head_dim) |d| {
                                const g = gate[d];
                                const sig = 1.0 / (1.0 + @exp(-g));
                                head_out[d] *= sig;
                            }
                        }
                    }
                }
            }
            parallel.parallelMatmat(self.weight_pool[o_proj_offset..].ptr, sc.ab.ptr, sc.ob.ptr, hidden, q_dim, T, dtypeForTensor(self.weight_dtypes, layer_idx, 4));
        }

        for (0 .. Ts * hidden) |i| H[i] = sc.r[i] + sc.ob[i];
        @memcpy(sc.r, H);
        for (0..Ts) |t| {
            const row = H[t * hidden ..][0..hidden];
            self.norm(row, row, ffn_norm);
        }
        self.ffnBatched(layer_idx, H, sc.ob, T, sc.gateb, sc.upb);
        for (0 .. Ts * hidden) |i| H[i] = sc.r[i] + sc.ob[i];
    }
    pub fn prefillBatched(self: *Engine, token_ids: []const u32) ![]f32 {
        try self.ensureKvCache();
        const T: u32 = @intCast(token_ids.len);
        if (T == 0) return self.logits;
        if (T == 1) {
            return self.forwardToken(token_ids[0]);
        }
        const hidden = self.config.hidden_dim;
        const Ts: usize = @intCast(T);
        const alloc = self.allocator;
        const value_dim = @max(self.config.ssm_inner_size, 1);
        const conv_dim_b = value_dim * 3;
        const num_v_heads = @max(self.config.ssm_dt_rank, 1);
        const is_qwen35 = self.config.full_attention_interval > 0;
        const q_dim = @max(self.config.num_heads * self.config.head_dim, 1);
        const q_proj_dim = if (is_qwen35) q_dim * 2 else q_dim;
        const kv_dim = @max(self.config.num_kv_heads * self.config.head_dim, 1);
        const ffn_h = @max(self.config.ffn_hidden_dim, 1);

        const H = try alloc.alloc(f32, Ts * hidden);
        defer alloc.free(H);
        var sc: BatchScratch = undefined;
        sc.r = try alloc.alloc(f32, Ts * hidden);
        defer alloc.free(sc.r);
        sc.mixed = try alloc.alloc(f32, Ts * conv_dim_b);
        defer alloc.free(sc.mixed);
        sc.zb = try alloc.alloc(f32, Ts * value_dim);
        defer alloc.free(sc.zb);
        sc.coreb = try alloc.alloc(f32, Ts * value_dim);
        defer alloc.free(sc.coreb);
        sc.alphab = try alloc.alloc(f32, Ts * num_v_heads);
        defer alloc.free(sc.alphab);
        sc.betab = try alloc.alloc(f32, Ts * num_v_heads);
        defer alloc.free(sc.betab);
        sc.qb = try alloc.alloc(f32, Ts * q_proj_dim);
        defer alloc.free(sc.qb);
        sc.kb = try alloc.alloc(f32, Ts * kv_dim);
        defer alloc.free(sc.kb);
        sc.vb = try alloc.alloc(f32, Ts * kv_dim);
        defer alloc.free(sc.vb);
        sc.ab = try alloc.alloc(f32, Ts * q_dim);
        defer alloc.free(sc.ab);
        sc.ob = try alloc.alloc(f32, Ts * hidden);
        defer alloc.free(sc.ob);
        sc.gateb = try alloc.alloc(f32, Ts * ffn_h);
        defer alloc.free(sc.gateb);
        sc.upb = try alloc.alloc(f32, Ts * ffn_h);
        defer alloc.free(sc.upb);

        const embedding_dtype = dtypeForGlobal(self.weight_dtypes, 0);
        const embedding_row_bytes = rowByteSize(hidden, embedding_dtype);
        for (token_ids, 0..) |tok_id, t| {
            const emb_bytes = self.weight_pool[self.weight_layout.token_embedding_offset + @as(usize, tok_id) * embedding_row_bytes ..][0..embedding_row_bytes];
            loadEmbeddingRowToF32(emb_bytes, embedding_dtype, hidden, H[t * hidden ..][0..hidden]);
        }

        for (0..self.config.num_layers) |layer_idx| {
            self.layerForwardBatched(@intCast(layer_idx), H, T, &sc);
        }

        @memcpy(self.hidden_state, H[(Ts - 1) * hidden ..][0..hidden]);
        const final_norm_bytes = self.weight_pool[self.weight_layout.final_norm_offset..][0 .. hidden * @sizeOf(types.fp16_t)];
        const final_norm = fp16SliceFromBytes(final_norm_bytes);
        self.norm(self.hidden_state, self.hidden_state, final_norm);

        @memset(self.logits, 0);
        const output_proj_byte_offset = self.weight_layout.output_proj_offset;
        const output_proj_size = self.weight_layout.ssm_region_offset - self.weight_layout.output_proj_offset;
        const output_proj_bytes = self.weight_pool[output_proj_byte_offset..][0..output_proj_size];
        const output_dtype = dtypeForGlobal(self.weight_dtypes, 2);
        parallel.parallelMatvec(output_proj_bytes.ptr, self.hidden_state.ptr, self.logits.ptr, self.config.vocab_size, hidden, output_dtype);

        self.seq_pos += T;
        return self.logits;
    }
    /// Prefill prompt tokens into KV cache, return logits of last token.
    pub fn prefill(self: *Engine, token_ids: []const u32) ![]f32 {
        var logits: []f32 = self.logits;
        for (token_ids) |tok_id| {
            logits = try self.forwardToken(tok_id);
        }
        return logits;
    }

    /// Decode one step: process token_id, return logits.
    pub fn decodeStep(self: *Engine, token_id: u32) ![]f32 {
        return self.forwardToken(token_id);
    }

    pub fn greedyNextToken(self: *const Engine) u32 {
        return self.sampleGreedy(self.logits);
    }

    fn sampleGreedy(self: *const Engine, logits: []const f32) u32 {
        _ = self;
        var max_idx: u32 = 0;
        for (1..logits.len) |i| {
            if (logits[i] > logits[max_idx]) max_idx = @intCast(i);
        }
        return max_idx;
    }

    /// Legacy alias
    fn sampleTopK(self: *const Engine, logits: []const f32, k: u32) u32 {
        _ = k;
        return self.sampleGreedy(logits);
    }
};

fn makeTinyConfig(num_layers: u32, max_seq_len: u32) config.ModelConfig {
    return .{
        .vocab_size = 4,
        .hidden_dim = 4,
        .num_heads = 1,
        .num_kv_heads = 1,
        .head_dim = 4,
        .num_layers = num_layers,
        .num_ssm_layers = 0,
        .ffn_hidden_dim = 4,
        .max_seq_len = max_seq_len,
        .ffn_type = .dense,
        .norm_type = .rms_norm,
        .pos_encoding = .rope,
        .use_kv_quantization = false,
        .full_attention_interval = 0,
        .ssm_conv_kernel = 4,
        .ssm_inner_size = 4,
        .ssm_num_groups = 1,
        .ssm_dt_rank = 1,
        .ssm_dt_scale = 1.0,
    };
}
