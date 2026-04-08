const std = @import("std");
const config = @import("../core/config.zig");
const memory = @import("../core/memory.zig");
const types = @import("../core/types.zig");
const matvec = @import("../kernels/matvec.zig");
const qwen_linear = @import("../kernels/qwen_linear.zig");
const ssm = @import("../kernels/ssm.zig");
const tokenizer = @import("../tokenizer/tokenizer.zig");
const kv_cache = @import("../kv_cache/cache.zig");
const math = @import("../core/math.zig");
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
    current_layer_base: usize,
    seq_pos: u32,

    pub fn init(cfg: config.ModelConfig, reader_opt: ?*const gguf.Reader, allocator: std.mem.Allocator) !Engine {
        std.debug.print("DEBUG: Engine.init() - starting WeightLayout.compute()\n", .{});
        var layout = try memory.WeightLayout.compute(cfg, reader_opt, allocator);
        errdefer layout.deinit(allocator);
        std.debug.print("DEBUG: WeightLayout.compute() completed - raw_pool_size={d} bytes ({d:.2} MB)\n", .{ layout.raw_pool_size, @as(f64, @floatFromInt(layout.raw_pool_size)) / (1024.0 * 1024.0) });

        const kv_dim = cfg.num_kv_heads * cfg.head_dim;
        const q_dim = cfg.num_heads * cfg.head_dim;

        // Unified byte pool for all weights (quantized + fp16)
        std.debug.print("DEBUG: Allocating weight_pool ({d:.2} MB)...\n", .{@as(f64, @floatFromInt(layout.raw_pool_size)) / (1024.0 * 1024.0)});
        const weight_pool = try allocator.alloc(u8, layout.raw_pool_size);
        errdefer allocator.free(weight_pool);
        @memset(weight_pool, 0);
        std.debug.print("DEBUG: weight_pool allocated successfully\n", .{});

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
        const q_proj = try allocator.alloc(f32, q_dim);
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

        const qwen_tmp_mixed_qkv = try allocator.alloc(f32, if (cfg.ssm_inner_size > 0) cfg.ssm_inner_size * 2 else 1);
        errdefer allocator.free(qwen_tmp_mixed_qkv);
        const qwen_tmp_conv_out = try allocator.alloc(f32, if (cfg.ssm_inner_size > 0) cfg.ssm_inner_size * 2 else 1);
        errdefer allocator.free(qwen_tmp_conv_out);
        const qwen_tmp_z = try allocator.alloc(f32, if (cfg.ssm_inner_size > 0) cfg.ssm_inner_size else 1);
        errdefer allocator.free(qwen_tmp_z);
        const qwen_tmp_core = try allocator.alloc(f32, if (cfg.ssm_inner_size > 0) cfg.ssm_inner_size else 1);
        errdefer allocator.free(qwen_tmp_core);
        const qwen_tmp_alpha = try allocator.alloc(f32, if (cfg.ssm_dt_rank > 0) cfg.ssm_dt_rank else 1);
        errdefer allocator.free(qwen_tmp_alpha);
        const qwen_tmp_beta = try allocator.alloc(f32, if (cfg.ssm_dt_rank > 0) cfg.ssm_dt_rank else 1);
        errdefer allocator.free(qwen_tmp_beta);

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
        matvec.matvecDispatchQuant(config.Intel13500H_Tiles.TILE_K, config.Intel13500H_Tiles.TILE_M, output_proj_bytes.ptr, self.hidden_state.ptr, self.logits.ptr, vocab, hidden, output_dtype);

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
            // Standard attention layer path
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
        const rope_base: f32 = 10000.0;

        // Apply to Q heads
        for (0..num_heads) |h| {
            const offset = h * head_dim;
            var i: usize = 0;
            while (i < head_dim) : (i += 2) {
                const freq = 1.0 / std.math.pow(f32, rope_base, @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(head_dim)));
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
            while (i < head_dim) : (i += 2) {
                const freq = 1.0 / std.math.pow(f32, rope_base, @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(head_dim)));
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
            var rms: f32 = 0;
            for (0..head_dim) |i| rms += proj[offset + i] * proj[offset + i];
            rms = @sqrt(rms / @as(f32, @floatFromInt(head_dim)) + 1e-6);
            const inv_rms = 1.0 / rms;
            for (0..head_dim) |i| {
                proj[offset + i] = proj[offset + i] * inv_rms * types.fp16_to_fp32(norm_weight[i]);
            }
        }
    }

    fn attention(self: *Engine, layer_idx: u32, input: []const f32, output: []f32) void {
        const hidden = self.config.hidden_dim;
        const num_heads = self.config.num_heads;
        const num_kv_heads = self.config.num_kv_heads;
        const head_dim = self.config.head_dim;
        const q_dim = num_heads * head_dim;
        const kv_dim = num_kv_heads * head_dim;
        const layer_base = self.weight_layout.layer_offsets[layer_idx];

        const q_proj_offset = layer_base + self.weight_layout.q_proj_offset;
        const k_proj_offset = layer_base + self.weight_layout.k_proj_offset;
        const v_proj_offset = layer_base + self.weight_layout.v_proj_offset;
        const o_proj_offset = layer_base + self.weight_layout.o_proj_offset;

        @memset(self.q_proj, 0);
        @memset(self.k_proj, 0);
        @memset(self.v_proj, 0);
        matvec.matvecDispatchQuant(config.Intel13500H_Tiles.TILE_K, config.Intel13500H_Tiles.TILE_M, self.weight_pool[q_proj_offset..].ptr, input.ptr, self.q_proj.ptr, q_dim, hidden, dtypeForTensor(self.weight_dtypes, layer_idx, 1));
        matvec.matvecDispatchQuant(config.Intel13500H_Tiles.TILE_K, config.Intel13500H_Tiles.TILE_M, self.weight_pool[k_proj_offset..].ptr, input.ptr, self.k_proj.ptr, kv_dim, hidden, dtypeForTensor(self.weight_dtypes, layer_idx, 2));
        matvec.matvecDispatchQuant(config.Intel13500H_Tiles.TILE_K, config.Intel13500H_Tiles.TILE_M, self.weight_pool[v_proj_offset..].ptr, input.ptr, self.v_proj.ptr, kv_dim, hidden, dtypeForTensor(self.weight_dtypes, layer_idx, 3));

        // Apply Q/K head norms if weights are non-zero
        const q_norm_byte_offset = layer_base + self.weight_layout.attn_q_norm_offset;
        const k_norm_byte_offset = layer_base + self.weight_layout.attn_k_norm_offset;
        const q_norm_bytes = self.weight_pool[q_norm_byte_offset..][0 .. head_dim * @sizeOf(types.fp16_t)];
        const k_norm_bytes = self.weight_pool[k_norm_byte_offset..][0 .. head_dim * @sizeOf(types.fp16_t)];
        const q_norm_w = fp16SliceFromBytes(q_norm_bytes);
        const k_norm_w = fp16SliceFromBytes(k_norm_bytes);
        if (!std.mem.allEqual(types.fp16_t, q_norm_w, 0)) {
            applyHeadNorm(self.q_proj, num_heads, head_dim, q_norm_w);
        }
        if (!std.mem.allEqual(types.fp16_t, k_norm_w, 0)) {
            applyHeadNorm(self.k_proj[0..kv_dim], num_kv_heads, head_dim, k_norm_w);
        }

        // Apply RoPE
        if (self.config.pos_encoding == .rope) {
            self.applyRoPE();
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
                const kv_head = h % num_kv_heads;
                @memcpy(self.head_q[0..head_dim], self.q_proj[@as(usize, h) * head_dim ..][0..head_dim]);

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

                math.softmax(self.scores[0..cache_len]);
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

                @memcpy(self.attn_accum[@as(usize, h) * head_dim ..][0..head_dim], self.head_out[0..head_dim]);
            }
        }

        @memset(self.attn_out, 0);
        matvec.matvecDispatchQuant(config.Intel13500H_Tiles.TILE_K, config.Intel13500H_Tiles.TILE_M, self.weight_pool[o_proj_offset..].ptr, self.attn_accum.ptr, self.attn_out.ptr, hidden, q_dim, dtypeForTensor(self.weight_dtypes, layer_idx, 4));
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
        const qkv_end = layer_base + self.weight_layout.k_proj_offset;
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
        const key_dim = value_dim / 2;
        const num_k_heads = if (head_v_dim > 0) key_dim / head_v_dim else 0;

        if (self.qwen_states[layer_idx].conv_state.len == 0) {
            self.qwen_states[layer_idx] = qwen_linear.QwenLinearState.init(
                value_dim * 2,
                self.config.ssm_conv_kernel,
                num_v_heads,
                if (num_k_heads > 0) key_dim / num_k_heads else 0,
                head_v_dim,
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
                matvec.matvecDispatchQuant(config.Intel13500H_Tiles.TILE_K, config.Intel13500H_Tiles.TILE_M, self.weight_pool[w1_offset..].ptr, input.ptr, self.ffn_scratch.ptr, ffn_h, hidden, dtypeForTensor(self.weight_dtypes, layer_idx, 6));
                for (self.ffn_scratch[0..ffn_h]) |*value| value.* = math.relu(value.*);
                @memset(output[0..hidden], 0);
                matvec.matvecDispatchQuant(config.Intel13500H_Tiles.TILE_K, config.Intel13500H_Tiles.TILE_M, self.weight_pool[w2_offset..].ptr, self.ffn_scratch.ptr, output.ptr, hidden, ffn_h, dtypeForTensor(self.weight_dtypes, layer_idx, 7));
            },
            .gated_swi_glu, .gated_gelu => {
                const w3_offset = self.current_layer_base + self.weight_layout.ffn_weight3_offset;
                @memset(self.ffn_gate_buf[0..ffn_h], 0);
                @memset(self.ffn_up_buf[0..ffn_h], 0);
                matvec.matvecDispatchQuant(config.Intel13500H_Tiles.TILE_K, config.Intel13500H_Tiles.TILE_M, self.weight_pool[w1_offset..].ptr, input.ptr, self.ffn_gate_buf.ptr, ffn_h, hidden, dtypeForTensor(self.weight_dtypes, layer_idx, 6));
                matvec.matvecDispatchQuant(config.Intel13500H_Tiles.TILE_K, config.Intel13500H_Tiles.TILE_M, self.weight_pool[w2_offset..].ptr, input.ptr, self.ffn_up_buf.ptr, ffn_h, hidden, dtypeForTensor(self.weight_dtypes, layer_idx, 7));

                for (0..ffn_h) |i| {
                    self.ffn_gate_buf[i] = switch (self.config.ffn_type) {
                        .gated_swi_glu => math.swish(self.ffn_gate_buf[i]) * self.ffn_up_buf[i],
                        .gated_gelu => math.gelu(self.ffn_gate_buf[i]) * self.ffn_up_buf[i],
                        else => unreachable,
                    };
                }

                @memset(output[0..hidden], 0);
                matvec.matvecDispatchQuant(config.Intel13500H_Tiles.TILE_K, config.Intel13500H_Tiles.TILE_M, self.weight_pool[w3_offset..].ptr, self.ffn_gate_buf.ptr, output.ptr, hidden, ffn_h, dtypeForTensor(self.weight_dtypes, layer_idx, 8));
            },
        }
    }

    fn norm(self: *const Engine, input: []const f32, output: []f32, norm_weight: []const types.fp16_t) void {
        const hidden = self.config.hidden_dim;
        switch (self.config.norm_type) {
            .rms_norm => {
                var rms: f32 = 0;
                for (0..hidden) |i| rms += input[i] * input[i];
                rms = @sqrt(rms / @as(f32, @floatFromInt(hidden)) + 1e-6);
                const inv_rms = 1.0 / rms;
                for (0..hidden) |i| output[i] = input[i] * inv_rms * types.fp16_to_fp32(norm_weight[i]);
            },
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
        for (ids.items) |tok_id| {
            _ = try self.forwardToken(tok_id);
        }

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
        .ssm_conv_kernel = 4,
        .ssm_inner_size = 4,
        .ssm_num_groups = 1,
        .ssm_dt_rank = 1,
        .ssm_dt_scale = 1.0,
    };
}

test "Engine init" {
    var eng = try Engine.init(makeTinyConfig(1, 8), null, std.testing.allocator);
    defer eng.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 4), eng.config.hidden_dim);
    try std.testing.expectEqual(@as(usize, 4), eng.logits.len);
}

test "Engine forward returns vocab logits and uses last token embedding" {
    const cfg = makeTinyConfig(0, 0);
    var eng = try Engine.init(cfg, null, std.testing.allocator);
    defer eng.deinit(std.testing.allocator);

    // Initialize weight_pool to zeros (as bytes)
    @memset(eng.weight_pool, 0);

    // Use byte offsets directly (not element offsets)
    const embed_offset = eng.weight_layout.token_embedding_offset;
    const final_norm_offset = eng.weight_layout.final_norm_offset;
    const output_offset = eng.weight_layout.output_proj_offset;

    // Set up embedding weights (one-hot identity)
    for (0..cfg.vocab_size) |tok| {
        for (0..cfg.hidden_dim) |i| {
            const byte_idx = embed_offset + (tok * cfg.hidden_dim + i) * @sizeOf(types.fp16_t);
            const fp16_val = types.fp32_to_fp16(if (tok == i) 1 else 0);
            const ptr: *u16 = @ptrCast(@alignCast(&eng.weight_pool[byte_idx]));
            ptr.* = fp16_val;
        }
    }

    // Set up final norm and output projection
    for (0..cfg.hidden_dim) |i| {
        const norm_byte_idx = final_norm_offset + i * @sizeOf(types.fp16_t);
        const norm_ptr: *u16 = @ptrCast(@alignCast(&eng.weight_pool[norm_byte_idx]));
        norm_ptr.* = types.fp32_to_fp16(1);

        for (0..cfg.hidden_dim) |j| {
            const out_byte_idx = output_offset + (i * cfg.hidden_dim + j) * @sizeOf(types.fp16_t);
            const fp16_val = types.fp32_to_fp16(if (i == j) 1 else 0);
            const out_ptr: *u16 = @ptrCast(@alignCast(&eng.weight_pool[out_byte_idx]));
            out_ptr.* = fp16_val;
        }
    }

    const logits = try eng.forward(&.{3});
    try std.testing.expectEqual(@as(usize, cfg.vocab_size), logits.len);

    var best: usize = 0;
    for (1..logits.len) |i| {
        if (logits[i] > logits[best]) best = i;
    }
    try std.testing.expectEqual(@as(usize, 3), best);
}

test "Engine forward advances kv cache across calls" {
    const cfg = makeTinyConfig(1, 8);
    var eng = try Engine.init(cfg, null, std.testing.allocator);
    defer eng.deinit(std.testing.allocator);

    // Fill weight_pool with fp16(0.1) pattern = 0x2E66 (little endian: 0x66, 0x2E)
    const fp16_pattern = types.fp32_to_fp16(0.1);
    var i: usize = 0;
    while (i < eng.weight_pool.len) : (i += 2) {
        if (i + 1 < eng.weight_pool.len) {
            const ptr: *u16 = @ptrCast(@alignCast(&eng.weight_pool[i]));
            ptr.* = fp16_pattern;
        }
    }

    _ = try eng.forward(&.{0});
    try std.testing.expect(eng.kv_cache != null);
    try std.testing.expectEqual(@as(u32, 1), eng.kv_cache.?.seqLen());

    _ = try eng.forward(&.{ 0, 1 });
    try std.testing.expectEqual(@as(u32, 2), eng.kv_cache.?.seqLen());
}

test "Engine lazily initializes kv cache on first forward" {
    const cfg = makeTinyConfig(1, 8);
    var eng = try Engine.init(cfg, null, std.testing.allocator);
    defer eng.deinit(std.testing.allocator);

    try std.testing.expect(eng.kv_cache == null);

    // Fill weight_pool with fp16(0.1) pattern
    const fp16_pattern = types.fp32_to_fp16(0.1);
    var j: usize = 0;
    while (j < eng.weight_pool.len) : (j += 2) {
        if (j + 1 < eng.weight_pool.len) {
            const ptr2: *u16 = @ptrCast(@alignCast(&eng.weight_pool[j]));
            ptr2.* = fp16_pattern;
        }
    }
    _ = try eng.forward(&.{0});

    try std.testing.expect(eng.kv_cache != null);
    try std.testing.expectEqual(@as(u32, 1), eng.kv_cache.?.seqLen());
}

test "Engine.greedyNextToken correctly identifies token with max logit" {
    const cfg = makeTinyConfig(0, 0);
    var eng = try Engine.init(cfg, null, std.testing.allocator);
    defer eng.deinit(std.testing.allocator);

    // Populate logits with known values
    eng.logits[0] = 0.1;
    eng.logits[1] = 0.5;
    eng.logits[2] = 2.0; // max
    eng.logits[3] = 1.1;

    try std.testing.expectEqual(@as(u32, 2), eng.greedyNextToken());

    // Change max
    eng.logits[1] = 5.0; // new max
    try std.testing.expectEqual(@as(u32, 1), eng.greedyNextToken());

    // Test early in the slice
    eng.logits[0] = 10.0;
    try std.testing.expectEqual(@as(u32, 0), eng.greedyNextToken());
}

test "Engine.greedyNextToken handles all negative logits" {
    const cfg = makeTinyConfig(0, 0);
    var eng = try Engine.init(cfg, null, std.testing.allocator);
    defer eng.deinit(std.testing.allocator);

    @memset(eng.logits, -1000.0);
    eng.logits[0] = -10.0;
    eng.logits[1] = -5.0;
    eng.logits[2] = -20.0;
    eng.logits[3] = -100.0;

    // -5.0 is the largest (closest to 0)
    try std.testing.expectEqual(@as(u32, 1), eng.greedyNextToken());
}

test "Engine.decodeStep advances seq_pos and returns logits" {
    const cfg = makeTinyConfig(1, 8);
    var eng = try Engine.init(cfg, null, std.testing.allocator);
    defer eng.deinit(std.testing.allocator);

    // Initialize weight_pool to zeros
    @memset(eng.weight_pool, 0);

    try std.testing.expectEqual(@as(u32, 0), eng.seq_pos);

    const logits1 = try eng.decodeStep(0);
    try std.testing.expectEqual(@as(usize, cfg.vocab_size), logits1.len);
    try std.testing.expectEqual(@as(u32, 1), eng.seq_pos);

    const logits2 = try eng.decodeStep(1);
    try std.testing.expectEqual(@as(usize, cfg.vocab_size), logits2.len);
    try std.testing.expectEqual(@as(u32, 2), eng.seq_pos);
}

test "Engine.prefill processes multiple tokens and advances seq_pos correctly" {
    const cfg = makeTinyConfig(1, 8);
    var eng = try Engine.init(cfg, null, std.testing.allocator);
    defer eng.deinit(std.testing.allocator);

    @memset(eng.weight_pool, 0);

    try std.testing.expectEqual(@as(u32, 0), eng.seq_pos);

    const prompt = [_]u32{ 0, 1, 2 };
    const logits = try eng.prefill(&prompt);

    try std.testing.expectEqual(@as(usize, cfg.vocab_size), logits.len);
    try std.testing.expectEqual(@as(u32, 3), eng.seq_pos);

    // Check that KV cache actually registered the tokens
    try std.testing.expect(eng.kv_cache != null);
    try std.testing.expectEqual(@as(u32, 3), eng.kv_cache.?.seqLen());
}
