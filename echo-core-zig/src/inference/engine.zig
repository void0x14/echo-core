const std = @import("std");
const config = @import("../core/config.zig");
const memory = @import("../core/memory.zig");
const types = @import("../core/types.zig");
const quant = @import("../kernels/quant.zig");
const matvec = @import("../kernels/matvec.zig");
const tokenizer = @import("../tokenizer/tokenizer.zig");
const kv_cache = @import("../kv_cache/cache.zig");
const gguf = @import("../gguf/reader.zig");
const math = @import("../core/math.zig");

pub const Engine = struct {
    config: config.ModelConfig,
    weight_layout: memory.WeightLayout,
    weight_pool: [*]types.fp16_t,
    kv_cache: ?kv_cache.KVCache,
    hidden_state: []f32,
    residual: []f32,
    attn_out: []f32,

    pub fn init(cfg: config.ModelConfig, allocator: std.mem.Allocator) !Engine {
        const layout = memory.WeightLayout.compute(cfg);
        const total_weights = layout.total_size / @sizeOf(types.fp16_t);
        const weight_pool = try allocator.alloc(types.fp16_t, total_weights);

        var cache: ?kv_cache.KVCache = null;
        if (cfg.max_seq_len > 0) {
            cache = try kv_cache.KVCache.init(cfg, allocator);
        }

        const hidden_size = cfg.hidden_dim;
        const hidden_state = try allocator.alloc(f32, hidden_size);
        const residual = try allocator.alloc(f32, hidden_size);
        const attn_out = try allocator.alloc(f32, hidden_size);

        return .{
            .config = cfg,
            .weight_layout = layout,
            .weight_pool = weight_pool,
            .kv_cache = cache,
            .hidden_state = hidden_state,
            .residual = residual,
            .attn_out = attn_out,
        };
    }

    pub fn deinit(self: *Engine, allocator: std.mem.Allocator) void {
        allocator.free(self.weight_pool[0 .. self.weight_layout.total_size / @sizeOf(types.fp16_t)]);
        if (self.kv_cache) |*cache| {
            cache.deinit(allocator);
        }
        allocator.free(self.hidden_state);
        allocator.free(self.residual);
        allocator.free(self.attn_out);
    }

    pub fn forward(self: *Engine, input_ids: []const u32) ![]f32 {
        if (input_ids.len == 0) return error.EmptyInput;

        const embed_offset = self.weight_layout.token_embedding_offset / @sizeOf(types.fp16_t);
        @memcpy(self.hidden_state, self.weight_pool[embed_offset..][0..self.config.hidden_dim]);

        for (0..self.config.num_layers) |layer_idx| {
            @memcpy(self.residual, self.hidden_state);

            self.layerNorm(layer_idx);

            self.attention(layer_idx);

            for (0..self.config.hidden_dim) |i| {
                self.hidden_state[i] += self.attn_out[i];
            }

            self.ffn(layer_idx);

            for (0..self.config.hidden_dim) |i| {
                self.hidden_state[i] += self.residual[i];
            }
        }

        const final_norm_offset = self.weight_layout.final_norm_offset / @sizeOf(types.fp16_t);
        self.rmsNorm(final_norm_offset);

        return self.hidden_state[0..self.config.vocab_size];
    }

    fn layerNorm(self: *Engine, layer_idx: u32) void {
        const mean = self.computeMean(self.hidden_state);
        const variance = self.computeVar(self.hidden_state, mean);
        const std_val = @sqrt(variance + 1e-6);
        for (0..self.config.hidden_dim) |i| {
            self.hidden_state[i] = (self.hidden_state[i] - mean) / std_val;
        }
    }

    fn rmsNorm(self: *Engine, norm_offset: usize) void {
        var sum: f32 = 0;
        for (0..self.config.hidden_dim) |i| {
            const v = @as(f32, @intCast(self.weight_pool[norm_offset + i]));
            sum += v * v;
        }
        const scale = 1.0 / @sqrt(sum / @as(f32, @intCast(self.config.hidden_dim)) + 1e-6);
        for (0..self.config.hidden_dim) |i| {
            self.hidden_state[i] *= scale;
        }
    }

    fn computeMean(self: *const Engine, data: []const f32) f32 {
        var sum: f32 = 0;
        for (data) |v| sum += v;
        return sum / @as(f32, @intCast(data.len));
    }

    fn computeVar(self: *const Engine, data: []const f32, mean: f32) f32 {
        var sum: f32 = 0;
        for (data) |v| {
            const diff = v - mean;
            sum += diff * diff;
        }
        return sum / @as(f32, @intCast(data.len));
    }

    fn attention(self: *Engine, layer_idx: u32) void {
        const q_offset = (self.weight_layout.q_proj_offset + layer_idx * self.weight_layout.per_layer_size) / @sizeOf(types.fp16_t);
        const k_offset = (self.weight_layout.k_proj_offset + layer_idx * self.weight_layout.per_layer_size) / @sizeOf(types.fp16_t);
        const v_offset = (self.weight_layout.v_proj_offset + layer_idx * self.weight_layout.per_layer_size) / @sizeOf(types.fp16_t);
        const o_offset = (self.weight_layout.o_proj_offset + layer_idx * self.weight_layout.per_layer_size) / @sizeOf(types.fp16_t);

        var q = try std.testing.allocator.alloc(f32, self.config.hidden_dim);
        defer std.testing.allocator.free(q);
        var k = try std.testing.allocator.alloc(f32, self.config.num_kv_heads * self.config.head_dim);
        defer std.testing.allocator.free(k);
        var v = try std.testing.allocator.alloc(f32, self.config.num_kv_heads * self.config.head_dim);
        defer std.testing.allocator.free(v);

        if (self.kv_cache) |*cache| {
            cache.append(layer_idx, k.ptr, v.ptr);
        }

        const cache_len = if (self.kv_cache) |c| c.seqLen() else 0;
        var scores = try std.testing.allocator.alloc(f32, cache_len);
        defer std.testing.allocator.free(scores);

        if (cache_len > 0) {
            quant.fusedDequantDotInt8(q.ptr, self.kv_cache.?.layers[layer_idx].keys_int8, self.kv_cache.?.layers[layer_idx].key_scales, scores.ptr, self.config.num_kv_heads * self.config.head_dim, cache_len);
        }

        for (scores) |*s| s.* = s.* / @sqrt(@as(f32, @intCast(self.config.head_dim)));
        math.softmax(scores);

        @memset(self.attn_out, 0);
        for (0..cache_len) |i| {
            for (0..self.config.num_kv_heads * self.config.head_dim) |j| {
                self.attn_out[j] += scores[i] * self.kv_cache.?.layers[layer_idx].values_fp32[i * self.config.num_kv_heads * self.config.head_dim + j];
            }
        }

        matvec.matvecFp16Fp32(1024, 512, self.weight_pool.ptr + o_offset, self.attn_out, self.attn_out.ptr, 1, self.config.hidden_dim);
    }

    fn ffn(self: *Engine, layer_idx: u32) void {
        const ffn1_offset = (self.weight_layout.ffn_weight1_offset + layer_idx * self.weight_layout.per_layer_size) / @sizeOf(types.fp16_t);
        const ffn2_offset = (self.weight_layout.ffn_weight2_offset + layer_idx * self.weight_layout.per_layer_size) / @sizeOf(types.fp16_t);
        const ffn3_offset = (self.weight_layout.ffn_weight3_offset + layer_idx * self.weight_layout.per_layer_size) / @sizeOf(types.fp16_t);

        var gate = try std.testing.allocator.alloc(f32, self.config.ffn_hidden_dim);
        defer std.testing.allocator.free(gate);
        var up = try std.testing.allocator.alloc(f32, self.config.ffn_hidden_dim);
        defer std.testing.allocator.free(up);

        matvec.matvecFp16Fp32(1024, 512, self.weight_pool.ptr + ffn1_offset, self.hidden_state, gate.ptr, 1, self.config.hidden_dim);
        matvec.matvecFp16Fp32(1024, 512, self.weight_pool.ptr + ffn2_offset, self.hidden_state, up.ptr, 1, self.config.hidden_dim);

        for (0..self.config.ffn_hidden_dim) |i| {
            gate[i] = math.swish(gate[i]) * up[i];
        }

        matvec.matvecFp16Fp32(1024, 512, self.weight_pool.ptr + ffn3_offset, gate, self.residual.ptr, 1, self.config.ffn_hidden_dim);
    }

    pub fn generate(self: *Engine, tokenizer_: *tokenizer.SimpleTokenizer, prompt: []const u8, max_tokens: u32) ![]u8 {
        const ids = try tokenizer_.encode(prompt);
        defer ids.deinit();

        var all_ids = std.ArrayList(u32).init(std.testing.allocator);
        defer all_ids.deinit();
        try all_ids.appendSlice(ids.items);
        try all_ids.append(tokenizer_.bos());

        while (all_ids.items.len < max_tokens) {
            const logits = try self.forward(all_ids.items);
            const next_id = self.sampleTopK(logits, 1);
            if (next_id == tokenizer_.eos()) break;
            try all_ids.append(next_id);
        }

        return tokenizer_.decode(all_ids.items);
    }

    fn sampleTopK(self: *const Engine, logits: []const f32, k: u32) u32 {
        var top_k = logits[0..k];
        var max_idx: u32 = 0;
        for (0..logits.len) |i| {
            if (logits[i] > logits[max_idx]) {
                max_idx = @intCast(i);
            }
        }
        return max_idx;
    }
};

test "Engine init" {
    const cfg = config.ModelConfig{
        .vocab_size = 32000,
        .hidden_dim = 256,
        .num_heads = 4,
        .num_kv_heads = 4,
        .head_dim = 64,
        .num_layers = 2,
        .ffn_hidden_dim = 512,
        .max_seq_len = 32,
        .ffn_type = .gated_swi_glu,
        .norm_type = .rms_norm,
        .pos_encoding = .rope,
        .use_kv_quantization = false,
    };
    var engine = try Engine.init(cfg, std.testing.allocator);
    defer engine.deinit(std.testing.allocator);
    try std.testing.expectEqual(engine.config.hidden_dim, 256);
}
