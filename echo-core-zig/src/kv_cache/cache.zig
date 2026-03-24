const std = @import("std");
const types = @import("../core/types.zig");
const config = @import("../core/config.zig");

pub const KVCacheLayer = extern struct {
    keys_int8: [*]i8,
    values_int8: [*]i8,
    key_scales: [*]f32,
    val_scales: [*]f32,
    keys_fp32: [*]f32,
    values_fp32: [*]f32,
    seq_len: u32,
};

pub const KVCache = struct {
    config: config.ModelConfig,
    layers: []KVCacheLayer,
    storage: [*]u8,

    pub fn init(cfg: config.ModelConfig, allocator: std.mem.Allocator) !KVCache {
        const num_kv_heads = cfg.num_kv_heads;
        const head_dim = cfg.head_dim;
        const max_seq_len = cfg.max_seq_len;
        const num_layers = cfg.num_layers;
        const kv_dim = num_kv_heads * head_dim;

        const keys_int8_size = max_seq_len * kv_dim * @sizeOf(i8);
        const values_int8_size = max_seq_len * kv_dim * @sizeOf(i8);
        const key_scales_size = max_seq_len * num_kv_heads * @sizeOf(f32);
        const val_scales_size = max_seq_len * num_kv_heads * @sizeOf(f32);
        const keys_fp32_size = max_seq_len * kv_dim * @sizeOf(f32);
        const values_fp32_size = max_seq_len * kv_dim * @sizeOf(f32);

        const total_size = (keys_int8_size + values_int8_size + key_scales_size + val_scales_size + keys_fp32_size + values_fp32_size) * num_layers;

        const storage = try allocator.alloc(u8, total_size);
        errdefer allocator.free(storage);

        const layers = try allocator.alloc(KVCacheLayer, num_layers);

        var offset: usize = 0;
        for (0..num_layers) |i| {
            layers[i] = .{
                .keys_int8 = @ptrCast(storage[offset..].ptr),
                .values_int8 = @ptrCast((storage[offset..].ptr + keys_int8_size)),
                .key_scales = @ptrCast((storage[offset..].ptr + keys_int8_size + values_int8_size)),
                .val_scales = @ptrCast((storage[offset..].ptr + keys_int8_size + values_int8_size + key_scales_size)),
                .keys_fp32 = @ptrCast((storage[offset..].ptr + keys_int8_size + values_int8_size + key_scales_size + val_scales_size)),
                .values_fp32 = @ptrCast((storage[offset..].ptr + keys_int8_size + values_int8_size + key_scales_size + val_scales_size + keys_fp32_size)),
                .seq_len = 0,
            };
            offset += keys_int8_size + values_int8_size + key_scales_size + val_scales_size + keys_fp32_size + values_fp32_size;
        }

        return .{
            .config = cfg,
            .layers = layers,
            .storage = storage.ptr,
        };
    }

    pub fn deinit(self: *KVCache, allocator: std.mem.Allocator) void {
        allocator.free(self.storage[0 .. self.config.num_layers * self.config.max_seq_len * self.config.num_kv_heads * self.config.head_dim * (@sizeOf(i8) * 2 + @sizeOf(f32) * 4)]);
        allocator.free(self.layers);
    }

    pub fn layer(self: *KVCache, layer_idx: u32) *KVCacheLayer {
        return &self.layers[layer_idx];
    }

    pub fn layerConst(self: *const KVCache, layer_idx: u32) *const KVCacheLayer {
        return &self.layers[layer_idx];
    }

    pub fn append(self: *KVCache, layer_idx: u32, k_proj: [*]const f32, v_proj: [*]const f32) void {
        const cache_layer = &self.layers[layer_idx];
        const seq_pos = cache_layer.seq_len;
        const kv_dim = self.config.num_kv_heads * self.config.head_dim;

        if (self.config.use_kv_quantization) {
            quant.quantizePerTokenSymmetric(k_proj, cache_layer.keys_int8 + seq_pos * kv_dim, cache_layer.key_scales + seq_pos * self.config.num_kv_heads, 1, kv_dim);
            quant.quantizePerTokenSymmetric(v_proj, cache_layer.values_int8 + seq_pos * kv_dim, cache_layer.val_scales + seq_pos * self.config.num_kv_heads, 1, kv_dim);
        } else {
            @memcpy(cache_layer.keys_fp32[seq_pos * kv_dim ..][0..kv_dim], k_proj[0..kv_dim]);
            @memcpy(cache_layer.values_fp32[seq_pos * kv_dim ..][0..kv_dim], v_proj[0..kv_dim]);
        }

        cache_layer.seq_len += 1;
    }

    pub fn seqLen(self: *const KVCache) u32 {
        if (self.layers.len == 0) return 0;
        return self.layers[0].seq_len;
    }

    pub fn reset(self: *KVCache) void {
        for (self.layers) |*l| {
            l.seq_len = 0;
        }
    }
};

const quant = @import("../kernels/quant.zig");

test "KVCache init" {
    const cfg = config.ModelConfig{
        .vocab_size = 32000,
        .hidden_dim = 4096,
        .num_heads = 32,
        .num_kv_heads = 32,
        .head_dim = 128,
        .num_layers = 2,
        .ffn_hidden_dim = 11008,
        .max_seq_len = 128,
        .ffn_type = .gated_swi_glu,
        .norm_type = .rms_norm,
        .pos_encoding = .rope,
        .use_kv_quantization = false,
    };
    var cache = try KVCache.init(cfg, std.testing.allocator);
    defer cache.deinit(std.testing.allocator);
    try std.testing.expectEqual(cache.seqLen(), 0);
}
