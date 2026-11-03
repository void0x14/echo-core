const std = @import("std");
const types = @import("../core/types.zig");
const config = @import("core_config");
const quant = @import("../kernels/quant.zig");

pub const KVCacheLayer = extern struct {
    keys_int8: ?[*]i8,
    values_int8: ?[*]i8,
    key_scales: ?[*]f32,
    val_scales: ?[*]f32,
    keys_fp32: ?[*]f32,
    values_fp32: ?[*]f32,
    seq_len: u32,
};

pub const KVCache = struct {
    config: config.ModelConfig,
    layers: []align(types.CACHE_LINE_SIZE) KVCacheLayer,
    storage: []align(types.CACHE_LINE_SIZE) u8,

    pub fn init(cfg: config.ModelConfig, allocator: std.mem.Allocator) !KVCache {
        const kv_dim = cfg.num_kv_heads * cfg.head_dim;
        const max_seq_len = cfg.max_seq_len;
        const num_layers = cfg.num_layers;

        const keys_int8_size = max_seq_len * kv_dim * @sizeOf(i8);
        const values_int8_size = max_seq_len * kv_dim * @sizeOf(i8);
        const scales_size = max_seq_len * cfg.num_kv_heads * @sizeOf(f32);
        const fp32_size = max_seq_len * kv_dim * @sizeOf(f32);

        const per_layer_size = if (cfg.use_kv_quantization)
            keys_int8_size + values_int8_size + scales_size * 2
        else
            fp32_size * 2;
        const total_size = per_layer_size * num_layers;

        const layers = try allocator.alignedAlloc(
            KVCacheLayer,
            std.mem.Alignment.fromByteUnits(types.CACHE_LINE_SIZE),
            num_layers,
        );
        errdefer allocator.free(layers);
        @memset(layers, std.mem.zeroes(KVCacheLayer));

        const storage = try allocator.alignedAlloc(
            u8,
            std.mem.Alignment.fromByteUnits(types.CACHE_LINE_SIZE),
            total_size,
        );
        errdefer allocator.free(storage);
        @memset(storage, 0);

        var ptr: [*]u8 = storage.ptr;
        for (layers) |*cache_layer| {
            if (cfg.use_kv_quantization) {
                cache_layer.keys_int8 = @ptrCast(ptr);
                ptr += keys_int8_size;
                cache_layer.values_int8 = @ptrCast(ptr);
                ptr += values_int8_size;
                cache_layer.key_scales = @ptrCast(@alignCast(ptr));
                ptr += scales_size;
                cache_layer.val_scales = @ptrCast(@alignCast(ptr));
                ptr += scales_size;
                cache_layer.keys_fp32 = null;
                cache_layer.values_fp32 = null;
            } else {
                cache_layer.keys_fp32 = @ptrCast(@alignCast(ptr));
                ptr += fp32_size;
                cache_layer.values_fp32 = @ptrCast(@alignCast(ptr));
                ptr += fp32_size;
                cache_layer.keys_int8 = null;
                cache_layer.values_int8 = null;
                cache_layer.key_scales = null;
                cache_layer.val_scales = null;
            }
            cache_layer.seq_len = 0;
        }

        return .{
            .config = cfg,
            .layers = layers,
            .storage = storage,
        };
    }

    pub fn deinit(self: *KVCache, allocator: std.mem.Allocator) void {
        allocator.free(self.storage);
        allocator.free(self.layers);
    }

    pub fn layer(self: *KVCache, layer_idx: u32) *KVCacheLayer {
        std.debug.assert(layer_idx < self.layers.len);
        return &self.layers[layer_idx];
    }

    pub fn layerConst(self: *const KVCache, layer_idx: u32) *const KVCacheLayer {
        std.debug.assert(layer_idx < self.layers.len);
        return &self.layers[layer_idx];
    }

    pub fn append(self: *KVCache, layer_idx: u32, k_proj: [*]const f32, v_proj: [*]const f32) void {
        const cache_layer = &self.layers[layer_idx];
        const seq_pos = cache_layer.seq_len;
        const kv_dim = self.config.num_kv_heads * self.config.head_dim;

        std.debug.assert(seq_pos < self.config.max_seq_len);

        if (self.config.use_kv_quantization) {
            quant.quantizePerTokenSymmetric(
                k_proj,
                cache_layer.keys_int8.? + seq_pos * kv_dim,
                cache_layer.key_scales.? + seq_pos * self.config.num_kv_heads,
                1,
                kv_dim,
            );
            quant.quantizePerTokenSymmetric(
                v_proj,
                cache_layer.values_int8.? + seq_pos * kv_dim,
                cache_layer.val_scales.? + seq_pos * self.config.num_kv_heads,
                1,
                kv_dim,
            );
        } else {
            @memcpy(cache_layer.keys_fp32.?[seq_pos * kv_dim ..][0..kv_dim], k_proj[0..kv_dim]);
            @memcpy(cache_layer.values_fp32.?[seq_pos * kv_dim ..][0..kv_dim], v_proj[0..kv_dim]);
        }

        cache_layer.seq_len += 1;
    }

    pub fn seqLen(self: *const KVCache) u32 {
        if (self.layers.len == 0) return 0;
        return self.layers[0].seq_len;
    }

    pub fn reset(self: *KVCache) void {
        for (self.layers) |*cache_layer| cache_layer.seq_len = 0;
    }
};
