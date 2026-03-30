const std = @import("std");
const types = @import("../core/types.zig");
const config = @import("../core/config.zig");
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

test "KVCache init" {
    const cfg = config.ModelConfig{
        .vocab_size = 32000,
        .hidden_dim = 4096,
        .num_heads = 32,
        .num_kv_heads = 32,
        .head_dim = 128,
        .num_layers = 2,
        .num_ssm_layers = 0,
        .ffn_hidden_dim = 11008,
        .max_seq_len = 128,
        .ffn_type = .gated_swi_glu,
        .norm_type = .rms_norm,
        .pos_encoding = .rope,
        .use_kv_quantization = false,
        .ssm_conv_kernel = 4,
        .ssm_inner_size = 16,
        .ssm_num_groups = 1,
        .ssm_dt_rank = 256,
        .ssm_dt_scale = 1.0,
    };
    var cache = try KVCache.init(cfg, std.testing.allocator);
    defer cache.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 0), cache.seqLen());
}

test "KVCache mode pointers and layer alignment" {
    const fp32_cfg = config.ModelConfig{
        .vocab_size = 1000,
        .hidden_dim = 64,
        .num_heads = 8,
        .num_kv_heads = 4,
        .head_dim = 8,
        .num_layers = 2,
        .num_ssm_layers = 0,
        .ffn_hidden_dim = 128,
        .max_seq_len = 64,
        .ffn_type = .gated_swi_glu,
        .norm_type = .rms_norm,
        .pos_encoding = .rope,
        .use_kv_quantization = false,
        .ssm_conv_kernel = 4,
        .ssm_inner_size = 16,
        .ssm_num_groups = 1,
        .ssm_dt_rank = 4,
        .ssm_dt_scale = 1.0,
    };
    var fp32_cache = try KVCache.init(fp32_cfg, std.testing.allocator);
    defer fp32_cache.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 0), @intFromPtr(fp32_cache.layers.ptr) % types.CACHE_LINE_SIZE);
    try std.testing.expect(fp32_cache.layer(0).keys_fp32 != null);
    try std.testing.expect(fp32_cache.layer(0).keys_int8 == null);

    const int8_cfg = config.ModelConfig{
        .vocab_size = 1000,
        .hidden_dim = 64,
        .num_heads = 8,
        .num_kv_heads = 4,
        .head_dim = 8,
        .num_layers = 2,
        .num_ssm_layers = 0,
        .ffn_hidden_dim = 128,
        .max_seq_len = 16,
        .ffn_type = .dense,
        .norm_type = .layer_norm,
        .pos_encoding = .rope,
        .use_kv_quantization = true,
        .ssm_conv_kernel = 4,
        .ssm_inner_size = 16,
        .ssm_num_groups = 1,
        .ssm_dt_rank = 4,
        .ssm_dt_scale = 1.0,
    };
    var int8_cache = try KVCache.init(int8_cfg, std.testing.allocator);
    defer int8_cache.deinit(std.testing.allocator);
    try std.testing.expect(int8_cache.layer(0).keys_int8 != null);
    try std.testing.expect(int8_cache.layer(0).keys_fp32 == null);
}

test "KVCache append fp32 stores values and reset clears seq len" {
    const cfg = config.ModelConfig{
        .vocab_size = 1000,
        .hidden_dim = 64,
        .num_heads = 8,
        .num_kv_heads = 4,
        .head_dim = 8,
        .num_layers = 2,
        .num_ssm_layers = 0,
        .ffn_hidden_dim = 128,
        .max_seq_len = 16,
        .ffn_type = .dense,
        .norm_type = .layer_norm,
        .pos_encoding = .rope,
        .use_kv_quantization = false,
        .ssm_conv_kernel = 4,
        .ssm_inner_size = 16,
        .ssm_num_groups = 1,
        .ssm_dt_rank = 4,
        .ssm_dt_scale = 1.0,
    };
    var cache = try KVCache.init(cfg, std.testing.allocator);
    defer cache.deinit(std.testing.allocator);

    var k_proj: [32]f32 = undefined;
    var v_proj: [32]f32 = undefined;
    for (0..32) |i| {
        k_proj[i] = @floatFromInt(i);
        v_proj[i] = @floatFromInt(i + 100);
    }

    cache.append(0, &k_proj, &v_proj);
    try std.testing.expectEqual(@as(u32, 1), cache.seqLen());
    try std.testing.expectEqual(@as(f32, 0), cache.layer(0).keys_fp32.?[0]);
    try std.testing.expectEqual(@as(f32, 31), cache.layer(0).keys_fp32.?[31]);
    try std.testing.expectEqual(@as(f32, 100), cache.layer(0).values_fp32.?[0]);
    try std.testing.expectEqual(@as(f32, 131), cache.layer(0).values_fp32.?[31]);

    cache.reset();
    try std.testing.expectEqual(@as(u32, 0), cache.seqLen());
}

test "KVCache append quantized stores int8 rows" {
    const cfg = config.ModelConfig{
        .vocab_size = 1000,
        .hidden_dim = 64,
        .num_heads = 8,
        .num_kv_heads = 4,
        .head_dim = 8,
        .num_layers = 2,
        .num_ssm_layers = 0,
        .ffn_hidden_dim = 128,
        .max_seq_len = 16,
        .ffn_type = .dense,
        .norm_type = .layer_norm,
        .pos_encoding = .rope,
        .use_kv_quantization = true,
        .ssm_conv_kernel = 4,
        .ssm_inner_size = 16,
        .ssm_num_groups = 1,
        .ssm_dt_rank = 4,
        .ssm_dt_scale = 1.0,
    };
    var cache = try KVCache.init(cfg, std.testing.allocator);
    defer cache.deinit(std.testing.allocator);

    var k_proj: [32]f32 = undefined;
    var v_proj: [32]f32 = undefined;
    for (0..32) |i| {
        k_proj[i] = @floatFromInt(i + 1);
        v_proj[i] = @floatFromInt(i + 33);
    }

    cache.append(0, &k_proj, &v_proj);
    try std.testing.expectEqual(@as(u32, 1), cache.seqLen());
    try std.testing.expect(cache.layer(0).keys_int8.?[0] != 0);
    try std.testing.expect(cache.layer(0).values_int8.?[0] != 0);
    try std.testing.expect(cache.layer(0).key_scales.?[0] > 0);
    try std.testing.expect(cache.layer(0).val_scales.?[0] > 0);
}
