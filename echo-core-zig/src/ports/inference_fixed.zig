const std = @import("std");
const types = @import("../core/types.zig");
const config = @import("../core/config.zig");
const memory = @import("../core/memory.zig");
const engine = @import("../inference/engine.zig");
const matvec = @import("../kernels/matvec.zig");
const quant = @import("../kernels/quant.zig");
const gguf = @import("../gguf/reader.zig");
const tokenizer = @import("../tokenizer/tokenizer.zig");
const kv_cache = @import("../kv_cache/cache.zig");

const ArrayList = std.array_list.Managed;

// Phase 4: Helper functions for computing tensor byte sizes
fn computeTensorByteSize(dtype: gguf.GGMLType, shape: []const u64) u64 {
    var n_elements: u64 = 1;
    for (shape) |dim| n_elements *= dim;

    return switch (dtype) {
        .f32 => n_elements * 4,
        .f16, .bf16 => n_elements * 2,
        .f64 => n_elements * 8,
        .i8 => n_elements,
        .i16 => n_elements * 2,
        .i32 => n_elements * 4,
        else => blk: {
            const block_bytes = blockSizeBytes(dtype);
            const block_elems = blockElements(dtype);
            if (block_bytes == 0 or block_elems == 0) unreachable;
            const n_blocks = (n_elements + block_elems - 1) / block_elems;
            break :blk n_blocks * block_bytes;
        },
    };
}

fn blockSizeBytes(dtype: gguf.GGMLType) u64 {
    return switch (dtype) {
        .f32 => 4,
        .f16, .bf16 => 2,
        .f64 => 8,
        .i8 => 1,
        .i16 => 2,
        .i32 => 4,
        .q4_0 => 18,
        .q4_1 => 20,
        .q5_0 => 22,
        .q5_1 => 24,
        .q8_0 => 34,
        .q8_1 => 36,
        .q2_k => 84,
        .q3_k => 110,
        .q4_k => 144,
        .q5_k => 176,
        .q6_k => 210,
        .q8_k => 34,
        .iq2_xxs => 66,
        .iq2_xs => 74,
        .iq3_xxs => 102,
        .iq1_s => 42,
        .iq4_nl => 144,
        .iq3_s => 109,
        .iq2_s => 74,
        .iq4_xs => 136,
        .i64 => 8,
        .iq1_m => 58,
        .tq1_0 => 34,
        .tq2_0 => 66,
        .mxfp4 => 32,
        .count => 0,
    };
}

fn blockElements(dtype: gguf.GGMLType) u64 {
    return switch (dtype) {
        .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q8_1 => 32,
        .q2_k, .q3_k, .q4_k, .q5_k, .q6_k => 256,
        .q8_k => 32,
        .iq2_xxs, .iq2_xs, .iq2_s, .iq3_xxs, .iq1_s, .iq4_nl, .iq4_xs, .iq3_s, .iq1_m => 256,
        .tq1_0, .tq2_0, .mxfp4 => 32,
        else => 1,
    };
}

// Per-layer tensor offsets - each layer may have different quantization types
const LayerTensorOffsets = struct {
    base_offset: usize, // Layer start offset in weight_pool
    norm: usize, // attn_norm offset (relative to base)
    q_proj: usize, // q_proj offset
    k_proj: usize, // k_proj offset
    v_proj: usize, // v_proj offset
    o_proj: usize, // o_proj offset
    ffn_norm: usize, // ffn_norm offset
    ffn_w1: usize, // ffn_weight1 offset
    ffn_w2: usize, // ffn_weight2 offset
    ffn_w3: usize, // ffn_weight3 offset
    q_norm: usize, // attn_q_norm offset
    k_norm: usize, // attn_k_norm offset
    total_size: usize, // Total layer size
};

pub const InferencePort = struct {
    config: config.ModelConfig,
    engine_: engine.Engine,

    pub fn init(model_path: []const u8, allocator: std.mem.Allocator) !InferencePort {
        const loader = try ModelLoader.init(model_path, allocator);
        return .{
            .config = loader.config,
            .engine_ = loader.engine_,
        };
    }

    pub fn deinit(self: *InferencePort, allocator: std.mem.Allocator) void {
        self.engine_.deinit(allocator);
    }

    pub fn generate(self: *InferencePort, tokenizer_: *tokenizer.SimpleTokenizer, prompt: []const u8, max_tokens: u32) ![]u8 {
        return self.engine_.generate(tokenizer_, prompt, max_tokens);
    }

    pub fn forward(self: *InferencePort, input_ids: []const u32) ![]f32 {
        return self.engine_.forward(input_ids);
    }

    pub fn getConfig(self: *const InferencePort) config.ModelConfig {
        return self.config;
    }
};

const ModelLoader = struct {
    config: config.ModelConfig,
    engine_: engine.Engine,

    pub fn init(model_path: []const u8, allocator: std.mem.Allocator) !ModelLoader {
        return try ModelLoader.load(model_path, allocator);
    }

    pub fn load(model_path: []const u8, allocator: std.mem.Allocator) !ModelLoader {
        var reader = try gguf.Reader.openWithAllocator(model_path, allocator);
        defer reader.deinit();

        // Check compatibility
        if (try buildCompatibilityReport(allocator, &reader)) |report| {
            std.debug.print("{s}", .{report});
            allocator.free(report);
            return error.ModelIncompatible;
        }

        // Get mutable copy of config so we can update it with actual dimensions
        var cfg = reader.config;

        // Detect actual tensor dimensions before computing weight layout
        detectActualDimensions(&cfg, &reader);

        var eng = try engine.Engine.init(cfg, &reader, allocator);
        errdefer eng.deinit(allocator);
        @memset(eng.weight_pool, types.fp32_to_fp16(0));

        try loadWeightsFromReader(&eng, &reader, allocator);

        return .{
            .config = cfg,
            .engine_ = eng,
        };
    }

    pub fn deinit(self: *ModelLoader, allocator: std.mem.Allocator) void {
        self.engine_.deinit(allocator);
    }
};

// Detect actual tensor dimensions that may differ from GGUF metadata
fn detectActualDimensions(cfg: *config.ModelConfig, reader: *const gguf.Reader) void {
    // Check attn_q.weight to detect actual hidden_dim
    if (reader.tensors.get("blk.0.attn_q.weight")) |tensor_info| {
        if (tensor_info.shape.len >= 2) {
            const actual_hidden = tensor_info.shape[0];
            if (actual_hidden > cfg.hidden_dim) {
                std.debug.print("INFO: Detected actual hidden_dim={d} from attn_q.weight, metadata reported {d}\n", .{ actual_hidden, cfg.hidden_dim });
                cfg.hidden_dim = actual_hidden;
                // Recalculate dependent values
                cfg.head_dim = cfg.hidden_dim / cfg.num_heads;
                cfg.num_kv_heads = @max(1, cfg.num_kv_heads * cfg.head_dim / cfg.head_dim);
            }
        }
    }

    // Check ssm_conv1d.weight for SSM models
    if (reader.tensors.get("blk.0.ssm_conv1d.weight")) |tensor_info| {
        if (tensor_info.shape.len >= 2) {
            const actual_ssm_hidden = tensor_info.shape[1];
            if (actual_ssm_hidden > cfg.hidden_dim) {
                std.debug.print("INFO: Detected actual SSM hidden_dim={d} from ssm_conv1d.weight, metadata reported {d}\n", .{ actual_ssm_hidden, cfg.hidden_dim });
                cfg.hidden_dim = actual_ssm_hidden;
            }
        }
    }

    // Check ssm_dt.weight for dt_rank
    if (reader.tensors.get("blk.0.ssm_dt.weight")) |tensor_info| {
        if (tensor_info.shape.len >= 2) {
            const actual_dt_rank = tensor_info.shape[1];
            if (actual_dt_rank != cfg.ssm_dt_rank) {
                cfg.ssm_dt_rank = actual_dt_rank;
            }
        }
    }

    // Check ssm_A.weight for ssm_inner_size
    if (reader.tensors.get("blk.0.ssm_A.weight")) |tensor_info| {
        if (tensor_info.shape.len >= 1) {
            const actual_ssm_inner = tensor_info.shape[0];
            if (actual_ssm_inner != cfg.ssm_inner_size) {
                cfg.ssm_inner_size = actual_ssm_inner;
            }
        }
    }

    // Count SSM layers for hybrid models
    var ssm_layer_count: u32 = 0;
    var buf: [64]u8 = undefined;
    for (0..cfg.num_layers) |layer_idx| {
        const tensor_name = std.fmt.bufPrint(&buf, "blk.{d}.ssm_out.weight", .{layer_idx}) catch continue;
        if (reader.tensors.get(tensor_name)) |_| {
            ssm_layer_count += 1;
        }
    }
    cfg.num_ssm_layers = ssm_layer_count;
    if (ssm_layer_count > 0) {
        std.debug.print("INFO: Detected hybrid model with {d} SSM layers out of {d} total layers\n", .{ ssm_layer_count, cfg.num_layers });
    }
}

// Find tensor name by suffix (handles both with and without architecture prefix)
fn findTensorNameBySuffix(reader: *const gguf.Reader, suffix: []const u8) ?[]const u8 {
    var it = reader.tensors.iterator();
    while (it.next()) |entry| {
        if (std.mem.endsWith(u8, entry.key_ptr.*, suffix)) {
            return entry.key_ptr.*;
        }
    }
    return null;
}

// Build tensor suffix for a specific layer
fn tensorSuffixForLayer(buf: []u8, layer_idx: usize, suffix: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "blk.{d}.{s}", .{ layer_idx, suffix });
}

// Compatibility checking structures
const CompatibilityTensor = struct {
    name: []const u8,
    dtype: gguf.GGMLType,
};

fn isSupportedTensorType(dtype: gguf.GGMLType) bool {
    return switch (dtype) {
        .f32, .f16, .bf16, .f64, .i8, .i16, .i32, .i64 => true,
        .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q8_1 => true,
        .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .q8_k => true,
        .iq2_xxs, .iq2_xs, .iq3_xxs, .iq1_s, .iq4_nl, .iq3_s, .iq2_s, .iq4_xs, .iq1_m => true,
        .tq1_0, .tq2_0, .mxfp4 => true,
        else => false,
    };
}

fn appendCompatibilityIssue(report: *ArrayList(u8), issue: []const u8) !void {
    try report.appendSlice(issue);
    try report.appendSlice("\n");
}

fn buildCompatibilityReportFromSummary(allocator: std.mem.Allocator, architecture: []const u8, is_hybrid: bool, tensors: []const CompatibilityTensor) !?[]u8 {
    var report = ArrayList(u8).init(allocator);
    errdefer report.deinit();

    for (tensors) |tensor| {
        if (!isSupportedTensorType(tensor.dtype)) {
            var buf: [192]u8 = undefined;
            const msg = try std.fmt.bufPrint(&buf, "unsupported tensor dtype {s} at {s}", .{ @tagName(tensor.dtype), tensor.name });
            try appendCompatibilityIssue(&report, msg);
            break;
        }
    }

    if (report.items.len == 0) return null;
    try report.appendSlice("Engine yuklemeyi durdurdu; once model uyumlulugu cozulmeli.\n");
    return try report.toOwnedSlice();
}

fn metadataString(reader: *const gguf.Reader, key: []const u8) []const u8 {
    if (reader.metadata.get(key)) |value| {
        return switch (value) {
            .string => |text| text,
            else => "",
        };
    }
    return "";
}

fn buildCompatibilityReport(allocator: std.mem.Allocator, reader: *const gguf.Reader) !?[]u8 {
    var it = reader.tensors.iterator();
    var count: usize = 0;
    while (it.next()) |_| count += 1;
    if (count == 0) return null;

    const tensors = try allocator.alloc(CompatibilityTensor, count);
    defer allocator.free(tensors);

    var idx: usize = 0;
    var tensor_it = reader.tensors.iterator();
    while (tensor_it.next()) |entry| : (idx += 1) {
        tensors[idx] = .{
            .name = entry.key_ptr.*,
            .dtype = entry.value_ptr.dtype,
        };
    }

    const architecture = metadataString(reader, "general.architecture");
    var key_buf: [96]u8 = undefined;
    const hybrid_key = if (architecture.len > 0)
        try std.fmt.bufPrint(&key_buf, "{s}.full_attention_interval", .{architecture})
    else
        "";

    return try buildCompatibilityReportFromSummary(
        allocator,
        architecture,
        hybrid_key.len > 0 and reader.metadata.get(hybrid_key) != null,
        tensors,
    );
}

// Load tensor if present - stores raw bytes without dequantization
fn loadTensorIfPresent(
    reader: *const gguf.Reader,
    suffix: []const u8,
    dst_bytes: []u8,
    dtype_slot_index: usize,
    weight_dtypes: []gguf.GGMLType,
) !bool {
    if (findTensorNameBySuffix(reader, suffix)) |tensor_name| {
        const info = reader.findTensorBySuffix(suffix).?;

        // Store the dtype for this tensor slot
        weight_dtypes[dtype_slot_index] = info.dtype;

        // Compute actual byte size from tensor shape and dtype
        const actual_bytes = computeTensorByteSize(info.dtype, info.shape);

        // Verify destination can hold the data
        if (actual_bytes > dst_bytes.len) {
            std.debug.print("WARN: tensor '{s}' needs {d} bytes but only {d} allocated\n", .{ tensor_name, actual_bytes, dst_bytes.len });
        }

        // Load raw tensor data directly - NO DEQUANTIZE
        const bytes_to_load = @min(actual_bytes, dst_bytes.len);
        try reader.loadTensorInto(tensor_name, dst_bytes[0..bytes_to_load]);

        return true;
    }
    return false;
}

fn loadTensorWithAliasIfPresent(
    reader: *const gguf.Reader,
    primary_suffix: []const u8,
    alias_suffix: []const u8,
    dst_bytes: []u8,
    dtype_slot_index: usize,
    weight_dtypes: []gguf.GGMLType,
) !bool {
    if (findTensorNameBySuffix(reader, primary_suffix)) |_| {
        return try loadTensorIfPresent(reader, primary_suffix, dst_bytes, dtype_slot_index, weight_dtypes);
    } else if (findTensorNameBySuffix(reader, alias_suffix)) |_| {
        return try loadTensorIfPresent(reader, alias_suffix, dst_bytes, dtype_slot_index, weight_dtypes);
    }
    return false;
}

// Calculate per-layer tensor offsets dynamically
fn calculateLayerTensorOffsets(
    reader: *const gguf.Reader,
    cfg: config.ModelConfig,
    allocator: std.mem.Allocator,
) ![]LayerTensorOffsets {
    const layer_offsets = try allocator.alloc(LayerTensorOffsets, cfg.num_layers);
    errdefer allocator.free(layer_offsets);

    var base_offset: usize = 0; // Will be set to token_embedding_size later

    for (0..cfg.num_layers) |layer_idx| {
        var buf: [64]u8 = undefined;
        const hidden = cfg.hidden_dim;
        const kv_dim = cfg.num_kv_heads * cfg.head_dim;
        const q_dim = cfg.num_heads * cfg.head_dim;
        const ffn_h = cfg.ffn_hidden_dim;

        var offsets = LayerTensorOffsets{
            .base_offset = base_offset,
            .norm = 0,
            .q_proj = 0,
            .k_proj = 0,
            .v_proj = 0,
            .o_proj = 0,
            .ffn_norm = 0,
            .ffn_w1 = 0,
            .ffn_w2 = 0,
            .ffn_w3 = 0,
            .q_norm = 0,
            .k_norm = 0,
            .total_size = 0,
        };

        var current_offset: usize = 0;

        // attn_norm
        const attn_norm_name = std.fmt.bufPrint(&buf, "blk.{d}.attn_norm.weight", .{layer_idx}) catch "";
        if (reader.tensors.get(attn_norm_name)) |info| {
            offsets.norm = current_offset;
            current_offset += computeTensorByteSize(info.dtype, info.shape);
        } else {
            offsets.norm = current_offset;
            current_offset += hidden * @sizeOf(types.fp16_t);
        }

        // q_proj
        const q_name = std.fmt.bufPrint(&buf, "blk.{d}.attn_q.weight", .{layer_idx}) catch "";
        if (reader.tensors.get(q_name)) |info| {
            offsets.q_proj = current_offset;
            current_offset += computeTensorByteSize(info.dtype, info.shape);
        } else {
            offsets.q_proj = current_offset;
            current_offset += hidden * q_dim * @sizeOf(types.fp16_t);
        }

        // k_proj
        const k_name = std.fmt.bufPrint(&buf, "blk.{d}.attn_k.weight", .{layer_idx}) catch "";
        if (reader.tensors.get(k_name)) |info| {
            offsets.k_proj = current_offset;
            current_offset += computeTensorByteSize(info.dtype, info.shape);
        } else {
            offsets.k_proj = current_offset;
            current_offset += hidden * kv_dim * @sizeOf(types.fp16_t);
        }

        // v_proj
        const v_name = std.fmt.bufPrint(&buf, "blk.{d}.attn_v.weight", .{layer_idx}) catch "";
        if (reader.tensors.get(v_name)) |info| {
            offsets.v_proj = current_offset;
            current_offset += computeTensorByteSize(info.dtype, info.shape);
        } else {
            offsets.v_proj = current_offset;
            current_offset += hidden * kv_dim * @sizeOf(types.fp16_t);
        }

        // o_proj
        const o_name = std.fmt.bufPrint(&buf, "blk.{d}.attn_output.weight", .{layer_idx}) catch "";
        if (reader.tensors.get(o_name)) |info| {
            offsets.o_proj = current_offset;
            current_offset += computeTensorByteSize(info.dtype, info.shape);
        } else {
            offsets.o_proj = current_offset;
            current_offset += q_dim * hidden * @sizeOf(types.fp16_t);
        }

        // ffn_norm
        const ffn_norm_name = std.fmt.bufPrint(&buf, "blk.{d}.ffn_norm.weight", .{layer_idx}) catch "";
        if (reader.tensors.get(ffn_norm_name)) |info| {
            offsets.ffn_norm = current_offset;
            current_offset += computeTensorByteSize(info.dtype, info.shape);
        } else {
            offsets.ffn_norm = current_offset;
            current_offset += hidden * @sizeOf(types.fp16_t);
        }

        // FFN weights based on type
        switch (cfg.ffn_type) {
            .dense => {
                const ffn_up_name = std.fmt.bufPrint(&buf, "blk.{d}.ffn_up.weight", .{layer_idx}) catch "";
                if (reader.tensors.get(ffn_up_name)) |info| {
                    offsets.ffn_w1 = current_offset;
                    current_offset += computeTensorByteSize(info.dtype, info.shape);
                } else {
                    offsets.ffn_w1 = current_offset;
                    current_offset += hidden * ffn_h * @sizeOf(types.fp16_t);
                }
                const ffn_down_name = std.fmt.bufPrint(&buf, "blk.{d}.ffn_down.weight", .{layer_idx}) catch "";
                if (reader.tensors.get(ffn_down_name)) |info| {
                    offsets.ffn_w2 = current_offset;
                    current_offset += computeTensorByteSize(info.dtype, info.shape);
                } else {
                    offsets.ffn_w2 = current_offset;
                    current_offset += ffn_h * hidden * @sizeOf(types.fp16_t);
                }
                offsets.ffn_w3 = offsets.ffn_w2; // Not used for dense
            },
            .gated_swi_glu, .gated_gelu => {
                const gate_name = std.fmt.bufPrint(&buf, "blk.{d}.ffn_gate.weight", .{layer_idx}) catch "";
                if (reader.tensors.get(gate_name)) |info| {
                    offsets.ffn_w1 = current_offset;
                    current_offset += computeTensorByteSize(info.dtype, info.shape);
                } else {
                    offsets.ffn_w1 = current_offset;
                    current_offset += hidden * ffn_h * @sizeOf(types.fp16_t);
                }
                const ffn_up_name = std.fmt.bufPrint(&buf, "blk.{d}.ffn_up.weight", .{layer_idx}) catch "";
                if (reader.tensors.get(ffn_up_name)) |info| {
                    offsets.ffn_w2 = current_offset;
                    current_offset += computeTensorByteSize(info.dtype, info.shape);
                } else {
                    offsets.ffn_w2 = current_offset;
                    current_offset += hidden * ffn_h * @sizeOf(types.fp16_t);
                }
                const ffn_down_name = std.fmt.bufPrint(&buf, "blk.{d}.ffn_down.weight", .{layer_idx}) catch "";
                if (reader.tensors.get(ffn_down_name)) |info| {
                    offsets.ffn_w3 = current_offset;
                    current_offset += computeTensorByteSize(info.dtype, info.shape);
                } else {
                    offsets.ffn_w3 = current_offset;
                    current_offset += ffn_h * hidden * @sizeOf(types.fp16_t);
                }
            },
        }

        // q_norm and k_norm
        const q_norm_name = std.fmt.bufPrint(&buf, "blk.{d}.attn_q_norm.weight", .{layer_idx}) catch "";
        if (reader.tensors.get(q_norm_name)) |info| {
            offsets.q_norm = current_offset;
            current_offset += computeTensorByteSize(info.dtype, info.shape);
        } else {
            offsets.q_norm = current_offset;
            current_offset += cfg.head_dim * @sizeOf(types.fp16_t);
        }
        const k_norm_name = std.fmt.bufPrint(&buf, "blk.{d}.attn_k_norm.weight", .{layer_idx}) catch "";
        if (reader.tensors.get(k_norm_name)) |info| {
            offsets.k_norm = current_offset;
            current_offset += computeTensorByteSize(info.dtype, info.shape);
        } else {
            offsets.k_norm = current_offset;
            current_offset += cfg.head_dim * @sizeOf(types.fp16_t);
        }

        offsets.total_size = current_offset;
        layer_offsets[layer_idx] = offsets;

        // Update base_offset for next layer (use max to ensure alignment)
        base_offset += @max(offsets.total_size, cfg.ffn_hidden_dim * 4); // Conservative alignment
    }

    return layer_offsets;
}

fn loadWeightsFromReader(eng: *engine.Engine, reader: *const gguf.Reader, allocator: std.mem.Allocator) !void {
    const layout = eng.weight_layout;
    const cfg = eng.config;

    // Slot mapping: 0 = token_embd, then 11 slots per layer, then final_norm, output_proj
    var current_slot: usize = 0;

    // Token embedding (slot 0)
    const token_embd_info = reader.tensors.get("token_embd.weight");
    const token_embd_bytes = if (token_embd_info) |info| computeTensorByteSize(info.dtype, info.shape) else cfg.vocab_size * cfg.hidden_dim * @sizeOf(types.fp16_t);
    _ = try loadTensorIfPresent(reader, "token_embd.weight", eng.weight_pool[layout.token_embedding_offset..][0..token_embd_bytes], current_slot, eng.weight_dtypes);
    current_slot += 1;

    // Calculate per-layer tensor offsets dynamically
    const layer_offsets = try calculateLayerTensorOffsets(reader, cfg, allocator);
    defer allocator.free(layer_offsets);

    var name_buf: [96]u8 = undefined;
    for (0..cfg.num_layers) |layer_idx| {
        const offsets = layer_offsets[layer_idx];
        const layer_base = layout.token_embedding_size + offsets.base_offset;

        // Helper for layer-specific tensors
        const getLayerTensorBytes = struct {
            fn call(r: *const gguf.Reader, l: usize, suffix: []const u8, fallback_elements: u64) usize {
                var buf: [64]u8 = undefined;
                const full_name = std.fmt.bufPrint(&buf, "blk.{d}.{s}", .{ l, suffix }) catch return fallback_elements * @sizeOf(types.fp16_t);
                if (r.tensors.get(full_name)) |info| {
                    return computeTensorByteSize(info.dtype, info.shape);
                }
                return fallback_elements * @sizeOf(types.fp16_t);
            }
        }.call;

        // attn_norm (slot current_slot + 0)
        const norm_bytes = getLayerTensorBytes(reader, layer_idx, "attn_norm.weight", cfg.hidden_dim);
        _ = try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "attn_norm.weight"), eng.weight_pool[layer_base + offsets.norm ..][0..norm_bytes], current_slot, eng.weight_dtypes);
        current_slot += 1;

        // q_proj (slot current_slot + 0)
        const q_dim = cfg.num_heads * cfg.head_dim;
        const q_proj_bytes = getLayerTensorBytes(reader, layer_idx, "attn_q.weight", cfg.hidden_dim * q_dim);
        const q_loaded = try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "attn_q.weight"), eng.weight_pool[layer_base + offsets.q_proj ..][0..q_proj_bytes], current_slot, eng.weight_dtypes);
        current_slot += 1;

        // k_proj (slot current_slot + 0)
        const kv_dim = cfg.num_kv_heads * cfg.head_dim;
        const k_proj_bytes = getLayerTensorBytes(reader, layer_idx, "attn_k.weight", cfg.hidden_dim * kv_dim);
        const k_loaded = try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "attn_k.weight"), eng.weight_pool[layer_base + offsets.k_proj ..][0..k_proj_bytes], current_slot, eng.weight_dtypes);
        current_slot += 1;

        // v_proj (slot current_slot + 0)
        const v_proj_bytes = getLayerTensorBytes(reader, layer_idx, "attn_v.weight", cfg.hidden_dim * kv_dim);
        const v_loaded = try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "attn_v.weight"), eng.weight_pool[layer_base + offsets.v_proj ..][0..v_proj_bytes], current_slot, eng.weight_dtypes);
        current_slot += 1;

        // Detect layer type: check for SSM tensor presence
        const ssm_out_suffix = try tensorSuffixForLayer(&name_buf, layer_idx, "ssm_out.weight");
        if (findTensorNameBySuffix(reader, ssm_out_suffix) != null) {
            eng.weight_layout.layer_types[layer_idx] = .ssm;
        }

        // o_proj (slot current_slot + 0)
        const o_proj_bytes = getLayerTensorBytes(reader, layer_idx, "attn_output.weight", q_dim * cfg.hidden_dim);
        const attn_output_suffix = try tensorSuffixForLayer(&name_buf, layer_idx, "attn_output.weight");
        _ = try loadTensorWithAliasIfPresent(reader, attn_output_suffix, try tensorSuffixForLayer(&name_buf, layer_idx, "attn_gate.weight"), eng.weight_pool[layer_base + offsets.o_proj ..][0..o_proj_bytes], current_slot, eng.weight_dtypes);
        current_slot += 1;

        // ffn_norm (slot current_slot + 0)
        const ffn_norm_bytes = getLayerTensorBytes(reader, layer_idx, "ffn_norm.weight", cfg.hidden_dim);
        _ = try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "ffn_norm.weight"), eng.weight_pool[layer_base + offsets.ffn_norm ..][0..ffn_norm_bytes], current_slot, eng.weight_dtypes);
        current_slot += 1;

        // attn_q_norm (slot current_slot + 0)
        const q_norm_bytes = getLayerTensorBytes(reader, layer_idx, "attn_q_norm.weight", cfg.head_dim);
        _ = try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "attn_q_norm.weight"), eng.weight_pool[layer_base + offsets.q_norm ..][0..q_norm_bytes], current_slot, eng.weight_dtypes);
        current_slot += 1;

        // attn_k_norm (slot current_slot + 0)
        const k_norm_bytes = getLayerTensorBytes(reader, layer_idx, "attn_k_norm.weight", cfg.head_dim);
        _ = try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "attn_k_norm.weight"), eng.weight_pool[layer_base + offsets.k_norm ..][0..k_norm_bytes], current_slot, eng.weight_dtypes);
        current_slot += 1;

        // FFN weights (3 slots)
        switch (cfg.ffn_type) {
            .dense => {
                const ffn_up_bytes = getLayerTensorBytes(reader, layer_idx, "ffn_up.weight", cfg.hidden_dim * cfg.ffn_hidden_dim);
                const ffn_down_bytes = getLayerTensorBytes(reader, layer_idx, "ffn_down.weight", cfg.ffn_hidden_dim * cfg.hidden_dim);
                _ = try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "ffn_up.weight"), eng.weight_pool[layer_base + offsets.ffn_w1 ..][0..ffn_up_bytes], current_slot, eng.weight_dtypes);
                current_slot += 1;
                _ = try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "ffn_down.weight"), eng.weight_pool[layer_base + offsets.ffn_w2 ..][0..ffn_down_bytes], current_slot, eng.weight_dtypes);
                current_slot += 1;
                current_slot += 1; // Skip ffn_weight3 slot for dense (not used)
            },
            .gated_swi_glu, .gated_gelu => {
                const gate_bytes = getLayerTensorBytes(reader, layer_idx, "ffn_gate.weight", cfg.hidden_dim * cfg.ffn_hidden_dim);
                const up_bytes = getLayerTensorBytes(reader, layer_idx, "ffn_up.weight", cfg.hidden_dim * cfg.ffn_hidden_dim);
                const down_bytes = getLayerTensorBytes(reader, layer_idx, "ffn_down.weight", cfg.ffn_hidden_dim * cfg.hidden_dim);
                _ = try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "ffn_gate.weight"), eng.weight_pool[layer_base + offsets.ffn_w1 ..][0..gate_bytes], current_slot, eng.weight_dtypes);
                current_slot += 1;
                _ = try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "ffn_up.weight"), eng.weight_pool[layer_base + offsets.ffn_w2 ..][0..up_bytes], current_slot, eng.weight_dtypes);
                current_slot += 1;
                _ = try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "ffn_down.weight"), eng.weight_pool[layer_base + offsets.ffn_w3 ..][0..down_bytes], current_slot, eng.weight_dtypes);
                current_slot += 1;
            },
        }
    }

    // Final norm (slot after all layers)
    const final_norm_info = reader.tensors.get("output_norm.weight");
    const final_norm_bytes = if (final_norm_info) |info| computeTensorByteSize(info.dtype, info.shape) else cfg.hidden_dim * @sizeOf(types.fp16_t);
    _ = try loadTensorIfPresent(reader, "output_norm.weight", eng.weight_pool[layout.final_norm_offset..][0..final_norm_bytes], current_slot, eng.weight_dtypes);
    current_slot += 1;

    // Output projection (final slot)
    const output_info = reader.tensors.get("output.weight");
    const output_bytes = if (output_info) |info| computeTensorByteSize(info.dtype, info.shape) else cfg.hidden_dim * cfg.vocab_size * @sizeOf(types.fp16_t);
    if (reader.findTensorBySuffix("output.weight") != null) {
        _ = try loadTensorIfPresent(reader, "output.weight", eng.weight_pool[layout.output_proj_offset..][0..output_bytes], current_slot, eng.weight_dtypes);
    } else {
        // Fallback: copy from token embedding
        const copy_bytes = @min(token_embd_bytes, output_bytes);
        @memcpy(eng.weight_pool[layout.output_proj_offset..][0..copy_bytes], eng.weight_pool[layout.token_embedding_offset..][0..copy_bytes]);
        eng.weight_dtypes[current_slot] = .f16;
    }
}
