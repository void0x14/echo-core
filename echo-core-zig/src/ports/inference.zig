const std = @import("std");
const builtin = @import("builtin");
const config = @import("../core/config.zig");
const types = @import("../core/types.zig");
const engine = @import("../inference/engine.zig");
const tokenizer = @import("../tokenizer/tokenizer.zig");
const gguf = @import("../gguf/reader.zig");
const quant = @import("../kernels/quant.zig");

const ArrayList = std.array_list.Managed;

const CompatibilityTensor = struct {
    name: []const u8,
    dtype: gguf.GGMLType,
};

pub const ModelLoader = struct {
    config: config.ModelConfig,
    engine_: engine.Engine,

    pub fn load(model_path: []const u8, allocator: std.mem.Allocator) !ModelLoader {
        var reader = try gguf.Reader.openWithAllocator(model_path, allocator);
        defer reader.deinit();

        if (try buildCompatibilityReport(allocator, &reader)) |report| {
            defer allocator.free(report);
            std.debug.print("{s}", .{report});
            return error.ModelIncompatible;
        }

        const cfg = reader.config;

        var eng = try engine.Engine.init(cfg, allocator);
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

pub const InferencePort = struct {
    loader: ModelLoader,
    tokenizer_: tokenizer.SimpleTokenizer,

    pub const BenchResult = struct {
        text: []u8,
        prompt_tokens: usize,
        generated_tokens: u32,
        prefill_ms: i64,
        decode_ms: i64,
    };

    pub fn init(model_path: []const u8, allocator: std.mem.Allocator) !InferencePort {
        var reader = try gguf.Reader.openWithAllocator(model_path, allocator);
        defer reader.deinit();

        const loader = try ModelLoader.load(model_path, allocator);
        errdefer {
            var loader_copy = loader;
            loader_copy.deinit(allocator);
        }
        const tok = try tokenizer.SimpleTokenizer.initFromReader(&reader, allocator);

        return .{
            .loader = loader,
            .tokenizer_ = tok,
        };
    }

    pub fn deinit(self: *InferencePort, allocator: std.mem.Allocator) void {
        self.loader.deinit(allocator);
        self.tokenizer_.deinit();
    }

    pub fn forward(self: *InferencePort, input_ids: []const u32) ![]f32 {
        return self.loader.engine_.forward(input_ids);
    }

    pub fn generate(self: *InferencePort, prompt: []const u8, max_tokens: u32) ![]u8 {
        return self.loader.engine_.generate(&self.tokenizer_, prompt, max_tokens);
    }

    pub fn benchmark(self: *InferencePort, prompt: []const u8, max_tokens: u32, io: std.Io) !BenchResult {
        self.reset();

        var ids = try self.tokenizer_.encode(prompt);
        defer ids.deinit();

        if (ids.items.len == 0) return error.EmptyInput;

        var all_ids = ArrayList(u32).init(self.tokenizer_.allocator);
        defer all_ids.deinit();
        try all_ids.appendSlice(ids.items);

        const prefill_start = std.Io.Timestamp.now(io, .awake);
        _ = try self.loader.engine_.prefill(ids.items);
        const prefill_elapsed = prefill_start.untilNow(io, .awake);

        const decode_start = std.Io.Timestamp.now(io, .awake);
        var generated_tokens: u32 = 0;
        while (generated_tokens < max_tokens) : (generated_tokens += 1) {
            const next_id = self.loader.engine_.greedyNextToken();
            if (next_id == self.tokenizer_.eos()) break;
            try all_ids.append(next_id);
            if (self.loader.engine_.kv_cache) |cache| {
                if (cache.seqLen() >= self.loader.engine_.config.max_seq_len) break;
            }
            _ = try self.loader.engine_.decodeStep(next_id);
        }
        const decode_elapsed = decode_start.untilNow(io, .awake);

        const text = try self.tokenizer_.decode(all_ids.items);
        return .{
            .text = text,
            .prompt_tokens = ids.items.len,
            .generated_tokens = generated_tokens,
            .prefill_ms = prefill_elapsed.toMilliseconds(),
            .decode_ms = decode_elapsed.toMilliseconds(),
        };
    }

    pub fn freeGenerated(self: *InferencePort, text: []u8) void {
        self.tokenizer_.allocator.free(text);
    }

    pub fn reset(self: *InferencePort) void {
        self.loader.engine_.reset();
    }

    pub fn getConfig(self: *const InferencePort) config.ModelConfig {
        return self.loader.config;
    }

    pub fn vocabSize(self: *const InferencePort) u32 {
        return self.loader.config.vocab_size;
    }
};

fn tensorSuffixForLayer(buf: []u8, layer_idx: usize, suffix: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "blk.{d}.{s}", .{ layer_idx, suffix });
}

fn findTensorNameBySuffix(reader: *const gguf.Reader, suffix: []const u8) ?[]const u8 {
    var it = reader.tensors.iterator();
    while (it.next()) |entry| {
        const name = entry.key_ptr.*;
        if (std.mem.eql(u8, name, suffix)) return name;
        if (name.len > suffix.len + 1 and name[name.len - suffix.len - 1] == '.' and std.mem.endsWith(u8, name, suffix)) {
            return name;
        }
    }
    return null;
}

fn isSupportedTensorType(dtype: gguf.GGMLType) bool {
    return switch (dtype) {
        .f16, .f32, .q8_0, .q6_k, .q5_k, .q4_k, .q2_k, .iq2_xs, .iq4_xs => true,
        else => false,
    };
}

fn hasTensorSuffix(tensors: []const CompatibilityTensor, suffix: []const u8) bool {
    for (tensors) |tensor| {
        if (std.mem.endsWith(u8, tensor.name, suffix)) return true;
    }
    return false;
}

fn dtypeForSuffix(tensors: []const CompatibilityTensor, suffix: []const u8) ?gguf.GGMLType {
    for (tensors) |tensor| {
        if (std.mem.endsWith(u8, tensor.name, suffix)) return tensor.dtype;
    }
    return null;
}

fn appendCompatibilityIssue(report: *ArrayList(u8), issue: []const u8) !void {
    if (report.items.len == 0) {
        try report.appendSlice("Model compatibility check FAILED.\n");
    }
    try report.appendSlice("  - ");
    try report.appendSlice(issue);
    try report.append('\n');
}

fn buildCompatibilityReportFromSummary(
    allocator: std.mem.Allocator,
    architecture: []const u8,
    has_hybrid_attention: bool,
    tensors: []const CompatibilityTensor,
) !?[]u8 {
    _ = architecture;
    _ = has_hybrid_attention;
    if (tensors.len == 0) return null;

    var report = ArrayList(u8).init(allocator);
    errdefer report.deinit();

    if (!hasTensorSuffix(tensors, "output.weight") and !hasTensorSuffix(tensors, "token_embd.weight")) {
        try appendCompatibilityIssue(&report, "ne output.weight ne de token_embd.weight bulundu; logits projection kurulamaz");
    }

    if (dtypeForSuffix(tensors, "token_embd.weight")) |dtype| {
        if (!isSupportedTensorType(dtype)) {
            var buf: [128]u8 = undefined;
            const msg = try std.fmt.bufPrint(&buf, "token_embd.weight unsupported dtype: {s}", .{@tagName(dtype)});
            try appendCompatibilityIssue(&report, msg);
        }
    }

    if (dtypeForSuffix(tensors, "output_norm.weight")) |dtype| {
        if (!isSupportedTensorType(dtype)) {
            var buf: [128]u8 = undefined;
            const msg = try std.fmt.bufPrint(&buf, "output_norm.weight unsupported dtype: {s}", .{@tagName(dtype)});
            try appendCompatibilityIssue(&report, msg);
        }
    }

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

// Phase 4: New byte-based tensor loading - stores raw bytes and dtype
fn loadTensorIfPresent(
    reader: *const gguf.Reader,
    suffix: []const u8,
    dst_bytes: []u8,
    dtype_slot_index: usize,
    weight_dtypes: []gguf.GGMLType,
    allocator: std.mem.Allocator,
) !bool {
    if (findTensorNameBySuffix(reader, suffix)) |tensor_name| {
        const info = reader.findTensorBySuffix(suffix).?;

        // Store the dtype for this tensor slot
        weight_dtypes[dtype_slot_index] = info.dtype;

        // Compute actual byte size from tensor shape and dtype
        const actual_bytes = computeTensorByteSize(info.dtype, info.shape);

        // Check if this is a simple fp16 load (most common case)
        if (info.dtype == .f16 and actual_bytes == dst_bytes.len) {
            // Direct load for fp16 with matching size
            try reader.loadTensorInto(tensor_name, dst_bytes);
            return true;
        }

        // For non-fp16 dtypes or size mismatches, need to convert
        if (actual_bytes != dst_bytes.len) {
            std.debug.print("INFO: tensor '{s}' has {d} bytes but destination has {d}, will convert from {s}\n", .{ tensor_name, actual_bytes, dst_bytes.len, @tagName(info.dtype) });
        }

        // Calculate number of elements in destination (always fp16)
        const num_elements = dst_bytes.len / @sizeOf(types.fp16_t);

        // Calculate number of elements in source tensor from shape
        var src_elements: u64 = 1;
        for (info.shape) |dim| {
            src_elements *= dim;
        }

        // Use the smaller of source and destination element counts
        const elements_to_convert = @min(num_elements, src_elements);

        if (src_elements > num_elements) {
            // Print shape info for debugging SSM size mismatches
            std.debug.print("WARN: tensor '{s}' has {d} elements (shape: ", .{ tensor_name, src_elements });
            for (info.shape, 0..) |dim, i| {
                if (i > 0) std.debug.print("×", .{});
                std.debug.print("{d}", .{dim});
            }
            std.debug.print(") but destination only has {d}, truncating to {d} elements\n", .{ num_elements, elements_to_convert });
        }

        // Load raw tensor data
        const raw = reader.loadTensor(tensor_name) catch |err| {
            std.debug.print("Failed to load tensor '{s}': {}\n", .{ tensor_name, err });
            return err;
        };
        defer allocator.free(raw);

        // Convert to fp16 and store in destination (only convert what fits)
        const dst_fp16 = @as([*]types.fp16_t, @ptrCast(@alignCast(dst_bytes.ptr)))[0..num_elements];
        // Calculate bytes per element for this dtype
        const bytes_per_element = if (info.dtype == .f32) @as(usize, 4) else if (info.dtype == .f16) @as(usize, 2) else blockSizeBytes(info.dtype) / blockElements(info.dtype);
        const raw_bytes_to_convert = elements_to_convert * bytes_per_element;
        try copyTensorToFp16(dst_fp16[0..elements_to_convert], info.dtype, raw[0..raw_bytes_to_convert]);

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
    allocator: std.mem.Allocator,
) !bool {
    if (findTensorNameBySuffix(reader, primary_suffix)) |_| {
        return try loadTensorIfPresent(reader, primary_suffix, dst_bytes, dtype_slot_index, weight_dtypes, allocator);
    } else if (findTensorNameBySuffix(reader, alias_suffix)) |_| {
        return try loadTensorIfPresent(reader, alias_suffix, dst_bytes, dtype_slot_index, weight_dtypes, allocator);
    }
    return false;
}

// Legacy dequantization function - kept for reference/testing
fn copyTensorToFp16(dst: []types.fp16_t, dtype: gguf.GGMLType, raw: []const u8) !void {
    switch (dtype) {
        .f16 => {
            if (raw.len != dst.len * @sizeOf(types.fp16_t)) return error.InvalidTensorSize;
            for (0..dst.len) |i| {
                dst[i] = std.mem.readInt(u16, raw[i * 2 ..][0..2], .little);
            }
        },
        .f32 => {
            if (raw.len != dst.len * @sizeOf(f32)) return error.InvalidTensorSize;
            for (0..dst.len) |i| {
                const value: f32 = @bitCast(std.mem.readInt(u32, raw[i * 4 ..][0..4], .little));
                dst[i] = types.fp32_to_fp16(value);
            }
        },
        .q8_0 => quant.dequantizeQ80ToFp16(raw.ptr, dst.ptr, dst.len),
        .q6_k => quant.dequantizeQ6KToFp16(raw.ptr, dst.ptr, dst.len),
        .q5_k => quant.dequantizeQ5KToFp16(raw.ptr, dst.ptr, dst.len),
        .q4_k => quant.dequantizeQ4KToFp16(raw.ptr, dst.ptr, dst.len),
        .q2_k => quant.dequantizeQ2KToFp16(raw.ptr, dst.ptr, dst.len),
        .iq2_xs => quant.dequantizeIQ2XSToFp16(raw.ptr, dst.ptr, dst.len),
        .iq4_xs => quant.dequantizeIQ4XSToFp16(raw.ptr, dst.ptr, dst.len),
        else => return error.UnsupportedTensorType,
    }
}

fn splitFusedQKV(
    reader: *const gguf.Reader,
    layer_idx: usize,
    q_dst: []types.fp16_t,
    k_dst: []types.fp16_t,
    v_dst: []types.fp16_t,
    hidden: usize,
    q_dim: usize,
    kv_dim: usize,
    allocator: std.mem.Allocator,
) !void {
    var name_buf: [96]u8 = undefined;
    const suffix = try tensorSuffixForLayer(&name_buf, layer_idx, "attn_qkv.weight");
    const tensor_name = findTensorNameBySuffix(reader, suffix) orelse return;
    const info = reader.findTensorBySuffix(suffix).?;

    const fused_elements = hidden * (q_dim + 2 * kv_dim);
    const temp_buf = try allocator.alloc(types.fp16_t, fused_elements);
    defer allocator.free(temp_buf);

    if (info.dtype == .f16) {
        try reader.loadTensorInto(tensor_name, std.mem.sliceAsBytes(temp_buf));
    } else {
        const raw = reader.loadTensor(tensor_name) catch |err| {
            std.debug.print("Failed to load fused QKV tensor '{s}': {}\n", .{ tensor_name, err });
            return err;
        };
        defer allocator.free(raw);
        try copyTensorToFp16(temp_buf, info.dtype, raw);
    }

    const q_size = hidden * q_dim;
    const kv_size = hidden * kv_dim;

    @memcpy(q_dst, temp_buf[0..q_size]);
    @memcpy(k_dst, temp_buf[q_size .. q_size + kv_size]);
    @memcpy(v_dst, temp_buf[q_size + kv_size .. fused_elements]);
}

fn loadWeightsFromReader(eng: *engine.Engine, reader: *const gguf.Reader, allocator: std.mem.Allocator) !void {
    const layout = eng.weight_layout;
    const cfg = eng.config;
    const hidden = cfg.hidden_dim;
    const kv_dim = cfg.num_kv_heads * cfg.head_dim;
    const ffn_h = cfg.ffn_hidden_dim;

    // Slot mapping: 0 = token_embd, then 11 slots per layer, then final_norm, output_proj
    var current_slot: usize = 0;

    // Token embedding (slot 0)
    const token_embd_bytes = cfg.vocab_size * hidden * @sizeOf(types.fp16_t);
    _ = try loadTensorIfPresent(reader, "token_embd.weight", eng.weight_pool[layout.token_embedding_offset..][0..token_embd_bytes], current_slot, eng.weight_dtypes, allocator);
    current_slot += 1;

    var name_buf: [96]u8 = undefined;
    for (0..cfg.num_layers) |layer_idx| {
        const layer_base = layout.token_embedding_size + layer_idx * layout.per_layer_size;

        // attn_norm (slot current_slot + 0)
        const norm_bytes = hidden * @sizeOf(types.fp16_t);
        _ = try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "attn_norm.weight"), eng.weight_pool[layer_base + layout.norm_weight_offset ..][0..norm_bytes], current_slot, eng.weight_dtypes, allocator);
        current_slot += 1;

        // q_proj (slot current_slot + 0)
        const q_dim = cfg.num_heads * cfg.head_dim;
        const q_proj_bytes = hidden * q_dim * @sizeOf(types.fp16_t);
        const q_loaded = try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "attn_q.weight"), eng.weight_pool[layer_base + layout.q_proj_offset ..][0..q_proj_bytes], current_slot, eng.weight_dtypes, allocator);
        current_slot += 1;

        // k_proj (slot current_slot + 0)
        const k_proj_bytes = hidden * kv_dim * @sizeOf(types.fp16_t);
        const k_loaded = try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "attn_k.weight"), eng.weight_pool[layer_base + layout.k_proj_offset ..][0..k_proj_bytes], current_slot, eng.weight_dtypes, allocator);
        current_slot += 1;

        // v_proj (slot current_slot + 0)
        const v_proj_bytes = hidden * kv_dim * @sizeOf(types.fp16_t);
        const v_loaded = try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "attn_v.weight"), eng.weight_pool[layer_base + layout.v_proj_offset ..][0..v_proj_bytes], current_slot, eng.weight_dtypes, allocator);
        current_slot += 1;

        // If any Q/K/V failed to load, try fused QKV split
        if (!q_loaded or !k_loaded or !v_loaded) {
            const q_f16 = @as([*]types.fp16_t, @ptrCast(@alignCast(eng.weight_pool[layer_base + layout.q_proj_offset ..].ptr)))[0 .. hidden * q_dim];
            const k_f16 = @as([*]types.fp16_t, @ptrCast(@alignCast(eng.weight_pool[layer_base + layout.k_proj_offset ..].ptr)))[0 .. hidden * kv_dim];
            const v_f16 = @as([*]types.fp16_t, @ptrCast(@alignCast(eng.weight_pool[layer_base + layout.v_proj_offset ..].ptr)))[0 .. hidden * kv_dim];
            try splitFusedQKV(reader, layer_idx, q_f16, k_f16, v_f16, hidden, q_dim, kv_dim, allocator);
        }

        // Detect layer type: check for SSM tensor presence
        const ssm_out_suffix = try tensorSuffixForLayer(&name_buf, layer_idx, "ssm_out.weight");
        if (findTensorNameBySuffix(reader, ssm_out_suffix) != null) {
            eng.weight_layout.layer_types[layer_idx] = .ssm;
        }

        // o_proj (slot current_slot + 0)
        const o_proj_bytes = hidden * q_dim * @sizeOf(types.fp16_t);
        const attn_output_suffix = try tensorSuffixForLayer(&name_buf, layer_idx, "attn_output.weight");
        _ = try loadTensorWithAliasIfPresent(reader, attn_output_suffix, try tensorSuffixForLayer(&name_buf, layer_idx, "attn_gate.weight"), eng.weight_pool[layer_base + layout.o_proj_offset ..][0..o_proj_bytes], current_slot, eng.weight_dtypes, allocator);
        current_slot += 1;

        // ffn_norm (slot current_slot + 0)
        const ffn_norm_bytes = hidden * @sizeOf(types.fp16_t);
        _ = try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "ffn_norm.weight"), eng.weight_pool[layer_base + layout.ffn_norm_offset ..][0..ffn_norm_bytes], current_slot, eng.weight_dtypes, allocator);
        // Check if ffn_norm is empty and try post_attention_norm fallback
        const ffn_norm_f16 = @as([*]types.fp16_t, @ptrCast(@alignCast(eng.weight_pool[layer_base + layout.ffn_norm_offset ..].ptr)))[0..hidden];
        if (std.mem.allEqual(types.fp16_t, ffn_norm_f16, types.fp32_to_fp16(0))) {
            _ = try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "post_attention_norm.weight"), eng.weight_pool[layer_base + layout.ffn_norm_offset ..][0..ffn_norm_bytes], current_slot, eng.weight_dtypes, allocator);
        }
        current_slot += 1;

        // attn_q_norm (slot current_slot + 0)
        const q_norm_bytes = cfg.head_dim * @sizeOf(types.fp16_t);
        _ = try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "attn_q_norm.weight"), eng.weight_pool[layer_base + layout.attn_q_norm_offset ..][0..q_norm_bytes], current_slot, eng.weight_dtypes, allocator);
        current_slot += 1;

        // attn_k_norm (slot current_slot + 0)
        const k_norm_bytes = cfg.head_dim * @sizeOf(types.fp16_t);
        _ = try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "attn_k_norm.weight"), eng.weight_pool[layer_base + layout.attn_k_norm_offset ..][0..k_norm_bytes], current_slot, eng.weight_dtypes, allocator);
        current_slot += 1;

        // FFN weights (3 slots)
        switch (cfg.ffn_type) {
            .dense => {
                const ffn_up_bytes = hidden * ffn_h * @sizeOf(types.fp16_t);
                const ffn_down_bytes = ffn_h * hidden * @sizeOf(types.fp16_t);
                _ = try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "ffn_up.weight"), eng.weight_pool[layer_base + layout.ffn_weight1_offset ..][0..ffn_up_bytes], current_slot, eng.weight_dtypes, allocator);
                current_slot += 1;
                _ = try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "ffn_down.weight"), eng.weight_pool[layer_base + layout.ffn_weight2_offset ..][0..ffn_down_bytes], current_slot, eng.weight_dtypes, allocator);
                current_slot += 1;
                current_slot += 1; // Skip ffn_weight3 slot for dense (not used)
            },
            .gated_swi_glu, .gated_gelu => {
                const gate_bytes = hidden * ffn_h * @sizeOf(types.fp16_t);
                const up_bytes = hidden * ffn_h * @sizeOf(types.fp16_t);
                const down_bytes = ffn_h * hidden * @sizeOf(types.fp16_t);
                _ = try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "ffn_gate.weight"), eng.weight_pool[layer_base + layout.ffn_weight1_offset ..][0..gate_bytes], current_slot, eng.weight_dtypes, allocator);
                current_slot += 1;
                _ = try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "ffn_up.weight"), eng.weight_pool[layer_base + layout.ffn_weight2_offset ..][0..up_bytes], current_slot, eng.weight_dtypes, allocator);
                current_slot += 1;
                _ = try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "ffn_down.weight"), eng.weight_pool[layer_base + layout.ffn_weight3_offset ..][0..down_bytes], current_slot, eng.weight_dtypes, allocator);
                current_slot += 1;
            },
        }

        // SSM weights (8 tensors) - only loaded if layer is SSM type
        if (eng.weight_layout.layer_types[layer_idx] == .ssm) {
            const ssm_base = layout.ssm_region_offset + layer_idx * layout.ssm_per_layer_size;
            const ssm_inner = cfg.ssm_inner_size;
            const dt_rank = cfg.ssm_dt_rank;
            const conv_kernel = cfg.ssm_conv_kernel;

            // Calculate ssm_out dimension (must match memory.zig calculation)
            const ssm_out_hidden_dim = if (cfg.ffn_hidden_dim > hidden and cfg.ffn_type != .dense)
                cfg.ffn_hidden_dim / 2
            else
                hidden * 2;

            // ssm_out: hidden × ssm_out_hidden_dim (conservative sizing for Qwen 3.5)
            const ssm_out_bytes = hidden * ssm_out_hidden_dim * @sizeOf(types.fp16_t);
            _ = try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "ssm_out.weight"), eng.weight_pool[ssm_base + layout.ssm_out_offset ..][0..ssm_out_bytes], current_slot, eng.weight_dtypes, allocator);
            current_slot += 1;

            // ssm_x: hidden x hidden
            const ssm_x_bytes = hidden * hidden * @sizeOf(types.fp16_t);
            _ = try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "ssm_x.weight"), eng.weight_pool[ssm_base + layout.ssm_x_offset ..][0..ssm_x_bytes], current_slot, eng.weight_dtypes, allocator);
            current_slot += 1;

            // ssm_dt: hidden x dt_rank
            const ssm_dt_bytes = hidden * dt_rank * @sizeOf(types.fp16_t);
            _ = try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "ssm_dt.weight"), eng.weight_pool[ssm_base + layout.ssm_dt_offset ..][0..ssm_dt_bytes], current_slot, eng.weight_dtypes, allocator);
            current_slot += 1;

            // ssm_A: ssm_inner
            const ssm_A_bytes = ssm_inner * @sizeOf(types.fp16_t);
            _ = try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "ssm_A.weight"), eng.weight_pool[ssm_base + layout.ssm_A_offset ..][0..ssm_A_bytes], current_slot, eng.weight_dtypes, allocator);
            current_slot += 1;

            // ssm_B: hidden x ssm_inner
            const ssm_B_bytes = hidden * ssm_inner * @sizeOf(types.fp16_t);
            _ = try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "ssm_B.weight"), eng.weight_pool[ssm_base + layout.ssm_B_offset ..][0..ssm_B_bytes], current_slot, eng.weight_dtypes, allocator);
            current_slot += 1;

            // ssm_C: hidden x ssm_inner
            const ssm_C_bytes = hidden * ssm_inner * @sizeOf(types.fp16_t);
            _ = try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "ssm_C.weight"), eng.weight_pool[ssm_base + layout.ssm_C_offset ..][0..ssm_C_bytes], current_slot, eng.weight_dtypes, allocator);
            current_slot += 1;

            // ssm_D: hidden
            const ssm_D_bytes = hidden * @sizeOf(types.fp16_t);
            _ = try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "ssm_D.weight"), eng.weight_pool[ssm_base + layout.ssm_D_offset ..][0..ssm_D_bytes], current_slot, eng.weight_dtypes, allocator);
            current_slot += 1;

            // ssm_conv1d: conv_kernel x hidden
            const ssm_conv1d_bytes = conv_kernel * hidden * @sizeOf(types.fp16_t);
            _ = try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "ssm_conv1d.weight"), eng.weight_pool[ssm_base + layout.ssm_conv1d_offset ..][0..ssm_conv1d_bytes], current_slot, eng.weight_dtypes, allocator);
            current_slot += 1;
        }
    }

    // Final norm (slot after all layers)
    const final_norm_bytes = hidden * @sizeOf(types.fp16_t);
    _ = try loadTensorIfPresent(reader, "output_norm.weight", eng.weight_pool[layout.final_norm_offset..][0..final_norm_bytes], current_slot, eng.weight_dtypes, allocator);
    current_slot += 1;

    // Output projection (final slot)
    const output_bytes = hidden * cfg.vocab_size * @sizeOf(types.fp16_t);
    if (reader.findTensorBySuffix("output.weight") != null) {
        _ = try loadTensorIfPresent(reader, "output.weight", eng.weight_pool[layout.output_proj_offset..][0..output_bytes], current_slot, eng.weight_dtypes, allocator);
    } else {
        // Fallback: copy from token embedding
        const token_embd_f16 = @as([*]types.fp16_t, @ptrCast(@alignCast(eng.weight_pool[layout.token_embedding_offset..].ptr)))[0..(cfg.vocab_size * hidden)];
        const output_f16 = @as([*]types.fp16_t, @ptrCast(@alignCast(eng.weight_pool[layout.output_proj_offset..].ptr)))[0..(hidden * cfg.vocab_size)];
        @memcpy(output_f16, token_embd_f16);
        // Mark as fp16 dtype for output projection
        eng.weight_dtypes[current_slot] = .f16;
    }
}

fn appendU32LE(list: *ArrayList(u8), value: u32) !void {
    // Cast u32 to u8 bytes - Zig 0.16 syntax
    try list.append(@truncate(value & 0xFF));
    try list.append(@truncate((value >> 8) & 0xFF));
    try list.append(@truncate((value >> 16) & 0xFF));
    try list.append(@truncate((value >> 24) & 0xFF));
}

fn appendU64LE(list: *ArrayList(u8), value: u64) !void {
    for (0..8) |i| {
        const shift: u6 = @intCast(i * 8);
        try list.append(@truncate(value >> shift));
    }
}

fn appendI32LE(list: *ArrayList(u8), value: i32) !void {
    try appendU32LE(list, @as(u32, @bitCast(value)));
}

fn appendF32LE(list: *ArrayList(u8), value: f32) !void {
    try appendU32LE(list, @bitCast(value));
}

fn appendString(list: *ArrayList(u8), value: []const u8) !void {
    try appendU64LE(list, value.len);
    try list.appendSlice(value);
}

fn buildSyntheticPortGguf(allocator: std.mem.Allocator) ![]u8 {
    var bytes = ArrayList(u8).init(allocator);
    errdefer bytes.deinit();

    try appendU32LE(&bytes, 0x46554747);
    try appendU32LE(&bytes, 3);
    try appendU64LE(&bytes, 0);
    try appendU64LE(&bytes, 12);

    try appendString(&bytes, "llama.context_length");
    try appendU32LE(&bytes, 4);
    try appendU32LE(&bytes, 128);
    try appendString(&bytes, "llama.embedding_length");
    try appendU32LE(&bytes, 4);
    try appendU32LE(&bytes, 8);
    try appendString(&bytes, "llama.block_count");
    try appendU32LE(&bytes, 4);
    try appendU32LE(&bytes, 1);
    try appendString(&bytes, "llama.feed_forward_length");
    try appendU32LE(&bytes, 4);
    try appendU32LE(&bytes, 16);
    try appendString(&bytes, "llama.attention.head_count");
    try appendU32LE(&bytes, 4);
    try appendU32LE(&bytes, 2);
    try appendString(&bytes, "llama.attention.head_count_kv");
    try appendU32LE(&bytes, 4);
    try appendU32LE(&bytes, 1);
    try appendString(&bytes, "tokenizer.ggml.model");
    try appendU32LE(&bytes, 8);
    try appendString(&bytes, "sp");
    try appendString(&bytes, "tokenizer.ggml.tokens");
    try appendU32LE(&bytes, 9);
    try appendU32LE(&bytes, 8);
    try appendU64LE(&bytes, 4);
    try appendString(&bytes, "<bos>");
    try appendString(&bytes, "<eos>");
    try appendString(&bytes, "\xE2\x96\x81hi");
    try appendString(&bytes, "<0x21>");
    try appendString(&bytes, "tokenizer.ggml.scores");
    try appendU32LE(&bytes, 9);
    try appendU32LE(&bytes, 6);
    try appendU64LE(&bytes, 4);
    try appendF32LE(&bytes, 0);
    try appendF32LE(&bytes, 0);
    try appendF32LE(&bytes, 1);
    try appendF32LE(&bytes, 0.5);
    try appendString(&bytes, "tokenizer.ggml.token_type");
    try appendU32LE(&bytes, 9);
    try appendU32LE(&bytes, 5);
    try appendU64LE(&bytes, 4);
    try appendI32LE(&bytes, 3);
    try appendI32LE(&bytes, 3);
    try appendI32LE(&bytes, 1);
    try appendI32LE(&bytes, 6);
    try appendString(&bytes, "tokenizer.ggml.bos_token_id");
    try appendU32LE(&bytes, 4);
    try appendU32LE(&bytes, 0);
    try appendString(&bytes, "tokenizer.ggml.eos_token_id");
    try appendU32LE(&bytes, 4);
    try appendU32LE(&bytes, 1);

    const aligned_len = std.mem.alignForward(usize, bytes.items.len, 32);
    while (bytes.items.len < aligned_len) try bytes.append(0);
    return try bytes.toOwnedSlice();
}

test "InferencePort init loads GGUF config and tokenizer" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;

    const data = try buildSyntheticPortGguf(std.testing.allocator);
    defer std.testing.allocator.free(data);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "port.gguf", .data = data });

    var path_buf: [256]u8 = undefined;
    const rel_path = try std.fmt.bufPrint(&path_buf, ".zig-cache/tmp/{s}/port.gguf", .{tmp.sub_path[0..]});

    var port = try InferencePort.init(rel_path, std.testing.allocator);
    defer port.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u32, 4), port.vocabSize());
    try std.testing.expectEqual(@as(u32, 8), port.getConfig().hidden_dim);
    try std.testing.expectEqual(@as(usize, 4), port.tokenizer_.vocabSize());
}

test "compatibility report accepts classic transformer summary" {
    const tensors = [_]CompatibilityTensor{
        .{ .name = "token_embd.weight", .dtype = .q8_0 },
        .{ .name = "output_norm.weight", .dtype = .f32 },
        .{ .name = "output.weight", .dtype = .q4_k },
        .{ .name = "blk.0.attn_norm.weight", .dtype = .f32 },
        .{ .name = "blk.0.attn_q.weight", .dtype = .q4_k },
        .{ .name = "blk.0.attn_k.weight", .dtype = .q4_k },
        .{ .name = "blk.0.attn_v.weight", .dtype = .q4_k },
        .{ .name = "blk.0.attn_output.weight", .dtype = .q4_k },
        .{ .name = "blk.0.ffn_gate.weight", .dtype = .q4_k },
        .{ .name = "blk.0.ffn_up.weight", .dtype = .q4_k },
        .{ .name = "blk.0.ffn_down.weight", .dtype = .q4_k },
    };

    const report = try buildCompatibilityReportFromSummary(std.testing.allocator, "qwen2", false, &tensors);
    defer if (report) |text| std.testing.allocator.free(text);
    try std.testing.expect(report == null);
}

test "findTensorNameBySuffix returns exact tensor key" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;

    const data = try gguf.buildSyntheticGgufForTests(std.testing.allocator);
    defer std.testing.allocator.free(data);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "sample.gguf", .data = data });

    var path_buf: [256]u8 = undefined;
    const rel_path = try std.fmt.bufPrint(&path_buf, ".zig-cache/tmp/{s}/sample.gguf", .{tmp.sub_path[0..]});

    var reader = try gguf.Reader.openWithAllocator(rel_path, std.testing.allocator);
    defer reader.deinit();

    try std.testing.expectEqualStrings("blk.0.attn_q.weight", findTensorNameBySuffix(&reader, "attn_q.weight").?);
}

test "compatibility report accepts fused QKV without separate Q/K/V" {
    const tensors = [_]CompatibilityTensor{
        .{ .name = "token_embd.weight", .dtype = .q8_0 },
        .{ .name = "output_norm.weight", .dtype = .f32 },
        .{ .name = "output.weight", .dtype = .q4_k },
        .{ .name = "blk.0.attn_norm.weight", .dtype = .f32 },
        .{ .name = "blk.0.attn_qkv.weight", .dtype = .q8_0 },
        .{ .name = "blk.0.attn_output.weight", .dtype = .q4_k },
        .{ .name = "blk.0.ffn_gate.weight", .dtype = .q4_k },
        .{ .name = "blk.0.ffn_up.weight", .dtype = .q4_k },
        .{ .name = "blk.0.ffn_down.weight", .dtype = .q4_k },
    };

    const report = try buildCompatibilityReportFromSummary(std.testing.allocator, "qwen2", false, &tensors);
    defer if (report) |text| std.testing.allocator.free(text);
    try std.testing.expect(report == null);
}

test "compatibility report rejects hybrid qwen35 style tensors" {
    const tensors = [_]CompatibilityTensor{
        .{ .name = "token_embd.weight", .dtype = .q8_0 },
        .{ .name = "output_norm.weight", .dtype = .f32 },
        .{ .name = "blk.0.attn_qkv.weight", .dtype = .q8_0 },
        .{ .name = "blk.0.attn_gate.weight", .dtype = .q8_0 },
        .{ .name = "blk.0.post_attention_norm.weight", .dtype = .f32 },
        .{ .name = "blk.0.ssm_out.weight", .dtype = .q8_0 },
    };

    const report = (try buildCompatibilityReportFromSummary(std.testing.allocator, "qwen35", true, &tensors)).?;
    defer std.testing.allocator.free(report);

    try std.testing.expect(std.mem.indexOf(u8, report, "qwen35") != null);
    try std.testing.expect(std.mem.indexOf(u8, report, "SSM") != null);
}

test "compatibility report rejects unsupported tensor dtype" {
    const tensors = [_]CompatibilityTensor{
        .{ .name = "token_embd.weight", .dtype = .iq3_xxs },
        .{ .name = "output_norm.weight", .dtype = .f32 },
        .{ .name = "output.weight", .dtype = .q4_k },
        .{ .name = "blk.0.attn_norm.weight", .dtype = .f32 },
        .{ .name = "blk.0.attn_q.weight", .dtype = .q4_k },
        .{ .name = "blk.0.attn_k.weight", .dtype = .q4_k },
        .{ .name = "blk.0.attn_v.weight", .dtype = .q4_k },
        .{ .name = "blk.0.attn_output.weight", .dtype = .q4_k },
        .{ .name = "blk.0.ffn_gate.weight", .dtype = .iq3_xxs },
        .{ .name = "blk.0.ffn_up.weight", .dtype = .q4_k },
        .{ .name = "blk.0.ffn_down.weight", .dtype = .q4_k },
    };

    const report = (try buildCompatibilityReportFromSummary(std.testing.allocator, "llama", false, &tensors)).?;
    defer std.testing.allocator.free(report);
    try std.testing.expect(std.mem.indexOf(u8, report, "iq3_xxs") != null);
}

test "compatibility report accepts qwen3 classic transformer summary" {
    const tensors = [_]CompatibilityTensor{
        .{ .name = "token_embd.weight", .dtype = .q6_k },
        .{ .name = "output_norm.weight", .dtype = .f32 },
        .{ .name = "blk.0.attn_norm.weight", .dtype = .f32 },
        .{ .name = "blk.0.attn_q.weight", .dtype = .q4_k },
        .{ .name = "blk.0.attn_k.weight", .dtype = .q4_k },
        .{ .name = "blk.0.attn_v.weight", .dtype = .iq2_xs },
        .{ .name = "blk.0.attn_output.weight", .dtype = .q4_k },
        .{ .name = "blk.0.attn_q_norm.weight", .dtype = .f32 },
        .{ .name = "blk.0.attn_k_norm.weight", .dtype = .f32 },
        .{ .name = "blk.0.ffn_norm.weight", .dtype = .f32 },
        .{ .name = "blk.0.ffn_gate.weight", .dtype = .q4_k },
        .{ .name = "blk.0.ffn_up.weight", .dtype = .q4_k },
        .{ .name = "blk.0.ffn_down.weight", .dtype = .q6_k },
    };

    const report = try buildCompatibilityReportFromSummary(std.testing.allocator, "qwen3", false, &tensors);
    defer if (report) |text| std.testing.allocator.free(text);
    try std.testing.expect(report == null);
}

test "compatibility report accepts q5_k and iq4_xs tensor dtypes" {
    const tensors = [_]CompatibilityTensor{
        .{ .name = "token_embd.weight", .dtype = .q5_k },
        .{ .name = "output_norm.weight", .dtype = .f32 },
        .{ .name = "output.weight", .dtype = .q4_k },
        .{ .name = "blk.0.attn_norm.weight", .dtype = .f32 },
        .{ .name = "blk.0.attn_q.weight", .dtype = .q4_k },
        .{ .name = "blk.0.attn_k.weight", .dtype = .q4_k },
        .{ .name = "blk.0.attn_v.weight", .dtype = .q5_k },
        .{ .name = "blk.0.attn_output.weight", .dtype = .q4_k },
        .{ .name = "blk.0.ffn_gate.weight", .dtype = .iq4_xs },
        .{ .name = "blk.0.ffn_up.weight", .dtype = .q4_k },
        .{ .name = "blk.0.ffn_down.weight", .dtype = .q4_k },
    };

    const report = try buildCompatibilityReportFromSummary(std.testing.allocator, "llama", false, &tensors);
    defer if (report) |text| std.testing.allocator.free(text);
    try std.testing.expect(report == null);
}
