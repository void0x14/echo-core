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
    if (tensors.len == 0) return null;

    var report = ArrayList(u8).init(allocator);
    errdefer report.deinit();

    if (architecture.len > 0 and
        !std.mem.eql(u8, architecture, "llama") and
        !std.mem.eql(u8, architecture, "qwen2") and
        !std.mem.eql(u8, architecture, "qwen3") and
        !std.mem.eql(u8, architecture, "qwen3vl") and
        !std.mem.eql(u8, architecture, "gemma3") and
        !std.mem.eql(u8, architecture, "olmo2"))
    {
        var buf: [192]u8 = undefined;
        const msg = try std.fmt.bufPrint(
            &buf,
            "general.architecture=\"{s}\"; engine sadece llama/qwen2/qwen3/gemma3/olmo2 benzeri klasik transformer akisini destekliyor",
            .{architecture},
        );
        try appendCompatibilityIssue(&report, msg);
    }

    if (has_hybrid_attention) {
        try appendCompatibilityIssue(&report, "model metadata attention+SSM hybrid akis gosteriyor; Zig engine SSM dalini hic uygulamiyor");
    }

    for (tensors) |tensor| {
        if (std.mem.indexOf(u8, tensor.name, ".ssm_") != null) {
            var buf: [192]u8 = undefined;
            const msg = try std.fmt.bufPrint(&buf, "SSM tensoru bulundu: {s}", .{tensor.name});
            try appendCompatibilityIssue(&report, msg);
            break;
        }
    }

    if (hasTensorSuffix(tensors, "attn_qkv.weight") and
        !(hasTensorSuffix(tensors, "attn_q.weight") and
            hasTensorSuffix(tensors, "attn_k.weight") and
            hasTensorSuffix(tensors, "attn_v.weight")))
    {
        try appendCompatibilityIssue(&report, "fused attn_qkv.weight var ama ayri Q/K/V tensorlari yok; Zig engine fused QKV ayirmiyor");
    }

    if (!hasTensorSuffix(tensors, "attn_output.weight") and hasTensorSuffix(tensors, "attn_gate.weight")) {
        try appendCompatibilityIssue(&report, "attn_output.weight yok, attn_gate.weight var; Zig engine attention output icin gate tensorunu kullanmiyor");
    }

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

fn loadTensorIfPresent(reader: *const gguf.Reader, suffix: []const u8, dst: []types.fp16_t, allocator: std.mem.Allocator) !void {
    if (findTensorNameBySuffix(reader, suffix)) |tensor_name| {
        const info = reader.findTensorBySuffix(suffix).?;
        // Compute actual element count from tensor shape
        var actual_elements: u64 = 1;
        for (info.shape) |dim| actual_elements *= dim;
        if (actual_elements != dst.len) {
            std.debug.print("WARN: tensor '{s}' shape has {d} elements but dst expects {d}, using min\n", .{ tensor_name, actual_elements, dst.len });
        }
        const n_to_copy = @min(dst.len, @as(usize, @intCast(actual_elements)));

        if (info.dtype == .f16 and n_to_copy == dst.len) {
            try reader.loadTensorInto(tensor_name, std.mem.sliceAsBytes(dst));
            return;
        }

        const raw = reader.loadTensor(tensor_name) catch |err| {
            std.debug.print("Failed to load tensor '{s}': {}\n", .{ tensor_name, err });
            return err;
        };
        defer allocator.free(raw);
        try copyTensorToFp16(dst[0..n_to_copy], info.dtype, raw);
    }
}

fn loadWeightsFromReader(eng: *engine.Engine, reader: *const gguf.Reader, allocator: std.mem.Allocator) !void {
    const layout = eng.weight_layout;
    const cfg = eng.config;
    const hidden = cfg.hidden_dim;
    const kv_dim = cfg.num_kv_heads * cfg.head_dim;
    const ffn_h = cfg.ffn_hidden_dim;

    const token_embd_off = layout.token_embedding_offset / @sizeOf(types.fp16_t);
    try loadTensorIfPresent(reader, "token_embd.weight", eng.weight_pool[token_embd_off .. token_embd_off + cfg.vocab_size * hidden], allocator);

    const final_norm_off = layout.final_norm_offset / @sizeOf(types.fp16_t);
    try loadTensorIfPresent(reader, "output_norm.weight", eng.weight_pool[final_norm_off .. final_norm_off + hidden], allocator);

    const output_off = layout.output_proj_offset / @sizeOf(types.fp16_t);
    if (reader.findTensorBySuffix("output.weight") != null) {
        try loadTensorIfPresent(reader, "output.weight", eng.weight_pool[output_off .. output_off + hidden * cfg.vocab_size], allocator);
    } else {
        @memcpy(
            eng.weight_pool[output_off .. output_off + hidden * cfg.vocab_size],
            eng.weight_pool[token_embd_off .. token_embd_off + hidden * cfg.vocab_size],
        );
    }

    var name_buf: [96]u8 = undefined;
    for (0..cfg.num_layers) |layer_idx| {
        const layer_base = layout.token_embedding_size + layer_idx * layout.per_layer_size;

        const norm_off = (layer_base + layout.norm_weight_offset) / @sizeOf(types.fp16_t);
        try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "attn_norm.weight"), eng.weight_pool[norm_off .. norm_off + hidden], allocator);

        const q_dim = cfg.num_heads * cfg.head_dim;
        const q_off = (layer_base + layout.q_proj_offset) / @sizeOf(types.fp16_t);
        try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "attn_q.weight"), eng.weight_pool[q_off .. q_off + hidden * q_dim], allocator);

        const k_off = (layer_base + layout.k_proj_offset) / @sizeOf(types.fp16_t);
        try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "attn_k.weight"), eng.weight_pool[k_off .. k_off + hidden * kv_dim], allocator);

        const v_off = (layer_base + layout.v_proj_offset) / @sizeOf(types.fp16_t);
        try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "attn_v.weight"), eng.weight_pool[v_off .. v_off + hidden * kv_dim], allocator);

        const o_off = (layer_base + layout.o_proj_offset) / @sizeOf(types.fp16_t);
        try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "attn_output.weight"), eng.weight_pool[o_off .. o_off + hidden * q_dim], allocator);

        const ffn_norm_off = (layer_base + layout.ffn_norm_offset) / @sizeOf(types.fp16_t);
        try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "ffn_norm.weight"), eng.weight_pool[ffn_norm_off .. ffn_norm_off + hidden], allocator);
        if (std.mem.allEqual(types.fp16_t, eng.weight_pool[ffn_norm_off .. ffn_norm_off + hidden], types.fp32_to_fp16(0))) {
            try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "post_attention_norm.weight"), eng.weight_pool[ffn_norm_off .. ffn_norm_off + hidden], allocator);
        }

        // Load attn Q/K head norms (qwen3, gemma3, olmo2 use these)
        const q_norm_off = (layer_base + layout.attn_q_norm_offset) / @sizeOf(types.fp16_t);
        try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "attn_q_norm.weight"), eng.weight_pool[q_norm_off .. q_norm_off + cfg.head_dim], allocator);

        const k_norm_off = (layer_base + layout.attn_k_norm_offset) / @sizeOf(types.fp16_t);
        try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "attn_k_norm.weight"), eng.weight_pool[k_norm_off .. k_norm_off + cfg.head_dim], allocator);

        const w1_off = (layer_base + layout.ffn_weight1_offset) / @sizeOf(types.fp16_t);
        const w2_off = (layer_base + layout.ffn_weight2_offset) / @sizeOf(types.fp16_t);
        const w3_off = (layer_base + layout.ffn_weight3_offset) / @sizeOf(types.fp16_t);
        switch (cfg.ffn_type) {
            .dense => {
                try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "ffn_up.weight"), eng.weight_pool[w1_off .. w1_off + hidden * ffn_h], allocator);
                try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "ffn_down.weight"), eng.weight_pool[w2_off .. w2_off + ffn_h * hidden], allocator);
            },
            .gated_swi_glu, .gated_gelu => {
                try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "ffn_gate.weight"), eng.weight_pool[w1_off .. w1_off + hidden * ffn_h], allocator);
                try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "ffn_up.weight"), eng.weight_pool[w2_off .. w2_off + hidden * ffn_h], allocator);
                try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "ffn_down.weight"), eng.weight_pool[w3_off .. w3_off + ffn_h * hidden], allocator);
            },
        }
    }
}

fn appendU32LE(list: *ArrayList(u8), value: u32) !void {
    try list.append(@truncate(value));
    try list.append(@truncate(value >> 8));
    try list.append(@truncate(value >> 16));
    try list.append(@truncate(value >> 24));
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
    try std.testing.expect(std.mem.indexOf(u8, report, "attn_qkv") != null);
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
