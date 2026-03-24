const std = @import("std");
const builtin = @import("builtin");
const config = @import("../core/config.zig");
const types = @import("../core/types.zig");
const engine = @import("../inference/engine.zig");
const tokenizer = @import("../tokenizer/tokenizer.zig");
const gguf = @import("../gguf/reader.zig");
const quant = @import("../kernels/quant.zig");

const ArrayList = std.array_list.Managed;

pub const ModelLoader = struct {
    config: config.ModelConfig,
    engine_: engine.Engine,

    pub fn load(model_path: []const u8, allocator: std.mem.Allocator) !ModelLoader {
        var reader = try gguf.Reader.openWithAllocator(model_path, allocator);
        defer reader.deinit();

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
        .q4_k => quant.dequantizeQ4KToFp16(raw.ptr, dst.ptr, dst.len),
        .q2_k => quant.dequantizeQ2KToFp16(raw.ptr, dst.ptr, dst.len),
        else => return error.UnsupportedTensorType,
    }
}

fn loadTensorIfPresent(reader: *const gguf.Reader, suffix: []const u8, dst: []types.fp16_t, allocator: std.mem.Allocator) !void {
    if (reader.findTensorBySuffix(suffix)) |info| {
        const raw = try reader.loadTensor(suffix);
        defer allocator.free(raw);
        try copyTensorToFp16(dst, info.dtype, raw);
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
    try loadTensorIfPresent(reader, "output.weight", eng.weight_pool[output_off .. output_off + hidden * cfg.vocab_size], allocator);

    var name_buf: [96]u8 = undefined;
    for (0..cfg.num_layers) |layer_idx| {
        const layer_base = layout.token_embedding_size + layer_idx * layout.per_layer_size;

        const norm_off = (layer_base + layout.norm_weight_offset) / @sizeOf(types.fp16_t);
        try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "attn_norm.weight"), eng.weight_pool[norm_off .. norm_off + hidden], allocator);

        const q_off = (layer_base + layout.q_proj_offset) / @sizeOf(types.fp16_t);
        try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "attn_q.weight"), eng.weight_pool[q_off .. q_off + hidden * hidden], allocator);

        const k_off = (layer_base + layout.k_proj_offset) / @sizeOf(types.fp16_t);
        try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "attn_k.weight"), eng.weight_pool[k_off .. k_off + hidden * kv_dim], allocator);

        const v_off = (layer_base + layout.v_proj_offset) / @sizeOf(types.fp16_t);
        try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "attn_v.weight"), eng.weight_pool[v_off .. v_off + hidden * kv_dim], allocator);

        const o_off = (layer_base + layout.o_proj_offset) / @sizeOf(types.fp16_t);
        try loadTensorIfPresent(reader, try tensorSuffixForLayer(&name_buf, layer_idx, "attn_output.weight"), eng.weight_pool[o_off .. o_off + hidden * hidden], allocator);

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
