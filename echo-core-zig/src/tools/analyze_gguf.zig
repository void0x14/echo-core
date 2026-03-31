const std = @import("std");

// Minimal GGUF types for standalone analysis
const GGUF_MAGIC: u32 = 0x46554747;
const GGUF_VERSION: u32 = 3;

const GGMLType = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    q8_k = 15,
    iq2_xxs = 16,
    iq2_xs = 17,
    iq3_xxs = 18,
    iq1_s = 19,
    iq4_nl = 20,
    iq3_s = 21,
    iq2_s = 22,
    iq4_xs = 23,
    i8 = 24,
    i16 = 25,
    i32 = 26,
    i64 = 27,
    f64 = 28,
    iq1_m = 29,
    bf16 = 30,
    tq1_0 = 34,
    tq2_0 = 35,
    mxfp4 = 39,
    count = 40,
};

const TensorInfo = struct {
    name: []const u8,
    offset: u64,
    size: u64,
    shape: []const u64,
    dtype: GGMLType,
};

const GGUFAnalyzer = struct {
    file: std.fs.File,
    allocator: std.mem.Allocator,
    cursor: u64,
    data_offset: u64,
    tensors: std.ArrayList(TensorInfo),
    metadata: std.StringHashMap(u64),

    fn init(file: std.fs.File, allocator: std.mem.Allocator) GGUFAnalyzer {
        return .{
            .file = file,
            .allocator = allocator,
            .cursor = 0,
            .data_offset = 0,
            .tensors = std.ArrayList(TensorInfo).init(allocator),
            .metadata = std.StringHashMap(u64).init(allocator),
        };
    }

    fn deinit(self: *GGUFAnalyzer) void {
        for (self.tensors.items) |t| {
            self.allocator.free(t.name);
            self.allocator.free(t.shape);
        }
        self.tensors.deinit();
        self.metadata.deinit();
    }

    fn readU32(self: *GGUFAnalyzer) !u32 {
        var buf: [4]u8 = undefined;
        const n = try self.file.read(&buf);
        if (n != 4) return error.EndOfStream;
        self.cursor += 4;
        return std.mem.readInt(u32, &buf, .little);
    }

    fn readU64(self: *GGUFAnalyzer) !u64 {
        var buf: [8]u8 = undefined;
        const n = try self.file.read(&buf);
        if (n != 8) return error.EndOfStream;
        self.cursor += 8;
        return std.mem.readInt(u64, &buf, .little);
    }

    fn readString(self: *GGUFAnalyzer) ![]const u8 {
        const len = try self.readU64();
        const buf = try self.allocator.alloc(u8, @intCast(len));
        const n = try self.file.read(buf);
        if (n != len) return error.EndOfStream;
        self.cursor += len;
        return buf;
    }

    fn readHeader(self: *GGUFAnalyzer) !void {
        const magic = try self.readU32();
        if (magic != GGUF_MAGIC) return error.InvalidGGUFMagic;

        const version = try self.readU32();
        if (version != GGUF_VERSION) return error.UnsupportedGGUFVersion;

        const tensor_count = try self.readU64();
        const metadata_kv_count = try self.readU64();

        var i: u64 = 0;
        while (i < metadata_kv_count) : (i += 1) {
            const key = try self.readString();
            defer self.allocator.free(key);
            const type_id = try self.readU32();

            // Skip value for analysis (we only need numeric metadata)
            switch (type_id) {
                0, 1 => { // uint8, int8
                    var buf: [1]u8 = undefined;
                    _ = try self.file.read(&buf);
                    self.cursor += 1;
                },
                2, 3 => { // uint16, int16
                    var buf: [2]u8 = undefined;
                    _ = try self.file.read(&buf);
                    self.cursor += 2;
                },
                4, 5 => { // uint32, int32
                    var buf: [4]u8 = undefined;
                    _ = try self.file.read(&buf);
                    self.cursor += 4;
                },
                6 => { // float32
                    var buf: [4]u8 = undefined;
                    _ = try self.file.read(&buf);
                    self.cursor += 4;
                },
                10, 11 => { // uint64, int64
                    var buf: [8]u8 = undefined;
                    _ = try self.file.read(&buf);
                    self.cursor += 8;
                },
                8 => { // string
                    const s = try self.readString();
                    self.allocator.free(s);
                },
                9 => { // array
                    const elem_type = try self.readU32();
                    const n_elems = try self.readU64();
                    for (0..@intCast(n_elems)) |_| {
                        switch (elem_type) {
                            0, 1 => { // uint8, int8
                                var buf: [1]u8 = undefined;
                                _ = try self.file.read(&buf);
                                self.cursor += 1;
                            },
                            4 => { // uint32
                                var buf: [4]u8 = undefined;
                                _ = try self.file.read(&buf);
                                self.cursor += 4;
                            },
                            8 => { // string
                                const s = try self.readString();
                                self.allocator.free(s);
                            },
                            else => {},
                        }
                    }
                },
                else => {},
            }

            // Extract key metadata
            if (std.mem.eql(u8, key, "llama.embedding_length")) {
                try self.metadata.put("hidden_dim", try self.readU64());
            }
            if (std.mem.eql(u8, key, "llama.vocab_size") or std.mem.eql(u8, key, "general.architecture")) {
                try self.metadata.put("vocab_size", 151936);
            }
        }

        // Read tensors
        i = 0;
        while (i < tensor_count) : (i += 1) {
            const name = try self.readString();
            const n_dims = try self.readU32();
            const shape = try self.allocator.alloc(u64, @intCast(n_dims));
            for (shape) |*dim| dim.* = try self.readU64();

            const dtype = try self.readU32();
            const offset = try self.readU64();
            const size = computeTensorByteSize(@enumFromInt(dtype), shape);

            try self.tensors.append(.{
                .name = name,
                .offset = offset,
                .size = size,
                .shape = shape,
                .dtype = @enumFromInt(dtype),
            });
        }

        self.data_offset = std.mem.alignForward(u64, self.cursor, 32);
    }
};

fn blockSizeBytes(dtype: GGMLType) u64 {
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

fn blockElements(dtype: GGMLType) u64 {
    return switch (dtype) {
        .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q8_1 => 32,
        .q2_k, .q3_k, .q4_k, .q5_k, .q6_k => 256,
        .q8_k => 32,
        .iq2_xxs, .iq2_xs, .iq2_s, .iq3_xxs, .iq1_s, .iq4_nl, .iq4_xs, .iq3_s, .iq1_m => 256,
        .tq1_0, .tq2_0, .mxfp4 => 32,
        else => 1,
    };
}

fn computeTensorByteSize(dtype: GGMLType, shape: []const u64) u64 {
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

fn detectMetadataFromTensors(analyzer: *GGUFAnalyzer) struct { vocab_size: u32, hidden_dim: u32, num_layers: u32 } {
    var vocab_size: u32 = 151936; // Default Qwen3.5
    var hidden_dim: u32 = 4096;
    var num_layers: u32 = 0;

    // Count layers
    for (analyzer.tensors.items) |t| {
        if (std.mem.startsWith(u8, t.name, "blk.")) {
            // Extract layer number
            if (std.mem.indexOf(u8, t.name[4..], ".")) |end| {
                const layer_str = t.name[4 .. 4 + end];
                if (std.fmt.parseInt(u32, layer_str, 10)) |layer| {
                    if (layer >= num_layers) num_layers = layer + 1;
                } else |_| {}
            }
        }
        if (std.mem.eql(u8, t.name, "token_embd.weight")) {
            // Shape is [vocab, hidden] or [hidden, vocab]
            if (t.shape.len >= 2) {
                hidden_dim = @intCast(t.shape[0]);
                vocab_size = @intCast(t.shape[1]);
            }
        }
    }

    return .{ .vocab_size = vocab_size, .hidden_dim = hidden_dim, .num_layers = num_layers };
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        std.debug.print("Usage: {s} <gguf_file>\n", .{args[0]});
        return;
    }

    const model_path = args[1];
    std.debug.print("\n=== GGUF Analiz: {s} ===\n\n", .{model_path});

    const file = try std.fs.cwd().openFile(model_path, .{});
    defer file.close();

    var analyzer = GGUFAnalyzer.init(file, allocator);
    defer analyzer.deinit();

    try analyzer.readHeader();

    const meta = detectMetadataFromTensors(&analyzer);

    // Default values for Qwen3.5 4B
    const hidden = meta.hidden_dim;
    const vocab = meta.vocab_size;
    const num_layers = meta.num_layers;
    const num_heads = 32;
    const num_kv_heads = 8;
    const head_dim = hidden / num_heads;
    const kv_dim = num_kv_heads * head_dim;
    const ffn_h = hidden * 4; // Estimate

    std.debug.print("--- Metadata (Tensor'lerden Tespit) ---\n", .{});
    std.debug.print("  vocab_size: {d}\n", .{vocab});
    std.debug.print("  hidden_dim: {d}\n", .{hidden});
    std.debug.print("  num_layers: {d}\n", .{num_layers});
    std.debug.print("  num_heads: {d} (tahmini)\n", .{num_heads});
    std.debug.print("  num_kv_heads: {d} (tahmini)\n", .{num_kv_heads});
    std.debug.print("  head_dim: {d} (tahmini)\n", .{head_dim});
    std.debug.print("  ffn_hidden_dim: {d} (tahmini)\n", .{ffn_h});

    // GGUF toplam boyut
    var total_gguf_bytes: u64 = 0;
    var q8_0_count: usize = 0;
    var f16_count: usize = 0;

    std.debug.print("\n--- Tensor'lar (ilk 30) ---\n", .{});
    for (analyzer.tensors.items, 0..) |t, i| {
        total_gguf_bytes += t.size;
        if (t.dtype == .q8_0) q8_0_count += 1 else if (t.dtype == .f16) f16_count += 1;

        if (i < 30) {
            std.debug.print("  {s}: shape=[", .{t.name});
            for (t.shape, 0..) |dim, j| {
                if (j > 0) std.debug.print(",", .{});
                std.debug.print("{d}", .{dim});
            }
            std.debug.print("], dtype={s}, size={d}\n", .{ @tagName(t.dtype), t.size });
        }
    }

    std.debug.print("\n--- Özet ---\n", .{});
    std.debug.print("  Toplam tensor sayısı: {d}\n", .{analyzer.tensors.items.len});
    std.debug.print("  Q8_0 tensor sayısı: {d}\n", .{q8_0_count});
    std.debug.print("  F16 tensor sayısı: {d}\n", .{f16_count});
    std.debug.print("  Toplam GGUF veri boyutu: {d} bytes ({d:.2} MB)\n", .{ total_gguf_bytes, @as(f64, @floatFromInt(total_gguf_bytes)) / (1024.0 * 1024.0) });

    // WeightLayout hesaplaması (FP16 varsayımı)
    const sizeof_fp16: usize = 2;

    // Token embedding
    const token_embedding_size = vocab * hidden * sizeof_fp16;

    // Per-layer size (attention)
    var per_layer: usize = 0;
    per_layer += hidden * sizeof_fp16; // norm
    const q_dim = num_heads * head_dim;
    per_layer += hidden * q_dim * sizeof_fp16; // q_proj
    per_layer += kv_dim * hidden * sizeof_fp16; // k_proj
    per_layer += kv_dim * hidden * sizeof_fp16; // v_proj
    per_layer += q_dim * hidden * sizeof_fp16; // o_proj
    per_layer += hidden * sizeof_fp16; // ffn_norm

    // FFN weights (gated_swi_glu assumed)
    per_layer += hidden * ffn_h * sizeof_fp16; // w1
    per_layer += hidden * ffn_h * sizeof_fp16; // w3
    per_layer += ffn_h * hidden * sizeof_fp16; // w2

    per_layer += head_dim * sizeof_fp16; // attn_q_norm
    per_layer += head_dim * sizeof_fp16; // attn_k_norm

    const final_norm = hidden * sizeof_fp16;
    const output_proj = hidden * vocab * sizeof_fp16;
    const total_layout = token_embedding_size + per_layer * num_layers + final_norm + output_proj;

    std.debug.print("\n--- WeightLayout Hesaplaması (FP16 varsayımı) ---\n", .{});
    std.debug.print("  token_embedding: {d} bytes ({d:.2} MB)\n", .{ token_embedding_size, @as(f64, @floatFromInt(token_embedding_size)) / (1024.0 * 1024.0) });
    std.debug.print("  per_layer_size: {d} bytes ({d:.2} MB)\n", .{ per_layer, @as(f64, @floatFromInt(per_layer)) / (1024.0 * 1024.0) });
    std.debug.print("  final_norm: {d} bytes\n", .{final_norm});
    std.debug.print("  output_proj: {d} bytes ({d:.2} MB)\n", .{ output_proj, @as(f64, @floatFromInt(output_proj)) / (1024.0 * 1024.0) });
    std.debug.print("  TOTAL (raw_pool_size): {d} bytes ({d:.2} MB)\n", .{ total_layout, @as(f64, @floatFromInt(total_layout)) / (1024.0 * 1024.0) });

    // Karşılaştırma
    std.debug.print("\n=== KARŞILAŞTIRMA ===\n", .{});
    const diff = @as(i64, @intCast(total_layout)) - @as(i64, @intCast(total_gguf_bytes));
    const ratio = @as(f64, @floatFromInt(total_layout)) / @as(f64, @floatFromInt(total_gguf_bytes));

    if (diff > 0) {
        std.debug.print("WeightLayout {d:.2}x BÜYÜK: +{d} bytes (+{d:.2} MB)\n", .{ ratio, diff, @as(f64, @floatFromInt(diff)) / (1024.0 * 1024.0) });
        std.debug.print("\nSONUÇ: FP16 layout hesaplanıyor ama quantized (Q8_0) veriler saklanıyor.\n", .{});
        std.debug.print("Allocation ~{d:.1}x fazla yer ayırıyor!\n", .{ratio});
    } else {
        std.debug.print("WeightLayout {d:.2}x KÜÇÜK: {d} bytes\n", .{ ratio, diff });
    }

    // Quantized boyut oranı
    std.debug.print("\n--- Quantization Analizi ---\n", .{});
    std.debug.print("  Q8_0: 34 bytes per 32 elements = {d:.3} bytes/elem\n", .{34.0 / 32.0});
    std.debug.print("  FP16: 2 bytes per element\n", .{});
    std.debug.print("  Teorik compression oranı: {d:.2}x\n", .{2.0 / (34.0 / 32.0)});
    std.debug.print("  Gerçek allocation/GGUF oranı: {d:.2}x\n", .{ratio});

    // Önemli tensor'ları kontrol et
    std.debug.print("\n--- Kritik Tensor Kontrolü ---\n", .{});
    for (analyzer.tensors.items) |t| {
        if (std.mem.endsWith(u8, t.name, "attn_q.weight")) {
            std.debug.print("  attn_q.weight: shape=[", .{});
            for (t.shape, 0..) |dim, j| {
                if (j > 0) std.debug.print(",", .{});
                std.debug.print("{d}", .{dim});
            }
            std.debug.print("], expected hidden×q_dim={d}×{d}={d} elements\n", .{ hidden, q_dim, hidden * q_dim });
        }
        if (std.mem.endsWith(u8, t.name, "ssm_conv1d.weight")) {
            std.debug.print("  ssm_conv1d.weight: shape=[", .{});
            for (t.shape, 0..) |dim, j| {
                if (j > 0) std.debug.print(",", .{});
                std.debug.print("{d}", .{dim});
            }
            std.debug.print("]\n", .{});
        }
    }
}
