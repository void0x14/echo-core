const std = @import("std");
const builtin = @import("builtin");
const config_mod = @import("../core/config.zig");

const Allocator = std.mem.Allocator;
const ArrayList = std.array_list.Managed;

const GGUF_MAGIC: u32 = 0x46554747;
const GGUF_VERSION: u32 = 3;

const GGUF_VAL_UINT8: u32 = 0;
const GGUF_VAL_INT8: u32 = 1;
const GGUF_VAL_UINT16: u32 = 2;
const GGUF_VAL_INT16: u32 = 3;
const GGUF_VAL_UINT32: u32 = 4;
const GGUF_VAL_INT32: u32 = 5;
const GGUF_VAL_FLOAT32: u32 = 6;
const GGUF_VAL_BOOL: u32 = 7;
const GGUF_VAL_STRING: u32 = 8;
const GGUF_VAL_ARRAY: u32 = 9;
const GGUF_VAL_UINT64: u32 = 10;
const GGUF_VAL_INT64: u32 = 11;
const GGUF_VAL_FLOAT64: u32 = 12;

const GGML_QUANT_BLOCK_SIZE: u64 = 32;

pub const GGMLType = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    // q4_2 = 4,
    // q4_3 = 5,
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

pub const TensorInfo = struct {
    offset: u64,
    size: u64,
    shape: []const u64,
    dtype: GGMLType,
};

pub const GGUFValue = union(enum) {
    uint: u64,
    int: i64,
    float: f64,
    bool: bool,
    string: []const u8,
    uints: []const u64,
    ints: []const i64,
    floats: []const f64,
    bools: []const bool,
    strings: []const []const u8,
};

pub const Reader = struct {
    file: std.Io.File,
    allocator: Allocator,
    cursor: u64,
    data_offset: u64,
    model_prefix: []const u8,
    metadata: std.StringHashMap(GGUFValue),
    tokens: ArrayList([]const u8),
    tensors: std.StringHashMap(TensorInfo),
    config: config_mod.ModelConfig,
    alignment: u32,

    fn runtimeIo() std.Io {
        return std.Io.Threaded.global_single_threaded.io();
    }

    pub fn open(path: []const u8) !Reader {
        return openWithAllocator(path, std.heap.page_allocator);
    }

    pub fn openWithAllocator(path: []const u8, allocator: Allocator) !Reader {
        const io = runtimeIo();
        const file = if (path.len > 0 and path[0] == '/')
            try std.Io.Dir.openFileAbsolute(io, path, .{})
        else
            try std.Io.Dir.cwd().openFile(io, path, .{});
        errdefer file.close(io);

        var reader = Reader{
            .file = file,
            .allocator = allocator,
            .cursor = 0,
            .data_offset = 0,
            .model_prefix = &.{},
            .metadata = std.StringHashMap(GGUFValue).init(allocator),
            .tokens = ArrayList([]const u8).init(allocator),
            .tensors = std.StringHashMap(TensorInfo).init(allocator),
            .config = std.mem.zeroes(config_mod.ModelConfig),
            .alignment = 32,
        };
        errdefer reader.deinit();

        try reader.readHeader();
        return reader;
    }

    fn readHeader(self: *Reader) !void {
        const magic = try self.readU32();
        if (magic != GGUF_MAGIC) return error.InvalidGGUFMagic;

        const version = try self.readU32();
        if (version != GGUF_VERSION) return error.UnsupportedGGUFVersion;

        const tensor_count = try self.readU64();
        const metadata_kv_count = try self.readU64();

        var i: u64 = 0;
        while (i < metadata_kv_count) : (i += 1) {
            const key = try self.readString();
            errdefer self.allocator.free(key);

            const type_id = try self.readU32();
            const value = try self.readValue(type_id);
            try self.metadata.put(key, value);

            if (std.mem.eql(u8, key, "general.alignment")) {
                if (self.numericAsU64(value)) |alignment| {
                    self.alignment = @intCast(alignment);
                }
            }
        }

        try self.detectModelPrefix();
        try self.populateConfig();
        try self.populateTokens();

        i = 0;
        while (i < tensor_count) : (i += 1) {
            const name = try self.readString();
            errdefer self.allocator.free(name);

            const n_dims = try self.readU32();
            const shape = try self.allocator.alloc(u64, @intCast(n_dims));
            errdefer self.allocator.free(shape);
            for (shape) |*dim| dim.* = try self.readU64();

            const dtype = try self.readGGMLType();
            const offset = try self.readU64();
            const size = computeTensorByteSize(dtype, shape);

            try self.tensors.put(name, .{
                .offset = offset,
                .size = size,
                .shape = shape,
                .dtype = dtype,
            });
        }

        self.data_offset = std.mem.alignForward(u64, self.cursor, 32);
    }

    fn numericAsU64(_: *const Reader, value: GGUFValue) ?u64 {
        return switch (value) {
            .uint => |v| v,
            .int => |v| if (v >= 0) @intCast(v) else null,
            else => null,
        };
    }

    fn prefixedLookup(self: *const Reader, suffix: []const u8) ?GGUFValue {
        var key_buf: [128]u8 = undefined;
        if (self.model_prefix.len + suffix.len <= key_buf.len) {
            @memcpy(key_buf[0..self.model_prefix.len], self.model_prefix);
            @memcpy(key_buf[self.model_prefix.len .. self.model_prefix.len + suffix.len], suffix);
            if (self.metadata.get(key_buf[0 .. self.model_prefix.len + suffix.len])) |value| {
                return value;
            }
        }
        return self.metadata.get(suffix);
    }

    fn detectModelPrefix(self: *Reader) !void {
        const suffixes = [_][]const u8{
            ".context_length",
            ".embedding_length",
            ".block_count",
            ".feed_forward_length",
            ".attention.head_count",
            ".attention.head_count_kv",
        };

        var it = self.metadata.iterator();
        while (it.next()) |entry| {
            const key = entry.key_ptr.*;
            for (suffixes) |suffix| {
                if (key.len > suffix.len and std.mem.endsWith(u8, key, suffix)) {
                    self.model_prefix = try self.allocator.dupe(u8, key[0 .. key.len - suffix.len + 1]);
                    return;
                }
            }
        }
    }

    fn populateConfig(self: *Reader) !void {
        self.config = std.mem.zeroes(config_mod.ModelConfig);

        if (self.prefixedLookup("context_length")) |value| {
            if (self.numericAsU64(value)) |v| self.config.max_seq_len = @intCast(v);
        }
        if (self.prefixedLookup("embedding_length")) |value| {
            if (self.numericAsU64(value)) |v| self.config.hidden_dim = @intCast(v);
        }
        if (self.prefixedLookup("block_count")) |value| {
            if (self.numericAsU64(value)) |v| self.config.num_layers = @intCast(v);
        }
        if (self.prefixedLookup("feed_forward_length")) |value| {
            if (self.numericAsU64(value)) |v| self.config.ffn_hidden_dim = @intCast(v);
        }
        if (self.prefixedLookup("attention.head_count")) |value| {
            if (self.numericAsU64(value)) |v| self.config.num_heads = @intCast(v);
        }
        if (self.prefixedLookup("attention.head_count_kv")) |value| {
            if (self.numericAsU64(value)) |v| self.config.num_kv_heads = @intCast(v);
        }

        if (self.config.num_kv_heads == 0) self.config.num_kv_heads = self.config.num_heads;
        if (self.config.num_heads != 0) self.config.head_dim = self.config.hidden_dim / self.config.num_heads;

        // Models with GQA (like Qwen3, Llama 3) often specify an explicit key_length (head_dim)
        // that isn't just hidden_dim / num_heads.
        if (self.prefixedLookup("attention.key_length")) |value| {
            if (self.numericAsU64(value)) |v| self.config.head_dim = @intCast(v);
        }

        if (self.prefixedLookup("vocab_size")) |value| {
            if (self.numericAsU64(value)) |v| self.config.vocab_size = @intCast(v);
        }

        self.config.ffn_type = if (self.config.ffn_hidden_dim > 0) .gated_swi_glu else .dense;
        self.config.norm_type = .rms_norm;
        self.config.pos_encoding = .rope;
        self.config.use_kv_quantization = false;

        // Cap max_seq_len to prevent OOM from models with huge context_length
        if (self.config.max_seq_len > 4096) self.config.max_seq_len = 4096;
    }

    fn populateTokens(self: *Reader) !void {
        if (self.metadata.get("tokenizer.ggml.tokens")) |value| {
            switch (value) {
                .strings => |items| {
                    try self.tokens.appendSlice(items);
                    if (self.config.vocab_size == 0) self.config.vocab_size = @intCast(items.len);
                },
                else => {},
            }
        }
    }

    fn readExact(self: *Reader, buf: []u8) !void {
        if (buf.len == 0) return;
        const n = try self.file.readPositionalAll(runtimeIo(), buf, self.cursor);
        if (n != buf.len) return error.EndOfStream;
        self.cursor += n;
    }

    fn readU8(self: *Reader) !u8 {
        var buf: [1]u8 = undefined;
        try self.readExact(&buf);
        return buf[0];
    }

    fn readU16(self: *Reader) !u16 {
        var buf: [2]u8 = undefined;
        try self.readExact(&buf);
        return std.mem.readInt(u16, &buf, .little);
    }

    fn readU32(self: *Reader) !u32 {
        var buf: [4]u8 = undefined;
        try self.readExact(&buf);
        return std.mem.readInt(u32, &buf, .little);
    }

    fn readI32(self: *Reader) !i32 {
        var buf: [4]u8 = undefined;
        try self.readExact(&buf);
        return std.mem.readInt(i32, &buf, .little);
    }

    fn readU64(self: *Reader) !u64 {
        var buf: [8]u8 = undefined;
        try self.readExact(&buf);
        return std.mem.readInt(u64, &buf, .little);
    }

    fn readI64(self: *Reader) !i64 {
        var buf: [8]u8 = undefined;
        try self.readExact(&buf);
        return std.mem.readInt(i64, &buf, .little);
    }

    fn readF32(self: *Reader) !f32 {
        return @bitCast(try self.readU32());
    }

    fn readF64(self: *Reader) !f64 {
        return @bitCast(try self.readU64());
    }

    fn readString(self: *Reader) ![]const u8 {
        const len = try self.readU64();
        const buf = try self.allocator.alloc(u8, @intCast(len));
        errdefer self.allocator.free(buf);
        try self.readExact(buf);
        return buf;
    }

    fn readGGMLType(self: *Reader) !GGMLType {
        const raw = try self.readU32();
        if (raw >= @intFromEnum(GGMLType.count)) return error.UnknownGGMLType;
        return @enumFromInt(raw);
    }

    fn readValue(self: *Reader, type_id: u32) !GGUFValue {
        return switch (type_id) {
            GGUF_VAL_UINT8 => .{ .uint = try self.readU8() },
            GGUF_VAL_INT8 => .{ .int = @as(i8, @bitCast(try self.readU8())) },
            GGUF_VAL_UINT16 => .{ .uint = try self.readU16() },
            GGUF_VAL_INT16 => .{ .int = @as(i16, @bitCast(try self.readU16())) },
            GGUF_VAL_UINT32 => .{ .uint = try self.readU32() },
            GGUF_VAL_INT32 => .{ .int = try self.readI32() },
            GGUF_VAL_FLOAT32 => .{ .float = try self.readF32() },
            GGUF_VAL_BOOL => .{ .bool = (try self.readU8()) != 0 },
            GGUF_VAL_STRING => .{ .string = try self.readString() },
            GGUF_VAL_UINT64 => .{ .uint = try self.readU64() },
            GGUF_VAL_INT64 => .{ .int = try self.readI64() },
            GGUF_VAL_FLOAT64 => .{ .float = try self.readF64() },
            GGUF_VAL_ARRAY => {
                const elem_type = try self.readU32();
                const n_elems = try self.readU64();
                return switch (elem_type) {
                    GGUF_VAL_UINT8, GGUF_VAL_UINT16, GGUF_VAL_UINT32, GGUF_VAL_UINT64 => blk: {
                        const items = try self.allocator.alloc(u64, @intCast(n_elems));
                        errdefer self.allocator.free(items);
                        for (items) |*item| {
                            item.* = switch (elem_type) {
                                GGUF_VAL_UINT8 => try self.readU8(),
                                GGUF_VAL_UINT16 => try self.readU16(),
                                GGUF_VAL_UINT32 => try self.readU32(),
                                GGUF_VAL_UINT64 => try self.readU64(),
                                else => unreachable,
                            };
                        }
                        break :blk .{ .uints = items };
                    },
                    GGUF_VAL_INT8, GGUF_VAL_INT16, GGUF_VAL_INT32, GGUF_VAL_INT64 => blk: {
                        const items = try self.allocator.alloc(i64, @intCast(n_elems));
                        errdefer self.allocator.free(items);
                        for (items) |*item| {
                            item.* = switch (elem_type) {
                                GGUF_VAL_INT8 => @as(i8, @bitCast(try self.readU8())),
                                GGUF_VAL_INT16 => @as(i16, @bitCast(try self.readU16())),
                                GGUF_VAL_INT32 => try self.readI32(),
                                GGUF_VAL_INT64 => try self.readI64(),
                                else => unreachable,
                            };
                        }
                        break :blk .{ .ints = items };
                    },
                    GGUF_VAL_FLOAT32, GGUF_VAL_FLOAT64 => blk: {
                        const items = try self.allocator.alloc(f64, @intCast(n_elems));
                        errdefer self.allocator.free(items);
                        for (items) |*item| {
                            item.* = switch (elem_type) {
                                GGUF_VAL_FLOAT32 => try self.readF32(),
                                GGUF_VAL_FLOAT64 => try self.readF64(),
                                else => unreachable,
                            };
                        }
                        break :blk .{ .floats = items };
                    },
                    GGUF_VAL_BOOL => blk: {
                        const items = try self.allocator.alloc(bool, @intCast(n_elems));
                        errdefer self.allocator.free(items);
                        for (items) |*item| item.* = (try self.readU8()) != 0;
                        break :blk .{ .bools = items };
                    },
                    GGUF_VAL_STRING => blk: {
                        const items = try self.allocator.alloc([]const u8, @intCast(n_elems));
                        errdefer self.allocator.free(items);
                        var filled: usize = 0;
                        errdefer {
                            for (items[0..filled]) |s| self.allocator.free(s);
                        }
                        for (items) |*item| {
                            item.* = try self.readString();
                            filled += 1;
                        }
                        break :blk .{ .strings = items };
                    },
                    else => error.UnknownGGUFValueType,
                };
            },
            else => error.UnknownGGUFValueType,
        };
    }

    pub fn loadTensor(self: *const Reader, name: []const u8) ![]u8 {
        const info = self.tensors.get(name) orelse return error.TensorNotFound;
        const data = try self.allocator.alloc(u8, @intCast(info.size));
        errdefer self.allocator.free(data);

        const n = try self.file.readPositionalAll(runtimeIo(), data, self.data_offset + info.offset);
        if (n != data.len) return error.EndOfStream;
        return data;
    }

    pub fn findTensorBySuffix(self: *const Reader, suffix: []const u8) ?TensorInfo {
        var it = self.tensors.iterator();
        while (it.next()) |entry| {
            const name = entry.key_ptr.*;
            if (std.mem.eql(u8, name, suffix)) return entry.value_ptr.*;
            if (name.len > suffix.len + 1 and name[name.len - suffix.len - 1] == '.' and std.mem.endsWith(u8, name, suffix)) {
                return entry.value_ptr.*;
            }
        }
        return null;
    }

    pub fn getAlignment(self: *const Reader) u32 {
        return self.alignment;
    }

    pub fn deinit(self: *Reader) void {
        var tensor_it = self.tensors.iterator();
        while (tensor_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.shape);
        }
        self.tensors.deinit();

        self.tokens.deinit();

        var metadata_it = self.metadata.iterator();
        while (metadata_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.freeValue(entry.value_ptr.*);
        }
        self.metadata.deinit();

        if (self.model_prefix.len > 0) self.allocator.free(self.model_prefix);
        self.file.close(runtimeIo());
        self.* = undefined;
    }

    fn freeValue(self: *Reader, value: GGUFValue) void {
        switch (value) {
            .string => |s| self.allocator.free(s),
            .uints => |items| self.allocator.free(items),
            .ints => |items| self.allocator.free(items),
            .floats => |items| self.allocator.free(items),
            .bools => |items| self.allocator.free(items),
            .strings => |items| {
                for (items) |s| self.allocator.free(s);
                self.allocator.free(items);
            },
            else => {},
        }
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

fn appendU8(list: *ArrayList(u8), value: u8) !void {
    try list.append(value);
}

fn appendU16LE(list: *ArrayList(u8), value: u16) !void {
    try list.append(@truncate(value));
    try list.append(@truncate(value >> 8));
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
    try appendU32LE(list, @bitCast(value));
}

fn appendF32LE(list: *ArrayList(u8), value: f32) !void {
    try appendU32LE(list, @bitCast(value));
}

fn appendString(list: *ArrayList(u8), value: []const u8) !void {
    try appendU64LE(list, value.len);
    try list.appendSlice(value);
}

pub fn buildSyntheticGgufForTests(allocator: Allocator) ![]u8 {
    var bytes = ArrayList(u8).init(allocator);
    errdefer bytes.deinit();

    try appendU32LE(&bytes, GGUF_MAGIC);
    try appendU32LE(&bytes, GGUF_VERSION);
    try appendU64LE(&bytes, 1); // tensor count
    try appendU64LE(&bytes, 9); // metadata count

    try appendString(&bytes, "general.alignment");
    try appendU32LE(&bytes, GGUF_VAL_UINT32);
    try appendU32LE(&bytes, 64);

    try appendString(&bytes, "llama.context_length");
    try appendU32LE(&bytes, GGUF_VAL_UINT32);
    try appendU32LE(&bytes, 128);

    try appendString(&bytes, "llama.embedding_length");
    try appendU32LE(&bytes, GGUF_VAL_UINT32);
    try appendU32LE(&bytes, 8);

    try appendString(&bytes, "llama.block_count");
    try appendU32LE(&bytes, GGUF_VAL_UINT32);
    try appendU32LE(&bytes, 1);

    try appendString(&bytes, "llama.feed_forward_length");
    try appendU32LE(&bytes, GGUF_VAL_UINT32);
    try appendU32LE(&bytes, 16);

    try appendString(&bytes, "llama.attention.head_count");
    try appendU32LE(&bytes, GGUF_VAL_UINT32);
    try appendU32LE(&bytes, 2);

    try appendString(&bytes, "llama.attention.head_count_kv");
    try appendU32LE(&bytes, GGUF_VAL_UINT32);
    try appendU32LE(&bytes, 1);

    try appendString(&bytes, "tokenizer.ggml.tokens");
    try appendU32LE(&bytes, GGUF_VAL_ARRAY);
    try appendU32LE(&bytes, GGUF_VAL_STRING);
    try appendU64LE(&bytes, 2);
    try appendString(&bytes, "hello");
    try appendString(&bytes, "world");

    try appendString(&bytes, "tokenizer.ggml.scores");
    try appendU32LE(&bytes, GGUF_VAL_ARRAY);
    try appendU32LE(&bytes, GGUF_VAL_FLOAT32);
    try appendU64LE(&bytes, 2);
    try appendF32LE(&bytes, 1.25);
    try appendF32LE(&bytes, -0.5);

    try appendString(&bytes, "blk.0.attn_q.weight");
    try appendU32LE(&bytes, 2);
    try appendU64LE(&bytes, 8);
    try appendU64LE(&bytes, 8);
    try appendU32LE(&bytes, @intFromEnum(GGMLType.f16));
    try appendU64LE(&bytes, 0);

    const aligned_len = std.mem.alignForward(usize, bytes.items.len, 32);
    try bytes.ensureTotalCapacity(aligned_len + 128);
    while (bytes.items.len < aligned_len) try bytes.append(0);
    for (0..128) |_| try appendU8(&bytes, 0xAB);

    return try bytes.toOwnedSlice();
}

test "GGMLType count" {
    try std.testing.expectEqual(@as(u32, 40), @intFromEnum(GGMLType.count));
}

test "Reader parses synthetic GGUF v3 file" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;

    const data = try buildSyntheticGgufForTests(std.testing.allocator);
    defer std.testing.allocator.free(data);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "sample.gguf", .data = data });

    var path_buf: [256]u8 = undefined;
    const rel_path = try std.fmt.bufPrint(&path_buf, ".zig-cache/tmp/{s}/sample.gguf", .{tmp.sub_path[0..]});

    var reader = try Reader.openWithAllocator(rel_path, std.testing.allocator);
    defer reader.deinit();

    try std.testing.expectEqual(@as(u32, 64), reader.getAlignment());
    try std.testing.expectEqual(@as(u32, 128), reader.config.max_seq_len);
    try std.testing.expectEqual(@as(u32, 8), reader.config.hidden_dim);
    try std.testing.expectEqual(@as(u32, 1), reader.config.num_layers);
    try std.testing.expectEqual(@as(u32, 2), reader.config.num_heads);
    try std.testing.expectEqual(@as(u32, 1), reader.config.num_kv_heads);
    try std.testing.expectEqual(@as(u32, 4), reader.config.head_dim);
    try std.testing.expectEqual(@as(usize, 2), reader.tokens.items.len);
    try std.testing.expectEqualStrings("hello", reader.tokens.items[0]);
    try std.testing.expectEqualStrings("world", reader.tokens.items[1]);
    try std.testing.expectEqual(@as(u64, @intCast(data.len - 128)), reader.data_offset);

    const exact = reader.findTensorBySuffix("blk.0.attn_q.weight");
    try std.testing.expect(exact != null);
    const suffix = reader.findTensorBySuffix("attn_q.weight");
    try std.testing.expect(suffix != null);
    try std.testing.expectEqual(@as(u64, 128), suffix.?.size);

    const tensor = try reader.loadTensor("blk.0.attn_q.weight");
    defer std.testing.allocator.free(tensor);
    try std.testing.expectEqual(@as(usize, 128), tensor.len);
    try std.testing.expectEqual(@as(u8, 0xAB), tensor[0]);

    const scores = reader.metadata.get("tokenizer.ggml.scores") orelse unreachable;
    switch (scores) {
        .floats => |vals| {
            try std.testing.expectEqual(@as(usize, 2), vals.len);
            try std.testing.expectApproxEqAbs(@as(f64, 1.25), vals[0], 0.0001);
            try std.testing.expectApproxEqAbs(@as(f64, -0.5), vals[1], 0.0001);
        },
        else => return error.TestUnexpectedResult,
    }
}
