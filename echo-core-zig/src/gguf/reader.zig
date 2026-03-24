const std = @import("std");

pub const GGMLType = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 4,
    q5_1 = 5,
    q8_0 = 6,
    q8_1 = 7,
    q2_k = 8,
    q3_k = 9,
    q4_k = 10,
    q5_k = 11,
    q6_k = 12,
    iq2_xxs = 13,
    iq2_xs = 14,
    i16 = 15,
    f64 = 16,
    iq1_s = 17,
    iq4_nl = 18,
    iq4_xs = 19,
    i8 = 20,
    i32 = 21,
    iq2_s = 22,
    iq3_xxs = 23,
    bf16 = 24,
    q4_0_4_4 = 25,
    q4_0_4_8 = 26,
    q4_0_8_8 = 27,
    count = 28,
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
    fd: std.fs.File,
    data_offset: u64,
    model_prefix: []const u8,
    metadata: std.StringHashMap(GGUFValue),
    tokens: std.ArrayList([]const u8),
    tensors: std.StringHashMap(TensorInfo),
    alignment: u32 = 32,

    pub fn open(path: []const u8) !Reader {
        const fd = try std.fs.openFileAbsolute(path, .{ .mode = .read_only });
        errdefer fd.close();

        var reader = Reader{
            .fd = fd,
            .data_offset = 0,
            .model_prefix = &.{},
            .metadata = std.StringHashMap(GGUFValue).init(fd.handle),
            .tokens = std.ArrayList([]const u8).init(std.heap.page_allocator),
            .tensors = std.StringHashMap(TensorInfo).init(fd.handle),
        };
        try reader.readHeader();
        return reader;
    }

    fn readHeader(self: *Reader) !void {
        const magic = try self.readU32();
        if (magic != 0x46554747) {
            return error.InvalidGGUFMagic;
        }
        const version = try self.readU32();
        if (version != 3) {
            return error.UnsupportedGGUFVersion;
        }
        const n_tensors = try self.readU64();
        const n_kv = try self.readU64();

        self.data_offset = @intCast(try self.fd.getPos());

        var i: u64 = 0;
        while (i < n_kv) : (i += 1) {
            const key = try self.readString();
            const type_id = try self.readU32();
            const value = try self.readValue(type_id);
            try self.metadata.put(key, value);
        }

        i = 0;
        while (i < n_tensors) : (i += 1) {
            const name = try self.readString();
            const n_dims = try self.readU32();
            var shape: [16]u64 = undefined;
            var j: u32 = 0;
            while (j < n_dims) : (j += 1) {
                shape[j] = try self.readU64();
            }
            const dtype_val = try self.readU32();
            const dtype: GGMLType = @enumFromInt(dtype_val);
            const offset = try self.readU64();
            const size = try self.readU64();

            try self.tensors.put(name, .{
                .offset = offset,
                .size = size,
                .shape = self.tensors.allocator.dupe(u64, shape[0..n_dims]),
                .dtype = dtype,
            });
        }
    }

    fn readU8(self: *Reader) !u8 {
        var buf: [1]u8 = undefined;
        try self.fd.read(&buf);
        return buf[0];
    }

    fn readU32(self: *Reader) !u32 {
        var buf: [4]u8 = undefined;
        try self.fd.read(&buf);
        return std.mem.readInt(u32, &buf, .little);
    }

    fn readI32(self: *Reader) !i32 {
        var buf: [4]u8 = undefined;
        try self.fd.read(&buf);
        return std.mem.readInt(i32, &buf, .little);
    }

    fn readU64(self: *Reader) !u64 {
        var buf: [8]u8 = undefined;
        try self.fd.read(&buf);
        return std.mem.readInt(u64, &buf, .little);
    }

    fn readI64(self: *Reader) !i64 {
        var buf: [8]u8 = undefined;
        try self.fd.read(&buf);
        return std.mem.readInt(i64, &buf, .little);
    }

    fn readF32(self: *Reader) !f32 {
        var buf: [4]u8 = undefined;
        try self.fd.read(&buf);
        return std.mem.readInt(u32, &buf, .little);
    }

    fn readF64(self: *Reader) !f64 {
        var buf: [8]u8 = undefined;
        try self.fd.read(&buf);
        return std.mem.readInt(u64, &buf, .little);
    }

    fn readString(self: *Reader) ![]const u8 {
        const len = try self.readU64();
        const buf = try self.tensors.allocator.alloc(u8, @intCast(len));
        try self.fd.read(buf);
        return buf;
    }

    fn readValue(self: *Reader, type_id: u32) !GGUFValue {
        return switch (type_id) {
            0 => .{ .uint = try self.readU64() },
            1 => .{ .int = try self.readI64() },
            2 => .{ .float = @bitCast(try self.readF32()) },
            3 => .{ .float = try self.readF64() },
            4 => .{ .bool = (try self.readU8()) != 0 },
            5 => .{ .string = try self.readString() },
            6 => {
                const n = try self.readU64();
                const arr = try self.tensors.allocator.alloc(u64, @intCast(n));
                try self.fd.read(arr);
                return .{ .uints = arr };
            },
            7 => {
                const n = try self.readU64();
                const arr = try self.tensors.allocator.alloc(i64, @intCast(n));
                try self.fd.read(arr);
                return .{ .ints = arr };
            },
            8 => {
                const n = try self.readU64();
                var arr = try self.tensors.allocator.alloc(f64, @intCast(n));
                for (arr) |*v| v.* = @bitCast(try self.readU64());
                return .{ .floats = arr };
            },
            9 => {
                const n = try self.readU64();
                const arr = try self.tensors.allocator.alloc(bool, @intCast(n));
                for (arr) |*v| v.* = (try self.readU8()) != 0;
                return .{ .bools = arr };
            },
            10 => {
                const n = try self.readU64();
                var arr = try self.tensors.allocator.alloc([]const u8, @intCast(n));
                for (arr) |*s| s.* = try self.readString();
                return .{ .strings = arr };
            },
            else => error.UnknownGGUFValueType,
        };
    }

    pub fn loadTensor(self: *Reader, name: []const u8) ![]u8 {
        const info = self.tensors.get(name) orelse return error.TensorNotFound;
        const data_offset = self.data_offset + info.offset;
        try self.fd.seek(data_offset);
        const data = try self.tensors.allocator.alloc(u8, @intCast(info.size));
        try self.fd.read(data);
        return data;
    }

    pub fn findTensorBySuffix(self: *Reader, suffix: []const u8) ?TensorInfo {
        for (self.tensors.keys()) |key| {
            if (key.len > suffix.len and std.mem.endsWith(u8, key, suffix)) {
                return self.tensors.get(key);
            }
        }
        return null;
    }

    pub fn getAlignment(self: *Reader) u32 {
        if (self.metadata.get("general.alignment")) |val| {
            if (val == .uint) return @intCast(val.uint);
        }
        return 32;
    }

    pub fn deinit(self: *Reader) void {
        self.fd.close();
        self.metadata.deinit();
        self.tensors.deinit();
        for (self.tokens.items) |t| {
            self.tokens.allocator.free(t);
        }
        self.tokens.deinit();
    }
};

test "GGMLType count" {
    try std.testing.expectEqual(@intFromEnum(GGMLType.count), 28);
}
