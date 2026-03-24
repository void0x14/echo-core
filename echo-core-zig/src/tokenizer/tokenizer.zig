const std = @import("std");
const builtin = @import("builtin");
const gguf = @import("../gguf/reader.zig");

const Allocator = std.mem.Allocator;
const ArrayList = std.array_list.Managed;

pub const TokenData = struct {
    text: []const u8,
    score: f32,
    type: i32,
};

const SortedToken = struct {
    text: []const u8,
    id: u32,
};

pub const SimpleTokenizer = struct {
    allocator: Allocator,
    id_to_token: ArrayList(TokenData),
    token_to_id: std.StringHashMap(u32),
    sorted_tokens: ArrayList(SortedToken),
    bos_id: u32,
    eos_id: u32,
    tokenizer_type: []const u8,

    pub fn init(allocator: Allocator) SimpleTokenizer {
        return .{
            .allocator = allocator,
            .id_to_token = ArrayList(TokenData).init(allocator),
            .token_to_id = std.StringHashMap(u32).init(allocator),
            .sorted_tokens = ArrayList(SortedToken).init(allocator),
            .bos_id = 0,
            .eos_id = 0,
            .tokenizer_type = "spm",
        };
    }

    pub fn initFromReader(reader: *const gguf.Reader, allocator: Allocator) !SimpleTokenizer {
        var tokenizer = SimpleTokenizer.init(allocator);
        errdefer tokenizer.deinit();

        if (reader.metadata.get("tokenizer.ggml.model")) |value| {
            switch (value) {
                .string => |model| {
                    if (std.mem.eql(u8, model, "llama") or std.mem.eql(u8, model, "sp")) {
                        tokenizer.tokenizer_type = "spm";
                    } else if (std.mem.eql(u8, model, "gpt2") or std.mem.eql(u8, model, "qwen2") or std.mem.eql(u8, model, "qwen")) {
                        tokenizer.tokenizer_type = "bpe";
                    }
                },
                else => {},
            }
        }

        if (reader.tokens.items.len == 0) return error.MissingTokenizerTokens;

        try tokenizer.id_to_token.ensureTotalCapacity(reader.tokens.items.len);
        for (reader.tokens.items, 0..) |token_text, i| {
            const owned = try allocator.dupe(u8, token_text);
            errdefer allocator.free(owned);
            try tokenizer.id_to_token.append(.{
                .text = owned,
                .score = 0,
                .type = 1,
            });
            try tokenizer.token_to_id.put(owned, @intCast(i));
        }

        if (reader.metadata.get("tokenizer.ggml.scores")) |value| {
            switch (value) {
                .floats => |scores| {
                    if (scores.len != tokenizer.id_to_token.items.len) return error.InvalidTokenizerScores;
                    for (scores, 0..) |score, i| {
                        tokenizer.id_to_token.items[i].score = @floatCast(score);
                    }
                },
                else => {},
            }
        }

        if (reader.metadata.get("tokenizer.ggml.token_type")) |value| {
            switch (value) {
                .ints => |types| {
                    if (types.len != tokenizer.id_to_token.items.len) return error.InvalidTokenizerTypes;
                    for (types, 0..) |token_type, i| {
                        tokenizer.id_to_token.items[i].type = @intCast(token_type);
                    }
                },
                else => {},
            }
        }

        if (reader.metadata.get("tokenizer.ggml.bos_token_id")) |value| {
            if (numericAsU32(value)) |id| tokenizer.bos_id = id;
        }
        if (reader.metadata.get("tokenizer.ggml.eos_token_id")) |value| {
            if (numericAsU32(value)) |id| tokenizer.eos_id = id;
        }

        try tokenizer.rebuildSortedTokens();
        return tokenizer;
    }

    pub fn deinit(self: *SimpleTokenizer) void {
        self.sorted_tokens.deinit();
        self.token_to_id.deinit();
        for (self.id_to_token.items) |token| self.allocator.free(token.text);
        self.id_to_token.deinit();
        self.* = undefined;
    }

    fn numericAsU32(value: gguf.GGUFValue) ?u32 {
        return switch (value) {
            .uint => |v| @intCast(v),
            .int => |v| if (v >= 0) @intCast(v) else null,
            else => null,
        };
    }

    fn rebuildSortedTokens(self: *SimpleTokenizer) !void {
        self.sorted_tokens.deinit();
        self.sorted_tokens = ArrayList(SortedToken).init(self.allocator);
        try self.sorted_tokens.ensureTotalCapacity(self.id_to_token.items.len);

        for (self.id_to_token.items, 0..) |token, i| {
            try self.sorted_tokens.append(.{ .text = token.text, .id = @intCast(i) });
        }

        std.sort.heap(SortedToken, self.sorted_tokens.items, {}, struct {
            fn lessThan(_: void, lhs: SortedToken, rhs: SortedToken) bool {
                return lhs.text.len > rhs.text.len;
            }
        }.lessThan);
    }

    pub fn encode(self: *const SimpleTokenizer, text: []const u8) !ArrayList(u32) {
        var result = ArrayList(u32).init(self.allocator);
        errdefer result.deinit();

        try result.append(self.bos_id);
        if (text.len == 0) return result;

        var normalized = ArrayList(u8).init(self.allocator);
        defer normalized.deinit();

        if (std.mem.eql(u8, self.tokenizer_type, "spm")) {
            try normalized.appendSlice("\xE2\x96\x81");
            for (text) |c| {
                if (c == ' ') {
                    try normalized.appendSlice("\xE2\x96\x81");
                } else {
                    try normalized.append(c);
                }
            }
        } else {
            try normalized.appendSlice(text);
        }

        var pos: usize = 0;
        while (pos < normalized.items.len) {
            var matched = false;
            for (self.sorted_tokens.items) |token| {
                if (token.text.len == 0) continue;
                if (pos + token.text.len > normalized.items.len) continue;
                if (std.mem.eql(u8, normalized.items[pos .. pos + token.text.len], token.text)) {
                    try result.append(token.id);
                    pos += token.text.len;
                    matched = true;
                    break;
                }
            }

            if (!matched) {
                const byte_val = normalized.items[pos];
                var byte_buf: [6]u8 = undefined;
                const byte_token = if (std.mem.eql(u8, self.tokenizer_type, "bpe"))
                    gpt2ByteToUnicode(byte_val, byte_buf[0..3])
                else
                    spmByteToken(byte_val, &byte_buf);

                const token_id = self.token_to_id.get(byte_token) orelse return error.UnknownToken;
                try result.append(token_id);
                pos += 1;
            }
        }

        return result;
    }

    pub fn decode(self: *const SimpleTokenizer, ids: []const u32) ![]u8 {
        var result = ArrayList(u8).init(self.allocator);
        errdefer result.deinit();

        for (ids) |id| {
            if (id == self.bos_id or id == self.eos_id) continue;
            if (id >= self.id_to_token.items.len) continue;

            const token = self.id_to_token.items[id];
            if (token.type == 3) continue;

            if (std.mem.eql(u8, self.tokenizer_type, "bpe")) {
                if (gpt2UnicodeToByte(token.text)) |byte_val| {
                    try result.append(byte_val);
                } else {
                    try result.appendSlice(token.text);
                }
            } else {
                try result.appendSlice(token.text);
            }
        }

        if (!std.mem.eql(u8, self.tokenizer_type, "spm")) {
            return try result.toOwnedSlice();
        }

        var cleaned = ArrayList(u8).init(self.allocator);
        errdefer cleaned.deinit();

        var i: usize = 0;
        while (i < result.items.len) {
            if (i + 2 < result.items.len and
                result.items[i] == 0xE2 and
                result.items[i + 1] == 0x96 and
                result.items[i + 2] == 0x81)
            {
                try cleaned.append(' ');
                i += 3;
            } else {
                try cleaned.append(result.items[i]);
                i += 1;
            }
        }
        result.deinit();

        if (cleaned.items.len > 0 and cleaned.items[0] == ' ') {
            _ = cleaned.orderedRemove(0);
        }

        return try cleaned.toOwnedSlice();
    }

    pub fn vocabSize(self: *const SimpleTokenizer) usize {
        return self.id_to_token.items.len;
    }

    pub fn bos(self: *const SimpleTokenizer) u32 {
        return self.bos_id;
    }

    pub fn eos(self: *const SimpleTokenizer) u32 {
        return self.eos_id;
    }

    pub fn tokenType(self: *const SimpleTokenizer) []const u8 {
        return self.tokenizer_type;
    }
};

fn gpt2ByteToUnicode(byte: u8, buf: []u8) []const u8 {
    if ((byte >= 33 and byte <= 126) or
        (byte >= 161 and byte <= 172) or
        (byte >= 174 and byte <= 255))
    {
        buf[0] = byte;
        return buf[0..1];
    }

    var n: u16 = 0;
    var b: u16 = 0;
    while (b < 256) : (b += 1) {
        const candidate: u8 = @intCast(b);
        const direct = (candidate >= 33 and candidate <= 126) or
            (candidate >= 161 and candidate <= 172) or
            (candidate >= 174 and candidate <= 255);
        if (!direct) {
            if (candidate == byte) {
                return codepointToUtf8(256 + n, buf);
            }
            n += 1;
        }
    }
    unreachable;
}

fn gpt2UnicodeToByte(token: []const u8) ?u8 {
    var buf: [3]u8 = undefined;
    for (0..256) |b| {
        const mapped = gpt2ByteToUnicode(@intCast(b), buf[0..]);
        if (std.mem.eql(u8, mapped, token)) return @intCast(b);
    }
    return null;
}

fn codepointToUtf8(cp: u16, buf: []u8) []const u8 {
    if (cp < 0x80) {
        buf[0] = @intCast(cp);
        return buf[0..1];
    }
    if (cp < 0x800) {
        buf[0] = 0xC0 | @as(u8, @intCast(cp >> 6));
        buf[1] = 0x80 | @as(u8, @intCast(cp & 0x3F));
        return buf[0..2];
    }

    buf[0] = 0xE0 | @as(u8, @intCast(cp >> 12));
    buf[1] = 0x80 | @as(u8, @intCast((cp >> 6) & 0x3F));
    buf[2] = 0x80 | @as(u8, @intCast(cp & 0x3F));
    return buf[0..3];
}

fn spmByteToken(byte: u8, buf: *[6]u8) []const u8 {
    const hex = "0123456789ABCDEF";
    buf.* = .{ '<', '0', 'x', hex[byte >> 4], hex[byte & 0x0F], '>' };
    return buf[0..6];
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

fn buildSyntheticTokenizerGguf(allocator: Allocator) ![]u8 {
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
    try appendString(&bytes, "\xE2\x96\x81hello");
    try appendString(&bytes, "<0x21>");

    try appendString(&bytes, "tokenizer.ggml.scores");
    try appendU32LE(&bytes, 9);
    try appendU32LE(&bytes, 6);
    try appendU64LE(&bytes, 4);
    try appendF32LE(&bytes, 0.0);
    try appendF32LE(&bytes, 0.0);
    try appendF32LE(&bytes, 1.0);
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

test "SimpleTokenizer vocab size" {
    var tokenizer = SimpleTokenizer.init(std.testing.allocator);
    defer tokenizer.deinit();

    const owned = try std.testing.allocator.dupe(u8, "hello");
    try tokenizer.id_to_token.append(.{ .text = owned, .score = 0, .type = 1 });
    try tokenizer.token_to_id.put(owned, 0);
    try tokenizer.rebuildSortedTokens();
    try std.testing.expectEqual(@as(usize, 1), tokenizer.vocabSize());
}

test "SimpleTokenizer SPM encode and decode" {
    var tokenizer = SimpleTokenizer.init(std.testing.allocator);
    defer tokenizer.deinit();

    const tok_bos = try std.testing.allocator.dupe(u8, "<bos>");
    const tok_eos = try std.testing.allocator.dupe(u8, "<eos>");
    const tok_hello = try std.testing.allocator.dupe(u8, "\xE2\x96\x81hello");
    const tok_bang = try std.testing.allocator.dupe(u8, "<0x21>");

    try tokenizer.id_to_token.append(.{ .text = tok_bos, .score = 0, .type = 3 });
    try tokenizer.id_to_token.append(.{ .text = tok_eos, .score = 0, .type = 3 });
    try tokenizer.id_to_token.append(.{ .text = tok_hello, .score = 0, .type = 1 });
    try tokenizer.id_to_token.append(.{ .text = tok_bang, .score = 0, .type = 6 });
    try tokenizer.token_to_id.put(tok_bos, 0);
    try tokenizer.token_to_id.put(tok_eos, 1);
    try tokenizer.token_to_id.put(tok_hello, 2);
    try tokenizer.token_to_id.put(tok_bang, 3);
    tokenizer.bos_id = 0;
    tokenizer.eos_id = 1;
    tokenizer.tokenizer_type = "spm";
    try tokenizer.rebuildSortedTokens();

    var encoded = try tokenizer.encode("hello!");
    defer encoded.deinit();
    try std.testing.expectEqualSlices(u32, &.{ 0, 2, 3 }, encoded.items);

    const decoded = try tokenizer.decode(encoded.items[1..]);
    defer std.testing.allocator.free(decoded);
    try std.testing.expectEqualStrings("hello<0x21>", decoded);
}

test "SimpleTokenizer BPE byte roundtrip" {
    var tokenizer = SimpleTokenizer.init(std.testing.allocator);
    defer tokenizer.deinit();

    const tok_bos = try std.testing.allocator.dupe(u8, "<bos>");
    const tok_eos = try std.testing.allocator.dupe(u8, "<eos>");
    const byte_token = try std.testing.allocator.dupe(u8, "\xC4\x80");

    try tokenizer.id_to_token.append(.{ .text = tok_bos, .score = 0, .type = 3 });
    try tokenizer.id_to_token.append(.{ .text = tok_eos, .score = 0, .type = 3 });
    try tokenizer.id_to_token.append(.{ .text = byte_token, .score = 0, .type = 6 });
    try tokenizer.token_to_id.put(tok_bos, 0);
    try tokenizer.token_to_id.put(tok_eos, 1);
    try tokenizer.token_to_id.put(byte_token, 2);
    tokenizer.bos_id = 0;
    tokenizer.eos_id = 1;
    tokenizer.tokenizer_type = "bpe";
    try tokenizer.rebuildSortedTokens();

    var encoded = try tokenizer.encode("\x00");
    defer encoded.deinit();
    try std.testing.expectEqualSlices(u32, &.{ 0, 2 }, encoded.items);

    const decoded = try tokenizer.decode(encoded.items[1..]);
    defer std.testing.allocator.free(decoded);
    try std.testing.expectEqual(@as(usize, 1), decoded.len);
    try std.testing.expectEqual(@as(u8, 0), decoded[0]);
}

test "SimpleTokenizer initFromReader loads GGUF vocab" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;

    const data = try buildSyntheticTokenizerGguf(std.testing.allocator);
    defer std.testing.allocator.free(data);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "tokenizer.gguf", .data = data });

    var path_buf: [256]u8 = undefined;
    const rel_path = try std.fmt.bufPrint(&path_buf, ".zig-cache/tmp/{s}/tokenizer.gguf", .{tmp.sub_path[0..]});

    var reader = try gguf.Reader.openWithAllocator(rel_path, std.testing.allocator);
    defer reader.deinit();

    var tokenizer = try SimpleTokenizer.initFromReader(&reader, std.testing.allocator);
    defer tokenizer.deinit();

    try std.testing.expectEqual(@as(usize, 4), tokenizer.vocabSize());
    try std.testing.expectEqual(@as(u32, 0), tokenizer.bos());
    try std.testing.expectEqual(@as(u32, 1), tokenizer.eos());
    try std.testing.expectEqualStrings("spm", tokenizer.tokenType());
    try std.testing.expectEqual(@as(f32, 1.0), tokenizer.id_to_token.items[2].score);
    try std.testing.expectEqual(@as(i32, 6), tokenizer.id_to_token.items[3].type);
}
