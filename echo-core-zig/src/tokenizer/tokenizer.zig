const std = @import("std");

pub const TokenData = struct {
    text: []const u8,
    score: f32,
    type: i32,
};

pub const SimpleTokenizer = struct {
    id_to_token: std.ArrayList(TokenData),
    token_to_id: std.StringHashMap(u32),
    sorted_tokens: std.ArrayList(struct { text: []const u8, id: u32 }),
    bos_id: u32,
    eos_id: u32,
    tokenizer_type: []const u8,

    pub fn init(allocator: std.mem.Allocator) SimpleTokenizer {
        return .{
            .id_to_token = std.ArrayList(TokenData).init(allocator),
            .token_to_id = std.StringHashMap(u32).init(allocator),
            .sorted_tokens = std.ArrayList(struct { text: []const u8, id: u32 }).init(allocator),
            .bos_id = 0,
            .eos_id = 1,
            .tokenizer_type = "spm",
        };
    }

    pub fn deinit(self: *SimpleTokenizer) void {
        self.id_to_token.deinit();
        self.token_to_id.deinit();
        self.sorted_tokens.deinit();
    }

    pub fn encode(self: *const SimpleTokenizer, text: []const u8) !std.ArrayList(u32) {
        var result = std.ArrayList(u32).init(self.id_to_token.allocator);

        if (self.tokenizer_type.len == 3 and std.mem.eql(u8, self.tokenizer_type, "spm")) {
            var normalized = std.ArrayList(u8).init(self.id_to_token.allocator);
            try normalized.append(0xE2); // U+2581 = ▁
            for (text) |c| {
                if (c == ' ') {
                    try normalized.append(0xE2);
                    try normalized.append(0x96);
                    try normalized.append(0x81);
                } else {
                    try normalized.append(c);
                }
            }

            var i: usize = 0;
            while (i < normalized.items.len) {
                var best_len: usize = 0;
                var best_id: u32 = 0;
                for (self.sorted_tokens.items) |t| {
                    const token_text = t.text;
                    if (i + token_text.len <= normalized.items.len) {
                        const slice = normalized.items[i..][0..token_text.len];
                        if (std.mem.eql(u8, slice, token_text)) {
                            if (token_text.len > best_len) {
                                best_len = token_text.len;
                                best_id = t.id;
                            }
                        }
                    }
                }
                if (best_len > 0) {
                    try result.append(best_id);
                    i += best_len;
                } else {
                    const hex = [_]u8{ '0', 'x', std.fmt.formatHexChar(text[i] >> 4), std.fmt.formatHexChar(text[i] & 0x0F) };
                    if (self.token_to_id.get(hex)) |id| {
                        try result.append(id);
                    } else {
                        try result.append(self.bos_id);
                    }
                    i += 1;
                }
            }
        } else {
            for (text) |byte| {
                const byte_str = [_]u8{byte};
                if (self.token_to_id.get(&byte_str)) |id| {
                    try result.append(id);
                } else {
                    try result.append(self.bos_id);
                }
            }
        }

        return result;
    }

    pub fn decode(self: *const SimpleTokenizer, ids: []const u32) ![]u8 {
        var result = std.ArrayList(u8).init(self.id_to_token.allocator);

        for (ids) |id| {
            if (id == self.bos_id or id == self.eos_id) continue;
            if (id >= self.id_to_token.items.len) continue;

            const token = self.id_to_token.items[id];
            if (token.type == 3) continue;

            try result.appendSlice(token.text);
        }

        if (self.tokenizer_type.len == 3 and std.mem.eql(u8, self.tokenizer_type, "spm")) {
            if (result.items.len > 0 and result.items[0] == 0xE2) {
                result.items[0] = ' ';
            }
        }

        return result.items;
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

test "SimpleTokenizer vocab size" {
    var tokenizer = SimpleTokenizer.init(std.testing.allocator);
    defer tokenizer.deinit();
    try tokenizer.id_to_token.append(.{ .text = "hello", .score = 0, .type = 1 });
    try tokenizer.token_to_id.put("hello", 0);
    try tokenizer.sorted_tokens.append(.{ .text = "hello", .id = 0 });
    try std.testing.expectEqual(tokenizer.vocabSize(), 1);
}
