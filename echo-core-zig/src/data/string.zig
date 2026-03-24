const std = @import("std");

pub const String = struct {
    data: []const u8,

    pub fn fromBytes(bytes: []const u8) String {
        return .{ .data = bytes };
    }

    pub fn toBytes(s: String) []const u8 {
        return s.data;
    }

    pub fn len(s: String) usize {
        return s.data.len;
    }

    pub fn eq(s1: String, s2: String) bool {
        return std.mem.eql(u8, s1.data, s2.data);
    }

    pub fn startsWith(s: String, prefix: String) bool {
        if (prefix.len() > s.len()) return false;
        return std.mem.eql(u8, s.data[0..prefix.len()], prefix.data);
    }

    pub fn endsWith(s: String, suffix: String) bool {
        if (suffix.len() > s.len()) return false;
        return std.mem.eql(u8, s.data[s.len() - suffix.len() ..], suffix.data);
    }

    pub fn concat(allocator: std.mem.Allocator, s1: String, s2: String) !String {
        const result = try allocator.alloc(u8, s1.len() + s2.len());
        @memcpy(result[0..s1.len()], s1.data);
        @memcpy(result[s1.len()..], s2.data);
        return .{ .data = result };
    }

    pub fn substring(s: String, start: usize, end: usize) String {
        if (start >= s.len() or end <= start) return .{ .data = &.{} };
        const real_end = if (end > s.len()) s.len() else end;
        return .{ .data = s.data[start..real_end] };
    }

    pub fn indexOf(s: String, needle: String) ?usize {
        if (needle.len() > s.len()) return null;
        for (0..s.len() - needle.len() + 1) |i| {
            if (std.mem.eql(u8, s.data[i..][0..needle.len()], needle.data)) {
                return i;
            }
        }
        return null;
    }
};

test "String basic" {
    const s = String.fromBytes("hello");
    try std.testing.expectEqual(s.len(), 5);
    try std.testing.expect(String.eq(s, String.fromBytes("hello")));
    try std.testing.expect(!String.eq(s, String.fromBytes("world")));
}

test "String startsWith/endsWith" {
    const s = String.fromBytes("hello world");
    try std.testing.expect(String.startsWith(s, String.fromBytes("hello")));
    try std.testing.expect(!String.startsWith(s, String.fromBytes("world")));
    try std.testing.expect(String.endsWith(s, String.fromBytes("world")));
    try std.testing.expect(!String.endsWith(s, String.fromBytes("hello")));
}

test "String substring with reversed bounds is empty" {
    const s = String.fromBytes("hello");
    const sub = String.substring(s, 4, 2);
    try std.testing.expectEqual(@as(usize, 0), sub.len());
}
