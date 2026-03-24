const std = @import("std");

pub fn HashMap(comptime K: type, comptime V: type) type {
    return struct {
        entries: []?Entry = &.{},
        count: usize = 0,
        allocator: std.mem.Allocator,

        const Entry = struct {
            key: K,
            value: V,
        };

        pub fn init(allocator: std.mem.Allocator, capacity: usize) !@This() {
            const size = nextPowerOfTwo(if (capacity < 8) 8 else capacity);
            const entries = try allocator.alloc(?Entry, size);
            @memset(entries, null);
            return .{
                .entries = entries,
                .count = 0,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *@This()) void {
            self.allocator.free(self.entries);
            self.* = undefined;
        }

        fn nextPowerOfTwo(n: usize) usize {
            if (n <= 1) return 1;
            var v = n;
            v -= 1;
            v |= v >> 1;
            v |= v >> 2;
            v |= v >> 4;
            v |= v >> 8;
            v |= v >> 16;
            v |= v >> 32;
            return v + 1;
        }

        fn keyBytes(key: K) []const u8 {
            return switch (@typeInfo(K)) {
                .pointer => |ptr| if (ptr.size == .slice and ptr.child == u8)
                    key
                else
                    std.mem.asBytes(&key),
                else => std.mem.asBytes(&key),
            };
        }

        fn keysEqual(a: K, b: K) bool {
            return std.mem.eql(u8, keyBytes(a), keyBytes(b));
        }

        fn hash(key: K) usize {
            var h: usize = 2166136261;
            const bytes = keyBytes(key);
            for (bytes) |byte| {
                h ^= byte;
                h *%= 16777619;
            }
            return h;
        }

        fn putAssumeCapacity(self: *@This(), key: K, value: V) void {
            const mask = self.entries.len - 1;
            var i = hash(key) & mask;
            while (true) : (i = (i + 1) & mask) {
                if (self.entries[i] == null) {
                    self.entries[i] = .{ .key = key, .value = value };
                    self.count += 1;
                    return;
                }
                if (keysEqual(self.entries[i].?.key, key)) {
                    self.entries[i].?.value = value;
                    return;
                }
            }
        }

        pub fn get(self: *const @This(), key: K) ?*const V {
            const mask = self.entries.len - 1;
            var i = hash(key) & mask;
            while (true) : (i = (i + 1) & mask) {
                if (self.entries[i] == null) return null;
                if (keysEqual(self.entries[i].?.key, key)) {
                    return &self.entries[i].?.value;
                }
            }
        }

        pub fn put(self: *@This(), key: K, value: V) !void {
            if (self.count >= self.entries.len * 3 / 4) {
                try self.resize();
            }
            self.putAssumeCapacity(key, value);
        }

        fn resize(self: *@This()) !void {
            const old_entries = self.entries;
            const new_size = old_entries.len * 2;
            const new_entries = try self.allocator.alloc(?Entry, new_size);
            @memset(new_entries, null);

            self.entries = new_entries;
            self.count = 0;

            for (old_entries) |entry| {
                if (entry) |e| {
                    self.putAssumeCapacity(e.key, e.value);
                }
            }

            self.allocator.free(old_entries);
        }

        fn reinsertCluster(self: *@This(), start_idx: usize) void {
            const mask = self.entries.len - 1;
            var i = (start_idx + 1) & mask;
            while (self.entries[i]) |entry| : (i = (i + 1) & mask) {
                self.entries[i] = null;
                self.count -= 1;
                self.putAssumeCapacity(entry.key, entry.value);
            }
        }

        pub fn remove(self: *@This(), key: K) bool {
            const mask = self.entries.len - 1;
            var i = hash(key) & mask;
            while (true) : (i = (i + 1) & mask) {
                if (self.entries[i] == null) return false;
                if (keysEqual(self.entries[i].?.key, key)) {
                    self.entries[i] = null;
                    self.count -= 1;
                    self.reinsertCluster(i);
                    return true;
                }
            }
        }
    };
}

test "HashMap basic" {
    var map = try HashMap([]const u8, i32).init(std.testing.allocator, 16);
    defer map.deinit();

    try map.put("answer", 42);
    try map.put("foo", 100);

    try std.testing.expectEqual(map.get("answer").?.*, 42);
    try std.testing.expectEqual(map.get("foo").?.*, 100);
    try std.testing.expect(map.get("missing") == null);
}

test "HashMap string keys compare by contents" {
    var map = try HashMap([]const u8, i32).init(std.testing.allocator, 8);
    defer map.deinit();

    const heap_key = try std.testing.allocator.dupe(u8, "answer");
    defer std.testing.allocator.free(heap_key);

    try map.put(heap_key, 42);
    try std.testing.expectEqual(@as(i32, 42), map.get("answer").?.*);
}

test "HashMap remove preserves probe chain" {
    var map = try HashMap([]const u8, i32).init(std.testing.allocator, 8);
    defer map.deinit();

    const candidates = [_][]const u8{ "aa", "ab", "ac", "ad", "ae", "af", "ag", "ah", "ai", "aj" };

    var first: ?[]const u8 = null;
    var second: ?[]const u8 = null;
    outer: for (candidates, 0..) |a, i| {
        for (candidates[(i + 1)..]) |b| {
            if ((HashMap([]const u8, i32).hash(a) & 7) == (HashMap([]const u8, i32).hash(b) & 7)) {
                first = a;
                second = b;
                break :outer;
            }
        }
    }

    try std.testing.expect(first != null and second != null);
    try map.put(first.?, 1);
    try map.put(second.?, 2);
    try std.testing.expect(map.remove(first.?));
    try std.testing.expectEqual(@as(i32, 2), map.get(second.?).?.*);
}
