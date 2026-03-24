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
            const size = nextPowerOfTwo(capacity);
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

        fn hash(key: K) usize {
            var h: usize = 2166136261;
            const bytes = std.mem.asBytes(&key);
            for (bytes) |byte| {
                h ^= byte;
                h *= 16777619;
            }
            return h;
        }

        pub fn get(self: *const @This(), key: K) ?*V {
            const idx = hash(key) & (self.entries.len - 1);
            var i = idx;
            while (i < self.entries.len) : (i += 1) {
                if (self.entries[i] == null) return null;
                if (std.mem.eql(u8, std.mem.asBytes(&self.entries[i].?.key), std.mem.asBytes(&key))) {
                    return &self.entries[i].?.value;
                }
            }
            i = 0;
            while (i < idx) : (i += 1) {
                if (self.entries[i] == null) return null;
                if (std.mem.eql(u8, std.mem.asBytes(&self.entries[i].?.key), std.mem.asBytes(&key))) {
                    return &self.entries[i].?.value;
                }
            }
            return null;
        }

        pub fn put(self: *@This(), key: K, value: V) !void {
            if (self.count >= self.entries.len * 3 / 4) {
                try self.resize();
            }
            const idx = hash(key) & (self.entries.len - 1);
            var i = idx;
            while (i < self.entries.len) : (i += 1) {
                if (self.entries[i] == null) {
                    self.entries[i] = .{ .key = key, .value = value };
                    self.count += 1;
                    return;
                }
                if (std.mem.eql(u8, std.mem.asBytes(&self.entries[i].?.key), std.mem.asBytes(&key))) {
                    self.entries[i].?.value = value;
                    return;
                }
            }
            i = 0;
            while (i < idx) : (i += 1) {
                if (self.entries[i] == null) {
                    self.entries[i] = .{ .key = key, .value = value };
                    self.count += 1;
                    return;
                }
                if (std.mem.eql(u8, std.mem.asBytes(&self.entries[i].?.key), std.mem.asBytes(&key))) {
                    self.entries[i].?.value = value;
                    return;
                }
            }
        }

        fn resize(self: *@This()) !void {
            const new_size = self.entries.len * 2;
            const new_entries = try self.allocator.alloc(?Entry, new_size);
            @memset(new_entries, null);

            for (self.entries) |entry| {
                if (entry) |e| {
                    const idx = hash(e.key) & (new_size - 1);
                    var i = idx;
                    while (i < new_size) : (i += 1) {
                        if (new_entries[i] == null) {
                            new_entries[i] = e;
                            break;
                        }
                    }
                }
            }

            self.allocator.free(self.entries);
            self.entries = new_entries;
        }

        pub fn remove(self: *@This(), key: K) bool {
            const idx = hash(key) & (self.entries.len - 1);
            var i = idx;
            while (i < self.entries.len) : (i += 1) {
                if (self.entries[i] == null) return false;
                if (std.mem.eql(u8, std.mem.asBytes(&self.entries[i].?.key), std.mem.asBytes(&key))) {
                    self.entries[i] = null;
                    self.count -= 1;
                    return true;
                }
            }
            i = 0;
            while (i < idx) : (i += 1) {
                if (self.entries[i] == null) return false;
                if (std.mem.eql(u8, std.mem.asBytes(&self.entries[i].?.key), std.mem.asBytes(&key))) {
                    self.entries[i] = null;
                    self.count -= 1;
                    return true;
                }
            }
            return false;
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
