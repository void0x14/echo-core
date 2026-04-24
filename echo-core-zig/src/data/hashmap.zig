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

        fn keyBytes(key: *const K) []const u8 {
            return switch (@typeInfo(K)) {
                .Pointer => |ptr| if (ptr.size == .Slice and ptr.child == u8)
                    key.*
                else
                    std.mem.asBytes(key),
                else => std.mem.asBytes(key),
            };
        }

        fn keysEqual(a: K, b: K) bool {
            return std.mem.eql(u8, keyBytes(&a), keyBytes(&b));
        }

        fn hash(key: K) usize {
            var h: usize = 2166136261;
            const bytes = keyBytes(&key);
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

test "HashMap resize logic" {
    var map = try HashMap([]const u8, i32).init(std.testing.allocator, 8);
    defer map.deinit();

    var keys = try std.ArrayList([]const u8).initCapacity(std.testing.allocator, 100);
    defer {
        for (keys.items) |key| {
            std.testing.allocator.free(key);
        }
        keys.deinit();
    }

    // Insert 100 items to trigger multiple resizes
    var i: i32 = 0;
    while (i < 100) : (i += 1) {
        const key = try std.fmt.allocPrint(std.testing.allocator, "item{d}", .{i});
        try keys.append(key);
        try map.put(key, i);
    }

    // Verify all 100 items are still present and values are correct
    i = 0;
    while (i < 100) : (i += 1) {
        const key = keys.items[@as(usize, @intCast(i))];
        try std.testing.expectEqual(i, map.get(key).?.*);
    }

    // Verify count is correct
    try std.testing.expectEqual(@as(usize, 100), map.count);

    // Verify it actually resized multiple times
    try std.testing.expect(map.entries.len >= 128);
}

test "HashMap resize multiple times (stress test)" {
    var map = try HashMap(usize, usize).init(std.testing.allocator, 8);
    defer map.deinit();

    const num_items: usize = 1000;

    // Insert enough items to trigger multiple resizes
    for (0..num_items) |i| {
        try map.put(i, i * 2);
    }

    // Verify all items are still present and values are correct
    for (0..num_items) |i| {
        const val = map.get(i);
        try std.testing.expect(val != null);
        try std.testing.expectEqual(@as(usize, i * 2), val.?.*);
    }

    // Verify count is correct
    try std.testing.expectEqual(num_items, map.count);

    // Verify it actually resized enough times (capacity should be > 1000 * 4 / 3)
    try std.testing.expect(map.entries.len >= 2048);
}

test "HashMap multiple resizes and collisions" {
    var map = try HashMap([]const u8, i32).init(std.testing.allocator, 8);
    defer map.deinit();

    // Insert 1000 items to trigger multiple resizes and force hash collisions
    var i: i32 = 0;
    while (i < 1000) : (i += 1) {
        // Create a heap-allocated key since primitive types passed by value shouldn't be used as keys directly
        // due to the dangling pointer issue mentioned in memory
        const key = try std.fmt.allocPrint(std.testing.allocator, "key_{d}", .{i});
        defer std.testing.allocator.free(key);

        // HashMap stores the slice directly, so the underlying memory must outlive the map.
        // We allocate and keep the keys until the map is destroyed.
        const stored_key = try std.testing.allocator.dupe(u8, key);
        errdefer std.testing.allocator.free(stored_key);
        try map.put(stored_key, i);
    }
    defer {
        // Clean up stored keys
        for (map.entries) |entry_opt| {
            if (entry_opt) |entry| {
                std.testing.allocator.free(entry.key);
            }
        }
    }

    // Verify all 1000 items are present and correct
    i = 0;
    while (i < 1000) : (i += 1) {
        const key = try std.fmt.allocPrint(std.testing.allocator, "key_{d}", .{i});
        defer std.testing.allocator.free(key);

        const value_ptr = map.get(key);
        try std.testing.expect(value_ptr != null);
        try std.testing.expectEqual(i, value_ptr.?.*);
    }

    try std.testing.expectEqual(@as(usize, 1000), map.count);

    // Initial capacity is 8.
    // It should resize multiple times and have enough capacity for 1000 items with 3/4 load factor.
    // 1000 / 0.75 = 1333.33 -> next power of 2 is 2048
    try std.testing.expect(map.entries.len >= 2048);
}

test "HashMap resize OutOfMemory" {
    // Wrap the standard allocator with a FailingAllocator
    // .fail_index = 1 allows the initial capacity allocation in init() to succeed,
    // but the next allocation (which happens during resize) will fail.
    var failing_allocator = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 1 });
    const allocator = failing_allocator.allocator();

    var map = try HashMap([]const u8, usize).init(allocator, 8);
    defer map.deinit();

    // With a capacity of 8 and a 75% load factor, the map will trigger a resize on the 7th insertion (count == 6)

    // Create an array to keep track of dynamically allocated string keys.
    var keys: [7][]const u8 = undefined;

    // Successfully insert 6 elements
    for (0..6) |i| {
        const key = try std.fmt.allocPrint(std.testing.allocator, "key_{d}", .{i});
        keys[i] = key;
        try map.put(key, i);
    }

    // Now insert the 7th element, which will trigger the failing resize
    const key_7 = try std.fmt.allocPrint(std.testing.allocator, "key_6", .{});
    keys[6] = key_7;

    const result = map.put(key_7, 6);
    try std.testing.expectError(error.OutOfMemory, result);

    // Verify map is in a consistent state and original items are present
    try std.testing.expectEqual(@as(usize, 6), map.count);

    for (0..6) |i| {
        const val = map.get(keys[i]);
        try std.testing.expect(val != null);
        try std.testing.expectEqual(i, val.?.*);
    }

    // Clean up all dynamically allocated keys
    for (0..7) |i| {
        std.testing.allocator.free(keys[i]);
    }
}
