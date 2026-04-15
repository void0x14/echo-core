const std = @import("std");

pub fn ArrayList(comptime T: type) type {
    return struct {
        items: []T = &.{},
        len: usize = 0,
        capacity: usize = 0,
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator) @This() {
            return .{
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *@This()) void {
            if (self.items.len > 0) {
                self.allocator.free(self.items);
            }
            self.* = undefined;
        }

        pub fn append(self: *@This(), item: T) !void {
            if (self.len >= self.capacity) {
                const new_capacity = if (self.capacity == 0) 8 else self.capacity * 2;
                const new_items = try self.allocator.alloc(T, new_capacity);
                @memcpy(new_items[0..self.len], self.items);
                if (self.items.len > 0) {
                    self.allocator.free(self.items);
                }
                self.items = new_items;
                self.capacity = new_capacity;
            }
            self.items[self.len] = item;
            self.len += 1;
        }

        pub fn pop(self: *@This()) ?T {
            if (self.len == 0) return null;
            self.len -= 1;
            return self.items[self.len];
        }

        pub fn get(self: *const @This(), index: usize) ?*const T {
            if (index >= self.len) return null;
            return &self.items[index];
        }

        pub fn clear(self: *@This()) void {
            self.len = 0;
        }
    };
}

test "ArrayList basic" {
    var list = ArrayList(i32).init(std.testing.allocator);
    defer list.deinit();

    try list.append(1);
    try list.append(2);
    try list.append(3);

    try std.testing.expectEqual(list.len, 3);
    try std.testing.expectEqual(list.pop(), 3);
    try std.testing.expectEqual(list.len, 2);
}

test "ArrayList const get returns const pointer" {
    var list = ArrayList(i32).init(std.testing.allocator);
    defer list.deinit();

    try list.append(123);

    const const_list = list;
    try std.testing.expectEqual(*const i32, @TypeOf(const_list.get(0).?));
}

test "ArrayList empty pop returns null" {
    var list = ArrayList(i32).init(std.testing.allocator);
    defer list.deinit();

    try std.testing.expectEqual(@as(?i32, null), list.pop());
}

test "ArrayList pop until empty" {
    var list = ArrayList(i32).init(std.testing.allocator);
    defer list.deinit();

    try list.append(10);
    try list.append(20);

    try std.testing.expectEqual(list.pop(), 20);
    try std.testing.expectEqual(list.len, 1);

    try std.testing.expectEqual(list.pop(), 10);
    try std.testing.expectEqual(list.len, 0);

    try std.testing.expectEqual(@as(?i32, null), list.pop());
    try std.testing.expectEqual(list.len, 0);
}

test "ArrayList get out of bounds" {
    var list = ArrayList(i32).init(std.testing.allocator);
    defer list.deinit();

    try std.testing.expectEqual(@as(?*const i32, null), list.get(0));

    try list.append(1);
    try std.testing.expectEqual(@as(?*const i32, null), list.get(1));
    try std.testing.expectEqual(@as(?*const i32, null), list.get(100));
}

test "ArrayList pop from newly initialized list returns null" {
    var list = ArrayList(f32).init(std.testing.allocator);
    defer list.deinit();
    try std.testing.expectEqual(@as(?f32, null), list.pop());
}

test "ArrayList pop after clear returns null" {
    var list = ArrayList(i32).init(std.testing.allocator);
    defer list.deinit();

    try list.append(10);
    list.clear();
    try std.testing.expectEqual(@as(?i32, null), list.pop());
}
