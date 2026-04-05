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

test "ArrayList pop multiple items" {
    var list = ArrayList(i32).init(std.testing.allocator);
    defer list.deinit();

    try list.append(10);
    try list.append(20);
    try list.append(30);

    try std.testing.expectEqual(@as(?i32, 30), list.pop());
    try std.testing.expectEqual(@as(usize, 2), list.len);

    try std.testing.expectEqual(@as(?i32, 20), list.pop());
    try std.testing.expectEqual(@as(usize, 1), list.len);

    try std.testing.expectEqual(@as(?i32, 10), list.pop());
    try std.testing.expectEqual(@as(usize, 0), list.len);

    try std.testing.expectEqual(@as(?i32, null), list.pop());
    try std.testing.expectEqual(@as(usize, 0), list.len);
}

test "ArrayList get retrieves items by index and handles out of bounds" {
    var list = ArrayList(i32).init(std.testing.allocator);
    defer list.deinit();

    try std.testing.expectEqual(@as(?*const i32, null), list.get(0));

    try list.append(100);
    try list.append(200);
    try list.append(300);

    const first = list.get(0);
    try std.testing.expect(first != null);
    try std.testing.expectEqual(@as(i32, 100), first.?.*);

    const second = list.get(1);
    try std.testing.expect(second != null);
    try std.testing.expectEqual(@as(i32, 200), second.?.*);

    const third = list.get(2);
    try std.testing.expect(third != null);
    try std.testing.expectEqual(@as(i32, 300), third.?.*);

    try std.testing.expectEqual(@as(?*const i32, null), list.get(3));
    try std.testing.expectEqual(@as(?*const i32, null), list.get(10));
}
