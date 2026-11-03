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
