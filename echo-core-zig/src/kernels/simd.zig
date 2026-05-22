const std = @import("std");

pub fn hsumF32_8(v: @Vector(8, f32)) f32 {
    return @reduce(.Add, v);
}

pub fn hsumI32_8(v: @Vector(8, i32)) i32 {
    return @reduce(.Add, v);
}

pub fn dotProductI8(a: @Vector(32, i8), b: @Vector(32, i8)) i32 {
    const a32: @Vector(32, i32) = @intCast(a);
    const b32: @Vector(32, i32) = @intCast(b);
    const prod = a32 * b32;
    return @reduce(.Add, prod);
}

test "hsumF32_8 basic" {
    const v: @Vector(8, f32) = .{ 1, 2, 3, 4, 5, 6, 7, 8 };
    try std.testing.expectEqual(@as(f32, 36), hsumF32_8(v));
}

test "hsumF32_8 negative" {
    const v: @Vector(8, f32) = .{ -1, -2, -3, -4, 0, 1, 2, 3 };
    try std.testing.expectEqual(@as(f32, -4), hsumF32_8(v));
}

test "hsumI32_8 basic" {
    const v: @Vector(8, i32) = .{ 1, 2, 3, 4, 5, 6, 7, 8 };
    try std.testing.expectEqual(@as(i32, 36), hsumI32_8(v));
}

test "hsumI32_8 negative" {
    const v: @Vector(8, i32) = .{ -10, -20, 5, 0, 3, -1, 2, -4 };
    try std.testing.expectEqual(@as(i32, -25), hsumI32_8(v));
}

test "dotProductI8 basic" {
    const a: @Vector(32, i8) = @splat(1);
    const b: @Vector(32, i8) = @splat(2);
    try std.testing.expectEqual(@as(i32, 64), dotProductI8(a, b));
}

test "dotProductI8 negative" {
    const a: @Vector(32, i8) = @splat(-1);
    const b: @Vector(32, i8) = @splat(2);
    try std.testing.expectEqual(@as(i32, -64), dotProductI8(a, b));
}

test "dotProductI8 mixed" {
    const a: @Vector(32, i8) = .{
        1, -1, 2, -2, 3, -3, 4, -4,
        5, -5, 6, -6, 7, -7, 8, -8,
        9, -9, 10, -10, 11, -11, 12, -12,
        13, -13, 14, -14, 15, -15, 16, -16,
    };
    const b: @Vector(32, i8) = @splat(1);
    try std.testing.expectEqual(@as(i32, 0), dotProductI8(a, b));
}
