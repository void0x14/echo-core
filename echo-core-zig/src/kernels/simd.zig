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
