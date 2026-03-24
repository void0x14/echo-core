const std = @import("std");

pub inline fn sqrt(x: f32) f32 {
    return @sqrt(x);
}

pub inline fn exp(x: f32) f32 {
    return @exp(x);
}

pub inline fn tanh(x: f32) f32 {
    return std.math.tanh(x);
}

pub inline fn sqrtf(x: f64) f64 {
    return @sqrt(x);
}

pub inline fn expf(x: f64) f64 {
    return @exp(x);
}

pub inline fn tanhF64(x: f64) f64 {
    return std.math.tanh(x);
}

pub inline fn relu(x: f32) f32 {
    return if (x > 0) x else 0;
}

pub inline fn gelu(x: f32) f32 {
    return x * 0.5 * (1.0 + std.math.tanh(0.7978845608028674 * (x + 0.044715 * x * x * x)));
}

pub inline fn swish(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

pub inline fn softmax(values: []f32) void {
    var max_val: f32 = -std.math.inf(f32);
    for (values) |v| {
        if (v > max_val) max_val = v;
    }
    var sum: f32 = 0;
    for (values) |*v| {
        v.* = @exp(v.* - max_val);
        sum += v.*;
    }
    for (values) |*v| {
        v.* /= sum;
    }
}

test "sqrt basic" {
    try std.testing.expectApproxEqAbs(sqrt(4.0), 2.0, 0.001);
    try std.testing.expectApproxEqAbs(sqrt(9.0), 3.0, 0.001);
}

test "exp basic" {
    try std.testing.expectApproxEqAbs(exp(0.0), 1.0, 0.001);
    try std.testing.expectApproxEqAbs(exp(1.0), std.math.e, 0.01);
}

test "gelu basic" {
    const result = gelu(1.0);
    try std.testing.expect(result > 0.7 and result < 0.8);
}

test "relu basic" {
    try std.testing.expectEqual(relu(5.0), 5.0);
    try std.testing.expectEqual(relu(-5.0), 0.0);
    try std.testing.expectEqual(relu(0.0), 0.0);
}
