const std = @import("std");
const gguf = @import("../gguf/reader.zig");
const matvec = @import("matvec.zig");

pub const NUM_THREADS: u32 = 16;
comptime {
    if (NUM_THREADS > 64) @compileError("too many threads");
}

const ThreadData = struct {
    W: [*]const u8,
    x: [*]const f32,
    y: [*]f32,
    M_start: u32,
    M_end: u32,
    K: u32,
    row_stride: usize,
    dtype: gguf.GGMLType,
};

fn worker(data: *const ThreadData) void {
    var m = data.M_start;
    while (m < data.M_end) : (m += 1) {
        const row_ptr = data.W + @as(usize, m) * data.row_stride;
        matvec.matvecDispatchQuant(0, 0, row_ptr, data.x, data.y + m, 1, data.K, data.dtype);
    }
}

fn rowStride(K: u32, dtype: gguf.GGMLType) usize {
    return switch (dtype) {
        .f16 => @as(usize, K) * 2,
        .f32 => @as(usize, K) * 4,
        .q8_0 => (@as(usize, K) / 32) * 34,
        .q4_k => (@as(usize, K) / 256) * 144,
        .q5_k => (@as(usize, K) / 256) * 176,
        .q6_k => (@as(usize, K) / 256) * 210,
        .q2_k => (@as(usize, K) / 256) * 84,
        .q3_k => (@as(usize, K) / 256) * 110,
        else => @as(usize, K) * 4,
    };
}

pub fn parallelMatvec(W: [*]const u8, x: [*]const f32, y: [*]f32, M: u32, K: u32, dtype: gguf.GGMLType) void {
    if (M < NUM_THREADS * 2) {
        matvec.matvecDispatchQuant(0, 0, W, x, y, M, K, dtype);
        return;
    }

    const stride = rowStride(K, dtype);
    const rows_per_thread = (M + NUM_THREADS - 1) / NUM_THREADS;

    var contexts: [NUM_THREADS]ThreadData = undefined;
    var threads: [NUM_THREADS]std.Thread = undefined;
    var active: u32 = 0;

    for (0..NUM_THREADS) |t| {
        const start = @as(u32, @intCast(t)) * rows_per_thread;
        const end = @min(start + rows_per_thread, M);
        if (start >= M) break;
        contexts[t] = .{
            .W = W,
            .x = x,
            .y = y,
            .M_start = start,
            .M_end = end,
            .K = K,
            .row_stride = stride,
            .dtype = dtype,
        };
        threads[t] = std.Thread.spawn(.{}, worker, .{&contexts[t]}) catch {
            matvec.matvecDispatchQuant(0, 0, W, x, y, M, K, dtype);
            return;
        };
        active += 1;
    }

    for (0..active) |t| {
        threads[t].join();
    }
}

test "parallel matvec identity" {
    if (@import("builtin").single_threaded) return error.SkipZigTest;

    const W: [16]f32 = @splat(1.0);
    const x: [16]f32 = @splat(2.0);
    var y: [16]f32 = undefined;
    @memset(&y, 0);

    parallelMatvec(@ptrCast(&W), &x, &y, 4, 4, .f32);
    try std.testing.expectEqual(@as(f32, 8.0), y[0]);
    try std.testing.expectEqual(@as(f32, 8.0), y[3]);
}
