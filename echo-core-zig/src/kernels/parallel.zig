const std = @import("std");
const gguf = @import("../gguf/reader.zig");
const matvec = @import("matvec.zig");

const ThreadData = struct {
    W: [*]const u8,
    x: [*]const f32,
    y: [*]f32,
    M_start: u32,
    M_end: u32,
    K: u32,
    row_stride: usize,
    block_stride: usize,
    dtype: gguf.GGMLType,
};

fn worker(data: *const ThreadData) void {
    _ = data.block_stride;
    const row_stride = data.row_stride;
    var m = data.M_start;
    while (m < data.M_end) : (m += 1) {
        const row_ptr = data.W + @as(usize, m) * row_stride;
        matvec.matvecDispatchQuant(0, 0, row_ptr, data.x, data.y + m, 1, data.K, data.dtype);
    }
}

pub fn parallelMatvec(W: [*]const u8, x: [*]const f32, y: [*]f32, M: u32, K: u32, dtype: gguf.GGMLType) !void {
    const num_threads = 4;
    const blocks_per_row = K / 256;
    const row_stride: usize = switch (dtype) {
        .q4_k => blocks_per_row * 144,
        .q5_k => blocks_per_row * 176,
        .q6_k => blocks_per_row * 210,
        .q2_k => blocks_per_row * 84,
        else => return matvec.matvecDispatchQuant(0, 0, W, x, y, M, K, dtype),
    };
    const blk_stride: usize = switch (dtype) {
        .q4_k => 144,
        .q5_k => 176,
        .q6_k => 210,
        .q2_k => 84,
        else => return matvec.matvecDispatchQuant(0, 0, W, x, y, M, K, dtype),
    };

    const rows_per_thread = (M + num_threads - 1) / num_threads;
    if (rows_per_thread < 2) return matvec.matvecDispatchQuant(0, 0, W, x, y, M, K, dtype);

    var contexts: [4]ThreadData = undefined;
    var threads: [4]std.Thread = undefined;
    var active: u32 = 0;

    for (0..num_threads) |t| {
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
            .row_stride = row_stride,
            .block_stride = blk_stride,
            .dtype = dtype,
        };
        threads[t] = try std.Thread.spawn(.{}, worker, .{&contexts[t]});
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

    try parallelMatvec(@ptrCast(&W), &x, &y, 4, 4, .f32);

    try std.testing.expectEqual(@as(f32, 8.0), y[0]);
    try std.testing.expectEqual(@as(f32, 8.0), y[3]);
}
