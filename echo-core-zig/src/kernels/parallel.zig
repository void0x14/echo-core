const std = @import("std");
const gguf = @import("../gguf/reader.zig");
const matvec = @import("matvec.zig");
const NUM_THREADS: u32 = 16;

fn rowStride(K: u32, dtype: gguf.GGMLType) usize {
    return switch (dtype) {
        .f16 => @as(usize, K) * 2, .f32 => @as(usize, K) * 4,
        .q8_0 => (@as(usize, K) / 32) * 34, .q4_k => (@as(usize, K) / 256) * 144,
        .q5_k => (@as(usize, K) / 256) * 176, .q6_k => (@as(usize, K) / 256) * 210,
        .q2_k => (@as(usize, K) / 256) * 84, .q3_k => (@as(usize, K) / 256) * 110,
        else => @as(usize, K) * 4,
    };
}

const ThreadData = struct {
    W: [*]const u8, x: [*]const f32, y: [*]f32,
    M_start: u32, M_end: u32, K: u32, row_stride: usize, dtype: gguf.GGMLType,
};

fn worker(data: *const ThreadData) void {
    const m0 = data.M_start;
    const rows = data.M_end - m0;
    for (0..rows) |i| {
        const row_ptr = data.W + @as(usize, m0 + i) * data.row_stride;
        matvec.matvecDispatchQuant(0, 0, row_ptr, data.x, data.y + m0 + @as(u32, @intCast(i)), 1, data.K, data.dtype);
    }
}

pub fn parallelMatvec(W: [*]const u8, x: [*]const f32, y: [*]f32, M: u32, K: u32, dtype: gguf.GGMLType) void {
    if (M < 32) { matvec.matvecDispatchQuant(0, 0, W, x, y, M, K, dtype); return; }
    const stride = rowStride(K, dtype);
    const rows_per_thread = (M + NUM_THREADS - 1) / NUM_THREADS;
    var ctx: [NUM_THREADS]ThreadData = undefined;
    var threads: [NUM_THREADS]std.Thread = undefined;
    var active: u32 = 0;
    for (0..NUM_THREADS) |t| {
        const start = @as(u32, @intCast(t)) * rows_per_thread;
        const end = @min(start + rows_per_thread, M);
        if (start >= M) break;
        ctx[t] = .{ .W = W, .x = x, .y = y, .M_start = start, .M_end = end, .K = K, .row_stride = stride, .dtype = dtype };
        threads[t] = std.Thread.spawn(.{}, worker, .{&ctx[t]}) catch { matvec.matvecDispatchQuant(0, 0, W, x, y, M, K, dtype); return; };
        active += 1;
    }
    for (0..active) |t| threads[t].join();
}
