const std = @import("std");
const builtin = @import("builtin");
const gguf = @import("../gguf/reader.zig");
const matvec = @import("matvec.zig");
const int_dot = @import("int_dot.zig");
const quantK = @import("quant.zig");

const MAX_TOTAL: u32 = 64;
const MIN_PARALLEL_ROWS: u32 = 32;

const JobFn = *const fn (ctx: *anyopaque, tid: u32, total: u32) void;

const Pool = struct {
    threads: [MAX_TOTAL - 1]std.Thread = undefined,
    n_workers: u32 = 0,
    generation: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    done: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    job_fn: JobFn = undefined,
    job_ctx: *anyopaque = undefined,

    fn workerMain(self: *Pool, tid: u32) void {
        var seen: u32 = 0;
        while (true) {
            var spins: u32 = 0;
            while (self.generation.load(.acquire) == seen) {
                if (spins < 2048) {
                    std.atomic.spinLoopHint();
                    spins += 1;
                } else if (builtin.os.tag == .linux) {
                    const ts = std.os.linux.timespec{ .sec = 0, .nsec = 1_000_000 };
                    _ = std.os.linux.futex_4arg(&self.generation.raw, .{ .cmd = .WAIT, .private = true }, seen, &ts);
                } else {
                    _ = std.Thread.yield() catch {};
                }
            }
            seen = self.generation.load(.acquire);
            const f = self.job_fn;
            const ctx = self.job_ctx;
            const total = self.n_workers + 1;
            f(ctx, tid, total);
            _ = self.done.fetchAdd(1, .release);
        }
    }

    fn run(self: *Pool, f: JobFn, ctx: *anyopaque) void {
        self.job_fn = f;
        self.job_ctx = ctx;
        self.done.store(0, .release);
        _ = self.generation.fetchAdd(1, .release);
        if (builtin.os.tag == .linux) {
            _ = std.os.linux.futex_3arg(&self.generation.raw, .{ .cmd = .WAKE, .private = true }, 0x7fffffff);
        }
        const total = self.n_workers + 1;
        f(ctx, 0, total);
        while (self.done.load(.acquire) < self.n_workers) {
            std.atomic.spinLoopHint();
        }
    }
};

var pool: Pool = .{};
var pool_state: std.atomic.Value(u32) = std.atomic.Value(u32).init(0);

fn getPool() ?*Pool {
    switch (pool_state.load(.acquire)) {
        2 => return &pool,
        3 => return null,
        else => {},
    }
    if (pool_state.cmpxchgStrong(0, 1, .acquire, .monotonic) != null) {
        while (pool_state.load(.acquire) == 1) std.atomic.spinLoopHint();
        return if (pool_state.load(.acquire) == 2) &pool else null;
    }
    const cpu = std.Thread.getCpuCount() catch 6;
    var total: u32 = @intCast(@min(@max((cpu + 1) / 2, 1), MAX_TOTAL));
    if (total < 2) total = 2;
    pool.n_workers = total - 1;
    var spawned: u32 = 0;
    for (0..pool.n_workers) |i| {
        pool.threads[i] = std.Thread.spawn(.{}, Pool.workerMain, .{ &pool, @as(u32, @intCast(i + 1)) }) catch break;
        spawned += 1;
    }
    pool.n_workers = spawned;
    if (spawned == 0) {
        pool_state.store(3, .release);
        return null;
    }
    pool_state.store(2, .release);
    return &pool;
}

fn chunkRange(m: u32, tid: u32, total: u32) struct { start: u32, end: u32 } {
    const rows_per = (m + total - 1) / total;
    const start = tid * rows_per;
    if (start >= m) return .{ .start = m, .end = m };
    return .{ .start = start, .end = @min(start + rows_per, m) };
}

fn rowStride(k: u32, dtype: gguf.GGMLType) usize {
    return switch (dtype) {
        .f16 => @as(usize, k) * 2,
        .f32 => @as(usize, k) * 4,
        .q8_0 => (@as(usize, k) / 32) * 34,
        .q4_k => (@as(usize, k) / 256) * 144,
        .q5_k => (@as(usize, k) / 256) * 176,
        .q6_k => (@as(usize, k) / 256) * 210,
        .q2_k => (@as(usize, k) / 256) * 84,
        .q3_k => (@as(usize, k) / 256) * 110,
        else => @as(usize, k) * 4,
    };
}

const MatvecJob = struct {
    w: [*]const u8,
    x: [*]const f32,
    y: [*]f32,
    m: u32,
    k: u32,
    stride: usize,
    dtype: gguf.GGMLType,

    fn work(ctx: *anyopaque, tid: u32, total: u32) void {
        const self: *MatvecJob = @ptrCast(@alignCast(ctx));
        const r = chunkRange(self.m, tid, total);
        if (r.end <= r.start) return;
        const row_ptr = self.w + @as(usize, r.start) * self.stride;
        matvec.matvecDispatchQuant(0, 0, row_ptr, self.x, self.y + r.start, r.end - r.start, self.k, self.dtype);
    }
};

pub fn parallelMatvec(w: [*]const u8, x: [*]const f32, y: [*]f32, m: u32, k: u32, dtype: gguf.GGMLType) void {
    if (dtype == .q8_0 and k % 32 == 0 and k <= 16384) {
        var q8_buf: [512]quantK.block_q8_0 = undefined;
        matvec.quantizeXToQ80(x, &q8_buf, k);
        parallelMatvecQ80Pre(w, &q8_buf, y, m, k);
        return;
    }
    if (m < MIN_PARALLEL_ROWS) {
        matvec.matvecDispatchQuant(0, 0, w, x, y, m, k, dtype);
        return;
    }
    const p = getPool() orelse {
        matvec.matvecDispatchQuant(0, 0, w, x, y, m, k, dtype);
        return;
    };
    var job = MatvecJob{ .w = w, .x = x, .y = y, .m = m, .k = k, .stride = rowStride(k, dtype), .dtype = dtype };
    p.run(MatvecJob.work, &job);
}

const PreJobQ80 = struct {
    w: [*]const u8,
    q8: [*]const quantK.block_q8_0,
    y: [*]f32,
    m: u32,
    k: u32,
    stride: usize,

    fn work(ctx: *anyopaque, tid: u32, total: u32) void {
        const self: *PreJobQ80 = @ptrCast(@alignCast(ctx));
        const r = chunkRange(self.m, tid, total);
        if (r.end <= r.start) return;
        const row_ptr = self.w + @as(usize, r.start) * self.stride;
        matvec.matvecQ80IntPre(row_ptr, self.q8, self.y + r.start, r.end - r.start, self.k);
    }
};

pub fn parallelMatvecQ80Pre(w: [*]const u8, q8: [*]const quantK.block_q8_0, y: [*]f32, m: u32, k: u32) void {
    if (m < MIN_PARALLEL_ROWS) {
        matvec.matvecQ80IntPre(w, q8, y, m, k);
        return;
    }
    const p = getPool() orelse {
        matvec.matvecQ80IntPre(w, q8, y, m, k);
        return;
    };
    var job = PreJobQ80{ .w = w, .q8 = q8, .y = y, .m = m, .k = k, .stride = (@as(usize, k) / 32) * 34 };
    p.run(PreJobQ80.work, &job);
}

const PreJob = struct {
    w: [*]const u8,
    q8: [*]const quantK.block_q8_K,
    y: [*]f32,
    m: u32,
    k: u32,
    stride: usize,

    fn work(ctx: *anyopaque, tid: u32, total: u32) void {
        const self: *PreJob = @ptrCast(@alignCast(ctx));
        const r = chunkRange(self.m, tid, total);
        if (r.end <= r.start) return;
        const row_ptr = self.w + @as(usize, r.start) * self.stride;
        int_dot.matvecQ4K_intPre(row_ptr, self.q8, self.y + r.start, r.end - r.start, self.k);
    }
};

pub fn parallelMatvecQ4KPre(w: [*]const u8, q8: [*]const quantK.block_q8_K, y: [*]f32, m: u32, k: u32) void {
    if (m < MIN_PARALLEL_ROWS) {
        int_dot.matvecQ4K_intPre(w, q8, y, m, k);
        return;
    }
    const p = getPool() orelse {
        int_dot.matvecQ4K_intPre(w, q8, y, m, k);
        return;
    };
    var job = PreJob{ .w = w, .q8 = q8, .y = y, .m = m, .k = k, .stride = (@as(usize, k) / 256) * 144 };
    p.run(PreJob.work, &job);
}
const MatmatJob = struct {
    w: [*]const u8,
    x: [*]const f32,
    y: [*]f32,
    m: u32,
    k: u32,
    t: u32,
    stride: usize,
    dtype: gguf.GGMLType,

    fn work(ctx: *anyopaque, tid: u32, total: u32) void {
        const self: *MatmatJob = @ptrCast(@alignCast(ctx));
        const r = chunkRange(self.m, tid, total);
        if (r.end <= r.start) return;
        const row_ptr = self.w + @as(usize, r.start) * self.stride;
        matvec.matmatDispatchQuantStrided(row_ptr, self.x, self.y, r.end - r.start, self.k, self.t, self.dtype, @intCast(self.m), @intCast(r.start));
    }
};

pub fn parallelMatmat(w: [*]const u8, x: [*]const f32, y: [*]f32, m: u32, k: u32, t: u32, dtype: gguf.GGMLType) void {
    if (t <= 1 or m < MIN_PARALLEL_ROWS) {
        matvec.matmatDispatchQuant(w, x, y, m, k, @max(t, 1), dtype);
        return;
    }
    const p = getPool() orelse {
        matvec.matmatDispatchQuant(w, x, y, m, k, t, dtype);
        return;
    };
    var job = MatmatJob{ .w = w, .x = x, .y = y, .m = m, .k = k, .t = t, .stride = rowStride(k, dtype), .dtype = dtype };
    p.run(MatmatJob.work, &job);
}
