//! ggml_threadpool: tüm thread'ler graph'ın tamamını yürütür, her node sonunda ggml_barrier.
//! Port: ggml-cpu.c (ggml_barrier, ggml_threadpool_chunk_set/add, ggml_graph_compute_thread).
const std = @import("std");
const linux = std.os.linux;
const Atomic = std.atomic.Value;

const SPIN_BEFORE_SLEEP = 300_000; // GOMP_SPINCOUNT varsayılanı

pub const Pool = struct {
    threads: []std.Thread,
    n_threads: u32,
    allocator: std.mem.Allocator,

    generation: Atomic(u32) align(64) = .init(0),
    sleepers: Atomic(u32) = .init(0),
    shutdown: bool = false,
    job_fn: *const fn (ctx: *anyopaque, ith: u32) void = undefined,
    job_ctx: *anyopaque = undefined,

    n_barrier: Atomic(u32) align(64) = .init(0),
    n_barrier_passed: Atomic(u32) align(64) = .init(0),
    current_chunk: Atomic(i32) align(64) = .init(0),

    pub fn init(allocator: std.mem.Allocator, n_threads: u32) !*Pool {
        const self = try allocator.create(Pool);
        errdefer allocator.destroy(self);
        self.* = .{
            .threads = try allocator.alloc(std.Thread, n_threads - 1),
            .n_threads = n_threads,
            .allocator = allocator,
        };
        for (self.threads, 0..) |*t, i| {
            t.* = try std.Thread.spawn(.{}, worker, .{ self, @as(u32, @intCast(i + 1)) });
        }
        return self;
    }

    pub fn deinit(self: *Pool) void {
        self.shutdown = true;
        _ = self.generation.fetchAdd(1, .release);
        _ = linux.futex_3arg(&self.generation.raw, .{ .cmd = .WAKE, .private = true }, 0x7fffffff);
        for (self.threads) |t| t.join();
        self.allocator.free(self.threads);
        self.allocator.destroy(self);
    }

    fn worker(self: *Pool, ith: u32) void {
        var seen: u32 = 0;
        while (true) {
            var spins: u32 = 0;
            while (self.generation.load(.acquire) == seen) {
                if (spins < SPIN_BEFORE_SLEEP) {
                    std.atomic.spinLoopHint();
                    spins += 1;
                } else {
                    _ = self.sleepers.fetchAdd(1, .seq_cst);
                    if (self.generation.load(.acquire) == seen) {
                        _ = linux.futex_4arg(&self.generation.raw, .{ .cmd = .WAIT, .private = true }, seen, null);
                    }
                    _ = self.sleepers.fetchSub(1, .seq_cst);
                }
            }
            seen = self.generation.load(.acquire);
            if (self.shutdown) return;
            self.job_fn(self.job_ctx, ith);
        }
    }

    /// İşi tüm thread'lerde başlatır; iş fonksiyonu son adımda barrier() çağırmalı.
    pub fn run(self: *Pool, ctx: *anyopaque, f: *const fn (ctx: *anyopaque, ith: u32) void) void {
        self.job_fn = f;
        self.job_ctx = ctx;
        _ = self.generation.fetchAdd(1, .seq_cst);
        if (self.sleepers.load(.seq_cst) > 0) {
            _ = linux.futex_3arg(&self.generation.raw, .{ .cmd = .WAKE, .private = true }, 0x7fffffff);
        }
        f(ctx, 0);
    }

    pub inline fn barrier(self: *Pool) void {
        const n = self.n_threads;
        if (n == 1) return;
        const passed = self.n_barrier_passed.load(.monotonic);
        const arrived = self.n_barrier.fetchAdd(1, .seq_cst);
        if (arrived == n - 1) {
            self.n_barrier.store(0, .monotonic);
            _ = self.n_barrier_passed.fetchAdd(1, .seq_cst);
            return;
        }
        // H46: uzun beklemede çekirdeği bırak (kirli ortamda önalınan thread'imiz daha çabuk döner)
        var spins: u32 = 0;
        while (self.n_barrier_passed.load(.monotonic) == passed) {
            spins += 1;
            if (spins & 2047 == 0) std.Thread.yield() catch {} else std.atomic.spinLoopHint();
        }
        // ggml: exit barrier (full seq-cst fence)
        _ = self.n_barrier_passed.fetchAdd(0, .seq_cst);
    }

    pub inline fn chunkSet(self: *Pool, v: i32) void {
        self.current_chunk.store(v, .monotonic);
    }
    pub inline fn chunkAdd(self: *Pool, v: i32) i32 {
        return self.current_chunk.fetchAdd(v, .monotonic);
    }
};

/// ggml_compute_params
pub const Params = struct {
    ith: u32,
    nth: u32,
    pool: *Pool,
    /// mul_mat: from_float çıktısı (paylaşımlı)
    wdata: [*]u8,
    /// flash_attn / gdn / rope: thread başına scratch (aynı buffer, ith ile bölünür)
    fdata: [*]f32,

    pub inline fn barrier(self: *const Params) void {
        self.pool.barrier();
    }
    /// dr = (nr+nth-1)/nth; ir0 = dr*ith; ir1 = min(ir0+dr, nr)
    pub inline fn range(self: *const Params, nr: usize) struct { usize, usize } {
        const dr = (nr + self.nth - 1) / self.nth;
        const ir0 = dr * self.ith;
        return .{ @min(ir0, nr), @min(ir0 + dr, nr) };
    }
};
