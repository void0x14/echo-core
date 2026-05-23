const std = @import("std");
const builtin = @import("builtin");

pub const NUM_WORKERS: u32 = 16;

pub const ThreadPool = struct {
    workers: [NUM_WORKERS]std.Thread = undefined,
    active: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    done: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    shutdown: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    f: *const fn (*anyopaque, u32) void = undefined,
    ctx: *anyopaque = undefined,

    inline fn spin() void {
        var i: u32 = 0;
        while (i < 16384) : (i += 1) {
            std.atomic.spinLoopHint();
        }
    }

    fn workerFn(self: *ThreadPool, tid: u32) void {
        while (self.shutdown.load(.acquire) == 0) {
            if (self.active.load(.acquire) == 0) {
                spin();
                continue;
            }
            if (tid < NUM_WORKERS) {
                self.f(self.ctx, tid);
            }
            _ = self.done.fetchAdd(1, .release);
            while (self.active.load(.acquire) != 0 and self.shutdown.load(.acquire) == 0) {
                spin();
            }
        }
    }

    fn init(self: *ThreadPool) void {
        for (0..NUM_WORKERS) |i| {
            self.workers[i] = std.Thread.spawn(.{}, workerFn, .{ self, @as(u32, @intCast(i)) }) catch @panic("tp spawn");
        }
    }

    pub fn submit(self: *ThreadPool, comptime F: anytype, args: anytype) void {
        const AT = @typeInfo(@TypeOf(args)).pointer.child;
        self.f = struct {
            fn run(d: *anyopaque, tid: u32) void {
                const t: *AT = @ptrCast(@alignCast(d));
                F(t, tid);
            }
        }.run;
        self.ctx = @ptrCast(@constCast(args));
        self.done.store(0, .release);
        self.active.store(1, .release);
        while (self.done.load(.acquire) < NUM_WORKERS) {
            std.atomic.spinLoopHint();
        }
        self.active.store(0, .release);
    }

    pub fn deinit(self: *ThreadPool) void {
        self.shutdown.store(1, .release);
        self.active.store(1, .release);
        for (&self.workers) |*t| t.join();
    }
};

var global_ptr: ?*ThreadPool = null;

pub fn get() *ThreadPool {
    if (builtin.is_test) {
        @compileError("Thread pool not available in tests");
    }
    if (global_ptr) |p| return p;
    const allocator = std.heap.page_allocator;
    const p = allocator.create(ThreadPool) catch @panic("tp alloc");
    p.init();
    global_ptr = p;
    return p;
}
