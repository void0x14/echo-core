const std = @import("std");
const builtin = @import("builtin");

const NUM_WORKERS: u32 = 16;

fn spin() void { var i: u32 = 0; while (i < 256) : (i += 1) { std.atomic.spinLoopHint(); } }

pub const ThreadPool = struct {
    workers: [NUM_WORKERS]std.Thread = undefined,
    active: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    done: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    shutdown: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    f: *const fn (*anyopaque, u32) void = undefined,
    ctx: *anyopaque = undefined,

    fn workerFn(self: *ThreadPool, tid: u32) void {
        while (self.shutdown.load(.acquire) == 0) {
            while (self.active.load(.acquire) == 0 and self.shutdown.load(.acquire) == 0) spin();
            if (self.shutdown.load(.acquire) != 0) return;
            self.f(self.ctx, tid);
            _ = self.done.fetchAdd(1, .release);
        }
    }
    fn init() ThreadPool {
        var self = ThreadPool{};
        for (0..NUM_WORKERS) |i| {
            self.workers[i] = std.Thread.spawn(.{}, workerFn, .{ &self, @as(u32, @intCast(i)) }) catch @panic("thread spawn");
        }
        return self;
    }
    pub fn submit(self: *ThreadPool, comptime F: anytype, args: anytype) void {
        const AT = @typeInfo(@TypeOf(args)).pointer.child;
        self.f = struct { fn run(d: *anyopaque, tid: u32) void { const t: *AT = @ptrCast(@alignCast(d)); F(t, tid); } }.run;
        self.ctx = @ptrCast(@constCast(args));
        self.active.store(1, .release);
        self.done.store(0, .release);
        while (self.done.load(.acquire) < NUM_WORKERS) spin();
        self.active.store(0, .release);
    }
};

var global: ThreadPool = undefined;
var inited = false;
pub fn get() *ThreadPool {
    if (!inited) { global = ThreadPool.init(); inited = true; }
    return &global;
}
