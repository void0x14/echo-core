const std = @import("std");

pub var pool: ?ThreadPool = null;

pub const Task = struct {
    ptr: ?*anyopaque = null,
    run: *const fn (ctx: ?*anyopaque, tid: u32) void,
};

pub const ThreadPool = struct {
    threads: []std.Thread,
    cond: std.Thread.Cond,
    mutex: std.Thread.Mutex,
    task: Task = .{ .ptr = null, .run = undefined },
    counter: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    total: u32 = 0,
    task_done: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    running: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    shutdown: bool = false,

    fn worker(self: *ThreadPool, tid: u32) void {
        while (true) {
            self.mutex.lock();
            while (!self.running.load(.acquire) and !self.shutdown) {
                self.cond.wait(&self.mutex);
            }
            if (self.shutdown) {
                self.mutex.unlock();
                return;
            }
            const my_task = self.task;
            self.mutex.unlock();

            self.task.run(my_task.ptr, tid);

            _ = self.task_done.fetchAdd(1, .release);
        }
    }

    pub fn init(allocator: std.mem.Allocator, num_threads: u32) !ThreadPool {
        var threads = try allocator.alloc(std.Thread, num_threads);
        errdefer allocator.free(threads);

        var tp = ThreadPool{
            .threads = threads,
            .cond = std.Thread.Cond{},
            .mutex = std.Thread.Mutex{},
        };

        for (0..threads.len) |i| {
            threads[i] = try std.Thread.spawn(.{}, worker, .{ @constCast(&tp), @as(u32, @intCast(i)) });
        }

        return tp;
    }

    pub fn submit(self: *ThreadPool, comptime runFn: anytype, ctx: *@TypeOf(runFn).Args) void {
        self.task_done.store(0, .release);
        self.task = .{
            .ptr = @ptrCast(ctx),
            .run = struct {
                fn wrapper(c: ?*anyopaque, tid: u32) void {
                    const typed: *@TypeOf(runFn).Args = @ptrCast(@alignCast(c));
                    @call(.auto, runFn, .{ typed, tid });
                }
            }.wrapper,
        };
        self.running.store(true, .release);
        self.mutex.lock();
        self.cond.broadcast();
        self.mutex.unlock();

        while (self.task_done.load(.acquire) < self.threads.len) {
            std.time.sleep(1);
        }
        self.running.store(false, .release);
    }

    pub fn deinit(self: *ThreadPool) void {
        self.shutdown = true;
        self.mutex.lock();
        self.cond.broadcast();
        self.mutex.unlock();
        for (self.threads) |t| t.join();
        std.posix.mem.free(self.threads.ptr, self.threads.len);
    }
};
