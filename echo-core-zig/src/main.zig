const std = @import("std");
const ports = @import("ports/inference.zig");
const config = @import("core/config.zig");

pub const REPL = struct {
    port: ports.InferencePort,
    running: bool,

    pub fn init(model_path: []const u8) !REPL {
        var port = try ports.InferencePort.init(model_path, std.heap.c_allocator);
        return .{
            .port = port,
            .running = false,
        };
    }

    pub fn deinit(self: *REPL) void {
        self.port.deinit(std.heap.c_allocator);
    }

    pub fn run(self: *REPL) !void {
        self.running = true;
        std.debug.print("Echo Core REPL\nType :quit to exit, :reset to clear KV cache\n", .{});

        const stdin = std.io.getStdIn();
        var buf: [4096]u8 = undefined;

        while (self.running) {
            std.debug.print("> ", .{});
            const n = stdin.read(&buf) catch continue;
            if (n == 0) break;

            const input = std.mem.trim(u8, buf[0..n], " \n\r");

            if (std.mem.eql(u8, input, ":quit")) {
                break;
            } else if (std.mem.eql(u8, input, ":reset")) {
                self.port.reset();
                std.debug.print("KV cache reset\n", .{});
            } else if (std.mem.eql(u8, input, ":stats")) {
                self.printStats();
            } else if (input.len > 0) {
                const output = self.port.generate(input, 100) catch |err| {
                    std.debug.print("Generation error: {}\n", .{err});
                    continue;
                };
                std.debug.print("{s}\n", .{output});
            }
        }
    }

    fn printStats(self: *const REPL) void {
        const cfg = self.port.getConfig();
        std.debug.print("Model Config:\n", .{});
        std.debug.print("  vocab_size: {}\n", .{cfg.vocab_size});
        std.debug.print("  hidden_dim: {}\n", .{cfg.hidden_dim});
        std.debug.print("  num_layers: {}\n", .{cfg.num_layers});
        std.debug.print("  max_seq_len: {}\n", .{cfg.max_seq_len});
    }
};

pub fn main() !void {
    const args = std.process.argsAlloc(std.heap.c_allocator) catch {
        std.debug.print("Usage: echo-core-zig <model.gguf>\n", .{});
        return error.InvalidArgs;
    };
    defer std.process.argsFree(std.heap.c_allocator, args);

    const model_path = if (args.len > 1) args[1] else {
        std.debug.print("Usage: echo-core-zig <model.gguf>\n", .{});
        return error.InvalidArgs;
    };

    var repl = try REPL.init(model_path);
    defer repl.deinit();
    try repl.run();
}

test {
    std.testing.refAllDecls(@This());
}
