const std = @import("std");
const ports = @import("ports/inference.zig");

const ArrayList = std.array_list.Managed;

pub const REPL = struct {
    port: ports.InferencePort,
    running: bool,

    pub fn init(model_path: []const u8, allocator: std.mem.Allocator) !REPL {
        const port = try ports.InferencePort.init(model_path, allocator);
        return .{
            .port = port,
            .running = false,
        };
    }

    pub fn deinit(self: *REPL, allocator: std.mem.Allocator) void {
        self.port.deinit(allocator);
    }

    pub fn run(self: *REPL, io: std.Io) !void {
        self.running = true;
        std.debug.print("Echo Core REPL\nType :quit to exit, :reset to clear KV cache\n", .{});

        var buf: [4096]u8 = undefined;
        var stdin_reader = std.Io.File.stdin().reader(io, &buf);

        while (self.running) {
            std.debug.print("> ", .{});
            const maybe_line = stdin_reader.interface.takeDelimiter('\n') catch |err| switch (err) {
                error.StreamTooLong => continue,
                else => return err,
            };

            const input = std.mem.trim(u8, maybe_line orelse break, " \n\r\t");
            if (input.len == 0) continue;

            if (std.mem.eql(u8, input, ":quit")) {
                break;
            } else if (std.mem.eql(u8, input, ":reset")) {
                self.port.reset();
                std.debug.print("KV cache reset\n", .{});
            } else if (std.mem.eql(u8, input, ":stats")) {
                self.printStats();
            } else {
                const output = self.port.generate(input, 100) catch |err| {
                    std.debug.print("Generation error: {}\n", .{err});
                    continue;
                };
                defer self.port.freeGenerated(output);
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

fn parseModelPath(args: []const []const u8) ?[]const u8 {
    var model_path: ?[]const u8 = null;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--model") or std.mem.eql(u8, args[i], "-m")) {
            if (i + 1 < args.len) {
                model_path = args[i + 1];
                i += 1;
            }
        } else if (args[i].len > 0 and args[i][0] != '-') {
            model_path = args[i];
        }
    }
    return model_path;
}

pub fn main(init: std.process.Init) !void {
    var args_it = try std.process.Args.Iterator.initAllocator(init.minimal.args, init.gpa);
    defer args_it.deinit();

    var args = ArrayList([]const u8).init(init.gpa);
    defer args.deinit();
    while (args_it.next()) |arg| try args.append(arg);

    const model_path = parseModelPath(args.items) orelse {
        std.debug.print("Usage: echo-core-zig <model.gguf>\n", .{});
        return error.InvalidArgs;
    };

    var repl = try REPL.init(model_path, init.gpa);
    defer repl.deinit(init.gpa);
    try repl.run(init.io);
}

test "parseModelPath positional" {
    try std.testing.expectEqualStrings("model.gguf", parseModelPath(&.{ "echo-core-zig", "model.gguf" }).?);
}

test "parseModelPath flag form" {
    try std.testing.expectEqualStrings("model.gguf", parseModelPath(&.{ "echo-core-zig", "--model", "model.gguf" }).?);
    try std.testing.expectEqualStrings("other.gguf", parseModelPath(&.{ "echo-core-zig", "-m", "other.gguf" }).?);
}

test {
    std.testing.refAllDecls(@This());
}
