const std = @import("std");
const ports = @import("ports/inference.zig");

const ArrayList = std.array_list.Managed;

fn formatBenchResult(allocator: std.mem.Allocator, result: ports.InferencePort.BenchResult, json: bool) ![]u8 {
    const pp_tps: f64 = if (result.prefill_ms > 0)
        @as(f64, @floatFromInt(result.prompt_tokens)) * 1000.0 / @as(f64, @floatFromInt(result.prefill_ms))
    else
        0;
    const tg_tps: f64 = if (result.decode_ms > 0)
        @as(f64, @floatFromInt(result.generated_tokens)) * 1000.0 / @as(f64, @floatFromInt(result.decode_ms))
    else
        0;

    if (json) {
        return std.fmt.allocPrint(
            allocator,
            "{{\"prompt_tokens\":{d},\"generated_tokens\":{d},\"prefill_ms\":{d},\"decode_ms\":{d},\"pp_tps\":{d:.2},\"tg_tps\":{d:.2}}}",
            .{ result.prompt_tokens, result.generated_tokens, result.prefill_ms, result.decode_ms, pp_tps, tg_tps },
        );
    }

    return std.fmt.allocPrint(
        allocator,
        "Output: {s}\nPrompt tokens: {d}\nGenerated tokens: {d}\nPrefill: {d} ms ({d:.2} tok/s)\nDecode: {d} ms ({d:.2} tok/s)\n",
        .{ result.text, result.prompt_tokens, result.generated_tokens, result.prefill_ms, pp_tps, result.decode_ms, tg_tps },
    );
}

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
        } else if (std.mem.eql(u8, args[i], "--prompt") or std.mem.eql(u8, args[i], "-p")) {
            if (i + 1 < args.len) {
                i += 1;
            }
        } else if (std.mem.eql(u8, args[i], "--max-tokens") or std.mem.eql(u8, args[i], "-n")) {
            if (i + 1 < args.len) {
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

    var prompt: ?[]const u8 = null;
    var max_tokens: u32 = 100;
    var bench = false;
    var json = false;
    var i: usize = 1;
    while (i < args.items.len) : (i += 1) {
        if (std.mem.eql(u8, args.items[i], "--prompt") or std.mem.eql(u8, args.items[i], "-p")) {
            if (i + 1 < args.items.len) {
                prompt = args.items[i + 1];
                i += 1;
            }
        } else if (std.mem.eql(u8, args.items[i], "--bench")) {
            bench = true;
        } else if (std.mem.eql(u8, args.items[i], "--json")) {
            json = true;
        } else if (std.mem.eql(u8, args.items[i], "--max-tokens") or std.mem.eql(u8, args.items[i], "-n")) {
            if (i + 1 < args.items.len) {
                max_tokens = try std.fmt.parseInt(u32, args.items[i + 1], 10);
                i += 1;
            }
        }
    }

    const model_path = parseModelPath(args.items) orelse {
        std.debug.print("Usage: echo-core-zig <model.gguf> [--prompt \"text\"] [--max-tokens N] [--bench] [--json]\n", .{});
        return error.InvalidArgs;
    };

    if (prompt) |p| {
        var port = try ports.InferencePort.init(model_path, init.gpa);
        defer port.deinit(init.gpa);

        if (bench) {
            const result = try port.benchmark(p, max_tokens, init.io);
            defer port.freeGenerated(result.text);

            const rendered = try formatBenchResult(init.gpa, result, json);
            defer init.gpa.free(rendered);
            std.debug.print("{s}", .{rendered});
            return;
        }

        const start = std.Io.Timestamp.now(init.io, .awake);
        const output = port.generate(p, max_tokens) catch |err| {
            std.debug.print("Generation error: {}\n", .{err});
            return;
        };
        const elapsed = start.untilNow(init.io, .awake);
        defer port.freeGenerated(output);

        std.debug.print("Output: {s}\n", .{output});
        std.debug.print("Time elapsed: {} ms\n", .{elapsed.toMilliseconds()});
        return;
    }

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

test "parseModelPath ignores benchmark option values" {
    try std.testing.expectEqualStrings(
        "model.gguf",
        parseModelPath(&.{
            "echo-core-zig",
            "--prompt",
            "Hello",
            "--max-tokens",
            "4",
            "--json",
            "--bench",
            "--model",
            "model.gguf",
        }).?,
    );
}

test "format bench result as json" {
    const result: ports.InferencePort.BenchResult = .{
        .text = @constCast("Helloorphic"),
        .prompt_tokens = 2,
        .generated_tokens = 1,
        .prefill_ms = 24636,
        .decode_ms = 12257,
    };

    const rendered = try formatBenchResult(std.testing.allocator, result, true);
    defer std.testing.allocator.free(rendered);

    try std.testing.expect(std.mem.indexOf(u8, rendered, "\"prompt_tokens\":2") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "\"generated_tokens\":1") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "\"prefill_ms\":24636") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "\"decode_ms\":12257") != null);
}

test {
    std.testing.refAllDecls(@This());
}
