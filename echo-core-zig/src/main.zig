const std = @import("std");
const ports = @import("ports/inference.zig");
const llama_bench = @import("llama/bench.zig");
const llama_port = @import("llama/port.zig");
const qwen35 = @import("llama/qwen35.zig");

const ArrayList = std.array_list.Managed;

fn formatBenchResult(allocator: std.mem.Allocator, result: anytype, json: bool) ![]u8 {
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

fn REPL(comptime Port: type) type {
    return struct {
    const Self = @This();
    port: Port,
    running: bool,

    pub fn init(model_path: []const u8, allocator: std.mem.Allocator) !Self {
        const port = try Port.init(model_path, allocator);
        return .{
            .port = port,
            .running = false,
        };
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        self.port.deinit(allocator);
    }

    pub fn run(self: *Self, io: std.Io) !void {
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

    fn printStats(self: *const Self) void {
        const cfg = self.port.getConfig();
        std.debug.print("Model Config:\n", .{});
        std.debug.print("  vocab_size: {}\n", .{cfg.vocab_size});
        std.debug.print("  hidden_dim: {}\n", .{cfg.hidden_dim});
        std.debug.print("  num_layers: {}\n", .{cfg.num_layers});
        std.debug.print("  max_seq_len: {}\n", .{cfg.max_seq_len});
    }
};
}

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
        } else if (std.mem.eql(u8, args[i], "--ngram")) {
            if (i + 1 < args.len) i += 1;
        } else if (args[i].len > 0 and args[i][0] != '-') {
            model_path = args[i];
        }
    }
    return model_path;
}

fn parseUsizeList(allocator: std.mem.Allocator, text: []const u8) ![]usize {
    var list: std.ArrayList(usize) = .empty;
    errdefer list.deinit(allocator);
    var it = std.mem.splitScalar(u8, text, ',');
    while (it.next()) |part| try list.append(allocator, try std.fmt.parseInt(usize, part, 10));
    return list.toOwnedSlice(allocator);
}

/// llama-bench ile aynı ölçüm: echo-core-zig --llama-bench -m M -t 6 -r 5 -p 128,512 -n 32,128
fn runLlamaBench(init: std.process.Init, args: []const []const u8) !void {
    var opt = llama_bench.Options{};
    var model: ?[]const u8 = null;
    var dump: ?[]const u8 = null;
    var tokens_text: ?[]const u8 = null;
    var steps_text: ?[]const u8 = null;
    var n_ctx: usize = 1024;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--llama-bench")) continue;
        i += 1;
        if (i >= args.len) return error.InvalidArgs;
        const v = args[i];
        if (std.mem.eql(u8, a, "-m")) {
            model = v;
        } else if (std.mem.eql(u8, a, "-t")) {
            opt.n_threads = try std.fmt.parseInt(u32, v, 10);
        } else if (std.mem.eql(u8, a, "-r")) {
            opt.reps = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, a, "-p")) {
            opt.prompts = try parseUsizeList(init.gpa, v);
        } else if (std.mem.eql(u8, a, "-n")) {
            opt.gens = try parseUsizeList(init.gpa, v);
        } else if (std.mem.eql(u8, a, "--lm-ivf")) {
            const lv = try parseUsizeList(init.gpa, v);
            if (lv.len != 2) return error.InvalidArgs;
            opt.lm_clusters = lv[0];
            opt.lm_nprobe = lv[1];
        } else if (std.mem.eql(u8, a, "--greedy")) {
            opt.greedy = !std.mem.eql(u8, v, "0");
        } else if (std.mem.eql(u8, a, "--gen-batch")) {
            opt.gen_batch = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, a, "--ngram")) {
            opt.ngram_k = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, a, "--grammar")) {
            opt.grammar = !std.mem.eql(u8, v, "0");
        } else if (std.mem.eql(u8, a, "--p6")) {
            opt.p6 = !std.mem.eql(u8, v, "0");
        } else if (std.mem.eql(u8, a, "--memory")) {
            opt.memory = !std.mem.eql(u8, v, "0");
        } else if (std.mem.eql(u8, a, "--min-draft")) {
            opt.min_draft = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, a, "--snap")) {
            opt.snap = !std.mem.eql(u8, v, "0");
        } else if (std.mem.eql(u8, a, "--dump-logits")) {
            dump = v;
        } else if (std.mem.eql(u8, a, "--tokens")) {
            tokens_text = v;
        } else if (std.mem.eql(u8, a, "--steps")) {
            steps_text = v;
        } else if (std.mem.eql(u8, a, "--ctx")) {
            n_ctx = try std.fmt.parseInt(usize, v, 10);
        } else return error.InvalidArgs;
    }
    const path = model orelse return error.InvalidArgs;
    var buf: [65536]u8 = undefined;
    if (dump) |dump_path| {
        const ids = try parseUsizeList(init.gpa, tokens_text orelse return error.InvalidArgs);
        const tokens = try init.gpa.alloc(u32, ids.len);
        for (ids, tokens) |id, *t| t.* = @intCast(id);
        const steps = try parseUsizeList(init.gpa, steps_text orelse return error.InvalidArgs);
        const f = try std.Io.Dir.cwd().createFile(init.io, dump_path, .{});
        defer f.close(init.io);
        var w = f.writer(init.io, &buf);
        try llama_bench.dumpLogits(path, tokens, steps, opt.n_threads, n_ctx, init.gpa, &w.interface);
        return;
    }
    var w = std.Io.File.stdout().writer(init.io, &buf);
    try llama_bench.run(path, opt, init.gpa, init.io, &w.interface);
}

pub fn main(init: std.process.Init) !void {
    var args_it = try std.process.Args.Iterator.initAllocator(init.minimal.args, init.gpa);
    defer args_it.deinit();

    var args = ArrayList([]const u8).init(init.gpa);
    defer args.deinit();
    while (args_it.next()) |arg| try args.append(arg);

    if (qwen35.supported and args.items.len > 5 and std.mem.eql(u8, args.items[1], "--ivf-eval")) {
        const m = try qwen35.Model.load(args.items[2], init.gpa);
        defer m.deinit();
        return @import("llama/ivf.zig").eval(args.items[2], m.output, args.items[3], args.items[4], try std.fmt.parseInt(usize, args.items[5], 10), init.gpa, init.io);
    }
    if (qwen35.supported and args.items.len > 1 and std.mem.eql(u8, args.items[1], "--logit-check")) {
        const save = args.items.len > 2 and std.mem.eql(u8, args.items[2], "--save");
        return @import("llama/logit_check.zig").run(save, 6, init.gpa, init.io);
    }
    if (qwen35.supported and args.items.len > 1 and std.mem.eql(u8, args.items[1], "--llama-bench")) {
        return runLlamaBench(init, args.items);
    }

    var prompt: ?[]const u8 = null;
    var max_tokens: u32 = 100;
    var bench = false;
    var ngram_k: usize = 8; // H01 varsayılan; --ngram 0 kapatır
    var grammar = true; // H06 varsayılan; --no-grammar kapatır
    var snap = true; // H13 varsayılan; --no-snap kapatır
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
        } else if (std.mem.eql(u8, args.items[i], "--ngram")) {
            if (i + 1 < args.items.len) {
                ngram_k = try std.fmt.parseInt(usize, args.items[i + 1], 10);
                i += 1;
            }
        } else if (std.mem.eql(u8, args.items[i], "--grammar")) {
            grammar = true;
        } else if (std.mem.eql(u8, args.items[i], "--no-grammar")) {
            grammar = false;
        } else if (std.mem.eql(u8, args.items[i], "--no-snap")) {
            snap = false;
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

    if (qwen35.supported and qwen35.Model.isQwen35(model_path, init.gpa)) {
        return drive(llama_port.Port, init, model_path, prompt, max_tokens, bench, json, ngram_k, grammar, snap);
    }
    return drive(ports.InferencePort, init, model_path, prompt, max_tokens, bench, json, ngram_k, grammar, snap);
}

fn drive(comptime Port: type, init: std.process.Init, model_path: []const u8, prompt: ?[]const u8, max_tokens: u32, bench: bool, json: bool, ngram_k: usize, grammar: bool, snap: bool) !void {
    if (prompt) |p| {
        var port = try Port.init(model_path, init.gpa);
        if (comptime @hasField(Port, "ngram_k")) {
            port.ngram_k = ngram_k;
            port.grammar = grammar;
            port.snap = snap;
        }
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

    var repl = try REPL(Port).init(model_path, init.gpa);
    defer repl.deinit(init.gpa);
    try repl.run(init.io);
}
