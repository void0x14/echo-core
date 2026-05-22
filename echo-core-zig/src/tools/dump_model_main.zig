const std = @import("std");
const gguf = @import("gguf");
const core = @import("core_config");

pub fn main(init: std.process.Init) !void {
    var args_it = try std.process.Args.Iterator.initAllocator(init.minimal.args, init.gpa);
    defer args_it.deinit();

    var args = std.array_list.Managed([]const u8).init(init.gpa);
    defer args.deinit();
    while (args_it.next()) |arg| try args.append(arg);

    if (args.items.len < 2) {
        std.debug.print("Usage: {s} <model.gguf>\n", .{args.items[0]});
        return error.MissingArgument;
    }

    try dumpModel(args.items[1], init.gpa);
}

fn dumpModel(model_path: []const u8, allocator: std.mem.Allocator) !void {
    const TensorInfo = gguf.TensorInfo;

    const printShape = struct {
        fn call(name: []const u8, info: *const TensorInfo) void {
            std.debug.print("  {s}: shape=", .{name});
            for (info.shape, 0..) |dim, j| {
                if (j > 0) std.debug.print("x", .{});
                std.debug.print("{d}", .{dim});
            }
            std.debug.print(" dtype={s} size={d}\n", .{ @tagName(info.dtype), info.size });
        }
    }.call;

    var reader = try gguf.Reader.openWithAllocator(model_path, allocator);
    defer reader.deinit();

    const cfg = reader.config;

    std.debug.print("\n=== MODEL METADATA ===\n", .{});
    std.debug.print("Architecture prefix: {s}\n", .{reader.model_prefix});
    std.debug.print("hidden_dim: {d}\n", .{cfg.hidden_dim});
    std.debug.print("num_layers: {d}\n", .{cfg.num_layers});
    std.debug.print("num_heads: {d}, num_kv_heads: {d}\n", .{ cfg.num_heads, cfg.num_kv_heads });
    std.debug.print("head_dim: {d}\n", .{cfg.head_dim});
    std.debug.print("ffn_hidden_dim: {d}\n", .{cfg.ffn_hidden_dim});
    std.debug.print("vocab_size: {d}\n", .{cfg.vocab_size});
    std.debug.print("max_seq_len: {d}\n", .{cfg.max_seq_len});
    std.debug.print("full_attention_interval: {d}\n", .{cfg.full_attention_interval});
    std.debug.print("\n=== SSM CONFIG ===\n", .{});
    std.debug.print("ssm_conv_kernel: {d}\n", .{cfg.ssm_conv_kernel});
    std.debug.print("ssm_inner_size: {d}\n", .{cfg.ssm_inner_size});
    std.debug.print("ssm_dt_rank: {d}\n", .{cfg.ssm_dt_rank});
    std.debug.print("ssm_num_groups: {d}\n", .{cfg.ssm_num_groups});

    var it = reader.tensors.iterator();
    var all_bytes: u64 = 0;
    var all_count: usize = 0;
    while (it.next()) |entry| {
        all_bytes += entry.value_ptr.size;
        all_count += 1;
    }

    std.debug.print("\n=== TENSORS ===\n", .{});
    std.debug.print("Total tensors: {d}\n", .{all_count});
    std.debug.print("Total size: {d:.2} GB\n", .{@as(f64, @floatFromInt(all_bytes)) / (1024.0 * 1024.0 * 1024.0)});

    var it2 = reader.tensors.iterator();
    while (it2.next()) |entry| {
        const name = entry.key_ptr.*;
        const info = entry.value_ptr;
        if (std.mem.indexOf(u8, name, "ssm_conv1d") != null) {
            printShape(name, info);
        }
        if (std.mem.eql(u8, name, "blk.0.ssm_out.weight")) {
            printShape(name, info);
        }
        if (std.mem.eql(u8, name, "blk.0.ssm_alpha.weight") or std.mem.eql(u8, name, "blk.0.ssm_beta.weight")) {
            printShape(name, info);
        }
        if (std.mem.eql(u8, name, "blk.0.ssm_a")) {
            printShape(name, info);
        }
        if (std.mem.eql(u8, name, "blk.0.ssm_dt.bias")) {
            printShape(name, info);
        }
        if (std.mem.eql(u8, name, "blk.0.ssm_norm.weight")) {
            printShape(name, info);
        }
        if (std.mem.indexOf(u8, name, "post_attention_norm") != null) {
            printShape(name, info);
        }
    }
}
