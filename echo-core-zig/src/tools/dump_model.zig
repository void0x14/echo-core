const std = @import("std");
const gguf = @import("gguf");
const config = @import("core_config");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        std.debug.print("Usage: {s} <model.gguf>\n", .{args[0]});
        return error.MissingArgument;
    }

    const model_path = args[1];
    
    var reader = try gguf.Reader.openWithAllocator(model_path, allocator);
    defer reader.deinit();

    const cfg = reader.config;
    
    // Print metadata
    std.debug.print("\n=== MODEL METADATA ===\n", .{});
    std.debug.print("Architecture prefix: {s}\n", .{reader.model_prefix});
    std.debug.print("hidden_dim (embedding_length): {d}\n", .{cfg.hidden_dim});
    std.debug.print("num_layers (block_count): {d}\n", .{cfg.num_layers});
    std.debug.print("num_heads: {d}\n", .{cfg.num_heads});
    std.debug.print("num_kv_heads: {d}\n", .{cfg.num_kv_heads});
    std.debug.print("head_dim: {d}\n", .{cfg.head_dim});
    std.debug.print("ffn_hidden_dim: {d}\n", .{cfg.ffn_hidden_dim});
    std.debug.print("vocab_size: {d}\n", .{cfg.vocab_size});
    std.debug.print("max_seq_len: {d}\n", .{cfg.max_seq_len});
    std.debug.print("\n=== SSM CONFIGURATION ===\n", .{});
    std.debug.print("ssm_conv_kernel: {d}\n", .{cfg.ssm_conv_kernel});
    std.debug.print("ssm_inner_size: {d}\n", .{cfg.ssm_inner_size});
    std.debug.print("ssm_dt_rank: {d}\n", .{cfg.ssm_dt_rank});
    std.debug.print("ssm_num_groups: {d}\n", .{cfg.ssm_num_groups});
    
    // Calculate total model size
    var total_bytes: u64 = 0;
    var tensor_count: usize = 0;
    
    std.debug.print("\n=== TENSOR SHAPES (first layer) ===\n", .{});
    
    // Check key tensors from first layer
    const key_tensors = [_][]const u8{
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight", 
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ssm_conv1d.weight",
        "blk.0.ssm_dt.weight",
        "blk.0.ssm_A.weight",
        "blk.0.ssm_B.weight",
        "blk.0.ssm_C.weight",
        "blk.0.ssm_D.weight",
        "blk.0.ssm_out.weight",
        "blk.0.ssm_x.weight",
        "token_embd.weight",
        "output_norm.weight",
        "output.weight",
    };
    
    for (key_tensors) |name| {
        if (reader.tensors.get(name)) |info| {
            std.debug.print("{s}:\n", .{name});
            std.debug.print("  dtype: {s}\n", .{@tagName(info.dtype)});
            std.debug.print("  shape: [", .{});
            for (info.shape, 0..) |dim, i| {
                if (i > 0) std.debug.print(", ", .{});
                std.debug.print("{d}", .{dim});
            }
            std.debug.print("]\n", .{});
            std.debug.print("  size: {d} bytes\n", .{info.size});
            total_bytes += info.size;
            tensor_count += 1;
        }
    }
    
    // Count all tensors
    std.debug.print("\n=== ALL TENSORS ===\n", .{});
    var it = reader.tensors.iterator();
    var all_bytes: u64 = 0;
    var all_count: usize = 0;
    while (it.next()) |entry| {
        all_bytes += entry.value_ptr.size;
        all_count += 1;
    }
    
    std.debug.print("Total tensors: {d}\n", .{all_count});
    std.debug.print("Total size: {d:.2} GB\n", .{@as(f64, @floatFromInt(all_bytes)) / (1024.0 * 1024.0 * 1024.0)});
    
    // Compare metadata vs actual
    std.debug.print("\n=== METADATA vs ACTUAL COMPARISON ===\n", .{});
    
    if (reader.tensors.get("blk.0.attn_q.weight")) |q_info| {
        if (q_info.shape.len >= 2) {
            const actual_hidden = q_info.shape[1];
            std.debug.print("attn_q.weight:\n", .{});
            std.debug.print("  metadata expects hidden_dim: {d}\n", .{cfg.hidden_dim});
            std.debug.print("  actual shape[1]: {d}\n", .{actual_hidden});
            if (actual_hidden != cfg.hidden_dim) {
                std.debug.print("  ⚠️  MISMATCH! Actual is {d}x larger\n", .{actual_hidden / cfg.hidden_dim});
            }
        }
    }
    
    if (reader.tensors.get("blk.0.ssm_conv1d.weight")) |conv_info| {
        if (conv_info.shape.len >= 2) {
            const actual_kernel = conv_info.shape[0];
            const actual_hidden = conv_info.shape[1];
            std.debug.print("\nssm_conv1d.weight:\n", .{});
            std.debug.print("  metadata expects ssm_conv_kernel: {d}\n", .{cfg.ssm_conv_kernel});
            std.debug.print("  actual shape[0]: {d}\n", .{actual_kernel});
            std.debug.print("  metadata expects hidden_dim: {d}\n", .{cfg.hidden_dim});
            std.debug.print("  actual shape[1]: {d}\n", .{actual_hidden});
            if (actual_kernel != cfg.ssm_conv_kernel) {
                std.debug.print("  ⚠️  KERNEL MISMATCH!\n", .{});
            }
            if (actual_hidden != cfg.hidden_dim) {
                std.debug.print("  ⚠️  HIDDEN_DIM MISMATCH! Actual is {d}x larger\n", .{actual_hidden / cfg.hidden_dim});
            }
        }
    }
    
    if (reader.tensors.get("blk.0.ssm_dt.weight")) |dt_info| {
        if (dt_info.shape.len >= 2) {
            const actual_dt_rank = dt_info.shape[1];
            std.debug.print("\nssm_dt.weight:\n", .{});
            std.debug.print("  metadata expects ssm_dt_rank: {d}\n", .{cfg.ssm_dt_rank});
            std.debug.print("  actual shape[1]: {d}\n", .{actual_dt_rank});
            if (actual_dt_rank != cfg.ssm_dt_rank) {
                std.debug.print("  ⚠️  DT_RANK MISMATCH!\n", .{});
            }
        }
    }
    
    if (reader.tensors.get("blk.0.ssm_A.weight")) |a_info| {
        const actual_inner = a_info.shape[a_info.shape.len - 1];
        std.debug.print("\nssm_A.weight:\n", .{});
        std.debug.print("  metadata expects ssm_inner_size: {d}\n", .{cfg.ssm_inner_size});
        std.debug.print("  actual last dim: {d}\n", .{actual_inner});
        if (actual_inner != cfg.ssm_inner_size) {
            std.debug.print("  ⚠️  INNER_SIZE MISMATCH!\n", .{});
        }
    }
}
