const std = @import("std");
const types = @import("types.zig");
const config = @import("config.zig");
const gguf = @import("../gguf/reader.zig");

/// Compute byte size for a tensor given its shape and dtype
pub fn computeTensorQuantizedBytes(shape: []const u64, dtype: gguf.GGMLType) usize {
    var n_elements: u64 = 1;
    for (shape) |dim| n_elements *= dim;

    return switch (dtype) {
        .f32 => n_elements * 4,
        .f16, .bf16 => n_elements * 2,
        .f64 => n_elements * 8,
        .i8 => n_elements,
        .i16 => n_elements * 2,
        .i32 => n_elements * 4,
        .i64 => n_elements * 8,
        else => blk: {
            // Quantized types: calculate blocks
            const block_bytes = blockSizeBytes(dtype);
            const block_elems = blockElements(dtype);
            if (block_bytes == 0 or block_elems == 0) unreachable;
            const n_blocks = (n_elements + block_elems - 1) / block_elems;
            break :blk n_blocks * block_bytes;
        },
    };
}

fn blockSizeBytes(dtype: gguf.GGMLType) u64 {
    return switch (dtype) {
        .q4_0 => 18, // 32 elements, 16 bytes quants + 2 bytes scale
        .q4_1 => 20, // 32 elements, 16 bytes quants + 2 bytes scale + 2 bytes min
        .q5_0 => 22, // 32 elements, 20 bytes quants + 2 bytes scale
        .q5_1 => 24, // 32 elements, 20 bytes quants + 2 bytes scale + 2 bytes min
        .q8_0 => 34, // 32 elements, 32 bytes quants + 2 bytes scale
        .q8_1 => 36, // 32 elements, 32 bytes quants + 2 bytes scale + 2 bytes min
        .q2_k => 84, // 256 elements, super-block
        .q3_k => 110, // 256 elements, super-block
        .q4_k => 144, // 256 elements, super-block
        .q5_k => 176, // 256 elements, super-block
        .q6_k => 210, // 256 elements, super-block
        .q8_k => 34, // 32 elements (note: different from Q8_0)
        .iq2_xxs => 66,
        .iq2_xs => 74,
        .iq2_s => 74,
        .iq3_xxs => 102,
        .iq3_s => 109,
        .iq1_s => 42,
        .iq1_m => 58,
        .iq4_nl => 144,
        .iq4_xs => 136,
        .tq1_0 => 34,
        .tq2_0 => 66,
        .mxfp4 => 32,
        else => 0,
    };
}

fn blockElements(dtype: gguf.GGMLType) u64 {
    return switch (dtype) {
        .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q8_1, .q8_k => 32,
        .q2_k, .q3_k, .q4_k, .q5_k, .q6_k => 256,
        .iq2_xxs, .iq2_xs, .iq2_s, .iq3_xxs, .iq3_s, .iq1_s, .iq1_m, .iq4_nl, .iq4_xs => 256,
        .tq1_0, .tq2_0, .mxfp4 => 32,
        else => 1,
    };
}

pub const AlignedMemoryPool = struct {
    base: [*]u8,
    heap_bytes: ?[]align(types.CACHE_LINE_SIZE) u8,
    mmap_bytes: ?[]align(std.heap.page_size_min) u8,
    total_size: usize,
    offset: usize,
    owns_mmap: bool,
    alignment: u32,
    page_delta: usize,

    pub fn init(total_bytes: usize) !AlignedMemoryPool {
        if (total_bytes == 0) {
            return AlignedMemoryPool{
                .base = undefined,
                .heap_bytes = null,
                .mmap_bytes = null,
                .total_size = 0,
                .offset = 0,
                .owns_mmap = false,
                .alignment = 32,
                .page_delta = 0,
            };
        }
        const bytes = try std.heap.page_allocator.alignedAlloc(
            u8,
            std.mem.Alignment.fromByteUnits(types.CACHE_LINE_SIZE),
            total_bytes,
        );
        return AlignedMemoryPool{
            .base = bytes.ptr,
            .heap_bytes = bytes,
            .mmap_bytes = null,
            .total_size = total_bytes,
            .offset = 0,
            .owns_mmap = false,
            .alignment = 32,
            .page_delta = 0,
        };
    }

    pub fn initMmap(fd: std.fs.File, data_offset: u64, data_size: usize, alignment: u32) !AlignedMemoryPool {
        if (data_size == 0) {
            return AlignedMemoryPool{
                .base = undefined,
                .heap_bytes = null,
                .mmap_bytes = null,
                .total_size = 0,
                .offset = 0,
                .owns_mmap = false,
                .alignment = alignment,
                .page_delta = 0,
            };
        }
        const page_size = std.heap.pageSize();
        const aligned_off = data_offset & ~(page_size - 1);
        const page_delta = data_offset - aligned_off;
        const map_size = data_size + page_delta;
        const ptr = try std.posix.mmap(null, map_size, std.posix.PROT.READ, .{ .TYPE = .PRIVATE }, fd.handle, aligned_off);
        return AlignedMemoryPool{
            .base = ptr.ptr,
            .heap_bytes = null,
            .mmap_bytes = ptr,
            .total_size = map_size,
            .offset = 0,
            .owns_mmap = true,
            .alignment = alignment,
            .page_delta = page_delta,
        };
    }

    pub fn deinit(self: *AlignedMemoryPool) void {
        if (self.total_size == 0) return;
        if (self.owns_mmap) {
            if (self.mmap_bytes) |bytes| std.posix.munmap(bytes);
        } else if (self.heap_bytes) |bytes| {
            std.heap.page_allocator.free(bytes);
        }
        self.base = undefined;
        self.heap_bytes = null;
        self.mmap_bytes = null;
        self.total_size = 0;
    }

    fn dataPtr(self: *const AlignedMemoryPool) [*]u8 {
        return self.base + self.page_delta;
    }

    pub fn alloc(self: *AlignedMemoryPool, comptime T: type, count: usize) [*]T {
        const align_bytes: usize = if (@alignOf(T) > types.CACHE_LINE_SIZE) @alignOf(T) else types.CACHE_LINE_SIZE;
        self.offset = (self.offset + align_bytes - 1) & ~(align_bytes - 1);
        const bytes = count * @sizeOf(T);
        std.debug.assert(self.offset + bytes + self.page_delta <= self.total_size);
        const ptr: [*]T = @ptrCast(@alignCast(self.dataPtr() + self.offset));
        self.offset += bytes;
        return ptr;
    }

    pub fn at(self: *const AlignedMemoryPool, comptime T: type, byte_offset: usize) *const T {
        std.debug.assert(byte_offset + @sizeOf(T) + self.page_delta <= self.total_size);
        return @ptrCast(@alignCast(self.dataPtr() + byte_offset));
    }

    pub fn atMut(self: *AlignedMemoryPool, comptime T: type, byte_offset: usize) *T {
        std.debug.assert(byte_offset + @sizeOf(T) + self.page_delta <= self.total_size);
        return @ptrCast(@alignCast(self.dataPtr() + byte_offset));
    }

    pub fn bytesUsed(self: *const AlignedMemoryPool) usize {
        return self.offset;
    }

    pub fn bytesTotal(self: *const AlignedMemoryPool) usize {
        return self.total_size - self.page_delta;
    }

    pub fn getAlignment(self: *const AlignedMemoryPool) u32 {
        return self.alignment;
    }
};

pub const WeightLayout = struct {
    token_embedding_offset: usize,
    token_embedding_size: usize,
    norm_weight_offset: usize,
    q_proj_offset: usize,
    k_proj_offset: usize,
    v_proj_offset: usize,
    o_proj_offset: usize,
    ffn_norm_offset: usize,
    ffn_weight1_offset: usize,
    ffn_weight2_offset: usize,
    ffn_weight3_offset: usize,
    attn_q_norm_offset: usize,
    attn_k_norm_offset: usize,

    // SSM weight offsets
    ssm_out_offset: usize,
    ssm_x_offset: usize,
    ssm_dt_offset: usize,
    ssm_A_offset: usize,
    ssm_B_offset: usize,
    ssm_C_offset: usize,
    ssm_D_offset: usize,
    ssm_conv1d_offset: usize,
    ssm_conv1d_bias_offset: usize,
    ssm_region_offset: usize, // Start of SSM weights region (after output projection)

    per_layer_size: usize,
    ssm_per_layer_size: usize,
    final_norm_offset: usize,
    output_proj_offset: usize,
    total_size: usize,
    raw_pool_size: usize,

    // Per-layer type tracking (dynamically allocated)
    layer_types: []config.ModelConfig.LayerType,
    layer_offsets: []usize, // Cumulative offset for each layer

    pub fn compute(config_: anytype, reader_opt: ?*const gguf.Reader, allocator: std.mem.Allocator) !WeightLayout {
        const hidden = config_.hidden_dim;
        const vocab = config_.vocab_size;
        const kv_dim = config_.num_kv_heads * config_.head_dim;
        const ffn_h = config_.ffn_hidden_dim;
        const sizeof_fp16 = @sizeOf(types.fp16_t);

        // Allocate layer types array
        const layer_types = try allocator.alloc(config.ModelConfig.LayerType, config_.num_layers);
        errdefer allocator.free(layer_types);

        // Allocate layer offsets array
        const layer_offsets = try allocator.alloc(usize, config_.num_layers);
        errdefer allocator.free(layer_offsets);

        // Default all layers to attention initially
        // Will be updated by loader based on tensor presence
        @memset(layer_types, .attention);

        var layout: WeightLayout = undefined;
        layout.layer_types = layer_types;
        layout.layer_offsets = layer_offsets;

        // Helper to get actual tensor size or fallback to FP16
        const getTensorBytes = struct {
            fn call(r: ?*const gguf.Reader, tensor_name: []const u8, fallback_elements: u64) usize {
                if (r) |reader| {
                    if (reader.tensors.get(tensor_name)) |tensor_info| {
                        return computeTensorQuantizedBytes(tensor_info.shape, tensor_info.dtype);
                    }
                }
                return fallback_elements * sizeof_fp16;
            }
        }.call;

        // Helper to get per-layer tensor size by looking up layer 0
        // Supports both old naming (attn_q, attn_k, attn_v, attn_output) and new naming (attn_qkv, attn_gate)
        const getLayerTensorBytes = struct {
            fn call(r: ?*const gguf.Reader, suffix: []const u8, fallback_elements: u64) usize {
                if (r) |reader| {
                    var name_buf: [64]u8 = undefined;

                    // Try the requested suffix first
                    const tensor_name = std.fmt.bufPrint(&name_buf, "blk.0.{s}", .{suffix}) catch return fallback_elements * @sizeOf(types.fp16_t);
                    if (reader.tensors.get(tensor_name)) |tensor_info| {
                        return computeTensorQuantizedBytes(tensor_info.shape, tensor_info.dtype);
                    }

                    // Special cases: handle naming differences
                    if (std.mem.eql(u8, suffix, "attn_q.weight")) {
                        // Try fused QKV instead of separate Q
                        const qkv_name = std.fmt.bufPrint(&name_buf, "blk.0.attn_qkv.weight", .{}) catch return fallback_elements * @sizeOf(types.fp16_t);
                        if (reader.tensors.get(qkv_name)) |tensor_info| {
                            // Fused QKV contains all 3 projections, so return appropriate portion
                            return computeTensorQuantizedBytes(tensor_info.shape, tensor_info.dtype);
                        }
                    }

                    if (std.mem.eql(u8, suffix, "attn_k.weight") or std.mem.eql(u8, suffix, "attn_v.weight")) {
                        // If model uses fused QKV, K and V are included in that tensor
                        // Return 0 since they're not separate
                        const qkv_name = std.fmt.bufPrint(&name_buf, "blk.0.attn_qkv.weight", .{}) catch return 0;
                        if (reader.tensors.get(qkv_name)) |_| {
                            return 0; // Included in QKV
                        }
                    }

                    if (std.mem.eql(u8, suffix, "attn_output.weight")) {
                        // Try attn_gate instead of attn_output
                        const gate_name = std.fmt.bufPrint(&name_buf, "blk.0.attn_gate.weight", .{}) catch return fallback_elements * @sizeOf(types.fp16_t);
                        if (reader.tensors.get(gate_name)) |tensor_info| {
                            return computeTensorQuantizedBytes(tensor_info.shape, tensor_info.dtype);
                        }
                    }

                    // SSM tensor name mappings for Qwen3.5 models
                    if (std.mem.eql(u8, suffix, "ssm_A.weight")) {
                        // Try ssm_a (lowercase, no .weight extension)
                        const ssm_a_name = std.fmt.bufPrint(&name_buf, "blk.0.ssm_a", .{}) catch return fallback_elements * @sizeOf(types.fp16_t);
                        if (reader.tensors.get(ssm_a_name)) |tensor_info| {
                            return computeTensorQuantizedBytes(tensor_info.shape, tensor_info.dtype);
                        }
                    }

                    if (std.mem.eql(u8, suffix, "ssm_B.weight")) {
                        // Try ssm_beta instead of ssm_B
                        const beta_name = std.fmt.bufPrint(&name_buf, "blk.0.ssm_beta.weight", .{}) catch return fallback_elements * @sizeOf(types.fp16_t);
                        if (reader.tensors.get(beta_name)) |tensor_info| {
                            return computeTensorQuantizedBytes(tensor_info.shape, tensor_info.dtype);
                        }
                    }

                    if (std.mem.eql(u8, suffix, "ssm_C.weight")) {
                        // Try ssm_alpha instead of ssm_C
                        const alpha_name = std.fmt.bufPrint(&name_buf, "blk.0.ssm_alpha.weight", .{}) catch return fallback_elements * @sizeOf(types.fp16_t);
                        if (reader.tensors.get(alpha_name)) |tensor_info| {
                            return computeTensorQuantizedBytes(tensor_info.shape, tensor_info.dtype);
                        }
                    }

                    if (std.mem.eql(u8, suffix, "ssm_x.weight")) {
                        // ssm_x doesn't exist in this model - fused into other tensors
                        // Return 0 since it's not needed
                        return 0;
                    }
                }
                return fallback_elements * @sizeOf(types.fp16_t);
            }
        }.call;

        layout.token_embedding_offset = 0;
        layout.token_embedding_size = getTensorBytes(reader_opt, "token_embd.weight", vocab * hidden);

        var offset: usize = 0;
        layout.norm_weight_offset = offset;
        offset += hidden * sizeof_fp16; // attn_norm is usually FP16/FP32

        layout.q_proj_offset = offset;
        const q_dim = config_.num_heads * config_.head_dim;
        offset += getLayerTensorBytes(reader_opt, "attn_q.weight", hidden * q_dim);

        layout.k_proj_offset = offset;
        offset += getLayerTensorBytes(reader_opt, "attn_k.weight", hidden * kv_dim);

        layout.v_proj_offset = offset;
        offset += getLayerTensorBytes(reader_opt, "attn_v.weight", hidden * kv_dim);

        layout.o_proj_offset = offset;
        offset += getLayerTensorBytes(reader_opt, "attn_output.weight", q_dim * hidden);

        layout.ffn_norm_offset = offset;
        offset += hidden * sizeof_fp16; // ffn_norm is usually FP16/FP32

        layout.ffn_weight1_offset = offset;
        if (config_.ffn_type == .dense) {
            offset += getLayerTensorBytes(reader_opt, "ffn_up.weight", hidden * ffn_h);
            layout.ffn_weight2_offset = offset;
            offset += getLayerTensorBytes(reader_opt, "ffn_down.weight", ffn_h * hidden);
            layout.ffn_weight3_offset = 0;
        } else {
            offset += getLayerTensorBytes(reader_opt, "ffn_gate.weight", hidden * ffn_h);
            layout.ffn_weight2_offset = offset;
            offset += getLayerTensorBytes(reader_opt, "ffn_up.weight", hidden * ffn_h);
            layout.ffn_weight3_offset = offset;
            offset += getLayerTensorBytes(reader_opt, "ffn_down.weight", ffn_h * hidden);
        }

        layout.attn_q_norm_offset = offset;
        offset += config_.head_dim * sizeof_fp16; // q_norm is usually FP16
        layout.attn_k_norm_offset = offset;
        offset += config_.head_dim * sizeof_fp16; // k_norm is usually FP16

        layout.per_layer_size = offset;

        std.debug.print("DEBUG: per_layer_size={d} bytes ({d:.2} MB), num_layers={d}\n", .{ layout.per_layer_size, @as(f64, @floatFromInt(layout.per_layer_size)) / (1024.0 * 1024.0), config_.num_layers });

        // Calculate SSM per-layer size using actual tensor sizes if available
        const ssm_conv_kernel = config_.ssm_conv_kernel;
        const ssm_inner = config_.ssm_inner_size;
        const dt_rank = config_.ssm_dt_rank;

        const ssm_out_hidden_dim = if (config_.ffn_hidden_dim > hidden * 2 and config_.ffn_type != .dense)
            config_.ffn_hidden_dim / 2
        else
            hidden * 2;

        var ssm_offset: usize = 0;
        layout.ssm_out_offset = ssm_offset;
        const ssm_out_bytes = getTensorBytes(reader_opt, "blk.0.ssm_out.weight", hidden * ssm_out_hidden_dim);
        ssm_offset += ssm_out_bytes;
        std.debug.print("DEBUG: ssm_out: {d} bytes (fallback: {d})\n", .{ ssm_out_bytes, hidden * ssm_out_hidden_dim * sizeof_fp16 });

        layout.ssm_x_offset = ssm_offset;
        // ssm_x doesn't exist in this model - use getLayerTensorBytes which will return 0
        const ssm_x_bytes = getLayerTensorBytes(reader_opt, "ssm_x.weight", hidden * hidden);
        ssm_offset += ssm_x_bytes;
        std.debug.print("DEBUG: ssm_x: {d} bytes (fallback: {d})\n", .{ ssm_x_bytes, hidden * hidden * sizeof_fp16 });

        layout.ssm_dt_offset = ssm_offset;
        const ssm_dt_bytes = getTensorBytes(reader_opt, "blk.0.ssm_dt.weight", hidden * dt_rank);
        ssm_offset += ssm_dt_bytes;
        std.debug.print("DEBUG: ssm_dt: {d} bytes (fallback: {d})\n", .{ ssm_dt_bytes, hidden * dt_rank * sizeof_fp16 });

        layout.ssm_A_offset = ssm_offset;
        // ssm_A maps to ssm_a (lowercase, no .weight)
        const ssm_A_bytes = getLayerTensorBytes(reader_opt, "ssm_A.weight", ssm_inner);
        ssm_offset += ssm_A_bytes;
        std.debug.print("DEBUG: ssm_A: {d} bytes (fallback: {d})\n", .{ ssm_A_bytes, ssm_inner * sizeof_fp16 });

        layout.ssm_B_offset = ssm_offset;
        // ssm_B maps to ssm_beta.weight
        const ssm_B_bytes = getLayerTensorBytes(reader_opt, "ssm_B.weight", hidden * ssm_inner);
        ssm_offset += ssm_B_bytes;
        std.debug.print("DEBUG: ssm_B: {d} bytes (fallback: {d})\n", .{ ssm_B_bytes, hidden * ssm_inner * sizeof_fp16 });

        layout.ssm_C_offset = ssm_offset;
        // ssm_C maps to ssm_alpha.weight
        const ssm_C_bytes = getLayerTensorBytes(reader_opt, "ssm_C.weight", hidden * ssm_inner);
        ssm_offset += ssm_C_bytes;
        std.debug.print("DEBUG: ssm_C: {d} bytes (fallback: {d})\n", .{ ssm_C_bytes, hidden * ssm_inner * sizeof_fp16 });

        layout.ssm_D_offset = ssm_offset;
        ssm_offset += getTensorBytes(reader_opt, "blk.0.ssm_D.weight", hidden);

        layout.ssm_conv1d_offset = ssm_offset;
        ssm_offset += getTensorBytes(reader_opt, "blk.0.ssm_conv1d.weight", ssm_conv_kernel * hidden);

        layout.ssm_conv1d_bias_offset = ssm_offset;
        ssm_offset += hidden * sizeof_fp16;

        layout.ssm_per_layer_size = ssm_offset;

        // Calculate total size including space for SSM weights
        // First, calculate actual per-layer sizes and fill layer_offsets
        var cumulative_offset: usize = layout.token_embedding_size;
        for (0..config_.num_layers) |i| {
            layout.layer_offsets[i] = cumulative_offset;

            // Calculate actual layer size for this specific layer
            var actual_layer_size: usize = 0;

            // Norm vectors are converted to fp16 when loaded into engine memory.
            var buf: [64]u8 = undefined;
            actual_layer_size += hidden * sizeof_fp16;

            // Attention projections - support both fused (qkv) and separate (q,k,v) naming
            // Try fused QKV first
            const qkv_name = std.fmt.bufPrint(&buf, "blk.{d}.attn_qkv.weight", .{i}) catch "";
            if (reader_opt) |reader| {
                if (reader.tensors.get(qkv_name)) |tensor| {
                    // Fused QKV: contains all three projections in one tensor
                    actual_layer_size += computeTensorQuantizedBytes(tensor.shape, tensor.dtype);
                } else {
                    // Fall back to separate Q, K, V projections
                    const q_name = std.fmt.bufPrint(&buf, "blk.{d}.attn_q.weight", .{i}) catch "";
                    if (reader.tensors.get(q_name)) |tensor| {
                        actual_layer_size += computeTensorQuantizedBytes(tensor.shape, tensor.dtype);
                    } else {
                        actual_layer_size += hidden * q_dim * sizeof_fp16;
                    }

                    const k_name = std.fmt.bufPrint(&buf, "blk.{d}.attn_k.weight", .{i}) catch "";
                    if (reader.tensors.get(k_name)) |tensor| {
                        actual_layer_size += computeTensorQuantizedBytes(tensor.shape, tensor.dtype);
                    } else {
                        actual_layer_size += hidden * kv_dim * sizeof_fp16;
                    }

                    const v_name = std.fmt.bufPrint(&buf, "blk.{d}.attn_v.weight", .{i}) catch "";
                    if (reader.tensors.get(v_name)) |tensor| {
                        actual_layer_size += computeTensorQuantizedBytes(tensor.shape, tensor.dtype);
                    } else {
                        actual_layer_size += hidden * kv_dim * sizeof_fp16;
                    }
                }
            } else {
                // No reader: use separate projections with FP16 fallback
                actual_layer_size += hidden * q_dim * sizeof_fp16;
                actual_layer_size += hidden * kv_dim * sizeof_fp16;
                actual_layer_size += hidden * kv_dim * sizeof_fp16;
            }

            // Output projection - support both attn_output and attn_gate naming
            // IMPORTANT: Use separate buffer to avoid overwriting with ffn_norm below
            var buf2: [64]u8 = undefined;
            const attn_gate_name = std.fmt.bufPrint(&buf2, "blk.{d}.attn_gate.weight", .{i}) catch "";
            const output_name = std.fmt.bufPrint(&buf, "blk.{d}.attn_output.weight", .{i}) catch "";
            if (reader_opt) |reader| {
                if (reader.tensors.get(attn_gate_name)) |tensor| {
                    actual_layer_size += computeTensorQuantizedBytes(tensor.shape, tensor.dtype);
                } else if (reader.tensors.get(output_name)) |tensor| {
                    actual_layer_size += computeTensorQuantizedBytes(tensor.shape, tensor.dtype);
                } else {
                    actual_layer_size += q_dim * hidden * sizeof_fp16;
                }
            } else {
                actual_layer_size += q_dim * hidden * sizeof_fp16;
            }

            // ffn_norm / post_attention_norm are also stored as fp16 in engine memory.
            actual_layer_size += hidden * sizeof_fp16;

            // FFN weights
            switch (config_.ffn_type) {
                .dense => {
                    const ffn_up_name = std.fmt.bufPrint(&buf, "blk.{d}.ffn_up.weight", .{i}) catch "";
                    if (reader_opt) |reader| {
                        if (reader.tensors.get(ffn_up_name)) |tensor| {
                            actual_layer_size += computeTensorQuantizedBytes(tensor.shape, tensor.dtype);
                        } else {
                            actual_layer_size += hidden * ffn_h * sizeof_fp16;
                        }
                    } else {
                        actual_layer_size += hidden * ffn_h * sizeof_fp16;
                    }
                    const ffn_down_name = std.fmt.bufPrint(&buf, "blk.{d}.ffn_down.weight", .{i}) catch "";
                    if (reader_opt) |reader| {
                        if (reader.tensors.get(ffn_down_name)) |tensor| {
                            actual_layer_size += computeTensorQuantizedBytes(tensor.shape, tensor.dtype);
                        } else {
                            actual_layer_size += ffn_h * hidden * sizeof_fp16;
                        }
                    } else {
                        actual_layer_size += ffn_h * hidden * sizeof_fp16;
                    }
                },
                .gated_swi_glu, .gated_gelu => {
                    const gate_name = std.fmt.bufPrint(&buf, "blk.{d}.ffn_gate.weight", .{i}) catch "";
                    if (reader_opt) |reader| {
                        if (reader.tensors.get(gate_name)) |tensor| {
                            actual_layer_size += computeTensorQuantizedBytes(tensor.shape, tensor.dtype);
                        } else {
                            actual_layer_size += hidden * ffn_h * sizeof_fp16;
                        }
                    } else {
                        actual_layer_size += hidden * ffn_h * sizeof_fp16;
                    }
                    const ffn_up_name = std.fmt.bufPrint(&buf, "blk.{d}.ffn_up.weight", .{i}) catch "";
                    if (reader_opt) |reader| {
                        if (reader.tensors.get(ffn_up_name)) |tensor| {
                            actual_layer_size += computeTensorQuantizedBytes(tensor.shape, tensor.dtype);
                        } else {
                            actual_layer_size += hidden * ffn_h * sizeof_fp16;
                        }
                    } else {
                        actual_layer_size += hidden * ffn_h * sizeof_fp16;
                    }
                    const ffn_down_name = std.fmt.bufPrint(&buf, "blk.{d}.ffn_down.weight", .{i}) catch "";
                    if (reader_opt) |reader| {
                        if (reader.tensors.get(ffn_down_name)) |tensor| {
                            actual_layer_size += computeTensorQuantizedBytes(tensor.shape, tensor.dtype);
                        } else {
                            actual_layer_size += ffn_h * hidden * sizeof_fp16;
                        }
                    } else {
                        actual_layer_size += ffn_h * hidden * sizeof_fp16;
                    }
                },
            }

            // q_norm and k_norm are stored as fp16 as well.
            actual_layer_size += config_.head_dim * sizeof_fp16;
            actual_layer_size += config_.head_dim * sizeof_fp16;

            // Update cumulative offset for next layer
            cumulative_offset += @max(actual_layer_size, layout.per_layer_size);
        }

        // Calculate final offsets based on actual layer sizes
        // cumulative_offset now contains the total size of all layers
        const output_proj_bytes = blk: {
            if (reader_opt) |reader| {
                if (reader.findTensorBySuffix("output.weight")) |tensor_info| {
                    break :blk computeTensorQuantizedBytes(tensor_info.shape, tensor_info.dtype);
                }
                if (reader.tensors.get("token_embd.weight")) |tensor_info| {
                    break :blk computeTensorQuantizedBytes(tensor_info.shape, tensor_info.dtype);
                }
            }
            break :blk vocab * hidden * sizeof_fp16;
        };

        layout.final_norm_offset = cumulative_offset;
        layout.output_proj_offset = layout.final_norm_offset + hidden * sizeof_fp16;
        layout.ssm_region_offset = layout.output_proj_offset + output_proj_bytes;

        const num_ssm_layers = if (config_.num_ssm_layers > 0) config_.num_ssm_layers else config_.num_layers;
        const ssm_region_size = layout.ssm_per_layer_size * num_ssm_layers;
        layout.total_size = layout.ssm_region_offset + ssm_region_size;

        layout.raw_pool_size = layout.total_size;

        return layout;
    }

    pub fn deinit(self: *WeightLayout, allocator: std.mem.Allocator) void {
        allocator.free(self.layer_types);
        allocator.free(self.layer_offsets);
    }
};

pub fn ssmPackedLayerIndex(layer_types: []const config.ModelConfig.LayerType, layer_idx: usize) ?usize {
    if (layer_types[layer_idx] == .attention) return null;

    var packed_idx: usize = 0;
    for (layer_types[0..layer_idx]) |layer_type| {
        if (layer_type != .attention) packed_idx += 1;
    }
    return packed_idx;
}

fn initTestReader(allocator: std.mem.Allocator) gguf.Reader {
    return .{
        .file = undefined,
        .allocator = allocator,
        .cursor = 0,
        .data_offset = 0,
        .model_prefix = &.{},
        .metadata = std.StringHashMap(gguf.GGUFValue).init(allocator),
        .tokens = std.array_list.Managed([]const u8).init(allocator),
        .tensors = std.StringHashMap(gguf.TensorInfo).init(allocator),
        .config = std.mem.zeroes(config.ModelConfig),
        .alignment = 32,
    };
}

fn deinitTestReader(reader: *gguf.Reader) void {
    var tensor_it = reader.tensors.iterator();
    while (tensor_it.next()) |entry| {
        reader.allocator.free(entry.key_ptr.*);
        reader.allocator.free(entry.value_ptr.shape);
    }
    reader.tensors.deinit();
    reader.tokens.deinit();
    reader.metadata.deinit();
}

fn addTestTensor(reader: *gguf.Reader, name: []const u8, dtype: gguf.GGMLType, shape_src: []const u64) !void {
    const owned_name = try reader.allocator.dupe(u8, name);
    errdefer reader.allocator.free(owned_name);

    const shape = try reader.allocator.alloc(u64, shape_src.len);
    errdefer reader.allocator.free(shape);
    @memcpy(shape, shape_src);

    try reader.tensors.put(owned_name, .{
        .offset = 0,
        .size = computeTensorQuantizedBytes(shape, dtype),
        .shape = shape,
        .dtype = dtype,
    });
}

test "ssmPackedLayerIndex compacts sparse SSM layers" {
    const layer_types = [_]config.ModelConfig.LayerType{ .ssm, .attention, .qwen_linear, .attention, .ssm };

    try std.testing.expectEqual(@as(?usize, 0), ssmPackedLayerIndex(&layer_types, 0));
    try std.testing.expectEqual(@as(?usize, null), ssmPackedLayerIndex(&layer_types, 1));
    try std.testing.expectEqual(@as(?usize, 1), ssmPackedLayerIndex(&layer_types, 2));
    try std.testing.expectEqual(@as(?usize, 2), ssmPackedLayerIndex(&layer_types, 4));
}

test "WeightLayout uses actual output tensor bytes" {
    var reader = initTestReader(std.testing.allocator);
    defer deinitTestReader(&reader);

    try addTestTensor(&reader, "token_embd.weight", .q8_0, &.{ 256, 8 });
    try addTestTensor(&reader, "output_norm.weight", .f32, &.{256});
    try addTestTensor(&reader, "output.weight", .q4_k, &.{ 256, 8 });

    const cfg = config.ModelConfig{
        .vocab_size = 8,
        .hidden_dim = 256,
        .num_heads = 8,
        .num_kv_heads = 8,
        .head_dim = 32,
        .num_layers = 1,
        .num_ssm_layers = 0,
        .ffn_hidden_dim = 512,
        .max_seq_len = 128,
        .ffn_type = .gated_swi_glu,
        .norm_type = .rms_norm,
        .pos_encoding = .rope,
        .use_kv_quantization = false,
        .ssm_conv_kernel = 4,
        .ssm_inner_size = 16,
        .ssm_num_groups = 1,
        .ssm_dt_rank = 16,
        .ssm_dt_scale = 1.0,
    };

    var layout = try WeightLayout.compute(cfg, &reader, std.testing.allocator);
    defer layout.deinit(std.testing.allocator);

    try std.testing.expectEqual(
        computeTensorQuantizedBytes(&.{ 256, 8 }, .q4_k),
        layout.ssm_region_offset - layout.output_proj_offset,
    );
}

test "AlignedMemoryPool basic alloc" {
    var pool = try AlignedMemoryPool.init(4096);
    defer pool.deinit();
    const arr = pool.alloc(f32, 256);
    try std.testing.expect(pool.bytesUsed() == 256 * 4);
    try std.testing.expectEqual(@as(usize, 0), @intFromPtr(arr) % types.CACHE_LINE_SIZE);
}

test "WeightLayout compute" {
    const test_config = config.ModelConfig{
        .vocab_size = 32000,
        .hidden_dim = 4096,
        .num_heads = 32,
        .num_kv_heads = 32,
        .head_dim = 128,
        .num_layers = 32,
        .num_ssm_layers = 0,
        .ffn_hidden_dim = 11008,
        .max_seq_len = 2048,
        .ffn_type = .gated_swi_glu,
        .norm_type = .rms_norm,
        .pos_encoding = .rope,
        .use_kv_quantization = false,
        .ssm_conv_kernel = 4,
        .ssm_inner_size = 16,
        .ssm_num_groups = 1,
        .ssm_dt_rank = 256,
        .ssm_dt_scale = 1.0,
    };
    var layout = try WeightLayout.compute(test_config, null, std.testing.allocator);
    defer layout.deinit(std.testing.allocator);
    try std.testing.expect(layout.token_embedding_size > 0);
    try std.testing.expect(layout.per_layer_size > 0);
    try std.testing.expect(layout.total_size > layout.token_embedding_size);
}
