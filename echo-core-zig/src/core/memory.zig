const std = @import("std");
const types = @import("types.zig");
const config = @import("config.zig");

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

    per_layer_size: usize,
    ssm_per_layer_size: usize,
    final_norm_offset: usize,
    output_proj_offset: usize,
    total_size: usize,
    raw_pool_size: usize,

    // Per-layer type tracking (dynamically allocated)
    layer_types: []config.ModelConfig.LayerType,

    pub fn compute(config_: anytype, allocator: std.mem.Allocator) !WeightLayout {
        const hidden = config_.hidden_dim;
        const vocab = config_.vocab_size;
        const kv_dim = config_.num_kv_heads * config_.head_dim;
        const ffn_h = config_.ffn_hidden_dim;
        const sizeof_fp16 = @sizeOf(types.fp16_t);

        // Allocate layer types array
        const layer_types = try allocator.alloc(config.ModelConfig.LayerType, config_.num_layers);
        errdefer allocator.free(layer_types);

        // Default all layers to attention initially
        // Will be updated by loader based on tensor presence
        @memset(layer_types, .attention);

        var layout: WeightLayout = undefined;
        layout.layer_types = layer_types;

        layout.token_embedding_offset = 0;
        layout.token_embedding_size = vocab * hidden * sizeof_fp16;

        var offset: usize = 0;
        layout.norm_weight_offset = offset;
        offset += hidden * sizeof_fp16;

        layout.q_proj_offset = offset;
        offset += (config_.num_heads * config_.head_dim) * hidden * sizeof_fp16;

        layout.k_proj_offset = offset;
        offset += kv_dim * hidden * sizeof_fp16;

        layout.v_proj_offset = offset;
        offset += kv_dim * hidden * sizeof_fp16;

        layout.o_proj_offset = offset;
        offset += hidden * (config_.num_heads * config_.head_dim) * sizeof_fp16;

        layout.ffn_norm_offset = offset;
        offset += hidden * sizeof_fp16;

        layout.ffn_weight1_offset = offset;
        if (config_.ffn_type == .dense) {
            offset += hidden * ffn_h * sizeof_fp16;
            layout.ffn_weight2_offset = offset;
            offset += ffn_h * hidden * sizeof_fp16;
            layout.ffn_weight3_offset = 0;
        } else {
            offset += hidden * ffn_h * sizeof_fp16;
            layout.ffn_weight2_offset = offset;
            offset += hidden * ffn_h * sizeof_fp16;
            layout.ffn_weight3_offset = offset;
            offset += ffn_h * hidden * sizeof_fp16;
        }

        layout.attn_q_norm_offset = offset;
        offset += config_.head_dim * sizeof_fp16;
        layout.attn_k_norm_offset = offset;
        offset += config_.head_dim * sizeof_fp16;

        layout.per_layer_size = offset;

        // Calculate SSM per-layer size
        const ssm_conv_kernel = config_.ssm_conv_kernel;
        const ssm_inner = config_.ssm_inner_size;
        const dt_rank = config_.ssm_dt_rank;

        var ssm_offset: usize = 0;
        layout.ssm_out_offset = ssm_offset;
        ssm_offset += hidden * hidden * sizeof_fp16; // ssm_out projection

        layout.ssm_x_offset = ssm_offset;
        ssm_offset += hidden * hidden * sizeof_fp16; // ssm_x projection

        layout.ssm_dt_offset = ssm_offset;
        ssm_offset += hidden * dt_rank * sizeof_fp16; // ssm_dt projection

        layout.ssm_A_offset = ssm_offset;
        ssm_offset += ssm_inner * sizeof_fp16; // A diagonal matrix

        layout.ssm_B_offset = ssm_offset;
        ssm_offset += hidden * ssm_inner * sizeof_fp16; // B matrix

        layout.ssm_C_offset = ssm_offset;
        ssm_offset += hidden * ssm_inner * sizeof_fp16; // C matrix

        layout.ssm_D_offset = ssm_offset;
        ssm_offset += hidden * sizeof_fp16; // D skip connection

        layout.ssm_conv1d_offset = ssm_offset;
        ssm_offset += ssm_conv_kernel * hidden * sizeof_fp16; // conv1d weights

        layout.ssm_conv1d_bias_offset = ssm_offset;
        ssm_offset += hidden * sizeof_fp16; // conv1d bias

        layout.ssm_per_layer_size = ssm_offset;

        // Calculate total size assuming all layers are attention (conservative)
        // Actual size will depend on mix of attention and SSM layers
        layout.final_norm_offset = layout.token_embedding_size + layout.per_layer_size * config_.num_layers;
        layout.output_proj_offset = layout.final_norm_offset + hidden * sizeof_fp16;
        layout.total_size = layout.output_proj_offset + hidden * vocab * sizeof_fp16;

        layout.raw_pool_size = layout.total_size;

        return layout;
    }

    pub fn deinit(self: *WeightLayout, allocator: std.mem.Allocator) void {
        allocator.free(self.layer_types);
    }
};

test "AlignedMemoryPool basic alloc" {
    var pool = try AlignedMemoryPool.init(4096);
    defer pool.deinit();
    const arr = pool.alloc(f32, 256);
    try std.testing.expect(pool.bytesUsed() == 256 * 4);
    try std.testing.expectEqual(@as(usize, 0), @intFromPtr(arr) % types.CACHE_LINE_SIZE);
}

test "WeightLayout compute" {
    const test_config = .{
        .vocab_size = 32000,
        .hidden_dim = 4096,
        .num_heads = 32,
        .num_kv_heads = 32,
        .head_dim = 128,
        .num_layers = 32,
        .ffn_hidden_dim = 11008,
        .ffn_type = .gated_swi_glu,
        .ssm_conv_kernel = 4,
        .ssm_inner_size = 16,
        .ssm_dt_rank = 256,
    };
    var layout = try WeightLayout.compute(test_config, std.testing.allocator);
    defer layout.deinit(std.testing.allocator);
    try std.testing.expect(layout.token_embedding_size > 0);
    try std.testing.expect(layout.per_layer_size > 0);
    try std.testing.expect(layout.total_size > layout.token_embedding_size);
}
