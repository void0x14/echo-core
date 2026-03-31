const std = @import("std");
const config = @import("../core/config.zig");
const types = @import("../core/types.zig");
const math = @import("../core/math.zig");
const matvec = @import("matvec.zig");
const gguf = @import("../gguf/reader.zig");

pub const TensorView = struct {
    bytes: []const u8,
    dtype: gguf.GGMLType,
};

pub const Weights = struct {
    qkv: TensorView,
    z: TensorView,
    alpha: TensorView,
    beta: TensorView,
    a_log: TensorView,
    dt_bias: TensorView,
    conv1d: TensorView,
    norm: TensorView,
    out: TensorView,
};

pub const TempBuffers = struct {
    mixed_qkv: []f32,
    conv_out: []f32,
    z: []f32,
    core: []f32,
    alpha: []f32,
    beta: []f32,
};

pub const QwenLinearState = struct {
    conv_state: []f32,
    recurrent_state: []f32,

    pub fn init(
        conv_dim: u32,
        conv_kernel: u32,
        num_v_heads: u32,
        head_k_dim: u32,
        head_v_dim: u32,
        allocator: std.mem.Allocator,
    ) !QwenLinearState {
        const conv_state = try allocator.alloc(f32, conv_dim * conv_kernel);
        errdefer allocator.free(conv_state);
        @memset(conv_state, 0);

        const recurrent_state = try allocator.alloc(f32, num_v_heads * head_k_dim * head_v_dim);
        errdefer allocator.free(recurrent_state);
        @memset(recurrent_state, 0);

        return .{
            .conv_state = conv_state,
            .recurrent_state = recurrent_state,
        };
    }

    pub fn deinit(self: *QwenLinearState, allocator: std.mem.Allocator) void {
        allocator.free(self.conv_state);
        allocator.free(self.recurrent_state);
    }

    pub fn reset(self: *QwenLinearState) void {
        @memset(self.conv_state, 0);
        @memset(self.recurrent_state, 0);
    }
};

fn softplus(x: f32) f32 {
    if (x > 20.0) return x;
    return std.math.log1p(std.math.exp(x));
}

fn scalarAt(tensor: TensorView, idx: usize) f32 {
    return switch (tensor.dtype) {
        .f32 => @as([*]const f32, @ptrCast(@alignCast(tensor.bytes.ptr)))[idx],
        .f16 => {
            const vals = @as([*]const types.fp16_t, @ptrCast(@alignCast(tensor.bytes.ptr)));
            return types.fp16_to_fp32(vals[idx]);
        },
        else => unreachable,
    };
}

fn normalizeHeads(buffer: []f32, num_heads: u32, head_dim: u32) void {
    for (0..num_heads) |head_idx| {
        const base = head_idx * head_dim;
        var sum_sq: f32 = 0;
        for (0..head_dim) |d| {
            const value = buffer[base + d];
            sum_sq += value * value;
        }

        const inv_norm = 1.0 / @sqrt(sum_sq + 1e-6);
        for (0..head_dim) |d| {
            buffer[base + d] *= inv_norm;
        }
    }
}

pub fn forward(
    cfg: config.ModelConfig,
    input: []const f32,
    output: []f32,
    state: *QwenLinearState,
    weights: Weights,
    temps: TempBuffers,
) void {
    const hidden = cfg.hidden_dim;
    const value_dim = cfg.ssm_inner_size;
    const num_v_heads = cfg.ssm_dt_rank;
    std.debug.assert(num_v_heads != 0);
    std.debug.assert(value_dim % num_v_heads == 0);

    const head_v_dim = value_dim / num_v_heads;
    const key_dim = value_dim / 2;
    std.debug.assert(key_dim % head_v_dim == 0);

    const head_k_dim = head_v_dim;
    const num_k_heads = key_dim / head_k_dim;
    const repeat_factor = num_v_heads / num_k_heads;
    const conv_dim = key_dim * 2 + value_dim;
    const conv_kernel = cfg.ssm_conv_kernel;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_k_dim)));

    std.debug.assert(temps.mixed_qkv.len >= conv_dim);
    std.debug.assert(temps.conv_out.len >= conv_dim);
    std.debug.assert(temps.z.len >= value_dim);
    std.debug.assert(temps.core.len >= value_dim);
    std.debug.assert(temps.alpha.len >= num_v_heads);
    std.debug.assert(temps.beta.len >= num_v_heads);

    @memset(temps.mixed_qkv[0..conv_dim], 0);
    @memset(temps.z[0..value_dim], 0);
    @memset(temps.alpha[0..num_v_heads], 0);
    @memset(temps.beta[0..num_v_heads], 0);
    @memset(temps.core[0..value_dim], 0);

    matvec.matvecDispatchQuant(
        config.Intel13500H_Tiles.TILE_K,
        config.Intel13500H_Tiles.TILE_M,
        weights.qkv.bytes.ptr,
        input.ptr,
        temps.mixed_qkv.ptr,
        conv_dim,
        hidden,
        weights.qkv.dtype,
    );
    matvec.matvecDispatchQuant(
        config.Intel13500H_Tiles.TILE_K,
        config.Intel13500H_Tiles.TILE_M,
        weights.z.bytes.ptr,
        input.ptr,
        temps.z.ptr,
        value_dim,
        hidden,
        weights.z.dtype,
    );
    matvec.matvecDispatchQuant(
        config.Intel13500H_Tiles.TILE_K,
        config.Intel13500H_Tiles.TILE_M,
        weights.alpha.bytes.ptr,
        input.ptr,
        temps.alpha.ptr,
        num_v_heads,
        hidden,
        weights.alpha.dtype,
    );
    matvec.matvecDispatchQuant(
        config.Intel13500H_Tiles.TILE_K,
        config.Intel13500H_Tiles.TILE_M,
        weights.beta.bytes.ptr,
        input.ptr,
        temps.beta.ptr,
        num_v_heads,
        hidden,
        weights.beta.dtype,
    );

    // Shift older depthwise-conv states and insert current projection at the front.
    if (conv_kernel > 1) {
        var kernel_idx: usize = conv_kernel - 1;
        while (kernel_idx > 0) : (kernel_idx -= 1) {
            const dst = kernel_idx * conv_dim;
            const src = (kernel_idx - 1) * conv_dim;
            @memcpy(state.conv_state[dst .. dst + conv_dim], state.conv_state[src .. src + conv_dim]);
        }
    }
    @memcpy(state.conv_state[0..conv_dim], temps.mixed_qkv[0..conv_dim]);

    for (0..conv_dim) |dim_idx| {
        var acc: f32 = 0;
        for (0..conv_kernel) |kernel_idx| {
            const state_idx = kernel_idx * conv_dim + dim_idx;
            const weight_idx = (conv_kernel - 1 - kernel_idx) * conv_dim + dim_idx;
            acc += state.conv_state[state_idx] * scalarAt(weights.conv1d, weight_idx);
        }
        temps.conv_out[dim_idx] = math.swish(acc);
    }

    normalizeHeads(temps.conv_out[0..key_dim], num_k_heads, head_k_dim);
    normalizeHeads(temps.conv_out[key_dim .. key_dim * 2], num_k_heads, head_k_dim);

    const norm_weight = weights.norm;
    for (0..num_v_heads) |head_idx| {
        const q_head_idx = head_idx / repeat_factor;
        const q_base = q_head_idx * head_k_dim;
        const k_base = key_dim + q_head_idx * head_k_dim;
        const v_base = key_dim * 2 + head_idx * head_v_dim;
        const core_base = head_idx * head_v_dim;
        const state_base = head_idx * head_k_dim * head_v_dim;

        const beta = 1.0 / (1.0 + std.math.exp(-temps.beta[head_idx]));
        const A_log = scalarAt(weights.a_log, head_idx);
        const dt_bias = scalarAt(weights.dt_bias, head_idx);
        const g = -std.math.exp(A_log) * softplus(temps.alpha[head_idx] + dt_bias);
        const decay = std.math.exp(g);

        const head_state = state.recurrent_state[state_base .. state_base + head_k_dim * head_v_dim];
        for (head_state) |*value| value.* *= decay;

        var delta: [256]f32 = undefined;
        std.debug.assert(head_v_dim <= delta.len);

        for (0..head_v_dim) |v_idx| {
            var kv_mem: f32 = 0;
            for (0..head_k_dim) |k_idx| {
                kv_mem += head_state[k_idx * head_v_dim + v_idx] * temps.conv_out[k_base + k_idx];
            }
            delta[v_idx] = (temps.conv_out[v_base + v_idx] - kv_mem) * beta;
        }

        for (0..head_k_dim) |k_idx| {
            const key_value = temps.conv_out[k_base + k_idx];
            for (0..head_v_dim) |v_idx| {
                head_state[k_idx * head_v_dim + v_idx] += key_value * delta[v_idx];
            }
        }

        var rms: f32 = 0;
        for (0..head_v_dim) |v_idx| {
            var attn_value: f32 = 0;
            for (0..head_k_dim) |k_idx| {
                attn_value += head_state[k_idx * head_v_dim + v_idx] * (temps.conv_out[q_base + k_idx] * scale);
            }
            temps.core[core_base + v_idx] = attn_value;
            rms += attn_value * attn_value;
        }

        rms = @sqrt(rms / @as(f32, @floatFromInt(head_v_dim)) + 1e-6);
        const inv_rms = 1.0 / rms;
        for (0..head_v_dim) |v_idx| {
            const normed = temps.core[core_base + v_idx] * inv_rms * scalarAt(norm_weight, v_idx);
            temps.core[core_base + v_idx] = normed * math.swish(temps.z[core_base + v_idx]);
        }
    }

    @memset(output[0..hidden], 0);
    matvec.matvecDispatchQuant(
        config.Intel13500H_Tiles.TILE_K,
        config.Intel13500H_Tiles.TILE_M,
        weights.out.bytes.ptr,
        temps.core.ptr,
        output.ptr,
        hidden,
        value_dim,
        weights.out.dtype,
    );
}
