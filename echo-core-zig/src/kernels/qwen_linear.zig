const std = @import("std");
const config = @import("core_config");
const types = @import("../core/types.zig");
const math = @import("../core/math.zig");
const matvec = @import("matvec.zig");
const parallel = @import("parallel.zig");
const quant = @import("quant.zig");
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
    dequant_scratch: []types.fp16_t,

    pub fn init(
        conv_dim: u32,
        conv_kernel: u32,
        num_v_heads: u32,
        head_k_dim: u32,
        head_v_dim: u32,
        hidden_dim: u32,
        allocator: std.mem.Allocator,
    ) !QwenLinearState {
        const conv_state = try allocator.alloc(f32, conv_dim * conv_kernel);
        errdefer allocator.free(conv_state);
        @memset(conv_state, 0);

        const recurrent_state = try allocator.alloc(f32, num_v_heads * head_k_dim * head_v_dim);
        errdefer allocator.free(recurrent_state);
        @memset(recurrent_state, 0);

        const dequant_scratch = try allocator.alloc(types.fp16_t, hidden_dim * num_v_heads * 2);
        errdefer allocator.free(dequant_scratch);

        return .{
            .conv_state = conv_state,
            .recurrent_state = recurrent_state,
            .dequant_scratch = dequant_scratch,
        };
    }

    pub fn deinit(self: *QwenLinearState, allocator: std.mem.Allocator) void {
        allocator.free(self.conv_state);
        allocator.free(self.recurrent_state);
        allocator.free(self.dequant_scratch);
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
    layer_hint: u32,
) void {
    _ = layer_hint;
    const hidden = cfg.hidden_dim;
    const value_dim = cfg.ssm_inner_size;
    const num_v_heads = cfg.ssm_dt_rank;
    std.debug.assert(num_v_heads != 0);
    std.debug.assert(value_dim % num_v_heads == 0);

    const head_v_dim = value_dim / num_v_heads;
    const conv_elem_bytes: usize = switch (weights.conv1d.dtype) { .f32 => 4, .f16, .bf16 => 2, else => 0 };
    const conv_dim: u32 = if (conv_elem_bytes > 0 and cfg.ssm_conv_kernel > 0) @intCast(weights.conv1d.bytes.len / (cfg.ssm_conv_kernel * conv_elem_bytes)) else value_dim * 3;
    const key_dim = (conv_dim - value_dim) / 2;
    std.debug.assert(key_dim % head_v_dim == 0);

    const head_k_dim = head_v_dim;
    const num_k_heads = key_dim / head_k_dim;
    const repeat_factor = num_v_heads / num_k_heads;
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

    parallel.parallelMatvec(weights.qkv.bytes.ptr, input.ptr, temps.mixed_qkv.ptr, conv_dim, hidden, weights.qkv.dtype);
    parallel.parallelMatvec(weights.z.bytes.ptr, input.ptr, temps.z.ptr, value_dim, hidden, weights.z.dtype);
    // alpha/beta projections: tensors stored as [hidden, N] (transposed layout)
    // Q4_K matvec assumes contiguous blocks per row, but these tensors have
    // small K (num_v_heads=32) → dequantize to FP16 and use FP16 matvec
    // Use state buffers to avoid stack overflow (each is ~256KB)
    {
        const n_weights = hidden * num_v_heads;
        const alpha_fp16 = @as([*]types.fp16_t, @ptrCast(@alignCast(state.dequant_scratch.ptr)))[0..n_weights];
        quant.dequantizeRow(weights.alpha.bytes.ptr, alpha_fp16.ptr, n_weights, weights.alpha.dtype);
        @memset(temps.alpha[0..num_v_heads], 0);
        parallel.parallelMatvec(@ptrCast(alpha_fp16.ptr), input.ptr, temps.alpha.ptr, num_v_heads, hidden, .f16);
    }
    {
        const n_weights = hidden * num_v_heads;
        const beta_fp16 = state.dequant_scratch[n_weights..];
        quant.dequantizeRow(weights.beta.bytes.ptr, beta_fp16.ptr, n_weights, weights.beta.dtype);
        @memset(temps.beta[0..num_v_heads], 0);
        parallel.parallelMatvec(@ptrCast(beta_fp16.ptr), input.ptr, temps.beta.ptr, num_v_heads, hidden, .f16);
    }

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

        var kv_mem: [256]f32 = undefined;
        @memset(kv_mem[0..head_v_dim], 0);
        for (0..head_k_dim) |k_idx| {
            const kval = temps.conv_out[k_base + k_idx];
            const srow = head_state[k_idx * head_v_dim ..][0..head_v_dim];
            for (0..head_v_dim) |v_idx| kv_mem[v_idx] += srow[v_idx] * kval;
        }
        var delta: [256]f32 = undefined;
        std.debug.assert(head_v_dim <= delta.len);
        for (0..head_v_dim) |v_idx| {
            delta[v_idx] = (temps.conv_out[v_base + v_idx] - kv_mem[v_idx]) * beta;
        }


        for (0..head_k_dim) |k_idx| {
            const key_value = temps.conv_out[k_base + k_idx];
            for (0..head_v_dim) |v_idx| {
                head_state[k_idx * head_v_dim + v_idx] += key_value * delta[v_idx];
            }
        }

        var attn_arr: [256]f32 = undefined;
        @memset(attn_arr[0..head_v_dim], 0);
        for (0..head_k_dim) |k_idx| {
            const qval = temps.conv_out[q_base + k_idx] * scale;
            const srow = head_state[k_idx * head_v_dim ..][0..head_v_dim];
            for (0..head_v_dim) |v_idx| attn_arr[v_idx] += srow[v_idx] * qval;
        }
        var rms: f32 = 0;
        for (0..head_v_dim) |v_idx| {
            temps.core[core_base + v_idx] = attn_arr[v_idx];
            rms += attn_arr[v_idx] * attn_arr[v_idx];
        }

        rms = @sqrt(rms / @as(f32, @floatFromInt(head_v_dim)) + 1e-6);
        const inv_rms = 1.0 / rms;
        for (0..head_v_dim) |v_idx| {
            const normed = temps.core[core_base + v_idx] * inv_rms * scalarAt(norm_weight, v_idx);
            temps.core[core_base + v_idx] = normed * math.swish(temps.z[core_base + v_idx]);
        }
    }

    @memset(output[0..hidden], 0);
    parallel.parallelMatvec(
        weights.out.bytes.ptr,
        temps.core.ptr,
        output.ptr,
        hidden,
        value_dim,
        weights.out.dtype,
    );
}

pub fn forwardCore(
    cfg: config.ModelConfig,
    state: *QwenLinearState,
    weights: Weights,
    mixed_qkv: []const f32,
    z_in: []const f32,
    alpha_in: []const f32,
    beta_in: []const f32,
    core: []f32,
) void {
    const hidden = cfg.hidden_dim;
    _ = hidden;
    const value_dim = cfg.ssm_inner_size;
    const num_v_heads = cfg.ssm_dt_rank;
    const head_v_dim = value_dim / num_v_heads;
    const conv_elem_bytes: usize = switch (weights.conv1d.dtype) { .f32 => 4, .f16, .bf16 => 2, else => 0 };
    const conv_dim: u32 = if (conv_elem_bytes > 0 and cfg.ssm_conv_kernel > 0) @intCast(weights.conv1d.bytes.len / (cfg.ssm_conv_kernel * conv_elem_bytes)) else value_dim * 3;
    const key_dim = (conv_dim - value_dim) / 2;
    const head_k_dim = head_v_dim;
    const num_k_heads = key_dim / head_k_dim;
    const repeat_factor = num_v_heads / num_k_heads;
    const conv_kernel = cfg.ssm_conv_kernel;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_k_dim)));

    var conv_out_buf: [8192]f32 = undefined;
    const conv_out = conv_out_buf[0..conv_dim];

    if (conv_kernel > 1) {
        var kernel_idx: usize = conv_kernel - 1;
        while (kernel_idx > 0) : (kernel_idx -= 1) {
            const dst = kernel_idx * conv_dim;
            const src = (kernel_idx - 1) * conv_dim;
            @memcpy(state.conv_state[dst .. dst + conv_dim], state.conv_state[src .. src + conv_dim]);
        }
    }
    @memcpy(state.conv_state[0..conv_dim], mixed_qkv[0..conv_dim]);

    for (0..conv_dim) |dim_idx| {
        var acc: f32 = 0;
        for (0..conv_kernel) |kernel_idx| {
            const state_idx = kernel_idx * conv_dim + dim_idx;
            const weight_idx = (conv_kernel - 1 - kernel_idx) * conv_dim + dim_idx;
            acc += state.conv_state[state_idx] * scalarAt(weights.conv1d, weight_idx);
        }
        conv_out[dim_idx] = math.swish(acc);
    }

    normalizeHeads(conv_out[0..key_dim], num_k_heads, head_k_dim);
    normalizeHeads(conv_out[key_dim .. key_dim * 2], num_k_heads, head_k_dim);

    const norm_weight = weights.norm;
    for (0..num_v_heads) |head_idx| {
        const q_head_idx = head_idx / repeat_factor;
        const q_base = q_head_idx * head_k_dim;
        const k_base = key_dim + q_head_idx * head_k_dim;
        const v_base = key_dim * 2 + head_idx * head_v_dim;
        const core_base = head_idx * head_v_dim;
        const state_base = head_idx * head_k_dim * head_v_dim;

        const beta = 1.0 / (1.0 + std.math.exp(-beta_in[head_idx]));
        const A_log = scalarAt(weights.a_log, head_idx);
        const dt_bias = scalarAt(weights.dt_bias, head_idx);
        const g = -std.math.exp(A_log) * softplus(alpha_in[head_idx] + dt_bias);
        const decay = std.math.exp(g);

        const head_state = state.recurrent_state[state_base .. state_base + head_k_dim * head_v_dim];
        for (head_state) |*value| value.* *= decay;

        var kv_mem: [256]f32 = undefined;
        @memset(kv_mem[0..head_v_dim], 0);
        for (0..head_k_dim) |k_idx| {
            const kval = conv_out[k_base + k_idx];
            const srow = head_state[k_idx * head_v_dim ..][0..head_v_dim];
            for (0..head_v_dim) |v_idx| kv_mem[v_idx] += srow[v_idx] * kval;
        }
        var delta: [256]f32 = undefined;
        for (0..head_v_dim) |v_idx| {
            delta[v_idx] = (conv_out[v_base + v_idx] - kv_mem[v_idx]) * beta;
        }

        for (0..head_k_dim) |k_idx| {
            const key_value = conv_out[k_base + k_idx];
            for (0..head_v_dim) |v_idx| {
                head_state[k_idx * head_v_dim + v_idx] += key_value * delta[v_idx];
            }
        }

        var attn_arr: [256]f32 = undefined;
        @memset(attn_arr[0..head_v_dim], 0);
        for (0..head_k_dim) |k_idx| {
            const qval = conv_out[q_base + k_idx] * scale;
            const srow = head_state[k_idx * head_v_dim ..][0..head_v_dim];
            for (0..head_v_dim) |v_idx| attn_arr[v_idx] += srow[v_idx] * qval;
        }
        var rms: f32 = 0;
        for (0..head_v_dim) |v_idx| {
            core[core_base + v_idx] = attn_arr[v_idx];
            rms += attn_arr[v_idx] * attn_arr[v_idx];
        }

        rms = @sqrt(rms / @as(f32, @floatFromInt(head_v_dim)) + 1e-6);
        const inv_rms = 1.0 / rms;
        for (0..head_v_dim) |v_idx| {
            const normed = core[core_base + v_idx] * inv_rms * scalarAt(norm_weight, v_idx);
            core[core_base + v_idx] = normed * math.swish(z_in[core_base + v_idx]);
        }
    }
}
