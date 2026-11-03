const std = @import("std");
const config = @import("core_config");
const types = @import("../core/types.zig");
const math = @import("../core/math.zig");
const matvec = @import("matvec.zig");
const parallel = @import("parallel.zig");
const gguf = @import("../gguf/reader.zig");

/// SSM state for a single layer
/// Maintains convolution state and SSM recurrent state
pub const SSMState = struct {
    conv_state: []f32, // [conv_kernel_size, hidden_dim]
    ssm_state: []f32, // [ssm_inner_size, hidden_dim/ssm_num_groups]
    conv_pos: usize, // Current position in circular conv buffer

    pub fn init(
        hidden_dim: u32,
        conv_kernel: u32,
        ssm_inner: u32,
        ssm_groups: u32,
        allocator: std.mem.Allocator,
    ) !SSMState {
        const state_per_group = hidden_dim / ssm_groups;
        const conv_state = try allocator.alloc(f32, conv_kernel * hidden_dim);
        errdefer allocator.free(conv_state);
        @memset(conv_state, 0);

        const ssm_state = try allocator.alloc(f32, ssm_inner * state_per_group);
        errdefer allocator.free(ssm_state);
        @memset(ssm_state, 0);

        return .{
            .conv_state = conv_state,
            .ssm_state = ssm_state,
            .conv_pos = 0,
        };
    }

    pub fn deinit(self: *SSMState, allocator: std.mem.Allocator) void {
        allocator.free(self.conv_state);
        allocator.free(self.ssm_state);
    }

    pub fn reset(self: *SSMState) void {
        @memset(self.conv_state, 0);
        @memset(self.ssm_state, 0);
        self.conv_pos = 0;
    }
};

/// Softplus activation: log(1 + exp(x))
/// For numerical stability, use: x + log(1 + exp(-x)) when x > 0
fn softplus(x: f32) f32 {
    if (x > 20.0) return x; // exp(-20) is negligible
    return std.math.log1p(std.math.exp(x));
}

/// Discretization step for state space model
/// Converts continuous parameters (A, B) to discrete (A_bar, B_bar)
/// Using zero-order hold (ZOH) discretization:
/// A_bar = exp(dt * A)
/// B_bar = (exp(dt * A) - 1) / A * B
fn discretizeZOH(
    A: f32, // Continuous state matrix diagonal element
    B: f32, // Continuous input matrix element
    dt: f32, // Time step (discretization rate)
) struct { A_bar: f32, B_bar: f32 } {
    const exp_adt = std.math.exp(dt * A);
    const A_bar = exp_adt;
    // Avoid division by zero when A is close to 0
    const B_bar = if (@abs(A) < 1e-6)
        dt * B
    else
        (exp_adt - 1.0) / A * B;
    return .{ .A_bar = A_bar, .B_bar = B_bar };
}

/// Selective scan recurrence
/// Computes: h_t = A_t * h_{t-1} + B_t * x_t
///           y_t = C_t * h_t
/// Where A, B, C are input-dependent (selective)
fn selectiveScanRecurrence(
    h_prev: f32, // Previous hidden state
    x: f32, // Input
    A_bar: f32, // Discretized A (scalar)
    B_bar: f32, // Discretized B (scalar)
    C: f32, // Output matrix element
) struct { h: f32, y: f32 } {
    const h = A_bar * h_prev + B_bar * x;
    const y = C * h;
    return .{ .h = h, .y = y };
}

/// Apply 1D convolution with SiLU activation
/// conv1d followed by SiLU (Swish): x * sigmoid(x)
fn applyConv1dSiLU(
    input: []const f32,
    conv_weights: []const f32, // [kernel_size]
    conv_bias: []const f32, // [hidden_dim]
    conv_state: []f32, // [kernel_size, hidden_dim] circular buffer
    conv_pos: usize,
    kernel_size: u32,
    output: []f32,
) void {
    const hidden_dim = input.len;

    // Store current input in circular buffer
    for (0..hidden_dim) |d| {
        const idx = (conv_pos % kernel_size) * hidden_dim + d;
        conv_state[idx] = input[d];
    }

    // Apply convolution
    @memset(output, 0);
    for (0..kernel_size) |k| {
        const buffer_pos = ((conv_pos + k + 1) % kernel_size) * hidden_dim;
        const weight = conv_weights[kernel_size - 1 - k]; // reversed for causal conv
        for (0..hidden_dim) |d| {
            output[d] += conv_state[buffer_pos + d] * weight;
        }
    }

    // Add bias and apply SiLU
    for (0..hidden_dim) |d| {
        const val = output[d] + conv_bias[d];
        output[d] = math.swish(val); // x * sigmoid(x)
    }
}

/// Mamba-2 SSM forward pass for a single token
/// Implements the selective state space model computation
pub fn ssmForward(
    hidden_dim: u32,
    ssm_inner: u32,
    ssm_groups: u32,
    dt_rank: u32,
    conv_kernel: u32,
    dt_scale: f32,

    input: []const f32,
    output: []f32,

    ssm_out_w: [*]const u8,
    ssm_x_w: [*]const u8,
    ssm_dt_w: [*]const u8,
    ssm_A: [*]const u8,
    ssm_B_w: [*]const u8,
    ssm_C_w: [*]const u8,
    ssm_D: [*]const u8,
    ssm_conv1d_w: [*]const u8,
    ssm_conv1d_b: [*]const u8,

    ssm_out_dtype: gguf.GGMLType,
    ssm_x_dtype: gguf.GGMLType,
    ssm_dt_dtype: gguf.GGMLType,
    ssm_B_dtype: gguf.GGMLType,
    ssm_C_dtype: gguf.GGMLType,

    state: *SSMState,

    tmp_x: []f32,
    tmp_z: []f32,
    tmp_dt: []f32,
    tmp_B: []f32,
    tmp_C: []f32,
) void {
    const state_per_group = hidden_dim / ssm_groups;

    // Step 1: Project input
    @memset(tmp_x, 0);
    @memset(tmp_z, 0);
    if (ssm_x_dtype != .f16) {
        parallel.parallelMatvec(
            ssm_x_w, input.ptr, tmp_x.ptr, hidden_dim, hidden_dim, ssm_x_dtype);
    } else {
        matvec.matvecDispatch(ssm_x_w, input.ptr, tmp_x.ptr, hidden_dim, hidden_dim, .{});
    }

    // Step 2: Apply convolution with SiLU
    // Extract conv weights and bias as fp16 slices
    const conv_w = @as([*]const types.fp16_t, @ptrCast(@alignCast(ssm_conv1d_w)))[0..conv_kernel];
    const conv_b = @as([*]const types.fp16_t, @ptrCast(@alignCast(ssm_conv1d_b)))[0..hidden_dim];

    var conv_weights: [8]f32 = undefined; // Max kernel size assumed 8
    var conv_bias: [8192]f32 = undefined; // Max hidden_dim

    for (0..conv_kernel) |k| {
        conv_weights[k] = types.fp16_to_fp32(conv_w[k]);
    }
    for (0..hidden_dim) |d| {
        conv_bias[d] = types.fp16_to_fp32(conv_b[d]);
    }

    var conv_out: [8192]f32 = undefined;
    applyConv1dSiLU(tmp_x, conv_weights[0..conv_kernel], conv_bias[0..hidden_dim], state.conv_state, state.conv_pos, conv_kernel, conv_out[0..hidden_dim]);

    // Step 3: Project to dt, B, C
    @memset(tmp_dt, 0);
    parallel.parallelMatvec(
        ssm_dt_w, conv_out[0..hidden_dim].ptr, tmp_dt.ptr, dt_rank, hidden_dim, ssm_dt_dtype);

    @memset(tmp_B, 0);
    parallel.parallelMatvec(
        ssm_B_w, conv_out[0..hidden_dim].ptr, tmp_B.ptr, ssm_inner, hidden_dim, ssm_B_dtype);

    @memset(tmp_C, 0);
    parallel.parallelMatvec(
        ssm_C_w, conv_out[0..hidden_dim].ptr, tmp_C.ptr, ssm_inner, hidden_dim, ssm_C_dtype);

    // Step 4: Apply selective scan for each group
    var scan_out: [8192]f32 = undefined; // [hidden_dim]

    for (0..ssm_groups) |g| {
        const group_offset = g * state_per_group;

        for (0..state_per_group) |d| {
            const idx = group_offset + d;

            // Get A diagonal (log-spaced initialization typically)
            const A_ptr = @as([*]const types.fp16_t, @ptrCast(@alignCast(ssm_A)));
            const A = types.fp16_to_fp32(A_ptr[d % ssm_inner]);

            // Get dt (with softplus and scale)
            const dt_idx = d % dt_rank;
            const dt_raw = tmp_dt[dt_idx] * dt_scale;
            const dt = softplus(dt_raw);

            // Discretize
            const B = tmp_B[d % ssm_inner];
            const disc = discretizeZOH(A, B, dt);

            // Get previous state
            const state_idx = (d % ssm_inner) * state_per_group + d;
            const h_prev = state.ssm_state[state_idx];

            // Recurrence
            const C = tmp_C[d % ssm_inner];
            const result = selectiveScanRecurrence(h_prev, conv_out[idx], disc.A_bar, disc.B_bar, C);

            // Update state and output
            state.ssm_state[state_idx] = result.h;
            scan_out[idx] = result.y;
        }
    }

    // Update conv position
    state.conv_pos += 1;

    // Step 5: Add skip connection (D * x)
    const D_ptr = @as([*]const types.fp16_t, @ptrCast(@alignCast(ssm_D)))[0..hidden_dim];
    for (0..hidden_dim) |d| {
        const D_val = types.fp16_to_fp32(D_ptr[d]);
        scan_out[d] += D_val * conv_out[d];
    }

    // Step 6: Output projection
    @memset(output, 0);
    parallel.parallelMatvec(
        ssm_out_w, scan_out[0..hidden_dim].ptr, output.ptr, hidden_dim, hidden_dim, ssm_out_dtype);
}

/// Initialize SSM A matrix with log-spaced values (typical Mamba initialization)
/// A_i = -exp(log_range_min + i * step)
pub fn initLogSpacedA(
    A: []f32,
    ssm_inner: u32,
    min_val: f32,
    max_val: f32,
) void {
    const step = (max_val - min_val) / @as(f32, @floatFromInt(ssm_inner - 1));
    for (0..ssm_inner) |i| {
        const val = min_val + @as(f32, @floatFromInt(i)) * step;
        A[i] = -std.math.exp(val); // Negative for stability
    }
}
