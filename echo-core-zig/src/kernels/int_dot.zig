const std = @import("std");
const types = @import("../core/types.zig");
const quant = @import("quant.zig");

pub fn quantizeXToQ8K(x: [*]const f32, q8: [*]quant.block_q8_K, K: usize) void {
    var b: usize = 0;
    while (b < K / 256) : (b += 1) {
        const bx = x + b * 256;
        var amax: f32 = 0;
        var ii: usize = 0;
        while (ii < 256) : (ii += 8) {
            const a = amax_v8(@as(*align(1) const [8]f32, @ptrCast(bx + ii)).*);
            if (a > amax) amax = a;
        }
        if (amax < 1e-30) { q8[b].d = 0; @memset(&q8[b].qs, 0); @memset(&q8[b].bsums, 0); continue; }
        q8[b].d = amax / 127.0;
        const isc_v = @as(@Vector(8, f32), @splat(127.0 / amax));
        ii = 0;
        while (ii < 256) : (ii += 8) {
            const xv: @Vector(8, f32) = @as(*align(1) const [8]f32, @ptrCast(bx + ii)).*;
            const sc: [8]f32 = xv * isc_v;
            for (0..8) |j_| {
                const v: i32 = @intFromFloat(@round(sc[j_]));
                q8[b].qs[ii + j_] = @intCast(@max(-128, @min(127, v)));
            }
        }
        var j: usize = 0;
        while (j < 16) : (j += 4) {
            const vi0: @Vector(16, i8) = @as(*align(1) const [16]i8, @ptrCast(&q8[b].qs[j * 16])).*;
            const vi1: @Vector(16, i8) = @as(*align(1) const [16]i8, @ptrCast(&q8[b].qs[(j + 1) * 16])).*;
            const vi2: @Vector(16, i8) = @as(*align(1) const [16]i8, @ptrCast(&q8[b].qs[(j + 2) * 16])).*;
            const vi3: @Vector(16, i8) = @as(*align(1) const [16]i8, @ptrCast(&q8[b].qs[(j + 3) * 16])).*;
            q8[b].bsums[j] = @intCast(@reduce(.Add, @as(@Vector(16, i32), @intCast(vi0))));
            q8[b].bsums[j + 1] = @intCast(@reduce(.Add, @as(@Vector(16, i32), @intCast(vi1))));
            q8[b].bsums[j + 2] = @intCast(@reduce(.Add, @as(@Vector(16, i32), @intCast(vi2))));
            q8[b].bsums[j + 3] = @intCast(@reduce(.Add, @as(@Vector(16, i32), @intCast(vi3))));
        }
    }
}

inline fn amax_v8(a: [8]f32) f32 {
    const x: @Vector(8, f32) = a;
    return @reduce(.Max, @select(f32, x < @as(@Vector(8, f32), @splat(0)), -x, x));
}

fn dotQ4K(q4: *align(1) const quant.block_q4_K, q8: *const quant.block_q8_K) f32 {
    const d0 = types.fp16_to_fp32(q4.d);
    const dmin0 = types.fp16_to_fp32(q4.dmin);
    const df = q8.d * d0;
    const dmf = -q8.d * dmin0;

    var sumi: i32 = 0;
    var mcorr: f32 = 0;

    var j: u32 = 0;
    while (j < 4) : (j += 1) {
        const si = j * 2;
        const mn0: u8 = if (si < 4) q4.scales[si + 4] & 63 else (q4.scales[si + 4] >> 4) | ((q4.scales[si] >> 6) << 4);
        const mn1: u8 = if (si + 1 < 4) q4.scales[si + 1 + 4] & 63 else (q4.scales[si + 1 + 4] >> 4) | ((q4.scales[si + 1] >> 6) << 4);

        const q4raw: @Vector(32, u8) = @as(*align(1) const [32]u8, @ptrCast(&q4.qs[j * 32])).*;
        const q4l: @Vector(32, i32) = @intCast(q4raw & @as(@Vector(32, u8), @splat(0x0F)));
        const q4h: @Vector(32, i32) = @intCast((q4raw >> @as(@Vector(32, u8), @splat(4))) & @as(@Vector(32, u8), @splat(0x0F)));
        const q8l_v: @Vector(32, i8) = @as(*align(1) const [32]i8, @ptrCast(&q8.qs[j * 64])).*;
        const q8h_v: @Vector(32, i8) = @as(*align(1) const [32]i8, @ptrCast(&q8.qs[j * 64 + 32])).*;
        const q8l: @Vector(32, i32) = @intCast(q8l_v);
        const q8h: @Vector(32, i32) = @intCast(q8h_v);

        sumi += @reduce(.Add, q4l * q8l);
        sumi += @reduce(.Add, q4h * q8h);

        mcorr += dmf * (@as(f32, @floatFromInt(mn0)) * @as(f32, @floatFromInt(@reduce(.Add, q8l))) +
                          @as(f32, @floatFromInt(mn1)) * @as(f32, @floatFromInt(@reduce(.Add, q8h))));
    }

    return df * @as(f32, @floatFromInt(sumi)) + mcorr;
}

pub fn matvecQ4K_int(blocks: [*]const u8, x: [*]const f32, y: [*]f32, M: u32, K: u32) void {
    const bpr = K / 256;
    var q8b: [64]quant.block_q8_K = undefined;
    quantizeXToQ8K(x, @ptrCast(&q8b), @intCast(K));
    var m: u32 = 0;
    while (m < M) : (m += 1) {
        var s: f32 = 0;
        const rp = blocks + @as(usize, m) * @as(usize, bpr) * 144;
        for (0..bpr) |b| s += dotQ4K(@ptrCast(rp + b * 144), &q8b[b]);
        y[m] += s;
    }
}

pub fn matvecQ4K_intPre(blocks: [*]const u8, q8_pre: [*]const quant.block_q8_K, y: [*]f32, M: u32, K: u32) void {
    const bpr = K / 256;
    var m: u32 = 0;
    while (m < M) : (m += 1) {
        var s: f32 = 0;
        const rp = blocks + @as(usize, m) * @as(usize, bpr) * 144;
        for (0..bpr) |b| s += dotQ4K(@ptrCast(rp + b * 144), &q8_pre[b]);
        y[m] += s;
    }
}
pub fn matmatQ4K_int(blocks: [*]const u8, X: [*]const f32, Y: [*]f32, M: u32, K: u32, T: u32, ys: usize, yo: usize) void {
    const bpr: usize = @intCast(K / 256);
    const nblocks = @as(usize, T) * bpr;
    const q8all = std.heap.smp_allocator.alloc(quant.block_q8_K, @max(nblocks, 1)) catch {
        for (0..T) |t| matvecQ4K_int(blocks, X + t * @as(usize, K), Y + t * ys + yo, M, K);
        return;
    };
    defer std.heap.smp_allocator.free(q8all);
    for (0..T) |t| quantizeXToQ8K(X + t * @as(usize, K), q8all.ptr + t * bpr, @intCast(K));
    var m: u32 = 0;
    while (m < M) : (m += 1) {
        const rp = blocks + @as(usize, m) * bpr * 144;
        for (0..T) |t| {
            var s: f32 = 0;
            for (0..bpr) |b| s += dotQ4K(@ptrCast(rp + b * 144), &q8all[t * bpr + b]);
            Y[t * ys + yo + m] = s;
        }
    }
}
