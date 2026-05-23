const std = @import("std");
const types = @import("../core/types.zig");
const quant = @import("quant.zig");

fn dotQ4Block(block: [*]const u8, x: [*]const f32) f32 {
    const d = types.fp16_to_fp32(std.mem.readInt(u16, block[0..2], .little));
    const dmin = types.fp16_to_fp32(std.mem.readInt(u16, block[2..4], .little));
    const scales: [*]const u8 = @ptrCast(block + 4);
    const qs_ptr = block + 16;

    var acc: @Vector(8, f32) = @splat(0);

    var blk: u32 = 0;
    while (blk < 4) : (blk += 1) {
        const js = blk * 2;
        const sc0: u8 = if (js < 4) scales[js] & 63 else (scales[js + 4] & 0x0F) | ((scales[js - 4] >> 6) << 4);
        const mn0: u8 = if (js < 4) scales[js + 4] & 63 else (scales[js + 4] >> 4) | ((scales[js] >> 6) << 4);
        const sc1: u8 = if (js + 1 < 4) scales[js + 1] & 63 else (scales[js + 1 + 4] & 0x0F) | ((scales[js + 1 - 4] >> 6) << 4);
        const mn1: u8 = if (js + 1 < 4) scales[js + 1 + 4] & 63 else (scales[js + 1 + 4] >> 4) | ((scales[js + 1] >> 6) << 4);
        const qoff: usize = blk * 32;
        const woff: usize = blk * 64;
        const rs0 = d * @as(f32, @floatFromInt(sc0));
        const rm0 = dmin * @as(f32, @floatFromInt(mn0));
        const rs1 = d * @as(f32, @floatFromInt(sc1));
        const rm1 = dmin * @as(f32, @floatFromInt(mn1));

        const qb_arr = load32u(qs_ptr + qoff);
        const qb: @Vector(32, u8) = qb_arr;
        const qlow: @Vector(32, u8) = qb & @as(@Vector(32, u8), @splat(0x0F));
        const qhigh: @Vector(32, u8) = (qb >> @as(@Vector(32, u8), @splat(4))) & @as(@Vector(32, u8), @splat(0x0F));

        const qla: [32]u8 = @bitCast(qlow);
        const qha: [32]u8 = @bitCast(qhigh);

        const wl0 = vec8uToF32(qla[0..8].*);
        const wl1 = vec8uToF32(qla[16..24].*);
        const wl2 = vec8uToF32(qla[8..16].*);
        const wl3 = vec8uToF32(qla[24..32].*);
        const wh0 = vec8uToF32(qha[0..8].*);
        const wh1 = vec8uToF32(qha[16..24].*);
        const wh2 = vec8uToF32(qha[8..16].*);
        const wh3 = vec8uToF32(qha[24..32].*);

        const R0 = @as(@Vector(8, f32), @splat(rs0));
        const R1 = @as(@Vector(8, f32), @splat(rs1));
        const M0 = @as(@Vector(8, f32), @splat(rm0));
        const M1 = @as(@Vector(8, f32), @splat(rm1));

        const x0_arr = load8f(x + woff + 0);
        const x1_arr = load8f(x + woff + 8);
        const x2_arr = load8f(x + woff + 16);
        const x3_arr = load8f(x + woff + 24);
        const x4_arr = load8f(x + woff + 32);
        const x5_arr = load8f(x + woff + 40);
        const x6_arr = load8f(x + woff + 48);
        const x7_arr = load8f(x + woff + 56);

        const x0: @Vector(8, f32) = x0_arr;
        const x1: @Vector(8, f32) = x1_arr;
        const x2: @Vector(8, f32) = x2_arr;
        const x3: @Vector(8, f32) = x3_arr;
        const x4: @Vector(8, f32) = x4_arr;
        const x5: @Vector(8, f32) = x5_arr;
        const x6: @Vector(8, f32) = x6_arr;
        const x7: @Vector(8, f32) = x7_arr;

        acc += (R0 * wl0 - M0) * x0;
        acc += (R1 * wl1 - M1) * x2;
        acc += (R0 * wl2 - M0) * x1;
        acc += (R1 * wl3 - M1) * x3;
        acc += (R0 * wh0 - M0) * x4;
        acc += (R1 * wh1 - M1) * x6;
        acc += (R0 * wh2 - M0) * x5;
        acc += (R1 * wh3 - M1) * x7;
    }

    return @reduce(.Add, acc);
}

inline fn load32u(p: [*]const u8) [32]u8 {
    const arr: *align(1) const [32]u8 = @ptrCast(p);
    return arr.*;
}

inline fn load8f(p: [*]const f32) [8]f32 {
    const arr: *align(1) const [8]f32 = @ptrCast(p);
    return arr.*;
}

inline fn vec8uToF32(a: [8]u8) @Vector(8, f32) {
    const v: @Vector(8, u8) = a;
    const i: @Vector(8, i32) = @intCast(v);
    return @floatFromInt(i);
}

pub fn matvecQ4K_avx2(blocks: [*]const u8, x: [*]const f32, y: [*]f32, M: u32, K: u32) void {
    var m: u32 = 0;
    while (m < M) : (m += 1) {
        var sum: f32 = 0;
        const blocks_per_row = K / 256;
        const row_ptr = blocks + @as(usize, m) * blocks_per_row * 144;
        var b: u32 = 0;
        while (b < blocks_per_row) : (b += 1) {
            sum += dotQ4Block(row_ptr + b * 144, x + b * 256);
        }
        y[m] += sum;
    }
}

test "Q4_K AVX2 dot basic" {
    var block = std.mem.zeroes(quant.block_q4_K);
    block.d = types.fp32_to_fp16(63); block.dmin = types.fp32_to_fp16(63);
    for (0..4) |j| { block.scales[j] = @intCast(j+1); block.scales[j+4] = @intCast(j); }
    block.scales[0] |= 1<<6; block.scales[1] |= 1<<6; block.scales[2] |= 1<<6; block.scales[3] |= 1<<6;
    block.scales[8] = 0x12; block.scales[9] = 0x23; block.scales[10] = 0x34; block.scales[11] = 0x45;
    @memset(&block.qs, 0x55);
    var x: [256]f32 = undefined;
    for (0..x.len) |i| x[i] = @as(f32, @floatFromInt((i % 11) + 1)) * 0.01;
    var y = [_]f32{0};
    matvecQ4K_avx2(std.mem.asBytes(&block).ptr, &x, &y, 1, 256);

    const df = types.fp16_to_fp32(block.d); const dmf = types.fp16_to_fp32(block.dmin);
    var exp: f32 = 0;
    for (0..4) |blk| { const js = blk*2;
        const sc0: u8 = if(js<4)block.scales[js]&63 else(block.scales[js+4]&0x0F)|((block.scales[js-4]>>6)<<4);
        const mn0: u8 = if(js<4)block.scales[js+4]&63 else(block.scales[js+4]>>4)|((block.scales[js]>>6)<<4);
        const sc1: u8 = if(js+1<4)block.scales[js+1]&63 else(block.scales[js+1+4]&0x0F)|((block.scales[js+1-4]>>6)<<4);
        const mn1: u8 = if(js+1<4)block.scales[js+1+4]&63 else(block.scales[js+1+4]>>4)|((block.scales[js+1]>>6)<<4);
        const rs0=df*@as(f32,@floatFromInt(sc0)); const rm0=dmf*@as(f32,@floatFromInt(mn0));
        const rs1=df*@as(f32,@floatFromInt(sc1)); const rm1=dmf*@as(f32,@floatFromInt(mn1));
        const qo=blk*32; const wo=blk*64;
        for (0..16) |k| {
            exp+=(rs0*@as(f32,@floatFromInt(block.qs[qo+k]&0x0F))-rm0)*x[wo+k];
            exp+=(rs1*@as(f32,@floatFromInt(block.qs[qo+16+k]&0x0F))-rm1)*x[wo+16+k];
            exp+=(rs0*@as(f32,@floatFromInt(block.qs[qo+k]>>4))-rm0)*x[wo+32+k];
            exp+=(rs1*@as(f32,@floatFromInt(block.qs[qo+16+k]>>4))-rm1)*x[wo+48+k];
        }
    }
    try std.testing.expectApproxEqAbs(exp, y[0], 0.01);
}
