const std = @import("std");
const types = @import("../core/types.zig");
const quant = @import("quant.zig");

pub fn matvecQ6K_avx2(blocks: [*]const u8, x: [*]const f32, y: [*]f32, M: u32, K: u32) void {
    const blocks_per_row: usize = @intCast(K / 256);
    const block_stride: usize = 210;
    var m: u32 = 0;
    while (m < M) : (m += 1) {
        const row_ptr = blocks + @as(usize, m) * blocks_per_row * block_stride;
        var acc0: @Vector(8, f32) = @splat(0);
        var acc1: @Vector(8, f32) = @splat(0);
        var acc2: @Vector(8, f32) = @splat(0);
        var acc3: @Vector(8, f32) = @splat(0);
        var b: usize = 0;
        while (b < blocks_per_row) : (b += 1) {
            const bp = row_ptr + b * block_stride;
            const ql = bp;
            const qh = bp + 128;
            const scales: [*]const i8 = @ptrCast(bp + 192);
            const d = types.fp16_to_fp32(std.mem.readInt(u16, bp[208..210], .little));
            const x_blk = x + b * 256;
            @prefetch(bp + 2 * block_stride, .{ .rw = .read, .locality = 2, .cache = .data });
            inline for (0..32) |c| {
                const half: usize = c / 16;
                const within_chunk: usize = c % 16;
                const within_start: usize = within_chunk * 8;
                const group: usize = within_start / 32;
                const idx: usize = within_start % 32;
                const ql_off: usize = half * 64 + (group % 2) * 32 + idx;
                const qbytes: @Vector(8, u8) = @as(*align(1) const [8]u8, @ptrCast(ql + ql_off)).*;
                const qhbytes: @Vector(8, u8) = @as(*align(1) const [8]u8, @ptrCast(qh + half * 32 + idx)).*;
                const m4: @Vector(8, u8) = @splat(0x0F);
                const nib = if (group < 2) qbytes & m4 else (qbytes >> @as(@Vector(8, u3), @splat(4))) & m4;
                const shift: u3 = @intCast(group * 2);
                const hi = ((qhbytes >> @as(@Vector(8, u3), @splat(shift))) & @as(@Vector(8, u8), @splat(3))) << @as(@Vector(8, u3), @splat(4));
                const qv = nib | hi;
                const qf: @Vector(8, f32) = @floatFromInt(@as(@Vector(8, i32), @intCast(qv)));
                const sc: f32 = @floatFromInt(scales[half * 8 + within_start / 16]);
                const wv = (qf - @as(@Vector(8, f32), @splat(32.0))) * @as(@Vector(8, f32), @splat(d * sc));
                const xv: @Vector(8, f32) = @as(*align(1) const [8]f32, @ptrCast(x_blk + c * 8)).*;
                switch (c % 4) {
                    0 => acc0 = @mulAdd(@Vector(8, f32), wv, xv, acc0),
                    1 => acc1 = @mulAdd(@Vector(8, f32), wv, xv, acc1),
                    2 => acc2 = @mulAdd(@Vector(8, f32), wv, xv, acc2),
                    3 => acc3 = @mulAdd(@Vector(8, f32), wv, xv, acc3),
                    else => unreachable,
                }
            }
        }
        y[m] += @reduce(.Add, acc0 + acc1 + acc2 + acc3);
    }
}
pub fn matmatQ6K_avx2(blocks: [*]const u8, X: [*]const f32, Y: [*]f32, M: u32, K: u32, T: u32, ys: usize, yo: usize) void {
    const blocks_per_row: usize = @intCast(K / 256);
    const block_stride: usize = 210;
    const Ks: usize = @intCast(K);
    const Ts: usize = @intCast(T);
    var m: u32 = 0;
    while (m < M) : (m += 1) {
        const row_ptr = blocks + @as(usize, m) * blocks_per_row * block_stride;
        var acc = [_]@Vector(8, f32){ @splat(0), @splat(0), @splat(0), @splat(0), @splat(0), @splat(0), @splat(0), @splat(0) };
        var tb: usize = 0;
        while (tb < Ts) : (tb += 8) {
            const tn = @min(@as(usize, 8), Ts - tb);
            inline for (0..8) |tt| acc[tt] = @splat(0);
            var b: usize = 0;
            while (b < blocks_per_row) : (b += 1) {
                const bp = row_ptr + b * block_stride;
                @prefetch(bp + 2 * block_stride, .{ .rw = .read, .locality = 2, .cache = .data });
                const ql = bp;
                const qh = bp + 128;
                const scales: [*]const i8 = @ptrCast(bp + 192);
                const d = types.fp16_to_fp32(std.mem.readInt(u16, bp[208..210], .little));
                inline for (0..32) |c| {
                    const half: usize = c / 16;
                    const within_chunk: usize = c % 16;
                    const within_start: usize = within_chunk * 8;
                    const group: usize = within_start / 32;
                    const idx: usize = within_start % 32;
                    const ql_off: usize = half * 64 + (group % 2) * 32 + idx;
                    const qbytes: @Vector(8, u8) = @as(*align(1) const [8]u8, @ptrCast(ql + ql_off)).*;
                    const qhbytes: @Vector(8, u8) = @as(*align(1) const [8]u8, @ptrCast(qh + half * 32 + idx)).*;
                    const m4: @Vector(8, u8) = @splat(0x0F);
                    const nib = if (group < 2) qbytes & m4 else (qbytes >> @as(@Vector(8, u3), @splat(4))) & m4;
                    const shift: u3 = @intCast(group * 2);
                    const hi = ((qhbytes >> @as(@Vector(8, u3), @splat(shift))) & @as(@Vector(8, u8), @splat(3))) << @as(@Vector(8, u3), @splat(4));
                    const qv = nib | hi;
                    const qf: @Vector(8, f32) = @floatFromInt(@as(@Vector(8, i32), @intCast(qv)));
                    const sc: f32 = @floatFromInt(scales[half * 8 + within_start / 16]);
                    const wv = (qf - @as(@Vector(8, f32), @splat(32.0))) * @as(@Vector(8, f32), @splat(d * sc));
                    inline for (0..8) |tt| {
                        if (tt < tn) {
                            const xv: @Vector(8, f32) = @as(*align(1) const [8]f32, @ptrCast(X + (tb + tt) * Ks + b * 256 + c * 8)).*;
                            acc[tt] = @mulAdd(@Vector(8, f32), wv, xv, acc[tt]);
                        }
                    }
                }
            }
            inline for (0..8) |tt| {
                if (tt < tn) Y[(tb + tt) * ys + yo + m] = @reduce(.Add, acc[tt]);
            }
        }
    }
}
