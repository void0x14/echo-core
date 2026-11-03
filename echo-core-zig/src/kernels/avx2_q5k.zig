const std = @import("std");
const types = @import("../core/types.zig");
const quant = @import("quant.zig");

pub fn matvecQ5K_avx2(blocks: [*]const u8, x: [*]const f32, y: [*]f32, M: u32, K: u32) void {
    const blocks_per_row: usize = @intCast(K / 256);
    const block_stride: usize = 176;
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
            const d = types.fp16_to_fp32(std.mem.readInt(u16, bp[0..2], .little));
            const dmin = types.fp16_to_fp32(std.mem.readInt(u16, bp[2..4], .little));
            const scales = bp + 4;
            const qh = bp + 16;
            const qs = bp + 48;
            const x_blk = x + b * 256;
            @prefetch(bp + 2 * block_stride, .{ .rw = .read, .locality = 2, .cache = .data });
            var rs: [8]f32 = undefined;
            var rm: [8]f32 = undefined;
            for (0..8) |s| {
                const sc: u8 = if (s < 4) scales[s] & 63 else (scales[s + 4] & 0x0F) | ((scales[s - 4] >> 6) << 4);
                const mn: u8 = if (s < 4) scales[s + 4] & 63 else (scales[s + 4] >> 4) | ((scales[s] >> 6) << 4);
                rs[s] = d * @as(f32, @floatFromInt(sc));
                rm[s] = dmin * @as(f32, @floatFromInt(mn));
            }
            inline for (0..32) |c| {
                const blk: usize = c / 8;
                const off: usize = c % 8;
                const high_seg: bool = off >= 4;
                const l: usize = (off % 4) * 8;
                const seg: usize = blk * 2 + @as(usize, @intFromBool(high_seg));
                const qbytes: @Vector(8, u8) = @as(*align(1) const [8]u8, @ptrCast(qs + blk * 32 + l)).*;
                const qhbytes: @Vector(8, u8) = @as(*align(1) const [8]u8, @ptrCast(qh + l)).*;
                const m4: @Vector(8, u8) = @splat(0x0F);
                const nib = if (!high_seg) qbytes & m4 else (qbytes >> @as(@Vector(8, u3), @splat(4))) & m4;
                const shift: u3 = @intCast(blk * 2 + @as(usize, @intFromBool(high_seg)));
                const hi = ((qhbytes >> @as(@Vector(8, u3), @splat(shift))) & @as(@Vector(8, u8), @splat(1))) << @as(@Vector(8, u3), @splat(4));
                const qv = nib | hi;
                const qf: @Vector(8, f32) = @floatFromInt(@as(@Vector(8, i32), @intCast(qv)));
                const wv = qf * @as(@Vector(8, f32), @splat(rs[seg])) - @as(@Vector(8, f32), @splat(rm[seg]));
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
pub fn matmatQ5K_avx2(blocks: [*]const u8, X: [*]const f32, Y: [*]f32, M: u32, K: u32, T: u32, ys: usize, yo: usize) void {
    const blocks_per_row: usize = @intCast(K / 256);
    const block_stride: usize = 176;
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
                const d = types.fp16_to_fp32(std.mem.readInt(u16, bp[0..2], .little));
                const dmin = types.fp16_to_fp32(std.mem.readInt(u16, bp[2..4], .little));
                const scales = bp + 4;
                const qh = bp + 16;
                const qs = bp + 48;
                var rs: [8]f32 = undefined;
                var rm: [8]f32 = undefined;
                for (0..8) |s| {
                    const sc: u8 = if (s < 4) scales[s] & 63 else (scales[s + 4] & 0x0F) | ((scales[s - 4] >> 6) << 4);
                    const mn: u8 = if (s < 4) scales[s + 4] & 63 else (scales[s + 4] >> 4) | ((scales[s] >> 6) << 4);
                    rs[s] = d * @as(f32, @floatFromInt(sc));
                    rm[s] = dmin * @as(f32, @floatFromInt(mn));
                }
                inline for (0..32) |c| {
                    const blk: usize = c / 8;
                    const off: usize = c % 8;
                    const high_seg: bool = off >= 4;
                    const l: usize = (off % 4) * 8;
                    const seg: usize = blk * 2 + @as(usize, @intFromBool(high_seg));
                    const qbytes: @Vector(8, u8) = @as(*align(1) const [8]u8, @ptrCast(qs + blk * 32 + l)).*;
                    const qhbytes: @Vector(8, u8) = @as(*align(1) const [8]u8, @ptrCast(qh + l)).*;
                    const m4: @Vector(8, u8) = @splat(0x0F);
                    const nib = if (!high_seg) qbytes & m4 else (qbytes >> @as(@Vector(8, u3), @splat(4))) & m4;
                    const shift: u3 = @intCast(blk * 2 + @as(usize, @intFromBool(high_seg)));
                    const hi = ((qhbytes >> @as(@Vector(8, u3), @splat(shift))) & @as(@Vector(8, u8), @splat(1))) << @as(@Vector(8, u3), @splat(4));
                    const qv = nib | hi;
                    const qf: @Vector(8, f32) = @floatFromInt(@as(@Vector(8, i32), @intCast(qv)));
                    const wv = qf * @as(@Vector(8, f32), @splat(rs[seg])) - @as(@Vector(8, f32), @splat(rm[seg]));
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
