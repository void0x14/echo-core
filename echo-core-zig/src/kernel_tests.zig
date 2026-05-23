const std = @import("std");
const quant = @import("kernels/quant.zig");
const matvec = @import("kernels/matvec.zig");
const simd = @import("kernels/simd.zig");
const avx2_q6k = @import("kernels/avx2_q6k.zig");
const avx2_q4k = @import("kernels/avx2_q4k.zig");
const avx2_q5k = @import("kernels/avx2_q5k.zig");
const avx2_softmax = @import("kernels/avx2_softmax.zig");
const avx2_norm = @import("kernels/avx2_norm.zig");

test {
    std.testing.refAllDecls(quant);
    std.testing.refAllDecls(matvec);
    std.testing.refAllDecls(simd);
    std.testing.refAllDecls(avx2_q6k);
    std.testing.refAllDecls(avx2_q4k);
    std.testing.refAllDecls(avx2_q5k);
    std.testing.refAllDecls(avx2_softmax);
    std.testing.refAllDecls(avx2_norm);
}
