const std = @import("std");
const quant = @import("kernels/quant.zig");
const matvec = @import("kernels/matvec.zig");

test {
    std.testing.refAllDecls(quant);
    std.testing.refAllDecls(matvec);
}
