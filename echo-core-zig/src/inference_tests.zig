const std = @import("std");
const engine = @import("inference/engine.zig");

test {
    std.testing.refAllDecls(engine);
}
