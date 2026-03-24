const std = @import("std");
const tokenizer = @import("tokenizer/tokenizer.zig");

test {
    std.testing.refAllDecls(tokenizer);
}
