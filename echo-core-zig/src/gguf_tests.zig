const std = @import("std");
const reader = @import("gguf/reader.zig");

test {
    std.testing.refAllDecls(reader);
}
