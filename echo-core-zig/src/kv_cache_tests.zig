const std = @import("std");
const cache = @import("kv_cache/cache.zig");

test {
    std.testing.refAllDecls(cache);
}
