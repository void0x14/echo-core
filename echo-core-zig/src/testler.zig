//! `zig build test`: kullanılabilirlik katmanının birim testleri.
test {
    _ = @import("tokenizer/bpe.zig");
}
