//! `zig build test`: kullanılabilirlik katmanının birim testleri.
test {
    _ = @import("tokenizer/bpe.zig");
    _ = @import("tokenizer/nfc.zig");
    _ = @import("llama/sampler.zig");
}
