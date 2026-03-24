const std = @import("std");

pub const ModelConfig = struct {
    vocab_size: u32,
    hidden_dim: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    num_layers: u32,
    ffn_hidden_dim: u32,
    max_seq_len: u32,
    ffn_type: FFNType,
    norm_type: NormType,
    pos_encoding: PosEncoding,
    use_kv_quantization: bool,

    pub const FFNType = enum {
        dense,
        gated_swi_glu,
        gated_gelu,
    };

    pub const NormType = enum {
        layer_norm,
        rms_norm,
    };

    pub const PosEncoding = enum {
        rope,
        learned,
        alibi,
        none,
    };
};

pub fn CacheTileConfig(comptime L3_SIZE_BYTES: u64) type {
    const budget = L3_SIZE_BYTES * 6 / 10;
    const tile_k: u32 = if (budget >= 18 * 1024 * 1024) 2048 else 1024;
    const tile_m: u32 = if (budget >= 18 * 1024 * 1024) 1024 else 512;
    const footprint = 2 * tile_m * tile_k + 4 * tile_k + 4 * tile_m;

    if (footprint > budget) {
        @compileError("Tile sizes exceed L3 cache budget");
    }

    return struct {
        pub const L3_BUDGET: u64 = budget;
        pub const TILE_K: u32 = tile_k;
        pub const TILE_M: u32 = tile_m;
        pub const TILE_FOOTPRINT: u64 = footprint;
    };
}

pub const Intel13500H_Tiles = CacheTileConfig(18 * 1024 * 1024);
pub const AMD_Ryzen3600_Tiles = CacheTileConfig(32 * 1024 * 1024);

test "CacheTileConfig compile-time validation" {
    try std.testing.expect(Intel13500H_Tiles.TILE_K == 1024);
    try std.testing.expect(Intel13500H_Tiles.TILE_M == 512);
    try std.testing.expect(AMD_Ryzen3600_Tiles.TILE_K == 2048);
    try std.testing.expect(AMD_Ryzen3600_Tiles.TILE_M == 1024);
}

test "ModelConfig basic creation" {
    const config = ModelConfig{
        .vocab_size = 32000,
        .hidden_dim = 4096,
        .num_heads = 32,
        .num_kv_heads = 32,
        .head_dim = 128,
        .num_layers = 32,
        .ffn_hidden_dim = 11008,
        .max_seq_len = 2048,
        .ffn_type = .gated_swi_glu,
        .norm_type = .rms_norm,
        .pos_encoding = .rope,
        .use_kv_quantization = true,
    };
    try std.testing.expect(config.vocab_size == 32000);
    try std.testing.expect(config.num_layers == 32);
}
