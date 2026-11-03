const std = @import("std");

pub const ModelConfig = extern struct {
    vocab_size: u32,
    hidden_dim: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    num_layers: u32,
    num_ssm_layers: u32,
    ffn_hidden_dim: u32,
    max_seq_len: u32,
    ffn_type: FFNType,
    norm_type: NormType,
    pos_encoding: PosEncoding,
    use_kv_quantization: bool,

    // Hybrid model configuration
    full_attention_interval: u32,

    // SSM configuration
    ssm_conv_kernel: u32,
    ssm_inner_size: u32,
    ssm_num_groups: u32,
    ssm_dt_rank: u32,
    ssm_dt_scale: f32,

    pub const FFNType = enum(c_int) {
        dense,
        gated_swi_glu,
        gated_gelu,
    };

    pub const NormType = enum(c_int) {
        layer_norm,
        rms_norm,
    };

    pub const PosEncoding = enum(c_int) {
        rope,
        learned,
        alibi,
        none,
    };

    pub const LayerType = enum(c_int) {
        attention,
        qwen_linear,
        ssm,
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
