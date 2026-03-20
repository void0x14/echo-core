#pragma once

#include <cstdint>
#include <cstddef>

struct ModelConfig {
    uint32_t vocab_size;
    uint32_t hidden_dim;       // d_model
    uint32_t num_heads;        // e.g. 32
    uint32_t num_kv_heads;     // for GQA support (same as num_heads if no GQA)
    uint32_t head_dim;         // e.g. 128
    uint32_t num_layers;
    uint32_t ffn_hidden_dim;   // intermediate size
    uint32_t max_seq_len;

    enum class FFNType { Dense, GatedSwiGLU, GatedGeLU };
    FFNType ffn_type;

    enum class NormType { LayerNorm, RMSNorm };
    NormType norm_type;

    enum class PosEncoding { RoPE, Learned, ALiBi, None };
    PosEncoding pos_encoding;

    bool use_kv_quantization;
};

// Compile-time cache tile configuration
// L3_SIZE_BYTES: total L3 cache size in bytes
// The 0.6 factor avoids OS-induced cache thrashing from targeting 100% of L3
template <uint64_t L3_SIZE_BYTES>
struct CacheTileConfig {
    static constexpr uint64_t L3_BUDGET = L3_SIZE_BYTES * 6 / 10;

    // Inner dimension tile (K): must be multiple of 8 for AVX2 (8x FP32 per 256-bit register)
    // Larger K = better vectorization efficiency per output element
    static constexpr uint32_t TILE_K = (L3_BUDGET >= 18ULL * 1024 * 1024) ? 2048 : 1024;

    // Outer dimension tile (M): rows of output processed simultaneously
    // Limited by accumulator cache residency
    static constexpr uint32_t TILE_M = (L3_BUDGET >= 18ULL * 1024 * 1024) ? 1024 : 512;

    // Verify constraint at compile time
    // footprint = 2*TILE_M*TILE_K + 4*TILE_K + 4*TILE_M <= L3_BUDGET
    // 2 from FP16 weights, 4 from FP32 vectors
    static_assert(
        2ULL * TILE_M * TILE_K + 4ULL * TILE_K + 4ULL * TILE_M <= L3_BUDGET,
        "Tile sizes exceed L3 cache budget"
    );

    static constexpr uint64_t TILE_FOOTPRINT = 2ULL * TILE_M * TILE_K + 4ULL * TILE_K + 4ULL * TILE_M;
};

// Hardware presets — INTEL has SMALLER tiles, AMD has LARGER tiles
// Intel i5-13500H: 18MB L3, single-channel DDR4 ~25GB/s → TILE_K=1024, TILE_M=512
using Intel13500H_Tiles = CacheTileConfig<18ULL * 1024 * 1024>;

// AMD Ryzen 5 3600: 32MB L3, dual-channel DDR4 ~50GB/s → TILE_K=2048, TILE_M=1024
using AMD_Ryzen3600_Tiles = CacheTileConfig<32ULL * 1024 * 1024>;
