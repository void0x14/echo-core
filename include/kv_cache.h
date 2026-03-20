#pragma once
#include "types.h"
#include "config.h"
#include "kernels/quant.h"

// Single layer's KV cache — supports both FP32 and INT8 modes
struct alignas(64) KVCacheLayer {
    // INT8 quantized storage (used when use_kv_quantization = true)
    int8_t* keys_int8;       // [max_seq_len * num_kv_heads * head_dim] contiguous
    int8_t* values_int8;     // [max_seq_len * num_kv_heads * head_dim] contiguous
    float*  key_scales;      // [max_seq_len * num_kv_heads]
    float*  val_scales;      // [max_seq_len * num_kv_heads]

    // FP32 fallback storage (used when use_kv_quantization = false)
    float*  keys_fp32;       // [max_seq_len * num_kv_heads * head_dim]
    float*  values_fp32;     // [max_seq_len * num_kv_heads * head_dim]

    uint32_t seq_len;        // current number of cached positions
};

// Full KV cache for all layers
class KVCache {
    ModelConfig config_;
    KVCacheLayer* layers_;   // [num_layers] contiguous array
    uint8_t* storage_;       // single contiguous allocation for ALL data

public:
    explicit KVCache(const ModelConfig& config);
    ~KVCache();

    // Get cache for a specific layer
    KVCacheLayer& layer(uint32_t layer_idx);
    const KVCacheLayer& layer(uint32_t layer_idx) const;

    // Append new K, V projections to the cache (one token at a time during decode)
    // k_proj: [num_kv_heads * head_dim] FP32 — the new token's K projection
    // v_proj: [num_kv_heads * head_dim] FP32 — the new token's V projection
    void append(uint32_t layer_idx, const float* k_proj, const float* v_proj);

    // Get current sequence length
    uint32_t seq_len() const;

    // Reset cache (for new sequence)
    void reset();

    // Non-copyable
    KVCache(const KVCache&) = delete;
    KVCache& operator=(const KVCache&) = delete;
};
