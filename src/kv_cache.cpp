#include "kv_cache.h"
#include <cstdlib>
#include <cstring>
#include <cassert>

KVCache::KVCache(const ModelConfig& config)
    : config_(config), layers_(nullptr), storage_(nullptr)
{
    const uint32_t num_layers  = config_.num_layers;
    const uint32_t kv_heads    = config_.num_kv_heads;
    const uint32_t head_dim    = config_.head_dim;
    const uint32_t max_seq     = config_.max_seq_len;
    const uint32_t row_bytes   = kv_heads * head_dim;

    // Allocate layers array (64-byte aligned for alignas(64) struct)
    void* layers_mem = nullptr;
    int ret = posix_memalign(&layers_mem, 64, sizeof(KVCacheLayer) * num_layers);
    assert(ret == 0 && "posix_memalign failed for KVCacheLayer array");
    layers_ = static_cast<KVCacheLayer*>(layers_mem);
    std::memset(layers_, 0, sizeof(KVCacheLayer) * num_layers);

    if (max_seq == 0 || num_layers == 0) return;

    // Compute total bytes needed
    size_t total_bytes;
    if (config_.use_kv_quantization) {
        size_t keys_bytes   = static_cast<size_t>(max_seq) * row_bytes;
        size_t values_bytes = keys_bytes;
        size_t scales_bytes = static_cast<size_t>(max_seq) * kv_heads * sizeof(float);
        size_t per_layer    = keys_bytes + values_bytes + scales_bytes * 2;
        total_bytes = per_layer * num_layers;
    } else {
        size_t per_layer = static_cast<size_t>(2) * max_seq * row_bytes * sizeof(float);
        total_bytes = per_layer * num_layers;
    }

    storage_ = static_cast<uint8_t*>(std::malloc(total_bytes));
    assert(storage_ && "malloc failed for KV cache storage");
    std::memset(storage_, 0, total_bytes);

    // Partition storage into layers
    uint8_t* ptr = storage_;

    if (config_.use_kv_quantization) {
        size_t keys_bytes   = static_cast<size_t>(max_seq) * row_bytes;
        size_t values_bytes = keys_bytes;
        size_t scales_count = static_cast<size_t>(max_seq) * kv_heads;
        size_t scales_bytes = scales_count * sizeof(float);

        for (uint32_t l = 0; l < num_layers; ++l) {
            layers_[l].keys_int8   = reinterpret_cast<int8_t*>(ptr); ptr += keys_bytes;
            layers_[l].values_int8 = reinterpret_cast<int8_t*>(ptr); ptr += values_bytes;
            layers_[l].key_scales  = reinterpret_cast<float*>(ptr);  ptr += scales_bytes;
            layers_[l].val_scales  = reinterpret_cast<float*>(ptr);  ptr += scales_bytes;
            layers_[l].keys_fp32   = nullptr;
            layers_[l].values_fp32 = nullptr;
            layers_[l].seq_len     = 0;
        }
    } else {
        size_t data_bytes = static_cast<size_t>(max_seq) * row_bytes * sizeof(float);

        for (uint32_t l = 0; l < num_layers; ++l) {
            layers_[l].keys_fp32   = reinterpret_cast<float*>(ptr); ptr += data_bytes;
            layers_[l].values_fp32 = reinterpret_cast<float*>(ptr); ptr += data_bytes;
            layers_[l].keys_int8   = nullptr;
            layers_[l].values_int8 = nullptr;
            layers_[l].key_scales  = nullptr;
            layers_[l].val_scales  = nullptr;
            layers_[l].seq_len     = 0;
        }
    }
}

KVCache::~KVCache() {
    std::free(layers_);
    std::free(storage_);
}

KVCacheLayer& KVCache::layer(uint32_t layer_idx) {
    assert(layer_idx < config_.num_layers);
    return layers_[layer_idx];
}

const KVCacheLayer& KVCache::layer(uint32_t layer_idx) const {
    assert(layer_idx < config_.num_layers);
    return layers_[layer_idx];
}

void KVCache::append(uint32_t layer_idx, const float* k_proj, const float* v_proj) {
    KVCacheLayer& l = layers_[layer_idx];
    assert(l.seq_len < config_.max_seq_len && "KV cache overflow");

    const uint32_t kv_heads  = config_.num_kv_heads;
    const uint32_t head_dim  = config_.head_dim;
    const uint32_t row_bytes = kv_heads * head_dim;
    const size_t   pos       = l.seq_len;

    if (config_.use_kv_quantization) {
        // Quantize 1 token's K projection
        int8_t* k_dst = l.keys_int8 + pos * row_bytes;
        quantize_per_token_symmetric(k_proj, k_dst, &l.key_scales[pos], 1, row_bytes);

        // Quantize 1 token's V projection
        int8_t* v_dst = l.values_int8 + pos * row_bytes;
        quantize_per_token_symmetric(v_proj, v_dst, &l.val_scales[pos], 1, row_bytes);
    } else {
        // FP32: plain copy
        std::memcpy(l.keys_fp32   + pos * row_bytes, k_proj, row_bytes * sizeof(float));
        std::memcpy(l.values_fp32 + pos * row_bytes, v_proj, row_bytes * sizeof(float));
    }

    l.seq_len++;
}

uint32_t KVCache::seq_len() const {
    if (config_.num_layers == 0) return 0;
    return layers_[0].seq_len;
}

void KVCache::reset() {
    for (uint32_t l = 0; l < config_.num_layers; ++l) {
        layers_[l].seq_len = 0;
    }
}

#ifdef KV_CACHE_TEST_MAIN
#include <cstdio>
#include <cmath>
#include <algorithm>

int main() {
    bool all_pass = true;

    // Small test config: 2 layers, 4 KV heads, head_dim=8, max_seq=16
    auto make_config = [](bool use_quant) -> ModelConfig {
        return ModelConfig{
            .vocab_size        = 1000,
            .hidden_dim        = 64,
            .num_heads         = 8,
            .num_kv_heads      = 4,
            .head_dim          = 8,
            .num_layers        = 2,
            .ffn_hidden_dim    = 128,
            .max_seq_len       = 16,
            .ffn_type          = ModelConfig::FFNType::Dense,
            .norm_type         = ModelConfig::NormType::LayerNorm,
            .pos_encoding      = ModelConfig::PosEncoding::RoPE,
            .use_kv_quantization = use_quant,
        };
    };

    const uint32_t kv_heads   = 4;
    const uint32_t head_dim   = 8;
    const uint32_t row_elems  = kv_heads * head_dim; // 32
    const uint32_t num_tokens = 5;

    // ----------------------------------------------------------------
    // Test 1: INT8 mode
    // ----------------------------------------------------------------
    printf("=== Test 1: KVCache INT8 mode ===\n");
    {
        ModelConfig cfg = make_config(true);
        KVCache cache(cfg);

        // Append 5 tokens to layer 0 with known values
        for (uint32_t t = 0; t < num_tokens; ++t) {
            float k_proj[32], v_proj[32];
            for (uint32_t i = 0; i < row_elems; ++i) {
                k_proj[i] = static_cast<float>(t * 10 + i);
                v_proj[i] = static_cast<float>(t * 100 + i);
            }
            cache.append(0, k_proj, v_proj);
        }

        bool seq_ok = cache.seq_len() == num_tokens;
        printf("  seq_len = %u (expected %u) %s\n",
               cache.seq_len(), num_tokens, seq_ok ? "OK" : "FAIL");
        if (!seq_ok) all_pass = false;

        // Verify INT8 pointers are set
        const KVCacheLayer& l0 = cache.layer(0);
        printf("  keys_int8   != NULL: %s\n", l0.keys_int8   ? "OK" : "FAIL");
        printf("  values_int8 != NULL: %s\n", l0.values_int8 ? "OK" : "FAIL");
        printf("  keys_fp32   == NULL: %s\n", l0.keys_fp32   == nullptr ? "OK" : "FAIL");
        if (!l0.keys_int8 || !l0.values_int8 || l0.keys_fp32 != nullptr) all_pass = false;

        // Verify dequantized K at position 0 matches original
        {
            float k_proj0[32];
            for (uint32_t i = 0; i < row_elems; ++i)
                k_proj0[i] = static_cast<float>(i); // token 0

            float max_err = 0.0f;
            for (uint32_t i = 0; i < row_elems; ++i) {
                float dq = static_cast<float>(l0.keys_int8[i]) * l0.key_scales[0];
                float err = std::fabs(dq - k_proj0[i]);
                if (err > max_err) max_err = err;
            }
            // Quantization error should be small (< scale)
            float scale0 = l0.key_scales[0];
            bool err_ok = max_err < (scale0 + 1e-4f);
            printf("  token 0 K dequant max_err=%.6f scale=%.6f %s\n",
                   max_err, scale0, err_ok ? "OK" : "FAIL");
            if (!err_ok) all_pass = false;
        }

        // Test fused_dequant_dot_int8 with a query against cached keys
        {
            float query[32];
            for (uint32_t i = 0; i < row_elems; ++i)
                query[i] = 1.0f;

            float scores[5] = {};
            fused_dequant_dot_int8(query, l0.keys_int8, l0.key_scales,
                                   scores, row_elems, num_tokens);

            // Compute expected from dequantized keys
            printf("  fused_dequant_dot_int8 scores:\n");
            for (uint32_t t = 0; t < num_tokens; ++t) {
                float expected = 0.0f;
                for (uint32_t i = 0; i < row_elems; ++i) {
                    float dq = static_cast<float>(l0.keys_int8[t * row_elems + i]) * l0.key_scales[t];
                    expected += dq * query[i];
                }
                float err = std::fabs(scores[t] - expected);
                bool ok = err < 1e-3f;
                if (!ok) all_pass = false;
                printf("    token %u: fused=%.6f expected=%.6f err=%.2e %s\n",
                       t, scores[t], expected, err, ok ? "OK" : "FAIL");
            }
        }

        // Verify reset
        cache.reset();
        bool reset_ok = cache.seq_len() == 0;
        printf("  after reset seq_len = %u %s\n",
               cache.seq_len(), reset_ok ? "OK" : "FAIL");
        if (!reset_ok) all_pass = false;

        // Append after reset
        {
            float k[32], v[32];
            for (uint32_t i = 0; i < row_elems; ++i) { k[i] = 1.0f; v[i] = 2.0f; }
            cache.append(1, k, v);
        }
        bool layer1_ok = cache.layer(1).seq_len == 1;
        printf("  layer 1 after reset+append seq_len = %u %s\n",
               cache.layer(1).seq_len, layer1_ok ? "OK" : "FAIL");
        if (!layer1_ok) all_pass = false;
    }

    // ----------------------------------------------------------------
    // Test 2: FP32 mode
    // ----------------------------------------------------------------
    printf("\n=== Test 2: KVCache FP32 mode ===\n");
    {
        ModelConfig cfg = make_config(false);
        KVCache cache(cfg);

        // Append same 5 tokens
        for (uint32_t t = 0; t < num_tokens; ++t) {
            float k_proj[32], v_proj[32];
            for (uint32_t i = 0; i < row_elems; ++i) {
                k_proj[i] = static_cast<float>(t * 10 + i);
                v_proj[i] = static_cast<float>(t * 100 + i);
            }
            cache.append(0, k_proj, v_proj);
        }

        bool seq_ok = cache.seq_len() == num_tokens;
        printf("  seq_len = %u (expected %u) %s\n",
               cache.seq_len(), num_tokens, seq_ok ? "OK" : "FAIL");
        if (!seq_ok) all_pass = false;

        const KVCacheLayer& l0 = cache.layer(0);
        printf("  keys_fp32   != NULL: %s\n", l0.keys_fp32   ? "OK" : "FAIL");
        printf("  keys_int8   == NULL: %s\n", l0.keys_int8   == nullptr ? "OK" : "FAIL");
        if (!l0.keys_fp32 || l0.keys_int8 != nullptr) all_pass = false;

        // Verify exact values at position 2
        {
            bool exact_ok = true;
            for (uint32_t i = 0; i < row_elems; ++i) {
                float expected_k = static_cast<float>(2 * 10 + i);
                float expected_v = static_cast<float>(2 * 100 + i);
                if (l0.keys_fp32[2 * row_elems + i] != expected_k) exact_ok = false;
                if (l0.values_fp32[2 * row_elems + i] != expected_v) exact_ok = false;
            }
            printf("  token 2 exact values: %s\n", exact_ok ? "OK" : "FAIL");
            if (!exact_ok) all_pass = false;
        }

        // Verify reset
        cache.reset();
        bool reset_ok = cache.seq_len() == 0;
        printf("  after reset seq_len = %u %s\n",
               cache.seq_len(), reset_ok ? "OK" : "FAIL");
        if (!reset_ok) all_pass = false;
    }

    printf("\n%s\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
#endif
