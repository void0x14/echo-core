#pragma once
#include <cstdint>
#include <memory>

#include "config.h"
#include "types.h"
#include "memory.h"
#include "kv_cache.h"
#include "kernels/matvec.h"

class InferenceEngine {
    ModelConfig config_;
    WeightLayout layout_;

    AlignedMemoryPool weight_pool_;     // all FP16 weights in one contiguous block
    std::unique_ptr<KVCache> kv_cache_; // per-layer KV cache

    // Scratch buffers (pre-allocated, reused across layers)
    float* hidden_state_;    // [hidden_dim] — current layer's output
    float* residual_;        // [hidden_dim] — skip connection
    float* attn_out_;        // [hidden_dim] — attention output
    float* ffn_out_;         // [hidden_dim] — FFN output
    float* q_proj_;          // [hidden_dim] — query projection
    float* k_proj_;          // [num_kv_heads * head_dim]
    float* v_proj_;          // [num_kv_heads * head_dim]
    float* scores_;          // [max_seq_len] — attention scores per head
    float* head_q_;          // [head_dim] — single head query
    float* head_out_;        // [head_dim] — single head attention output
    float* ffn_scratch_;     // [ffn_hidden_dim] — FFN intermediate buffer (Dense FFN)
    float* ffn_gate_buf_;    // [ffn_hidden_dim] — gated FFN gate projection scratch
    float* ffn_up_buf_;      // [ffn_hidden_dim] — gated FFN up projection scratch

    size_t current_layer_base_; // byte offset of current layer's weights (set by layer_forward)

public:
    InferenceEngine(const ModelConfig& config);
    ~InferenceEngine();

    // Load weights from a contiguous FP16 buffer into the weight pool
    // weights: pointer to [total_elements] FP16 values laid out per WeightLayout
    void load_weights(const fp16_t* weights);

    // Generate logits for a sequence of token IDs
    // tokens: [seq_len] input token IDs
    // seq_len: number of input tokens
    // logits: [vocab_size] output — unnormalized log probabilities
    void forward(const uint32_t* tokens, uint32_t seq_len, float* logits);

    const ModelConfig& config() const { return config_; }

private:
    // Single transformer layer forward pass
    void layer_forward(uint32_t layer_idx, float* input, float* output);

    // Self-attention for one layer
    void attention(uint32_t layer_idx, const float* input, float* output);

    // Feed-forward network for one layer
    void ffn(const float* input, float* output);

    // Normalization: RMSNorm or LayerNorm based on config
    void norm(const float* input, float* output, const fp16_t* norm_weight);

    // Matrix-vector multiply using tiled AVX2 kernel
    void matvec(const float* input, float* output,
                const fp16_t* weight, uint32_t rows, uint32_t cols);
};
