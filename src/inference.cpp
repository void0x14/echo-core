#include "inference.h"

#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cmath>
#include <algorithm>

#include "kernels/quant.h"

static float* alloc_floats(size_t count) {
    void* ptr = nullptr;
    int rc = posix_memalign(&ptr, CACHE_LINE_SIZE, count * sizeof(float));
    if (rc != 0) throw std::bad_alloc();
    std::memset(ptr, 0, count * sizeof(float));
    return static_cast<float*>(ptr);
}

InferenceEngine::InferenceEngine(const ModelConfig& config)
    : config_(config)
    , layout_(WeightLayout::compute(config))
    , weight_pool_(layout_.total_size)
    , hidden_state_(nullptr)
    , residual_(nullptr)
    , attn_out_(nullptr)
    , ffn_out_(nullptr)
    , q_proj_(nullptr)
    , k_proj_(nullptr)
    , v_proj_(nullptr)
    , scores_(nullptr)
    , head_q_(nullptr)
    , head_out_(nullptr)
    , ffn_scratch_(nullptr)
    , ffn_gate_buf_(nullptr)
    , ffn_up_buf_(nullptr)
    , current_layer_base_(0)
{
    kv_cache_ = std::make_unique<KVCache>(config);

    const uint32_t hidden  = config_.hidden_dim;
    const uint32_t kv_dim  = config_.num_kv_heads * config_.head_dim;
    const uint32_t max_seq = config_.max_seq_len;

    hidden_state_ = alloc_floats(hidden);
    residual_     = alloc_floats(hidden);
    attn_out_     = alloc_floats(hidden);
    ffn_out_      = alloc_floats(hidden);
    q_proj_       = alloc_floats(hidden);
    k_proj_       = alloc_floats(kv_dim);
    v_proj_       = alloc_floats(kv_dim);
    scores_       = alloc_floats(max_seq > 0 ? max_seq : 1);
    head_q_       = alloc_floats(config_.head_dim);
    head_out_     = alloc_floats(config_.head_dim);
    ffn_scratch_  = alloc_floats(config_.ffn_hidden_dim);
    ffn_gate_buf_ = alloc_floats(config_.ffn_hidden_dim);
    ffn_up_buf_   = alloc_floats(config_.ffn_hidden_dim);
}

InferenceEngine::~InferenceEngine() {
    std::free(hidden_state_);
    std::free(residual_);
    std::free(attn_out_);
    std::free(ffn_out_);
    std::free(q_proj_);
    std::free(k_proj_);
    std::free(v_proj_);
    std::free(scores_);
    std::free(head_q_);
    std::free(head_out_);
    std::free(ffn_scratch_);
    std::free(ffn_gate_buf_);
    std::free(ffn_up_buf_);
}

void InferenceEngine::load_weights(const fp16_t* weights) {
    std::memcpy(weight_pool_.at<fp16_t>(0), weights, layout_.total_size);
}

void InferenceEngine::forward(const uint32_t* tokens, uint32_t seq_len, float* logits) {
    const uint32_t hidden = config_.hidden_dim;
    const uint32_t vocab  = config_.vocab_size;

    // Token embedding lookup: convert FP16 row of last token to FP32
    const fp16_t* emb_row = weight_pool_.at<fp16_t>(
        layout_.token_embedding_offset) + static_cast<size_t>(tokens[seq_len - 1]) * hidden;
    fp16_to_fp32_row(emb_row, hidden_state_, hidden);

    // Run through each transformer layer
    for (uint32_t l = 0; l < config_.num_layers; ++l) {
        layer_forward(l, hidden_state_, hidden_state_);
    }

    // Final normalization
    const fp16_t* final_norm = weight_pool_.at<fp16_t>(layout_.final_norm_offset);
    norm(hidden_state_, hidden_state_, final_norm);

    // Output projection: hidden -> vocab
    const fp16_t* out_w = weight_pool_.at<fp16_t>(layout_.output_proj_offset);
    std::memset(logits, 0, vocab * sizeof(float));
    matvec(hidden_state_, logits, out_w, vocab, hidden);
}

void InferenceEngine::layer_forward(uint32_t layer_idx, float* input, float* output) {
    const uint32_t hidden = config_.hidden_dim;

    // Absolute byte offsets for this layer's weights
    size_t layer_base = layout_.token_embedding_size
                      + static_cast<size_t>(layer_idx) * layout_.per_layer_size;
    current_layer_base_ = layer_base;

    // Get norm weight for this layer
    const fp16_t* layer_norm = weight_pool_.at<fp16_t>(
        layer_base + layout_.norm_weight_offset);

    // 1. residual = input (skip connection)
    std::memcpy(residual_, input, hidden * sizeof(float));

    // 2. norm(input, normed, layer_norm_weight)
    norm(input, output, layer_norm);

    // 3. attention(layer_idx, normed, attn_out)
    attention(layer_idx, output, attn_out_);

    // 4. hidden = residual + attn_out
    for (uint32_t i = 0; i < hidden; ++i) {
        output[i] = residual_[i] + attn_out_[i];
    }

    // 5. residual = hidden
    std::memcpy(residual_, output, hidden * sizeof(float));

    // 6. norm(hidden, normed, ffn_norm_weight)
    //    For simplicity: use the SAME norm weight for both pre-attn and pre-ffn norm
    norm(output, output, layer_norm);

    // 7. ffn(normed, ffn_out)
    ffn(output, ffn_out_);

    // 8. output = residual + ffn_out
    for (uint32_t i = 0; i < hidden; ++i) {
        output[i] = residual_[i] + ffn_out_[i];
    }
}

void InferenceEngine::attention(uint32_t layer_idx, const float* input, float* output) {
    const uint32_t hidden  = config_.hidden_dim;
    const uint32_t num_heads    = config_.num_heads;
    const uint32_t num_kv_heads = config_.num_kv_heads;
    const uint32_t head_dim     = config_.head_dim;
    const uint32_t kv_dim  = num_kv_heads * head_dim;
    const uint32_t seq_len = kv_cache_->seq_len();

    size_t layer_base = layout_.token_embedding_size
                      + static_cast<size_t>(layer_idx) * layout_.per_layer_size;

    // Q, K, V projections
    const fp16_t* W_q = weight_pool_.at<fp16_t>(layer_base + layout_.q_proj_offset);
    const fp16_t* W_k = weight_pool_.at<fp16_t>(layer_base + layout_.k_proj_offset);
    const fp16_t* W_v = weight_pool_.at<fp16_t>(layer_base + layout_.v_proj_offset);

    matvec(input, q_proj_, W_q, hidden, hidden);
    matvec(input, k_proj_, W_k, kv_dim,  hidden);
    matvec(input, v_proj_, W_v, kv_dim,  hidden);

    // Scale Q by 1/sqrt(head_dim) for numerical stability
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (uint32_t i = 0; i < hidden; ++i) {
        q_proj_[i] *= scale;
    }

    // Clear attention output
    std::memset(attn_out_, 0, hidden * sizeof(float));

    if (seq_len > 0) {
        KVCacheLayer& cache_layer = kv_cache_->layer(layer_idx);

        // Per-head attention
        for (uint32_t h = 0; h < num_heads; ++h) {
            uint32_t kv_head = h % num_kv_heads;

            // Extract head h's query
            std::memcpy(head_q_, q_proj_ + h * head_dim, head_dim * sizeof(float));

            // Compute attention scores for this head
            if (config_.use_kv_quantization) {
                // INT8 mode: dequantize key head slice per position and dot with head_q_
                for (uint32_t pos = 0; pos < seq_len; ++pos) {
                    float score = 0.0f;
                    const int8_t* key_head = cache_layer.keys_int8
                        + pos * kv_dim + kv_head * head_dim;
                    float k_scale = cache_layer.key_scales[pos * num_kv_heads + kv_head];
                    for (uint32_t d = 0; d < head_dim; ++d) {
                        score += head_q_[d] * static_cast<float>(key_head[d]) * k_scale;
                    }
                    scores_[pos] = score;
                }
            } else {
                // FP32 mode: dot product per position
                for (uint32_t pos = 0; pos < seq_len; ++pos) {
                    float score = 0.0f;
                    const float* key_head = cache_layer.keys_fp32
                        + pos * kv_dim + kv_head * head_dim;
                    for (uint32_t d = 0; d < head_dim; ++d) {
                        score += head_q_[d] * key_head[d];
                    }
                    scores_[pos] = score;
                }
            }

            // Softmax over seq_len
            float max_score = scores_[0];
            for (uint32_t i = 1; i < seq_len; ++i) {
                if (scores_[i] > max_score) max_score = scores_[i];
            }
            float sum = 0.0f;
            for (uint32_t i = 0; i < seq_len; ++i) {
                scores_[i] = std::exp(scores_[i] - max_score);
                sum += scores_[i];
            }
            float inv_sum = 1.0f / sum;
            for (uint32_t i = 0; i < seq_len; ++i) {
                scores_[i] *= inv_sum;
            }

            // Weighted sum of values -> head_out_
            std::memset(head_out_, 0, head_dim * sizeof(float));
            if (config_.use_kv_quantization) {
                for (uint32_t pos = 0; pos < seq_len; ++pos) {
                    const int8_t* val_head = cache_layer.values_int8
                        + pos * kv_dim + kv_head * head_dim;
                    float v_scale = cache_layer.val_scales[pos * num_kv_heads + kv_head];
                    for (uint32_t d = 0; d < head_dim; ++d) {
                        head_out_[d] += scores_[pos]
                            * (static_cast<float>(val_head[d]) * v_scale);
                    }
                }
            } else {
                for (uint32_t pos = 0; pos < seq_len; ++pos) {
                    const float* val_head = cache_layer.values_fp32
                        + pos * kv_dim + kv_head * head_dim;
                    for (uint32_t d = 0; d < head_dim; ++d) {
                        head_out_[d] += scores_[pos] * val_head[d];
                    }
                }
            }

            // Copy head output to attn_out_
            std::memcpy(attn_out_ + h * head_dim, head_out_, head_dim * sizeof(float));
        }
    }

    // O projection: attn_out_ -> output [hidden]
    const fp16_t* W_o = weight_pool_.at<fp16_t>(layer_base + layout_.o_proj_offset);
    std::memset(output, 0, hidden * sizeof(float));
    matvec(attn_out_, output, W_o, hidden, hidden);

    // KV cache append
    kv_cache_->append(layer_idx, k_proj_, v_proj_);
}

void InferenceEngine::ffn(const float* input, float* output) {
    const uint32_t hidden = config_.hidden_dim;
    const uint32_t ffn_h  = config_.ffn_hidden_dim;

    const fp16_t* W1 = weight_pool_.at<fp16_t>(current_layer_base_ + layout_.ffn_weight1_offset);
    const fp16_t* W2 = weight_pool_.at<fp16_t>(current_layer_base_ + layout_.ffn_weight2_offset);

    switch (config_.ffn_type) {
        case ModelConfig::FFNType::Dense:
            // y = W2 * relu(W1 * x)
            std::memset(ffn_scratch_, 0, ffn_h * sizeof(float));
            matvec(input, ffn_scratch_, W1, ffn_h, hidden);
            for (uint32_t i = 0; i < ffn_h; ++i) {
                ffn_scratch_[i] = (ffn_scratch_[i] > 0.0f) ? ffn_scratch_[i] : 0.0f; // ReLU
            }
            std::memset(output, 0, hidden * sizeof(float));
            matvec(ffn_scratch_, output, W2, hidden, ffn_h);
            break;

        case ModelConfig::FFNType::GatedSwiGLU: {
            // y = W3 * (swish(W1_gate * x) * W2_up * x)
            const fp16_t* W_gate = W1;
            const fp16_t* W_up   = W2;
            const fp16_t* W_down = weight_pool_.at<fp16_t>(
                current_layer_base_ + layout_.ffn_weight3_offset);

            std::memset(ffn_gate_buf_, 0, ffn_h * sizeof(float));
            std::memset(ffn_up_buf_, 0, ffn_h * sizeof(float));
            matvec(input, ffn_gate_buf_, W_gate, ffn_h, hidden);
            matvec(input, ffn_up_buf_, W_up, ffn_h, hidden);

            // swish(gate) * up
            for (uint32_t i = 0; i < ffn_h; ++i) {
                float sigmoid = 1.0f / (1.0f + std::exp(-ffn_gate_buf_[i]));
                float swish = ffn_gate_buf_[i] * sigmoid;
                ffn_gate_buf_[i] = swish * ffn_up_buf_[i];
            }

            std::memset(output, 0, hidden * sizeof(float));
            matvec(ffn_gate_buf_, output, W_down, hidden, ffn_h);
            break;
        }

        case ModelConfig::FFNType::GatedGeLU: {
            // y = W3 * (gelu(W1_gate * x) * W2_up * x)
            const fp16_t* W_gate = W1;
            const fp16_t* W_up   = W2;
            const fp16_t* W_down = weight_pool_.at<fp16_t>(
                current_layer_base_ + layout_.ffn_weight3_offset);

            std::memset(ffn_gate_buf_, 0, ffn_h * sizeof(float));
            std::memset(ffn_up_buf_, 0, ffn_h * sizeof(float));
            matvec(input, ffn_gate_buf_, W_gate, ffn_h, hidden);
            matvec(input, ffn_up_buf_, W_up, ffn_h, hidden);

            // gelu(gate) * up
            const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
            for (uint32_t i = 0; i < ffn_h; ++i) {
                float x = ffn_gate_buf_[i];
                float inner = sqrt_2_over_pi * (x + 0.044715f * x * x * x);
                float gelu = 0.5f * x * (1.0f + std::tanh(inner));
                ffn_gate_buf_[i] = gelu * ffn_up_buf_[i];
            }

            std::memset(output, 0, hidden * sizeof(float));
            matvec(ffn_gate_buf_, output, W_down, hidden, ffn_h);
            break;
        }
    }
}

void InferenceEngine::norm(const float* input, float* output, const fp16_t* norm_weight) {
    const uint32_t hidden = config_.hidden_dim;

    // Convert norm weight from FP16 to FP32
    float scale_buf[hidden];
    fp16_to_fp32_row(norm_weight, scale_buf, hidden);

    if (config_.norm_type == ModelConfig::NormType::RMSNorm) {
        // RMSNorm: output = (input / rms(input)) * scale
        float rms = 0.0f;
        for (uint32_t i = 0; i < hidden; ++i) {
            rms += input[i] * input[i];
        }
        rms = std::sqrt(rms / static_cast<float>(hidden) + 1e-6f);
        float inv_rms = 1.0f / rms;
        for (uint32_t i = 0; i < hidden; ++i) {
            output[i] = input[i] * inv_rms * scale_buf[i];
        }
    } else {
        // LayerNorm (simplified — no bias, scale only):
        // output = ((input - mean) / sqrt(var + eps)) * scale
        float mean = 0.0f;
        for (uint32_t i = 0; i < hidden; ++i) {
            mean += input[i];
        }
        mean /= static_cast<float>(hidden);

        float var = 0.0f;
        for (uint32_t i = 0; i < hidden; ++i) {
            float diff = input[i] - mean;
            var += diff * diff;
        }
        var /= static_cast<float>(hidden);

        float inv_std = 1.0f / std::sqrt(var + 1e-6f);
        for (uint32_t i = 0; i < hidden; ++i) {
            output[i] = (input[i] - mean) * inv_std * scale_buf[i];
        }
    }
}

void InferenceEngine::matvec(const float* input, float* output,
                              const fp16_t* weight, uint32_t rows, uint32_t cols) {
    // matvec_fp16_fp32 accumulates, so caller must zero output first
    matvec_dispatch(weight, input, output, rows, cols, config_);
}

#ifdef INFERENCE_TEST_MAIN
#include <cstdio>

int main() {
    printf("=== InferenceEngine Smoke Test ===\n\n");

    // Tiny model config
    ModelConfig cfg{};
    cfg.vocab_size        = 64;
    cfg.hidden_dim        = 32;
    cfg.num_heads         = 2;
    cfg.num_kv_heads      = 2;
    cfg.head_dim          = 16;
    cfg.num_layers        = 2;
    cfg.ffn_hidden_dim    = 64;
    cfg.max_seq_len       = 16;
    cfg.ffn_type          = ModelConfig::FFNType::Dense;
    cfg.norm_type         = ModelConfig::NormType::RMSNorm;
    cfg.pos_encoding      = ModelConfig::PosEncoding::None;
    cfg.use_kv_quantization = false;

    WeightLayout layout = WeightLayout::compute(cfg);
    printf("Config: hidden=%u heads=%u layers=%u ffn=%u vocab=%u\n",
           cfg.hidden_dim, cfg.num_heads, cfg.num_layers,
           cfg.ffn_hidden_dim, cfg.vocab_size);
    printf("Weight layout total_size: %zu bytes\n\n", layout.total_size);

    // Allocate synthetic weights (all 0.1)
    fp16_t* weights = new fp16_t[layout.total_size / sizeof(fp16_t)];
    fp16_t val01 = fp32_to_fp16(0.1f);
    size_t total_elements = layout.total_size / sizeof(fp16_t);
    for (size_t i = 0; i < total_elements; ++i) {
        weights[i] = val01;
    }

    InferenceEngine engine(cfg);
    engine.load_weights(weights);

    float logits[64];

    // Test 1: single token forward pass
    printf("--- Test 1: forward([0]) ---\n");
    {
        uint32_t token = 0;
        engine.forward(&token, 1, logits);
        printf("First 8 logits: ");
        for (int i = 0; i < 8; ++i) {
            printf("%.4f ", logits[i]);
        }
        printf("\n");
    }

    // Test 2: second token (KV cache usage)
    printf("\n--- Test 2: forward([0, 1]) ---\n");
    {
        uint32_t tokens[2] = {0, 1};
        engine.forward(tokens, 2, logits);
        printf("First 8 logits: ");
        for (int i = 0; i < 8; ++i) {
            printf("%.4f ", logits[i]);
        }
        printf("\n");
    }

    delete[] weights;

    printf("\nPASS\n");
    return 0;
}
#endif
