#include "inference.h"

#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <vector>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>

#include "kernels/quant.h"

void InferenceEngine::reset() {
    if (kv_cache_) kv_cache_->reset();
}

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
    std::memcpy(resolve_weight<fp16_t>(0), weights, layout_.total_size);

    // Populate weight_dtype_ and weight_ptr_map_ for matvec_d() dispatch
    auto reg = [&](size_t off) {
        weight_dtype_[off]   = static_cast<uint32_t>(GGMLType::F16);
        weight_ptr_map_[off] = weight_pool_.at<void>(off);
    };

    reg(layout_.token_embedding_offset);
    reg(layout_.final_norm_offset);
    reg(layout_.output_proj_offset);

    for (uint32_t l = 0; l < config_.num_layers; ++l) {
        size_t lb = layout_.token_embedding_size
                   + static_cast<size_t>(l) * layout_.per_layer_size;
        reg(lb + layout_.norm_weight_offset);
        reg(lb + layout_.q_proj_offset);
        reg(lb + layout_.k_proj_offset);
        reg(lb + layout_.v_proj_offset);
        reg(lb + layout_.o_proj_offset);
        reg(lb + layout_.ffn_weight1_offset);
        reg(lb + layout_.ffn_weight2_offset);
        if (layout_.ffn_weight3_offset != 0)
            reg(lb + layout_.ffn_weight3_offset);
    }
}

void InferenceEngine::load_weights_from_gguf(const GGUFReader& reader,
                                              const std::string& model_path) {
    int align = reader.alignment();
    reader.assert_alignment(align);

    int fd = ::open(model_path.c_str(), O_RDONLY);
    if (fd < 0)
        throw std::runtime_error("InferenceEngine: cannot open '" + model_path + "'");

    off_t data_off = reader.data_offset();
    size_t data_size = 0;
    for (const auto& [name, info] : reader.tensors()) {
        size_t end = static_cast<size_t>(info.offset) + static_cast<size_t>(info.size);
        if (end > data_size) data_size = end;
    }

    weight_pool_ = AlignedMemoryPool(fd, data_off, data_size, align);
    ::close(fd);

    auto pool_off = [&](const TensorInfo* ti) -> size_t {
        return static_cast<size_t>(ti->offset);
    };

    // Search for tensor with blk.{idx}.{suffix} or just {suffix}
    auto find_layer_tensor = [&](uint32_t idx, const char* suffix) -> const TensorInfo* {
        std::string prefixed = "blk." + std::to_string(idx) + "." + suffix;
        const TensorInfo* ti = reader.find_tensor_by_suffix(prefixed);
        if (!ti) ti = reader.find_tensor_by_suffix(suffix);
        return ti;
    };

    // Embedding
    {
        const TensorInfo* ti = reader.find_tensor_by_suffix("token_embd.weight");
        if (ti)
            gguf_offset_map_[layout_.token_embedding_offset] = pool_off(ti);
    }

    // Per-layer tensors
    for (uint32_t l = 0; l < config_.num_layers; ++l) {
        size_t layer_base = layout_.token_embedding_size
                          + static_cast<size_t>(l) * layout_.per_layer_size;

        auto try_map = [&](const char* suffix, size_t layout_rel_offset) {
            const TensorInfo* ti = find_layer_tensor(l, suffix);
            if (ti)
                gguf_offset_map_[layer_base + layout_rel_offset] = pool_off(ti);
        };

        // Norm
        try_map("attn_norm.weight", layout_.norm_weight_offset);

        // Attention projections — try separate Q/K/V first, then fused QKV
        const TensorInfo* q = find_layer_tensor(l, "attn_q.weight");
        const TensorInfo* k = find_layer_tensor(l, "attn_k.weight");
        const TensorInfo* v = find_layer_tensor(l, "attn_v.weight");
        const TensorInfo* qkv = find_layer_tensor(l, "attn_qkv.weight");

        if (q && k && v) {
            gguf_offset_map_[layer_base + layout_.q_proj_offset] = pool_off(q);
            gguf_offset_map_[layer_base + layout_.k_proj_offset] = pool_off(k);
            gguf_offset_map_[layer_base + layout_.v_proj_offset] = pool_off(v);
        } else if (qkv) {
            // Fused QKV: point Q/K/V to the same tensor (caller must handle splitting)
            size_t qkv_off = pool_off(qkv);
            gguf_offset_map_[layer_base + layout_.q_proj_offset] = qkv_off;
            gguf_offset_map_[layer_base + layout_.k_proj_offset] = qkv_off;
            gguf_offset_map_[layer_base + layout_.v_proj_offset] = qkv_off;
        }

        // Output projection (attention output or gate)
        try_map("attn_output.weight", layout_.o_proj_offset);
        if (!gguf_offset_map_.count(layer_base + layout_.o_proj_offset))
            try_map("attn_gate.weight", layout_.o_proj_offset);

        // FFN
        switch (config_.ffn_type) {
            case ModelConfig::FFNType::Dense:
                try_map("ffn_up.weight",   layout_.ffn_weight1_offset);
                try_map("ffn_down.weight", layout_.ffn_weight2_offset);
                break;
            case ModelConfig::FFNType::GatedSwiGLU:
            case ModelConfig::FFNType::GatedGeLU:
                try_map("ffn_gate.weight", layout_.ffn_weight1_offset);
                try_map("ffn_up.weight",   layout_.ffn_weight2_offset);
                try_map("ffn_down.weight", layout_.ffn_weight3_offset);
                break;
        }
    }

    // Final norm + output projection
    {
        const TensorInfo* ti = reader.find_tensor_by_suffix("output_norm.weight");
        if (ti)
            gguf_offset_map_[layout_.final_norm_offset] = pool_off(ti);
    }
    {
        const TensorInfo* ti = reader.find_tensor_by_suffix("output.weight");
        if (ti)
            gguf_offset_map_[layout_.output_proj_offset] = pool_off(ti);
    }

    // Populate weight_dtype_ and weight_ptr_map_ (layout_offset keys)
    std::unordered_map<size_t, uint32_t> gguf_off_to_dtype;
    for (const auto& [name, info] : reader.tensors()) {
        gguf_off_to_dtype[static_cast<size_t>(info.offset)] =
            static_cast<uint32_t>(info.dtype);
    }
    for (const auto& [layout_off, gguf_off] : gguf_offset_map_) {
        auto it = gguf_off_to_dtype.find(gguf_off);
        if (it == gguf_off_to_dtype.end())
            throw std::runtime_error("load_weights: dtype not found for gguf_off="
                                     + std::to_string(gguf_off));
        weight_dtype_[layout_off]   = it->second;
        weight_ptr_map_[layout_off] = weight_pool_.at<void>(gguf_off);
    }
}

void InferenceEngine::forward(const uint32_t* tokens, uint32_t seq_len, float* logits) {
    const uint32_t hidden = config_.hidden_dim;
    const uint32_t vocab  = config_.vocab_size;

    // Token embedding lookup
    if (has_weight(layout_.token_embedding_offset)) {
        const fp16_t* emb_row = resolve_weight<fp16_t>(
            layout_.token_embedding_offset) + static_cast<size_t>(tokens[seq_len - 1]) * hidden;
        fp16_to_fp32_row(emb_row, hidden_state_, hidden);
    }

    // Run through each transformer layer
    for (uint32_t l = 0; l < config_.num_layers; ++l) {
        layer_forward(l, hidden_state_, hidden_state_);
    }

    // Final normalization
    if (has_weight(layout_.final_norm_offset)) {
        const fp16_t* final_norm = resolve_weight<fp16_t>(layout_.final_norm_offset);
        norm(hidden_state_, hidden_state_, final_norm);
    }

    // Output projection: hidden -> vocab
    if (has_weight(layout_.output_proj_offset)) {
        std::memset(logits, 0, vocab * sizeof(float));
        matvec_d(hidden_state_, logits, layout_.output_proj_offset, vocab, hidden);
    }
}

void InferenceEngine::layer_forward(uint32_t layer_idx, float* input, float* output) {
    const uint32_t hidden = config_.hidden_dim;

    // Absolute byte offsets for this layer's weights
    size_t layer_base = layout_.token_embedding_size
                      + static_cast<size_t>(layer_idx) * layout_.per_layer_size;
    current_layer_base_ = layer_base;

    // Get norm weight for this layer
    const fp16_t* layer_norm = resolve_weight<fp16_t>(
        layer_base + layout_.norm_weight_offset);

    // 1. residual = input (skip connection)
    std::memcpy(residual_, input, hidden * sizeof(float));

    // 2. norm(input, normed, layer_norm_weight)
    if (layer_norm)
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
    if (layer_norm)
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
    if (has_weight(layer_base + layout_.q_proj_offset) &&
        has_weight(layer_base + layout_.k_proj_offset) &&
        has_weight(layer_base + layout_.v_proj_offset)) {
        matvec_d(input, q_proj_, layer_base + layout_.q_proj_offset, hidden, hidden);
        matvec_d(input, k_proj_, layer_base + layout_.k_proj_offset, kv_dim,  hidden);
        matvec_d(input, v_proj_, layer_base + layout_.v_proj_offset, kv_dim,  hidden);
    }

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
    std::memset(output, 0, hidden * sizeof(float));
    if (has_weight(layer_base + layout_.o_proj_offset))
        matvec_d(attn_out_, output, layer_base + layout_.o_proj_offset, hidden, hidden);

    // KV cache append
    kv_cache_->append(layer_idx, k_proj_, v_proj_);
}

void InferenceEngine::ffn(const float* input, float* output) {
    const uint32_t hidden = config_.hidden_dim;
    const uint32_t ffn_h  = config_.ffn_hidden_dim;

    if (!has_weight(current_layer_base_ + layout_.ffn_weight1_offset) ||
        !has_weight(current_layer_base_ + layout_.ffn_weight2_offset)) return;

    switch (config_.ffn_type) {
        case ModelConfig::FFNType::Dense:
            std::memset(ffn_scratch_, 0, ffn_h * sizeof(float));
            matvec_d(input, ffn_scratch_, current_layer_base_ + layout_.ffn_weight1_offset, ffn_h, hidden);
            for (uint32_t i = 0; i < ffn_h; ++i) {
                ffn_scratch_[i] = (ffn_scratch_[i] > 0.0f) ? ffn_scratch_[i] : 0.0f;
            }
            std::memset(output, 0, hidden * sizeof(float));
            matvec_d(ffn_scratch_, output, current_layer_base_ + layout_.ffn_weight2_offset, hidden, ffn_h);
            break;

        case ModelConfig::FFNType::GatedSwiGLU: {
            if (!has_weight(current_layer_base_ + layout_.ffn_weight3_offset)) break;

            std::memset(ffn_gate_buf_, 0, ffn_h * sizeof(float));
            std::memset(ffn_up_buf_, 0, ffn_h * sizeof(float));
            matvec_d(input, ffn_gate_buf_, current_layer_base_ + layout_.ffn_weight1_offset, ffn_h, hidden);
            matvec_d(input, ffn_up_buf_, current_layer_base_ + layout_.ffn_weight2_offset, ffn_h, hidden);

            // swish(gate) * up
            for (uint32_t i = 0; i < ffn_h; ++i) {
                float sigmoid = 1.0f / (1.0f + std::exp(-ffn_gate_buf_[i]));
                float swish = ffn_gate_buf_[i] * sigmoid;
                ffn_gate_buf_[i] = swish * ffn_up_buf_[i];
            }

            std::memset(output, 0, hidden * sizeof(float));
            matvec_d(ffn_gate_buf_, output, current_layer_base_ + layout_.ffn_weight3_offset, hidden, ffn_h);
            break;
        }

        case ModelConfig::FFNType::GatedGeLU: {
            if (!has_weight(current_layer_base_ + layout_.ffn_weight3_offset)) break;

            std::memset(ffn_gate_buf_, 0, ffn_h * sizeof(float));
            std::memset(ffn_up_buf_, 0, ffn_h * sizeof(float));
            matvec_d(input, ffn_gate_buf_, current_layer_base_ + layout_.ffn_weight1_offset, ffn_h, hidden);
            matvec_d(input, ffn_up_buf_, current_layer_base_ + layout_.ffn_weight2_offset, ffn_h, hidden);

            // gelu(gate) * up
            const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
            for (uint32_t i = 0; i < ffn_h; ++i) {
                float x = ffn_gate_buf_[i];
                float inner = sqrt_2_over_pi * (x + 0.044715f * x * x * x);
                float gelu = 0.5f * x * (1.0f + std::tanh(inner));
                ffn_gate_buf_[i] = gelu * ffn_up_buf_[i];
            }

            std::memset(output, 0, hidden * sizeof(float));
            matvec_d(ffn_gate_buf_, output, current_layer_base_ + layout_.ffn_weight3_offset, hidden, ffn_h);
            break;
        }
    }
}

void InferenceEngine::norm(const float* input, float* output, const fp16_t* norm_weight) {
    const uint32_t hidden = config_.hidden_dim;

    // Convert norm weight from FP16 to FP32
    std::vector<float> scale_buf(hidden);
    fp16_to_fp32_row(norm_weight, scale_buf.data(), hidden);

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

void InferenceEngine::matvec_d(const float* input, float* output,
                                size_t layout_offset, uint32_t rows, uint32_t cols) {
    auto it = weight_dtype_.find(layout_offset);
    if (it == weight_dtype_.end()) [[unlikely]]
        throw std::runtime_error("matvec_d: no dtype for offset " +
                                 std::to_string(layout_offset));
    const void* w = resolve_weight_ptr(layout_offset);
    switch (static_cast<GGMLType>(it->second)) {
        case GGMLType::Q8_0: matvec_q8_0(w, input, output, rows, cols); break;
        case GGMLType::Q4_K: matvec_q4_K(w, input, output, rows, cols); break;
        case GGMLType::Q2_K: matvec_q2_K(w, input, output, rows, cols); break;
        case GGMLType::F16:
            matvec_dispatch(static_cast<const fp16_t*>(w),
                            input, output, rows, cols, config_);
            break;
        default:
            throw std::runtime_error("matvec_d: unsupported dtype " +
                                     std::to_string(it->second));
    }
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
