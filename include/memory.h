#pragma once

#include <cstdlib>
#include <cassert>
#include <stdexcept>

#include "types.h"
#include "config.h"

class AlignedMemoryPool {
    uint8_t* base_;
    size_t   total_size_;
    size_t   offset_;

public:
    explicit AlignedMemoryPool(size_t total_bytes)
        : base_(nullptr), total_size_(total_bytes), offset_(0)
    {
        if (total_bytes == 0) return;
        void* ptr = nullptr;
        int rc = posix_memalign(&ptr, CACHE_LINE_SIZE, total_bytes);
        if (rc != 0)
            throw std::bad_alloc();
        base_ = static_cast<uint8_t*>(ptr);
    }

    ~AlignedMemoryPool() {
        free(base_);
    }

    template<typename T>
    T* alloc(size_t count) {
        // Align the current offset to alignof(T) or CACHE_LINE_SIZE, whichever is larger
        constexpr size_t align = (alignof(T) > CACHE_LINE_SIZE) ? alignof(T) : CACHE_LINE_SIZE;
        offset_ = (offset_ + align - 1) & ~(align - 1);
        size_t bytes = count * sizeof(T);
        assert(offset_ + bytes <= total_size_);
        T* p = reinterpret_cast<T*>(base_ + offset_);
        offset_ += bytes;
        return p;
    }

    template<typename T>
    T* at(size_t byte_offset) {
        assert(byte_offset + sizeof(T) <= total_size_);  // note: element count is caller's concern for array access
        return reinterpret_cast<T*>(base_ + byte_offset);
    }

    size_t bytes_used() const { return offset_; }
    size_t bytes_total() const { return total_size_; }

    AlignedMemoryPool(const AlignedMemoryPool&) = delete;
    AlignedMemoryPool& operator=(const AlignedMemoryPool&) = delete;
};

struct WeightLayout {
    size_t token_embedding_offset;
    size_t token_embedding_size;

    // Per-layer weights:
    size_t norm_weight_offset;        // [hidden_dim] fp16
    size_t q_proj_offset;             // [hidden_dim * hidden_dim] fp16
    size_t k_proj_offset;             // [hidden_dim * (num_kv_heads * head_dim)] fp16
    size_t v_proj_offset;             // [hidden_dim * (num_kv_heads * head_dim)] fp16
    size_t o_proj_offset;             // [hidden_dim * hidden_dim] fp16
    size_t ffn_weight1_offset;
    size_t ffn_weight2_offset;
    size_t ffn_weight3_offset;        // gate weight for Gated FFN (0 if Dense)
    size_t per_layer_size;

    size_t final_norm_offset;
    size_t output_proj_offset;

    size_t total_size;

    static WeightLayout compute(const ModelConfig& config) {
        WeightLayout layout{};
        const size_t hidden = config.hidden_dim;
        const size_t vocab  = config.vocab_size;
        const size_t kv_dim = config.num_kv_heads * config.head_dim;
        const size_t ffn_h  = config.ffn_hidden_dim;

        auto bytes = [](size_t elements) -> size_t {
            return elements * sizeof(fp16_t);
        };

        // Token embedding: [vocab, hidden]
        layout.token_embedding_offset = 0;
        layout.token_embedding_size   = bytes(vocab * hidden);

        // Per-layer layout — norm, q, k, v, o, ffn weights
        size_t offset = 0;

        // Layer norm weight: [hidden]
        layout.norm_weight_offset = offset;
        offset += bytes(hidden);

        // Attention projections
        layout.q_proj_offset = offset;
        offset += bytes(hidden * hidden);

        layout.k_proj_offset = offset;
        offset += bytes(hidden * kv_dim);

        layout.v_proj_offset = offset;
        offset += bytes(hidden * kv_dim);

        layout.o_proj_offset = offset;
        offset += bytes(hidden * hidden);

        // FFN weights
        layout.ffn_weight1_offset = offset;
        switch (config.ffn_type) {
            case ModelConfig::FFNType::Dense:
                // Dense: 2 matrices — up [hidden, ffn_h] + down [ffn_h, hidden]
                offset += bytes(hidden * ffn_h);
                layout.ffn_weight2_offset = offset;
                offset += bytes(ffn_h * hidden);
                layout.ffn_weight3_offset = 0;
                break;
            case ModelConfig::FFNType::GatedSwiGLU:
            case ModelConfig::FFNType::GatedGeLU:
                // Gated: 3 matrices — gate [hidden, ffn_h] + up [hidden, ffn_h] + down [ffn_h, hidden]
                offset += bytes(hidden * ffn_h);
                layout.ffn_weight2_offset = offset;
                offset += bytes(hidden * ffn_h);
                layout.ffn_weight3_offset = offset;
                offset += bytes(ffn_h * hidden);
                break;
        }

        layout.per_layer_size = offset;

        // Final norm: [hidden]
        layout.final_norm_offset = layout.token_embedding_size + layout.per_layer_size * config.num_layers;
        layout.output_proj_offset = layout.final_norm_offset + bytes(hidden);
        layout.total_size = layout.output_proj_offset + bytes(hidden * vocab);

        return layout;
    }
};
