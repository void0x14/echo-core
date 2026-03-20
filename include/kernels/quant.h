#pragma once
#include <cstdint>

// Per-token symmetric quantization: FP32 → INT8
// For each row, finds max(abs(row)), computes scale = max_abs / 127.0,
// and quantizes each element to int8 via round(x / scale) clamped to [-127, 127].
void quantize_per_token_symmetric(
    const float* input,     // [num_tokens * num_elements]
    int8_t* output,         // [num_tokens * num_elements]
    float* scales,          // [num_tokens]
    uint32_t num_tokens,
    uint32_t num_elements
);

// Fused dequantization + dot product for attention scoring.
// Computes scores[pos] = dot(query, dequant(key_cache[pos])) without
// materializing dequantized FP32 values in memory.
void fused_dequant_dot_int8(
    const float* query,         // [dim]
    const int8_t* key_cache,    // [seq_len * dim]
    const float* scales,        // [seq_len]
    float* scores,              // [seq_len]
    uint32_t dim,
    uint32_t seq_len
);
