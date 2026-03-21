#pragma once
#include <cstdint>
#include <cstddef>

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

// --- GGUF Q8_0 block format ---
// 34 bytes per block, 32 weights per block
// Dequant: weight[i] = fp16_to_fp32(d) * qs[i]
struct block_q8_0 {
    uint16_t d;      // FP16 scale (little-endian)
    int8_t   qs[32]; // 32 signed INT8 quantized weights
};

// Fused dequantization + dot product for Q8_0 GGUF blocks.
// Computes dot(query, dequant(blocks)) without materializing FP32 weights.
// n_blocks = total_weights / 32
float fused_dequant_dot_q8_0(const block_q8_0* blocks, uint32_t n_blocks,
                              const float* query_fp32);

// --- GGUF Q4_K block format ---
// 144 bytes per block, 256 weights per block (4 blocks x 64 weights)
// Dequant: weight = real_scale * q - real_min
// where q is unsigned 4-bit (0-15), nibble-packed in qs[128]
struct block_q4_K {
    uint16_t d;           // FP16 super-scale
    uint16_t dmin;        // FP16 super-minimum
    uint8_t  scales[12];  // 6-bit packed scale+min (canonical get_scale_min_k4)
    uint8_t  qs[128];     // 256 unsigned 4-bit weights, nibble-packed
};

// Fused dequantization + dot product for Q4_K GGUF blocks.
// Computes dot(query, dequant(blocks)) without materializing FP32 weights.
// n_blocks = total_weights / 256
float fused_dequant_dot_q4_K(const block_q4_K* blocks, uint32_t n_blocks,
                              const float* query_fp32);

// Dequantize Q8_0 blocks to FP16 array.
// src: pointer to Q8_0 blocks (34 bytes each, 32 weights per block)
// dst: output FP16 array (n_weights elements)
// n_weights: total number of weights (must be multiple of 32)
void dequantize_q8_0_to_fp16(const void* src, uint16_t* dst, size_t n_weights);

// Dequantize Q4_K blocks to FP16 array.
void dequantize_q4_K_to_fp16(const void* src, uint16_t* dst, size_t n_weights);

// Dequantize Q2_K blocks to FP16 array.
void dequantize_q2_K_to_fp16(const void* src, uint16_t* dst, size_t n_weights);
