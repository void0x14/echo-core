#pragma once
#include "types.h"
#include "config.h"

// Tiled AVX2 matrix-vector multiply: y = W * x
// W: M×K matrix stored row-major as FP16
// x: K-length vector as FP32
// y: M-length vector as FP32 (accumulated, not overwritten — caller zeroes y)
// TILE_K, TILE_M: compile-time tile sizes from CacheTileConfig
template <uint32_t TILE_K, uint32_t TILE_M>
void matvec_fp16_fp32(const fp16_t* W, const float* x, float* y, uint32_t M, uint32_t K);

// Convenience overload that dispatches to the correct tile config
void matvec_dispatch(const fp16_t* W, const float* x, float* y,
                     uint32_t M, uint32_t K, const ModelConfig& config);

// Fused dequant+matvec for Q8_0 blocks (avoids FP16 dequantization)
void matvec_q8_0(const void* blocks, const float* x, float* y,
                 uint32_t M, uint32_t K);

// Fused dequant+matvec for Q4_K blocks
void matvec_q4_K(const void* blocks, const float* x, float* y,
                 uint32_t M, uint32_t K);

// Fused dequant+matvec for Q2_K blocks
void matvec_q2_K(const void* blocks, const float* x, float* y,
                 uint32_t M, uint32_t K);
