#include "kernels/matvec.h"
#include <immintrin.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>

// Horizontal sum of __m256 to float
inline float hsum256_ps(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum = _mm_add_ps(hi, lo);        // [a0+a4, a1+a5, a2+a6, a3+a7]
    sum = _mm_hadd_ps(sum, sum);            // [a0+a4+a1+a5, ...]
    sum = _mm_hadd_ps(sum, sum);            // [a0+a1+a2+a3+a4+a5+a6+a7, ...]
    return _mm_cvtss_f32(sum);
}

template <uint32_t TILE_K, uint32_t TILE_M>
void matvec_fp16_fp32(const fp16_t* W, const float* x, float* y, uint32_t M, uint32_t K) {
    // Loop order: for m → for k (NOT for k → for m)
    // Bu sıra input vector x'in L1 cache'te kalmasını sağlar.
    // K=2560 × 4 bytes = 10KB, L1 = 48KB → x tamamen L1'de kalır.
    for (uint32_t m = 0; m < M; ++m) {
        const fp16_t* W_row = W + static_cast<size_t>(m) * K;

        __m256 acc_vec = _mm256_setzero_ps();
        float acc_scalar = 0.0f;

        uint32_t k = 0;
        for (; k + 8 <= K; k += 8) {
            __m128i w16 = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(W_row + k));
            __m256 w32 = _mm256_cvtph_ps(w16);
            __m256 xv  = _mm256_loadu_ps(x + k);
            acc_vec = _mm256_fmadd_ps(w32, xv, acc_vec);
        }

        // Scalar tail for remaining elements
        for (; k < K; ++k) {
            acc_scalar += fp16_to_fp32(W_row[k]) * x[k];
        }

        y[m] += hsum256_ps(acc_vec) + acc_scalar;
    }
}

// Explicit instantiations for known tile configs
template void matvec_fp16_fp32<1024, 512>(const fp16_t*, const float*, float*, uint32_t, uint32_t);
template void matvec_fp16_fp32<2048, 1024>(const fp16_t*, const float*, float*, uint32_t, uint32_t);

// Dispatch: pick tile config based on which platform preset we want.
void matvec_dispatch(const fp16_t* W, const float* x, float* y,
                     uint32_t M, uint32_t K, const ModelConfig& config) {
    (void)config;
    matvec_fp16_fp32<Intel13500H_Tiles::TILE_K, Intel13500H_Tiles::TILE_M>(W, x, y, M, K);
}

// --- Fused dequant + matvec for Q8_0 ---
// block_q8_0: 2 bytes d (FP16 scale) + 32 bytes qs (INT8 weights) = 34 bytes
// Each block covers 32 columns of one row
// Row of K columns uses K/32 blocks = K*34/32 bytes
void matvec_q8_0(const void* blocks, const float* x, float* y,
                 uint32_t M, uint32_t K) {
    const uint8_t* base = static_cast<const uint8_t*>(blocks);
    const uint32_t blocks_per_row = K / 32;
    const size_t block_stride = 34; // sizeof(block_q8_0)

    for (uint32_t m = 0; m < M; ++m) {
        float sum = 0.0f;
        const uint8_t* row_ptr = base + static_cast<size_t>(m) * blocks_per_row * block_stride;

        for (uint32_t b = 0; b < blocks_per_row; ++b) {
            const uint8_t* bp = row_ptr + b * block_stride;
            float d = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(bp));
            const int8_t* qs = reinterpret_cast<const int8_t*>(bp + 2);
            const float* x_blk = x + b * 32;

            // Accumulate dot product for this 32-element block
            float block_sum = 0.0f;
            for (int j = 0; j < 32; ++j) {
                block_sum += static_cast<float>(qs[j]) * x_blk[j];
            }
            sum += d * block_sum;
        }
        y[m] += sum;
    }
}

// --- Fused dequant + matvec for Q4_K ---
// block_q4_K: d(FP16) + dmin(FP16) + scales[12] + qs[128] = 144 bytes
// Each block covers 256 columns of one row
void matvec_q4_K(const void* blocks, const float* x, float* y,
                 uint32_t M, uint32_t K) {
    const uint8_t* base = static_cast<const uint8_t*>(blocks);
    const uint32_t blocks_per_row = K / 256;
    const size_t block_stride = 144;

    for (uint32_t m = 0; m < M; ++m) {
        float sum = 0.0f;
        const uint8_t* row_ptr = base + static_cast<size_t>(m) * blocks_per_row * block_stride;

        for (uint32_t b = 0; b < blocks_per_row; ++b) {
            const uint8_t* bp = row_ptr + b * block_stride;
            float d = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(bp));
            float dmin = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(bp + 2));
            const uint8_t* scales = bp + 4;
            const uint8_t* qs = bp + 16;
            const float* x_blk = x + b * 256;

            for (int blk = 0; blk < 4; ++blk) {
                int js = blk * 2;
                uint8_t sc0, mn0, sc1, mn1;
                if (js < 4) {
                    sc0 = scales[js] & 63;
                    mn0 = scales[js + 4] & 63;
                } else {
                    sc0 = (scales[js + 4] & 0x0F) | ((scales[js - 4] >> 6) << 4);
                    mn0 = (scales[js + 4] >> 4)    | ((scales[js] >> 6) << 4);
                }
                if (js + 1 < 4) {
                    sc1 = scales[js + 1] & 63;
                    mn1 = scales[js + 1 + 4] & 63;
                } else {
                    sc1 = (scales[js + 1 + 4] & 0x0F) | ((scales[js + 1 - 4] >> 6) << 4);
                    mn1 = (scales[js + 1 + 4] >> 4)    | ((scales[js + 1] >> 6) << 4);
                }
                float rs0 = d * sc0, rm0 = dmin * mn0;
                float rs1 = d * sc1, rm1 = dmin * mn1;

                int qoff = blk * 32;
                int woff = blk * 64;
                for (int j = 0; j < 16; ++j) {
                    sum += (rs0 * (qs[qoff+j] & 0x0F) - rm0) * x_blk[woff + j];
                    sum += (rs1 * (qs[qoff+16+j] & 0x0F) - rm1) * x_blk[woff + 16 + j];
                    sum += (rs0 * (qs[qoff+j] >> 4) - rm0) * x_blk[woff + 32 + j];
                    sum += (rs1 * (qs[qoff+16+j] >> 4) - rm1) * x_blk[woff + 48 + j];
                }
            }
        }
        y[m] += sum;
    }
}

// --- Fused dequant + matvec for Q2_K ---
// 86 bytes per block, 256 weights per block
void matvec_q2_K(const void* blocks, const float* x, float* y,
                 uint32_t M, uint32_t K) {
    const uint8_t* base = static_cast<const uint8_t*>(blocks);
    const uint32_t blocks_per_row = K / 256;
    const size_t block_stride = 86;

    for (uint32_t m = 0; m < M; ++m) {
        float sum = 0.0f;
        const uint8_t* row_ptr = base + static_cast<size_t>(m) * blocks_per_row * block_stride;

        for (uint32_t b = 0; b < blocks_per_row; ++b) {
            const uint8_t* bp = row_ptr + b * block_stride;
            float d_all = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(bp));
            float m_all = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(bp + 2));
            const uint8_t* scales = bp + 4;
            const uint8_t* qs = bp + 20;
            const float* x_blk = x + b * 256;

            for (int j = 0; j < 256; ++j) {
                int sb = j / 8;
                int s = sb / 2;
                float scale, min_val;
                if (sb % 2 == 0) {
                    scale = d_all * (scales[s] & 0x0F);
                    min_val = m_all * (scales[s] >> 4);
                } else {
                    scale = d_all * (scales[s + 8] & 0x0F);
                    min_val = m_all * (scales[s + 8] >> 4);
                }
                int byte_idx = j / 4;
                int bit_off = (j % 4) * 2;
                int q = (qs[byte_idx] >> bit_off) & 0x03;
                sum += (scale * q - min_val) * x_blk[j];
            }
        }
        y[m] += sum;
    }
}

#ifdef MATVEC_TEST_MAIN
int main() {
    // 4×8 FP16 weight matrix (row-major)
    fp16_t W[4 * 8];
    // 8-length FP32 input vector
    float x[8];
    // 4-length FP32 output vector
    float y[4] = {0};

    // Fill W with simple values: row m has all values = m + 1.0
    for (uint32_t m = 0; m < 4; ++m) {
        for (uint32_t k = 0; k < 8; ++k) {
            W[m * 8 + k] = fp32_to_fp16(static_cast<float>(m + 1));
        }
    }

    // Fill x with values 1..8
    for (uint32_t k = 0; k < 8; ++k) {
        x[k] = static_cast<float>(k + 1);
    }

    matvec_fp16_fp32<8, 4>(W, x, y, 4, 8);

    // Expected:
    // y[0] = 1*(1+2+3+4+5+6+7+8) = 1*36 = 36
    // y[1] = 2*(1+2+3+4+5+6+7+8) = 2*36 = 72
    // y[2] = 3*(1+2+3+4+5+6+7+8) = 3*36 = 108
    // y[3] = 4*(1+2+3+4+5+6+7+8) = 4*36 = 144
    printf("matvec test results:\n");
    for (uint32_t m = 0; m < 4; ++m) {
        printf("  y[%u] = %.1f (expected %.1f) %s\n",
               m, y[m], static_cast<float>(m + 1) * 36.0f,
               y[m] == static_cast<float>(m + 1) * 36.0f ? "OK" : "FAIL");
    }

    // Test with non-tile-aligned K (scalar tail path)
    fp16_t W2[4 * 5];
    float x2[5];
    float y2[4] = {0};

    for (uint32_t m = 0; m < 4; ++m)
        for (uint32_t k = 0; k < 5; ++k)
            W2[m * 5 + k] = fp32_to_fp16(static_cast<float>(m + 1));
    for (uint32_t k = 0; k < 5; ++k)
        x2[k] = static_cast<float>(k + 1);

    matvec_fp16_fp32<8, 4>(W2, x2, y2, 4, 5);

    // Expected: sum(1..5) = 15
    printf("\nTail test (K=5) results:\n");
    bool all_pass = true;
    for (uint32_t m = 0; m < 4; ++m) {
        float expected = static_cast<float>(m + 1) * 15.0f;
        bool ok = (y2[m] == expected);
        if (!ok) all_pass = false;
        printf("  y[%u] = %.1f (expected %.1f) %s\n",
               m, y2[m], expected, ok ? "OK" : "FAIL");
    }
    printf("\n%s\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");

    return all_pass ? 0 : 1;
}
#endif
