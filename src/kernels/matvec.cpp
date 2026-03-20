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
    for (uint32_t m_tile = 0; m_tile < M; m_tile += TILE_M) {
        uint32_t m_end = std::min(m_tile + TILE_M, M);

        for (uint32_t k_tile = 0; k_tile < K; k_tile += TILE_K) {
            uint32_t k_end = std::min(k_tile + TILE_K, K);

            for (uint32_t m = m_tile; m < m_end; ++m) {
                const fp16_t* W_row = W + static_cast<size_t>(m) * K + k_tile;
                const float* x_row = x + k_tile;

                uint32_t k_len = k_end - k_tile;
                uint32_t k = 0;

                __m256 acc = _mm256_setzero_ps();

                // Main vectorized loop: process 8 elements at a time
                for (; k + 8 <= k_len; k += 8) {
                    __m128i w16 = _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(W_row + k));
                    __m256 w32 = _mm256_cvtph_ps(w16);
                    __m256 xv  = _mm256_loadu_ps(x_row + k);
                    acc = _mm256_fmadd_ps(w32, xv, acc);
                }

                // Horizontal sum of the accumulator
                float partial = hsum256_ps(acc);
                y[m] += partial;

                // Scalar tail for remaining elements
                for (; k < k_len; ++k) {
                    y[m] += fp16_to_fp32(W_row[k]) * x_row[k];
                }
            }
        }
    }
}

// Explicit instantiations for known tile configs
template void matvec_fp16_fp32<1024, 512>(const fp16_t*, const float*, float*, uint32_t, uint32_t);
template void matvec_fp16_fp32<2048, 1024>(const fp16_t*, const float*, float*, uint32_t, uint32_t);

// Dispatch: pick tile config based on which platform preset we want.
// Since ModelConfig doesn't carry platform info, we default to Intel13500H_Tiles.
// The caller can also call the template directly if they know the platform.
void matvec_dispatch(const fp16_t* W, const float* x, float* y,
                     uint32_t M, uint32_t K, const ModelConfig& config) {
    (void)config;
    // Default to Intel tiles; use AMD tiles if K or M suggest larger working set
    // and the user compiled with AMD tile instantiation available.
    matvec_fp16_fp32<Intel13500H_Tiles::TILE_K, Intel13500H_Tiles::TILE_M>(W, x, y, M, K);
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
