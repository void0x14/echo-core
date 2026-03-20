#include "kernels/quant.h"
#include <immintrin.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <algorithm>

// Horizontal sum of __m256 to float
static inline float hsum256_ps(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum = _mm_add_ps(hi, lo);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

void quantize_per_token_symmetric(
    const float* input,
    int8_t* output,
    float* scales,
    uint32_t num_tokens,
    uint32_t num_elements
) {
    for (uint32_t t = 0; t < num_tokens; ++t) {
        const float* row = input + static_cast<size_t>(t) * num_elements;
        int8_t* out_row  = output + static_cast<size_t>(t) * num_elements;

        // Pass 1: find max absolute value across the row using AVX2
        __m256 abs_max = _mm256_setzero_ps();
        uint32_t i = 0;

        // Standard AVX2 trick: mask off sign bit for abs
        static const uint32_t sign_mask_arr[8] = {
            0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF,
            0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF
        };
        __m256 sign_mask = _mm256_loadu_ps(
            reinterpret_cast<const float*>(sign_mask_arr));

        for (; i + 8 <= num_elements; i += 8) {
            __m256 v = _mm256_loadu_ps(row + i);
            __m256 abs_v = _mm256_and_ps(sign_mask, v);
            abs_max = _mm256_max_ps(abs_max, abs_v);
        }

        // Horizontal max across the 8-wide AVX accumulator
        // (reduce: max(abs_max[0..3], abs_max[4..7]) → scalar)
        __m128 hi128 = _mm256_extractf128_ps(abs_max, 1);
        __m128 lo128 = _mm256_castps256_ps128(abs_max);
        __m128 max128 = _mm_max_ps(lo128, hi128);
        max128 = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
        max128 = _mm_max_ss(max128, _mm_shuffle_ps(max128, max128, 1));
        float max_abs = _mm_cvtss_f32(max128);

        // Scalar tail
        for (; i < num_elements; ++i) {
            float a = std::fabs(row[i]);
            if (a > max_abs) max_abs = a;
        }

        // Compute scale (protect against all-zero rows)
        float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
        scales[t] = scale;

        // Pass 2: quantize row — load 8 FP32, divide, round, pack to INT8
        // packssdw with (x, x) duplicates each half in its lane:
        //   lane0: [x[0..3], x[0..3]]  lane1: [x[4..7], x[4..7]]
        // We need [x[0..3], x[4..7]], so extract both lanes and concat.
        __m256 scale_vec = _mm256_set1_ps(1.0f / scale);
        i = 0;

        for (; i + 8 <= num_elements; i += 8) {
            __m256 vf = _mm256_loadu_ps(row + i);
            vf = _mm256_round_ps(
                _mm256_mul_ps(vf, scale_vec),
                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256i vi32 = _mm256_cvtps_epi32(vf);
            // pack int32 → int16: lane0=[vi32[0..3],vi32[0..3]], lane1=[vi32[4..7],vi32[4..7]]
            __m256i vi16 = _mm256_packs_epi32(vi32, vi32);
            // concat low half of lane0 + low half of lane1 → [vi32[0..3], vi32[4..7]]
            __m128i lo16 = _mm256_castsi256_si128(vi16);
            __m128i hi16 = _mm256_extractf128_si256(vi16, 1);
            __m128i packed16 = _mm_unpacklo_epi64(lo16, hi16);
            // pack int16 → int8 (8 values), store low 64 bits
            __m128i vi8 = _mm_packs_epi16(packed16, packed16);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(out_row + i), vi8);
        }

        // Scalar tail
        for (; i < num_elements; ++i) {
            int v = static_cast<int>(std::round(row[i] / scale));
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            out_row[i] = static_cast<int8_t>(v);
        }
    }
}

void fused_dequant_dot_int8(
    const float* query,
    const int8_t* key_cache,
    const float* scales,
    float* scores,
    uint32_t dim,
    uint32_t seq_len
) {
    for (uint32_t pos = 0; pos < seq_len; ++pos) {
        const int8_t* key = key_cache + static_cast<size_t>(pos) * dim;
        __m256 scale_vec = _mm256_set1_ps(scales[pos]);
        __m256 score_acc = _mm256_setzero_ps();

        uint32_t k = 0;
        for (; k + 16 <= dim; k += 16) {
            // Load 16 INT8 values
            __m128i k8 = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(key + k));

            // First 8: widen int8 → int32 → float, dequant, FMADD
            __m256i k32_lo = _mm256_cvtepi8_epi32(k8);
            __m256 kf_lo   = _mm256_cvtepi32_ps(k32_lo);
            kf_lo = _mm256_mul_ps(kf_lo, scale_vec);
            __m256 q_lo = _mm256_loadu_ps(query + k);
            score_acc = _mm256_fmadd_ps(kf_lo, q_lo, score_acc);

            // Next 8: shift right by 8 bytes, then same pattern
            __m128i k8_hi = _mm_srli_si128(k8, 8);
            __m256i k32_hi = _mm256_cvtepi8_epi32(k8_hi);
            __m256 kf_hi   = _mm256_cvtepi32_ps(k32_hi);
            kf_hi = _mm256_mul_ps(kf_hi, scale_vec);
            __m256 q_hi = _mm256_loadu_ps(query + k + 8);
            score_acc = _mm256_fmadd_ps(kf_hi, q_hi, score_acc);
        }

        float score = hsum256_ps(score_acc);

        // Scalar tail for non-aligned dim
        for (; k < dim; ++k) {
            score += static_cast<float>(key[k]) * scales[pos] * query[k];
        }

        scores[pos] = score;
    }
}

#ifdef QUANT_TEST_MAIN
int main() {
    bool all_pass = true;

    // ---- Test 1: quantize_per_token_symmetric ----
    printf("=== Test 1: quantize_per_token_symmetric ===\n");
    {
        float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        int8_t output[4] = {};
        float scale = 0.0f;

        quantize_per_token_symmetric(input, output, &scale, 1, 4);

        float expected_scale = 4.0f / 127.0f;
        printf("  scale = %.9f (expected %.9f) %s\n",
               scale, expected_scale,
               std::fabs(scale - expected_scale) < 1e-6f ? "OK" : "FAIL");
        if (std::fabs(scale - expected_scale) >= 1e-6f) all_pass = false;

        // Dequantize and check error
        float max_err = 0.0f;
        for (int i = 0; i < 4; ++i) {
            float dq = static_cast<float>(output[i]) * scale;
            float err = std::fabs(dq - input[i]);
            if (err > max_err) max_err = err;
            printf("  input[%.1f] -> int8[%d] -> dq[%.6f] err=%.6f\n",
                   input[i], (int)output[i], dq, err);
        }
        // Per-token symmetric INT8: max quantization error = scale/2 ≈ 4/127/2 ≈ 0.0157
        bool q_ok = max_err < (scale * 0.5f + 1e-6f);
        printf("  max_err=%.6f threshold=%.6f %s\n",
               max_err, scale * 0.5f, q_ok ? "OK" : "FAIL");
        if (!q_ok) all_pass = false;
    }

    // ---- Test 2: fused_dequant_dot_int8 ----
    printf("\n=== Test 2: fused_dequant_dot_int8 ===\n");
    {
        const uint32_t dim = 4;
        const uint32_t seq_len = 3;

        // Query vector
        float query[4] = {1.0f, 2.0f, 3.0f, 4.0f};

        // Quantize 3 key rows
        float keys_fp32[3 * 4] = {
            1.0f, 0.0f, -1.0f, 0.5f,   // row 0
            0.0f, 1.0f,  0.0f, 0.0f,   // row 1
            2.0f, 2.0f,  2.0f, 2.0f,   // row 2
        };
        int8_t keys_i8[3 * 4] = {};
        float k_scales[3] = {};

        quantize_per_token_symmetric(keys_fp32, keys_i8, k_scales, seq_len, dim);

        // Compute scores with fused kernel
        float scores[3] = {};
        fused_dequant_dot_int8(query, keys_i8, k_scales, scores, dim, seq_len);

        // Compute expected scores from dequantized values
        bool fused_ok = true;
        for (uint32_t pos = 0; pos < seq_len; ++pos) {
            float expected = 0.0f;
            for (uint32_t k = 0; k < dim; ++k) {
                float dq = static_cast<float>(keys_i8[pos * dim + k]) * k_scales[pos];
                expected += dq * query[k];
            }
            float err = std::fabs(scores[pos] - expected);
            // Small rounding differences are expected from different summation order
            bool ok = err < 1e-4f;
            if (!ok) fused_ok = false;
            printf("  pos[%u]: fused=%.6f expected=%.6f err=%.2e %s\n",
                   pos, scores[pos], expected, err, ok ? "OK" : "FAIL");
        }
        if (!fused_ok) all_pass = false;

        // Cross-check: fused score vs. exact FP32 dot product (with quantization tolerance)
        printf("\n  Cross-check vs exact FP32 dot:\n");
        for (uint32_t pos = 0; pos < seq_len; ++pos) {
            float exact = 0.0f;
            for (uint32_t k = 0; k < dim; ++k) {
                exact += keys_fp32[pos * dim + k] * query[k];
            }
            float err = std::fabs(scores[pos] - exact);
            printf("  pos[%u]: fused=%.6f exact=%.6f err=%.2e\n",
                   pos, scores[pos], exact, err);
        }
    }

    // ---- Test 3: larger dim (dim=16 to exercise AVX2 16-wide loop) ----
    printf("\n=== Test 3: fused_dequant_dot_int8 dim=16 ===\n");
    {
        const uint32_t dim = 16;
        const uint32_t seq_len = 2;

        float query[16];
        float keys[2 * 16];
        for (uint32_t i = 0; i < 16; ++i) {
            query[i] = static_cast<float>(i + 1);
            keys[i] = static_cast<float>(i + 1);           // row 0 = 1..16
            keys[16 + i] = static_cast<float>(16 - i);     // row 1 = 16..1
        }

        int8_t keys_i8[2 * 16] = {};
        float k_scales[2] = {};
        quantize_per_token_symmetric(keys, keys_i8, k_scales, seq_len, dim);

        float scores[2] = {};
        fused_dequant_dot_int8(query, keys_i8, k_scales, scores, dim, seq_len);

        for (uint32_t pos = 0; pos < seq_len; ++pos) {
            float exact = 0.0f;
            for (uint32_t k = 0; k < dim; ++k) {
                exact += keys[pos * dim + k] * query[k];
            }
            float err = std::fabs(scores[pos] - exact);
            // Allow larger tolerance for quantization error
            bool ok = err < (k_scales[pos] * 8.0f + 1.0f); // loose bound
            printf("  pos[%u]: fused=%.2f exact=%.2f err=%.4f %s\n",
                   pos, scores[pos], exact, err, ok ? "OK" : "FAIL");
            if (!ok) all_pass = false;
        }
    }

    printf("\n%s\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
#endif
