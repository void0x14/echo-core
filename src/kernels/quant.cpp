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

// Fused dequantization + dot product for GGUF Q8_0 blocks.
// Block layout (34 bytes): { uint16_t d (FP16 scale), int8_t qs[32] }
// Dequant formula: weight[i] = fp16_to_fp32(d) * qs[i]
// FP32 values are never written to memory — all accumulation in AVX registers.
float fused_dequant_dot_q8_0(const block_q8_0* blocks, uint32_t n_blocks,
                              const float* query_fp32) {
    __m256 acc = _mm256_setzero_ps();

    for (uint32_t b = 0; b < n_blocks; ++b) {
        float scale = _cvtsh_ss(blocks[b].d);
        __m256 scale_vec = _mm256_set1_ps(scale);

        const int8_t* qs = blocks[b].qs;

        // Chunk 0: qs[0..7]
        __m128i chunk0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qs + 0));
        __m256i i32_0 = _mm256_cvtepi8_epi32(chunk0);
        __m256 f32_0 = _mm256_cvtepi32_ps(i32_0);
        f32_0 = _mm256_mul_ps(f32_0, scale_vec);
        __m256 q0 = _mm256_loadu_ps(query_fp32 + b * 32 + 0);
        acc = _mm256_fmadd_ps(f32_0, q0, acc);

        // Chunk 1: qs[8..15]
        __m128i chunk1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qs + 8));
        __m256i i32_1 = _mm256_cvtepi8_epi32(chunk1);
        __m256 f32_1 = _mm256_cvtepi32_ps(i32_1);
        f32_1 = _mm256_mul_ps(f32_1, scale_vec);
        __m256 q1 = _mm256_loadu_ps(query_fp32 + b * 32 + 8);
        acc = _mm256_fmadd_ps(f32_1, q1, acc);

        // Chunk 2: qs[16..23]
        __m128i chunk2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qs + 16));
        __m256i i32_2 = _mm256_cvtepi8_epi32(chunk2);
        __m256 f32_2 = _mm256_cvtepi32_ps(i32_2);
        f32_2 = _mm256_mul_ps(f32_2, scale_vec);
        __m256 q2 = _mm256_loadu_ps(query_fp32 + b * 32 + 16);
        acc = _mm256_fmadd_ps(f32_2, q2, acc);

        // Chunk 3: qs[24..31]
        __m128i chunk3 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qs + 24));
        __m256i i32_3 = _mm256_cvtepi8_epi32(chunk3);
        __m256 f32_3 = _mm256_cvtepi32_ps(i32_3);
        f32_3 = _mm256_mul_ps(f32_3, scale_vec);
        __m256 q3 = _mm256_loadu_ps(query_fp32 + b * 32 + 24);
        acc = _mm256_fmadd_ps(f32_3, q3, acc);
    }

    return hsum256_ps(acc);
}

// Fused dequantization + dot product for GGUF Q4_K blocks.
// Block layout (144 bytes, 256 weights = 4 blocks x 64):
//   { uint16_t d (FP16 super-scale), uint16_t dmin (FP16 super-minimum),
//     uint8_t scales[12] (bit-packed 6-bit scale+min per canonical get_scale_min_k4),
//     uint8_t qs[128] (nibble-packed per canonical llama.cpp) }
//
// Canonical nibble packing (per 64-weight block, 32 bytes qs):
//   qs[0..15]:   low nibbles  -> weights 0..15   (scale/min pair 0)
//   qs[16..31]:  low nibbles  -> weights 16..31  (scale/min pair 1)
//   qs[0..15]:   high nibbles -> weights 32..47  (scale/min pair 0)
//   qs[16..31]:  high nibbles -> weights 48..63  (scale/min pair 1)
//
// Dequant formula (canonical): weight = real_scale * q - real_min
// FP32 values are never written to memory -- all accumulation in AVX registers.
float fused_dequant_dot_q4_K(const block_q4_K* blocks, uint32_t n_blocks,
                              const float* query_fp32) {
    __m256 acc = _mm256_setzero_ps();

    for (uint32_t b = 0; b < n_blocks; ++b) {
        float d_f32    = _cvtsh_ss(blocks[b].d);
        float dmin_f32 = _cvtsh_ss(blocks[b].dmin);

        const uint8_t* scales = blocks[b].scales;
        const uint8_t* qs     = blocks[b].qs;

        // Process 4 blocks of 64 weights each
        for (int blk = 0; blk < 4; ++blk) {
            int js = blk * 2;
            uint8_t sc0, mn0, sc1, mn1;

            // Scale/min pair 0
            if (js < 4) {
                sc0 = scales[js]     & 63;
                mn0 = scales[js + 4] & 63;
            } else {
                sc0 = (scales[js + 4] & 0x0F) | ((scales[js - 4] >> 6) << 4);
                mn0 = (scales[js + 4] >>    4) | ((scales[js] >> 6) << 4);
            }

            // Scale/min pair 1
            if (js + 1 < 4) {
                sc1 = scales[js + 1]     & 63;
                mn1 = scales[js + 1 + 4] & 63;
            } else {
                sc1 = (scales[js + 1 + 4] & 0x0F) | ((scales[js + 1 - 4] >> 6) << 4);
                mn1 = (scales[js + 1 + 4] >>    4) | ((scales[js + 1] >> 6) << 4);
            }

            float rs0 = d_f32 * static_cast<float>(sc0);
            float rm0 = dmin_f32 * static_cast<float>(mn0);
            float rs1 = d_f32 * static_cast<float>(sc1);
            float rm1 = dmin_f32 * static_cast<float>(mn1);

            __m256 scale0 = _mm256_set1_ps(rs0);
            __m256 scale1 = _mm256_set1_ps(rs1);
            __m256 neg_min0 = _mm256_set1_ps(-rm0);
            __m256 neg_min1 = _mm256_set1_ps(-rm1);

            int qoff = blk * 32;
            int woff = blk * 64;

            // Load the 32 bytes for this 64-weight block
            __m128i raw0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs + qoff));
            __m128i raw1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs + qoff + 16));

            // Chunk 0: raw0 low nibbles -> weights 0-15 (scale/min pair 0)
            __m128i lo0 = _mm_and_si128(raw0, _mm_set1_epi8(0x0F));
            __m256i i32_0 = _mm256_cvtepu8_epi32(lo0);
            __m256 f32_0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i32_0), scale0, neg_min0);
            acc = _mm256_fmadd_ps(f32_0, _mm256_loadu_ps(query_fp32 + b * 256 + woff), acc);

            __m256i i32_1 = _mm256_cvtepu8_epi32(_mm_srli_si128(lo0, 8));
            __m256 f32_1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i32_1), scale0, neg_min0);
            acc = _mm256_fmadd_ps(f32_1, _mm256_loadu_ps(query_fp32 + b * 256 + woff + 8), acc);

            // Chunk 1: raw1 low nibbles -> weights 16-31 (scale/min pair 1)
            __m128i lo1 = _mm_and_si128(raw1, _mm_set1_epi8(0x0F));
            __m256i i32_2 = _mm256_cvtepu8_epi32(lo1);
            __m256 f32_2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i32_2), scale1, neg_min1);
            acc = _mm256_fmadd_ps(f32_2, _mm256_loadu_ps(query_fp32 + b * 256 + woff + 16), acc);

            __m256i i32_3 = _mm256_cvtepu8_epi32(_mm_srli_si128(lo1, 8));
            __m256 f32_3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i32_3), scale1, neg_min1);
            acc = _mm256_fmadd_ps(f32_3, _mm256_loadu_ps(query_fp32 + b * 256 + woff + 24), acc);

            // Chunk 2: raw0 high nibbles -> weights 32-47 (scale/min pair 0)
            __m128i hi0 = _mm_and_si128(_mm_srli_epi16(raw0, 4), _mm_set1_epi8(0x0F));
            __m256i i32_4 = _mm256_cvtepu8_epi32(hi0);
            __m256 f32_4 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i32_4), scale0, neg_min0);
            acc = _mm256_fmadd_ps(f32_4, _mm256_loadu_ps(query_fp32 + b * 256 + woff + 32), acc);

            __m256i i32_5 = _mm256_cvtepu8_epi32(_mm_srli_si128(hi0, 8));
            __m256 f32_5 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i32_5), scale0, neg_min0);
            acc = _mm256_fmadd_ps(f32_5, _mm256_loadu_ps(query_fp32 + b * 256 + woff + 40), acc);

            // Chunk 3: raw1 high nibbles -> weights 48-63 (scale/min pair 1)
            __m128i hi1 = _mm_and_si128(_mm_srli_epi16(raw1, 4), _mm_set1_epi8(0x0F));
            __m256i i32_6 = _mm256_cvtepu8_epi32(hi1);
            __m256 f32_6 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i32_6), scale1, neg_min1);
            acc = _mm256_fmadd_ps(f32_6, _mm256_loadu_ps(query_fp32 + b * 256 + woff + 48), acc);

            __m256i i32_7 = _mm256_cvtepu8_epi32(_mm_srli_si128(hi1, 8));
            __m256 f32_7 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i32_7), scale1, neg_min1);
            acc = _mm256_fmadd_ps(f32_7, _mm256_loadu_ps(query_fp32 + b * 256 + woff + 56), acc);
        }
    }

    return hsum256_ps(acc);
}

#include "types.h"

void dequantize_q8_0_to_fp16(const void* src, uint16_t* dst, size_t n_weights) {
    const uint8_t* p = static_cast<const uint8_t*>(src);
    size_t n_blocks = n_weights / 32;
    for (size_t b = 0; b < n_blocks; ++b) {
        const block_q8_0* block = reinterpret_cast<const block_q8_0*>(p + b * sizeof(block_q8_0));
        float d = fp16_to_fp32(block->d);
        for (int j = 0; j < 32; ++j) {
            float val = d * static_cast<float>(block->qs[j]);
            dst[b * 32 + j] = fp32_to_fp16(val);
        }
    }
}

void dequantize_q4_K_to_fp16(const void* src, uint16_t* dst, size_t n_weights) {
    const uint8_t* p = static_cast<const uint8_t*>(src);
    size_t n_blocks = n_weights / 256;
    for (size_t b = 0; b < n_blocks; ++b) {
        const block_q4_K* block = reinterpret_cast<const block_q4_K*>(p + b * sizeof(block_q4_K));
        float d_f32 = fp16_to_fp32(block->d);
        float dmin_f32 = fp16_to_fp32(block->dmin);

        for (int blk = 0; blk < 4; ++blk) {
            int js = blk * 2;
            const uint8_t* scales = block->scales;
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
            float rs0 = d_f32 * sc0, rm0 = dmin_f32 * mn0;
            float rs1 = d_f32 * sc1, rm1 = dmin_f32 * mn1;

            int qoff = blk * 32;
            int woff = blk * 64;
            for (int j = 0; j < 16; ++j) {
                dst[b*256 + woff + j]      = fp32_to_fp16(rs0 * (block->qs[qoff+j] & 0x0F) - rm0);
                dst[b*256 + woff + 16 + j] = fp32_to_fp16(rs1 * (block->qs[qoff+16+j] & 0x0F) - rm1);
                dst[b*256 + woff + 32 + j] = fp32_to_fp16(rs0 * (block->qs[qoff+j] >> 4) - rm0);
                dst[b*256 + woff + 48 + j] = fp32_to_fp16(rs1 * (block->qs[qoff+16+j] >> 4) - rm1);
            }
        }
    }
}

void dequantize_q2_K_to_fp16(const void* src, uint16_t* dst, size_t n_weights) {
    const uint8_t* p = static_cast<const uint8_t*>(src);
    // Q2_K: 84 bytes per block, 256 weights per block
    // d (FP16), dmin (FP16), scales[16], qs[64]
    struct block_q2_k_local {
        uint16_t d, dmin;
        uint8_t scales[16];
        uint8_t qs[64];
    };
    size_t n_blocks = n_weights / 256;
    for (size_t b = 0; b < n_blocks; ++b) {
        const block_q2_k_local* block =
            reinterpret_cast<const block_q2_k_local*>(p + b * 84);
        float d_all = fp16_to_fp32(block->d);
        float m_all = fp16_to_fp32(block->dmin);

        for (int j = 0; j < 256; ++j) {
            int sb = j / 8;           // sub-block index (0..31)
            int idx = j % 8;          // position within sub-block
            int s = sb / 2;
            float scale, min_val;
            if (sb % 2 == 0) {
                scale = d_all * (block->scales[s] & 0x0F);
                min_val = m_all * (block->scales[s] >> 4);
            } else {
                scale = d_all * (block->scales[s + 8] & 0x0F);
                min_val = m_all * (block->scales[s + 8] >> 4);
            }
            int byte_idx = j / 4;
            int bit_off = (j % 4) * 2;
            int q = (block->qs[byte_idx] >> bit_off) & 0x03;
            dst[b * 256 + j] = fp32_to_fp16(scale * q - min_val);
        }
    }
}

#ifdef QUANT_TEST_MAIN
#include "types.h"
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

    // ---- Test 4: fused_dequant_dot_q8_0 ----
    printf("\n=== Test 4: fused_dequant_dot_q8_0 ===\n");
    {
        const uint32_t n_blocks = 2;
        const uint32_t total_weights = n_blocks * 32;

        float query[64];
        for (uint32_t i = 0; i < total_weights; ++i)
            query[i] = static_cast<float>((i % 7) + 1) * 0.1f;

        block_q8_0 blocks[2];
        blocks[0].d = fp32_to_fp16(0.5f);
        for (int i = 0; i < 32; ++i)
            blocks[0].qs[i] = static_cast<int8_t>(i - 16);
        blocks[1].d = fp32_to_fp16(1.0f);
        for (int i = 0; i < 32; ++i)
            blocks[1].qs[i] = static_cast<int8_t>((i % 32) - 16);

        float fused = fused_dequant_dot_q8_0(blocks, n_blocks, query);

        float expected = 0.0f;
        for (uint32_t b = 0; b < n_blocks; ++b) {
            float scale = fp16_to_fp32(blocks[b].d);
            for (uint32_t i = 0; i < 32; ++i)
                expected += scale * static_cast<float>(blocks[b].qs[i]) * query[b * 32 + i];
        }

        float err = std::fabs(fused - expected);
        bool ok = err < 1e-2f;  // tolerance for floating-point accumulate order differences
        printf("  fused=%.6f expected=%.6f err=%.2e %s\n",
               fused, expected, err, ok ? "OK" : "FAIL");
        if (!ok) all_pass = false;
    }

    // ---- Test 5: fused_dequant_dot_q4_K ----
    printf("\n=== Test 5: fused_dequant_dot_q4_K ===\n");
    {
        const uint32_t n_blocks = 1;
        const uint32_t total_weights = 256;

        float query[256];
        for (uint32_t i = 0; i < total_weights; ++i)
            query[i] = static_cast<float>((i % 11) + 1) * 0.01f;

        block_q4_K block{};
        block.d    = fp32_to_fp16(63.0f);   // super-scale
        block.dmin = fp32_to_fp16(63.0f);   // super-minimum

        // Set scales: for sb=0..3, scales[sb]=sb+1, scales[sb+4]=sb
        //             for sb=4..7, use bit-packing via scales[sb+4] and upper bits
        for (int j = 0; j < 4; ++j) {
            block.scales[j]     = static_cast<uint8_t>(j + 1);  // scale for sb=j
            block.scales[j + 4] = static_cast<uint8_t>(j);      // min for sb=j
        }
        // For sb=4..7: sc = (scales[sb+4] & 0xF) | ((scales[sb-4] >> 6) << 4)
        //              mn = (scales[sb+4] >> 4) | ((scales[sb-0] >> 6) << 4)
        // Set upper 2 bits of scales[0..3] for sb=4..7 scale extraction
        block.scales[0] |= (1 << 6);  // sc for sb=4: upper bits = 1
        block.scales[1] |= (1 << 6);  // sc for sb=5
        block.scales[2] |= (1 << 6);  // sc for sb=6
        block.scales[3] |= (1 << 6);  // sc for sb=7
        block.scales[8]  = 0x10 | 0x02;  // sb=4: sc low bits=2, mn low bits=1 (high nibble)
        block.scales[9]  = 0x20 | 0x03;  // sb=5: sc=3, mn=2
        block.scales[10] = 0x30 | 0x04;  // sb=6: sc=4, mn=3
        block.scales[11] = 0x40 | 0x05;  // sb=7: sc=5, mn=4

        // Set quantized weights: each nibble = 5 (for simplicity)
        for (int i = 0; i < 128; ++i)
            block.qs[i] = 0x55;  // both nibbles = 5

        float fused = fused_dequant_dot_q4_K(&block, n_blocks, query);

                // Compute expected using canonical 4 blocks x 64 weights
        float d_f32    = fp16_to_fp32(block.d);
        float dmin_f32 = fp16_to_fp32(block.dmin);
        float expected = 0.0f;

        for (int blk = 0; blk < 4; ++blk) {
            int js = blk * 2;
            uint8_t sc0, mn0, sc1, mn1;

            if (js < 4) {
                sc0 = block.scales[js]     & 63;
                mn0 = block.scales[js + 4] & 63;
            } else {
                sc0 = (block.scales[js + 4] & 0x0F) | ((block.scales[js - 4] >> 6) << 4);
                mn0 = (block.scales[js + 4] >>    4) | ((block.scales[js] >> 6) << 4);
            }
            if (js + 1 < 4) {
                sc1 = block.scales[js + 1]     & 63;
                mn1 = block.scales[js + 1 + 4] & 63;
            } else {
                sc1 = (block.scales[js + 1 + 4] & 0x0F) | ((block.scales[js + 1 - 4] >> 6) << 4);
                mn1 = (block.scales[js + 1 + 4] >>    4) | ((block.scales[js + 1] >> 6) << 4);
            }

            float rs0 = d_f32 * static_cast<float>(sc0);
            float rm0 = dmin_f32 * static_cast<float>(mn0);
            float rs1 = d_f32 * static_cast<float>(sc1);
            float rm1 = dmin_f32 * static_cast<float>(mn1);

            int qoff = blk * 32;
            int woff = blk * 64;

            // qs[qoff..qoff+15] low nibbles -> weights woff..woff+15 (pair 0)
            for (int j = 0; j < 16; ++j) {
                float w = rs0 * static_cast<float>(block.qs[qoff + j] & 0x0F) - rm0;
                expected += w * query[woff + j];
            }
            // qs[qoff+16..qoff+31] low nibbles -> weights woff+16..woff+31 (pair 1)
            for (int j = 0; j < 16; ++j) {
                float w = rs1 * static_cast<float>(block.qs[qoff + 16 + j] & 0x0F) - rm1;
                expected += w * query[woff + 16 + j];
            }
            // qs[qoff..qoff+15] high nibbles -> weights woff+32..woff+47 (pair 0)
            for (int j = 0; j < 16; ++j) {
                float w = rs0 * static_cast<float>(block.qs[qoff + j] >> 4) - rm0;
                expected += w * query[woff + 32 + j];
            }
            // qs[qoff+16..qoff+31] high nibbles -> weights woff+48..woff+63 (pair 1)
            for (int j = 0; j < 16; ++j) {
                float w = rs1 * static_cast<float>(block.qs[qoff + 16 + j] >> 4) - rm1;
                expected += w * query[woff + 48 + j];
            }
        }

        float err = std::fabs(fused - expected);
        bool ok = err < 1e-2f;
        printf("  fused=%.6f expected=%.6f err=%.2e %s\n",
               fused, expected, err, ok ? "OK" : "FAIL");
        if (!ok) all_pass = false;
    }

    printf("\n%s\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
#endif
