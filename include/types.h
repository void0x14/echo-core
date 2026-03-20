#pragma once

#include <cstdint>
#include <cstddef>
#include <immintrin.h>

using fp16_t = uint16_t;

constexpr size_t CACHE_LINE_SIZE = 64;

#define EC_ALIGN alignas(64)

inline float fp16_to_fp32(fp16_t h) {
    return _cvtsh_ss(h);
}

inline fp16_t fp32_to_fp16(float f) {
    return _cvtss_sh(f, 0);
}

inline void fp16_to_fp32_row(const fp16_t* src, float* dst, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
        __m256 f = _mm256_cvtph_ps(h);
        _mm256_storeu_ps(dst + i, f);
    }
    for (; i < n; ++i) {
        dst[i] = _cvtsh_ss(src[i]);
    }
}

inline void fp32_to_fp16_row(const float* src, fp16_t* dst, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 f = _mm256_loadu_ps(src + i);
        __m128i h = _mm256_cvtps_ph(f, 0);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), h);
    }
    for (; i < n; ++i) {
        dst[i] = _cvtss_sh(src[i], 0);
    }
}
