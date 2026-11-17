// SIMD helpers plus bit-exact ports of the Eigen math kernels TensorFlow uses on CPU.
//
// TensorFlow's x86 wheels compile Eigen for AVX without FMA, so every pmadd() in Eigen is a
// separately rounded multiply and add. The functions below reproduce Eigen's float paths
// (plog_impl_float, pexp_float, scalar_logistic_op) operation for operation; the scalar and
// AVX2 versions return identical bits. This file must be compiled with -ffp-contract=off.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

// MSVC's /arch:AVX2 implies FMA3 but never defines __FMA__.
#if defined(__AVX2__) && (defined(__FMA__) || defined(_MSC_VER))
#define BPP_AVX2 1
#include <immintrin.h>
#else
#define BPP_AVX2 0
#endif

namespace bpp::detail {

// 64-byte aligned, zero-initialised float buffer.
class FloatBuffer {
public:
    FloatBuffer() = default;
    explicit FloatBuffer(size_t n) { resize(n); }
    FloatBuffer(const FloatBuffer&) = delete;
    FloatBuffer& operator=(const FloatBuffer&) = delete;
    FloatBuffer(FloatBuffer&& o) noexcept : p_(o.p_), n_(o.n_) { o.p_ = nullptr; o.n_ = 0; }
    FloatBuffer& operator=(FloatBuffer&& o) noexcept {
        std::swap(p_, o.p_);
        std::swap(n_, o.n_);
        return *this;
    }
    ~FloatBuffer() { ::operator delete[](p_, std::align_val_t(64)); }

    void resize(size_t n) {
        ::operator delete[](p_, std::align_val_t(64));
        // round up so vector loops may over-read a little past the end
        n_ = (n + 63) & ~size_t(63);
        p_ = static_cast<float*>(::operator new[](n_ * sizeof(float), std::align_val_t(64)));
        std::memset(p_, 0, n_ * sizeof(float));
    }
    float* data() { return p_; }
    const float* data() const { return p_; }
    float& operator[](size_t i) { return p_[i]; }
    const float& operator[](size_t i) const { return p_[i]; }
    size_t size() const { return n_; }

private:
    float* p_ = nullptr;
    size_t n_ = 0;
};

namespace eigen {

inline float bits_to_float(uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
inline uint32_t float_to_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

// Eigen plog_impl_float (rational 3/3 approximation), positive normal inputs.
inline float log(float x) {
    const uint32_t bits = float_to_bits(x);
    float e = static_cast<float>(static_cast<int>((bits >> 23) & 0xff) - 126);
    float m = bits_to_float((bits & 0x807fffffu) | 0x3f000000u);
    const bool small = m < 0.707106781186547524f;
    const float tmp = small ? m : 0.0f;
    m = m - 1.0f;
    e = e - (small ? 1.0f : 0.0f);
    m = m + tmp;
    float p = m * 0.18256296349849254f + 1.0000000190281063f;
    p = m * p + 1.0000000190281136f;
    p = m * p;
    float q = m * 0.049616247954120038f + 0.59923249590823520f;
    q = m * q + 1.4999999999999927f;
    q = m * q + 1.0f;
    const float r = p / q;
    return e * 0.693147182464599609375f + r;  // float(M_LN2)
}

// Eigen pldexp_generic: a * 2^e, exact except for the final rounding into the subnormal range.
inline float ldexp(float a, float e_in) {
    const float max_exponent = 277.0f;
    const float e = std::fmax(std::fmin(e_in, max_exponent), -max_exponent);
    const int ei = static_cast<int>(e);
    int b = ei >> 2;  // floor(e / 4)
    float c = bits_to_float(static_cast<uint32_t>(b + 127) << 23);
    float out = ((a * c) * c) * c;
    b = ei - b - b - b;
    c = bits_to_float(static_cast<uint32_t>(b + 127) << 23);
    return out * c;
}

// Eigen pexp_float (degree-6 minimax polynomial).
inline float exp(float x_in) {
    if (x_in < -104.0f) return 0.0f;
    const float x = std::fmin(x_in, 88.723f);
    const float m = std::floor(x * 1.44269504088896341f + 0.5f);
    float r = m * -0.693359375f + x;
    r = m * 2.12194440e-4f + r;
    const float r2 = r * r;
    float p_even = r2 * 1.37449637986719608306884765625e-3f + 4.166965186595916748046875e-2f;
    const float p_odd = r2 * 8.36894474923610687255859375e-3f + 0.16666518151760101318359375f;
    p_even = r2 * p_even + 0.49999988079071044921875f;
    const float p_low = r + 1.0f;
    float y = r * p_odd + p_even;
    y = r2 * y + p_low;
    return std::fmax(ldexp(y, m), x_in);
}

// Eigen scalar_logistic_op (generic form): e^x / (1 + e^x), 1 where e^x overflows.
inline float logistic(float x) {
    const float e = exp(x);
    if (e == std::numeric_limits<float>::infinity()) return 1.0f;
    return e / (1.0f + e);
}

#if BPP_AVX2

inline __m256 ldexp_ps(__m256 a, __m256 e_in) {
    const __m256 maxe = _mm256_set1_ps(277.0f);
    const __m256 e = _mm256_max_ps(_mm256_min_ps(e_in, maxe), _mm256_sub_ps(_mm256_setzero_ps(), maxe));
    const __m256i ei = _mm256_cvttps_epi32(e);
    __m256i b = _mm256_srai_epi32(ei, 2);
    const __m256i bias = _mm256_set1_epi32(127);
    __m256 c = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(b, bias), 23));
    __m256 out = _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(a, c), c), c);
    b = _mm256_sub_epi32(_mm256_sub_epi32(_mm256_sub_epi32(ei, b), b), b);
    c = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(b, bias), 23));
    return _mm256_mul_ps(out, c);
}

inline __m256 madd(__m256 a, __m256 b, __m256 c) { return _mm256_add_ps(_mm256_mul_ps(a, b), c); }

inline __m256 exp_ps(__m256 x_in) {
    const __m256 zero_mask = _mm256_cmp_ps(x_in, _mm256_set1_ps(-104.0f), _CMP_LT_OQ);
    const __m256 x = _mm256_min_ps(x_in, _mm256_set1_ps(88.723f));
    const __m256 m = _mm256_floor_ps(madd(x, _mm256_set1_ps(1.44269504088896341f), _mm256_set1_ps(0.5f)));
    __m256 r = madd(m, _mm256_set1_ps(-0.693359375f), x);
    r = madd(m, _mm256_set1_ps(2.12194440e-4f), r);
    const __m256 r2 = _mm256_mul_ps(r, r);
    __m256 p_even = madd(r2, _mm256_set1_ps(1.37449637986719608306884765625e-3f),
                         _mm256_set1_ps(4.166965186595916748046875e-2f));
    const __m256 p_odd = madd(r2, _mm256_set1_ps(8.36894474923610687255859375e-3f),
                              _mm256_set1_ps(0.16666518151760101318359375f));
    p_even = madd(r2, p_even, _mm256_set1_ps(0.49999988079071044921875f));
    const __m256 p_low = _mm256_add_ps(r, _mm256_set1_ps(1.0f));
    __m256 y = madd(r, p_odd, p_even);
    y = madd(r2, y, p_low);
    const __m256 res = _mm256_max_ps(ldexp_ps(y, m), x_in);
    return _mm256_andnot_ps(zero_mask, res);
}

inline __m256 logistic_ps(__m256 x) {
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 e = exp_ps(x);
    const __m256 inf_mask = _mm256_cmp_ps(e, _mm256_set1_ps(std::numeric_limits<float>::infinity()), _CMP_EQ_OQ);
    return _mm256_blendv_ps(_mm256_div_ps(e, _mm256_add_ps(one, e)), one, inf_mask);
}

inline __m256 log_ps(__m256 x) {
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256i xi = _mm256_castps_si256(x);
    __m256 e = _mm256_cvtepi32_ps(
        _mm256_sub_epi32(_mm256_and_si256(_mm256_srli_epi32(xi, 23), _mm256_set1_epi32(0xff)), _mm256_set1_epi32(126)));
    __m256 m = _mm256_castsi256_ps(_mm256_or_si256(_mm256_and_si256(xi, _mm256_set1_epi32(static_cast<int>(0x807fffffu))),
                                                    _mm256_set1_epi32(0x3f000000)));
    const __m256 mask = _mm256_cmp_ps(m, _mm256_set1_ps(0.707106781186547524f), _CMP_LT_OQ);
    const __m256 tmp = _mm256_and_ps(m, mask);
    m = _mm256_sub_ps(m, one);
    e = _mm256_sub_ps(e, _mm256_and_ps(one, mask));
    m = _mm256_add_ps(m, tmp);
    __m256 p = madd(m, _mm256_set1_ps(0.18256296349849254f), _mm256_set1_ps(1.0000000190281063f));
    p = madd(m, p, _mm256_set1_ps(1.0000000190281136f));
    p = _mm256_mul_ps(m, p);
    __m256 q = madd(m, _mm256_set1_ps(0.049616247954120038f), _mm256_set1_ps(0.59923249590823520f));
    q = madd(m, q, _mm256_set1_ps(1.4999999999999927f));
    q = madd(m, q, one);
    return madd(e, _mm256_set1_ps(0.693147182464599609375f), _mm256_div_ps(p, q));
}

#endif

}  // namespace eigen

}  // namespace bpp::detail
