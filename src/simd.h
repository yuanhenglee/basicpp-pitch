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

// WebAssembly SIMD128 kernels (when there is no AVX2).
#if !BPP_AVX2 && defined(__wasm_simd128__)
#define BPP_SIMD128 1
#include <wasm_simd128.h>
#else
#define BPP_SIMD128 0
#endif

namespace bpp::detail {

// x * y + z with one rounding, like std::fma. Without hardware FMA (WebAssembly, x86-64 below v3)
// std::fma is a slow library call. A float product is exact in double, so the double sum rounded
// to float is already the fused result unless it lies exactly halfway between two floats; that is
// musl's / FreeBSD's fmaf fast path, and the rare halfway case still goes to std::fma.
inline float fmadd(float x, float y, float z) {
#if defined(__FMA__) || defined(__ARM_FEATURE_FMA) || defined(_MSC_VER)
    return std::fma(x, y, z);
#else
    const double r = static_cast<double>(x) * y + z;
    uint64_t bits;
    std::memcpy(&bits, &r, sizeof bits);
    if ((bits & 0x1fffffff) != 0x10000000) return static_cast<float>(r);
    return std::fma(x, y, z);
#endif
}

#if BPP_SIMD128

// Fused float FMA chains on WebAssembly SIMD128, which has no deterministic fused multiply-add.
// Accumulators are f64x2 vectors holding float values exactly. A step computes x * w + acc in
// double, where the product of two floats is exact, and rounds that once more to float: this is
// the fused result unless the double sum lies exactly halfway between two floats (the case
// musl's fmaf handles separately). Such sums set bits in `flags`; the caller then recomputes the
// block with the scalar fmadd().
inline v128_t fma_f64x2(v128_t x, v128_t w, v128_t acc, v128_t& flags) {
    const v128_t r = wasm_f64x2_add(wasm_f64x2_mul(x, w), acc);
    const v128_t low = wasm_v128_and(r, wasm_i64x2_const_splat(0x1fffffff));
    flags = wasm_v128_or(flags, wasm_i64x2_eq(low, wasm_i64x2_const_splat(0x10000000)));
    v128_t f = wasm_f32x4_demote_f64x2_zero(r);
    __asm__("" : "+r"(f));  // keeps LLVM from folding demote + promote into scalar code
    return wasm_f64x2_promote_low_f32x4(f);
}

// Float lanes 0-1 / 2-3 of v as f64x2, and two float-exact f64x2 back as one f32x4.
inline v128_t f32x4_lo_f64x2(v128_t v) { return wasm_f64x2_promote_low_f32x4(v); }
inline v128_t f32x4_hi_f64x2(v128_t v) { return wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(v, v, 2, 3, 2, 3)); }
inline v128_t f64x2_pair_f32x4(v128_t lo, v128_t hi) {
    return wasm_i32x4_shuffle(wasm_f32x4_demote_f64x2_zero(lo), wasm_f32x4_demote_f64x2_zero(hi), 0, 1, 4, 5);
}

// x86 maxps / minps (second operand on ties and NaN); wasm's max / min follow IEEE instead.
inline v128_t max_x86(v128_t a, v128_t b) { return wasm_f32x4_pmax(b, a); }
inline v128_t min_x86(v128_t a, v128_t b) { return wasm_f32x4_pmin(b, a); }

#endif

// ReLU as x86's max(v, 0): +0 for -0 and NaN.
inline float relu(float v) { return v > 0.0f ? v : 0.0f; }

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

#if BPP_SIMD128

// 4-lane versions of the AVX2 functions above, operation for operation.
inline v128_t madd_ps4(v128_t a, v128_t b, v128_t c) { return wasm_f32x4_add(wasm_f32x4_mul(a, b), c); }

inline v128_t ldexp_ps4(v128_t a, v128_t e_in) {
    const v128_t maxe = wasm_f32x4_splat(277.0f);
    const v128_t e = max_x86(min_x86(e_in, maxe), wasm_f32x4_sub(wasm_f32x4_splat(0.0f), maxe));
    const v128_t ei = wasm_i32x4_trunc_sat_f32x4(e);
    v128_t b = wasm_i32x4_shr(ei, 2);
    const v128_t bias = wasm_i32x4_splat(127);
    v128_t c = wasm_i32x4_shl(wasm_i32x4_add(b, bias), 23);
    const v128_t out = wasm_f32x4_mul(wasm_f32x4_mul(wasm_f32x4_mul(a, c), c), c);
    b = wasm_i32x4_sub(wasm_i32x4_sub(wasm_i32x4_sub(ei, b), b), b);
    c = wasm_i32x4_shl(wasm_i32x4_add(b, bias), 23);
    return wasm_f32x4_mul(out, c);
}

inline v128_t exp_ps4(v128_t x_in) {
    const v128_t zero_mask = wasm_f32x4_lt(x_in, wasm_f32x4_splat(-104.0f));
    const v128_t x = min_x86(x_in, wasm_f32x4_splat(88.723f));
    const v128_t m = wasm_f32x4_floor(madd_ps4(x, wasm_f32x4_splat(1.44269504088896341f), wasm_f32x4_splat(0.5f)));
    v128_t r = madd_ps4(m, wasm_f32x4_splat(-0.693359375f), x);
    r = madd_ps4(m, wasm_f32x4_splat(2.12194440e-4f), r);
    const v128_t r2 = wasm_f32x4_mul(r, r);
    v128_t p_even = madd_ps4(r2, wasm_f32x4_splat(1.37449637986719608306884765625e-3f),
                             wasm_f32x4_splat(4.166965186595916748046875e-2f));
    const v128_t p_odd = madd_ps4(r2, wasm_f32x4_splat(8.36894474923610687255859375e-3f),
                                  wasm_f32x4_splat(0.16666518151760101318359375f));
    p_even = madd_ps4(r2, p_even, wasm_f32x4_splat(0.49999988079071044921875f));
    const v128_t p_low = wasm_f32x4_add(r, wasm_f32x4_splat(1.0f));
    v128_t y = madd_ps4(r, p_odd, p_even);
    y = madd_ps4(r2, y, p_low);
    const v128_t res = max_x86(ldexp_ps4(y, m), x_in);
    return wasm_v128_andnot(res, zero_mask);
}

inline v128_t logistic_ps4(v128_t x) {
    const v128_t one = wasm_f32x4_splat(1.0f);
    const v128_t e = exp_ps4(x);
    const v128_t inf_mask = wasm_f32x4_eq(e, wasm_f32x4_splat(std::numeric_limits<float>::infinity()));
    return wasm_v128_bitselect(one, wasm_f32x4_div(e, wasm_f32x4_add(one, e)), inf_mask);
}

inline v128_t log_ps4(v128_t x) {
    const v128_t one = wasm_f32x4_splat(1.0f);
    v128_t e = wasm_f32x4_convert_i32x4(
        wasm_i32x4_sub(wasm_v128_and(wasm_u32x4_shr(x, 23), wasm_i32x4_splat(0xff)), wasm_i32x4_splat(126)));
    v128_t m = wasm_v128_or(wasm_v128_and(x, wasm_i32x4_splat(static_cast<int>(0x807fffffu))), wasm_i32x4_splat(0x3f000000));
    const v128_t mask = wasm_f32x4_lt(m, wasm_f32x4_splat(0.707106781186547524f));
    const v128_t tmp = wasm_v128_and(m, mask);
    m = wasm_f32x4_sub(m, one);
    e = wasm_f32x4_sub(e, wasm_v128_and(one, mask));
    m = wasm_f32x4_add(m, tmp);
    v128_t p = madd_ps4(m, wasm_f32x4_splat(0.18256296349849254f), wasm_f32x4_splat(1.0000000190281063f));
    p = madd_ps4(m, p, wasm_f32x4_splat(1.0000000190281136f));
    p = wasm_f32x4_mul(m, p);
    v128_t q = madd_ps4(m, wasm_f32x4_splat(0.049616247954120038f), wasm_f32x4_splat(0.59923249590823520f));
    q = madd_ps4(m, q, wasm_f32x4_splat(1.4999999999999927f));
    q = madd_ps4(m, q, one);
    return madd_ps4(e, wasm_f32x4_splat(0.693147182464599609375f), wasm_f32x4_div(p, q));
}

#endif

}  // namespace eigen

}  // namespace bpp::detail
