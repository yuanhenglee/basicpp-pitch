#include "window_model.h"

#include "model_data.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#ifdef BPP_PROFILE
#include <chrono>
#include <cstdio>
#endif

namespace bpp::detail {

#ifdef BPP_PROFILE
// Per-stage wall time, summed over all windows (single-threaded profiling builds only).
namespace prof {
inline double stage[8];
inline const char* names[8] = {"cqt", "normalize+stack", "conv2d_1", "contours-reduced", "conv2d_2+conv2d_4",
                               "conv2d_3", "conv2d_5", "copy-out"};
struct Report {
    ~Report() {
        double total = 0;
        for (double v : stage) total += v;
        for (int i = 0; i < 8; ++i)
            std::fprintf(stderr, "%-20s %8.1f ms  %5.1f%%\n", names[i], stage[i] * 1e3, 100 * stage[i] / total);
    }
} inline report;
struct Scope {
    int id;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    ~Scope() { stage[id] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
};
}  // namespace prof
#define BPP_STAGE(i) prof::Scope bpp_scope_##i{i}
#else
#define BPP_STAGE(i)
#endif

namespace {

// Decimation output length of nnAudio's downsampling_by_n (pad 127 each side, 256 taps, stride 2).
constexpr int decimated_length(int n) { return (n - 2) / 2 + 1; }

constexpr int octave_length(int octave) {
    int n = kWindowSamples;
    for (int i = 0; i < octave; ++i) n = decimated_length(n);
    return n;
}

constexpr int kReflectPad = kCqtKernelLen / 2;  // 128
constexpr int kDecimatePad = (kLowpassLen - 1) / 2;  // 127

// Tensor layouts. "Pad" columns on the frequency axis are zero and implement "SAME" padding.
constexpr int kHsPad = 19, kHsRow = (kContourBins + 2 * kHsPad) * kHarmonics;  // NHWC
constexpr int kC1Pad = 2, kC1Row = 272, kC1Ch = kFramesPerWindow * kC1Row;      // channel-major
constexpr int kYpPad = 2, kYpRow = kContourBins + 2 * kYpPad;                   // single channel
constexpr int kN1Pad = 1, kN1Row = 96, kN1Ch = kFramesPerWindow * kN1Row;       // channel-major
constexpr int kCatPad = 1, kCatRow = 96, kCatCh = kFramesPerWindow * kCatRow;   // channel-major

// Rows each stage must produce so that output frames [15, 157) are exact. Every convolution
// only reads rows inside the window, so no time padding is ever touched.
constexpr int kOutBegin = kHalfOverlap, kOutEnd = kHalfOverlap + kUsedFrames;  // [15, 157)
constexpr int kCatBegin = kOutBegin - 1, kCatEnd = kOutEnd + 1;                // conv2d_5 is 3 tall
constexpr int kN1Begin = kCatBegin - 3, kN1End = kCatEnd + 3;                  // conv2d_3 is 7 tall
constexpr int kYpBegin = kN1Begin - 3, kYpEnd = kN1End + 3;                    // conv2d_2 is 7 tall
constexpr int kC1Begin = kYpBegin - 2, kC1End = kYpEnd + 2;                    // contours-reduced is 5 tall
constexpr int kHsBegin = std::min(kC1Begin - 1, kCatBegin - 2);
constexpr int kHsEnd = std::max(kC1End + 1, kCatEnd + 2);
static_assert(kHsBegin >= 0 && kHsEnd <= kFramesPerWindow, "receptive field exceeds window");

constexpr int kHarmonicShift[kHarmonics] = {-36, 0, 36, 57, 72, 84, 93, 101};  // round(36*log2(h))


constexpr float kLn10 = 2.30258512f;  // float32 log(10), as TensorFlow evaluates it
// Grappler's constant push-down rewrites 10 * (log(x) / log(10)) into log(x) * (10 / log(10)).
const float kDbPerNeper = 10.0f / kLn10;
constexpr float kBnEps = 0.001f;

inline void run_rows(const RowRunner* runner, int b, int e, const std::function<void(int, int)>& fn) {
    if (runner) (*runner)(b, e, fn);
    else fn(b, e);
}

// ---------------------------------------------------------------------------------------------
// Convolutions. Each output value is one FMA chain whose order matches the oneDNN kernel that
// TensorFlow selects for that layer; only the vectorisation (which outputs share a register)
// is free to differ.

// conv2d_1 (oneDNN brgconv, 3x39, 8 -> 8). oneDNN keeps one partial sum per kernel column:
// part[kw] = FMA chain from 0 over (kh, ic); out = ((part[0] + part[1]) + ...) + bias.
// Then batch norm (bnorm_tbb_jit) and ReLU. Lanes = output channels, 12 pixels per block.
void conv_contour1(const float* hs, const float* inv_std, int t0, int t1, float* c1) {
    using namespace data;
#if BPP_AVX2
    const __m256 vb = _mm256_loadu_ps(kConvContour1_b), vm = _mm256_loadu_ps(kBnContour_mean);
    const __m256 vi = _mm256_loadu_ps(inv_std), vg = _mm256_loadu_ps(kBnContour_gamma);
    const __m256 vbeta = _mm256_loadu_ps(kBnContour_beta), zero = _mm256_setzero_ps();
    for (int t = t0; t < t1; ++t) {
        for (int f0 = 0; f0 < kContourBins; f0 += 12) {
            __m256 total[12];
            for (auto& a : total) a = zero;
            for (int kw = 0; kw < 39; ++kw) {
                __m256 part[12];
                for (auto& a : part) a = zero;
                for (int kh = 0; kh < 3; ++kh) {
                    const float* a = hs + (t + kh - 1) * kHsRow + (f0 + kw) * 8;  // logical column f + kw - 19
                    const float* wk = kConvContour1_w + (kh * 39 + kw) * 64;
                    for (int ic = 0; ic < 8; ++ic) {
                        const __m256 wv = _mm256_loadu_ps(wk + 8 * ic);
                        for (int i = 0; i < 12; ++i)
                            part[i] = _mm256_fmadd_ps(_mm256_broadcast_ss(a + 8 * i + ic), wv, part[i]);
                    }
                }
                for (int i = 0; i < 12; ++i) total[i] = _mm256_add_ps(total[i], part[i]);
            }
            alignas(32) float y[12][8];
            for (int i = 0; i < 12; ++i) {
                __m256 v = _mm256_add_ps(total[i], vb);
                v = _mm256_mul_ps(_mm256_sub_ps(v, vm), vi);
                v = _mm256_fmadd_ps(v, vg, vbeta);
                _mm256_store_ps(y[i], _mm256_max_ps(v, zero));
            }
            for (int c = 0; c < 8; ++c) {
                float* dst = c1 + c * kC1Ch + t * kC1Row + kC1Pad + f0;
                for (int i = 0; i < 12; ++i) dst[i] = y[i][c];
            }
        }
    }
#else
    for (int t = t0; t < t1; ++t)
        for (int f = 0; f < kContourBins; ++f) {
            float total[8] = {};
            for (int kw = 0; kw < 39; ++kw) {
                float part[8] = {};
                for (int kh = 0; kh < 3; ++kh) {
                    const float* a = hs + (t + kh - 1) * kHsRow + (f + kw) * 8;
                    const float* wk = kConvContour1_w + (kh * 39 + kw) * 64;
                    for (int ic = 0; ic < 8; ++ic)
                        for (int c = 0; c < 8; ++c) part[c] = fmadd(a[ic], wk[8 * ic + c], part[c]);
                }
                for (int c = 0; c < 8; ++c) total[c] += part[c];
            }
            for (int c = 0; c < 8; ++c) {
                const float v = ((total[c] + kConvContour1_b[c]) - kBnContour_mean[c]) * inv_std[c];
                c1[c * kC1Ch + t * kC1Row + kC1Pad + f] =
                    std::max(fmadd(v, kBnContour_gamma[c], kBnContour_beta[c]), 0.0f);
            }
        }
#endif
}

// 32 output channels, frequency stride 3, lanes = output channels, 2 pixels per block.
//  conv2d_2 (oneDNN jit:avx2, 7x7, 1 -> 32): acc starts at the bias, ReLU post-op.
//  conv2d_4 (oneDNN brgconv, 5x5, 8 -> 32): acc starts at 0, + bias, batch norm, ReLU.
// Output is scattered to a channel-major tensor at out + c * out_ch + t * out_row + f.
template <bool kBiasFirst, bool kBatchNorm>
void conv_out32_s3(const float* in, int in_row, int in_c, int col_off, const float* w, int kh, int k_len,
                   int row_off, int t0, int t1, const float* bias, const float* inv_std, float* out, int out_ch,
                   int out_row) {
    using namespace data;
#if BPP_AVX2
    const __m256 zero = _mm256_setzero_ps();
    for (int t = t0; t < t1; ++t) {
        for (int f0 = 0; f0 < kNoteBins; f0 += 2) {
            __m256 acc[2][4];
            for (int j = 0; j < 4; ++j) acc[0][j] = acc[1][j] = kBiasFirst ? _mm256_loadu_ps(bias + 8 * j) : zero;
            for (int kt = 0; kt < kh; ++kt) {
                const float* a0 = in + (t + kt + row_off) * in_row + (3 * f0 + col_off) * in_c;
                const float* a1 = a0 + 3 * in_c;
                const float* wk = w + kt * k_len * 32;
                for (int k = 0; k < k_len; ++k) {
                    const __m256 w0 = _mm256_loadu_ps(wk + 32 * k), w1 = _mm256_loadu_ps(wk + 32 * k + 8);
                    const __m256 w2 = _mm256_loadu_ps(wk + 32 * k + 16), w3 = _mm256_loadu_ps(wk + 32 * k + 24);
                    const __m256 b0 = _mm256_broadcast_ss(a0 + k), b1 = _mm256_broadcast_ss(a1 + k);
                    acc[0][0] = _mm256_fmadd_ps(b0, w0, acc[0][0]);
                    acc[0][1] = _mm256_fmadd_ps(b0, w1, acc[0][1]);
                    acc[0][2] = _mm256_fmadd_ps(b0, w2, acc[0][2]);
                    acc[0][3] = _mm256_fmadd_ps(b0, w3, acc[0][3]);
                    acc[1][0] = _mm256_fmadd_ps(b1, w0, acc[1][0]);
                    acc[1][1] = _mm256_fmadd_ps(b1, w1, acc[1][1]);
                    acc[1][2] = _mm256_fmadd_ps(b1, w2, acc[1][2]);
                    acc[1][3] = _mm256_fmadd_ps(b1, w3, acc[1][3]);
                }
            }
            alignas(32) float y[2][32];
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 4; ++j) {
                    __m256 v = acc[i][j];
                    if (!kBiasFirst) v = _mm256_add_ps(v, _mm256_loadu_ps(bias + 8 * j));
                    if (kBatchNorm) {
                        v = _mm256_mul_ps(_mm256_sub_ps(v, _mm256_loadu_ps(kBnOnset_mean + 8 * j)),
                                          _mm256_loadu_ps(inv_std + 8 * j));
                        v = _mm256_fmadd_ps(v, _mm256_loadu_ps(kBnOnset_gamma + 8 * j), _mm256_loadu_ps(kBnOnset_beta + 8 * j));
                    }
                    _mm256_store_ps(y[i] + 8 * j, _mm256_max_ps(v, zero));
                }
            for (int c = 0; c < 32; ++c) {
                float* dst = out + c * out_ch + t * out_row + f0;
                dst[0] = y[0][c];
                dst[1] = y[1][c];
            }
        }
    }
#else
    for (int t = t0; t < t1; ++t)
        for (int f = 0; f < kNoteBins; ++f) {
            float acc[32];
            for (int c = 0; c < 32; ++c) acc[c] = kBiasFirst ? bias[c] : 0.0f;
            for (int kt = 0; kt < kh; ++kt) {
                const float* a = in + (t + kt + row_off) * in_row + (3 * f + col_off) * in_c;
                const float* wk = w + kt * k_len * 32;
                for (int k = 0; k < k_len; ++k)
                    for (int c = 0; c < 32; ++c) acc[c] = fmadd(a[k], wk[32 * k + c], acc[c]);
            }
            for (int c = 0; c < 32; ++c) {
                float v = acc[c];
                if (!kBiasFirst) v += bias[c];
                if (kBatchNorm) v = fmadd((v - kBnOnset_mean[c]) * inv_std[c], kBnOnset_gamma[c], kBnOnset_beta[c]);
                out[c * out_ch + t * out_row + f] = std::max(v, 0.0f);
            }
        }
#endif
}

// Single output channel (oneDNN brgconv) + bias + Eigen logistic, on a channel-major input.
// acc = 0; for kh, kw, ic: acc = fma(x, w, acc). Lanes = 8 consecutive output pixels.
struct Out1Conv {
    const float* in;  // channel-major, logical column f maps to in[c * ch + t * row + f + col_off]
    int ch, row, n_in, col_off;
    int kh, kw, row_off;
    const float* w;  // HWIO [kh][kw][n_in]
    float bias;
};

template <int NV>
void out1_block(const Out1Conv& p, int t, int f0, float* out, int out_pix) {
#if BPP_AVX2
    __m256 acc[NV];
    for (auto& a : acc) a = _mm256_setzero_ps();
    for (int i = 0; i < p.kh; ++i)
        for (int j = 0; j < p.kw; ++j) {
            const float* src = p.in + (t + i + p.row_off) * p.row + f0 + j + p.col_off;
            const float* wk = p.w + (i * p.kw + j) * p.n_in;
            for (int c = 0; c < p.n_in; ++c) {
                const __m256 wv = _mm256_broadcast_ss(wk + c);
                const float* s = src + c * p.ch;
                for (int v = 0; v < NV; ++v) acc[v] = _mm256_fmadd_ps(_mm256_loadu_ps(s + 8 * v), wv, acc[v]);
            }
        }
    const __m256 vb = _mm256_set1_ps(p.bias);
    for (int v = 0; v < NV; ++v) {
        const __m256 y = eigen::logistic_ps(_mm256_add_ps(acc[v], vb));
        if (out_pix == 1) {
            _mm256_storeu_ps(out + f0 + 8 * v, y);
        } else {
            alignas(32) float tmp[8];
            _mm256_store_ps(tmp, y);
            for (int l = 0; l < 8; ++l) out[(f0 + 8 * v + l) * out_pix] = tmp[l];
        }
    }
#else
    for (int l = 0; l < 8 * NV; ++l) {
        float acc = 0.0f;
        for (int i = 0; i < p.kh; ++i)
            for (int j = 0; j < p.kw; ++j)
                for (int c = 0; c < p.n_in; ++c)
                    acc = fmadd(p.in[c * p.ch + (t + i + p.row_off) * p.row + f0 + l + j + p.col_off],
                                   p.w[(i * p.kw + j) * p.n_in + c], acc);
        out[(f0 + l) * out_pix] = eigen::logistic(acc + p.bias);
    }
#endif
}

// Runs rows [t0, t1) over n_out output pixels (88 or 264). Output row t goes to
// out + (t - out_t0) * out_row, pixel f at offset f * out_pix. Blocks of 11 vectors keep
// enough independent FMA chains in flight to hide FMA latency.
void conv_out1(const Out1Conv& p, int t0, int t1, int n_out, float* out, int out_row, int out_pix, int out_t0) {
    for (int t = t0; t < t1; ++t) {
        float* o = out + (t - out_t0) * out_row;
        for (int f0 = 0; f0 < n_out; f0 += 88) out1_block<11>(p, t, f0, o, out_pix);
    }
}

}  // namespace

// ---------------------------------------------------------------------------------------------

Workspace::Workspace() {
    for (int o = 0; o < kOctaves; ++o) sig[o].resize(octave_length(o) + 2 * kReflectPad + 64);
    const int max_out = (decimated_length(kWindowSamples) + 31) / 32 * 32;
    even.resize(max_out + kLowpassLen / 2 + 64);
    odd.resize(max_out + kLowpassLen / 2 + 64);
    down.resize(max_out + 64);
    cqt_re.resize(kFramesPerWindow * kCqtBins);
    cqt_im.resize(kFramesPerWindow * kCqtBins);
    logp.resize(kFramesPerWindow * kCqtBins);
    hs.resize(kFramesPerWindow * kHsRow);
    c1.resize(8 * kC1Ch);
    yp.resize(kFramesPerWindow * kYpRow);
    n1.resize(32 * kN1Ch);
    cat.resize(33 * kCatCh);
}

const WindowModel& WindowModel::get() {
    static const WindowModel model;
    return model;
}

WindowModel::WindowModel() {
    using namespace data;
    // Pack the CQT kernel per 8 bins, trimmed to the union of the bins' nonzero taps. Skipping
    // zero taps leaves every FMA chain bit-identical (fma(x, 0, acc) == acc).
    for (int g = 0; g < 5; ++g) {
        int k0 = kCqtKernelLen, k1 = 0;
        for (int l = 0; l < 8; ++l) {
            const int b = 8 * g + l;
            if (b >= kBinsPerOctave) break;
            for (int k = 0; k < kCqtKernelLen; ++k)
                if (kCqtKernelReal[b * kCqtKernelLen + k] != 0.0f || kCqtKernelImag[b * kCqtKernelLen + k] != 0.0f) {
                    k0 = std::min(k0, k);
                    k1 = std::max(k1, k + 1);
                }
        }
        CqtGroup& grp = cqt_groups_[g];
        grp.k0 = k0;
        grp.k1 = k1;
        grp.w.resize(std::max(0, k1 - k0) * 16);
        for (int k = k0; k < k1; ++k)
            for (int l = 0; l < 8; ++l) {
                const int b = 8 * g + l;
                if (b >= kBinsPerOctave) continue;
                grp.w[(k - k0) * 16 + l] = kCqtKernelReal[b * kCqtKernelLen + k];
                grp.w[(k - k0) * 16 + 8 + l] = kCqtKernelImag[b * kCqtKernelLen + k];
            }
    }
    // oneDNN nspc_bnorm (1 channel): (x - mean) * (gamma / sqrt(var + eps)) + beta
    bn_cqt_scale_ = kBnCqt_gamma[0] / std::sqrt(kBnCqt_var[0] + kBnEps);
    // oneDNN bnorm_tbb_jit: fma((x - mean) * (1 / sqrt(var + eps)), gamma, beta)
    for (int c = 0; c < 8; ++c) c1_inv_std_[c] = 1.0f / std::sqrt(kBnContour_var[c] + kBnEps);
    for (int c = 0; c < 32; ++c) o1_inv_std_[c] = 1.0f / std::sqrt(kBnOnset_var[c] + kBnEps);
}

namespace {

// Low-pass + decimate by 2 (nnAudio downsampling_by_n, zero padding of 127 on each side).
// oneDNN jit:avx2 conv: acc = 0, then taps 0..255 in order.
int decimate(const float* x, int n, Workspace& ws) {
    const int n_out = decimated_length(n);
    const int n_out_pad = (n_out + 31) / 32 * 32;
    const int m_needed = n_out_pad + kLowpassLen / 2;
    float* e = ws.even.data();
    float* o = ws.odd.data();
    for (int m = 0; m < m_needed; ++m) {
        const int ie = 2 * m - kDecimatePad, io = ie + 1;
        e[m] = (ie >= 0 && ie < n) ? x[ie] : 0.0f;
        o[m] = (io >= 0 && io < n) ? x[io] : 0.0f;
    }
    const float* h = data::kLowpass;
    float* out = ws.down.data();
#if BPP_AVX2
    for (int j = 0; j < n_out_pad; j += 32) {
        __m256 acc0 = _mm256_setzero_ps(), acc1 = acc0, acc2 = acc0, acc3 = acc0;
        for (int i = 0; i < kLowpassLen / 2; ++i) {
            const __m256 he = _mm256_broadcast_ss(h + 2 * i), ho = _mm256_broadcast_ss(h + 2 * i + 1);
            const float* pe = e + j + i;
            const float* po = o + j + i;
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pe), he, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(pe + 8), he, acc1);
            acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(pe + 16), he, acc2);
            acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(pe + 24), he, acc3);
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(po), ho, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(po + 8), ho, acc1);
            acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(po + 16), ho, acc2);
            acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(po + 24), ho, acc3);
        }
        _mm256_storeu_ps(out + j, acc0);
        _mm256_storeu_ps(out + j + 8, acc1);
        _mm256_storeu_ps(out + j + 16, acc2);
        _mm256_storeu_ps(out + j + 24, acc3);
    }
#else
    for (int j = 0; j < n_out; ++j) {
        float acc = 0;
        for (int i = 0; i < kLowpassLen / 2; ++i) {
            acc = fmadd(e[j + i], h[2 * i], acc);
            acc = fmadd(o[j + i], h[2 * i + 1], acc);
        }
        out[j] = acc;
    }
#endif
    return n_out;
}

void reflect_pad(const float* x, int n, float* r) {
    for (int i = 0; i < kReflectPad; ++i) r[i] = x[kReflectPad - i];
    std::memcpy(r + kReflectPad, x, sizeof(float) * n);
    for (int i = 0; i < kReflectPad; ++i) r[kReflectPad + n + i] = x[n - 2 - i];
}

// CQT conv1d (oneDNN brgconv / jit:avx2 with a single input channel): acc = 0, taps in order.
template <int MR>
void cqt_block(const float* sig, int hop, int f0, const float* w, int k0, int k1, float* re, float* im) {
#if BPP_AVX2
    __m256 ar[MR], ai[MR];
    for (int r = 0; r < MR; ++r) ar[r] = ai[r] = _mm256_setzero_ps();
    for (int k = k0; k < k1; ++k) {
        const __m256 wr = _mm256_loadu_ps(w + (k - k0) * 16), wi = _mm256_loadu_ps(w + (k - k0) * 16 + 8);
        for (int r = 0; r < MR; ++r) {
            const __m256 a = _mm256_broadcast_ss(sig + (f0 + r) * hop + k);
            ar[r] = _mm256_fmadd_ps(a, wr, ar[r]);
            ai[r] = _mm256_fmadd_ps(a, wi, ai[r]);
        }
    }
    for (int r = 0; r < MR; ++r) {
        _mm256_storeu_ps(re + 8 * r, ar[r]);
        _mm256_storeu_ps(im + 8 * r, ai[r]);
    }
#else
    for (int r = 0; r < MR; ++r)
        for (int l = 0; l < 8; ++l) {
            float sr = 0, si = 0;
            for (int k = k0; k < k1; ++k) {
                const float a = sig[(f0 + r) * hop + k];
                sr = fmadd(a, w[(k - k0) * 16 + l], sr);
                si = fmadd(a, w[(k - k0) * 16 + 8 + l], si);
            }
            re[8 * r + l] = sr;
            im[8 * r + l] = si;
        }
#endif
}

}  // namespace

void WindowModel::cqt(const float* window, Workspace& ws) const {
    const float* x = window;
    int n = kWindowSamples;
    for (int oct = 0; oct < kOctaves; ++oct) {
        float* sig = ws.sig[oct].data();
        reflect_pad(x, n, sig);
        const int hop = kFftHop >> oct;
        const int bin0 = kCqtBins - kBinsPerOctave * (oct + 1);  // global index of this octave's bin 0
        for (int g = 0; g < 5; ++g) {
            const CqtGroup& grp = cqt_groups_[g];
            const int b_lo = 8 * g, b_hi = std::min(8 * g + 8, kBinsPerOctave);
            if (bin0 + b_hi <= 0) continue;  // whole group below the lowest kept bin
            alignas(32) float re[6 * 8], im[6 * 8];
            for (int f0 = 0; f0 < kFramesPerWindow;) {
                const int mr = kFramesPerWindow - f0 >= 6 ? 6 : 4;
                if (mr == 6) cqt_block<6>(sig, hop, f0, grp.w.data(), grp.k0, grp.k1, re, im);
                else cqt_block<4>(sig, hop, f0, grp.w.data(), grp.k0, grp.k1, re, im);
                for (int r = 0; r < mr; ++r)
                    for (int b = b_lo; b < b_hi; ++b) {
                        const int gb = bin0 + b;
                        if (gb < 0) continue;
                        ws.cqt_re[(f0 + r) * kCqtBins + gb] = re[8 * r + b - b_lo];
                        ws.cqt_im[(f0 + r) * kCqtBins + gb] = im[8 * r + b - b_lo];
                    }
                f0 += mr;
            }
        }
        if (oct + 1 < kOctaves) {
            n = decimate(sig + kReflectPad, n, ws);
            x = ws.down.data();
        }
    }
}

void WindowModel::normalize_and_stack(Workspace& ws) const {
    constexpr int total = kFramesPerWindow * kCqtBins;  // 53148 = 8 * 6643 + 4
    float* lp = ws.logp.data();
    const float* re = ws.cqt_re.data();
    const float* im = ws.cqt_im.data();
    // Magnitude, power and dB exactly as the TF graph computes them. Eigen evaluates the flat
    // tensor in 8-wide packets and the final 4 elements with the C library's logf.
    int i = 0;
#if BPP_AVX2
    const __m256 eps = _mm256_set1_ps(1e-10f), db = _mm256_set1_ps(kDbPerNeper);
    for (; i + 8 <= total; i += 8) {
        const int b = i % kCqtBins;
        alignas(32) float sl[8];
        for (int l = 0; l < 8; ++l) sl[l] = data::kCqtSqrtLengths[(b + l) % kCqtBins];
        const __m256 s = _mm256_load_ps(sl);
        const __m256 r = _mm256_mul_ps(_mm256_loadu_ps(re + i), s);
        const __m256 q = _mm256_mul_ps(_mm256_loadu_ps(im + i), s);
        const __m256 mag = _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(r, r), _mm256_mul_ps(q, q)));
        const __m256 pw = _mm256_add_ps(_mm256_mul_ps(mag, mag), eps);
        _mm256_storeu_ps(lp + i, _mm256_mul_ps(eigen::log_ps(pw), db));
    }
#else
    for (; i < total - total % 8; ++i) {
        const float s = data::kCqtSqrtLengths[i % kCqtBins];
        const float r = re[i] * s, q = im[i] * s;
        const float mag = std::sqrt(r * r + q * q);
        lp[i] = eigen::log(mag * mag + 1e-10f) * kDbPerNeper;
    }
#endif
    for (; i < total; ++i) {
        const float s = data::kCqtSqrtLengths[i % kCqtBins];
        const float r = re[i] * s, q = im[i] * s;
        const float mag = std::sqrt(r * r + q * q);
        const float pw = mag * mag + 1e-10f;
        lp[i] = static_cast<float>(std::log(static_cast<double>(pw))) * kDbPerNeper;
    }

    float mn = lp[0];
    for (int k = 1; k < total; ++k) mn = std::min(mn, lp[k]);
    float mx = lp[0] - mn;
    for (int k = 0; k < total; ++k) {
        lp[k] -= mn;
        mx = std::max(mx, lp[k]);
    }
    // div_no_nan, then the CQT batch norm
    const float mean = data::kBnCqt_mean[0], beta = data::kBnCqt_beta[0];
    for (int k = 0; k < total; ++k) {
        const float v = mx != 0.0f ? lp[k] / mx : 0.0f;
        lp[k] = (v - mean) * bn_cqt_scale_ + beta;
    }
    // harmonic stacking into the padded NHWC buffer
    for (int t = kHsBegin; t < kHsEnd; ++t) {
        const float* src = lp + t * kCqtBins;
        float* dst = ws.hs.data() + t * kHsRow + kHsPad * kHarmonics;
        for (int f = 0; f < kContourBins; ++f)
            for (int h = 0; h < kHarmonics; ++h) {
                const int b = f + kHarmonicShift[h];
                dst[f * kHarmonics + h] = (b >= 0 && b < kCqtBins) ? src[b] : 0.0f;
            }
    }
}

void WindowModel::run(const float* window, Workspace& ws, float* contour, float* note, float* onset,
                      const RowRunner* runner) const {
    using namespace data;
    {
        BPP_STAGE(0);
        cqt(window, ws);
    }
    {
        BPP_STAGE(1);
        normalize_and_stack(ws);
    }
    {
        // conv2d_1 (3x39, 8 -> 8) + BN + ReLU
        BPP_STAGE(2);
        run_rows(runner, kC1Begin, kC1End,
                 [&](int b, int e) { conv_contour1(ws.hs.data(), c1_inv_std_.data(), b, e, ws.c1.data()); });
    }
    {
        // contours-reduced (5x5, 8 -> 1) + sigmoid -> Yp
        BPP_STAGE(3);
        const Out1Conv p{ws.c1.data(), kC1Ch, kC1Row, 8, kC1Pad - 2, 5, 5, -2, kConvContour2_w, kConvContour2_b[0]};
        run_rows(runner, kYpBegin, kYpEnd,
                 [&](int b, int e) { conv_out1(p, b, e, kContourBins, ws.yp.data() + kYpPad, kYpRow, 1, 0); });
    }
    {
        // conv2d_2 (7x7 /3, 1 -> 32) + ReLU, and conv2d_4 (5x5 /3, 8 -> 32) + BN + ReLU
        BPP_STAGE(4);
        run_rows(runner, kN1Begin, kN1End, [&](int b, int e) {
            conv_out32_s3<true, false>(ws.yp.data(), kYpRow, 1, kYpPad - 2, kConvNote1_w, 7, 7, -3, b, e, kConvNote1_b,
                                       nullptr, ws.n1.data() + kN1Pad, kN1Ch, kN1Row);
            const int ob = std::max(b, kCatBegin), oe = std::min(e, kCatEnd);
            if (ob < oe)
                conv_out32_s3<false, true>(ws.hs.data(), kHsRow, 8, kHsPad - 1, kConvOnset1_w, 5, 5 * 8, -2, ob, oe,
                                           kConvOnset1_b, o1_inv_std_.data(), ws.cat.data() + kCatCh + kCatPad, kCatCh,
                                           kCatRow);
        });
    }
    {
        // conv2d_3 (7x3, 32 -> 1) + sigmoid -> Yn, stored as channel 0 of the concat tensor
        BPP_STAGE(5);
        const Out1Conv p{ws.n1.data(), kN1Ch, kN1Row, 32, kN1Pad - 1, 7, 3, -3, kConvNote2_w, kConvNote2_b[0]};
        run_rows(runner, kCatBegin, kCatEnd,
                 [&](int b, int e) { conv_out1(p, b, e, kNoteBins, ws.cat.data() + kCatPad, kCatRow, 1, 0); });
    }
    {
        // conv2d_5 (3x3, 33 -> 1) + sigmoid -> Yo
        BPP_STAGE(6);
        const Out1Conv p{ws.cat.data(), kCatCh, kCatRow, 33, kCatPad - 1, 3, 3, -1, kConvOnset2_w, kConvOnset2_b[0]};
        run_rows(runner, kOutBegin, kOutEnd,
                 [&](int b, int e) { conv_out1(p, b, e, kNoteBins, onset, kNoteBins, 1, kOutBegin); });
    }
    BPP_STAGE(7);
    for (int t = kOutBegin; t < kOutEnd; ++t) {
        std::memcpy(contour + (t - kOutBegin) * kContourBins, ws.yp.data() + t * kYpRow + kYpPad,
                    sizeof(float) * kContourBins);
        std::memcpy(note + (t - kOutBegin) * kNoteBins, ws.cat.data() + t * kCatRow + kCatPad,
                    sizeof(float) * kNoteBins);
    }
}

}  // namespace bpp::detail
