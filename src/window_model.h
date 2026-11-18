// One analysis window of the basic-pitch network: CQT -> harmonic stacking -> CNN.
//
// Every operation reproduces the arithmetic TensorFlow 2.15 performs on x86 CPUs for the
// official SavedModel (oneDNN convolution/batch-norm kernels, Eigen element-wise kernels),
// including the order of floating-point accumulation, so outputs match bit for bit.
#pragma once

#include "constants.h"
#include "simd.h"

#include <array>
#include <functional>
#include <vector>

namespace bpp::detail {

// Per-thread scratch memory for one window. Allocate once, reuse for every window.
struct Workspace {
    Workspace();

    std::array<FloatBuffer, kOctaves> sig;  // reflect-padded signal per octave
    FloatBuffer even, odd;                  // deinterleaved signal for decimation
    FloatBuffer down;                       // decimator output
    FloatBuffer cqt_re, cqt_im;             // [172][309]
    FloatBuffer logp;                       // [172][309]
    FloatBuffer hs;                         // harmonic stack, NHWC [172][19+264+19][8]
    FloatBuffer c1;                         // contour hidden, channel-major [8][172][272]
    FloatBuffer yp;                         // contour output, [172][2+264+2]
    FloatBuffer n1;                         // note hidden, channel-major [32][172][96]
    FloatBuffer cat;                        // [note, onset hidden], channel-major [33][172][96]
};

const char* kernel_name();  // see bpp::kernel_name()

// Called with a row range; must run fn over all of it, possibly split across threads.
using RowRunner = std::function<void(int begin, int end, const std::function<void(int, int)>& fn)>;

class WindowModel {
public:
    static const WindowModel& get();

    // Runs the network on kWindowSamples samples. Writes the kUsedFrames frames that
    // basic-pitch keeps (window frames 15..156) to contour[142][264], note[142][88], onset[142][88].
    void run(const float* window, Workspace& ws, float* contour, float* note, float* onset,
             const RowRunner* runner = nullptr) const;

private:
    WindowModel();

    void cqt(const float* window, Workspace& ws) const;
    void normalize_and_stack(Workspace& ws) const;

    // CQT kernel packed per group of 8 bins: [k][re0..re7, im0..im7], k restricted to nonzero support
    struct CqtGroup {
        int k0 = 0, k1 = 0;
        FloatBuffer w;
        std::vector<double> w64;  // w in double for the SIMD128 kernels (fma_f64x2); empty otherwise
    };
    std::array<CqtGroup, 5> cqt_groups_;

    // Weights in double for the SIMD128 kernels; empty in other builds.
    std::vector<double> contour1_w64_, note1_w64_, onset1_w64_, lowpass_w64_;

    float bn_cqt_scale_ = 0;                     // gamma / sqrt(var + eps)
    std::array<float, 8> c1_inv_std_{};          // 1 / sqrt(var + eps)
    std::array<float, 32> o1_inv_std_{};
};

}  // namespace bpp::detail
