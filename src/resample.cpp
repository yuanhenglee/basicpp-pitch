// Resampling to 22050 Hz exactly as librosa.load() does by default (res_type="soxr_hq"):
// python-soxr 1.1's split-channel path (csoxr_split_ch) driving the bundled libsoxr 0.1.3 with
// SOXR_HQ, followed by librosa.util.fix_length to ceil(n * ratio) samples.
#include "resample.h"

#include "constants.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#ifdef BPP_HAVE_SOXR
#include <soxr.h>
#endif

namespace bpp::detail {

size_t resampled_length(size_t n, uint32_t sample_rate) {
    if (sample_rate == 0) throw std::invalid_argument("sample rate must be positive");
    const double ratio = static_cast<double>(kSampleRate) / static_cast<double>(sample_rate);
    return static_cast<size_t>(std::ceil(static_cast<double>(n) * ratio));
}

size_t resampler_capacity(size_t n, uint32_t sample_rate) {
    if (sample_rate == 0) throw std::invalid_argument("sample rate must be positive");
    const double in_rate = sample_rate, out_rate = kSampleRate;
    const size_t olen = n * out_rate / in_rate + 1;  // csoxr_split_ch(), verbatim
    return std::max(olen, resampled_length(n, sample_rate));
}

void resample_to_model_rate(const float* mono, size_t ilen, uint32_t sample_rate, float* out,
                            const std::function<void(size_t)>& progress) {
    if (sample_rate == 0) throw std::invalid_argument("sample rate must be positive");
#ifdef BPP_HAVE_SOXR
    const double in_rate = sample_rate, out_rate = kSampleRate;
    const size_t olen = ilen * out_rate / in_rate + 1;
    const size_t div_len = std::max(1000., 48000 * in_rate / out_rate);
    const size_t n_result = resampled_length(ilen, sample_rate);

    soxr_error_t err = nullptr;
    const soxr_io_spec_t io_spec = soxr_io_spec(SOXR_FLOAT32_S, SOXR_FLOAT32_S);
    const soxr_quality_spec_t quality_spec = soxr_quality_spec(SOXR_HQ, 0);
    soxr_t soxr = soxr_create(in_rate, out_rate, 1, &err, &io_spec, &quality_spec, nullptr);
    if (err) throw std::runtime_error(std::string("soxr_create: ") + err);

    size_t out_pos = 0, odone = 0;
    for (size_t idx = 0; idx < ilen && !err; idx += div_len) {
        const float* in_ptr = mono + idx;
        float* out_ptr = out + out_pos;
        err = soxr_process(soxr, &in_ptr, std::min(div_len, ilen - idx), nullptr, &out_ptr, olen - out_pos, &odone);
        out_pos += odone;
        if (progress) progress(std::min(out_pos, n_result));
    }
    if (!err) {
        float* out_ptr = out + out_pos;
        err = soxr_process(soxr, nullptr, 0, nullptr, &out_ptr, olen - out_pos, &odone);
        out_pos += odone;
    }
    soxr_delete(soxr);
    if (err) throw std::runtime_error(std::string("soxr_process: ") + err);
    // fix_length: samples past out_pos stay zero (the buffer starts zeroed); extra ones are ignored
    if (progress) progress(n_result);
#else
    (void)mono, (void)ilen, (void)out, (void)progress;
    throw std::runtime_error("sample rate " + std::to_string(sample_rate) +
                             " Hz needs resampling, but this build has no libsoxr (BPP_WITH_SOXR=OFF)");
#endif
}

std::vector<float> resample_to_model_rate(const std::vector<float>& mono, uint32_t sample_rate) {
    std::vector<float> y(resampler_capacity(mono.size(), sample_rate), 0.0f);
    resample_to_model_rate(mono.data(), mono.size(), sample_rate, y.data(), nullptr);
    y.resize(resampled_length(mono.size(), sample_rate));
    return y;
}

}  // namespace bpp::detail
