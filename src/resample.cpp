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

#ifdef BPP_HAVE_SOXR

struct StreamResampler::Impl {
    soxr_t soxr = nullptr;
    uint32_t sample_rate = 0;
    size_t n_in = 0;      // input samples consumed
    size_t n_out = 0;     // result samples handed out
    std::vector<float> pending;  // resampler output not yet handed out

    // Feeds in[0, n) (all of it, like the offline path: no idone) or, if in == nullptr, flushes; then
    // collects output until the resampler has none left.
    void drain(const float* in, size_t n) {
        const float dummy = 0;
        const float* in_ptr = in;
        size_t ilen = n;
        for (;;) {
            const size_t cap = static_cast<size_t>(double(ilen) * kSampleRate / sample_rate) + 4096;
            const size_t old = pending.size();
            pending.resize(old + cap);
            float* out_ptr = pending.data() + old;
            size_t odone = 0;
            const soxr_error_t err = soxr_process(soxr, in_ptr ? &in_ptr : nullptr, ilen, nullptr, &out_ptr, cap, &odone);
            pending.resize(old + odone);
            if (err) throw std::runtime_error(std::string("soxr_process: ") + err);
            if (odone < cap) return;
            if (in_ptr) in_ptr = &dummy, ilen = 0;  // more output pending: drain without input or flushing
        }
    }

    // Hands out pending samples, never more than the final result length allows so far.
    void emit(size_t limit, std::vector<float>& out) {
        const size_t k = std::min(pending.size(), limit - std::min(limit, n_out));
        out.insert(out.end(), pending.begin(), pending.begin() + static_cast<std::ptrdiff_t>(k));
        pending.erase(pending.begin(), pending.begin() + static_cast<std::ptrdiff_t>(k));
        n_out += k;
    }
};

StreamResampler::StreamResampler(uint32_t sample_rate) : impl_(std::make_unique<Impl>()) {
    if (sample_rate == 0) throw std::invalid_argument("sample rate must be positive");
    impl_->sample_rate = sample_rate;
    soxr_error_t err = nullptr;
    const soxr_io_spec_t io_spec = soxr_io_spec(SOXR_FLOAT32_S, SOXR_FLOAT32_S);
    const soxr_quality_spec_t quality_spec = soxr_quality_spec(SOXR_HQ, 0);
    impl_->soxr = soxr_create(sample_rate, kSampleRate, 1, &err, &io_spec, &quality_spec, nullptr);
    if (err) throw std::runtime_error(std::string("soxr_create: ") + err);
}

StreamResampler::~StreamResampler() { soxr_delete(impl_->soxr); }

void StreamResampler::push(const float* mono, size_t n, std::vector<float>& out) {
    if (n == 0) return;
    impl_->drain(mono, n);
    impl_->n_in += n;
    // the result is finally trimmed to resampled_length(total input), which only grows
    impl_->emit(resampled_length(impl_->n_in, impl_->sample_rate), out);
}

void StreamResampler::finish(std::vector<float>& out) {
    Impl& s = *impl_;
    if (s.n_in > 0) s.drain(nullptr, 0);
    const size_t n_result = resampled_length(s.n_in, s.sample_rate);
    s.emit(n_result, out);
    out.resize(out.size() + (n_result - s.n_out), 0.0f);
    s.n_out = n_result;
    s.pending.clear();
}

#else

struct StreamResampler::Impl {};

StreamResampler::StreamResampler(uint32_t sample_rate) {
    throw std::runtime_error("sample rate " + std::to_string(sample_rate) +
                             " Hz needs resampling, but this build has no libsoxr (BPP_WITH_SOXR=OFF)");
}

StreamResampler::~StreamResampler() = default;
void StreamResampler::push(const float*, size_t, std::vector<float>&) {}
void StreamResampler::finish(std::vector<float>&) {}

#endif

}  // namespace bpp::detail
