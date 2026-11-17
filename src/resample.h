// Sample-rate conversion to the model rate (22050 Hz), identical to librosa.load(sr=22050).
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace bpp::detail {

// Output length of librosa.resample: ceil(n * 22050 / sample_rate).
size_t resampled_length(size_t n, uint32_t sample_rate);

// Capacity the resampler writes into before trimming (python-soxr's output buffer size).
size_t resampler_capacity(size_t n, uint32_t sample_rate);

// Resamples mono[0, n) into out, which must hold resampler_capacity(n, sample_rate) zeroed floats;
// the first resampled_length(n, sample_rate) of them are the result. progress(k) reports that the
// first k result samples are final; it is called with increasing k and finally the full length.
void resample_to_model_rate(const float* mono, size_t n, uint32_t sample_rate, float* out,
                            const std::function<void(size_t)>& progress);

std::vector<float> resample_to_model_rate(const std::vector<float>& mono, uint32_t sample_rate);

}  // namespace bpp::detail
