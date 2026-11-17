// WAV decoding to mono float32 at the native sample rate (librosa / libsndfile semantics).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace bpp::detail {

struct WavAudio {
    std::vector<float> mono;
    uint32_t sample_rate = 0;
};

WavAudio decode_wav(const std::string& path);

// librosa.to_mono: np.mean(y, axis=0) in float32, i.e. a sequential sum over channels divided by
// the count. sample(i, c) returns channel c of frame i.
template <typename Sample>
void mix_to_mono(size_t frames, int channels, Sample&& sample, float* out) {
    if (channels == 1) {
        for (size_t i = 0; i < frames; ++i) out[i] = sample(i, 0);
        return;
    }
    for (size_t i = 0; i < frames; ++i) {
        float s = sample(i, 0);
        for (int c = 1; c < channels; ++c) s += sample(i, c);
        out[i] = s / static_cast<float>(channels);
    }
}

// Interleaved float32 PCM to mono; rejects channel counts and sample rates decode_wav() rejects.
std::vector<float> interleaved_to_mono(const float* audio, size_t frames, int channels);
void check_pcm_format(int sample_rate, int channels);

}  // namespace bpp::detail
