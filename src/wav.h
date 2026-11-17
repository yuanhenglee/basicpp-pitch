// WAV decoding to mono float32 at the native sample rate (librosa / libsndfile semantics).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace bpp::detail {

struct WavAudio {
    std::vector<float> mono;
    uint32_t sample_rate = 0;
};

WavAudio decode_wav(const std::string& path);

}  // namespace bpp::detail
