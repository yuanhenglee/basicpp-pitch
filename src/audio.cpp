// WAV decoding that matches librosa.load(path, sr=22050, mono=True) sample for sample:
// libsndfile's integer -> float32 scaling, then librosa.to_mono (float32 channel mean).
#include "bpp/bpp.h"

#include "constants.h"
#include "resample.h"
#include "wav.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace bpp {

namespace {

uint32_t le32(const uint8_t* p) { return p[0] | (p[1] << 8) | (p[2] << 16) | (uint32_t(p[3]) << 24); }
uint16_t le16(const uint8_t* p) { return static_cast<uint16_t>(p[0] | (p[1] << 8)); }

}  // namespace

namespace detail {

WavAudio decode_wav(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    const std::streamoff size = f.tellg();
    if (size < 0) throw std::runtime_error("cannot read " + path);
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (f.gcount() != static_cast<std::streamsize>(bytes.size())) throw std::runtime_error("short read on " + path);
    if (bytes.size() < 12 || std::memcmp(bytes.data(), "RIFF", 4) || std::memcmp(bytes.data() + 8, "WAVE", 4))
        throw std::runtime_error(path + ": not a RIFF/WAVE file");

    int format = 0, channels = 0, bits = 0, block_align = 0;
    uint32_t sample_rate = 0;
    const uint8_t* pcm = nullptr;
    size_t pcm_size = 0;
    for (size_t pos = 12; pos + 8 <= bytes.size();) {
        const uint8_t* chunk = bytes.data() + pos;
        const uint32_t size = le32(chunk + 4);
        const size_t avail = std::min<size_t>(size, bytes.size() - pos - 8);
        if (!std::memcmp(chunk, "fmt ", 4) && avail >= 16) {
            format = le16(chunk + 8);
            channels = le16(chunk + 10);
            sample_rate = le32(chunk + 12);
            block_align = le16(chunk + 20);
            bits = le16(chunk + 22);
            if (format == 0xfffe && avail >= 40) format = le16(chunk + 8 + 24);  // WAVE_FORMAT_EXTENSIBLE subformat
        } else if (!std::memcmp(chunk, "data", 4)) {
            pcm = chunk + 8;
            pcm_size = avail;
        }
        pos += 8 + size + (size & 1);
    }
    if (!pcm || channels <= 0 || block_align <= 0) throw std::runtime_error(path + ": missing fmt/data chunk");
    const bool is_float = format == 3;
    if (!(format == 1 || is_float) || (is_float && bits != 32 && bits != 64) ||
        (!is_float && bits != 8 && bits != 16 && bits != 24 && bits != 32))
        throw std::runtime_error(path + ": unsupported WAV encoding (format " + std::to_string(format) + ", " +
                                 std::to_string(bits) + " bit)");

    const int bytes_per_sample = bits / 8;
    if (block_align < channels * bytes_per_sample)
        throw std::runtime_error(path + ": inconsistent WAV header (block align " + std::to_string(block_align) +
                                 " < " + std::to_string(channels) + " channels x " + std::to_string(bits) + " bit)");
    // Up to 1 MHz; lower bound keeps the resampled length reasonable (librosa handles 8 kHz audio and up).
    if (sample_rate < 1000 || sample_rate > 1000000)
        throw std::runtime_error(path + ": unsupported sample rate " + std::to_string(sample_rate) + " Hz");
    const size_t frames = pcm_size / block_align;
    auto sample = [&](size_t frame, int ch) -> float {
        const uint8_t* p = pcm + frame * block_align + ch * bytes_per_sample;
        if (is_float) {
            if (bits == 32) {
                float v;
                std::memcpy(&v, p, 4);
                return v;
            }
            double v;
            std::memcpy(&v, p, 8);
            return static_cast<float>(v);
        }
        switch (bits) {
            case 8: return static_cast<float>(int(p[0]) - 128) * (1.0f / 0x80);
            case 16: return static_cast<float>(static_cast<int16_t>(le16(p))) * (1.0f / 0x8000);
            case 24: {
                const int32_t v = static_cast<int32_t>((uint32_t(p[0]) << 8) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 24));
                return static_cast<float>(v) * (1.0f / 2147483648.0f);
            }
            default: return static_cast<float>(static_cast<int32_t>(le32(p))) * (1.0f / 2147483648.0f);
        }
    };

    std::vector<float> mono(frames);
    mix_to_mono(frames, channels, sample, mono.data());
    return {std::move(mono), sample_rate};
}

void check_pcm_format(int sample_rate, int channels) {
    if (channels < 1 || channels > 65535) throw std::invalid_argument("channel count must be 1..65535");
    if (sample_rate < 1000 || sample_rate > 1000000)
        throw std::invalid_argument("unsupported sample rate " + std::to_string(sample_rate) + " Hz");
}

std::vector<float> interleaved_to_mono(const float* audio, size_t frames, int channels) {
    std::vector<float> mono(frames);
    const size_t stride = static_cast<size_t>(channels);
    mix_to_mono(frames, channels, [&](size_t i, int c) { return audio[i * stride + c]; }, mono.data());
    return mono;
}

}  // namespace detail

std::vector<float> convert_pcm(const float* audio, size_t n_frames, int sample_rate, int channels) {
    detail::check_pcm_format(sample_rate, channels);
    std::vector<float> mono = channels == 1 ? std::vector<float>(audio, audio + n_frames)
                                            : detail::interleaved_to_mono(audio, n_frames, channels);
    if (sample_rate == kSampleRate) return mono;
    return detail::resample_to_model_rate(mono, static_cast<uint32_t>(sample_rate));
}

std::vector<float> load_audio(const std::string& path) {
    detail::WavAudio wav = detail::decode_wav(path);
    if (wav.sample_rate == static_cast<uint32_t>(kSampleRate)) return std::move(wav.mono);
    return detail::resample_to_model_rate(wav.mono, wav.sample_rate);
}

}  // namespace bpp
