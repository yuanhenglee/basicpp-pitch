// Self-contained regression test: transcribes tests/fixtures/<name>.wav (44.1 kHz stereo) and requires
// bit-identical notes and MIDI bytes to basic-pitch's golden output, offline and streaming, from the
// decoded 22050 Hz signal and from the raw 44.1 kHz stereo PCM (Engine::infer_pcm, resampling Stream).
//   test_fixture <fixture_dir> <name>
#include "bpp/bpp.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    failures += !ok;
}

std::vector<uint8_t> read_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

struct GoldenNote {
    double start, end;
    int pitch;
    float amplitude;
    std::vector<int> bends;
};

std::vector<GoldenNote> read_notes(const std::string& path) {
    std::vector<GoldenNote> notes;
    std::ifstream f(path);
    for (std::string line; std::getline(f, line);) {
        std::istringstream in(line);
        std::string s, e, a;
        GoldenNote n;
        in >> s >> e >> n.pitch >> a;
        n.start = std::strtod(s.c_str(), nullptr);
        n.end = std::strtod(e.c_str(), nullptr);
        n.amplitude = static_cast<float>(std::strtod(a.c_str(), nullptr));
        for (int b; in >> b;) n.bends.push_back(b);
        notes.push_back(std::move(n));
    }
    return notes;
}

// Interleaved float PCM of a 16-bit WAV file, scaled like libsndfile (x / 32768).
struct Pcm {
    std::vector<float> samples;
    int sample_rate = 0, channels = 0;
};

Pcm read_pcm16(const std::string& path) {
    const auto b = read_bytes(path);
    auto u16 = [&](size_t i) { return uint16_t(b[i] | (b[i + 1] << 8)); };
    auto u32 = [&](size_t i) { return uint32_t(u16(i)) | (uint32_t(u16(i + 2)) << 16); };
    Pcm pcm;
    for (size_t pos = 12; pos + 8 <= b.size(); pos += 8 + u32(pos + 4) + (u32(pos + 4) & 1)) {
        const uint32_t size = u32(pos + 4);
        if (!std::memcmp(&b[pos], "fmt ", 4)) {
            if (u16(pos + 8) != 1 || u16(pos + 22) != 16) return {};
            pcm.channels = u16(pos + 10);
            pcm.sample_rate = static_cast<int>(u32(pos + 12));
        } else if (!std::memcmp(&b[pos], "data", 4)) {
            const size_t n = std::min<size_t>(size, b.size() - pos - 8) / 2;
            for (size_t i = 0; i < n; ++i) pcm.samples.push_back(int16_t(u16(pos + 8 + 2 * i)) * (1.0f / 32768));
        }
    }
    return pcm;
}

bool same_output(const bpp::ModelOutput& a, const bpp::ModelOutput& b) {
    return a.n_frames == b.n_frames && a.note == b.note && a.onset == b.onset && a.contour == b.contour;
}

bool same_notes(const std::vector<bpp::NoteEvent>& got, const std::vector<GoldenNote>& ref, float amplitude_tol = 0) {
    if (got.size() != ref.size()) return false;
    for (size_t i = 0; i < got.size(); ++i) {
        const bpp::NoteEvent& g = got[i];
        const GoldenNote& r = ref[i];
        if (g.start_s != r.start || g.end_s != r.end || g.pitch != r.pitch ||
            !(std::fabs(g.amplitude - r.amplitude) <= amplitude_tol) || g.pitch_bends != r.bends)
            return false;
    }
    return true;
}

// The golden output comes from basic-pitch on Windows, where libsoxr designs its resampling filter
// with the Universal CRT's libm. Other C libraries (musl in WebAssembly, glibc) round a few of those
// math calls differently, which moves resampled samples by about one ulp: note times, pitches,
// bends and MIDI bytes stay identical, amplitudes move by ~1e-5.
#if defined(_WIN32)
constexpr float kResampledAmplitudeTol = 0;
#else
constexpr float kResampledAmplitudeTol = 1e-4f;
#endif

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: test_fixture <fixture_dir> <name>\n");
        return 2;
    }
    const std::string base = std::string(argv[1]) + "/" + argv[2];
    const auto golden_midi = read_bytes(base + ".mid");
    const auto golden_notes = read_notes(base + ".notes");
    check(!golden_midi.empty() && !golden_notes.empty(), "fixture loaded");

    const auto audio = bpp::load_audio(base + ".wav");
    bpp::Engine engine(0);
    const auto out = engine.infer(audio.data(), audio.size());
    const auto notes = bpp::output_to_notes(out);
    check(same_notes(notes, golden_notes, kResampledAmplitudeTol),
          "offline note events " + std::string(kResampledAmplitudeTol ? "identical, amplitudes within 1e-4" : "bit-identical") +
              " (" + std::to_string(notes.size()) + ")");
    check(bpp::notes_to_midi(notes) == golden_midi, "offline MIDI bytes identical");

    for (size_t chunk : {size_t(1) << 9, size_t(30000)}) {
        bpp::Stream stream(chunk > 1000 ? 2 : 1);
        for (size_t pos = 0; pos < audio.size(); pos += chunk)
            stream.push(audio.data() + pos, std::min(chunk, audio.size() - pos));
        stream.finish();
        check(same_output(stream.output(), out), "stream (chunk " + std::to_string(chunk) + ") equals offline model output");
    }

    // Raw PCM at the native rate: mixed to mono and resampled inside the engine.
    const Pcm pcm = read_pcm16(base + ".wav");
    check(pcm.channels > 1 && pcm.sample_rate != bpp::kModelSampleRate, "fixture is multichannel, not 22050 Hz");
    const size_t frames = pcm.samples.size() / std::max(1, pcm.channels);
    check(same_output(engine.infer_pcm(pcm.samples.data(), frames, pcm.sample_rate, pcm.channels), out),
          "infer_pcm (" + std::to_string(pcm.sample_rate) + " Hz, " + std::to_string(pcm.channels) +
              " ch) equals offline model output");
    for (size_t chunk : {size_t(37), size_t(441), size_t(4096), size_t(60000)}) {
        bpp::Stream stream(chunk > 1000 ? 2 : 1, pcm.sample_rate, pcm.channels);
        for (size_t pos = 0; pos < frames; pos += chunk)
            stream.push(pcm.samples.data() + pos * pcm.channels, std::min(chunk, frames - pos));
        stream.finish();
        check(same_output(stream.output(), out),
              "resampling stream (chunk " + std::to_string(chunk) + " frames) equals offline model output");
    }

    std::printf("%s\n", failures ? "FAILED" : "ALL PASSED");
    return failures ? 1 : 0;
}
