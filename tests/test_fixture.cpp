// Self-contained regression test: transcribes tests/fixtures/<name>.wav (44.1 kHz stereo) and requires
// bit-identical notes and MIDI bytes to basic-pitch's golden output, offline and streaming.
//   test_fixture <fixture_dir> <name>
#include "bpp/bpp.h"

#include <cstdio>
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

bool same_notes(const std::vector<bpp::NoteEvent>& got, const std::vector<GoldenNote>& ref) {
    if (got.size() != ref.size()) return false;
    for (size_t i = 0; i < got.size(); ++i) {
        const bpp::NoteEvent& g = got[i];
        const GoldenNote& r = ref[i];
        if (g.start_s != r.start || g.end_s != r.end || g.pitch != r.pitch || g.amplitude != r.amplitude ||
            g.pitch_bends != r.bends)
            return false;
    }
    return true;
}

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
    check(same_notes(notes, golden_notes), "offline note events bit-identical (" + std::to_string(notes.size()) + ")");
    check(bpp::notes_to_midi(notes) == golden_midi, "offline MIDI bytes identical");

    for (size_t chunk : {size_t(1) << 9, size_t(30000)}) {
        bpp::Stream stream(chunk > 1000 ? 2 : 1);
        for (size_t pos = 0; pos < audio.size(); pos += chunk)
            stream.push(audio.data() + pos, std::min(chunk, audio.size() - pos));
        stream.finish();
        const auto& so = stream.output();
        check(so.n_frames == out.n_frames && so.note == out.note && so.onset == out.onset && so.contour == out.contour,
              "stream (chunk " + std::to_string(chunk) + ") equals offline model output");
    }

    std::printf("%s\n", failures ? "FAILED" : "ALL PASSED");
    return failures ? 1 : 0;
}
