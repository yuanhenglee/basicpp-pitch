// Command-line front end.
//
//   bpp transcribe <audio.wav> [-o out.mid] [--csv notes.csv] [--dump DIR] [--threads N] [note options]
//   bpp bench      <audio.wav> [--threads N] [--repeat R]
//   bpp stream     <audio.wav> [--chunk-ms MS] [--threads N] [--realtime] [-o out.mid] [note options]
//
// Note options mirror basic-pitch's CLI: --onset-threshold, --frame-threshold, --minimum-note-length (ms),
// --minimum-frequency, --maximum-frequency (Hz), --multiple-pitch-bends, --no-melodia, --midi-tempo.
#include "bpp/bpp.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

struct Args {
    std::string command, input, midi, csv, dump;
    int threads = 0;
    int repeat = 5;
    double chunk_ms = 20.0;
    bool realtime = false;
    bpp::NoteOptions notes;
};

[[noreturn]] void usage() {
    std::fprintf(stderr,
                 "usage:\n"
                 "  bpp transcribe <audio.wav> [-o out.mid] [--csv notes.csv] [--dump DIR] [--threads N] [note options]\n"
                 "  bpp bench      <audio.wav> [--threads N] [--repeat R]\n"
                 "  bpp stream     <audio.wav> [--chunk-ms MS] [--threads N] [--realtime] [-o out.mid] [note options]\n"
                 "note options (defaults as basic-pitch):\n"
                 "  --onset-threshold 0.5  --frame-threshold 0.3  --minimum-note-length 127.70 (ms)\n"
                 "  --minimum-frequency HZ  --maximum-frequency HZ  --multiple-pitch-bends  --no-melodia\n"
                 "  --midi-tempo 120\n");
    std::exit(2);
}

Args parse(int argc, char** argv) {
    if (argc < 3) usage();
    Args a;
    a.command = argv[1];
    a.input = argv[2];
    for (int i = 3; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) usage();
            return argv[++i];
        };
        if (k == "-o") a.midi = next();
        else if (k == "--csv") a.csv = next();
        else if (k == "--dump") a.dump = next();
        else if (k == "--threads") a.threads = std::stoi(next());
        else if (k == "--repeat") a.repeat = std::stoi(next());
        else if (k == "--chunk-ms") a.chunk_ms = std::stod(next());
        else if (k == "--realtime") a.realtime = true;
        else if (k == "--onset-threshold") a.notes.onset_threshold = std::stod(next());
        else if (k == "--frame-threshold") a.notes.frame_threshold = std::stod(next());
        else if (k == "--minimum-note-length") a.notes.minimum_note_length_ms = std::stod(next());
        else if (k == "--minimum-frequency") a.notes.minimum_frequency = std::stod(next());
        else if (k == "--maximum-frequency") a.notes.maximum_frequency = std::stod(next());
        else if (k == "--multiple-pitch-bends") a.notes.multiple_pitch_bends = true;
        else if (k == "--no-melodia") a.notes.melodia_trick = false;
        else if (k == "--midi-tempo") a.notes.midi_tempo = std::stod(next());
        else usage();
    }
    if (a.threads < 0 || a.repeat < 1 || !(a.chunk_ms > 0) || !(a.chunk_ms < 1e7)) {
        std::fprintf(stderr, "error: --threads must be >= 0, --repeat >= 1 and --chunk-ms > 0\n");
        std::exit(2);
    }
    return a;
}

void write_file(const std::string& path, const void* data, size_t n) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path);
    f.write(static_cast<const char*>(data), static_cast<std::streamsize>(n));
}

void write_npy(const std::string& path, const std::vector<float>& v, int rows, int cols) {
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + std::to_string(rows) + ", " +
                         std::to_string(cols) + "), }";
    while ((10 + header.size() + 1) % 64) header += ' ';
    header += '\n';
    std::string out = "\x93NUMPY\x01";
    out += '\0';
    out += static_cast<char>(header.size() & 0xff);
    out += static_cast<char>(header.size() >> 8);
    out += header;
    out.append(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(float));
    write_file(path, out.data(), out.size());
}

void write_notes_json(const std::string& path, const std::vector<bpp::NoteEvent>& notes) {
    std::string s = "[";
    char buf[96];
    for (size_t i = 0; i < notes.size(); ++i) {
        const auto& n = notes[i];
        std::snprintf(buf, sizeof buf, "%s[%.17g, %.17g, %d, %.17g, ", i ? ", " : "", n.start_s, n.end_s, n.pitch,
                      static_cast<double>(n.amplitude));
        s += buf;
        if (n.pitch_bends.empty()) {
            s += "null]";
        } else {
            s += "[";
            for (size_t j = 0; j < n.pitch_bends.size(); ++j) s += (j ? ", " : "") + std::to_string(n.pitch_bends[j]);
            s += "]]";
        }
    }
    s += "]";
    write_file(path, s.data(), s.size());
}

// Python's repr(float): shortest round-trip digits, scientific notation outside 1e-4 <= |x| < 1e16.
std::string py_repr(double v) {
    if (v == 0) return std::signbit(v) ? "-0.0" : "0.0";
    char sci[64];
    const auto end = std::to_chars(sci, sci + sizeof sci, v, std::chars_format::scientific).ptr;
    const std::string str(sci, end);  // [-]d[.ddd]e[+-]XX
    const size_t epos = str.find('e');
    const int exp = std::stoi(str.substr(epos + 1));
    const bool neg = str[0] == '-';
    std::string digits;
    for (size_t i = neg; i < epos; ++i)
        if (str[i] != '.') digits += str[i];
    std::string out = neg ? "-" : "";
    if (exp < -4 || exp >= 16) {
        out += digits.substr(0, 1);
        if (digits.size() > 1) out += "." + digits.substr(1);
        char e[8];
        std::snprintf(e, sizeof e, "e%c%02d", exp < 0 ? '-' : '+', std::abs(exp));
        return out + e;
    }
    if (exp < 0) return out + "0." + std::string(-exp - 1, '0') + digits;
    if (digits.size() <= size_t(exp) + 1) return out + digits + std::string(exp + 1 - digits.size(), '0') + ".0";
    return out + digits.substr(0, exp + 1) + "." + digits.substr(exp + 1);
}

// basic-pitch's save_note_events(): csv.writer rows with "\r\n" line endings.
void write_csv(const std::string& path, const std::vector<bpp::NoteEvent>& notes) {
    std::string s = "start_time_s,end_time_s,pitch_midi,velocity,pitch_bend\r\n";
    for (const auto& n : notes) {
        s += py_repr(n.start_s) + "," + py_repr(n.end_s) + "," + std::to_string(n.pitch) + "," +
             std::to_string(static_cast<int>(std::nearbyint(127.0 * double(n.amplitude))));
        for (int b : n.pitch_bends) s += "," + std::to_string(b);
        s += "\r\n";
    }
    write_file(path, s.data(), s.size());
}

void emit(const Args& a, const bpp::ModelOutput& out, const std::vector<bpp::NoteEvent>& notes) {
    const bpp::NoteOptions& opt = a.notes;
    if (!a.midi.empty()) {
        const auto midi = bpp::notes_to_midi(notes, opt);
        write_file(a.midi, midi.data(), midi.size());
    }
    if (!a.csv.empty()) write_csv(a.csv, notes);
    if (!a.dump.empty()) {
        write_npy(a.dump + "/note.npy", out.note, out.n_frames, bpp::kNoteBinCount);
        write_npy(a.dump + "/onset.npy", out.onset, out.n_frames, bpp::kNoteBinCount);
        write_npy(a.dump + "/contour.npy", out.contour, out.n_frames, bpp::kContourBinCount);
        write_notes_json(a.dump + "/notes.json", notes);
        const auto midi = bpp::notes_to_midi(notes, opt);
        write_file(a.dump + "/out.mid", midi.data(), midi.size());
    }
}

int cmd_transcribe(const Args& a) {
    const auto t0 = Clock::now();
    bpp::Engine engine(a.threads);
    const auto out = engine.infer_file(a.input);  // decoding + resampling overlap with inference
    const double t_infer = seconds_since(t0);
    const auto t1 = Clock::now();
    const auto notes = bpp::output_to_notes(out, a.notes);
    const double t_notes = seconds_since(t1);
    emit(a, out, notes);
    const double dur = out.n_frames * 256.0 / bpp::kModelSampleRate;
    std::printf("%zu notes from %.2f s of audio | decode+model %.1f ms, notes %.1f ms, %d threads | %.0fx real time\n",
                notes.size(), dur, t_infer * 1e3, t_notes * 1e3, engine.threads(), dur / (t_infer + t_notes));
    return 0;
}

int cmd_bench(const Args& a) {
    const auto audio = bpp::load_audio(a.input);
    const double dur = double(audio.size()) / bpp::kModelSampleRate;
    bpp::Engine engine(a.threads);
    engine.infer(audio.data(), std::min<size_t>(audio.size(), 100000));  // warm-up
    std::vector<double> times;
    size_t n_notes = 0;
    for (int r = 0; r < a.repeat; ++r) {
        const auto t0 = Clock::now();
        const auto out = engine.infer(audio.data(), audio.size());
        const auto notes = bpp::output_to_notes(out);
        times.push_back(seconds_since(t0));
        n_notes = notes.size();
    }
    std::sort(times.begin(), times.end());
    const double best = times.front(), median = times[times.size() / 2];
    std::printf("audio %.2f s, %d threads, %zu notes | best %.1f ms (%.0fx real time), median %.1f ms\n", dur,
                engine.threads(), n_notes, best * 1e3, dur / best, median * 1e3);
    return 0;
}

int cmd_stream(const Args& a) {
    const auto audio = bpp::load_audio(a.input);
    const size_t chunk = std::max<size_t>(1, static_cast<size_t>(a.chunk_ms * 1e-3 * bpp::kModelSampleRate));
    bpp::Stream stream(std::max(1, a.threads));
    std::vector<double> lat;  // compute time of pushes that ran a window
    const auto t_start = Clock::now();
    for (size_t pos = 0; pos < audio.size(); pos += chunk) {
        const size_t n = std::min(chunk, audio.size() - pos);
        if (a.realtime) {
            const auto due = t_start + std::chrono::duration<double>(double(pos + n) / bpp::kModelSampleRate);
            std::this_thread::sleep_until(due);
        }
        const int windows_before = stream.windows_processed();
        const auto t0 = Clock::now();
        stream.push(audio.data() + pos, n);
        if (stream.windows_processed() > windows_before) lat.push_back(seconds_since(t0));
    }
    const auto t0 = Clock::now();
    stream.finish();
    const double t_finish = seconds_since(t0);
    const auto t1 = Clock::now();
    const auto notes = bpp::output_to_notes(stream.output(), a.notes);
    const double t_notes = seconds_since(t1);
    const double total = seconds_since(t_start);
    std::sort(lat.begin(), lat.end());
    const double dur = double(audio.size()) / bpp::kModelSampleRate;
    std::printf("streamed %.2f s in %.0f ms chunks: %zu window pushes, per-window compute median %.2f ms, max %.2f ms "
                "(budget %.0f ms); finish %.1f ms, notes %.1f ms, wall %.2f s, %zu notes\n",
                dur, a.chunk_ms, lat.size(), lat.empty() ? 0 : lat[lat.size() / 2] * 1e3,
                lat.empty() ? 0 : lat.back() * 1e3, 36164.0 / bpp::kModelSampleRate * 1e3, t_finish * 1e3,
                t_notes * 1e3, total, notes.size());
    emit(a, stream.output(), notes);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args a = parse(argc, argv);
        if (a.command == "transcribe") return cmd_transcribe(a);
        if (a.command == "bench") return cmd_bench(a);
        if (a.command == "stream") return cmd_stream(a);
        usage();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
