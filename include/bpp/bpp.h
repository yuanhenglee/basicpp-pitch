// BasiCPP Pitch: a fast, dependency-free C++ engine for Spotify's basic-pitch
// polyphonic transcription model. Results reproduce the official Python library
// (basic-pitch 0.4.0): identical note events, pitch bends and MIDI bytes.
//
// Input audio is mono float32 PCM at 22050 Hz (bpp::kSampleRate); use
// bpp::load_audio() to decode WAV files the same way basic-pitch does.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace bpp {

inline constexpr int kModelSampleRate = 22050;
inline constexpr int kNoteBinCount = 88;
inline constexpr int kContourBinCount = 264;

// Frame-level model output, row-major [n_frames][bins], 86.13 frames per second.
struct ModelOutput {
    int n_frames = 0;
    std::vector<float> note;     // [n_frames][88]
    std::vector<float> onset;    // [n_frames][88]
    std::vector<float> contour;  // [n_frames][264]
};

// Note decoding parameters; defaults equal basic_pitch.inference.predict().
struct NoteOptions {
    double onset_threshold = 0.5;
    double frame_threshold = 0.3;
    double minimum_note_length_ms = 127.70;
    std::optional<double> minimum_frequency;  // Hz
    std::optional<double> maximum_frequency;  // Hz
    bool infer_onsets = true;
    bool melodia_trick = true;
    bool include_pitch_bends = true;
    bool multiple_pitch_bends = false;
    double midi_tempo = 120.0;
};

struct NoteEvent {
    double start_s = 0;
    double end_s = 0;
    int start_frame = 0;
    int end_frame = 0;
    int pitch = 0;          // MIDI note number
    float amplitude = 0;    // mean note activation in [0, 1]; MIDI velocity = round(127 * amplitude)
    std::vector<int> pitch_bends;  // one per frame, units of 1/3 semitone; empty if none
};

// Runs the neural network over audio. Thread-safe; one instance can serve many callers.
class Engine {
public:
    // n_threads <= 0 uses all hardware threads.
    explicit Engine(int n_threads = 0);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // Whole-signal inference, parallelised over analysis windows.
    ModelOutput infer(const float* audio, size_t n_samples) const;

    // Decodes a WAV file like load_audio() and runs inference, overlapping resampling with the
    // network. Same result as infer() on load_audio(path).
    ModelOutput infer_file(const std::string& path) const;

    int threads() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Incremental inference for live audio. Push samples as they arrive; every completed analysis
// window (one per ~1.64 s of audio) appends 142 frames to output() immediately. Frame values are
// bit-identical to Engine::infer() on the complete signal. basic-pitch trims the end of the frame
// sequence to floor(n_samples * 86 / 22050) frames, so the newest frames are only guaranteed to
// survive once enough audio follows them: final_frames() is that guaranteed prefix, and finish()
// applies the trim.
class Stream {
public:
    // n_threads > 1 lets a single window be split across threads to cut latency (clamped to 64).
    explicit Stream(int n_threads = 1);
    ~Stream();
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    // Appends audio and runs every analysis window that became complete.
    // Returns the number of frames appended to output(). Throws std::logic_error after finish().
    int push(const float* audio, size_t n_samples);

    // Flushes the final (zero-padded) windows and trims the output to basic-pitch's length.
    // Returns the net change in output().n_frames (may be negative).
    int finish();

    // Frames computed so far. After finish() this equals Engine::infer() of all audio pushed.
    const ModelOutput& output() const;

    // Leading frames of output() that are final no matter when the stream ends.
    int final_frames() const;

    size_t samples_pushed() const;

    // Number of analysis windows run so far.
    int windows_processed() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Decodes model output into note events (basic_pitch.note_creation.model_output_to_notes).
std::vector<NoteEvent> output_to_notes(const ModelOutput& output, const NoteOptions& options = {});

// Serialises notes to a Standard MIDI File, byte-identical to basic-pitch's pretty_midi output.
std::vector<uint8_t> notes_to_midi(const std::vector<NoteEvent>& notes, const NoteOptions& options = {});

// Decodes a WAV file to 22050 Hz mono float32 the way librosa.load(sr=22050, mono=True) does.
// Throws std::runtime_error on unsupported input.
std::vector<float> load_audio(const std::string& path);

}  // namespace bpp
