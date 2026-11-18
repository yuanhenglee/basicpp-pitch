#include "bpp/bpp_c.h"

#include "bpp/bpp.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>

// js/index.js writes bpp_note_options through these byte offsets.
static_assert(offsetof(bpp_note_options, maximum_frequency) == 32 && offsetof(bpp_note_options, infer_onsets) == 40 &&
                  offsetof(bpp_note_options, multiple_pitch_bends) == 52 && offsetof(bpp_note_options, midi_tempo) == 56,
              "update js/index.js (Wasm.withOptions) when bpp_note_options changes");

struct bpp_engine {
    bpp::Engine engine;
    explicit bpp_engine(int n) : engine(n) {}
};
struct bpp_stream {
    bpp::Stream stream;
    bpp_stream(int n, int sample_rate, int channels) : stream(n, sample_rate, channels) {}
};
struct bpp_notes {
    std::vector<bpp::NoteEvent> notes;
};
struct bpp_output {
    bpp::ModelOutput out;
};

namespace {

thread_local std::string g_error;

template <typename F>
auto guard(F&& f, decltype(f()) on_error) -> decltype(f()) {
    try {
        g_error.clear();
        return f();
    } catch (const std::exception& e) {
        g_error = e.what();
    } catch (...) {
        g_error = "unknown error";
    }
    return on_error;
}

bpp::NoteOptions to_cpp(const bpp_note_options* o) {
    bpp::NoteOptions r;
    if (!o) return r;
    r.onset_threshold = o->onset_threshold;
    r.frame_threshold = o->frame_threshold;
    r.minimum_note_length_ms = o->minimum_note_length_ms;
    if (o->minimum_frequency > 0) r.minimum_frequency = o->minimum_frequency;
    if (o->maximum_frequency > 0) r.maximum_frequency = o->maximum_frequency;
    r.infer_onsets = o->infer_onsets != 0;
    r.melodia_trick = o->melodia_trick != 0;
    r.include_pitch_bends = o->include_pitch_bends != 0;
    r.multiple_pitch_bends = o->multiple_pitch_bends != 0;
    r.midi_tempo = o->midi_tempo;
    return r;
}

constexpr size_t kNote = bpp::kNoteBinCount, kContour = bpp::kContourBinCount;

void copy_frames(const bpp::ModelOutput& out, int first, int count, float* note, float* onset, float* contour) {
    std::memcpy(note, out.note.data() + size_t(first) * kNote, sizeof(float) * size_t(count) * kNote);
    std::memcpy(onset, out.onset.data() + size_t(first) * kNote, sizeof(float) * size_t(count) * kNote);
    std::memcpy(contour, out.contour.data() + size_t(first) * kContour, sizeof(float) * size_t(count) * kContour);
}

// Hands audio to C callers in malloc'ed memory (freed with bpp_free_audio).
void export_audio(const std::vector<float>& audio, float** samples, size_t* n) {
    *n = audio.size();
    *samples = static_cast<float*>(std::malloc(sizeof(float) * std::max<size_t>(1, audio.size())));
    if (!*samples) throw std::bad_alloc();
    std::memcpy(*samples, audio.data(), sizeof(float) * audio.size());
}

}  // namespace

extern "C" {

const char* bpp_last_error(void) { return g_error.c_str(); }

const char* bpp_kernel_name(void) { return bpp::kernel_name(); }

void bpp_note_options_default(bpp_note_options* o) {
    const bpp::NoteOptions d;
    o->onset_threshold = d.onset_threshold;
    o->frame_threshold = d.frame_threshold;
    o->minimum_note_length_ms = d.minimum_note_length_ms;
    o->minimum_frequency = 0;
    o->maximum_frequency = 0;
    o->infer_onsets = d.infer_onsets;
    o->melodia_trick = d.melodia_trick;
    o->include_pitch_bends = d.include_pitch_bends;
    o->multiple_pitch_bends = d.multiple_pitch_bends;
    o->midi_tempo = d.midi_tempo;
}

int bpp_frame_count(size_t n_samples) {
    return bpp::frame_count(n_samples);
}

int bpp_window_count(size_t n_samples) { return bpp::window_count(n_samples); }

bpp_engine* bpp_engine_new(int n_threads) {
    return guard([&] { return new bpp_engine(n_threads); }, nullptr);
}

void bpp_engine_free(bpp_engine* e) { delete e; }

int bpp_engine_infer(bpp_engine* e, const float* audio, size_t n, float* note, float* onset, float* contour) {
    return guard(
        [&] {
            const bpp::ModelOutput out = e->engine.infer(audio, n);
            copy_frames(out, 0, out.n_frames, note, onset, contour);
            return out.n_frames;
        },
        -1);
}

bpp_output* bpp_engine_infer_file(bpp_engine* e, const char* path) {
    return guard([&] { return new bpp_output{e->engine.infer_file(path)}; }, static_cast<bpp_output*>(nullptr));
}

bpp_output* bpp_engine_infer_pcm(bpp_engine* e, const float* audio, size_t n_frames, int sample_rate, int channels) {
    return guard([&] { return new bpp_output{e->engine.infer_pcm(audio, n_frames, sample_rate, channels)}; },
                 static_cast<bpp_output*>(nullptr));
}

bpp_output* bpp_output_new(size_t n_samples) {
    return guard(
        [&] {
            auto* o = new bpp_output;
            const int n = bpp::frame_count(n_samples);
            o->out.n_frames = n;
            o->out.note.resize(size_t(n) * kNote);
            o->out.onset.resize(size_t(n) * kNote);
            o->out.contour.resize(size_t(n) * kContour);
            return o;
        },
        static_cast<bpp_output*>(nullptr));
}

int bpp_engine_infer_windows(bpp_engine* e, const float* audio, size_t n_samples, int first, int count, bpp_output* o) {
    return guard(
        [&] {
            e->engine.infer_windows(audio, n_samples, first, count, o->out);
            return 0;
        },
        -1);
}

int bpp_output_frames(const bpp_output* o) { return o->out.n_frames; }

void bpp_output_copy(const bpp_output* o, float* note, float* onset, float* contour) {
    copy_frames(o->out, 0, o->out.n_frames, note, onset, contour);
}

void bpp_output_free(bpp_output* o) { delete o; }

bpp_stream* bpp_stream_new(int n_threads) {
    return bpp_stream_new_pcm(n_threads, bpp::kModelSampleRate, 1);
}

bpp_stream* bpp_stream_new_pcm(int n_threads, int sample_rate, int channels) {
    return guard([&] { return new bpp_stream(n_threads, sample_rate, channels); }, nullptr);
}

void bpp_stream_free(bpp_stream* s) { delete s; }

int bpp_stream_push(bpp_stream* s, const float* audio, size_t n) {
    return guard(
        [&] {
            s->stream.push(audio, n);
            return s->stream.output().n_frames;
        },
        -1);
}

int bpp_stream_final_frames(const bpp_stream* s) { return s->stream.final_frames(); }

int bpp_stream_finish(bpp_stream* s) {
    return guard(
        [&] {
            s->stream.finish();
            return s->stream.output().n_frames;
        },
        -1);
}

int bpp_stream_read(bpp_stream* s, int first, int count, float* note, float* onset, float* contour) {
    return guard(
        [&] {
            const bpp::ModelOutput& out = s->stream.output();
            if (first < 0 || count < 0 || first > out.n_frames || count > out.n_frames - first)
                throw std::out_of_range("frame range");
            copy_frames(out, first, count, note, onset, contour);
            return 0;
        },
        -1);
}

bpp_notes* bpp_notes_decode(const float* note, const float* onset, const float* contour, int n_frames,
                            const bpp_note_options* options) {
    return guard(
        [&] {
            if (n_frames < 0) throw std::invalid_argument("n_frames must be >= 0");
            bpp::ModelOutput out;
            out.n_frames = n_frames;
            out.note.assign(note, note + size_t(n_frames) * kNote);
            out.onset.assign(onset, onset + size_t(n_frames) * kNote);
            out.contour.assign(contour, contour + size_t(n_frames) * kContour);
            auto* r = new bpp_notes;
            r->notes = bpp::output_to_notes(out, to_cpp(options));
            return r;
        },
        static_cast<bpp_notes*>(nullptr));
}

void bpp_notes_free(bpp_notes* n) { delete n; }

int bpp_notes_count(const bpp_notes* n) { return static_cast<int>(n->notes.size()); }

int bpp_notes_get(const bpp_notes* n, int i, double* start_s, double* end_s, int* start_frame, int* end_frame,
                  int* pitch, float* amplitude) {
    if (i < 0 || i >= static_cast<int>(n->notes.size())) {
        g_error = "note index out of range";
        return -1;
    }
    const bpp::NoteEvent& e = n->notes[i];
    *start_s = e.start_s;
    *end_s = e.end_s;
    *start_frame = e.start_frame;
    *end_frame = e.end_frame;
    *pitch = e.pitch;
    *amplitude = e.amplitude;
    return static_cast<int>(e.pitch_bends.size());
}

void bpp_notes_get_bends(const bpp_notes* n, int i, int* bends) {
    if (i < 0 || i >= static_cast<int>(n->notes.size())) return;
    const auto& b = n->notes[i].pitch_bends;
    std::copy(b.begin(), b.end(), bends);
}

size_t bpp_notes_midi(const bpp_notes* n, const bpp_note_options* options, unsigned char* buf, size_t cap) {
    return guard(
        [&] {
            const auto midi = bpp::notes_to_midi(n->notes, to_cpp(options));
            if (buf) std::memcpy(buf, midi.data(), std::min(cap, midi.size()));
            return midi.size();
        },
        size_t(0));
}

int bpp_convert_pcm(const float* audio, size_t n_frames, int sample_rate, int channels, float** samples, size_t* n) {
    return guard(
        [&] {
            export_audio(bpp::convert_pcm(audio, n_frames, sample_rate, channels), samples, n);
            return 0;
        },
        -1);
}

int bpp_load_audio(const char* path, float** samples, size_t* n) {
    return guard(
        [&] {
            export_audio(bpp::load_audio(path), samples, n);
            return 0;
        },
        -1);
}

void bpp_free_audio(float* samples) { std::free(samples); }

}  // extern "C"
