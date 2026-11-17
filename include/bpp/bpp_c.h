/* Plain C interface to BasiCPP Pitch, used by the Python binding (ctypes) and other FFIs. */
#ifndef BPP_C_H
#define BPP_C_H

#include <stddef.h>

#if defined(_WIN32)
#if defined(BPP_BUILDING_LIBRARY)
#define BPP_API __declspec(dllexport)
#else
#define BPP_API __declspec(dllimport)
#endif
#else
#define BPP_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bpp_engine bpp_engine;
typedef struct bpp_stream bpp_stream;
typedef struct bpp_notes bpp_notes;
typedef struct bpp_output bpp_output;

typedef struct {
    double onset_threshold;         /* 0.5 */
    double frame_threshold;         /* 0.3 */
    double minimum_note_length_ms;  /* 127.70 */
    double minimum_frequency;       /* Hz, <= 0 for none */
    double maximum_frequency;       /* Hz, <= 0 for none */
    int infer_onsets;               /* 1 */
    int melodia_trick;              /* 1 */
    int include_pitch_bends;        /* 1 */
    int multiple_pitch_bends;       /* 0 */
    double midi_tempo;              /* 120 */
} bpp_note_options;

/* Returns a message for the last error on this thread, or "" */
BPP_API const char* bpp_last_error(void);

BPP_API void bpp_note_options_default(bpp_note_options* options);

/* Number of output frames for n_samples of 22050 Hz audio. */
BPP_API int bpp_frame_count(size_t n_samples);

BPP_API bpp_engine* bpp_engine_new(int n_threads);
BPP_API void bpp_engine_free(bpp_engine* engine);
/* Writes bpp_frame_count(n) rows into note[.][88], onset[.][88], contour[.][264]. Returns frames or -1. */
BPP_API int bpp_engine_infer(bpp_engine* engine, const float* audio, size_t n, float* note, float* onset, float* contour);

/* Decodes a WAV file (librosa-identical) and runs inference with resampling overlapped. */
BPP_API bpp_output* bpp_engine_infer_file(bpp_engine* engine, const char* path);
/* Interleaved float32 PCM, n_frames frames of `channels` samples at any rate from 1 kHz to 1 MHz:
   mixed to mono and resampled like librosa.load(sr=22050), then inferred. */
BPP_API bpp_output* bpp_engine_infer_pcm(bpp_engine* engine, const float* audio, size_t n_frames, int sample_rate,
                                         int channels);
BPP_API int bpp_output_frames(const bpp_output* output);
BPP_API void bpp_output_copy(const bpp_output* output, float* note, float* onset, float* contour);
BPP_API void bpp_output_free(bpp_output* output);

/* 22050 Hz mono input. */
BPP_API bpp_stream* bpp_stream_new(int n_threads);
/* Interleaved float32 input at any rate / channel count, converted like bpp_engine_infer_pcm. */
BPP_API bpp_stream* bpp_stream_new_pcm(int n_threads, int sample_rate, int channels);
BPP_API void bpp_stream_free(bpp_stream* stream);
/* push takes n frames (n * channels floats). Both return the total number of frames computed so far (see bpp::Stream), or -1 on error
   (including push after finish). finish() trims the output to basic-pitch's length. */
BPP_API int bpp_stream_push(bpp_stream* stream, const float* audio, size_t n);
BPP_API int bpp_stream_finish(bpp_stream* stream);
/* Leading frames that are final no matter when the stream ends. */
BPP_API int bpp_stream_final_frames(const bpp_stream* stream);
/* Copies frames [first, first + count) of the stream output. Returns 0 on success. */
BPP_API int bpp_stream_read(bpp_stream* stream, int first, int count, float* note, float* onset, float* contour);

BPP_API bpp_notes* bpp_notes_decode(const float* note, const float* onset, const float* contour, int n_frames,
                                    const bpp_note_options* options);
BPP_API void bpp_notes_free(bpp_notes* notes);
BPP_API int bpp_notes_count(const bpp_notes* notes);
/* Returns the number of pitch bend values of note i (0 if none), -1 if i is out of range. */
BPP_API int bpp_notes_get(const bpp_notes* notes, int i, double* start_s, double* end_s, int* start_frame,
                          int* end_frame, int* pitch, float* amplitude);
BPP_API void bpp_notes_get_bends(const bpp_notes* notes, int i, int* bends);
/* MIDI bytes; call with buf == NULL to get the size. */
BPP_API size_t bpp_notes_midi(const bpp_notes* notes, const bpp_note_options* options, unsigned char* buf, size_t cap);

/* Decodes a WAV file to 22050 Hz mono; free the samples with bpp_free_audio. Returns 0 on success. */
BPP_API int bpp_load_audio(const char* path, float** samples, size_t* n);
BPP_API void bpp_free_audio(float* samples);

#ifdef __cplusplus
}
#endif

#endif /* BPP_C_H */
