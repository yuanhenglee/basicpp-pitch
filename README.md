# BasiCPP Pitch

A fast, dependency-free C++ engine for Spotify's [basic-pitch](https://github.com/spotify/basic-pitch)
polyphonic, instrument-agnostic music transcription model
([paper](https://arxiv.org/abs/2203.09893)): audio in, notes / pitch bends / MIDI out.

* **Bit-true.** Model activations, note events (including float32 amplitudes), pitch bends, MIDI
  bytes and CSV bytes are identical to `basic_pitch.inference.predict()` (basic-pitch 0.4.0,
  TensorFlow 2.15 backend), starting from the original WAV file. See
  [docs/BIT_EXACTNESS.md](docs/BIT_EXACTNESS.md) for how and why.
* **Fast.** About 160× real time on one core and 450–600× on a laptop CPU, roughly 20× faster
  than basic-pitch itself.
* **Real time.** A streaming API turns live audio into frames as each 1.64 s analysis window
  completes, using under 1 % of the time budget. Its output is bit-identical to offline processing.
* **Self-contained.** No TensorFlow, Python, Eigen or MKL. The weights are compiled in, AVX2/FMA
  kernels are hand-written with a portable scalar fallback, and libsoxr is vendored for
  librosa-identical resampling. C++17 library, CLI, C API and Python bindings.

## Results

Intel Core Ultra 5 135U (15 W laptop, 2P + 8E + 2LP cores), Windows 11, 160.6 s song
(`data/Undertale-Megalovania.wav`, 44.1 kHz stereo):

| | time | real-time factor |
|---|---|---|
| basic-pitch 0.4.0, TF 2.15 (`predict`, excluding ~10 s import / model load) | 5.73 s | 28× |
| basic-pitch with ONNX Runtime / TFLite backend | 11.4 s / 8.7 s | 14× / 18× |
| previous C++ version of this repo, 1 thread / 14 threads (pre-resampled input) | 35.0 s / 6.7 s | 5× / 24× |
| **BasiCPP Pitch, 1 thread** (model + notes) | **1.03 s** | **157×** |
| **BasiCPP Pitch, 4 threads** | **0.44 s** | **369×** |
| **BasiCPP Pitch, 14 threads** | **0.26 s** | **622×** |
| **BasiCPP Pitch, WAV → MIDI end-to-end** (decode + resample overlapped, 14 threads) | **0.35 s** | **~450×** |

Streaming: one analysis window (1.64 s of new audio) costs 11 ms on one thread or about 6.5 ms
split across 4 threads. Final notes are available about 20 ms after the stream ends.

Bit-exactness is verified on 38 cases (11 real recordings, 15 synthetic edge cases such as
silence, window-boundary lengths, −80 dB audio, clipping, noise and a 100 s sustained chord,
plus 12 WAV encodings / sample rates from 8 to 96 kHz). The checks cover offline, streaming,
scalar-build and end-to-end-from-WAV runs, and 7 note-option variants checked against
basic-pitch's own decoder.

## Build

Requirements: CMake ≥ 3.16 and a C++17 compiler (GCC, Clang, MSVC). The default target is x86-64-v3
(AVX2 + FMA); use `-DBPP_ARCH=native`, or `-DBPP_ARCH=x86-64` for the portable scalar build.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build          # self-contained bit-exactness regression
```

Windows without a compiler installed: Zig's bundled clang works (`pip install ziglang cmake ninja`):

```sh
cmake -G Ninja -S . -B build -DCMAKE_TOOLCHAIN_FILE=cmake/zig-toolchain.cmake -DBPP_ARCH=native
cmake --build build
```

Outputs: `bpp` (CLI), `libbpp` (static library), `basicpp_pitch` shared library (C API, used by the
Python package).

## Usage

### Command line

```sh
bpp transcribe song.wav -o song.mid --csv song.csv          # same files as `basic-pitch --save-midi --save-note-events`
bpp transcribe song.wav -o song.mid --minimum-frequency 80 --no-melodia
bpp stream song.wav --chunk-ms 20 --realtime                # simulate live input, report latency
bpp bench song.wav --threads 1
```

Note options mirror basic-pitch: `--onset-threshold`, `--frame-threshold`, `--minimum-note-length`,
`--minimum-frequency`, `--maximum-frequency`, `--multiple-pitch-bends`, `--no-melodia`, `--midi-tempo`.

### C++

```cpp
#include "bpp/bpp.h"

bpp::Engine engine;                                   // all cores; thread-safe, reusable
bpp::ModelOutput out = engine.infer_file("song.wav"); // or engine.infer(samples, n) for 22050 Hz mono float
std::vector<bpp::NoteEvent> notes = bpp::output_to_notes(out);
std::vector<uint8_t> midi = bpp::notes_to_midi(notes);

// live input
bpp::Stream stream(/*threads=*/2);
while (capturing) stream.push(chunk, n);              // 22050 Hz mono float32
stream.finish();
auto live_notes = bpp::output_to_notes(stream.output());
```

### Python

```python
import sys; sys.path.insert(0, "python")   # or copy python/basicpp_pitch next to your code
import basicpp_pitch as bpp

result = bpp.predict("song.wav")            # same outputs as basic_pitch.inference.predict
open("song.mid", "wb").write(result.midi_bytes)
for start_s, end_s, pitch, amplitude, bends in result.note_events:
    ...
```

`bpp.Engine`, `bpp.Stream`, `bpp.load_audio` and `bpp.output_to_notes` expose the same pieces as the
C++ API. The package finds the shared library next to itself, in `build/`, or via
`BASICPP_PITCH_LIB`.

## How it works

```
WAV ─ decode ─ resample (libsoxr, as librosa) ─┐
                                               ▼
         windows of 43 844 samples every 36 164 (3840 zeros prepended), processed in parallel
                                               ▼
  CQT: 9 octaves of reflect-padded 256-tap complex conv + decimation ─ |·|² ─ dB ─ min/max norm ─ BN
                                               ▼
  harmonic stacking (8 harmonics, 264 bins) ─┬─ conv 3×39 ─ BN ─ ReLU ─ conv 5×5 ─ sigmoid ── contour
                                             │                              └ conv 7×7/3 ─ ReLU ─ conv 7×3 ─ sigmoid ── note
                                             └─ conv 5×5/3 ─ BN ─ ReLU ─ concat(note) ─ conv 3×3 ─ sigmoid ── onset
                                               ▼
  note decoding (onset peaks, energy tracking, melodia trick, pitch bends) ─ MIDI / CSV
```

Design choices that differ from the previous version of this repository:

| previous | now | why |
|---|---|---|
| weights loaded from JSON / `.npy` at runtime | weights exported from the official SavedModel into the binary | no I/O or parsing, exact constants |
| Eigen + MKL, `VecMatrixf` copies, im2col | NHWC / channel-major buffers, register-blocked AVX2 micro-kernels | 34× faster single-threaded |
| whole CNN on all 172 frames per window | only the rows the kept frames depend on; skip windows that get truncated | ~10 % less work, same result |
| one `std::thread` per window per batch | work-stealing over windows; row splitting inside a window for streaming | scales across hybrid cores; low latency |
| results within 1e-3 of TF; note decoding with several divergences | arithmetic identical to TF's kernels; decoding replicated with NumPy's dtypes | bit-true |
| pybind11 module | C API + ctypes package | no build-time Python dependency |
| caller resamples audio | librosa-identical WAV decoding and resampling built in | end-to-end exactness |

`conv2d_1` (about 60 % of the work) runs at roughly two thirds of one P-core's FMA peak. The accumulation order is
fixed by the exactness contract, so remaining speedups come from parallelism.

## Verification workflow

The reference tools need Python 3.11 with `basic-pitch==0.4.0` (which pulls in TensorFlow 2.15):

```sh
python tools/make_corpus.py data/corpus                    # synthetic edge cases (+ librosa examples)
python tools/golden.py data/corpus/*.wav --out golden/tf   # reference outputs from basic-pitch
python tools/verify.py                                     # C++ vs reference: every float, note and byte
python tools/verify.py --source                            # end to end from the original WAV files
python tools/verify.py --stream-chunk-ms 10                # streaming API
python -m pytest tests                                     # Python binding + note options vs basic-pitch
```

`tools/debug/` contains the tools used to reverse engineer TensorFlow's arithmetic (layer-by-layer
diff, oneDNN reduction-order probes).

## Layout

```
include/bpp/      public C++ (bpp.h) and C (bpp_c.h) API
src/              engine: window_model (CQT + CNN), engine (windowing, threads, streaming),
                  notes, midi, audio + resample, simd (Eigen-exact math), model_data (generated)
apps/bpp_cli.cpp  command line tool
python/           ctypes bindings
tests/            CTest fixture test, pytest suite
tools/            weight export, golden generation, verification, debugging probes
third_party/      libsoxr 0.1.3 (LGPL-2.1) as bundled by python-soxr 1.1; UCRT math shim
docs/             BIT_EXACTNESS.md
```

## License notes

The model weights and the algorithms come from basic-pitch (Apache-2.0). libsoxr is LGPL-2.1 and is
linked statically by default; build with `-DBPP_WITH_SOXR=OFF` to leave it out (22050 Hz input only).

## References

- [basic-pitch](https://github.com/spotify/basic-pitch)
- R. Bittner et al., [A Lightweight Instrument-Agnostic Model for Polyphonic Note Transcription and Multipitch Estimation](https://arxiv.org/abs/2203.09893), ICASSP 2022
