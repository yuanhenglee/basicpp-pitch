# How BasiCPP Pitch matches basic-pitch bit for bit

The reference is `basic_pitch.inference.predict()` from **basic-pitch 0.4.0** with its default
backend on Python ≥ 3.11 / x86-64: **TensorFlow 2.15 CPU** running the bundled ICASSP 2022
SavedModel, audio loaded by **librosa 0.11** (soundfile + python-soxr 1.1). Every intermediate
tensor, the note events (including the float32 amplitude) and the MIDI / CSV bytes are identical.

This document records what had to be reproduced and how each detail was established.

## 0. What "bit-true" can mean here

basic-pitch's own backends do not agree with each other at the float level. On the 160 s example:

| backend vs TF (oneDNN) | identical floats | max abs diff | notes / MIDI |
|---|---|---|---|
| TF without oneDNN (`TF_ENABLE_ONEDNN_OPTS=0`) | 4–12 % | 1.1e-4 | identical here |
| ONNX Runtime | 4–6 % | 1.8e-6 | identical here |
| TFLite | 3–10 % | 9.2e-5 | identical here |

On `dense_poly_45s` TFLite and TF-without-oneDNN even produce **5 different notes**: a frame
activation of 0.29999989 vs 0.30000010 lands on opposite sides of the 0.3 threshold. So any
implementation that is merely "accurate" will occasionally disagree with the reference on notes.
The only robust contract is to reproduce the reference arithmetic exactly, which is what this
project does for the default backend on AVX2 CPUs.

## 1. Model weights and constants

`tools/export_model.py` reads the official SavedModel: conv kernels and biases, batch-norm
statistics, and the CQT constants that were frozen into the graph at export time (complex kernel,
256-tap low-pass filter, bin lengths). They are emitted as hex-float literals in
`src/model_data.cpp`, so the engine needs no files at runtime. The recomputed nnAudio kernel and
filter were checked to be bit-identical to the frozen ones.

## 2. Which kernels TensorFlow actually runs

`ONEDNN_VERBOSE=1` on the SavedModel shows the implementation of every primitive:

| op | kernel | reduction order (per output value) |
|---|---|---|
| CQT conv1d, hop ≥ 8 | `brgconv:avx2` | FMA chain from 0 over taps 0…255 |
| CQT conv1d, hop ≤ 4; decimation FIR | `jit:avx2` | FMA chain from 0 over taps 0…255 |
| `conv2d_1` 3×39, 8→8 | `brgconv:avx2` | **one chain per kernel column kw** over (kh, ic), partials summed in kw order, then + bias |
| `conv2d_4` 5×5/3, 8→32; 1-channel convs | `brgconv:avx2` | chain from 0 over (kh, kw, ic), then + bias |
| `conv2d_2` 7×7/3, 1→32 (+ReLU post-op) | `jit:avx2` | chain **starting at the bias**, over (kh, kw) |
| CQT batch norm (1 ch) | `nspc_bnorm` | `(x − μ) · (γ / √(σ²+ε)) + β`, no FMA |
| batch norm 8 / 32 ch (+ReLU) | `bnorm_tbb_jit:avx2` | `fma((x − μ) · (1/√(σ²+ε)), γ, β)` |

The reduction structures were measured, not guessed: `tools/debug/conv_structure.py` feeds a
same-shaped `tf.nn.conv2d` unit weights and inputs that are all 1.0 except one tap set to 2^24.
Later additions of 1.0 are absorbed by round-to-nearest-even, so the output reveals where the
probe sits in the reduction tree; the measured pattern is matched against float32 simulations of
candidate loop nests. That is how the per-column partial sums of `conv2d_1` were found.

Zero padding needs no special care: skipping a tap whose input is +0 is equivalent to adding it,
because an accumulator that starts at +0 can never become −0.

## 3. Element-wise ops come from Eigen (built for AVX, without FMA)

TensorFlow's x86 wheels compile Eigen for AVX without FMA, so every `pmadd` is a separately
rounded multiply and add. Each function was identified empirically against `tf.math.*` on
500 000 random inputs (100 % match required):

* **log**: Eigen `plog_impl_float`, the rational (3,3) approximation, no FMA. Eigen evaluates
  tensors in 8-wide packets; the final `size % 8` elements go through the scalar path, i.e. the C
  runtime's `logf` (the CQT dB tensor has 53 148 = 8·6643 + 4 elements).
* **sigmoid**: Eigen's generic `scalar_logistic_op`, `e / (1 + e)` with `e = pexp(x)` (degree-6
  minimax `pexp_float`, Cody-Waite reduction, `pldexp` scaling), returning 1 where `e` overflows.
* **sqrt, square, add, mul, div, min, max**: plain IEEE operations.

Implementations are in `src/simd.h` (scalar and AVX2 versions return identical bits).

## 4. Grappler rewrites the graph

The graph TensorFlow executes is not the one the Python code describes:

* `10 * (log(x) / log(10))` is folded by constant push-down into `log(x) * (10 / log(10))` with
  the constant computed in float32. Using the unfolded form changes 13 % of the dB values.
* `pow(x, 2)` is exact squaring.
* Batch-norm parameters are variables in the SavedModel, so the batch norms are **not** folded
  into Mul/Add; they run in oneDNN with the formulas above. (A test with constant parameters
  shows a third, folded formula: an easy trap when reverse engineering.)

## 5. Windowing

basic-pitch prepends 3840 zeros, cuts 43 844-sample windows every 36 164 samples, keeps frames
15…156 of each and truncates to `floor(n · 86 / 22050)` frames. The engine reproduces this exactly,
skips windows whose frames are all truncated away, and computes each CNN layer only for the rows
those frames depend on (the receptive field is ±10 frames, so no time padding is ever read).

## 6. Note decoding (float32 / float64 exactly as NumPy)

`src/notes.cpp` follows `note_creation.py` line by line, keeping NumPy's dtypes:

* `get_infered_onsets` promotes to float64; with `max(diff) == 0` NumPy produces NaN everywhere
  and therefore no onset peaks (silence). Reproduced.
* `scipy.signal.argrelmax` is strict on both sides with clipped edges.
* Thresholds are Python floats (float64): comparing float32 activations against `0.3` and against
  `float32(0.3)` differs in general, so all options are doubles.
* The melodia loop (repeated `np.argmax` over remaining energy) equals visiting cells by
  (value desc, flat index asc), because cells are only ever zeroed.
* `np.mean` on a float32 column: pairwise summation (8-way unrolled blocks of ≤ 128, recursive
  split) over 8192-element buffer chunks, then a float32 division. Verified on random data.
* Pitch bends use the exact float64 values of `scipy.signal.windows.gaussian(51, 5)`.
* Frame times: `frames * 256 / 22050 − window_offset * floor(frames / 172)` in float64.

## 7. MIDI and CSV

`src/midi.cpp` reproduces `note_events_to_midi` + pretty_midi's `write` + mido's serialisation:
tuple sorting in `drop_overlapping_pitch_bends`, `np.linspace` for bend times, round-half-even
tick quantisation (`time / (60 / (tempo · 220))`), pretty_midi's event ordering keys, the swap
pass, set_tempo before time_signature, and mido's running status. The CLI's `--csv` matches
`save_note_events` (Python `repr` float formatting, `\r\n` rows).

## 8. Audio loading and resampling

`librosa.load(path, sr=22050, mono=True)`: libsndfile's integer→float scaling, `np.mean` over
channels in float32, then `soxr.resample(..., quality="soxr_hq")` and `fix_length` to
`ceil(n · ratio)`. The project vendors the exact libsoxr source from the python-soxr 1.1 sdist,
configures it like python-soxr (no OpenMP, PFFFT, SIMD engines), and drives it with the same call
sequence (split-channel I/O, 48000·ratio input chunks, flush). Two traps:

* `-ffp-contract=off` is required: clang would otherwise fuse multiply-adds that MSVC does not.
* The MSVC-built wheels use the Universal CRT's `sin/cos/sinf/cosf/exp/log/pow`. MinGW links its
  own libm, whose `cosf` rounds one 2048-point FFT twiddle factor differently, which perturbs every
  output sample by up to an ulp. MinGW builds therefore route those calls to `ucrtbase.dll`
  (`third_party/ucrt_math`). Native MSVC builds use the UCRT anyway.

## 9. Scope and caveats

* The reduction orders are those of oneDNN's **AVX2** kernels. On AVX-512 or AMX machines
  TensorFlow selects different kernels, so its own output there differs at the ulp level (and,
  rarely, in notes) from TF on AVX2 machines — and from this engine, which always reproduces the
  AVX2 arithmetic. The scalar fallback (any CPU without AVX2/FMA) reproduces the same arithmetic,
  so model output, notes and MIDI are identical wherever the code runs (verified on x86-64 with the
  AVX2 and the scalar build).
* Resampling depends on the C runtime's math library, exactly like the python-soxr wheels do. It is
  verified against the Windows (MSVC/UCRT) wheel. On Linux, building with GCC/Clang against glibc
  mirrors how the Linux wheels are built, but that combination has not been verified here.
  Input that is already 22050 Hz is never resampled and is unaffected.
* Other basic-pitch backends (CoreML on macOS, ONNX, TFLite) are not bit-compatible with TF and
  therefore not with this engine either (see §0).
* Audio input is WAV. Other formats must be decoded to WAV or 22050 Hz float32 first.

## Verification

* `tools/golden.py` produces reference outputs with basic-pitch; `tools/verify.py` runs the C++ CLI
  (offline, streaming, or from the original WAV) and compares every float, note field and byte.
* `tests/test_vs_basic_pitch.py` (pytest) checks the Python binding, note options against
  basic-pitch's own decoder, and streaming vs offline.
* `tests/test_fixture.cpp` (CTest) is a self-contained regression on a 44.1 kHz stereo clip.
* `tools/debug/layer_diff.py` compares one window layer by layer against a Keras rebuild of the
  network loaded with the SavedModel's variables (itself verified bit-identical to the SavedModel).
