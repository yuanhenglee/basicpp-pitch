// Fixed parameters of the basic-pitch ICASSP 2022 model and its inference pipeline.
#pragma once

namespace bpp {

inline constexpr int kSampleRate = 22050;
inline constexpr int kFftHop = 256;
inline constexpr int kAnnotationsFps = kSampleRate / kFftHop;  // 86
inline constexpr int kWindowSamples = kSampleRate * 2 - kFftHop;  // 43844, AUDIO_N_SAMPLES
inline constexpr int kFramesPerWindow = 172;  // ANNOT_N_FRAMES
inline constexpr int kOverlapFrames = 30;
inline constexpr int kOverlapSamples = kOverlapFrames * kFftHop;  // 7680
inline constexpr int kWindowHop = kWindowSamples - kOverlapSamples;  // 36164
inline constexpr int kHalfOverlap = kOverlapFrames / 2;  // frames dropped at each window edge
inline constexpr int kUsedFrames = kFramesPerWindow - kOverlapFrames;  // 142 frames kept per window

inline constexpr int kCqtBins = 309;  // 103 semitones * 3
inline constexpr int kBinsPerOctave = 36;
inline constexpr int kOctaves = 9;
inline constexpr int kCqtKernelLen = 256;
inline constexpr int kLowpassLen = 256;

inline constexpr int kHarmonics = 8;
inline constexpr int kContourBins = 264;  // 88 * 3
inline constexpr int kNoteBins = 88;
inline constexpr int kMidiOffset = 21;

}  // namespace bpp
