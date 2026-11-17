// Note decoding: an exact replica of basic_pitch.note_creation (basic-pitch 0.4.0).
// Precision follows the Python code: activations are float32, intermediate matrices that
// NumPy promotes to float64 are float64 here too, and np.mean uses NumPy's pairwise sum.
#include "bpp/bpp.h"

#include "constants.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace bpp {

namespace {

constexpr int kMaxFreqIdx = 87;
constexpr int kEnergyTol = 11;
constexpr int kBinsTolerance = 25;

// scipy.signal.windows.gaussian(51, std=5), exact float64 values.
constexpr double kPitchBendGaussian[2 * kBinsTolerance + 1] = {
    0x1.f42ed3f68e690p-19, 0x1.4d2dcae279bfdp-17, 0x1.aa774310a5b40p-16, 0x1.063bf418bbb1cp-14, 0x1.35d9cc01eb2d5p-13,
    0x1.5fc21041027adp-12, 0x1.7facdba659766p-11, 0x1.92144ad2e5a28p-10, 0x1.94d817547e93ep-9,  0x1.87a506e4ea251p-8,
    0x1.6c0504695c417p-7,  0x1.45139542cf62fp-6,  0x1.16eab0131bceep-5,  0x1.cbdb2151527fap-5,  0x1.6c3912ee53ea2p-4,
    0x1.152aaa3bf81ccp-3,  0x1.954be9c3e16c7p-3,  0x1.1cb5cf620ef1fp-2,  0x1.80518d7d9201ap-2,  0x1.f26f2ef268c13p-2,
    0x1.368b2fc6f960ap-1,  0x1.73c9ce7c81bcep-1,  0x1.aba88982ab390p-1,  0x1.d8a2b4ac44685p-1,  0x1.f5dc99badec5bp-1,
    0x1.0000000000000p+0,  0x1.f5dc99badec5bp-1,  0x1.d8a2b4ac44685p-1,  0x1.aba88982ab390p-1,  0x1.73c9ce7c81bcep-1,
    0x1.368b2fc6f960ap-1,  0x1.f26f2ef268c13p-2,  0x1.80518d7d9201ap-2,  0x1.1cb5cf620ef1fp-2,  0x1.954be9c3e16c7p-3,
    0x1.152aaa3bf81ccp-3,  0x1.6c3912ee53ea2p-4,  0x1.cbdb2151527fap-5,  0x1.16eab0131bceep-5,  0x1.45139542cf62fp-6,
    0x1.6c0504695c417p-7,  0x1.87a506e4ea251p-8,  0x1.94d817547e93ep-9,  0x1.92144ad2e5a28p-10, 0x1.7facdba659766p-11,
    0x1.5fc21041027adp-12, 0x1.35d9cc01eb2d5p-13, 0x1.063bf418bbb1cp-14, 0x1.aa774310a5b40p-16, 0x1.4d2dcae279bfdp-17,
    0x1.f42ed3f68e690p-19,
};

// NumPy's float32 pairwise summation (numpy/_core/src/umath/loops_utils.h.src) over a strided column.
float pairwise_sum(const float* a, long long n, long long stride) {
    if (n < 8) {
        float res = 0.0f;
        for (long long i = 0; i < n; ++i) res += a[i * stride];
        return res;
    }
    if (n <= 128) {
        float r[8];
        for (int j = 0; j < 8; ++j) r[j] = a[j * stride];
        long long i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; ++j) r[j] += a[(i + j) * stride];
        float res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i * stride];
        return res;
    }
    long long n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise_sum(a, n2, stride) + pairwise_sum(a + n2 * stride, n - n2, stride);
}

// np.mean(frames[begin:end, col]) for a float32 matrix: the reduction runs in 8192-element
// buffer chunks, each summed pairwise, then the float32 sum is divided by the count.
float column_mean(const float* m, int cols, int begin, int end, int col) {
    const long long n = end - begin;
    if (n <= 0) return std::numeric_limits<float>::quiet_NaN();
    float sum = 0.0f;
    for (long long i = 0; i < n; i += 8192)
        sum += pairwise_sum(m + (begin + i) * cols + col, std::min<long long>(8192, n - i), cols);
    return sum / static_cast<float>(n);
}

// Python slice bound normalisation for a[:, idx:] / a[:, :idx] on an axis of length len.
int py_slice_index(int idx, int len) {
    if (idx < 0) idx += len;
    return std::clamp(idx, 0, len);
}

double hz_to_midi(double hz) { return 12.0 * (std::log2(hz) - std::log2(440.0)) + 69.0; }

// model_frames_to_time(): frame index -> seconds, with basic-pitch's per-window offset correction.
double frame_time(int frame) {
    const double original = static_cast<double>(static_cast<long long>(frame) * kFftHop) / double(kSampleRate);
    const double window_number = std::floor(static_cast<double>(frame) / double(kFramesPerWindow));
    const double window_offset =
        (double(kFftHop) / double(kSampleRate)) * (double(kFramesPerWindow) - (double(kWindowSamples) / double(kFftHop))) +
        0.0018;
    return original - window_offset * window_number;
}

struct RawNote {
    int start, end, pitch;
    float amplitude;
};

std::vector<RawNote> output_to_notes_polyphonic(const float* frame, const float* onset_in, int n_frames,
                                                const NoteOptions& opt, int min_note_len) {
    constexpr int F = kNoteBins;

    // get_infered_onsets() works in float64: onsets' = max(onsets, max(onsets) * diff / max(diff)),
    // diff = max(0, min(frames[t] - frames[t-1], frames[t] - frames[t-2])), zero for t < 2.
    float max_onset = -std::numeric_limits<float>::infinity();
    double max_diff = 0.0;  // frame_diff[:2] = 0, so the maximum is never below 0
    if (opt.infer_onsets) {
        for (size_t i = 0; i < size_t(n_frames) * F; ++i) max_onset = std::max(max_onset, onset_in[i]);
        for (int t = 2; t < n_frames; ++t) {
            const float* c = frame + size_t(t) * F;
            double row_max = 0.0;
            for (int f = 0; f < F; ++f) {
                const double cur = c[f];
                const double d = std::min(cur - double(c[f - F]), cur - double(c[f - 2 * F]));
                row_max = d > row_max ? d : row_max;
            }
            max_diff = std::max(max_diff, row_max);
        }
    }
    // With max(diff) == 0 NumPy computes 0/0: every inferred onset is NaN and argrelmax finds no peak.
    const bool all_nan = opt.infer_onsets && !(max_diff > 0.0);
    const double mo = max_onset;
    auto onset_row = [&](int t, double* out) {
        const float* o = onset_in + size_t(t) * F;
        if (!opt.infer_onsets) {
            for (int f = 0; f < F; ++f) out[f] = o[f];
        } else if (t < 2) {
            for (int f = 0; f < F; ++f) out[f] = std::max(double(o[f]), (mo * 0.0) / max_diff);
        } else {
            const float* c = frame + size_t(t) * F;
            const float* p1 = c - F;
            const float* p2 = c - 2 * F;
            for (int f = 0; f < F; ++f) {
                const double cur = c[f];
                double d = std::min(cur - double(p1[f]), cur - double(p2[f]));
                d = d < 0 ? 0.0 : d;
                const double scaled = (mo * d) / max_diff;
                out[f] = double(o[f]) < scaled ? scaled : double(o[f]);
            }
        }
    };

    // scipy.signal.argrelmax(onsets, axis=0) (edges clipped), then threshold; rows are produced
    // on the fly in a 3-row window. Onsets are visited backwards in time.
    std::vector<std::pair<int, int>> onset_idx;
    if (!all_nan) {
        double buf[3][kNoteBins];
        double *prev = buf[0], *cur = buf[1], *next = buf[2];
        onset_row(0, cur);
        std::copy(cur, cur + F, prev);  // row -1 clips to row 0
        for (int t = 0; t < n_frames; ++t) {
            if (t + 1 < n_frames) onset_row(t + 1, next);
            else std::copy(cur, cur + F, next);  // row n clips to row n - 1
            // peak_thresh_mat[peaks] = onsets[peaks]; np.where(peak_thresh_mat >= onset_thresh)
            const bool zero_passes = 0.0 >= opt.onset_threshold;
            uint8_t hit[kNoteBins];
            bool any = false;
            for (int f = 0; f < F; ++f) {
                const double v = cur[f];
                const bool is_peak = (v > prev[f]) & (v > next[f]);
                hit[f] = is_peak ? (v >= opt.onset_threshold) : zero_passes;
                any |= hit[f] != 0;
            }
            if (any)
                for (int f = 0; f < F; ++f)
                    if (hit[f]) onset_idx.emplace_back(t, f);
            double* recycled = prev;
            prev = cur;
            cur = next;
            next = recycled;
        }
    }
    std::reverse(onset_idx.begin(), onset_idx.end());

    // remaining_energy holds float32 values; every comparison is done in float64 like NumPy.
    const double frame_thresh = opt.frame_threshold;
    std::vector<float> energy(frame, frame + size_t(n_frames) * F);
    auto E = [&](int t, int f) -> float& { return energy[size_t(t) * F + f]; };

    std::vector<RawNote> notes;
    for (auto [start, f] : onset_idx) {
        if (start >= n_frames - 1) continue;
        int i = start + 1, k = 0;
        while (i < n_frames - 1 && k < kEnergyTol) {
            if (double(E(i, f)) < frame_thresh) ++k;
            else k = 0;
            ++i;
        }
        i -= k;
        if (i - start <= min_note_len) continue;
        for (int t = start; t < i; ++t) {
            E(t, f) = 0;
            if (f < kMaxFreqIdx) E(t, f + 1) = 0;
            if (f > 0) E(t, f - 1) = 0;
        }
        notes.push_back({start, i, f + kMidiOffset, column_mean(frame, F, start, i, f)});
    }

    if (opt.melodia_trick) {
        // Repeated np.argmax over the remaining energy == visiting cells by (value desc, flat index
        // asc), because cells only ever get zeroed. Keys: float bits (positive floats sort like
        // integers) in the high word, inverted index in the low word.
        std::vector<uint64_t> order;
        for (size_t i = 0; i < energy.size(); ++i)
            if (double(energy[i]) > frame_thresh) {
                uint32_t bits;
                std::memcpy(&bits, &energy[i], 4);
                order.push_back((uint64_t(bits) << 32) | (0xffffffffu - uint32_t(i)));
            }
        std::sort(order.begin(), order.end(), std::greater<uint64_t>());
        for (uint64_t key : order) {
            const int cell = static_cast<int>(0xffffffffu - uint32_t(key));
            if (energy[cell] == 0.0f) continue;  // zeroed by an earlier note
            const int i_mid = cell / F, f = cell % F;
            E(i_mid, f) = 0;

            int i = i_mid + 1, k = 0;
            while (i < n_frames - 1 && k < kEnergyTol) {
                if (double(E(i, f)) < frame_thresh) ++k;
                else k = 0;
                E(i, f) = 0;
                if (f < kMaxFreqIdx) E(i, f + 1) = 0;
                if (f > 0) E(i, f - 1) = 0;
                ++i;
            }
            const int i_end = i - 1 - k;

            i = i_mid - 1;
            k = 0;
            while (i > 0 && k < kEnergyTol) {
                if (double(E(i, f)) < frame_thresh) ++k;
                else k = 0;
                E(i, f) = 0;
                if (f < kMaxFreqIdx) E(i, f + 1) = 0;
                if (f > 0) E(i, f - 1) = 0;
                --i;
            }
            const int i_start = i + 1 + k;

            if (i_end - i_start <= min_note_len) continue;
            notes.push_back({i_start, i_end, f + kMidiOffset, column_mean(frame, F, i_start, i_end, f)});
        }
    }
    return notes;
}

std::vector<int> pitch_bends(const std::vector<float>& contour, const RawNote& n) {
    constexpr int W = 2 * kBinsTolerance + 1;
    const int freq_idx = 3 * (n.pitch - kMidiOffset);  // round(midi_pitch_to_contour_bin(pitch))
    const int f0 = std::max(freq_idx - kBinsTolerance, 0);
    const int f1 = std::min(kContourBins, freq_idx + kBinsTolerance + 1);
    const int g0 = std::max(0, kBinsTolerance - freq_idx);
    const int shift = kBinsTolerance - std::max(0, kBinsTolerance - freq_idx);
    (void)W;
    std::vector<int> bends;
    bends.reserve(std::max(0, n.end - n.start));
    for (int t = n.start; t < n.end; ++t) {
        const float* row = contour.data() + size_t(t) * kContourBins;
        int best = 0;
        double best_v = -std::numeric_limits<double>::infinity();
        for (int f = f0; f < f1; ++f) {
            const double v = double(row[f]) * kPitchBendGaussian[g0 + (f - f0)];
            if (v > best_v || (std::isnan(v) && !std::isnan(best_v))) {
                best_v = v;
                best = f - f0;
                if (std::isnan(v)) break;  // np.argmax returns the first NaN
            }
        }
        bends.push_back(best - shift);
    }
    return bends;
}

}  // namespace

std::vector<NoteEvent> output_to_notes(const ModelOutput& output, const NoteOptions& opt) {
    const int n = output.n_frames;
    if (n <= 0) return {};
    if (static_cast<long long>(n) * kNoteBins > std::numeric_limits<int32_t>::max())
        throw std::length_error("output_to_notes: too many frames");
    if (output.note.size() != size_t(n) * kNoteBins || output.onset.size() != size_t(n) * kNoteBins ||
        output.contour.size() != size_t(n) * kContourBins)
        throw std::invalid_argument("output_to_notes: matrix sizes do not match n_frames");
    // min_note_len = int(np.round(minimum_note_length / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP)))
    const int min_note_len =
        static_cast<int>(std::nearbyint(opt.minimum_note_length_ms / 1000.0 * (double(kSampleRate) / double(kFftHop))));
    // constrain_frequency() zeroes columns of both matrices (in place in Python); copy only if needed
    std::vector<float> frames_c, onsets_c;
    const float* frames = output.note.data();
    const float* onsets = output.onset.data();
    if (opt.maximum_frequency || opt.minimum_frequency) {
        frames_c = output.note;
        onsets_c = output.onset;
        constexpr int F = kNoteBins;
        auto zero_cols = [&](int f0, int f1) {
            for (int t = 0; t < n; ++t)
                for (int f = f0; f < f1; ++f) onsets_c[size_t(t) * F + f] = frames_c[size_t(t) * F + f] = 0.0f;
        };
        if (opt.maximum_frequency)
            zero_cols(py_slice_index(static_cast<int>(std::nearbyint(hz_to_midi(*opt.maximum_frequency) - kMidiOffset)), F), F);
        if (opt.minimum_frequency)
            zero_cols(0, py_slice_index(static_cast<int>(std::nearbyint(hz_to_midi(*opt.minimum_frequency) - kMidiOffset)), F));
        frames = frames_c.data();
        onsets = onsets_c.data();
    }
    const std::vector<RawNote> raw = output_to_notes_polyphonic(frames, onsets, n, opt, min_note_len);

    std::vector<NoteEvent> events;
    events.reserve(raw.size());
    for (const RawNote& r : raw) {
        NoteEvent e;
        e.start_frame = r.start;
        e.end_frame = r.end;
        e.start_s = frame_time(r.start);
        e.end_s = frame_time(r.end);
        e.pitch = r.pitch;
        e.amplitude = r.amplitude;
        if (opt.include_pitch_bends) e.pitch_bends = pitch_bends(output.contour, r);
        events.push_back(std::move(e));
    }
    return events;
}

}  // namespace bpp
