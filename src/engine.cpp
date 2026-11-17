// Windowing, multi-threaded whole-file inference, and streaming inference.
#include "bpp/bpp.h"

#include "constants.h"
#include "resample.h"
#include "wav.h"
#include "window_model.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>

namespace bpp {

namespace {

using detail::FloatBuffer;
using detail::WindowModel;
using detail::Workspace;

// basic-pitch prepends half the overlap as zeros before windowing.
constexpr int kLeadPad = kOverlapSamples / 2;  // 3840

// int(np.floor(n_samples * (ANNOTATIONS_FPS / AUDIO_SAMPLE_RATE)))
int output_frame_count(size_t n_samples) {
    return static_cast<int>(std::floor(static_cast<double>(n_samples) * (double(kAnnotationsFps) / kSampleRate)));
}

// Windows start every kWindowHop samples of the padded signal: range(0, n + 3840, hop).
int window_count(size_t n_samples) {
    const size_t padded = n_samples + kLeadPad;
    return static_cast<int>((padded + kWindowHop - 1) / kWindowHop);
}

// Windows whose kept frames survive the final trim to output_frame_count().
int useful_window_count(size_t n_samples) {
    const int frames = output_frame_count(n_samples);
    return std::min(window_count(n_samples), (frames + kUsedFrames - 1) / kUsedFrames);
}

// One past the last audio sample window w reads.
size_t window_end_sample(int w, size_t n_samples) {
    const long long end = static_cast<long long>(w) * kWindowHop - kLeadPad + kWindowSamples;
    return static_cast<size_t>(std::clamp<long long>(end, 0, static_cast<long long>(n_samples)));
}

// Copies window w of the padded signal (3840 zeros + audio, zero-extended) into dst.
void extract_window(const float* audio, size_t n_samples, int w, float* dst) {
    const long long start = static_cast<long long>(w) * kWindowHop - kLeadPad;  // in audio coordinates
    for (int i = 0; i < kWindowSamples; ++i) {
        const long long s = start + i;
        dst[i] = (s >= 0 && s < static_cast<long long>(n_samples)) ? audio[s] : 0.0f;
    }
}

void resize_output(ModelOutput& out, int frames) {
    out.n_frames = frames;
    out.note.resize(size_t(frames) * kNoteBins);
    out.onset.resize(size_t(frames) * kNoteBins);
    out.contour.resize(size_t(frames) * kContourBins);
}

// Runs window w and stores its kept frames at frame offset w * 142, clipped to out.n_frames.
void run_window(const float* window, int w, Workspace& ws, ModelOutput& out, float* tmp,
                const detail::RowRunner* runner = nullptr) {
    const int first = w * kUsedFrames;
    const int rows = std::min(kUsedFrames, out.n_frames - first);
    if (rows <= 0) return;
    float* contour = tmp;
    float* note = contour + kUsedFrames * kContourBins;
    float* onset = note + kUsedFrames * kNoteBins;
    WindowModel::get().run(window, ws, contour, note, onset, runner);
    std::memcpy(out.contour.data() + size_t(first) * kContourBins, contour, sizeof(float) * rows * kContourBins);
    std::memcpy(out.note.data() + size_t(first) * kNoteBins, note, sizeof(float) * rows * kNoteBins);
    std::memcpy(out.onset.data() + size_t(first) * kNoteBins, onset, sizeof(float) * rows * kNoteBins);
}

constexpr size_t kTmpFloats = size_t(kUsedFrames) * (kContourBins + 2 * kNoteBins);

// Per-thread state reused across windows and calls.
struct ThreadState {
    Workspace ws;
    FloatBuffer window{size_t(kWindowSamples)};
    FloatBuffer tmp{kTmpFloats};
};

}  // namespace

// ---------------------------------------------------------------------------------------------

struct Engine::Impl {
    int n_threads = 1;

    // Runs every window; wait_for(k), if given, blocks until audio[0, k) is available.
    ModelOutput run(const float* audio, size_t n_samples, const std::function<void(size_t)>* wait_for);
    std::mutex mu;
    std::vector<std::unique_ptr<ThreadState>> idle;  // cached scratch memory

    std::unique_ptr<ThreadState> acquire() {
        {
            std::lock_guard<std::mutex> lock(mu);
            if (!idle.empty()) {
                auto s = std::move(idle.back());
                idle.pop_back();
                return s;
            }
        }
        return std::make_unique<ThreadState>();
    }
    void release(std::unique_ptr<ThreadState> s) {
        std::lock_guard<std::mutex> lock(mu);
        idle.push_back(std::move(s));
    }
};

Engine::Engine(int n_threads) : impl_(std::make_unique<Impl>()) {
    const int hw = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    impl_->n_threads = n_threads > 0 ? n_threads : hw;
    WindowModel::get();  // build packed weights up front
}

Engine::~Engine() = default;

int Engine::threads() const { return impl_->n_threads; }

ModelOutput Engine::infer(const float* audio, size_t n_samples) const { return impl_->run(audio, n_samples, nullptr); }

ModelOutput Engine::infer_file(const std::string& path) const {
    detail::WavAudio wav = detail::decode_wav(path);
    if (wav.sample_rate == static_cast<uint32_t>(kSampleRate)) return infer(wav.mono.data(), wav.mono.size());

    // Resample on a producer thread; each window starts as soon as its samples exist.
    const size_t n = detail::resampled_length(wav.mono.size(), wav.sample_rate);
    std::vector<float> audio(detail::resampler_capacity(wav.mono.size(), wav.sample_rate), 0.0f);
    std::mutex mu;
    std::condition_variable cv;
    size_t ready = 0;
    std::exception_ptr error;
    auto produce = [&] {
        try {
            detail::resample_to_model_rate(wav.mono.data(), wav.mono.size(), wav.sample_rate, audio.data(),
                                           [&](size_t k) {
                                               std::lock_guard<std::mutex> lock(mu);
                                               ready = k;
                                               cv.notify_all();
                                           });
        } catch (...) {
            std::lock_guard<std::mutex> lock(mu);
            error = std::current_exception();
            ready = n;  // release waiting workers
            cv.notify_all();
        }
    };
    const std::function<void(size_t)> wait_for = [&](size_t k) {
        std::unique_lock<std::mutex> lock(mu);
        cv.wait(lock, [&] { return ready >= k; });
    };
    std::thread producer(produce);
    ModelOutput out;
    try {
        out = impl_->run(audio.data(), n, &wait_for);
    } catch (...) {
        producer.join();
        throw;
    }
    producer.join();
    if (error) std::rethrow_exception(error);
    return out;
}

ModelOutput Engine::Impl::run(const float* audio, size_t n_samples, const std::function<void(size_t)>* wait_for) {
    ModelOutput out;
    resize_output(out, output_frame_count(n_samples));
    const int n_windows = useful_window_count(n_samples);
    if (n_windows <= 0) return out;

    std::atomic<int> next{0};
    std::mutex error_mu;
    std::exception_ptr error;
    auto worker = [&] {
        try {
            auto state = acquire();
            for (int w; (w = next.fetch_add(1)) < n_windows;) {
                if (wait_for) (*wait_for)(window_end_sample(w, n_samples));
                extract_window(audio, n_samples, w, state->window.data());
                run_window(state->window.data(), w, state->ws, out, state->tmp.data());
            }
            release(std::move(state));
        } catch (...) {
            std::lock_guard<std::mutex> lock(error_mu);
            if (!error) error = std::current_exception();
            next.store(n_windows);  // stop the other workers early
        }
    };
    const int n_workers = std::min(n_threads, n_windows);
    std::vector<std::thread> pool;
    pool.reserve(n_workers - 1);
    try {
        for (int i = 1; i < n_workers; ++i) pool.emplace_back(worker);
    } catch (...) {
        // could not start every thread: the started ones and this one finish the job
    }
    worker();
    for (auto& t : pool) t.join();
    if (error) std::rethrow_exception(error);
    return out;
}

// ---------------------------------------------------------------------------------------------

namespace {

// Fixed set of helper threads that split one window's row loops; used by Stream for low latency.
class RowPool {
public:
    explicit RowPool(int n_threads) {
        try {
            for (int i = 1; i < n_threads; ++i) threads_.emplace_back([this, i] { loop(i); });
        } catch (...) {
            shutdown();
            throw;
        }
        n_ = n_threads;
    }
    ~RowPool() { shutdown(); }

    void run(int begin, int end, const std::function<void(int, int)>& fn) {
        if (n_ <= 1 || end - begin < 2 * n_) {
            fn(begin, end);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mu_);
            fn_ = &fn;
            begin_ = begin;
            end_ = end;
            pending_ = n_ - 1;
            ++generation_;
        }
        cv_.notify_all();
        slice(0, fn);
        std::unique_lock<std::mutex> lock(mu_);
        done_cv_.wait(lock, [this] { return pending_ == 0; });
    }

private:
    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_ = true;
            ++generation_;
        }
        cv_.notify_all();
        for (auto& t : threads_) t.join();
        threads_.clear();
    }

    void slice(int idx, const std::function<void(int, int)>& fn) const {
        const int span = end_ - begin_;
        const int b = begin_ + span * idx / n_, e = begin_ + span * (idx + 1) / n_;
        if (b < e) fn(b, e);
    }
    void loop(int idx) {
        unsigned long long seen = 0;
        for (;;) {
            const std::function<void(int, int)>* fn;
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_.wait(lock, [&] { return generation_ != seen; });
                seen = generation_;
                if (stop_) return;
                fn = fn_;
            }
            slice(idx, *fn);
            {
                std::lock_guard<std::mutex> lock(mu_);
                if (--pending_ == 0) done_cv_.notify_one();
            }
        }
    }

    std::vector<std::thread> threads_;
    int n_ = 1;
    std::mutex mu_;
    std::condition_variable cv_, done_cv_;
    const std::function<void(int, int)>* fn_ = nullptr;
    int begin_ = 0, end_ = 0, pending_ = 0;
    unsigned long long generation_ = 0;
    bool stop_ = false;
};

}  // namespace

struct Stream::Impl {
    explicit Impl(int threads) : n_threads(std::clamp(threads, 1, kMaxStreamThreads)), pool(n_threads) {
        runner = [this](int b, int e, const std::function<void(int, int)>& fn) { pool.run(b, e, fn); };
        buffer.assign(kLeadPad, 0.0f);
    }

    static constexpr int kMaxStreamThreads = 64;
    const int n_threads;
    ThreadState state;
    RowPool pool;
    detail::RowRunner runner;
    std::vector<float> buffer;    // padded-signal samples from window `next_window` onwards
    long long buffer_start = 0;   // padded-signal index of buffer[0]
    long long padded_len = kLeadPad;  // samples of the padded signal seen so far
    size_t n_samples = 0;
    int next_window = 0;
    bool finished = false;
    ModelOutput out;  // every computed frame; trimmed to basic-pitch's length by finish()

    void process(int w) {
        const long long start = static_cast<long long>(w) * kWindowHop;
        float* win = state.window.data();
        for (int i = 0; i < kWindowSamples; ++i) {
            const long long idx = start + i - buffer_start;
            win[i] = idx < static_cast<long long>(buffer.size()) ? buffer[idx] : 0.0f;
        }
        float* contour = state.tmp.data();
        float* note = contour + kUsedFrames * kContourBins;
        float* onset = note + kUsedFrames * kNoteBins;
        WindowModel::get().run(win, state.ws, contour, note, onset, n_threads > 1 ? &runner : nullptr);
        out.n_frames += kUsedFrames;
        out.contour.insert(out.contour.end(), contour, contour + kUsedFrames * kContourBins);
        out.note.insert(out.note.end(), note, note + kUsedFrames * kNoteBins);
        out.onset.insert(out.onset.end(), onset, onset + kUsedFrames * kNoteBins);

        // drop samples no later window needs
        const long long keep_from = static_cast<long long>(w + 1) * kWindowHop;
        const long long drop = std::min<long long>(keep_from - buffer_start, static_cast<long long>(buffer.size()));
        if (drop > 0) {
            buffer.erase(buffer.begin(), buffer.begin() + drop);
            buffer_start += drop;
        }
    }
};

Stream::Stream(int n_threads) : impl_(std::make_unique<Impl>(n_threads)) {}

Stream::~Stream() = default;

int Stream::push(const float* audio, size_t n) {
    Impl& s = *impl_;
    if (s.finished) throw std::logic_error("Stream::push after finish()");
    const int before = s.out.n_frames;
    s.buffer.insert(s.buffer.end(), audio, audio + n);
    s.n_samples += n;
    s.padded_len += static_cast<long long>(n);
    while (static_cast<long long>(s.next_window) * kWindowHop + kWindowSamples <= s.padded_len)
        s.process(s.next_window++);
    return s.out.n_frames - before;
}

int Stream::finish() {
    Impl& s = *impl_;
    if (s.finished) return 0;
    s.finished = true;
    const int before = s.out.n_frames;
    const int n_windows = useful_window_count(s.n_samples);
    while (s.next_window < n_windows) s.process(s.next_window++);
    // basic-pitch keeps floor(n_samples * 86 / 22050) frames; the unwrapped grid runs slightly ahead
    resize_output(s.out, std::min(s.out.n_frames, output_frame_count(s.n_samples)));
    s.buffer.clear();
    s.buffer.shrink_to_fit();
    return s.out.n_frames - before;
}

const ModelOutput& Stream::output() const { return impl_->out; }

int Stream::final_frames() const {
    const Impl& s = *impl_;
    return s.finished ? s.out.n_frames : std::min(s.out.n_frames, output_frame_count(s.n_samples));
}

size_t Stream::samples_pushed() const { return impl_->n_samples; }

int Stream::windows_processed() const { return impl_->next_window; }

}  // namespace bpp
