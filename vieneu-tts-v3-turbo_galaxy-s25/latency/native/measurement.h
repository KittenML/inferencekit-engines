#ifndef VIENEU_BENCH_MEASUREMENT_H_
#define VIENEU_BENCH_MEASUREMENT_H_

#include <aaudio/AAudio.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <memory>
#include <mutex>
#include <semaphore.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <thread>
#include <vector>

namespace vieneu_bench {
namespace detail {

inline int64_t monotonic_ns() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1'000'000'000 + ts.tv_nsec;
}

inline std::string number_or_null(int64_t value) {
    return value >= 0 ? std::to_string(value) : "null";
}

inline std::string ms_or_null(int64_t ns) {
    return ns >= 0 ? std::to_string(ns / 1'000'000.0) : "null";
}

}  // namespace detail

struct MemoryReading {
    int64_t pss_kib = -1;
    int64_t rss_kib = -1;
    int64_t vmhwm_kib = -1;

    std::string json() const {
        using detail::number_or_null;
        return "{\"pss_kib\":" + number_or_null(pss_kib) +
               ",\"rss_kib\":" + number_or_null(rss_kib) +
               ",\"vmhwm_kib\":" + number_or_null(vmhwm_kib) + "}";
    }
};

inline MemoryReading read_memory() {
    MemoryReading reading;
    std::string line;
    std::ifstream rollup("/proc/self/smaps_rollup");
    while (std::getline(rollup, line)) {
        if (line.compare(0, 4, "Pss:") == 0)
            reading.pss_kib = std::strtoll(line.c_str() + 4, nullptr, 10);
        if (line.compare(0, 4, "Rss:") == 0)
            reading.rss_kib = std::strtoll(line.c_str() + 4, nullptr, 10);
        if (reading.pss_kib >= 0 && reading.rss_kib >= 0) break;
    }
    std::ifstream status("/proc/self/status");
    while (std::getline(status, line)) {
        if (line.compare(0, 6, "VmHWM:") == 0) {
            reading.vmhwm_kib = std::strtoll(line.c_str() + 6, nullptr, 10);
            break;
        }
    }
    return reading;
}

struct MemoryStats {
    MemoryReading initial;
    MemoryReading peak;
    MemoryReading current;
    int samples = 0;
    int smaps_samples = 0;

    void observe(const MemoryReading& reading) {
        if (samples == 0) initial = reading;
        peak.pss_kib = std::max(peak.pss_kib, reading.pss_kib);
        peak.rss_kib = std::max(peak.rss_kib, reading.rss_kib);
        peak.vmhwm_kib = std::max(peak.vmhwm_kib, reading.vmhwm_kib);
        current = reading;
        ++samples;
        if (reading.pss_kib >= 0 && reading.rss_kib >= 0) ++smaps_samples;
    }

    std::string json() const {
        return "{\"initial\":" + initial.json() + ",\"peak\":" + peak.json() +
               ",\"current\":" + current.json() + ",\"samples\":" + std::to_string(samples) +
               ",\"smaps_samples\":" + std::to_string(smaps_samples) +
               ",\"interval_ms\":100,\"vmhwm_scope\":\"process_lifetime\"}";
    }
};

class MemorySampler {
public:
    ~MemorySampler() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
        }
        condition_.notify_one();
        if (worker_.joinable()) worker_.join();
    }

    void start() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (worker_.joinable()) throw std::logic_error("memory sampler already started");
        observe();
        worker_ = std::thread([this] {
            setpriority(PRIO_PROCESS, 0, 10);
            std::unique_lock<std::mutex> lock(mutex_);
            while (!condition_.wait_for(lock, std::chrono::milliseconds(100), [this] { return stopped_; })) {
                observe();
            }
        });
    }

    void reset_window() {
        std::lock_guard<std::mutex> lock(mutex_);
        window_ = {};
        observe();
    }

    MemoryStats snapshot() {
        std::lock_guard<std::mutex> lock(mutex_);
        observe();
        return window_;
    }

    MemoryStats lifetime() {
        std::lock_guard<std::mutex> lock(mutex_);
        observe();
        return lifetime_;
    }

private:
    void observe() {
        const auto reading = read_memory();
        window_.observe(reading);
        lifetime_.observe(reading);
    }

    std::thread worker_;
    std::mutex mutex_;
    std::condition_variable condition_;
    bool stopped_ = false;
    MemoryStats window_;
    MemoryStats lifetime_;
};

struct PlaybackStats {
    const char* status = "not_started";
    int aaudio_result = AAUDIO_OK;
    bool start_on_admission = false;
    int source_rate = 0;
    int output_rate = -1;
    int source_channels = 1;
    int output_channels = -1;
    int format = AAUDIO_FORMAT_INVALID;
    int device_id = -1;
    int performance_mode = -1;
    int sharing_mode = -1;
    int frames_per_burst = -1;
    int buffer_size_frames = -1;
    int xruns = -1;
    int64_t first_pcm_ns = -1;
    int64_t stream_start_requested_ns = -1;
    int64_t first_audible_ns = -1;
    int64_t synthesis_complete_ns = -1;
    int64_t playback_complete_ns = -1;
    int64_t source_frames = 0;
    int64_t invalid_source_samples = 0;
    int64_t clipped_source_samples = 0;
    int64_t consumed_pcm_frames = 0;
    int64_t first_pcm_frame = -1;
    int64_t last_pcm_frame = -1;
    int64_t gap_count = 0;
    int64_t gap_frames = 0;
    int64_t onset_frame = -1;
    int64_t onset_timestamp_frame = -1;
    int64_t onset_timestamp_monotonic_ns = -1;
    int64_t request_monotonic_ns = -1;
    int64_t max_timestamp_projection_frames = 0;

    std::string json() const {
        using detail::ms_or_null;
        using detail::number_or_null;
        std::ostringstream out;
        out << "{\"status\":\"" << status << "\",\"aaudio_result\":" << aaudio_result
            << ",\"start_on_admission\":" << (start_on_admission ? "true" : "false")
            << ",\"source_rate\":" << source_rate << ",\"source_channels\":" << source_channels
            << ",\"output_rate\":" << number_or_null(output_rate)
            << ",\"output_channels\":" << number_or_null(output_channels) << ",\"aaudio_format\":" << format
            << ",\"device_id\":" << number_or_null(device_id)
            << ",\"performance_mode\":" << number_or_null(performance_mode)
            << ",\"sharing_mode\":" << number_or_null(sharing_mode)
            << ",\"frames_per_burst\":" << number_or_null(frames_per_burst)
            << ",\"buffer_size_frames\":" << number_or_null(buffer_size_frames)
            << ",\"first_pcm_ms\":" << ms_or_null(first_pcm_ns)
            << ",\"stream_start_requested_ms\":" << ms_or_null(stream_start_requested_ns)
            << ",\"first_audible_ms\":" << ms_or_null(first_audible_ns)
            << ",\"synthesis_complete_ms\":" << ms_or_null(synthesis_complete_ns)
            << ",\"playback_complete_ms\":" << ms_or_null(playback_complete_ns)
            << ",\"source_frames\":" << source_frames
            << ",\"invalid_source_samples\":" << invalid_source_samples
            << ",\"clipped_source_samples\":" << clipped_source_samples
            << ",\"consumed_pcm_frames\":" << consumed_pcm_frames
            << ",\"first_pcm_frame\":" << number_or_null(first_pcm_frame)
            << ",\"last_pcm_frame\":" << number_or_null(last_pcm_frame)
            << ",\"app_gap_count\":" << gap_count << ",\"app_gap_frames\":" << gap_frames
            << ",\"app_gap_ms\":" << ms_or_null(output_rate > 0 ? gap_frames * 1'000'000'000 / output_rate : -1)
            << ",\"device_xruns\":" << number_or_null(xruns)
            << ",\"onset_frame\":" << number_or_null(onset_frame)
            << ",\"onset_timestamp_frame\":" << number_or_null(onset_timestamp_frame)
            << ",\"onset_timestamp_monotonic_ns\":" << number_or_null(onset_timestamp_monotonic_ns)
            << ",\"request_monotonic_ns\":" << number_or_null(request_monotonic_ns)
            << ",\"max_timestamp_projection_frames\":" << max_timestamp_projection_frames
            << ",\"onset_window_ms\":5,\"onset_rms_dbfs\":-45}";
        return out.str();
    }
};

// Construct before request release so opening/configuring the common output path
// is outside the timing interval. The default starts playback on the first PCM.
class Playback {
public:
    explicit Playback(int source_rate, bool start_on_admission = false) {
        if (source_rate != 24'000 && source_rate != kRate) {
            throw std::invalid_argument("playback requires 24 kHz or 48 kHz mono source");
        }
        stats_.source_rate = source_rate;
        stats_.start_on_admission = start_on_admission;
        stats_.max_timestamp_projection_frames = kMaxProjectionFrames;
        if (sem_init(&ready_, 0, 0) != 0) throw std::runtime_error("sem_init failed");
        open_stream();
        if (stream_ && stats_.aaudio_result == AAUDIO_OK) {
            try {
                worker_ = std::thread([this] { present(); });
            } catch (...) {
                close_stream();
                sem_destroy(&ready_);
                throw;
            }
        }
    }

    ~Playback() {
        if (worker_.joinable()) {
            cancelled_.store(true);
            sem_post(&ready_);
            worker_.join();
        }
        close_stream();
        sem_destroy(&ready_);
        Node* node = head_.next.load();
        while (node) {
            Node* next = node->next.load();
            delete node;
            node = next;
        }
    }

    Playback(const Playback&) = delete;
    Playback& operator=(const Playback&) = delete;

    void start() {
        if (stats_.request_monotonic_ns >= 0) throw std::logic_error("playback already started");
        stats_.request_monotonic_ns = detail::monotonic_ns();
        if (stats_.start_on_admission && worker_.joinable()) sem_post(&ready_);
    }

    // Single producer. Work on this path is bounded by the submitted packet.
    // The consumer never locks or allocates.
    void emit(const float* pcm, size_t n) {
        if (!n) return;
        if (stats_.request_monotonic_ns < 0 || done_.load()) {
            throw std::logic_error("PCM emitted outside request");
        }
        const bool first = stats_.source_frames == 0;
        if (first) first_pcm_ns_.store(detail::monotonic_ns() - stats_.request_monotonic_ns);
        stats_.source_frames += n;
        if (!worker_.joinable() || stream_error_.load() != AAUDIO_OK) return;
        auto node = std::make_unique<Node>();
        auto sanitize = [this](float sample) {
            if (!std::isfinite(sample)) {
                ++stats_.invalid_source_samples;
                return 0.0f;
            }
            if (std::abs(sample) > 1.0f) ++stats_.clipped_source_samples;
            return std::clamp(sample, -1.0f, 1.0f);
        };
        if (stats_.source_rate == kRate) {
            node->samples.resize(n);
            std::transform(pcm, pcm + n, node->samples.begin(), sanitize);
        } else {
            node->samples.resize(n * 2);
            for (size_t i = 0; i < n; ++i) {
                const float sample = sanitize(pcm[i]);
                if (first && i == 0) previous_sample_ = sample;
                node->samples[2 * i] = (previous_sample_ + sample) * 0.5f;
                node->samples[2 * i + 1] = sample;
                previous_sample_ = sample;
            }
        }
        published_frames_ += node->samples.size();
        Node* published = node.release();
        tail_->next.store(published, std::memory_order_release);
        tail_ = published;
        if (first && !stats_.start_on_admission) sem_post(&ready_);
    }

    // Records synthesis completion before waiting for the playback tail.
    PlaybackStats finish() {
        if (stats_.request_monotonic_ns < 0 || done_.load()) {
            throw std::logic_error("finish requires an active request");
        }
        const int64_t completed_ns = detail::monotonic_ns();
        stats_.synthesis_complete_ns = completed_ns - stats_.request_monotonic_ns;
        const auto pending = std::max<int64_t>(0, published_frames_ - consumed_frames_.load());
        drain_deadline_ns_ = completed_ns + frames_to_ns(pending) + 3'000'000'000LL;
        done_.store(true, std::memory_order_release);
        if (!stats_.start_on_admission && !head_.next.load(std::memory_order_acquire)) sem_post(&ready_);
        if (worker_.joinable()) worker_.join();
        close_stream();
        stats_.first_pcm_ns = first_pcm_ns_.load();
        stats_.consumed_pcm_frames = consumed_frames_.load();
        stats_.first_pcm_frame = first_pcm_frame_.load();
        stats_.last_pcm_frame = last_pcm_frame_.load();
        stats_.gap_count = gap_count_;
        stats_.gap_frames = gap_frames_;
        stats_.onset_frame = onset_frame_.load();
        return stats_;
    }

private:
    static constexpr int kRate = 48'000;
    static constexpr int kOnsetWindow = kRate * 5 / 1'000;
    static constexpr double kMeanSquareThreshold = 0.000031622776601684;
    static constexpr int kMaxProjectionFrames = kRate / 4;

    struct Node {
        std::vector<float> samples;
        std::atomic<Node*> next{nullptr};
    };

    static int64_t frames_to_ns(int64_t frames) {
        return frames * 1'000'000'000 / kRate;
    }

    static aaudio_data_callback_result_t audio_callback(AAudioStream*, void* user, void* data,
                                                        int32_t frames) {
        auto& self = *static_cast<Playback*>(user);
        self.fill(static_cast<float*>(data), frames);
        return AAUDIO_CALLBACK_RESULT_CONTINUE;
    }

    static void error_callback(AAudioStream*, void* user, aaudio_result_t result) {
        static_cast<Playback*>(user)->stream_error_.store(result);
    }

    void open_stream() {
        AAudioStreamBuilder* builder = nullptr;
        stats_.aaudio_result = AAudio_createStreamBuilder(&builder);
        if (stats_.aaudio_result != AAUDIO_OK) {
            stats_.status = "builder_error";
            return;
        }
        AAudioStreamBuilder_setDirection(builder, AAUDIO_DIRECTION_OUTPUT);
        AAudioStreamBuilder_setSharingMode(builder, AAUDIO_SHARING_MODE_SHARED);
        AAudioStreamBuilder_setPerformanceMode(builder, AAUDIO_PERFORMANCE_MODE_LOW_LATENCY);
        AAudioStreamBuilder_setUsage(builder, AAUDIO_USAGE_MEDIA);
        AAudioStreamBuilder_setContentType(builder, AAUDIO_CONTENT_TYPE_SPEECH);
        AAudioStreamBuilder_setSampleRate(builder, kRate);
        AAudioStreamBuilder_setChannelCount(builder, 1);
        AAudioStreamBuilder_setFormat(builder, AAUDIO_FORMAT_PCM_FLOAT);
        AAudioStreamBuilder_setDataCallback(builder, audio_callback, this);
        AAudioStreamBuilder_setErrorCallback(builder, error_callback, this);
        stats_.aaudio_result = AAudioStreamBuilder_openStream(builder, &stream_);
        AAudioStreamBuilder_delete(builder);
        if (stats_.aaudio_result != AAUDIO_OK) {
            stats_.status = "open_error";
            return;
        }
        stats_.output_rate = AAudioStream_getSampleRate(stream_);
        stats_.output_channels = AAudioStream_getChannelCount(stream_);
        stats_.format = AAudioStream_getFormat(stream_);
        stats_.device_id = AAudioStream_getDeviceId(stream_);
        stats_.performance_mode = AAudioStream_getPerformanceMode(stream_);
        stats_.sharing_mode = AAudioStream_getSharingMode(stream_);
        stats_.frames_per_burst = AAudioStream_getFramesPerBurst(stream_);
        if (stats_.frames_per_burst > 0) {
            AAudioStream_setBufferSizeInFrames(stream_, 2 * stats_.frames_per_burst);
        }
        stats_.buffer_size_frames = AAudioStream_getBufferSizeInFrames(stream_);
        if (stats_.output_rate != kRate || stats_.output_channels != 1 || stats_.format != AAUDIO_FORMAT_PCM_FLOAT) {
            stats_.aaudio_result = AAUDIO_ERROR_INVALID_FORMAT;
            stats_.status = "unexpected_output_format";
            close_stream();
        }
    }

    void close_stream() {
        if (stream_) {
            AAudioStream_close(stream_);
            stream_ = nullptr;
        }
    }

    void fill(float* output, int32_t frames) {
        int written = 0;
        while (written < frames) {
            if (head_pos_ == head_read_->samples.size()) {
                Node* next = head_read_->next.load(std::memory_order_acquire);
                if (!next) break;
                head_read_ = next;
                head_pos_ = 0;
            }
            const size_t count = std::min(head_read_->samples.size() - head_pos_,
                                          static_cast<size_t>(frames - written));
            std::copy_n(head_read_->samples.data() + head_pos_, count, output + written);
            head_pos_ += count;
            written += static_cast<int>(count);
        }
        if (written > 0) {
            if (consumed_frames_.load(std::memory_order_relaxed) == 0) {
                first_pcm_frame_.store(device_frames_, std::memory_order_relaxed);
            }
            if (pending_gap_ > 0) {
                ++gap_count_;
                gap_frames_ += pending_gap_;
                pending_gap_ = 0;
            }
            last_pcm_frame_.store(device_frames_ + written - 1, std::memory_order_relaxed);
            consumed_frames_.fetch_add(written, std::memory_order_release);
        }
        if (written < frames) {
            std::fill(output + written, output + frames, 0.0f);
            if (consumed_frames_.load(std::memory_order_relaxed) > 0) pending_gap_ += frames - written;
        }
        const auto first_pcm_frame = first_pcm_frame_.load(std::memory_order_relaxed);
        if (first_pcm_frame >= 0 && onset_frame_.load(std::memory_order_relaxed) < 0) {
            for (int i = 0; i < frames; ++i) {
                const double x = output[i];
                window_sum_ += x * x - energy_window_[window_pos_];
                energy_window_[window_pos_] = x * x;
                window_pos_ = (window_pos_ + 1) % kOnsetWindow;
                if (device_frames_ + i - first_pcm_frame >= kOnsetWindow - 1 &&
                    window_sum_ >= kOnsetWindow * kMeanSquareThreshold) {
                    onset_frame_.store(device_frames_ + i - kOnsetWindow + 1);
                    break;
                }
            }
        }
        device_frames_ += frames;
    }

    void present() {
        int wait_result;
        do {
            wait_result = sem_wait(&ready_);
        } while (wait_result != 0 && errno == EINTR);
        if (wait_result != 0) {
            stats_.status = "semaphore_error";
            return;
        }
        if (cancelled_.load()) return;
        if ((!stats_.start_on_admission || done_.load(std::memory_order_acquire)) &&
                !head_.next.load(std::memory_order_acquire)) {
            stats_.aaudio_result = stream_error_.load();
            stats_.status = stats_.aaudio_result == AAUDIO_OK ? "no_pcm" : "stream_error";
            return;
        }

        const int initial_xruns = AAudioStream_getXRunCount(stream_);
        stats_.stream_start_requested_ns = detail::monotonic_ns() - stats_.request_monotonic_ns;
        stats_.aaudio_result = AAudioStream_requestStart(stream_);
        if (stats_.aaudio_result != AAUDIO_OK) {
            stats_.status = "start_error";
            return;
        }
        while (!cancelled_.load()) {
            if (const auto error = stream_error_.load(); error != AAUDIO_OK) {
                stats_.status = "stream_error";
                stats_.aaudio_result = error;
                break;
            }
            int64_t ts_frame = -1;
            int64_t ts_ns = -1;
            const bool timestamp = AAudioStream_getTimestamp(stream_, CLOCK_MONOTONIC, &ts_frame, &ts_ns) ==
                                   AAUDIO_OK && ts_frame >= 0 && ts_ns > 0;
            const int64_t onset = onset_frame_.load();
            const int64_t first_pcm = first_pcm_ns_.load();
            if (timestamp && stats_.first_audible_ns < 0 && onset >= 0 && ts_frame >= onset &&
                    ts_frame - onset <= kMaxProjectionFrames && first_pcm >= 0) {
                const auto estimate = ts_ns - frames_to_ns(ts_frame - onset) - stats_.request_monotonic_ns;
                if (estimate >= first_pcm) {
                    stats_.first_audible_ns = estimate;
                    stats_.onset_timestamp_frame = ts_frame;
                    stats_.onset_timestamp_monotonic_ns = ts_ns;
                }
            }
            if (done_.load(std::memory_order_acquire)) {
                if (published_frames_ == 0) {
                    stats_.status = "no_pcm";
                    break;
                }
                const auto consumed = consumed_frames_.load(std::memory_order_acquire);
                const auto last = last_pcm_frame_.load(std::memory_order_relaxed);
                if (timestamp && consumed == published_frames_ && last >= 0 && ts_frame > last) {
                    const auto estimate = ts_ns - frames_to_ns(ts_frame - last - 1) - stats_.request_monotonic_ns;
                    if (estimate >= first_pcm && first_pcm >= 0 && ts_frame - last - 1 <= kMaxProjectionFrames)
                        stats_.playback_complete_ns = estimate;
                    if (onset < 0) stats_.status = "onset_threshold_not_reached";
                    else if (stats_.first_audible_ns < 0) stats_.status = "onset_timestamp_unavailable";
                    else if (stats_.playback_complete_ns < 0) stats_.status = "tail_timestamp_unavailable";
                    else stats_.status = "ok";
                    break;
                }
                if (detail::monotonic_ns() > drain_deadline_ns_) {
                    stats_.status = consumed == published_frames_ ? "drain_timestamp_timeout" : "drain_pcm_timeout";
                    break;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(4));
        }
        const int final_xruns = AAudioStream_getXRunCount(stream_);
        if (initial_xruns >= 0 && final_xruns >= initial_xruns) stats_.xruns = final_xruns - initial_xruns;
        stats_.device_id = AAudioStream_getDeviceId(stream_);
        AAudioStream_requestStop(stream_);
    }

    PlaybackStats stats_;
    AAudioStream* stream_ = nullptr;
    std::thread worker_;
    sem_t ready_{};
    std::atomic<bool> cancelled_{false};
    std::atomic<bool> done_{false};
    std::atomic<int> stream_error_{AAUDIO_OK};
    std::atomic<int64_t> first_pcm_ns_{-1};
    int64_t drain_deadline_ns_ = 0;

    Node head_;
    Node* tail_ = &head_;
    int64_t published_frames_ = 0;
    float previous_sample_ = 0.0f;

    Node* head_read_ = &head_;
    size_t head_pos_ = 0;
    int64_t device_frames_ = 0;
    int64_t pending_gap_ = 0;
    int64_t gap_count_ = 0;
    int64_t gap_frames_ = 0;
    std::atomic<int64_t> consumed_frames_{0};
    std::atomic<int64_t> first_pcm_frame_{-1};
    std::atomic<int64_t> last_pcm_frame_{-1};
    std::atomic<int64_t> onset_frame_{-1};
    std::array<double, kOnsetWindow> energy_window_{};
    double window_sum_ = 0;
    size_t window_pos_ = 0;
};

}  // namespace vieneu_bench
#endif  // VIENEU_BENCH_MEASUREMENT_H_
