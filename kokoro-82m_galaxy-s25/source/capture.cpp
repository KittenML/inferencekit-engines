#include <audiocpp.h>
#include <sherpa-onnx/c-api/c-api.h>

#include "kokoro_pcm_process.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <sched.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#ifdef __ANDROID__
#include <android/thermal.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;
using Milliseconds = std::chrono::duration<double, std::milli>;

bool enabled(const char *name) {
    const char *value = std::getenv(name);
    return value && std::string(value) == "1";
}

struct Fixture {
    std::string id;
    std::array<std::string, 2> sentences;
};

struct Chunk {
    double arrival_ms;
    size_t frames;
    size_t leading_silent_frames;
    std::vector<int16_t> pcm;
};

struct Observation {
    std::string id;
    std::vector<Chunk> chunks;
    double complete_ms = 0.0;
    std::string error;
};

std::vector<Fixture> load_fixtures(const char *path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("could not open fixtures");
    }
    std::vector<Fixture> fixtures;
    for (std::string line; std::getline(file, line);) {
        std::istringstream row(line);
        Fixture fixture;
        if (!std::getline(row, fixture.id, '\t') ||
            !std::getline(row, fixture.sentences[0], '\t') ||
            !std::getline(row, fixture.sentences[1], '\t') ||
            fixture.id.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789-") != std::string::npos) {
            throw std::runtime_error("invalid fixture row");
        }
        fixtures.emplace_back(std::move(fixture));
    }
    return fixtures;
}

Chunk pcm_chunk(Clock::time_point release, int expected_rate, const float *samples,
                size_t frames, int rate, int channels, bool keep_pcm) {
    if (!samples || !frames || rate != expected_rate || channels < 1 || channels > 2) {
        throw std::runtime_error("unexpected waveform layout");
    }
    const auto returned = Clock::now();
    Chunk chunk{Milliseconds(returned - release).count(), frames, frames, {}};
    if (keep_pcm) {
        chunk.pcm.resize(frames * channels);
    }
    for (size_t frame = 0; frame < frames; ++frame) {
        for (int channel = 0; channel < channels; ++channel) {
            const size_t index = frame * channels + channel;
            const float sample = samples[index];
            if (!std::isfinite(sample)) {
                throw std::runtime_error("nonfinite waveform sample");
            }
            const long value = std::lround(std::clamp(sample, -1.0f, 1.0f) * 32767.0f);
            if (value != 0 && chunk.leading_silent_frames == frames) {
                chunk.leading_silent_frames = frame;
            }
            if (keep_pcm) {
                chunk.pcm[index] = static_cast<int16_t>(value);
            }
        }
    }
    if (chunk.leading_silent_frames == frames) {
        throw std::runtime_error("silent waveform");
    }
    return chunk;
}

[[maybe_unused]] Chunk pcm_chunk(Clock::time_point release, std::vector<int16_t> pcm, bool keep) {
    const auto arrived = Clock::now();
    const auto signal = std::find_if(pcm.begin(), pcm.end(), [](int16_t sample) { return sample != 0; });
    if (signal == pcm.end()) {
        throw std::runtime_error("silent waveform from the Kokoro PCM worker");
    }
    Chunk chunk{Milliseconds(arrived - release).count(), pcm.size(),
                static_cast<size_t>(signal - pcm.begin()), {}};
    if (keep) {
        chunk.pcm = std::move(pcm);
    }
    return chunk;
}

void check(audiocpp_status status) {
    if (status != AUDIOCPP_OK) {
        throw std::runtime_error(audiocpp_last_error());
    }
}

class Engines {
public:
    Engines(bool kokoro, std::string model, int threads)
        : kokoro_(kokoro), sherpa_(kokoro && !enabled("KOKORO_AUDIOCPP")),
          adreno_(kokoro && enabled("KOKORO_ADRENO")),
          model_(std::move(model)), threads_(threads) {
        if (adreno_) {
            std::signal(SIGPIPE, SIG_IGN);
            return;
        }
        if (sherpa_) {
            const char *provider = std::getenv("KOKORO_PROVIDER");
            provider_ = provider ? provider : "cpu";
            if (enabled("KOKORO_SHARED_SESSION")) {
                shared_kokoro_ = kokoro_worker();
            }
            std::cerr << "kokoro_provider=" << provider_
                      << " sherpa_onnx=" << SherpaOnnxGetVersionStr()
                      << " onnxruntime=" << SherpaOnnxGetOnnxruntimeVersionStr()
                      << " kokoro_shared_session=" << bool(shared_kokoro_) << '\n';
            return;
        }
        const auto slash = model_.find_last_of('/');
        directory_ = model_.substr(0, slash + 1);
        if (!kokoro_) {
            // Rust G2P registers TLS destructors on inference workers.
            g2p_library_.reset(dlopen((directory_ + "libsea_g2p_rs.so").c_str(),
                                      RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE));
            if (!g2p_library_) {
                const char *error = dlerror();
                throw std::runtime_error(error ? error : "could not load SEA-G2P");
            }
        }
        audiocpp_registry *registry = nullptr;
        check(audiocpp_registry_create(nullptr, &registry));
        registry_.reset(registry);
        audiocpp_model_config config{kokoro_ ? "kokoro_tts" : "vieneu_v3_turbo", nullptr, nullptr, nullptr};
        audiocpp_model *loaded = nullptr;
        check(audiocpp_model_load(registry_.get(), model_.c_str(), &config, nullptr, &loaded));
        model_handle_.reset(loaded);
    }

    using Generate = std::function<Chunk(const std::string &, Clock::time_point, bool)>;

    Generate worker() const {
        if (adreno_) {
            const std::vector<std::string> command{
                "./Kokoro_82M_inference_fp16", "--serve-stream", "--voice", "en-us",
            };
            const auto engine = std::make_shared<KokoroPcmProcess>(model_, command);
            return [engine](const std::string &text, Clock::time_point release, bool keep) {
                return pcm_chunk(release, engine->synthesize(text), keep);
            };
        }
        if (sherpa_) {
            return shared_kokoro_ ? shared_kokoro_ : kokoro_worker();
        }
        return audiocpp_worker();
    }

private:
    Generate kokoro_worker() const {
        const auto slash = model_.find_last_of('/');
        const std::string directory = model_.substr(0, slash + 1);
        const std::string voices = directory + "voices.bin";
        const std::string tokens = directory + "tokens.txt";
        const std::string g2p = directory + "espeak-ng-data";
        SherpaOnnxOfflineTtsConfig config{};
        config.model.kokoro.model = model_.c_str();
        config.model.kokoro.voices = voices.c_str();
        config.model.kokoro.tokens = tokens.c_str();
        config.model.kokoro.data_dir = g2p.c_str();
        config.model.kokoro.lang = "en-us";
        config.model.num_threads = threads_;
        config.model.provider = provider_.c_str();
        config.max_num_sentences = 1;
        config.silence_scale = 0.2f;
        const std::shared_ptr<const SherpaOnnxOfflineTts> tts(
            SherpaOnnxCreateOfflineTts(&config), SherpaOnnxDestroyOfflineTts);
        if (!tts) {
            throw std::runtime_error("sherpa-onnx failed to load Kokoro");
        }
        return [tts](const std::string &text, Clock::time_point release, bool keep) {
            SherpaOnnxGenerationConfig generation{};
            generation.sid = 3;
            generation.speed = 1.0f;
            generation.silence_scale = 0.2f;
            const std::unique_ptr<const SherpaOnnxGeneratedAudio, decltype(&SherpaOnnxDestroyOfflineTtsGeneratedAudio)>
                audio(SherpaOnnxOfflineTtsGenerateWithConfig(tts.get(), text.c_str(), &generation, nullptr, nullptr),
                      SherpaOnnxDestroyOfflineTtsGeneratedAudio);
            if (!audio || audio->n <= 0) {
                throw std::runtime_error("Kokoro returned no audio");
            }
            return pcm_chunk(release, 24000, audio->samples, static_cast<size_t>(audio->n), audio->sample_rate, 1, keep);
        };
    }

    Generate audiocpp_worker() const {
        const std::unique_ptr<audiocpp_options, decltype(&audiocpp_options_free)>
            options(audiocpp_options_create(), audiocpp_options_free);
        if (kokoro_) {
            if (enabled("KOKORO_AUDIOCPP_NEON")) {
                check(audiocpp_options_set(options.get(), "kokoro_tts.exec_plan",
                                           R"({"generator":{"impl":"neon"}})"));
            }
        } else {
            check(audiocpp_options_set(options.get(), "vieneu_v3_turbo.g2p_dict_file",
                                       (directory_ + "sea_g2p.bin").c_str()));
            check(audiocpp_options_set(options.get(), "vieneu_v3_turbo.g2p_library",
                                       (directory_ + "libsea_g2p_rs.so").c_str()));
        }
        const char *backend_name = std::getenv(kokoro_ ? "KOKORO_AUDIOCPP_BACKEND" : "VIENEU_BACKEND");
        const audiocpp_backend_config backend{backend_name ? backend_name : "cpu", 0, threads_};
        std::cerr << (kokoro_ ? "kokoro_audiocpp_backend=" : "vieneu_backend=") << backend.backend << '\n';
        audiocpp_session *loaded = nullptr;
        check(audiocpp_session_create(model_handle_.get(), "tts", "offline", &backend, options.get(), &loaded));
        const std::shared_ptr<audiocpp_session> session(loaded, audiocpp_session_free);
        const std::string codes = directory_ + "ref_codes.txt";
        const std::string speaker = directory_ + "speaker.emb.txt";
        const bool kokoro = kokoro_;
        return [session, codes, speaker, kokoro](const std::string &text, Clock::time_point release, bool keep) {
            const std::unique_ptr<audiocpp_request, decltype(&audiocpp_request_free)>
                request(audiocpp_request_create(), audiocpp_request_free);
            check(audiocpp_request_set_text(request.get(), text.c_str(), kokoro ? "en-us" : nullptr));
            if (kokoro) {
                check(audiocpp_request_set_voice_id(request.get(), "af_heart"));
            } else {
                check(audiocpp_request_set_option(request.get(), "reference_codes_file", codes.c_str()));
                check(audiocpp_request_set_option(request.get(), "speaker_embedding_file", speaker.c_str()));
            }
            check(audiocpp_request_set_option(request.get(), "seed", "61118"));
            audiocpp_result *out = nullptr;
            check(audiocpp_session_run(session.get(), request.get(), &out));
            const std::unique_ptr<audiocpp_result, decltype(&audiocpp_result_free)> result(out, audiocpp_result_free);
            const float *samples = nullptr;
            size_t frames = 0;
            int rate = 0;
            int channels = 0;
            check(audiocpp_result_audio(result.get(), &samples, &frames, &rate, &channels));
            return pcm_chunk(release, kokoro ? 24000 : 48000, samples, frames, rate, channels, keep);
        };
    }

    bool kokoro_;
    bool sherpa_;
    bool adreno_;
    std::string model_;
    int threads_;
    std::string provider_;
    Generate shared_kokoro_;
    std::string directory_;
    std::unique_ptr<void, decltype(&dlclose)> g2p_library_{nullptr, dlclose};
    std::unique_ptr<audiocpp_registry, decltype(&audiocpp_registry_free)>
        registry_{nullptr, audiocpp_registry_free};
    std::unique_ptr<audiocpp_model, decltype(&audiocpp_model_free)>
        model_handle_{nullptr, audiocpp_model_free};
};

std::vector<int> cpus_by_frequency() {
    cpu_set_t cpus;
    CPU_ZERO(&cpus);
    if (sched_getaffinity(0, sizeof(cpus), &cpus) != 0) {
        throw std::runtime_error("could not read CPU affinity");
    }
    std::vector<std::pair<int, int>> frequencies;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &cpus)) {
            std::ifstream source("/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                                 "/cpufreq/cpuinfo_max_freq");
            int frequency = 0;
            if (source >> frequency) {
                frequencies.emplace_back(-frequency, cpu);
            }
        }
    }
    std::sort(frequencies.begin(), frequencies.end());
    std::vector<int> ordered;
    for (auto frequency : frequencies) {
        ordered.push_back(frequency.second);
    }
    return ordered;
}

void bind_worker(const std::vector<int> &cpus, int stream, int streams, int threads, bool spread) {
    if (cpus.empty()) {
        return;
    }
    cpu_set_t mask;
    CPU_ZERO(&mask);
    for (int thread = 0; thread < threads; ++thread) {
        const size_t index = spread ? thread * streams + stream : stream * threads + thread;
        CPU_SET(cpus.at(index % cpus.size()), &mask);
    }
    if (sched_setaffinity(0, sizeof(mask), &mask) != 0) {
        throw std::runtime_error("could not set worker CPU affinity");
    }
}

Observation generate(const Engines::Generate &engine, const Fixture &fixture,
                     bool streaming, Clock::time_point release, bool keep_pcm) {
    Observation result;
    result.id = fixture.id;
    try {
        if (streaming) {
            for (const auto &sentence : fixture.sentences) {
                result.chunks.push_back(engine(sentence, release, keep_pcm));
            }
        } else {
            const std::string joined = fixture.sentences[0] + " " + fixture.sentences[1];
            result.chunks.push_back(engine(joined, release, keep_pcm));
        }
        result.complete_ms = Milliseconds(Clock::now() - release).count();
    } catch (const std::exception &error) {
        result.error = error.what();
    }
    return result;
}

class SentenceQueue {
public:
    struct Job {
        size_t stream;
        size_t request;
        size_t sentence;
        Clock::time_point release;
    };

    SentenceQueue(size_t streams, size_t waves, size_t workers, int rate, Clock::duration period,
                  std::vector<Observation> &observations)
        : waves_(waves), workers_(workers), rate_(rate), period_(period),
          observations_(observations), cursors_(streams) {}

    bool ready(const std::string &error) {
        std::unique_lock lock(mutex_);
        if (!error.empty() && warmup_error_.empty()) {
            warmup_error_ = error;
        }
        if (++ready_ == workers_) {
            origin_ = Clock::now() + std::chrono::milliseconds(100);
            changed_.notify_all();
        } else {
            changed_.wait(lock, [&] { return ready_ == workers_; });
        }
        return warmup_error_.empty();
    }

    std::optional<Job> take() {
        std::unique_lock lock(mutex_);
        while (finished_ != observations_.size()) {
            const auto now = Clock::now();
            auto wake = Clock::time_point::max();
            auto best = Clock::time_point::max();
            std::optional<Job> selected;
            for (size_t stream = 0; stream < cursors_.size(); ++stream) {
                const Cursor &cursor = cursors_[stream];
                if (cursor.busy || cursor.wave == waves_) {
                    continue;
                }
                const auto release = origin_ + period_ * static_cast<Clock::duration::rep>(cursor.wave);
                if (release > now) {
                    wake = std::min(wake, release);
                    continue;
                }
                const bool is_continuation = cursor.sentence == 1;
                const auto deadline = is_continuation ? cursor.playback_deadline : release + period_;
                if (!selected || (is_continuation && selected->sentence == 0) ||
                    (cursor.sentence == selected->sentence && deadline < best)) {
                    selected = Job{stream, cursor.wave * cursors_.size() + stream,
                                   cursor.sentence, release};
                    best = deadline;
                }
            }
            if (selected) {
                cursors_[selected->stream].busy = true;
                return selected;
            }
            if (wake == Clock::time_point::max()) {
                changed_.wait(lock);
            } else {
                changed_.wait_until(lock, wake);
            }
        }
        return std::nullopt;
    }

    void finish(const Job &job, std::optional<Chunk> chunk, const std::string &error) {
        std::lock_guard lock(mutex_);
        Cursor &cursor = cursors_[job.stream];
        Observation &observation = observations_[job.request];
        if (chunk) {
            if (job.sentence == 0) {
                const auto duration = Milliseconds(chunk->arrival_ms + 1000.0 * chunk->frames / rate_);
                cursor.playback_deadline = job.release + std::chrono::duration_cast<Clock::duration>(duration);
            }
            observation.chunks.push_back(std::move(*chunk));
        } else {
            observation.error = error;
        }
        if (!chunk || job.sentence == 1) {
            observation.complete_ms = Milliseconds(Clock::now() - job.release).count();
            ++cursor.wave;
            cursor.sentence = 0;
            ++finished_;
        } else {
            cursor.sentence = 1;
        }
        cursor.busy = false;
        changed_.notify_all();
    }

    const std::string &warmup_error() const { return warmup_error_; }

private:
    struct Cursor {
        size_t wave = 0;
        size_t sentence = 0;
        bool busy = false;
        Clock::time_point playback_deadline;
    };

    size_t waves_;
    size_t workers_;
    int rate_;
    Clock::duration period_;
    std::vector<Observation> &observations_;
    std::vector<Cursor> cursors_;
    std::mutex mutex_;
    std::condition_variable changed_;
    Clock::time_point origin_;
    size_t ready_ = 0;
    size_t finished_ = 0;
    std::string warmup_error_;
};

void run_sentence_pool(const Engines &engines, const std::vector<Fixture> &fixtures,
                       const std::vector<int> &cpus, int pin, int threads, int workers,
                       int concurrency, int waves, int rate, Clock::duration period,
                       std::vector<Observation> &observations) {
    SentenceQueue queue(concurrency, waves, workers, rate, period, observations);
    const bool save_all = enabled("INFERENCEKIT_SAVE_PCM");
    std::mutex initialization;
    std::vector<std::thread> pool;
    for (size_t i = 0; i < observations.size(); ++i) {
        observations[i].id = fixtures[i].id;
    }
    for (int index = 0; index < workers; ++index) {
        pool.emplace_back([&, index] {
            Engines::Generate engine;
            std::string error;
            try {
                bind_worker(cpus, index, workers, threads, pin == 2);
                {
                    const std::lock_guard lock(initialization);
                    engine = engines.worker();
                }
                error = generate(engine, fixtures.back(), true, Clock::now(), false).error;
            } catch (const std::exception &failure) {
                error = failure.what();
            }
            if (!queue.ready(error)) {
                return;
            }
            while (auto job = queue.take()) {
                std::optional<Chunk> chunk;
                try {
                    chunk = engine(fixtures[job->request].sentences[job->sentence], job->release,
                                   save_all || job->request < static_cast<size_t>(concurrency));
                } catch (const std::exception &failure) {
                    queue.finish(*job, std::nullopt, failure.what());
                    continue;
                }
                queue.finish(*job, std::move(chunk), "");
            }
        });
    }
    for (auto &worker : pool) {
        worker.join();
    }
    if (!queue.warmup_error().empty()) {
        throw std::runtime_error("sentence worker warmup failed: " + queue.warmup_error());
    }
}

void write_observations(const std::vector<Observation> &observations, int round, int concurrency,
                        const std::string &output) {
    for (size_t index = 0; index < observations.size(); ++index) {
        const Observation &row = observations[index];
        const int wave = static_cast<int>(index / concurrency);
        const int stream = static_cast<int>(index % concurrency);
        std::cout << "{\"id\":\"" << row.id << "\",\"round\":" << round
                  << ",\"concurrency\":" << concurrency << ",\"wave\":" << wave
                  << ",\"stream\":" << stream << ",\"complete_ms\":";
        if (row.error.empty()) {
            std::cout << row.complete_ms;
        } else {
            std::cerr << "fixture " << row.id << ": " << row.error << '\n';
            std::cout << "null";
        }
        std::cout << ",\"chunks\":[";
        bool first = true;
        for (size_t chunk_index = 0; chunk_index < row.chunks.size(); ++chunk_index) {
            const Chunk &chunk = row.chunks[chunk_index];
            if (!first) {
                std::cout << ',';
            }
            std::cout << '[' << chunk.arrival_ms << ',' << chunk.frames << ',' << chunk.leading_silent_frames << ']';
            first = false;
            if (!chunk.pcm.empty()) {
                const std::string filename = output + "/" + row.id + "-r" + std::to_string(round) +
                    "-n" + std::to_string(concurrency) + "-s" + std::to_string(stream) +
                    "-c" + std::to_string(chunk_index) + ".s16le";
                std::ofstream pcm(filename, std::ios::binary);
                pcm.write(reinterpret_cast<const char *>(chunk.pcm.data()), chunk.pcm.size() * sizeof(int16_t));
                if (!pcm) {
                    throw std::runtime_error("could not save " + filename);
                }
            }
        }
        std::cout << "]}\n";
    }
}

void process_stats(const char *stage) {
    std::cerr << "process_stats_stage=" << stage << '\n';
#ifdef __ANDROID__
    if (auto *library = dlopen("libandroid.so", RTLD_LAZY | RTLD_LOCAL)) {
        auto acquire = reinterpret_cast<AThermalManager *(*)()>(dlsym(library, "AThermal_acquireManager"));
        auto read = reinterpret_cast<AThermalStatus (*)(AThermalManager *)>(
            dlsym(library, "AThermal_getCurrentThermalStatus"));
        auto release = reinterpret_cast<void (*)(AThermalManager *)>(dlsym(library, "AThermal_releaseManager"));
        if (acquire && read && release) {
            if (auto *manager = acquire()) {
                std::cerr << "process_thermal_status=" << read(manager) << '\n';
                release(manager);
            }
        }
        dlclose(library);
    }
#endif
    std::ifstream status("/proc/self/status");
    for (std::string line; std::getline(status, line);) {
        if (line.rfind("VmHWM", 0) == 0 || line.rfind("Cpus_allowed_list", 0) == 0 ||
            line.rfind("Threads:", 0) == 0) {
            std::cerr << line << '\n';
        }
    }
    if (enabled("INFERENCEKIT_MEMORY_ROLLUP")) {
        std::ifstream memory("/proc/self/smaps_rollup");
        if (!memory) {
            std::cerr << "smaps_rollup=UNAVAILABLE\n";
        }
        for (std::string line; std::getline(memory, line);) {
            if (line.rfind("Pss:", 0) == 0 || line.rfind("Private_Dirty:", 0) == 0 ||
                line.rfind("Private_Clean:", 0) == 0) {
                std::cerr << line << '\n';
            }
        }
    }
}

int synthesize(int argc, char **argv) {
    if (argc != 7) {
        throw std::runtime_error("usage: capture synthesize MODEL MODEL_FILE THREADS TEXT OUTPUT.wav");
    }
    const std::string family = argv[2];
    if (family != "kokoro" && family != "vieneu") {
        throw std::runtime_error("invalid model family");
    }
    const int threads = std::stoi(argv[4]);
    if (threads < 1) {
        throw std::runtime_error("thread count must be positive");
    }
    const bool kokoro = family == "kokoro";
    Engines engines(kokoro, argv[3], threads);
    const Chunk chunk = engines.worker()(argv[5], Clock::now(), true);
    const int channels = static_cast<int>(chunk.pcm.size() / chunk.frames);
    const int rate = kokoro ? 24000 : 48000;
    std::vector<std::vector<float>> audio(channels, std::vector<float>(chunk.frames));
    std::vector<const float *> data;
    for (int channel = 0; channel < channels; ++channel) {
        for (size_t frame = 0; frame < chunk.frames; ++frame) {
            audio[channel][frame] = chunk.pcm[frame * channels + channel] / 32768.0f;
        }
        data.push_back(audio[channel].data());
    }
    if (!SherpaOnnxWriteWaveMultiChannel(data.data(), static_cast<int32_t>(chunk.frames), rate, channels, argv[6])) {
        throw std::runtime_error("could not write WAV");
    }
    return 0;
}

int run(int argc, char **argv) {
    if (argc != 12) {
        throw std::runtime_error("usage: capture MODEL MODEL_FILE FIXTURES MODE THREADS PIN ROUND CONCURRENCY WAVES PERIOD_MS PCM_DIR");
    }
    const std::string family = argv[1];
    if (family != "kokoro" && family != "vieneu") {
        throw std::runtime_error("invalid model family");
    }
    const std::string mode = argv[4];
    if (mode != "stream" && mode != "full") {
        throw std::runtime_error("invalid mode");
    }
    const bool streaming = mode == "stream";
    const int threads = std::stoi(argv[5]);
    const int pin = std::stoi(argv[6]);
    const int round = std::stoi(argv[7]);
    const int concurrency = std::stoi(argv[8]);
    const int waves = std::stoi(argv[9]);
    const std::chrono::milliseconds period(std::stoi(argv[10]));
    const char *pool_option = std::getenv("INFERENCEKIT_SENTENCE_WORKERS");
    const int pool_size = pool_option ? std::stoi(pool_option) : 0;
    const auto fixtures = load_fixtures(argv[3]);
    if (threads < 1 || concurrency < 1 || waves < 1 || pin < 0 || pin > 2 || period.count() < 1 ||
        pool_size < 0 || (pool_size && !streaming) ||
        fixtures.size() < static_cast<size_t>(waves) * concurrency + 1) {
        throw std::runtime_error("invalid plan dimensions");
    }
    const auto cpus = pin ? cpus_by_frequency() : std::vector<int>{};
    if (pin && cpus.empty()) {
        throw std::runtime_error("missing CPU frequency topology for pinned mode");
    }
    std::cerr << "family=" << family << " model=" << argv[2] << " mode=" << mode
              << " threads_per_worker=" << threads << " concurrency=" << concurrency
              << " affinity_policy=" << pin << " cpu_frequency_order=";
    for (const auto cpu : cpus) {
        std::cerr << cpu << ',';
    }
    std::cerr << '\n';
    process_stats("before-init");

    if (family == "vieneu" && enabled("INFERENCEKIT_PROFILE")) {
        audiocpp_log_set([](audiocpp_log_level, const char *line, void *) {
            if (std::string_view(line).find("[TIMING") == 0) {
                fputs(line, stderr);
            }
        }, nullptr);
    }
    Engines engines(family == "kokoro", argv[2], threads);
    std::vector<Observation> observations(static_cast<size_t>(concurrency) * waves);
    if (pool_size) {
        const int workers = std::min(pool_size, concurrency);
        std::cerr << "sentence_pool_workers=" << workers << " policy=playback_deadline\n";
        run_sentence_pool(engines, fixtures, cpus, pin, threads, workers, concurrency, waves,
                          family == "kokoro" ? 24000 : 48000, period, observations);
        process_stats("after-pool-close");
        write_observations(observations, round, concurrency, argv[11]);
        return 0;
    }
    Clock::time_point origin;
    std::mutex gate;
    std::mutex initialization;
    std::condition_variable changed;
    int ready = 0;
    const bool save_all = enabled("INFERENCEKIT_SAVE_PCM");
    std::vector<std::thread> workers;
    for (int stream = 0; stream < concurrency; ++stream) {
        workers.emplace_back([&, stream] {
            Engines::Generate engine;
            std::string worker_error;
            try {
                bind_worker(cpus, stream, concurrency, threads, pin == 2);
                {
                    const std::lock_guard lock(initialization);
                    engine = engines.worker();
                }
                auto warmup = generate(engine, fixtures.back(), streaming, Clock::now(), false);
                worker_error = std::move(warmup.error);
            } catch (const std::exception &error) {
                worker_error = error.what();
            }
            {
                std::unique_lock lock(gate);
                ++ready;
                changed.notify_all();
                changed.wait(lock, [&] { return origin != Clock::time_point{}; });
            }
            for (int wave = 0; wave < waves; ++wave) {
                const int index = wave * concurrency + stream;
                const auto release = origin + wave * period;
                std::this_thread::sleep_until(release);
                if (worker_error.empty()) {
                    Observation row = generate(engine, fixtures[index], streaming, release, save_all || wave == 0);
                    observations[index] = std::move(row);
                } else {
                    observations[index].id = fixtures[index].id;
                    observations[index].error = worker_error;
                }
            }
        });
    }
    {
        std::unique_lock lock(gate);
        changed.wait(lock, [&] { return ready == concurrency; });
        process_stats("warmed");
        origin = Clock::now() + std::chrono::milliseconds(100);
    }
    changed.notify_all();
    for (auto &worker : workers) {
        worker.join();
    }
    process_stats("after-workers-close");
    write_observations(observations, round, concurrency, argv[11]);
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    try {
        if (argc > 1 && std::string_view(argv[1]) == "synthesize") {
            return synthesize(argc, argv);
        }
        return run(argc, argv);
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
