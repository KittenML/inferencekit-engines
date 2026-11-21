#include "measurement.h"
#include "official_onnx_engine.h"
#include "pico_engine.h"
#include "vieneu/v3_native/vieneu_v3_native.h"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using Millis = std::chrono::duration<double, std::milli>;
using Sink = std::function<void(const float*, size_t)>;

struct Fixture {
    std::string id;
    std::string bucket;
    std::string language;
    std::string text;
};

struct EngineConfig {
    std::string engine;
    int threads;
    int first_frames;
    int follow_frames;
    int pico_chars;
    bool pico_int8;
    bool pico_retain_buffers;
};

EngineConfig settings(const std::string& label) {
    if (label == "full-t2") return {"vieneu", 2, 0, 0, 0, false, false};
    if (label == "full-t4") return {"vieneu", 4, 0, 0, 0, false, false};
    if (label == "full-t6") return {"vieneu", 6, 0, 0, 0, false, false};
    if (label == "step4-6-t2") return {"vieneu", 2, 4, 6, 0, false, false};
    if (label == "step4-6-t4") return {"vieneu", 4, 4, 6, 0, false, false};
    if (label == "step4-6-t6") return {"vieneu", 6, 4, 6, 0, false, false};
    if (label == "step2-4-t4") return {"vieneu", 4, 2, 4, 0, false, false};
    if (label == "step2-6-t4") return {"vieneu", 4, 2, 6, 0, false, false};
    if (label == "step3-4-t4") return {"vieneu", 4, 3, 4, 0, false, false};
    if (label == "step3-3-t4") return {"vieneu", 4, 3, 3, 0, false, false};
    if (label == "adaptive3-t4") return {"vieneu", 4, 3, 25, 0, false, false};
    if (label == "adaptive4-t4") return {"vieneu", 4, 4, 25, 0, false, false};
    if (label == "onnx-full-t4") return {"publisher-onnx", 4, 0, 0, 0, false, false};
    if (label == "onnx-step2-4-t4") return {"publisher-onnx", 4, 2, 4, 0, false, false};
    if (label == "onnx-step4-6-t4") return {"publisher-onnx", 4, 4, 6, 0, false, false};
    if (label == "onnx-step2-4-t2") return {"publisher-onnx", 2, 2, 4, 0, false, false};
    if (label == "pico-i8-t1") return {"pico", 1, 0, 0, 64, true, false};
    if (label == "pico-i8-t2") return {"pico", 2, 0, 0, 64, true, false};
    if (label == "pico-i8-t4") return {"pico", 4, 0, 0, 64, true, false};
    if (label == "pico-i8-t4-latency") return {"pico", 4, 0, 0, 64, true, true};
    if (label == "pico-fp32-t4") return {"pico", 4, 0, 0, 64, false, false};
    throw std::runtime_error("Unknown engine label " + label);
}

std::vector<Fixture> fixtures(const std::string& path) {
    std::ifstream source(path);
    if (!source) throw std::runtime_error("Can't open " + path);
    std::vector<Fixture> rows;
    for (std::string line; std::getline(source, line);) {
        const auto id_end = line.find('\t');
        const auto bucket_end = line.find('\t', id_end + 1);
        const auto lang_end = line.find('\t', bucket_end + 1);
        if (id_end == std::string::npos || bucket_end == std::string::npos ||
            lang_end == std::string::npos || lang_end + 1 >= line.size()) {
            throw std::runtime_error("Malformed workload line");
        }
        rows.push_back({line.substr(0, id_end), line.substr(id_end + 1, bucket_end - id_end - 1),
                        line.substr(bucket_end + 1, lang_end - bucket_end - 1), line.substr(lang_end + 1)});
    }
    if (rows.empty()) throw std::runtime_error("Empty workload");
    return rows;
}

void write_wav(const std::string& path, int rate, const std::vector<int16_t>& samples) {
    std::ofstream file(path, std::ios::binary);
    const uint32_t pcm_bytes = static_cast<uint32_t>(samples.size() * sizeof(int16_t));
    auto field = [&](uint32_t word, int bytes) {
        for (int i = 0; i < bytes; ++i) file.put(static_cast<char>(word >> (8 * i)));
    };
    file.write("RIFF", 4);
    field(36 + pcm_bytes, 4);
    file.write("WAVEfmt ", 8);
    field(16, 4);
    field(1, 2);
    field(1, 2);
    field(rate, 4);
    field(rate * 2, 4);
    field(2, 2);
    field(16, 2);
    file.write("data", 4);
    field(pcm_bytes, 4);
    file.write(reinterpret_cast<const char*>(samples.data()), pcm_bytes);
    if (!file) throw std::runtime_error("Could not write " + path);
}

uint32_t fixture_seed(const Fixture& fixture) {
    uint32_t seed = 42;
    for (unsigned char c : fixture.id) seed = seed * 33 + c;
    return seed;
}

int measure(const std::string& payload, const std::string& output,
            const std::string& label, int round) {
    const std::string eager_suffix = "-eager";
    const bool eager = label.size() > eager_suffix.size() &&
        label.compare(label.size() - eager_suffix.size(), eager_suffix.size(), eager_suffix) == 0;
    const auto base_label = eager ? label.substr(0, label.size() - eager_suffix.size()) : label;
    const auto config = settings(base_label);
    const bool adaptive = base_label.rfind("adaptive", 0) == 0;
    auto prompts = fixtures(payload + "/workload.tsv");
    // Keep the Vietnamese quality check outside the matched English timing rounds.
    const bool vietnamese_quality = round == -3;
    const bool screen = round == -1 || round == -2;
    const bool retain = round == -1 || vietnamese_quality || round == 0;
    prompts.erase(std::remove_if(prompts.begin(), prompts.end(), [&](const Fixture& fixture) {
        if (vietnamese_quality) return fixture.language != "vi";
        return fixture.language != "en" || (screen && fixture.id.back() != '1');
    }), prompts.end());
    if (round >= 0) {
        std::rotate(prompts.begin(), prompts.begin() + (static_cast<size_t>(round) % 3) * 3, prompts.end());
    }

    vieneu_bench::MemorySampler memory;
    memory.start();
    const auto before_load = Clock::now();
    std::unique_ptr<PicoEngine> pico;
    std::unique_ptr<VieneuV3NativeEngine> vieneu;
    std::unique_ptr<OfficialOnnxEngine> publisher_onnx;
    int sample_rate;
    if (config.engine == "pico") {
        pico = std::make_unique<PicoEngine>(payload + "/pico", config.threads,
                                            config.pico_int8, config.pico_chars, config.pico_retain_buffers);
        sample_rate = pico->sample_rate();
    } else {
        const std::string model = payload + "/vieneu";
        if (setenv("VIENEU_SEA_G2P_DICT", (model + "/sea_g2p.bin").c_str(), 1) != 0)
            throw std::runtime_error("Could not configure the phonemizer");
        if (setenv("VIENEU_SEA_G2P_DEBUG", "1", 1) != 0) throw std::runtime_error("Could not configure phonemizer error logging");
        if (config.engine == "publisher-onnx") {
            const auto release = payload + "/vieneu-int8";
            publisher_onnx = std::make_unique<OfficialOnnxEngine>(release, model + "/codec",
                release + "/voices.json", config.threads, config.first_frames, config.follow_frames);
            sample_rate = publisher_onnx->sample_rate();
        } else {
            vieneu = std::make_unique<VieneuV3NativeEngine>();
            VieneuV3NativeInit init;
            init.model_dir = model;
            init.codec_dir = model + "/codec";
            init.n_threads = config.threads;
            init.preset_only = true;
            init.codec_step = config.first_frames > 0;
            std::string error;
            if (!vieneu->initialize(init, error)) throw std::runtime_error(error);
            sample_rate = vieneu->sample_rate();
        }
    }
    const double load_ms = Millis(Clock::now() - before_load).count();
    const auto loaded = vieneu_bench::read_memory();

    auto synthesize = [&](const Fixture& fixture, const Sink& sink) {
        if (pico) {
            pico->synthesize(fixture.text, sink);
            return;
        }
        if (publisher_onnx) {
            publisher_onnx->synthesize(fixture.text, fixture_seed(fixture), sink, fixture.language == "en");
            return;
        }
        VieneuV3NativeParams params;
        params.text = fixture.text;
        params.voice_id = "adam";
        params.temperature = 0.8f;
        params.top_k = 25;
        params.top_p = 0.95f;
        params.repetition_penalty = 1.2f;
        params.apply_watermark = false;
        params.codec_first_frames = config.first_frames;
        params.codec_follow_frames = config.follow_frames;
        params.codec_adaptive_groups = adaptive;
        params.reset_sampling = true;
        params.request_seed = fixture_seed(fixture);
        params.force_english = fixture.language == "en";
        params.require_eos = true;
        // Use fixed internal sentence/short-clause partitions in both full and step arms.
        params.max_chars = 96;
        params.pcm = sink;
        std::vector<float> unused;
        std::string error;
        if (!vieneu->synthesize(params, unused, error)) throw std::runtime_error(error);
    };

    const auto before_warmup = Clock::now();
    const Fixture warmup = {"warmup", "short", "en", "Welcome to our morning briefing."};
    synthesize(warmup, [](const float*, size_t) {});
    const double warmup_ms = Millis(Clock::now() - before_warmup).count();
    const Json cell = {{"engine", config.engine}, {"threads", config.threads},
                       {"codec_first_frames", config.first_frames}, {"codec_follow_frames", config.follow_frames},
                       {"pico_chunk_chars", config.pico_chars}, {"pico_int8", config.pico_int8},
                       {"pico_retain_buffers", config.pico_retain_buffers}, {"label", label},
                       {"round", round}, {"pid", getpid()}, {"sample_rate", sample_rate},
                       {"start_output_on_admission", eager}, {"codec_adaptive_groups", adaptive},
                       {"load_ms", load_ms}, {"loaded_memory", Json::parse(loaded.json())},
                       {"warmup_ms", warmup_ms}, {"pre_requests_memory", Json::parse(memory.lifetime().json())}};
    std::ofstream cell_file(output + "/cell.json");
    cell_file << cell.dump(2) << '\n';
    cell_file.close();

    std::ofstream requests(output + "/requests.jsonl");
    int errors = 0;
    for (const auto& fixture : prompts) {
        if (pico && fixture.language != "en") continue;
        Json row = {{"id", fixture.id}, {"bucket", fixture.bucket},
                    {"language", fixture.language}, {"seed", fixture_seed(fixture)},
                    {"engine", config.engine}, {"label", label}, {"round", round}};
        try {
            std::vector<int16_t> saved;
            if (retain) saved.reserve(20 * sample_rate);
            vieneu_bench::Playback playback(sample_rate, eager);
            memory.reset_window();
            playback.start();
            int64_t frames = 0;
            int64_t clips = 0;
            double squares = 0;
            size_t chunks = 0;
            synthesize(fixture, [&](const float* pcm, size_t count) {
                if (count == 0) return;
                playback.emit(pcm, count);
                for (size_t i = 0; i < count; ++i) {
                    const float sample = pcm[i];
                    if (!std::isfinite(sample)) throw std::runtime_error("Nonfinite PCM");
                    squares += static_cast<double>(sample) * sample;
                    clips += std::abs(sample) > 1.0f;
                    if (retain) saved.push_back(static_cast<int16_t>(std::lrintf(std::clamp(sample, -1.0f, 1.0f) * 32767.0f)));
                }
                frames += count;
                ++chunks;
            });
            const auto played = playback.finish();
            const auto request_memory = memory.snapshot();
            if (!frames || squares == 0.0) throw std::runtime_error("Empty or silent PCM");
            row["audio_s"] = static_cast<double>(frames) / sample_rate;
            row["source_frames"] = frames;
            row["audio_chunks"] = chunks;
            row["clipped_samples"] = clips;
            row["rms_dbfs"] = 10.0 * std::log10(squares / frames);
            row["playback"] = Json::parse(played.json());
            row["memory"] = Json::parse(request_memory.json());
            row["rtf"] = row["playback"]["synthesis_complete_ms"].is_number() ?
                Json(row["playback"]["synthesis_complete_ms"].get<double>() / 1000 / row["audio_s"].get<double>()) : Json(nullptr);
            if (retain) {
                const std::string name = fixture.id + ".wav";
                write_wav(output + "/" + name, sample_rate, saved);
                row["wav"] = name;
            }
        } catch (const std::exception& exception) {
            row["error"] = exception.what();
            ++errors;
        }
        requests << row.dump() << '\n' << std::flush;
    }
    std::ofstream peaks(output + "/lifetime-memory.json");
    peaks << memory.lifetime().json() << '\n';
    return errors ? 2 : 0;
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: study_runner PAYLOAD OUTPUT ENGINE ROUND\n";
        return 64;
    }
    try {
        return measure(argv[1], argv[2], argv[3], std::stoi(argv[4]));
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        std::ofstream error(std::string(argv[2]) + "/error.txt");
        error << exception.what() << '\n';
        return 1;
    }
}
