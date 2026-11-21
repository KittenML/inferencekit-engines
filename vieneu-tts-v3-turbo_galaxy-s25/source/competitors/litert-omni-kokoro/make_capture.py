"""Bind the native timing harness to LiteRT-LM's public Kokoro Omni engine."""
from pathlib import Path
import sys


source = Path(sys.argv[1]).read_text()
source = source.replace(
    '#include <audiocpp.h>\n#include <sherpa-onnx/c-api/c-api.h>',
    '#include "omni/omni_engine.h"\n#include "omni/omni_session.h"\n'
    '#include "runtime/proto/text2speech_session_config.pb.h"\n#include <variant>',
)
# Match the Android MediaPipe capture's Math.round conversion.
source = source.replace(
    'std::lround(std::clamp(sample, -1.0f, 1.0f) * 32767.0f)',
    'static_cast<long>(std::floor(static_cast<double>(std::clamp(sample, -1.0f, 1.0f) * 32767.0f) + 0.5))',
)
start = source.index('void check(audiocpp_status')
end = source.index('std::vector<int> cpus_by_frequency()', start)
source = source[:start] + r'''class Engines {
public:
    Engines(bool kokoro, std::string model, int threads)
        : model_(std::move(model)), threads_(threads) {
        if (!kokoro) {
            throw std::runtime_error("the Omni adapter expects Kokoro");
        }
        if (enabled("INFERENCEKIT_OMNI_PROCESS_PER_STREAM")) {
            std::signal(SIGPIPE, SIG_IGN);
        }
    }

    using Generate = std::function<Chunk(const std::string &, Clock::time_point, bool)>;

    Generate worker() const {
        if (enabled("INFERENCEKIT_OMNI_PROCESS_PER_STREAM")) {
            const std::vector<std::string> command{
                "/proc/self/exe", "--pcm-worker", model_, std::to_string(threads_),
            };
            const auto process = std::make_shared<KokoroPcmProcess>(".", command);
            std::cerr << "LiteRT Omni initialized a process-isolated PCM worker\n";
            return [process](const std::string &text, Clock::time_point release, bool keep) {
                return pcm_chunk(release, process->synthesize(text), keep);
            };
        }
        return in_process_worker();
    }

    int serve_pcm() const {
        const Generate generate = in_process_worker();
        std::fputs("ready.\n", stderr);
        std::fflush(stderr);
        std::string text;
        while (std::getline(std::cin, text)) {
            try {
                const auto result = generate(text, Clock::now(), true);
                std::fprintf(stderr, "KOKORO_PCM_BEGIN %zu 24000\n", result.pcm.size());
                std::fflush(stderr);
                if (std::fwrite(result.pcm.data(), sizeof(int16_t), result.pcm.size(), stdout) != result.pcm.size() ||
                    std::fflush(stdout) != 0) {
                    return 1;
                }
            } catch (const std::exception &error) {
                std::fprintf(stderr, "Kokoro PCM error: %s\n", error.what());
            }
            std::fputs("KOKORO_UTT_END\n", stderr);
            std::fflush(stderr);
        }
        return 0;
    }

private:
    Generate in_process_worker() const {
        using ::litert::omni::OmniEngine;
        using ::litert::omni::OmniSession;
        using ::litert::omni::PushInputSource;

        OmniEngine::Settings settings;
        settings.backend = OmniEngine::Settings::Backend::kGpu;
        settings.model_path = model_;
        settings.cache_dir = "cache-omni";
        settings.num_threads = threads_;
        litert::lm::proto::Text2SpeechSessionConfig config;
        config.set_language("en-US");
        config.set_voice("af_heart");
        auto created = OmniEngine::Create("kokoro", settings, &config);
        if (!created.ok()) {
            throw std::runtime_error(created.status().ToString());
        }
        const std::shared_ptr<OmniEngine> engine(std::move(*created));
        return [engine](const std::string &text, Clock::time_point release, bool keep) {
            auto source = std::make_unique<PushInputSource>();
            auto *input = source.get();
            auto session = engine->CreateSession(std::move(source));
            if (!session.ok()) {
                throw std::runtime_error(session.status().ToString());
            }
            auto pushed = input->PushInput(OmniSession::TextInput{text});
            if (!pushed.ok()) {
                throw std::runtime_error(pushed.ToString());
            }
            input->Finish();

            std::vector<float> samples;
            const auto append = [&](const OmniSession::Output &output) {
                if (const auto *audio = std::get_if<litert::omni::AudioOutput>(&output)) {
                    if (audio->sample_rate_hz != 24000) {
                        throw std::runtime_error("unexpected Omni Kokoro sample rate");
                    }
                    samples.insert(samples.end(), audio->pcm_samples.begin(), audio->pcm_samples.end());
                }
            };
            while (true) {
                auto next = (*session)->ProcessNext();
                if (next.ok()) {
                    append(*next);
                    continue;
                }
                if (!absl::IsOutOfRange(next.status()) && !absl::IsNotFound(next.status())) {
                    throw std::runtime_error(next.status().ToString());
                }
                auto flushed = (*session)->Flush();
                if (!flushed.ok()) {
                    throw std::runtime_error(flushed.status().ToString());
                }
                append(*flushed);
                break;
            }
            return pcm_chunk(release, 24000, samples.data(), samples.size(), 24000, 1, keep);
        };
    }

    std::string model_;
    int threads_;
};

''' + source[end:]
start = source.index('int synthesize(int')
end = source.index('int run(int', start)
source = source[:start] + source[end:]
start = source.index('    if (family == "vieneu" && enabled("INFERENCEKIT_PROFILE"))')
end = source.index('    Engines engines', start)
source = source[:start] + source[end:]
source = source.replace(
    '        if (argc > 1 && std::string_view(argv[1]) == "synthesize") {\n'
    '            return synthesize(argc, argv);\n        }\n',
    '        if (argc == 4 && std::string_view(argv[1]) == "--pcm-worker") {\n'
    '            return Engines(true, argv[2], std::stoi(argv[3])).serve_pcm();\n        }\n',
)
Path(sys.argv[2]).write_text(source)
