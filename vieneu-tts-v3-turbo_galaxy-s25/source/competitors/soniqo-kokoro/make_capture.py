"""Attach the pinned Soniqo Kokoro engine to the native timing harness."""
from pathlib import Path
import sys


source = Path(sys.argv[1]).read_text()
source = source.replace(
    '#include <audiocpp.h>\n#include <sherpa-onnx/c-api/c-api.h>',
    '#include <speech_core/models/kokoro_tts.h>\n#include <speech_core/models/onnx_engine.h>\n#include <unistd.h>',
)
start = source.index('void check(audiocpp_status')
end = source.index('std::vector<int> cpus_by_frequency()', start)
source = source[:start] + r'''class Engines {
public:
    Engines(bool kokoro, std::string model, int threads)
        : model_(std::move(model)), nnapi_(enabled("SONIQO_KOKORO_NNAPI")) {
        if (!kokoro) {
            throw std::runtime_error("the Soniqo adapter expects Kokoro");
        }
        const std::string width = std::to_string(threads);
        setenv("SPEECH_CORE_KOKORO_ORT_THREADS", width.c_str(), 1);
        std::cerr << "runtime=soniqo-speech-core pid=" << getpid()
                  << " requested_nnapi=" << nnapi_ << '\n';
    }

    using Generate = std::function<Chunk(const std::string &, Clock::time_point, bool)>;

    Generate worker() const {
        auto engine = std::make_shared<speech_core::KokoroTts>(
            model_ + "/kokoro-e2e-realtime.onnx", model_ + "/voices", model_, nnapi_);
        engine->set_voice("af_heart");
        engine->set_speed(1.0f);
        auto &backend = OnnxEngine::get();
        std::cerr << "soniqo_nnapi_retry_cpu=" << backend.had_nnapi_fallback() << '\n';
        return [engine](const std::string &text, Clock::time_point release, bool keep) {
            std::vector<float> samples;
            bool saw_final = false;
            engine->synthesize(text, "en", [&](const float *block, size_t size, bool final) {
                if (saw_final || !block) {
                    throw std::runtime_error("invalid Soniqo synthesis callback");
                }
                samples.insert(samples.end(), block, block + size);
                saw_final = final;
            });
            if (!saw_final) {
                throw std::runtime_error("Soniqo synthesis did not report a complete sentence");
            }
            return pcm_chunk(release, 24000, samples.data(), samples.size(),
                             engine->output_sample_rate(), 1, keep);
        };
    }

private:
    std::string model_;
    bool nnapi_;
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
    '            return synthesize(argc, argv);\n        }\n', '',
)
source = source.replace('#include "kokoro_pcm_process.h"\n', '')
Path(sys.argv[2]).write_text(source)
