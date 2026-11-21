#!/usr/bin/env python3
"""Reuse the study's arrival clock and playback measurements with VieNeu-TTS.cpp."""
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text()
source = source.replace('#include <audiocpp.h>\n#include <sherpa-onnx/c-api/c-api.h>', '#include <vieneu/vieneu_tts.h>')
start = source.index('void check(audiocpp_status')
end = source.index('std::vector<int> cpus_by_frequency()', start)
source = source[:start] + '''class Engines {
public:
    Engines(bool kokoro, std::string model, int threads)
        : model_(std::move(model)), threads_(threads) {
        if (kokoro) {
            throw std::runtime_error("VieNeu-TTS.cpp only supports VieNeu");
        }
        std::cerr << "runtime=vieneu-tts.cpp version=" << vieneu_version() << '\\n';
    }

    using Generate = std::function<Chunk(const std::string &, Clock::time_point, bool)>;

    Generate worker() const {
        const std::string codec_dir = model_ + "/codec";
        vieneu_init_params_v2 init{};
        vieneu_init_v2_default_params(&init);
        init.profile = "vieneu-v3-native";
        init.model_dir = model_.c_str();
        init.codec_dir = codec_dir.c_str();
        init.n_threads = threads_;
        const std::shared_ptr<vieneu_context> engine(vieneu_init_v2(&init), vieneu_free);
        if (!engine) {
            throw std::runtime_error(vieneu_last_error());
        }
        return [engine](const std::string &text, Clock::time_point release, bool keep) {
            vieneu_tts_params_v3 params{};
            vieneu_tts_v3_default_params(&params);
            params.text = text.c_str();
            params.voice_id = "minh_quan_pro";
            params.apply_watermark = false;
            vieneu_audio waveform{};
            const std::unique_ptr<vieneu_audio, decltype(&vieneu_audio_free)>
                guard(&waveform, vieneu_audio_free);
            if (vieneu_synthesize_v3(engine.get(), &params, &waveform) != 0) {
                throw std::runtime_error(vieneu_last_error());
            }
            return pcm_chunk(release, 48000, waveform.samples, static_cast<size_t>(waveform.n_samples),
                             waveform.sample_rate, waveform.channels, keep);
        };
    }

private:
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
source = source.replace('        if (argc > 1 && std::string_view(argv[1]) == "synthesize") {\n            return synthesize(argc, argv);\n        }\n', '')
source = source.replace('#include "kokoro_pcm_process.h"\n', "")
Path(sys.argv[2]).write_text(source)
