#pragma once

#include "vieneu/vieneu_v3_onnx.h"

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

class OfficialOnnxEngine {
public:
    using TokenizerTrace = std::function<void(const std::string&, const std::vector<int64_t>&)>;

    OfficialOnnxEngine(const std::string& model_dir, const std::string& codec_dir,
                       const std::string& voices_path, int threads, int first_frames = 4, int follow_frames = 6)
        : first_frames_(first_frames), follow_frames_(follow_frames) {
        if (first_frames < 0 || (first_frames > 0 && follow_frames <= 0)) {
            throw std::invalid_argument("invalid MOSS frame groups");
        }
        VieneuV3OnnxInit options;
        options.model_dir = model_dir;
        options.codec_dir = codec_dir;
        options.voices_json_path = voices_path;
        options.n_threads = threads;
        options.codec_step = first_frames > 0;
        options.allow_spinning = false;
        std::string error;
        if (!engine_.initialize(options, error)) throw std::runtime_error(error);
    }

    int sample_rate() const { return engine_.sample_rate(); }

    void synthesize(const std::string& text, uint32_t seed,
                    const std::function<void(const float*, size_t)>& emit, bool force_english,
                    bool phonemes = false, float temperature = 0.8f, const TokenizerTrace& trace = {}) {
        VieneuV3OnnxParams params;
        params.text = text;
        params.voice_id = "adam";
        params.apply_watermark = false;
        params.phoneme_input = phonemes;
        params.force_english = force_english;
        params.require_eos = true;
        params.seed = seed;
        params.temperature = temperature;
        params.max_chars = 96;
        params.codec_first_frames = first_frames_;
        params.codec_follow_frames = follow_frames_;
        params.pcm = emit;
        params.trace_tokenizer = trace;
        std::vector<float> samples;
        std::string error;
        if (!engine_.synthesize(params, samples, error)) throw std::runtime_error(error);
        if (first_frames_ == 0) emit(samples.data(), samples.size());
    }

private:
    VieneuV3OnnxEngine engine_;
    int first_frames_;
    int follow_frames_;
};
