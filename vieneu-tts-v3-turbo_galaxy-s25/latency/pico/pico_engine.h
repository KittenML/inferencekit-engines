#pragma once

#include "kitten_tts08.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

class PicoEngine {
public:
    // chunk_chars = 0 uses Kitten's default. A smaller limit starts playback
    // sooner by synthesizing shorter chunks from the same complete request.
    PicoEngine(const std::string& assets, int threads, bool int8 = true,
               int chunk_chars = 64, bool retain_buffers = false) {
        if (threads < 0 || chunk_chars < 0) {
            throw std::invalid_argument("Pico threads and chunk size must be nonnegative");
        }
        if (kt08_api_version() != KT08_API_VERSION) {
            throw std::runtime_error("Pico runtime and streaming header API versions differ");
        }
        kt08_options model_options;
        kt08_options_init(&model_options);
        model_options.threads = threads;
        model_options.memory_mode = retain_buffers ? KT08_MEMORY_LATENCY : KT08_MEMORY_EXTREME_LOW;
        kt08_stream_options_init(&stream_options_);
        stream_options_.chunk_chars = chunk_chars;

        const std::string prefix = assets + "/kitten_tts_2_pico_en_" + (int8 ? "int8" : "fp32");
        kt08_model* model = nullptr;
        check(kt08_model_load((prefix + "_arch.json").c_str(), (prefix + ".bin").c_str(),
                              &model_options, &model), "load model");
        model_.reset(model);

        const std::string vocab_path = assets + "/vocab.json";
        kt08_vocab* vocab = nullptr;
        check(kt08_vocab_load(vocab_path.c_str(), &vocab), "load vocabulary");
        vocab_.reset(vocab);

        kt08_tiny_frontend* frontend = nullptr;
        check(kt08_tiny_frontend_load("v1", vocab_path.c_str(), (assets + "/g2p").c_str(),
                                     nullptr, &frontend), "load English frontend");
        frontend_.reset(frontend);

        float* voice = nullptr;
        check(kt08_voice_load((assets + "/eleanor.bin").c_str(), &voice, &voice_rows_),
              "load eleanor_en voice");
        voice_.reset(voice);
    }

    int sample_rate() const { return KT08_SAMPLE_RATE; }

    // The callback is synchronous. Its sample storage expires when it returns.
    void synthesize(const std::string& text, const std::function<void(const float*, size_t)>& emit) {
        kt08_stream* stream = nullptr;
        check(kt08_stream_open(model_.get(), vocab_.get(), frontend_.get(), text.c_str(), "en",
                               voice_.get(), voice_rows_, 1.0f, &stream_options_, &stream),
              "open English stream");
        const std::unique_ptr<kt08_stream, decltype(&kt08_stream_free)> handle(stream, kt08_stream_free);
        bool emitted = false;
        int last = 0;
        do {
            float* pcm = nullptr;
            size_t count = 0;
            check(kt08_stream_next(stream, &pcm, &count, &last), "synthesize stream chunk");
            const std::unique_ptr<float, decltype(&kt08_free)> audio(pcm, kt08_free);
            if (count != 0) {
                if (!pcm) {
                    throw std::runtime_error("Pico returned a chunk without PCM");
                }
                emit(pcm, count);
                emitted = true;
            }
        } while (!last);
        if (!emitted) {
            throw std::runtime_error("Pico produced no audio for the request");
        }
    }

private:
    static void check(kt08_status status, const char* operation) {
        if (status != KT08_OK) {
            throw std::runtime_error(std::string("Pico ") + operation + ": " + kt08_last_error());
        }
    }

    std::unique_ptr<kt08_model, decltype(&kt08_model_free)> model_{nullptr, kt08_model_free};
    std::unique_ptr<kt08_vocab, decltype(&kt08_vocab_free)> vocab_{nullptr, kt08_vocab_free};
    std::unique_ptr<kt08_tiny_frontend, decltype(&kt08_tiny_frontend_free)> frontend_{
        nullptr, kt08_tiny_frontend_free};
    std::unique_ptr<float, decltype(&kt08_free)> voice_{nullptr, kt08_free};
    size_t voice_rows_ = 0;
    kt08_stream_options stream_options_{};
};
