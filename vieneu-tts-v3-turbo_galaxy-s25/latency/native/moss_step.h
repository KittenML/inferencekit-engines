#pragma once

#include "onnxruntime_cxx_api.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

class MossStep {
    struct CacheTensor {
        std::string input;
        std::string output;
        std::vector<int64_t> shape;
        bool floating;
        int initial;
    };

public:
    MossStep(const std::string& codec_dir, int threads)
        : env_(ORT_LOGGING_LEVEL_WARNING, "MossStep"),
          memory_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
        std::ifstream file(codec_dir + "/codec_browser_onnx_meta.json");
        const auto config = nlohmann::json::parse(file);
        const auto& stream = config.at("streaming_decode");
        vq_ = config.at("codec_config").at("num_quantizers").get<int>();
        samples_per_frame_ = config.at("codec_config").at("downsample_rate").get<int>();
        for (const auto& offset : stream.at("transformer_offsets")) {
            cache_.push_back({offset.at("input_name"), offset.at("output_name"), offset.at("shape"), false, 0});
        }
        for (const auto& attention : stream.at("attention_caches")) {
            add_cache(attention, "offset", "offset_shape", false, 0);
            add_cache(attention, "cached_keys", "cache_shape", true, 0);
            add_cache(attention, "cached_values", "cache_shape", true, 0);
            add_cache(attention, "cached_positions", "positions_shape", false, -1);
        }
        for (const auto& tensor : cache_) {
            inputs_.push_back(tensor.input.c_str());
            outputs_.push_back(tensor.output.c_str());
        }
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(threads);
        options.SetInterOpNumThreads(1);
        options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
        options.AddConfigEntry("session.intra_op.allow_spinning", "0");
        session_ = Ort::Session(env_, (codec_dir + "/moss_audio_tokenizer_decode_step.onnx").c_str(), options);
    }

    std::vector<Ort::Value> new_state() const {
        Ort::AllocatorWithDefaultOptions allocator;
        std::vector<Ort::Value> state;
        state.reserve(cache_.size());
        for (const auto& tensor : cache_) {
            if (tensor.floating) {
                auto value = Ort::Value::CreateTensor<float>(allocator, tensor.shape.data(), tensor.shape.size());
                std::fill_n(value.GetTensorMutableData<float>(), value.GetTensorTypeAndShapeInfo().GetElementCount(), 0.0f);
                state.push_back(std::move(value));
            } else {
                auto value = Ort::Value::CreateTensor<int32_t>(allocator, tensor.shape.data(), tensor.shape.size());
                std::fill_n(value.GetTensorMutableData<int32_t>(), value.GetTensorTypeAndShapeInfo().GetElementCount(), tensor.initial);
                state.push_back(std::move(value));
            }
        }
        return state;
    }

    std::vector<float> decode(int32_t* codes, int32_t frame_count, std::vector<Ort::Value>& state) {
        const std::array<int64_t, 3> code_shape = {1, frame_count, vq_};
        const std::array<int64_t, 1> length_shape = {1};
        std::vector<Ort::Value> values;
        values.reserve(inputs_.size());
        values.push_back(Ort::Value::CreateTensor<int32_t>(memory_, codes, frame_count * vq_, code_shape.data(), code_shape.size()));
        values.push_back(Ort::Value::CreateTensor<int32_t>(memory_, &frame_count, 1, length_shape.data(), length_shape.size()));
        for (auto& tensor : state) values.push_back(std::move(tensor));

        auto result = session_.Run(Ort::RunOptions{nullptr}, inputs_.data(), values.data(), values.size(),
                                   outputs_.data(), outputs_.size());
        const auto shape = result[0].GetTensorTypeAndShapeInfo().GetShape();
        const int32_t length = result[1].GetTensorData<int32_t>()[0];
        if (shape.size() != 3 || shape[0] != 1 || shape[1] != 2 ||
            length != frame_count * samples_per_frame_ || shape[2] < length) {
            throw std::runtime_error("Unexpected MOSS decode-step audio shape or duration");
        }
        for (size_t i = 0; i < state.size(); ++i) state[i] = std::move(result[i + 2]);
        const float* first = result[0].GetTensorData<float>();
        const float* second = first + shape[2];
        std::vector<float> mono(length);
        for (int i = 0; i < length; ++i) mono[i] = first[i] * 0.5f + second[i] * 0.5f;
        return mono;
    }

private:
    void add_cache(const nlohmann::json& description, const std::string& prefix, const char* shape,
                   bool floating, int initial) {
        cache_.push_back({description.at(prefix + "_input_name"), description.at(prefix + "_output_name"),
                          description.at(shape), floating, initial});
    }

    Ort::Env env_;
    Ort::MemoryInfo memory_;
    Ort::Session session_{nullptr};
    int vq_;
    int samples_per_frame_;
    std::vector<CacheTensor> cache_;
    std::vector<const char*> inputs_{"audio_codes", "audio_code_lengths"};
    std::vector<const char*> outputs_{"audio", "audio_lengths"};
};
