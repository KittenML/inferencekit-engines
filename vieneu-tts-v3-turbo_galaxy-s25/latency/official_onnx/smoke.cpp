#include "official_onnx_engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc != 8) {
        std::cerr << "Usage: smoke MODEL_DIR CODEC_DIR VOICES TEXT OUT_F32 {phonemes|en|vi} FIRST_FRAMES\n";
        return 2;
    }
    try {
        const int first_frames = std::stoi(argv[7]);
        const std::string mode = argv[6];
        OfficialOnnxEngine engine(argv[1], argv[2], argv[3], 2, first_frames, 6);
        float peak = 0.0f;
        size_t samples = 0;
        size_t blocks = 0;
        std::ofstream out(argv[5], std::ios::binary);
        const auto start = std::chrono::steady_clock::now();
        engine.synthesize(argv[4], 42, [&](const float* pcm, size_t size) {
            if (blocks++ == 0) {
                const auto first_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - start).count();
                std::cout << "first_pcm_samples=" << size << " first_pcm_host_ms=" << first_ms << '\n';
            }
            samples += size;
            for (size_t i = 0; i < size; ++i) {
                if (!std::isfinite(pcm[i])) throw std::runtime_error("nonfinite PCM");
                peak = std::max(peak, std::abs(pcm[i]));
            }
            out.write(reinterpret_cast<const char*>(pcm), sizeof(float) * size);
            if (!out) throw std::runtime_error("cannot save PCM");
        }, mode == "en", mode == "phonemes", 0.0f, [](const std::string& phonemes, const std::vector<int64_t>& ids) {
            std::cout << "phonemes=" << phonemes << "\tids=";
            for (size_t i = 0; i < ids.size(); ++i) std::cout << (i == 0 ? "" : ",") << ids[i];
            std::cout << '\n';
        });
        if (peak < 0.01f || samples < 3840) throw std::runtime_error("bad PCM waveform");
        const auto complete_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
        std::cout << "sample_rate=" << engine.sample_rate() << " samples=" << samples
                  << " blocks=" << blocks << " peak=" << peak << " complete_host_ms=" << complete_ms << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
