#include "pico_engine.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " ASSETS {int8|fp32} THREADS\n";
        return 2;
    }
    const std::string precision = argv[2];
    if (precision != "int8" && precision != "fp32") {
        std::cerr << "Invalid precision\n";
        return 2;
    }

    try {
        const auto utterance = "We left the city just before sunrise. "
                               "Our first stop was the quiet village beside the lake. "
                               "By the time we reached the old bridge, the fog had lifted.";
        PicoEngine engine(argv[1], std::stoi(argv[3]), precision == "int8");
        int chunks = 0;
        size_t samples = 0;
        float peak = 0.0F;
        engine.synthesize(utterance, [&](const float* data, size_t count) {
            ++chunks;
            samples += count;
            for (size_t i = 0; i < count; ++i) {
                if (!std::isfinite(data[i])) throw std::runtime_error("nonfinite PCM");
                peak = std::fmax(peak, std::abs(data[i]));
            }
        });
        if (chunks < 2 || samples < static_cast<size_t>(engine.sample_rate()) || peak < 0.01F) {
            throw std::runtime_error("expected multiple audible chunks");
        }
        std::cout << precision << ' ' << engine.sample_rate() << " Hz " << chunks
                  << " chunks " << samples << " samples peak=" << peak << '\n';
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
