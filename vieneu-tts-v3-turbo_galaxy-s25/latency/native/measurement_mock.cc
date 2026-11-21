// Host-only checks for frame accounting and the asynchronous protocol.
// The shim simulates timestamps and cannot validate Android device latency.
#include "measurement.h"

#include <cassert>
#include <cstring>
#include <iostream>

struct AAudioStreamBuilderStruct {
    AAudioStream_dataCallback data = nullptr;
    void* user = nullptr;
};

struct AAudioStreamStruct {
    AAudioStreamBuilder config;
    std::atomic<bool> stop{false};
    std::atomic<int64_t> read{0};
    int64_t start_ns = 0;
    std::thread worker;
};

static bool timestamp_available = true;
static std::atomic<int64_t> mock_presented_frames{0};

extern "C" {
aaudio_result_t AAudio_createStreamBuilder(AAudioStreamBuilder** builder) {
    *builder = new AAudioStreamBuilder;
    return AAUDIO_OK;
}
aaudio_result_t AAudioStreamBuilder_delete(AAudioStreamBuilder* b) { delete b; return AAUDIO_OK; }
void AAudioStreamBuilder_setDirection(AAudioStreamBuilder*, aaudio_direction_t) {}
void AAudioStreamBuilder_setSharingMode(AAudioStreamBuilder*, aaudio_sharing_mode_t) {}
void AAudioStreamBuilder_setPerformanceMode(AAudioStreamBuilder*, aaudio_performance_mode_t) {}
void AAudioStreamBuilder_setUsage(AAudioStreamBuilder*, aaudio_usage_t) {}
void AAudioStreamBuilder_setContentType(AAudioStreamBuilder*, aaudio_content_type_t) {}
void AAudioStreamBuilder_setSampleRate(AAudioStreamBuilder*, int32_t) {}
void AAudioStreamBuilder_setChannelCount(AAudioStreamBuilder*, int32_t) {}
void AAudioStreamBuilder_setFormat(AAudioStreamBuilder*, aaudio_format_t) {}
void AAudioStreamBuilder_setErrorCallback(AAudioStreamBuilder*, AAudioStream_errorCallback, void*) {}

void AAudioStreamBuilder_setDataCallback(AAudioStreamBuilder* b, AAudioStream_dataCallback cb, void* user) {
    b->data = cb;
    b->user = user;
}

aaudio_result_t AAudioStreamBuilder_openStream(AAudioStreamBuilder* b, AAudioStream** stream) {
    *stream = new AAudioStream;
    (*stream)->config = *b;
    return AAUDIO_OK;
}

aaudio_result_t AAudioStream_requestStart(AAudioStream* stream) {
    stream->start_ns = vieneu_bench::detail::monotonic_ns() + 30'000'000;
    stream->worker = std::thread([stream] {
        float frames[96];
        while (!stream->stop.load()) {
            stream->config.data(stream, stream->config.user, frames, 96);
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            mock_presented_frames.store(stream->read.fetch_add(96) + 96);
        }
    });
    return AAUDIO_OK;
}

aaudio_result_t AAudioStream_getTimestamp(AAudioStream* stream, clockid_t,
                                         int64_t* frame, int64_t* ns) {
    *frame = stream->read.load();
    if (!timestamp_available || *frame == 0) return AAUDIO_ERROR_INVALID_STATE;
    *ns = stream->start_ns + *frame * 1'000'000'000 / 48'000;
    return AAUDIO_OK;
}

aaudio_result_t AAudioStream_requestStop(AAudioStream* stream) { stream->stop.store(true); return AAUDIO_OK; }
aaudio_result_t AAudioStream_close(AAudioStream* stream) {
    AAudioStream_requestStop(stream);
    if (stream->worker.joinable()) stream->worker.join();
    delete stream;
    return AAUDIO_OK;
}

int32_t AAudioStream_getSampleRate(AAudioStream*) { return 48'000; }
int32_t AAudioStream_getChannelCount(AAudioStream*) { return 1; }
aaudio_format_t AAudioStream_getFormat(AAudioStream*) { return AAUDIO_FORMAT_PCM_FLOAT; }
int32_t AAudioStream_getDeviceId(AAudioStream*) { return 9; }
aaudio_performance_mode_t AAudioStream_getPerformanceMode(AAudioStream*) { return AAUDIO_PERFORMANCE_MODE_LOW_LATENCY; }
aaudio_sharing_mode_t AAudioStream_getSharingMode(AAudioStream*) { return AAUDIO_SHARING_MODE_SHARED; }
int32_t AAudioStream_getFramesPerBurst(AAudioStream*) { return 96; }
aaudio_result_t AAudioStream_setBufferSizeInFrames(AAudioStream*, int32_t n) { return n; }
int32_t AAudioStream_getBufferSizeInFrames(AAudioStream*) { return 192; }
int32_t AAudioStream_getXRunCount(AAudioStream*) { return 0; }
}

static void wait_for_mock_frames(int64_t frames) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (mock_presented_frames.load() < frames && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    assert(mock_presented_frames.load() >= frames);
}

int main() {
    using namespace vieneu_bench;
    MemorySampler memory;
    memory.start();
    memory.reset_window();

    {
        Playback playback(48000);
        std::vector<float> pcm(1920, 0.006f);
        std::fill_n(pcm.begin(), 480, 0.0f);
        playback.start();
        playback.emit(pcm.data(), pcm.size());
        const auto r = playback.finish();
        std::cout << r.json() << '\n';
        assert(std::strcmp(r.status, "ok") == 0);
        assert(r.onset_frame == 451);
        assert(r.first_audible_ns > r.first_pcm_ns);
        assert(r.playback_complete_ns > r.synthesis_complete_ns);
        assert(r.consumed_pcm_frames == 1920 && r.last_pcm_frame == 1919);
        assert(r.gap_count == 0 && r.xruns == 0);
        assert(!r.start_on_admission && r.first_pcm_frame == 0);
        assert(r.stream_start_requested_ns >= r.first_pcm_ns);
    }
    {
        Playback playback(24000);
        const std::vector<float> tone(240, 0.1f);
        mock_presented_frames.store(0);
        playback.start();
        playback.emit(tone.data(), tone.size());
        wait_for_mock_frames(1920);
        playback.emit(tone.data(), tone.size());
        const auto r = playback.finish();
        std::cout << r.json() << '\n';
        assert(std::strcmp(r.status, "ok") == 0);
        assert(r.consumed_pcm_frames == 960 && r.onset_frame == 0);
        assert(r.gap_count == 1 && r.gap_frames > 0);
        assert(r.last_pcm_frame == r.consumed_pcm_frames + r.gap_frames - 1);
    }
    {
        Playback playback(48000);
        const std::vector<float> silence(480, 0.0f);
        playback.start();
        playback.emit(silence.data(), silence.size());
        const auto r = playback.finish();
        std::cout << r.json() << '\n';
        assert(std::strcmp(r.status, "onset_threshold_not_reached") == 0);
        assert(r.first_audible_ns == -1);
    }
    {
        Playback playback(48000);
        playback.start();
        assert(std::strcmp(playback.finish().status, "no_pcm") == 0);
    }
    {
        Playback playback(48000, true);
        const std::vector<float> tone(480, 0.1f);
        mock_presented_frames.store(0);
        playback.start();
        wait_for_mock_frames(480);
        playback.emit(tone.data(), tone.size());
        const auto r = playback.finish();
        std::cout << r.json() << '\n';
        assert(std::strcmp(r.status, "ok") == 0);
        assert(r.start_on_admission && r.first_pcm_frame >= 480);
        assert(r.stream_start_requested_ns < r.first_pcm_ns);
        assert(r.first_audible_ns >= r.first_pcm_ns);
        assert(r.onset_frame == r.first_pcm_frame);
        assert(r.gap_count == 0 && r.consumed_pcm_frames == 480);
    }
    {
        Playback playback(48000, true);
        mock_presented_frames.store(0);
        playback.start();
        wait_for_mock_frames(480);
        const auto before = std::chrono::steady_clock::now();
        const auto r = playback.finish();
        assert(std::strcmp(r.status, "no_pcm") == 0);
        assert(r.first_pcm_ns == -1 && r.first_audible_ns == -1);
        assert(std::chrono::steady_clock::now() - before < std::chrono::seconds(1));
    }
    {
        timestamp_available = false;
        Playback playback(48000);
        const std::vector<float> tone(480, 0.2f);
        playback.start();
        playback.emit(tone.data(), tone.size());
        const auto r = playback.finish();
        std::cout << r.json() << '\n';
        assert(std::strcmp(r.status, "drain_timestamp_timeout") == 0);
        assert(r.first_audible_ns == -1 && r.playback_complete_ns == -1);
    }
    const auto snapshot = memory.snapshot();
    std::cout << snapshot.json() << '\n';
    assert(snapshot.samples >= 2 && snapshot.smaps_samples > 0);
    assert(snapshot.peak.pss_kib >= snapshot.initial.pss_kib);
}
