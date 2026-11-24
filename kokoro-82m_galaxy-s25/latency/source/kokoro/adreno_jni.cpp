#include <jni.h>

#include <algorithm>
#include <cstdlib>
#include <memory>
#include "kokoro_pcm_process.h"

namespace {
void fail(JNIEnv* env, const char* message) {
    env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), message);
}

std::string read_string(JNIEnv* env, jstring input) {
    const char* value = env->GetStringUTFChars(input, nullptr);
    if (!value) return {};
    std::string result(value);
    env->ReleaseStringUTFChars(input, value);
    return result;
}
}

extern "C" JNIEXPORT jlong JNICALL Java_org_kittenml_s25study_Adreno_create(
    JNIEnv* env, jclass, jstring data_path, jstring library_path) {
    const auto path = read_string(env, library_path);
    const auto data = read_string(env, data_path);
    if (env->ExceptionCheck()) return 0;
    std::signal(SIGPIPE, SIG_IGN);
    setenv("LD_LIBRARY_PATH", (path + ":/vendor/lib64").c_str(), 1);
    setenv("NNOPT_GPU_FP32_GENERATOR", "1", 1);
    setenv("NNOPT_DOT8", "0", 1);
    setenv("NNOPT_DOT8_GEMM", "0", 1);
    try {
        auto worker = std::make_unique<KokoroPcmProcess>(data, std::vector<std::string>{
            path + "/libstudy_adreno_worker.so", "--serve-stream", "--voice", "en-us",
        });
        return reinterpret_cast<jlong>(worker.release());
    } catch (const std::exception& error) {
        fail(env, error.what());
        return 0;
    }
}

extern "C" JNIEXPORT void JNICALL Java_org_kittenml_s25study_Adreno_generate(
    JNIEnv* env, jclass, jlong handle, jstring input, jobject sink) {
    const auto text = read_string(env, input);
    if (env->ExceptionCheck()) return;
    try {
        auto& worker = *reinterpret_cast<KokoroPcmProcess*>(handle);
        const auto pcm = worker.synthesize(text);
        std::vector<float> audio(pcm.size());
        std::transform(pcm.begin(), pcm.end(), audio.begin(),
                       [](int16_t x) { return x / 32768.0f; });
        jfloatArray samples = env->NewFloatArray(audio.size());
        if (!samples) return;
        env->SetFloatArrayRegion(samples, 0, audio.size(), audio.data());
        jmethodID accept = env->GetMethodID(env->GetObjectClass(sink), "accept", "([F)V");
        if (accept && !env->ExceptionCheck()) env->CallVoidMethod(sink, accept, samples);
    } catch (const std::exception& error) {
        fail(env, error.what());
    }
}

extern "C" JNIEXPORT void JNICALL Java_org_kittenml_s25study_Adreno_destroy(
    JNIEnv*, jclass, jlong handle) {
    delete reinterpret_cast<KokoroPcmProcess*>(handle);
}
