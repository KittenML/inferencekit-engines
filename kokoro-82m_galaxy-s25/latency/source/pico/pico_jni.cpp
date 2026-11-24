#include <jni.h>
#include <kitten_tts08.h>

#include <memory>
#include <string>
#ifdef BENCH_PRECISION_MATRIX
#include <array>
#include <cstring>
#endif

namespace {

struct Pico {
    kt08_model* model = nullptr;
    kt08_vocab* vocab = nullptr;
    kt08_tiny_frontend* frontend = nullptr;
    float* voice = nullptr;
    size_t voice_rows = 0;

    ~Pico() {
        kt08_free(voice);
        kt08_tiny_frontend_free(frontend);
        kt08_vocab_free(vocab);
        kt08_model_free(model);
    }
};

bool check(JNIEnv* env, kt08_status status) {
    if (status == KT08_OK) return true;
    env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), kt08_last_error());
    return false;
}

std::string utf8(JNIEnv* env, jstring value) {
    jclass type = env->FindClass("java/lang/String");
    jmethodID get_bytes = env->GetMethodID(type, "getBytes", "(Ljava/lang/String;)[B");
    jstring encoding = env->NewStringUTF("UTF-8");
    auto bytes = static_cast<jbyteArray>(env->CallObjectMethod(value, get_bytes, encoding));
    if (env->ExceptionCheck()) return {};
    std::string result(env->GetArrayLength(bytes), '\0');
    env->GetByteArrayRegion(bytes, 0, result.size(), reinterpret_cast<jbyte*>(result.data()));
    return result;
}

bool deliver(JNIEnv* env, jobject sink, jmethodID accept, const float* samples, jsize count,
             jsize leading_silence) {
    jfloatArray audio = env->NewFloatArray(count + leading_silence);
    if (!audio) return false;
    env->SetFloatArrayRegion(audio, leading_silence, count, samples);
    if (!env->ExceptionCheck()) env->CallVoidMethod(sink, accept, audio);
    env->DeleteLocalRef(audio);
    return !env->ExceptionCheck();
}

jlong create(JNIEnv* env, jstring model_dir, const std::string& model_name,
             const std::string& phonemizer, jint threads) {
    const std::string dir = utf8(env, model_dir);
    if (env->ExceptionCheck()) return 0;
    const std::string model = dir + "/" + model_name;
    const std::string vocab = dir + "/vocab.json";
    const std::string g2p = dir + "/g2p";
    const std::string voice = dir + "/eleanor.bin";
    auto pico = std::make_unique<Pico>();
    kt08_options options;
    kt08_options_init(&options);
    options.threads = threads;
    if (!check(env, kt08_model_load((model + "_arch.json").c_str(), (model + ".bin").c_str(),
                                   &options, &pico->model)) ||
        !check(env, kt08_vocab_load(vocab.c_str(), &pico->vocab)) ||
        !check(env, kt08_tiny_frontend_load(phonemizer.c_str(), vocab.c_str(), g2p.c_str(), nullptr,
                                           &pico->frontend)) ||
        !check(env, kt08_voice_load(voice.c_str(), &pico->voice, &pico->voice_rows))) {
        return 0;
    }
    return reinterpret_cast<jlong>(pico.release());
}

}  // namespace

#ifndef BENCH_PRECISION_MATRIX
extern "C" JNIEXPORT jlong JNICALL
Java_org_kittenml_s25study_Pico_create(JNIEnv* env, jclass, jstring model_dir, jint threads) {
    return create(env, model_dir, "kitten_tts_2_pico_en_" PICO_PRECISION, "v1", threads);
}
#else
extern "C" JNIEXPORT jlong JNICALL
Java_org_kittenml_s25study_Pico_createVariant(JNIEnv* env, jclass, jstring model_dir, jstring prefix,
                                             jstring precision, jstring phonemizer, jint threads) {
    std::string filename = utf8(env, prefix);
    filename += "_";
    filename += utf8(env, precision);
    return create(env, model_dir, filename, utf8(env, phonemizer), threads);
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_kittenml_s25study_Pico_buildInfo(JNIEnv* env, jclass) {
    return env->NewStringUTF(kt08_build_info());
}
#endif

extern "C" JNIEXPORT void JNICALL
Java_org_kittenml_s25study_Pico_generate(JNIEnv* env, jclass, jlong handle, jstring input,
                                       jfloat speed, jobject sink) {
    auto* pico = reinterpret_cast<Pico*>(handle);
    const std::string text = utf8(env, input);
    if (env->ExceptionCheck()) return;
    jmethodID accept = env->GetMethodID(env->GetObjectClass(sink), "accept", "([F)V");
    if (!accept) return;

#ifdef BENCH_PRECISION_MATRIX
    char* raw_phonemes = nullptr;
    if (!check(env, kt08_tiny_text_to_phonemes(pico->frontend, text.c_str(), "en", &raw_phonemes))) {
        return;
    }
    std::unique_ptr<char, decltype(&kt08_free)> phonemes(raw_phonemes, kt08_free);
    if (!phonemes || std::strchr(phonemes.get(), '\n')) {
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"),
                      "Expected one native phoneme chunk for the supplied sentence");
        return;
    }
    std::array<int32_t, 512> ids;
    size_t token_count = 0;
    if (!check(env, kt08_vocab_ids(pico->vocab, phonemes.get(), ids.data(), ids.size(), &token_count))) {
        return;
    }
    float* raw_samples = nullptr;
    size_t count = 0;
    const size_t style = kt08_style_row_for_ids(token_count, pico->voice_rows);
    const auto status = kt08_synthesize_tiny(pico->model, ids.data(), token_count, pico->voice,
                                             pico->voice_rows, style, speed, 1, &raw_samples, &count);
    std::unique_ptr<float, decltype(&kt08_free)> samples(raw_samples, kt08_free);
    if (check(env, status) && count > 0) deliver(env, sink, accept, samples.get(), count, 0);
#else
    kt08_stream* raw_stream = nullptr;
    if (!check(env, kt08_stream_open(pico->model, pico->vocab, pico->frontend, text.c_str(),
                                    "en", pico->voice, pico->voice_rows, speed, nullptr,
                                    &raw_stream))) {
        return;
    }
    std::unique_ptr<kt08_stream, decltype(&kt08_stream_free)> stream(raw_stream, kt08_stream_free);
    constexpr jsize inter_chunk_silence = KT08_SAMPLE_RATE * 15 / 100;
    bool emitted = false;
    int last = 0;
    while (!last) {
        float* raw_samples = nullptr;
        size_t count = 0;
        const auto status = kt08_stream_next(stream.get(), &raw_samples, &count, &last);
        std::unique_ptr<float, decltype(&kt08_free)> samples(raw_samples, kt08_free);
        if (!check(env, status)) return;
        if (count == 0) continue;
        if (!deliver(env, sink, accept, samples.get(), count, emitted ? inter_chunk_silence : 0)) {
            return;
        }
        emitted = true;
    }
#endif
}

extern "C" JNIEXPORT void JNICALL
Java_org_kittenml_s25study_Pico_destroy(JNIEnv*, jclass, jlong handle) {
    delete reinterpret_cast<Pico*>(handle);
}
