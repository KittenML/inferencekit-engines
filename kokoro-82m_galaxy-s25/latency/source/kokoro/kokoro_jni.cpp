#include <jni.h>

#include <string>

#include "sherpa-onnx/c-api/c-api.h"

namespace {

void fail(JNIEnv* env, const char* message) {
  env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), message);
}

std::string read_string(JNIEnv* env, jstring input) {
  const char* converted = env->GetStringUTFChars(input, nullptr);
  if (!converted) return {};
  std::string value(converted);
  env->ReleaseStringUTFChars(input, converted);
  return value;
}

struct Callback {
  JNIEnv* env;
  jobject sink;
  jmethodID accept;
  int64_t frames = 0;
};

int32_t forward_audio(const float* audio, int32_t frames, float, void* user) {
  auto& cb = *static_cast<Callback*>(user);
  if (frames <= 0) return 1;
  jfloatArray array = cb.env->NewFloatArray(frames);
  if (!array) return 0;
  cb.env->SetFloatArrayRegion(array, 0, frames, audio);
  if (!cb.env->ExceptionCheck()) cb.env->CallVoidMethod(cb.sink, cb.accept, array);
  cb.env->DeleteLocalRef(array);
  cb.frames += frames;
  return !cb.env->ExceptionCheck();
}

jlong create(JNIEnv* env, jstring directory, const std::string& model_file, jint threads) {
  const std::string base = read_string(env, directory);
  if (env->ExceptionCheck()) return 0;
  const std::string model = base + "/" + model_file;
  const std::string voices = base + "/voices.bin";
  const std::string tokens = base + "/tokens.txt";
  const std::string espeak = base + "/espeak-ng-data";
  const std::string lexicon = base + "/lexicon-us-en.txt";
  SherpaOnnxOfflineTtsConfig config{};
  config.model.kokoro.model = model.c_str();
  config.model.kokoro.voices = voices.c_str();
  config.model.kokoro.tokens = tokens.c_str();
  config.model.kokoro.data_dir = espeak.c_str();
  config.model.kokoro.lexicon = lexicon.c_str();
  config.model.kokoro.lang = "en-us";
  config.model.num_threads = threads;
  config.model.provider = "cpu";
  config.max_num_sentences = 1;
  config.silence_scale = 0.2f;
  const auto* handle = SherpaOnnxCreateOfflineTts(&config);
  if (!handle) {
    fail(env, "Sherpa could not initialize Kokoro");
    return 0;
  }
  if (SherpaOnnxOfflineTtsSampleRate(handle) != 24000) {
    SherpaOnnxDestroyOfflineTts(handle);
    fail(env, "Kokoro sample rate differs from the 24 kHz playback contract");
    return 0;
  }
  return reinterpret_cast<jlong>(handle);
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL Java_org_kittenml_s25study_Sherpa_create(
    JNIEnv* env, jclass, jstring directory, jint threads) {
  return create(env, directory, "model.onnx", threads);
}

extern "C" JNIEXPORT jlong JNICALL Java_org_kittenml_s25study_Sherpa_createVariant(
    JNIEnv* env, jclass, jstring directory, jstring model_file, jint threads) {
  return create(env, directory, read_string(env, model_file), threads);
}

extern "C" JNIEXPORT void JNICALL Java_org_kittenml_s25study_Sherpa_generate(
    JNIEnv* env, jclass, jlong handle, jstring input, jfloat speed, jobject sink) {
  const std::string text = read_string(env, input);
  if (env->ExceptionCheck()) return;
  const auto sink_class = env->GetObjectClass(sink);
  const auto accept = env->GetMethodID(sink_class, "accept", "([F)V");
  if (!accept) return;
  Callback cb{env, sink, accept};
  SherpaOnnxGenerationConfig config{};
  config.speed = speed;
  config.sid = 3;  // af_heart in sherpa-onnx's v1.0 voice map.
  config.silence_scale = 0.2f;
  const auto* audio = SherpaOnnxOfflineTtsGenerateWithConfig(
      reinterpret_cast<const SherpaOnnxOfflineTts*>(handle), text.c_str(), &config,
      forward_audio, &cb);
  const bool complete = audio && audio->n > 0 && audio->sample_rate == 24000 &&
                        audio->n == cb.frames;
  if (audio) SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
  if (!complete && !env->ExceptionCheck()) fail(env, "Kokoro callback frame count is incomplete");
}

extern "C" JNIEXPORT void JNICALL Java_org_kittenml_s25study_Sherpa_destroy(
    JNIEnv*, jclass, jlong handle) {
  SherpaOnnxDestroyOfflineTts(reinterpret_cast<const SherpaOnnxOfflineTts*>(handle));
}

extern "C" JNIEXPORT jstring JNICALL Java_org_kittenml_s25study_Sherpa_version(
    JNIEnv* env, jclass) {
  return env->NewStringUTF(SherpaOnnxGetVersionStr());
}
