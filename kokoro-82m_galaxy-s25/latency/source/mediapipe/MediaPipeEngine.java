package org.kittenml.s25study;

import android.util.Log;

import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.core.Delegate;
import com.google.mediapipe.tasks.audio.texttospeech.AudioResult;
import com.google.mediapipe.tasks.audio.texttospeech.TextToSpeech;
import com.google.mediapipe.tasks.audio.texttospeech.TextToSpeechOptions;

public final class MediaPipeEngine implements SpeechRequest.Engine, AutoCloseable {
    private final TextToSpeech engine;

    public MediaPipeEngine(String modelPath, String cachePath) throws Exception {
        Log.i("S25Release", "MediaPipe Tasks Audio 1.2.2 requested GPU, af_heart, target bucket 64");
        BaseOptions base = BaseOptions.builder().setModelAssetPath(modelPath)
            .setDelegate(Delegate.GPU).build();
        engine = TextToSpeech.create(new TextToSpeechOptions(
            base, "af_heart", "en-US", 1.0f, 64, cachePath));
    }

    @Override public void generate(String text, Sink sink) {
        try {
            AudioResult audio = engine.synthesize(text);
            if (audio.getSampleRateHz() != Playback.RATE || audio.getSamples().length == 0) {
                throw new IllegalStateException("Unexpected MediaPipe waveform");
            }
            sink.accept(audio.getSamples());
        } catch (Exception error) {
            throw new IllegalStateException("MediaPipe synthesis failed", error);
        }
    }

    @Override public void close() throws Exception {
        engine.close();
    }
}
