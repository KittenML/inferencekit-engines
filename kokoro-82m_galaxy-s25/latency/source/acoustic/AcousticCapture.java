package org.kittenml.s25study;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.media.AudioDeviceInfo;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioRecord;
import android.media.AudioRecordingConfiguration;
import android.media.AudioTimestamp;
import android.media.MediaRecorder;
import android.media.audiofx.AudioEffect;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

final class AcousticCapture implements AutoCloseable {
    private static final int RATE = 48000;
    private static final int ANALYSIS_RATE = 6000;
    private static final int STRIDE = RATE / ANALYSIS_RATE;
    private static final int READ_FRAMES = RATE / 100;
    private static final int CAPACITY = RATE * 15;
    private static final long NS = 1_000_000_000L;

    private final AudioRecord record;
    private final String source;
    private final short[] audio;
    private final CountDownLatch ready = new CountDownLatch(1);
    private final long[] stampFrames = new long[CAPACITY / READ_FRAMES];
    private final long[] stampTimes = new long[stampFrames.length];
    private final Thread reader;
    private volatile String failure;
    private volatile boolean stop;
    private int frames;
    private int timestamps;
    private int micType = -1;
    private boolean closed;

    private AcousticCapture(AudioRecord record, String source, String failure) {
        this.record = record;
        this.source = source;
        this.failure = failure;
        audio = record == null ? null : new short[CAPACITY];
        reader = record == null ? null : new Thread(this::capture, "study-microphone");
    }

    static AcousticCapture start(Context context) throws InterruptedException {
        if (context.checkSelfPermission(Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            return new AcousticCapture(null, null, "permission-denied");
        }
        AudioManager audioManager = context.getSystemService(AudioManager.class);
        boolean raw = Boolean.parseBoolean(audioManager.getProperty(
            AudioManager.PROPERTY_SUPPORT_AUDIO_SOURCE_UNPROCESSED));
        int sourceType = raw ? MediaRecorder.AudioSource.UNPROCESSED
                             : MediaRecorder.AudioSource.VOICE_RECOGNITION;
        int min = AudioRecord.getMinBufferSize(RATE, AudioFormat.CHANNEL_IN_MONO,
                                               AudioFormat.ENCODING_PCM_16BIT);
        if (min <= 0) return new AcousticCapture(null, null, "unsupported-input-format");

        AudioRecord record = null;
        AcousticCapture capture;
        try {
            record = new AudioRecord.Builder().setAudioSource(sourceType)
                .setAudioFormat(new AudioFormat.Builder().setSampleRate(RATE)
                    .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT).build())
                .setBufferSizeInBytes(Math.max(min, RATE / 10 * Short.BYTES))
                .build();
            if (record.getState() != AudioRecord.STATE_INITIALIZED || record.getSampleRate() != RATE) {
                record.release();
                return new AcousticCapture(null, null, "input-initialization-failed");
            }
            for (AudioDeviceInfo device : audioManager.getDevices(AudioManager.GET_DEVICES_INPUTS)) {
                if (device.getType() == AudioDeviceInfo.TYPE_BUILTIN_MIC) {
                    record.setPreferredDevice(device);
                    break;
                }
            }
            record.startRecording();
            capture = new AcousticCapture(record, raw ? "unprocessed" : "voice-recognition", null);
        } catch (RuntimeException error) {
            if (record != null) record.release();
            return new AcousticCapture(null, null, "input-start-failed");
        }
        capture.reader.start();
        try {
            if (!capture.ready.await(750, TimeUnit.MILLISECONDS)) {
                capture.failure = "input-not-ready";
                capture.close();
            }
            return capture;
        } catch (InterruptedException error) {
            capture.close();
            throw error;
        }
    }

    private void capture() {
        AudioTimestamp stamp = new AudioTimestamp();
        int nextRouteCheck = RATE / 20;
        try {
            while (frames < audio.length && !stop) {
                int count = record.read(audio, frames, Math.min(READ_FRAMES, audio.length - frames),
                                        AudioRecord.READ_BLOCKING);
                if (stop) break;
                if (count <= 0) {
                    failure = "input-read-failed";
                    break;
                }
                frames += count;
                if (timestamps < stampTimes.length
                        && record.getTimestamp(stamp, AudioTimestamp.TIMEBASE_MONOTONIC) == AudioRecord.SUCCESS
                        && stamp.framePosition >= 0 && stamp.nanoTime > 0
                        && (timestamps == 0 || stamp.framePosition > stampFrames[timestamps - 1]
                            && stamp.nanoTime > stampTimes[timestamps - 1])) {
                    stampFrames[timestamps] = stamp.framePosition;
                    stampTimes[timestamps++] = stamp.nanoTime;
                }
                if (frames >= nextRouteCheck) {
                    inspectRoute();
                    nextRouteCheck = frames + RATE / 10;
                }
                if (frames >= RATE / 20 && timestamps >= 2) ready.countDown();
                if (failure != null) break;
            }
            if (!stop && failure == null) inspectRoute();
        } catch (RuntimeException error) {
            if (!stop) failure = "input-read-failed";
        } finally {
            if (!stop) stopRecord();
            ready.countDown();
        }
    }

    private void inspectRoute() {
        AudioDeviceInfo device = record.getRoutedDevice();
        if (device != null) micType = device.getType();
        AudioRecordingConfiguration config = record.getActiveRecordingConfiguration();
        if (micType >= 0 && micType != AudioDeviceInfo.TYPE_BUILTIN_MIC) {
            failure = "not-built-in-microphone";
        } else if (config != null && config.isClientSilenced()) {
            failure = "microphone-silenced";
        } else if (config != null && config.getEffects().stream().anyMatch(effect ->
                       AudioEffect.EFFECT_TYPE_AEC.equals(effect.type)
                       || AudioEffect.EFFECT_TYPE_AGC.equals(effect.type)
                       || AudioEffect.EFFECT_TYPE_NS.equals(effect.type))) {
            failure = "microphone-voice-effects-active";
        }
    }

    JSONObject finish(long requestStartNs, long playCallNs, List<short[]> knownPcm,
                      int knownRate, int onsetFrame, int outputRoute) throws JSONException {
        JSONObject result = new JSONObject().put("source", source == null ? JSONObject.NULL : source);
        try {
            if (record == null || closed) {
                return result.put("status", failure == null ? "closed" : failure);
            }
            halt();
            if (failure != null) return result.put("status", failure);
            if (micType != AudioDeviceInfo.TYPE_BUILTIN_MIC) {
                return result.put("status", "no-confirmed-microphone-route");
            }
            if (outputRoute != AudioDeviceInfo.TYPE_BUILTIN_SPEAKER
                    && outputRoute != AudioDeviceInfo.TYPE_BUILTIN_SPEAKER_SAFE) {
                return result.put("status", "not-confirmed-built-in-speaker");
            }
            if (timestamps < 2 || playCallNs < requestStartNs) {
                return result.put("status", "no-stable-capture-clock");
            }
            float[] reference = reference(knownPcm, knownRate, onsetFrame);
            if (reference.length < ANALYSIS_RATE * 80 / 1000) {
                return result.put("status", "short-reference-speech");
            }
            long roughEpoch = stampTimes[0] - stampFrames[0] * NS / RATE;
            // Search from just before the earliest possible emission to the end of the microphone window.
            int begin = Math.max(0, (int) ((playCallNs - roughEpoch - 5_000_000) * ANALYSIS_RATE / NS));
            float[] mic = downsample(audio, frames, STRIDE);
            Match match;
            try {
                match = match(reference, mic, begin);
            } finally {
                Arrays.fill(mic, 0);
            }
            if (match == null) return result.put("status", "no-acoustic-match");
            result.put("correlation", match.correlation).put("competing_correlation", match.competing);
            if (match.correlation < 0.45 || match.correlation - match.competing < 0.055) {
                return result.put("status", "weak-or-ambiguous-match");
            }
            int matchedFrame = match.position * STRIDE;
            long captureNs = timeAt(matchedFrame);
            if (captureNs <= 0 || captureNs < playCallNs - 2_000_000) {
                return result.put("status", "no-stable-clock-at-match");
            }
            return result.put("status", "matched-speaker-at-phone-mic")
                .put("matched_after_request_ms", (captureNs - requestStartNs) / 1e6)
                .put("matched_after_play_call_ms", (captureNs - playCallNs) / 1e6);
        } finally {
            close();
        }
    }

    private long timeAt(int frame) {
        long[] estimates = new long[timestamps];
        int nearby = 0;
        for (int i = 0; i < timestamps; i++) {
            if (Math.abs(frame - stampFrames[i]) <= RATE / 4) {
                estimates[nearby++] = stampTimes[i] + (frame - stampFrames[i]) * NS / RATE;
            }
        }
        if (nearby < 3) return -1;
        Arrays.sort(estimates, 0, nearby);
        if (estimates[nearby - 1] - estimates[0] > 2_000_000L) return -1;
        return estimates[nearby / 2];
    }

    private static float[] reference(List<short[]> chunks, int rate, int onset) {
        if (rate <= 0 || rate % ANALYSIS_RATE != 0 || onset < 0) return new float[0];
        short[] samples = new short[rate * 150 / 1000];
        int written = 0;
        int skip = onset;
        for (short[] chunk : chunks) {
            if (skip >= chunk.length) {
                skip -= chunk.length;
                continue;
            }
            int count = Math.min(chunk.length - skip, samples.length - written);
            System.arraycopy(chunk, skip, samples, written, count);
            written += count;
            skip = 0;
            if (written == samples.length) break;
        }
        return downsample(samples, written, rate / ANALYSIS_RATE);
    }

    static float[] downsample(short[] samples, int count, int stride) {
        float[] signal = new float[count / stride];
        float previous = 0;
        for (int i = 0; i < signal.length; i++) {
            int sum = 0;
            for (int j = 0; j < stride; j++) sum += samples[i * stride + j];
            float average = (float) sum / stride;
            if (i > 0) signal[i] = average - 0.9f * previous;
            previous = average;
        }
        return signal;
    }

    static final class Match {
        final int position;
        final double correlation;
        final double competing;

        Match(int position, double correlation, double competing) {
            this.position = position;
            this.correlation = correlation;
            this.competing = competing;
        }
    }

    static Match match(float[] reference, float[] mic, int begin) {
        final int length = reference.length;
        final int end = mic.length - length;
        if (begin > end) return null;
        double referenceEnergy = 0;
        for (float value : reference) referenceEnergy += value * value;
        if (referenceEnergy <= 0) return null;
        double micEnergy = 0;
        for (int i = begin; i < begin + length; i++) micEnergy += mic[i] * mic[i];
        double[] scores = new double[end - begin + 1];
        double best = 0;
        int bestPosition = -1;
        for (int p = begin; p <= end; p++) {
            if (micEnergy > 16 * length) {
                double dot = 0;
                for (int i = 0; i < length; i++) dot += reference[i] * mic[p + i];
                double score = Math.abs(dot) / Math.sqrt(referenceEnergy * micEnergy);
                scores[p - begin] = score;
                if (score > best) {
                    best = score;
                    bestPosition = p;
                }
            }
            if (p < end) micEnergy += mic[p + length] * mic[p + length] - mic[p] * mic[p];
        }
        if (bestPosition < 0) return null;
        double competing = 0;
        for (int p = begin; p <= end; p++) {
            if (Math.abs(p - bestPosition) >= ANALYSIS_RATE * 15 / 1000) {
                competing = Math.max(competing, scores[p - begin]);
            }
        }
        return new Match(bestPosition, best, competing);
    }

    private void halt() {
        if (record == null || stop) return;
        stop = true;
        stopRecord();
        try {
            reader.join(500);
            if (reader.isAlive()) failure = "input-stop-timed-out";
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            failure = "input-stop-interrupted";
        }
    }

    private void stopRecord() {
        try {
            record.stop();
        } catch (IllegalStateException ignored) {}
    }

    @Override public void close() {
        if (closed) return;
        closed = true;
        halt();
        if (record != null) record.release();
        if (audio != null) Arrays.fill(audio, (short) 0);
    }
}
