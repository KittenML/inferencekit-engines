package org.inferencekit.mediapipecapture;

import android.app.Activity;
import android.app.Instrumentation;
import android.content.Intent;
import android.os.Build;
import android.os.Bundle;
import android.os.PowerManager;
import android.os.Process;
import android.os.SystemClock;
import android.util.Log;

import com.google.mediapipe.tasks.audio.texttospeech.AudioResult;
import com.google.mediapipe.tasks.audio.texttospeech.TextToSpeech;
import com.google.mediapipe.tasks.audio.texttospeech.TextToSpeechOptions;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.core.Delegate;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.LockSupport;
import org.json.JSONArray;
import org.json.JSONObject;

public final class MediaPipeCapture extends Instrumentation {
    private static final int RATE = 24000;
    private static final String TAG = "InferenceKitTTS";
    private Bundle arguments;
    private Activity foreground;

    private static final class Chunk {
        final JSONArray timing;
        final byte[] pcm;
        final long arrival;
        final int frames;

        Chunk(JSONArray timing, byte[] pcm, long arrival, int frames) {
            this.timing = timing;
            this.pcm = pcm;
            this.arrival = arrival;
            this.frames = frames;
        }
    }

    @Override
    public void onCreate(Bundle arguments) {
        super.onCreate(arguments);
        this.arguments = arguments;
        start();
    }

    @Override
    public void onStart() {
        Bundle result = new Bundle();
        result.putInt("pid", Process.myPid());
        int status = Activity.RESULT_CANCELED;
        try {
            if (arguments.getString("foreground", "0").equals("1")) {
                Intent intent = new Intent(getTargetContext(), CaptureActivity.class)
                        .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
                foreground = startActivitySync(intent);
                waitForIdleSync();
                Log.i(TAG, "foreground_activity=" + foreground.getClass().getSimpleName());
            }
            capture();
            result.putString("status", "complete");
            status = Activity.RESULT_OK;
        } catch (Exception | LinkageError error) {
            Log.e(TAG, "capture failed", error);
            result.putString("error", error.toString());
        } finally {
            if (foreground != null) {
                foreground.finish();
            }
            finish(status, result);
        }
    }

    private void logSchedulingState(String stage) {
        Log.i(TAG, "process_stats_stage=" + stage + " activity=" +
                (foreground == null ? "none" : foreground.getClass().getSimpleName()));
        if (foreground != null) {
            runOnMainSync(() -> Log.i(TAG, "activity_window_focus=" + foreground.hasWindowFocus()));
        }
        if (Build.VERSION.SDK_INT >= 29) {
            PowerManager power = getTargetContext().getSystemService(PowerManager.class);
            if (power != null) {
                Log.i(TAG, "process_thermal_status=" + power.getCurrentThermalStatus());
            }
        }
        try {
            for (String line : Files.readAllLines(new File("/proc/self/status").toPath(), StandardCharsets.UTF_8)) {
                if (line.startsWith("Cpus_allowed_list:") || line.startsWith("Threads:") ||
                        line.startsWith("VmHWM:")) {
                    Log.i(TAG, line);
                }
            }
        } catch (IOException error) {
            Log.w(TAG, "/proc/self/status unavailable", error);
        }
    }

    private int integer(String key) {
        return Integer.parseInt(arguments.getString(key));
    }

    private static long now() {
        return SystemClock.elapsedRealtimeNanos();
    }

    private static double elapsed(long start) {
        return (now() - start) / 1.0e6;
    }

    private static String[] synthesisInputs(String[] fixture, boolean streaming) {
        if (streaming) {
            return new String[]{fixture[1], fixture[2]};
        }
        return new String[]{fixture[1] + " " + fixture[2]};
    }

    private static void waitUntil(long release) {
        long remaining;
        while ((remaining = release - now()) > 0) {
            LockSupport.parkNanos(remaining);
        }
    }

    private static Chunk encodeChunk(TextToSpeech engine, String text, long release, boolean savePcm)
            throws Exception {
        AudioResult audio = engine.synthesize(text);
        long arrival = now();
        double firstPcm = (arrival - release) / 1.0e6;
        float[] samples = audio.getSamples();
        if (samples.length == 0 || audio.getSampleRateHz() != RATE) {
            throw new IllegalStateException("unexpected MediaPipe waveform");
        }
        int leading = samples.length;
        ByteBuffer pcm = savePcm
                ? ByteBuffer.allocate(samples.length * 2).order(ByteOrder.LITTLE_ENDIAN) : null;
        for (int frame = 0; frame < samples.length; ++frame) {
            float sample = samples[frame];
            if (!Float.isFinite(sample)) {
                throw new IllegalStateException("nonfinite MediaPipe audio");
            }
            short value = (short) Math.round(32767.0f * Math.max(-1.0f, Math.min(1.0f, sample)));
            if (leading == samples.length && value != 0) {
                leading = frame;
            }
            if (pcm != null) {
                pcm.putShort(value);
            }
        }
        if (leading == samples.length) {
            throw new IllegalStateException("silent MediaPipe audio");
        }
        JSONArray timing = new JSONArray().put(firstPcm).put(samples.length).put(leading);
        return new Chunk(timing, pcm == null ? null : pcm.array(), arrival, samples.length);
    }

    private void capture() throws Exception {
        File internal = getTargetContext().getFilesDir();
        Path files = internal.toPath();
        String delegateName = arguments.getString("delegate");
        Delegate delegate = Delegate.valueOf(delegateName);
        String bundle = arguments.getString("bundle", delegateName);
        if (!bundle.equals("CPU") && !bundle.equals("GPU")) {
            throw new IllegalArgumentException("unknown MediaPipe bundle");
        }
        File model = new File(internal, "kokoro-" + bundle + ".litertlm");
        List<String> fixtures = Files.readAllLines(files.resolve("kokoro.tsv"), StandardCharsets.UTF_8);
        String[] warmup = fixtures.get(fixtures.size() - 1).split("\t", -1);
        int concurrency = integer("concurrency");
        int waves = integer("waves");
        int round = integer("round");
        long period = Math.multiplyExact(integer("period_ms"), 1000000L);
        String mode = arguments.getString("mode", "stream");
        boolean streaming = mode.equals("stream");
        int chunkCount = streaming ? 2 : 1;
        if (concurrency < 1 || waves < 1 || period < 1 || fixtures.size() <= waves * concurrency
                || (!streaming && !mode.equals("full"))) {
            throw new IllegalArgumentException("invalid capture dimensions");
        }
        int poolSize = Integer.parseInt(arguments.getString("pool_size", "0"));
        int bucket = Integer.parseInt(arguments.getString("target_bucket", "64"));
        String policy = arguments.getString("pool_policy", "deadline");
        if (poolSize < 0 || (poolSize > 0 && delegate != Delegate.GPU)
                || (!policy.equals("deadline") && !policy.equals("fair"))) {
            throw new IllegalArgumentException("invalid GPU pool configuration");
        }
        if (bucket < 0 || bucket > 120) {
            throw new IllegalArgumentException("invalid Kokoro text bucket");
        }
        Path pcm = Files.createDirectories(files.resolve("pcm"));
        String[] rows = new String[concurrency * waves];
        boolean saveAll = arguments.getString("save_pcm", "0").equals("1");
        byte[][] savedPcm = new byte[(saveAll ? rows.length : concurrency) * chunkCount][];
        AtomicReference<Throwable> failed = new AtomicReference<>();
        Log.i(TAG, "MediaPipe Tasks Audio " + RuntimeVersion.NAME + " delegate=" + delegateName +
                " bundle=" + bundle + " target_bucket=" + bucket + " threads=runtime-default");
        logSchedulingState("before-init");
        BaseOptions base = BaseOptions.builder().setModelAssetPath(model.getAbsolutePath())
                .setDelegate(delegate).build();
        TextToSpeechOptions options = new TextToSpeechOptions(
                base, "af_heart", "en-US", 1.0f, bucket, getTargetContext().getCacheDir().getAbsolutePath());
        Log.i(TAG, "scheduler=" + (poolSize == 0 ? "stock" : policy)
                + " engines=" + (poolSize == 0 ? concurrency : poolSize)
                + " concurrency=" + concurrency + " waves=" + waves + " mode=" + mode);
        if (poolSize > 0) {
            capturePooled(options, warmup, fixtures, streaming, period, concurrency, waves,
                    round, poolSize, policy.equals("deadline"), rows, savedPcm, failed);
        } else {
            captureStock(options, warmup, fixtures, streaming, period, concurrency, waves,
                    round, rows, savedPcm, failed);
        }
        logSchedulingState("after-workers-close");
        for (int index = 0; index < savedPcm.length / chunkCount; ++index) {
            int stream = index % concurrency;
            String id = fixtures.get(index).split("\t", 2)[0];
            String stem = id + "-r" + round + "-n" + concurrency + "-s" + stream;
            for (int chunk = 0; chunk < chunkCount; ++chunk) {
                byte[] data = savedPcm[index * chunkCount + chunk];
                if (data != null) {
                    Files.write(pcm.resolve(stem + "-c" + chunk + ".s16le"), data);
                }
            }
        }
        try (BufferedWriter output = Files.newBufferedWriter(files.resolve("capture.jsonl"))) {
            for (String row : rows) {
                output.write(row);
                output.newLine();
            }
        }
        if (failed.get() != null) {
            throw new IllegalStateException("MediaPipe worker failed", failed.get());
        }
    }

    private void captureStock(TextToSpeechOptions options, String[] warmup, List<String> fixtures,
            boolean streaming, long period, int concurrency, int waves, int round,
            String[] rows, byte[][] savedPcm, AtomicReference<Throwable> failed) throws Exception {
        int chunkCount = streaming ? 2 : 1;
        Thread[] workers = new Thread[concurrency];
        CountDownLatch ready = new CountDownLatch(concurrency);
        CountDownLatch start = new CountDownLatch(1);
        AtomicLong origin = new AtomicLong();
        for (int stream = 0; stream < concurrency; ++stream) {
            final int slot = stream;
            workers[stream] = new Thread(() -> {
                boolean warmed = false;
                try (TextToSpeech engine = TextToSpeech.create(options)) {
                    for (String input : synthesisInputs(warmup, streaming)) {
                        encodeChunk(engine, input, now(), false);
                    }
                    warmed = true;
                    ready.countDown();
                    start.await();
                    if (failed.get() != null) {
                        return;
                    }
                    for (int wave = 0; wave < waves; ++wave) {
                        int index = wave * concurrency + slot;
                        String[] fixture = fixtures.get(index).split("\t", -1);
                        long release = origin.get() + period * wave;
                        waitUntil(release);
                        JSONArray chunks = new JSONArray();
                        Object completed = JSONObject.NULL;
                        try {
                            String[] inputs = synthesisInputs(fixture, streaming);
                            for (int part = 0; part < inputs.length; ++part) {
                                boolean save = index < savedPcm.length / chunkCount;
                                Chunk chunk = encodeChunk(engine, inputs[part], release, save);
                                chunks.put(chunk.timing);
                                if (save) {
                                    savedPcm[index * chunkCount + part] = chunk.pcm;
                                }
                            }
                            completed = elapsed(release);
                        } catch (Exception error) {
                            Log.w(TAG, "synthesis failed for " + fixture[0], error);
                        }
                        rows[index] = new JSONObject().put("round", round).put("concurrency", concurrency)
                                .put("wave", wave).put("stream", slot).put("id", fixture[0])
                                .put("complete_ms", completed).put("chunks", chunks).toString();
                    }
                } catch (Exception | LinkageError error) {
                    failed.compareAndSet(null, error);
                } finally {
                    if (!warmed) {
                        ready.countDown();
                    }
                }
            }, "sentence-stream-" + stream);
            workers[stream].start();
        }
        ready.await();
        logSchedulingState("warmed");
        origin.set(now() + 100000000L);
        start.countDown();
        for (Thread worker : workers) {
            worker.join();
        }
        fillMissingRows(fixtures, concurrency, round, rows, null);
    }

    private void capturePooled(TextToSpeechOptions options, String[] warmup, List<String> fixtures,
            boolean streaming, long period, int concurrency, int waves, int round, int poolSize,
            boolean deadlineOrder, String[] rows, byte[][] savedPcm, AtomicReference<Throwable> failed)
            throws Exception {
        int chunkCount = streaming ? 2 : 1;
        SentenceQueue queue = new SentenceQueue(concurrency, rows.length, period, deadlineOrder);
        CountDownLatch ready = new CountDownLatch(poolSize);
        Thread[] workers = new Thread[poolSize];
        String[][] inputs = new String[rows.length][];
        JSONArray[] timings = new JSONArray[rows.length];
        for (int index = 0; index < rows.length; ++index) {
            inputs[index] = synthesisInputs(fixtures.get(index).split("\t", -1), streaming);
            timings[index] = new JSONArray();
        }
        for (int worker = 0; worker < poolSize; ++worker) {
            workers[worker] = new Thread(() -> {
                boolean warmed = false;
                try (TextToSpeech engine = TextToSpeech.create(options)) {
                    for (String input : synthesisInputs(warmup, streaming)) {
                        encodeChunk(engine, input, now(), false);
                    }
                    warmed = true;
                    ready.countDown();
                    SentenceQueue.Request request;
                    while ((request = queue.take()) != null) {
                        int index = request.index;
                        int part = request.part;
                        Chunk chunk;
                        try {
                            chunk = encodeChunk(engine, inputs[index][part], request.release,
                                    index < savedPcm.length / chunkCount);
                        } catch (Exception error) {
                            Log.w(TAG, "synthesis failed at request " + index + " part " + part, error);
                            rows[index] = row(fixtures, round, concurrency, index,
                                    JSONObject.NULL, timings[index]);
                            queue.finish(request);
                            continue;
                        }
                        timings[index].put(chunk.timing);
                        if (index < savedPcm.length / chunkCount) {
                            savedPcm[index * chunkCount + part] = chunk.pcm;
                        }
                        if (part + 1 == chunkCount) {
                            rows[index] = row(fixtures, round, concurrency, index,
                                    elapsed(request.release), timings[index]);
                            queue.finish(request);
                        } else {
                            // Playback starts at first PCM, including its leading silence.
                            long playbackDeadline = chunk.arrival + chunk.frames * 1000000000L / RATE;
                            queue.continueAt(request, playbackDeadline);
                        }
                    }
                } catch (Exception | LinkageError error) {
                    failed.compareAndSet(null, error);
                    queue.abort();
                } finally {
                    if (!warmed) {
                        ready.countDown();
                    }
                }
            }, "gpu-engine-" + worker);
            workers[worker].start();
        }
        boolean releasedAll = false;
        try {
            ready.await();
            logSchedulingState("warmed");
            long origin = now() + 100000000L;
            for (int wave = 0; wave < waves && failed.get() == null; ++wave) {
                long release = origin + period * wave;
                waitUntil(release);
                queue.releaseWave(release);
            }
            releasedAll = failed.get() == null;
        } finally {
            // An interrupted instrumentation thread must release sleeping owners.
            if (!releasedAll) {
                queue.abort();
            }
            for (Thread worker : workers) {
                worker.join();
            }
        }
        fillMissingRows(fixtures, concurrency, round, rows, timings);
    }

    private static void fillMissingRows(List<String> fixtures, int concurrency, int round,
            String[] rows, JSONArray[] timings) throws Exception {
        for (int index = 0; index < rows.length; ++index) {
            if (rows[index] == null) {
                rows[index] = row(fixtures, round, concurrency, index, JSONObject.NULL,
                        timings == null ? new JSONArray() : timings[index]);
            }
        }
    }

    private static String row(List<String> fixtures, int round, int concurrency, int index,
            Object completed, JSONArray chunks) throws Exception {
        return new JSONObject().put("round", round).put("concurrency", concurrency)
                .put("wave", index / concurrency).put("stream", index % concurrency)
                .put("id", fixtures.get(index).split("\t", 2)[0])
                .put("complete_ms", completed).put("chunks", chunks).toString();
    }
}
