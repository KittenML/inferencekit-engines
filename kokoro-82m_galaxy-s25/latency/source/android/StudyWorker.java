package org.kittenml.s25study;

import android.app.Service;
import android.content.Intent;
import android.media.AudioManager;
import android.os.Debug;
import android.os.IBinder;
import android.os.Process;

import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public final class StudyWorker extends Service {
    private static final String WARMUP = "The green umbrella is waiting by the stairs.";
    private boolean running;

    @Override public IBinder onBind(Intent intent) { return null; }

    @Override public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent != null && !running) {
            running = true;
            new Thread(() -> execute(intent), "study-synthesis").start();
        }
        return START_NOT_STICKY;
    }

    private static long memory(String key) throws Exception {
        try (BufferedReader reader = new BufferedReader(new FileReader("/proc/self/status"))) {
            for (String line; (line = reader.readLine()) != null;) {
                if (line.startsWith(key + ":")) return Long.parseLong(line.split("\\s+")[1]);
            }
        }
        return -1;
    }

    private static JSONObject warm(SpeechRequest.Engine engine, JSONObject fixture,
                                   boolean stream, boolean play, boolean tolerateFailure) throws Exception {
        try {
            return SpeechRequest.run(engine, fixture, stream, play, null);
        } catch (Exception error) {
            if (!tolerateFailure) throw error;
            return new JSONObject().put("status", "ERROR").put("error", error.toString());
        }
    }

    private void execute(Intent intent) {
        int pid = Process.myPid();
        String label = intent.getStringExtra("label");
        File results = new File(getFilesDir(), "results");
        File destination = new File(results, label);
        File completed = new File(destination, "done.json");
        JSONObject summary = new JSONObject();
        long handle = 0;
        AutoCloseable external = null;
        boolean adreno = false;
        boolean pico = intent.getBooleanExtra("pico", false);
        try {
            StudyFiles.mkdir(destination);
            summary.put("pid", pid).put("conditions_start", StudyFiles.conditions(this));
            int threads = intent.getIntExtra("threads", 2);
            int round = intent.getIntExtra("round", 0);
            boolean screen = intent.getBooleanExtra("screen", false);
            boolean stream = intent.getBooleanExtra("stream", true);
            boolean matchedChunks = intent.getBooleanExtra("matched_chunks", false);
            boolean precisionMatrix = intent.getBooleanExtra("precision_matrix", false);
            boolean releasePanel = intent.getBooleanExtra("release_panel", false);
            boolean timeInvalidAudio = precisionMatrix || releasePanel;
            if (timeInvalidAudio) StudyFiles.put(new File(destination, "worker-pid"), Integer.toString(pid));
            boolean play = !screen && !intent.getBooleanExtra("compute_only", false);
            boolean acoustic = intent.getBooleanExtra("acoustic", false);
            AudioManager audio = getSystemService(AudioManager.class);
            summary.put("threads", threads).put("stream", stream).put("play", play)
                .put("acoustic_enabled", acoustic).put("screen", screen).put("round", round).put("label", label)
                .put("label_engine", pico ? "pico" : "kokoro")
                .put("available_processors", Runtime.getRuntime().availableProcessors())
                .put("media_volume", audio.getStreamVolume(AudioManager.STREAM_MUSIC))
                .put("media_volume_max", audio.getStreamMaxVolume(AudioManager.STREAM_MUSIC))
                .put("pss_before_load_kib", Debug.getPss());
            File models = new File(getFilesDir(), "models");
            long serviceLaunch = intent.getLongExtra("launch_monotonic_ns", 0);
            long loadStart = System.nanoTime();
            String arm = label.substring(0, label.indexOf('/'));
            String runtime = pico ? "kitten-inference native C API" : "";
            if (releasePanel) {
                summary.put("release_arm", arm);
                switch (arm) {
                    case "mediapipe_opencl":
                        external = (AutoCloseable) Class.forName("org.kittenml.s25study.MediaPipeEngine")
                            .getConstructor(String.class, String.class)
                            .newInstance(new File(models, "mediapipe/kokoro-GPU.litertlm").getPath(),
                                getCacheDir().getPath());
                        runtime = "MediaPipe Tasks Audio 1.2.2";
                        summary.put("requested_delegate", "GPU").put("threads", "runtime default");
                        break;
                    case "adreno_fp16":
                        adreno = true;
                        handle = Adreno.create(new File(models, "adreno").getPath(),
                            getApplicationInfo().nativeLibraryDir);
                        runtime = "Adreno-LLMs FP16 with FP32 generator";
                        summary.put("threads", "runtime default").put("requested_delegate", "OpenCL");
                        break;
                    case "sherpa_official_int8":
                        handle = Sherpa.createVariant(new File(models, "sherpa").getPath(),
                            "model.int8.onnx", threads);
                        runtime = "Sherpa-ONNX " + Sherpa.version();
                        break;
                    case "pico_int8":
                        handle = Pico.createVariant(new File(models, "pico").getPath(),
                            "kitten_tts_2_pico_en", "int8", "v1", threads);
                        summary.put("native_build", Pico.buildInfo());
                        break;
                    default:
                        throw new IllegalArgumentException("Unknown release arm " + arm);
                }
            } else if (precisionMatrix) {
                String[] variant = arm.split("_", 2);
                String model = variant[0];
                String precision = variant[1];
                String directory = new File(models, model).getPath();
                summary.put("matrix_arm", arm).put("label_engine", model).put("precision", precision);
                if (pico) {
                    boolean mini = model.equals("mini");
                    String prefix = mini ? "kitten_tts_2_mini" : "kitten_tts_2_pico_en";
                    handle = Pico.createVariant(directory, prefix, precision, mini ? "v2" : "v1", threads);
                    summary.put("native_build", Pico.buildInfo());
                } else {
                    handle = Sherpa.createVariant(directory,
                        precision.equals("int8") ? "model.int8.onnx" : "model.onnx", threads);
                }
            } else {
                handle = pico ? Pico.create(new File(models, "pico").getPath(), threads)
                              : Sherpa.create(new File(models, "kokoro").getPath(), threads);
            }
            long loadEnd = System.nanoTime();
            if (handle == 0 && external == null) throw new IllegalStateException("Invalid engine handle");
            final long session = handle;
            summary.put("runtime", runtime.isEmpty() ? Sherpa.version() : runtime)
                .put("load_ms", (loadEnd - loadStart) / 1e6)
                .put("start_service_to_loaded_ms", (loadEnd - serviceLaunch) / 1e6)
                .put("pss_after_load_kib", Debug.getPss())
                .put("rss_after_load_kib", memory("VmRSS"));
            SpeechRequest.Engine engine;
            if (external != null) engine = (SpeechRequest.Engine) external;
            else if (adreno) engine = (text, sink) -> Adreno.generate(session, text, sink);
            else if (pico) engine = (text, sink) -> Pico.generate(session, text, 1.0f, sink);
            else engine = (text, sink) -> Sherpa.generate(session, text, 1.0f, sink);
            JSONObject warmup = new JSONObject().put("id", "warmup").put("bucket", "untimed")
                .put("text", WARMUP).put("time_invalid_audio", timeInvalidAudio);
            summary.put("first_request_compute", warm(engine, warmup, stream, false, timeInvalidAudio));
            // A second distinct text warms the same audio route before measured requests.
            if (play) {
                JSONObject audioWarmup = new JSONObject().put("id", "audio_warmup")
                    .put("bucket", "untimed").put("text", "Come in.")
                    .put("time_invalid_audio", timeInvalidAudio);
                if (acoustic) {
                    summary.put("muted_control", SpeechRequest.run(engine, audioWarmup, stream,
                        true, null, this, true));
                }
                summary.put("audio_warmup", acoustic
                    ? SpeechRequest.run(engine, audioWarmup, stream, true, null, this, false)
                    : warm(engine, audioWarmup, stream, true, timeInvalidAudio));
            }
            summary.put("pss_after_warmup_kib", Debug.getPss());
            String fixtureAsset = matchedChunks ? "matched-chunks.jsonl"
                : (screen ? "screen.jsonl" : "requests.jsonl");
            List<JSONObject> fixtures = StudyFiles.fixtures(this, fixtureAsset);
            Collections.shuffle(fixtures, new Random(584399L + round));
            File rowsFile = new File(destination, "requests.jsonl");
            long highestPss = 0;
            boolean allValid = true;
            try (PrintWriter rows = new PrintWriter(rowsFile, "UTF-8")) {
                for (JSONObject fixture : fixtures) {
                    JSONObject row;
                    try {
                        if (timeInvalidAudio) fixture.put("time_invalid_audio", true);
                        File wav = !screen && !label.startsWith("acoustic/") && round == 0
                            ? new File(destination, fixture.getString("id") + ".wav") : null;
                        row = SpeechRequest.run(engine, fixture, stream, play, wav,
                            acoustic ? this : null, false);
                        row.put("status", "OK");
                    } catch (Exception error) {
                        allValid = false;
                        row = new JSONObject().put("id", fixture.getString("id"))
                            .put("bucket", fixture.getString("bucket"))
                            .put("error", error.toString()).put("status", "ERROR");
                    }
                    long pss = Debug.getPss();
                    highestPss = Math.max(highestPss, pss);
                    row.put("post_request_pss_kib", pss)
                        .put("vm_hwm_kib", memory("VmHWM"))
                        .put("thermal_status", getSystemService(android.os.PowerManager.class)
                            .getCurrentThermalStatus());
                    rows.println(row);
                    rows.flush();
                }
                if (rows.checkError()) throw new IllegalStateException("Failed to write request rows");
            }
            summary.put("max_post_request_pss_kib", highestPss)
                .put("vm_hwm_kib", memory("VmHWM"))
                .put("conditions_end", StudyFiles.conditions(this))
                .put("status", allValid ? "OK" : "PARTIAL");
        } catch (Throwable error) {
            StringWriter stack = new StringWriter();
            error.printStackTrace(new PrintWriter(stack));
            try {
                summary.put("status", "ERROR").put("error", stack.toString());
            } catch (Exception ignored) {}
        } finally {
            try {
                if (external != null) external.close();
                if (handle != 0) {
                    if (adreno) Adreno.destroy(handle);
                    else if (pico) Pico.destroy(handle);
                    else Sherpa.destroy(handle);
                }
            } catch (Throwable error) {
                android.util.Log.e("StudyWorker", "Engine shutdown failed", error);
            }
            try {
                StudyFiles.put(completed, summary.toString(2));
            } catch (Exception error) {
                android.util.Log.e("StudyWorker", "Could not write completion status", error);
            }
            Process.killProcess(pid);
        }
    }
}
