package org.kittenml.s25study;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.os.SystemClock;
import android.view.WindowManager;
import android.widget.TextView;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

public final class StudyActivity extends Activity {
    private static final AtomicBoolean STARTED = new AtomicBoolean();
    private static final String[] ARMS = {"kokoro_buffered", "kokoro_stream", "pico_stream"};
    private static final String[] MATRIX_ARMS = {
        "kokoro_fp32", "kokoro_int8", "mini_fp32", "mini_int8", "mini_int4",
        "pico_fp32", "pico_int8", "pico_int4"
    };
    private static final int[] MATRIX_ORDER = {0, 1, 7, 2, 6, 3, 5, 4};
    private static final String[] RELEASE_ARMS = {
        "sherpa_official_int8", "mediapipe_opencl", "adreno_fp16", "pico_int8"
    };
    private static final int[] RELEASE_ORDER = {0, 1, 3, 2};
    private static final int[][] ORDER = {
        {0, 1, 2}, {2, 0, 1}, {1, 2, 0}, {2, 1, 0}, {1, 0, 2}, {0, 2, 1}
    };
    private TextView statusView;
    private File files;
    private File results;
    private boolean acoustic;
    private boolean uiExport;
    private boolean seeded;
    private boolean matchedChunks;
    private boolean precisionMatrix;
    private boolean releasePanel;

    @Override public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        statusView = new TextView(this);
        statusView.setContentDescription("study-status");
        statusView.setTextSize(20);
        statusView.setPadding(32, 64, 32, 32);
        statusView.setText("STARTING");
        setContentView(statusView);
        if (STARTED.compareAndSet(false, true)) new Thread(this::study, "study-controller").start();
    }

    private void status(String value) throws Exception {
        StudyFiles.put(new File(files, "status"), value);
        runOnUiThread(() -> statusView.setText(value));
    }

    private JSONObject process(String label, boolean pico, boolean stream, boolean screen,
                               boolean computeOnly, int threads, int round) throws Exception {
        status("RUNNING " + label);
        File destination = new File(results, label);
        File completed = new File(destination, "done.json");
        if (completed.exists()) throw new IllegalStateException("Refusing to merge a stale round " + label);
        Intent intent = new Intent(this, StudyWorker.class)
            .putExtra("label", label).putExtra("pico", pico).putExtra("stream", stream)
            .putExtra("screen", screen).putExtra("compute_only", computeOnly)
            .putExtra("acoustic", acoustic)
            .putExtra("matched_chunks", matchedChunks)
            .putExtra("precision_matrix", precisionMatrix)
            .putExtra("release_panel", releasePanel)
            .putExtra("threads", threads).putExtra("round", round)
            .putExtra("launch_monotonic_ns", System.nanoTime());
        startService(intent);
        long deadline = SystemClock.elapsedRealtime() + 170_000;
        while (!completed.isFile() && SystemClock.elapsedRealtime() < deadline) {
            File workerPid = new File(destination, "worker-pid");
            if ((precisionMatrix || releasePanel) && workerPid.isFile()) {
                String pid = StudyFiles.read(workerPid).trim();
                if (!new File("/proc/" + pid).exists() && !completed.isFile()) {
                    JSONObject failed = new JSONObject().put("pid", Integer.parseInt(pid))
                        .put("status", "NOT RUN").put("error", "Native worker exited before writing done.json");
                    StudyFiles.put(completed, failed.toString(2));
                    return failed;
                }
            }
            SystemClock.sleep(175);
        }
        if (!completed.isFile()) throw new IllegalStateException("Worker timed out " + label);
        JSONObject result = new JSONObject(StudyFiles.read(completed));
        int pid = result.getInt("pid");
        // Wait until the previous measured process has exited before starting another.
        long exitDeadline = SystemClock.elapsedRealtime() + 5000;
        File proc = new File("/proc/" + pid);
        while (proc.exists() && SystemClock.elapsedRealtime() < exitDeadline) {
            SystemClock.sleep(70);
        }
        if (proc.exists()) throw new IllegalStateException("Worker did not exit " + label);
        return result;
    }

    private Map<String, JSONObject> requests(String label) throws Exception {
        HashMap<String, JSONObject> indexed = new HashMap<>();
        try (BufferedReader in = new BufferedReader(new FileReader(new File(results, label + "/requests.jsonl")))) {
            for (String line; (line = in.readLine()) != null;) {
                JSONObject row = new JSONObject(line);
                indexed.put(row.getString("id"), row);
            }
        }
        return indexed;
    }

    private int screen(boolean pico, boolean[] engines) throws Exception {
        String kind = pico ? "pico" : "kokoro";
        int[] threads = pico ? new int[]{2, 1, 4} : new int[]{2, 1, 4, 6, 8};
        Map<Integer, String> labels = new HashMap<>();
        Map<Integer, String> verdict = new HashMap<>();
        for (int count : threads) {
            String label = "screen/" + kind + "-" + count;
            JSONObject run = process(label, pico, true, true, true, count, 0);
            String status = run.getString("status");
            verdict.put(count, status);
            if (status.equals("OK")) labels.put(count, label);
            if (seeded && !pico && count == 2 && status.equals("OK")) {
                String repeated = "screen/kokoro-2-repeat";
                JSONObject second = process(repeated, false, true, true, true, count, 0);
                boolean exact = second.getString("status").equals("OK")
                    && equalPcm(requests(label), requests(repeated));
                StudyFiles.put(new File(results, "kokoro-seeded-aa.json"), new JSONObject()
                    .put("exact", exact).put("first_pid", run.getInt("pid"))
                    .put("second_pid", second.getInt("pid")).toString(2));
                if (!exact) throw new IllegalStateException("Seeded Kokoro failed fresh-process A/A PCM parity");
            }
        }
        int referenceThreads = 2;
        if (!labels.containsKey(referenceThreads)) {
            for (int count : threads) {
                if (labels.containsKey(count)) {
                    referenceThreads = count;
                    break;
                }
            }
        }
        if (!labels.containsKey(referenceThreads)) {
            engines[pico ? 1 : 0] = false;
            return 2;
        }
        Map<String, JSONObject> reference = requests(labels.get(referenceThreads));
        double best = Double.POSITIVE_INFINITY;
        int winner = referenceThreads;
        JSONArray scores = new JSONArray();
        for (int count : threads) {
            JSONObject row = new JSONObject().put("threads", count).put("status", verdict.get(count));
            if (labels.containsKey(count)) {
                Map<String, JSONObject> candidate = requests(labels.get(count));
                double[] ratios = new double[reference.size()];
                int index = 0;
                boolean exact = equalPcm(reference, candidate);
                for (Map.Entry<String, JSONObject> item : reference.entrySet()) {
                    JSONObject baseline = item.getValue();
                    JSONObject arm = candidate.get(item.getKey());
                    if (arm == null) continue;
                    ratios[index++] = arm.getDouble("first_signal_ready_ms") /
                        baseline.getDouble("first_signal_ready_ms");
                }
                Arrays.sort(ratios);
                double median = ratios[ratios.length / 2];
                if (exact && median < best) {
                    best = median;
                    winner = count;
                }
                row.put("eligible_exact_pcm", exact).put("median_first_signal_ready_ratio", median);
            }
            scores.put(row);
        }
        StudyFiles.put(new File(results, kind + "-selection.json"), new JSONObject()
            .put("screen_reference_threads", referenceThreads)
            .put("selected_threads", winner).put("observations", scores).toString(2));
        return winner;
    }

    private boolean equalPcm(Map<String, JSONObject> first, Map<String, JSONObject> second)
            throws Exception {
        if (first.isEmpty() || !first.keySet().equals(second.keySet())) return false;
        for (Map.Entry<String, JSONObject> item : first.entrySet()) {
            JSONObject a = item.getValue();
            JSONObject b = second.get(item.getKey());
            if (!a.getString("float32_pcm_sha256").equals(b.getString("float32_pcm_sha256"))
                    || !a.getString("pcm16_sha256").equals(b.getString("pcm16_sha256"))) return false;
        }
        return true;
    }

    private int[] orderForRound(int round) {
        if (releasePanel) {
            return Arrays.stream(RELEASE_ORDER)
                .map(arm -> (arm + round) % RELEASE_ARMS.length).toArray();
        }
        if (precisionMatrix) {
            return Arrays.stream(MATRIX_ORDER)
                .map(arm -> (arm + round) % MATRIX_ARMS.length).toArray();
        }
        if (matchedChunks) return round % 2 == 0 ? new int[]{1, 2} : new int[]{2, 1};
        return ORDER[round];
    }

    private void study() {
        files = getFilesDir();
        results = new File(files, "results");
        File bundle = new File(files, "study.zip");
        try {
            StudyFiles.mkdir(results);
            status("UNPACKING");
            JSONObject start = StudyFiles.conditions(this);
            StudyFiles.put(new File(results, "device-start.json"), start.toString(2));
            String model = start.getString("model");
            if (!model.startsWith("SM-S931") || !start.getString("abis").startsWith("arm64-v8a")) {
                throw new IllegalStateException("Expected a physical base Galaxy S25, received " + model);
            }
            File models = new File(files, "models");
            StudyFiles.mkdir(models);
            StudyFiles.unpack(this, "models.zip", models);
            String mode = StudyFiles.readAsset(this, "mode.txt").trim();
            precisionMatrix = mode.equals("precision-matrix");
            releasePanel = mode.equals("release-panel");
            matchedChunks = mode.equals("matched-chunks") || precisionMatrix || releasePanel;
            seeded = mode.equals("seeded") || (matchedChunks && !releasePanel);
            acoustic = mode.equals("acoustic") || mode.equals("acoustic-ui");
            uiExport = mode.equals("seeded") || mode.equals("ui") || mode.equals("acoustic-ui");
            StudyFiles.put(new File(results, "capture-mode.json"), new JSONObject()
                .put("mode", mode).put("kokoro_seeded", seeded).toString(2));
            boolean smoke = mode.equals("smoke");
            boolean computeOnly = mode.equals("compute");
            boolean[] supported = {true, true};
            int kokoroThreads = smoke || matchedChunks ? 2 : screen(false, supported);
            int picoThreads = smoke || matchedChunks ? 2 : screen(true, supported);
            int rounds = smoke ? 1 : (releasePanel ? RELEASE_ARMS.length
                : (precisionMatrix ? MATRIX_ARMS.length : ORDER.length));
            JSONArray schedule = new JSONArray();
            int acousticAvailable = 0;
            for (int round = 0; round < rounds; round++) {
                for (int index : orderForRound(round)) {
                    boolean nativeEngine = releasePanel ? index == 3
                        : (precisionMatrix ? index >= 2 : index == 2);
                    if (!supported[nativeEngine ? 1 : 0]) continue;
                    int threads = nativeEngine ? picoThreads : kokoroThreads;
                    String arm = releasePanel ? RELEASE_ARMS[index]
                        : (precisionMatrix ? MATRIX_ARMS[index] : ARMS[index]);
                    String label = arm + "/round-" + round;
                    if (smoke && index == 1) continue;
                    JSONObject result = process(label, nativeEngine, releasePanel || precisionMatrix || index != 0, smoke,
                                                computeOnly, threads, round);
                    if (acoustic) {
                        JSONObject warm = result.optJSONObject("audio_warmup");
                        JSONObject capture = warm == null ? null : warm.optJSONObject("acoustic");
                        if (capture != null && "matched-speaker-at-phone-mic".equals(
                                capture.optString("status"))) acousticAvailable++;
                    }
                    schedule.put(new JSONObject().put("label", label)
                        .put("pid", result.getInt("pid")).put("status", result.getString("status"))
                        .put("error", result.opt("error")));
                    StudyFiles.put(new File(results, "schedule.json"), schedule.toString(2));
                    SystemClock.sleep(350);
                }
                if (acoustic && acousticAvailable == 0) break;
            }
            StudyFiles.put(new File(results, "device-end.json"), StudyFiles.conditions(this).toString(2));
            if (seeded && !matchedChunks && supported[0] && supported[1]) {
                acoustic = true;
                JSONArray acousticSchedule = new JSONArray();
                for (int index : ORDER[0]) {
                    boolean pico = index == 2;
                    String label = "acoustic/" + ARMS[index] + "/round-0";
                    JSONObject result = process(label, pico, index != 0, false, false,
                        pico ? picoThreads : kokoroThreads, 0);
                    acousticSchedule.put(new JSONObject().put("label", label)
                        .put("pid", result.getInt("pid")).put("status", result.getString("status")));
                    StudyFiles.put(new File(results, "acoustic/schedule.json"),
                        acousticSchedule.toString(2));
                }
            }
            export(bundle);
            status("DONE");
        } catch (Throwable error) {
            try {
                StringWriter stack = new StringWriter();
                error.printStackTrace(new PrintWriter(stack));
                StudyFiles.put(new File(results, "error.txt"), stack.toString());
                export(bundle);
                status("ERROR " + error);
            } catch (Exception nested) {
                android.util.Log.e("StudyActivity", "Could not export the failure", nested);
            }
        }
    }
    private void export(File output) throws Exception {
        StudyFiles.bundle(results, output);
        if (uiExport) {
            StudyUiExport publisher = StudyUiExport.prepare(output);
            runOnUiThread(() -> publisher.attach(this, statusView));
        }
    }

}
