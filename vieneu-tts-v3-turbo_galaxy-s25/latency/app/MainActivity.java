package org.inferencekit.s25study;

import android.app.Activity;
import android.content.Intent;
import android.content.IntentFilter;
import android.media.AudioDeviceInfo;
import android.media.AudioManager;
import android.os.BatteryManager;
import android.os.Build;
import android.os.Bundle;
import android.os.PowerManager;
import android.os.SystemClock;
import android.text.Editable;
import android.text.TextWatcher;
import android.util.Base64;
import android.view.WindowManager;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.RandomAccessFile;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

public final class MainActivity extends Activity {
    private static final int BUNDLE_PART_BYTES = 2 * 1024 * 1024;
    private static final int PAGE_BYTES = 45_000;
    private static final AtomicBoolean RUNNING = new AtomicBoolean();
    private TextView statusView;
    private TextView indexView;
    private TextView pageView;

    @Override
    public void onCreate(Bundle saved) {
        super.onCreate(saved);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        layout.setPadding(32, 60, 32, 32);
        statusView = new TextView(this);
        statusView.setTextSize(22);
        statusView.setContentDescription("study-status");
        statusView.setText("STARTING");
        layout.addView(statusView);
        indexView = new TextView(this);
        indexView.setTextSize(10);
        indexView.setMaxLines(3);
        indexView.setContentDescription("study-index");
        layout.addView(indexView);
        EditText pageRequest = new EditText(this);
        pageRequest.setSingleLine(true);
        pageRequest.setContentDescription("study-page");
        pageRequest.setHint("Artifact page");
        pageRequest.setShowSoftInputOnFocus(false);
        layout.addView(pageRequest);
        pageView = new TextView(this);
        pageView.setTextSize(8);
        pageView.setMaxLines(2);
        pageView.setContentDescription("study-page-data");
        layout.addView(pageView);
        pageRequest.addTextChangedListener(new TextWatcher() {
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {}
            public void onTextChanged(CharSequence s, int start, int before, int count) {}
            public void afterTextChanged(Editable value) { showPage(value.toString()); }
        });
        setContentView(layout);
        if (RUNNING.compareAndSet(false, true)) new Thread(this::runPlan, "study").start();
    }

    private void showPage(String request) {
        if (!request.matches("[abm][0-9]{1,6}")) return;
        String file = request.charAt(0) == 'a' ? "index.json"
                : request.charAt(0) == 'b' ? "bundle.zip" : "metrics.zip";
        File source = new File(getFilesDir(), "study/latest/" + file);
        try (RandomAccessFile data = new RandomAccessFile(source, "r")) {
            long first = PAGE_BYTES * Long.parseLong(request.substring(1));
            int length = (int) Math.min(PAGE_BYTES, data.length() - first);
            if (length <= 0) {
                pageView.setText(request + "|");
                return;
            }
            byte[] buffer = new byte[length];
            data.seek(first);
            data.readFully(buffer);
            pageView.setText(request + "|" + Base64.encodeToString(buffer, Base64.NO_WRAP));
        } catch (IOException failure) {
            pageView.setText(request + "|error");
        }
    }

    private void status(String message) {
        runOnUiThread(() -> statusView.setText(message));
    }

    private static void replaceFile(Path path, byte[] bytes) throws IOException {
        Path temp = path.resolveSibling(path.getFileName() + ".partial");
        Files.write(temp, bytes);
        Files.move(temp, path, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
    }

    private void saveIndex(Path path, JSONObject index) throws IOException {
        byte[] bytes = index.toString().getBytes(StandardCharsets.UTF_8);
        replaceFile(path, bytes);
        String compact = Base64.encodeToString(bytes, Base64.NO_WRAP);
        runOnUiThread(() -> indexView.setText(compact));
    }

    private static void removeTree(Path root) throws IOException {
        if (!Files.exists(root)) return;
        List<Path> files;
        try (Stream<Path> tree = Files.walk(root)) {
            files = tree.sorted(java.util.Comparator.reverseOrder()).collect(Collectors.toList());
        }
        for (Path file : files) Files.delete(file);
    }

    private void copyAsset(String asset, Path destination) throws IOException {
        String[] children = getAssets().list(asset);
        if (children != null && children.length != 0) {
            Files.createDirectories(destination);
            for (String child : children) copyAsset(asset + "/" + child, destination.resolve(child));
            return;
        }
        try (InputStream input = getAssets().open(asset)) {
            Files.copy(input, destination, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private static final class Cell {
        final String engine;
        final String round;
        final int cooldownSeconds;

        Cell(String line) {
            String[] fields = line.split("\t", -1);
            if (fields.length < 2 || fields.length > 3
                    || !fields[0].matches("[a-zA-Z0-9_-]{1,50}")
                    || !fields[1].matches("[a-zA-Z0-9_-]{1,24}")) {
                throw new IllegalArgumentException("Expected engine<TAB>round[<TAB>cooldown_seconds]");
            }
            engine = fields[0];
            round = fields[1];
            cooldownSeconds = fields.length == 3 ? Integer.parseInt(fields[2]) : 0;
            if (cooldownSeconds < 0 || cooldownSeconds > 180) {
                throw new IllegalArgumentException("Cooldown exceeds range 0–180 s");
            }
        }
    }

    private List<Cell> plan(String name) throws IOException {
        if (!name.matches("[a-zA-Z0-9_-]{1,40}")) throw new IllegalArgumentException("Invalid plan name");
        List<Cell> cells = new ArrayList<>();
        try (BufferedReader r = new BufferedReader(new InputStreamReader(
                getAssets().open(name + ".tsv"), StandardCharsets.UTF_8))) {
            String line;
            while ((line = r.readLine()) != null) {
                if (!line.trim().isEmpty() && !line.trim().startsWith("#")) cells.add(new Cell(line));
            }
        }
        if (cells.isEmpty()) throw new IllegalArgumentException("Empty plan");
        return cells;
    }

    private JSONObject conditions() throws JSONException {
        PowerManager power = getSystemService(PowerManager.class);
        AudioManager audio = getSystemService(AudioManager.class);
        Intent battery = registerReceiver(null, new IntentFilter(Intent.ACTION_BATTERY_CHANGED));
        JSONArray outputs = new JSONArray();
        JSONArray routes = new JSONArray();
        for (AudioDeviceInfo device : audio.getDevices(AudioManager.GET_DEVICES_OUTPUTS)) {
            outputs.put(device.getType());
            routes.put(new JSONObject().put("id", device.getId()).put("type", device.getType()));
        }
        JSONObject result = new JSONObject()
                .put("elapsed_realtime_ms", SystemClock.elapsedRealtime())
                .put("thermal_status", Build.VERSION.SDK_INT >= 29
                        ? power.getCurrentThermalStatus() : JSONObject.NULL)
                .put("power_save_mode", power.isPowerSaveMode())
                .put("music_volume", audio.getStreamVolume(AudioManager.STREAM_MUSIC))
                .put("music_volume_max", audio.getStreamMaxVolume(AudioManager.STREAM_MUSIC))
                .put("music_muted", audio.isStreamMute(AudioManager.STREAM_MUSIC))
                .put("audio_output_types", outputs)
                .put("audio_output_routes", routes);
        if (battery != null) {
            result.put("battery_temperature_tenths_c", battery.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1))
                    .put("battery_level", battery.getIntExtra(BatteryManager.EXTRA_LEVEL, -1))
                    .put("battery_plugged", battery.getIntExtra(BatteryManager.EXTRA_PLUGGED, -1));
        }
        return result;
    }

    private JSONObject runCell(Cell cell, int number, Path payload, Path cells, int timeoutSeconds)
            throws IOException, JSONException, InterruptedException {
        String label = String.format(Locale.ROOT, "%03d-%s-%s", number, cell.engine, cell.round);
        Path output = Files.createDirectories(cells.resolve(label));
        JSONObject result = new JSONObject().put("engine", cell.engine).put("round", cell.round)
                .put("directory", "cells/" + label).put("cooldown_seconds", cell.cooldownSeconds);
        if (cell.cooldownSeconds > 0) {
            status("COOLDOWN " + label + " (" + cell.cooldownSeconds + " s)");
            Thread.sleep(cell.cooldownSeconds * 1000L);
        }
        status("RUNNING " + label);
        result.put("before", conditions());
        String libraries = getApplicationInfo().nativeLibraryDir;
        try {
            ProcessBuilder runner = new ProcessBuilder(new File(libraries, "libstudy_runner.so").getPath(),
                    payload.toString(), output.toString(), cell.engine, cell.round).directory(output.toFile());
            runner.environment().put("LD_LIBRARY_PATH", libraries);
            runner.redirectErrorStream(true).redirectOutput(output.resolve("runner.log").toFile());
            Process process = runner.start();
            try {
                if (!process.waitFor(timeoutSeconds, TimeUnit.SECONDS)) {
                    result.put("timed_out", true);
                    process.destroyForcibly().waitFor(10, TimeUnit.SECONDS);
                }
                result.put("exit_code", process.isAlive() ? JSONObject.NULL : process.exitValue());
            } finally {
                if (process.isAlive()) process.destroyForcibly();
            }
        } catch (IOException failure) {
            result.put("launch_error", failure.toString());
        }
        return result.put("after", conditions());
    }

    private static void zipCells(Path output, Path target, boolean audio) throws IOException {
        Path pending = target.resolveSibling(target.getFileName() + ".partial");
        Path cells = output.resolve("cells");
        try (ZipOutputStream archive = new ZipOutputStream(new BufferedOutputStream(Files.newOutputStream(pending)));
                Stream<Path> files = Files.walk(cells)) {
            archive.setLevel(1);
            for (Path file : files.filter(Files::isRegularFile).collect(Collectors.toList())) {
                if (!audio && file.toString().toLowerCase(Locale.ROOT).matches(".*\\.(wav|pcm|flac|ogg)")) continue;
                archive.putNextEntry(new ZipEntry(output.relativize(file).toString()));
                Files.copy(file, archive);
                archive.closeEntry();
            }
        }
        Files.move(pending, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
    }

    private static void makeBundle(Path output, JSONObject index) throws IOException, JSONException {
        Path target = output.resolve("bundle.zip");
        Path metrics = output.resolve("metrics.zip");
        zipCells(output, target, true);
        zipCells(output, metrics, false);
        index.put("metrics_bytes", Files.size(metrics));
        index.put("page_bytes", PAGE_BYTES);
        long bytes = Files.size(target);
        index.put("bundle_bytes", bytes);
        if (bytes > BUNDLE_PART_BYTES) {
            int part = 0;
            try (InputStream source = Files.newInputStream(target)) {
                byte[] buffer = new byte[BUNDLE_PART_BYTES];
                while (true) {
                    int read = 0;
                    while (read < buffer.length) {
                        int received = source.read(buffer, read, buffer.length - read);
                        if (received == -1) break;
                        read += received;
                    }
                    if (read == 0) break;
                    Path file = output.resolve(String.format(Locale.ROOT, "bundle-%03d.part", part++));
                    try (FileOutputStream stream = new FileOutputStream(file.toFile())) {
                        stream.write(buffer, 0, read);
                    }
                }
            }
            index.put("bundle_parts", part);
        }
    }

    private void mirrorArtifacts(Path index, Path output) throws IOException {
        File external = getExternalFilesDir(null);
        if (external == null) return;
        Path base = external.toPath().resolve("study");
        removeTree(base);
        Files.createDirectories(base);
        try (Stream<Path> files = Files.list(output)) {
            for (Path file : files.filter(Files::isRegularFile).collect(Collectors.toList())) {
                if (!file.equals(index)) Files.copy(file, base.resolve(file.getFileName()));
            }
        }
        Files.copy(index, base.resolve(index.getFileName()));
    }

    private void runPlan() {
        Path base = getFilesDir().toPath().resolve("study");
        Path output = base.resolve("latest");
        Path indexPath = output.resolve("index.json");
        JSONObject index = new JSONObject();
        try {
            removeTree(output);
            Path cells = Files.createDirectories(output.resolve("cells"));
            String name = getIntent().getStringExtra("plan_name");
            if (name == null) name = "plan";
            int timeoutSeconds = getIntent().getIntExtra("cell_timeout_s", 180);
            if (timeoutSeconds < 1 || timeoutSeconds > 600) throw new IllegalArgumentException("Invalid cell timeout");
            index.put("plan", name).put("cell_timeout_seconds", timeoutSeconds)
                    .put("model", Build.MODEL).put("device", Build.DEVICE).put("manufacturer", Build.MANUFACTURER)
                    .put("soc_model", Build.VERSION.SDK_INT >= 31 ? Build.SOC_MODEL : JSONObject.NULL)
                    .put("android_release", Build.VERSION.RELEASE).put("android_build", Build.ID)
                    .put("build_fingerprint", Build.FINGERPRINT).put("sdk", Build.VERSION.SDK_INT)
                    .put("abis", new JSONArray(Build.SUPPORTED_ABIS)).put("start", conditions());
            List<Cell> plan = plan(name);
            JSONArray results = new JSONArray();
            index.put("planned_cells", plan.size()).put("cells", results);
            saveIndex(indexPath, index);
            status("STAGING");
            Path payload = base.resolve("payload");
            removeTree(payload);
            copyAsset("payload", payload);
            int failed = 0;
            int number = 0;
            for (Cell cell : plan) {
                JSONObject result = runCell(cell, ++number, payload, cells, timeoutSeconds);
                results.put(result);
                if (result.optInt("exit_code", -1) != 0 || result.optBoolean("timed_out")) failed++;
                saveIndex(indexPath, index);
            }
            status("PACKAGING");
            makeBundle(output, index);
            index.put("failed_cells", failed).put("end", conditions()).put("complete", true);
            saveIndex(indexPath, index);
            try {
                mirrorArtifacts(indexPath, output);
            } catch (IOException failure) {
                android.util.Log.w("S25Study", "External mirror unavailable", failure);
            }
            status("DONE " + number + " cells, " + failed + " failed");
        } catch (Exception failure) {
            try {
                index.put("error", failure.toString());
                saveIndex(indexPath, index);
            } catch (Exception suppressed) {
                failure.addSuppressed(suppressed);
            }
            android.util.Log.e("S25Study", "Study failed", failure);
            status("ERROR " + failure.getMessage());
        }
    }
}
