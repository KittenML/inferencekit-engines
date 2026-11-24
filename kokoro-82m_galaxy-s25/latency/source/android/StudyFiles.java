package org.kittenml.s25study;

import android.content.Context;
import android.os.Build;
import android.os.PowerManager;

import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

final class StudyFiles {
    static List<JSONObject> fixtures(Context context, String name) throws Exception {
        ArrayList<JSONObject> rows = new ArrayList<>();
        try (BufferedReader in = new BufferedReader(new InputStreamReader(
                context.getAssets().open(name), StandardCharsets.UTF_8))) {
            for (String line; (line = in.readLine()) != null;) {
                if (!line.isEmpty()) rows.add(new JSONObject(line));
            }
        }
        return rows;
    }

    static void put(File output, String text) throws IOException {
        File temporary = new File(output.getPath() + ".tmp");
        Files.write(temporary.toPath(), text.getBytes(StandardCharsets.UTF_8));
        Files.move(temporary.toPath(), output.toPath(), StandardCopyOption.REPLACE_EXISTING,
                   StandardCopyOption.ATOMIC_MOVE);
    }

    static String read(File input) throws IOException {
        return new String(Files.readAllBytes(input.toPath()), StandardCharsets.UTF_8);
    }

    static String readAsset(Context context, String name) throws IOException {
        try (BufferedReader in = new BufferedReader(new InputStreamReader(
                context.getAssets().open(name), StandardCharsets.UTF_8))) {
            return in.readLine();
        }
    }

    static void mkdir(File directory) throws IOException {
        if (!directory.isDirectory() && !directory.mkdirs()) {
            throw new IOException("Could not create directory " + directory);
        }
    }

    static void unpack(Context context, String asset, File models) throws IOException {
        try (ZipInputStream in = new ZipInputStream(context.getAssets().open(asset))) {
            for (ZipEntry entry; (entry = in.getNextEntry()) != null;) {
                File output = new File(models, entry.getName());
                if (!output.getCanonicalPath().startsWith(models.getCanonicalPath() + "/")) {
                    throw new IOException("Rejected model path " + entry.getName());
                }
                if (entry.isDirectory()) mkdir(output);
                else {
                    mkdir(output.getParentFile());
                    try (FileOutputStream stream = new FileOutputStream(output)) {
                        byte[] bytes = new byte[128 * 1024];
                        for (int count; (count = in.read(bytes)) != -1;) stream.write(bytes, 0, count);
                    }
                }
            }
        }
    }

    private static void writeEntries(File directory, String base, ZipOutputStream stream)
            throws IOException {
        for (File file : directory.listFiles()) {
            String name = base + file.getName();
            if (file.isDirectory()) {
                writeEntries(file, name + "/", stream);
            } else {
                stream.putNextEntry(new ZipEntry(name));
                try (FileInputStream in = new FileInputStream(file)) {
                    byte[] bytes = new byte[64 * 1024];
                    for (int count; (count = in.read(bytes)) != -1;) stream.write(bytes, 0, count);
                }
                stream.closeEntry();
            }
        }
    }

    static void bundle(File results, File output) throws IOException {
        File temporary = new File(output.getPath() + ".tmp");
        try (ZipOutputStream stream = new ZipOutputStream(new FileOutputStream(temporary))) {
            writeEntries(results, "", stream);
        }
        Files.move(temporary.toPath(), output.toPath(), StandardCopyOption.REPLACE_EXISTING,
                   StandardCopyOption.ATOMIC_MOVE);
    }

    static JSONObject conditions(Context context) throws Exception {
        PowerManager power = context.getSystemService(PowerManager.class);
        android.content.Intent battery = context.registerReceiver(null,
            new android.content.IntentFilter(android.content.Intent.ACTION_BATTERY_CHANGED));
        return new JSONObject()
            .put("model", Build.MODEL)
            .put("device", Build.DEVICE)
            .put("board", Build.BOARD)
            .put("soc", Build.SOC_MODEL)
            .put("android", Build.VERSION.RELEASE)
            .put("fingerprint", Build.FINGERPRINT)
            .put("abis", String.join(",", Build.SUPPORTED_ABIS))
            .put("thermal_status", power.getCurrentThermalStatus())
            .put("power_save", power.isPowerSaveMode())
            .put("battery_temperature_tenths_c", battery == null ? JSONObject.NULL
                : battery.getIntExtra(android.os.BatteryManager.EXTRA_TEMPERATURE, -1))
            .put("battery_plugged", battery == null ? JSONObject.NULL
                : battery.getIntExtra(android.os.BatteryManager.EXTRA_PLUGGED, -1));
    }
}
