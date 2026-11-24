package org.kittenml.s25study;

import android.app.Activity;
import android.text.Editable;
import android.text.InputFilter;
import android.text.InputType;
import android.text.TextWatcher;
import android.util.Base64;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.zip.CRC32;

final class StudyUiExport {
    private static final int MAX_PAGE = 24 * 1024;
    private static final Pattern QUERY = Pattern.compile("([0-9]{1,18}):([0-9]{1,6})!");

    private final File archive;
    private final long bytes;
    private final String sha256;

    private StudyUiExport(File archive, long bytes, String sha256) {
        this.archive = archive;
        this.bytes = bytes;
        this.sha256 = sha256;
    }

    // Prepare after the result archive has been closed, on the benchmark controller thread.
    static StudyUiExport prepare(File archive) throws IOException {
        File immutableArchive = archive.getCanonicalFile();
        long bytes = immutableArchive.length();
        if (!immutableArchive.isFile() || bytes == 0) throw new IOException("No exported ZIP");
        MessageDigest digest;
        try {
            digest = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException error) {
            throw new AssertionError(error);
        }
        long hashed = 0;
        try (FileInputStream input = new FileInputStream(immutableArchive)) {
            byte[] buffer = new byte[64 * 1024];
            for (int count; (count = input.read(buffer)) != -1;) {
                digest.update(buffer, 0, count);
                hashed += count;
            }
        }
        if (hashed != bytes) throw new IOException("ZIP changed while hashing");
        StringBuilder hex = new StringBuilder(64);
        for (byte value : digest.digest()) {
            int sample = value & 0xff;
            hex.append(Character.forDigit(sample >>> 4, 16));
            hex.append(Character.forDigit(sample & 0xf, 16));
        }
        return new StudyUiExport(immutableArchive, bytes, hex.toString());
    }

    // Call on the main thread. Reuse the existing study status view so polling still works.
    void attach(Activity activity, TextView status) {
        if (status.getParent() instanceof ViewGroup) {
            ((ViewGroup) status.getParent()).removeView(status);
        }
        LinearLayout layout = new LinearLayout(activity);
        layout.setOrientation(LinearLayout.VERTICAL);
        status.setMaxLines(3);
        layout.addView(status, new LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        TextView metadata = new TextView(activity);
        metadata.setContentDescription("study-export-meta");
        metadata.setText("V1|" + bytes + "|" + MAX_PAGE + "|" + sha256);
        metadata.setImportantForAccessibility(View.IMPORTANT_FOR_ACCESSIBILITY_YES);
        layout.addView(metadata);

        EditText query = new EditText(activity);
        query.setContentDescription("study-export-query");
        query.setHint("Result page");
        query.setSingleLine(true);
        query.setShowSoftInputOnFocus(false);
        query.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS);
        query.setFilters(new InputFilter[]{new InputFilter.LengthFilter(32)});
        query.setImportantForAccessibility(View.IMPORTANT_FOR_ACCESSIBILITY_YES);
        layout.addView(query);

        TextView output = new TextView(activity);
        output.setContentDescription("study-export-page");
        output.setImportantForAccessibility(View.IMPORTANT_FOR_ACCESSIBILITY_YES);
        output.setHorizontallyScrolling(true);
        output.setMaxLines(1);
        output.setText("WAIT");
        layout.addView(output);
        query.addTextChangedListener(new TextWatcher() {
            @Override public void beforeTextChanged(CharSequence text, int start, int count, int after) {}
            @Override public void onTextChanged(CharSequence text, int start, int before, int count) {}

            @Override public void afterTextChanged(Editable text) {
                output.setText(page(text));
            }
        });
        activity.setContentView(layout);
    }

    private String page(CharSequence text) {
        Matcher match = QUERY.matcher(text);
        if (!match.matches()) return "WAIT";
        long offset = Long.parseLong(match.group(1));
        int requested = Integer.parseInt(match.group(2));
        if (offset >= bytes || requested < 1 || requested > MAX_PAGE) return "RANGE";
        int length = (int) Math.min(requested, bytes - offset);
        byte[] data = new byte[length];
        try (RandomAccessFile source = new RandomAccessFile(archive, "r")) {
            if (source.length() != bytes) return "ARCHIVE_CHANGED";
            source.seek(offset);
            source.readFully(data);
        } catch (IOException error) {
            return "READ_ERROR";
        }
        CRC32 checksum = new CRC32();
        checksum.update(data);
        String crc = String.format(Locale.ROOT, "%08x", checksum.getValue());
        return "V1|" + offset + "|" + length + "|" + crc + "|"
            + Base64.encodeToString(data, Base64.NO_WRAP);
    }
}
