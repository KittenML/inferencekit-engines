package org.kittenml.s25study;

import android.content.Context;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.security.MessageDigest;
import java.util.ArrayList;

final class SpeechRequest implements Sink {
    interface Engine {
        void generate(String text, Sink sink);
    }

    private static final int WINDOW = Playback.RATE / 200;
    private static final double SIGNAL_SQUARED = 327.68 * 327.68;
    private final Playback playback;
    private final boolean stream;
    private final boolean timeInvalidAudio;
    private final ArrayList<short[]> chunks = new ArrayList<>();
    private final ArrayList<float[]> nativeChunks = new ArrayList<>();
    private final ArrayList<Long> chunkReadyNs;
    private long firstCallback;
    private long firstSignal;
    private int onset = -1;
    private int frames;
    private int windowFrames;
    private long windowEnergy;
    private double maxAbs;
    private int clips;
    private int nonfinite;

    private SpeechRequest(Playback playback, boolean stream, boolean traceChunks,
                          boolean timeInvalidAudio) {
        this.playback = playback;
        this.stream = stream;
        this.chunkReadyNs = traceChunks ? new ArrayList<>() : null;
        this.timeInvalidAudio = timeInvalidAudio;
    }

    @Override public void accept(float[] input) {
        if (input.length == 0) return;
        if (firstCallback == 0) firstCallback = System.nanoTime();
        short[] pcm = new short[input.length];
        for (int i = 0; i < pcm.length; i++) {
            float value = input[i];
            if (!Float.isFinite(value)) {
                if (!timeInvalidAudio) throw new IllegalStateException("Nonfinite audio");
                nonfinite++;
                value = 0;
            }
            maxAbs = Math.max(maxAbs, Math.abs(value));
            if (value >= 1 || value <= -1) clips++;
            int sample = Math.max(-32768, Math.min(32767, Math.round(value * 32768)));
            pcm[i] = (short) sample;
            if (onset < 0) {
                windowEnergy += (long) sample * sample;
                windowFrames++;
                if (windowFrames == WINDOW) {
                    if (windowEnergy >= SIGNAL_SQUARED * WINDOW) onset = frames + i + 1 - WINDOW;
                    windowFrames = 0;
                    windowEnergy = 0;
                }
            }
        }
        frames += pcm.length;
        if (onset >= 0 && firstSignal == 0) {
            firstSignal = System.nanoTime();
            if (playback != null) playback.onset(onset);
        }
        nativeChunks.add(input);
        chunks.add(pcm);
        if (chunkReadyNs != null) chunkReadyNs.add(System.nanoTime());
        if (stream && playback != null) playback.put(pcm);
    }

    static JSONObject run(Engine engine, JSONObject fixture, boolean stream, boolean play,
                          File wav) throws Exception {
        return run(engine, fixture, stream, play, wav, null, false);
    }

    static JSONObject run(Engine engine, JSONObject fixture, boolean stream, boolean play,
                          File wav, Context acousticContext, boolean muted) throws Exception {
        String text = fixture.getString("text");
        JSONArray segments = fixture.optJSONArray("segments");
        boolean timeInvalidAudio = fixture.optBoolean("time_invalid_audio");
        try (Playback playback = play ? new Playback(muted, segments != null) : null;
             AcousticCapture mic = acousticContext != null && play
                 ? AcousticCapture.start(acousticContext) : null) {
            SpeechRequest request = new SpeechRequest(playback, stream, segments != null, timeInvalidAudio);
            long start = System.nanoTime();
            if (segments == null) {
                engine.generate(text, request);
            } else {
                for (int i = 0; i < segments.length(); i++) {
                    engine.generate(segments.getString(i), request);
                    if (request.chunks.size() != i + 1) {
                        throw new IllegalStateException("Expected one audio chunk for text segment " + i);
                    }
                }
            }
            long synthesisComplete = System.nanoTime();
            if (request.frames == 0 || (request.onset < 0 && !timeInvalidAudio)) {
                throw new IllegalStateException("Signal gate found no speech");
            }
            String playbackError = null;
            if (playback != null) {
                if (!stream) for (short[] chunk : request.chunks) playback.put(chunk);
                try {
                    playback.finish();
                } catch (Exception error) {
                    Throwable cause = error.getCause();
                    playbackError = cause == null ? error.toString() : cause.toString();
                }
            }
            double duration = (double) request.frames / Playback.RATE;
            double synthMs = (synthesisComplete - start) / 1e6;
            JSONObject row = playback == null ? new JSONObject() : playback.metrics(start);
            if (playback != null) {
                if (playbackError == null && row.getInt("audio_written_frames") != request.frames) {
                    playbackError = "Incomplete AudioTrack PCM";
                }
                row.put("audio_playback_error", playbackError == null ? JSONObject.NULL : playbackError);
            }
            if (mic != null && playbackError == null) {
                row.put("acoustic", mic.finish(start, playback.playCallNs(), request.chunks,
                    Playback.RATE, request.onset, row.getInt("audio_route_type")));
            }
            // Hash and export PCM after the timed section.
            String digest = request.save(wav);
            String nativeDigest = request.hashFloatPcm();
            if (segments != null) {
                JSONArray events = new JSONArray();
                for (int i = 0; i < segments.length(); i++) {
                    events.put(new JSONObject()
                        .put("text", segments.getString(i))
                        .put("frames", request.chunks.get(i).length)
                        .put("ready_ms", (request.chunkReadyNs.get(i) - start) / 1e6));
                }
                row.put("chunk_events", events);
            }
            return row.put("id", fixture.getString("id"))
                .put("bucket", fixture.getString("bucket"))
                .put("chars", text.length())
                .put("start_monotonic_ns", start)
                .put("first_callback_ms", (request.firstCallback - start) / 1e6)
                .put("first_signal_ready_ms", request.onset >= 0
                    ? (request.firstSignal - start) / 1e6 : JSONObject.NULL)
                .put("synthesis_complete_ms", synthMs)
                .put("rtf", synthMs / (1000 * duration))
                .put("duration_s", duration)
                .put("frames", request.frames)
                .put("onset_frame", request.onset)
                .put("sample_rate", Playback.RATE)
                .put("callbacks", request.chunks.size())
                .put("peak_abs_before_clip", request.maxAbs)
                .put("clipped_samples", request.clips)
                .put("nonfinite_samples", request.nonfinite)
                .put("pcm16_sha256", digest)
                .put("float32_pcm_sha256", nativeDigest);
        }
    }

    private String save(File path) throws Exception {
        MessageDigest hash = MessageDigest.getInstance("SHA-256");
        try (DataOutputStream out = path == null ? null
                : new DataOutputStream(new BufferedOutputStream(new FileOutputStream(path)))) {
            if (out != null) {
                int size = frames * 2;
                out.writeBytes("RIFF");
                out.writeInt(Integer.reverseBytes(36 + size));
                out.writeBytes("WAVEfmt ");
                out.writeInt(Integer.reverseBytes(16));
                out.writeShort(Short.reverseBytes((short) 1));
                out.writeShort(Short.reverseBytes((short) 1));
                out.writeInt(Integer.reverseBytes(Playback.RATE));
                out.writeInt(Integer.reverseBytes(Playback.RATE * 2));
                out.writeShort(Short.reverseBytes((short) 2));
                out.writeShort(Short.reverseBytes((short) 16));
                out.writeBytes("data");
                out.writeInt(Integer.reverseBytes(size));
            }
            for (short[] chunk : chunks) {
                byte[] bytes = new byte[chunk.length * 2];
                for (int i = 0; i < chunk.length; i++) {
                    bytes[i * 2] = (byte) chunk[i];
                    bytes[i * 2 + 1] = (byte) (chunk[i] >>> 8);
                }
                hash.update(bytes);
                if (out != null) out.write(bytes);
            }
        }
        return hex(hash.digest());
    }

    private String hashFloatPcm() throws Exception {
        MessageDigest hash = MessageDigest.getInstance("SHA-256");
        java.nio.ByteBuffer buffer = java.nio.ByteBuffer.allocate(8192)
            .order(java.nio.ByteOrder.LITTLE_ENDIAN);
        for (float[] chunk : nativeChunks) {
            for (float value : chunk) {
                if (!buffer.hasRemaining()) {
                    hash.update(buffer.array(), 0, buffer.position());
                    buffer.clear();
                }
                buffer.putInt(Float.floatToRawIntBits(value));
            }
        }
        hash.update(buffer.array(), 0, buffer.position());
        return hex(hash.digest());
    }

    private static String hex(byte[] digest) {
        StringBuilder result = new StringBuilder(64);
        for (byte value : digest) result.append(String.format(java.util.Locale.ROOT, "%02x", value & 255));
        return result.toString();
    }
}
