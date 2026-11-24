package org.kittenml.s25study;

import android.media.AudioAttributes;
import android.media.AudioDeviceInfo;
import android.media.AudioFormat;
import android.media.AudioTimestamp;
import android.media.AudioTrack;
import android.os.Process;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.concurrent.LinkedBlockingQueue;

final class Playback implements AutoCloseable {
    static final int RATE = 24000;
    private static final long NS = 1_000_000_000L;
    private static final short[] END = new short[0];

    private final AudioTrack track;
    private final LinkedBlockingQueue<short[]> queue = new LinkedBlockingQueue<>();
    private final Thread thread;
    private volatile int onsetFrame = -1;
    private Throwable failure;
    private long playedAt;
    private long drainedAt;
    private long onsetAt;
    private long anchorNs;
    private long anchorFrame;
    private long lastTimestampNs;
    private long lastTimestampFrame;
    private long nextTimestampPollNs;
    private int written;
    private int payloadFrames = -1;
    private int headAtDrain;
    private int underrunsAtStart;
    private int underrunsBeforeOnset;
    private int underruns;
    private int route = -1;
    private String routeName;
    private final int bufferFrames;
    private final int thresholdFrames;
    private final int performanceMode;
    private final ArrayList<Long> emptyQueueWaitNs;

    Playback(boolean muted, boolean traceBoundaries) {
        emptyQueueWaitNs = traceBoundaries ? new ArrayList<>() : null;
        int minimum = AudioTrack.getMinBufferSize(RATE, AudioFormat.CHANNEL_OUT_MONO,
                                                  AudioFormat.ENCODING_PCM_16BIT);
        if (minimum <= 0) throw new IllegalStateException("No 24 kHz mono playback configuration");
        track = new AudioTrack.Builder()
            .setAudioAttributes(new AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_MEDIA)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
            .setAudioFormat(new AudioFormat.Builder().setSampleRate(RATE)
                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                .setEncoding(AudioFormat.ENCODING_PCM_16BIT).build())
            .setTransferMode(AudioTrack.MODE_STREAM)
            .setPerformanceMode(AudioTrack.PERFORMANCE_MODE_LOW_LATENCY)
            .setBufferSizeInBytes(Math.max(minimum, RATE / 10 * 2))
            .build();
        if (track.getState() != AudioTrack.STATE_INITIALIZED) {
            track.release();
            throw new IllegalStateException("AudioTrack initialization failed");
        }
        if (muted) track.setVolume(0f);
        bufferFrames = track.getBufferSizeInFrames();
        thresholdFrames = track.setStartThresholdInFrames(Math.min(RATE / 50, bufferFrames));
        performanceMode = track.getPerformanceMode();
        thread = new Thread(this::pump, "study-audio");
        thread.start();
    }

    void put(short[] data) {
        if (data.length > 0) queue.add(data);
    }

    void onset(int frame) {
        onsetFrame = frame;
    }

    void finish() throws Exception {
        queue.add(END);
        thread.join(60_000);
        if (thread.isAlive()) throw new IllegalStateException("AudioTrack playback timed out");
        if (failure != null) throw new IllegalStateException("AudioTrack playback failed", failure);
    }

    private int write(short[] data, int offset) {
        int count = track.write(data, offset, data.length - offset, AudioTrack.WRITE_NON_BLOCKING);
        if (count < 0) throw new IllegalStateException("AudioTrack.write returned " + count);
        written += count;
        return count;
    }

    private void sampleClock(AudioTimestamp stamp) {
        if (route < 0) {
            AudioDeviceInfo device = track.getRoutedDevice();
            if (device != null) {
                route = device.getType();
                routeName = device.getProductName().toString();
            }
        }
        long now = System.nanoTime();
        if (anchorNs != 0 || now < nextTimestampPollNs) return;
        nextTimestampPollNs = now + 12_000_000L;
        if (!track.getTimestamp(stamp) || stamp.nanoTime < playedAt ||
                stamp.framePosition <= 0 || stamp.framePosition > written ||
                stamp.nanoTime <= lastTimestampNs || stamp.framePosition <= lastTimestampFrame) {
            return;
        }
        lastTimestampNs = stamp.nanoTime;
        lastTimestampFrame = stamp.framePosition;
        if (anchorNs == 0 && onsetFrame >= 0 && stamp.framePosition > onsetFrame) {
            long presentation = stamp.nanoTime - (stamp.framePosition - onsetFrame) * NS / RATE;
            if (presentation >= playedAt && presentation <= System.nanoTime()) {
                anchorNs = stamp.nanoTime;
                anchorFrame = stamp.framePosition;
                underrunsBeforeOnset = track.getUnderrunCount() - underrunsAtStart;
                if (underrunsBeforeOnset == 0) onsetAt = presentation;
            }
        }
    }

    private void pump() {
        try {
            Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO);
            short[] data = queue.take();
            if (data == END) return;
            // Prime the same AudioTrack buffer for each arm before starting playback.
            int offset = write(data, 0);
            underrunsAtStart = track.getUnderrunCount();
            playedAt = System.nanoTime();
            track.play();
            AudioTimestamp stamp = new AudioTimestamp();
            long emptySince = 0;
            while (data != END) {
                if (data != null && offset < data.length) {
                    int accepted = write(data, offset);
                    if (emptyQueueWaitNs != null && offset == 0 && accepted > 0) {
                        emptyQueueWaitNs.add(emptySince == 0 ? 0 : System.nanoTime() - emptySince);
                        emptySince = 0;
                    }
                    offset += accepted;
                }
                if (data == null || offset == data.length) {
                    data = queue.poll();
                    offset = 0;
                }
                if (emptyQueueWaitNs != null && data == null && emptySince == 0
                        && written > 0
                        && Integer.toUnsignedLong(track.getPlaybackHeadPosition()) >= written) {
                    emptySince = System.nanoTime();
                }
                sampleClock(stamp);
                if (data != END) Thread.sleep(4);
            }
            payloadFrames = written;
            // Keep the mixer fed past the last content frame, including at a resampler boundary.
            short[] tail = new short[RATE / 20];
            int tailOffset = 0;
            long deadline = System.nanoTime() + 5 * NS;
            for (;;) {
                headAtDrain = track.getPlaybackHeadPosition();
                if (Integer.toUnsignedLong(headAtDrain) >= payloadFrames) break;
                if (System.nanoTime() > deadline) {
                    throw new IllegalStateException("AudioTrack head " + headAtDrain
                        + " did not reach " + payloadFrames + " content frames");
                }
                if (tailOffset < tail.length) tailOffset += write(tail, tailOffset);
                sampleClock(stamp);
                Thread.sleep(4);
            }
            drainedAt = System.nanoTime();
            sampleClock(stamp);
            underruns = Math.max(0, track.getUnderrunCount() - underrunsAtStart);
            track.stop();
        } catch (Throwable error) {
            failure = error;
        }
    }

    private static Object elapsed(long start, long end) {
        return end > 0 ? (end - start) / 1e6 : JSONObject.NULL;
    }

    long playCallNs() {
        return playedAt;
    }

    JSONObject metrics(long start) throws Exception {
        JSONObject row = new JSONObject()
            .put("play_call_ms", elapsed(start, playedAt))
            .put("first_presented_signal_ms", elapsed(start, onsetAt))
            .put("playback_drained_ms", elapsed(start, drainedAt))
            .put("timestamp_frame", anchorFrame)
            .put("timestamp_ns", anchorNs)
            .put("last_timestamp_frame", lastTimestampFrame)
            .put("audio_start_underruns", underrunsBeforeOnset)
            .put("audio_write_underruns", underruns)
            .put("audio_written_frames", payloadFrames < 0 ? written : payloadFrames)
            .put("audio_padding_written_frames", payloadFrames < 0 ? 0 : written - payloadFrames)
            .put("audio_playback_head_frames", headAtDrain)
            .put("audio_buffer_frames", bufferFrames)
            .put("audio_start_threshold_frames", thresholdFrames)
            .put("audio_performance_mode", performanceMode)
            .put("audio_route_type", route)
            .put("audio_route_name", routeName == null ? JSONObject.NULL : routeName);
        if (emptyQueueWaitNs != null) {
            JSONArray waits = new JSONArray();
            for (long ns : emptyQueueWaitNs) waits.put(ns / 1e6);
            row.put("observed_queue_empty_wait_ms", waits);
        }
        return row;
    }

    @Override public void close() {
        if (thread.isAlive()) {
            thread.interrupt();
            try {
                thread.join(250);
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
            }
        }
        track.release();
    }
}
