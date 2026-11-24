package org.kittenml.s25study;

final class Sherpa {
    static {
        System.loadLibrary("kokoro_bench");
    }

    static native long create(String modelDir, int threads);
    static native long createVariant(String modelDir, String modelFile, int threads);
    static native void generate(long handle, String text, float speed, Sink sink);
    static native void destroy(long handle);
    static native String version();
}
