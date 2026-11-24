package org.kittenml.s25study;

final class Pico {
    static {
        System.loadLibrary("pico_bench");
    }

    static native long create(String modelDir, int threads);
    static native long createVariant(String modelDir, String prefix, String precision,
                                     String phonemizer, int threads);
    static native String buildInfo();
    static native void generate(long handle, String text, float speed, Sink sink);
    static native void destroy(long handle);
}
