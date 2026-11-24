package org.kittenml.s25study;

final class Adreno {
    static { System.loadLibrary("study_adreno"); }

    static native long create(String dataPath, String libraryPath);
    static native void generate(long handle, String text, Sink sink);
    static native void destroy(long handle);
}
