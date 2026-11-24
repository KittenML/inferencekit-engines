package org.inferencekit.mediapipecapture;

import android.app.Activity;
import android.os.Bundle;
import android.view.WindowManager;

public final class CaptureActivity extends Activity {
    @Override
    protected void onCreate(Bundle savedState) {
        super.onCreate(savedState);
        setShowWhenLocked(true);
        setTurnScreenOn(true);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    }
}
