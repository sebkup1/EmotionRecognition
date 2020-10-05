package com.skupis.emotionrecognition.app;
import android.app.Application
import android.util.Log
import com.google.firebase.FirebaseApp
import org.opencv.android.OpenCVLoader

open class EmotionRecognitionApp : Application() {
    private val TAG = "EmotionRecognitionApp"

    override fun onCreate() {
        super.onCreate()
        FirebaseApp.initializeApp(this)
        if (!OpenCVLoader.initDebug()) {
            Log.e(TAG,"OpenCV not loaded")
        }
    }
}
