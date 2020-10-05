package com.skupis.emotionrecognition.ui

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.graphics.drawable.BitmapDrawable
import android.os.Bundle
import android.util.DisplayMetrics
import android.util.Log
import android.util.Size
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.skupis.emotionrecognition.R
import com.skupis.emotionrecognition.domain.FaceEmotionClassifier
import com.skupis.emotionrecognition.domain.ImageAnalyzer
import com.skupis.emotionrecognition.domain.RecognitionResult
import kotlinx.android.synthetic.main.activity_main.*
import org.opencv.android.Utils
import org.opencv.core.Mat
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors


@Suppress("DEPRECATION")
class MainActivity : AppCompatActivity() {
    private var imageCapture: ImageCapture? = null
    private lateinit var cameraExecutor: ExecutorService
    private var faceEmotionClassifier = FaceEmotionClassifier(this)
    private val faceAnalyzer = ImageAnalyzer(faceEmotionClassifier)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        setWindowSettings()
        adjustLayoutParams()
        initializeVideoProcessing()
        setImageProcessingObservers()
    }

    private fun setImageProcessingObservers() {
        faceAnalyzer.foundFaces.observe(this, { result ->
            processRecognitionResults(result)
        })

        faceAnalyzer.quickPreview.observe(this, {
            showQuickPreview(it)
        })
    }

    private fun initializeVideoProcessing() {
        // Request camera permissions
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }

        // Setup digit classifier.
        faceEmotionClassifier
            .initialize()
            .addOnFailureListener { e -> Log.e(TAG, "Error to setting up digit classifier.", e) }
    }

    private fun adjustLayoutParams() {
        val displayMetrics = DisplayMetrics()
        windowManager.defaultDisplay.getMetrics(displayMetrics)
        val width: Int = displayMetrics.widthPixels
        viewFinder.layoutParams.height = width
        viewFinder.layoutParams.width = width
        shapesSurface.layoutParams.height = width
        shapesSurface.layoutParams.width = width
    }

    private fun setWindowSettings() {
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        supportActionBar?.hide()
        makeFullScreen()
    }

    private fun showQuickPreview(mat: Mat) {
        val bitmap =
            Bitmap.createBitmap(mat.width(), mat.height(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(mat, bitmap)
        quickPreview.setImageDrawable(BitmapDrawable(resources, bitmap))
    }

    private fun startCamera() {
        cameraExecutor = Executors.newSingleThreadExecutor()

        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Preview
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(viewFinder.createSurfaceProvider())
                }

            imageCapture = ImageCapture.Builder()
                .build()

            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetResolution(Size(1088, 1088))
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, faceAnalyzer)
                }

            // Select back camera as a default
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture, imageAnalyzer
                )

            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it
        ) == PackageManager.PERMISSION_GRANTED
    }

    private val faceContoursPaint =
        Paint().apply {
            isAntiAlias = true
            color = Color.YELLOW
            style = Paint.Style.STROKE
            strokeWidth = 5.0F
            textSize = 70.0F
        }

    private fun processRecognitionResults(result: RecognitionResult) {
        val ratio = viewFinder.height / result.processImageSize.height.toFloat()
        val bitmap =
            Bitmap.createBitmap(viewFinder.width, viewFinder.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        val heightOffset = 0//(shapesSurface.height - result.processImageSize.height) / 4
        val widthOffset = -result.processImageSize.width * ratio / 2 + viewFinder.width / 2
        result.faceList?.forEach {
            canvas.drawRect(
                Rect(
                    (it.bound.left * ratio + widthOffset).toInt(),
                    (it.bound.top.toFloat() * ratio).toInt(),
                    (it.bound.right * ratio + widthOffset).toInt(),
                    (it.bound.bottom.toFloat() * ratio).toInt()
                ), faceContoursPaint
            )
            if (it.emotion != FaceEmotionClassifier.Emotion.Unclassified)
                canvas.drawText(
                    it.emotion.toString(),
                    it.bound.left * ratio + widthOffset,
                    it.bound.top * ratio + heightOffset - 20.toFloat(),
                    faceContoursPaint
                )
        }
        shapesSurface.setImageDrawable(BitmapDrawable(resources, bitmap))
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        faceEmotionClassifier.close()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults:
        IntArray
    ) {
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(
                    this,
                    "Permissions not granted by the user.",
                    Toast.LENGTH_SHORT
                ).show()
                finish()
            }
        }
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus) makeFullScreen()
    }

    override fun onTouchEvent(event: MotionEvent?): Boolean {
        if (event?.action == MotionEvent.ACTION_DOWN) {
            makeFullScreen()
        }
        return super.onTouchEvent(event)
    }

    private fun makeFullScreen() {
        window.decorView.systemUiVisibility = (View.SYSTEM_UI_FLAG_IMMERSIVE
                or View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                or View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                or View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                or View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                or View.SYSTEM_UI_FLAG_FULLSCREEN)
    }

    companion object {
        internal const val TAG = "EmotionRecognitionMainActivity"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }

}