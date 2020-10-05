package com.skupis.emotionrecognition.domain

import android.graphics.Rect
import android.util.Log
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import com.google.firebase.ml.vision.FirebaseVision
import com.google.firebase.ml.vision.common.FirebaseVisionImage
import com.google.firebase.ml.vision.face.FirebaseVisionFace
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetectorOptions
import com.skupis.emotionrecognition.*
import kotlinx.coroutines.*
import org.opencv.core.*
import org.opencv.imgproc.Imgproc


class ImageAnalyzer(private val faceEmotionClassifier: FaceEmotionClassifier) :
    ImageAnalysis.Analyzer {

    private val options = FirebaseVisionFaceDetectorOptions.Builder()
        .setMinFaceSize(smallestFaceSize)
        .build()

    private val detector = FirebaseVision.getInstance()
        .getVisionFaceDetector(options)

    private val _foundFaces = MutableLiveData<RecognitionResult>()
    val foundFaces: LiveData<RecognitionResult>
        get() = _foundFaces

    private val _quickPreview = MutableLiveData<Mat>()
    val quickPreview: LiveData<Mat>
        get() = _quickPreview

    private var classificationJob = Job()
    private val classificationScope = CoroutineScope(Dispatchers.Main + classificationJob)

    private var framesCounter = 0

    @ExperimentalGetImage
    override fun analyze(imageProxy: ImageProxy) {
        if (++framesCounter != frameProcessingRatio) {
            imageProxy.close()
            return
        }
        framesCounter = 0
        val mediaImage = imageProxy.image ?: return
        val image = FirebaseVisionImage.fromMediaImage(
            mediaImage,
            degreesToFirebaseRotation(imageProxy.imageInfo.rotationDegrees)
        )

        val rgbaMat = mediaImage.toOpenCvMat()
        mediaImage.close()
        Core.flip(rgbaMat.t(), rgbaMat, 1)
//            _quickPreview.postValue(rgbaMat)

        detector.detectInImage(image)
            .addOnSuccessListener { faces ->
                val facesToClassify = ArrayList<FaceEmotionClassifier.ClassificationInput>()
                for (face in faces) {
                    if (!isFaceInFrameBound(rgbaMat, face.boundingBox)) continue
                    facesToClassify.add(
                        FaceEmotionClassifier.ClassificationInput(
                            face.boundingBox,
                            convertFaceImageToClassifiableFormat(rgbaMat, face)
                        )
                    )
                }

                if (faceEmotionClassifier.isInitialized) {
                    runClassificationAsync(rgbaMat, facesToClassify)
                }
            }
            .addOnFailureListener { e ->
                Log.e(TAG, "result failure $e")
            }
        imageProxy.close()

    }

    private fun isFaceInFrameBound(frame: Mat, faceBoundingBox: Rect): Boolean =
        with(faceBoundingBox) {
            if (this.left >= 0 && this.top >= 0 && this.right <= frame.width() && this.bottom <= frame.height()) {
                return true
            } else false
        }

    private fun runClassificationAsync(
        rgbaMat: Mat,
        facesToClassify: ArrayList<FaceEmotionClassifier.ClassificationInput>
    ) {
        classificationScope.launch {
            withContext(Dispatchers.Main) {
                _foundFaces.postValue(
                    RecognitionResult(
                        android.util.Size(
                            rgbaMat.width(),
                            rgbaMat.height()
                        ),
                        faceEmotionClassifier.classifyMultipleFaces(
                            facesToClassify
                        )
                    )
                )
            }
        }
    }

    private fun convertFaceImageToClassifiableFormat(
        rgbaMat: Mat,
        face: FirebaseVisionFace
    ): Mat {
        val input = Mat(rgbaMat, androidToOpenCvRect(face.boundingBox))
//        _quickPreview.postValue(input)
        val mat = Mat()
        Imgproc.cvtColor(input, mat, Imgproc.COLOR_RGB2GRAY)
        Imgproc.resize(mat, mat, Size(64.0, 64.0), -1.0, -1.0, Imgproc.INTER_AREA)
        mat.convertTo(mat, CvType.CV_32F)
        Core.divide(mat, Scalar(127.5, 127.5, 127.5), mat)
        Core.subtract(mat, Scalar(1.0, 1.0, 1.0), mat)
        return mat
    }

    companion object {
        private const val TAG = "EmotionRecognitionImageAnalyzer"
    }
}
