package com.skupis.emotionrecognition.domain

import android.graphics.Rect
import android.util.Size

class RecognitionResult(
    val processImageSize: Size,
    val faceList: List<FaceEmotionClassifyResult>?
)

class FaceEmotionClassifyResult(val bound : Rect,
                                val emotion : FaceEmotionClassifier.Emotion)

