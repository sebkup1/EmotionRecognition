package com.skupis.emotionrecognition.domain

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Rect
import android.util.Log
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.TaskCompletionSource
import com.skupis.emotionrecognition.frameProcessingTimeout
import org.opencv.core.Mat
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.concurrent.Callable
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

class FaceEmotionClassifier(private val context: Context) {
    private var interpreter: Interpreter? = null
    var isInitialized = false
        private set

    /** Executor to run inference task in the background. */
    private val executorService: ExecutorService = Executors.newCachedThreadPool()

    private var inputImageWidth: Int = 0 // will be inferred from TF Lite model.
    private var inputImageHeight: Int = 0 // will be inferred from TF Lite model.
    private var modelInputSize: Int = 0 // will be inferred from TF Lite model.

    fun initialize(): Task<Void> {
        val task = TaskCompletionSource<Void>()
        executorService.execute {
            try {
                initializeInterpreter()
                task.setResult(null)
            } catch (e: IOException) {
                task.setException(e)
            }
        }
        return task.task
    }

    @Throws(IOException::class)
    private fun initializeInterpreter() {
        // Load the TF Lite model from asset folder and initialize TF Lite Interpreter with NNAPI enabled.
        val assetManager = context.assets
        val model = "model_facial_expression_quant (1) (1).tflite".loadModelFile(assetManager)
        val options = Interpreter.Options()
        options.setUseNNAPI(true)
        val interpreter = Interpreter(model, options)

        // Read input shape from model file.
        val inputShape = interpreter.getInputTensor(0).shape()
        inputImageWidth = inputShape[1]
        inputImageHeight = inputShape[2]
        modelInputSize = FLOAT_TYPE_SIZE * inputImageWidth *
                inputImageHeight * PIXEL_SIZE

        // Finish interpreter initialization.
        this.interpreter = interpreter

        isInitialized = true
        Log.d(TAG, "Initialized TFLite interpreter.")
    }

    @Throws(IOException::class)
    private fun String.loadModelFile(assetManager: AssetManager): ByteBuffer {
        val fileDescriptor = assetManager.openFd(this)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun classify(classificationInput: ClassificationInput): FaceEmotionClassifyResult {
        check(isInitialized) { "TF Lite Interpreter is not initialized yet." }

        // Define an array to store the model output.
        val output = Array(1) { FloatArray(OUTPUT_CLASSES_COUNT) }

        // Run inference with the input data.
        interpreter?.run(classificationInput.mat.toByteBuffer(), output)

        // Post-processing: find the digit that has the highest probability
        // and return it a human-readable string.
        val result = output[0]
        val maxIndex = result.indices.maxByOrNull { result[it] } ?: -1
        return FaceEmotionClassifyResult(
            classificationInput.onParentBound,
            Emotion.values()[maxIndex + 1]
        )
    }

    fun classifyMultipleFaces(inputList: List<ClassificationInput>): List<FaceEmotionClassifyResult>? =
        executorService.invokeAll(inputList.map {
            Callable {
                classify(it)
            }
        }, frameProcessingTimeout.toLong(), TimeUnit.MILLISECONDS).mapIndexed { idx, value ->
            if (!value.isCancelled)
                value.get()
            else FaceEmotionClassifyResult(inputList[idx].onParentBound, Emotion.Unclassified)
        }

    fun close() {
        executorService.execute {
            interpreter?.close()
            Log.d(TAG, "Closed TFLite interpreter.")
        }
    }

    private fun Mat.toByteBuffer(): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(modelInputSize)
        byteBuffer.order(ByteOrder.nativeOrder())
        for (r in 0 until rows()) {
            for (c in 0 until cols()) {
                byteBuffer.putFloat(this.get(r, c)[0].toFloat())
            }
        }
        return byteBuffer
    }

    enum class Emotion {
        Unclassified, Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
    }

    class ClassificationInput(val onParentBound: Rect, val mat: Mat)

    companion object {
        private const val TAG = "FaceEmotionClassifier"
        private const val FLOAT_TYPE_SIZE = 4
        private const val PIXEL_SIZE = 1
        private const val OUTPUT_CLASSES_COUNT = 7
    }
}