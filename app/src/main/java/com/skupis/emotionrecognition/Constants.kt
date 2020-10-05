package com.skupis.emotionrecognition

const val  smallestFaceSize = 0.15f// smallest desired face size, expressed as a proportion of the width of the head to the image width
const val frameProcessingTimeout = 200 // in milliseconds
const val frameProcessingRatio = 4 // value indicates how much frames would be skipped, app would process 1 frame per value
