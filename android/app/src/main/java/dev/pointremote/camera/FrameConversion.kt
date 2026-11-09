package dev.pointremote.camera

import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.camera.core.ImageProxy
import dev.pointremote.recognition.centerSquare
import dev.pointremote.recognition.toNchw
import java.io.ByteArrayOutputStream
import kotlin.math.roundToInt

fun ImageProxy.uprightBitmap(): Bitmap {
    val bitmap = toBitmap()
    val rotation = imageInfo.rotationDegrees
    if (rotation == 0) return bitmap
    val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
    return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
}

/** Centre square crop → resize → normalised NCHW, matching the server's eval transform. */
fun Bitmap.toModelInput(size: Int, mean: FloatArray, std: FloatArray): FloatArray {
    val crop = centerSquare(width, height)
    val square = Bitmap.createBitmap(this, crop.left, crop.top, crop.size, crop.size)
    val scaled = Bitmap.createScaledBitmap(square, size, size, true)
    val pixels = IntArray(size * size)
    scaled.getPixels(pixels, 0, size, 0, 0, size, size)
    return toNchw(pixels, size, mean, std)
}

fun Bitmap.toCaptureJpeg(maxSide: Int = 512, quality: Int = 90): ByteArray {
    val scale = maxSide.toFloat() / maxOf(width, height)
    val out = if (scale < 1f) {
        Bitmap.createScaledBitmap(this, (width * scale).roundToInt(), (height * scale).roundToInt(), true)
    } else {
        this
    }
    return ByteArrayOutputStream().also { out.compress(Bitmap.CompressFormat.JPEG, quality, it) }.toByteArray()
}
