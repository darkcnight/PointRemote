package dev.pointremote.setup

import androidx.camera.core.ImageProxy
import com.google.zxing.BarcodeFormat
import com.google.zxing.BinaryBitmap
import com.google.zxing.DecodeHintType
import com.google.zxing.MultiFormatReader
import com.google.zxing.NotFoundException
import com.google.zxing.PlanarYUVLuminanceSource
import com.google.zxing.ReaderException
import com.google.zxing.common.HybridBinarizer

/** Finds a QR code in an RGBA camera frame. Not thread-safe; use one per analyzer thread. */
class QrDecoder {
    private val reader = MultiFormatReader().apply {
        setHints(mapOf(DecodeHintType.POSSIBLE_FORMATS to listOf(BarcodeFormat.QR_CODE), DecodeHintType.TRY_HARDER to true))
    }
    private var luma = ByteArray(0)

    fun decode(image: ImageProxy): String? {
        val plane = image.planes[0]
        val w = image.width
        val h = image.height
        if (luma.size != w * h) luma = ByteArray(w * h)
        val buf = plane.buffer
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        for (y in 0 until h) {
            val row = y * rowStride
            for (x in 0 until w) {
                val i = row + x * pixelStride
                val r = buf.get(i).toInt() and 0xFF
                val g = buf.get(i + 1).toInt() and 0xFF
                val b = buf.get(i + 2).toInt() and 0xFF
                luma[y * w + x] = ((r + 2 * g + b) shr 2).toByte()
            }
        }
        val source = PlanarYUVLuminanceSource(luma, w, h, 0, 0, w, h, false)
        return try {
            reader.decodeWithState(BinaryBitmap(HybridBinarizer(source))).text
        } catch (_: NotFoundException) {
            null
        } catch (_: ReaderException) {
            null
        } finally {
            reader.reset()
        }
    }
}
