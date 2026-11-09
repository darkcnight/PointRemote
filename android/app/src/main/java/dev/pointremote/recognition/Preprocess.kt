package dev.pointremote.recognition

data class CropRect(val left: Int, val top: Int, val size: Int)

fun centerSquare(width: Int, height: Int): CropRect {
    val size = minOf(width, height)
    return CropRect((width - size) / 2, (height - size) / 2, size)
}

/** ARGB pixels (row-major, size × size) → normalised NCHW floats (R, G, B planes). */
fun toNchw(pixels: IntArray, size: Int, mean: FloatArray, std: FloatArray): FloatArray {
    require(pixels.size == size * size) { "expected ${size * size} pixels, got ${pixels.size}" }
    val plane = size * size
    val out = FloatArray(3 * plane)
    for (i in 0 until plane) {
        val c = pixels[i]
        val r = ((c shr 16) and 0xFF) / 255f
        val g = ((c shr 8) and 0xFF) / 255f
        val b = (c and 0xFF) / 255f
        out[i] = (r - mean[0]) / std[0]
        out[plane + i] = (g - mean[1]) / std[1]
        out[2 * plane + i] = (b - mean[2]) / std[2]
    }
    return out
}
