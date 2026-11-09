package dev.pointremote.recognition

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class PreprocessTest {
    @Test fun centerSquareLandscape() = assertEquals(CropRect(80, 0, 480), centerSquare(640, 480))

    @Test fun centerSquarePortrait() = assertEquals(CropRect(0, 80, 480), centerSquare(480, 640))

    @Test fun centerSquareSquare() = assertEquals(CropRect(0, 0, 100), centerSquare(100, 100))

    private val pixels = intArrayOf(
        0xFFFF0000.toInt(), 0xFF00FF00.toInt(),
        0xFF0000FF.toInt(), 0xFFFFFFFF.toInt(),
    )

    @Test fun toNchwSplitsPlanes() {
        val out = toNchw(pixels, 2, floatArrayOf(0f, 0f, 0f), floatArrayOf(1f, 1f, 1f))
        assertArrayEquals(
            floatArrayOf(
                1f, 0f, 0f, 1f, // R
                0f, 1f, 0f, 1f, // G
                0f, 0f, 1f, 1f, // B
            ),
            out, 1e-6f,
        )
    }

    @Test fun toNchwNormalises() {
        val out = toNchw(pixels, 2, floatArrayOf(0.5f, 0.5f, 0.5f), floatArrayOf(0.5f, 0.5f, 0.5f))
        assertEquals(1f, out[0], 1e-6f)   // red pixel, R plane
        assertEquals(-1f, out[4], 1e-6f)  // red pixel, G plane
    }
}
