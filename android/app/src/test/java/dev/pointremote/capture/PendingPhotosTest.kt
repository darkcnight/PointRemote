package dev.pointremote.capture

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class PendingPhotosTest {
    @get:Rule val tmp = TemporaryFolder()

    @Test(expected = IllegalArgumentException::class)
    fun rejectsUnsafeLabelId() {
        PendingPhotos(tmp.root).add("../escape", byteArrayOf(1))
    }

    @Test fun addsAndCounts() {
        val pending = PendingPhotos(tmp.root)
        assertEquals(emptyMap<String, Int>(), pending.counts())
        val f = pending.add("light_a", byteArrayOf(1, 2))
        pending.add("light_a", byteArrayOf(3))
        pending.add("unknown", byteArrayOf(4))
        assertArrayEquals(byteArrayOf(1, 2), f.readBytes())
        assertEquals("light_a", f.parentFile!!.name)
        assertEquals(mapOf("light_a" to 2, "unknown" to 1), pending.counts())
        f.delete()
        assertEquals(mapOf("light_a" to 1, "unknown" to 1), pending.counts())
    }
}
