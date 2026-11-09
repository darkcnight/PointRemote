package dev.pointremote.recognition

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LockTrackerTest {
    private val a = floatArrayOf(0.05f, 0.9f, 0.05f)
    private val b = floatArrayOf(0.05f, 0.05f, 0.9f)
    private val unsure = floatArrayOf(0.2f, 0.5f, 0.3f)
    private val unknown = floatArrayOf(0.9f, 0.05f, 0.05f)

    private fun LockTracker.feed(frame: FloatArray, times: Int): LockState {
        var state: LockState = LockState.None
        repeat(times) { state = update(frame) }
        return state
    }

    @Test fun locksOnlyAfterStableFrames() {
        val t = LockTracker(0.8f, 3)
        assertEquals(LockState.None, t.update(a))
        assertEquals(LockState.None, t.update(a))
        assertEquals(LockState.Locked(1, 0.9f), t.update(a))
    }

    @Test fun unknownNeverLocks() {
        assertEquals(LockState.None, LockTracker(0.8f, 3).feed(unknown, 10))
    }

    @Test fun lowConfidenceNeverLocks() {
        assertEquals(LockState.None, LockTracker(0.8f, 3).feed(unsure, 10))
    }

    @Test fun interruptedStreakStartsOver() {
        val t = LockTracker(0.8f, 3)
        t.feed(a, 2)
        t.update(unsure)
        assertEquals(LockState.None, t.feed(a, 2))
        assertTrue(t.update(a) is LockState.Locked)
    }

    @Test fun singleBadFrameKeepsLock() {
        val t = LockTracker(0.8f, 3)
        t.feed(a, 3)
        assertEquals(LockState.Locked(1, 0.9f), t.update(unsure))
    }

    @Test fun unlocksAfterStableBadFrames() {
        val t = LockTracker(0.8f, 3)
        t.feed(a, 3)
        assertTrue(t.feed(unsure, 2) is LockState.Locked)
        assertEquals(LockState.None, t.update(unsure))
    }

    @Test fun switchesToNewTarget() {
        val t = LockTracker(0.8f, 3)
        t.feed(a, 3)
        assertEquals(LockState.Locked(1, 0.9f), t.update(b))
        assertEquals(LockState.Locked(1, 0.9f), t.update(b))
        assertEquals(LockState.Locked(2, 0.9f), t.update(b))
    }

    @Test fun confidenceTracksLatestFrame() {
        val t = LockTracker(0.8f, 3)
        t.feed(a, 3)
        assertEquals(LockState.Locked(1, 0.95f), t.update(floatArrayOf(0.03f, 0.95f, 0.02f)))
    }

    @Test fun resetClearsLock() {
        val t = LockTracker(0.8f, 3)
        t.feed(a, 3)
        t.reset()
        assertEquals(LockState.None, t.update(a))
    }

    @Test fun softmaxIsNormalisedAndStable() {
        val p = softmax(floatArrayOf(1f, 2f, 3f))
        assertEquals(1f, p.sum(), 1e-5f)
        assertTrue(p[2] > p[1] && p[1] > p[0])
        val big = softmax(floatArrayOf(1000f, 1001f))
        assertEquals(1f, big.sum(), 1e-5f)
        assertTrue(big.none { it.isNaN() })
    }
}
