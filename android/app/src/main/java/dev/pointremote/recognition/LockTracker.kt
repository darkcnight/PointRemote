package dev.pointremote.recognition

import kotlin.math.exp

sealed interface LockState {
    data object None : LockState
    data class Locked(val classIndex: Int, val confidence: Float) : LockState
}

fun softmax(logits: FloatArray): FloatArray {
    val max = logits.max()
    val exps = FloatArray(logits.size) { exp((logits[it] - max).toDouble()).toFloat() }
    val sum = exps.sum()
    return FloatArray(exps.size) { exps[it] / sum }
}

/**
 * Turns a stream of per-frame class probabilities into a stable "locked target".
 * A class locks after [stableFrames] consecutive confident frames and unlocks
 * after [stableFrames] consecutive frames that don't confirm it.
 */
class LockTracker(
    private val threshold: Float,
    private val stableFrames: Int,
    private val unknownIndex: Int = 0,
) {
    private var streakClass = -1
    private var streakCount = 0
    private var locked = -1
    private var lockedConfidence = 0f
    private var failCount = 0

    fun reset() {
        streakClass = -1
        streakCount = 0
        locked = -1
        failCount = 0
    }

    fun update(probabilities: FloatArray): LockState {
        var best = 0
        for (i in probabilities.indices) if (probabilities[i] > probabilities[best]) best = i
        val p = probabilities[best]
        val qualifies = best != unknownIndex && p >= threshold

        if (qualifies) {
            if (best == streakClass) streakCount++ else {
                streakClass = best
                streakCount = 1
            }
        } else {
            streakClass = -1
            streakCount = 0
        }

        if (locked >= 0) {
            if (qualifies && best == locked) {
                failCount = 0
                lockedConfidence = p
            } else if (++failCount >= stableFrames) {
                locked = -1
                failCount = 0
            }
        }
        if (locked < 0 && streakCount >= stableFrames) {
            locked = streakClass
            lockedConfidence = p
            failCount = 0
        }
        return if (locked >= 0) LockState.Locked(locked, lockedConfidence) else LockState.None
    }
}
