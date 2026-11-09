package dev.pointremote.quick

/**
 * When quick-point toggles. At most one toggle per session, so holding the phone on a
 * device never flips it back and forth; a failed toggle can be retried by tapping.
 */
class QuickPointRules(private val autoToggle: Boolean) {
    private var fired = false

    /** A device just locked. True if it should be toggled now. */
    fun onLocked(): Boolean = (autoToggle && !fired).also { if (it) fired = true }

    /** The screen was tapped. True if the locked device should be toggled now. */
    fun onTap(hasTarget: Boolean): Boolean = (hasTarget && !fired).also { if (it) fired = true }

    fun onToggleFailed() {
        fired = false
    }
}
