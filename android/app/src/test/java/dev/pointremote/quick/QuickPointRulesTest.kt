package dev.pointremote.quick

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class QuickPointRulesTest {
    @Test fun autoTogglesOnFirstLockOnly() {
        val rules = QuickPointRules(autoToggle = true)
        assertTrue(rules.onLocked())
        assertFalse(rules.onLocked())
        assertFalse(rules.onTap(hasTarget = true))
    }

    @Test fun tapModeIgnoresLocksAndTogglesOnTap() {
        val rules = QuickPointRules(autoToggle = false)
        assertFalse(rules.onLocked())
        assertFalse(rules.onTap(hasTarget = false))
        assertTrue(rules.onTap(hasTarget = true))
        assertFalse(rules.onTap(hasTarget = true))
    }

    @Test fun failedToggleCanBeRetried() {
        val rules = QuickPointRules(autoToggle = true)
        assertTrue(rules.onLocked())
        rules.onToggleFailed()
        assertTrue(rules.onTap(hasTarget = true))
    }
}
