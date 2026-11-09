package dev.pointremote.settings

import android.content.Context
import android.content.SharedPreferences
import androidx.core.content.edit
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

data class AppSettings(
    val haUrl: String = "",
    val haToken: String = "",
    val serverUrl: String = "",
    val serverKey: String = "",
    val threshold: Float = 0.80f,
    val stableFrames: Int = 5,
    /** Quick-point (widget/tile) switches the device as soon as it locks, without a tap. */
    val quickAutoToggle: Boolean = true,
) {
    val haConfigured: Boolean get() = haUrl.isNotBlank() && haToken.isNotBlank()
    val serverConfigured: Boolean get() = serverUrl.isNotBlank() && serverKey.isNotBlank()
}

@Suppress("DEPRECATION") // security-crypto is deprecated upstream but still the simplest encrypted store.
class SettingsRepository(context: Context) {
    private val prefs: SharedPreferences = EncryptedSharedPreferences.create(
        context,
        "pointremote_settings",
        MasterKey.Builder(context).setKeyScheme(MasterKey.KeyScheme.AES256_GCM).build(),
        EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
        EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM,
    )
    private val _settings = MutableStateFlow(load())
    val settings: StateFlow<AppSettings> = _settings.asStateFlow()

    private fun load() = AppSettings(
        haUrl = prefs.getString(HA_URL, "") ?: "",
        haToken = prefs.getString(HA_TOKEN, "") ?: "",
        serverUrl = prefs.getString(SERVER_URL, "") ?: "",
        serverKey = prefs.getString(SERVER_KEY, "") ?: "",
        threshold = prefs.getFloat(THRESHOLD, 0.80f),
        stableFrames = prefs.getInt(STABLE_FRAMES, 5),
        quickAutoToggle = prefs.getBoolean(QUICK_AUTO_TOGGLE, true),
    )

    fun save(settings: AppSettings) {
        prefs.edit {
            putString(HA_URL, settings.haUrl.trim())
            putString(HA_TOKEN, settings.haToken.trim())
            putString(SERVER_URL, settings.serverUrl.trim())
            putString(SERVER_KEY, settings.serverKey.trim())
            putFloat(THRESHOLD, settings.threshold)
            putInt(STABLE_FRAMES, settings.stableFrames)
            putBoolean(QUICK_AUTO_TOGGLE, settings.quickAutoToggle)
        }
        _settings.value = load()
    }

    private companion object {
        const val HA_URL = "ha_url"
        const val HA_TOKEN = "ha_token"
        const val SERVER_URL = "server_url"
        const val SERVER_KEY = "server_key"
        const val THRESHOLD = "threshold"
        const val STABLE_FRAMES = "stable_frames"
        const val QUICK_AUTO_TOGGLE = "quick_auto_toggle"
    }
}
