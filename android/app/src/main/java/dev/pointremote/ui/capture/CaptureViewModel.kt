package dev.pointremote.ui.capture

import android.os.SystemClock
import androidx.camera.core.ImageProxy
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dev.pointremote.AppContainer
import dev.pointremote.camera.toCaptureJpeg
import dev.pointremote.camera.uprightBitmap
import dev.pointremote.net.HaEntity
import dev.pointremote.net.HaException
import dev.pointremote.net.ServerException
import dev.pointremote.net.ServerLabel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

val UNKNOWN_LABEL = ServerLabel("unknown", null, "Unknown", 0)

data class CaptureUiState(
    val labels: List<ServerLabel> = listOf(UNKNOWN_LABEL),
    val selected: String = UNKNOWN_LABEL.id,
    val serverOnline: Boolean = false,
    val message: String? = null,
    val entities: List<HaEntity>? = null,
    val capturing: Boolean = false,
    val sessionCount: Int = 0,
)

private const val CAPTURE_INTERVAL_MS = 333L

class CaptureViewModel(private val container: AppContainer) : ViewModel() {
    private val _state = MutableStateFlow(CaptureUiState())
    val state: StateFlow<CaptureUiState> = _state.asStateFlow()
    val pending: StateFlow<Map<String, Int>> = container.photoQueue.counts

    @Volatile private var capturing = false
    @Volatile private var lastShot = 0L

    init {
        refresh()
    }

    fun refresh() {
        viewModelScope.launch {
            val server = container.server()
            if (server == null) {
                _state.update { it.copy(serverOnline = false, message = "Set up the PC server in Settings") }
                return@launch
            }
            try {
                val labels = server.labels()
                _state.update { s ->
                    s.copy(
                        labels = labels,
                        serverOnline = true,
                        message = null,
                        selected = if (labels.any { it.id == s.selected }) s.selected else UNKNOWN_LABEL.id,
                    )
                }
            } catch (e: ServerException) {
                _state.update { it.copy(serverOnline = false, message = e.message) }
            }
            container.photoQueue.refresh()
        }
    }

    fun select(id: String) = _state.update { it.copy(selected = id) }

    fun setCapturing(on: Boolean) {
        capturing = on
        _state.update { it.copy(capturing = on, sessionCount = if (on) 0 else it.sessionCount) }
        if (!on) refresh()
    }

    /** Runs on the camera executor. */
    fun analyze(image: ImageProxy) {
        if (!capturing) return
        val now = SystemClock.elapsedRealtime()
        if (now - lastShot < CAPTURE_INTERVAL_MS) return
        lastShot = now
        val jpeg = image.uprightBitmap().toCaptureJpeg()
        container.photoQueue.enqueue(_state.value.selected, jpeg)
        _state.update { it.copy(sessionCount = it.sessionCount + 1) }
    }

    fun startNewLabel() {
        viewModelScope.launch {
            val ha = container.ha()
            if (ha == null) {
                _state.update { it.copy(message = "Set up Home Assistant in Settings") }
                return@launch
            }
            try {
                val existing = _state.value.labels.mapNotNull { it.entityId }.toSet()
                val entities = ha.entities().filter { it.entityId !in existing }
                _state.update { it.copy(entities = entities, message = null) }
            } catch (e: HaException) {
                _state.update { it.copy(message = e.message) }
            }
        }
    }

    fun cancelNewLabel() = _state.update { it.copy(entities = null) }

    fun createLabel(entity: HaEntity) {
        viewModelScope.launch {
            val server = container.server() ?: return@launch
            try {
                val label = server.createLabel(entity.entityId, entity.friendlyName)
                _state.update { it.copy(entities = null, labels = it.labels + label, selected = label.id) }
            } catch (e: ServerException) {
                _state.update { it.copy(entities = null, message = e.message) }
            }
        }
    }
}
