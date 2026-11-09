package dev.pointremote.ui.point

import androidx.camera.core.ImageProxy
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dev.pointremote.AppContainer
import dev.pointremote.camera.toModelInput
import dev.pointremote.camera.uprightBitmap
import dev.pointremote.classify.OnnxClassifier
import dev.pointremote.model.ModelClass
import dev.pointremote.model.ModelMeta
import dev.pointremote.net.HaException
import dev.pointremote.recognition.LockState
import dev.pointremote.recognition.LockTracker
import dev.pointremote.recognition.softmax
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.receiveAsFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class PointUiState(
    val hasModel: Boolean = false,
    val target: ModelClass? = null,
    val confidence: Float = 0f,
    val toggling: Boolean = false,
    val message: String? = null,
)

sealed interface PointEvent {
    data object Locked : PointEvent
    data object ToggleOk : PointEvent
    data object Error : PointEvent
}

class PointViewModel(private val container: AppContainer) : ViewModel() {
    private val _state = MutableStateFlow(PointUiState())
    val state: StateFlow<PointUiState> = _state.asStateFlow()
    private val _events = Channel<PointEvent>(Channel.BUFFERED)
    val events: Flow<PointEvent> = _events.receiveAsFlow()

    private val lock = Any()
    private var classifier: OnnxClassifier? = null
    private var meta: ModelMeta? = null
    private var tracker = newTracker()

    init {
        viewModelScope.launch { container.modelStore.installed.collect { loadModel(it) } }
        viewModelScope.launch {
            container.settings.settings.collect { synchronized(lock) { tracker = newTracker() } }
        }
    }

    private fun newTracker() = container.settings.settings.value.let { LockTracker(it.threshold, it.stableFrames) }

    private fun loadModel(newMeta: ModelMeta?) {
        val loaded = synchronized(lock) {
            classifier?.close()
            classifier = null
            meta = newMeta
            val file = container.modelStore.modelFile()
            if (newMeta != null && file != null) {
                classifier = runCatching { OnnxClassifier(file, newMeta.inputSize) }.getOrNull()
            }
            tracker = newTracker()
            classifier != null
        }
        _state.update {
            it.copy(
                hasModel = loaded,
                target = null,
                message = if (newMeta != null && !loaded) "Couldn't load model v${newMeta.version}" else it.message,
            )
        }
    }

    /** Runs on the camera executor. */
    fun analyze(image: ImageProxy) {
        val m = synchronized(lock) { meta } ?: return
        val input = image.uprightBitmap().toModelInput(
            m.inputSize,
            m.normalization.mean.toFloatArray(),
            m.normalization.std.toFloatArray(),
        )
        val lockState = synchronized(lock) {
            val c = classifier ?: return
            if (meta !== m) return
            val logits = runCatching { c.logits(input) }.getOrNull() ?: return
            if (logits.size != m.classes.size) return
            tracker.update(softmax(logits))
        }
        val previous = _state.value.target
        when (lockState) {
            LockState.None -> _state.update { it.copy(target = null, confidence = 0f) }
            is LockState.Locked -> {
                val cls = m.classes[lockState.classIndex]
                _state.update { it.copy(target = cls, confidence = lockState.confidence) }
                if (previous?.id != cls.id) _events.trySend(PointEvent.Locked)
            }
        }
    }

    fun toggle() {
        val target = _state.value.target ?: return
        val entityId = target.entityId ?: return
        val ha = container.ha()
        if (ha == null) {
            fail("Set up Home Assistant in Settings")
            return
        }
        _state.update { it.copy(toggling = true) }
        viewModelScope.launch {
            try {
                val newState = ha.toggle(entityId)
                _state.update { it.copy(toggling = false, message = "${target.friendlyName} → ${newState ?: "toggled"}") }
                _events.send(PointEvent.ToggleOk)
            } catch (e: HaException) {
                _state.update { it.copy(toggling = false) }
                fail(e.message ?: "Toggle failed")
            }
        }
    }

    private fun fail(message: String) {
        _state.update { it.copy(message = message) }
        _events.trySend(PointEvent.Error)
    }

    fun clearMessage() = _state.update { it.copy(message = null) }

    override fun onCleared() {
        synchronized(lock) {
            classifier?.close()
            classifier = null
        }
    }
}
