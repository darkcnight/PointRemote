package dev.pointremote.ui.train

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dev.pointremote.AppContainer
import dev.pointremote.model.ModelMeta
import dev.pointremote.model.UpdateResult
import dev.pointremote.net.ServerClient
import dev.pointremote.net.ServerException
import dev.pointremote.net.ServerLabel
import dev.pointremote.net.TrainStatus
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

const val MIN_PHOTOS = 30
const val RECOMMENDED_PHOTOS = 100

data class TrainUiState(
    val online: Boolean? = null,
    val labels: List<ServerLabel> = emptyList(),
    val status: TrainStatus? = null,
    val meta: ModelMeta? = null,
    val message: String? = null,
    val busy: Boolean = false,
) {
    val canTrain: Boolean
        get() = online == true && !busy && status?.running != true &&
            labels.size >= 2 && labels.all { it.photoCount >= MIN_PHOTOS }
}

class TrainViewModel(private val container: AppContainer) : ViewModel() {
    private val _state = MutableStateFlow(TrainUiState())
    val state: StateFlow<TrainUiState> = _state.asStateFlow()
    private var pollJob: Job? = null

    init {
        viewModelScope.launch { container.modelStore.installed.collect { m -> _state.update { it.copy(meta = m) } } }
        refresh()
    }

    fun refresh() {
        viewModelScope.launch {
            val server = container.server()
            if (server == null) {
                _state.update { it.copy(online = false, message = "Set up the PC server in Settings") }
                return@launch
            }
            try {
                server.health()
                val labels = server.labels()
                val status = server.trainStatus()
                _state.update { it.copy(online = true, labels = labels, status = status, message = null) }
                if (status.running) startPolling(server) else syncModel(server)
            } catch (e: ServerException) {
                _state.update { it.copy(online = false, message = e.message) }
            }
        }
    }

    fun train() {
        val server = container.server() ?: return
        viewModelScope.launch {
            _state.update { it.copy(busy = true, message = null) }
            try {
                val status = server.startTraining()
                _state.update { it.copy(status = status) }
                startPolling(server)
            } catch (e: ServerException) {
                _state.update { it.copy(message = e.message) }
            } finally {
                _state.update { it.copy(busy = false) }
            }
        }
    }

    private fun startPolling(server: ServerClient) {
        if (pollJob?.isActive == true) return
        pollJob = viewModelScope.launch {
            try {
                while (true) {
                    val status = server.trainStatus()
                    _state.update { it.copy(status = status) }
                    if (!status.running) {
                        if (status.state == "done") syncModel(server)
                        if (status.state == "failed") _state.update { it.copy(message = "Training failed: ${status.message}") }
                        break
                    }
                    delay(2000)
                }
            } catch (e: ServerException) {
                _state.update { it.copy(message = e.message) }
            }
        }
    }

    private suspend fun syncModel(server: ServerClient) {
        try {
            val result = container.modelStore.updateFrom(server)
            if (result is UpdateResult.Updated) {
                _state.update { it.copy(message = "Model v${result.meta.version} installed") }
            }
        } catch (e: Exception) {
            _state.update { it.copy(message = "Model download failed: ${e.message}") }
        }
    }
}
