package dev.pointremote.ui.point

import android.os.Build
import android.view.HapticFeedbackConstants
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import dev.pointremote.AppContainer
import dev.pointremote.camera.CameraPreview
import dev.pointremote.camera.WithCameraPermission
import kotlinx.coroutines.delay
import kotlin.math.roundToInt

private val CONFIRM = if (Build.VERSION.SDK_INT >= 30) HapticFeedbackConstants.CONFIRM else HapticFeedbackConstants.LONG_PRESS
private val REJECT = if (Build.VERSION.SDK_INT >= 30) HapticFeedbackConstants.REJECT else HapticFeedbackConstants.LONG_PRESS

@Composable
fun PointScreen(container: AppContainer, onOpenCapture: () -> Unit) {
    val vm: PointViewModel = viewModel { PointViewModel(container) }
    val state by vm.state.collectAsStateWithLifecycle()
    val view = LocalView.current

    LaunchedEffect(vm) {
        vm.events.collect { event ->
            view.performHapticFeedback(
                when (event) {
                    PointEvent.Locked -> HapticFeedbackConstants.CLOCK_TICK
                    PointEvent.ToggleOk -> CONFIRM
                    PointEvent.Error -> REJECT
                },
            )
        }
    }
    LaunchedEffect(state.message) {
        if (state.message != null) {
            delay(3000)
            vm.clearMessage()
        }
    }

    if (!state.hasModel) {
        Box(Modifier.fillMaxSize().padding(24.dp), contentAlignment = Alignment.Center) {
            Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(16.dp)) {
                Text("Capture and train first", style = MaterialTheme.typography.titleLarge)
                state.message?.let { Text(it, textAlign = TextAlign.Center) }
                Button(onClick = onOpenCapture) { Text("Go to Capture") }
            }
        }
    } else {
        WithCameraPermission {
            Box(Modifier.fillMaxSize()) {
                CameraPreview(onFrame = vm::analyze, modifier = Modifier.fillMaxSize())
                state.target?.let { target ->
                    Surface(
                        modifier = Modifier.align(Alignment.TopCenter).padding(top = 24.dp),
                        shape = RoundedCornerShape(50),
                        color = MaterialTheme.colorScheme.primaryContainer,
                    ) {
                        Text(
                            "${target.friendlyName} · ${(state.confidence * 100).roundToInt()}%",
                            modifier = Modifier.padding(horizontal = 20.dp, vertical = 10.dp),
                            style = MaterialTheme.typography.titleMedium,
                        )
                    }
                }
                Column(
                    modifier = Modifier.align(Alignment.BottomCenter).fillMaxWidth().padding(24.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    state.message?.let {
                        Surface(shape = RoundedCornerShape(12.dp), color = MaterialTheme.colorScheme.surfaceVariant) {
                            Text(it, modifier = Modifier.padding(12.dp).fillMaxWidth(), textAlign = TextAlign.Center)
                        }
                    }
                    Button(
                        onClick = vm::toggle,
                        enabled = state.target != null && !state.toggling,
                        modifier = Modifier.fillMaxWidth().height(72.dp),
                    ) {
                        Text(if (state.toggling) "…" else "Toggle", style = MaterialTheme.typography.headlineSmall)
                    }
                }
            }
        }
    }
}
