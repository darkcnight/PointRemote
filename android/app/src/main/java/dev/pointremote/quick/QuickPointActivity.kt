package dev.pointremote.quick

import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.view.HapticFeedbackConstants
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import dev.pointremote.AppContainer
import dev.pointremote.MainActivity
import dev.pointremote.PointRemoteApp
import dev.pointremote.camera.CameraPreview
import dev.pointremote.camera.WithCameraPermission
import dev.pointremote.ui.point.PointEvent
import dev.pointremote.ui.point.PointViewModel
import kotlinx.coroutines.delay
import kotlin.math.roundToInt

private const val CLOSE_AFTER_TOGGLE_MS = 1200L

/** Stripped-down Point: camera only, toggles the locked device, then closes. Opened by the widget and tile. */
class QuickPointActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        val container = (application as PointRemoteApp).container
        setContent {
            MaterialTheme(colorScheme = if (isSystemInDarkTheme()) darkColorScheme() else lightColorScheme()) {
                QuickPointScreen(
                    container,
                    onDone = ::finish,
                    onOpenApp = {
                        startActivity(Intent(this, MainActivity::class.java))
                        finish()
                    },
                )
            }
        }
    }

    companion object {
        fun intent(context: Context): Intent = Intent(context, QuickPointActivity::class.java)
    }
}

private val CONFIRM = if (Build.VERSION.SDK_INT >= 30) HapticFeedbackConstants.CONFIRM else HapticFeedbackConstants.LONG_PRESS
private val REJECT = if (Build.VERSION.SDK_INT >= 30) HapticFeedbackConstants.REJECT else HapticFeedbackConstants.LONG_PRESS

@Composable
private fun QuickPointScreen(container: AppContainer, onDone: () -> Unit, onOpenApp: () -> Unit) {
    val vm: PointViewModel = viewModel { PointViewModel(container) }
    val state by vm.state.collectAsStateWithLifecycle()
    val autoToggle = remember { container.settings.settings.value.quickAutoToggle }
    val rules = remember { QuickPointRules(autoToggle) }
    val view = LocalView.current

    LaunchedEffect(vm) {
        vm.events.collect { event ->
            when (event) {
                PointEvent.Locked -> {
                    view.performHapticFeedback(HapticFeedbackConstants.CLOCK_TICK)
                    if (rules.onLocked()) vm.toggle()
                }
                PointEvent.ToggleOk -> {
                    view.performHapticFeedback(CONFIRM)
                    delay(CLOSE_AFTER_TOGGLE_MS)
                    onDone()
                }
                PointEvent.Error -> {
                    view.performHapticFeedback(REJECT)
                    rules.onToggleFailed()
                }
            }
        }
    }

    if (!state.hasModel) {
        Box(Modifier.fillMaxSize().padding(24.dp), contentAlignment = Alignment.Center) {
            Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(16.dp)) {
                Text("No model yet", style = MaterialTheme.typography.titleLarge)
                Text("Capture photos and train in the app first.", textAlign = TextAlign.Center)
                Button(onClick = onOpenApp) { Text("Open PointRemote") }
            }
        }
        return
    }

    WithCameraPermission {
        Box(
            Modifier.fillMaxSize().clickable(
                interactionSource = remember { MutableInteractionSource() },
                indication = null,
            ) { if (rules.onTap(state.target != null)) vm.toggle() },
        ) {
            CameraPreview(onFrame = vm::analyze, modifier = Modifier.fillMaxSize())
            Column(
                Modifier.align(Alignment.TopCenter).safeDrawingPadding().padding(top = 16.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                Surface(shape = RoundedCornerShape(50), color = MaterialTheme.colorScheme.primaryContainer) {
                    val target = state.target
                    Text(
                        when {
                            state.toggling -> "Switching ${target?.friendlyName ?: ""}…"
                            target != null -> "${target.friendlyName} · ${(state.confidence * 100).roundToInt()}%"
                            else -> "Aim at a device"
                        },
                        modifier = Modifier.padding(horizontal = 20.dp, vertical = 10.dp),
                        style = MaterialTheme.typography.titleMedium,
                    )
                }
            }
            Column(
                Modifier.align(Alignment.BottomCenter).fillMaxWidth().safeDrawingPadding().padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                val hint = state.message ?: if (!autoToggle && state.target != null) "Tap anywhere to switch it" else null
                hint?.let {
                    Surface(shape = RoundedCornerShape(12.dp), color = MaterialTheme.colorScheme.surfaceVariant) {
                        Text(it, modifier = Modifier.padding(12.dp), textAlign = TextAlign.Center)
                    }
                }
                TextButton(onClick = onDone) { Text("Close") }
            }
        }
    }
}
