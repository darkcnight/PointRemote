package dev.pointremote.setup

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import dev.pointremote.camera.CameraPreview
import dev.pointremote.camera.WithCameraPermission
import java.util.concurrent.atomic.AtomicBoolean

/** Full-screen camera that closes as soon as it sees the PC app's setup QR code. */
@Composable
fun ScanSetupDialog(onScanned: (ServerSetup) -> Unit, onDismiss: () -> Unit) {
    val decoder = remember { QrDecoder() }
    val done = remember { AtomicBoolean(false) }
    val scanned by rememberUpdatedState(onScanned)
    Dialog(onDismissRequest = onDismiss, properties = DialogProperties(usePlatformDefaultWidth = false)) {
        Surface(Modifier.fillMaxSize(), color = Color.Black) {
            WithCameraPermission {
                Box(Modifier.fillMaxSize()) {
                    CameraPreview(onFrame = { frame ->
                        if (done.get()) return@CameraPreview
                        val setup = decoder.decode(frame)?.let(::parseSetupQr) ?: return@CameraPreview
                        if (done.compareAndSet(false, true)) {
                            android.os.Handler(android.os.Looper.getMainLooper()).post { scanned(setup) }
                        }
                    }, modifier = Modifier.fillMaxSize())
                    Box(
                        Modifier.align(Alignment.Center).size(240.dp)
                            .border(3.dp, Color.White.copy(alpha = 0.8f), RoundedCornerShape(20.dp)),
                    )
                    Column(
                        Modifier.align(Alignment.TopCenter).padding(top = 48.dp, start = 24.dp, end = 24.dp)
                            .background(Color.Black.copy(alpha = 0.55f), RoundedCornerShape(12.dp)).padding(12.dp),
                    ) {
                        Text("Point at the QR code in the PointRemote window on your PC",
                            color = Color.White, style = MaterialTheme.typography.bodyLarge)
                    }
                    TextButton(onClick = onDismiss, modifier = Modifier.align(Alignment.BottomCenter).padding(bottom = 40.dp)) {
                        Text("Cancel", color = Color.White)
                    }
                }
            }
        }
    }
}
