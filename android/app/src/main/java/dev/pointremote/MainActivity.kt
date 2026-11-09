package dev.pointremote

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.lifecycle.lifecycleScope
import dev.pointremote.launch.refreshPointWidgets
import dev.pointremote.ui.AppRoot
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        val container = (application as PointRemoteApp).container
        // Check for a newer model on every launch; failure is silent (the installed model keeps working).
        lifecycleScope.launch {
            container.server()?.let { server -> runCatching { container.modelStore.updateFrom(server) } }
        }
        lifecycleScope.launch { refreshPointWidgets(applicationContext) }
        setContent { AppRoot(container) }
    }
}
