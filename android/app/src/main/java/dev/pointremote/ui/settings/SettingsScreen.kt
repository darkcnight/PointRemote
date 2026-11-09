package dev.pointremote.ui.settings

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import dev.pointremote.AppContainer
import dev.pointremote.net.HaClient
import dev.pointremote.net.HaException
import dev.pointremote.net.ServerClient
import dev.pointremote.net.ServerException
import kotlin.math.roundToInt
import kotlinx.coroutines.launch

@Composable
fun SettingsScreen(container: AppContainer) {
    val saved by container.settings.settings.collectAsStateWithLifecycle()
    var draft by remember(saved) { mutableStateOf(saved) }
    var haResult by remember { mutableStateOf<String?>(null) }
    var serverResult by remember { mutableStateOf<String?>(null) }
    val scope = rememberCoroutineScope()

    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("Home Assistant", style = MaterialTheme.typography.titleMedium)
        OutlinedTextField(
            value = draft.haUrl, onValueChange = { draft = draft.copy(haUrl = it) },
            label = { Text("URL, e.g. http://192.168.1.10:8123") }, singleLine = true,
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri), modifier = Modifier.fillMaxWidth(),
        )
        OutlinedTextField(
            value = draft.haToken, onValueChange = { draft = draft.copy(haToken = it) },
            label = { Text("Long-lived access token") }, singleLine = true,
            visualTransformation = PasswordVisualTransformation(), modifier = Modifier.fillMaxWidth(),
        )
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            OutlinedButton(onClick = {
                haResult = "Testing…"
                scope.launch {
                    haResult = try {
                        "Connected: ${HaClient(draft.haUrl, draft.haToken, container.http).entities().size} controllable entities"
                    } catch (e: HaException) {
                        e.message
                    } catch (e: IllegalArgumentException) {
                        "Invalid URL"
                    }
                }
            }) { Text("Test") }
            haResult?.let { Text(it) }
        }

        Text("PC training server", style = MaterialTheme.typography.titleMedium)
        OutlinedTextField(
            value = draft.serverUrl, onValueChange = { draft = draft.copy(serverUrl = it) },
            label = { Text("URL, e.g. http://192.168.1.20:8765") }, singleLine = true,
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri), modifier = Modifier.fillMaxWidth(),
        )
        OutlinedTextField(
            value = draft.serverKey, onValueChange = { draft = draft.copy(serverKey = it) },
            label = { Text("Access key") }, singleLine = true,
            visualTransformation = PasswordVisualTransformation(), modifier = Modifier.fillMaxWidth(),
        )
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            OutlinedButton(onClick = {
                serverResult = "Testing…"
                scope.launch {
                    serverResult = try {
                        val health = ServerClient(draft.serverUrl, draft.serverKey, container.http).health()
                        "Connected: " + (health.latestVersion?.let { "model v$it" } ?: "no model yet")
                    } catch (e: ServerException) {
                        e.message
                    } catch (e: IllegalArgumentException) {
                        "Invalid URL"
                    }
                }
            }) { Text("Test") }
            serverResult?.let { Text(it) }
        }

        Text("Recognition", style = MaterialTheme.typography.titleMedium)
        Text("Confidence threshold: ${(draft.threshold * 100).roundToInt()}%")
        Slider(value = draft.threshold, onValueChange = { draft = draft.copy(threshold = it) }, valueRange = 0.5f..0.99f)
        Text("Stable frames before lock: ${draft.stableFrames}")
        Slider(
            value = draft.stableFrames.toFloat(),
            onValueChange = { draft = draft.copy(stableFrames = it.roundToInt()) },
            valueRange = 2f..15f, steps = 12,
        )

        Text("Widget and tile", style = MaterialTheme.typography.titleMedium)
        Row(verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text("Switch without tapping")
                Text(
                    if (draft.quickAutoToggle) "The device switches as soon as it's recognised."
                    else "Tap the screen to switch the recognised device.",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
            Switch(checked = draft.quickAutoToggle, onCheckedChange = { draft = draft.copy(quickAutoToggle = it) })
        }

        Button(onClick = { container.settings.save(draft) }, enabled = draft != saved, modifier = Modifier.fillMaxWidth()) {
            Text(if (draft == saved) "Saved" else "Save")
        }
    }
}
