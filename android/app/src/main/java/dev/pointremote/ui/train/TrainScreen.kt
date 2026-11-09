package dev.pointremote.ui.train

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import dev.pointremote.AppContainer
import kotlin.math.roundToInt

private fun pct(x: Double) = "${(x * 100).roundToInt()}%"

@Composable
fun TrainScreen(container: AppContainer) {
    val vm: TrainViewModel = viewModel { TrainViewModel(container) }
    val state by vm.state.collectAsStateWithLifecycle()
    LaunchedEffect(Unit) { vm.refresh() }

    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Text(
                when (state.online) {
                    true -> "PC server online"
                    false -> "PC offline"
                    null -> "Checking PC server…"
                },
                style = MaterialTheme.typography.titleMedium,
            )
            OutlinedButton(onClick = vm::refresh) { Text("Refresh") }
        }

        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Text("Photos per label", style = MaterialTheme.typography.titleMedium)
                if (state.labels.size < 2) Text("Add at least one device label in Capture.")
                state.labels.forEach { label ->
                    val color = when {
                        label.photoCount < MIN_PHOTOS -> MaterialTheme.colorScheme.error
                        label.photoCount < RECOMMENDED_PHOTOS -> Color(0xFFF9A825)
                        else -> MaterialTheme.colorScheme.onSurface
                    }
                    val hint = when {
                        label.photoCount < MIN_PHOTOS -> " (needs $MIN_PHOTOS)"
                        label.photoCount < RECOMMENDED_PHOTOS -> " (aim for $RECOMMENDED_PHOTOS)"
                        else -> ""
                    }
                    Text("${label.friendlyName}: ${label.photoCount}$hint", color = color)
                }
            }
        }

        Button(onClick = vm::train, enabled = state.canTrain, modifier = Modifier.fillMaxWidth()) { Text("Train") }

        state.status?.let { status ->
            if (status.running) {
                val progress = if (status.epochs > 0) status.epoch.toFloat() / status.epochs else 0f
                LinearProgressIndicator(progress = { progress }, modifier = Modifier.fillMaxWidth())
                Text(status.message)
            }
        }
        state.message?.let { Text(it) }

        state.meta?.let { meta ->
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text("Installed model v${meta.version}", style = MaterialTheme.typography.titleMedium)
                    Text("Validation accuracy: ${pct(meta.valAccuracy)}")
                    meta.classes.forEach { cls ->
                        meta.perClassAccuracy[cls.id]?.let { Text("• ${cls.friendlyName}: ${pct(it)}") }
                    }
                    if (meta.confusions.isNotEmpty()) {
                        Text("Most confused", style = MaterialTheme.typography.titleSmall)
                        val names = meta.classes.associate { it.id to it.friendlyName }
                        meta.confusions.forEach { c ->
                            Text("• ${names[c.trueId] ?: c.trueId} seen as ${names[c.predicted] ?: c.predicted}: ${c.count}")
                        }
                    }
                }
            }
        }
    }
}
