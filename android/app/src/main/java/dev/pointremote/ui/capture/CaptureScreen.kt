@file:OptIn(ExperimentalMaterial3Api::class)

package dev.pointremote.ui.capture

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Checkbox
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import dev.pointremote.AppContainer
import dev.pointremote.camera.CameraPreview
import dev.pointremote.camera.WithCameraPermission
import dev.pointremote.net.HaEntity
import dev.pointremote.net.ServerLabel
import dev.pointremote.net.filterEntities

@Composable
fun CaptureScreen(container: AppContainer) {
    val vm: CaptureViewModel = viewModel { CaptureViewModel(container) }
    val state by vm.state.collectAsStateWithLifecycle()
    val pending by vm.pending.collectAsStateWithLifecycle()
    LaunchedEffect(Unit) { vm.refresh() }
    // Plain remember (not saveable): the tag sheet opens every time you come to Capture,
    // so you always confirm what you're about to photograph.
    var showTags by remember { mutableStateOf(true) }
    val current = state.labels.firstOrNull { it.id == state.selected } ?: UNKNOWN_LABEL

    WithCameraPermission {
        Box(Modifier.fillMaxSize()) {
            CameraPreview(onFrame = vm::analyze, modifier = Modifier.fillMaxSize())

            Column(
                Modifier.align(Alignment.TopCenter).fillMaxWidth()
                    .background(MaterialTheme.colorScheme.surface.copy(alpha = 0.85f))
                    .padding(vertical = 8.dp),
            ) {
                Surface(
                    onClick = { showTags = true },
                    enabled = !state.capturing,
                    modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp),
                    shape = RoundedCornerShape(12.dp),
                    color = MaterialTheme.colorScheme.primaryContainer,
                ) {
                    Row(Modifier.padding(horizontal = 16.dp, vertical = 12.dp), verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f)) {
                            Text("Photos go to", style = MaterialTheme.typography.labelMedium)
                            Text(current.friendlyName, style = MaterialTheme.typography.titleLarge)
                            Text(countText(current, pending), style = MaterialTheme.typography.bodySmall)
                        }
                        Text("Change ▾", style = MaterialTheme.typography.labelLarge)
                    }
                }
                state.message?.let {
                    Text(it, modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp), color = MaterialTheme.colorScheme.error)
                }
            }

            Column(
                Modifier.align(Alignment.BottomCenter).padding(bottom = 32.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Surface(shape = RoundedCornerShape(12.dp), color = MaterialTheme.colorScheme.surface.copy(alpha = 0.85f)) {
                    Text(
                        if (state.capturing) "${current.friendlyName}: ${state.sessionCount}" else "Hold to capture · ${state.sessionCount} last burst",
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
                    )
                }
                Box(
                    Modifier.size(88.dp).clip(CircleShape)
                        .background(if (state.capturing) Color(0xFFE53935) else Color.White)
                        .pointerInput(Unit) {
                            detectTapGestures(onPress = {
                                vm.setCapturing(true)
                                tryAwaitRelease()
                                vm.setCapturing(false)
                            })
                        },
                )
            }
        }
    }

    if (showTags && state.entities == null) {
        TagSheet(
            labels = state.labels,
            selected = state.selected,
            pending = pending,
            onPick = { vm.select(it); showTags = false },
            onAddDevice = { showTags = false; vm.startNewLabel() },
            onDismiss = { showTags = false },
        )
    }
    state.entities?.let { entities ->
        EntityPicker(
            entities,
            onPick = vm::createLabel,
            onCancel = { vm.cancelNewLabel(); showTags = true },
        )
    }
}

private fun countText(label: ServerLabel, pending: Map<String, Int>): String {
    val waiting = pending[label.id] ?: 0
    val photos = if (label.photoCount == 1) "1 photo" else "${label.photoCount} photos"
    return if (waiting > 0) "$photos · $waiting waiting to upload" else photos
}

@Composable
private fun TagSheet(
    labels: List<ServerLabel>,
    selected: String,
    pending: Map<String, Int>,
    onPick: (String) -> Unit,
    onAddDevice: () -> Unit,
    onDismiss: () -> Unit,
) {
    ModalBottomSheet(onDismissRequest = onDismiss, sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)) {
        Text(
            "What are you photographing?",
            style = MaterialTheme.typography.titleLarge,
            modifier = Modifier.padding(horizontal = 24.dp, vertical = 8.dp),
        )
        LazyColumn(Modifier.fillMaxWidth(), contentPadding = PaddingValues(bottom = 24.dp)) {
            items(labels, key = { it.id }) { label ->
                Row(
                    Modifier.fillMaxWidth().clickable { onPick(label.id) }.padding(horizontal = 12.dp, vertical = 4.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    RadioButton(selected = label.id == selected, onClick = { onPick(label.id) })
                    Column(Modifier.padding(start = 4.dp)) {
                        Text(label.friendlyName, style = MaterialTheme.typography.bodyLarge)
                        Text(
                            if (label.id == UNKNOWN_LABEL.id) "Anything that isn't one of your devices · ${countText(label, pending)}"
                            else countText(label, pending),
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                }
            }
            item {
                Row(
                    Modifier.fillMaxWidth().clickable(onClick = onAddDevice).padding(horizontal = 24.dp, vertical = 16.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text("+  Add a device from Home Assistant", style = MaterialTheme.typography.bodyLarge, color = MaterialTheme.colorScheme.primary)
                }
            }
        }
    }
}

@Composable
private fun EntityPicker(entities: List<HaEntity>, onPick: (HaEntity) -> Unit, onCancel: () -> Unit) {
    var query by rememberSaveable { mutableStateOf("") }
    var showAll by rememberSaveable { mutableStateOf(false) }
    val shown = remember(entities, query, showAll) { filterEntities(entities, query, showAll) }
    val hiddenCount = remember(entities) { entities.count { it.clutter } }
    AlertDialog(
        onDismissRequest = onCancel,
        title = { Text("Pick a device") },
        text = {
            Column {
                OutlinedTextField(
                    value = query,
                    onValueChange = { query = it },
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true,
                    placeholder = { Text("Search, e.g. aircon or bedroom") },
                )
                if (hiddenCount > 0) {
                    Row(
                        Modifier.fillMaxWidth().clickable { showAll = !showAll }.padding(vertical = 4.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Checkbox(checked = showAll, onCheckedChange = { showAll = it })
                        Text("Show $hiddenCount settings, hidden and offline entities", style = MaterialTheme.typography.bodySmall)
                    }
                }
                if (shown.isEmpty()) {
                    Text(
                        if (entities.isEmpty()) "No controllable entities left to add." else "Nothing matches.",
                        Modifier.padding(top = 12.dp),
                    )
                } else {
                    LazyColumn(Modifier.heightIn(max = 400.dp)) {
                        items(shown, key = { it.entityId }) { entity ->
                            Column(Modifier.fillMaxWidth().clickable { onPick(entity) }.padding(vertical = 10.dp)) {
                                Text(entity.friendlyName, style = MaterialTheme.typography.bodyLarge)
                                Text("${entity.entityId} · ${entity.state}", style = MaterialTheme.typography.bodySmall)
                            }
                        }
                    }
                }
            }
        },
        confirmButton = {},
        dismissButton = { TextButton(onClick = onCancel) { Text("Cancel") } },
    )
}
