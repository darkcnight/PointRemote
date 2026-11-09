package dev.pointremote.ui

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import dev.pointremote.AppContainer
import dev.pointremote.ui.capture.CaptureScreen
import dev.pointremote.ui.point.PointScreen
import dev.pointremote.ui.settings.SettingsScreen
import dev.pointremote.ui.train.TrainScreen

enum class Screen(val label: String, val glyph: String) {
    Point("Point", "◎"),
    Capture("Capture", "◉"),
    Train("Train", "▲"),
    Settings("Settings", "⚙"),
}

@Composable
fun AppRoot(container: AppContainer) {
    var screen by rememberSaveable { mutableStateOf(Screen.Point) }
    MaterialTheme(colorScheme = if (isSystemInDarkTheme()) darkColorScheme() else lightColorScheme()) {
        Scaffold(
            bottomBar = {
                NavigationBar {
                    Screen.entries.forEach { s ->
                        NavigationBarItem(
                            selected = screen == s,
                            onClick = { screen = s },
                            icon = { Text(s.glyph) },
                            label = { Text(s.label) },
                        )
                    }
                }
            },
        ) { padding ->
            Box(Modifier.padding(padding)) {
                when (screen) {
                    Screen.Point -> PointScreen(container, onOpenCapture = { screen = Screen.Capture })
                    Screen.Capture -> CaptureScreen(container)
                    Screen.Train -> TrainScreen(container)
                    Screen.Settings -> SettingsScreen(container)
                }
            }
        }
    }
}
