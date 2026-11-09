package dev.pointremote.launch

import android.content.Context
import android.content.Intent
import androidx.compose.ui.unit.dp
import androidx.glance.GlanceId
import androidx.glance.GlanceModifier
import androidx.glance.Image
import androidx.glance.ImageProvider
import androidx.glance.appwidget.action.actionStartActivity
import androidx.glance.action.clickable
import androidx.glance.appwidget.GlanceAppWidget
import androidx.glance.appwidget.GlanceAppWidgetReceiver
import androidx.glance.appwidget.provideContent
import androidx.glance.appwidget.updateAll
import androidx.glance.background
import androidx.glance.layout.Alignment
import androidx.glance.layout.Box
import androidx.glance.layout.fillMaxSize
import androidx.glance.layout.size
import dev.pointremote.R
import dev.pointremote.quick.QuickPointActivity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

/**
 * A round 1x1 button: smoky glass tinted with the wallpaper's accent colour (Android 12+)
 * and a viewfinder reticle. The tint comes from system colour resources, so the launcher
 * updates it when the wallpaper changes without the app redrawing.
 */
class PointWidget : GlanceAppWidget() {
    override suspend fun provideGlance(context: Context, id: GlanceId) {
        provideContent {
            Box(GlanceModifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Box(
                    modifier = GlanceModifier.size(58.dp)
                        .background(ImageProvider(R.drawable.widget_glass))
                        .clickable(actionStartActivity(QuickPointActivity.intent(context))),
                    contentAlignment = Alignment.Center,
                ) {
                    Image(
                        provider = ImageProvider(R.drawable.ic_reticle),
                        contentDescription = context.getString(R.string.widget_point),
                        modifier = GlanceModifier.size(30.dp),
                    )
                }
            }
        }
    }
}

class PointWidgetReceiver : GlanceAppWidgetReceiver() {
    override val glanceAppWidget: GlanceAppWidget = PointWidget()

    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action == Intent.ACTION_MY_PACKAGE_REPLACED) {
            // A new APK renumbers resources, so the launcher's saved copy of the widget
            // can point at ones that no longer exist ("Can't load widget"). Redraw it.
            val pending = goAsync()
            CoroutineScope(Dispatchers.Default).launch {
                try { refreshPointWidgets(context) } finally { pending.finish() }
            }
            return
        }
        super.onReceive(context, intent)
    }
}

/** Redraws every placed widget; safe to call when none are placed. */
suspend fun refreshPointWidgets(context: Context) {
    runCatching { PointWidget().updateAll(context) }
}
