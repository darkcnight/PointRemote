package dev.pointremote

import android.app.Application
import android.content.Context
import dev.pointremote.capture.PhotoQueue
import dev.pointremote.model.ModelStore
import dev.pointremote.net.HaClient
import dev.pointremote.net.ServerClient
import dev.pointremote.settings.SettingsRepository
import okhttp3.OkHttpClient
import java.io.File
import java.util.concurrent.TimeUnit

class AppContainer(context: Context) {
    val settings = SettingsRepository(context)
    val modelStore = ModelStore(File(context.filesDir, "model"))
    val photoQueue = PhotoQueue(context)
    val http: OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(5, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .build()

    fun server(): ServerClient? = settings.settings.value.let {
        if (it.serverConfigured) ServerClient(it.serverUrl, it.serverKey, http) else null
    }

    fun ha(): HaClient? = settings.settings.value.let {
        if (it.haConfigured) HaClient(it.haUrl, it.haToken, http) else null
    }
}

class PointRemoteApp : Application() {
    lateinit var container: AppContainer
        private set

    override fun onCreate() {
        super.onCreate()
        container = AppContainer(this)
    }
}
