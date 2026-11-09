package dev.pointremote.capture

import android.content.Context
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import dev.pointremote.PointRemoteApp
import dev.pointremote.net.ServerException
import java.io.File

class UploadWorker(context: Context, params: WorkerParameters) : CoroutineWorker(context, params) {
    override suspend fun doWork(): Result {
        val path = inputData.getString(KEY_PATH) ?: return Result.failure()
        val label = inputData.getString(KEY_LABEL) ?: return Result.failure()
        val root = File(applicationContext.filesDir, "pending").canonicalFile
        val file = File(path).canonicalFile
        if (file.parentFile?.parentFile != root || file.extension != "jpg") return Result.failure()
        val container = (applicationContext as PointRemoteApp).container
        if (!file.exists()) return Result.success()
        val server = container.server() ?: return Result.retry()
        return try {
            // Ok or LabelGone: either way this file is finished with.
            server.uploadPhoto(label, file)
            file.delete()
            Result.success()
        } catch (e: ServerException) {
            val code = e.code
            val permanent = code != null && code in 400..499 && code !in setOf(401, 408, 429)
            if (permanent) {
                file.delete()
                Result.failure()
            } else {
                Result.retry()
            }
        } finally {
            container.photoQueue.refresh()
        }
    }

    companion object {
        const val KEY_PATH = "path"
        const val KEY_LABEL = "label"
    }
}
