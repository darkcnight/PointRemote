package dev.pointremote.capture

import android.content.Context
import androidx.work.BackoffPolicy
import androidx.work.Constraints
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.workDataOf
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.File
import java.util.concurrent.TimeUnit

class PhotoQueue(private val context: Context) {
    private val pending = PendingPhotos(File(context.filesDir, "pending"))
    private val _counts = MutableStateFlow(pending.counts())
    val counts: StateFlow<Map<String, Int>> = _counts.asStateFlow()

    fun enqueue(labelId: String, jpeg: ByteArray) {
        val file = pending.add(labelId, jpeg)
        val request = OneTimeWorkRequestBuilder<UploadWorker>()
            .setInputData(workDataOf(UploadWorker.KEY_PATH to file.path, UploadWorker.KEY_LABEL to labelId))
            .setConstraints(Constraints.Builder().setRequiredNetworkType(NetworkType.CONNECTED).build())
            .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, 15, TimeUnit.SECONDS)
            .build()
        WorkManager.getInstance(context).enqueue(request)
        refresh()
    }

    fun refresh() {
        _counts.value = pending.counts()
    }
}
