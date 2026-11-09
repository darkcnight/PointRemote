package dev.pointremote.model

import dev.pointremote.net.AppJson
import dev.pointremote.net.isControllableEntity
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException
import java.security.MessageDigest

interface ModelSource {
    suspend fun latestMeta(): ModelMeta?
    /** Must fail (and may leave a partial file) once more than [maxBytes] arrive. */
    suspend fun downloadModel(dest: File, maxBytes: Long)
}

class CorruptModelException(message: String) : Exception(message)

sealed interface UpdateResult {
    data object NoModel : UpdateResult
    data object UpToDate : UpdateResult
    data class Updated(val meta: ModelMeta) : UpdateResult
}

fun sha256Hex(bytes: ByteArray): String =
    MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

fun sha256Hex(file: File): String {
    val digest = MessageDigest.getInstance("SHA-256")
    file.inputStream().use { input ->
        val buffer = ByteArray(64 * 1024)
        while (true) {
            val n = input.read(buffer)
            if (n < 0) break
            digest.update(buffer, 0, n)
        }
    }
    return digest.digest().joinToString("") { "%02x".format(it) }
}

const val MAX_MODEL_BYTES = 200L * 1024 * 1024

/** Why this metadata can't be used, or null if it looks sane. */
internal fun ModelMeta.problem(): String? = when {
    classes.size < 2 -> "model needs at least one device class"
    classes[0].id != "unknown" -> "class 0 must be unknown"
    inputSize !in 32..1024 -> "unsupported input size $inputSize"
    normalization.mean.size != 3 || normalization.std.size != 3 || normalization.std.any { it <= 0f } ->
        "invalid normalization"
    sizeBytes !in 1..MAX_MODEL_BYTES -> "unexpected model size $sizeBytes"
    classes.drop(1).any { it.entityId?.let(::isControllableEntity) != true } -> "model refers to an entity this app can't control"
    else -> null
}

/** Keeps the installed model in `dir/current/{model.onnx,meta.json}`. */
class ModelStore(private val dir: File) {
    private val current get() = File(dir, "current")
    private val mutex = Mutex()
    private val _installed = MutableStateFlow(readInstalled())
    val installed: StateFlow<ModelMeta?> = _installed.asStateFlow()

    fun modelFile(): File? = File(current, MODEL).takeIf { it.isFile && _installed.value != null }

    private fun readInstalled(): ModelMeta? {
        val metaFile = File(current, META)
        if (!metaFile.isFile || !File(current, MODEL).isFile) return null
        val meta = runCatching { AppJson.decodeFromString<ModelMeta>(metaFile.readText()) }.getOrNull() ?: return null
        // Re-check the file on disk so a damaged or swapped model is never loaded.
        val model = File(current, MODEL)
        if (meta.problem() != null || model.length() != meta.sizeBytes || sha256Hex(model) != meta.sha256) return null
        return meta
    }

    suspend fun updateFrom(source: ModelSource): UpdateResult = mutex.withLock {
        val meta = source.latestMeta() ?: return UpdateResult.NoModel
        if (_installed.value?.version == meta.version) return UpdateResult.UpToDate
        meta.problem()?.let { throw CorruptModelException("Rejected model v${meta.version}: $it") }
        withContext(Dispatchers.IO) {
            dir.mkdirs()
            val incoming = File(dir, "incoming").apply { deleteRecursively(); mkdirs() }
            try {
                val model = File(incoming, MODEL)
                source.downloadModel(model, meta.sizeBytes)
                if (model.length() != meta.sizeBytes || sha256Hex(model) != meta.sha256) {
                    throw CorruptModelException("Downloaded model failed verification")
                }
                File(incoming, META).writeText(AppJson.encodeToString(ModelMeta.serializer(), meta))
                val old = File(dir, "old").apply { deleteRecursively() }
                if (current.exists() && !current.renameTo(old)) throw IOException("Could not move the current model aside")
                if (!incoming.renameTo(current)) {
                    old.renameTo(current)
                    throw IOException("Could not install the new model")
                }
                old.deleteRecursively()
            } finally {
                incoming.deleteRecursively()
            }
        }
        _installed.value = meta
        UpdateResult.Updated(meta)
    }

    private companion object {
        const val MODEL = "model.onnx"
        const val META = "meta.json"
    }
}
