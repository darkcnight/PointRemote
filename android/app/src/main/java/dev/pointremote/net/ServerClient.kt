package dev.pointremote.net

import dev.pointremote.model.ModelMeta
import dev.pointremote.model.ModelSource
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import java.io.File
import java.io.IOException

class ServerException(message: String, val code: Int? = null) : Exception(message)

@Serializable
data class Health(
    val ok: Boolean,
    val training: Boolean,
    @SerialName("latest_version") val latestVersion: Int? = null,
)

@Serializable
data class ServerLabel(
    val id: String,
    @SerialName("entity_id") val entityId: String? = null,
    @SerialName("friendly_name") val friendlyName: String,
    @SerialName("photo_count") val photoCount: Int = 0,
)

@Serializable
data class TrainStatus(
    val state: String,
    val epoch: Int = 0,
    val epochs: Int = 0,
    val message: String = "",
    val version: Int? = null,
) {
    val running: Boolean get() = state == "running"
}

@Serializable
private data class NewLabelBody(
    @SerialName("entity_id") val entityId: String,
    @SerialName("friendly_name") val friendlyName: String,
)

sealed interface UploadResult {
    data class Ok(val count: Int) : UploadResult
    data object LabelGone : UploadResult
}

internal fun serverErrorMessage(code: Int, body: String): String {
    val detail = runCatching { AppJson.parseToJsonElement(body).jsonObject["detail"] }.getOrNull()
    val blockers = (detail as? JsonObject)?.get("blockers")?.jsonArray?.map { it.jsonPrimitive.content }
    return when {
        code == 401 -> "Server rejected the access key"
        blockers != null -> "Not ready: " + blockers.joinToString("; ")
        detail is JsonPrimitive -> detail.content
        else -> "Server error $code"
    }
}

class ServerClient(
    baseUrl: String,
    private val apiKey: String,
    private val http: OkHttpClient,
) : ModelSource {
    private val base = baseUrl.trimEnd('/')

    private fun request(path: String) = Request.Builder().url(base + path).header("X-Api-Key", apiKey)

    private suspend fun <T> call(request: Request, handle: (Response) -> T): T = withContext(Dispatchers.IO) {
        try {
            http.newCall(request).execute().use(handle)
        } catch (e: IOException) {
            throw ServerException("Can't reach the PC server (${e.message})")
        }
    }

    private fun Response.bodyOrThrow(): String {
        val text = body.string()
        if (!isSuccessful) throw ServerException(serverErrorMessage(code, text), code)
        return text
    }

    suspend fun health(): Health =
        call(request("/health").build()) { AppJson.decodeFromString(it.bodyOrThrow()) }

    suspend fun labels(): List<ServerLabel> =
        call(request("/labels").build()) { AppJson.decodeFromString(it.bodyOrThrow()) }

    suspend fun createLabel(entityId: String, friendlyName: String): ServerLabel {
        val body = AppJson.encodeToString(NewLabelBody.serializer(), NewLabelBody(entityId, friendlyName))
        return call(request("/labels").post(body.toRequestBody(JSON_TYPE)).build()) {
            AppJson.decodeFromString(it.bodyOrThrow())
        }
    }

    suspend fun uploadPhoto(labelId: String, jpeg: File): UploadResult {
        if (!LABEL_ID_RE.matches(labelId)) throw ServerException("Invalid label id \"$labelId\"")
        val body = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("file", jpeg.name, jpeg.asRequestBody(JPEG_TYPE))
            .build()
        return call(request("/photos/$labelId").post(body).build()) { r ->
            if (r.code == 404) {
                UploadResult.LabelGone
            } else {
                val json = AppJson.parseToJsonElement(r.bodyOrThrow()).jsonObject
                UploadResult.Ok(json.getValue("count").jsonPrimitive.int)
            }
        }
    }

    suspend fun startTraining(): TrainStatus =
        call(request("/train").post(ByteArray(0).toRequestBody()).build()) {
            AppJson.decodeFromString(it.bodyOrThrow())
        }

    suspend fun trainStatus(): TrainStatus =
        call(request("/train/status").build()) { AppJson.decodeFromString(it.bodyOrThrow()) }

    override suspend fun latestMeta(): ModelMeta? =
        call(request("/model/latest/meta").build()) { r ->
            if (r.code == 404) null else AppJson.decodeFromString<ModelMeta>(r.bodyOrThrow())
        }

    override suspend fun downloadModel(dest: File, maxBytes: Long) {
        call(request("/model/latest/file").build()) { r ->
            if (!r.isSuccessful) throw ServerException(serverErrorMessage(r.code, r.body.string()), r.code)
            val declared = r.body.contentLength()
            if (declared > maxBytes) throw ServerException("Model larger than expected")
            dest.outputStream().use { out ->
                val input = r.body.byteStream()
                val buffer = ByteArray(64 * 1024)
                var total = 0L
                while (true) {
                    val n = input.read(buffer)
                    if (n < 0) break
                    total += n
                    if (total > maxBytes) throw ServerException("Model larger than expected")
                    out.write(buffer, 0, n)
                }
            }
        }
    }

    private companion object {
        val JSON_TYPE = "application/json".toMediaType()
        val JPEG_TYPE = "image/jpeg".toMediaType()
        val LABEL_ID_RE = Regex("^[a-z0-9_]{1,128}$")
    }
}
