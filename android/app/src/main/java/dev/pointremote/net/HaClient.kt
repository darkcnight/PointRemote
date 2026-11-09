package dev.pointremote.net

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import java.io.IOException
import kotlin.coroutines.resume

class HaException(message: String, val kind: Kind) : Exception(message) {
    enum class Kind { Unauthorized, Network, Http, Invalid }
}

/**
 * [clutter] marks entities the device picker hides by default: settings/diagnostic entities,
 * ones hidden in Home Assistant, unavailable ones, and input_boolean helpers.
 */
data class HaEntity(
    val entityId: String,
    val friendlyName: String,
    val state: String,
    val clutter: Boolean = false,
)

val CONTROLLABLE_DOMAINS = setOf("light", "switch", "cover", "fan", "media_player", "input_boolean", "climate")

private val CLUTTER_STATES = setOf("unavailable", "unknown")
private const val REGISTRY_TIMEOUT_MS = 5_000L

private val ENTITY_ID_RE = Regex("^[a-z_]+\\.[a-z0-9_]+$")

/** True for well-formed ids in a domain this app is allowed to control. */
fun isControllableEntity(entityId: String): Boolean =
    ENTITY_ID_RE.matches(entityId) && entityId.substringBefore('.') in CONTROLLABLE_DOMAINS

private fun requireControllable(entityId: String) {
    if (!isControllableEntity(entityId)) {
        throw HaException("Refusing to control \"$entityId\"", HaException.Kind.Invalid)
    }
}

private fun stateIn(json: JsonElement, entityId: String): String? =
    (json as? JsonArray)
        ?.firstOrNull { it.jsonObject["entity_id"]?.jsonPrimitive?.content == entityId }
        ?.jsonObject?.get("state")?.jsonPrimitive?.content

class HaClient(baseUrl: String, private val token: String, private val http: OkHttpClient) {
    private val base = baseUrl.trimEnd('/')

    private fun request(path: String) =
        Request.Builder().url(base + path).header("Authorization", "Bearer $token")

    private suspend fun <T> call(request: Request, handle: (String) -> T): T = withContext(Dispatchers.IO) {
        try {
            http.newCall(request).execute().use { r ->
                val body = r.body.string()
                when {
                    r.code == 401 -> throw HaException("Home Assistant rejected the token", HaException.Kind.Unauthorized)
                    !r.isSuccessful -> throw HaException("Home Assistant error ${r.code}", HaException.Kind.Http)
                    else -> handle(body)
                }
            }
        } catch (e: IOException) {
            throw HaException("Can't reach Home Assistant (${e.message})", HaException.Kind.Network)
        }
    }

    suspend fun toggle(entityId: String): String? {
        requireControllable(entityId)
        val payload = buildJsonObject { put("entity_id", entityId) }.toString()
        val post = request("/api/services/homeassistant/toggle").post(payload.toRequestBody(JSON_TYPE)).build()
        val changed = call(post) { body -> stateIn(AppJson.parseToJsonElement(body), entityId) }
        return changed ?: state(entityId)
    }

    suspend fun state(entityId: String): String? {
        requireControllable(entityId)
        return call(request("/api/states/$entityId").build()) { body ->
            AppJson.parseToJsonElement(body).jsonObject["state"]?.jsonPrimitive?.content
        }
    }

    suspend fun entities(): List<HaEntity> {
        val all = call(request("/api/states").build()) { body ->
            AppJson.parseToJsonElement(body).jsonArray.mapNotNull { element ->
                val o = element.jsonObject
                val id = o["entity_id"]?.jsonPrimitive?.content ?: return@mapNotNull null
                if (id.substringBefore('.') !in CONTROLLABLE_DOMAINS) return@mapNotNull null
                val name = o["attributes"]?.jsonObject?.get("friendly_name")?.jsonPrimitive?.content ?: id
                HaEntity(id, name, o["state"]?.jsonPrimitive?.content ?: "unknown")
            }.sortedBy { it.friendlyName.lowercase() }
        }
        // The REST API doesn't say which entities are settings or hidden; the websocket
        // registry does. If it can't be read, only the state-based rules apply.
        val registryClutter = registryClutter().orEmpty()
        return all.map { e ->
            e.copy(
                clutter = e.entityId in registryClutter ||
                    e.state in CLUTTER_STATES ||
                    e.entityId.startsWith("input_boolean."),
            )
        }
    }

    /** Entity ids that have an entity category (config/diagnostic) or are hidden, or null if unavailable. */
    private suspend fun registryClutter(): Set<String>? = withContext(Dispatchers.IO) {
        withTimeoutOrNull(REGISTRY_TIMEOUT_MS) {
            suspendCancellableCoroutine { cont ->
                fun finish(ws: WebSocket, result: Set<String>?) {
                    ws.close(1000, null)
                    if (cont.isActive) cont.resume(result)
                }
                val ws = http.newWebSocket(Request.Builder().url("$base/api/websocket").build(), object : WebSocketListener() {
                    override fun onMessage(webSocket: WebSocket, text: String) {
                        val msg = runCatching { AppJson.parseToJsonElement(text).jsonObject }.getOrNull()
                            ?: return finish(webSocket, null)
                        when (msg["type"]?.jsonPrimitive?.content) {
                            "auth_required" -> webSocket.send(
                                buildJsonObject { put("type", "auth"); put("access_token", token) }.toString(),
                            )
                            "auth_ok" -> webSocket.send(
                                buildJsonObject { put("id", 1); put("type", "config/entity_registry/list_for_display") }.toString(),
                            )
                            "result" -> finish(webSocket, runCatching { parseRegistryClutter(msg) }.getOrNull())
                            else -> finish(webSocket, null)
                        }
                    }

                    override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                        if (cont.isActive) cont.resume(null)
                    }

                    override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                        if (cont.isActive) cont.resume(null)
                    }
                })
                cont.invokeOnCancellation { ws.cancel() }
            }
        }
    }

    private fun parseRegistryClutter(msg: JsonObject): Set<String>? {
        if (msg["success"]?.jsonPrimitive?.content != "true") return null
        val entities = msg["result"]?.jsonObject?.get("entities")?.jsonArray ?: return null
        return entities.mapNotNull { e ->
            val o = e.jsonObject
            val id = o["ei"]?.jsonPrimitive?.content ?: return@mapNotNull null
            val categorised = o["ec"]?.let { it !is JsonNull } == true
            val hidden = o["hb"]?.jsonPrimitive?.content == "true"
            id.takeIf { categorised || hidden }
        }.toSet()
    }

    private companion object {
        val JSON_TYPE = "application/json".toMediaType()
    }
}
