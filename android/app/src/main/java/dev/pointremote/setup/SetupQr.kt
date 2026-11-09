package dev.pointremote.setup

import java.net.URI
import java.net.URLDecoder

/** What the PC app's QR code carries: the training server address and its access key. */
data class ServerSetup(val url: String, val key: String) {
    val host: String get() = URI(url).host

    /** True for addresses on a home network (192.168.x, 10.x, 172.16-31.x, .local names). */
    val isHomeNetwork: Boolean get() = isPrivateHost(host)
}

private val KEY_RE = Regex("^[A-Za-z0-9_\\-+/=.~]{8,200}$")

internal fun isPrivateHost(host: String): Boolean {
    if (host.endsWith(".local") || host.endsWith(".lan") || host.endsWith(".home.arpa")) return true
    val parts = host.split('.').map { it.toIntOrNull() ?: return false }
    if (parts.size != 4 || parts.any { it !in 0..255 }) return false
    val (a, b) = parts
    return a == 10 || (a == 192 && b == 168) || (a == 172 && b in 16..31) || (a == 169 && b == 254)
}

/** Reads `pointremote://setup?url=…&key=…`; anything else (any other QR code) gives null. */
fun parseSetupQr(text: String): ServerSetup? {
    val uri = runCatching { URI(text.trim()) }.getOrNull() ?: return null
    if (uri.scheme != "pointremote" || uri.host != "setup") return null
    val params = (uri.rawQuery ?: return null).split('&').mapNotNull { part ->
        val i = part.indexOf('=')
        if (i <= 0) null else part.substring(0, i) to URLDecoder.decode(part.substring(i + 1), Charsets.UTF_8)
    }.toMap()
    val url = params["url"]?.trimEnd('/') ?: return null
    val key = params["key"] ?: return null
    val server = runCatching { URI(url) }.getOrNull() ?: return null
    if (server.scheme !in setOf("http", "https") || server.host.isNullOrEmpty() || !KEY_RE.matches(key)) return null
    return ServerSetup(url, key)
}
