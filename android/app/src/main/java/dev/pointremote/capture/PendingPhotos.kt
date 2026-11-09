package dev.pointremote.capture

import java.io.File
import java.util.UUID

/** Photos waiting to be uploaded, stored as `root/<label>/<uuid>.jpg`. */
class PendingPhotos(private val root: File) {
    fun add(labelId: String, jpeg: ByteArray): File {
        require(SAFE_ID.matches(labelId)) { "unsafe label id: $labelId" }
        val dir = File(root, labelId).apply { mkdirs() }
        return File(dir, "${UUID.randomUUID()}.jpg").apply { writeBytes(jpeg) }
    }

    fun counts(): Map<String, Int> =
        (root.listFiles() ?: emptyArray())
            .filter { it.isDirectory }
            .associate { dir -> dir.name to (dir.listFiles { f -> f.extension == "jpg" }?.size ?: 0) }
            .filterValues { it > 0 }

    private companion object {
        val SAFE_ID = Regex("^[a-z0-9_]{1,128}$")
    }
}
