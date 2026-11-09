package dev.pointremote.net

/**
 * Entities for the device picker: clutter is left out unless [showAll], and every word
 * of [query] must appear in the friendly name or entity id (case-insensitive).
 */
fun filterEntities(all: List<HaEntity>, query: String, showAll: Boolean): List<HaEntity> {
    val words = query.lowercase().split(Regex("\\s+")).filter { it.isNotEmpty() }
    return all.filter { e ->
        (showAll || !e.clutter) && words.all { w ->
            e.friendlyName.lowercase().contains(w) || e.entityId.contains(w)
        }
    }
}
