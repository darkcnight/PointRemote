package dev.pointremote.net

import org.junit.Assert.assertEquals
import org.junit.Test

class EntityFilterTest {
    private val all = listOf(
        HaEntity("climate.main_bedroom_aircon", "Main Bedroom Aircon", "cool"),
        HaEntity("light.main_bedroom_lights", "Main Bedroom Lights", "off"),
        HaEntity("light.kitchen_tracklights", "Kitchen Tracklights", "on"),
        HaEntity("switch.kitchen_plug_led", "Kitchen Plug LED", "on", clutter = true),
    )

    private fun ids(query: String, showAll: Boolean = false) =
        filterEntities(all, query, showAll).map { it.entityId }

    @Test fun hidesClutterUnlessShowAll() {
        assertEquals(3, ids("").size)
        assertEquals(4, ids("", showAll = true).size)
    }

    @Test fun everyWordMustMatchNameOrIdIgnoringCase() {
        assertEquals(listOf("climate.main_bedroom_aircon"), ids("AIRCON"))
        assertEquals(listOf("climate.main_bedroom_aircon", "light.main_bedroom_lights"), ids("main bed"))
        assertEquals(listOf("climate.main_bedroom_aircon"), ids("climate."))
        assertEquals(emptyList<String>(), ids("kitchen led"))
        assertEquals(listOf("switch.kitchen_plug_led"), ids("kitchen led", showAll = true))
    }

    @Test fun blankQueryKeepsOrder() {
        assertEquals(ids(""), ids("   "))
    }
}
