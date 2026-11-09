package dev.pointremote.net

import kotlinx.coroutines.test.runTest
import okhttp3.OkHttpClient
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test

class HaClientTest {
    private val server = MockWebServer()
    private lateinit var ha: HaClient

    @Before fun setUp() {
        server.start()
        ha = HaClient(server.url("/").toString(), "tok", OkHttpClient())
    }

    @After fun tearDown() = server.shutdown()

    private fun respond(code: Int, body: String) =
        server.enqueue(MockResponse().setResponseCode(code).setBody(body))

    private suspend fun failure(block: suspend () -> Unit): HaException =
        runCatching { block() }.exceptionOrNull() as HaException

    @Test fun toggleReturnsNewStateFromResponse() = runTest {
        respond(200, """[{"entity_id":"light.kitchen","state":"on","attributes":{}}]""")
        assertEquals("on", ha.toggle("light.kitchen"))
        val req = server.takeRequest()
        assertEquals("POST", req.method)
        assertEquals("/api/services/homeassistant/toggle", req.path)
        assertEquals("Bearer tok", req.getHeader("Authorization"))
        assertEquals("""{"entity_id":"light.kitchen"}""", req.body.readUtf8())
    }

    @Test fun toggleFallsBackToStateLookup() = runTest {
        respond(200, "[]")
        respond(200, """{"entity_id":"cover.big","state":"opening","attributes":{}}""")
        assertEquals("opening", ha.toggle("cover.big"))
        server.takeRequest()
        assertEquals("/api/states/cover.big", server.takeRequest().path)
    }

    @Test fun unauthorized() = runTest {
        respond(401, "401: Unauthorized")
        assertEquals(HaException.Kind.Unauthorized, failure { ha.toggle("light.kitchen") }.kind)
    }

    @Test fun httpError() = runTest {
        respond(500, "boom")
        assertEquals(HaException.Kind.Http, failure { ha.toggle("light.kitchen") }.kind)
    }

    @Test fun network() = runTest {
        server.shutdown()
        assertEquals(HaException.Kind.Network, failure { ha.entities() }.kind)
    }

    @Test fun refusesNonControllableEntities() = runTest {
        for (bad in listOf("sensor.temp", "light.a/../../api/config", "light.kitchen?x=1", "lock.front_door")) {
            assertEquals(HaException.Kind.Invalid, failure { ha.toggle(bad) }.kind)
        }
        assertEquals(0, server.requestCount)
    }

    @Test fun climateIsControllable() = runTest {
        respond(200, """[{"entity_id":"climate.bedroom_aircon","state":"off","attributes":{}}]""")
        assertEquals("off", ha.toggle("climate.bedroom_aircon"))
        assertEquals(true, isControllableEntity("climate.bedroom_aircon"))
    }

    /** Fake HA websocket: asks for auth, checks the token, then answers the registry request. */
    private fun registry(entities: String) = server.enqueue(MockResponse().withWebSocketUpgrade(object : WebSocketListener() {
        override fun onOpen(webSocket: WebSocket, response: Response) {
            webSocket.send("""{"type":"auth_required"}""")
        }
        override fun onMessage(webSocket: WebSocket, text: String) {
            when {
                text.contains("\"type\":\"auth\"") && text.contains("\"access_token\":\"tok\"") ->
                    webSocket.send("""{"type":"auth_ok"}""")
                text.contains("config/entity_registry/list_for_display") ->
                    webSocket.send("""{"id":1,"type":"result","success":true,"result":{"entity_categories":{"0":"config","1":"diagnostic"},"entities":$entities}}""")
                else -> webSocket.send("""{"type":"auth_invalid"}""")
            }
        }
        override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
            webSocket.close(1000, null)
        }
    }))

    @Test fun entitiesMarksClutterFromRegistryAndState() = runTest {
        respond(200, """[
            {"entity_id":"light.kitchen","state":"off","attributes":{"friendly_name":"Kitchen"}},
            {"entity_id":"switch.plug_led","state":"on","attributes":{"friendly_name":"Plug LED"}},
            {"entity_id":"switch.old","state":"on","attributes":{"friendly_name":"Old"}},
            {"entity_id":"switch.gone","state":"unavailable","attributes":{"friendly_name":"Gone"}},
            {"entity_id":"input_boolean.flag","state":"off","attributes":{"friendly_name":"Flag"}},
            {"entity_id":"climate.aircon","state":"cool","attributes":{"friendly_name":"Aircon"}}
        ]""")
        registry("""[
            {"ei":"light.kitchen"},
            {"ei":"switch.plug_led","ec":0},
            {"ei":"switch.old","hb":true},
            {"ei":"sensor.x","ec":1}
        ]""")
        val byId = ha.entities().associate { it.entityId to it.clutter }
        assertEquals(
            mapOf(
                "climate.aircon" to false,
                "input_boolean.flag" to true,
                "light.kitchen" to false,
                "switch.gone" to true,
                "switch.old" to true,
                "switch.plug_led" to true,
            ),
            byId,
        )
    }

    @Test fun entitiesStillWorkWithoutRegistry() = runTest {
        respond(200, """[{"entity_id":"switch.plug","state":"on","attributes":{"friendly_name":"Plug"}}]""")
        respond(404, "no websocket")
        assertEquals(listOf(HaEntity("switch.plug", "Plug", "on", clutter = false)), ha.entities())
    }

    @Test fun entitiesFiltersAndSorts() = runTest {
        respond(200, """[
            {"entity_id":"sensor.temp","state":"21","attributes":{"friendly_name":"Temp"}},
            {"entity_id":"light.kitchen","state":"off","attributes":{"friendly_name":"kitchen lights"}},
            {"entity_id":"cover.big","state":"open","attributes":{"friendly_name":"Big shutter"}},
            {"entity_id":"switch.tv","state":"on","attributes":{}}
        ]""")
        respond(404, "no websocket")
        assertEquals(
            listOf(
                HaEntity("cover.big", "Big shutter", "open"),
                HaEntity("light.kitchen", "kitchen lights", "off"),
                HaEntity("switch.tv", "switch.tv", "on"),
            ),
            ha.entities(),
        )
    }
}
