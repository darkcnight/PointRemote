package dev.pointremote.net

import kotlinx.coroutines.test.runTest
import okhttp3.OkHttpClient
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class ServerClientTest {
    @get:Rule val tmp = TemporaryFolder()
    private val server = MockWebServer()
    private lateinit var client: ServerClient

    @Before fun setUp() {
        server.start()
        client = ServerClient(server.url("/").toString(), "k", OkHttpClient())
    }

    @After fun tearDown() = server.shutdown()

    private fun respond(code: Int, body: String) =
        server.enqueue(MockResponse().setResponseCode(code).setBody(body))

    private suspend fun failure(block: suspend () -> Unit): ServerException =
        runCatching { block() }.exceptionOrNull() as ServerException

    @Test fun healthSendsKey() = runTest {
        respond(200, """{"ok":true,"training":false,"latest_version":3}""")
        assertEquals(Health(ok = true, training = false, latestVersion = 3), client.health())
        val req = server.takeRequest()
        assertEquals("/health", req.path)
        assertEquals("k", req.getHeader("X-Api-Key"))
    }

    @Test fun labels() = runTest {
        respond(200, """[{"id":"unknown","entity_id":null,"friendly_name":"Unknown","photo_count":4}]""")
        assertEquals(listOf(ServerLabel("unknown", null, "Unknown", 4)), client.labels())
    }

    @Test fun createLabelPostsJson() = runTest {
        respond(201, """{"id":"light_kitchen","entity_id":"light.kitchen","friendly_name":"Kitchen","photo_count":0}""")
        val label = client.createLabel("light.kitchen", "Kitchen")
        assertEquals("light_kitchen", label.id)
        val req = server.takeRequest()
        assertEquals("POST", req.method)
        assertEquals("/labels", req.path)
        val body = req.body.readUtf8()
        assertTrue(body.contains("\"entity_id\":\"light.kitchen\""))
        assertTrue(body.contains("\"friendly_name\":\"Kitchen\""))
    }

    @Test fun uploadPhoto() = runTest {
        val jpeg = tmp.newFile("a.jpg").apply { writeBytes(byteArrayOf(1, 2, 3)) }
        respond(201, """{"name":"x.jpg","count":7}""")
        assertEquals(UploadResult.Ok(7), client.uploadPhoto("light_kitchen", jpeg))
        val req = server.takeRequest()
        assertEquals("/photos/light_kitchen", req.path)
        assertTrue(req.getHeader("Content-Type")!!.startsWith("multipart/form-data"))
        assertTrue(req.body.readUtf8().contains("name=\"file\""))
        respond(404, """{"detail":"label not found"}""")
        assertEquals(UploadResult.LabelGone, client.uploadPhoto("gone", jpeg))
    }

    @Test fun uploadRejectsUnsafeLabelId() = runTest {
        val jpeg = tmp.newFile("b.jpg").apply { writeBytes(byteArrayOf(1)) }
        assertTrue(failure { client.uploadPhoto("../labels", jpeg) }.message!!.contains("Invalid label"))
        assertEquals(0, server.requestCount)
    }

    @Test fun downloadStopsAtSizeLimit() = runTest {
        server.enqueue(MockResponse().setResponseCode(200).setBody(okio.Buffer().write(ByteArray(10))))
        val dest = tmp.newFile("big.onnx")
        assertTrue(failure { client.downloadModel(dest, maxBytes = 4) }.message!!.contains("larger than expected"))
    }

    @Test fun trainingBlockedMessage() = runTest {
        respond(400, """{"detail":{"blockers":["unknown: 3/30 photos","light_a: 1/30 photos"]}}""")
        val e = failure { client.startTraining() }
        assertEquals(400, e.code)
        assertEquals("Not ready: unknown: 3/30 photos; light_a: 1/30 photos", e.message)
    }

    @Test fun trainingConflictUsesDetail() = runTest {
        respond(409, """{"detail":"training already running"}""")
        assertEquals("training already running", failure { client.startTraining() }.message)
    }

    @Test fun trainStatus() = runTest {
        respond(200, """{"state":"running","epoch":3,"epochs":20,"message":"epoch 3/20","version":null}""")
        val s = client.trainStatus()
        assertTrue(s.running)
        assertEquals(3, s.epoch)
    }

    @Test fun unauthorized() = runTest {
        respond(401, """{"detail":"invalid api key"}""")
        assertEquals("Server rejected the access key", failure { client.health() }.message)
    }

    @Test fun unreachable() = runTest {
        server.shutdown()
        assertTrue(failure { client.health() }.message!!.startsWith("Can't reach the PC server"))
    }

    @Test fun latestMetaAndDownload() = runTest {
        respond(404, """{"detail":"no model trained yet"}""")
        assertNull(client.latestMeta())
        respond(200, """{"version":1,"classes":[],"input_size":224,"normalization":{"mean":[0,0,0],"std":[1,1,1]},"val_accuracy":1.0,"sha256":"s","size_bytes":3}""")
        assertEquals(1, client.latestMeta()!!.version)
        server.enqueue(MockResponse().setResponseCode(200).setBody(okio.Buffer().write(byteArrayOf(9, 8, 7))))
        val dest = tmp.newFile("model.onnx")
        client.downloadModel(dest, maxBytes = 3)
        assertArrayEquals(byteArrayOf(9, 8, 7), dest.readBytes())
    }
}
