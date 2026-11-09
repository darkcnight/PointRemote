package dev.pointremote.model

import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class ModelStoreTest {
    @get:Rule val tmp = TemporaryFolder()

    private class FakeSource(var meta: ModelMeta?, var bytes: ByteArray) : ModelSource {
        var downloads = 0
        override suspend fun latestMeta() = meta
        override suspend fun downloadModel(dest: File, maxBytes: Long) {
            downloads++
            dest.writeBytes(bytes)
        }
    }

    private fun metaFor(version: Int, bytes: ByteArray, sha: String = sha256Hex(bytes)) = ModelMeta(
        version = version,
        classes = listOf(ModelClass("unknown", null, "Unknown"), ModelClass("light_a", "light.a", "A")),
        inputSize = 224,
        normalization = Normalization(listOf(0.485f, 0.456f, 0.406f), listOf(0.229f, 0.224f, 0.225f)),
        valAccuracy = 1.0,
        sha256 = sha,
        sizeBytes = bytes.size.toLong(),
    )

    @Test fun noModelOnServer() = runTest {
        val store = ModelStore(tmp.root)
        assertEquals(UpdateResult.NoModel, store.updateFrom(FakeSource(null, byteArrayOf())))
        assertNull(store.installed.value)
        assertNull(store.modelFile())
    }

    @Test fun installsAndPersists() = runTest {
        val bytes = byteArrayOf(1, 2, 3)
        val store = ModelStore(tmp.root)
        val result = store.updateFrom(FakeSource(metaFor(1, bytes), bytes))
        assertTrue(result is UpdateResult.Updated)
        assertEquals(1, store.installed.value?.version)
        val reopened = ModelStore(tmp.root)
        assertEquals(1, reopened.installed.value?.version)
        assertArrayEquals(bytes, reopened.modelFile()!!.readBytes())
    }

    @Test fun skipsDownloadWhenUpToDate() = runTest {
        val bytes = byteArrayOf(1, 2, 3)
        val source = FakeSource(metaFor(1, bytes), bytes)
        val store = ModelStore(tmp.root)
        store.updateFrom(source)
        assertEquals(UpdateResult.UpToDate, store.updateFrom(source))
        assertEquals(1, source.downloads)
    }

    @Test fun replacesWithNewerVersion() = runTest {
        val store = ModelStore(tmp.root)
        store.updateFrom(FakeSource(metaFor(1, byteArrayOf(1)), byteArrayOf(1)))
        store.updateFrom(FakeSource(metaFor(2, byteArrayOf(2, 2)), byteArrayOf(2, 2)))
        assertEquals(2, store.installed.value?.version)
        assertArrayEquals(byteArrayOf(2, 2), store.modelFile()!!.readBytes())
        assertFalse(File(tmp.root, "old").exists())
    }

    @Test fun rejectsInvalidMetaWithoutDownloading() = runTest {
        val bytes = byteArrayOf(1)
        val good = metaFor(1, bytes)
        val bad = listOf(
            good.copy(classes = good.classes.take(1)),
            good.copy(classes = listOf(good.classes[1], good.classes[0])),
            good.copy(inputSize = 100_000),
            good.copy(normalization = Normalization(listOf(0f, 0f), listOf(1f, 1f))),
            good.copy(classes = good.classes + ModelClass("lock_door", "lock.door", "Door")),
        )
        for (meta in bad) {
            val source = FakeSource(meta, bytes)
            assertTrue(runCatching { ModelStore(tmp.root).updateFrom(source) }.exceptionOrNull() is CorruptModelException)
            assertEquals(0, source.downloads)
        }
    }

    @Test fun tamperedInstalledModelIsIgnored() = runTest {
        ModelStore(tmp.root).updateFrom(FakeSource(metaFor(1, byteArrayOf(1, 2)), byteArrayOf(1, 2)))
        File(tmp.root, "current/model.onnx").writeBytes(byteArrayOf(6, 6))
        assertNull(ModelStore(tmp.root).installed.value)
    }

    @Test fun corruptDownloadKeepsOldModel() = runTest {
        val store = ModelStore(tmp.root)
        store.updateFrom(FakeSource(metaFor(1, byteArrayOf(1)), byteArrayOf(1)))
        val bad = FakeSource(metaFor(2, byteArrayOf(2), sha = "0".repeat(64)), byteArrayOf(2))
        val error = runCatching { store.updateFrom(bad) }.exceptionOrNull()
        assertTrue(error is CorruptModelException)
        assertEquals(1, store.installed.value?.version)
        assertArrayEquals(byteArrayOf(1), store.modelFile()!!.readBytes())
        assertFalse(File(tmp.root, "incoming").exists())
    }
}
