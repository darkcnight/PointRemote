# PointRemote Android App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Plan 2 of 2.** Consumes the HTTP API built in Plan 1 (`docs/superpowers/plans/2026-11-09-training-server.md`).

**Goal:** A native Android app that recognises the appliance the camera is pointing at (using the ONNX model trained by the PC server) and toggles it through Home Assistant, plus capture/train/settings screens, a home-screen widget and a Quick Settings tile.

**Architecture:** Pure-Kotlin core (lock decision, preprocessing, model metadata/store, HTTP clients, pending-photo store) is unit-tested on the JVM. Android-specific shells (CameraX, ONNX Runtime, EncryptedSharedPreferences, WorkManager, Compose screens, Glance widget, TileService) are thin and verified by building and on-device checks. A manual `AppContainer` owned by the `Application` wires everything; each screen has a `ViewModel` created with the container.

**Tech Stack:** AGP 9.5.1 (built-in Kotlin), Kotlin 2.4.21, Gradle 9.9.0 (wrapper, run on JDK 21), compileSdk 37 / targetSdk 36 / minSdk 29, Jetpack Compose (BOM 2026.09.00, Material 3), CameraX 1.6.2, ONNX Runtime Android 1.31.1, Glance 1.2.0, WorkManager 2.12.0, security-crypto 1.1.0, OkHttp 5.5.1 (+ MockWebServer), kotlinx.serialization 1.12.0, coroutines 1.11.0, JUnit 4.

**Spec:** `docs/superpowers/specs/2026-11-09-point-remote-design.md`

## Global Constraints

- Package / applicationId `dev.pointremote`; sources under `android/app/src/main/java/dev/pointremote/`, tests under `android/app/src/test/java/dev/pointremote/`.
- All Gradle commands run from `android/` as `./gradlew --console=plain ...`; JDK 21 is selected by `org.gradle.java.home` in `gradle.properties`.
- Server API (Plan 1): header `X-Api-Key`; `GET /health`, `GET /labels`, `POST /labels {entity_id, friendly_name}`, `POST /photos/{label}` (multipart field `file`, 404 = label gone), `POST /train` (400 `{"detail":{"blockers":[...]}}`, 409 running), `GET /train/status`, `GET /model/latest/meta` (404 none), `GET /model/latest/file`.
- Model: input name `input`, shape `[1,3,224,224]` float NCHW; output `logits`; preprocessing = centre square crop → resize to `input_size` → RGB/255 → `(x-mean)/std` using `meta.normalization`. Class 0 is `unknown`.
- Lock rule: argmax ≠ unknown, p ≥ threshold (default 0.80), same class for ≥ N consecutive frames (default 5); lose lock after N consecutive non-matching frames.
- HA: `POST {ha}/api/services/homeassistant/toggle {"entity_id"}` with `Authorization: Bearer <token>`; no automatic retry of toggles.
- Controllable HA domains: `light, switch, cover, fan, media_player, input_boolean`.
- Capture: ~3 fps while the shutter is held; JPEG, longest side 512 px, quality 90.
- Train button needs ≥ 30 photos for every label (warn below 100).
- Model download: verify `size_bytes` and `sha256` before an atomic swap; keep the old model on failure.
- Cleartext HTTP is allowed (LAN-only app).
- Do not install on the user's phone without asking them first.

## File Structure

```
android/
  settings.gradle.kts, build.gradle.kts, gradle.properties, gradle/libs.versions.toml, gradlew, gradle/wrapper/*
  app/build.gradle.kts
  app/src/main/AndroidManifest.xml
  app/src/main/res/drawable/ic_point.xml
  app/src/main/res/values/strings.xml
  app/src/main/res/xml/point_widget_info.xml
  app/src/main/java/dev/pointremote/
    PointRemoteApp.kt        Application + AppContainer (wiring)
    MainActivity.kt          entry; model update on start
    recognition/LockTracker.kt   softmax + lock decision (pure)
    recognition/Preprocess.kt    crop rect + NCHW normalisation (pure)
    model/ModelMeta.kt           meta.json types
    model/ModelStore.kt          versioned install with verify + atomic swap
    net/Json.kt                  shared Json config
    net/ServerClient.kt          PC server API client (also a ModelSource)
    net/HaClient.kt              Home Assistant client
    settings/AppSettings.kt      settings value + encrypted repository
    capture/PendingPhotos.kt     pending upload files (pure)
    capture/PhotoQueue.kt        enqueue + counts (WorkManager)
    capture/UploadWorker.kt      background upload
    classify/OnnxClassifier.kt   ONNX Runtime session
    camera/FrameConversion.kt    ImageProxy/Bitmap → model input / JPEG
    camera/CameraPreview.kt      CameraX preview + analysis composable, permission gate
    ui/AppRoot.kt                theme + bottom navigation
    ui/point/PointViewModel.kt, ui/point/PointScreen.kt
    ui/capture/CaptureViewModel.kt, ui/capture/CaptureScreen.kt
    ui/train/TrainViewModel.kt, ui/train/TrainScreen.kt
    ui/settings/SettingsScreen.kt
    launch/PointWidget.kt        Glance widget + receiver
    launch/PointTileService.kt   Quick Settings tile
  app/src/test/java/dev/pointremote/
    recognition/LockTrackerTest.kt, recognition/PreprocessTest.kt
    model/ModelMetaTest.kt, model/ModelStoreTest.kt
    net/ServerClientTest.kt, net/HaClientTest.kt
    capture/PendingPhotosTest.kt
```

---

### Task 1: Project scaffold (toolchain verified)

The scaffold was created and built during the toolchain check: root Gradle files, version catalog, wrapper (Gradle 9.9.0), `gradle.properties` (`org.gradle.java.home=/usr/lib/jvm/java-21-openjdk`), `local.properties` (SDK path, git-ignored), `app/build.gradle.kts` (compileSdk 37, minSdk 29, targetSdk 36, all dependencies above, `unitTests.isReturnDefaultValues = true`), a minimal manifest, `MainActivity` and `SmokeTest`.

**Files:** as listed above under `android/` (already present).

- [ ] **Step 1: Verify the build**

Run: `cd android && ./gradlew --console=plain -q testDebugUnitTest assembleDebug`
Expected: exits 0; `app/build/outputs/apk/debug/app-debug.apk` exists.

- [ ] **Step 2: Commit**

```bash
git add android/
git commit -m "feat(android): scaffold Compose app on AGP 9.5 / compileSdk 37"
```

---

### Task 2: Recognition core (softmax, lock tracker, preprocessing)

**Files:**
- Create: `recognition/LockTracker.kt`, `recognition/Preprocess.kt`
- Delete: `app/src/test/java/dev/pointremote/SmokeTest.kt`
- Test: `recognition/LockTrackerTest.kt`, `recognition/PreprocessTest.kt`

**Interfaces:**
- Produces: `sealed interface LockState { data object None; data class Locked(classIndex: Int, confidence: Float) }`; `fun softmax(logits: FloatArray): FloatArray`; `class LockTracker(threshold: Float, stableFrames: Int, unknownIndex: Int = 0)` with `update(probabilities: FloatArray): LockState` and `reset()`.
- Produces: `data class CropRect(left: Int, top: Int, size: Int)`; `fun centerSquare(width: Int, height: Int): CropRect`; `fun toNchw(pixels: IntArray, size: Int, mean: FloatArray, std: FloatArray): FloatArray` (ARGB row-major in, R plane, G plane, B plane out).

- [ ] **Step 1: Write the failing tests**

File: `android/app/src/test/java/dev/pointremote/recognition/LockTrackerTest.kt`
```kotlin
package dev.pointremote.recognition

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LockTrackerTest {
    private val a = floatArrayOf(0.05f, 0.9f, 0.05f)
    private val b = floatArrayOf(0.05f, 0.05f, 0.9f)
    private val unsure = floatArrayOf(0.2f, 0.5f, 0.3f)
    private val unknown = floatArrayOf(0.9f, 0.05f, 0.05f)

    private fun LockTracker.feed(frame: FloatArray, times: Int): LockState {
        var state: LockState = LockState.None
        repeat(times) { state = update(frame) }
        return state
    }

    @Test fun locksOnlyAfterStableFrames() {
        val t = LockTracker(0.8f, 3)
        assertEquals(LockState.None, t.update(a))
        assertEquals(LockState.None, t.update(a))
        assertEquals(LockState.Locked(1, 0.9f), t.update(a))
    }

    @Test fun unknownNeverLocks() {
        assertEquals(LockState.None, LockTracker(0.8f, 3).feed(unknown, 10))
    }

    @Test fun lowConfidenceNeverLocks() {
        assertEquals(LockState.None, LockTracker(0.8f, 3).feed(unsure, 10))
    }

    @Test fun interruptedStreakStartsOver() {
        val t = LockTracker(0.8f, 3)
        t.feed(a, 2)
        t.update(unsure)
        assertEquals(LockState.None, t.feed(a, 2))
        assertTrue(t.update(a) is LockState.Locked)
    }

    @Test fun singleBadFrameKeepsLock() {
        val t = LockTracker(0.8f, 3)
        t.feed(a, 3)
        assertEquals(LockState.Locked(1, 0.9f), t.update(unsure))
    }

    @Test fun unlocksAfterStableBadFrames() {
        val t = LockTracker(0.8f, 3)
        t.feed(a, 3)
        assertTrue(t.feed(unsure, 2) is LockState.Locked)
        assertEquals(LockState.None, t.update(unsure))
    }

    @Test fun switchesToNewTarget() {
        val t = LockTracker(0.8f, 3)
        t.feed(a, 3)
        assertEquals(LockState.Locked(1, 0.9f), t.update(b))
        assertEquals(LockState.Locked(1, 0.9f), t.update(b))
        assertEquals(LockState.Locked(2, 0.9f), t.update(b))
    }

    @Test fun confidenceTracksLatestFrame() {
        val t = LockTracker(0.8f, 3)
        t.feed(a, 3)
        assertEquals(LockState.Locked(1, 0.95f), t.update(floatArrayOf(0.03f, 0.95f, 0.02f)))
    }

    @Test fun resetClearsLock() {
        val t = LockTracker(0.8f, 3)
        t.feed(a, 3)
        t.reset()
        assertEquals(LockState.None, t.update(a))
    }

    @Test fun softmaxIsNormalisedAndStable() {
        val p = softmax(floatArrayOf(1f, 2f, 3f))
        assertEquals(1f, p.sum(), 1e-5f)
        assertTrue(p[2] > p[1] && p[1] > p[0])
        val big = softmax(floatArrayOf(1000f, 1001f))
        assertEquals(1f, big.sum(), 1e-5f)
        assertTrue(big.none { it.isNaN() })
    }
}
```

File: `android/app/src/test/java/dev/pointremote/recognition/PreprocessTest.kt`
```kotlin
package dev.pointremote.recognition

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class PreprocessTest {
    @Test fun centerSquareLandscape() = assertEquals(CropRect(80, 0, 480), centerSquare(640, 480))

    @Test fun centerSquarePortrait() = assertEquals(CropRect(0, 80, 480), centerSquare(480, 640))

    @Test fun centerSquareSquare() = assertEquals(CropRect(0, 0, 100), centerSquare(100, 100))

    private val pixels = intArrayOf(
        0xFFFF0000.toInt(), 0xFF00FF00.toInt(),
        0xFF0000FF.toInt(), 0xFFFFFFFF.toInt(),
    )

    @Test fun toNchwSplitsPlanes() {
        val out = toNchw(pixels, 2, floatArrayOf(0f, 0f, 0f), floatArrayOf(1f, 1f, 1f))
        assertArrayEquals(
            floatArrayOf(
                1f, 0f, 0f, 1f, // R
                0f, 1f, 0f, 1f, // G
                0f, 0f, 1f, 1f, // B
            ),
            out, 1e-6f,
        )
    }

    @Test fun toNchwNormalises() {
        val out = toNchw(pixels, 2, floatArrayOf(0.5f, 0.5f, 0.5f), floatArrayOf(0.5f, 0.5f, 0.5f))
        assertEquals(1f, out[0], 1e-6f)   // red pixel, R plane
        assertEquals(-1f, out[4], 1e-6f)  // red pixel, G plane
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./gradlew --console=plain -q testDebugUnitTest`
Expected: FAIL — compilation errors `Unresolved reference 'LockTracker'`, `'centerSquare'`.

- [ ] **Step 3: Implement**

File: `android/app/src/main/java/dev/pointremote/recognition/LockTracker.kt`
```kotlin
package dev.pointremote.recognition

import kotlin.math.exp

sealed interface LockState {
    data object None : LockState
    data class Locked(val classIndex: Int, val confidence: Float) : LockState
}

fun softmax(logits: FloatArray): FloatArray {
    val max = logits.max()
    val exps = FloatArray(logits.size) { exp((logits[it] - max).toDouble()).toFloat() }
    val sum = exps.sum()
    return FloatArray(exps.size) { exps[it] / sum }
}

/**
 * Turns a stream of per-frame class probabilities into a stable "locked target".
 * A class locks after [stableFrames] consecutive confident frames and unlocks
 * after [stableFrames] consecutive frames that don't confirm it.
 */
class LockTracker(
    private val threshold: Float,
    private val stableFrames: Int,
    private val unknownIndex: Int = 0,
) {
    private var streakClass = -1
    private var streakCount = 0
    private var locked = -1
    private var lockedConfidence = 0f
    private var failCount = 0

    fun reset() {
        streakClass = -1
        streakCount = 0
        locked = -1
        failCount = 0
    }

    fun update(probabilities: FloatArray): LockState {
        var best = 0
        for (i in probabilities.indices) if (probabilities[i] > probabilities[best]) best = i
        val p = probabilities[best]
        val qualifies = best != unknownIndex && p >= threshold

        if (qualifies) {
            if (best == streakClass) streakCount++ else {
                streakClass = best
                streakCount = 1
            }
        } else {
            streakClass = -1
            streakCount = 0
        }

        if (locked >= 0) {
            if (qualifies && best == locked) {
                failCount = 0
                lockedConfidence = p
            } else if (++failCount >= stableFrames) {
                locked = -1
                failCount = 0
            }
        }
        if (locked < 0 && streakCount >= stableFrames) {
            locked = streakClass
            lockedConfidence = p
            failCount = 0
        }
        return if (locked >= 0) LockState.Locked(locked, lockedConfidence) else LockState.None
    }
}
```

File: `android/app/src/main/java/dev/pointremote/recognition/Preprocess.kt`
```kotlin
package dev.pointremote.recognition

data class CropRect(val left: Int, val top: Int, val size: Int)

fun centerSquare(width: Int, height: Int): CropRect {
    val size = minOf(width, height)
    return CropRect((width - size) / 2, (height - size) / 2, size)
}

/** ARGB pixels (row-major, size × size) → normalised NCHW floats (R, G, B planes). */
fun toNchw(pixels: IntArray, size: Int, mean: FloatArray, std: FloatArray): FloatArray {
    require(pixels.size == size * size) { "expected ${size * size} pixels, got ${pixels.size}" }
    val plane = size * size
    val out = FloatArray(3 * plane)
    for (i in 0 until plane) {
        val c = pixels[i]
        val r = ((c shr 16) and 0xFF) / 255f
        val g = ((c shr 8) and 0xFF) / 255f
        val b = (c and 0xFF) / 255f
        out[i] = (r - mean[0]) / std[0]
        out[plane + i] = (g - mean[1]) / std[1]
        out[2 * plane + i] = (b - mean[2]) / std[2]
    }
    return out
}
```

Also: `git rm android/app/src/test/java/dev/pointremote/SmokeTest.kt`

- [ ] **Step 4: Run tests to verify they pass**

Run: `./gradlew --console=plain -q testDebugUnitTest`
Expected: exits 0.

- [ ] **Step 5: Commit**

```bash
git add android/app/src
git commit -m "feat(android): lock tracker, softmax and preprocessing"
```

---

### Task 3: Model metadata and model store

**Files:**
- Create: `net/Json.kt`, `model/ModelMeta.kt`, `model/ModelStore.kt`
- Test: `model/ModelMetaTest.kt`, `model/ModelStoreTest.kt`

**Interfaces:**
- Produces: `val AppJson: Json` (ignoreUnknownKeys). `@Serializable data class ModelClass(id, entityId: String?, friendlyName)`, `Normalization(mean: List<Float>, std: List<Float>)`, `Confusion(trueId, predicted, count)`, `ModelMeta(version, classes, inputSize, normalization, valAccuracy: Double, perClassAccuracy: Map<String, Double>, confusions: List<Confusion>, sha256, sizeBytes: Long)` (snake_case JSON names).
- Produces: `interface ModelSource { suspend fun latestMeta(): ModelMeta?; suspend fun downloadModel(dest: File) }`; `class CorruptModelException`; `sealed interface UpdateResult { NoModel; UpToDate; Updated(meta) }`; `fun sha256Hex(file: File): String`, `fun sha256Hex(bytes: ByteArray): String`; `class ModelStore(dir: File)` with `installed: StateFlow<ModelMeta?>`, `modelFile(): File?`, `suspend fun updateFrom(source: ModelSource): UpdateResult`.

- [ ] **Step 1: Write the failing tests**

File: `android/app/src/test/java/dev/pointremote/model/ModelMetaTest.kt`
```kotlin
package dev.pointremote.model

import dev.pointremote.net.AppJson
import org.junit.Assert.assertEquals
import org.junit.Test

class ModelMetaTest {
    @Test fun parsesServerMeta() {
        val json = """
            {"version": 3, "created_at": "2026-11-09T10:00:00+00:00",
             "classes": [{"id": "unknown", "entity_id": null, "friendly_name": "Unknown"},
                         {"id": "light_kitchen", "entity_id": "light.kitchen", "friendly_name": "Kitchen"}],
             "input_size": 224,
             "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
             "preprocessing": "center_square_crop_then_resize_rgb", "output": "logits",
             "val_accuracy": 0.95, "per_class_accuracy": {"unknown": 0.9, "light_kitchen": 1.0},
             "confusion_matrix": [[9, 1], [0, 10]],
             "confusions": [{"true": "unknown", "predicted": "light_kitchen", "count": 1}],
             "train_count": 80, "val_count": 20, "seed": 1234, "epochs": 20,
             "sha256": "abc", "size_bytes": 123}
        """.trimIndent()
        val meta = AppJson.decodeFromString<ModelMeta>(json)
        assertEquals(3, meta.version)
        assertEquals(null, meta.classes[0].entityId)
        assertEquals("light.kitchen", meta.classes[1].entityId)
        assertEquals("Kitchen", meta.classes[1].friendlyName)
        assertEquals(224, meta.inputSize)
        assertEquals(0.229f, meta.normalization.std[0], 1e-6f)
        assertEquals(1.0, meta.perClassAccuracy.getValue("light_kitchen"), 1e-9)
        assertEquals(Confusion("unknown", "light_kitchen", 1), meta.confusions[0])
        assertEquals(123L, meta.sizeBytes)
    }
}
```

File: `android/app/src/test/java/dev/pointremote/model/ModelStoreTest.kt`
```kotlin
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
        override suspend fun downloadModel(dest: File) {
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./gradlew --console=plain -q testDebugUnitTest`
Expected: FAIL — `Unresolved reference 'AppJson'`, `'ModelStore'`.

- [ ] **Step 3: Implement**

File: `android/app/src/main/java/dev/pointremote/net/Json.kt`
```kotlin
package dev.pointremote.net

import kotlinx.serialization.json.Json

val AppJson = Json {
    ignoreUnknownKeys = true
    explicitNulls = false
}
```

File: `android/app/src/main/java/dev/pointremote/model/ModelMeta.kt`
```kotlin
package dev.pointremote.model

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class ModelClass(
    val id: String,
    @SerialName("entity_id") val entityId: String? = null,
    @SerialName("friendly_name") val friendlyName: String,
)

@Serializable
data class Normalization(val mean: List<Float>, val std: List<Float>)

@Serializable
data class Confusion(
    @SerialName("true") val trueId: String,
    val predicted: String,
    val count: Int,
)

@Serializable
data class ModelMeta(
    val version: Int,
    val classes: List<ModelClass>,
    @SerialName("input_size") val inputSize: Int,
    val normalization: Normalization,
    @SerialName("val_accuracy") val valAccuracy: Double,
    @SerialName("per_class_accuracy") val perClassAccuracy: Map<String, Double> = emptyMap(),
    val confusions: List<Confusion> = emptyList(),
    val sha256: String,
    @SerialName("size_bytes") val sizeBytes: Long,
)
```

File: `android/app/src/main/java/dev/pointremote/model/ModelStore.kt`
```kotlin
package dev.pointremote.model

import dev.pointremote.net.AppJson
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
    suspend fun downloadModel(dest: File)
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
        return runCatching { AppJson.decodeFromString<ModelMeta>(metaFile.readText()) }.getOrNull()
    }

    suspend fun updateFrom(source: ModelSource): UpdateResult = mutex.withLock {
        val meta = source.latestMeta() ?: return UpdateResult.NoModel
        if (_installed.value?.version == meta.version) return UpdateResult.UpToDate
        withContext(Dispatchers.IO) {
            dir.mkdirs()
            val incoming = File(dir, "incoming").apply { deleteRecursively(); mkdirs() }
            try {
                val model = File(incoming, MODEL)
                source.downloadModel(model)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./gradlew --console=plain -q testDebugUnitTest`
Expected: exits 0.

- [ ] **Step 5: Commit**

```bash
git add android/app/src
git commit -m "feat(android): model metadata and verified atomic model store"
```

---

### Task 4: PC server client

**Files:**
- Create: `net/ServerClient.kt`
- Test: `net/ServerClientTest.kt`

**Interfaces:**
- Consumes: `AppJson`, `ModelMeta`, `ModelSource` (Task 3).
- Produces: `class ServerException(message: String, val code: Int? = null)`; `@Serializable Health(ok, training, latestVersion: Int?)`, `ServerLabel(id, entityId: String?, friendlyName, photoCount: Int)`, `TrainStatus(state, epoch, epochs, message, version: Int?)` with `val running: Boolean`; `sealed interface UploadResult { Ok(count: Int); LabelGone }`; `class ServerClient(baseUrl: String, apiKey: String, http: OkHttpClient) : ModelSource` with `health()`, `labels()`, `createLabel(entityId, friendlyName)`, `uploadPhoto(labelId, jpeg: File)`, `startTraining()`, `trainStatus()`, `latestMeta()`, `downloadModel(dest)`. All suspend; network failures → `ServerException("Can't reach the PC server (...)")`; 401 → `"Server rejected the access key"`; 400 with blockers → `"Not ready: a; b"`.

- [ ] **Step 1: Write the failing tests**

File: `android/app/src/test/java/dev/pointremote/net/ServerClientTest.kt`
```kotlin
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
        client.downloadModel(dest)
        assertArrayEquals(byteArrayOf(9, 8, 7), dest.readBytes())
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./gradlew --console=plain -q testDebugUnitTest`
Expected: FAIL — `Unresolved reference 'ServerClient'`.

- [ ] **Step 3: Implement**

File: `android/app/src/main/java/dev/pointremote/net/ServerClient.kt`
```kotlin
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

    override suspend fun downloadModel(dest: File) {
        call(request("/model/latest/file").build()) { r ->
            if (!r.isSuccessful) throw ServerException(serverErrorMessage(r.code, r.body.string()), r.code)
            dest.outputStream().use { out -> r.body.byteStream().copyTo(out) }
        }
    }

    private companion object {
        val JSON_TYPE = "application/json".toMediaType()
        val JPEG_TYPE = "image/jpeg".toMediaType()
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./gradlew --console=plain -q testDebugUnitTest`
Expected: exits 0.

- [ ] **Step 5: Commit**

```bash
git add android/app/src
git commit -m "feat(android): PC server API client"
```

---

### Task 5: Home Assistant client

**Files:**
- Create: `net/HaClient.kt`
- Test: `net/HaClientTest.kt`

**Interfaces:**
- Produces: `class HaException(message, val kind: Kind)` with `enum Kind { Unauthorized, Network, Http }`; `data class HaEntity(entityId, friendlyName, state)`; `val CONTROLLABLE_DOMAINS: Set<String>`; `class HaClient(baseUrl, token, http: OkHttpClient)` with `suspend toggle(entityId): String?` (new state; falls back to `GET /api/states/{id}` when the response doesn't include it), `suspend state(entityId): String?`, `suspend entities(): List<HaEntity>` (controllable only, sorted by friendly name, case-insensitive).

- [ ] **Step 1: Write the failing tests**

File: `android/app/src/test/java/dev/pointremote/net/HaClientTest.kt`
```kotlin
package dev.pointremote.net

import kotlinx.coroutines.test.runTest
import okhttp3.OkHttpClient
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
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

    @Test fun entitiesFiltersAndSorts() = runTest {
        respond(200, """[
            {"entity_id":"sensor.temp","state":"21","attributes":{"friendly_name":"Temp"}},
            {"entity_id":"light.kitchen","state":"off","attributes":{"friendly_name":"kitchen lights"}},
            {"entity_id":"cover.big","state":"open","attributes":{"friendly_name":"Big shutter"}},
            {"entity_id":"switch.tv","state":"on","attributes":{}}
        ]""")
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./gradlew --console=plain -q testDebugUnitTest`
Expected: FAIL — `Unresolved reference 'HaClient'`.

- [ ] **Step 3: Implement**

File: `android/app/src/main/java/dev/pointremote/net/HaClient.kt`
```kotlin
package dev.pointremote.net

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.IOException

class HaException(message: String, val kind: Kind) : Exception(message) {
    enum class Kind { Unauthorized, Network, Http }
}

data class HaEntity(val entityId: String, val friendlyName: String, val state: String)

val CONTROLLABLE_DOMAINS = setOf("light", "switch", "cover", "fan", "media_player", "input_boolean")

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
        val payload = buildJsonObject { put("entity_id", entityId) }.toString()
        val post = request("/api/services/homeassistant/toggle").post(payload.toRequestBody(JSON_TYPE)).build()
        val changed = call(post) { body -> stateIn(AppJson.parseToJsonElement(body), entityId) }
        return changed ?: state(entityId)
    }

    suspend fun state(entityId: String): String? =
        call(request("/api/states/$entityId").build()) { body ->
            AppJson.parseToJsonElement(body).jsonObject["state"]?.jsonPrimitive?.content
        }

    suspend fun entities(): List<HaEntity> =
        call(request("/api/states").build()) { body ->
            AppJson.parseToJsonElement(body).jsonArray.mapNotNull { element ->
                val o = element.jsonObject
                val id = o["entity_id"]?.jsonPrimitive?.content ?: return@mapNotNull null
                if (id.substringBefore('.') !in CONTROLLABLE_DOMAINS) return@mapNotNull null
                val name = o["attributes"]?.jsonObject?.get("friendly_name")?.jsonPrimitive?.content ?: id
                HaEntity(id, name, o["state"]?.jsonPrimitive?.content ?: "unknown")
            }.sortedBy { it.friendlyName.lowercase() }
        }

    private companion object {
        val JSON_TYPE = "application/json".toMediaType()
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./gradlew --console=plain -q testDebugUnitTest`
Expected: exits 0.

- [ ] **Step 5: Commit**

```bash
git add android/app/src
git commit -m "feat(android): Home Assistant client"
```

---

### Task 6: Settings, pending photos, upload worker, app wiring

**Files:**
- Create: `settings/AppSettings.kt`, `capture/PendingPhotos.kt`, `capture/PhotoQueue.kt`, `capture/UploadWorker.kt`, `PointRemoteApp.kt`
- Modify: `app/src/main/AndroidManifest.xml` (application class, permissions, cleartext)
- Test: `capture/PendingPhotosTest.kt`

**Interfaces:**
- Produces: `data class AppSettings(haUrl, haToken, serverUrl, serverKey, threshold = 0.80f, stableFrames = 5)` with `haConfigured`, `serverConfigured`; `class SettingsRepository(context)` with `settings: StateFlow<AppSettings>`, `save(AppSettings)`.
- Produces: `class PendingPhotos(root: File)` with `add(labelId, jpeg: ByteArray): File`, `counts(): Map<String, Int>`; `class PhotoQueue(context)` with `counts: StateFlow<Map<String, Int>>`, `enqueue(labelId, jpeg)`, `refresh()`; `class UploadWorker` (`KEY_PATH`, `KEY_LABEL`).
- Produces: `class AppContainer(context)` with `settings`, `modelStore`, `photoQueue`, `http: OkHttpClient`, `server(): ServerClient?`, `ha(): HaClient?`; `class PointRemoteApp : Application` with `container`.

- [ ] **Step 1: Write the failing test**

File: `android/app/src/test/java/dev/pointremote/capture/PendingPhotosTest.kt`
```kotlin
package dev.pointremote.capture

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class PendingPhotosTest {
    @get:Rule val tmp = TemporaryFolder()

    @Test fun addsAndCounts() {
        val pending = PendingPhotos(tmp.root)
        assertEquals(emptyMap<String, Int>(), pending.counts())
        val f = pending.add("light_a", byteArrayOf(1, 2))
        pending.add("light_a", byteArrayOf(3))
        pending.add("unknown", byteArrayOf(4))
        assertArrayEquals(byteArrayOf(1, 2), f.readBytes())
        assertEquals("light_a", f.parentFile!!.name)
        assertEquals(mapOf("light_a" to 2, "unknown" to 1), pending.counts())
        f.delete()
        assertEquals(mapOf("light_a" to 1, "unknown" to 1), pending.counts())
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./gradlew --console=plain -q testDebugUnitTest`
Expected: FAIL — `Unresolved reference 'PendingPhotos'`.

- [ ] **Step 3: Implement**

File: `android/app/src/main/java/dev/pointremote/capture/PendingPhotos.kt`
```kotlin
package dev.pointremote.capture

import java.io.File
import java.util.UUID

/** Photos waiting to be uploaded, stored as `root/<label>/<uuid>.jpg`. */
class PendingPhotos(private val root: File) {
    fun add(labelId: String, jpeg: ByteArray): File {
        val dir = File(root, labelId).apply { mkdirs() }
        return File(dir, "${UUID.randomUUID()}.jpg").apply { writeBytes(jpeg) }
    }

    fun counts(): Map<String, Int> =
        (root.listFiles() ?: emptyArray())
            .filter { it.isDirectory }
            .associate { dir -> dir.name to (dir.listFiles { f -> f.extension == "jpg" }?.size ?: 0) }
            .filterValues { it > 0 }
}
```

File: `android/app/src/main/java/dev/pointremote/settings/AppSettings.kt`
```kotlin
package dev.pointremote.settings

import android.content.Context
import android.content.SharedPreferences
import androidx.core.content.edit
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

data class AppSettings(
    val haUrl: String = "",
    val haToken: String = "",
    val serverUrl: String = "",
    val serverKey: String = "",
    val threshold: Float = 0.80f,
    val stableFrames: Int = 5,
) {
    val haConfigured: Boolean get() = haUrl.isNotBlank() && haToken.isNotBlank()
    val serverConfigured: Boolean get() = serverUrl.isNotBlank() && serverKey.isNotBlank()
}

@Suppress("DEPRECATION") // security-crypto is deprecated upstream but still the simplest encrypted store.
class SettingsRepository(context: Context) {
    private val prefs: SharedPreferences = EncryptedSharedPreferences.create(
        context,
        "pointremote_settings",
        MasterKey.Builder(context).setKeyScheme(MasterKey.KeyScheme.AES256_GCM).build(),
        EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
        EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM,
    )
    private val _settings = MutableStateFlow(load())
    val settings: StateFlow<AppSettings> = _settings.asStateFlow()

    private fun load() = AppSettings(
        haUrl = prefs.getString(HA_URL, "") ?: "",
        haToken = prefs.getString(HA_TOKEN, "") ?: "",
        serverUrl = prefs.getString(SERVER_URL, "") ?: "",
        serverKey = prefs.getString(SERVER_KEY, "") ?: "",
        threshold = prefs.getFloat(THRESHOLD, 0.80f),
        stableFrames = prefs.getInt(STABLE_FRAMES, 5),
    )

    fun save(settings: AppSettings) {
        prefs.edit {
            putString(HA_URL, settings.haUrl.trim())
            putString(HA_TOKEN, settings.haToken.trim())
            putString(SERVER_URL, settings.serverUrl.trim())
            putString(SERVER_KEY, settings.serverKey.trim())
            putFloat(THRESHOLD, settings.threshold)
            putInt(STABLE_FRAMES, settings.stableFrames)
        }
        _settings.value = load()
    }

    private companion object {
        const val HA_URL = "ha_url"
        const val HA_TOKEN = "ha_token"
        const val SERVER_URL = "server_url"
        const val SERVER_KEY = "server_key"
        const val THRESHOLD = "threshold"
        const val STABLE_FRAMES = "stable_frames"
    }
}
```

File: `android/app/src/main/java/dev/pointremote/capture/PhotoQueue.kt`
```kotlin
package dev.pointremote.capture

import android.content.Context
import androidx.work.BackoffPolicy
import androidx.work.Constraints
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.workDataOf
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.File
import java.util.concurrent.TimeUnit

class PhotoQueue(private val context: Context) {
    private val pending = PendingPhotos(File(context.filesDir, "pending"))
    private val _counts = MutableStateFlow(pending.counts())
    val counts: StateFlow<Map<String, Int>> = _counts.asStateFlow()

    fun enqueue(labelId: String, jpeg: ByteArray) {
        val file = pending.add(labelId, jpeg)
        val request = OneTimeWorkRequestBuilder<UploadWorker>()
            .setInputData(workDataOf(UploadWorker.KEY_PATH to file.path, UploadWorker.KEY_LABEL to labelId))
            .setConstraints(Constraints.Builder().setRequiredNetworkType(NetworkType.CONNECTED).build())
            .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, 15, TimeUnit.SECONDS)
            .build()
        WorkManager.getInstance(context).enqueue(request)
        refresh()
    }

    fun refresh() {
        _counts.value = pending.counts()
    }
}
```

File: `android/app/src/main/java/dev/pointremote/capture/UploadWorker.kt`
```kotlin
package dev.pointremote.capture

import android.content.Context
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import dev.pointremote.PointRemoteApp
import dev.pointremote.net.ServerException
import java.io.File

class UploadWorker(context: Context, params: WorkerParameters) : CoroutineWorker(context, params) {
    override suspend fun doWork(): Result {
        val path = inputData.getString(KEY_PATH) ?: return Result.failure()
        val label = inputData.getString(KEY_LABEL) ?: return Result.failure()
        val file = File(path)
        val container = (applicationContext as PointRemoteApp).container
        if (!file.exists()) return Result.success()
        val server = container.server() ?: return Result.retry()
        return try {
            // Ok or LabelGone: either way this file is finished with.
            server.uploadPhoto(label, file)
            file.delete()
            Result.success()
        } catch (e: ServerException) {
            val code = e.code
            val permanent = code != null && code in 400..499 && code !in setOf(401, 408, 429)
            if (permanent) {
                file.delete()
                Result.failure()
            } else {
                Result.retry()
            }
        } finally {
            container.photoQueue.refresh()
        }
    }

    companion object {
        const val KEY_PATH = "path"
        const val KEY_LABEL = "label"
    }
}
```

File: `android/app/src/main/java/dev/pointremote/PointRemoteApp.kt`
```kotlin
package dev.pointremote

import android.app.Application
import android.content.Context
import dev.pointremote.capture.PhotoQueue
import dev.pointremote.model.ModelStore
import dev.pointremote.net.HaClient
import dev.pointremote.net.ServerClient
import dev.pointremote.settings.SettingsRepository
import okhttp3.OkHttpClient
import java.io.File
import java.util.concurrent.TimeUnit

class AppContainer(context: Context) {
    val settings = SettingsRepository(context)
    val modelStore = ModelStore(File(context.filesDir, "model"))
    val photoQueue = PhotoQueue(context)
    val http: OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(5, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .build()

    fun server(): ServerClient? = settings.settings.value.let {
        if (it.serverConfigured) ServerClient(it.serverUrl, it.serverKey, http) else null
    }

    fun ha(): HaClient? = settings.settings.value.let {
        if (it.haConfigured) HaClient(it.haUrl, it.haToken, http) else null
    }
}

class PointRemoteApp : Application() {
    lateinit var container: AppContainer
        private set

    override fun onCreate() {
        super.onCreate()
        container = AppContainer(this)
    }
}
```

File: `android/app/src/main/AndroidManifest.xml`
```xml
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android">

    <uses-permission android:name="android.permission.INTERNET" />
    <uses-permission android:name="android.permission.CAMERA" />
    <uses-feature android:name="android.hardware.camera.any" android:required="true" />

    <application
        android:name=".PointRemoteApp"
        android:label="PointRemote"
        android:usesCleartextTraffic="true"
        android:theme="@android:style/Theme.Material.NoActionBar">

        <activity
            android:name=".MainActivity"
            android:exported="true"
            android:launchMode="singleTask">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>

    </application>
</manifest>
```

- [ ] **Step 4: Run tests and build**

Run: `./gradlew --console=plain -q testDebugUnitTest assembleDebug`
Expected: exits 0.

- [ ] **Step 5: Commit**

```bash
git add android/app/src
git commit -m "feat(android): settings, upload queue and app container"
```

---

### Task 7: Classifier and camera plumbing

**Files:**
- Create: `classify/OnnxClassifier.kt`, `camera/FrameConversion.kt`, `camera/CameraPreview.kt`

**Interfaces:**
- Consumes: `centerSquare`, `toNchw` (Task 2).
- Produces: `class OnnxClassifier(modelFile: File, inputSize: Int) : AutoCloseable` with `logits(input: FloatArray): FloatArray`; `fun ImageProxy.uprightBitmap(): Bitmap`; `fun Bitmap.toModelInput(size: Int, mean: FloatArray, std: FloatArray): FloatArray`; `fun Bitmap.toCaptureJpeg(maxSide: Int = 512, quality: Int = 90): ByteArray`; `@Composable fun CameraPreview(onFrame: (ImageProxy) -> Unit, modifier: Modifier = Modifier)` (frames closed automatically after `onFrame` returns); `@Composable fun WithCameraPermission(content: @Composable () -> Unit)`.

These wrap Android/native APIs and are verified by compiling here and on-device in Task 12.

- [ ] **Step 1: Implement**

File: `android/app/src/main/java/dev/pointremote/classify/OnnxClassifier.kt`
```kotlin
package dev.pointremote.classify

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.FloatBuffer

class OnnxClassifier(modelFile: File, private val inputSize: Int) : AutoCloseable {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession = env.createSession(modelFile.path, OrtSession.SessionOptions())
    private val shape = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())

    fun logits(input: FloatArray): FloatArray {
        OnnxTensor.createTensor(env, FloatBuffer.wrap(input), shape).use { tensor ->
            session.run(mapOf("input" to tensor)).use { result ->
                @Suppress("UNCHECKED_CAST")
                return (result.get(0).value as Array<FloatArray>)[0]
            }
        }
    }

    override fun close() = session.close()
}
```

File: `android/app/src/main/java/dev/pointremote/camera/FrameConversion.kt`
```kotlin
package dev.pointremote.camera

import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.camera.core.ImageProxy
import dev.pointremote.recognition.centerSquare
import dev.pointremote.recognition.toNchw
import java.io.ByteArrayOutputStream
import kotlin.math.roundToInt

fun ImageProxy.uprightBitmap(): Bitmap {
    val bitmap = toBitmap()
    val rotation = imageInfo.rotationDegrees
    if (rotation == 0) return bitmap
    val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
    return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
}

/** Centre square crop → resize → normalised NCHW, matching the server's eval transform. */
fun Bitmap.toModelInput(size: Int, mean: FloatArray, std: FloatArray): FloatArray {
    val crop = centerSquare(width, height)
    val square = Bitmap.createBitmap(this, crop.left, crop.top, crop.size, crop.size)
    val scaled = Bitmap.createScaledBitmap(square, size, size, true)
    val pixels = IntArray(size * size)
    scaled.getPixels(pixels, 0, size, 0, 0, size, size)
    return toNchw(pixels, size, mean, std)
}

fun Bitmap.toCaptureJpeg(maxSide: Int = 512, quality: Int = 90): ByteArray {
    val scale = maxSide.toFloat() / maxOf(width, height)
    val out = if (scale < 1f) {
        Bitmap.createScaledBitmap(this, (width * scale).roundToInt(), (height * scale).roundToInt(), true)
    } else {
        this
    }
    return ByteArrayOutputStream().also { out.compress(Bitmap.CompressFormat.JPEG, quality, it) }.toByteArray()
}
```

File: `android/app/src/main/java/dev/pointremote/camera/CameraPreview.kt`
```kotlin
package dev.pointremote.camera

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import java.util.concurrent.Executors

@Composable
fun CameraPreview(onFrame: (ImageProxy) -> Unit, modifier: Modifier = Modifier) {
    val lifecycleOwner = LocalLifecycleOwner.current
    val currentOnFrame by rememberUpdatedState(onFrame)
    val executor = remember { Executors.newSingleThreadExecutor() }
    DisposableEffect(Unit) { onDispose { executor.shutdown() } }

    AndroidView(
        modifier = modifier,
        factory = { context ->
            val view = PreviewView(context).apply { scaleType = PreviewView.ScaleType.FILL_CENTER }
            val providerFuture = ProcessCameraProvider.getInstance(context)
            providerFuture.addListener({
                val provider = providerFuture.get()
                val preview = Preview.Builder().build().also { it.setSurfaceProvider(view.surfaceProvider) }
                val analysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .build()
                analysis.setAnalyzer(executor) { image -> image.use { currentOnFrame(it) } }
                provider.unbindAll()
                provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
            }, ContextCompat.getMainExecutor(context))
            view
        },
    )
}

@Composable
fun WithCameraPermission(content: @Composable () -> Unit) {
    val context = LocalContext.current
    var granted by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED,
        )
    }
    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted = it }
    LaunchedEffect(Unit) { if (!granted) launcher.launch(Manifest.permission.CAMERA) }
    if (granted) {
        content()
    } else {
        Box(Modifier.fillMaxSize().padding(24.dp), contentAlignment = Alignment.Center) {
            Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(16.dp)) {
                Text("PointRemote needs the camera to see what you're pointing at.", textAlign = TextAlign.Center)
                Button(onClick = { launcher.launch(Manifest.permission.CAMERA) }) { Text("Allow camera") }
            }
        }
    }
}
```

- [ ] **Step 2: Build**

Run: `./gradlew --console=plain -q assembleDebug`
Expected: exits 0.

- [ ] **Step 3: Commit**

```bash
git add android/app/src
git commit -m "feat(android): ONNX classifier and CameraX plumbing"
```

---

### Task 8: Point screen

**Files:**
- Create: `ui/point/PointViewModel.kt`, `ui/point/PointScreen.kt`

**Interfaces:**
- Consumes: `AppContainer`, `ModelStore.installed`, `OnnxClassifier`, `LockTracker`, `softmax`, frame conversion, `CameraPreview`, `WithCameraPermission`, `HaClient`/`HaException`.
- Produces: `data class PointUiState(hasModel, target: ModelClass?, confidence: Float, toggling: Boolean, message: String?)`; `sealed interface PointEvent { Locked; ToggleOk; Error }`; `class PointViewModel(container)` with `state`, `events: Flow<PointEvent>`, `analyze(ImageProxy)` (camera thread), `toggle()`, `clearMessage()`; `@Composable fun PointScreen(container: AppContainer, onOpenCapture: () -> Unit)`.

- [ ] **Step 1: Implement the view model**

File: `android/app/src/main/java/dev/pointremote/ui/point/PointViewModel.kt`
```kotlin
package dev.pointremote.ui.point

import androidx.camera.core.ImageProxy
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dev.pointremote.AppContainer
import dev.pointremote.camera.toModelInput
import dev.pointremote.camera.uprightBitmap
import dev.pointremote.classify.OnnxClassifier
import dev.pointremote.model.ModelClass
import dev.pointremote.model.ModelMeta
import dev.pointremote.net.HaException
import dev.pointremote.recognition.LockState
import dev.pointremote.recognition.LockTracker
import dev.pointremote.recognition.softmax
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.receiveAsFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class PointUiState(
    val hasModel: Boolean = false,
    val target: ModelClass? = null,
    val confidence: Float = 0f,
    val toggling: Boolean = false,
    val message: String? = null,
)

sealed interface PointEvent {
    data object Locked : PointEvent
    data object ToggleOk : PointEvent
    data object Error : PointEvent
}

class PointViewModel(private val container: AppContainer) : ViewModel() {
    private val _state = MutableStateFlow(PointUiState())
    val state: StateFlow<PointUiState> = _state.asStateFlow()
    private val _events = Channel<PointEvent>(Channel.BUFFERED)
    val events: Flow<PointEvent> = _events.receiveAsFlow()

    private val lock = Any()
    private var classifier: OnnxClassifier? = null
    private var meta: ModelMeta? = null
    private var tracker = newTracker()

    init {
        viewModelScope.launch { container.modelStore.installed.collect { loadModel(it) } }
        viewModelScope.launch {
            container.settings.settings.collect { synchronized(lock) { tracker = newTracker() } }
        }
    }

    private fun newTracker() = container.settings.settings.value.let { LockTracker(it.threshold, it.stableFrames) }

    private fun loadModel(newMeta: ModelMeta?) {
        val loaded = synchronized(lock) {
            classifier?.close()
            classifier = null
            meta = newMeta
            val file = container.modelStore.modelFile()
            if (newMeta != null && file != null) {
                classifier = runCatching { OnnxClassifier(file, newMeta.inputSize) }.getOrNull()
            }
            tracker = newTracker()
            classifier != null
        }
        _state.update {
            it.copy(
                hasModel = loaded,
                target = null,
                message = if (newMeta != null && !loaded) "Couldn't load model v${newMeta.version}" else it.message,
            )
        }
    }

    /** Runs on the camera executor. */
    fun analyze(image: ImageProxy) {
        val m = synchronized(lock) { meta } ?: return
        val input = image.uprightBitmap().toModelInput(
            m.inputSize,
            m.normalization.mean.toFloatArray(),
            m.normalization.std.toFloatArray(),
        )
        val lockState = synchronized(lock) {
            val c = classifier ?: return
            if (meta !== m) return
            tracker.update(softmax(c.logits(input)))
        }
        val previous = _state.value.target
        when (lockState) {
            LockState.None -> _state.update { it.copy(target = null, confidence = 0f) }
            is LockState.Locked -> {
                val cls = m.classes[lockState.classIndex]
                _state.update { it.copy(target = cls, confidence = lockState.confidence) }
                if (previous?.id != cls.id) _events.trySend(PointEvent.Locked)
            }
        }
    }

    fun toggle() {
        val target = _state.value.target ?: return
        val entityId = target.entityId ?: return
        val ha = container.ha()
        if (ha == null) {
            fail("Set up Home Assistant in Settings")
            return
        }
        _state.update { it.copy(toggling = true) }
        viewModelScope.launch {
            try {
                val newState = ha.toggle(entityId)
                _state.update { it.copy(toggling = false, message = "${target.friendlyName} → ${newState ?: "toggled"}") }
                _events.send(PointEvent.ToggleOk)
            } catch (e: HaException) {
                _state.update { it.copy(toggling = false) }
                fail(e.message ?: "Toggle failed")
            }
        }
    }

    private fun fail(message: String) {
        _state.update { it.copy(message = message) }
        _events.trySend(PointEvent.Error)
    }

    fun clearMessage() = _state.update { it.copy(message = null) }

    override fun onCleared() {
        synchronized(lock) {
            classifier?.close()
            classifier = null
        }
    }
}
```

- [ ] **Step 2: Implement the screen**

File: `android/app/src/main/java/dev/pointremote/ui/point/PointScreen.kt`
```kotlin
package dev.pointremote.ui.point

import android.os.Build
import android.view.HapticFeedbackConstants
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import dev.pointremote.AppContainer
import dev.pointremote.camera.CameraPreview
import dev.pointremote.camera.WithCameraPermission
import kotlinx.coroutines.delay
import kotlin.math.roundToInt

private val CONFIRM = if (Build.VERSION.SDK_INT >= 30) HapticFeedbackConstants.CONFIRM else HapticFeedbackConstants.LONG_PRESS
private val REJECT = if (Build.VERSION.SDK_INT >= 30) HapticFeedbackConstants.REJECT else HapticFeedbackConstants.LONG_PRESS

@Composable
fun PointScreen(container: AppContainer, onOpenCapture: () -> Unit) {
    val vm: PointViewModel = viewModel { PointViewModel(container) }
    val state by vm.state.collectAsStateWithLifecycle()
    val view = LocalView.current

    LaunchedEffect(vm) {
        vm.events.collect { event ->
            view.performHapticFeedback(
                when (event) {
                    PointEvent.Locked -> HapticFeedbackConstants.CLOCK_TICK
                    PointEvent.ToggleOk -> CONFIRM
                    PointEvent.Error -> REJECT
                },
            )
        }
    }
    LaunchedEffect(state.message) {
        if (state.message != null) {
            delay(3000)
            vm.clearMessage()
        }
    }

    if (!state.hasModel) {
        Box(Modifier.fillMaxSize().padding(24.dp), contentAlignment = Alignment.Center) {
            Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(16.dp)) {
                Text("Capture and train first", style = MaterialTheme.typography.titleLarge)
                state.message?.let { Text(it, textAlign = TextAlign.Center) }
                Button(onClick = onOpenCapture) { Text("Go to Capture") }
            }
        }
    } else {
        WithCameraPermission {
            Box(Modifier.fillMaxSize()) {
                CameraPreview(onFrame = vm::analyze, modifier = Modifier.fillMaxSize())
                state.target?.let { target ->
                    Surface(
                        modifier = Modifier.align(Alignment.TopCenter).padding(top = 24.dp),
                        shape = RoundedCornerShape(50),
                        color = MaterialTheme.colorScheme.primaryContainer,
                    ) {
                        Text(
                            "${target.friendlyName} · ${(state.confidence * 100).roundToInt()}%",
                            modifier = Modifier.padding(horizontal = 20.dp, vertical = 10.dp),
                            style = MaterialTheme.typography.titleMedium,
                        )
                    }
                }
                Column(
                    modifier = Modifier.align(Alignment.BottomCenter).fillMaxWidth().padding(24.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    state.message?.let {
                        Surface(shape = RoundedCornerShape(12.dp), color = MaterialTheme.colorScheme.surfaceVariant) {
                            Text(it, modifier = Modifier.padding(12.dp).fillMaxWidth(), textAlign = TextAlign.Center)
                        }
                    }
                    Button(
                        onClick = vm::toggle,
                        enabled = state.target != null && !state.toggling,
                        modifier = Modifier.fillMaxWidth().height(72.dp),
                    ) {
                        Text(if (state.toggling) "…" else "Toggle", style = MaterialTheme.typography.headlineSmall)
                    }
                }
            }
        }
    }
}
```

- [ ] **Step 3: Build**

Run: `./gradlew --console=plain -q assembleDebug`
Expected: exits 0.

- [ ] **Step 4: Commit**

```bash
git add android/app/src
git commit -m "feat(android): point screen with live recognition and toggle"
```

---

### Task 9: Capture screen

**Files:**
- Create: `ui/capture/CaptureViewModel.kt`, `ui/capture/CaptureScreen.kt`

**Interfaces:**
- Consumes: `AppContainer`, `ServerClient.labels/createLabel`, `HaClient.entities`, `PhotoQueue`, frame conversion, `CameraPreview`, `WithCameraPermission`.
- Produces: `val UNKNOWN_LABEL: ServerLabel`; `data class CaptureUiState(labels, selected, serverOnline, message, entities: List<HaEntity>?, capturing, sessionCount)`; `class CaptureViewModel(container)` with `state`, `pending: StateFlow<Map<String, Int>>`, `refresh()`, `select(id)`, `setCapturing(Boolean)`, `analyze(ImageProxy)`, `startNewLabel()`, `cancelNewLabel()`, `createLabel(HaEntity)`; `@Composable fun CaptureScreen(container: AppContainer)`.

- [ ] **Step 1: Implement the view model**

File: `android/app/src/main/java/dev/pointremote/ui/capture/CaptureViewModel.kt`
```kotlin
package dev.pointremote.ui.capture

import android.os.SystemClock
import androidx.camera.core.ImageProxy
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dev.pointremote.AppContainer
import dev.pointremote.camera.toCaptureJpeg
import dev.pointremote.camera.uprightBitmap
import dev.pointremote.net.HaEntity
import dev.pointremote.net.HaException
import dev.pointremote.net.ServerException
import dev.pointremote.net.ServerLabel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

val UNKNOWN_LABEL = ServerLabel("unknown", null, "Unknown", 0)

data class CaptureUiState(
    val labels: List<ServerLabel> = listOf(UNKNOWN_LABEL),
    val selected: String = UNKNOWN_LABEL.id,
    val serverOnline: Boolean = false,
    val message: String? = null,
    val entities: List<HaEntity>? = null,
    val capturing: Boolean = false,
    val sessionCount: Int = 0,
)

private const val CAPTURE_INTERVAL_MS = 333L

class CaptureViewModel(private val container: AppContainer) : ViewModel() {
    private val _state = MutableStateFlow(CaptureUiState())
    val state: StateFlow<CaptureUiState> = _state.asStateFlow()
    val pending: StateFlow<Map<String, Int>> = container.photoQueue.counts

    @Volatile private var capturing = false
    @Volatile private var lastShot = 0L

    init {
        refresh()
    }

    fun refresh() {
        viewModelScope.launch {
            val server = container.server()
            if (server == null) {
                _state.update { it.copy(serverOnline = false, message = "Set up the PC server in Settings") }
                return@launch
            }
            try {
                val labels = server.labels()
                _state.update { s ->
                    s.copy(
                        labels = labels,
                        serverOnline = true,
                        message = null,
                        selected = if (labels.any { it.id == s.selected }) s.selected else UNKNOWN_LABEL.id,
                    )
                }
            } catch (e: ServerException) {
                _state.update { it.copy(serverOnline = false, message = e.message) }
            }
            container.photoQueue.refresh()
        }
    }

    fun select(id: String) = _state.update { it.copy(selected = id) }

    fun setCapturing(on: Boolean) {
        capturing = on
        _state.update { it.copy(capturing = on, sessionCount = if (on) 0 else it.sessionCount) }
        if (!on) refresh()
    }

    /** Runs on the camera executor. */
    fun analyze(image: ImageProxy) {
        if (!capturing) return
        val now = SystemClock.elapsedRealtime()
        if (now - lastShot < CAPTURE_INTERVAL_MS) return
        lastShot = now
        val jpeg = image.uprightBitmap().toCaptureJpeg()
        container.photoQueue.enqueue(_state.value.selected, jpeg)
        _state.update { it.copy(sessionCount = it.sessionCount + 1) }
    }

    fun startNewLabel() {
        viewModelScope.launch {
            val ha = container.ha()
            if (ha == null) {
                _state.update { it.copy(message = "Set up Home Assistant in Settings") }
                return@launch
            }
            try {
                val existing = _state.value.labels.mapNotNull { it.entityId }.toSet()
                val entities = ha.entities().filter { it.entityId !in existing }
                _state.update { it.copy(entities = entities, message = null) }
            } catch (e: HaException) {
                _state.update { it.copy(message = e.message) }
            }
        }
    }

    fun cancelNewLabel() = _state.update { it.copy(entities = null) }

    fun createLabel(entity: HaEntity) {
        viewModelScope.launch {
            val server = container.server() ?: return@launch
            try {
                val label = server.createLabel(entity.entityId, entity.friendlyName)
                _state.update { it.copy(entities = null, labels = it.labels + label, selected = label.id) }
            } catch (e: ServerException) {
                _state.update { it.copy(entities = null, message = e.message) }
            }
        }
    }
}
```

- [ ] **Step 2: Implement the screen**

File: `android/app/src/main/java/dev/pointremote/ui/capture/CaptureScreen.kt`
```kotlin
@file:OptIn(ExperimentalMaterial3Api::class)

package dev.pointremote.ui.capture

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.AssistChip
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import dev.pointremote.AppContainer
import dev.pointremote.camera.CameraPreview
import dev.pointremote.camera.WithCameraPermission

@Composable
fun CaptureScreen(container: AppContainer) {
    val vm: CaptureViewModel = viewModel { CaptureViewModel(container) }
    val state by vm.state.collectAsStateWithLifecycle()
    val pending by vm.pending.collectAsStateWithLifecycle()
    LaunchedEffect(Unit) { vm.refresh() }

    WithCameraPermission {
        Box(Modifier.fillMaxSize()) {
            CameraPreview(onFrame = vm::analyze, modifier = Modifier.fillMaxSize())

            Column(
                Modifier.align(Alignment.TopCenter).fillMaxWidth()
                    .background(MaterialTheme.colorScheme.surface.copy(alpha = 0.85f))
                    .padding(vertical = 8.dp),
            ) {
                LazyRow(contentPadding = PaddingValues(horizontal = 12.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    items(state.labels, key = { it.id }) { label ->
                        val waiting = pending[label.id] ?: 0
                        val count = if (waiting > 0) "${label.photoCount}+$waiting" else "${label.photoCount}"
                        FilterChip(
                            selected = label.id == state.selected,
                            onClick = { vm.select(label.id) },
                            label = { Text("${label.friendlyName} ($count)") },
                        )
                    }
                    item { AssistChip(onClick = vm::startNewLabel, label = { Text("+ New label") }) }
                }
                state.message?.let {
                    Text(it, modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp), color = MaterialTheme.colorScheme.error)
                }
            }

            Column(
                Modifier.align(Alignment.BottomCenter).padding(bottom = 32.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Surface(shape = RoundedCornerShape(12.dp), color = MaterialTheme.colorScheme.surface.copy(alpha = 0.85f)) {
                    Text(
                        if (state.capturing) "Capturing… ${state.sessionCount}" else "Hold to capture · ${state.sessionCount} last burst",
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
                    )
                }
                Box(
                    Modifier.size(88.dp).clip(CircleShape)
                        .background(if (state.capturing) Color(0xFFE53935) else Color.White)
                        .pointerInput(Unit) {
                            detectTapGestures(onPress = {
                                vm.setCapturing(true)
                                tryAwaitRelease()
                                vm.setCapturing(false)
                            })
                        },
                )
            }
        }
    }

    state.entities?.let { entities ->
        AlertDialog(
            onDismissRequest = vm::cancelNewLabel,
            title = { Text("Pick a Home Assistant entity") },
            text = {
                if (entities.isEmpty()) {
                    Text("No controllable entities left to add.")
                } else {
                    LazyColumn(Modifier.heightIn(max = 400.dp)) {
                        items(entities, key = { it.entityId }) { entity ->
                            Column(Modifier.fillMaxWidth().clickable { vm.createLabel(entity) }.padding(vertical = 10.dp)) {
                                Text(entity.friendlyName, style = MaterialTheme.typography.bodyLarge)
                                Text(entity.entityId, style = MaterialTheme.typography.bodySmall)
                            }
                        }
                    }
                }
            },
            confirmButton = {},
            dismissButton = { TextButton(onClick = vm::cancelNewLabel) { Text("Cancel") } },
        )
    }
}
```

- [ ] **Step 3: Build**

Run: `./gradlew --console=plain -q assembleDebug`
Expected: exits 0.

- [ ] **Step 4: Commit**

```bash
git add android/app/src
git commit -m "feat(android): capture screen with burst capture and new labels"
```

---

### Task 10: Train screen

**Files:**
- Create: `ui/train/TrainViewModel.kt`, `ui/train/TrainScreen.kt`

**Interfaces:**
- Consumes: `AppContainer`, `ServerClient.health/labels/trainStatus/startTraining`, `ModelStore.updateFrom/installed`.
- Produces: `const val MIN_PHOTOS = 30`, `const val RECOMMENDED_PHOTOS = 100`; `data class TrainUiState(online: Boolean?, labels, status: TrainStatus?, meta: ModelMeta?, message: String?, busy: Boolean)` with `canTrain`; `class TrainViewModel(container)` with `state`, `refresh()`, `train()`; `@Composable fun TrainScreen(container: AppContainer)`.

- [ ] **Step 1: Implement the view model**

File: `android/app/src/main/java/dev/pointremote/ui/train/TrainViewModel.kt`
```kotlin
package dev.pointremote.ui.train

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dev.pointremote.AppContainer
import dev.pointremote.model.ModelMeta
import dev.pointremote.model.UpdateResult
import dev.pointremote.net.ServerClient
import dev.pointremote.net.ServerException
import dev.pointremote.net.ServerLabel
import dev.pointremote.net.TrainStatus
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

const val MIN_PHOTOS = 30
const val RECOMMENDED_PHOTOS = 100

data class TrainUiState(
    val online: Boolean? = null,
    val labels: List<ServerLabel> = emptyList(),
    val status: TrainStatus? = null,
    val meta: ModelMeta? = null,
    val message: String? = null,
    val busy: Boolean = false,
) {
    val canTrain: Boolean
        get() = online == true && !busy && status?.running != true &&
            labels.size >= 2 && labels.all { it.photoCount >= MIN_PHOTOS }
}

class TrainViewModel(private val container: AppContainer) : ViewModel() {
    private val _state = MutableStateFlow(TrainUiState())
    val state: StateFlow<TrainUiState> = _state.asStateFlow()
    private var pollJob: Job? = null

    init {
        viewModelScope.launch { container.modelStore.installed.collect { m -> _state.update { it.copy(meta = m) } } }
        refresh()
    }

    fun refresh() {
        viewModelScope.launch {
            val server = container.server()
            if (server == null) {
                _state.update { it.copy(online = false, message = "Set up the PC server in Settings") }
                return@launch
            }
            try {
                server.health()
                val labels = server.labels()
                val status = server.trainStatus()
                _state.update { it.copy(online = true, labels = labels, status = status, message = null) }
                if (status.running) startPolling(server) else syncModel(server)
            } catch (e: ServerException) {
                _state.update { it.copy(online = false, message = e.message) }
            }
        }
    }

    fun train() {
        val server = container.server() ?: return
        viewModelScope.launch {
            _state.update { it.copy(busy = true, message = null) }
            try {
                val status = server.startTraining()
                _state.update { it.copy(status = status) }
                startPolling(server)
            } catch (e: ServerException) {
                _state.update { it.copy(message = e.message) }
            } finally {
                _state.update { it.copy(busy = false) }
            }
        }
    }

    private fun startPolling(server: ServerClient) {
        if (pollJob?.isActive == true) return
        pollJob = viewModelScope.launch {
            try {
                while (true) {
                    val status = server.trainStatus()
                    _state.update { it.copy(status = status) }
                    if (!status.running) {
                        if (status.state == "done") syncModel(server)
                        if (status.state == "failed") _state.update { it.copy(message = "Training failed: ${status.message}") }
                        break
                    }
                    delay(2000)
                }
            } catch (e: ServerException) {
                _state.update { it.copy(message = e.message) }
            }
        }
    }

    private suspend fun syncModel(server: ServerClient) {
        try {
            val result = container.modelStore.updateFrom(server)
            if (result is UpdateResult.Updated) {
                _state.update { it.copy(message = "Model v${result.meta.version} installed") }
            }
        } catch (e: Exception) {
            _state.update { it.copy(message = "Model download failed: ${e.message}") }
        }
    }
}
```

- [ ] **Step 2: Implement the screen**

File: `android/app/src/main/java/dev/pointremote/ui/train/TrainScreen.kt`
```kotlin
package dev.pointremote.ui.train

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import dev.pointremote.AppContainer
import kotlin.math.roundToInt

private fun pct(x: Double) = "${(x * 100).roundToInt()}%"

@Composable
fun TrainScreen(container: AppContainer) {
    val vm: TrainViewModel = viewModel { TrainViewModel(container) }
    val state by vm.state.collectAsStateWithLifecycle()
    LaunchedEffect(Unit) { vm.refresh() }

    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Text(
                when (state.online) {
                    true -> "PC server online"
                    false -> "PC offline"
                    null -> "Checking PC server…"
                },
                style = MaterialTheme.typography.titleMedium,
            )
            OutlinedButton(onClick = vm::refresh) { Text("Refresh") }
        }

        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Text("Photos per label", style = MaterialTheme.typography.titleMedium)
                if (state.labels.size < 2) Text("Add at least one device label in Capture.")
                state.labels.forEach { label ->
                    val color = when {
                        label.photoCount < MIN_PHOTOS -> MaterialTheme.colorScheme.error
                        label.photoCount < RECOMMENDED_PHOTOS -> Color(0xFFF9A825)
                        else -> MaterialTheme.colorScheme.onSurface
                    }
                    val hint = when {
                        label.photoCount < MIN_PHOTOS -> " (needs $MIN_PHOTOS)"
                        label.photoCount < RECOMMENDED_PHOTOS -> " (aim for $RECOMMENDED_PHOTOS)"
                        else -> ""
                    }
                    Text("${label.friendlyName}: ${label.photoCount}$hint", color = color)
                }
            }
        }

        Button(onClick = vm::train, enabled = state.canTrain, modifier = Modifier.fillMaxWidth()) { Text("Train") }

        state.status?.let { status ->
            if (status.running) {
                val progress = if (status.epochs > 0) status.epoch.toFloat() / status.epochs else 0f
                LinearProgressIndicator(progress = { progress }, modifier = Modifier.fillMaxWidth())
                Text(status.message)
            }
        }
        state.message?.let { Text(it) }

        state.meta?.let { meta ->
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text("Installed model v${meta.version}", style = MaterialTheme.typography.titleMedium)
                    Text("Validation accuracy: ${pct(meta.valAccuracy)}")
                    meta.classes.forEach { cls ->
                        meta.perClassAccuracy[cls.id]?.let { Text("• ${cls.friendlyName}: ${pct(it)}") }
                    }
                    if (meta.confusions.isNotEmpty()) {
                        Text("Most confused", style = MaterialTheme.typography.titleSmall)
                        val names = meta.classes.associate { it.id to it.friendlyName }
                        meta.confusions.forEach { c ->
                            Text("• ${names[c.trueId] ?: c.trueId} seen as ${names[c.predicted] ?: c.predicted}: ${c.count}")
                        }
                    }
                }
            }
        }
    }
}
```

- [ ] **Step 3: Build**

Run: `./gradlew --console=plain -q assembleDebug`
Expected: exits 0.

- [ ] **Step 4: Commit**

```bash
git add android/app/src
git commit -m "feat(android): train screen with progress and model report"
```

---

### Task 11: Settings screen, navigation and launcher icon

**Files:**
- Create: `ui/settings/SettingsScreen.kt`, `ui/AppRoot.kt`, `app/src/main/res/drawable/ic_point.xml`, `app/src/main/res/values/strings.xml`
- Modify: `MainActivity.kt` (replace), `AndroidManifest.xml` (icon)

**Interfaces:**
- Consumes: `AppContainer`, `AppSettings`, `HaClient`, `ServerClient`, all screens.
- Produces: `@Composable fun SettingsScreen(container: AppContainer)`; `enum class Screen`; `@Composable fun AppRoot(container: AppContainer)`; resources `@drawable/ic_point`, `@string/app_name`, `@string/widget_description`.

- [ ] **Step 1: Implement the settings screen**

File: `android/app/src/main/java/dev/pointremote/ui/settings/SettingsScreen.kt`
```kotlin
package dev.pointremote.ui.settings

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import dev.pointremote.AppContainer
import dev.pointremote.net.HaClient
import dev.pointremote.net.HaException
import dev.pointremote.net.ServerClient
import dev.pointremote.net.ServerException
import kotlinx.coroutines.launch
import kotlin.math.roundToInt

@Composable
fun SettingsScreen(container: AppContainer) {
    val saved by container.settings.settings.collectAsStateWithLifecycle()
    var draft by remember(saved) { mutableStateOf(saved) }
    var haResult by remember { mutableStateOf<String?>(null) }
    var serverResult by remember { mutableStateOf<String?>(null) }
    val scope = rememberCoroutineScope()

    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("Home Assistant", style = MaterialTheme.typography.titleMedium)
        OutlinedTextField(
            value = draft.haUrl, onValueChange = { draft = draft.copy(haUrl = it) },
            label = { Text("URL, e.g. http://192.168.1.10:8123") }, singleLine = true,
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri), modifier = Modifier.fillMaxWidth(),
        )
        OutlinedTextField(
            value = draft.haToken, onValueChange = { draft = draft.copy(haToken = it) },
            label = { Text("Long-lived access token") }, singleLine = true,
            visualTransformation = PasswordVisualTransformation(), modifier = Modifier.fillMaxWidth(),
        )
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            OutlinedButton(onClick = {
                haResult = "Testing…"
                scope.launch {
                    haResult = try {
                        "Connected: ${HaClient(draft.haUrl, draft.haToken, container.http).entities().size} controllable entities"
                    } catch (e: HaException) {
                        e.message
                    } catch (e: IllegalArgumentException) {
                        "Invalid URL"
                    }
                }
            }) { Text("Test") }
            haResult?.let { Text(it) }
        }

        Text("PC training server", style = MaterialTheme.typography.titleMedium)
        OutlinedTextField(
            value = draft.serverUrl, onValueChange = { draft = draft.copy(serverUrl = it) },
            label = { Text("URL, e.g. http://192.168.1.20:8765") }, singleLine = true,
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri), modifier = Modifier.fillMaxWidth(),
        )
        OutlinedTextField(
            value = draft.serverKey, onValueChange = { draft = draft.copy(serverKey = it) },
            label = { Text("Access key") }, singleLine = true,
            visualTransformation = PasswordVisualTransformation(), modifier = Modifier.fillMaxWidth(),
        )
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            OutlinedButton(onClick = {
                serverResult = "Testing…"
                scope.launch {
                    serverResult = try {
                        val health = ServerClient(draft.serverUrl, draft.serverKey, container.http).health()
                        "Connected: " + (health.latestVersion?.let { "model v$it" } ?: "no model yet")
                    } catch (e: ServerException) {
                        e.message
                    } catch (e: IllegalArgumentException) {
                        "Invalid URL"
                    }
                }
            }) { Text("Test") }
            serverResult?.let { Text(it) }
        }

        Text("Recognition", style = MaterialTheme.typography.titleMedium)
        Text("Confidence threshold: ${(draft.threshold * 100).roundToInt()}%")
        Slider(value = draft.threshold, onValueChange = { draft = draft.copy(threshold = it) }, valueRange = 0.5f..0.99f)
        Text("Stable frames before lock: ${draft.stableFrames}")
        Slider(
            value = draft.stableFrames.toFloat(),
            onValueChange = { draft = draft.copy(stableFrames = it.roundToInt()) },
            valueRange = 2f..15f, steps = 12,
        )

        Button(onClick = { container.settings.save(draft) }, enabled = draft != saved, modifier = Modifier.fillMaxWidth()) {
            Text(if (draft == saved) "Saved" else "Save")
        }
    }
}
```

- [ ] **Step 2: Implement navigation and the activity**

File: `android/app/src/main/java/dev/pointremote/ui/AppRoot.kt`
```kotlin
package dev.pointremote.ui

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import dev.pointremote.AppContainer
import dev.pointremote.ui.capture.CaptureScreen
import dev.pointremote.ui.point.PointScreen
import dev.pointremote.ui.settings.SettingsScreen
import dev.pointremote.ui.train.TrainScreen

enum class Screen(val label: String, val glyph: String) {
    Point("Point", "◎"),
    Capture("Capture", "◉"),
    Train("Train", "▲"),
    Settings("Settings", "⚙"),
}

@Composable
fun AppRoot(container: AppContainer) {
    var screen by rememberSaveable { mutableStateOf(Screen.Point) }
    MaterialTheme(colorScheme = if (isSystemInDarkTheme()) darkColorScheme() else lightColorScheme()) {
        Scaffold(
            bottomBar = {
                NavigationBar {
                    Screen.entries.forEach { s ->
                        NavigationBarItem(
                            selected = screen == s,
                            onClick = { screen = s },
                            icon = { Text(s.glyph) },
                            label = { Text(s.label) },
                        )
                    }
                }
            },
        ) { padding ->
            Box(Modifier.padding(padding)) {
                when (screen) {
                    Screen.Point -> PointScreen(container, onOpenCapture = { screen = Screen.Capture })
                    Screen.Capture -> CaptureScreen(container)
                    Screen.Train -> TrainScreen(container)
                    Screen.Settings -> SettingsScreen(container)
                }
            }
        }
    }
}
```

File: `android/app/src/main/java/dev/pointremote/MainActivity.kt`
```kotlin
package dev.pointremote

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.lifecycle.lifecycleScope
import dev.pointremote.ui.AppRoot
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        val container = (application as PointRemoteApp).container
        // Check for a newer model on every launch; failure is silent (the installed model keeps working).
        lifecycleScope.launch {
            container.server()?.let { server -> runCatching { container.modelStore.updateFrom(server) } }
        }
        setContent { AppRoot(container) }
    }
}
```

- [ ] **Step 3: Add resources and the icon**

File: `android/app/src/main/res/drawable/ic_point.xml`
```xml
<vector xmlns:android="http://schemas.android.com/apk/res/android"
    android:width="24dp"
    android:height="24dp"
    android:viewportWidth="24"
    android:viewportHeight="24">
    <path
        android:fillColor="#FF1E88E5"
        android:pathData="M12,8c-2.21,0 -4,1.79 -4,4s1.79,4 4,4 4,-1.79 4,-4 -1.79,-4 -4,-4zM20.94,11c-0.46,-4.17 -3.77,-7.48 -7.94,-7.94L13,1h-2v2.06C6.83,3.52 3.52,6.83 3.06,11L1,11v2h2.06c0.46,4.17 3.77,7.48 7.94,7.94L11,23h2v-2.06c4.17,-0.46 7.48,-3.77 7.94,-7.94L23,13v-2h-2.06zM12,19c-3.87,0 -7,-3.13 -7,-7s3.13,-7 7,-7 7,3.13 7,7 -3.13,7 -7,7z" />
</vector>
```

File: `android/app/src/main/res/values/strings.xml`
```xml
<resources>
    <string name="app_name">PointRemote</string>
    <string name="widget_description">Open PointRemote and point at something to control it</string>
    <string name="tile_label">Point</string>
</resources>
```

In `AndroidManifest.xml`, change the `<application` attributes `android:label="PointRemote"` to:
```xml
        android:label="@string/app_name"
        android:icon="@drawable/ic_point"
```

- [ ] **Step 4: Build and run unit tests**

Run: `./gradlew --console=plain -q testDebugUnitTest assembleDebug`
Expected: exits 0.

- [ ] **Step 5: Commit**

```bash
git add android/app/src
git commit -m "feat(android): settings screen, navigation and icon"
```

---

### Task 12: Home-screen widget and Quick Settings tile

**Files:**
- Create: `launch/PointWidget.kt`, `launch/PointTileService.kt`, `app/src/main/res/xml/point_widget_info.xml`
- Modify: `AndroidManifest.xml` (receiver + service)

**Interfaces:**
- Consumes: `MainActivity`, `@drawable/ic_point`, strings.
- Produces: `PointWidget`, `PointWidgetReceiver`, `PointTileService` — all open `MainActivity` (which starts on the Point screen).

- [ ] **Step 1: Implement**

File: `android/app/src/main/java/dev/pointremote/launch/PointWidget.kt`
```kotlin
package dev.pointremote.launch

import android.content.Context
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.glance.GlanceId
import androidx.glance.GlanceModifier
import androidx.glance.action.actionStartActivity
import androidx.glance.action.clickable
import androidx.glance.appwidget.GlanceAppWidget
import androidx.glance.appwidget.GlanceAppWidgetReceiver
import androidx.glance.appwidget.cornerRadius
import androidx.glance.appwidget.provideContent
import androidx.glance.background
import androidx.glance.layout.Alignment
import androidx.glance.layout.Box
import androidx.glance.layout.fillMaxSize
import androidx.glance.text.FontWeight
import androidx.glance.text.Text
import androidx.glance.text.TextStyle
import androidx.glance.unit.ColorProvider
import dev.pointremote.MainActivity

class PointWidget : GlanceAppWidget() {
    override suspend fun provideGlance(context: Context, id: GlanceId) {
        provideContent {
            Box(
                modifier = GlanceModifier.fillMaxSize()
                    .background(Color(0xFF1E88E5))
                    .cornerRadius(24.dp)
                    .clickable(actionStartActivity<MainActivity>()),
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    "◎  Point",
                    style = TextStyle(color = ColorProvider(Color.White), fontSize = 20.sp, fontWeight = FontWeight.Bold),
                )
            }
        }
    }
}

class PointWidgetReceiver : GlanceAppWidgetReceiver() {
    override val glanceAppWidget: GlanceAppWidget = PointWidget()
}
```

File: `android/app/src/main/java/dev/pointremote/launch/PointTileService.kt`
```kotlin
package dev.pointremote.launch

import android.app.PendingIntent
import android.content.Intent
import android.os.Build
import android.service.quicksettings.Tile
import android.service.quicksettings.TileService
import dev.pointremote.MainActivity

class PointTileService : TileService() {
    override fun onStartListening() {
        qsTile?.apply {
            state = Tile.STATE_INACTIVE
            updateTile()
        }
    }

    override fun onClick() {
        val intent = Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        if (Build.VERSION.SDK_INT >= 34) {
            startActivityAndCollapse(PendingIntent.getActivity(this, 0, intent, PendingIntent.FLAG_IMMUTABLE))
        } else {
            @Suppress("DEPRECATION")
            startActivityAndCollapse(intent)
        }
    }
}
```

File: `android/app/src/main/res/xml/point_widget_info.xml`
```xml
<?xml version="1.0" encoding="utf-8"?>
<appwidget-provider xmlns:android="http://schemas.android.com/apk/res/android"
    android:description="@string/widget_description"
    android:initialLayout="@layout/glance_default_loading_layout"
    android:minWidth="110dp"
    android:minHeight="50dp"
    android:resizeMode="horizontal|vertical"
    android:targetCellWidth="2"
    android:targetCellHeight="1"
    android:updatePeriodMillis="0"
    android:widgetCategory="home_screen" />
```

In `AndroidManifest.xml`, insert before `</application>`:
```xml
        <receiver
            android:name=".launch.PointWidgetReceiver"
            android:exported="true">
            <intent-filter>
                <action android:name="android.appwidget.action.APPWIDGET_UPDATE" />
            </intent-filter>
            <meta-data
                android:name="android.appwidget.provider"
                android:resource="@xml/point_widget_info" />
        </receiver>

        <service
            android:name=".launch.PointTileService"
            android:exported="true"
            android:icon="@drawable/ic_point"
            android:label="@string/tile_label"
            android:permission="android.permission.BIND_QUICK_SETTINGS_TILE">
            <intent-filter>
                <action android:name="android.service.quicksettings.action.QS_TILE" />
            </intent-filter>
        </service>
```

- [ ] **Step 2: Build, test, lint**

Run: `./gradlew --console=plain -q testDebugUnitTest assembleDebug lintDebug`
Expected: exits 0 (lint warnings are acceptable; errors are not).

- [ ] **Step 3: Commit**

```bash
git add android/app/src
git commit -m "feat(android): home-screen widget and Quick Settings tile"
```

---

### Task 13: On-device acceptance (needs the user)

- [ ] **Step 1: Ask the user before installing.** With their OK: `adb install -r android/app/build/outputs/apk/debug/app-debug.apk`.
- [ ] **Step 2: Setup.** Start the PC server (`server/run.sh`), make sure the phone can reach `http://<pc-ip>:8765/ui` (the firewall is the user's call), and fill in Settings (HA URL + token, server URL + key); both Test buttons should report "Connected".
- [ ] **Step 3: Capture and train.** Add each device label from HA in Capture, take ~100 photos per device and per "unknown" (varied angles, distances, lighting), then Train. Expect ≥ 95% validation accuracy and the model to install automatically.
- [ ] **Step 4: Point.** Add the widget and the Quick Settings tile; from each, time widget tap → first lock (< 1.5 s). For each device, 20 point-and-toggle attempts: record correct locks and wrong-device locks (target: no wrong-device toggles, ≥ 90 % correct locks).
- [ ] **Step 5: Failure paths.** With the PC server stopped, Point still works and Capture queues (counts show `+N`), and uploads drain when the server is back. With a wrong HA token, Toggle shows "Home Assistant rejected the token" with an error buzz.
