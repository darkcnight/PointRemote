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
