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
