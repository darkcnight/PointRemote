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
