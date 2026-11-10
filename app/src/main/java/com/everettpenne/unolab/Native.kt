package com.everettpenne.unolab

/** Receives each complete piece of UTF-8 the model produces. */
fun interface PieceSink { fun onPiece(bytes: ByteArray) }

/** The model, through llama.cpp on the CPU (see llm_jni.cpp). Not thread-safe: call it from one thread. */
object Llm {
    init { System.loadLibrary("unolab") }
    external fun nativeLoad(path: String, nCtx: Int, threads: Int): Long
    external fun nativeFree(handle: Long)
    external fun nativeReset(handle: Long)
    external fun nativeUsed(handle: Long): Int
    external fun nativeCtx(handle: Long): Int
    external fun nativeStop(handle: Long)
    /** {tokens, ms} */
    external fun nativePrefill(handle: Long, text: String): DoubleArray
    /** {tokens, ms, endedNaturally} */
    external fun nativeGenerate(handle: Long, maxTokens: Int, temp: Float, topP: Float, topK: Int, sink: PieceSink): DoubleArray
    external fun nativeSystemInfo(): String
}

/** The NPU, through raw NNAPI (see npu_jni.cpp). */
object Npu {
    init { System.loadLibrary("unolab") }
    /** "name|TYPE|featureLevel" per device */
    external fun nativeDevices(): Array<String>
    /** {ok, compileMs, firstMs, bestMs, weightBytes}; detail[0] gets the check or the error */
    external fun nativeGemm(index: Int, k: Int, n: Int, iters: Int, batch: Int, detail: Array<String?>): DoubleArray
}
