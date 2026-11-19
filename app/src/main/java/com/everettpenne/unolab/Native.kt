package com.everettpenne.unolab

/** Receives each complete piece of UTF-8 the model produces. */
fun interface PieceSink { fun onPiece(bytes: ByteArray) }

/** The model, through llama.cpp on the CPU (see llm_jni.cpp). Not thread-safe: call it from one thread. */
object Llm {
    init { System.loadLibrary("unolab") }
    external fun nativeLoad(path: String, nCtx: Int, threads: Int): Long
    external fun nativeFree(handle: Long)
    external fun nativeReset(handle: Long)
    /** Another context on [handle]'s model; free it before [handle]. */
    external fun nativeNewContext(handle: Long, nCtx: Int, threads: Int): Long
    /** The planner's LoRA for [handle]'s model; 0 when it does not load (trained on other weights). */
    external fun nativeLoadAdapter(handle: Long, path: String): Long
    /** Applies [adapter] to context [handle] at [scale], or no adapter at all when [adapter] is 0. */
    external fun nativeUseAdapter(handle: Long, adapter: Long, scale: Float): Boolean
    /** Frees an adapter before its model is freed (the model frees any left). */
    external fun nativeFreeAdapter(adapter: Long)
    external fun nativeSaveState(handle: Long): ByteArray
    external fun nativeRestoreState(handle: Long, state: ByteArray, nPast: Int): Boolean
    external fun nativeUsed(handle: Long): Int
    external fun nativeCtx(handle: Long): Int
    external fun nativeStop(handle: Long)
    /** {tokens, ms} */
    external fun nativePrefill(handle: Long, text: String): DoubleArray
    /** {tokens, ms, endedNaturally} */
    external fun nativeGenerate(handle: Long, maxTokens: Int, temp: Float, topP: Float, topK: Int, grammar: String?, sink: PieceSink): DoubleArray
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
