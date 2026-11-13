package com.everettpenne.unolab

import android.content.Context
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.security.MessageDigest
import java.util.concurrent.Executors

/**
 * The model, for the whole process: the chat screen and the assistant service share one copy of the weights, loaded once. llama.cpp is not
 * thread-safe, so every call to it runs on [dispatcher]'s one thread. There are two contexts on the weights: the chat's conversation ([chat])
 * and the assistant's one-shot questions ([helper]), so neither resets the other.
 *
 * Whoever needs the model holds it ([acquire] / [release]); with nobody holding it for [IDLE_MS] it is freed, since the weights and their
 * repacked copy take about 2.5 GB.
 */
object AssistantEngine {
    sealed interface State {
        data object Missing : State
        data class Loading(val name: String) : State
        data class Ready(val name: String, val ctx: Int) : State
        data class Failed(val why: String) : State
    }

    interface Listener {
        fun onPiece(piece: String)
        fun onDone(ms: Long, tokens: Int)
        fun onError(why: String)
    }

    const val CONTEXT = 4096
    /** The assistant's context: a system prompt, up to 2,000 characters of phone context, and one short question and answer. */
    private const val HELPER_CONTEXT = 2048
    /**
     * How long the weights stay loaded with nobody holding them. Search is opened in bursts; a 3-minute window meant the next question
     * after a short pause paid the whole model load again. The weights (and their repacked copy) are ~2.5 GB, so this is a real trade,
     * but the launcher is a foreground app and Android reclaims the process under pressure anyway.
     */
    const val IDLE_MS = 10 * 60_000L
    /** A saved prefix state is the KV and recurrent state for a few hundred tokens: a few MB. Anything larger is not worth keeping. */
    private const val MAX_PREFIX_BYTES = 64L * 1024 * 1024
    private const val TAG = "UnoLabTiming"
    private const val PLAN_TAG = "UnoLabPlan"

    val dispatcher = Executors.newSingleThreadExecutor { r -> Thread(r, "llm").apply { priority = Thread.MAX_PRIORITY } }.asCoroutineDispatcher()
    private val scope = CoroutineScope(SupervisorJob() + dispatcher)
    /** Writing a prefix state must not sit in front of the next question on the model's thread. */
    private val io = Executors.newSingleThreadExecutor { r -> Thread(r, "prefix-io") }.asCoroutineDispatcher()
    private val ioScope = CoroutineScope(SupervisorJob() + io)
    private lateinit var store: ModelStore
    private lateinit var appContext: Context

    private val _state = MutableStateFlow<State>(State.Missing)
    val state: StateFlow<State> = _state

    /** The most threads the model is given; the native side uses one per fast core the process is allowed at that moment (three in
     *  the background on a Tensor G4, four in front), in a pool it keeps. Never more than there are cores. */
    @Volatile var threads = minOf(4, Runtime.getRuntime().availableProcessors())

    /** The chat's context, which owns the weights; 0 when nothing is loaded. Read and used on [dispatcher] only. */
    var chat = 0L; private set
    private var helper = 0L
    /** Each fixed prompt decoded once, keyed by mode ("m0", "m1") or by a planner catalogue hash ("plan-…"): the state after it, and how
     *  many tokens it holds. */
    private val prefixes = HashMap<String, Pair<ByteArray, Int>>()
    /** What the loaded weights are, so a prefix state saved from a different model is never restored into this one. */
    private var modelKey = ""

    private var holders = 0
    private var idleJob: Job? = null
    private var askJob: Job? = null

    fun init(context: Context) {
        if (!::store.isInitialized) {
            appContext = context.applicationContext
            store = ModelStore(appContext)
        }
    }

    @Synchronized fun acquire() { holders++; idleJob?.cancel(); idleJob = null }

    @Synchronized fun release() {
        holders = (holders - 1).coerceAtLeast(0)
        if (holders == 0) { idleJob?.cancel(); idleJob = scope.launch { delay(IDLE_MS); unloadNow() } }
    }

    /** Loads the model if it is not loaded; the state says how that went. Call on [dispatcher]. */
    fun loadNow(): State {
        (_state.value as? State.Ready)?.let { if (chat != 0L) return it }
        // A model pushed to the drop folder is moved in here too, so Uno's search can use it before Uno Lab has been opened.
        if (store.find() == null) store.waiting()?.let { runCatching { store.moveIn(it) { } } }
        val file = store.find() ?: return State.Missing.also { _state.value = it }
        _state.value = State.Loading(file.name)
        val t0 = System.nanoTime()
        val next = runCatching {
            unloadNow(keepState = true)
            chat = Llm.nativeLoad(file.absolutePath, CONTEXT, threads)
            modelKey = "${file.name}:${file.length()}:${file.lastModified()}"
            State.Ready(file.name, Llm.nativeCtx(chat))
        }.getOrElse { State.Failed(it.message ?: it.toString()) }
        _state.value = next
        Log.i(TAG, "model load ${elapsed(t0)} ms -> $next")
        return next
    }

    suspend fun load(): State = withContext(dispatcher) { loadNow() }

    /** [loadNow] from any thread, waiting for it (for tests). */
    fun loadNowBlocking(): State = java.util.concurrent.FutureTask { loadNow() }.also { dispatcher.executor.execute(it) }.get()

    /** Frees the model (the helper first: it does not own the weights). Call on [dispatcher]. */
    fun unloadNow(keepState: Boolean = false) {
        if (helper != 0L) { Llm.nativeFree(helper); helper = 0L }
        if (chat != 0L) { Llm.nativeFree(chat); chat = 0L }
        prefixes.clear()
        if (!keepState) _state.value = State.Missing
    }

    suspend fun unload() = withContext(dispatcher) { unloadNow() }

    fun warm() { scope.launch { loadNow() } }

    /** Stops whatever the assistant is writing. Safe from any thread. */
    fun cancel() {
        val h = helper
        if (h != 0L) Llm.nativeStop(h)
        askJob?.cancel()
    }

    /**
     * Asks the assistant one question in [mode] (see [Assistant]), with optional phone [context] from Uno, replacing any still running;
     * [listener] is called on the model's thread. The context is used for this question only and kept nowhere.
     */
    fun ask(query: String, context: String?, mode: Int, listener: Listener) {
        cancel()
        askJob = scope.launch {
            val t0 = System.nanoTime()
            val job = coroutineContext[Job]
            val state = loadNow()
            if (state !is State.Ready) { listener.onError(if (state is State.Failed) state.why else "No model in Uno Lab"); return@launch }
            runCatching {
                if (helper == 0L) helper = Llm.nativeNewContext(chat, HELPER_CONTEXT, threads)
                startFrom("m$mode", Assistant.prefix(mode))
                val tPre = System.nanoTime()
                val pre = Llm.nativePrefill(helper, Assistant.turn(query, context))
                Log.i(TAG, "prefill ${elapsed(tPre)} ms (${pre[0].toInt()} tok)")
                // A question replaced while it was still being read is dropped here, before anything is written for it.
                if (job?.isActive != true) return@launch
                val h = helper
                var first = 0L
                val gen = Llm.nativeGenerate(h, Assistant.maxTokens(mode), Assistant.temperature(mode), .9f, 20,
                    Assistant.grammar(mode)) { bytes ->
                    if (first == 0L) {
                        first = System.nanoTime()
                        Log.i(TAG, "first token ${elapsed(t0)} ms after the ask")
                    }
                    if (job.isActive) listener.onPiece(String(bytes, Charsets.UTF_8)) else Llm.nativeStop(h)
                }
                if (job.isActive) listener.onDone((pre[1] + gen[1]).toLong(), gen[0].toInt())
                Log.i(TAG, "turn ${elapsed(t0)} ms, mode $mode, ${gen[0].toInt()} tok")
            }.onFailure { listener.onError(it.message ?: it.toString()) }
        }
    }

    /** Puts the helper at the end of a fixed prompt: from the in-memory state, the one saved on disk, or decoded afresh. */
    private fun startFrom(key: String, prefixText: String) {
        val t0 = System.nanoTime()
        prefixes[key]?.let { (bytes, n) ->
            if (Llm.nativeRestoreState(helper, bytes, n)) { Log.i(TAG, "prefix $key from memory ${elapsed(t0)} ms"); return }
        }
        savedPrefix(key, prefixText)?.let { (bytes, n) ->
            if (Llm.nativeRestoreState(helper, bytes, n)) {
                prefixes[key] = bytes to n
                Log.i(TAG, "prefix $key from disk ${elapsed(t0)} ms")
                return
            }
        }
        Llm.nativeReset(helper)
        val pre = Llm.nativePrefill(helper, prefixText)
        val bytes = Llm.nativeSaveState(helper)
        prefixes[key] = bytes to pre[0].toInt()
        Log.i(TAG, "prefix $key decoded ${elapsed(t0)} ms (${pre[0].toInt()} tok, ${bytes.size / 1024} KB)")
        savePrefixAsync(key, prefixText, bytes, pre[0].toInt())
    }

    /**
     * Answers the launcher's planner request: the tool catalogue and the grammar arrive in [requestJson], so a new tool needs no change
     * here. The prompt (instructions and catalogue) is decoded once per catalogue and cached under its hash, like the other prefixes, and the
     * reply is held to the grammar the launcher generated.
     */
    suspend fun plan(requestJson: String): String = withContext(dispatcher) {
        val request = Planner.parse(requestJson) ?: return@withContext Planner.error("I couldn't read the planner request.")
        val state = loadNow()
        if (state !is State.Ready) {
            return@withContext Planner.error(if (state is State.Failed) state.why else "No model in Uno Lab")
        }
        if (helper == 0L) helper = Llm.nativeNewContext(chat, HELPER_CONTEXT, threads)
        val t0 = System.nanoTime()
        val prepared = runCatching {
            startFrom("plan-${request.hash}", Planner.prefix(request))
            Llm.nativePrefill(helper, Planner.turn(request))
        }.getOrElse { return@withContext Planner.error(it.message ?: "The request was too long for the assistant.") }
        val reply = StringBuilder()
        runCatching {
            Llm.nativeGenerate(helper, Planner.MAX_TOKENS, Planner.TEMPERATURE, .95f, 20, request.grammar) { bytes ->
                reply.append(String(bytes, Charsets.UTF_8))
            }
        }.onFailure { return@withContext Planner.error(it.message ?: "The planner failed.") }
        Log.i(TAG, "plan ${elapsed(t0)} ms (${prepared[0].toInt()} tok in, ${reply.length} chars out)")
        // The plan carries what was asked for (a note's words, a number), so it is logged only when a test asks:
        // adb shell setprop log.tag.UnoLabPlan DEBUG
        if (Log.isLoggable(PLAN_TAG, Log.DEBUG)) Log.d(PLAN_TAG, "plan reply: ${reply.toString().trim()}")
        reply.toString().trim()
    }

    private fun elapsed(t0: Long) = (System.nanoTime() - t0) / 1_000_000

    private fun prefixFile(key: String) = File(File(appContext.filesDir, "prefix"), "$key.bin")

    /** A prefix state is only valid for the weights and the prompt it was decoded from. */
    private fun fingerprint(key: String, prefixText: String): String {
        val digest = MessageDigest.getInstance("SHA-256")
        digest.update(modelKey.toByteArray())
        digest.update(byteArrayOf(0))
        digest.update(key.toByteArray())
        digest.update(byteArrayOf(0))
        digest.update(prefixText.toByteArray())
        return digest.digest().joinToString("") { "%02x".format(it) }.take(16)
    }

    private fun savedPrefix(key: String, prefixText: String): Pair<ByteArray, Int>? = runCatching {
        val file = prefixFile(key)
        if (!file.exists()) return@runCatching null
        DataInputStream(file.inputStream().buffered()).use { input ->
            val n = input.readInt()
            val fp = ByteArray(input.readInt()).also { input.readFully(it) }
            if (String(fp, Charsets.UTF_8) != fingerprint(key, prefixText)) return@runCatching null
            val len = input.readInt()
            if (len <= 0 || len > MAX_PREFIX_BYTES) return@runCatching null
            ByteArray(len).also { input.readFully(it) } to n
        }
    }.getOrNull()

    private fun savePrefixAsync(key: String, prefixText: String, bytes: ByteArray, n: Int) {
        if (bytes.size > MAX_PREFIX_BYTES) return
        val fp = fingerprint(key, prefixText).toByteArray()
        val file = prefixFile(key)
        ioScope.launch {
            runCatching {
                file.parentFile?.mkdirs()
                val tmp = File(file.parentFile, file.name + ".tmp")
                DataOutputStream(tmp.outputStream().buffered()).use { out ->
                    out.writeInt(n); out.writeInt(fp.size); out.write(fp); out.writeInt(bytes.size); out.write(bytes)
                }
                if (!tmp.renameTo(file)) tmp.delete()
            }
        }
    }
}
