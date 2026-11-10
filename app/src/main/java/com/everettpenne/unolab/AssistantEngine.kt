package com.everettpenne.unolab

import android.content.Context
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
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
    /** The assistant's context: a system prompt and one short question and answer. */
    private const val HELPER_CONTEXT = 1536
    const val IDLE_MS = 3 * 60_000L

    val dispatcher = Executors.newSingleThreadExecutor { r -> Thread(r, "llm").apply { priority = Thread.MAX_PRIORITY } }.asCoroutineDispatcher()
    private val scope = CoroutineScope(SupervisorJob() + dispatcher)
    private lateinit var store: ModelStore

    private val _state = MutableStateFlow<State>(State.Missing)
    val state: StateFlow<State> = _state

    /** Four was fastest on a Tensor G4 (more threads ran slower); never more than there are cores. */
    @Volatile var threads = minOf(4, Runtime.getRuntime().availableProcessors())

    /** The chat's context, which owns the weights; 0 when nothing is loaded. Read and used on [dispatcher] only. */
    var chat = 0L; private set
    private var helper = 0L
    /** Each mode's system prompt decoded once: the state after it, and how many tokens it holds. */
    private val prefixes = HashMap<Int, Pair<ByteArray, Int>>()

    private var holders = 0
    private var idleJob: Job? = null
    private var askJob: Job? = null

    fun init(context: Context) { if (!::store.isInitialized) store = ModelStore(context.applicationContext) }

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
        val next = runCatching {
            unloadNow(keepState = true)
            chat = Llm.nativeLoad(file.absolutePath, CONTEXT, threads)
            State.Ready(file.name, Llm.nativeCtx(chat))
        }.getOrElse { State.Failed(it.message ?: it.toString()) }
        _state.value = next
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

    /** Asks the assistant one question in [mode] (see [Assistant]), replacing any still running; [listener] is called on the model's thread. */
    fun ask(query: String, mode: Int, listener: Listener) {
        cancel()
        askJob = scope.launch {
            val job = coroutineContext[Job]
            val state = loadNow()
            if (state !is State.Ready) { listener.onError(if (state is State.Failed) state.why else "No model in Uno Lab"); return@launch }
            runCatching {
                if (helper == 0L) helper = Llm.nativeNewContext(chat, HELPER_CONTEXT, threads)
                startFrom(mode)
                val pre = Llm.nativePrefill(helper, Assistant.turn(query))
                // A question replaced while it was still being read is dropped here, before anything is written for it.
                if (job?.isActive != true) return@launch
                val h = helper
                val gen = Llm.nativeGenerate(h, Assistant.maxTokens(mode), Assistant.temperature(mode), .9f, 20,
                    if (mode == Assistant.ACTION) Assistant.GRAMMAR else null) { bytes ->
                    if (job.isActive) listener.onPiece(String(bytes, Charsets.UTF_8)) else Llm.nativeStop(h)
                }
                if (job.isActive) listener.onDone((pre[1] + gen[1]).toLong(), gen[0].toInt())
            }.onFailure { listener.onError(it.message ?: it.toString()) }
        }
    }

    /** Puts the helper at the end of [mode]'s system prompt: from its saved state, or decoded and saved the first time. */
    private fun startFrom(mode: Int) {
        prefixes[mode]?.let { (bytes, n) -> if (Llm.nativeRestoreState(helper, bytes, n)) return }
        Llm.nativeReset(helper)
        val pre = Llm.nativePrefill(helper, Assistant.prefix(mode))
        prefixes[mode] = Llm.nativeSaveState(helper) to pre[0].toInt()
    }
}
