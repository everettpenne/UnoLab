package com.everettpenne.unolab

import android.app.Application
import android.net.Uri
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors

/** One message in the chat. [stats] is filled in when a reply is done. */
class Message(val user: Boolean, text: String = "", thinking: String = "") {
    var text by mutableStateOf(text)
    var thinking by mutableStateOf(thinking)
    var stats by mutableStateOf<String?>(null)
}

/** One row of the NPU sweep. */
data class GemmRow(val k: Int, val n: Int, val batch: Int, val ok: Boolean, val compileMs: Double, val bestMs: Double, val weightMb: Double,
                   val gbPerS: Double, val gmacPerS: Double, val detail: String)

sealed interface ModelState {
    data object Missing : ModelState
    data class Importing(val fraction: Float) : ModelState
    data class Loading(val name: String) : ModelState
    data class Ready(val name: String, val ctx: Int) : ModelState
    data class Failed(val why: String) : ModelState
}

class LabViewModel(app: Application) : AndroidViewModel(app) {
    private val store = ModelStore(app)
    val power = Power(app)
    /** llama.cpp is not thread-safe: every call to it runs on the engine's one thread, shared with the assistant service. */
    private val llmThread = AssistantEngine.dispatcher
    private val npuThread = Executors.newSingleThreadExecutor { r -> Thread(r, "npu") }.asCoroutineDispatcher()
    /** The chat's context, owned by the engine (which the assistant service shares). */
    private val handle get() = AssistantEngine.chat

    var model by mutableStateOf<ModelState>(ModelState.Missing); private set
    val messages = mutableStateListOf<Message>()
    var busy by mutableStateOf(false); private set
    var thinking by mutableStateOf(false)
    /** Four was fastest on a Tensor G4 (more threads ran slower); never more than there are cores (the emulator has one). */
    var threads by mutableStateOf(AssistantEngine.threads)
    var used by mutableStateOf(0); private set
    var systemInfo by mutableStateOf(""); private set
    val modelDir: String get() = store.dir.absolutePath

    /** The phone's draw while nothing of ours runs, kept up to date between tasks, to take off a task's draw. */
    var idleWatts by mutableStateOf<Double?>(null); private set
    var wattsNow by mutableStateOf<Double?>(null); private set
    var charging by mutableStateOf(false); private set

    private var firstTurn = true
    private var closePrevious = false
    private var genJob: Job? = null

    init {
        AssistantEngine.init(app)
        AssistantEngine.acquire()
        viewModelScope.launch {
            while (isActive) {
                val w = withContext(Dispatchers.Default) { power.watts() }
                wattsNow = w; charging = power.charging()
                if (!busy && !npuBusy && w != null) idleWatts = idleWatts?.let { it * .8 + w * .2 } ?: w
                delay(1_000)
            }
        }
        viewModelScope.launch { systemInfo = withContext(llmThread) { runCatching { Llm.nativeSystemInfo() }.getOrDefault("") } }
        refreshModel()
    }

    fun refreshModel() {
        if (model is ModelState.Loading || model is ModelState.Importing) return
        val dropped = store.waiting()
        if (dropped != null) {
            // A newly pushed model: moved into internal storage first, then loaded.
            if (busy) return
            model = ModelState.Importing(0f)
            viewModelScope.launch {
                runCatching { withContext(Dispatchers.IO) { store.moveIn(dropped) { f -> model = ModelState.Importing(f) } } }
                    .onSuccess { model = ModelState.Missing; unload(); refreshModel() }
                    .onFailure { model = ModelState.Failed(it.message ?: it.toString()) }
            }
            return
        }
        val file = store.find()
        if (file == null) { model = ModelState.Missing; return }
        if (model is ModelState.Ready) return
        model = ModelState.Loading(file.name)
        viewModelScope.launch {
            AssistantEngine.threads = threads
            model = when (val s = AssistantEngine.load()) {
                is AssistantEngine.State.Ready -> ModelState.Ready(s.name, s.ctx)
                is AssistantEngine.State.Failed -> ModelState.Failed(s.why)
                is AssistantEngine.State.Loading -> ModelState.Loading(s.name)
                AssistantEngine.State.Missing -> ModelState.Missing
            }
            askPending()
        }
    }

    /** A question handed over by another app, asked once the model is ready (it may still be moving in or loading when it arrives). */
    private var pending: String? = null

    fun ask(question: String) { pending = question; askPending() }

    private fun askPending() {
        val q = pending ?: return
        if (model is ModelState.Ready && !busy) { pending = null; send(q) }
    }

    fun import(uri: Uri) {
        model = ModelState.Importing(0f)
        viewModelScope.launch {
            runCatching { withContext(Dispatchers.IO) { store.import(uri) { f -> model = ModelState.Importing(f) } } }
                .onSuccess { model = ModelState.Missing; unload(); refreshModel() }
                .onFailure { model = ModelState.Failed(it.message ?: it.toString()) }
        }
    }

    private suspend fun unload() {
        AssistantEngine.unload()
        messages.clear(); firstTurn = true; closePrevious = false; used = 0
    }

    fun newChat() {
        if (busy) return
        messages.clear(); firstTurn = true; closePrevious = false
        viewModelScope.launch { withContext(llmThread) { if (handle != 0L) Llm.nativeReset(handle) }; used = 0 }
    }

    fun stop() { if (handle != 0L) Llm.nativeStop(handle) }

    fun send(text: String) {
        val h = handle
        if (busy || h == 0L || text.isBlank()) return
        busy = true
        val think = thinking
        messages += Message(user = true, text = text.trim())
        val reply = Message(user = false)
        messages += reply
        genJob = viewModelScope.launch {
            val meter = PowerMeter()
            val idle = idleWatts
            val sampler = launch { while (isActive) { meter.add(withContext(Dispatchers.Default) { power.watts() }); delay(500) } }
            val raw = StringBuilder()
            val result = withContext(llmThread) {
                runCatching {
                    val pre = Llm.nativePrefill(h, Prompt.turn(text, firstTurn, closePrevious, think))
                    firstTurn = false
                    val gen = Llm.nativeGenerate(h, if (think) 2048 else 768, if (think) .6f else .7f, if (think) .95f else .8f, 20, null) { bytes ->
                        val piece = String(bytes, Charsets.UTF_8)
                        synchronized(raw) { raw.append(piece) }
                        val (t, a) = Prompt.split(synchronized(raw) { raw.toString() }, think)
                        reply.thinking = t; reply.text = a
                    }
                    closePrevious = true
                    pre to gen
                }
            }
            sampler.cancel()
            used = withContext(llmThread) { if (handle != 0L) Llm.nativeUsed(handle) else 0 }
            result.onSuccess { (pre, gen) ->
                val prefillTps = pre[0] / (pre[1] / 1000.0)
                val decodeTps = if (gen[1] > 0) gen[0] / (gen[1] / 1000.0) else 0.0
                val seconds = (pre[1] + gen[1]) / 1000.0
                val mean = meter.mean
                val energy = if (mean != null && idle != null && !charging) Energy.joules(mean, idle, seconds) else null
                val perToken = energy?.let { Energy.perUnit(it, gen[0]) }
                reply.stats = buildString {
                    append("prefill ${pre[0].toInt()} tok at %.1f tok/s · decode ${gen[0].toInt()} tok at %.1f tok/s".format(prefillTps, decodeTps))
                    if (mean != null) append(" · %.2f W".format(mean))
                    if (perToken != null) append(" · %.2f J/token".format(perToken))
                    if (charging) append(" · on charger, no energy figure")
                    if (gen[2] == 0.0) append(" · stopped")
                }
                if (reply.text.isBlank() && reply.thinking.isBlank()) reply.text = "(no reply)"
            }.onFailure { reply.text = "Error: ${it.message}"; closePrevious = false }
            busy = false
            askPending()
        }
    }

    // ---- NPU lab ----

    var npuBusy by mutableStateOf(false); private set
    var devices by mutableStateOf<List<String>>(emptyList()); private set
    val rows = mutableStateListOf<GemmRow>()
    var npuNote by mutableStateOf<String?>(null); private set

    fun loadDevices() {
        viewModelScope.launch { devices = withContext(npuThread) { runCatching { Npu.nativeDevices().toList() }.getOrDefault(emptyList()) } }
    }

    private fun acceleratorIndex(): Int = devices.indexOfFirst { it.startsWith("google-edgetpu|") }.takeIf { it >= 0 }
        ?: devices.indexOfFirst { it.split('|').getOrNull(1) == "ACCELERATOR" }

    private fun gemm(index: Int, k: Int, n: Int, batch: Int, iters: Int): GemmRow {
        val detail = arrayOfNulls<String>(1)
        val r = Npu.nativeGemm(index, k, n, iters, batch, detail)
        val bytes = r[4]; val best = r[3]
        val gbs = if (best > 0) bytes / 1e9 / (best / 1000) else 0.0
        val gmac = if (best > 0) k.toDouble() * n * batch / 1e9 / (best / 1000) else 0.0
        return GemmRow(k, n, batch, r[0] == 1.0, r[1], best, bytes / 1e6, gbs, gmac, detail[0] ?: "")
    }

    /** The probe's sweep: int8 FC layers of growing size, pinned to the accelerator, decode-shaped (batch 1) and one prefill-shaped run. */
    fun sweep(big: Boolean) {
        if (npuBusy) return
        val index = acceleratorIndex()
        if (index < 0) { npuNote = "No NNAPI accelerator on this device."; return }
        npuBusy = true; npuNote = null; rows.clear()
        val sizes = listOf(512 to 512, 2048 to 2048, 4096 to 4096, 8192 to 2048, 8192 to 8192, 16384 to 4096) + if (big) listOf(2048 to 248320) else emptyList()
        viewModelScope.launch {
            withContext(npuThread) {
                for ((k, n) in sizes) { val row = gemm(index, k, n, 1, 5); withContext(Dispatchers.Main) { rows += row } }
                val pre = gemm(index, 4096, 4096, 128, 3); withContext(Dispatchers.Main) { rows += pre }
            }
            npuBusy = false
        }
    }

    /** Streams one 67 MB layer over and over for [seconds], with the draw sampled, for energy per GB. */
    fun sustained(seconds: Int = 10) {
        if (npuBusy) return
        val index = acceleratorIndex()
        if (index < 0) { npuNote = "No NNAPI accelerator on this device."; return }
        npuBusy = true; npuNote = "Streaming for $seconds s…"
        viewModelScope.launch {
            val idle = idleWatts
            val meter = PowerMeter()
            val sampler = launch { while (isActive) { meter.add(withContext(Dispatchers.Default) { power.watts() }); delay(500) } }
            val (gb, secs) = withContext(npuThread) {
                val t0 = System.nanoTime(); var gb = 0.0
                while ((System.nanoTime() - t0) / 1e9 < seconds) {
                    val r = Npu.nativeGemm(index, 8192, 8192, 50, 1, arrayOfNulls(1))
                    if (r[0] != 1.0) break
                    gb += r[4] / 1e9 * 50
                }
                gb to (System.nanoTime() - t0) / 1e9
            }
            sampler.cancel()
            val mean = meter.mean
            npuNote = buildString {
                append("Streamed %.1f GB in %.1f s (%.1f GB/s with set-up).".format(gb, secs, gb / secs))
                if (mean != null) append(" Draw %.2f W".format(mean))
                if (mean != null && idle != null && !charging) {
                    val j = Energy.joules(mean, idle, secs)
                    append(", %.2f W over idle: %.3f J/GB.".format(mean - idle, Energy.perUnit(j, gb) ?: 0.0))
                    append(" At 1.84 GB a token (Qwen3.8-2B in int8) that is %.2f J/token.".format((Energy.perUnit(j, gb) ?: 0.0) * 1.84))
                } else if (charging) append(". On a charger, so no energy figure.")
            }
            npuBusy = false
        }
    }

    // The model stays with the engine: freed a few minutes after nothing holds it (this screen or Uno's search).
    override fun onCleared() { AssistantEngine.release(); super.onCleared() }
}
