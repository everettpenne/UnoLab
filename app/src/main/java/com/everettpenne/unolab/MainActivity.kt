package com.everettpenne.unolab

import android.app.role.RoleManager
import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.Send
import androidx.compose.material.icons.rounded.Add
import androidx.compose.material.icons.rounded.Stop
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel

class MainActivity : ComponentActivity() {
    /** A question handed over by another app (Uno's search, or "Ask Qwen" on selected text), asked once the model is ready. */
    private var handed by mutableStateOf<String?>(null)
    /** Bumped on every resume, so returning from the system's assistant chooser re-reads whether the role was granted. */
    private var resumes by mutableIntStateOf(0)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        handed = questionFrom(intent)
        setContent {
            MaterialTheme(colorScheme = darkColorScheme(primary = Color(0xFF9BC5D7), secondary = Color(0xFFD1BE98))) {
                Surface(Modifier.fillMaxSize(), color = Color(0xFF0E1214), contentColor = Color.White) { Lab(handed, resumes) { handed = null } }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        resumes++
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        questionFrom(intent)?.let { handed = it }
    }

    private fun questionFrom(intent: Intent?): String? = when (intent?.action) {
        Intent.ACTION_PROCESS_TEXT -> intent.getCharSequenceExtra(Intent.EXTRA_PROCESS_TEXT)?.toString()
        Intent.ACTION_SEND, ACTION_ASK -> intent.getStringExtra(Intent.EXTRA_TEXT)
        else -> null
    }?.takeIf { it.isNotBlank() }

    companion object {
        /** What another app (Uno) sends, with the question in Intent.EXTRA_TEXT. */
        const val ACTION_ASK = "com.everettpenne.unolab.ASK"
    }
}

@Composable
private fun Lab(handed: String?, resumes: Int, onHandedTaken: () -> Unit) {
    val vm: LabViewModel = viewModel()
    var tab by rememberSaveable { mutableIntStateOf(0) }
    // A handed-over question goes to the chat; the view model asks it as soon as the model is ready.
    LaunchedEffect(handed) { if (handed != null) { tab = 0; vm.ask(handed); onHandedTaken() } }
    Column(Modifier.fillMaxSize().systemBarsPadding()) {
        TabRow(tab, containerColor = Color.Transparent) {
            listOf("Chat", "NPU lab", "Model").forEachIndexed { i, label ->
                Tab(tab == i, { tab = i }, text = { Text(label) }, modifier = Modifier.testTag("tab-$i"))
            }
        }
        PowerBar(vm)
        when (tab) {
            0 -> Chat(vm, Modifier.weight(1f))
            1 -> NpuLab(vm, Modifier.weight(1f))
            else -> ModelTab(vm, resumes, Modifier.weight(1f))
        }
    }
}

@Composable
private fun PowerBar(vm: LabViewModel) {
    val now = vm.wattsNow; val idle = vm.idleWatts
    Text(buildString {
        append(if (now != null) "Draw %.2f W".format(now) else "Draw unknown")
        if (idle != null) append(" · idle %.2f W".format(idle))
        if (vm.charging) append(" · charging: energy figures off")
    }, Modifier.padding(horizontal = 16.dp, vertical = 6.dp), style = MaterialTheme.typography.labelSmall, color = Color.White.copy(alpha = .55f))
}

@Composable
private fun rememberSaveable(init: () -> MutableIntState): MutableIntState = androidx.compose.runtime.saveable.rememberSaveable { init() }

@Composable
private fun Chat(vm: LabViewModel, modifier: Modifier) {
    val model = vm.model
    if (model !is ModelState.Ready) { Box(modifier.fillMaxWidth().padding(24.dp)) { ModelStatus(vm) }; return }
    var draft by remember { mutableStateOf("") }
    val list = rememberLazyListState()
    LaunchedEffect(vm.messages.size, vm.messages.lastOrNull()?.text?.length) { if (vm.messages.isNotEmpty()) list.scrollToItem(vm.messages.size - 1) }
    Column(modifier.fillMaxWidth().imePadding()) {
        LazyColumn(Modifier.weight(1f).fillMaxWidth(), state = list, contentPadding = PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            if (vm.messages.isEmpty()) item {
                Text("Qwen3.8-2B, running on this phone's CPU with llama.cpp. Nothing leaves the phone.\n\nSelect text in any app and choose “Ask Qwen”, or ask below.",
                    color = Color.White.copy(alpha = .6f))
            }
            items(vm.messages) { m -> Bubble(m) }
        }
        Row(Modifier.fillMaxWidth().padding(horizontal = 12.dp), verticalAlignment = Alignment.CenterVertically) {
            FilterChip(vm.thinking, { vm.thinking = !vm.thinking }, label = { Text("Think") }, enabled = !vm.busy)
            Spacer(Modifier.width(8.dp))
            AssistChip({ vm.newChat() }, label = { Text("New chat") }, leadingIcon = { Icon(Icons.Rounded.Add, null) }, enabled = !vm.busy)
            Spacer(Modifier.weight(1f))
            Text("${vm.used} / ${model.ctx}", style = MaterialTheme.typography.labelSmall, color = Color.White.copy(alpha = .5f))
        }
        Row(Modifier.fillMaxWidth().padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
            OutlinedTextField(draft, { draft = it }, Modifier.weight(1f).testTag("chat-field"), placeholder = { Text("Ask anything") }, maxLines = 5)
            Spacer(Modifier.width(8.dp))
            if (vm.busy) IconButton({ vm.stop() }) { Icon(Icons.Rounded.Stop, "Stop") }
            else IconButton({ vm.send(draft); draft = "" }, enabled = draft.isNotBlank(), modifier = Modifier.testTag("chat-send")) { Icon(Icons.AutoMirrored.Rounded.Send, "Send") }
        }
    }
}

@Composable
private fun Bubble(m: Message) {
    Column(Modifier.fillMaxWidth(), horizontalAlignment = if (m.user) Alignment.End else Alignment.Start) {
        if (m.thinking.isNotBlank()) Text(m.thinking, Modifier.padding(bottom = 4.dp).fillMaxWidth(.9f), color = Color.White.copy(alpha = .45f),
            style = MaterialTheme.typography.bodySmall)
        SelectionContainer {
            Text(m.text.ifEmpty { if (m.user) "" else "…" }, Modifier.widthIn(max = 320.dp).background(
                if (m.user) Color(0xFF2C4A57) else Color(0xFF1C2226), RoundedCornerShape(16.dp)).padding(horizontal = 14.dp, vertical = 10.dp)
                .testTag(if (m.user) "msg-user" else "msg-reply"), color = Color.White)
        }
        m.stats?.let { Text(it, Modifier.padding(top = 3.dp), style = MaterialTheme.typography.labelSmall, color = Color.White.copy(alpha = .45f)) }
    }
}

@Composable
private fun NpuLab(vm: LabViewModel, modifier: Modifier) {
    LaunchedEffect(Unit) { vm.loadDevices() }
    var big by remember { mutableStateOf(false) }
    Column(modifier.fillMaxWidth().verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Text("Raw NNAPI, no Google SDK: one int8 fully-connected layer at a time, pinned to the accelerator so nothing falls back to the CPU. ", color = Color.White.copy(alpha = .6f), style = MaterialTheme.typography.bodySmall)
        Text("Devices", style = MaterialTheme.typography.titleSmall)
        if (vm.devices.isEmpty()) Text("None visible to this app.", color = Color.White.copy(alpha = .6f))
        vm.devices.forEach { d -> val (name, type, level) = (d.split('|') + listOf("", "", "")).take(3)
            Text("$name · $type · feature level $level", fontFamily = FontFamily.Monospace, fontSize = 13.sp, modifier = Modifier.testTag("npu-device")) }
        Row(verticalAlignment = Alignment.CenterVertically) {
            Button({ vm.sweep(big) }, enabled = !vm.npuBusy, modifier = Modifier.testTag("npu-sweep")) { Text("Run sweep") }
            Spacer(Modifier.width(8.dp))
            OutlinedButton({ vm.sustained() }, enabled = !vm.npuBusy) { Text("Energy, 10 s") }
        }
        Row(verticalAlignment = Alignment.CenterVertically) {
            Checkbox(big, { big = it }, enabled = !vm.npuBusy)
            Text("Include the 509 MB layer (the size of Qwen's embedding; compiling it takes about 20 s)", style = MaterialTheme.typography.bodySmall)
        }
        if (vm.npuBusy) LinearProgressIndicator(Modifier.fillMaxWidth())
        vm.npuNote?.let { Text(it, color = MaterialTheme.colorScheme.secondary, modifier = Modifier.testTag("npu-note")) }
        if (vm.rows.isNotEmpty()) {
            Text("  K x N (batch)       weights   compile     run     rate", fontFamily = FontFamily.Monospace, fontSize = 11.sp, color = Color.White.copy(alpha = .5f))
            vm.rows.forEach { r ->
                val shape = "${r.k}x${r.n}" + if (r.batch > 1) " (${r.batch})" else ""
                val rate = if (r.batch > 1) "%5.0f GMAC/s".format(r.gmacPerS) else "%5.1f GB/s".format(r.gbPerS)
                Text(if (r.ok) "%-19s %6.1f MB %7.0f ms %6.2f ms %s".format(shape, r.weightMb, r.compileMs, r.bestMs, rate)
                    else "%-19s %6.1f MB  failed: %s".format(shape, r.weightMb, r.detail),
                    fontFamily = FontFamily.Monospace, fontSize = 11.sp, modifier = Modifier.testTag("npu-row"))
                if (r.ok && r.detail.isNotEmpty()) Text("    check: ${r.detail}", fontFamily = FontFamily.Monospace, fontSize = 10.sp, color = Color.White.copy(alpha = .45f))
            }
        }
        Text("What it showed on a Pixel 10a: 20–26 GB/s and about 3x less energy per byte than the CPU, but int8 only, so Qwen's activations " +
            "collapse by its third layer: the chat runs on the CPU.", color = Color.White.copy(alpha = .5f), style = MaterialTheme.typography.bodySmall)
    }
}

/**
 * The system assistant role. Taking it is what lets the assist gesture open Uno's search from any app, with that app's screen as a fact for
 * the answer. It is asked for, never assumed: the system dialog is the consent, and Settings can take it back at any time.
 */
@Composable
private fun AssistantRole(resumes: Int) {
    val context = LocalContext.current
    val roles = remember { context.getSystemService(RoleManager::class.java) }
    val available = remember(resumes) { roles?.isRoleAvailable(RoleManager.ROLE_ASSISTANT) == true }
    val held = remember(resumes) { roles?.isRoleHeld(RoleManager.ROLE_ASSISTANT) == true }
    val ask = rememberLauncherForActivityResult(ActivityResultContracts.StartActivityForResult()) { }
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text("System assistant", style = MaterialTheme.typography.titleSmall)
        Text(when {
            !available -> "This phone offers no assistant role to take."
            held -> "Uno Lab is the assistant: the assist gesture opens Uno's search from any app, with what is on the screen."
            else -> "The assistant gesture can open Uno's search from any app, with what is on the screen."
        }, style = MaterialTheme.typography.bodySmall)
        if (available && !held) {
            Button({ roles?.let { ask.launch(it.createRequestRoleIntent(RoleManager.ROLE_ASSISTANT)) } }, Modifier.testTag("assistant-role")) {
                Text("Make Uno Lab the assistant")
            }
        }
        Text("No hotword: \"Hey Qwen\" needs a privileged permission a sideloaded app cannot hold. The gesture — or Uno's own search — is how it opens.",
            style = MaterialTheme.typography.bodySmall, color = Color.White.copy(alpha = .6f))
    }
}

@Composable
private fun ModelStatus(vm: LabViewModel) {
    when (val m = vm.model) {
        ModelState.Missing -> Text("No model yet. Push a .gguf with\n\nadb push Qwen3.8-2B-Q4_K_M.gguf ${vm.modelDir}/\n\nor import one on the Model tab.",
            color = Color.White.copy(alpha = .7f), modifier = Modifier.testTag("model-missing"))
        is ModelState.Importing -> Column { Text("Moving the model into the app… ${(m.fraction * 100).toInt()}%"); LinearProgressIndicator({ m.fraction }, Modifier.fillMaxWidth()) }
        is ModelState.Loading -> Row(verticalAlignment = Alignment.CenterVertically) { CircularProgressIndicator(Modifier.size(20.dp)); Spacer(Modifier.width(10.dp)); Text("Loading ${m.name}…") }
        is ModelState.Ready -> Text("${m.name} is loaded, ${m.ctx} tokens of context.", modifier = Modifier.testTag("model-ready"))
        is ModelState.Failed -> Text("The model could not be loaded: ${m.why}", color = MaterialTheme.colorScheme.error)
    }
}

@Composable
private fun ModelTab(vm: LabViewModel, resumes: Int, modifier: Modifier) {
    val pick = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri -> uri?.let(vm::import) }
    Column(modifier.fillMaxWidth().verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        AssistantRole(resumes)
        HorizontalDivider()
        ModelStatus(vm)
        Row {
            Button({ pick.launch(arrayOf("application/octet-stream", "*/*")) }, enabled = vm.model !is ModelState.Importing && !vm.busy) { Text("Import model") }
            Spacer(Modifier.width(8.dp))
            OutlinedButton({ vm.refreshModel() }, enabled = !vm.busy) { Text("Look again") }
        }
        Text("Folder: ${vm.modelDir}", fontFamily = FontFamily.Monospace, fontSize = 11.sp, color = Color.White.copy(alpha = .6f))
        Text("CPU threads: ${vm.threads} (measured best on a Tensor G4; takes effect on the next load)", style = MaterialTheme.typography.bodySmall)
        Text("llama.cpp build", style = MaterialTheme.typography.titleSmall)
        Text(vm.systemInfo, fontFamily = FontFamily.Monospace, fontSize = 10.sp, color = Color.White.copy(alpha = .6f))
    }
}
