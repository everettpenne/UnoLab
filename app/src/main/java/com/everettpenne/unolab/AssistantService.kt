package com.everettpenne.unolab

import android.app.Service
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Binder
import android.os.IBinder
import android.os.RemoteException

/**
 * Uno Lab's assistant, for Uno Launcher's search to bind to (see IAssistant.aidl). The model runs here, in Uno Lab's process, never in the
 * launcher's. Only Uno is served: each call checks that the calling app is Uno (or Uno Dev) signed with Uno's own key, by certificate.
 */
class AssistantService : Service() {
    override fun onCreate() { super.onCreate(); AssistantEngine.init(this) }

    override fun onBind(intent: Intent?): IBinder { AssistantEngine.acquire(); return binder }

    override fun onUnbind(intent: Intent?): Boolean { AssistantEngine.release(); return false }

    private val binder = object : IAssistant.Stub() {
        override fun state(): Int { check(); return when (AssistantEngine.state.value) {
            AssistantEngine.State.Missing -> 0; is AssistantEngine.State.Loading -> 1; is AssistantEngine.State.Ready -> 2; is AssistantEngine.State.Failed -> 3 } }

        override fun warm() { check(); AssistantEngine.warm() }

        override fun ask(query: String?, mode: Int, sink: IAnswerSink?) { check(); answer(query, null, mode, sink) }

        override fun version(): Int { check(); return Assistant.VERSION }

        override fun askWithContext(query: String?, context: String?, mode: Int, sink: IAnswerSink?) { check(); answer(query, context, mode, sink) }

        override fun cancel() { check(); AssistantEngine.cancel() }
    }

    private fun answer(query: String?, context: String?, mode: Int, sink: IAnswerSink?) {
        if (query.isNullOrBlank() || sink == null || mode !in Assistant.ANSWER..Assistant.ACTION) return
        AssistantEngine.ask(query, context, mode, object : AssistantEngine.Listener {
            // The caller going away (a RemoteException) stops the answer rather than writing it for nobody.
            override fun onPiece(piece: String) { try { sink.onPiece(piece) } catch (_: RemoteException) { AssistantEngine.cancel() } }
            override fun onDone(ms: Long, tokens: Int) { runCatching { sink.onDone(ms, tokens) } }
            override fun onError(why: String) { runCatching { sink.onError(why) } }
        })
    }

    private fun check() {
        val uid = Binder.getCallingUid()
        if (!CallerPolicy.allowed(packageManager.getPackagesForUid(uid)?.toList().orEmpty()) { pkg, cert -> signedWith(pkg, cert) })
            throw SecurityException("Uno Lab's assistant only answers Uno Launcher")
    }

    private fun signedWith(pkg: String, sha256: ByteArray): Boolean =
        runCatching { packageManager.hasSigningCertificate(pkg, sha256, PackageManager.CERT_INPUT_SHA256) }.getOrDefault(false)
}

/** Who may use the assistant: Uno, by package and signing certificate. Pure (the certificate check is passed in), so it is unit-tested. */
object CallerPolicy {
    /** Package to its signing certificate's SHA-256: Uno's release key, and the debug key Uno Dev is built with. */
    val ALLOWED: Map<String, String> = mapOf(
        "com.jake.duolauncher" to "784b26f76660f3a66533818793389ed9daf9c93fc91f95fb7a1da6a7bf2b69e6",
        "com.jake.duolauncher.dev" to "ae50491a2ed73e7b67fd492d5d7729df5d7f26cbed1149d9f129d00086f57899",
    )

    fun allowed(packages: List<String>, signedWith: (String, ByteArray) -> Boolean): Boolean =
        packages.any { pkg -> ALLOWED[pkg]?.let { signedWith(pkg, hex(it)) } == true }

    fun hex(s: String): ByteArray = ByteArray(s.length / 2) { i -> s.substring(i * 2, i * 2 + 2).toInt(16).toByte() }
}
