package com.everettpenne.unolab

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.service.voice.VoiceInteractionService
import android.service.voice.VoiceInteractionSession
import android.service.voice.VoiceInteractionSessionService

/**
 * Where "what's on the screen" goes into. Uno Launcher's search owns the assistant's UI, so the session does one thing: read the screen's
 * text, then open Uno's own search with it. Nothing is drawn here and no model is touched here — the load happens on the ask.
 *
 * Uno Lab holds the role rather than the launcher because this app declares no INTERNET permission: the text of whatever app is in front
 * lands in the one app that structurally cannot send it anywhere.
 */
internal object AssistEntry {
    const val LAUNCHER = "com.jake.duolauncher"
    /** The first Uno Launcher whose search opens over the app in front (0.132.5-uno01). */
    const val OVER_APP_VERSION = 208L

    /** Where the assist gesture should land: Uno's search, or — when the launcher is not installed — Uno Lab itself. */
    fun intent(context: Context, screenText: String?): Intent {
        val launcher = launcherIntent(context)
        val intent = launcher ?: Intent(context, MainActivity::class.java)
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        if (launcher != null) {
            intent.putExtra("duo_destination", "spotlight")
            if (!screenText.isNullOrBlank()) intent.putExtra("uno_assist_text", screenText)
        } else if (!screenText.isNullOrBlank()) {
            intent.putExtra(Intent.EXTRA_TEXT, screenText)
        }
        return intent
    }
}

/**
 * Uno Launcher's own activity. A home app carries no `CATEGORY_LAUNCHER` filter, so `getLaunchIntentForPackage` returns null for it and the
 * hand-off would land in Uno Lab itself; the activity is named directly instead.
 */
private fun launcherIntent(context: Context): Intent? {
    val pm = context.packageManager
    fun named(activity: String): Intent? =
        Intent(Intent.ACTION_MAIN).setClassName(AssistEntry.LAUNCHER, AssistEntry.LAUNCHER + "." + activity)
            .takeIf { runCatching { pm.resolveActivity(it, 0) }.getOrNull() != null }
    // From Uno Launcher 0.132.5 (versionCode 208) its .search.AssistActivity puts search over the app in front, so dismissing it returns there;
    // in older versions that activity is the floating window that could crash, so Home's activity is named instead, as it always was.
    val overApp = launcherVersion(context) >= AssistEntry.OVER_APP_VERSION
    return (if (overApp) named("search.AssistActivity") else null)
        ?: named("MainActivity")
        ?: runCatching { pm.getLaunchIntentForPackage(AssistEntry.LAUNCHER) }.getOrNull()
}

private fun launcherVersion(context: Context): Long = runCatching {
    context.packageManager.getPackageInfo(AssistEntry.LAUNCHER, 0).longVersionCode
}.getOrDefault(0L)

/** The role is only offered when the app has a service for it; this is that service. It has nothing of its own to do. */
class AssistService : VoiceInteractionService()

/** Makes a session for each assist request, which forwards it to Uno and closes itself. */
class AssistSessionService : VoiceInteractionSessionService() {
    override fun onNewSession(args: Bundle?): VoiceInteractionSession = AssistSession(this)
}

/**
 * Opens Uno's search as soon as the session is shown, and takes the screen's text with it if it arrives quickly.
 *
 * `onHandleAssist` is not dependable on its own: a launcher or a home screen often hands over no assist structure at all, and then the
 * session would sit there waiting for data that never comes while the user stares at an empty overlay. So `onShow` opens search straight
 * away, a short grace period later, and `onHandleAssist` jumps that queue when it does fire.
 */
internal class AssistSession(context: Context) : VoiceInteractionSession(context) {
    private var text: String? = null
    private var opened = false
    private val handler = android.os.Handler(android.os.Looper.getMainLooper())
    private val open = Runnable { launch() }

    override fun onShow(args: Bundle?, showFlags: Int) {
        // Assist data usually lands just after the session is shown; wait for it, but never for long.
        handler.postDelayed(open, GRACE_MS)
    }

    override fun onHandleAssist(data: Bundle?, structure: android.app.assist.AssistStructure?, content: android.app.assist.AssistContent?) {
        text = runCatching { ScreenText.from(structure) }.getOrNull()
        handler.removeCallbacks(open)
        launch()
    }

    override fun onHide() {
        handler.removeCallbacks(open)
    }

    private fun launch() {
        if (opened) return
        opened = true
        val app = context.applicationContext
        runCatching { startAssistantActivity(AssistEntry.intent(app, text)) }
            .onFailure { runCatching { startAssistantActivity(AssistEntry.intent(app, null)) } }
        finish()
    }

    private companion object {
        /** How long a session waits for the screen's text before opening search without it. */
        const val GRACE_MS = 250L
    }
}
