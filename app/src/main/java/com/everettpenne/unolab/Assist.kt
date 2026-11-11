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

    /** Where the assist gesture should land: Uno's search, or — when the launcher is not installed — Uno Lab itself. */
    fun intent(context: Context, screenText: String?): Intent {
        val launcher = runCatching { context.packageManager.getLaunchIntentForPackage(LAUNCHER) }.getOrNull()
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

/** The role is only offered when the app has a service for it; this is that service. It has nothing of its own to do. */
class AssistService : VoiceInteractionService()

/** Makes a session for each assist request, which forwards it to Uno and closes itself. */
class AssistSessionService : VoiceInteractionSessionService() {
    override fun onNewSession(args: Bundle?): VoiceInteractionSession = AssistSession(this)
}

/** Reads the screen once, opens Uno's search with the text, and gets out of the way. */
internal class AssistSession(context: Context) : VoiceInteractionSession(context) {
    override fun onHandleAssist(data: Bundle?, structure: android.app.assist.AssistStructure?, content: android.app.assist.AssistContent?) {
        val app = context.applicationContext
        val text = runCatching { ScreenText.from(structure) }.getOrNull()
        runCatching { startAssistantActivity(AssistEntry.intent(app, text)) }
            .onFailure { runCatching { startAssistantActivity(AssistEntry.intent(app, null)) } }
        finish()
    }
}
