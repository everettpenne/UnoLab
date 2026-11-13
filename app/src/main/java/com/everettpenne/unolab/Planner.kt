package com.everettpenne.unolab

import org.json.JSONObject

/** What Uno Launcher asks the planner for: the question, the facts, the tools and the grammar Uno generated from them. */
internal class PlannerRequest(
    val query: String,
    val context: String,
    val grounding: String,
    val catalogue: String,
    val grammar: String,
    /** Changes only when the tool catalogue does, so the decoded prompt is reused while the tools stay the same. */
    val hash: String,
)

/**
 * The planner's prompts. Uno Launcher sends the tool catalogue and the GBNF grammar as strings, so this side never learns what a tool does or
 * which tools exist: a new capability is a change in the launcher alone. Pure, so it is unit-tested.
 */
internal object Planner {
    /** Plans are short — a few steps of a handful of fields each. */
    const val MAX_TOKENS = 256

    /** Planning wants the most likely reading, not variety. */
    const val TEMPERATURE = .2f

    private const val SYSTEM =
        "You are the planner for Uno, a phone assistant. Turn what the person asks into the JSON described below. " +
            "Use only the tools listed. Prefer the fewest steps. If a detail is missing (which person, which time), ask a question instead " +
            "of guessing. If it cannot be done with these tools, refuse in one sentence and say what is missing. Reply with JSON only."

    /** The fixed start of every planner turn: its instructions and the generated catalogue. Decoded once per catalogue and cached. */
    fun prefix(request: PlannerRequest): String = buildString {
        append("<|im_start|>system\n")
        append(SYSTEM)
        append("\n\n")
        append(request.catalogue.trim())
        append("\n<|im_end|>\n")
    }

    /** The question, after [prefix]: the phone's facts and anything grounding resolved, then the request itself. */
    fun turn(request: PlannerRequest): String = buildString {
        append("<|im_start|>user\n")
        if (request.context.isNotBlank()) {
            append("<context>\n").append(request.context.trim()).append("\n</context>\n\n")
        }
        if (request.grounding.isNotBlank()) {
            append("<known>\n").append(request.grounding.trim()).append("\n</known>\n\n")
        }
        append("Request: ").append(request.query.trim())
        append("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    }

    /** A refusal the launcher can parse, for a request this side cannot serve. */
    fun error(why: String): String = "{\"refuse\":" + quote(why) + "}"

    /** Reads the launcher's request; null when it is not one. */
    fun parse(json: String): PlannerRequest? {
        val obj = runCatching { JSONObject(json) }.getOrNull() ?: return null
        val query = obj.optString("query").trim()
        if (query.isEmpty()) return null
        return PlannerRequest(
            query = query,
            context = obj.optString("context"),
            grounding = obj.optString("grounding"),
            catalogue = obj.optString("catalogue"),
            grammar = obj.optString("grammar"),
            hash = obj.optString("hash").ifBlank { "nocache" },
        )
    }

    private fun quote(text: String): String = buildString {
        append('"')
        for (ch in text) {
            when {
                ch == '"' -> append("\\\"")
                ch == '\\' -> append("\\\\")
                ch == '\n' -> append("\\n")
                ch == '\r' -> append("\\r")
                ch == '\t' -> append("\\t")
                ch < ' ' -> append("\\u").append("%04x".format(ch.code))
                else -> append(ch)
            }
        }
        append('"')
    }
}
