package com.everettpenne.unolab

/**
 * Qwen3.5's chat format (ChatML), written out the way its own template does, so only the new text of each turn is decoded. Thinking off is the
 * template's own way: the assistant turn opens with an empty think block. Pure, so it is unit-tested.
 */
object Prompt {
    const val SYSTEM = "You are a helpful assistant running entirely on this phone. Answer briefly."

    fun turn(user: String, first: Boolean, closePrevious: Boolean, thinking: Boolean): String = buildString {
        if (closePrevious) append("<|im_end|>\n")
        if (first) append("<|im_start|>system\n").append(SYSTEM).append("<|im_end|>\n")
        append("<|im_start|>user\n").append(user.trim()).append("<|im_end|>\n")
        append("<|im_start|>assistant\n")
        append(if (thinking) "<think>\n" else "<think>\n\n</think>\n\n")
    }

    /** Splits a reply into its thinking and its answer. With thinking on, everything before </think> is thinking (all of it, until that arrives). */
    fun split(raw: String, thinking: Boolean): Pair<String, String> {
        if (!thinking) return "" to raw.trimStart()
        val end = raw.indexOf("</think>")
        return if (end < 0) raw.trim() to "" else raw.substring(0, end).trim() to raw.substring(end + "</think>".length).trimStart()
    }
}
