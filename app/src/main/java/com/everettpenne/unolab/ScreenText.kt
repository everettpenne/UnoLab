package com.everettpenne.unolab

import android.app.assist.AssistStructure
import android.text.InputType

/**
 * What is on the screen when the assistant gesture is used, as plain lines.
 *
 * It is written by other apps, so it is treated as a quote everywhere downstream: it is only ever a fact for an answer, never something
 * that reaches a turn which may become an action, and it is capped here before it travels.
 *
 * Pure apart from the traversal, so the joining and the cap are unit-tested.
 */
internal object ScreenText {
    /** Well under the Binder limit for an intent extra, and enough for an article's visible text. */
    const val MAX_CHARS = 6000

    /** The text of [structure], or null when it says nothing usable. */
    fun from(structure: AssistStructure?): String? {
        if (structure == null) return null
        val lines = ArrayList<String>()
        for (w in 0 until structure.windowNodeCount) {
            val window = runCatching { structure.getWindowNodeAt(w) }.getOrNull() ?: continue
            runCatching { collect(window.rootViewNode, lines) }
        }
        return cap(join(lines))
    }

    /** Text, hint and description of every visible node, skipping secret fields and their subtrees. */
    private fun collect(node: AssistStructure.ViewNode?, out: MutableList<String>) {
        if (node == null) return
        // A password field's contents never go anywhere near a model.
        if (secret(node)) return
        add(node.text, out)
        add(node.hint, out)
        add(node.contentDescription, out)
        for (i in 0 until node.childCount) collect(runCatching { node.getChildAt(i) }.getOrNull(), out)
    }

    /** Whether a field holds something secret: a password variation, or the autofill hints that say so. */
    fun secret(node: AssistStructure.ViewNode): Boolean {
        val input = runCatching { node.inputType }.getOrDefault(0)
        val kind = input and InputType.TYPE_MASK_CLASS
        val variation = input and InputType.TYPE_MASK_VARIATION
        if (kind == InputType.TYPE_CLASS_TEXT &&
            variation in setOf(
                InputType.TYPE_TEXT_VARIATION_PASSWORD,
                InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD,
                InputType.TYPE_TEXT_VARIATION_WEB_PASSWORD,
            )
        ) return true
        if (kind == InputType.TYPE_CLASS_NUMBER && variation == InputType.TYPE_NUMBER_VARIATION_PASSWORD) return true
        val hints = runCatching { node.autofillHints }.getOrNull()
        return hints != null && hints.any { it.contains("password", ignoreCase = true) }
    }

    private fun add(value: CharSequence?, out: MutableList<String>) {
        val line = value?.toString()?.trim()?.replace(Regex("""\s+"""), " ")?.takeIf { it.isNotEmpty() } ?: return
        out += line
    }

    /** The lines without the repeats the same label produces at two levels of the tree, and without blank runs. */
    fun join(lines: List<String>): String? {
        val kept = ArrayList<String>(lines.size)
        var last: String? = null
        for (line in lines) {
            val t = line.trim()
            if (t.isEmpty() || t == last) continue
            kept += t
            last = t
        }
        return kept.joinToString("\n").takeIf { it.isNotBlank() }
    }

    /** [text] cut to [MAX_CHARS] at a word boundary, so a truncated word is not sent as if it were real. */
    fun cap(text: String?): String? {
        val t = text?.trim()?.takeIf { it.isNotEmpty() } ?: return null
        if (t.length <= MAX_CHARS) return t
        val cut = t.substring(0, MAX_CHARS)
        val space = cut.lastIndexOfAny(charArrayOf(' ', '\n'))
        return (if (space > MAX_CHARS / 2) cut.substring(0, space) else cut).trim().takeIf { it.isNotEmpty() }
    }
}
