package com.everettpenne.unolab

/**
 * What Uno Lab's assistant says to the model, for Uno Launcher's search: two prompts (a short answer, or the request turned into one of Uno's
 * actions), and the grammar that keeps an action to one JSON object of a fixed shape. Pure, so it is unit-tested.
 */
object Assistant {
    const val ANSWER = 0
    const val ACTION = 1
    /** Longer questions are cut: search queries are short, and a long one would only cost prefill time. */
    const val MAX_QUERY = 500
    /** Phone context longer than this is cut: Uno sends a few short lines, and every token of it is read before the answer starts. */
    const val MAX_CONTEXT = 2000
    /** What [IAssistant.version] answers: 1 added askWithContext. An older Uno Lab has no version() and so reads as 0. */
    const val VERSION = 1

    /** Said in both prompts, since Uno may put what the phone knows (date, battery, alarm…) before the question. */
    private const val CONTEXT_RULE = "The user's message may start with phone context from this phone; use it when it answers the question and never invent it."

    const val ANSWER_SYSTEM = "You are Uno, an assistant running entirely on this phone, answering from the phone's search. " +
        "Answer in one to three short sentences, plainly, without lists or headings. If you are not sure, say so. " + CONTEXT_RULE

    const val ACTION_SYSTEM = "You turn a request typed into a phone's search into one action for the phone, as JSON. Actions:\n" +
        "timer: seconds, and an optional label\n" +
        "alarm: at (HH:MM, 24-hour) or in_minutes\n" +
        "stopwatch\n" +
        "torch: the flashlight\n" +
        "focus: on (true or false), do not disturb\n" +
        "note: text to write down\n" +
        "open_app: name of the app\n" +
        "call: name of the person\n" +
        "text: name of the person to message\n" +
        "web: query, to search the web\n" +
        "none: a question to answer, or anything else.\n" +
        "Examples:\n" +
        "pasta timer 12 minutes -> {\"action\":\"timer\",\"seconds\":720,\"label\":\"pasta\"}\n" +
        "wake me at 6:45 -> {\"action\":\"alarm\",\"at\":\"06:45\"}\n" +
        "turn on the flashlight -> {\"action\":\"torch\"}\n" +
        "open the camera -> {\"action\":\"open_app\",\"name\":\"Camera\"}\n" +
        "why is the sky blue -> {\"action\":\"none\"}\n" +
        CONTEXT_RULE + "\n" +
        "Reply with the JSON only."

    /** GBNF: one action object, nothing else. Strings are short and have no quotes or backslashes, so Uno can read them without surprises. */
    const val GRAMMAR = """root ::= "{" "\"action\":" act "}"
act ::= timer | alarmat | alarmin | bare | focus | note | openapp | call | textmsg | web
timer ::= "\"timer\",\"seconds\":" int ( ",\"label\":" str )?
alarmat ::= "\"alarm\",\"at\":\"" [0-2] [0-9] ":" [0-5] [0-9] "\""
alarmin ::= "\"alarm\",\"in_minutes\":" int
bare ::= "\"stopwatch\"" | "\"torch\"" | "\"none\""
focus ::= "\"focus\",\"on\":" ( "true" | "false" )
note ::= "\"note\",\"text\":" str
openapp ::= "\"open_app\",\"name\":" str
call ::= "\"call\",\"name\":" str
textmsg ::= "\"text\",\"name\":" str
web ::= "\"web\",\"query\":" str
int ::= [0-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
str ::= "\"" [^"\\\n]{1,80} "\""
"""

    /** The fixed start of every question in [mode]: decoded once, its state saved and put back before each question. */
    fun prefix(mode: Int): String = "<|im_start|>system\n" + (if (mode == ACTION) ACTION_SYSTEM else ANSWER_SYSTEM) + "<|im_end|>\n"

    /**
     * The question, after [prefix]: thinking off (an empty think block), as the chat does. Phone [context], when there is any, opens the user
     * turn (`<context>\n\nQuestion: <query>`), so it comes after the saved system prompt and the prompt's state can still be reused.
     */
    fun turn(query: String, context: String? = null): String {
        val ctx = context?.let { clean(it, MAX_CONTEXT) }.orEmpty()
        val user = if (ctx.isEmpty()) clean(query) else ctx + "\n\nQuestion: " + clean(query)
        return "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    }

    /** Text as the model sees it: control characters (and so any chat-format token text) taken out, trimmed, and capped at [max]. */
    fun clean(text: String, max: Int = MAX_QUERY): String =
        text.filter { it == '\n' || !it.isISOControl() }.replace("<|", "< |").trim().take(max).trimEnd()

    /** Sampling: an action is near-greedy (it should be the most likely reading), an answer a little freer. */
    fun maxTokens(mode: Int) = if (mode == ACTION) 64 else 220
    fun temperature(mode: Int) = if (mode == ACTION) .1f else .6f
}
