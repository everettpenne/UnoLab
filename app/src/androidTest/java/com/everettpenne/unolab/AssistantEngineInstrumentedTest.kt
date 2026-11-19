package com.everettpenne.unolab

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.IBinder
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/** The assistant with the real model (it must already be in Uno Lab), and the service's refusal of anyone but Uno. Slow on a one-core emulator. */
@RunWith(AndroidJUnit4::class)
class AssistantEngineInstrumentedTest {
    private val context: Context get() = InstrumentationRegistry.getInstrumentation().targetContext

    private fun ask(q: String, mode: Int, phone: String? = null): Pair<String, Long> {
        val out = StringBuilder(); val done = CountDownLatch(1); var error: String? = null
        val t0 = System.nanoTime()
        AssistantEngine.ask(q, phone, mode, object : AssistantEngine.Listener {
            override fun onPiece(piece: String) { out.append(piece) }
            override fun onDone(ms: Long, tokens: Int) { done.countDown() }
            override fun onError(why: String) { error = why; done.countDown() }
        })
        assertTrue("answered in time", done.await(40, TimeUnit.MINUTES))
        error?.let { fail(it) }
        return out.toString() to (System.nanoTime() - t0) / 1_000_000
    }

    @Test fun actionsComeBackAsOneJsonObjectAndTheSecondQuestionSkipsTheSystemPrompt() {
        AssistantEngine.init(context)
        AssistantEngine.threads = minOf(4, Runtime.getRuntime().availableProcessors())
        val state = AssistantEngine.loadNowBlocking()
        assumeTrue("the model is in Uno Lab: $state", state is AssistantEngine.State.Ready)
        val (first, firstMs) = ask("pasta timer 12 minutes", Assistant.ACTION)
        val (second, secondMs) = ask("turn on the flashlight", Assistant.ACTION)
        Log.i("AssistantTest", "first ${firstMs} ms: $first | second ${secondMs} ms: $second")
        val shape = Regex("""^\{"action":"[a-z_]+".*\}$""")
        assertTrue(first, shape.matches(first))
        assertTrue(second, shape.matches(second))
        assertTrue("the second question reused the decoded system prompt: $firstMs vs $secondMs ms", secondMs < firstMs)
    }

    /**
     * The stock weights with the planner's adapter beside them (`*.lora.gguf`): an answer is the model's own, so a follow-up is read
     * against what came before it, and a plan is written with the adapter on. Merged into the weights, the adapter answered this
     * follow-up "I don't have a person in mind" nearly every time.
     */
    @Test fun answersComeFromTheWeightsAndPlansFromTheAdapter() {
        AssistantEngine.init(context)
        AssistantEngine.threads = minOf(4, Runtime.getRuntime().availableProcessors())
        val state = AssistantEngine.loadNowBlocking()
        assumeTrue("the model is in Uno Lab: $state", state is AssistantEngine.State.Ready)
        assumeTrue("a planner adapter is beside it", AssistantEngine.hasAdapter)
        val earlier = "Earlier in this conversation (oldest first):\n- They said: who painted the starry night\n  You answered: Vincent van Gogh painted the Starry Night."
        val (answer, ms) = ask("where did he live", Assistant.ANSWER, earlier)
        Log.i("AssistantTest", "follow-up ${ms} ms: $answer")
        assertTrue(answer, Regex("Netherlands|France|Arles|Amsterdam|Paris|Auvers|Holland|Saint-R", RegexOption.IGNORE_CASE).containsMatchIn(answer))
        val plan = kotlinx.coroutines.runBlocking { AssistantEngine.plan("""{"query":"set a timer for 5 minutes","hash":"adapter-test"}""") }
        Log.i("AssistantTest", "plan: $plan")
        assertTrue(plan, plan.contains("\"steps\"") && plan.contains("timer"))
        // And back: the next answer is the weights' own again.
        val (again, _) = ask("what is the capital of France", Assistant.ANSWER)
        assertTrue(again, again.contains("Paris"))
    }

    @Test fun anAnswerUsesThePhoneContext() {
        AssistantEngine.init(context)
        AssistantEngine.threads = minOf(4, Runtime.getRuntime().availableProcessors())
        val state = AssistantEngine.loadNowBlocking()
        assumeTrue("the model is in Uno Lab: $state", state is AssistantEngine.State.Ready)
        val (answer, ms) = ask("What day is it?", Assistant.ANSWER, "Phone context (from this phone, now):\n- Date: Tuesday 10 November 2026")
        Log.i("AssistantTest", "with context ${ms} ms: $answer")
        assertTrue(answer, answer.contains("Tuesday", ignoreCase = true))
    }

    @Test fun theServiceRefusesAnAppThatIsNotUno() {
        val bound = CountDownLatch(1); var binder: IBinder? = null
        val conn = object : ServiceConnection {
            override fun onServiceConnected(name: ComponentName?, service: IBinder?) { binder = service; bound.countDown() }
            override fun onServiceDisconnected(name: ComponentName?) {}
        }
        assertTrue(context.bindService(Intent(context, AssistantService::class.java), conn, Context.BIND_AUTO_CREATE))
        assertTrue(bound.await(10, TimeUnit.SECONDS))
        try {
            // In-process the call is local, but the check is the same: this app is not Uno.
            IAssistant.Stub.asInterface(binder).state()
            fail("Uno Lab itself should not be let in")
        } catch (_: SecurityException) {
        } finally { context.unbindService(conn) }
    }
}
