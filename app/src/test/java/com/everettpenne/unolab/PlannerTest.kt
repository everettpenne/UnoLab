package com.everettpenne.unolab

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class PlannerTest {
    private val json = """{"v":1,"query":"timer for 20 minutes","context":"Today: Friday 13 November",""" +
        """"grounding":"Clock is installed","catalogue":"Tools:\n- timer: starts a timer","grammar":"root ::= plan","hash":"abc123"}"""

    @Test fun aRequestFromTheLauncherIsRead() {
        val request = Planner.parse(json)!!
        assertEquals("timer for 20 minutes", request.query)
        assertEquals("Today: Friday 13 November", request.context)
        assertEquals("Clock is installed", request.grounding)
        assertEquals("root ::= plan", request.grammar)
        // The hash keys the cached prompt, so a change of tools re-decodes it and nothing else does.
        assertEquals("abc123", request.hash)
    }

    @Test fun thePromptIsTheInstructionsPlusTheGeneratedCatalogue() {
        val prefix = Planner.prefix(Planner.parse(json)!!)
        assertTrue(prefix.startsWith("<|im_start|>system\n"))
        assertTrue(prefix.contains("You are the planner for Uno"))
        assertTrue(prefix.contains("- timer: starts a timer"))
        assertTrue(prefix.endsWith("<|im_end|>\n"))
    }

    @Test fun theTurnCarriesWhatIsKnownAndTheRequest() {
        val turn = Planner.turn(Planner.parse(json)!!)
        assertTrue(turn.contains("<context>\nToday: Friday 13 November\n</context>"))
        assertTrue(turn.contains("<known>\nClock is installed\n</known>"))
        assertTrue(turn.contains("Request: timer for 20 minutes"))
        // Thinking off, as elsewhere, so the model goes straight to the plan.
        assertTrue(turn.endsWith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
    }

    @Test fun anEmptyContextIsLeftOutRatherThanSentAsAnEmptyBlock() {
        val bare = Planner.parse("""{"query":"torch","catalogue":"x","grammar":"y","hash":"h"}""")!!
        val turn = Planner.turn(bare)
        assertTrue(!turn.contains("<context>"))
        assertTrue(!turn.contains("<known>"))
        assertTrue(turn.contains("Request: torch"))
    }

    @Test fun aRequestThisSideCannotServeComesBackAsARefusal() {
        val refusal = JSONObject(Planner.error("No model in Uno Lab"))
        assertEquals("No model in Uno Lab", refusal.getString("refuse"))
        assertTrue(JSONObject(Planner.error("say \"hello\"\n")).getString("refuse").contains("hello"))
    }

    @Test fun aBlankOrBrokenRequestIsNotAPlan() {
        assertNull(Planner.parse(""))
        assertNull(Planner.parse("{}"))
        assertNull(Planner.parse("not json"))
    }
}
