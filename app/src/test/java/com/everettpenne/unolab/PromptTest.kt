package com.everettpenne.unolab

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class PromptTest {
    @Test fun theFirstTurnOpensWithTheSystemPromptAndThinkingOffIsAnEmptyThinkBlock() {
        val t = Prompt.turn("Hi", first = true, closePrevious = false, thinking = false)
        assertEquals("<|im_start|>system\n${Prompt.SYSTEM}<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", t)
    }

    @Test fun aLaterTurnClosesTheLastReplyFirstAndThinkingOnLeavesTheBlockOpen() {
        val t = Prompt.turn(" Why? ", first = false, closePrevious = true, thinking = true)
        assertEquals("<|im_end|>\n<|im_start|>user\nWhy?<|im_end|>\n<|im_start|>assistant\n<think>\n", t)
    }

    @Test fun aReplyIsSplitIntoItsThinkingAndItsAnswer() {
        assertEquals("" to "Answer", Prompt.split("  Answer", thinking = false))
        assertEquals("working it out" to "", Prompt.split("working it out", thinking = true))
        assertEquals("step 1" to "Done.", Prompt.split("step 1\n</think>\n\nDone.", thinking = true))
    }

    @Test fun energyIsTheDrawOverIdleForTheTime() {
        assertEquals(10.0, Energy.joules(7.0, 2.0, 2.0), 1e-9)
        assertEquals(0.0, Energy.joules(1.0, 2.0, 5.0), 1e-9)
        assertEquals(0.5, Energy.perUnit(10.0, 20.0)!!, 1e-9)
        assertNull(Energy.perUnit(10.0, 0.0))
        assertTrue(Energy.joules(8.29, .65, 1.0) > 7.6)
    }
}
