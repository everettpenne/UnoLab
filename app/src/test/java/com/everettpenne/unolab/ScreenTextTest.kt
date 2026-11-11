package com.everettpenne.unolab

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ScreenTextTest {
    @Test fun repeatedLabelsAreKeptOnceAndBlanksDropped() {
        // A view and its container often carry the same label; the model does not need it twice.
        assertEquals("Battery\n68%", ScreenText.join(listOf("Battery", "Battery", "", "68%", "  ")))
        assertNull(ScreenText.join(listOf("", "   ")))
    }

    @Test fun aLongScreenIsCutAtAWordRatherThanMidWord() {
        val line = "the quick brown fox jumps over the lazy dog ".repeat(200)
        val capped = ScreenText.cap(line)!!
        assertEquals(ScreenText.MAX_CHARS >= capped.length, true)
        // Cut on a space, not inside a word: the last thing in it is a whole word.
        assertTrue(capped.substringAfterLast(' ') in setOf("the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"))
        assertTrue(capped.length < ScreenText.MAX_CHARS)
    }

    @Test fun aShortScreenIsLeftAloneAndEmptyOnesAreNull() {
        assertEquals("Hello", ScreenText.cap("  Hello  "))
        assertNull(ScreenText.cap(null))
        assertNull(ScreenText.cap("   "))
    }
}
