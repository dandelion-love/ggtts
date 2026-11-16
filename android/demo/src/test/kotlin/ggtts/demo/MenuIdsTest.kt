// The ⋮ menu's ids: distinct, and used by name. Two items with one id open each other's action, as Saved reads and
// the Licences sheet would have, both taking 5. Each test names the planted fault it catches.
package ggtts.demo

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class MenuIdsTest {
    /** Plant: two items given one id. */
    @Test
    fun everyMenuIdIsDistinct() {
        val ids = MenuIds::class.java.declaredFields
            .filter { java.lang.reflect.Modifier.isStatic(it.modifiers) && it.type == Int::class.javaPrimitiveType }
            .associate { it.name to it.getInt(null) }
        assertEquals("ALL lists every id constant", ids.values.sorted(), MenuIds.ALL.sorted())
        val clashes = ids.entries.groupBy { it.value }.filterValues { it.size > 1 }
        assertTrue("menu ids shared: $clashes", clashes.isEmpty())
    }

    /** Plant: an item added or matched by a bare number, which no id check can see. */
    @Test
    fun theMenuUsesTheNamesNotNumbers() {
        val src = File("src/main/kotlin/ggtts/demo/MainActivity.kt").readText()
        val bare = Regex("""header\.menu\.add\(\s*\d+\s*,\s*\d+""").findAll(src).map { it.value }.toList() +
            Regex("""^\s*\d+\s*->""", RegexOption.MULTILINE).findAll(src).map { it.value.trim() }.toList()
        assertTrue("menu items by number in MainActivity.kt: $bare", bare.isEmpty())
        assertTrue(Regex("""header\.menu\.add\(0, MenuIds\.""").containsMatchIn(src))
    }
}
