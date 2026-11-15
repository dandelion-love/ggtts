// The options document as the app holds it: the library's sections (load, render, output, observe, ...) of keys, as
// the options schema names them (Ggtts.optionsSchema()). The app never checks a key itself: the binding refuses one it
// does not take, by its key, so an option the binding gains reaches the app's requests and saved voices with no new
// field here. Plain Kotlin, no Android and no JSON library, so the JVM tests and the host binding test run it as is.
package ggtts.demo

/**
 * An options document: section name to key to value. A value is a String, a Long, a Double, a Boolean or a List of
 * those; an Int is kept as a Long, so a document built in code equals the same document read back. Immutable, equal by
 * content, and written with its keys sorted, so equal documents write the same JSON.
 */
class Options private constructor(private val sections: Map<String, Map<String, Any>>) {
    /** The value of [section].[key], or null when the document does not give it. */
    operator fun get(section: String, key: String): Any? = sections[section]?.get(key)

    /** Whether the document has [section] at all. */
    fun has(section: String): Boolean = sections.containsKey(section)

    /** This document with [section].[key] set to [value]; a null [value] removes the key (and an emptied section). */
    fun with(section: String, key: String, value: Any?): Options {
        val s = (sections[section] ?: emptyMap()).toMutableMap()
        if (value == null) s.remove(key) else s[key] = normal(value)
        val all = sections.toMutableMap()
        if (s.isEmpty()) all.remove(section) else all[section] = s.toSortedMap()
        return Options(all.toSortedMap())
    }

    /** Only the [names] sections of this document. */
    fun only(vararg names: String): Options = Options(sections.filterKeys { it in names }.toSortedMap())

    /** Every section of this document but the [names] ones. */
    fun without(vararg names: String): Options = Options(sections.filterKeys { it !in names }.toSortedMap())

    /** The document as JSON, keys sorted. */
    fun toJson(): String = StringBuilder().also { write(it, sections) }.toString()

    override fun equals(other: Any?) = other is Options && other.sections == sections
    override fun hashCode() = sections.hashCode()
    override fun toString() = toJson()

    companion object {
        val EMPTY = Options(sortedMapOf())

        /**
         * Read a document. Throws IllegalArgumentException when [json] is not an object of section objects, or a value
         * is not one the app holds (an object below a section). Unknown sections and keys are kept: the binding is the
         * one that refuses them.
         */
        fun parse(json: String): Options {
            val p = Parser(json)
            val top = p.value()
            p.end()
            require(top is Map<*, *>) { "an options document is a JSON object" }
            var o = EMPTY
            for ((section, keys) in top) {
                require(keys is Map<*, *>) { "options section $section is not an object" }
                for ((key, value) in keys) {
                    require(value != null) { "options key $section.$key is null" }
                    o = o.with(section as String, key as String, value)
                }
            }
            return o
        }

        private fun normal(v: Any): Any = when (v) {
            is Int -> v.toLong()
            is Float -> v.toDouble()
            is String, is Long, is Double, is Boolean -> v
            is List<*> -> v.map { normal(requireNotNull(it) { "a null in an options list" }) }
            else -> throw IllegalArgumentException("an options value is a string, number, boolean or list, not $v")
        }

        private fun write(b: StringBuilder, v: Any) {
            when (v) {
                is Map<*, *> -> {
                    b.append('{')
                    v.entries.forEachIndexed { i, (k, x) ->
                        if (i > 0) b.append(", ")
                        write(b, k as String)
                        b.append(": ")
                        write(b, x!!)
                    }
                    b.append('}')
                }
                is List<*> -> {
                    b.append('[')
                    v.forEachIndexed { i, x ->
                        if (i > 0) b.append(", ")
                        write(b, x!!)
                    }
                    b.append(']')
                }
                is String -> {
                    b.append('"')
                    for (c in v) when {
                        c == '"' -> b.append("\\\"")
                        c == '\\' -> b.append("\\\\")
                        c == '\n' -> b.append("\\n")
                        c == '\r' -> b.append("\\r")
                        c == '\t' -> b.append("\\t")
                        c < ' ' -> b.append("\\u%04x".format(c.code))
                        else -> b.append(c)
                    }
                    b.append('"')
                }
                else -> b.append(v.toString())
            }
        }
    }

    /** A small JSON reader: objects, arrays, strings, numbers, booleans and null. */
    private class Parser(private val s: String) {
        private var i = 0

        fun end() {
            space()
            require(i == s.length) { "options: unexpected text at $i" }
        }

        fun value(): Any? {
            space()
            require(i < s.length) { "options: the document ends early" }
            return when (val c = s[i]) {
                '{' -> obj()
                '[' -> arr()
                '"' -> str()
                't' -> word("true", true)
                'f' -> word("false", false)
                'n' -> word("null", null)
                else -> if (c == '-' || c in '0'..'9') num() else throw IllegalArgumentException("options: '$c' at $i")
            }
        }

        private fun obj(): Map<String, Any?> {
            val m = linkedMapOf<String, Any?>()
            i++
            space()
            if (peek('}')) return m
            while (true) {
                space()
                require(i < s.length && s[i] == '"') { "options: a key is a string, at $i" }
                val k = str()
                space()
                expect(':')
                require(!m.containsKey(k)) { "options: $k is given twice" }
                m[k] = value()
                space()
                if (peek('}')) return m
                expect(',')
            }
        }

        private fun arr(): List<Any?> {
            val l = mutableListOf<Any?>()
            i++
            space()
            if (peek(']')) return l
            while (true) {
                l += value()
                space()
                if (peek(']')) return l
                expect(',')
            }
        }

        private fun str(): String {
            val b = StringBuilder()
            i++
            while (true) {
                require(i < s.length) { "options: a string does not end" }
                val c = s[i++]
                when (c) {
                    '"' -> return b.toString()
                    '\\' -> {
                        require(i < s.length) { "options: a string does not end" }
                        when (val e = s[i++]) {
                            'n' -> b.append('\n')
                            't' -> b.append('\t')
                            'r' -> b.append('\r')
                            'b' -> b.append('\b')
                            'f' -> b.append('\u000c')
                            'u' -> {
                                require(i + 4 <= s.length) { "options: a short \\u escape" }
                                b.append(s.substring(i, i + 4).toInt(16).toChar())
                                i += 4
                            }
                            else -> b.append(e)
                        }
                    }
                    else -> b.append(c)
                }
            }
        }

        private fun num(): Any {
            val start = i
            while (i < s.length && (s[i] in "+-.eE" || s[i] in '0'..'9')) i++
            val t = s.substring(start, i)
            return t.toLongOrNull() ?: t.toDoubleOrNull() ?: throw IllegalArgumentException("options: bad number $t")
        }

        private fun word(w: String, v: Any?): Any? {
            require(s.startsWith(w, i)) { "options: unexpected text at $i" }
            i += w.length
            return v
        }

        private fun space() {
            while (i < s.length && s[i].isWhitespace()) i++
        }

        private fun peek(c: Char): Boolean = (i < s.length && s[i] == c).also { if (it) i++ }

        private fun expect(c: Char) {
            require(i < s.length && s[i] == c) { "options: expected '$c' at $i" }
            i++
        }
    }
}
