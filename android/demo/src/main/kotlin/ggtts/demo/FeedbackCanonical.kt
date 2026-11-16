package ggtts.demo

/** Python json.dumps(ensure_ascii=True, sort_keys=True, separators=(",", ":")) plus HTML-safe escaping. */
internal fun feedbackCanonical(value: Any?): String = when (value) {
    null -> "null"
    is Boolean -> value.toString()
    is Long -> value.toString()
    is String -> buildString {
        append('"')
        for (ch in value) {
            when (ch) {
                '"' -> append("\\\"")
                '\\' -> append("\\\\")
                '\b' -> append("\\b")
                '\t' -> append("\\t")
                '\n' -> append("\\n")
                '\u000c' -> append("\\f")
                '\r' -> append("\\r")
                else -> if (ch.code !in 0x20..0x7e || ch == '<' || ch == '>' || ch == '&')
                    append("\\u" + ch.code.toString(16).padStart(4, '0'))
                else append(ch)
            }
        }
        append('"')
    }
    is List<*> -> value.joinToString(",", "[", "]") { feedbackCanonical(it) }
    is Map<*, *> -> {
        require(value.keys.all { it is String })
        value.keys.map { it as String }.sorted().joinToString(",", "{", "}") {
            feedbackCanonical(it) + ":" + feedbackCanonical(value[it])
        }
    }
    else -> error("unsupported JSON value")
}

/** Decode strict base64 and inspect the FLAC STREAMINFO required by the shared page contract. */
internal fun validFeedbackFlac(uri: String): Boolean {
    val prefix = "data:audio/flac;base64,"
    if (!uri.startsWith(prefix)) return false
    val s = uri.substring(prefix.length)
    if (s.isEmpty() || s.length % 4 != 0 || !s.matches(Regex("^[A-Za-z0-9+/]+={0,2}$"))) return false
    val alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    val out = ArrayList<Int>()
    for (i in s.indices step 4) {
        val last = i + 4 == s.length
        val a = alphabet.indexOf(s[i]); val b = alphabet.indexOf(s[i + 1])
        val c = if (s[i + 2] == '=') -1 else alphabet.indexOf(s[i + 2])
        val d = if (s[i + 3] == '=') -1 else alphabet.indexOf(s[i + 3])
        if (a < 0 || b < 0 || (!last && (c < 0 || d < 0)) ||
            (c < 0 && d >= 0) || (c < 0 && b and 15 != 0) || (d < 0 && c >= 0 && c and 3 != 0)) return false
        if (out.size < 42) out.add((a shl 2) or (b shr 4))
        if (c >= 0 && out.size < 42) out.add(((b and 15) shl 4) or (c shr 2))
        if (d >= 0 && out.size < 42) out.add(((c and 3) shl 6) or d)
    }
    if (out.size < 42 || out.subList(0, 4) != listOf(102, 76, 97, 67) ||
        out[4] and 0x7f != 0 || out[5] != 0 || out[6] != 0 || out[7] != 34) return false
    val rate = (out[18] shl 12) or (out[19] shl 4) or (out[20] shr 4)
    val count = (out[21] and 15) or out[22] or out[23] or out[24] or out[25]
    return rate > 0 && count != 0
}
