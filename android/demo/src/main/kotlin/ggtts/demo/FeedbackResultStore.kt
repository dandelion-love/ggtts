package ggtts.demo

import java.io.File
import java.io.FileOutputStream
import java.security.MessageDigest
import java.util.UUID

/** An already prepared, canonical ASCII input; the host never invents read evidence. */
class PreparedFeedbackInput(val canonicalJson: String) {
    val id: String
    val digest: String
    val cjkCounts: Map<String, Int>

    init {
        require(canonicalJson.all { it.code in 0x20..0x7e })
        val parsed = JsonReader(canonicalJson).readDocument()
        require(feedbackCanonical(parsed) == canonicalJson) { "input must be canonical JSON" }
        val root = obj(parsed)
        require(root.keys == setOf("format", "id", "items") && root["format"] == "ggtts-feedback-input/1")
        id = validId(root["id"])
        val items = root["items"] as? List<*> ?: error("items must be an array")
        require(items.isNotEmpty())
        val counts = linkedMapOf<String, Int>()
        for (raw in items) {
            val item = obj(raw)
            require(item.keys == setOf("id", "text", "audio"))
            val itemId = validId(item["id"])
            val text = item["text"] as? String ?: error("text must be a string")
            require(text.isNotEmpty())
            val audio = item["audio"] as? String ?: error("audio must be a string")
            require(validFeedbackFlac(audio)) { "invalid FLAC" }
            require(counts.put(itemId, text.codePoints().filter { cjk(it) }.count().toInt()) == null)
        }
        cjkCounts = counts.toMap()
        digest = sha(canonicalJson.toByteArray(Charsets.US_ASCII))
    }

    fun validateResult(json: String) {
        require(json.length <= 1_000_000)
        val root = obj(JsonReader(json).readDocument())
        require(root.keys == setOf("format", "id", "input_sha256", "note", "items"))
        require(root["format"] == "ggtts-feedback-result/1" && root["id"] == id && root["input_sha256"] == digest)
        require(root["note"] is String)
        val items = root["items"] as? List<*> ?: error("items must be an array")
        require(items.size == cjkCounts.size)
        val seen = mutableSetOf<String>()
        for (raw in items) {
            val item = obj(raw)
            require(item.keys == setOf("id", "marks", "rating", "note"))
            val itemId = item["id"] as? String ?: error("missing item id")
            val count = cjkCounts[itemId] ?: error("unknown item")
            require(seen.add(itemId) && item["rating"] in setOf("accept", "reject", "unsure") && item["note"] is String)
            val marks = item["marks"] as? List<*> ?: error("marks must be an array")
            val unique = mutableSetOf<Long>()
            for (mark in marks) {
                require(mark is Long && mark >= 0 && mark < count && unique.add(mark))
            }
        }
    }
}

private fun obj(value: Any?): Map<String, Any?> {
    @Suppress("UNCHECKED_CAST")
    return value as? Map<String, Any?> ?: error("expected object")
}
private fun validId(value: Any?): String {
    val id = value as? String ?: error("missing id")
    require(id.matches(Regex("^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")))
    return id
}
private fun cjk(cp: Int): Boolean = cp in 0x3400..0x4dbf || cp in 0x4e00..0x9fff ||
    cp in 0xf900..0xfaff || cp in 0x20000..0x2fa1f || cp in 0x30000..0x323af
private fun sha(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
    .digest(bytes).joinToString("") { "%02x".format(it) }

/** Validated result files are keyed by exact input digest; writes are atomic within this directory. */
class FeedbackResultStore(private val dir: File) {
    companion object { private val lock = Any() }

    fun save(input: PreparedFeedbackInput, json: String) = synchronized(lock) {
        input.validateResult(json)
        write(input, json)
    }

    /** Preserve both sources when an existing private result differs or is unreadable. */
    fun restoreExport(input: PreparedFeedbackInput, json: String): Boolean = synchronized(lock) {
        input.validateResult(json)
        val target = File(dir, input.digest + ".json")
        if (target.exists()) return@synchronized load(input) == json
        write(input, json)
        true
    }

    private fun write(input: PreparedFeedbackInput, json: String) {
        check(dir.isDirectory || dir.mkdirs())
        val target = File(dir, input.digest + ".json")
        val temp = File(dir, UUID.randomUUID().toString() + ".tmp")
        try {
            FileOutputStream(temp).use { it.write(json.toByteArray(Charsets.UTF_8)); it.fd.sync() }
            check(temp.renameTo(target)) { "cannot publish feedback" }
        } finally { temp.delete() }
    }

    fun load(input: PreparedFeedbackInput): String? = try {
        val json = File(dir, input.digest + ".json").readText(Charsets.UTF_8)
        input.validateResult(json)
        json
    } catch (_: Exception) { null }
}
