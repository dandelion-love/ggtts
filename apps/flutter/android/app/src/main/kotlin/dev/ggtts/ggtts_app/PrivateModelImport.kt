package dev.ggtts.ggtts_app

import java.io.File
import java.io.InputStream
import java.util.UUID

/**
 * Streams a selected document to a unique app-owned directory. The target is
 * exposed only after validation and rename; no provider name becomes a path.
 * Bounds apply to bytes actually read, even when provider metadata is absent.
 */
internal object PrivateModelImport {
    private const val MODEL_LIMIT = 8L * 1024 * 1024 * 1024
    private const val VOICE_LIMIT = 64L * 1024 * 1024
    private const val PREFIX_LIMIT = 65536

    fun import(
        root: File,
        kind: String,
        input: InputStream,
        cancelled: () -> Boolean = { false },
        limitOverride: Long? = null,
    ): File = input.use { source -> importOpened(root, kind, source, cancelled, limitOverride) }

    private fun importOpened(
        root: File,
        kind: String,
        input: InputStream,
        cancelled: () -> Boolean,
        limitOverride: Long?,
    ): File {
        require(kind == "model" || kind == "codec" || kind == "voice")
        val limit = limitOverride ?: if (kind == "voice") VOICE_LIMIT else MODEL_LIMIT
        require(limit > 0)
        check(root.isDirectory || root.mkdirs()) { "Cannot create import directory" }
        var dir: File? = null
        repeat(10) {
            if (dir == null) {
                val candidate = File(root, UUID.randomUUID().toString())
                if (candidate.mkdir()) dir = candidate
            }
        }
        val owned = checkNotNull(dir) { "Cannot reserve import directory" }
        val partial = File(owned, "incoming.part")
        val target = File(owned, if (kind == "voice") "voice.json" else "model.gguf")
        var complete = false
        try {
            input.use { source ->
                partial.outputStream().use { out ->
                    val buffer = ByteArray(64 * 1024)
                    var total = 0L
                    var validated = false
                    val prefix = ArrayList<Byte>()
                    while (true) {
                        check(!cancelled()) { "Import cancelled" }
                        val n = source.read(buffer)
                        if (n < 0) break
                        check(n > 0) { "Document returned no data" }
                        check(n.toLong() <= limit - total) { "Selected file exceeds the import limit" }
                        total += n
                        if (!validated && prefix.size < PREFIX_LIMIT) {
                            val take = minOf(n, PREFIX_LIMIT - prefix.size)
                            repeat(take) { prefix.add(buffer[it]) }
                        }
                        if (!validated) {
                        val probe = prefix.toByteArray()
                        if (kind != "voice" && probe.size >= 4) {
                            check(probe[0] == 'G'.code.toByte() && probe[1] == 'G'.code.toByte() &&
                                probe[2] == 'U'.code.toByte() && probe[3] == 'F'.code.toByte()) {
                                "Selected file is not GGUF"
                            }
                            validated = true
                        }
                        if (kind == "voice") {
                            val first = probe.firstOrNull { it.toInt() !in listOf(9, 10, 13, 32) }
                            if (first != null) {
                                check(first == '{'.code.toByte()) { "Selected voice is not a JSON object" }
                                validated = true
                            } else check(probe.size < PREFIX_LIMIT) { "Voice JSON prefix is too long" }
                        }
                        }
                        out.write(buffer, 0, n)
                    }
                    check(total > 0) { "Selected file is empty" }
                    val bytes = prefix.toByteArray()
                    if (kind == "voice") {
                        val first = bytes.firstOrNull { it.toInt() !in listOf(9, 10, 13, 32) }
                        check(first == '{'.code.toByte()) { "Selected voice is not a JSON object" }
                    } else {
                        check(bytes.size >= 4 && bytes[0] == 'G'.code.toByte() && bytes[1] == 'G'.code.toByte() &&
                            bytes[2] == 'U'.code.toByte() && bytes[3] == 'F'.code.toByte()) { "Selected file is not GGUF" }
                    }
                    out.fd.sync()
                }
            }
            check(!cancelled()) { "Import cancelled" }
            check(!target.exists() && partial.renameTo(target)) { "Cannot publish imported file" }
            check(!cancelled()) { "Import cancelled" }
            complete = true
            return target
        } finally {
            if (!complete) {
                partial.delete()
                target.delete()
                owned.delete()
            }
        }
    }
}
