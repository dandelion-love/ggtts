// Copying a model file the user picked into the models directory, checked on the way in: only a GGUF or a voice JSON,
// never an empty file, never over a file already there, and for the default Qwen3-TTS files, only the exact bytes
// their published sha256 names. A copy goes to NAME.part and is renamed when complete, so a cancelled or failed copy
// never appears in the lists.
package ggtts.demo

import java.io.ByteArrayOutputStream
import java.io.File
import java.io.IOException
import java.io.InputStream
import java.security.DigestInputStream
import java.security.MessageDigest

class ImportException(message: String) : IOException(message)

class ModelImport(
    private val dir: File,
    /** File name to sha256 (lowercase hex) for files whose bytes are known. */
    private val known: Map<String, String> = DEFAULT_MODEL_SHA256,
) {
    companion object {
        /** The default Qwen3-TTS model's two files, as the README lists them. */
        val DEFAULT_MODEL_SHA256 = mapOf(
            "qwen-talker-hk-cantonese-cc0-v0.0.5-q8_0.gguf" to
                "4eb27e5a3434d15ca818814b8e4f0f1941716654afb8543866732b62f9227041",
            "qwen-tokenizer-12hz-q8_0.gguf" to "7771c653aea5d26cdbf461cf00279b78e6c97af5f313da3a8cb3d2b61b837824",
        )
        const val PART = ".part"
        private val GGUF_MAGIC = "GGUF".toByteArray()

        /**
         * Whether [name] names a file directly inside the models directory: not empty, no path separator, not `.` or
         * `..`, no NUL. The adb extras take only such names, so another app can only choose among models the user
         * has already put there, never point the demo (and through it the system engine) at a file elsewhere.
         */
        fun isPlainName(name: String): Boolean =
            name.isNotEmpty() && name != "." && name != ".." && name.none { it == '/' || it == '\\' || it == '\u0000' }

        /** The scripted extras among [names] (null where absent) that [isPlainName] refuses, in order. */
        fun refusedNames(names: List<String?>): List<String> = names.filterNotNull().filterNot { isPlainName(it) }

        /** The files the screen lists: complete ones only, never a copy in progress. */
        fun listed(dir: File): List<String> =
            dir.listFiles()?.filter { it.isFile && !it.name.endsWith(PART) }?.map { it.name }?.sorted() ?: emptyList()
    }

    /**
     * Copy [input] into the directory as [name]. [size] is the size the picker reported, or null if it gave none.
     * Throws [ImportException] with a message for the screen; on any failure nothing new is listed.
     */
    fun importFile(name: String, size: Long?, input: InputStream): File {
        val base = File(name).name
        val kind = when {
            base.endsWith(".gguf", ignoreCase = true) -> "gguf"
            base.endsWith(".json", ignoreCase = true) -> "json"
            else -> throw ImportException("$base is not a model file: import a .gguf model or a .json voice")
        }
        val target = File(dir, base)
        if (target.exists()) throw ImportException("$base is already imported; it was not replaced")
        dir.mkdirs()
        val part = File(dir, base + PART)
        val digest = MessageDigest.getInstance("SHA-256")
        var copied = 0L
        try {
            DigestInputStream(input, digest).use { src ->
                part.outputStream().use { out ->
                    // The bytes read to recognise the file, written out before the rest.
                    val head = ByteArrayOutputStream()
                    when (kind) {
                        "gguf" -> {
                            val magic = ByteArray(4)
                            val n = src.readNBytesCompat(magic)
                            head.write(magic, 0, n)
                            if (n == 0) throw ImportException("$base is empty")
                            if (n < 4 || !magic.contentEquals(GGUF_MAGIC)) {
                                throw ImportException("$base is not a GGUF file")
                            }
                        }
                        "json" -> {
                            // Any amount of leading whitespace (up to 64 KiB), then the object's '{'.
                            var c: Int
                            do {
                                c = src.read()
                                if (c >= 0) head.write(c)
                            } while (c >= 0 && c.toChar().isWhitespace() && head.size() < 65536)
                            if (head.size() == 0) throw ImportException("$base is empty")
                            if (c != '{'.code) throw ImportException("$base is not a voice JSON file")
                        }
                    }
                    head.writeTo(out)
                    copied = head.size() + src.copyTo(out, 1 shl 20)
                }
            }
            if (size != null && copied != size) {
                throw ImportException("$base: copied $copied bytes of $size; the copy is incomplete")
            }
            val want = known[base]
            if (want != null) {
                val got = digest.digest().joinToString("") { "%02x".format(it) }
                if (got != want) throw ImportException("$base does not match its published sha256 ($got, not $want)")
            }
            if (!part.renameTo(target)) throw ImportException("could not move $base into place")
            return target
        } catch (e: Exception) {
            part.delete()
            throw if (e is ImportException) e else ImportException("$base: ${e.message}")
        }
    }
}

/** Read up to buf.size bytes (InputStream.readNBytes is API 33; this app supports 24). */
private fun InputStream.readNBytesCompat(buf: ByteArray): Int {
    var n = 0
    while (n < buf.size) {
        val r = read(buf, n, buf.size - n)
        if (r < 0) break
        n += r
    }
    return n
}
