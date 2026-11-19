package dev.ggtts.ggtts_app

import org.junit.Assert.*
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.File
import java.io.IOException
import java.io.InputStream
import java.nio.file.Files

class PrivateModelImportTest {
    private fun temp(): File = Files.createTempDirectory("model-import-").toFile()

    @Test fun importsDistinctFilesAndNeverOverwrites() {
        val root = temp()
        try {
            val one = PrivateModelImport.import(root, "model", ByteArrayInputStream("GGUFone".toByteArray()))
            val two = PrivateModelImport.import(root, "model", ByteArrayInputStream("GGUFtwo".toByteArray()))
            assertNotEquals(one.path, two.path)
            assertEquals("GGUFone", one.readText())
            assertEquals("GGUFtwo", two.readText())
        } finally { root.deleteRecursively() }
    }

    @Test fun malformedOversizedCancelledAndInterruptedLeaveNoArtifacts() {
        val cases: List<Pair<String, () -> InputStream>> = listOf(
            "model" to { ByteArrayInputStream("bad".toByteArray()) },
            "voice" to { ByteArrayInputStream("[]".toByteArray()) },
            "model" to { ByteArrayInputStream("GGUFtoo-long".toByteArray()) },
            "model" to { object : InputStream() {
                override fun read(): Int = throw IOException("broken")
                override fun read(b: ByteArray, off: Int, len: Int): Int = throw IOException("broken")
            } },
        )
        for ((index, case) in cases.withIndex()) {
            val root = temp()
            try {
                assertThrows(Exception::class.java) {
                    PrivateModelImport.import(root, case.first, case.second(), limitOverride = if (index == 2) 4 else 64)
                }
                assertTrue(root.listFiles()!!.isEmpty())
            } finally { root.deleteRecursively() }
        }
        val root = temp()
        try {
            assertThrows(Exception::class.java) {
                PrivateModelImport.import(root, "model", ByteArrayInputStream("GGUFvalid".toByteArray()), cancelled = { true })
            }
            assertTrue(root.listFiles()!!.isEmpty())
        } finally { root.deleteRecursively() }
    }

    @Test fun acceptsWhitespacePrefixedVoiceObject() {
        val root = temp()
        try {
            val file = PrivateModelImport.import(root, "voice", ByteArrayInputStream(" \n { }".toByteArray()))
            assertEquals(" \n { }", file.readText())
        } finally { root.deleteRecursively() }
    }
    @Test fun midstreamFailureAndCancellationRemovePartial() {
        val root = temp()
        try {
            var calls = 0
            val broken = object : InputStream() {
                override fun read(): Int = throw UnsupportedOperationException()
                override fun read(b: ByteArray, off: Int, len: Int): Int {
                    if (calls++ > 0) throw IOException("provider stopped")
                    val prefix = "GGUFpayload".toByteArray()
                    prefix.copyInto(b, off)
                    return prefix.size
                }
            }
            assertThrows(IOException::class.java) { PrivateModelImport.import(root, "model", broken) }
            assertTrue(root.listFiles()!!.isEmpty())

            var checks = 0
            assertThrows(IllegalStateException::class.java) {
                PrivateModelImport.import(root, "model", ByteArrayInputStream("GGUFdata".toByteArray()),
                    cancelled = { ++checks >= 2 })
            }
            assertTrue(root.listFiles()!!.isEmpty())
        } finally { root.deleteRecursively() }
    }

    @Test fun closesSourceWhenArgumentsAreRejected() {
        val root = temp()
        var closed = false
        val source = object : ByteArrayInputStream("GGUF".toByteArray()) {
            override fun close() { closed = true; super.close() }
        }
        try {
            assertThrows(IllegalArgumentException::class.java) { PrivateModelImport.import(root, "unknown", source) }
            assertTrue(closed)
            assertTrue(root.listFiles()!!.isEmpty())
        } finally { root.deleteRecursively() }
    }

}
