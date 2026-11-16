package ggtts.demo

import org.junit.Assert.*
import org.junit.Test

class AudioBoundReadManifestIndependentTest {
    private val audio = byteArrayOf(0x66, 0x4c, 0x61, 0x43, 0x01, 0x02)
    private val chunks = listOf(
        DeliveredChunk(0, "First passage.", 1, -1L, ReadStop.EOS, SampleSpan(0, 12), SampleSpan(0, 10)),
        DeliveredChunk(1, "Second passage.", 0, Long.MIN_VALUE, ReadStop.CAP, SampleSpan(12, 20), SampleSpan(12, 20)),
    )
    private fun make(chunks: List<DeliveredChunk> = this.chunks) =
        AudioBoundReadManifest.create("read-1", audio, 24000, 20L, chunks)
    private fun refuses(block: () -> Unit) {
        try { block(); fail("invalid evidence was accepted") } catch (_: IllegalArgumentException) {}
    }

    @Test fun roundTripPreservesUnsignedSeedsStopAndSpeechGap() {
        val manifest = make()
        val json = manifest.encodeJson()
        assertTrue(json.contains("18446744073709551615"))
        assertTrue(json.contains("9223372036854775808"))
        val decoded = AudioBoundReadManifest.decodeJson(json)
        assertEquals(manifest, decoded)
        decoded.validateAudio(audio, 24000, 20L)
    }

    @Test fun audioCorruptionRateAndCountAreRejected() {
        val manifest = make()
        val planted = audio.clone().also { it[4] = (it[4].toInt() xor 1).toByte() }
        refuses { manifest.validateAudio(planted, 24000, 20L) }
        refuses { manifest.validateAudio(audio, 22050, 20L) }
        refuses { manifest.validateAudio(audio, 24000, 19L) }
        refuses { manifest.validateAudio(audio.copyOf(audio.size - 1), 24000, 20L) }
        manifest.validateAudio(audio, 24000, 20L)
    }

    @Test fun gapsOverlapsAndOutOfBoundsDeliveryAreRefused() {
        refuses { make(listOf(chunks[0], chunks[1].copy(delivered = SampleSpan(13, 20), speech = SampleSpan(13, 20)))) }
        refuses { make(listOf(chunks[0], chunks[1].copy(delivered = SampleSpan(11, 20), speech = SampleSpan(11, 20)))) }
        refuses { make(listOf(chunks[0], chunks[1].copy(delivered = SampleSpan(12, 21), speech = SampleSpan(12, 20)))) }
        refuses { make(listOf(chunks[0], chunks[1].copy(delivered = SampleSpan(12, 19), speech = SampleSpan(12, 19)))) }
        refuses { make(listOf(chunks[0].copy(delivered = SampleSpan(1, 12), speech = SampleSpan(1, 10)), chunks[1])) }
        refuses { make(listOf(chunks[0].copy(speech = SampleSpan(0, 13)), chunks[1])) }
        refuses { make(listOf(chunks[0].copy(speech = SampleSpan(1, 10)), chunks[1])) }
    }

    @Test fun invalidIdentityAndAudioOnlyLegacyAreRefused() {
        refuses { make(listOf(chunks[0], chunks[1].copy(index = 0))) }
        refuses { make(listOf(chunks[0].copy(text = ""), chunks[1])) }
        refuses { make(listOf(chunks[0].copy(draw = -1), chunks[1])) }
        refuses { AudioBoundReadManifest.decodeJson("{}") }
        refuses { AudioBoundReadManifest.decodeJson("") }
    }

    @Test fun serializedCorruptionCannotSilentlyChangeStopSeedOrSpan() {
        val json = make().encodeJson()
        val plants = listOf(
            json.replace("\"stop\":\"eos\"", "\"stop\":\"abort\""),
            json.replace("18446744073709551615", "18446744073709551616"),
            json.replace("18446744073709551615", "018446744073709551615"),
            json.replace("\"index\":1", "\"index\":0"),
            json.replace("\"version\":1", "\"version\":2"),
        )
        for (planted in plants) {
            assertNotEquals("plant must alter serialized record", json, planted)
            refuses { AudioBoundReadManifest.decodeJson(planted) }
        }
    }

    @Test fun unicodeTextAndDefensiveChunkCopySurviveRoundTrip() {
        val mutable = chunks.toMutableList()
        mutable[0] = mutable[0].copy(text = "Test \u4e2d\u6587 \uD83D\uDE00")
        val manifest = make(mutable)
        val before = manifest.encodeJson()
        mutable.clear()
        assertEquals(before, manifest.encodeJson())
        assertEquals(manifest, AudioBoundReadManifest.decodeJson(before))
    }
}
