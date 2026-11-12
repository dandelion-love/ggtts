// The demo's engine as Android's TextToSpeechService: other apps (and the demo's own "Speak through Android"
// button) reach ggtts through the platform's TextToSpeech API. All the work is SynthesisSession's; this adapts it.
//
// It declares every voice LATENCY_VERY_HIGH: the first audio takes tens of seconds on a phone, and a client that
// reads voice latency can see that before choosing it. It is not meant for screen readers.
package ggtts.demo

import android.media.AudioFormat
import android.speech.tts.SynthesisCallback
import android.speech.tts.SynthesisRequest
import android.speech.tts.TextToSpeech
import android.speech.tts.TextToSpeechService
import android.speech.tts.Voice
import android.util.Log
import ggtts.Ggtts
import java.io.File
import java.util.Locale

class GgttsTtsService : TextToSpeechService() {
    private lateinit var session: SynthesisSession
    private lateinit var voices: VoiceStore
    private var current = arrayOf("yue", "HKG", "")

    override fun onCreate() {
        voices = VoiceStore(prefs(this))
        session = SynthesisSession(
            GgttsBinding(File(filesDir, "ggtts-cache").path),
            voices,
            log = { Log.i(MainActivity.TAG, it) },
            // The engine the demo's screen loaded, if it is the same request: one model resident, not two.
            cache = Reader.shared(this),
        )
        super.onCreate()
    }

    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        Reader.trim(this, level)
    }

    override fun onIsLanguageAvailable(lang: String?, country: String?, variant: String?): Int =
        session.availability(lang ?: "", country ?: "")

    override fun onGetLanguage(): Array<String> = current

    override fun onLoadLanguage(lang: String?, country: String?, variant: String?): Int {
        val r = onIsLanguageAvailable(lang, country, variant)
        if (r >= TtsCodes.LANG_AVAILABLE) current = arrayOf(lang ?: "", country ?: "", "")
        return r
    }

    override fun onStop() = session.stop()

    override fun onSynthesizeText(request: SynthesisRequest, callback: SynthesisCallback) {
        val text = request.charSequenceText?.toString() ?: request.text ?: ""
        Log.i(MainActivity.TAG, "system engine: synthesize ${request.language}-${request.country}, ${text.length} chars")
        session.synthesize(request.language ?: "", request.country ?: "", text, Ggtts.DEFAULT_SEED, object : SynthesisOut {
            override val maxBytes = callback.maxBufferSize
            override fun start(sampleRate: Int) {
                callback.start(sampleRate, AudioFormat.ENCODING_PCM_16BIT, 1)
            }
            override fun audio(pcm16: ByteArray, offset: Int, length: Int) {
                callback.audioAvailable(pcm16, offset, length)
            }
            override fun done() {
                callback.done()
            }
            override fun error(code: Int) {
                callback.error(code)
            }
        })
    }

    override fun onGetVoices(): List<Voice> = VOICES.mapNotNull { (name, locale, engine) ->
        if (voices.load(engine) == null) return@mapNotNull null
        Voice(name, locale, Voice.QUALITY_VERY_HIGH, Voice.LATENCY_VERY_HIGH, false, emptySet())
    }

    override fun onIsValidVoiceName(voiceName: String?): Int =
        if (VOICES.any { it.first == voiceName }) TextToSpeech.SUCCESS else TextToSpeech.ERROR

    override fun onLoadVoice(voiceName: String?): Int {
        val v = VOICES.firstOrNull { it.first == voiceName } ?: return TextToSpeech.ERROR
        current = arrayOf(v.second.isO3Language, v.second.isO3Country, "")
        return TextToSpeech.SUCCESS
    }

    override fun onGetDefaultVoiceNameFor(lang: String?, country: String?, variant: String?): String? {
        val engine = Languages.engineFor(lang ?: "", country ?: "") ?: return null
        return VOICES.firstOrNull { it.third == engine }?.first
    }

    companion object {
        /** Voice name, its locale, and the engine that speaks it. */
        val VOICES = listOf(
            Triple("yue-hk-ggtts-qwen3", Locale("yue", "HK"), "qwen3-tts"),
            Triple("en-us-ggtts-kitten", Locale.US, "kitten-tts-2"),
        )

        fun prefs(context: android.content.Context): KeyValue {
            val p = context.getSharedPreferences("ggtts-system-engine", MODE_PRIVATE)
            return object : KeyValue {
                override fun get(key: String) = p.getString(key, null)
                override fun put(key: String, value: String) {
                    p.edit().putString(key, value).apply()
                }
            }
        }
    }
}
