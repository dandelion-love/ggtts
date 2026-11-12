// FirstRender.kt: load an engine, render one line, write it to a WAV file. It runs on a desktop JVM as it is; in an
// Android app, the same calls run off the UI thread, with cacheDir under context.filesDir.
//
//   main(arrayOf())                                        Qwen3-TTS with its default model (downloaded once, 2.33 GB)
//   main(arrayOf("qwen3-tts", "talker.gguf", "codec.gguf"))  model files you already have
//   main(arrayOf("kitten-tts-2", "lm.gguf", "s3gen.gguf", "voice.json"))
//   main(arrayOf("fake"))                                  the test engine: no files, a tone instead of speech
package firstrender

import ggtts.Ggtts
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

fun main(args: Array<String>) {
    val engine = args.getOrElse(0) { "qwen3-tts" }
    val cacheDir = File(System.getProperty("java.io.tmpdir"), "ggtts-cache").path  // Android: File(context.filesDir, "ggtts").path

    Ggtts.load(
        engine,
        model = args.getOrNull(1),          // null: the default model (qwen3-tts only)
        codec = args.getOrNull(2),
        voice = args.getOrNull(3),          // kitten-tts-2's prepared reference voice
        cacheDir = cacheDir,
        allowDownload = true,               // ask the user before this on a phone: it is 2.33 GB
    ).use { tts ->
        val pcm = tts.render("今日天氣好好。")    // seed = Ggtts.DEFAULT_SEED unless you pass one
        writeWav(File("first_render.wav"), pcm.samples, pcm.sampleRate)
        println("${pcm.samples.size} samples at ${pcm.sampleRate} Hz, written to first_render.wav")
    }
}

/** Mono float samples as a 16-bit PCM WAV file. */
fun writeWav(file: File, samples: FloatArray, rate: Int) {
    val data = samples.size * 2
    val b = ByteBuffer.allocate(44 + data).order(ByteOrder.LITTLE_ENDIAN)
    b.put("RIFF".toByteArray()).putInt(36 + data).put("WAVEfmt ".toByteArray())
    b.putInt(16).putShort(1).putShort(1).putInt(rate).putInt(rate * 2).putShort(2).putShort(16)
    b.put("data".toByteArray()).putInt(data)
    for (x in samples) b.putShort((x.coerceIn(-1f, 1f) * 32767f).toInt().toShort())
    file.writeBytes(b.array())
}
