// The demo's one screen. It lists the model files in the app's models directory (put there with "Import a model
// file", or with adb push; see android/README.md), loads the ones picked, and speaks one line with the chosen engine.
//
// A debug build can also be driven from adb, for scripted checks (a release build ignores these extras): every extra
// is optional, and `speak` presses Speak once loaded.
//   adb shell am start -n ggtts.demo/.MainActivity --es engine kitten-tts-2 --es model lm.gguf \
//       --es codec s3gen.gguf --es voice voice.json --es text "Hello." --ei threads 4 --ez speak true
//   adb shell am start -n ggtts.demo/.MainActivity --ez stop true      # presses Stop
// A file name is looked up in the models directory; an absolute path is used as it is.
package ggtts.demo

import android.app.Activity
import android.app.ActivityManager
import android.content.Intent
import android.content.pm.ApplicationInfo
import android.graphics.Color
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.OpenableColumns
import android.util.Log
import android.view.View
import android.view.Window
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import ggtts.Ggtts
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors
import java.util.concurrent.LinkedBlockingQueue

class MainActivity : Activity(), DemoView {
    companion object {
        const val TAG = "ggtts-demo"
        private const val PICK = 1
        private const val NONE = "(none)"
        val ENGINES = listOf("qwen3-tts", "kitten-tts-2")
        val LINES = mapOf("qwen3-tts" to "今日天氣好好。", "kitten-tts-2" to "Hello from ggtts, running on this phone.")
        val THREADS = listOf(2, 4, 6, 8)
        /** The binding's default. See android/README.md for what more threads buy on a phone. */
        const val DEFAULT_THREADS = 4
    }

    private val worker = Executors.newSingleThreadExecutor()
    private val handler = Handler(Looper.getMainLooper())
    private lateinit var controller: DemoController
    private lateinit var modelsDir: File

    private lateinit var engines: RadioGroup
    private lateinit var model: Spinner
    private lateinit var codec: Spinner
    private lateinit var voice: Spinner
    private lateinit var speaker: EditText
    private lateinit var text: EditText
    private lateinit var threads: Spinner
    private lateinit var speak: Button
    private lateinit var stop: Button
    private lateinit var importButton: Button
    private lateinit var status: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // The screen draws its own title; Android 15 and later lay the window out edge to edge, under the bars.
        requestWindowFeature(Window.FEATURE_NO_TITLE)
        modelsDir = File(getExternalFilesDir(null) ?: filesDir, "models").apply { mkdirs() }
        controller = DemoController(
            binding = GgttsBinding(File(filesDir, "ggtts-cache").path),
            view = this,
            audio = DeviceAudio(File(getExternalFilesDir(null) ?: filesDir, "out")),
            background = { worker.execute(it) },
            main = { handler.post(it) },
            log = { Log.i(TAG, it) },
            memory = {
                val info = ActivityManager.MemoryInfo()
                (getSystemService(ACTIVITY_SERVICE) as ActivityManager).getMemoryInfo(info)
                info.availMem to info.lowMemory
            },
        )
        setContentView(layout())
        refreshFiles()
        applyIntent(intent)
    }

    override fun onNewIntent(intent: Intent) {
        // singleTop: an `am start` while the screen is open arrives here rather than as a second screen.
        super.onNewIntent(intent)
        setIntent(intent)
        applyIntent(intent)
    }

    override fun onDestroy() {
        worker.execute { controller.close() }
        worker.shutdown()
        super.onDestroy()
    }

    // ---- DemoView

    override fun setBusy(busy: Boolean) {
        speak.isEnabled = !busy
        stop.isEnabled = busy
        importButton.isEnabled = !busy
    }

    override fun showStatus(text: String) {
        status.setTextColor(Color.DKGRAY)
        status.text = text
    }

    override fun showError(text: String) {
        Log.w(TAG, "shown: $text")
        status.setTextColor(Color.rgb(0xB0, 0x00, 0x20))
        status.text = text
    }

    // ---- the screen

    private fun layout(): View {
        val pad = (16 * resources.displayMetrics.density).toInt()
        val col = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad, pad, pad)
        }
        fun label(s: String) = TextView(this).apply { text = s; setPadding(0, pad / 2, 0, 0) }
        col.addView(TextView(this).apply { text = "ggtts demo"; textSize = 22f })
        col.addView(label("No model ships with this app. Model files are read from:\n${modelsDir.path}"))

        engines = RadioGroup(this).apply { orientation = RadioGroup.HORIZONTAL }
        ENGINES.forEachIndexed { i, name ->
            engines.addView(RadioButton(this).apply { id = 100 + i; text = name })
        }
        // What the chosen engine costs, said before anyone waits for it.
        val cost = TextView(this).apply { setPadding(0, 0, 0, pad / 2) }
        engines.setOnCheckedChangeListener { _, id ->
            val engine = ENGINES[id - 100]
            text.setText(LINES[engine])
            cost.text = EngineCosts.describe(engine)
        }
        col.addView(label("Engine"))
        col.addView(engines)
        col.addView(cost)

        model = Spinner(this); codec = Spinner(this); voice = Spinner(this)
        col.addView(label("Model (Qwen3-TTS talker, or Kitten-TTS 2 LM)")); col.addView(model)
        col.addView(label("Codec (Qwen3-TTS 12 Hz codec, or Kitten-TTS 2 S3Gen)")); col.addView(codec)
        col.addView(label("Voice (Kitten-TTS 2 only: the prepared voice JSON)")); col.addView(voice)
        speaker = EditText(this).apply { hint = "the model's default"; setSingleLine() }
        col.addView(label("Speaker (Qwen3-TTS only, for example hk_voice_343)")); col.addView(speaker)

        importButton = Button(this).apply { text = "Import a model file..." }
        importButton.setOnClickListener {
            startActivityForResult(
                Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType("*/*"),
                PICK,
            )
        }
        col.addView(importButton)

        text = EditText(this)
        col.addView(label("Line to speak")); col.addView(text)
        threads = Spinner(this).apply {
            adapter = ArrayAdapter(this@MainActivity, android.R.layout.simple_spinner_dropdown_item, THREADS.map { "$it threads" })
            setSelection(THREADS.indexOf(DEFAULT_THREADS))
        }
        col.addView(label("CPU threads")); col.addView(threads)
        val buttons = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        speak = Button(this).apply { text = "Speak" }
        speak.setOnClickListener { speakNow() }
        stop = Button(this).apply { text = "Stop"; isEnabled = false }
        stop.setOnClickListener { controller.stop() }
        buttons.addView(speak, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        buttons.addView(stop, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        col.addView(buttons)
        status = TextView(this).apply { textSize = 16f; setPadding(0, pad, 0, 0) }
        col.addView(status)
        engines.check(100)
        return ScrollView(this).apply {
            addView(col)
            // Keep the content clear of the status and navigation bars.
            @Suppress("DEPRECATION")
            setOnApplyWindowInsetsListener { v, insets ->
                v.setPadding(0, insets.systemWindowInsetTop, 0, insets.systemWindowInsetBottom)
                insets
            }
        }
    }

    private fun refreshFiles() {
        val files = listOf(NONE) + ModelImport.listed(modelsDir)
        for (s in listOf(model, codec, voice)) {
            val keep = s.selectedItem as String?
            s.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, files)
            keep?.let { files.indexOf(it).takeIf { i -> i >= 0 }?.let(s::setSelection) }
        }
    }

    private fun pick(s: Spinner, name: String?) {
        if (name == null) return
        @Suppress("UNCHECKED_CAST")
        val adapter = s.adapter as ArrayAdapter<String>
        var i = (0 until adapter.count).firstOrNull { adapter.getItem(it) == name }
        if (i == null) {
            adapter.add(name) // an absolute path, or a name not (yet) in the directory: the binding will say
            i = adapter.count - 1
        }
        s.setSelection(i)
    }

    private fun applyIntent(intent: Intent?) {
        // The scripted extras (any path, and Speak pressed for you) are for testing a debug build from adb. Another
        // app can send this exported activity an intent too, so a release build ignores them.
        if (applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE == 0) return
        val extras = intent?.extras ?: return
        extras.getString("engine")?.let { e -> ENGINES.indexOf(e).takeIf { it >= 0 }?.let { engines.check(100 + it) } }
        pick(model, extras.getString("model"))
        pick(codec, extras.getString("codec"))
        pick(voice, extras.getString("voice"))
        extras.getString("speaker")?.let { speaker.setText(it) }
        if (extras.containsKey("threads")) {
            THREADS.indexOf(extras.getInt("threads")).takeIf { it >= 0 }?.let { threads.setSelection(it) }
        }
        if (extras.getBoolean("stop", false)) controller.stop()
        extras.getString("text")?.let { text.setText(it) }
        if (extras.getBoolean("speak", false)) speakNow()
    }

    private fun path(s: Spinner): String? {
        val name = s.selectedItem as String? ?: return null
        if (name == NONE) return null
        return if (name.startsWith("/")) name else File(modelsDir, name).path
    }

    private fun speakNow() {
        val engine = ENGINES[engines.checkedRadioButtonId - 100]
        val request = LoadRequest(
            engine = engine,
            model = path(model),
            codec = path(codec),
            voice = if (engine == "kitten-tts-2") path(voice) else null,
            speaker = if (engine == "qwen3-tts") speaker.text.toString().trim().ifEmpty { null } else null,
            threads = THREADS[threads.selectedItemPosition],
        )
        Log.i(TAG, "speak: $request")
        controller.speak(request, text.text.toString(), Ggtts.DEFAULT_SEED)
    }

    // ---- importing a file through the system picker (copied into the models directory)

    @Deprecated("The platform Activity API; this app has no AndroidX")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        val uri = data?.data
        if (requestCode != PICK || resultCode != RESULT_OK || uri == null) return
        setBusy(true)
        showStatus("Copying...")
        worker.execute {
            val result = runCatching { copyIn(uri) }
            handler.post {
                setBusy(false)
                refreshFiles()
                result.fold({ showStatus("Imported ${it.name} (${it.length() / 1_000_000} MB)") },
                    { showError("Import failed: ${it.message}") })
            }
        }
    }

    private fun copyIn(uri: Uri): File {
        var name: String? = null
        var size: Long? = null
        contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE), null, null, null)?.use {
            if (it.moveToFirst()) {
                name = it.getString(0)
                size = if (it.isNull(1)) null else it.getLong(1)
            }
        }
        val input = contentResolver.openInputStream(uri) ?: throw ImportException("the picked file cannot be read")
        val out = input.use { ModelImport(modelsDir).importFile(name ?: "unnamed", size, it) }
        Log.i(TAG, "imported $out (${out.length()} bytes)")
        return out
    }
}

/**
 * Plays a stream of pieces through one AudioTrack, on its own thread, so rendering the next chunk never waits for
 * playback; saves a whole read as a 16-bit WAV in [dir] (one file per engine, overwritten).
 */
class DeviceAudio(private val dir: File) : AudioOut {
    private val queue = LinkedBlockingQueue<FloatArray>()
    private val end = FloatArray(0)
    @Volatile private var track: AudioTrack? = null
    @Volatile private var player: Thread? = null

    override fun begin(rate: Int) {
        stop()
        val t = AudioTrack.Builder()
            .setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
            .setAudioFormat(AudioFormat.Builder().setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                .setSampleRate(rate).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
            .setTransferMode(AudioTrack.MODE_STREAM)
            .setBufferSizeInBytes(AudioTrack.getMinBufferSize(rate, AudioFormat.CHANNEL_OUT_MONO,
                AudioFormat.ENCODING_PCM_FLOAT).coerceAtLeast(rate))
            .build()
        track = t
        t.play()
        player = Thread({
            try {
                while (true) {
                    val piece = queue.take()
                    if (piece === end) break
                    t.write(piece, 0, piece.size, AudioTrack.WRITE_BLOCKING)
                }
                Thread.sleep(250) // let the last buffer drain
            } catch (_: InterruptedException) {
            }
        }, "ggtts-player").apply { start() }
    }

    override fun play(samples: FloatArray) {
        if (samples.isNotEmpty()) queue.put(samples)
    }

    override fun finish() {
        val p = player ?: return
        queue.put(end)
        p.join()
        release()
    }

    override fun stop() {
        queue.clear()
        player?.interrupt()
        track?.let { runCatching { it.pause(); it.flush() } }
        player?.join(1000)
        release()
    }

    private fun release() {
        player = null
        track?.let { runCatching { it.stop() }; it.release() }
        track = null
        queue.clear()
    }

    override fun save(samples: FloatArray, rate: Int, engine: String): String {
        dir.mkdirs()
        val file = File(dir, "$engine.wav")
        val data = samples.size * 2
        val b = ByteBuffer.allocate(44 + data).order(ByteOrder.LITTLE_ENDIAN)
        b.put("RIFF".toByteArray()).putInt(36 + data).put("WAVEfmt ".toByteArray())
        b.putInt(16).putShort(1).putShort(1).putInt(rate).putInt(rate * 2).putShort(2).putShort(16)
        b.put("data".toByteArray()).putInt(data)
        for (x in samples) b.putShort((x.coerceIn(-1f, 1f) * 32767f).toInt().toShort())
        file.writeBytes(b.array())
        return file.path
    }
}
