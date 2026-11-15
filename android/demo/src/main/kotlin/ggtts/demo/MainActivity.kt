// The demo's one screen, laid out as a reading screen: a voice card (a sheet of the voices the model files in the
// app's models directory make; Voices says how), the text, and a player docked at the bottom. Model files come in
// through the ⋮ menu's import, or adb push (android/README.md). The engine and its files stay under Advanced.
//
// A debug build can also be driven from adb, for scripted checks (a release build ignores these extras): every extra
// is optional, and `speak` presses Speak once loaded.
//   adb shell am start -n ggtts.demo/.MainActivity --es engine kitten-tts-2 --es model lm.gguf \
//       --es codec s3gen.gguf --es voice voice.json --es text "Hello." --ei threads 4 --ez speak true
//   adb shell am start -n ggtts.demo/.MainActivity --ez stop true      # presses Stop
// Model, codec and voice extras are file names in the models directory; a path is refused (ModelImport.isPlainName).
package ggtts.demo

import android.app.Activity
import android.app.Dialog
import android.content.ClipboardManager
import android.app.ActivityManager
import android.content.Intent
import android.content.pm.ApplicationInfo
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.content.res.Configuration
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.ColorDrawable
import android.graphics.drawable.GradientDrawable
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.media.MediaPlayer
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.text.Editable
import android.text.InputType
import android.text.TextWatcher
import android.view.Gravity
import android.widget.ProgressBar
import android.provider.OpenableColumns
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import android.view.View
import android.view.ViewGroup
import android.widget.PopupMenu
import android.view.Window
import android.view.WindowManager
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
        private const val REQUEST_NOTIFICATIONS = 2
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
    private lateinit var systemSpeak: Button
    private var platformTts: TextToSpeech? = null
    private var askedForNotifications = false
    private lateinit var importButton: Button
    private lateinit var status: TextView
    private lateinit var gettingStarted: TextView
    private lateinit var voiceCard: LinearLayout
    private lateinit var voiceBadge: TextView
    private lateinit var voiceName: TextView
    private lateinit var voiceDetail: TextView
    private lateinit var voiceNote: TextView
    private lateinit var estimate: TextView
    private lateinit var count: TextView
    private lateinit var about: LinearLayout
    private lateinit var adv: LinearLayout
    private lateinit var progress: ProgressBar
    /** The status the controller last gave, shown with the seconds since Speak while a read runs. */
    private var lastStatus = ""
    private var statusIsError = false
    /** When the running read started (elapsedRealtime), or 0 when none runs. */
    private var busySince = 0L
    private val tick = object : Runnable {
        override fun run() {
            if (busySince == 0L) return
            renderStatus()
            handler.postDelayed(this, 1000)
        }
    }
    private var found = Found(emptyList(), emptyList())
    /** The voice last applied to the Advanced controls, so a refresh does not undo a choice made there. */
    private var applied: Voice? = null
    private var sample: MediaPlayer? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // The screen draws its own title; Android 15 and later lay the window out edge to edge, under the bars.
        requestWindowFeature(Window.FEATURE_NO_TITLE)
        modelsDir = File(getExternalFilesDir(null) ?: filesDir, "models").apply { mkdirs() }
        // The read belongs to the process (Reader), not to this screen: it goes on when the screen is left, and a
        // screen opened later shows its progress.
        controller = Reader.controller(this)
        setContentView(layout())
        refreshFiles()
        Reader.screen = this
        applyIntent(intent)
    }

    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        Reader.trim(this, level)
    }

    override fun onNewIntent(intent: Intent) {
        // singleTop: an `am start` while the screen is open arrives here rather than as a second screen.
        super.onNewIntent(intent)
        setIntent(intent)
        applyIntent(intent)
    }

    /**
     * Speak the line the way another app would: through Android's TextToSpeech API, naming ggtts as the engine (so no
     * system setting changes). The system engine uses the model files of this engine's last finished Speak.
     */
    private fun speakThroughAndroid() {
        val engine = ENGINES[engines.checkedRadioButtonId - 100]
        val locale = SystemVoices.ALL.first { it.third == engine }.second
        val line = text.text.toString()
        val t0 = System.nanoTime()
        fun ms() = (System.nanoTime() - t0) / 1_000_000
        showStatus("Asking Android's TextToSpeech for engine $packageName...")
        platformTts?.shutdown()
        lateinit var tts: TextToSpeech
        tts = TextToSpeech(this, { status ->
            if (status != TextToSpeech.SUCCESS) {
                showError("Android's TextToSpeech could not start the ggtts engine (status $status)")
                return@TextToSpeech
            }
            val lang = tts.setLanguage(locale)
            Log.i(TAG, "platform tts: engine ${tts.defaultEngine.let { "default $it" }}, setLanguage($locale) = $lang")
            if (lang < TextToSpeech.LANG_AVAILABLE) {
                showError("ggtts has no model files for $locale yet (code $lang): press Speak with $engine first.")
                return@TextToSpeech
            }
            tts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                override fun onStart(id: String?) {
                    Log.i(TAG, "platform tts: utterance started after ${ms()} ms")
                    handler.post { showStatus("Android's TTS is speaking through ggtts (started after ${ms() / 1000} s)...") }
                }
                override fun onDone(id: String?) {
                    Log.i(TAG, "platform tts: utterance done after ${ms()} ms")
                    handler.post { showStatus("Android's TTS spoke the line through ggtts in ${ms() / 1000} s.") }
                }
                @Deprecated("Deprecated in Java")
                override fun onError(id: String?) = onError(id, TextToSpeech.ERROR)
                override fun onError(id: String?, code: Int) {
                    Log.w(TAG, "platform tts: utterance error $code after ${ms()} ms")
                    handler.post { showError("Android's TTS reported error $code from ggtts.") }
                }
            })
            tts.speak(line, TextToSpeech.QUEUE_FLUSH, null, "ggtts-check")
        }, packageName)
        platformTts = tts
    }

    override fun onDestroy() {
        platformTts?.shutdown()
        sample?.release()
        // The read and the loaded engine stay with the process (see Reader); only this screen goes.
        if (Reader.screen === this) Reader.screen = null
        worker.shutdown()
        super.onDestroy()
    }

    // ---- DemoView

    override fun setBusy(busy: Boolean) {
        // The docked player: Read aloud while idle; while a read runs, its stage, a progress bar, the seconds since
        // the press, and Stop in Read aloud's place.
        speak.visibility = if (busy) View.GONE else View.VISIBLE
        stop.visibility = if (busy) View.VISIBLE else View.GONE
        speak.isEnabled = !busy
        stop.isEnabled = busy
        progress.visibility = if (busy) View.VISIBLE else View.GONE
        if (busy) status.visibility = View.VISIBLE
        handler.removeCallbacks(tick)
        busySince = if (busy) SystemClock.elapsedRealtime() else 0L
        if (busy) handler.post(tick) else renderStatus()
        // While a read runs on this screen, keep the screen on. Off screen, Android moves the app to the phone's slow
        // cores (measured: the render threads ran on the three Cortex-A510s only), and the read takes several times
        // longer; the reading service keeps it going there, but slowly.
        if (busy) window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        else window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        importButton.isEnabled = !busy
    }

    override fun showStatus(text: String) {
        lastStatus = text
        statusIsError = false
        status.visibility = View.VISIBLE
        status.setTextColor(themeColor(android.R.attr.textColorSecondary))
        renderStatus()
    }

    private fun renderStatus() {
        if (statusIsError) return
        val secs = if (busySince == 0L) -1 else (SystemClock.elapsedRealtime() - busySince) / 1000
        status.text = if (secs < 0) lastStatus else "$lastStatus  ·  $secs s"
    }

    /**
     * A list for a Spinner whose text takes the theme's colour: the platform's item layout keeps a light theme's dark
     * text, which disappears on a dark background.
     */
    private fun themed(items: List<String>): ArrayAdapter<String> {
        val colour = themeColor(android.R.attr.textColorPrimary)
        // The closed Spinner shows simple_spinner_item: the dropdown row layout measured 0 px high there under the
        // DayNight theme.
        return object : ArrayAdapter<String>(this, android.R.layout.simple_spinner_item, items.toMutableList()) {
            override fun getView(position: Int, convertView: View?, parent: ViewGroup) =
                super.getView(position, convertView, parent).also { (it as? TextView)?.setTextColor(colour) }
            override fun getDropDownView(position: Int, convertView: View?, parent: ViewGroup) =
                super.getDropDownView(position, convertView, parent).also { (it as? TextView)?.setTextColor(colour) }
        }.apply { setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item) }
    }

    private fun themeColor(attr: Int): Int {
        val a = obtainStyledAttributes(intArrayOf(attr))
        try { return a.getColor(0, Color.GRAY) } finally { a.recycle() }
    }

    private val night get() =
        resources.configuration.uiMode and Configuration.UI_MODE_NIGHT_MASK == Configuration.UI_MODE_NIGHT_YES

    override fun showError(text: String) {
        Log.w(TAG, "shown: $text")
        statusIsError = true
        status.visibility = View.VISIBLE
        status.setTextColor(if (night) Color.rgb(0xF2, 0xB8, 0xB5) else Color.rgb(0xB0, 0x00, 0x20))
        status.text = text
    }

    // ---- the screen: a reading screen, not a form. A voice card, the text, and a player docked at the bottom
    // (docs/android/knowledge/research/tts-app-screens.md). The engine's parts stay under Advanced, in the ⋮ menu.

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()
    /** Card surfaces and outlines, by theme. */
    private val surface get() = if (night) Color.rgb(0x1B, 0x1E, 0x22) else Color.rgb(0xF1, 0xF3, 0xF4)
    private val outline get() = if (night) Color.rgb(0x2C, 0x30, 0x36) else Color.rgb(0xDA, 0xDC, 0xE0)
    private val accent get() = themeColor(android.R.attr.colorAccent)
    /** Primary text: on this theme a plain TextView otherwise takes the secondary grey. */
    private val fg get() = themeColor(android.R.attr.textColorPrimary)
    private val onAccent get() = if (Color.luminance(accent) > 0.5f) Color.BLACK else Color.WHITE

    private fun rounded(fill: Int, radiusDp: Int, stroke: Int? = null, strokeDp: Int = 1) = GradientDrawable().apply {
        setColor(fill)
        cornerRadius = dp(radiusDp).toFloat()
        stroke?.let { setStroke(dp(strokeDp), it) }
    }

    /** A small outlined pill button: Paste, Clear, a voice's sample. */
    private fun chip(label: String, onClick: () -> Unit) = Button(this, null, android.R.attr.borderlessButtonStyle).apply {
        text = label
        isAllCaps = false
        textSize = 14f
        minHeight = dp(40); minimumHeight = dp(40); minWidth = 0; minimumWidth = 0
        setPadding(dp(16), 0, dp(16), 0)
        background = rounded(Color.TRANSPARENT, 20, outline)
        setOnClickListener { onClick() }
    }

    private fun secondary(s: String, size: Float = 13f) = TextView(this).apply {
        text = s; textSize = size; setTextColor(themeColor(android.R.attr.textColorSecondary))
    }

    private fun layout(): View {
        val pad = dp(16)
        fun label(s: String) = TextView(this).apply { text = s; setPadding(0, pad / 2, 0, 0) }
        fun lp(w: Int = ViewGroup.LayoutParams.MATCH_PARENT, h: Int = ViewGroup.LayoutParams.WRAP_CONTENT, top: Int = 0) =
            LinearLayout.LayoutParams(w, h).apply { topMargin = top }

        // The header: what this is, and a ⋮ menu for everything that is not reading.
        val header = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(20), dp(12), dp(4), dp(4))
        }
        val titles = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        titles.addView(TextView(this).apply { text = "Read aloud"; textSize = 22f; setTextColor(fg); setTypeface(typeface, Typeface.BOLD) })
        titles.addView(secondary("On this phone · no internet"))
        header.addView(titles, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        val more = Button(this, null, android.R.attr.borderlessButtonStyle).apply {
            text = "⋮"; textSize = 22f; contentDescription = "More: import a voice, about, advanced"
            minWidth = dp(48); minimumWidth = dp(48)
        }
        header.addView(more, LinearLayout.LayoutParams(dp(56), dp(56)))

        val col = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(pad, 0, pad, pad) }
        gettingStarted = TextView(this).apply { textSize = 16f; setPadding(0, pad, 0, 0) }
        col.addView(gettingStarted)

        // The voice, as a card: tap it for the sheet of voices.
        voiceCard = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(14), dp(12), dp(14), dp(12))
            background = rounded(surface, 16, outline)
            isClickable = true; isFocusable = true
            contentDescription = "Change the voice"
            setOnClickListener { showVoiceSheet() }
        }
        voiceBadge = TextView(this).apply {
            gravity = Gravity.CENTER; textSize = 15f; setTextColor(onAccent); background = rounded(accent, 20)
        }
        voiceCard.addView(voiceBadge, LinearLayout.LayoutParams(dp(40), dp(40)))
        val names = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(dp(12), 0, dp(8), 0) }
        voiceName = TextView(this).apply { textSize = 16f; setTextColor(fg) }
        voiceDetail = secondary("")
        names.addView(voiceName); names.addView(voiceDetail)
        voiceCard.addView(names, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        voiceCard.addView(TextView(this).apply { text = "Change"; textSize = 14f; setTextColor(accent) })
        col.addView(voiceCard, lp(top = dp(8)))
        voiceNote = secondary("")
        col.addView(voiceNote)

        // The text: the main surface, in a card with Paste, Clear and a count.
        col.addView(secondary("What to read").apply { setPadding(dp(4), dp(20), 0, dp(8)) })
        val textCard = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL; background = rounded(surface, 16, outline)
        }
        text = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
            minLines = 6
            gravity = Gravity.TOP or Gravity.START
            hint = "Type or paste what to read aloud"
            textSize = 20f
            setLineSpacing(0f, 1.3f)
            background = null
            setPadding(pad, pad, pad, dp(8))
        }
        textCard.addView(text)
        val tools = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(12), 0, pad, dp(12))
        }
        tools.addView(chip("Paste") { pasteText() })
        tools.addView(chip("Clear") { text.setText("") }, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { leftMargin = dp(8) })
        count = secondary("", 12f).apply { gravity = Gravity.END }
        tools.addView(count, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        textCard.addView(tools)
        text.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
            override fun onTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
            override fun afterTextChanged(s: Editable?) {
                val n = s?.length ?: 0
                count.text = if (n == 0) "" else "$n characters"
            }
        })
        col.addView(textCard)

        // About and Advanced: closed, opened from the ⋮ menu.
        about = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; visibility = View.GONE }
        about.addView(TextView(this).apply { text = "About this app"; textSize = 18f; setPadding(0, dp(24), 0, 0) })
        about.addView(label(About.TAGLINE))
        About.PURPOSE.forEach { about.addView(label(it)) }
        about.addView(label("No model ships with this app. Model files are read from:\n${modelsDir.path}"))
        adv = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; visibility = View.GONE }
        adv.addView(TextView(this).apply { text = "Advanced"; textSize = 18f; setPadding(0, dp(24), 0, 0) })

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
        adv.addView(label("Engine"))
        adv.addView(engines)
        adv.addView(cost)
        model = Spinner(this); codec = Spinner(this); voice = Spinner(this)
        adv.addView(label("Model (Qwen3-TTS talker, or Kitten-TTS 2 LM)")); adv.addView(model)
        adv.addView(label("Codec (Qwen3-TTS 12 Hz codec, or Kitten-TTS 2 S3Gen)")); adv.addView(codec)
        adv.addView(label("Voice file (Kitten-TTS 2 only: the prepared voice JSON)")); adv.addView(voice)
        speaker = EditText(this).apply { hint = "the model's default"; setSingleLine() }
        adv.addView(label("Speaker (Qwen3-TTS only, for example hk_voice_343)")); adv.addView(speaker)
        threads = Spinner(this).apply {
            adapter = themed(THREADS.map { "$it threads" })
            setSelection(THREADS.indexOf(DEFAULT_THREADS))
        }
        adv.addView(label("CPU threads")); adv.addView(threads)
        importButton = Button(this).apply { text = "Import a model file..."; setOnClickListener { pickFile() } }
        adv.addView(importButton)
        systemSpeak = Button(this).apply { text = "Speak through Android's TTS (ggtts as the engine)" }
        systemSpeak.setOnClickListener { speakThroughAndroid() }
        adv.addView(systemSpeak)
        col.addView(about)
        col.addView(adv)

        val scroll = ScrollView(this).apply { addView(col) }
        more.setOnClickListener {
            PopupMenu(this, more).apply {
                menu.add(0, 1, 0, "Import a voice file…")
                menu.add(0, 2, 1, if (about.visibility == View.VISIBLE) "Hide About" else "About this app")
                menu.add(0, 3, 2, if (adv.visibility == View.VISIBLE) "Hide Advanced" else "Advanced")
                setOnMenuItemClickListener { item ->
                    when (item.itemId) {
                        1 -> pickFile()
                        2 -> toggleSection(scroll, about)
                        3 -> toggleSection(scroll, adv)
                    }
                    true
                }
                show()
            }
        }

        // The player, docked: it stays in view while the text scrolls.
        val footer = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, dp(12), pad, dp(16))
            background = GradientDrawable().apply { setColor(surface) }
        }
        estimate = secondary("", 12f)
        footer.addView(estimate)
        status = TextView(this).apply { textSize = 14f; visibility = View.GONE; setPadding(0, dp(4), 0, 0) }
        footer.addView(status)
        progress = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
            isIndeterminate = true; visibility = View.GONE
        }
        footer.addView(progress)
        speak = Button(this).apply {
            text = "▶  Read aloud"; isAllCaps = false; textSize = 18f
            setTextColor(onAccent); background = rounded(accent, 28)
        }
        speak.setOnClickListener {
            askForNotifications()
            speakNow()
        }
        stop = Button(this).apply {
            text = "■  Stop"; isAllCaps = false; textSize = 18f; visibility = View.GONE; isEnabled = false
            background = rounded(Color.TRANSPARENT, 28, outline, 2)
        }
        stop.setOnClickListener { controller.stop() }
        footer.addView(speak, lp(h = dp(56), top = dp(10)))
        footer.addView(stop, lp(h = dp(56), top = dp(10)))

        engines.check(100)
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(themeColor(android.R.attr.colorBackground))
            addView(header)
            addView(scroll, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
            addView(View(this@MainActivity).apply { setBackgroundColor(outline) }, lp(h = 1))
            addView(footer)
            // Keep the content clear of the status and navigation bars.
            @Suppress("DEPRECATION")
            setOnApplyWindowInsetsListener { v, insets ->
                v.setPadding(0, insets.systemWindowInsetTop, 0, insets.systemWindowInsetBottom)
                insets
            }
        }
    }

    private fun toggleSection(scroll: ScrollView, section: View) {
        val open = section.visibility != View.VISIBLE
        section.visibility = if (open) View.VISIBLE else View.GONE
        if (open) scroll.post { scroll.smoothScrollTo(0, section.top) }
    }

    private fun pickFile() {
        startActivityForResult(
            Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType("*/*"),
            PICK,
        )
    }

    private fun pasteText() {
        val clip = (getSystemService(CLIPBOARD_SERVICE) as ClipboardManager).primaryClip
        val pasted = clip?.takeIf { it.itemCount > 0 }?.getItemAt(0)?.coerceToText(this)?.toString() ?: return
        text.setText(pasted)
        text.setSelection(text.text.length)
    }

    private fun refreshFiles() {
        val listed = ModelImport.listed(modelsDir)
        val steps = About.gettingStarted(listed.size)
        gettingStarted.text = if (steps.isEmpty()) "" else "To start:\n" + steps.joinToString("\n")
        gettingStarted.visibility = if (steps.isEmpty()) View.GONE else View.VISIBLE
        val files = listOf(NONE) + listed
        for (s in listOf(model, codec, voice)) {
            val keep = s.selectedItem as String?
            s.adapter = themed(files)
            keep?.let { files.indexOf(it).takeIf { i -> i >= 0 }?.let(s::setSelection) }
        }
        refreshVoices(listed)
    }

    /** Lists the voices the files make; keeps the voice chosen, else selects the first complete one. */
    private fun refreshVoices(listed: List<String>) {
        val samples = modelsDir.list()?.filter { it.endsWith(Voices.SAMPLE_SUFFIX) }?.toSet() ?: emptySet()
        found = Voices.group(Voices.scan(modelsDir, listed.filter { !it.endsWith(Voices.SAMPLE_SUFFIX) }), samples)
        val keep = applied?.let { a -> found.voices.firstOrNull { it.label == a.label } }
        // Applied here, so adb extras applied after this still win.
        (keep ?: found.first)?.let { if (keep == null) applyVoice(it) else showVoice(it) } ?: showVoice(null)
        val gaps = found.incomplete.joinToString("\n") { "Not complete yet: ${it.name} needs ${it.missing}." }
        voiceNote.text = gaps
        voiceNote.visibility = if (gaps.isEmpty()) View.GONE else View.VISIBLE
        Log.i(TAG, "voices: ${found.voices.map { it.label }}, incomplete: ${found.incomplete}")
    }

    /** Sets the Advanced controls to the voice's files, which is all Speak reads. */
    private fun applyVoice(v: Voice) {
        applied = v
        val current = text.text.toString()
        if (current.isBlank() || current in EXAMPLES.values) EXAMPLES[v.language]?.let { text.setText(it) }
        engines.check(100 + ENGINES.indexOf(v.engine))
        pick(model, v.model)
        pick(codec, v.codec)
        pick(voice, v.voiceFile ?: NONE)
        speaker.setText(v.speaker ?: "")
        showVoice(v)
        Log.i(TAG, "voice: ${v.label}")
    }

    /** The voice card and the player's expected wait, for [v] (null: no complete voice). */
    private fun showVoice(v: Voice?) {
        voiceCard.isEnabled = found.voices.isNotEmpty()
        if (v == null) {
            voiceBadge.text = "?"
            voiceName.text = "No voice yet"
            voiceDetail.text = "Import a voice's model files from the ⋮ menu"
            estimate.text = ""
            return
        }
        voiceBadge.text = badge(v)
        voiceName.text = v.name
        voiceDetail.text = "${v.language} · ${Voices.ENGINE_NAMES[v.engine] ?: v.engine}"
        estimate.text = EngineCosts.BY_ENGINE[v.engine]?.let { "First sound in about ${it.firstAudioS} s on a Galaxy Z Fold5" } ?: ""
    }

    private fun badge(v: Voice) = when (v.language) { "Cantonese" -> "粵"; "English" -> "EN"; else -> v.language.take(2) }

    /** The voices as a sheet of cards from the bottom: tap one to choose it; a sample plays without a render. */
    private fun showVoiceSheet() {
        if (found.voices.isEmpty()) return
        val dialog = Dialog(this)
        val sheet = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(12), dp(16), dp(24))
            background = GradientDrawable().apply {
                setColor(surface)
                cornerRadii = floatArrayOf(dp(24).toFloat(), dp(24).toFloat(), dp(24).toFloat(), dp(24).toFloat(), 0f, 0f, 0f, 0f)
            }
        }
        sheet.addView(View(this).apply { background = rounded(outline, 2) },
            LinearLayout.LayoutParams(dp(36), dp(4)).apply { gravity = Gravity.CENTER_HORIZONTAL; bottomMargin = dp(12) })
        val head = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.BOTTOM }
        head.addView(TextView(this).apply { text = "Choose a voice"; textSize = 20f; setTextColor(fg); setTypeface(typeface, Typeface.BOLD) },
            LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        head.addView(secondary("All run offline", 12f))
        sheet.addView(head)
        val list = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        for (v in found.voices) {
            val card = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
                setPadding(dp(14), dp(12), dp(10), dp(12))
                val chosen = v.label == applied?.label
                background = rounded(themeColor(android.R.attr.colorBackground), 16, if (chosen) accent else outline, if (chosen) 2 else 1)
                isClickable = true; isFocusable = true
                setOnClickListener { applyVoice(v); dialog.dismiss() }
            }
            card.addView(TextView(this).apply {
                text = badge(v); gravity = Gravity.CENTER; textSize = 15f; setTextColor(accent); background = rounded(outline, 20)
            }, LinearLayout.LayoutParams(dp(40), dp(40)))
            val words = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(dp(12), 0, dp(8), 0) }
            words.addView(TextView(this).apply { text = v.name; textSize = 16f; setTextColor(fg) })
            words.addView(secondary("${v.language} · ${Voices.ENGINE_NAMES[v.engine] ?: v.engine}"))
            EngineCosts.BY_ENGINE[v.engine]?.let {
                words.addView(secondary("%.1f GB · first sound %s s".format(it.downloadGb, it.firstAudioS), 12f))
            }
            card.addView(words, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
            card.addView(chip(if (v.sample != null) "▶ Sample" else "No sample") { playSample(v) }.apply { isEnabled = v.sample != null })
            list.addView(card, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
                .apply { topMargin = dp(10) })
        }
        sheet.addView(ScrollView(this).apply { addView(list) }, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
        sheet.addView(chip("+  Add a voice from a file") { dialog.dismiss(); pickFile() }, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, dp(48)).apply { topMargin = dp(12) })
        dialog.requestWindowFeature(Window.FEATURE_NO_TITLE)
        dialog.setContentView(sheet)
        dialog.window?.apply {
            setBackgroundDrawable(ColorDrawable(Color.TRANSPARENT))
            setGravity(Gravity.BOTTOM)
            setLayout(ViewGroup.LayoutParams.MATCH_PARENT, (resources.displayMetrics.heightPixels * 0.7).toInt())
        }
        dialog.show()
    }

    private fun playSample(v: Voice) {
        val name = v.sample ?: return
        sample?.release()
        sample = MediaPlayer().apply {
            setDataSource(File(modelsDir, name).path)
            setOnCompletionListener { it.release(); if (sample === it) sample = null }
            prepare()
            start()
        }
    }

    private fun pick(s: Spinner, name: String?) {
        if (name == null) return
        @Suppress("UNCHECKED_CAST")
        val adapter = s.adapter as ArrayAdapter<String>
        var i = (0 until adapter.count).firstOrNull { adapter.getItem(it) == name }
        if (i == null) {
            adapter.add(name) // a name not (yet) in the directory: the binding will say
            i = adapter.count - 1
        }
        s.setSelection(i)
    }

    private fun applyIntent(intent: Intent?) {
        // The scripted extras (file names in the models directory, and Speak pressed for you) are for testing a debug
        // build from adb. Another app can send this exported activity an intent too, so a release build ignores them,
        // and a debug build takes only names of files already in the models directory (pick).
        if (applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE == 0) return
        val extras = intent?.extras ?: return
        // Only file names inside the models directory: another app can send this exported activity extras too, and a
        // path it chose would be loaded and then saved as the system engine's model. One refused name refuses the whole
        // intent, Speak included, so the error stays on screen and nothing speaks with the previous selection.
        val refused = ModelImport.refusedNames(listOf("model", "codec", "voice").map { extras.getString(it) })
        if (refused.isNotEmpty()) {
            Log.w(TAG, "adb extras refused: $refused are not file names in the models directory; nothing applied")
            showError("Refused ${refused.joinToString()}: scripted model files must be names in the models directory, not paths. Nothing was applied.")
            return
        }
        extras.getString("engine")?.let { e -> ENGINES.indexOf(e).takeIf { it >= 0 }?.let { engines.check(100 + it) } }
        pick(model, extras.getString("model"))
        pick(codec, extras.getString("codec"))
        pick(voice, extras.getString("voice"))
        extras.getString("speaker")?.let { speaker.setText(it) }
        if (extras.containsKey("threads")) {
            THREADS.indexOf(extras.getInt("threads")).takeIf { it >= 0 }?.let { threads.setSelection(it) }
        }
        if (extras.getBoolean("stop", false)) controller.stop()
        if (extras.getBoolean("speak_system", false)) speakThroughAndroid()
        extras.getString("text")?.let { text.setText(it) }
        if (extras.getBoolean("speak", false)) speakNow()
    }

    private fun path(s: Spinner): String? {
        val name = s.selectedItem as String? ?: return null
        if (name == NONE) return null
        return File(modelsDir, name).path
    }

    /**
     * Android 13 and later hide the read's notification, and its Stop, until the app may post notifications. Ask once,
     * at the first Speak pressed on the screen (not one driven by adb extras, which must not wait on a dialog). The
     * read starts either way; refusing hides only the notification.
     */
    private fun askForNotifications() {
        if (Build.VERSION.SDK_INT < 33 || askedForNotifications) return
        askedForNotifications = true
        if (checkSelfPermission(android.Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(android.Manifest.permission.POST_NOTIFICATIONS), REQUEST_NOTIFICATIONS)
        }
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
 * Plays a stream of pieces through one AudioTrack per stream (see [StreamPlayer]: the player thread owns the track, and
 * Stop never waits for it); saves a whole read as a 16-bit WAV in [dir] (one file per engine, overwritten).
 */
class DeviceAudio(private val dir: File) : AudioOut {
    private val player = StreamPlayer({ rate -> TrackSink(rate) })

    override fun begin(rate: Int) = player.begin(rate)
    override fun play(samples: FloatArray) = player.play(samples)
    override fun finish() = player.finish()
    override fun stop() = player.stop()

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

/** One stream's AudioTrack, as [StreamPlayer] drives it. */
private class TrackSink(rate: Int) : PcmSink {
    private val track = AudioTrack.Builder()
        .setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA)
            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
        .setAudioFormat(AudioFormat.Builder().setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
            .setSampleRate(rate).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
        .setTransferMode(AudioTrack.MODE_STREAM)
        .setBufferSizeInBytes(AudioTrack.getMinBufferSize(rate, AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_FLOAT).coerceAtLeast(rate))
        .build()

    override fun play() = track.play()
    // Never a blocking write: one begun just after a Stop's pause would wait for good (see StreamPlayer).
    override fun write(samples: FloatArray, offset: Int, count: Int): Int =
        track.write(samples, offset, count, AudioTrack.WRITE_NON_BLOCKING)
    override fun pause() = track.pause()
    override fun flush() = track.flush()
    override fun stop() = track.stop()
    override fun release() = track.release()
}
