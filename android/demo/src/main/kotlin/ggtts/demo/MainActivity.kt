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

import android.content.ClipboardManager
import android.content.Intent
import android.content.pm.ApplicationInfo
import android.content.pm.PackageManager
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
import android.provider.OpenableColumns
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import android.view.MenuItem
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.ArrayAdapter
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.RadioGroup
import android.widget.Spinner
import android.widget.TextView
import androidx.activity.enableEdgeToEdge
import androidx.annotation.AttrRes
import androidx.annotation.StyleRes
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.AppCompatSpinner
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.widget.NestedScrollView
import com.google.android.material.bottomsheet.BottomSheetDialog
import com.google.android.material.bottomsheet.BottomSheetDragHandleView
import com.google.android.material.button.MaterialButton
import com.google.android.material.card.MaterialCardView
import com.google.android.material.chip.Chip
import com.google.android.material.color.DynamicColors
import com.google.android.material.color.MaterialColors
import com.google.android.material.divider.MaterialDivider
import com.google.android.material.appbar.MaterialToolbar
import com.google.android.material.progressindicator.LinearProgressIndicator
import com.google.android.material.radiobutton.MaterialRadioButton
import com.google.android.material.shape.MaterialShapeDrawable
import com.google.android.material.shape.RelativeCornerSize
import com.google.android.material.shape.ShapeAppearanceModel
import android.content.res.ColorStateList
import com.google.android.material.textfield.TextInputEditText
import com.google.android.material.textfield.TextInputLayout
import ggtts.Ggtts
import com.google.android.material.R as M
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), DemoView {
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
        private const val PICK_FOLDER = 3
    }

    // Storage (Storage.kt): the picked models folder, Music/ggtts/, and every model file listed from both.
    private lateinit var folder: ModelsFolder
    private lateinit var music: MusicFolder
    private var catalog = Catalog.EMPTY
    private lateinit var readActions: LinearLayout

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
    private lateinit var speak: MaterialButton
    private lateinit var stop: MaterialButton
    private lateinit var systemSpeak: MaterialButton
    private var platformTts: TextToSpeech? = null
    private var askedForNotifications = false
    private lateinit var importButton: MaterialButton
    private lateinit var status: TextView
    private lateinit var gettingStarted: TextView
    private lateinit var voiceCard: MaterialCardView
    private lateinit var voiceBadge: TextView
    private lateinit var voiceName: TextView
    private lateinit var voiceDetail: TextView
    private lateinit var voiceNote: TextView
    private lateinit var estimate: TextView
    private lateinit var count: TextView
    private lateinit var about: LinearLayout
    private lateinit var adv: LinearLayout
    private lateinit var progress: LinearProgressIndicator
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
        // Material You: the wallpaper's colours on Android 12 and later, the theme's own before. Edge to edge on every
        // version (Android 15 and later do it anyway); the screen keeps its content clear of the bars (layout()).
        DynamicColors.applyToActivityIfAvailable(this)
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        modelsDir = File(getExternalFilesDir(null) ?: filesDir, "models").apply { mkdirs() }
        folder = ModelsFolder(this)
        music = MusicFolder(this)
        // The read belongs to the process (Reader), not to this screen: it goes on when the screen is left, and a
        // screen opened later shows its progress.
        controller = Reader.controller(this)
        setContentView(layout())
        refreshFiles()
        Reader.screen = this
        applyIntent(intent)
        syncFolder()
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
        showReadActions(busy)
    }

    override fun showStatus(text: String) {
        lastStatus = text
        statusIsError = false
        status.visibility = View.VISIBLE
        status.setTextColor(color(M.attr.colorOnSurfaceVariant))
        renderStatus()
    }

    private fun renderStatus() {
        if (statusIsError) return
        val secs = if (busySince == 0L) -1 else (SystemClock.elapsedRealtime() - busySince) / 1000
        status.text = if (secs < 0) lastStatus else "$lastStatus  ·  $secs s"
    }

    /** A Spinner's list: under the Material theme the platform's item layouts take the theme's text colours. */
    private fun choices(items: List<String>) =
        ArrayAdapter(this, android.R.layout.simple_spinner_item, items.toMutableList())
            .apply { setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item) }

    /** A colour role of the current (dynamic) theme, such as colorPrimary or colorOnSurfaceVariant. */
    private fun color(@AttrRes role: Int) = MaterialColors.getColor(this, role, javaClass.simpleName)

    override fun showError(text: String) {
        Log.w(TAG, "shown: $text")
        statusIsError = true
        status.visibility = View.VISIBLE
        status.setTextColor(color(androidx.appcompat.R.attr.colorError))
        status.text = text
    }

    // ---- the screen: a reading screen, not a form. A voice card, the text, and a player docked at the bottom
    // (docs/android/knowledge/research/tts-app-screens.md). The engine's parts stay under Advanced, in the ⋮ menu.

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

    /** Text in one of the theme's type styles (Material 3's type scale), in its colour role. */
    private fun words(s: String, @StyleRes style: Int, @AttrRes role: Int = M.attr.colorOnSurface) = TextView(this).apply {
        text = s
        setTextAppearance(style)
        setTextColor(color(role))
    }

    private fun secondary(s: String, @StyleRes style: Int = M.style.TextAppearance_Material3_BodyMedium) =
        words(s, style, M.attr.colorOnSurfaceVariant)

    /** The voice's language as a round badge: 粵, EN. */
    private fun badgeView(filled: Boolean) = TextView(this).apply {
        gravity = Gravity.CENTER
        setTextAppearance(M.style.TextAppearance_Material3_TitleMedium)
        setTextColor(color(if (filled) M.attr.colorOnPrimary else M.attr.colorOnSecondaryContainer))
        background = MaterialShapeDrawable(ShapeAppearanceModel.builder().setAllCornerSizes(RelativeCornerSize(0.5f)).build())
            .apply {
                fillColor = ColorStateList.valueOf(
                    color(if (filled) androidx.appcompat.R.attr.colorPrimary else M.attr.colorSecondaryContainer))
            }
    }

    private fun layout(): View {
        val pad = dp(16)
        fun label(s: String) = words(s, M.style.TextAppearance_Material3_BodyLarge).apply { setPadding(0, pad / 2, 0, 0) }
        fun lp(w: Int = ViewGroup.LayoutParams.MATCH_PARENT, h: Int = ViewGroup.LayoutParams.WRAP_CONTENT, top: Int = 0) =
            LinearLayout.LayoutParams(w, h).apply { topMargin = top }

        // The header: what this is, and a ⋮ menu for everything that is not reading.
        val header = MaterialToolbar(this).apply {
            title = "Read aloud"
            // What the app is for, before anything technical (About.kt; the full sentences are under About).
            subtitle = About.HEADER_SUBTITLE
            setTitleTextAppearance(context, M.style.TextAppearance_Material3_TitleLarge)
            setSubtitleTextAppearance(context, M.style.TextAppearance_Material3_BodyMedium)
            setSubtitleTextColor(color(M.attr.colorOnSurfaceVariant))
        }

        val col = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(pad, 0, pad, pad) }
        gettingStarted = words("", M.style.TextAppearance_Material3_BodyLarge).apply { setPadding(0, pad, 0, 0) }
        col.addView(gettingStarted)

        // The voice, as a card: tap it for the sheet of voices.
        voiceCard = MaterialCardView(this, null, M.attr.materialCardViewFilledStyle).apply {
            radius = dp(16).toFloat()
            isClickable = true; isFocusable = true
            contentDescription = "Change the voice"
            setOnClickListener { showVoiceSheet() }
        }
        val voiceRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(14), dp(12), dp(14), dp(12))
        }
        voiceBadge = badgeView(filled = true)
        voiceRow.addView(voiceBadge, LinearLayout.LayoutParams(dp(40), dp(40)))
        val names = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(dp(12), 0, dp(8), 0) }
        voiceName = words("", M.style.TextAppearance_Material3_TitleMedium)
        voiceDetail = secondary("")
        names.addView(voiceName); names.addView(voiceDetail)
        voiceRow.addView(names, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        voiceRow.addView(words("Change", M.style.TextAppearance_Material3_LabelLarge, androidx.appcompat.R.attr.colorPrimary))
        voiceCard.addView(voiceRow)
        col.addView(voiceCard, lp(top = dp(8)))
        voiceNote = secondary("")
        col.addView(voiceNote)

        // The text: the main surface, an outlined field with Paste, Clear and a count under it.
        val field = TextInputLayout(this, null, M.attr.textInputOutlinedStyle).apply {
            hint = "What to read"
            placeholderText = "Type or paste what to read aloud"
            val r = dp(16).toFloat()
            setBoxCornerRadii(r, r, r, r)
        }
        text = TextInputEditText(field.context).apply {
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
            minLines = 6
            gravity = Gravity.TOP or Gravity.START
            setTextAppearance(M.style.TextAppearance_Material3_BodyLarge)
            textSize = 20f
            setLineSpacing(0f, 1.3f)
        }
        field.addView(text)
        col.addView(field, lp(top = dp(16)))
        val tools = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(4), dp(4), dp(4), 0)
        }
        tools.addView(Chip(this).apply { text = "Paste"; setOnClickListener { pasteText() } })
        tools.addView(Chip(this).apply { text = "Clear"; setOnClickListener { this@MainActivity.text.setText("") } },
            LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT)
                .apply { leftMargin = dp(8) })
        count = secondary("", M.style.TextAppearance_Material3_BodySmall).apply { gravity = Gravity.END }
        tools.addView(count, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        col.addView(tools)
        text.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
            override fun onTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
            override fun afterTextChanged(s: Editable?) {
                val n = s?.length ?: 0
                count.text = if (n == 0) "" else "$n characters"
                field.helperText = if (n == 0) About.EMPTY_TEXT_HINT else null
            }
        })
        field.helperText = if (text.text.isNullOrEmpty()) About.EMPTY_TEXT_HINT else null

        // About and Advanced: closed, opened from the ⋮ menu.
        fun heading(s: String) = words(s, M.style.TextAppearance_Material3_TitleLarge).apply { setPadding(0, dp(24), 0, 0) }
        about = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; visibility = View.GONE }
        about.addView(heading("About this app"))
        about.addView(label(About.TAGLINE))
        About.PURPOSE.forEach { about.addView(label(it)) }
        about.addView(label("No model ships with this app. Model files are read from:\n${modelsDir.path}"))
        adv = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; visibility = View.GONE }
        adv.addView(heading("Advanced"))

        engines = RadioGroup(this).apply { orientation = RadioGroup.HORIZONTAL }
        ENGINES.forEachIndexed { i, name ->
            engines.addView(MaterialRadioButton(this).apply { id = 100 + i; text = name })
        }
        // What the chosen engine costs, said before anyone waits for it.
        val cost = secondary("").apply { setPadding(0, 0, 0, pad / 2) }
        engines.setOnCheckedChangeListener { _, id ->
            val engine = ENGINES[id - 100]
            text.setText(LINES[engine])
            cost.text = EngineCosts.describe(engine)
        }
        adv.addView(label("Engine"))
        adv.addView(engines)
        adv.addView(cost)
        model = AppCompatSpinner(this); codec = AppCompatSpinner(this); voice = AppCompatSpinner(this)
        adv.addView(label("Model (Qwen3-TTS talker, or Kitten-TTS 2 LM)")); adv.addView(model)
        adv.addView(label("Codec (Qwen3-TTS 12 Hz codec, or Kitten-TTS 2 S3Gen)")); adv.addView(codec)
        adv.addView(label("Voice file (Kitten-TTS 2 only: the prepared voice JSON)")); adv.addView(voice)
        val speakerField = TextInputLayout(this, null, M.attr.textInputOutlinedStyle).apply {
            hint = "Speaker (Qwen3-TTS only, for example hk_voice_343)"
            placeholderText = "the model's default"
        }
        speaker = TextInputEditText(speakerField.context).apply { setSingleLine() }
        speakerField.addView(speaker)
        adv.addView(speakerField, lp(top = pad))
        threads = AppCompatSpinner(this).apply {
            adapter = choices(THREADS.map { "$it threads" })
            setSelection(THREADS.indexOf(DEFAULT_THREADS))
        }
        adv.addView(label("CPU threads")); adv.addView(threads)
        importButton = MaterialButton(this, null, M.attr.materialButtonOutlinedStyle).apply {
            text = "Import a model file..."; setOnClickListener { pickFile() }
        }
        adv.addView(importButton, lp(top = pad / 2))
        systemSpeak = MaterialButton(this, null, M.attr.materialButtonOutlinedStyle).apply {
            text = "Speak through Android's TTS (ggtts as the engine)"
        }
        systemSpeak.setOnClickListener { speakThroughAndroid() }
        adv.addView(systemSpeak)
        col.addView(about)
        col.addView(adv)

        val scroll = NestedScrollView(this).apply { addView(col) }
        // The ⋮ menu: everything that is not reading. The two section items say whether they open or close.
        header.menu.add(0, 1, 0, "Import a voice file…")
        header.menu.add(0, 4, 1, "Choose a models folder…")
        val aboutItem = header.menu.add(0, 2, 2, "About this app")
        val advItem = header.menu.add(0, 3, 3, "Advanced")
        header.setOnMenuItemClickListener { item: MenuItem ->
            when (item.itemId) {
                1 -> pickFile()
                2 -> aboutItem.title = if (toggleSection(scroll, about)) "Hide About" else "About this app"
                3 -> advItem.title = if (toggleSection(scroll, adv)) "Hide Advanced" else "Advanced"
                4 -> chooseFolder()
            }
            true
        }

        // The player, docked: it stays in view while the text scrolls.
        val footer = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, dp(12), pad, dp(16))
            setBackgroundColor(color(M.attr.colorSurfaceContainer))
        }
        estimate = secondary("", M.style.TextAppearance_Material3_BodySmall)
        footer.addView(estimate)
        status = words("", M.style.TextAppearance_Material3_BodyMedium).apply {
            visibility = View.GONE; setPadding(0, dp(4), 0, dp(4))
        }
        footer.addView(status)
        progress = LinearProgressIndicator(this).apply {
            isIndeterminate = true; visibility = View.GONE
        }
        footer.addView(progress)
        speak = MaterialButton(this).apply {
            text = "▶  Read aloud"
            textSize = 18f
        }
        speak.setOnClickListener {
            askForNotifications()
            speakNow()
        }
        stop = MaterialButton(this, null, M.attr.materialButtonOutlinedStyle).apply {
            text = "■  Stop"; visibility = View.GONE; isEnabled = false
            textSize = 18f
        }
        stop.setOnClickListener { controller.stop() }
        footer.addView(speak, lp(h = dp(56), top = dp(10)))
        footer.addView(stop, lp(h = dp(56), top = dp(10)))
        footer.addView(readActionsRow(), lp(top = dp(8)))

        engines.check(100)
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(color(M.attr.colorSurface))
            addView(header)
            addView(scroll, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
            addView(MaterialDivider(this@MainActivity))
            addView(footer)
            // Edge to edge: keep the content clear of the status and navigation bars, and of a cutout to the side.
            ViewCompat.setOnApplyWindowInsetsListener(this) { v, insets ->
                val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout())
                v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
                insets
            }
        }
    }

    /** Opens or closes [section]; true when it is now open. */
    private fun toggleSection(scroll: NestedScrollView, section: View): Boolean {
        val open = section.visibility != View.VISIBLE
        section.visibility = if (open) View.VISIBLE else View.GONE
        if (open) scroll.post { scroll.smoothScrollTo(0, section.top) }
        return open
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
        // The app's own directory, and the picked folder's samples (its model files are copied in: syncFolder).
        catalog = Catalog.merge(Catalog.appEntries(modelsDir), folder.entries().filter { it.name.endsWith(Voices.SAMPLE_SUFFIX) })
        exportSamples()
        val listed = catalog.names
        val steps = About.gettingStarted(listed.size)
        gettingStarted.text = if (steps.isEmpty()) "" else "To start:\n" + steps.joinToString("\n")
        gettingStarted.visibility = if (steps.isEmpty()) View.GONE else View.VISIBLE
        val files = listOf(NONE) + listed
        for (s in listOf(model, codec, voice)) {
            val keep = s.selectedItem as String?
            s.adapter = choices(files)
            keep?.let { files.indexOf(it).takeIf { i -> i >= 0 }?.let(s::setSelection) }
        }
        refreshVoices(listed)
    }

    /** Lists the voices the files make; keeps the voice chosen, else selects the first complete one. */
    private fun refreshVoices(listed: List<String>) {
        val samples = listed.filter { it.endsWith(Voices.SAMPLE_SUFFIX) }.toSet()
        found = Voices.group(Voices.scan(listed.filter { !it.endsWith(Voices.SAMPLE_SUFFIX) }) { catalog.open(it, ::openLocation) }, samples)
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
        val dialog = BottomSheetDialog(this)
        val sheet = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), 0, dp(16), dp(24))
        }
        sheet.addView(BottomSheetDragHandleView(this))
        val head = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.BOTTOM }
        head.addView(words("Choose a voice", M.style.TextAppearance_Material3_TitleLarge),
            LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        head.addView(secondary("All run offline", M.style.TextAppearance_Material3_BodySmall))
        sheet.addView(head)
        val list = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        for (v in found.voices) {
            val chosen = v.label == applied?.label
            val card = MaterialCardView(this, null, M.attr.materialCardViewOutlinedStyle).apply {
                radius = dp(16).toFloat()
                if (chosen) { strokeColor = color(androidx.appcompat.R.attr.colorPrimary); strokeWidth = dp(2) }
                isClickable = true; isFocusable = true
                setOnClickListener { applyVoice(v); dialog.dismiss() }
            }
            val row = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
                setPadding(dp(14), dp(12), dp(10), dp(12))
            }
            row.addView(badgeView(filled = false).apply { text = badge(v) }, LinearLayout.LayoutParams(dp(40), dp(40)))
            val lines = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(dp(12), 0, dp(8), 0) }
            lines.addView(words(v.name, M.style.TextAppearance_Material3_TitleMedium))
            lines.addView(secondary("${v.language} · ${Voices.ENGINE_NAMES[v.engine] ?: v.engine}"))
            EngineCosts.BY_ENGINE[v.engine]?.let {
                lines.addView(secondary("%.1f GB · first sound %s s".format(it.downloadGb, it.firstAudioS),
                    M.style.TextAppearance_Material3_BodySmall))
            }
            row.addView(lines, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
            row.addView(Chip(this).apply {
                text = if (v.sample != null) "▶ Sample" else "No sample"
                isEnabled = v.sample != null
                setOnClickListener { playSample(v) }
            })
            card.addView(row)
            list.addView(card, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
                .apply { topMargin = dp(10) })
        }
        sheet.addView(list)
        sheet.addView(MaterialButton(this, null, M.attr.materialButtonOutlinedStyle).apply {
            text = "+  Add a voice from a file"
            setOnClickListener { dialog.dismiss(); pickFile() }
        }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
            .apply { topMargin = dp(12) })
        // The sheet scrolls itself (it is a NestedScrollView's child), and opens fully: it is short.
        dialog.setContentView(NestedScrollView(this).apply { addView(sheet) })
        dialog.behavior.state = com.google.android.material.bottomsheet.BottomSheetBehavior.STATE_EXPANDED
        dialog.behavior.skipCollapsed = true
        dialog.show()
    }

    private fun playSample(v: Voice) {
        val location = v.sample?.let { catalog.location(it) } ?: return
        sample?.release()
        sample = MediaPlayer().apply {
            if (Storage.isContent(location)) setDataSource(this@MainActivity, Uri.parse(location)) else setDataSource(location)
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
        // A listed name is wherever it is (the app directory, or the picked folder as a content URI); a name that is
        // not listed (an adb extra for a file not there) stays a path in the app directory, for the binding's error.
        return catalog.location(name) ?: File(modelsDir, name).path
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

    @Deprecated("The platform Activity API; the result API would be the AndroidX one")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == PICK_FOLDER) {
            data?.data?.takeIf { resultCode == RESULT_OK }?.let { onFolderPicked(it) }
            return
        }
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

    // ---- storage a person can reach (Storage.kt): a models folder picked once, and Music/ggtts/ for audio out.
    // No storage permission: the folder comes through the system picker, and MediaStore takes the app's own audio.

    private fun chooseFolder() {
        startActivityForResult(ModelsFolder.pickIntent(), PICK_FOLDER)
    }

    private fun onFolderPicked(uri: Uri) {
        runCatching { folder.choose(uri) }.onFailure {
            showError("Could not keep access to that folder: ${it.message}")
            return
        }
        Log.i(TAG, "models folder: $uri (${folder.entries().size} files)")
        refreshFiles()
        syncFolder()
    }

    private var syncing = false

    /**
     * Copy the picked folder's model files that the app directory lacks into it, checked as an import is, one at a
     * time on the worker; the voices refresh after each. The engine cannot open a picked file in place (Storage.kt).
     */
    private fun syncFolder() {
        if (syncing) return
        val todo = Storage.toCopy(folder.entries(), ModelImport.listed(modelsDir).toSet())
        if (todo.isEmpty()) return
        syncing = true
        val where = folder.describe() ?: "the models folder"
        worker.execute {
            var failed = 0
            todo.forEachIndexed { i, e ->
                handler.post { showStatus("Copying ${e.name} from $where (${i + 1} of ${todo.size})...") }
                val t0 = SystemClock.elapsedRealtime()
                val r = runCatching {
                    (contentResolver.openInputStream(Uri.parse(e.location)) ?: throw ImportException("cannot read ${e.name}"))
                        .use { ModelImport(modelsDir).importFile(e.name, e.size, it) }
                }
                val ms = SystemClock.elapsedRealtime() - t0
                if (r.isFailure) failed++
                r.fold({ Log.i(TAG, "folder copy: ${e.name} (${it.length()} bytes) in $ms ms") },
                    { Log.w(TAG, "folder copy: ${e.name} refused after $ms ms: ${it.message}") })
                handler.post {
                    refreshFiles()
                    r.exceptionOrNull()?.let { showError("Not copied from $where: ${it.message}") }
                }
            }
            val copied = todo.size - failed
            handler.post {
                syncing = false
                // A refusal stays on screen; a clean run says so.
                if (failed == 0) showStatus("Copied $copied model file(s) from $where; their voices are listed.")
            }
        }
    }

    /** Save and Share for the last finished read, under the player. */
    private fun readActionsRow(): View {
        readActions = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; visibility = View.GONE }
        readActions.addView(MaterialButton(this, null, M.attr.materialButtonOutlinedStyle).apply {
            text = "Save to Music"; setOnClickListener { saveRead(share = false) }
        })
        readActions.addView(MaterialButton(this, null, M.attr.materialButtonOutlinedStyle).apply {
            text = "Share"; setOnClickListener { saveRead(share = true) }
        }, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { leftMargin = dp(8) })
        return readActions
    }

    private fun showReadActions(busy: Boolean) {
        if (!::readActions.isInitialized) return
        readActions.visibility = if (!busy && music.available && LastRead.file?.exists() == true) View.VISIBLE else View.GONE
    }

    /** Copy the last read into Music/ggtts/ (once), then share it if [share]. */
    private fun saveRead(share: Boolean) {
        val file = LastRead.file ?: return
        val engine = LastRead.engine ?: "read"
        worker.execute {
            val result = runCatching {
                LastRead.savedAs ?: music.save(Storage.readName(engine, System.currentTimeMillis())) { out ->
                    file.inputStream().use { it.copyTo(out) }
                }.also { LastRead.savedAs = it }
            }
            handler.post {
                result.fold({ uri ->
                    Log.i(TAG, "saved read: $uri in ${Storage.MUSIC_PATH}")
                    if (share) startActivity(MusicFolder.shareIntent(uri))
                    else showStatus("Saved to ${Storage.MUSIC_PATH} (the Files app, and over USB).")
                }, { showError("Could not save the read: ${it.message}") })
            }
        }
    }

    /** Each voice's sample goes to Music/ggtts/ too, once, so a person can play or send it from there. */
    private fun exportSamples() {
        if (!music.available) return
        val samples = catalog.names.filter { it.endsWith(Voices.SAMPLE_SUFFIX) }
        if (samples.isEmpty()) return
        val snapshot = catalog
        worker.execute {
            for (name in Storage.samplesToExport(samples, music.saved())) {
                runCatching {
                    music.save(name) { out -> snapshot.open(name, ::openLocation).use { it.copyTo(out) } }
                }.fold({ Log.i(TAG, "sample saved: $name to ${Storage.MUSIC_PATH}") }, { Log.w(TAG, "sample $name not saved: $it") })
            }
        }
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
        LastRead.wrote(file, engine)
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
