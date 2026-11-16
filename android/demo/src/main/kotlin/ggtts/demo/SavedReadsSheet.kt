// The saved-reads sheet: each read with its title, length and date, and Play, Share and Delete. Reads from an earlier
// install appear once Music/ggtts is picked with the system's folder picker (ReadsLibrary); the sheet offers that when
// none is picked.
package ggtts.demo

import android.app.Activity
import android.media.MediaPlayer
import android.net.Uri
import android.util.Log
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.webkit.WebView
import android.app.Dialog
import java.io.File
import java.lang.ref.WeakReference
import android.os.Handler
import android.os.Looper
import com.google.android.material.bottomsheet.BottomSheetDialog
import com.google.android.material.button.MaterialButton
import com.google.android.material.R as M

class SavedReadsSheet(
    private val activity: Activity,
    private val library: ReadsLibrary,
    /** Opens the folder picker for Music/ggtts; the sheet is shown again when a folder comes back. */
    private val pickFolder: () -> Unit,
) {
    private var player: MediaPlayer? = null
    private var dialog: BottomSheetDialog? = null

    private fun dp(v: Int) = (v * activity.resources.displayMetrics.density).toInt()

    fun show() {
        dialog?.dismiss()
        val reads = library.list()
        Log.i(MainActivity.TAG, "saved reads: ${reads.size} (${reads.count { it.owned }} this install's)")
        val d = BottomSheetDialog(activity)
        val handler = Handler(Looper.getMainLooper())
        val tasks = FeedbackTaskSession(post = { handler.post(it) })
        val col = LinearLayout(activity).apply { orientation = LinearLayout.VERTICAL; setPadding(dp(20), dp(8), dp(20), dp(24)) }
        col.addView(TextView(activity).apply {
            text = "Saved reads"; setTextAppearance(M.style.TextAppearance_Material3_TitleLarge)
        })
        col.addView(TextView(activity).apply {
            text = if (reads.isEmpty()) "No saved reads yet. Save a read to Music, or render one for later; they go to ${Storage.MUSIC_PATH}."
            else "In ${Storage.MUSIC_PATH}, newest first."
            setTextAppearance(M.style.TextAppearance_Material3_BodyMedium); setPadding(0, dp(4), 0, dp(8))
        })
        if (library.folder == null) {
            // Reads an earlier install made are not this app's to MediaStore any more; the folder picker reaches them.
            col.addView(TextView(activity).apply {
                text = "Reads saved before the app was reinstalled are listed once you choose the ${Storage.MUSIC_PATH} folder."
                setTextAppearance(M.style.TextAppearance_Material3_BodySmall)
            })
            col.addView(MaterialButton(activity, null, M.attr.materialButtonOutlinedStyle).apply {
                text = "Choose the ${Storage.MUSIC_PATH.trimEnd('/')} folder…"
                setOnClickListener { d.dismiss(); pickFolder() }
            })
        }
        val privateDir = File(activity.getExternalFilesDir(null) ?: activity.filesDir, "out")
        val privateReads = privateDir.listFiles()?.filter { it.isFile && it.extension.equals("wav", true) }
            ?.sortedByDescending { it.lastModified() } ?: emptyList()
        val checks = mutableListOf<Triple<File, TextView, MaterialButton>>()
        if (privateReads.isNotEmpty()) {
            col.addView(TextView(activity).apply {
                text = activity.getString(R.string.feedback_private_reads)
                setTextAppearance(M.style.TextAppearance_Material3_TitleMedium)
            })
            for (audio in privateReads) {
                val line = LinearLayout(activity).apply { orientation = LinearLayout.VERTICAL }
                val status = TextView(activity).apply {
                    text = activity.getString(R.string.feedback_checking)
                    setTextAppearance(M.style.TextAppearance_Material3_BodySmall)
                }
                line.addView(TextView(activity).apply { text = audio.name })
                line.addView(status)
                val button = MaterialButton(activity, null, M.attr.materialButtonOutlinedStyle).apply {
                    text = activity.getString(R.string.feedback_open)
                    isEnabled = false
                }
                val weakDialog = WeakReference(d)
                val weakButton = WeakReference(button)
                val gate = FeedbackOpenGate { enabled ->
                    if (weakDialog.get()?.isShowing == true) weakButton.get()?.isEnabled = enabled
                }
                button.setOnClickListener {
                    if (gate.begin()) openFeedback(audio, d, tasks, gate)
                }
                line.addView(button)
                val export = MaterialButton(activity, null, M.attr.materialButtonOutlinedStyle).apply {
                    text = activity.getString(R.string.feedback_export)
                    setOnClickListener {
                        val folder = library.folder
                        if (folder == null) {
                            d.dismiss()
                            pickFolder()
                        } else exportFeedback(audio, folder, d, tasks)
                    }
                }
                line.addView(export)
                col.addView(line)
                checks.add(Triple(audio, status, button))
            }
        }
        val list = LinearLayout(activity).apply { orientation = LinearLayout.VERTICAL }
        for (r in reads) list.addView(row(r, d, tasks))
        col.addView(ScrollView(activity).apply { addView(list) }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        d.setContentView(col)
        d.setOnDismissListener { tasks.close(); player?.release(); player = null }
        d.show()
        for ((audio, status, button) in checks) {
            val weakDialog = WeakReference(d)
            val weakStatus = WeakReference(status)
            val weakButton = WeakReference(button)
            val weakActivity = WeakReference(activity)
            tasks.submit({ FeedbackReadSelector().eligible(audio) }) { eligible ->
                val owner = weakActivity.get()
                val sheet = weakDialog.get()
                if (owner != null && !owner.isFinishing && !owner.isDestroyed && sheet?.isShowing == true) {
                    weakStatus.get()?.text = owner.getString(if (eligible) R.string.feedback_available else R.string.feedback_unavailable)
                    weakButton.get()?.isEnabled = eligible
                }
            }
        }
        dialog = d
    }

    private fun row(r: SavedRead, d: BottomSheetDialog, tasks: FeedbackTaskSession) = LinearLayout(activity).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(0, dp(12), 0, dp(4))
        addView(TextView(activity).apply { text = r.title; setTextAppearance(M.style.TextAppearance_Material3_TitleMedium) })
        addView(TextView(activity).apply {
            text = SavedReads.describe(r) + if (r.owned) "" else " · from the chosen folder"
            setTextAppearance(M.style.TextAppearance_Material3_BodySmall)
        })
        if (library.folder != null) {
            val exportedButton = MaterialButton(activity, null, M.attr.materialButtonOutlinedStyle).apply {
                text = activity.getString(R.string.feedback_open_export)
            }
            val weakDialog = WeakReference(d)
            val weakButton = WeakReference(exportedButton)
            val gate = FeedbackOpenGate { enabled ->
                if (weakDialog.get()?.isShowing == true) weakButton.get()?.isEnabled = enabled
            }
            exportedButton.setOnClickListener {
                if (gate.begin()) openExportedFeedback(r, d, tasks, gate)
            }
            addView(exportedButton)
        } else {
            addView(TextView(activity).apply {
                text = activity.getString(R.string.feedback_export_unavailable)
                setTextAppearance(M.style.TextAppearance_Material3_BodySmall)
            })
        }
        val buttons = LinearLayout(activity).apply { orientation = LinearLayout.HORIZONTAL }
        fun button(label: String, onClick: () -> Unit) = MaterialButton(activity, null, M.attr.materialButtonOutlinedStyle).apply {
            text = label; setOnClickListener { onClick() }
        }
        buttons.addView(button("▶ Play") { play(r) })
        buttons.addView(button("Share") { activity.startActivity(MusicFolder.shareIntent(Uri.parse(r.uri), activity.getString(R.string.share_title), r.mime)) },
            LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { leftMargin = dp(8) })
        buttons.addView(button("Delete") {
            library.delete(r).fold(
                { Log.i(MainActivity.TAG, "saved_read_delete outcome=ok"); show() },
                { Log.w(MainActivity.TAG, "saved_read_delete outcome=failure"); (activity as? DemoView)?.showError("Not deleted: ${it.message}") },
            )
        }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { leftMargin = dp(8) })
        addView(buttons)
    }

    private fun openFeedback(audio: File, sheet: BottomSheetDialog, tasks: FeedbackTaskSession, gate: FeedbackOpenGate) {
        val weakActivity = WeakReference(activity)
        val weakSheet = WeakReference(sheet)
        val diagnostics = DiagnosticLog(sink = { Log.i(MainActivity.TAG, it) })
        tasks.submit({ FeedbackReadSelector(diagnostics = diagnostics).prepare(audio) }) { input ->
            val owner = weakActivity.get()
            if (owner == null || owner.isFinishing || owner.isDestroyed || weakSheet.get()?.isShowing != true)
                return@submit
            if (input == null) {
                gate.finish()
                (owner as? DemoView)?.showError(owner.getString(R.string.feedback_unavailable))
                return@submit
            }
            val lease = FeedbackSessionRegistry.acquire(input.digest)
            if (lease == null) {
                gate.finish()
                (owner as? DemoView)?.showError(owner.getString(R.string.feedback_in_use))
                return@submit
            }
            var panel: Dialog? = null
            var host: OfflineFeedbackHost? = null
            try {
                val dialog = Dialog(owner)
                val web = WebView(owner)
                val pageHost = OfflineFeedbackHost(owner, web, input, lease)
                panel = dialog
                host = pageHost
                dialog.setContentView(web)
                dialog.setOnDismissListener {
                    try { pageHost.close() } finally { lease.close(); gate.finish() }
                }
                pageHost.show(owner)
                dialog.show()
                dialog.window?.setLayout(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)
            } catch (e: Exception) {
                try { if (panel?.isShowing == true) panel.dismiss() else host?.close() }
                finally { lease.close(); gate.finish() }
                (owner as? DemoView)?.showError(owner.getString(R.string.feedback_unavailable))
            }
        }
    }

    private fun exportFeedback(audio: File, folder: Uri, sheet: BottomSheetDialog, tasks: FeedbackTaskSession) {
        val owner = WeakReference(activity)
        val visible = WeakReference(sheet)
        val resolver = activity.contentResolver
        val resultDir = File(activity.filesDir, "feedback-results")
        tasks.submit({
            runCatching {
                FeedbackSidecarExport(FileSavedEvidenceStore(), FeedbackResultStore(resultDir),
                    DiagnosticLog(sink = { Log.i(MainActivity.TAG, it) }))
                    .export(SafFeedbackDocumentProvider(resolver, folder), audio)
            }
        }) { result ->
            val a = owner.get()
            if (a != null && !a.isFinishing && !a.isDestroyed && visible.get()?.isShowing == true) {
                result.fold(
                    { (a as? DemoView)?.showStatus(a.getString(R.string.feedback_export_done)) },
                    { (a as? DemoView)?.showError(a.getString(R.string.feedback_export_failed)) },
                )
            }
        }
    }

    private fun openExportedFeedback(read: SavedRead, sheet: BottomSheetDialog, tasks: FeedbackTaskSession, gate: FeedbackOpenGate) {
        val folder = library.folder ?: run { gate.finish(); return }
        val owner = WeakReference(activity)
        val visible = WeakReference(sheet)
        val resolver = activity.contentResolver
        val resultDir = File(activity.filesDir, "feedback-results")
        tasks.submit({
            runCatching {
                val provider = SafFeedbackDocumentProvider(resolver, folder)
                val id = provider.find(read.name) ?: error("audio missing")
                val listed = resolver.openInputStream(Uri.parse(read.uri))?.use { it.readBytes() } ?: error("audio unreadable")
                check(provider.read(id).contentEquals(listed)) { "listed audio differs" }
                val pair = FeedbackSidecarExport(FileSavedEvidenceStore(), FeedbackResultStore(resultDir),
                    DiagnosticLog(sink = { Log.i(MainActivity.TAG, it) }))
                    .reopenPage(provider, read.name) ?: error("feedback unavailable")
                pair
            }
        }) { result ->
            val a = owner.get()
            if (a == null || a.isFinishing || a.isDestroyed || visible.get()?.isShowing != true) return@submit
            val pair = result.getOrElse {
                gate.finish()
                (a as? DemoView)?.showError(a.getString(R.string.feedback_unavailable))
                return@submit
            }
            val input = pair.input
            val lease = FeedbackSessionRegistry.acquire(input.digest)
            if (lease == null) {
                gate.finish()
                (a as? DemoView)?.showError(a.getString(R.string.feedback_in_use))
                return@submit
            }
            try {
                if (!FeedbackResultStore(resultDir).restoreExport(input, pair.result)) throw FeedbackConflict()
            } catch (e: Exception) {
                lease.close()
                gate.finish()
                val message = if (e is FeedbackConflict) R.string.feedback_conflict else R.string.feedback_unavailable
                (a as? DemoView)?.showError(a.getString(message))
                return@submit
            }
            var panel: Dialog? = null
            var host: OfflineFeedbackHost? = null
            try {
                val dialog = Dialog(a)
                panel = dialog
                val web = WebView(a)
                val actual = OfflineFeedbackHost(a, web, input, lease)
                host = actual
                dialog.setContentView(web)
                dialog.setOnDismissListener {
                    try { actual.close() } finally { lease.close(); gate.finish() }
                }
                actual.show(a)
                dialog.show()
                dialog.window?.setLayout(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)
            } catch (_: Exception) {
                try { if (panel?.isShowing == true) panel.dismiss() else host?.close() }
                finally { lease.close(); gate.finish() }
                (a as? DemoView)?.showError(a.getString(R.string.feedback_unavailable))
            }
        }
    }

    private fun play(r: SavedRead) {
        player?.release()
        player = MediaPlayer().apply {
            setDataSource(activity, Uri.parse(r.uri))
            setOnCompletionListener { it.release(); if (player === it) player = null }
            prepare()
            start()
        }
        Log.i(MainActivity.TAG, "saved_read_play outcome=start")
    }
}

private class FeedbackConflict : Exception()
