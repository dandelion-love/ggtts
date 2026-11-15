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
        val list = LinearLayout(activity).apply { orientation = LinearLayout.VERTICAL }
        for (r in reads) list.addView(row(r, d))
        col.addView(ScrollView(activity).apply { addView(list) }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        d.setContentView(col)
        d.setOnDismissListener { player?.release(); player = null }
        d.show()
        dialog = d
    }

    private fun row(r: SavedRead, d: BottomSheetDialog) = LinearLayout(activity).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(0, dp(12), 0, dp(4))
        addView(TextView(activity).apply { text = r.title; setTextAppearance(M.style.TextAppearance_Material3_TitleMedium) })
        addView(TextView(activity).apply {
            text = SavedReads.describe(r) + if (r.owned) "" else " · from the chosen folder"
            setTextAppearance(M.style.TextAppearance_Material3_BodySmall)
        })
        val buttons = LinearLayout(activity).apply { orientation = LinearLayout.HORIZONTAL }
        fun button(label: String, onClick: () -> Unit) = MaterialButton(activity, null, M.attr.materialButtonOutlinedStyle).apply {
            text = label; setOnClickListener { onClick() }
        }
        buttons.addView(button("▶ Play") { play(r) })
        buttons.addView(button("Share") { activity.startActivity(MusicFolder.shareIntent(Uri.parse(r.uri), r.mime)) },
            LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { leftMargin = dp(8) })
        buttons.addView(button("Delete") {
            library.delete(r).fold(
                { Log.i(MainActivity.TAG, "saved reads: deleted ${r.name}"); show() },
                { Log.w(MainActivity.TAG, "saved reads: ${r.name} not deleted: $it"); (activity as? DemoView)?.showError("Not deleted: ${it.message}") },
            )
        }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { leftMargin = dp(8) })
        addView(buttons)
    }

    private fun play(r: SavedRead) {
        player?.release()
        player = MediaPlayer().apply {
            setDataSource(activity, Uri.parse(r.uri))
            setOnCompletionListener { it.release(); if (player === it) player = null }
            prepare()
            start()
        }
        Log.i(MainActivity.TAG, "saved reads: playing ${r.name}")
    }
}
