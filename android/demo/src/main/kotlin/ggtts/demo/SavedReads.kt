// The reads the app saved to Music/ggtts/, listed with title, length and date, to play, share or delete.
//
// Where the list comes from, with no permission asked:
// - This install's own saves, through MediaStore (an app sees and deletes its own media with no permission).
// - Reads an earlier install left: MediaStore no longer counts them as this app's, and listing them would need a read
//   permission (a prompt). Instead the person can pick Music/ggtts once with the system's folder picker, as the models
//   folder is picked (a picker, not a permission); the app then lists and deletes every read in it.
//
// The title and the length come from the file itself (a FLAC's TITLE tag and STREAMINFO, a WAV's header), falling
// back to MediaStore's TITLE and DURATION, so the list is right after a restart or a reinstall.
package ggtts.demo

import android.content.ContentUris
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.provider.DocumentsContract
import android.provider.MediaStore
import java.io.InputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone

/** One saved read. [owned]: saved by this install (MediaStore); otherwise found in the picked Music/ggtts folder. */
data class SavedRead(
    val uri: String,
    val name: String,
    val title: String,
    val seconds: Double?,
    val dateMillis: Long,
    val owned: Boolean,
) {
    val mime: String get() = if (name.endsWith(".flac", true)) Flac.MIME else Storage.WAV_MIME
}

object SavedReads {
    /** How much of a file's start is read for its title and length. */
    const val HEAD = 64 * 1024

    /** Whether [name] is a read the app saved (Save to Music, or rendered for later), not a voice sample. */
    fun isRead(name: String): Boolean =
        name.startsWith("ggtts-") && !name.endsWith(Voices.SAMPLE_SUFFIX) &&
            (name.endsWith(".flac", true) || name.endsWith(".wav", true))

    /** The title and length (seconds) a file's [head] gives, each null where it gives none. */
    fun fromHead(head: ByteArray): Pair<String?, Double?> = when {
        head.size >= 4 && String(head, 0, 4, Charsets.ISO_8859_1) == "fLaC" -> Flac.title(head) to Flac.seconds(head)
        head.size >= 12 && String(head, 0, 4, Charsets.ISO_8859_1) == "RIFF" && String(head, 8, 4, Charsets.ISO_8859_1) == "WAVE" ->
            null to wavSeconds(head)
        else -> null to null
    }

    /** A read's row: [fromFile] wins; MediaStore's [storeTitle] and [storeMs] fill what the file does not say. */
    fun read(uri: String, name: String, head: ByteArray?, storeTitle: String?, storeMs: Long?, dateMillis: Long, owned: Boolean): SavedRead {
        val (title, seconds) = head?.let { fromHead(it) } ?: (null to null)
        return SavedRead(
            uri, name,
            title ?: storeTitle?.takeIf { it.isNotBlank() && it != name.substringBeforeLast('.') } ?: name.substringBeforeLast('.'),
            seconds ?: storeMs?.takeIf { it > 0 }?.let { it / 1000.0 },
            dateMillis, owned,
        )
    }

    /** This install's reads and the picked folder's, newest first; a read in both is listed once, as this install's. */
    fun merge(owned: List<SavedRead>, picked: List<SavedRead>): List<SavedRead> {
        val names = owned.map { it.name }.toSet()
        return (owned + picked.filter { it.name !in names }).filter { isRead(it.name) }.sortedByDescending { it.dateMillis }
    }

    /** The row's second line: its length and when it was saved. */
    fun describe(r: SavedRead, zone: TimeZone = TimeZone.getDefault()): String {
        val length = r.seconds?.let { if (it < 60) "%.1f s".format(it) else "%d min %02d s".format((it / 60).toInt(), (it % 60).toInt()) } ?: "length unknown"
        val date = SimpleDateFormat("d MMM yyyy HH:mm", Locale.ROOT).apply { timeZone = zone }.format(Date(r.dateMillis))
        return "$length · $date"
    }

    private fun le32(b: ByteArray, at: Int) =
        (b[at].toLong() and 0xff) or ((b[at + 1].toLong() and 0xff) shl 8) or ((b[at + 2].toLong() and 0xff) shl 16) or ((b[at + 3].toLong() and 0xff) shl 24)

    /** A WAV's length from its fmt byte rate and its data chunk's size (the data need not be in [head]). */
    fun wavSeconds(head: ByteArray): Double? {
        var p = 12
        var byteRate = 0L
        while (p + 8 <= head.size) {
            val id = String(head, p, 4, Charsets.ISO_8859_1)
            val size = le32(head, p + 4)
            if (id == "fmt " && p + 16 <= head.size) byteRate = le32(head, p + 16)
            if (id == "data") return if (byteRate > 0) size.toDouble() / byteRate else null
            p += 8 + size.toInt() + (size.toInt() and 1)
        }
        return null
    }
}

/** The reads on this phone: this install's through MediaStore, and the picked Music/ggtts folder's. */
class ReadsLibrary(context: Context) {
    private val app = context.applicationContext
    private val resolver = app.contentResolver
    private val prefs = app.getSharedPreferences("storage", Context.MODE_PRIVATE)
    private val collection: Uri =
        if (Build.VERSION.SDK_INT >= 29) MediaStore.Audio.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
        else MediaStore.Audio.Media.EXTERNAL_CONTENT_URI

    /** The picked Music/ggtts folder, if one is picked and its grant is still held. */
    val folder: Uri?
        get() {
            val s = prefs.getString(KEY, null) ?: return null
            val uri = Uri.parse(s)
            return uri.takeIf { u -> resolver.persistedUriPermissions.any { it.uri == u && it.isReadPermission } }
        }

    /** Remember the folder picked for reads, with read and write (delete) access across restarts. */
    fun choose(uri: Uri) {
        resolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION)
        prefs.edit().putString(KEY, uri.toString()).apply()
    }

    fun list(): List<SavedRead> = SavedReads.merge(owned(), picked())

    private fun head(uri: Uri): ByteArray? = runCatching {
        resolver.openInputStream(uri)?.use { it.readUpTo(SavedReads.HEAD) }
    }.getOrNull()

    private fun owned(): List<SavedRead> {
        if (Build.VERSION.SDK_INT < 29) return emptyList()
        val out = mutableListOf<SavedRead>()
        runCatching {
            resolver.query(collection, arrayOf(MediaStore.MediaColumns._ID, MediaStore.MediaColumns.DISPLAY_NAME,
                MediaStore.MediaColumns.TITLE, MediaStore.MediaColumns.DURATION, MediaStore.MediaColumns.DATE_ADDED),
                "${MediaStore.MediaColumns.RELATIVE_PATH} = ?", arrayOf(Storage.MUSIC_PATH), null)?.use { c ->
                while (c.moveToNext()) {
                    val name = c.getString(1) ?: continue
                    if (!SavedReads.isRead(name)) continue
                    val uri = ContentUris.withAppendedId(collection, c.getLong(0))
                    out += SavedReads.read(uri.toString(), name, head(uri), c.getString(2),
                        if (c.isNull(3)) null else c.getLong(3), c.getLong(4) * 1000, owned = true)
                }
            }
        }
        return out
    }

    private fun picked(): List<SavedRead> {
        val t = folder ?: return emptyList()
        val children = DocumentsContract.buildChildDocumentsUriUsingTree(t, DocumentsContract.getTreeDocumentId(t))
        val out = mutableListOf<SavedRead>()
        runCatching {
            resolver.query(children, arrayOf(DocumentsContract.Document.COLUMN_DOCUMENT_ID, DocumentsContract.Document.COLUMN_DISPLAY_NAME,
                DocumentsContract.Document.COLUMN_LAST_MODIFIED), null, null, null)?.use { c ->
                while (c.moveToNext()) {
                    val name = c.getString(1) ?: continue
                    if (!SavedReads.isRead(name)) continue
                    val uri = DocumentsContract.buildDocumentUriUsingTree(t, c.getString(0))
                    out += SavedReads.read(uri.toString(), name, head(uri), null, null, if (c.isNull(2)) 0 else c.getLong(2), owned = false)
                }
            }
        }
        return out
    }

    /** Delete [r]: this install's through MediaStore, the picked folder's through the folder's grant. */
    fun delete(r: SavedRead): Result<Unit> = runCatching {
        val uri = Uri.parse(r.uri)
        val gone = if (r.owned) resolver.delete(uri, null, null) > 0 else DocumentsContract.deleteDocument(resolver, uri)
        check(gone) { "Android did not delete ${r.name}" }
    }

    companion object {
        private const val KEY = "reads.tree"

        /** The folder picker, opened at Music/ggtts where the provider allows it. */
        fun pickIntent(): Intent = Intent(Intent.ACTION_OPEN_DOCUMENT_TREE).apply {
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION or Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION)
            if (Build.VERSION.SDK_INT >= 26) {
                putExtra(DocumentsContract.EXTRA_INITIAL_URI,
                    DocumentsContract.buildDocumentUri("com.android.externalstorage.documents", "primary:" + Storage.MUSIC_PATH.trimEnd('/')))
            }
        }
    }
}

/** Up to [n] bytes from the start of the stream (InputStream.readNBytes is API 33). */
private fun InputStream.readUpTo(n: Int): ByteArray {
    val buf = ByteArray(n)
    var at = 0
    while (at < n) {
        val r = read(buf, at, n - at)
        if (r < 0) break
        at += r
    }
    return buf.copyOf(at)
}
