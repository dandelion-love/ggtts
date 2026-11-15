// Where the app's files live so a person can reach them, with no storage permission. Audio goes out to Music/ggtts/
// through MediaStore (Android 10 and later let an app add its own media there without asking), where the Files app
// and a computer over USB both see it. Model files come in from a folder the user picks once with the system picker
// (ACTION_OPEN_DOCUMENT_TREE, permission persisted) and load in place, never copied: [ContentPaths] opens each one
// with openFileDescriptor and hands the engine `/proc/self/fd/<n>`, which the engine reads through that descriptor
// rather than reopening it (ggtts_core::fdpath; reopening fails with "Permission denied" without a storage
// permission). The app's own models directory stays, and a name in both places is its.
//
// The parts apart from Android (naming, merging the two listings, what to export) are in [Storage] and [Catalog], so
// the JVM tests drive them; the Android parts ([MusicFolder], [ModelsFolder], [ContentPaths]) are thin.
package ggtts.demo

import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.provider.DocumentsContract
import android.provider.MediaStore
import java.io.File
import java.io.FileNotFoundException
import java.io.InputStream
import java.io.OutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

object Storage {
    /** Where saved reads and voice samples go, as MediaStore's RELATIVE_PATH. */
    const val MUSIC_PATH = "Music/ggtts/"
    const val WAV_MIME = "audio/wav"

    /** Whether [location] is a document in the picked folder (a content URI) rather than a file path. */
    fun isContent(location: String): Boolean = location.startsWith("content://")

    /** The path the engine reads a descriptor this process holds through (never reopened: ggtts_core::fdpath). */
    fun fdPath(fd: Int): String = "/proc/self/fd/$fd"

    /**
     * The cache file name for a picked document whose descriptor cannot seek ([ContentPaths]): the document's own
     * name, after a hash of its whole location, so two documents of one name in different places never share a copy.
     */
    fun fallbackName(location: String): String {
        val base = java.net.URLDecoder.decode(location.substringAfterLast('/'), "UTF-8").substringAfterLast('/')
            .filter { it != '\\' && it != '\u0000' }.ifEmpty { "picked" }
        return "%08x-%s".format(location.hashCode(), base)
    }

    /** A saved read's file name: the engine and when the read was saved, so saves never collide or overwrite. */
    fun readName(engine: String, timeMillis: Long): String {
        val f = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.ROOT).apply { timeZone = java.util.TimeZone.getDefault() }
        return "ggtts-$engine-${f.format(Date(timeMillis))}.wav"
    }

    /** The voice samples among [samples] not yet in Music/ggtts/ ([saved]: the names already there), in order. */
    fun samplesToExport(samples: Collection<String>, saved: Set<String>): List<String> =
        samples.filter { it.endsWith(Voices.SAMPLE_SUFFIX) && it !in saved }.sorted()
}

/**
 * One file the screen can list: its name, where it is (a path, or a document URI in the picked folder), and its size
 * where the folder's provider gives one.
 */
data class Entry(val name: String, val location: String, val size: Long? = null)

/**
 * The model files the screen lists, from the app's models directory and the picked folder, by name. A name in both
 * is the app directory's (its copy was put there on purpose); a copy in progress (`.part`) and a name that is not a
 * plain file name are never listed.
 */
class Catalog private constructor(private val byName: Map<String, String>) {
    /** Every name, sorted. */
    val names: List<String> = byName.keys.sorted()

    /** Where [name] is, or null when it is not listed. */
    fun location(name: String): String? = byName[name]

    fun isEmpty() = byName.isEmpty()

    /** Opens [name] for reading: [reader] opens its location (a path or a content URI). */
    fun open(name: String, reader: (String) -> InputStream): InputStream =
        reader(location(name) ?: throw FileNotFoundException(name))

    companion object {
        val EMPTY = Catalog(emptyMap())

        fun merge(app: List<Entry>, folder: List<Entry>): Catalog {
            val m = LinkedHashMap<String, String>()
            for (e in app + folder) {
                if (!ModelImport.isPlainName(e.name) || e.name.endsWith(ModelImport.PART)) continue
                m.putIfAbsent(e.name, e.location)
            }
            return Catalog(m)
        }

        /** The app directory's complete files, as entries. */
        fun appEntries(dir: File): List<Entry> = ModelImport.listed(dir).map { Entry(it, File(dir, it).path) }
    }
}

/** A path the engine can open, and what keeps it open (closed when the engine is). */
class Opened(val path: String, private val handle: AutoCloseable? = null) : AutoCloseable {
    override fun close() {
        handle?.close()
    }
}

/** Turns a model location into a path the engine opens. */
fun interface PathResolver {
    fun open(location: String): Opened

    companion object {
        /** Paths only: what the JVM tests and a binding with no picked folder use. */
        val PLAIN = PathResolver { Opened(it) }
    }
}

/**
 * A picked-folder document as `/proc/self/fd/<n>`: the descriptor stays open as long as the engine that loaded it,
 * and the engine reads through it, never reopening the path. A plain path passes through.
 *
 * The fallback: a provider may hand back a descriptor that cannot seek (a pipe, for a document it streams from
 * elsewhere), and a model is read at offsets. Such a document is copied once into the app's cache
 * ([Storage.fallbackName]) and that copy is loaded instead. A local folder such as Download/ggtts gives a seekable
 * descriptor, so it is never copied.
 */
class ContentPaths(context: Context) : PathResolver {
    private val app = context.applicationContext
    private val resolver = app.contentResolver

    override fun open(location: String): Opened {
        if (!Storage.isContent(location)) return Opened(location)
        val uri = Uri.parse(location)
        val pfd = resolver.openFileDescriptor(uri, "r") ?: throw FileNotFoundException("$location could not be opened")
        if (seekable(pfd)) return Opened(Storage.fdPath(pfd.fd), pfd)
        pfd.close()
        val dir = File(app.cacheDir, "picked").apply { mkdirs() }
        val copy = File(dir, Storage.fallbackName(location))
        if (!copy.isFile) {
            val part = File(dir, copy.name + ModelImport.PART)
            try {
                (resolver.openInputStream(uri) ?: throw FileNotFoundException("$location could not be read"))
                    .use { input -> part.outputStream().use { input.copyTo(it, 1 shl 20) } }
                if (!part.renameTo(copy)) throw FileNotFoundException("could not move ${copy.name} into place")
            } finally {
                part.delete()
            }
        }
        android.util.Log.i(MainActivity.TAG, "picked file does not seek; loading a copy: $copy")
        return Opened(copy.path)
    }

    private fun seekable(pfd: android.os.ParcelFileDescriptor): Boolean =
        runCatching { android.system.Os.lseek(pfd.fileDescriptor, 0, android.system.OsConstants.SEEK_CUR) }.isSuccess
}

/** Opens a location (a path or a content URI) for reading: what [Voices.scan] and the samples read through. */
fun Context.openLocation(location: String): InputStream =
    if (Storage.isContent(location)) {
        contentResolver.openInputStream(Uri.parse(location)) ?: throw FileNotFoundException(location)
    } else {
        File(location).inputStream()
    }

/** The folder the user picked once for model files, remembered with its permission across restarts. */
class ModelsFolder(private val context: Context) {
    private val prefs = context.getSharedPreferences("storage", Context.MODE_PRIVATE)

    /** The picked folder's tree URI, or null when none is picked (or its permission was taken back). */
    val tree: Uri?
        get() {
            val s = prefs.getString(KEY, null) ?: return null
            val uri = Uri.parse(s)
            val held = context.contentResolver.persistedUriPermissions.any { it.uri == uri && it.isReadPermission }
            return if (held) uri else null
        }

    /** Remember [uri], picked with ACTION_OPEN_DOCUMENT_TREE, and keep read access to it across restarts. */
    fun choose(uri: Uri) {
        context.contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
        // One folder at a time: give back the one chosen before, so its grant does not linger.
        prefs.getString(KEY, null)?.let { old ->
            if (old != uri.toString()) {
                runCatching { context.contentResolver.releasePersistableUriPermission(Uri.parse(old), Intent.FLAG_GRANT_READ_URI_PERMISSION) }
            }
        }
        prefs.edit().putString(KEY, uri.toString()).apply()
    }

    /** The folder's files (not its subfolders), as entries whose location is each file's document URI. */
    fun entries(): List<Entry> {
        val t = tree ?: return emptyList()
        val children = DocumentsContract.buildChildDocumentsUriUsingTree(t, DocumentsContract.getTreeDocumentId(t))
        val cols = arrayOf(DocumentsContract.Document.COLUMN_DOCUMENT_ID, DocumentsContract.Document.COLUMN_DISPLAY_NAME,
            DocumentsContract.Document.COLUMN_MIME_TYPE, DocumentsContract.Document.COLUMN_SIZE)
        val out = mutableListOf<Entry>()
        runCatching {
            context.contentResolver.query(children, cols, null, null, null)?.use { c ->
                while (c.moveToNext()) {
                    if (c.getString(2) == DocumentsContract.Document.MIME_TYPE_DIR) continue
                    val name = c.getString(1) ?: continue
                    val size = if (c.isNull(3)) null else c.getLong(3)
                    out += Entry(name, DocumentsContract.buildDocumentUriUsingTree(t, c.getString(0)).toString(), size)
                }
            }
        }
        return out
    }

    /** A short name for the folder, for the screen: its path inside the storage it is on, where the provider says. */
    fun describe(): String? = tree?.let { DocumentsContract.getTreeDocumentId(it).substringAfter(':') }

    companion object {
        private const val KEY = "models.tree"

        fun pickIntent(): Intent = Intent(Intent.ACTION_OPEN_DOCUMENT_TREE)
            .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION)
    }
}

/**
 * Music/ggtts/ through MediaStore: the app adds its own audio there with no permission on Android 10 and later, and
 * finds what it added (its own items need no read permission either). Below Android 10 that would need a storage
 * permission, which this app never asks for, so [available] is false there.
 */
class MusicFolder(context: Context) {
    private val resolver = context.applicationContext.contentResolver
    private val collection: Uri =
        if (Build.VERSION.SDK_INT >= 29) MediaStore.Audio.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
        else MediaStore.Audio.Media.EXTERNAL_CONTENT_URI

    val available: Boolean get() = Build.VERSION.SDK_INT >= 29

    /** Adds [name] to Music/ggtts/ with the bytes [write] gives; returns its content URI (shareable as it is). */
    fun save(name: String, write: (OutputStream) -> Unit): Uri {
        check(available) { "saving to Music needs Android 10 or later" }
        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, Storage.WAV_MIME)
            put(MediaStore.MediaColumns.RELATIVE_PATH, Storage.MUSIC_PATH)
            put(MediaStore.MediaColumns.IS_PENDING, 1)
        }
        val uri = resolver.insert(collection, values) ?: throw FileNotFoundException("MediaStore refused $name")
        try {
            (resolver.openOutputStream(uri) ?: throw FileNotFoundException("cannot write $name")).use(write)
            resolver.update(uri, ContentValues().apply { put(MediaStore.MediaColumns.IS_PENDING, 0) }, null, null)
            return uri
        } catch (e: Exception) {
            resolver.delete(uri, null, null)
            throw e
        }
    }

    /** The names this app has already put in Music/ggtts/. */
    fun saved(): Set<String> {
        if (!available) return emptySet()
        val out = mutableSetOf<String>()
        runCatching {
            resolver.query(collection, arrayOf(MediaStore.MediaColumns.DISPLAY_NAME),
                "${MediaStore.MediaColumns.RELATIVE_PATH} = ?", arrayOf(Storage.MUSIC_PATH), null)?.use { c ->
                while (c.moveToNext()) c.getString(0)?.let { out += it }
            }
        }
        return out
    }

    companion object {
        /** A chooser that shares [uri] (audio) with the app the user picks, which may read it and nothing else. */
        fun shareIntent(uri: Uri): Intent = Intent.createChooser(
            Intent(Intent.ACTION_SEND).setType(Storage.WAV_MIME).putExtra(Intent.EXTRA_STREAM, uri)
                .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION),
            "Share the read",
        )
    }
}

/** The last finished read the app wrote (DeviceAudio.save), for the screen's Save and Share. */
object LastRead {
    @Volatile var file: File? = null
    @Volatile var engine: String? = null
    /** Where Save put it in Music/ggtts/, so Share does not add a second copy. */
    @Volatile var savedAs: Uri? = null

    fun wrote(file: File, engine: String) {
        this.file = file
        this.engine = engine
        savedAs = null
    }
}
