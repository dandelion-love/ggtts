// The licence notices the app shows: the ones the binding's AAR carries under META-INF/ggtts/ (ggtts's own licence, the
// third-party notices and the licence texts they name), read from the APK at run time, never copied by hand. Plain
// Kotlin, so the JVM tests read the same files from the AAR's resources.
package ggtts.demo

import java.io.InputStream

object Licences {
    /** Where the AAR puts its notices; they travel into the APK as Java resources. */
    const val DIR = "META-INF/ggtts/"

    /** Files in [DIR] that are a build record, not a notice. */
    private val NOT_NOTICES = setOf("BUILDINFO.json")

    /** A notice: its path under [DIR], and its text, or null when the file could not be read. */
    data class Notice(val name: String, val text: String?)

    /**
     * The notices among [entries] (the APK's, or the resources', paths), ggtts's own LICENSE first, then the rest in
     * order, each read with [open]. A file [open] cannot read is a [Notice] with no text, so the screen says which
     * one is missing and still opens.
     */
    fun notices(entries: List<String>, open: (String) -> InputStream?): List<Notice> =
        entries.asSequence()
            .filter { it.startsWith(DIR) && !it.endsWith("/") }
            .map { it.removePrefix(DIR) }
            .filter { it !in NOT_NOTICES }
            .distinct()
            .sortedWith(compareBy({ it != "LICENSE" }, { it }))
            .map { name ->
                Notice(name, runCatching { open(DIR + name)?.use { s -> s.readBytes().toString(Charsets.UTF_8) } }.getOrNull())
            }
            .toList()

    /**
     * Whether [manifest] (AndroidManifest.xml's text) asks for no internet permission: the privacy statement says the
     * app collects no data because it cannot send any, and this is what holds it to that.
     */
    fun hasNoInternetPermission(manifest: String): Boolean = "android.permission.INTERNET" !in manifest
}
