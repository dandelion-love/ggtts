package ggtts.demo

import android.content.ContentResolver
import android.net.Uri
import android.provider.DocumentsContract

/**
 * Selected SAF tree adapter. Names and document IDs are verified on every operation; providers
 * can still fail or leave orphans, so no two-document atomicity is claimed.
 */
class SafFeedbackDocumentProvider(private val resolver: ContentResolver, private val tree: Uri) : FeedbackDocumentProvider {
    private val root = DocumentsContract.buildDocumentUriUsingTree(tree, DocumentsContract.getTreeDocumentId(tree))
    private val owned = mutableSetOf<String>()

    private fun children(): List<Pair<String, String>> {
        val uri = DocumentsContract.buildChildDocumentsUriUsingTree(tree, DocumentsContract.getTreeDocumentId(tree))
        val rows = mutableListOf<Pair<String, String>>()
        val columns = arrayOf(DocumentsContract.Document.COLUMN_DOCUMENT_ID, DocumentsContract.Document.COLUMN_DISPLAY_NAME)
        val cursor = resolver.query(uri, columns, null, null, null) ?: error("cannot list selected folder")
        cursor.use {
            while (it.moveToNext()) {
                val id = it.getString(0) ?: error("document has no ID")
                val name = it.getString(1) ?: error("document has no name")
                rows.add(DocumentsContract.buildDocumentUriUsingTree(tree, id).toString() to name)
            }
        }
        return rows
    }

    override fun find(name: String): String? {
        val matches = children().filter { it.second == name }
        check(matches.size <= 1) { "duplicate document name" }
        return matches.singleOrNull()?.first
    }

    override fun createStaging(name: String, mime: String): String {
        val before = children()
        check(before.none { it.second == name }) { "staging name exists" }
        val uri = DocumentsContract.createDocument(resolver, root, mime, name) ?: error("cannot create staging document")
        val id = uri.toString()
        // Do not treat an existing document as ours, even if a provider returns it.
        check(before.none { it.first == id }) { "provider returned an existing document" }
        owned.add(id)
        try {
            check(find(name) == id) { "provider changed or duplicated staging name" }
            return id
        } catch (t: Throwable) {
            runCatching { delete(id) }
            throw t
        }
    }

    override fun write(id: String, bytes: ByteArray) {
        check(id in owned)
        (resolver.openOutputStream(Uri.parse(id), "wt") ?: error("cannot write staging document")).use { it.write(bytes) }
    }

    override fun read(id: String): ByteArray =
        (resolver.openInputStream(Uri.parse(id)) ?: error("cannot read document")).use { it.readBytes() }

    override fun publish(id: String, name: String): String {
        check(id in owned)
        val before = children()
        check(before.none { it.second == name }) { "destination exists" }
        val renamed = DocumentsContract.renameDocument(resolver, Uri.parse(id), name) ?: error("cannot publish document")
        val next = renamed.toString()
        // A changed ID already in the folder is not ours to remove.
        check(next == id || before.none { it.first == next && it.first !in owned }) { "provider returned an existing document" }
        owned.remove(id); owned.add(next)
        try {
            check(find(name) == next) { "provider changed or duplicated destination name" }
            return next
        } catch (t: Throwable) {
            runCatching { delete(next) }
            throw t
        }
    }

    override fun delete(id: String) {
        if (owned.remove(id)) DocumentsContract.deleteDocument(resolver, Uri.parse(id))
    }
}
