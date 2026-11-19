package dev.ggtts.ggtts_app

import android.app.Activity
import android.content.Intent
import android.os.Handler
import android.os.Looper
import io.flutter.plugin.common.MethodChannel
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/** Owns one system-picker request and its background import for this Activity. */
internal class ModelFilePicker(private val activity: Activity) {
    private val worker = Executors.newSingleThreadExecutor()
    private val main = Handler(Looper.getMainLooper())
    private var pending: MethodChannel.Result? = null
    private var kind: String? = null
    private var cancellation: AtomicBoolean? = null

    fun pick(kind: String?, result: MethodChannel.Result) {
        if (pending != null || kind !in listOf("model", "codec", "voice")) {
            result.error("unavailable", "A file selection is already active or invalid.", null)
            return
        }
        pending = result
        this.kind = kind
        cancellation = AtomicBoolean(false)
        try {
            val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                addCategory(Intent.CATEGORY_OPENABLE)
                type = "*/*"
            }
            @Suppress("DEPRECATION")
            activity.startActivityForResult(intent, REQUEST)
        } catch (_: Exception) {
            finishError("Could not open the system file picker.")
        }
    }

    fun result(requestCode: Int, resultCode: Int, data: Intent?): Boolean {
        if (requestCode != REQUEST) return false
        val current = pending ?: return true
        if (resultCode != Activity.RESULT_OK) {
            pending = null; kind = null; cancellation = null
            current.success(null)
            return true
        }
        val uri = data?.data
        val selectedKind = kind
        val cancel = cancellation
        if (uri == null || uri.scheme != "content" || selectedKind == null || cancel == null) {
            finishError("The selected document is unavailable.")
            return true
        }
        try {
            worker.execute {
                val outcome = runCatching {
                    val stream = activity.contentResolver.openInputStream(uri)
                        ?: error("Document could not be opened")
                    PrivateModelImport.import(File(activity.filesDir, "model-imports"), selectedKind, stream, cancel::get)
                }
                main.post {
                    if (pending !== current || cancel.get()) {
                        outcome.getOrNull()?.let { it.delete(); it.parentFile?.delete() }
                    } else {
                        pending = null; kind = null; cancellation = null
                        outcome.fold({ current.success(it.absolutePath) }, {
                            current.error("import_failed", "Could not import the selected file.", null)
                        })
                    }
                }
            }
        } catch (_: Exception) {
            finishError("Could not start the file import.")
        }
        return true
    }

    fun close() {
        cancellation?.set(true)
        pending?.error("cancelled", "File selection was interrupted.", null)
        pending = null; kind = null; cancellation = null
        worker.shutdownNow()
    }

    private fun finishError(message: String) {
        val reply = pending
        pending = null; kind = null; cancellation = null
        reply?.error("picker_failed", message, null)
    }

    companion object { private const val REQUEST = 49312 }
}
