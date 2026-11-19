package ggtts.demo

import android.content.Context
import android.graphics.Bitmap
import android.webkit.JavascriptInterface
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebView
import android.webkit.WebViewClient
import java.io.ByteArrayInputStream

/**
 * Hosts only the bundled feedback page with caller-supplied validated input.
 * This is not wired to a read selector and does not establish feedback eligibility.
 */
internal class OfflineFeedbackHost(context: Context, private val view: WebView,
                          private val input: PreparedFeedbackInput,
                          private val lease: FeedbackSessionRegistry.Lease) {
    private val store = FeedbackResultStore(java.io.File(context.filesDir, "feedback-results"))
    private val diagnostics = DiagnosticLog(sink = { android.util.Log.i(MainActivity.TAG, it) })
    private val saveDiagnostics = BridgeDiagnosticLimiter()

    private inner class Bridge {
        @JavascriptInterface fun save(json: String): Boolean {
            val event = diagnostics.start(DiagnosticStage.BRIDGE_SAVE)
            val ok = try { lease.runIfActive { store.save(input, json); true } ?: false }
                catch (_: Exception) { false }
            saveDiagnostics.report(event, ok)
            return ok
        }

        @JavascriptInterface fun load(): String {
            val event = diagnostics.start(DiagnosticStage.BRIDGE_RESTORE)
            val result = try { lease.runIfActive { store.load(input) ?: "" } ?: "" }
                catch (_: Exception) { "" }
            event.finish(if (result.isEmpty()) DiagnosticOutcome.UNAVAILABLE else DiagnosticOutcome.OK)
            return result
        }
    }

    fun show(context: Context) {
        val event = diagnostics.start(DiagnosticStage.HOST_SETUP)
        try {
        val template = context.assets.open("feedback-page.html").bufferedReader(Charsets.UTF_8).use { it.readText() }
        val page = feedbackHostPage(template, input)
        view.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = false
            allowFileAccess = false
            allowContentAccess = false
            allowFileAccessFromFileURLs = false
            allowUniversalAccessFromFileURLs = false
            blockNetworkLoads = true
        }
        view.webViewClient = object : WebViewClient() {
            override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest) = true
            @Suppress("DEPRECATION")
            override fun shouldOverrideUrlLoading(view: WebView, url: String) = true
            override fun shouldInterceptRequest(view: WebView, request: WebResourceRequest): WebResourceResponse? =
                if (feedbackMediaAllowed(request.url.toString())) null
                else WebResourceResponse("text/plain", "utf-8", ByteArrayInputStream(ByteArray(0)))
            override fun onPageStarted(view: WebView, url: String?, favicon: Bitmap?) {
                if (url != ORIGIN) view.stopLoading()
            }
        }
        view.addJavascriptInterface(Bridge(), "FeedbackHost")
        view.loadDataWithBaseURL(ORIGIN, page, "text/html", "UTF-8", null)
        event.finish(DiagnosticOutcome.OK)
        } catch (t: Throwable) {
            event.finish(DiagnosticOutcome.FAILURE)
            throw t
        }
    }

    fun close() {
        view.removeJavascriptInterface("FeedbackHost")
        view.stopLoading()
        try { view.destroy() } finally { lease.close() }
    }

    companion object { private const val ORIGIN = "https://offline.ggtts.invalid/" }
}

internal fun feedbackMediaAllowed(url: String): Boolean = url.startsWith("data:audio/flac;base64,")
