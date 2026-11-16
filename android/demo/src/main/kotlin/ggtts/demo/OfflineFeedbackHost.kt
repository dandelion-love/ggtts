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
class OfflineFeedbackHost(context: Context, private val view: WebView,
                          private val input: PreparedFeedbackInput) {
    private val store = FeedbackResultStore(java.io.File(context.filesDir, "feedback-results"))

    private inner class Bridge {
        @JavascriptInterface fun save(json: String): Boolean = try {
            store.save(input, json); true
        } catch (_: Exception) { false }

        @JavascriptInterface fun load(): String = store.load(input) ?: ""
    }

    fun show(context: Context) {
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
    }

    fun close() {
        view.removeJavascriptInterface("FeedbackHost")
        view.stopLoading()
        view.destroy()
    }

    companion object { private const val ORIGIN = "https://offline.ggtts.invalid/" }
}

internal fun feedbackMediaAllowed(url: String): Boolean = url.startsWith("data:audio/flac;base64,")
