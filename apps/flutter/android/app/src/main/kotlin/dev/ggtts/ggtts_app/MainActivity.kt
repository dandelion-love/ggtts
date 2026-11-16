package dev.ggtts.ggtts_app

import android.app.Dialog
import android.content.Intent
import android.graphics.Bitmap
import android.view.ViewGroup
import android.webkit.JavascriptInterface
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebView
import android.webkit.WebViewClient
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import java.io.ByteArrayInputStream
import java.io.File
import java.security.MessageDigest

class MainActivity : FlutterActivity() {
    private val modelPicker by lazy { ModelFilePicker(this) }
    private var active = false
    private var closeActive: (() -> Unit)? = null

    override fun onStop() {
        AndroidPlaybackBridge.interruptActive()
        super.onStop()
    }

    override fun onDestroy() {
        modelPicker.close()
        closeActive?.invoke()
        super.onDestroy()
    }

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, "ggtts/model_picker")
            .setMethodCallHandler { call, result ->
                if (call.method != "pick") result.notImplemented()
                else modelPicker.pick(call.argument<String>("kind"), result)
            }
        AndroidPlaybackBridge.configure(applicationContext)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, "ggtts/shared_feedback")
            .setMethodCallHandler { call, result ->
                if (call.method != "open") { result.notImplemented(); return@setMethodCallHandler }
                val input = call.argument<String>("input")
                val digest = call.argument<String>("digest")
                if (active || input == null || digest == null || input.length > 16000000 ||
                    !digest.matches(Regex("[0-9a-f]{64}")) ||
                    input.any { it == '<' || it == '>' || it == '&' } ||
                    sha(input.toByteArray(Charsets.UTF_8)) != digest) {
                    result.error("invalid", "Feedback input unavailable", null)
                    return@setMethodCallHandler
                }
                try {
                    // The template is read from the APK's Flutter asset bundle, never from the channel.
                    val assetKey = io.flutter.FlutterInjector.instance().flutterLoader()
                        .getLookupKeyForAsset("assets/feedback-page.html")
                    val template = assets.open(assetKey).bufferedReader(Charsets.UTF_8).use { it.readText() }
                    val page = FeedbackPage.adapt(template, input, digest)
                    showPage(page, digest, result)
                } catch (_: Exception) {
                    result.error("host", "Feedback page unavailable", null)
                }
            }
    }

    @Deprecated("Uses FlutterActivity's activity-result callback")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        modelPicker.result(requestCode, resultCode, data)
    }

    private fun showPage(page: String, digest: String, result: MethodChannel.Result) {
        active = true
        val draft = FeedbackDraft(File(filesDir, "feedback-draft-" + digest + ".json"))
        val dialog = Dialog(this, android.R.style.Theme_Material_Light_NoActionBar_Fullscreen)
        val view = WebView(this)
        var returned = false
        fun finish(value: String?) {
            if (returned) return
            returned = true
            active = false
            closeActive = null
            result.success(value)
            view.removeJavascriptInterface("FeedbackHost")
            view.stopLoading()
            dialog.dismiss()
            view.destroy()
        }
        closeActive = { finish(null) }
        val bridge = object {
            @JavascriptInterface fun save(json: String): Boolean = draft.save(json)
            @JavascriptInterface fun load(): String = draft.load()
            @JavascriptInterface fun submit(json: String) {
                if (json.toByteArray(Charsets.UTF_8).size > 1048576) return
                runOnUiThread { finish(json) }
            }
        }
        view.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = false
            allowFileAccess = false
            allowContentAccess = false
            @Suppress("DEPRECATION")
            allowFileAccessFromFileURLs = false
            @Suppress("DEPRECATION")
            allowUniversalAccessFromFileURLs = false
            blockNetworkLoads = true
            mediaPlaybackRequiresUserGesture = true
        }
        view.webViewClient = object : WebViewClient() {
            override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest) = true
            @Suppress("DEPRECATION")
            override fun shouldOverrideUrlLoading(view: WebView, url: String) = true
            override fun shouldInterceptRequest(view: WebView, request: WebResourceRequest): WebResourceResponse? =
                if (FeedbackPage.mediaAllowed(request.url.toString())) null
                else WebResourceResponse("text/plain", "utf-8", ByteArrayInputStream(ByteArray(0)))
            override fun onPageStarted(view: WebView, url: String?, favicon: Bitmap?) {
                if (url != ORIGIN) view.stopLoading()
            }
        }
        view.addJavascriptInterface(bridge, "FeedbackHost")
        dialog.setContentView(view, ViewGroup.LayoutParams(-1, -1))
        dialog.setOnCancelListener { finish(null) }
        view.loadDataWithBaseURL(ORIGIN, page, "text/html", "UTF-8", null)
        dialog.show()
    }

    companion object {
        private const val ORIGIN = "https://offline.ggtts.invalid/"
        private fun sha(data: ByteArray) = MessageDigest.getInstance("SHA-256")
            .digest(data).joinToString("") { "%02x".format(it) }

    }
}
