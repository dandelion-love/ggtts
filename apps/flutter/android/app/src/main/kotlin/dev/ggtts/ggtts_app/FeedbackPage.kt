package dev.ggtts.ggtts_app

import java.io.File

internal object FeedbackPage {
    fun mediaAllowed(url: String) = url.startsWith("data:audio/flac;base64,")
    fun adapt(template: String, input: String, digest: String): String {
            fun String.once(old: String, replacement: String): String {
                require(indexOf(old) >= 0 && indexOf(old) == lastIndexOf(old))
                return replace(old, replacement)
            }
            var page = template.once("<!-- GGTTS_INPUT -->",
                "<meta name=\"feedback-input-sha256\" content=\"" + digest +
                    "\"><script id=\"feedback-input\" type=\"application/json\">" + input + "</script>")
            page = page.once(">Export feedback</button>", ">Submit feedback</button>")
            page = page.once("<label for=\"import-file\">Restore feedback from a JSON file</label>", "")
            page = page.once("<input id=\"import-file\" type=\"file\" accept=\"application/json,.json\">",
                "<input id=\"import-file\" type=\"file\" hidden disabled>")
            val start = page.indexOf("    function persist() {")
            val end = page.indexOf("    function render() {", start)
            require(start >= 0 && end > start && page.indexOf("    function persist() {", start + 1) < 0)
            page = page.substring(0, start) + """
    function persist() {
      if (FeedbackHost.save(JSON.stringify(state))) notify('Draft saved on this device.');
      else notify('Draft could not be saved.', true);
    }
""" + page.substring(end)
            val blob = page.indexOf("        const blob = new Blob(")
            val exported = page.indexOf("        notify('Feedback exported.');", blob)
            require(blob >= 0 && exported > blob && page.indexOf("        const blob = new Blob(", blob + 1) < 0 &&
                page.lastIndexOf("        notify('Feedback exported.');") == exported)
            page = page.substring(0, blob) + "        FeedbackHost.submit(JSON.stringify(state));\n" +
                page.substring(exported).once("notify('Feedback exported.');", "notify('Submitting feedback.');")
            return page.once("const raw = localStorage.getItem(storageKey);",
                "const savedResult = FeedbackHost.load(); const raw = savedResult ? JSON.stringify({input_sha256: inputDigest, result: JSON.parse(savedResult)}) : null;")
        }
}

internal class FeedbackDraft(private val file: File) {
    fun save(json: String): Boolean = try {
        require(json.toByteArray(Charsets.UTF_8).size <= 1048576)
        val tmp = File(file.parentFile, file.name + ".tmp")
        tmp.writeText(json, Charsets.UTF_8)
        if (!tmp.renameTo(file)) { tmp.delete(); false } else true
    } catch (_: Exception) { false }

    fun load(): String = try {
        if (file.length() > 1048576) "" else file.takeIf { it.exists() }?.readText(Charsets.UTF_8) ?: ""
    } catch (_: Exception) { "" }
}
