package ggtts.demo

/** Adapts the exact trusted shared template; fails closed if the shared structure changes. */
internal fun feedbackHostPage(template: String, input: PreparedFeedbackInput): String {
    fun String.once(old: String, replacement: String): String {
        check(indexOf(old) >= 0 && indexOf(old) == lastIndexOf(old)) { "shared feedback template changed" }
        return replace(old, replacement)
    }
    require(input.canonicalJson.none { it == '<' || it == '>' || it == '&' })
    val embedded = "<meta name=\"feedback-input-sha256\" content=\"" + input.digest +
        "\"><script id=\"feedback-input\" type=\"application/json\">" + input.canonicalJson + "</script>"
    var page = template.once("<!-- GGTTS_INPUT -->", embedded)
    page = page.once(">Export feedback</button>", ">Save feedback</button>")
    page = page.once("<label for=\"import-file\">Restore feedback from a JSON file</label>", "")
    page = page.once("<input id=\"import-file\" type=\"file\" accept=\"application/json,.json\">",
        "<input id=\"import-file\" type=\"file\" hidden disabled>")
    val persistStart = page.indexOf("    function persist() {")
    val persistEnd = page.indexOf("    function render() {", persistStart)
    check(persistStart >= 0 && persistEnd > persistStart)
    page = page.substring(0, persistStart) + """
    function persist() {
      if (FeedbackHost.save(JSON.stringify(state))) notify('Feedback saved on this device.');
      else notify('Feedback could not be saved on this device.', true);
    }
""" + page.substring(persistEnd)
    val start = page.indexOf("        const blob = new Blob(")
    val end = page.indexOf("        notify('Feedback exported.');", start)
    check(start >= 0 && end > start)
    page = page.substring(0, start) +
        "        if (!FeedbackHost.save(JSON.stringify(state))) throw new Error('Could not save feedback.');\n" +
        page.substring(end).once("notify('Feedback exported.');", "notify('Feedback saved.');")
    page = page.once("const raw = localStorage.getItem(storageKey);",
        "const savedResult = FeedbackHost.load(); const raw = savedResult ? JSON.stringify({input_sha256: inputDigest, result: JSON.parse(savedResult)}) : null;")
    return page
}
