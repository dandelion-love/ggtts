package ggtts.demo

import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.util.concurrent.TimeUnit

/** Exercises the actual transformed page in Chromium; the bridge is a test fake, not Android WebView. */
class FeedbackHostBrowserTest {
    @Test fun hostPageSavesRestoresAndPlaysOffline() {
        val chrome = listOf("/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser")
            .firstOrNull { File(it).canExecute() }
        assumeTrue("Chromium required", chrome != null)
        var root = File(System.getProperty("user.dir") ?: error("missing user.dir")).absoluteFile
        while (!File(root, "android/demo/src/main/assets/feedback-page.html").isFile) root = root.parentFile ?: break
        val template = File(root, "android/demo/src/main/assets/feedback-page.html")
        assumeTrue("bundled page template required", template.isFile)

        val samples = FloatArray(7200) { i -> (kotlin.math.sin(i * 0.05) * 0.15).toFloat() }
        val wav = Wav.encode(samples, 24000)
        val evidence = VerifiedWavEvidence()
        evidence.add(0, "A中😀𠀀，文", 1, 0, 0, 0, samples.size.toLong(), samples, 24000)
        evidence.complete(true)
        val manifest = evidence.verify("test-read", wav)
        val input = FeedbackInputAdapter.fromWav(wav, manifest, "browser-test")
        val prepared = PreparedFeedbackInput(input.canonicalJson)
        var page = feedbackHostPage(template.readText(), prepared)
        val bridge = """
<script>
window.FeedbackHost = {
 save: function(json) { localStorage.setItem('fake-bridge', json); return true; },
 load: function() { return localStorage.getItem('fake-bridge') || ''; }
};
</script>
"""
        page = page.replace("<script>", bridge + "<script>", ignoreCase = false)
        val harness = """
<script>
(async () => {
 const errors=[], check=(b,s)=>{if(!b) errors.push(s);};
 const finish=()=>{document.body.dataset.hostTest=errors.length?'FAIL':'PASS';document.body.dataset.hostErrors=errors.join(' | ');};
 const controls=[...document.querySelectorAll('button.character')];
 check(controls.length===3, 'CJK count');
 check(document.querySelectorAll('section.item').length===1, 'page initialization');
 check(performance.getEntriesByType('resource').every(x=>/^(data:|file:|blob:)/.test(x.name)), 'network resource');
 const phase=sessionStorage.getItem('host-phase');
 if (!phase) {
   controls[1].click();
   const item=document.querySelector('section.item textarea');
   item.value='sentence note'; item.dispatchEvent(new Event('input',{bubbles:true}));
   const note=document.getElementById('read-note');
   note.value='read note'; note.dispatchEvent(new Event('input',{bubbles:true}));
   document.getElementById('export').click();
   const saved=JSON.parse(localStorage.getItem('fake-bridge') || '{}');
   check(saved.format==='ggtts-feedback-result/1' && saved.id==='browser-test', 'bridge result identity');
   check(saved.input_sha256===document.querySelector('meta[name=feedback-input-sha256]').content, 'digest');
   check(JSON.stringify(saved.items?.[0]?.marks)==='[1]' && saved.items[0].note==='sentence note' &&
     saved.note==='read note', 'marks/notes not saved');
   const audio=document.querySelector('audio');
   let playing=false; audio.addEventListener('playing',()=>playing=true,{once:true});
   const ended=new Promise(r=>audio.addEventListener('ended',r,{once:true}));
   try {
     await audio.play();
     await Promise.race([ended, new Promise(r=>setTimeout(r,5000))]);
     check(playing && audio.ended && audio.currentTime>0, 'FLAC playback did not reach ended');
   } catch(e) { errors.push('FLAC play failed: '+e.name); }
   if(errors.length){finish(); return;}
   sessionStorage.setItem('host-phase','restored');
   location.reload();
 } else {
   check(controls[1]?.getAttribute('aria-pressed')==='true', 'mark not restored');
   check(document.querySelector('section.item textarea')?.value==='sentence note', 'sentence note not restored');
   check(document.getElementById('read-note').value==='read note', 'read note not restored');
   check(document.getElementById('message').textContent.includes('restored'), 'restore not reported');
   finish();
 }
})().catch(e=>{document.body.dataset.hostTest='FAIL';document.body.dataset.hostErrors=String(e.stack||e);});
</script>
"""
        page = page.replace("</body>", harness + "</body>")
        val temp = kotlin.io.path.createTempDirectory("ggtts-host-test-").toFile()
        try {
            val path = File(temp, "feedback.html").apply { writeText(page) }
            val out = File(temp, "dom.txt")
            val err = File(temp, "chrome.err")
            val driver = File(root, "scripts/test_feedback_host_browser.py")
            val process = ProcessBuilder("python3", driver.path, chrome!!, path.path)
                .redirectOutput(out).redirectError(err).start()
            if (!process.waitFor(35, TimeUnit.SECONDS)) { process.destroyForcibly(); fail("Chromium timed out") }
            assertEquals("browser driver: " + out.readText() + err.readText(), 0, process.exitValue())
        } finally { temp.deleteRecursively() }
    }
}
