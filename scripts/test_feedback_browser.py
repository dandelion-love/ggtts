#!/usr/bin/env python3
"""Opt-in offline Chromium regression tests for feedback/page.html.

Requires ffmpeg and Chromium (CHROME_BIN or google-chrome). Uses synthetic tones only.
"""
import base64, json, os, shutil, subprocess, tempfile
from pathlib import Path
from scripts.feedback_page import generate, input_digest

HARNESS = r"""
<script>
try { Object.defineProperty(window, "crypto", {value:undefined, configurable:true}); } catch (_) {}
</script>
<script>
(async () => {
 const errors=[], check=(ok,msg)=>{if(!ok)errors.push(msg);};
 const done=()=>{document.body.dataset.feedbackTest=errors.length?'FAIL':'PASS';
   document.body.dataset.feedbackErrors=errors.join(' | ');};
 const phase=sessionStorage.getItem('ggtts-test-phase');
 const external=performance.getEntriesByType('resource').map(x=>x.name)
  .filter(x=>!x.startsWith('data:')&&!x.startsWith('file:')&&!x.startsWith('blob:'));
 check(external.length===0,'external resource request');
 check(window.crypto===undefined,'test did not disable WebCrypto');
 const csp=document.querySelector('meta[http-equiv="Content-Security-Policy"]')?.content||'';
 check(csp.includes("default-src 'none'")&&csp.includes("connect-src 'none'"),'offline CSP');
 if(!phase){
  const chars=[...document.querySelectorAll('button.character')];
  check(chars.length===3&&chars.map(x=>x.textContent).join('')===String.fromCodePoint(0x4e2d,0x20000,0x6587),
   'CJK scalar controls');
  if(chars.length===3){
   chars[2].click(); check(chars[2].getAttribute('aria-pressed')==='true','supplementary CJK index 2');
   const rating=document.querySelector('select[aria-label="Rating for Clip 1"]');
   rating.value='accept'; rating.dispatchEvent(new Event('change',{bubbles:true}));
   check(chars[2].getAttribute('aria-pressed')==='true','rating changed marks');
   const n=document.querySelector('section.item textarea');n.value='item note';n.dispatchEvent(new Event('input',{bubbles:true}));
   const rn=document.getElementById('read-note');rn.value='read note';rn.dispatchEvent(new Event('input',{bubbles:true}));
  }
  const audio=document.querySelector('audio');
  check(audio?.src.startsWith('data:audio/flac;base64,'),'audio not inlined');
  if(audio){audio.preload='metadata';audio.load();await Promise.race([
   new Promise(r=>audio.addEventListener('loadedmetadata',r,{once:true})),new Promise(r=>setTimeout(r,1500))]);
   check(Number.isFinite(audio.duration)&&audio.duration>0,'synthetic FLAC did not decode offline');}
  let blob=null;URL.createObjectURL=b=>{blob=b;return 'blob:feedback-test';};
  HTMLAnchorElement.prototype.click=function(){};
  document.getElementById('export').click();
  if(!blob){errors.push('export did not create blob');done();return;}
  const good=JSON.parse(await blob.text());
  check(good.format==='ggtts-feedback-result/1'&&good.id==='browser-test'&&
   /^[0-9a-f]{64}$/.test(good.input_sha256),'export identity/digest');
  check(JSON.stringify(good.items?.[0]?.marks)==='[2]'&&good.items[0].rating==='accept'&&
   good.items[0].note==='item note'&&good.note==='read note','export fields/rating independence');
  const invalid=JSON.parse(JSON.stringify(good));invalid.items[0].marks=[3];
  await importResult(invalid);
  check(document.getElementById('message').classList.contains('error'),'out-of-range import mark accepted');
  const mismatch=JSON.parse(JSON.stringify(good));mismatch.input_sha256='__CHANGED_DIGEST__';
  await importResult(mismatch);
  check(document.getElementById('message').classList.contains('error'),'same-ID changed-input result accepted');
  blob=null;document.getElementById('export').click();
  const still=JSON.parse(await blob.text());
  check(JSON.stringify(still.items?.[0]?.marks)==='[2]','invalid import replaced feedback');
  if(audio){audio.src='data:audio/flac;base64,Zm9v';audio.load();await new Promise(r=>setTimeout(r,700));
   check(document.getElementById('message').classList.contains('error'),'audio error was not surfaced');}
  const key=Object.keys(localStorage).find(k=>k.startsWith('ggtts-feedback-v1:'));
  check(Boolean(key),'draft not persisted');
  if(key){const saved=JSON.parse(localStorage.getItem(key));saved.result.items[0].marks=[3];
   localStorage.setItem(key,JSON.stringify(saved));}
  sessionStorage.setItem('ggtts-test-phase','corrupt-reload');location.reload();return;
 }
 if(phase==='corrupt-reload'){
  const chars=[...document.querySelectorAll('button.character')];
  check(document.getElementById('message').classList.contains('error'),'invalid draft index unreported');
  check(chars.length===3&&chars.every(x=>x.getAttribute('aria-pressed')==='false'),'invalid draft partially restored');
  check(document.querySelector('select[aria-label="Rating for Clip 1"]')?.value==='unsure','invalid draft changed rating');
  done();return;
 }
 errors.push('unexpected phase');done();
})().catch(e=>{document.body.dataset.feedbackTest='FAIL';document.body.dataset.feedbackErrors=String(e.stack||e);});
async function importResult(value){
 const d=new DataTransfer();d.items.add(new File([JSON.stringify(value)],'feedback.json',{type:'application/json'}));
 const f=document.getElementById('import-file');f.files=d.files;f.dispatchEvent(new Event('change',{bubbles:true}));
 await new Promise(r=>setTimeout(r,60));
}
</script>
"""

CHANGED_HARNESS = r"""
<script>
(async()=>{
 const errors=[],check=(ok,s)=>{if(!ok)errors.push(s);};
 const d=new DataTransfer();
 d.items.add(new File([JSON.stringify(__OLD_RESULT__)],'old.json',{type:'application/json'}));
 const f=document.getElementById('import-file');f.files=d.files;f.dispatchEvent(new Event('change',{bubbles:true}));
 await new Promise(r=>setTimeout(r,80));
 check(document.getElementById('message').classList.contains('error'),'changed text/audio with same IDs accepted old result');
 check([...document.querySelectorAll('button.character')].every(x=>x.getAttribute('aria-pressed')==='false'),'changed input restored old marks');
 check(document.querySelector('select[aria-label="Rating for Clip 1"]')?.value==='unsure','changed input restored old rating');
 document.body.dataset.feedbackTest=errors.length?'FAIL':'PASS';document.body.dataset.feedbackErrors=errors.join(' | ');
})().catch(e=>{document.body.dataset.feedbackTest='FAIL';document.body.dataset.feedbackErrors=String(e.stack||e);});
</script>
"""

BAD_DIGEST_HARNESS = r"""
<script>
document.body.dataset.digestTest =
 document.getElementById('message').textContent === 'Feedback input identity mismatch.' &&
 document.querySelector('#export').disabled ? 'PASS' : 'FAIL';
</script>
"""

def run(chrome, page, profile):
 pagepath=profile.parent/'feedback.html'
 pagepath.write_text(page,encoding='utf-8')
 cmd=[chrome,'--headless=new','--no-sandbox','--disable-gpu','--disable-background-networking',
  '--disable-sync','--no-first-run','--host-resolver-rules=MAP * ~NOTFOUND',
  f'--user-data-dir={profile}','--virtual-time-budget=5000','--dump-dom',pagepath.as_uri()]
 p=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
 if p.returncode:raise RuntimeError(f'Chromium exit {p.returncode}: {p.stderr[-2500:]}')
 return p.stdout,p.stderr

def assert_pass(dom, attr='data-feedback-test'):
 marker=attr+'="PASS"'
 if marker not in dom:
  key='data-feedback-errors="'
  detail=dom.split(key,1)[1].split('"',1)[0] if key in dom else 'result marker missing'
  raise AssertionError(detail)

def main():
 ffmpeg=shutil.which('ffmpeg')
 chrome=os.environ.get('CHROME_BIN') or shutil.which('google-chrome') or shutil.which('chromium')
 if not ffmpeg or not chrome:raise SystemExit('SKIP: requires ffmpeg and Chromium/google-chrome')
 with tempfile.TemporaryDirectory(prefix='ggtts-feedback-browser-') as tmp:
  root=Path(tmp)
  def audio(freq,name):
   path=root/name
   subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-f','lavfi','-i',
    f'sine=frequency={freq}:sample_rate=24000:duration=0.25','-c:a','flac','-f','flac',str(path)],check=True)
   return 'data:audio/flac;base64,'+base64.b64encode(path.read_bytes()).decode('ascii')
  source={'format':'ggtts-feedback-input/1','id':'browser-test','items':[
   {'id':'clip-1','text':'A'+chr(0x4e2d)+chr(0x1f600)+chr(0x20000)+chr(0xff0c)+chr(0x6587),'audio':audio(440,'a.flac')}]}
  changed={'format':'ggtts-feedback-input/1','id':'browser-test','items':[
   {'id':'clip-1','text':source['items'][0]['text']+chr(0x6587),'audio':audio(660,'b.flac')}]}
  old_digest=input_digest(source);changed_digest=input_digest(changed)
  if old_digest==changed_digest:raise AssertionError('changed source digest did not change')
  old_result={'format':'ggtts-feedback-result/1','id':'browser-test','input_sha256':old_digest,'note':'',
   'items':[{'id':'clip-1','marks':[2],'rating':'accept','note':'item note'}]}
  harness=HARNESS.replace('__CHANGED_DIGEST__',changed_digest)
  first=generate(source).replace('<script>\n  \'use strict\';',
   '<script>\n  try { Object.defineProperty(window,"crypto",{value:undefined,configurable:true}); } catch (_) {}\n  \'use strict\';')
  first=first.replace('</body>',harness+'\n</body>')
  profile=root/'profile'
  dom,_=run(chrome,first,profile);assert_pass(dom)
  changed_page=generate(changed).replace('</body>',CHANGED_HARNESS.replace('__OLD_RESULT__',json.dumps(old_result,separators=(',',':')))+'\n</body>')
  dom,_=run(chrome,changed_page,profile);assert_pass(dom)
  wrong=generate(source).replace('name="feedback-input-sha256" content="'+old_digest+'"',
   'name="feedback-input-sha256" content="'+'0'*64+'"').replace('</body>',BAD_DIGEST_HARNESS+'\n</body>')
  dom,_=run(chrome,wrong,root/'bad-profile')
  if 'data-digest-test="PASS"' not in dom:raise AssertionError('tampered embedded-input digest accepted')
  mutant=generate(source).replace("const position = index++;","const position = index++ + 1;")
  if mutant==generate(source):raise AssertionError('wrong-index mutant did not alter page code')
  mutant=mutant.replace('</body>',harness+'\n</body>')
  dom,_=run(chrome,mutant,root/'mutant-profile')
  if 'data-feedback-test="FAIL"' not in dom or 'export did not create blob' not in dom:
   raise AssertionError('off-by-one CJK index plant escaped the browser oracle')
  print('PASS offline Chromium: offline/CSP, no-WebCrypto boot, CJK scalar marks, rating independence, draft reload, export, invalid index, audio error, changed-input result/draft isolation, digest tamper; planted off-by-one index was detected')
if __name__=='__main__':main()
