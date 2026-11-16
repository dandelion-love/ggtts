import hashlib, os, pathlib, shutil, subprocess, tempfile, unittest
SCRIPT = pathlib.Path(__file__).resolve().parents[1] / 'android_media_recovery_test.sh'
AAPT = """#!/usr/bin/env python3
import os,sys
if sys.argv[2]=='badging':
 print("package: name='%s'" % (os.getenv('TEST_ID','ggtts.demo.recovery.test') if 'androidTest' in sys.argv[3] else os.getenv('APP_ID','ggtts.demo.recovery')))
else:
 print('android:targetPackage(0x01010021)="%s" ' % os.getenv('TARGET','ggtts.demo.recovery'))
 print('android:name(0x01010003)="%s"' % os.getenv('RUNNER','androidx.test.runner.AndroidJUnitRunner'))
"""
ADB = """#!/usr/bin/env python3
import os,sys
a=sys.argv[1:]
with open(os.environ['LOG'],'a') as f: f.write(' '.join(a)+'\\n')
if 'services' in a and os.getenv('ACTIVE'): print('ggtts.demo.recovery/.RenderService')
if 'activities' in a and os.getenv('VISIBLE'): print('topResumedActivity: ggtts.demo.recovery/.MainActivity')
if 'list' in a:
 if os.getenv('LIST_FAIL'): sys.exit(1)
 if os.getenv('BAD_LIST'): print('Error: package manager unavailable')
 elif os.getenv('INSTALLED'): print('package:' + a[-1])
 elif os.getenv('SIMILAR'): print('package:' + a[-1] + '.other')
if 'path' in a:
 if os.getenv('PATH_FAIL'): sys.exit(1)
 if os.getenv('INSTALLED'):
  print('package:/data/app/base.apk')
  if os.getenv('SPLIT'): print('package:/data/app/split.apk')
if 'run-as' in a and os.getenv('STALE'): print('files/queue-recovery-old.txt')
if 'sha256sum' in a: print(os.getenv('HASH','differenthash') + ' /data/app/base.apk')
if 'instrument' in a:
 if 'cleanupExactRowsAndJournals' in ' '.join(a) and os.getenv('FAIL'):
  print('INSTRUMENTATION_FAILED'); sys.exit(1)
 print('OK (1 test)')
"""
class Safety(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); r=pathlib.Path(self.tmp.name)
  (r/'scripts').mkdir(); shutil.copy(SCRIPT,r/'scripts')
  for p in ('debug/demo-debug.apk','androidTest/debug/demo-debug-androidTest.apk'):
   f=r/'android/demo/build/outputs/apk'/p; f.parent.mkdir(parents=True,exist_ok=True); f.write_bytes(b'fake')
  sdk=r/'sdk'
  for p,c in [('platform-tools/adb',ADB),('build-tools/35.0.0/aapt',AAPT)]:
   f=sdk/p; f.parent.mkdir(parents=True,exist_ok=True); f.write_text(c); f.chmod(0o755)
  self.log=r/'log'; self.script=r/'scripts'/SCRIPT.name
  self.env=dict(os.environ,ANDROID_HOME=str(sdk),ANDROID_SERIAL='fake',LOG=str(self.log))
 def run_script(self,**kw): return subprocess.run(['bash',str(self.script)],env=dict(self.env,**kw),capture_output=True,text=True)
 def calls(self): return self.log.read_text().splitlines() if self.log.exists() else []
 def test_identity_refuses_without_adb(self):
  for k,v in [('APP_ID','ggtts.demo'),('TEST_ID','other'),('TARGET','ggtts.demo'),('RUNNER','other')]:
   with self.subTest(k=k):
    p=self.run_script(**{k:v}); self.assertNotEqual(p.returncode,0); self.assertIn('refusing install',p.stderr); self.assertEqual(self.calls(),[])
 def test_busy_or_stale_refuses_install(self):
  for k in ('ACTIVE','VISIBLE','STALE'):
   with self.subTest(k=k):
    self.log.unlink(missing_ok=True); p=self.run_script(**{k:'1','INSTALLED':'1'})
    self.assertNotEqual(p.returncode,0); self.assertFalse(any(' install ' in ' '+c+' ' for c in self.calls()))
 def test_only_isolated_updates_and_force_stop(self):
  p=self.run_script(INSTALLED='1'); self.assertEqual(p.returncode,0,p.stdout+p.stderr)
  calls=self.calls(); ins=[c for c in calls if ' install ' in ' '+c+' ']
  self.assertEqual(len(ins),2); self.assertTrue(all(' install -r -t ' in ' '+c+' ' for c in ins))
  self.assertTrue(any('demo-debug.apk' in c and 'androidTest' not in c for c in ins))
  self.assertTrue(any('demo-debug-androidTest.apk' in c for c in ins))
  self.assertEqual([c for c in calls if 'force-stop' in c],['-s fake shell am force-stop ggtts.demo.recovery'])
 def test_inspection_failure_refuses_install(self):
  for kw in ({'LIST_FAIL':'1'}, {'BAD_LIST':'1'}, {'PATH_FAIL':'1','INSTALLED':'1'}):
   with self.subTest(kw=kw):
    self.log.unlink(missing_ok=True); p=self.run_script(**kw)
    self.assertNotEqual(p.returncode,0,p.stdout+p.stderr)
    self.assertFalse(any(' install ' in ' '+c+' ' for c in self.calls()))
 def test_absent_and_similar_package_are_installed_without_replace(self):
  for kw in ({}, {'SIMILAR':'1'}):
   with self.subTest(kw=kw):
    self.log.unlink(missing_ok=True); p=self.run_script(**kw)
    self.assertEqual(p.returncode,0,p.stdout+p.stderr)
    ins=[c for c in self.calls() if ' install ' in ' '+c+' ']
    self.assertEqual(len(ins),2); self.assertTrue(all(' install -t ' in ' '+c+' ' for c in ins))
 def test_split_refused_and_matching_apks_untouched(self):
  p=self.run_script(INSTALLED='1',SPLIT='1')
  self.assertNotEqual(p.returncode,0); self.assertIn('multi-split',p.stderr)
  self.assertFalse(any(' install ' in ' '+c+' ' for c in self.calls()))
  self.log.unlink(); digest=hashlib.sha256(b'fake').hexdigest()
  p=self.run_script(INSTALLED='1',HASH=digest)
  self.assertEqual(p.returncode,0,p.stdout+p.stderr)
  self.assertFalse(any(' install ' in ' '+c+' ' for c in self.calls()))
 def test_cleanup_failure_preserves_fixture(self):
  p=self.run_script(FAIL='1'); self.assertNotEqual(p.returncode,0); self.assertIn('cleanup failed',p.stderr)
  fixture=p.stdout.split('GGTTS recovery fixture ID: ')[1].splitlines()[0]
  calls=[c for c in self.calls() if 'cleanupExactRowsAndJournals' in c]
  self.assertTrue(calls); self.assertTrue(all(fixture in c for c in calls))
if __name__=='__main__': unittest.main()
