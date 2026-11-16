import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:material_ui/material_ui.dart';
import 'package:ggtts_app/listening_page.dart';
import 'package:ggtts_app/listening_source.dart';

import 'support/listening_source_fake.dart';

/// Escape flutter_test's mock HttpClient only for the loopback fake in this test.
class _RealHttp extends HttpOverrides {}

class _LoopbackSource implements ListeningSource {
  _LoopbackSource(Uri url) : inner = HttpListeningSource(url);
  final HttpListeningSource inner;
  final overrides = _RealHttp();
  Future<T> run<T>(Future<T> Function() f) =>
      HttpOverrides.runWithHttpOverrides(f, overrides);
  @override
  String get identity => inner.identity;
  @override
  Future<void> signIn(String user, String password) =>
      run(() => inner.signIn(user, password));
  @override
  Future<List<ListeningSession>> sessions() => run(inner.sessions);
  @override
  Future<ListeningAudio> audio(String sessionId, ListeningClip clip) =>
      run(() => inner.audio(sessionId, clip));
  @override
  Future<void> submit(
    String sessionId,
    Map<String, Object?> body,
    String key,
  ) => run(() => inner.submit(sessionId, body, key));
}

Future<void> pumpUntil(WidgetTester tester, bool Function() condition) async {
  final end = DateTime.now().add(const Duration(seconds: 10));
  while (!condition() && DateTime.now().isBefore(end)) {
    await Future<void>.delayed(const Duration(milliseconds: 40));
    await tester.pump();
  }
  expect(condition(), isTrue, reason: 'timed out waiting for Flutter state');
}

const harness = r"""
<script>
(async()=>{
 const done=(ok,why,result)=>{document.body.dataset.hostTest=ok?'PASS':'FAIL';
   document.body.dataset.hostErrors=why; if(result) document.body.dataset.hostResult=result;};
 try {
   const chars=[...document.querySelectorAll('button.character')];
   if(chars.length!==3) return done(false,'wrong CJK count');
   chars[0].click(); chars[1].click(); chars[2].click();
   const item=document.querySelector('section.item textarea');
   item.value='item note'; item.dispatchEvent(new Event('input',{bubbles:true}));
   const note=document.getElementById('read-note');
   note.value='page note'; note.dispatchEvent(new Event('input',{bubbles:true}));
   const rating=document.querySelector('section.item select');
   rating.value='accept'; rating.dispatchEvent(new Event('change',{bubbles:true}));
   const audio=document.querySelector('audio');
   let playing=false; audio.addEventListener('playing',()=>playing=true,{once:true});
   const ended=new Promise(r=>audio.addEventListener('ended',r,{once:true}));
   await audio.play();
   await Promise.race([ended,new Promise(r=>setTimeout(r,5000))]);
   if(!playing || !audio.ended || audio.currentTime<=0) return done(false,'audio did not reach ended');
   let blob=null; URL.createObjectURL=b=>{blob=b;return 'blob:test';};
   HTMLAnchorElement.prototype.click=function(){};
   document.getElementById('export').click();
   if(!blob) return done(false,'no exported blob');
   const external=performance.getEntriesByType('resource').filter(x=>!(/^(data:|file:|blob:)/.test(x.name)));
   if(external.length) return done(false,'external page resource');
   done(true,'',await blob.text());
 } catch(e) { done(false,String(e.stack||e)); }
})();
</script>
""";

// Linux-only: this test relies on POSIX process-group cleanup.
// Set GGTTS_REQUIRE_CHROMIUM=1 to require the test (rather than skip).
// Set CHROME_BIN to an explicit Chromium executable. CI is not automatically configured to require it.
void main() {
  testWidgets('generated page plays and exports into real source retry queue', (
    tester,
  ) async {
    if (!Platform.isLinux) {
      if (Platform.environment['GGTTS_REQUIRE_CHROMIUM'] == '1') {
        fail('This required Chromium test supports Linux only');
      }
      markTestSkipped('Joined Chromium test is Linux-only');
      return;
    }
    final chrome =
        [
          Platform.environment['CHROME_BIN'],
          '/usr/bin/google-chrome',
          '/usr/bin/chromium',
          '/usr/bin/chromium-browser',
        ].whereType<String>().firstWhere(
          (path) => File(path).existsSync(),
          orElse: () => '',
        );
    if (chrome.isEmpty) {
      if (Platform.environment['GGTTS_REQUIRE_CHROMIUM'] == '1') {
        fail('Chromium required by GGTTS_REQUIRE_CHROMIUM=1');
      }
      markTestSkipped(
        'Chromium unavailable; set CHROME_BIN or GGTTS_REQUIRE_CHROMIUM=1',
      );
      return;
    }
    final driver = File('../../scripts/test_flutter_feedback_browser.py')
        .absolute;
    if (!driver.existsSync()) {
      fail('Required browser driver missing: ${driver.path}');
    }
    final temp = Directory.systemTemp.createTempSync('ggtts-joined-');
    addTearDown(() async => temp.delete(recursive: true));
    final fake = (await tester.runAsync(
      () => ListeningSourceFake.start(clipText: 'A中😀𠀀，文'),
    ))!;
    addTearDown(fake.close);
    final queue = ListeningQueue(File('${temp.path}/queue.json'));
    await tester.pumpWidget(
      MaterialApp(
        home: ListeningPage(
          sourceFactory: _LoopbackSource.new,
          queue: queue,
          feedbackDirectory: () async => temp,
        ),
      ),
    );
    await tester.enterText(
      find.byKey(const Key('listening-source')),
      fake.baseUrl.toString(),
    );
    await tester.enterText(find.byKey(const Key('listening-user')), 'listener');
    await tester.enterText(
      find.byKey(const Key('listening-password')),
      'secret',
    );
    await tester.runAsync(() async {
      await tester.tap(find.text('Sign in'));
      await tester.pump();
      await pumpUntil(
        tester,
        () => find.text('Assigned test session').evaluate().isNotEmpty,
      );
      await tester.tap(find.text('Assigned test session'));
      await tester.pump();
      await pumpUntil(
        tester,
        () => find.text('A中😀𠀀，文').evaluate().isNotEmpty,
      );
      await tester.tap(find.text('A中😀𠀀，文'));
      await tester.pump();
      await pumpUntil(
        tester,
        () => find.text('Prepare shared feedback page').evaluate().isNotEmpty,
      );
      await tester.ensureVisible(find.text('Prepare shared feedback page'));
      await tester.tap(find.text('Prepare shared feedback page'));
      await tester.pump();
      await pumpUntil(
        tester,
        () => find
            .byKey(const Key('shared-feedback-result'))
            .evaluate()
            .isNotEmpty,
      );
      final html = temp.listSync().whereType<File>().singleWhere(
        (f) => f.path.endsWith('.html'),
      );
      final page = await html.readAsString();
      expect(page, isNot(contains('fake-listener-token')));
      expect(page, isNot(contains('sig=expired')));
      expect(page, isNot(contains('sig=fresh')));
      final instrumented = File('${temp.path}/browser-test.html');
      await instrumented.writeAsString(
        page.replaceFirst('</body>', '$harness</body>'),
      );
      // The driver can spend 20 s starting and 20 s running the page; allow
      // 55 s plus its cleanup, and terminate it explicitly on timeout.
      final groupFile = File('${temp.path}/browser-process-group');
      final browser = await Process.start('python3', [
        driver.path,
        chrome,
        instrumented.path,
        groupFile.path,
      ]);
      final stdoutFuture = browser.stdout.transform(utf8.decoder).join();
      final stderrFuture = browser.stderr.transform(utf8.decoder).join();
      int exit;
      try {
        exit = await browser.exitCode.timeout(const Duration(seconds: 55));
      } on TimeoutException {
        browser.kill(ProcessSignal.sigterm);
        try {
          await browser.exitCode.timeout(const Duration(seconds: 8));
        } on TimeoutException {
          // The Python driver normally terminates its Chrome group in finally.
          // If it is unresponsive, stop that recorded group before killing it.
          if (!Platform.isWindows && await groupFile.exists()) {
            final group = int.tryParse((await groupFile.readAsString()).trim());
            if (group != null && group > 1) {
              await Process.run('kill', ['-KILL', '--', '-$group']);
            }
          }
          browser.kill(ProcessSignal.sigkill);
          await browser.exitCode;
        }
        fail('Browser driver timed out and was terminated');
      }
      final stdout = await stdoutFuture;
      final stderr = await stderrFuture;
      expect(exit, 0, reason: '$stdout\n$stderr');
      final output = jsonDecode(stdout) as Map;
      expect(output['state'], 'PASS');
      final result = output['result'] as String;
      await tester.enterText(
        find.byKey(const Key('shared-feedback-result')),
        result,
      );
      await tester.ensureVisible(find.text('Submit feedback'));
      await tester.tap(find.text('Submit feedback'));
      await tester.pump();
      await pumpUntil(
        tester,
        () => fake.requests.any((r) => r.path.endsWith('/marks')),
      );
      await pumpUntil(
        tester,
        () => find.textContaining('Feedback is queued').evaluate().isNotEmpty,
      );
      final saved = await queue.read();
      expect(saved, hasLength(1));
      expect(jsonEncode(saved), isNot(contains('fake-listener-token')));
      expect(jsonEncode(saved), isNot(contains('sig=')));
      await tester.tap(find.byTooltip('Retry pending'));
      await tester.pump();
      await pumpUntil(
        tester,
        () => find
            .textContaining('Pending feedback uploaded')
            .evaluate()
            .isNotEmpty,
      );
      final marks = fake.submissions.single['marks'] as List;
      expect(marks.map((e) => (e as Map)['start']), [1, 3, 5]);
      expect(fake.submissions.single['verdict'], 'accept');
      final notes =
          jsonDecode(fake.submissions.single['note'] as String) as Map;
      expect(notes['page_note'], 'page note');
      expect(notes['item_note'], 'item note');
      final posts = fake.requests
          .where((r) => r.path.endsWith('/marks'))
          .toList();
      expect(posts, hasLength(2));
      expect(posts[0].idempotencyKey, posts[1].idempotencyKey);
      final audio = fake.requests
          .where((r) => r.path.startsWith('/audio/'))
          .toList();
      expect(audio.map((r) => r.path), ['/audio/expired', '/audio/fresh']);
      expect(audio.every((r) => !r.hasAuthorization), isTrue);
      expect(await queue.read(), isEmpty);
    });
  });
}
