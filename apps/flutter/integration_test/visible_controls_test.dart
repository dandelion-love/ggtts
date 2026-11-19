import 'dart:convert';
import 'dart:io';
import 'dart:ui' as ui;

import 'package:material_ui/material_ui.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:ggtts_app/main.dart';
import 'package:ggtts_app/settings_store.dart';

const passage =
    '今朝落過雨，樓下啲石級仲係濕嘅。我拎住把遮行去街市，行到轉角先發現天已經放晴。\n\n'
    '老闆娘話今日啲橙幾甜，我就揀咗三個。佢用紙袋裝好，仲提我行返屋企嗰陣要小心地下跣。\n\n'
    '返到屋企，我將啲橙擺喺窗邊，開窗畀啲風吹入嚟。隔籬有人彈琴，啱啱彈完最後一句。';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  testWidgets('visible Windows controls', (tester) async {
    final env = Platform.environment;
    final dir = Directory(env['VISIBLE_RUN_DIR']!);
    final library = Directory(env['VISIBLE_LIBRARY']!);
    final mode = env['VISIBLE_MODE']!;
    final idFile = File(env['VISIBLE_READ_ID_FILE']!);
    final options = File(env['VISIBLE_OPTIONS']!);
    final log = File('${dir.path}${Platform.pathSeparator}events.jsonl');
    final boundaryKey = GlobalKey();
    final watch = Stopwatch()..start();
    Future<void> event(
      String name, [
      Map<String, Object?> extra = const {},
    ]) async {
      await log.writeAsString(
        '${jsonEncode({'event': name, 'wall_utc': DateTime.now().toUtc().toIso8601String(), 'elapsed_ms': watch.elapsedMilliseconds, ...extra})}\n',
        mode: FileMode.append,
        flush: true,
      );
    }

    Future<void> screenshot(String name) async {
      await tester.pump();
      final boundary =
          boundaryKey.currentContext!.findRenderObject()!
              as RenderRepaintBoundary;
      final image = await boundary.toImage(pixelRatio: 1);
      final bytes = await image.toByteData(format: ui.ImageByteFormat.png);
      await File('${dir.path}${Platform.pathSeparator}$name.png')
          .writeAsBytes(bytes!.buffer.asUint8List(), flush: true);
      image.dispose();
    }

    Future<void> waitFor(Finder finder, Duration limit) async {
      final clock = Stopwatch()..start();
      while (finder.evaluate().isEmpty) {
        if (clock.elapsed > limit) {
          throw StateError('Control did not appear: $finder');
        }
        await tester.pump(const Duration(milliseconds: 200));
      }
    }

    Future<void> tap(Finder finder) async {
      await tester.ensureVisible(finder);
      await tester.pump(const Duration(milliseconds: 100));
      await tester.tap(finder);
      await tester.pump(const Duration(milliseconds: 100));
    }

    Future<void> captureReady() async {
      await File('${dir.path}${Platform.pathSeparator}app-pid.txt')
          .writeAsString('$pid', flush: true);
      final marker = File('${dir.path}${Platform.pathSeparator}capture-ready');
      final clock = Stopwatch()..start();
      while (!await marker.exists()) {
        if (clock.elapsed > const Duration(seconds: 45)) {
          throw StateError('Capture was not ready');
        }
        await tester.pump(const Duration(milliseconds: 200));
      }
    }

    await dir.create(recursive: true);
    await File('${dir.path}${Platform.pathSeparator}process-pid.txt')
        .writeAsString('$pid', flush: true);
    await tester.pumpWidget(
      RepaintBoundary(
        key: boundaryKey,
        child: MaterialApp(
          theme: ThemeData(colorSchemeSeed: Colors.teal, useMaterial3: true),
          home: ReadPage(
            readDirectory: () async => library,
            loadSettings: () => SettingsStore(options).load(),
          ),
        ),
      ),
    );
    await tester.pumpAndSettle();
    await event('app_started');
    await screenshot('start');
    if (mode == 'read') {
      await tester.enterText(find.byType(TextField).first, passage);
      await event('prepare_clicked');
      await tap(find.text('Prepare offline (selected or cached files)'));
      await waitFor(
        find.text('Prepare model again (download if needed)'),
        const Duration(seconds: 120),
      );
      await event('model_prepared');
      await screenshot('prepared');
      await captureReady();
      await event('speak_clicked');
      await tap(find.text('Speak'));
      await waitFor(find.text('Save read'), const Duration(minutes: 5));
      await event('read_finished');
      await screenshot('read_finished');
      final before = (await library.list().toList()).map((e) => e.path).toSet();
      await tap(find.text('Save read'));
      final clock = Stopwatch()..start();
      String? saved;
      while (saved == null) {
        await tester.pump(const Duration(milliseconds: 200));
        final added = (await library.list().toList())
            .where(
              (e) =>
                  e is Directory &&
                  !before.contains(e.path) &&
                  !e.uri.pathSegments
                      .where((x) => x.isNotEmpty)
                      .last
                      .startsWith('.'),
            )
            .toList();
        if (added.length == 1 && find.text('Save read').evaluate().isEmpty) {
          saved = added.single.uri.pathSegments.where((x) => x.isNotEmpty).last;
        }
        if (clock.elapsed > const Duration(seconds: 30)) {
          throw StateError('Saved read did not appear');
        }
      }
      await idFile.writeAsString(saved, flush: true);
      await event('saved', {'id': saved});
      await screenshot('saved');
    } else if (mode == 'replay' || mode == 'replay-only') {
      await event('saved_reads_clicked');
      await tap(find.byTooltip('Saved reads'));
      final id = (await idFile.readAsString()).trim();
      final row = find.byKey(ValueKey('saved-$id'));
      await waitFor(row, const Duration(seconds: 30));
      await screenshot('saved_reads');
      final play = find.descendant(of: row, matching: find.byTooltip('Play'));
      await captureReady();
      await event('replay_clicked', {'id': id});
      await tap(play);
      final clock = Stopwatch()..start();
      final seconds = double.parse(env['VISIBLE_REPLAY_SECONDS']!);
      while (clock.elapsedMilliseconds < ((seconds + 2) * 1000).round()) {
        await tester.pump(const Duration(milliseconds: 200));
      }
      await event('replay_observation_complete');
      await screenshot('replay_complete');
    } else {
      throw StateError('Unknown mode');
    }
    await event('complete');
  }, timeout: const Timeout(Duration(minutes: 8)));
}
