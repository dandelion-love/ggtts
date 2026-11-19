import 'package:flutter_test/flutter_test.dart';
import 'package:ggtts_app/read_options.dart';

void main() {
  test('model preparation is explicit and selected speaker reaches the load document', () {
    expect((loadOptions('/cache')['load'] as Map)['offline'], true);
    final load =
        loadOptions('/cache', offline: false, speaker: 'voice')['load'] as Map;
    expect(load['offline'], false);
    expect(load['speaker'], 'voice');
  });
  test('Android defaults to four threads and preserves explicit values', () {
    final defaultDoc = composeLoadDocument({}, '/cache', isAndroid: true);
    expect((defaultDoc['load'] as Map)['threads'], 4);
    final saved = <String, Object?>{
      'load': <String, Object?>{'threads': 7},
    };
    final explicit = composeLoadDocument(saved, '/cache', isAndroid: true);
    expect((explicit['load'] as Map)['threads'], 7);
    expect((saved['load'] as Map)['threads'], 7);
    final other = composeLoadDocument({}, '/cache', isAndroid: false);
    expect((other['load'] as Map).containsKey('threads'), isFalse);
  });

  test('render sends a shared streaming document and omits blank steering', () {
    expect(renderOptions()['output'], {'stream': 'raw'});
    expect(
      (renderOptions(steering: '  ')['render'] as Map).containsKey('steer'),
      false,
    );
    expect(
      (renderOptions(steering: ' preset.safetensors ')['render']
          as Map)['steer'],
      'preset.safetensors',
    );
  });
  test('controls depend on the actual schema', () {
    final schema = <String, Object?>{
      'properties': {
        'load': {
          'properties': {'speaker': {}},
        },
      },
    };
    expect(hasSchemaOption(schema, 'load', 'speaker'), true);
    expect(hasSchemaOption(schema, 'render', 'steer'), false);
    expect(hasSchemaOption({}, 'load', 'speaker'), false);
  });
}
