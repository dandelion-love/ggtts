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
