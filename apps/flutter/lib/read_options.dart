import 'dart:convert';
import 'dart:io';

/// Combine saved options with the explicit action's cache, offline and voice choices.
Map<String, Object?> composeLoadDocument(
  Map<String, Object?> saved,
  String cacheDir, {
  bool offline = true,
  String? speaker,
  bool? isAndroid,
}) {
  final doc = Map<String, Object?>.from(jsonDecode(jsonEncode(saved)) as Map);
  final load = Map<String, Object?>.from((doc['load'] as Map?) ?? {});
  load.putIfAbsent('cache_dir', () => cacheDir);
  if (isAndroid ?? Platform.isAndroid) {
    load.putIfAbsent('threads', () => 4);
  }
  load['offline'] = offline;
  if (speaker != null && speaker.isNotEmpty) load['speaker'] = speaker;
  doc['load'] = load;
  return doc;
}

Map<String, Object?> loadOptions(
  String cacheDir, {
  bool offline = true,
  String? speaker,
}) => composeLoadDocument({}, cacheDir, offline: offline, speaker: speaker);

Map<String, Object?> renderOptions({String? steering}) => {
  'render': {
    if (steering != null && steering.trim().isNotEmpty)
      'steer': steering.trim(),
  },
  'output': {'stream': 'raw'},
};

bool hasSchemaOption(Map<String, Object?> schema, String section, String name) {
  final sections = schema['properties'];
  if (sections is! Map) return false;
  final group = sections[section];
  if (group is! Map) return false;
  final properties = group['properties'];
  return properties is Map && properties.containsKey(name);
}
