/// Build only schema-defined options for the shared C ABI.
Map<String, Object?> loadOptions(
  String cacheDir, {
  bool offline = true,
  String? speaker,
}) => {
  'load': {
    'offline': offline,
    'cache_dir': cacheDir,
    if (speaker != null && speaker.isNotEmpty) 'speaker': speaker,
  },
};

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
