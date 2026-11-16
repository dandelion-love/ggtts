import 'package:flutter_test/flutter_test.dart';
import 'package:ggtts_app/read_control_gate.dart';

void main() {
  test('stop is admitted once before terminal and never after it', () {
    final gate = ReadControlGate();
    expect(gate.requestStop(), isTrue);
    expect(gate.requestStop(), isFalse);
    gate.finish();
    expect(gate.requestStop(), isFalse);
  });

  test('terminal outcome closes stop admission before async cleanup', () {
    final gate = ReadControlGate();
    gate.finish();
    expect(gate.requestStop(), isFalse);
    expect(gate.stopRequested, isFalse);
  });
}
