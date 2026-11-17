import 'dart:io';

import 'package:flutter/services.dart';

abstract interface class SharedTextInbox {
  Future<String?> take();
  void onAvailable(void Function()? callback);
  void close();
}

class AndroidSharedTextInbox implements SharedTextInbox {
  static const _channel = MethodChannel('ggtts/shared_text');
  void Function()? _callback;

  @override
  void onAvailable(void Function()? callback) {
    _callback = callback;
    _channel.setMethodCallHandler((call) async {
      if (call.method == 'available') _callback?.call();
    });
  }

  @override
  Future<String?> take() => _channel.invokeMethod<String>('take');

  @override
  void close() {
    _callback = null;
    _channel.setMethodCallHandler(null);
  }
}

SharedTextInbox? platformSharedTextInbox() =>
    Platform.isAndroid ? AndroidSharedTextInbox() : null;
