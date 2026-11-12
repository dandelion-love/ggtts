# The ggtts binding: C-ABI and Kotlin

This page was the binding's design, written before the code. How to use the bindings is now in
[docs/bindings](../bindings/README.md):

* [ggtts from C](../bindings/c.md): the C ABI in `libggtts_ffi` and `ggtts.h`, its settings, codes, threading rules
  and versions.
* [ggtts from Kotlin and Android](../bindings/kotlin-android.md): the `ggtts.Ggtts` class over JNI, and how to build
  it into an app.

The header, [`crates/ggtts-ffi/include/ggtts.h`](../../crates/ggtts-ffi/include/ggtts.h), carries the same rules in
its comments. The design's history is in this file's git log.
