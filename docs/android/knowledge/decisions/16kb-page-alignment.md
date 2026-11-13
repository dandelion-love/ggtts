---
type: Decision
title: Native libraries are aligned for 16 KB pages
description: The Android builds link with -z max-page-size=16384, and the build refuses a library whose LOAD segments are aligned below 16 KB.
tags: [android, build]
status: draft
generated: { by: claude-code/opus-5.5, at: 2026-11-12T07:30:00-07:00 }
sources:
  - { id: android16k, resource: "https://developer.android.com/guide/practices/page-sizes", title: "Support 16 KB page sizes" }
---
# Decision

`scripts/android_build.sh` links `libggtts_ffi.so` and the Android `ggtts-render` with
`-C link-arg=-Wl,-z,max-page-size=16384`. Its `check_align` step refuses any library with a LOAD segment aligned
below 16 KB. A planted 4 KB-aligned library is refused on every build.

# Why

It was found on the device. The first install of the demo on a Galaxy Z Fold5 (Android 16) showed a compatibility
warning: `libggtts_ffi.so`'s LOAD segments were aligned to 0x1000. Android 15 and later flag such a library, and a
device with 16 KB pages cannot load it at all.[^android16k] The NDK's `libc++_shared.so` was already 16 KB-aligned.

After the fix and a clean reinstall, the package reports `pageSizeCompat=0`. The first install's result had been
stored: reinstalling over it with `-r` kept the warning.

# Undo

Remove the `RUSTFLAGS` line and the `check_align` calls from `scripts/android_build.sh`. The libraries then load on
4 KB-page devices only.
