# Windows null-backend rejection

From the repository root, with CMake and the Visual Studio C++ tools installed:

```powershell
cmake -S apps/flutter/vendor/mp_audio_stream/ggtts_tests/windows_null -B build/windows-null-test -A x64
cmake --build build/windows-null-test --config Release
ctest --test-dir build/windows-null-test -C Release --output-on-failure
```

The executable forces miniaudio to offer only its null backend. The production
Windows initialization path must refuse it and release the failed session. This
does not test physical output or establish which real backend is present.
