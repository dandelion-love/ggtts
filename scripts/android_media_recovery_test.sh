#!/usr/bin/env bash
# On-device MediaStore recovery validation. Requires an explicitly leased, idle device:
#   ANDROID_HOME=... ANDROID_SERIAL=<serial> scripts/android_media_recovery_test.sh
# It force-stops ggtts.demo between phases. Close ggtts first; do not run during a read or queue job.
# It never clears app data and installs only missing packages; a different installed APK is refused.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
if [ -z "${ANDROID_HOME:-}" ]; then echo "set ANDROID_HOME" >&2; exit 2; fi
if [ -z "${ANDROID_SERIAL:-}" ]; then echo "set ANDROID_SERIAL to the leased device serial; no implicit device selection" >&2; exit 2; fi
adb="$ANDROID_HOME/platform-tools/adb"
app="$root/android/demo/build/outputs/apk/debug/demo-debug.apk"
test_apk="$root/android/demo/build/outputs/apk/androidTest/debug/demo-debug-androidTest.apk"
test_package="ggtts.demo.recoverytest"
component="$test_package/androidx.test.runner.AndroidJUnitRunner"
class="ggtts.demo.MediaStoreRecoveryInstrumentationTest"
fixture_id=$(tr -d '-' < /proc/sys/kernel/random/uuid)
echo "GGTTS recovery fixture ID: $fixture_id"

run_phase() {
    local method="$1" output
    output=$("$adb" -s "$ANDROID_SERIAL" shell am instrument -w -e fixture "$fixture_id" -e class "$class#$method" "$component" 2>&1) || {
        printf '%s
' "$output" >&2
        return 1
    }
    printf '%s
' "$output"
    if grep -Eq 'FAILURES!!!|INSTRUMENTATION_FAILED|shortMsg=' <<<"$output"; then return 1; fi
    if ! grep -Eq 'OK \([0-9]+ test[s]?\)' <<<"$output"; then
        echo "instrumentation did not report a passing test" >&2
        return 1
    fi
}

install_if_absent_or_identical() {
    local apk="$1" package="$2" local_hash remote_paths remote_path remote_hash
    local_hash=$(sha256sum "$apk" | awk '{print $1}')
    remote_paths=$("$adb" -s "$ANDROID_SERIAL" shell pm path "$package" 2>/dev/null | sed -n 's/^package://p' | tr -d '
' || true)
    if [ -z "$remote_paths" ]; then
        "$adb" -s "$ANDROID_SERIAL" install "$apk"
        return
    fi
    if [ "$(wc -l <<<"$remote_paths")" -ne 1 ]; then
        echo "refusing to replace multi-split installation of $package" >&2
        return 1
    fi
    remote_path=$(head -n 1 <<<"$remote_paths")
    remote_hash=$("$adb" -s "$ANDROID_SERIAL" shell sha256sum "$remote_path" | awk '{print $1}' | tr -d '
')
    if [ "$remote_hash" != "$local_hash" ]; then
        echo "refusing to replace installed $package because its APK differs" >&2
        return 1
    fi
    echo "$package already has the exact APK; leaving it untouched"
}

cleanup() {
    local status=0
    run_phase cleanupExactRowsAndJournals || status=$?
    return "$status"
}
on_exit() {
    local original=$?
    trap - EXIT
    if ! cleanup; then
        echo "test cleanup failed; preserve the printed fixture ID and retry cleanup manually" >&2
        exit 1
    fi
    exit "$original"
}
trap on_exit EXIT

test -f "$app" || { echo "build the debug app first: $app" >&2; exit 1; }
test -f "$test_apk" || { echo "build the test APK first: $test_apk" >&2; exit 1; }

services=$("$adb" -s "$ANDROID_SERIAL" shell dumpsys activity services ggtts.demo 2>&1)
if grep -q 'ggtts.demo/.RenderService' <<<"$services"; then
    echo "refusing to continue while ggtts RenderService is active" >&2
    exit 1
fi
activities=$("$adb" -s "$ANDROID_SERIAL" shell dumpsys activity activities 2>&1)
if grep -E 'ResumedActivity|topResumedActivity|mResumedActivity' <<<"$activities" | grep -q 'ggtts.demo/'; then
    echo "refusing to continue while ggtts is visible" >&2
    exit 1
fi

install_if_absent_or_identical "$app" ggtts.demo
install_if_absent_or_identical "$test_apk" "$test_package"
run_phase preparePublishedPendingMissingAndUnrelatedPending
"$adb" -s "$ANDROID_SERIAL" shell am force-stop ggtts.demo
run_phase recoverAfterForceStopWithoutDuplicates
cleanup
trap - EXIT
