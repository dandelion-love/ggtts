#!/usr/bin/env bash
# On-device MediaStore recovery validation. Requires an explicitly leased, idle device:
#   Build with -Pggtts.recoveryTest=true, then set ANDROID_HOME and ANDROID_SERIAL before running.
# It force-stops only the isolated ggtts.demo.recovery target between phases.
# It never installs, force-stops, or clears data for ggtts.demo; mismatched test APKs are refused.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
if [ -z "${ANDROID_HOME:-}" ]; then echo "set ANDROID_HOME" >&2; exit 2; fi
if [ -z "${ANDROID_SERIAL:-}" ]; then echo "set ANDROID_SERIAL to the leased device serial; no implicit device selection" >&2; exit 2; fi
adb="$ANDROID_HOME/platform-tools/adb"
aapt="$ANDROID_HOME/build-tools/35.0.0/aapt"
app="$root/android/demo/build/outputs/apk/debug/demo-debug.apk"
test_apk="$root/android/demo/build/outputs/apk/androidTest/debug/demo-debug-androidTest.apk"
target_package="ggtts.demo.recovery"
test_package="ggtts.demo.recovery.test"
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

assert_no_stale_test_state() {
    local package="$1" files
    if [ -z "$("$adb" -s "$ANDROID_SERIAL" shell pm path "$package" 2>/dev/null || true)" ]; then
        return 0
    fi
    case "$package" in
        "$target_package")
            files=$("$adb" -s "$ANDROID_SERIAL" shell run-as "$package" sh -c "if [ -d files ]; then find files -maxdepth 1 -type f \( -name 'queue-recovery-*.txt' -o -name 'queue-recovery-*.txt.part' \) -print; fi") || {
                echo "cannot inspect isolated recovery journals; refusing APK update for $package" >&2
                return 1
            }
            ;;
        "$test_package")
            files=$("$adb" -s "$ANDROID_SERIAL" shell run-as "$package" sh -c "if [ -d files ]; then find files -maxdepth 1 -type f -name 'queue-recovery-fixture-*.json*' -print; fi") || {
                echo "cannot inspect isolated fixture ledger; refusing APK update for $package" >&2
                return 1
            }
            ;;
        *)
            echo "refusing to inspect or update non-test package $package" >&2
            return 1
            ;;
    esac
    files=$(tr -d '\r' <<<"$files")
    if [ -n "$files" ]; then
        echo "refusing APK update: unresolved test state remains in $package: $files" >&2
        return 1
    fi
}

install_isolated_test_apk() {
    local apk="$1" package="$2" local_hash remote_paths remote_path remote_hash
    case "$package" in "$target_package"|"$test_package") ;; *) echo "refusing update outside isolated test packages: $package" >&2; return 1 ;; esac
    local_hash=$(sha256sum "$apk" | awk '{print $1}')
    remote_paths=$("$adb" -s "$ANDROID_SERIAL" shell pm path "$package" 2>/dev/null | sed -n 's/^package://p' | tr -d '\r' || true)
    if [ -z "$remote_paths" ]; then
        "$adb" -s "$ANDROID_SERIAL" install -t "$apk"
        return
    fi
    if [ "$(wc -l <<<"$remote_paths")" -ne 1 ]; then
        echo "refusing to replace multi-split installation of $package" >&2
        return 1
    fi
    remote_path=$(head -n 1 <<<"$remote_paths")
    remote_hash=$("$adb" -s "$ANDROID_SERIAL" shell sha256sum "$remote_path" | awk '{print $1}' | tr -d '\r')
    if [ "$remote_hash" = "$local_hash" ]; then
        echo "$package already has the exact APK; leaving it untouched"
        return 0
    fi
    echo "updating only isolated test package $package; app data is preserved"
    "$adb" -s "$ANDROID_SERIAL" install -r -t "$apk"
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

test -f "$app" || { echo "build the isolated debug app first with -Pggtts.recoveryTest=true: $app" >&2; exit 1; }
test -f "$test_apk" || { echo "build the isolated test APK first with -Pggtts.recoveryTest=true: $test_apk" >&2; exit 1; }
test -x "$aapt" || { echo "Android build-tools 35.0.0 aapt is required to verify package identities: $aapt" >&2; exit 1; }

app_id=$("$aapt" dump badging "$app" | sed -n "s/^package: name='\([^']*\)'.*/\1/p" | head -n 1)
test_id=$("$aapt" dump badging "$test_apk" | sed -n "s/^package: name='\([^']*\)'.*/\1/p" | head -n 1)
if [ "$app_id" != "$target_package" ]; then
    echo "refusing install: app APK package is '$app_id', expected '$target_package'" >&2
    exit 1
fi
if [ "$test_id" != "$test_package" ]; then
    echo "refusing install: test APK package is '$test_id', expected '$test_package'" >&2
    exit 1
fi
instrumentation_manifest=$("$aapt" dump xmltree "$test_apk" AndroidManifest.xml)
if ! grep -F "android:targetPackage(0x01010021)=\"$target_package\" " <<<"$instrumentation_manifest" >/dev/null; then
    echo "refusing install: instrumentation target is not '$target_package'" >&2
    exit 1
fi
if ! grep -F 'android:name(0x01010003)="androidx.test.runner.AndroidJUnitRunner"' <<<"$instrumentation_manifest" >/dev/null; then
    echo "refusing install: test APK does not declare the expected AndroidJUnitRunner" >&2
    exit 1
fi

services=$("$adb" -s "$ANDROID_SERIAL" shell dumpsys activity services "$target_package" 2>&1)
if grep -q "$target_package/.RenderService" <<<"$services"; then
    echo "refusing to continue while isolated recovery RenderService is active" >&2
    exit 1
fi
activities=$("$adb" -s "$ANDROID_SERIAL" shell dumpsys activity activities 2>&1)
if grep -E 'ResumedActivity|topResumedActivity|mResumedActivity' <<<"$activities" | grep -q "$target_package/"; then
    echo "refusing to continue while isolated recovery app is visible" >&2
    exit 1
fi

assert_no_stale_test_state "$target_package"
assert_no_stale_test_state "$test_package"
install_isolated_test_apk "$app" "$target_package"
install_isolated_test_apk "$test_apk" "$test_package"
trap on_exit EXIT
run_phase preparePublishedPendingMissingAndUnrelatedPending
"$adb" -s "$ANDROID_SERIAL" shell am force-stop "$target_package"
run_phase recoverAfterForceStopWithoutDuplicates
cleanup
trap - EXIT
