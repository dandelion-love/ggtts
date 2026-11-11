#!/usr/bin/env bash
# Runs both identity checks and every planted defect's expected-failure test, one test per cargo invocation, and
# prints a record: what ran, on what, and each test's result lines and verdict. The committed record is
# records/identity-cpu.log.
#
#   GGTTS_TALKER=<talker.gguf> GGTTS_CODEC=<codec.gguf> scripts/identity_runs.sh > records/identity-cpu.log
#
# Optional: GGTTS_DEVICE and GGTTS_FIXTURES as the tests read them; FEATURES (default ggml,plant); NICE
# ("nice -n 19"); JOBS (4). Exit status 0 only when every identity check passes and every plant is seen.
set -uo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
root=$(cd "$here/../.." && pwd)
: "${GGTTS_TALKER:?}" "${GGTTS_CODEC:?}"
export GGTTS_TALKER GGTTS_CODEC
features=${FEATURES:-ggml,plant}
nice=${NICE:-nice -n 19}
jobs=${JOBS:-4}
cargo=(cargo test -j "$jobs" --release --locked -p ggtts-qwen3-tts --features "$features")

(cd "$root" && $nice "${cargo[@]}" --no-run) > /dev/null 2>&1 || { echo "build failed" >&2; exit 2; }

echo "# ggtts-qwen3-tts identity checks and planted defects"
echo "commit: $(git -C "$root" rev-parse HEAD)$(git -C "$root" diff --quiet HEAD -- crates || echo ' (with uncommitted changes)')"
echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "cpu: $(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -n 1)"
echo "rustc: $(rustc --version)"
echo "device: ${GGTTS_DEVICE:-cpu}; fixtures: ${GGTTS_FIXTURES:-fixtures/cpu (committed)}; features: $features"
echo "talker sha256: $(sha256sum "$GGTTS_TALKER" | cut -d' ' -f1)"
echo "codec sha256: $(sha256sum "$GGTTS_CODEC" | cut -d' ' -f1)"

status=0
for t in decoder_identity:decoder_samples_are_bit_identical_to_the_reference \
         decoder_identity:planted::rope_normal_is_seen \
         decoder_identity:planted::snake_no_epsilon_is_seen \
         decoder_identity:planted::window_off_by_one_is_seen \
         token_identity:codec_tokens_are_identical_to_the_reference \
         token_identity:planted::rope_normal_is_seen \
         token_identity:planted::sampler_off_by_one_is_seen \
         token_identity:instructed_codec_tokens_are_identical_to_the_reference \
         token_identity:an_empty_instruction_is_no_instruction \
         token_identity:an_instruction_changes_the_codes \
         token_identity:planted::instruction_after_role_is_seen; do
  file=${t%%:*} name=${t#*:}
  echo
  echo "## $file $name"
  log=$(mktemp)
  start=$(date +%s)
  if (cd "$root" && $nice "${cargo[@]}" --test "$file" -- --ignored --exact "$name" --nocapture) > "$log" 2>&1; then
    verdict=passed
  else
    verdict=FAILED
    status=1
  fi
  # Identity checks print IDENTICAL/DIFFERS per case; difference checks print ' frames with the instruction' per case.
  grep -E '^(IDENTICAL|DIFFERS) |identity \(| frames with the instruction, |^test |panicked|assertion' "$log" \
    | grep -v '^test result:'
  case $name in
    planted::*) meaning="the plant was seen" ;;
    # A test that asserts a difference, not an identity.
    an_instruction_changes_the_codes) meaning="the instruction changed the codes of every case" ;;
    *) meaning="identical to the reference" ;;
  esac
  [ $verdict = passed ] && echo "verdict: passed ($meaning), $(($(date +%s) - start)) s" \
    || echo "verdict: FAILED, $(($(date +%s) - start)) s"
  rm -f "$log"
done
echo
echo "exit: $status"
exit $status
