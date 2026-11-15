#!/usr/bin/env bash
# Makes a sample for each voice with the demo app itself, on a phone connected over adb, and keeps it beside the
# models as <voice name>.sample.wav, where the voice sheet's Sample button plays it. A sample made this way is what
# that voice sounds like from these exact files on this phone. Needs a debug build of the demo (it takes adb extras)
# and the model files already in its models directory.
#
#   scripts/android_voice_samples.sh                       # the default voices below
#   TEXT="Hello from this phone." scripts/android_voice_samples.sh
#
# Each line of VOICES is: <voice name> <engine> <model file> <codec file> <speaker or voice-JSON file>.
set -u
ADB=${ADB:-adb}
D=/sdcard/Android/data/ggtts.demo/files
TEXT=${TEXT:-你好！今日天氣幾好，不如一齊出去行吓街，食個下午茶。}
VOICES=${VOICES:-"hk_voice_343 qwen3-tts qwen-talker-hk-cantonese-cc0-v0.0.5-q8_0.gguf qwen-tokenizer-12hz-q8_0.gguf hk_voice_343
hk_voice_777 qwen3-tts qwen-talker-hk-cantonese-cc0-v0.0.5-q8_0.gguf qwen-tokenizer-12hz-q8_0.gguf hk_voice_777
voice-yue kitten-tts-2 kitten-cantonese-qat-tq1_0.gguf kitten-s3gen.gguf voice-yue.json
voice kitten-tts-2 kitten-cantonese-qat-tq1_0.gguf kitten-s3gen.gguf voice.json"}

log() { "$ADB" logcat -d -s ggtts-demo:V; }
status=0
# The list is read from descriptor 3: adb shell reads stdin, and would swallow the rest of the list.
while read -r -u 3 name engine model codec who; do
  [ -z "$name" ] && continue
  if "$ADB" shell ls "$D/models/$model" "$D/models/$codec" >/dev/null 2>&1; then :; else
    echo "$name: skipped, $model or $codec is not on the phone"; continue
  fi
  key=speaker; [ "$engine" = kitten-tts-2 ] && key=voice
  "$ADB" logcat -c
  speak() {
    "$ADB" shell am start -n ggtts.demo/.MainActivity --es engine "$engine" --es model "$model" --es codec "$codec" \
      --es "$key" "$who" --es text "'$TEXT'" --ez speak true >/dev/null 2>&1
  }
  speak
  done_=""; pressed_again=""
  for _ in $(seq 1 100); do
    sleep 3
    # The app warns once before a load when memory looks short, and a second Speak loads anyway.
    if [ -z "$pressed_again" ] && log | grep -q "shown: Low memory"; then
      pressed_again=1; "$ADB" logcat -c; speak; continue
    fi
    if log | grep -q "rendered $engine"; then
      "$ADB" shell cp "$D/out/$engine.wav" "$D/models/$name.sample.wav"
      echo "$name: $(log | grep -o "rendered $engine: [^,]*([^)]*) in [0-9]* ms")"
      done_=1; sleep 3; break
    fi
    if log | grep -q "shown: "; then echo "$name: failed: $(log | grep -o 'shown: .*' | tail -1)"; break; fi
  done
  [ -z "$done_" ] && status=1
done 3<<< "$VOICES"
"$ADB" shell ls "$D/models/" | grep '\.sample\.wav$'
exit $status
