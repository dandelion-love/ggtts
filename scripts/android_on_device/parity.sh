#!/system/bin/sh
# parity.sh SEED: the phone half of the parity runs, with the same flags as the desktop half.
cd /data/local/tmp/ggtts
M=/sdcard/Android/data/ggtts.demo/files/models
s=$1
common="--seed $s --threads 4 --trace-level 2 --stream-raw"
rm -f qwen-$s.* kitten-$s.*
./run.sh qwen-$s --model $M/qwen-talker-hk-cantonese-cc0-v0.0.5-q8_0.gguf --codec $M/qwen-tokenizer-12hz-q8_0.gguf \
    --speaker hk_voice_343 --story qwen-story.txt $common --trace qwen-$s.trace.jsonl --stream qwen-$s.raw.wav \
    --out qwen-$s.flac --timing-report qwen-$s.timing.json
./run.sh kitten-$s --model $M/kitten-lm.gguf --codec $M/kitten-s3gen.gguf --voice $M/voice.json \
    --story kitten-story.txt $common --trace kitten-$s.trace.jsonl --stream kitten-$s.raw.wav \
    --out kitten-$s.flac --timing-report kitten-$s.timing.json
echo PARITY_DONE
