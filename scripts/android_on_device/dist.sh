#!/system/bin/sh
# dist.sh N: the phone half of the distribution, titles "Parity 1".."Parity N", and a repeat of title "Parity" to
# show whether the phone repeats itself. Same flags as the desktop half. Stops at the first failed render.
set -e
cd /data/local/tmp/ggtts
M=/sdcard/Android/data/ggtts.demo/files/models
QA="--model $M/qwen-talker-hk-cantonese-cc0-v0.0.5-q8_0.gguf --codec $M/qwen-tokenizer-12hz-q8_0.gguf --speaker hk_voice_343"
KA="--model $M/kitten-lm.gguf --codec $M/kitten-s3gen.gguf --voice $M/voice.json"
one() { # NAME STORY ENGINE-ARGS...
    n=$1; st=$2; shift 2
    rm -f $n.trace.jsonl $n.raw.wav $n.flac $n.timing.json $n.log $n.mem
    ./run.sh $n "$@" --story $st --threads 4 --trace $n.trace.jsonl --trace-level 2 --stream $n.raw.wav --stream-raw \
        --out $n.flac --timing-report $n.timing.json
}
one qwen-rep qwen-story.txt $QA
one kitten-rep kitten-story.txt $KA
i=1
while [ $i -le "$1" ]; do
    { echo "Parity $i"; tail -n +2 qwen-story.txt; } > story-qwen-t$i.txt
    { echo "Parity $i"; tail -n +2 kitten-story.txt; } > story-kitten-t$i.txt
    one qwen-t$i story-qwen-t$i.txt $QA
    one kitten-t$i story-kitten-t$i.txt $KA
    i=$((i + 1))
done
echo DIST_DONE
