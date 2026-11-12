#!/system/bin/sh
# threads_rev.sh: Qwen3-TTS at 8 threads, then 4, on the same story: the sweep's order reversed, to separate the
# thread count from the phone warming up over a sweep.
cd /data/local/tmp/ggtts
M=/sdcard/Android/data/ggtts.demo/files/models
QA="--model $M/qwen-talker-hk-cantonese-cc0-v0.0.5-q8_0.gguf --codec $M/qwen-tokenizer-12hz-q8_0.gguf --speaker hk_voice_343"
for n in 8 4; do
    rm -f rev-qwen-$n.*
    ./run.sh rev-qwen-$n $QA --story story-qwen-t1.txt --threads $n --stream rev-qwen-$n.raw.wav --stream-raw \
        --out rev-qwen-$n.flac --timing-report rev-qwen-$n.timing.json
    grep -o '"render_rtf":[0-9.]*' rev-qwen-$n.timing.json
done
sha256sum th-qwen-*.raw.wav rev-qwen-*.raw.wav th-kitten-*.raw.wav | cut -c1-16,65-
echo REV_DONE
