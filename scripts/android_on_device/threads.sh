#!/system/bin/sh
# threads.sh: the same story ("Parity 1") at 2, 4, 6 and 8 threads, both engines, for the real-time factor and the
# time each takes. Peak memory per run in NAME.mem.
cd /data/local/tmp/ggtts
M=/sdcard/Android/data/ggtts.demo/files/models
QA="--model $M/qwen-talker-hk-cantonese-cc0-v0.0.5-q8_0.gguf --codec $M/qwen-tokenizer-12hz-q8_0.gguf --speaker hk_voice_343"
KA="--model $M/kitten-lm.gguf --codec $M/kitten-s3gen.gguf --voice $M/voice.json"
for n in 2 4 6 8; do
    for e in kitten qwen; do
        if [ $e = qwen ]; then a=$QA; else a=$KA; fi
        rm -f th-$e-$n.trace.jsonl th-$e-$n.raw.wav th-$e-$n.flac th-$e-$n.timing.json th-$e-$n.log th-$e-$n.mem
        ./run.sh th-$e-$n $a --story story-$e-t1.txt --threads $n --trace th-$e-$n.trace.jsonl --trace-level 2 \
            --stream th-$e-$n.raw.wav --stream-raw --out th-$e-$n.flac --timing-report th-$e-$n.timing.json
    done
done
echo THREADS_DONE
