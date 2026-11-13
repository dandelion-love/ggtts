#!/system/bin/sh
# armfeat-measure.sh: ggml built with its default ARM flags (base) against armv8.6-a+dotprod+i8mm+fp16 (armv86), for
# Kitten-TTS 2's Q4_0 and TQ1_0 LMs. Same S3Gen, voice, story, seed and 4 threads as tq1-measure.sh. ABBA order per
# model so warming reaches both builds. Each run's raw stream is hashed: the builds should give the same audio.
cd /data/local/tmp/ggtts
M=/sdcard/Android/data/ggtts.demo/files/models
[ -f tq1-story.txt ] || { echo "tq1-story.txt missing (tq1-measure.sh writes it)" >&2; exit 1; }
fail=0
for run in base-q4-1 armv86-q4-1 armv86-q4-2 base-q4-2 base-tq1-1 armv86-tq1-1 armv86-tq1-2 base-tq1-2; do
    build=${run%%-*}
    case $run in
        *-q4-*) lm=$M/kitten-lm.gguf ;;
        *-tq1-*) lm=$M/kitten-cantonese-qat-tq1_0.gguf ;;
    esac
    cp ggtts-render-$build ggtts-render
    echo "== $run thermal status $(dumpsys thermalservice 2>/dev/null | awk -F: '/Thermal Status/{print $2; exit}')"
    ./run-mem.sh af-$run --model $lm --codec $M/kitten-s3gen.gguf --voice $M/voice-yue.json \
        --story tq1-story.txt --threads 4 --stream af-$run.raw.wav --stream-raw \
        --timing-report af-$run.timing.json --out af-$run.flac || fail=1
    echo "$run raw $(sha256sum af-$run.raw.wav | cut -c1-16)"
    tail -1 af-$run.log
    sleep 60
done
[ $fail = 0 ] || { echo "armfeat-measure.sh: a run FAILED" >&2; exit 1; }
echo ARMFEAT_MEASURE_DONE
