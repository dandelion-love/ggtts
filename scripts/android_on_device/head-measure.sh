#!/system/bin/sh
# head-measure.sh: on the phone, Kitten-TTS 2's restricted output head against the whole head (GGTTS_KITTEN_FULL_HEAD=1),
# with the released TQ1_0 LM and the default ggml build, interleaved full, restricted, restricted, full. Then the
# per-op profile (op_profile, whole head) for both ggml builds and both LMs. Same S3Gen, voice, story and 4 threads
# as tq1-measure.sh.
cd /data/local/tmp/ggtts
M=/sdcard/Android/data/ggtts.demo/files/models
[ -f tq1-story.txt ] || { echo "tq1-story.txt missing" >&2; exit 1; }
tail -n +2 tq1-story.txt > op-lines.txt
cp ggtts-render-head ggtts-render
fail=0
for run in full-1 restricted-1 restricted-2 full-2; do
    if [ ${run%%-*} = full ]; then export GGTTS_KITTEN_FULL_HEAD=1; else unset GGTTS_KITTEN_FULL_HEAD; fi
    echo "== head $run thermal status $(dumpsys thermalservice 2>/dev/null | awk -F: '/Thermal Status/{print $2; exit}')"
    ./run-mem.sh hd-$run --model $M/kitten-cantonese-qat-tq1_0.gguf --codec $M/kitten-s3gen.gguf --voice $M/voice-yue.json \
        --story tq1-story.txt --threads 4 --stream hd-$run.raw.wav --stream-raw \
        --timing-report hd-$run.timing.json --out hd-$run.flac || fail=1
    echo "head $run raw $(sha256sum hd-$run.raw.wav | cut -c1-16)"
    tail -1 hd-$run.log
    sleep 45
done
export GGTTS_KITTEN_FULL_HEAD=1
for run in base-tq1 armv86-tq1 armv86-q4 base-q4; do
    build=${run%%-*}
    case $run in
        *-q4) lm=$M/kitten-lm.gguf ;;
        *-tq1) lm=$M/kitten-cantonese-qat-tq1_0.gguf ;;
    esac
    echo "== op $run thermal status $(dumpsys thermalservice 2>/dev/null | awk -F: '/Thermal Status/{print $2; exit}')"
    LD_LIBRARY_PATH=. ./op_profile-$build --model $lm --codec $M/kitten-s3gen.gguf --voice $M/voice-yue.json \
        --lines op-lines.txt --count 2 --threads 4 --json op-$run.json > op-$run.log 2>&1 || fail=1
    grep -v '^ggtts' op-$run.log | head -14
    sleep 45
done
[ $fail = 0 ] || { echo "head-measure.sh: a run FAILED" >&2; exit 1; }
echo HEAD_MEASURE_DONE
