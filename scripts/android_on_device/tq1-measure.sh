#!/system/bin/sh
# tq1-measure.sh: Kitten-TTS 2 on the phone, the base Q4_0 LM against the released Cantonese QAT TQ1_0 LM. Same
# S3Gen, voice, story, seed and 4 threads; alternating Q4_0, TQ1_0, Q4_0, TQ1_0 so warming affects both; a raw stream,
# so first audio is the first chunk. Each run records its memory timeline (run-mem.sh) and the thermal status.
cd /data/local/tmp/ggtts
M=/sdcard/Android/data/ggtts.demo/files/models
cat > tq1-story.txt <<'EOF'
測試
今日天氣好好，我哋去公園行下啦。
公園有好多人，有啲喺度跑步，有啲坐低食嘢。
EOF
fail=0
for run in q4-1 tq1-1 q4-2 tq1-2; do
    case $run in
        q4-*) lm=$M/kitten-lm.gguf ;;
        tq1-*) lm=$M/kitten-cantonese-qat-tq1_0.gguf ;;
    esac
    echo "== $run thermal status $(dumpsys thermalservice 2>/dev/null | awk -F: '/Thermal Status/{print $2; exit}')"
    ./run-mem.sh tq1m-$run --model $lm --codec $M/kitten-s3gen.gguf --voice $M/voice-yue.json \
        --story tq1-story.txt --threads 4 --stream tq1m-$run.raw.wav --stream-raw \
        --timing-report tq1m-$run.timing.json --out tq1m-$run.flac || fail=1
    tail -2 tq1m-$run.log
    sleep 60
done
[ $fail = 0 ] || { echo "tq1-measure.sh: a run FAILED" >&2; exit 1; }
echo TQ1_MEASURE_DONE
