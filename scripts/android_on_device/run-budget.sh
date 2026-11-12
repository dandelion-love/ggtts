#!/system/bin/sh
# run-budget.sh: the resource budget's run (its name must not match budget.*, which it clears). Kitten-TTS 2 (Q4_0) reads a six-paragraph story at the binding's default four
# threads, streaming as a listener would hear it, while thermal.sh samples the thermal status and temperatures.
cd /data/local/tmp/ggtts
M=/sdcard/Android/data/ggtts.demo/files/models
cat > budget-story.txt <<'EOF'
Budget
The morning bus was late again, so we walked along the river instead.
A small boat drifted past, and someone on board waved at us.
By the bridge, an old man was selling roasted chestnuts from a cart.
We bought a bag and shared them while the sun climbed over the hills.
At the station, the board said every train was running twenty minutes behind.
So we sat on a bench, finished the chestnuts, and watched the pigeons argue.
EOF
rm -f budget.* thermal-budget.txt
./thermal.sh 330 thermal-budget.txt &
tpid=$!
./run.sh budget --model $M/kitten-lm.gguf --codec $M/kitten-s3gen.gguf --voice $M/voice.json \
    --story budget-story.txt --threads 4 --trace budget.trace.jsonl --stream budget.stream.wav \
    --out budget.flac --timing-report budget.timing.json
render=$?
wait $tpid
thermal=$?
cat thermal-budget.txt
if [ $render -ne 0 ] || [ $thermal -ne 0 ]; then
    echo "run-budget.sh: FAILED (render exit $render, thermal exit $thermal)" >&2
    exit 1
fi
echo BUDGET_DONE
