#!/system/bin/sh
# run.sh OUT ARGS...: run ggtts-render ARGS from this directory, logging to OUT.log, and record the process's peak
# resident memory (VmHWM, a high-water mark the kernel keeps) and exit code in OUT.mem.
cd /data/local/tmp/ggtts
out=$1
shift
LD_LIBRARY_PATH=. ./ggtts-render "$@" > "$out.log" 2>&1 &
pid=$!
hwm=0
while kill -0 $pid 2>/dev/null; do
    v=$(grep VmHWM /proc/$pid/status 2>/dev/null | awk '{print $2}')
    if [ -n "$v" ] && [ "$v" -gt "$hwm" ]; then hwm=$v; fi
    sleep 0.5
done
wait $pid
rc=$?
echo "rc=$rc vmhwm_kb=$hwm" > "$out.mem"
cat "$out.mem"
