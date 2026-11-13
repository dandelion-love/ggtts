#!/system/bin/sh
# run-mem.sh OUT ARGS...: like run.sh, and also a memory timeline. Every 0.25 s it writes "uptime_s VmRSS_kB VmHWM_kB
# RssAnon_kB RssFile_kB" to OUT.rss, so the peak can be placed in the run (load or render). Exits with ggtts-render's
# exit code.
cd /data/local/tmp/ggtts
out=$1
shift
LD_LIBRARY_PATH=. ./ggtts-render "$@" > "$out.log" 2>&1 &
pid=$!
: > "$out.rss"
hwm=0
while kill -0 $pid 2>/dev/null; do
    s=$(cat /proc/$pid/status 2>/dev/null)
    rss=$(echo "$s" | awk '/^VmRSS/{print $2}')
    h=$(echo "$s" | awk '/^VmHWM/{print $2}')
    an=$(echo "$s" | awk '/^RssAnon/{print $2}')
    fi=$(echo "$s" | awk '/^RssFile/{print $2}')
    if [ -n "$h" ]; then
        [ "$h" -gt "$hwm" ] && hwm=$h
        echo "$(cut -d' ' -f1 /proc/uptime) $rss $h $an $fi" >> "$out.rss"
    fi
    sleep 0.25
done
wait $pid
rc=$?
echo "rc=$rc vmhwm_kb=$hwm" > "$out.mem"
cat "$out.mem"
exit $rc
