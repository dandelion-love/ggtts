#!/system/bin/sh
# thermal.sh SECONDS OUT: every 10 s for SECONDS, append one line to OUT: elapsed seconds, the Android thermal status,
# the thermal HAL's AP, SKIN and BAT temperatures (C), and the battery service's temperature (C). Reads only.
#
# A sample with any field missing stops the run with exit 1 and names the field: an empty column must never pass for
# a measurement (the first budget run lost its temperatures that way).
end=$1
out=$2
t=0
: > "$out"
while [ $t -le "$end" ]; do
    d=$(dumpsys thermalservice)
    st=$(echo "$d" | awk '/^Thermal Status:/ { print $3; exit }')
    temps=$(echo "$d" | awk '/Current temperatures from HAL/ { on = 1; next } on && !/Temperature/ { exit }
        on { match($0, /mValue=[0-9.]+/); v = substr($0, RSTART + 7, RLENGTH - 7)
             match($0, /mName=[A-Z]+/); n = substr($0, RSTART + 6, RLENGTH - 6)
             if (n == "AP" || n == "SKIN" || n == "BAT") printf "%s=%s ", n, v }')
    bat=$(dumpsys battery | awk '/^  temperature:/ { printf "%.1f", $2 / 10; exit }')
    missing=""
    [ -n "$st" ] || missing="$missing status"
    for n in AP SKIN BAT; do
        case "$temps" in *"$n="[0-9]*) ;; *) missing="$missing $n" ;; esac
    done
    [ -n "$bat" ] || missing="$missing battery"
    if [ -n "$missing" ]; then
        echo "t=$t ERROR missing:$missing" >> "$out"
        echo "thermal.sh: at t=$t s these fields could not be read:$missing" >&2
        exit 1
    fi
    echo "t=$t status=$st ${temps}battery=$bat" >> "$out"
    sleep 10
    t=$((t + 10))
done
