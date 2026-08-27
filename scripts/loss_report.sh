#!/usr/bin/env bash
# Train and validation loss side by side for each run log matching RUNS.
set -uo pipefail

LOG_DIR=${LOG_DIR:-logs}
RUNS=${RUNS:-all swe_only tmax_only sol_v1}

for g in $RUNS; do
    log=$LOG_DIR/${g}_rank0.log
    [ -f "$log" ] || continue

    read -r step epoch tr < <(
        grep -oE 'step [0-9]+ \| epoch [0-9]+ \| loss [0-9.]+ \| grad_norm' "$log" |
            tail -1 | awk '{print $2, $5, $8}'
    )
    read -r vstep vloss < <(
        grep '\[val\]' "$log" | tail -1 |
            sed -nE 's/.*step ([0-9]+) \| epoch [0-9]+ \| loss ([0-9.]+).*/\1 \2/p'
    )
    best=$(grep '\[val\]' "$log" | sed -nE 's/.*loss ([0-9.]+).*/\1/p' | sort -g | head -1)
    total=$(grep -oE '[0-9]+/[0-9]+ \[' "$log" | tail -1 | tr -d ' [' | cut -d/ -f2)

    gap=$(awk -v a="${tr:-0}" -v b="${vloss:-0}" 'BEGIN{printf "%+.4f", a-b}')
    printf '%-10s step %-4s/%-4s epoch %-3s | train %-7s val %-7s gap %-8s bestval %s\n' \
        "$g" "${step:-?}" "${total:-?}" "${epoch:-?}" "${tr:-?}" "${vloss:-?}" "$gap" "${best:-?}"
done
