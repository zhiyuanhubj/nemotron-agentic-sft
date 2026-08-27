#!/usr/bin/env bash
# Choose checkpoints to evaluate for a training group.
#
# Best and runner-up alone are not informative here: validation is evaluated
# every two or three steps, so the top of the ranking is a run of adjacent
# steps whose losses differ by less than the noise. The third pick is
# therefore the best checkpoint from the last quarter of the run.
#
#   LOG=logs/sol_v1_rank0.log CKDIR=checkpoints/run_sol_v1 ./scripts/pick_checkpoints.sh
set -uo pipefail

LOG="${LOG:?set LOG to the rank0 training log}"
CKDIR="${CKDIR:?set CKDIR to the run checkpoint directory}"
[ -f "$LOG" ] || { echo "no log $LOG"; exit 1; }

VALS=$(grep '\[val\]' "$LOG" |
    sed -nE 's/.*step ([0-9]+) \| epoch ([0-9]+) \| loss ([0-9.]+).*/\1\t\2\t\3/p')
[ -n "$VALS" ] || { echo "no validation points in $LOG"; exit 1; }

HAVE=$(for d in "$CKDIR"/epoch_*; do
    [ -f "$d/model/adapter_config.json" ] || continue
    basename "$d" | sed -nE 's/^epoch_[0-9]+_step_([0-9]+)$/\1/p'
done | sort -n)
[ -n "$HAVE" ] || { echo "no saved adapters under $CKDIR"; exit 1; }
VALS=$(awk -F'\t' 'NR==FNR{keep[$1];next} $1 in keep' <(printf '%s\n' "$HAVE") <(printf '%s\n' "$VALS"))
[ -n "$VALS" ] || { echo "no validation point coincides with a saved adapter"; exit 1; }

LAST=$(printf '%s\n' "$VALS" | awk -F'\t' '{print $1}' | sort -n | tail -1)
CUT=$((LAST * 3 / 4))

pick() { awk -F'\t' -v lo="$1" -v hi="$2" '$1>=lo && $1<=hi' <<<"$VALS" | sort -t$'\t' -k3,3g; }

mapfile -t TOP < <(pick 0 "$LAST" | head -2)
LATE=$(pick $((CUT + 1)) "$LAST" | head -1)

late_step=${LATE%%$'\t'*}
if [ "$late_step" = "${TOP[0]%%$'\t'*}" ] || [ "$late_step" = "${TOP[1]%%$'\t'*}" ]; then
    LATE=$(awk -F'\t' -v s="$LAST" '$1==s' <<<"$VALS" | head -1)
    LATE_NOTE="(best already in the late window; using the final checkpoint)"
fi

emit() {
    [ -n "$1" ] || return 0
    local step epoch loss
    IFS=$'\t' read -r step epoch loss <<<"$1"
    printf '%-10s epoch_%s_step_%s  val=%s\n' "$2" "$epoch" "$step" "$loss"
}

echo "validation points $(printf '%s\n' "$VALS" | wc -l), last step $LAST, late window step>$CUT"
emit "${TOP[0]:-}" "best"
emit "${TOP[1]:-}" "runner-up"
emit "$LATE" "late"
[ -n "${LATE_NOTE:-}" ] && echo "           $LATE_NOTE"
