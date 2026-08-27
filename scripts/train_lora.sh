#!/usr/bin/env bash
# LoRA SFT Nemotron-3-Ultra on 4 nodes × 8 GPUs via NeMo Automodel.
#
# Two ways to supply ranks:
#   HOLD_JOBS=id,id,id,id   four already-running 1-node allocations (overlap srun)
#   SLURM_JOB_ID set        a single 4-node allocation (this script is then a no-op
#                           wrapper; use scripts/train.sbatch instead)
#
# Required:
#   AUTOMODEL   clone of NVIDIA-NeMo/Automodel (with pretok_jsonl.py dropped in)
#   CONFIG      yaml under this repo, e.g. configs/run_sol_v1.yaml
#
# Optional:
#   NEMOTRON_MODEL  HF id or local path (default: nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16)
#   DATA_DIR        tokenized jsonl directory
#   CKPT_DIR        checkpoint parent directory
#   IMAGE           nvcr.io/nvidia/nemo-automodel:26.06.00
#   NNODES NPROC PORT RUN NCCL_SUBNET HF_HOME
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
AUTOMODEL=${AUTOMODEL:?set AUTOMODEL to the Automodel checkout}
CONFIG=${CONFIG:?set CONFIG, e.g. configs/run_sol_v1.yaml}
IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-automodel:26.06.00}
NEMOTRON_MODEL=${NEMOTRON_MODEL:-nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16}
DATA_DIR=${DATA_DIR:-$ROOT/data}
CKPT_DIR=${CKPT_DIR:-$ROOT/checkpoints}
NNODES=${NNODES:-4}
NPROC=${NPROC:-8}
PORT=${PORT:-29531}
RUN=${RUN:-sft}
NCCL_SUBNET=${NCCL_SUBNET:-10.1.}
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

if [ -z "${HOLD_JOBS:-}" ]; then
  echo "set HOLD_JOBS=jobid,jobid,jobid,jobid (four 1-node allocations)" >&2
  echo "or submit scripts/train.sbatch on a 4-node job" >&2
  exit 2
fi

IFS=',' read -r -a JOBS <<<"$HOLD_JOBS"
if [ "${#JOBS[@]}" -ne "$NNODES" ]; then
  echo "HOLD_JOBS has ${#JOBS[@]} ids, expected $NNODES" >&2
  exit 2
fi

export NEMOTRON_MODEL DATA_DIR CKPT_DIR
resolved=$ROOT/.resolved/${RUN}.yaml
mkdir -p "$ROOT/.resolved" "$ROOT/logs" "$CKPT_DIR"
if ! command -v envsubst >/dev/null; then
  echo "envsubst (gettext) is required to fill ${NEMOTRON_MODEL} / ${DATA_DIR} / ${CKPT_DIR}" >&2
  exit 2
fi
envsubst '${NEMOTRON_MODEL} ${DATA_DIR} ${CKPT_DIR}' <"$ROOT/$CONFIG" >"$resolved"

nodes=()
for job in "${JOBS[@]}"; do
  state=$(squeue -j "$job" -h -o '%T')
  node=$(squeue -j "$job" -h -o '%N')
  if [ "$state" != RUNNING ] || [ -z "$node" ] || [ "$node" = "(null)" ]; then
    echo "job $job is not RUNNING (state=${state:-empty} node=${node:-empty})" >&2
    exit 1
  fi
  nodes+=("$job:$node")
done

master_node=${nodes[0]##*:}
master_addr=$(sed 's/^ip-//; s/-/./g' <<<"$master_node")
echo "allocations ${nodes[*]}"
echo "master $master_addr resolved config $resolved"

for rank in "${!nodes[@]}"; do
  pair=${nodes[$rank]}
  job=${pair%%:*}
  node=${pair##*:}
  cpus=96
  [ "$rank" -eq 0 ] && cpus=94
  setsid srun --jobid="$job" --overlap -N1 -n1 -w "$node" \
    --cpus-per-task="$cpus" --mem=0 \
    bash -c "
docker rm -f automodel >/dev/null 2>&1 || true
NIC=\$(ip -o -4 addr show | awk '\$4 ~ /^${NCCL_SUBNET//./\\.}/ {print \$2; exit}')
if [ -z \"\$NIC\" ]; then echo 'no NIC matching ${NCCL_SUBNET}' >&2; exit 1; fi
docker run --rm --name automodel --gpus all \
  --ipc host --network host --shm-size 64g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --device=/dev/infiniband \
  -v $ROOT:$ROOT -v $AUTOMODEL:$AUTOMODEL \
  -e HF_HOME=$HF_HOME \
  -e HF_HUB_OFFLINE=\${HF_HUB_OFFLINE:-0} \
  -e TRANSFORMERS_OFFLINE=\${TRANSFORMERS_OFFLINE:-0} \
  -e MASTER_ADDR=$master_addr -e MASTER_PORT=$PORT \
  -e NCCL_SOCKET_IFNAME=\$NIC -e GLOO_SOCKET_IFNAME=\$NIC \
  -e FI_PROVIDER=\${FI_PROVIDER:-efa} -e FI_EFA_USE_HUGE_PAGE=0 \
  -e NCCL_DEBUG=WARN -e TOKENIZERS_PARALLELISM=false \
  -e PYTHONFAULTHANDLER=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint bash $IMAGE -c '
    cd $AUTOMODEL && exec torchrun \
      --nnodes=$NNODES --nproc-per-node=$NPROC --node-rank=$rank \
      --master_addr=$master_addr --master_port=$PORT \
      -m nemo_automodel.cli.app $resolved
  '
" >"$ROOT/logs/${RUN}_rank${rank}.log" 2>&1 </dev/null &
  echo "rank $rank -> $node (job $job)"
done

echo "started; logs at $ROOT/logs/${RUN}_rank*.log"
