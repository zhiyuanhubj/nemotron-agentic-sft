#!/bin/bash
#SBATCH --job-name=nemotron-sft
#SBATCH --nodes=4
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --exclusive
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#
# Submit from the repo root after exporting AUTOMODEL, DATA_DIR, CKPT_DIR.
# CONFIG defaults to configs/run_sol_v1.yaml.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
AUTOMODEL=${AUTOMODEL:?set AUTOMODEL}
CONFIG=${CONFIG:-configs/run_sol_v1.yaml}
IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-automodel:26.06.00}
NEMOTRON_MODEL=${NEMOTRON_MODEL:-nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16}
DATA_DIR=${DATA_DIR:-$ROOT/data}
CKPT_DIR=${CKPT_DIR:-$ROOT/checkpoints}
PORT=${PORT:-29531}
NCCL_SUBNET=${NCCL_SUBNET:-10.1.}
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

mkdir -p logs "$CKPT_DIR" .resolved
export NEMOTRON_MODEL DATA_DIR CKPT_DIR
resolved=$ROOT/.resolved/${SLURM_JOB_ID}.yaml
envsubst '${NEMOTRON_MODEL} ${DATA_DIR} ${CKPT_DIR}' <"$ROOT/$CONFIG" >"$resolved"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_ADDR=${MASTER_ADDR/#ip-/}
MASTER_ADDR=${MASTER_ADDR//-/.}

srun --ntasks-per-node=1 bash -c "
NIC=\$(ip -o -4 addr show | awk '\$4 ~ /^${NCCL_SUBNET//./\\.}/ {print \$2; exit}')
docker rm -f automodel >/dev/null 2>&1 || true
docker run --rm --name automodel --gpus all \
  --ipc host --network host --shm-size 64g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --device=/dev/infiniband \
  -v $ROOT:$ROOT -v $AUTOMODEL:$AUTOMODEL \
  -e HF_HOME=$HF_HOME \
  -e MASTER_ADDR=$MASTER_ADDR -e MASTER_PORT=$PORT \
  -e NCCL_SOCKET_IFNAME=\$NIC -e GLOO_SOCKET_IFNAME=\$NIC \
  -e FI_PROVIDER=\${FI_PROVIDER:-efa} -e FI_EFA_USE_HUGE_PAGE=0 \
  -e NCCL_DEBUG=WARN -e TOKENIZERS_PARALLELISM=false \
  -e PYTHONFAULTHANDLER=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint bash $IMAGE -c '
    cd $AUTOMODEL && exec torchrun \
      --nnodes=$SLURM_NNODES --nproc-per-node=8 \
      --node-rank=\$SLURM_NODEID \
      --master_addr=$MASTER_ADDR --master_port=$PORT \
      -m nemo_automodel.cli.app $resolved
  '
"
