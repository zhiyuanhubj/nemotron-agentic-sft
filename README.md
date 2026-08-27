# Nemotron-3-Ultra agentic SFT

Recipes for LoRA SFT of [NVIDIA Nemotron-3-Ultra 550B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16) on agent trajectories (SWE-rebench, TMax, and a Sol-thinking mix). Training uses [NeMo Automodel](https://github.com/NVIDIA-NeMo/Automodel) (`nvcr.io/nvidia/nemo-automodel:26.06.00`) on 4×8 H200 with FSDP2 + expert parallel 32 + context parallel 8.

This repo is the cluster-independent version of the pipeline: configs, launchers, pretokenization, and the small Automodel patches that made tool-call trajectories trainable. Weights, trajectories, and checkpoints are not included.

## What actually ran

| Run | Config | Data | Steps / epochs | Notes |
|---|---|---|---|---|
| V4 mix | `configs/run_all.yaml` | SWE-rebench + TMax | 688 / 16 | Main V4-teacher mix |
| SWE only | `configs/run_swe_only.yaml` | SWE-rebench | 352 / 16 | Ablation |
| TMax only | `configs/run_tmax_only.yaml` | TMax | 160 / 16 | Ablation |
| Sol v1 | `configs/run_sol_v1.yaml` | decoded GPT-5.6-sol thinking on the same task pool | 192 / 8 | Fresh run, do not resume from `run_all` |

Shared training knobs that mattered:

- LoRA rank/alpha 32, Triton LoRA **off** (SIGILL on H200)
- Exclude `*.out_proj` (Mamba kernels read that weight directly)
- MTP depth 2 with `mtp_use_repeated_layer: true`, loss scale 0.1
- Experts: grouped GEMM, dispatcher `torch`
- `use_triton: false` on LoRA; NCCL bound to the 10.1 fabric NIC, not docker/veth
- Pretokenized `input_ids`/`labels` rather than Automodel `ChatDataset` (see below)

Validated V4 teacher trajectories used for the mix are also published as [`zhiyuanhucs/agentic-sft-v4-teacher-v2`](https://huggingface.co/datasets/zhiyuanhucs/agentic-sft-v4-teacher-v2).

## Why pretokenize

Handing raw `messages` to Automodel's `ChatDataset` broke two things:

1. It serializes tool-call arguments to a JSON **string**. Nemotron's chat template iterates `arguments|items`, so Jinja raises `Can only get item pairs from a mapping` on every tool-using trajectory.
2. The only mask control is `start_of_turn_token`. That cannot train `reasoning_content` while still masking the scaffold-prefixed `<|im_start|>assistant` header.

`scripts/pretokenize.py` renders the Nemotron template by hand and writes aligned `input_ids`/`labels`:

- **masked:** system, user, tool observations, assistant turn header, empty `<think></think>`
- **trained:** reasoning, visible content, tool calls, `<|im_end|>`

Empty thinking is kept (Nemotron has a first-class empty think block). Steps with very short thinking plus a tool call, or a verbatim repeated action, are rendered as context but not trained.

`automodel_dropins/pretok_jsonl.py` is the corresponding dataset class. It **shifts** `labels` by one so the recipe does not predict token `i` from token `i`. Drop it into Automodel:

```bash
cp automodel_dropins/pretok_jsonl.py \
  $AUTOMODEL/nemo_automodel/components/datasets/llm/pretok_jsonl.py
```

If you skip pretokenization and go through `ChatDataset`, apply `patches/chat_dataset-tool-arguments.patch` so tool arguments stay a mapping.

## Layout

```
configs/                 Automodel YAML recipes (${NEMOTRON_MODEL}, ${DATA_DIR}, ${CKPT_DIR})
scripts/train_lora.sh    4-node launch over four single-node Slurm holds, or one allocation
scripts/train.sbatch     same recipe as a 4-node Slurm job
scripts/pretokenize.py   messages JSONL -> input_ids/labels
scripts/extract_sft_data.py
scripts/build_sol_sft_data.py
scripts/balance_sol_sft.py
automodel_dropins/       PretokenizedJsonl dataset class
patches/                 ChatDataset tool-argument fix
```

## Setup

```bash
git clone https://github.com/NVIDIA-NeMo/Automodel.git third_party/Automodel
git -C third_party/Automodel checkout 8cb12a3   # pin used for these runs
cp automodel_dropins/pretok_jsonl.py \
  third_party/Automodel/nemo_automodel/components/datasets/llm/pretok_jsonl.py

# optional, only if you train from messages instead of pretok JSONL
git -C third_party/Automodel apply ../../patches/chat_dataset-tool-arguments.patch
```

Pull the base model (`nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16`) and put tokenized JSONL under `$DATA_DIR`. A V4 mix can start from the HF dataset:

```bash
python scripts/pretokenize.py \
  --model nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 \
  --inp path/to/messages.jsonl \
  --out data/all_train_tok.jsonl
```

## Launch

Export paths, then either hold four 8-GPU jobs and overlap `srun`, or submit `scripts/train.sbatch`.

```bash
export AUTOMODEL=$PWD/third_party/Automodel
export NEMOTRON_MODEL=nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16
export DATA_DIR=$PWD/data
export CKPT_DIR=$PWD/checkpoints
export IMAGE=nvcr.io/nvidia/nemo-automodel:26.06.00

# Four already-running 1-node allocations:
HOLD_JOBS=1001,1002,1003,1004 \
CONFIG=configs/run_sol_v1.yaml \
RUN=sol_v1 \
  bash scripts/train_lora.sh
```

`train_lora.sh` resolves the 10.1.* NIC per node so NCCL does not bind to docker/veth. Override with `NCCL_SUBNET` if your fabric is different. `FI_PROVIDER=efa` is set for AWS EFA; drop it on non-EFA clusters.

Do not resume `run_sol_v1` from a `run_all` checkpoint: that run carried a scheduler stuck at `min_lr`.

## Data prep (teacher trajectories)

Harbor trial directories (`*/agent/mini-swe-agent.trajectory.json`) become message JSONL with:

```bash
srun --overlap ... python scripts/extract_sft_data.py \
  --root /path/to/harbor_root \
  --results-glob 'results/swerebench_v4_s*' \
  --out sft_data/swerebench.jsonl
```

The Sol mix additionally joins decoded Responses reasoning (`build_sol_sft_data.py`) and rebalances source weights to match the V4 token mix (`balance_sol_sft.py`). GPT-5.6-sol hidden reasoning is encrypted in the raw traces; the SFT `reasoning_content` is the decoded transcript, not the API plaintext.

## License

Apache-2.0. Automodel and several YAML headers are NVIDIA's; see `NOTICE`.
