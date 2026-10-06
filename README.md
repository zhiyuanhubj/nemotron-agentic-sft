# Nemotron-3.5-Super Agentic SFT

Data preparation and full-parameter supervised fine-tuning for Nemotron-3.5-Super using NVIDIA NeMo Automodel. The pipeline accepts text conversations with reasoning and tool calls, applies the model's native chat template, and trains on assistant targets only.

The training entry point supports a configurable number of nodes and GPUs through `torchrun`. Model, dataset, output, and checkpoint paths are supplied by the user. No cluster scheduler, pre-existing job allocations, benchmark metadata, or trajectory review service is required.

## Features

- Messages JSONL input, including `reasoning_content`, tool calls, and tool observations.
- Deterministic validation splits by task ID, or a separate validation file.
- Native-template tokenization with assistant-only loss masking.
- Rolling windows for long trajectories, with each original target supervised once.
- Memory-mapped token arrays and a single causal label shift.
- Full-parameter language-model and embedding training with FSDP2; configurable context and expert parallelism.
- Validation loss, early stopping, assistant-token accounting, and resumable training state.
- Standard filesystem checkpoints by default; optional node-local checkpoint storage.

The vision and audio towers are frozen for this text/tool training recipe. No LoRA or other parameter-efficient adapter is configured.

## Requirements

- Linux, NVIDIA GPUs, and a compatible CUDA/PyTorch environment.
- Python 3.12 and the pinned Automodel checkout below, with this repository's compatibility patch.
- A local model directory containing weights, the safetensors index, tokenizer/chat template, processor configuration, and any model-provided Python files.
- Enough GPU memory for the model parameters, gradients, optimizer state, and activations across the selected devices. A configurable topology does not imply that a large model fits on one GPU.
- For multi-node training: code, environment, model, prepared data, run state, and checkpoint storage available at the same absolute paths on each host, plus a reachable rendezvous address.

The reference environment uses PyTorch `2.10.0+cu130` and CUDA 13.0. Supporting package versions are listed in [super/requirements-reference.txt](super/requirements-reference.txt). These are a compatibility reference; GPU dependencies should be installed through Automodel for your platform.

## Installation

```bash
git clone --branch nemotron-3.5-super-20261006 \
  https://github.com/zhiyuanhubj/nemotron-agentic-sft.git
cd nemotron-agentic-sft

git clone https://github.com/NVIDIA-NeMo/Automodel.git third_party/Automodel
git -C third_party/Automodel checkout 8cb12a35b65eda421d8f1fa485c658eb06d22c76
git -C third_party/Automodel apply --check ../../patches/nemotron35-super-automodel.patch
git -C third_party/Automodel apply ../../patches/nemotron35-super-automodel.patch

(cd third_party/Automodel && uv sync --extra vlm --extra cuda)
export AUTOMODEL="$PWD/third_party/Automodel"
export TRAIN_PYTHON="$AUTOMODEL/.venv/bin/python"
```

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if it is not already available. Verify the resolved PyTorch wheel and CUDA runtime before training. The patch adds Super architecture support, RADIO weight mappings, FSDP2/MTP tensor compatibility, and a checkpoint consolidation timeout option.

Model weights and model-provided remote code are not bundled with this repository. Both preprocessing and training use a local model directory. Preprocessing requires a fast tokenizer with character offsets.

## Input data

Each line is a conversation with a nonempty string `task_id` and a `messages` array:

```json
{"task_id":"task-001","messages":[{"role":"system","content":"You are a helpful coding assistant."},{"role":"user","content":"Explain what git status does."},{"role":"assistant","reasoning_content":"The user needs a brief description of working-tree state.","content":"git status shows staged, unstaged, and untracked changes."}]}
```

An initial system message is optional. Supported roles are `system`, `user`, `assistant`, and `tool`. Message content is text. Assistant messages may include `reasoning_content` and OpenAI-style `tool_calls`; tool messages may include `tool_call_id` and `name`. Function arguments can be dictionaries or JSON-encoded strings. Harness `exit` messages are ignored.

Use the same task ID for every trajectory belonging to the same underlying task. Automatic splitting keeps all trajectories for a task in one split. Explicit validation input must have no task IDs or identical conversations in common with training input. Exact duplicate conversations are rejected.

Small format examples are provided in [examples/train.jsonl](examples/train.jsonl) and [examples/validation.jsonl](examples/validation.jsonl). They illustrate the schema and are not a training dataset.

## Prepare the dataset

```bash
export MODEL_DIR=/path/to/nemotron-model
export PREPARED_DIR="$PWD/prepared/my_dataset"

"$TRAIN_PYTHON" super/prepare_data.py \
  --model "$MODEL_DIR" \
  --train /path/to/train.jsonl \
  --validation /path/to/validation.jsonl \
  --output "$PREPARED_DIR" \
  --max-length 8192
```

Omit `--validation` to select a deterministic held-out fraction of tasks using `--validation-fraction` (default `0.1`) and `--seed` (default `42`). Automatic splitting requires at least two tasks. Plain JSONL and gzip-compressed JSONL are supported. Use a new or empty output directory.

The processor renders the complete native template and tokenizes it once, checking equality with `apply_chat_template(tokenize=True)`. System/user text, tool observations, assistant headers, and opening or empty reasoning tags are masked with `-100`. Assistant reasoning, responses, tool calls, and end-of-message tokens remain targets.

Overlength trajectories are split into rolling windows. Windows preserve the initial task prefix and recent context, prefer assistant-round boundaries, and fall back to token boundaries for oversized rounds. Repeated context is masked. A task prefix occupying half the configured window or more is rejected; increase `--max-length` for that input. `--context-tokens` controls the recent-context budget, which is capped further to reserve room for new targets.

Output:

```text
prepared/my_dataset/
  train/                  input_ids.npy, labels.npy, offsets.npy, manifest.jsonl
  validation/             input_ids.npy, labels.npy, offsets.npy, manifest.jsonl
  summary.json            source hashes, split statistics, preprocessing settings
  READY.json              verified dataset marker
```

Arrays are stored without a label shift. The collator applies `input_ids[:, :-1]` and `labels[:, 1:]` once and masks padding. Data verification checks hashes, array structure, target counts, and task separation before training. Tokenization assets are fingerprinted by content, so prepared data can be moved to another machine using an identical tokenizer.

## Configure training

The baseline in [super/configs/full_sft.yaml](super/configs/full_sft.yaml) uses BF16 weights, FSDP2, activation checkpointing, AdamW, a cosine schedule, and three epochs. Context and expert parallelism default to one. Hyperparameters are editable in the template before generating a run configuration.

This example uses one host with four GPUs; replace the worker count and parallelism settings with values appropriate for your hardware:

```bash
export RUN_DIR="$PWD/runs/my_sft"

"$TRAIN_PYTHON" super/configure.py \
  --model "$MODEL_DIR" \
  --prepared-dir "$PREPARED_DIR" \
  --run-dir "$RUN_DIR" \
  --run-name my_sft \
  --nnodes 1 --nproc-per-node 4 \
  --cp-size 1 --ep-size 1 \
  --global-batch-size 8 --local-batch-size 1 \
  --epochs 3 --learning-rate 1e-5
```

There is no fixed node count or GPU count per node. Use `--nproc-per-node 1` for a single GPU when the model and training state fit. Increase the number of devices or adjust sequence length and parallelism for larger workloads.

`cp_size` and `ep_size` must divide the total worker count; expert parallelism must also divide the model's routed-expert count. With tensor and pipeline parallelism fixed at one, data parallel size is `world_size / cp_size`. Global batch size must be a multiple of `local_batch_size * dp_size`; the remaining factor sets gradient accumulation. The configuration generator validates these constraints.

The collator's sequence limit is taken from the prepared dataset unless `--max-length` specifies a larger value. To lower it, reprocess the dataset so supervised tokens are preserved.

By default, checkpoints are written to `RUN_DIR/checkpoints`, and W&B logging is disabled. Override the checkpoint path with `--checkpoint-dir`. Enable W&B with `--wandb` and authenticate through your usual W&B environment or login; `WANDB_KEY_FILE` is also supported. Credentials are never embedded in generated YAML.

## Launch training

```bash
# Validate the configuration and inspect the torchrun command.
"$TRAIN_PYTHON" super/launch.py \
  --run-dir "$RUN_DIR" --automodel "$AUTOMODEL" --dry-run

# Start training on the current host.
"$TRAIN_PYTHON" super/launch.py \
  --run-dir "$RUN_DIR" --automodel "$AUTOMODEL"
```

The launcher uses the Python interpreter that runs it and waits for `torchrun` to finish. Logs stream to the terminal; redirect them or use your scheduler's log capture if needed. Per-host startup records and training monitor state are saved under `RUN_DIR/state`.

### Multiple hosts

Generate the configuration with your chosen `--nnodes` and `--nproc-per-node`. Run the launcher once on each allocated host, using a unique zero-based node rank and the same master address and port:

```bash
# Set NODE_RANK separately on each host: 0, 1, ..., nnodes - 1.
export NODE_RANK=0
export MASTER_ADDR=training-host-0
export MASTER_PORT=29500

"$TRAIN_PYTHON" super/launch.py \
  --run-dir "$RUN_DIR" --automodel "$AUTOMODEL" \
  --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT"
```

Your scheduler allocates resources and starts one launcher per host. The launcher does not select nodes or inspect allocation lifetimes. Configure NCCL interfaces and transport libraries through the environment for your network.

## Checkpoints and resume

Automodel saves the model, optimizer, learning-rate scheduler, RNG, dataloader, and step scheduler. The additional hooks save early-stopping history and token accounting alongside each synchronous checkpoint. Default retention keeps two recent checkpoints and checkpoints referenced by Automodel's latest/best pointers.

To resume, stop the previous workers, archive `RUN_DIR/train.yaml`, and regenerate the configuration with `--restore-from` pointing to a completed checkpoint in the same checkpoint directory. Reuse the original model, data, node/worker counts, context/expert parallelism, batch sizes, and hook settings:

```bash
mv "$RUN_DIR/train.yaml" "$RUN_DIR/train.previous.yaml"

"$TRAIN_PYTHON" super/configure.py \
  --model "$MODEL_DIR" --prepared-dir "$PREPARED_DIR" \
  --run-dir "$RUN_DIR" --run-name my_sft \
  --nnodes 1 --nproc-per-node 4 --cp-size 1 --ep-size 1 \
  --global-batch-size 8 --local-batch-size 1 \
  --epochs 3 --learning-rate 1e-5 \
  --restore-from "$RUN_DIR/checkpoints/epoch_0_step_100"
```

Then launch with the same commands as above. Model weights alone are insufficient for full-state resume. Resume is limited to the original topology and data; changing those requires a new run.

Early stopping uses sustained validation-loss degradation together with improving training loss. The monitor state is restored on resume. Validation loss measures prediction on held-out trajectories; evaluate agent task success separately.

Token accounting defaults to `track_only` and counts actual assistant targets consumed by optimizer updates, including sampler repetitions. Context-parallel replicas are not counted twice. Set `token_budget.mode: exact` and `token_budget.supervised_tokens` in the template to stop at an exact target-token budget. Set `enabled: false` to disable accounting.

### Optional node-local storage

For installations with node-local disks and shared backup storage, `configure.py --nvme-root /path/to/local/storage` enables the optional node-local checkpoint hook. Stage the model and prepared data on **every** host at the `local_model` and `local_data` paths recorded in `RUN_DIR/state/READY` before launching. The code does not automatically copy these inputs or require NVMe in the default workflow.

Each host retains the union of its latest two and best three shard sets. A better checkpoint is assembled and verified in the shared checkpoint directory before the previous complete backup is removed. This mode requires `rsync` and sufficient local/shared disk capacity. Verify a backup with:

```bash
"$TRAIN_PYTHON" super/verify_checkpoint.py "$RUN_DIR/checkpoints/BEST"
```

## Repository layout

```text
examples/                   minimal messages JSONL examples
super/
  prepare_data.py           public data-preparation CLI
  prepare_native.py         native-template tokenization and masking
  rolling_windows.py        long-trajectory segmentation
  agent_sft_data.py          memory-mapped dataset and collator
  data_gate.py              dataset integrity verification
  configure.py              resolved configuration generation
  topology.py               parallelism and batch validation
  launch.py                 single-host and multi-host torchrun launcher
  run_training.py           training entry point
  checkpoint_state.py       checkpoint persistence for optional hooks
  early_stopping.py         validation monitor
  token_budget.py           assistant-target accounting and exact budgets
  nvme_checkpoint.py        optional node-local checkpoint backup
  verify_checkpoint.py      node-local backup verification
  configs/full_sft.yaml      editable full-parameter SFT baseline
  tests/                    CPU pipeline and distributed-accounting checks
patches/                    pinned Automodel compatibility patch
```

## Verification

```bash
"$TRAIN_PYTHON" -m unittest discover -s super/tests -p 'test_*.py' -v
"$TRAIN_PYTHON" super/tests/test_token_budget_distributed.py
```

Set `TEST_MODEL` to a local Nemotron model directory to include the native-tokenizer integration test. Other pipeline tests run on CPU without model weights. The distributed check uses four CPU/Gloo workers to verify DP/CP accounting and exact-budget termination.

The portable launch and configuration paths are covered by CPU checks. A full GPU training run on each supported hardware topology is not part of these tests.

