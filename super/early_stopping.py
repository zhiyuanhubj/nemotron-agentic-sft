"""Stop only sustained validation degradation while training loss improves."""

import json
import math
import os
from collections import deque
from pathlib import Path


class OverfitMonitor:
    def __init__(
        self,
        min_steps=20,
        relative_val_increase=0.10,
        consecutive_evals=5,
        train_loss_drop=0.05,
        train_window=10,
    ):
        self.min_steps = min_steps
        self.relative_val_increase = relative_val_increase
        self.consecutive_evals = consecutive_evals
        self.train_loss_drop = train_loss_drop
        self.training = deque(maxlen=train_window)
        self.best = float("inf")
        self.best_train = None
        self.bad = 0
        self.history = []

    def observe_training(self, loss):
        self.training.append(float(loss))

    def observe_validation(self, step, loss):
        train = sum(self.training) / len(self.training) if self.training else None
        if loss < self.best:
            self.best = loss
            self.best_train = train
            self.bad = 0
        overfit = (
            step + 1 >= self.min_steps
            and loss >= self.best * (1 + self.relative_val_increase)
            and train is not None
            and self.best_train is not None
            and train <= self.best_train * (1 - self.train_loss_drop)
        )
        self.bad = self.bad + 1 if overfit else 0
        stop = self.bad >= self.consecutive_evals or not math.isfinite(loss)
        record = {
            "step": step,
            "val_loss": loss,
            "best_val_loss": self.best,
            "recent_train_loss": train,
            "train_loss_at_best": self.best_train,
            "consecutive_overfit_evals": self.bad,
            "stop": stop,
        }
        self.history.append(record)
        return record


def install(config):
    options = config.get("early_stopping")
    if not options or options.get("enabled") is False:
        return
    import checkpoint_state
    import torch.distributed as dist
    from nemo_automodel.recipes.vlm.finetune import FinetuneRecipeForVLM

    options = {k: v for k, v in options.items() if k != "enabled"}
    monitor = OverfitMonitor(**options)
    restore = config.get("checkpoint", {}).get("restore_from")
    if restore:
        saved = json.loads((Path(restore) / "early_stopping.json").read_text())
        if saved["options"] != options:
            raise ValueError("Resume requires unchanged early-stopping options")
        monitor.training.extend(saved["training"])
        monitor.best = saved["best"] if saved["best"] is not None else float("inf")
        monitor.best_train, monitor.bad = saved["best_train"], saved["bad"]
        monitor.history = saved["history"]
    original_train = FinetuneRecipeForVLM.log_train_metrics
    original_val = FinetuneRecipeForVLM.log_val_metrics
    state = Path(os.environ["RUN_DIR"]) / "state/early_stopping.json"

    def snapshot():
        return {
            "options": options,
            "history": monitor.history,
            "training": list(monitor.training),
            "best": monitor.best if math.isfinite(monitor.best) else None,
            "best_train": monitor.best_train,
            "bad": monitor.bad,
        }

    checkpoint_state.register("early_stopping", snapshot)

    def train(recipe, data):
        original_train(recipe, data)
        if recipe.dist_env.is_main:
            monitor.observe_training(data.metrics["loss"])

    def val(recipe, data):
        original_val(recipe, data)
        outcome = [None]
        if recipe.dist_env.is_main:
            outcome[0] = monitor.observe_validation(data.step, float(data.metrics["val_loss"]))
            state.write_text(json.dumps({"options": options, "history": monitor.history}, indent=2))
            import wandb

            if wandb.run is not None:
                wandb.log(
                    {
                        "early_stop/bad_evals": monitor.bad,
                        "early_stop/best_val_loss": monitor.best,
                        "early_stop/triggered": int(outcome[0]["stop"]),
                    },
                    step=data.step,
                )
        dist.broadcast_object_list(outcome, src=0)
        if outcome[0]["stop"]:
            recipe.step_scheduler.sigterm_flag = True
            if recipe.dist_env.is_main:
                print("EARLY_STOP_OVERFITTING", json.dumps(outcome[0]), flush=True)

    FinetuneRecipeForVLM.log_train_metrics = train
    FinetuneRecipeForVLM.log_val_metrics = val
