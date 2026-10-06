"""Stop only sustained validation degradation while training loss improves."""

from collections import deque
from pathlib import Path
import json, math, os


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
    import torch.distributed as dist
    from nemo_automodel.recipes.vlm.finetune import FinetuneRecipeForVLM
    import nvme_checkpoint as nv

    options = config["early_stopping"]
    monitor = OverfitMonitor(**options)
    original_train = FinetuneRecipeForVLM.log_train_metrics
    original_val = FinetuneRecipeForVLM.log_val_metrics
    original_backup = nv.backup_checkpoint
    state = Path(os.environ["RUN_DIR"]) / "state/early_stopping.json"

    def train(recipe, data):
        original_train(recipe, data)
        if recipe.dist_env.is_main:
            monitor.observe_training(data.metrics["loss"])

    def val(recipe, data):
        original_val(recipe, data)
        outcome = [None]
        if recipe.dist_env.is_main:
            outcome[0] = monitor.observe_validation(
                data.step, float(data.metrics["val_loss"])
            )
            state.write_text(
                json.dumps({"options": options, "history": monitor.history}, indent=2)
            )
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

    def backup(path):
        def persist():
            if state.is_file():
                (path / "early_stopping.json").write_text(state.read_text())

        nv.collective_step(persist, int(os.environ.get("LOCAL_RANK", "0")) == 0)
        return original_backup(path)

    FinetuneRecipeForVLM.log_train_metrics = train
    FinetuneRecipeForVLM.log_val_metrics = val
    nv.backup_checkpoint = backup
