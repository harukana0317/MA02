import json
import os
import random
import subprocess
import sys
import time

import torch


def set_seed(seed: int):
    """Fix every source of randomness we use. Call this before building
    the dataset AND before building the model, since both consume the
    random state (data augmentation/shuffling vs. weight init)."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Makes cuDNN conv algorithms deterministic (slower, but reproducible).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_git_commit():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet"], stderr=subprocess.DEVNULL
        ) != 0
        return commit + ("-dirty" if dirty else "")
    except Exception:
        return None


class ExperimentLogger:
    """
    Writes one JSON file per run to `run_dir`, e.g. runs/2026-07-01_143000_seed0.json
    Contains: config, env info (git commit, torch version, python version),
    and a growing list of per-epoch metric dicts.

    This is intentionally dependency-free (no wandb/tensorboard requirement)
    so it works offline during data collection.
    """

    def __init__(self, config: dict, run_dir="runs", tag=None):
        os.makedirs(run_dir, exist_ok=True)
        ts = time.strftime("%Y-%m-%d_%H%M%S")
        suffix = f"_{tag}" if tag else ""
        self.path = os.path.join(run_dir, f"{ts}{suffix}.json")

        self.record = {
            "config": config,
            "env": {
                "git_commit": _get_git_commit(),
                "torch_version": torch.__version__,
                "python_version": sys.version,
                "cuda_available": torch.cuda.is_available(),
                "start_time": ts,
            },
            "epochs": [],
            "final": {},
        }
        self._flush()

    def log_epoch(self, epoch: int, metrics: dict):
        entry = {"epoch": epoch, **metrics}
        self.record["epochs"].append(entry)
        self._flush()

    def log_final(self, metrics: dict):
        self.record["final"] = metrics
        self._flush()

    def _flush(self):
        with open(self.path, "w") as f:
            json.dump(self.record, f, indent=2)


def summarize_runs(run_dir="runs", metric_key="val_time_loss"):
    """
    Aggregate multiple run JSONs (e.g. from different seeds) into
    mean/std of a final metric — for reporting "X ± Y" in a paper.
    """
    import glob
    import statistics

    values = []
    for path in glob.glob(os.path.join(run_dir, "*.json")):
        with open(path) as f:
            rec = json.load(f)
        if metric_key in rec.get("final", {}):
            values.append(rec["final"][metric_key])

    if not values:
        return {"n_runs": 0, "mean": None, "std": None, "values": []}

    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"n_runs": len(values), "mean": mean, "std": std, "values": values}
