import hashlib
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


# Settings that identify WHICH EXPERIMENT a run belongs to. Everything else in a run's config (seed, output
# paths, split sizes that follow from the seed) may differ between runs of the same experiment.
_NOT_PART_OF_CONFIG_ID = {"seed", "n_seeds", "checkpoint", "run_dir", "n_train_systems", "n_val_systems",
                          "n_test_systems", "config_id"}


def config_id(config: dict) -> str:
    """Short hash of the experiment settings (all of `config` except seed, paths and derived counts)."""
    items = {k: v for k, v in sorted(config.items()) if k not in _NOT_PART_OF_CONFIG_ID}
    return hashlib.sha1(json.dumps(items, sort_keys=True, default=str).encode()).hexdigest()[:10]


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
            "config_id": config.get("config_id") or config_id(config),
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


def summarize_runs(run_dir="runs", metric_key="val_time_loss", config_id_filter=None):
    """
    Aggregate the run JSONs of ONE experiment (e.g. different seeds) into mean/std of a final metric.

    Only runs with the same config_id are combined. Give config_id_filter to pick the experiment; without it
    the folder must contain a single configuration, otherwise a ValueError lists the configurations found,
    because averaging runs with different settings would not mean anything.
    """
    import glob
    import statistics

    by_config = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
        with open(path) as f:
            rec = json.load(f)
        cid = rec.get("config_id") or config_id(rec.get("config", {}))
        if metric_key in rec.get("final", {}):
            by_config.setdefault(cid, []).append((path, rec["final"][metric_key]))

    if config_id_filter is not None:
        by_config = {k: v for k, v in by_config.items() if k == config_id_filter}
    if len(by_config) > 1:
        raise ValueError(f"{run_dir} holds runs from {len(by_config)} different configurations "
                         f"({', '.join(by_config)}); pass config_id_filter to choose one.")
    if not by_config:
        return {"n_runs": 0, "mean": None, "std": None, "values": [], "config_id": config_id_filter}

    cid, runs = next(iter(by_config.items()))
    values = [v for _, v in runs]
    return {"n_runs": len(values), "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": values, "config_id": cid, "files": [p for p, _ in runs]}
