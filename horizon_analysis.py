from queue_rules import VALID_MODES
import argparse
import csv
import math
import json
import statistics
import subprocess
import sys

from diagnose_generation import match_with_counts, stream_report
from numerical_studies import _customers_to_event_dt, HAVE_TRANSFORMER
from o2des_gg1 import simulate_gg1_o2des
from validator import validate_stream

try:
    from numerical_studies import scipy_stats
except ImportError:
    scipy_stats = None

try:  # only defined when PyTorch and generate.py are importable
    from numerical_studies import _load_transformer_model, _transformer_generate
except ImportError:
    pass

MIN_COMPLETED = 5   # fewer completed customers than this in a prefix: nothing meaningful to compare


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    pos = q * (len(values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def completed_cycles(stream, n_seed=0):
    """
    Cycle times of the customers completed in `stream`. With n_seed > 0 only customers who ARRIVE AFTER the
    first n_seed (conditioning) events count, so that the true history the model was given is not scored.
    """
    t_seed_end = sum(dt for _, dt in stream[:n_seed])
    records = match_with_counts(stream)[0]
    return [r["leave"] - r["arrival"] for r in records if n_seed == 0 or r["arrival"] > t_seed_end]


def prefix_w1(stream, n_events, reference, n_seed=0):
    """W1 (seconds) between the cycle times of generated customers completed within the first n_events and the reference."""
    cycles = completed_cycles(stream[:n_events], n_seed)
    if len(cycles) < MIN_COMPLETED:
        return None
    return float(scipy_stats.wasserstein_distance(cycles, reference))


def reference_cycles(params, n_customers=20000, drop=500, seed=4242):
    """Long ground-truth run; the first customers (empty-system start) are dropped."""
    records = simulate_gg1_o2des(*params, n_customers=n_customers, seed=seed)
    return [r["leave"] - r["arrival"] for r in records[drop:]]


def perfect_band(params, horizons, n_seed, reference, n_runs=200, seed0=20000):
    """
    What a PERFECT generator would score at each horizon: fresh ground-truth runs, cut to the same
    number of events and scored with exactly the same procedure as the model. The 95th percentile of
    their W1 is the acceptance threshold at that horizon.
    """
    n_customers = (n_seed + max(horizons)) // 3 + 30
    scores = {h: [] for h in horizons}
    n_invalid = 0
    for k in range(n_runs):
        stream = _customers_to_event_dt(simulate_gg1_o2des(*params, n_customers=n_customers, seed=seed0 + k))
        if not validate_stream(stream[:n_seed + max(horizons)])["valid"]:
            n_invalid += 1
        for h in horizons:
            w = prefix_w1(stream, n_seed + h, reference, n_seed)
            if w is not None:
                scores[h].append(w)
    if n_invalid:
        print(f"WARNING: {n_invalid} of {n_runs} reference simulator runs fail validator.py; "
              "the validator and the simulator disagree about the queue rules.")
    return {h: {"median": quantile(v, 0.5), "p95": quantile(v, 0.95), "n": len(v)} for h, v in scores.items()}


def analyse_runs(streams, n_seed, horizons, band, reference, labels=None):
    """
    streams: generated event streams (each starts with its n_seed conditioning events).
    labels: optional list (same length) naming the conditioning history of each stream.
    Returns (rows per horizon, per-trajectory records). Invalid trajectories stay in the table:
    "valid" is judged by validator.py on the first n_seed+h events, and W1 is reported both for all
    runs (reconstructed cycle times, NOT trustworthy for invalid runs) and for valid runs only.
    """
    rows, per_run = [], []
    for h in horizons:
        w1_all, w1_valid, completed, unfinished = [], [], [], []
        n_valid, n_in_band, n_ok = 0, 0, 0
        fail_cats = {}
        for k, s in enumerate(streams):
            prefix = s[:n_seed + h]
            check = validate_stream(prefix)
            valid = check["valid"]
            w = prefix_w1(s, n_seed + h, reference, n_seed)
            in_band = w is not None and w <= band[h]["p95"]
            n_valid += valid
            n_in_band += in_band
            n_ok += (valid and in_band)
            if w is not None:
                w1_all.append(w)
                if valid:
                    w1_valid.append(w)
            t_seed = sum(dt for _, dt in s[:n_seed])
            n_new = sum(1 for c in check["customers"] if c["arrival"] is not None and c["arrival"] > t_seed)
            n_done = sum(1 for c in check["customers"]
                         if c["arrival"] is not None and c["arrival"] > t_seed and c["leave"] is not None)
            completed.append(n_done)
            unfinished.append(n_new - n_done)
            if not valid:
                c = check["first_failure"]["category"]
                fail_cats[c] = fail_cats.get(c, 0) + 1
            per_run.append({"run": k, "history": labels[k] if labels else "", "horizon": h, "valid": int(valid),
                            "first_failure": "" if valid else check["first_failure"]["category"],
                            "first_failure_index": "" if valid else check["first_failure"]["index"],
                            "w1": "" if w is None else round(w, 3), "in_band": int(in_band),
                            "acceptable": int(valid and in_band), "completed_new": n_done,
                            "unfinished_new": n_new - n_done, "queue_end": check["queue_end"]})
        rows.append({
            "horizon": h, "band_median": band[h]["median"], "band_p95": band[h]["p95"], "runs": len(streams),
            "valid": n_valid, "in_band": n_in_band, "acceptable": n_ok,
            "model_w1_median": statistics.median(w1_all) if w1_all else None,
            "model_w1_median_valid": statistics.median(w1_valid) if w1_valid else None,
            "completed_median": statistics.median(completed) if completed else None,
            "unfinished_median": statistics.median(unfinished) if unfinished else None,
            "first_failures": "; ".join(f"{k}:{v}" for k, v in sorted(fail_cats.items())),
        })
    return rows, per_run


def boundary(rows, accept_fraction):
    """Largest horizon up to which at least accept_fraction of runs are acceptable at every horizon so far."""
    last_ok, first_bad = None, None
    for r in rows:
        if r["acceptable"] >= accept_fraction * r["runs"]:
            if first_bad is None:
                last_ok = r["horizon"]
        elif first_bad is None:
            first_bad = r["horizon"]
    return last_ok, first_bad


def fmt(v, width, digits=0):
    return " " * (width - 3) + "n/a" if v is None else f"{v:>{width}.{digits}f}"


def print_band(band):
    print("Perfect-generator reference: W1 (s) of fresh ground-truth runs scored the same way")
    print(f"{'horizon':>8}{'median':>10}{'95th pct':>10}{'runs':>7}")
    for h, b in band.items():
        print(f"{h:>8}{fmt(b['median'], 10, 1)}{fmt(b['p95'], 10, 1)}{b['n']:>7}")


def print_rows(rows, accept_fraction):
    print(f"\n{'horizon':>8} | {'perfect W1':>11}{'accept if':>10} | {'valid':>6}{'in band':>9}{'acceptable':>12} | "
          f"{'W1 all':>8}{'W1 valid':>10} | {'done':>6}{'unfin.':>7}")
    print(f"{'(steps)':>8} | {'median':>11}{'W1 <=':>10} | {'':>6}{'':>9}{'':>12} | {'median':>8}{'median':>10} | {'median':>6}{'median':>7}")
    print("-" * 110)
    for r in rows:
        n = r["runs"]
        print(f"{r['horizon']:>8} | {fmt(r['band_median'], 11, 1)}{fmt(r['band_p95'], 10, 1)} | "
              f"{str(r['valid']) + '/' + str(n):>6}{str(r['in_band']) + '/' + str(n):>9}{str(r['acceptable']) + '/' + str(n):>12} | "
              f"{fmt(r['model_w1_median'], 8, 1)}{fmt(r['model_w1_median_valid'], 10, 1)} | "
              f"{fmt(r['completed_median'], 6)}{fmt(r['unfinished_median'], 7)}")
        if r["first_failures"]:
            print(f"{'':>8}   first failures of invalid runs: {r['first_failures']}")
    last_ok, first_bad = boundary(rows, accept_fraction)
    need = f"at least {accept_fraction:.0%} of runs"
    print(f"\nA horizon counts as acceptable when {need} are valid (validator.py: all G/G/1 rules hold) AND have W1 inside the perfect-generator band.")
    print("W1 'all' uses reconstructed cycle times and is not trustworthy for invalid runs; 'valid' uses valid runs only.")
    print("done / unfin. = customers (arriving after the conditioning events) who finished / had not finished by the cutoff.")
    if first_bad is None:
        print(f"Result: acceptable at every tested horizon (up to {rows[-1]['horizon']} steps).")
    elif last_ok is None:
        print(f"Result: not acceptable even at the shortest tested horizon ({first_bad} steps).")
    else:
        print(f"Result: acceptable up to {last_ok} steps; performance drops by {first_bad} steps.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--band_only", action="store_true", help="only print the perfect-generator reference (no checkpoint or PyTorch needed)")
    p.add_argument("--horizons", type=int, nargs="+", default=[100, 200, 300, 450, 600, 900],
                   help="generated steps after the conditioning events")
    p.add_argument("--n_repeats", type=int, default=10, help="continuations generated per conditioning history")
    p.add_argument("--n_seed_events", type=int, default=16, help="ground-truth events the generator is conditioned on "
                   "(independent of the model's own context window)")
    p.add_argument("--history_seeds", type=int, nargs="+", default=[999],
                   help="seeds of the ground-truth runs used as conditioning histories; several seeds = several "
                        "different queue states (each gets --n_repeats continuations)")
    p.add_argument("--constrain", choices=["off"] + list(VALID_MODES), default="off")
    p.add_argument("--accept_fraction", type=float, default=0.8)
    p.add_argument("--band_runs", type=int, default=200)
    p.add_argument("--arrival_shape", type=float, default=2.0)
    p.add_argument("--arrival_scale", type=float, default=40.0)
    p.add_argument("--service_shape", type=float, default=2.0)
    p.add_argument("--service_scale", type=float, default=25.0)
    p.add_argument("--out", default=None, help="write the table to this CSV file (a .meta.json and a .runs.csv with "
                   "one row per trajectory and horizon are written next to it)")
    return p.parse_args()


def main():
    args = parse_args()
    if scipy_stats is None:
        sys.exit("scipy is required (pip3 install scipy).")
    horizons = sorted(args.horizons)
    params = (args.arrival_shape, args.arrival_scale, args.service_shape, args.service_scale)
    rho = (args.service_shape * args.service_scale) / (args.arrival_shape * args.arrival_scale)
    print(f"Ground truth: O2DES G/G/1, utilisation {rho:.3f}; {args.band_runs} reference runs per horizon")

    reference = reference_cycles(params)
    band = perfect_band(params, horizons, args.n_seed_events, reference, n_runs=args.band_runs)
    print_band(band)
    if args.band_only:
        return
    if not args.checkpoint:
        sys.exit("Give --checkpoint, or use --band_only.")
    if not HAVE_TRANSFORMER:
        sys.exit("PyTorch / generate.py are not available in this environment.")

    model, dt_mean, dt_std, seq_len = _load_transformer_model(args.checkpoint)
    constrain = None if args.constrain == "off" else args.constrain
    n_hist_customers = max(300, args.n_seed_events // 3 + 20)
    histories = {}
    for hs in args.history_seeds:
        truth_stream = _customers_to_event_dt(simulate_gg1_o2des(*params, n_customers=n_hist_customers, seed=hs))
        if len(truth_stream) < args.n_seed_events:
            sys.exit(f"History seed {hs} has only {len(truth_stream)} events, fewer than --n_seed_events.")
        histories[hs] = truth_stream[:args.n_seed_events]
    print(f"\nCheckpoint {args.checkpoint} (training window {seq_len}, time head {model.time_mode}), "
          f"generation rule: {args.constrain}; {len(histories)} histories x {args.n_repeats} runs of {max(horizons)} steps; "
          f"conditioned on {args.n_seed_events} true events")
    streams, labels = [], []
    for hs, seed_events in histories.items():
        for r in range(args.n_repeats):
            streams.append(_transformer_generate(model, seed_events, dt_mean, dt_std, seq_len,
                                                 steps=max(horizons), rng_seed=r, constrain=constrain))
            labels.append(hs)
            print(f"  generated run {len(streams)}/{len(histories) * args.n_repeats}")

    rows, per_run = analyse_runs(streams, args.n_seed_events, horizons, band, reference, labels)
    print_rows(rows, args.accept_fraction)
    if len(histories) == 1:
        print("Note: all runs share ONE conditioning history; the runs measure sampling variability for that history only.")
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        base = args.out[:-4] if args.out.endswith(".csv") else args.out
        with open(base + ".runs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_run[0].keys()))
            w.writeheader()
            w.writerows(per_run)
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            commit = None
        with open(base + ".meta.json", "w") as f:
            json.dump({"args": vars(args), "training_window": seq_len, "time_head": model.time_mode,
                       "git_commit": commit, "band_runs": args.band_runs, "ground_truth_params": params}, f, indent=2)
        print(f"Saved {args.out}, {base}.runs.csv and {base}.meta.json")


if __name__ == "__main__":
    main()
