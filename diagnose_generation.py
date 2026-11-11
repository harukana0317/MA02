from queue_rules import VALID_MODES
import argparse
import statistics

from numerical_studies import (
    simulate_customers_gg1,
    _customers_to_event_dt,
    HAVE_TRANSFORMER,
)

try:  # scipy is optional; without it the distance section is skipped
    from numerical_studies import scipy_stats
except ImportError:
    scipy_stats = None

try:  # only defined when PyTorch + generate.py are importable
    from numerical_studies import _load_transformer_model, _transformer_generate
except ImportError:
    pass

EVENT_NAMES = ["Arrival", "Ordering", "Leave"]


def match_with_counts(stream):
    """
    Exactly the same FIFO rule as numerical_studies.match_events_to_customers(),
    but it also COUNTS the events that could not be matched, because that
    count is itself a symptom (a Leave with nobody present is impossible).
    stream: list of [event_id, dt]. Returns (records, orphan_order, orphan_leave, leftover).
    """
    t = 0.0
    waiting_to_order, waiting_to_leave, records = [], [], []
    orphan_order = orphan_leave = 0
    for event_id, dt in stream:
        t += dt
        if event_id == 0:
            waiting_to_order.append({"arrival": t})
        elif event_id == 1:
            if waiting_to_order:
                c = waiting_to_order.pop(0)
                c["order"] = t
                waiting_to_leave.append(c)
            else:
                orphan_order += 1
        elif event_id == 2:
            if waiting_to_leave:
                c = waiting_to_leave.pop(0)
                c["leave"] = t
                records.append(c)
            elif waiting_to_order:
                c = waiting_to_order.pop(0)
                c["leave"] = t
                records.append(c)
            else:
                orphan_leave += 1
    leftover = len(waiting_to_order) + len(waiting_to_leave)
    return records, orphan_order, orphan_leave, leftover


def stream_report(stream):
    """Summary numbers for ONE event stream (list of [event_id, dt])."""
    n = len(stream)
    counts = [0, 0, 0]
    dt_sum = [0.0, 0.0, 0.0]
    in_system, path = 0, []
    for event_id, dt in stream:
        counts[event_id] += 1
        dt_sum[event_id] += dt
        if event_id == 0:
            in_system += 1
        elif event_id == 2:
            in_system -= 1
        path.append(in_system)

    records, orphan_o, orphan_d, leftover = match_with_counts(stream)
    cycles = [r["leave"] - r["arrival"] for r in records]

    third = len(cycles) // 3
    first_third = statistics.mean(cycles[:third]) if third >= 1 else None
    last_third = statistics.mean(cycles[-third:]) if third >= 1 else None

    return {
        "n_events": n,
        "pct": [100.0 * c / n if n else 0.0 for c in counts],
        "mean_dt": [dt_sum[k] / counts[k] if counts[k] else None for k in range(3)],
        "N_mean": statistics.mean(path) if path else None,
        "N_max": max(path) if path else None,
        "N_end": path[-1] if path else None,
        "N_neg_steps": sum(1 for v in path if v < 0),
        "orphan_o": orphan_o,
        "orphan_d": orphan_d,
        "leftover": leftover,
        "n_cust": len(cycles),
        "cycle_mean": statistics.mean(cycles) if cycles else None,
        "cycle_median": statistics.median(cycles) if cycles else None,
        "first_third": first_third,
        "last_third": last_third,
        "cycles": cycles,
    }


def distance_to_truth(cycles_gen, cycles_true):
    """
    The comparison metric:
      W1 = Wasserstein-1 distance between the generated and ground-truth
           cycle-time distributions, in SECONDS. It is the average horizontal
           gap between the two cumulative distribution curves, so 0 means the
           distributions are identical and 100 means they are typically 100 s apart.
      KS D = Kolmogorov-Smirnov statistic (largest vertical gap, between 0 and 1).
    Returns (W1, KS_D), or (None, None) if there is nothing to compare.
    """
    if scipy_stats is None or not cycles_gen or not cycles_true:
        return None, None
    w1 = float(scipy_stats.wasserstein_distance(cycles_gen, cycles_true))
    ks = float(scipy_stats.ks_2samp(cycles_gen, cycles_true).statistic)
    return w1, ks


def _f(v, w, d=1):
    return f"{'n/a':>{w}}" if v is None else f"{v:>{w}.{d}f}"


def print_table(rows):
    print(f"{'run':<7}{'A%':>6}{'O%':>6}{'D%':>6} |{'N mean':>7}{'N max':>7}{'N end':>7}{'N<0':>6} |"
          f"{'orphO':>6}{'orphD':>6}{'left':>6} |{'cust':>5} |{'cyc mean':>9}{'median':>8} |"
          f"{'1st 1/3':>8}{'last 1/3':>9}")
    print("-" * 124)
    for name, r in rows:
        print(f"{name:<7}{r['pct'][0]:>6.1f}{r['pct'][1]:>6.1f}{r['pct'][2]:>6.1f} |"
              f"{_f(r['N_mean'], 7)}{r['N_max']:>7}{r['N_end']:>7}{r['N_neg_steps']:>6} |"
              f"{r['orphan_o']:>6}{r['orphan_d']:>6}{r['leftover']:>6} |{r['n_cust']:>5} |"
              f"{_f(r['cycle_mean'], 9)}{_f(r['cycle_median'], 8)} |"
              f"{_f(r['first_third'], 8)}{_f(r['last_third'], 9)}")


def run_diagnostics(truth_stream, generated_streams):
    truth = stream_report(truth_stream)
    reports = [stream_report(s) for s in generated_streams]

    print("\n=== 1) One row per run (truth = the ground-truth stream through the same code) ===")
    rows = [("truth", truth)] + [(f"gen {i}", r) for i, r in enumerate(reports)]
    print_table(rows)
    print("""
Legend: A%/O%/D% = share of Arrival/Ordering/Leave events (truth ~33.3 each).
  N = customers in the system = (#Arrivals - #Leaves so far): mean / max / value at the end / steps with N<0.
  orphO, orphD = Ordering / Leave events with nobody to attach to (impossible in a real queue).
  left = customers still unfinished when the stream ends.
  cyc mean / median = Arrival->Leave time (s); 1st 1/3 and last 1/3 = mean cycle of the earliest vs latest third of customers.""")

    print("\n=== 2) Mean time gap (s) BEFORE each event type ===")
    print(f"{'':<22}{'Arrival':>10}{'Ordering':>10}{'Leave':>10}")
    print(f"{'truth':<22}" + "".join(_f(truth['mean_dt'][k], 10) for k in range(3)))
    avg = []
    for k in range(3):
        vals = [r["mean_dt"][k] for r in reports if r["mean_dt"][k] is not None]
        avg.append(statistics.mean(vals) if vals else None)
    print(f"{'generated (avg runs)':<22}" + "".join(_f(avg[k], 10) for k in range(3)))

    print("\n=== 3) Across the generated runs ===")
    cyc = [r["cycle_mean"] for r in reports if r["cycle_mean"] is not None]
    if cyc and truth["cycle_mean"]:
        over2 = sum(1 for v in cyc if v > 2 * truth["cycle_mean"])
        print(f"cycle mean per run (s): " + ", ".join(f"{v:.0f}" for v in cyc))
        print(f"truth cycle mean: {truth['cycle_mean']:.1f}s | median of run means: {statistics.median(cyc):.1f}s | "
              f"runs with mean > 2x truth: {over2} of {len(cyc)}")
    bad_state = sum(1 for r in reports if r["N_neg_steps"] > 0 or r["orphan_o"] + r["orphan_d"] > 0)
    print(f"runs containing impossible states (N<0 or orphan events): {bad_state} of {len(reports)}")

    print("\n=== 4) Distance between generated and ground-truth cycle-time distributions ===")
    if scipy_stats is None:
        print("scipy is not installed, so this section is skipped (pip3 install scipy).")
    else:
        pairs = [distance_to_truth(r["cycles"], truth["cycles"]) for r in reports]
        w1s = [p[0] for p in pairs if p[0] is not None]
        kss = [p[1] for p in pairs if p[1] is not None]
        if w1s:
            print("W1 per run (s):  " + ", ".join(f"{v:.0f}" for v in w1s))
            print("KS D per run:    " + ", ".join(f"{v:.2f}" for v in kss))
            sd = lambda v: statistics.stdev(v) if len(v) > 1 else 0.0
            print(f"W1  mean +/- std over {len(w1s)} runs: {statistics.mean(w1s):.1f} +/- {sd(w1s):.1f} s")
            print(f"KS D mean +/- std over {len(kss)} runs: {statistics.mean(kss):.3f} +/- {sd(kss):.3f}")
            if len(w1s) < len(reports):
                print(f"({len(reports) - len(w1s)} run(s) skipped: no completed customers to compare)")
        else:
            print("No run produced completed customers, nothing to compare.")

    print("""
HOW TO READ IT (what each pattern would point to):
  - A%/O%/D% far from 33%, or large 'left'  -> the model's event-type choices drift (it loses track of who is in the system).
  - N end / N max grow large, or N<0 / orphans > 0 -> the generated queue leaves the set of states seen in training.
  - last 1/3 much bigger than 1st 1/3        -> error accumulates the longer generation runs (exposure bias).
  - mean >> median, or only some runs huge   -> a few runs blow up; the average is dominated by them.
  - gap before each event type far from truth -> the time model (dt given event type) is still off.""")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoint.pt")
    p.add_argument("--n_customers", type=int, default=300)
    p.add_argument("--arrival_shape", type=float, default=2.0)
    p.add_argument("--arrival_scale", type=float, default=40.0)
    p.add_argument("--service_shape", type=float, default=2.0)
    p.add_argument("--service_scale", type=float, default=25.0)
    p.add_argument("--n_repeats", type=int, default=10)
    p.add_argument("--truth", choices=["proxy", "o2des"], default="proxy",
                   help="Ground truth source: 'o2des' for checkpoints trained with --data gg1_o2des, "
                        "'proxy' for the earlier simulator (--data gg1).")
    p.add_argument("--constrain", choices=["off"] + list(VALID_MODES), default="off",
                   help="Constrained generation (hard rule). With 'strict' the orphan and "
                        "N<0 columns MUST be zero - if not, something is wrong with the rule code.")
    return p.parse_args()


def main():
    args = parse_args()
    if not HAVE_TRANSFORMER:
        print("PyTorch / generate.py are not available in this environment.")
        return

    # Same ground truth, same seed and same conditioning rule as sanity_check_step2.py
    if args.truth == "o2des":
        from o2des_gg1 import simulate_gg1_o2des
        truth_fn = simulate_gg1_o2des
    else:
        truth_fn = simulate_customers_gg1
    ground_truth = truth_fn(
        args.arrival_shape, args.arrival_scale, args.service_shape, args.service_scale,
        n_customers=args.n_customers, seed=999)
    truth_stream = _customers_to_event_dt(ground_truth)
    seed_events = _customers_to_event_dt(ground_truth[:20])

    model, dt_mean, dt_std, seq_len = _load_transformer_model(args.checkpoint)
    seed = seed_events[:seq_len]
    constrain_mode = None if args.constrain == "off" else args.constrain
    print(f"Checkpoint: {args.checkpoint} | time head: {model.time_mode} | "
          f"ground truth: {args.truth} | constrained generation: {args.constrain} | "
          f"generating {args.n_repeats} runs of {args.n_customers * 3} steps each...")

    streams = []
    for r in range(args.n_repeats):
        streams.append(_transformer_generate(model, seed, dt_mean, dt_std, seq_len,
                                             steps=args.n_customers * 3, rng_seed=r,
                                             constrain=constrain_mode))
    run_diagnostics(truth_stream, streams)


if __name__ == "__main__":
    main()
