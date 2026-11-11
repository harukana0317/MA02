import argparse
import csv
import os
import re
import statistics
import subprocess
import sys
import time

# Seed-0 results already obtained (per-run W1 in seconds and KS D, ten generations each),
# copied from the diagnose_generation.py outputs. "bad" = runs containing impossible events.
SEED0 = {
    ("all", "off"):      {"w1": [511, 772, 1114, 43, 816, 1895, 1347, 499, 1125, 1419],
                          "ks": [0.74, 0.79, 0.81, 0.19, 0.94, 0.87, 0.96, 0.82, 0.51, 0.75], "bad": 8},
    ("all", "strict"):   {"w1": [23, 772, 1099, 12, 836, 2149, 1347, 368, 1225, 972],
                          "ks": [0.24, 0.79, 0.80, 0.08, 0.87, 0.89, 0.96, 0.57, 0.64, 0.55], "bad": 0},
    ("stable", "off"):   {"w1": [49, 58, 86, 6, 695, 279, 878, 82, 119, 459],
                          "ks": [0.28, 0.22, 0.38, 0.07, 0.77, 0.73, 0.78, 0.36, 0.28, 0.61], "bad": 9},
    ("stable", "strict"): {"w1": [210, 74, 23, 6, 797, 276, 878, 81, 182, 453],
                           "ks": [0.58, 0.25, 0.10, 0.06, 0.80, 0.73, 0.78, 0.36, 0.44, 0.56], "bad": 0},
}

SETTINGS = {
    "all": [],                                  # original ranges, about 23% unstable systems
    "stable": ["--rho_range", "0.3", "0.9"],    # stable systems only
}
RULES = {
    "off": [],
    "strict": ["--constrain", "strict"],
}
NEAR_S = 30    # a run counts as "close to the ground truth" if its W1 is at most this many seconds
FAR_S = 200    # ... and as "far off" if its W1 is above this


# ------------------------------------------------------------------ running steps
def run_logged(cmd, log_path):
    """Run cmd, copy its output to log_path, print only short progress lines. Returns the exit code."""
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True)
        for line in proc.stdout:
            log.write(line)
            log.flush()
            m = re.search(r"Epoch (\d+)/(\d+)", line)
            if m and int(m.group(1)) % 10 == 0:
                print("      epoch %s/%s" % (m.group(1), m.group(2)))
            elif "Best validation loss" in line or "Error" in line or "Traceback" in line:
                print("      " + line.strip()[:150])
        return proc.wait()


def file_text(path):
    if not os.path.exists(path):
        return ""
    with open(path) as f:
        return f.read()


def train_done(log_path):
    return "Saved checkpoint to" in file_text(log_path)


def diag_done(log_path):
    return "=== 4)" in file_text(log_path)


def parse_diag(text):
    """Pull W1, KS D and the impossible-run count out of a diagnose_generation.py output."""
    w1 = re.search(r"^W1 per run \(s\):\s+(.+)$", text, re.M)
    ks = re.search(r"^KS D per run:\s+(.+)$", text, re.M)
    bad = re.search(r"runs containing impossible states \(N<0 or orphan events\): (\d+) of (\d+)", text)
    if not (w1 and ks and bad):
        return None
    return {"w1": [float(x) for x in w1.group(1).split(",")],
            "ks": [float(x) for x in ks.group(1).split(",")],
            "bad": int(bad.group(1)), "n_runs": int(bad.group(2))}


# ------------------------------------------------------------------ summary
def _sd(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize(results, seeds):
    """results[(setting, rule, seed)] = parsed dict. Returns (text, csv_rows)."""
    lines, rows = [], []
    lines.append("SUMMARY: distance between generated and ground-truth cycle-time distributions")
    lines.append("W1 = Wasserstein-1 distance in seconds (0 = identical); each seed = ten free-running generations.")
    lines.append("close = runs with W1 <= %d s; far = runs with W1 > %d s; impossible = runs containing impossible events." % (NEAR_S, FAR_S))

    seed_means = {}
    for rule in RULES:
        lines.append("")
        lines.append("=== generation rule: %s %s ===" % (rule, "(hard rule: forbid impossible events)" if rule == "strict" else "(no rule)"))
        for setting in SETTINGS:
            lines.append("  training systems: %s" % ("stable only (rho 0.3-0.9)" if setting == "stable" else "all (about 23% unstable)"))
            lines.append("    %-6s %10s %10s %7s %6s %10s %11s" % ("seed", "W1 mean", "W1 median", "close", "far", "KS D mean", "impossible"))
            means = []
            for seed in seeds:
                r = results.get((setting, rule, seed))
                if r is None:
                    lines.append("    %-6s %s" % (seed, "-- missing (step failed or not run) --"))
                    continue
                w = r["w1"]
                m = statistics.mean(w)
                means.append(m)
                n = r.get("n_runs", len(w))
                lines.append("    %-6s %10.1f %10.1f %7s %6s %10.3f %11s" % (
                    seed, m, statistics.median(w), "%d/%d" % (sum(x <= NEAR_S for x in w), len(w)),
                    "%d/%d" % (sum(x > FAR_S for x in w), len(w)), statistics.mean(r["ks"]),
                    "%d/%d" % (r["bad"], n)))
                rows.append({"setting": setting, "rule": rule, "seed": seed, "w1_mean": round(m, 1),
                             "w1_median": round(statistics.median(w), 1),
                             "runs_close": sum(x <= NEAR_S for x in w), "runs_far": sum(x > FAR_S for x in w),
                             "ks_mean": round(statistics.mean(r["ks"]), 3), "runs_impossible": r["bad"]})
            seed_means[(setting, rule)] = dict(zip([s for s in seeds if results.get((setting, rule, s))], means))
            if means:
                lines.append("    across %d seeds: mean W1 = %.1f +/- %.1f   (lowest %.1f, highest %.1f)" % (
                    len(means), statistics.mean(means), _sd(means), min(means), max(means)))

    lines.append("")
    lines.append("=== does training on stable systems only help? (mean W1, stable minus all, per seed) ===")
    for rule in RULES:
        a, b = seed_means.get(("all", rule), {}), seed_means.get(("stable", rule), {})
        common = [s for s in seeds if s in a and s in b]
        if not common:
            continue
        diffs = [b[s] - a[s] for s in common]
        better = sum(d < 0 for d in diffs)
        lines.append("  rule %-6s: " % rule + ", ".join("seed %s: %+.0f s" % (s, d) for s, d in zip(common, diffs))
                     + "   -> stable-only had the lower W1 in %d of %d seeds" % (better, len(common)))
    lines.append("")
    lines.append("How to read: with only a few seeds no significance test is meaningful. Compare the size of the")
    lines.append("stable-vs-all difference with the spread BETWEEN seeds (the +/- above). If the lowest W1 of one")
    lines.append("setting is still above the highest W1 of the other, the difference is not just seed luck.")
    return "\n".join(lines), rows


# ------------------------------------------------------------------ main
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--num_sim_systems", type=int, default=20)
    p.add_argument("--n_repeats", type=int, default=10)
    p.add_argument("--n_customers", type=int, default=300)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--quick", action="store_true", help="tiny version to test that everything runs (numbers are meaningless)")
    p.add_argument("--force", action="store_true", help="redo steps even if their output already exists")
    p.add_argument("--no_seed0", action="store_true", help="leave the typed-in seed-0 results out of the table")
    p.add_argument("--train_script", default="train.py", help=argparse.SUPPRESS)   # for testing only
    p.add_argument("--diag_script", default="diagnose_generation.py", help=argparse.SUPPRESS)
    p.add_argument("--skip_preflight", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.epochs, args.num_sim_systems, args.n_repeats, args.n_customers = 2, 6, 2, 60
        args.seeds, args.no_seed0 = [1], True
    out_dir = args.out_dir or ("replicates_quick" if args.quick else "replicates")
    os.makedirs(out_dir, exist_ok=True)

    for f in (args.train_script, args.diag_script):
        if not os.path.exists(f):
            sys.exit("Cannot find %s in this folder. Run this script from the folder that contains it." % f)
    if not args.skip_preflight:
        try:
            import o2des_gg1
            if not o2des_gg1.HAVE_O2DES:
                sys.exit("O2DESpy is not available: %s\nInstall it with:  pip3 install o2despy pandas sortedcontainers" % o2des_gg1._IMPORT_ERROR)
        except ImportError:
            sys.exit("Cannot find o2des_gg1.py in this folder.")

    steps = len(args.seeds) * len(SETTINGS) * (1 + len(RULES))
    print("Output folder: %s | seeds %s | %d epochs | %d steps in total" % (out_dir, args.seeds, args.epochs, steps))
    t_start, step, failures, results = time.time(), 0, [], {}

    for seed in args.seeds:
        for setting, extra in SETTINGS.items():
            tag = "%s_s%d" % (setting, seed)
            ckpt = os.path.join(out_dir, "ck_%s.pt" % tag)
            tlog = os.path.join(out_dir, "train_%s.log" % tag)

            step += 1
            if train_done(tlog) and os.path.exists(ckpt) and not args.force:
                print("[%d/%d] train %s: already done, skipping" % (step, steps, tag))
            else:
                print("[%d/%d] train %s ..." % (step, steps, tag))
                t0 = time.time()
                cmd = [sys.executable, "-u", args.train_script, "--data", "gg1_o2des",
                       "--epochs", str(args.epochs), "--num_sim_systems", str(args.num_sim_systems),
                       "--seed", str(seed), "--regenerate_each_epoch", "--time_head", "lognormal",
                       "--checkpoint", ckpt] + extra
                code = run_logged(cmd, tlog)
                if code != 0 or not train_done(tlog):
                    print("      FAILED (exit code %s). Last lines of %s:" % (code, tlog))
                    print("".join("        " + l for l in file_text(tlog).splitlines(True)[-12:]))
                    failures.append(tlog)
                    step += len(RULES)
                    continue
                print("      done in %.1f min" % ((time.time() - t0) / 60))

            for rule, rule_args in RULES.items():
                step += 1
                dlog = os.path.join(out_dir, "diag_%s_%s.log" % (tag, rule))
                if diag_done(dlog) and not args.force:
                    print("[%d/%d] diagnose %s, rule %s: already done, skipping" % (step, steps, tag, rule))
                else:
                    print("[%d/%d] diagnose %s, rule %s ..." % (step, steps, tag, rule))
                    cmd = [sys.executable, "-u", args.diag_script, "--checkpoint", ckpt, "--truth", "o2des",
                           "--n_repeats", str(args.n_repeats), "--n_customers", str(args.n_customers)] + rule_args
                    code = run_logged(cmd, dlog)
                    if code != 0 or not diag_done(dlog):
                        print("      FAILED (exit code %s), see %s" % (code, dlog))
                        failures.append(dlog)
                        continue
                parsed = parse_diag(file_text(dlog))
                if parsed is None:
                    print("      could not read the numbers from %s" % dlog)
                    failures.append(dlog)
                else:
                    results[(setting, rule, seed)] = parsed

    seeds = list(args.seeds)
    if not args.no_seed0:
        for (setting, rule), d in SEED0.items():
            results[(setting, rule, 0)] = dict(d, n_runs=len(d["w1"]))
        seeds = [0] + seeds
    text, rows = summarize(results, seeds)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["setting", "rule", "seed", "w1_mean", "w1_median", "runs_close",
                                          "runs_far", "ks_mean", "runs_impossible"])
        w.writeheader()
        w.writerows(rows)

    print("\n" + text)
    print("\nTotal time: %.1f min. Saved %s and %s" % ((time.time() - t_start) / 60,
          os.path.join(out_dir, "summary.txt"), os.path.join(out_dir, "summary.csv")))
    if failures:
        print("\n%d step(s) failed; their logs: %s" % (len(failures), ", ".join(failures)))
        print("Run the same command again to retry only the failed steps.")


if __name__ == "__main__":
    main()
