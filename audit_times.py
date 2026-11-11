from queue_rules import VALID_MODES
import argparse
import statistics
import sys

from diagnose_generation import match_with_counts
from numerical_studies import _customers_to_event_dt, HAVE_TRANSFORMER
from o2des_gg1 import simulate_gg1_o2des

try:
    from numerical_studies import scipy_stats
except ImportError:
    scipy_stats = None

try:  # only defined when PyTorch and generate.py are importable
    from numerical_studies import _load_transformer_model, _transformer_generate
except ImportError:
    pass

COMPONENTS = ("interarrival", "wait", "service", "cycle")
LABELS = {"interarrival": "inter-arrival (Arrival -> next Arrival)", "wait": "wait (Arrival -> Ordering)",
          "service": "service (Ordering -> Leave)", "cycle": "cycle time (Arrival -> Leave)"}


def audit_stream(stream, tol=1.0):
    """
    Rebuild the four time quantities from an event stream [[event_id, dt], ...] and test one rule of a
    single FIFO server: service starts (Ordering) at max(own arrival, previous customer's Leave).
    """
    t, arrivals = 0.0, []
    for event_id, dt in stream:
        t += dt
        if event_id == 0:
            arrivals.append(t)
    records, orphan_o, orphan_d, left = match_with_counts(stream)
    records = sorted(records, key=lambda r: r["arrival"])

    out = {
        "interarrival": [b - a for a, b in zip(arrivals[:-1], arrivals[1:])],
        "wait": [r["order"] - r["arrival"] for r in records if "order" in r],
        "service": [r["leave"] - r["order"] for r in records if "order" in r],
        "cycle": [r["leave"] - r["arrival"] for r in records],
        "orphans": orphan_o + orphan_d,
    }
    checked = late = early = 0
    server_time = []          # leave - (moment the server could first have started this customer)
    prev_leave = None
    for r in records:
        if "order" in r:
            expected = r["arrival"] if prev_leave is None else max(r["arrival"], prev_leave)
            checked += 1
            diff = r["order"] - expected
            late += diff > tol        # service started later than the server was free: idle server, waiting customer
            early += diff < -tol      # service started while the previous customer was still being served
            server_time.append(r["leave"] - expected)
        prev_leave = r["leave"]
    out["checked"], out["late"], out["early"], out["violated"] = checked, late, early, late + early
    mean = lambda v: statistics.mean(v) if v else None
    ia = mean(out["interarrival"])
    out["rho_eff"] = mean(out["service"]) / ia if out["service"] and ia else None
    out["rho_server"] = mean(server_time) / ia if server_time and ia else None
    return out


def w1(values, reference):
    if scipy_stats is None or len(values) < 5:
        return None
    return float(scipy_stats.wasserstein_distance(values, reference))


def reference_components(params, n_customers=20000, drop=500, seed=4242):
    records = simulate_gg1_o2des(*params, n_customers=n_customers, seed=seed)
    ref = audit_stream(_customers_to_event_dt(records))
    return {k: ref[k][drop:] for k in COMPONENTS}, ref["rho_eff"]


def f(v, w, d=1):
    return " " * (w - 3) + "n/a" if v is None else f"{v:>{w}.{d}f}"


def report(streams, reference, ref_rho, tol):
    ref_mean = {k: statistics.mean(reference[k]) for k in COMPONENTS}
    truth_rows = None
    print(f"\n{'run':<7}{'rho_eff':>8}{'rho_srv':>8} |{'mean gap':>9}{'service':>9}{'wait':>8} |"
          f"{'W1 gap':>8}{'W1 svc':>8}{'W1 wait':>9}{'W1 cycle':>9} |{'late':>7}{'early':>7}{'impossible':>11}")
    print(f"{'':<7}{'':>8}{'':>8} |{'(s)':>9}{'(s)':>9}{'(s)':>8} |{'(s)':>8}{'(s)':>8}{'(s)':>9}{'(s)':>9} |{'%':>7}{'%':>7}{'events':>11}")
    print("-" * 119)
    print(f"{'truth':<7}{f(ref_rho, 8, 3)}{f(ref_rho, 8, 3)} |{f(ref_mean['interarrival'], 9)}{f(ref_mean['service'], 9)}{f(ref_mean['wait'], 8)} |"
          f"{'':>8}{'':>8}{'':>9}{'':>9} |{'0.0':>7}{'0.0':>7}{'0':>11}")
    audits = [audit_stream(s, tol) for s in streams]
    rel = {k: [] for k in COMPONENTS}
    for i, a in enumerate(audits):
        ws = {k: w1(a[k], reference[k]) for k in COMPONENTS}
        for k in COMPONENTS:
            if ws[k] is not None:
                rel[k].append(ws[k] / ref_mean[k])
        pct = lambda k: 100.0 * a[k] / a["checked"] if a["checked"] else None
        mean = lambda v: statistics.mean(v) if v else None
        print(f"{'gen ' + str(i):<7}{f(a['rho_eff'], 8, 3)}{f(a['rho_server'], 8, 3)} |{f(mean(a['interarrival']), 9)}{f(mean(a['service']), 9)}{f(mean(a['wait']), 8)} |"
              f"{f(ws['interarrival'], 8)}{f(ws['service'], 8)}{f(ws['wait'], 9)}{f(ws['cycle'], 9)} |{f(pct('late'), 7)}{f(pct('early'), 7)}{a['orphans']:>11}")

    print("\nLegend: rho_eff = mean service / mean inter-arrival measured from the generated events (truth is the first row).")
    print("        rho_srv = the same, but the server's time per customer also counts any delay before service started,")
    print("        i.e. the load the generated queue really experiences (equal to rho_eff when the server starts at once).")
    print(f"        late % = service started more than {tol:g} s after the server was free and the customer was waiting;")
    print(f"        early % = service started more than {tol:g} s before the previous customer had left.")
    print("        W1 columns = Wasserstein-1 distance of each time quantity to the ground-truth distribution.")

    print("\n=== Summary over the generated runs ===")
    rhos = [a["rho_eff"] for a in audits if a["rho_eff"] is not None]
    if rhos:
        print(f"rho_eff: mean {statistics.mean(rhos):.3f}, range {min(rhos):.3f}-{max(rhos):.3f} (truth {ref_rho:.3f}); "
              f"runs with rho_eff >= 1: {sum(r >= 1 for r in rhos)} of {len(rhos)}")
    rsrv = [a["rho_server"] for a in audits if a["rho_server"] is not None]
    if rsrv:
        print(f"rho_srv: mean {statistics.mean(rsrv):.3f}, range {min(rsrv):.3f}-{max(rsrv):.3f}; "
              f"runs with rho_srv >= 1: {sum(r >= 1 for r in rsrv)} of {len(rsrv)}")
    for key, label in (("late", "late starts"), ("early", "early starts")):
        share = [100.0 * a[key] / a["checked"] for a in audits if a["checked"]]
        if share:
            print(f"{label}: mean {statistics.mean(share):.1f}% of customers (truth 0%), "
                  f"runs with more than 5%: {sum(x > 5 for x in share)} of {len(share)}")
    print("Average W1 as a share of the truth mean (how far off each quantity is, in relative terms):")
    ranked = sorted(((statistics.mean(v), k) for k, v in rel.items() if v), reverse=True)
    for share, k in ranked:
        print(f"   {LABELS[k]:<42} {100 * share:6.0f}%   (truth mean {ref_mean[k]:.1f} s)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--steps", type=int, default=900)
    p.add_argument("--n_repeats", type=int, default=10)
    p.add_argument("--n_seed_events", type=int, default=16)
    p.add_argument("--constrain", choices=["off"] + list(VALID_MODES), default="off")
    p.add_argument("--tol", type=float, default=1.0)
    p.add_argument("--arrival_shape", type=float, default=2.0)
    p.add_argument("--arrival_scale", type=float, default=40.0)
    p.add_argument("--service_shape", type=float, default=2.0)
    p.add_argument("--service_scale", type=float, default=25.0)
    return p.parse_args()


def main():
    args = parse_args()
    if scipy_stats is None:
        sys.exit("scipy is required (pip3 install scipy).")
    if not HAVE_TRANSFORMER:
        sys.exit("PyTorch / generate.py are not available in this environment.")
    params = (args.arrival_shape, args.arrival_scale, args.service_shape, args.service_scale)
    reference, ref_rho = reference_components(params)
    ground_truth = simulate_gg1_o2des(*params, n_customers=300, seed=999)
    seed_events = _customers_to_event_dt(ground_truth)[:args.n_seed_events]
    model, dt_mean, dt_std, seq_len = _load_transformer_model(args.checkpoint)
    constrain = None if args.constrain == "off" else args.constrain
    print(f"Checkpoint {args.checkpoint} (training window {seq_len}, time head {model.time_mode}), generation rule: {args.constrain}")
    streams = [_transformer_generate(model, seed_events, dt_mean, dt_std, seq_len, steps=args.steps, rng_seed=r, constrain=constrain)
               for r in range(args.n_repeats)]
    report(streams, reference, ref_rho, args.tol)


if __name__ == "__main__":
    main()
