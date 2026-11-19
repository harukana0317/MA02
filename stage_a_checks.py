"""
Stage A: checks that the evaluation pipeline can be trusted. Run it before any experiment:

    python3 stage_a_checks.py

Prints one PASS/FAIL line per check (a table for the report). The checks that need PyTorch are skipped,
and said to be skipped, when it is not installed.
"""
import math
import sys

from baseline import ExponentialMarkovBaseline
from validator import validate_stream, completed_cycles

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))


# ---- 1. validator: valid simulator trajectories pass -------------------------------------------------
try:
    from o2des_gg1 import simulate_gg1_o2des, HAVE_O2DES
    from numerical_studies import _customers_to_event_dt
except ImportError:
    HAVE_O2DES = False

if HAVE_O2DES:
    n_bad, n_total = 0, 0
    for params in [(2, 40, 2, 25), (2, 25, 2, 40), (1, 60, 3, 15)]:     # stable, overloaded, other shapes
        for seed in range(40):
            stream = _customers_to_event_dt(simulate_gg1_o2des(*params, n_customers=150, seed=seed))
            n_total += 1
            n_bad += not validate_stream(stream)["valid"]
    check("validator accepts simulator trajectories", n_bad == 0, f"{n_total - n_bad}/{n_total} pass")

    # ---- 2. customer records survive event conversion and reconstruction ------------------------------
    worst = 0.0
    n_customers = 0
    for seed in range(20):
        records = simulate_gg1_o2des(2, 40, 2, 25, n_customers=120, seed=seed)
        stream = _customers_to_event_dt(records)
        rebuilt = validate_stream(stream)["customers"]
        n_customers += len(records)
        if len(rebuilt) != len(records):
            worst = float("inf")
            continue
        t0 = records[0]["arrival"]        # the event stream starts its clock at the first arrival
        for a, b in zip(records, rebuilt):
            for key in ("arrival", "order", "leave"):
                worst = max(worst, abs((a[key] - t0) - b[key]))
    check("event conversion + FIFO reconstruction returns the true customer times", worst < 1e-6,
          f"{n_customers} customers, largest difference {worst:.2e} s")
else:
    check("validator accepts simulator trajectories", False, "SKIPPED: o2despy not installed")

# ---- 3. the validator rejects each broken rule for the expected reason ----------------------------------
cases = [
    ("arrival then leave, service skipped", [[0, 0], [2, 10]], "leave_without_service"),
    ("two customers in service at once", [[0, 0], [1, 0], [0, 5], [1, 0]], "order_while_busy"),
    ("service starts 5 s after the server could start", [[0, 0], [1, 5]], "late_service_start"),
    ("negative time gap", [[0, 0], [1, -1]], "bad_gap"),
    ("waiting customer leaves (abandonment)", [[0, 0], [1, 0], [0, 5], [2, 5], [2, 1]], "leave_without_service"),
    ("ordering with nobody waiting", [[1, 0]], "order_without_wait"),
    ("not-a-number gap", [[0, float("nan")]], "bad_gap"),
]
for name, stream, expected in cases:
    r = validate_stream(stream)
    got = r["first_failure"]["category"] if r["first_failure"] else None
    check(f"validator rejects: {name}", (not r["valid"]) and got == expected, f"category {got}")

# the unfinished customers of an invalid-prefix-free stream stay visible
r = validate_stream([[0, 0], [1, 0], [0, 10], [0, 10]])
check("unfinished customers are counted", r["valid"] and r["n_unfinished"] == 3 and r["queue_end"] == 2 and r["busy_end"] == 1,
      f"unfinished {r['n_unfinished']}, waiting {r['queue_end']}, in service {r['busy_end']}")

# ---- 4. baseline probabilities ---------------------------------------------------------------------------
seqs = [[[0, 0.0], [1, 0.0], [2, 40.0]] * 20 for _ in range(3)]
base = ExponentialMarkovBaseline().fit(seqs)
sums = [sum(base.event_row(p).values()) for p in (None, 0, 1, 2)]
check("baseline: every next-event row sums to 1", all(abs(s - 1) < 1e-12 for s in sums),
      "row sums " + ", ".join(f"{s:.6f}" for s in sums))

# ---- 5. conditioning prefix: chronological stream first, then the prefix ------------------------------------
if HAVE_O2DES:
    stream = _customers_to_event_dt(simulate_gg1_o2des(2, 40, 2, 25, n_customers=300, seed=999))
    prefix = stream[:400]
    check("a 400-event conditioning prefix really has 400 events", len(prefix) == 400 and validate_stream(prefix)["valid"],
          f"{len(prefix)} events, valid={validate_stream(prefix)['valid']}")

# ---- 6. PyTorch checks: likelihood vs sampling, masking ------------------------------------------------------
try:
    import torch
    from model import QueueTransformer
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

if HAVE_TORCH:
    torch.manual_seed(0)
    model = QueueTransformer(d_model=16, nhead=2, num_layers=1, time_mode="lognormal")
    for mu_v, ls_v in [(0.0, 0.0), (1.0, -0.5), (-1.5, 0.3), (3.0, -1.0)]:
        out = torch.zeros(1, 6)
        out[0, 0], out[0, 1] = mu_v, ls_v                     # event type 0: mu, log_sigma
        ev = torch.zeros(200000, dtype=torch.long)
        draws = model.sample_dt(out.expand(200000, 6), ev)
        frac_zero = (draws <= 1e-3).float().mean().item()
        sigma = math.exp(ls_v)
        p0 = 0.5 * math.erfc((mu_v / sigma) / math.sqrt(2))   # Phi(-mu/sigma)
        nll0 = model.time_nll(out, torch.zeros(1, dtype=torch.long), torch.zeros(1)).item()
        # total probability: point mass + integral of the density over dt > 0 must be 1
        grid = torch.logspace(-6, 7, 400000, dtype=torch.float64)
        nll_pos = model.time_nll(out.double().expand(len(grid), 6), torch.zeros(len(grid), dtype=torch.long), grid,
                                 reduction="none")
        density = torch.exp(-nll_pos)
        integral = torch.trapz(density, grid).item()
        total = math.exp(-nll0) + integral
        ok = abs(frac_zero - p0) < 0.01 and abs(nll0 + math.log(p0)) < 1e-4 and abs(total - 1) < 5e-3
        check(f"time likelihood matches sampling (mu={mu_v}, log_sigma={ls_v})", ok,
              f"sampled zero share {frac_zero:.3f} vs {p0:.3f}; zero-gap NLL {nll0:.4f}; mass+density = {total:.4f}")

    from queue_rules import allowed_events
    logits = torch.tensor([[2.0, 1.0, -3.0]])
    forbidden = torch.tensor([[False, True, False]])
    masked = torch.softmax(logits.masked_fill(forbidden, float("-inf")), dim=-1)
    ref = torch.softmax(logits[:, [0, 2]], dim=-1)
    check("masking before softmax gives the exact conditional distribution",
          masked[0, 1].item() == 0.0 and abs(masked[0, 0].item() - ref[0, 0].item()) < 1e-7)
    extreme = torch.tensor([[-200.0, 100.0, -300.0]]).masked_fill(torch.tensor([[False, True, False]]), float("-inf"))
    p = torch.softmax(extreme, dim=-1)
    check("masking survives extreme logits (no uniform fallback needed)", abs(p.sum().item() - 1) < 1e-6 and p[0, 0] > 0.99,
          f"probabilities {[round(v, 4) for v in p[0].tolist()]}")
else:
    check("PyTorch checks (likelihood vs sampling, masking)", False, "SKIPPED: PyTorch not installed")

# ---- report --------------------------------------------------------------------------------------------------
width = max(len(n) for n, _, _ in results)
failed = 0
for name, ok, detail in results:
    skipped = detail.startswith("SKIPPED")
    status = "SKIP" if skipped else ("PASS" if ok else "FAIL")
    failed += (status == "FAIL")
    print(f"{status}  {name:<{width}}  {detail}")
print(f"\n{sum(1 for _, ok, _ in results if ok)} passed, {failed} failed, "
      f"{sum(1 for _, _, d in results if d.startswith('SKIPPED'))} skipped")
sys.exit(1 if failed else 0)
