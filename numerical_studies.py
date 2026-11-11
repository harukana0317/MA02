from queue_rules import VALID_MODES
import argparse
import csv
import random
import statistics
from collections import defaultdict
from datetime import datetime

try:
    from scipy import stats as scipy_stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


# =====================================================
# 1. LOAD REAL DATA (from data_logger.html's CSV export)
# =====================================================

def load_real_customers(csv_path):
    """
    Reads a CSV with columns: system_id, customer_id, event_type, timestamp
    (exactly what data_logger.html exports) and reconstructs each
    customer's arrival / order / leave times.

    Returns: dict of {system_id: [ {arrival, order, leave}, ... ]}
    Only customers with BOTH an arrival and a leave time are included
    (a customer who never left cannot contribute a cycle time).
    """
    by_system_cust = defaultdict(lambda: defaultdict(dict))

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sys_id = row["system_id"].strip()
            cust_id = row["customer_id"].strip()
            etype = row["event_type"].strip().upper()
            ts = datetime.strptime(row["timestamp"].strip(), "%Y-%m-%d %H:%M:%S")
            key = {"A": "arrival", "O": "order", "D": "leave"}.get(etype)
            if key:
                by_system_cust[sys_id][cust_id][key] = ts

    result = {}
    for sys_id, custs in by_system_cust.items():
        records = []
        for cust_id, times in custs.items():
            if "arrival" in times and "leave" in times:
                records.append(times)
        result[sys_id] = records
    return result


# =====================================================
# 2. SIMULATE CUSTOMERS (DES, keeping per-customer identity)
# =====================================================

def simulate_customers(n_servers, arrival_rate, service_rate, n_customers=200, seed=None):
    """
    Discrete-event simulation of a G/G/n queue, returning one record per
    customer: {arrival, order, leave} as floats (seconds from t=0).

    This is the CLASSICAL queueing-theory structure: n servers work
    independently and in parallel, each one capable of handling a
    customer from start to finish on its own (like n identical,
    interchangeable checkout counters).

    arrival_rate: customers per unit time (e.g. per minute)
    service_rate: customers a single server can finish per unit time
    """
    rng = random.Random(seed)

    t = 0.0
    arrivals = []
    for _ in range(n_customers):
        t += rng.expovariate(arrival_rate)
        arrivals.append(t)

    servers = [0.0] * n_servers
    records = []
    for arrival_time in arrivals:
        server_idx = min(range(n_servers), key=lambda i: servers[i])
        start_time = max(arrival_time, servers[server_idx])
        service_time = rng.expovariate(service_rate)
        leave_time = start_time + service_time
        servers[server_idx] = leave_time

        # "Order" placed shortly after arrival, before service starts/ends
        order_time = arrival_time + (leave_time - arrival_time) * rng.uniform(0.05, 0.3)

        records.append({"arrival": arrival_time, "order": order_time, "leave": leave_time})

    return records


def simulate_customers_gg1(arrival_shape, arrival_scale, service_shape, service_scale,
                            n_customers=200, seed=None):
    """
    STEP 2 SANITY-CHECK GROUND TRUTH — matches data.py's
    generate_gg1_event_log() but returns per-customer records directly
    (arrival/order/leave in seconds), for statistical comparison against
    the Transformer's generated output.

    Single server (genuine contention/waiting), Gamma-distributed
    inter-arrival and service times — NOT restricted to the memoryless
    exponential case.
    """
    rng = random.Random(seed)

    arrivals = []
    t = 0.0
    for _ in range(n_customers):
        t += rng.gammavariate(arrival_shape, arrival_scale)
        arrivals.append(t)

    server_free = 0.0
    records = []
    for arrival_time in arrivals:
        start_time = max(arrival_time, server_free)
        service_time = rng.gammavariate(service_shape, service_scale)
        leave_time = start_time + service_time
        server_free = leave_time

        order_time = arrival_time + (leave_time - arrival_time) * rng.uniform(0.05, 0.3)

        records.append({"arrival": arrival_time, "order": order_time, "leave": leave_time})

    return records


def simulate_customers_random(arrival_rate, mean_wait, mean_service, n_customers=200, seed=None):
    """
    STEP 1 SANITY-CHECK GROUND TRUTH — matches data.py's
    generate_random_event_log() but returns per-customer records
    directly (arrival/order/leave in seconds), for statistical
    comparison against the Transformer's generated output.

    NO server, NO queueing: every customer's three event times are
    drawn independently of every other customer. This is the simplest
    possible ground truth — if the Transformer can't reproduce this,
    the training pipeline itself needs fixing before testing it on any
    real queueing structure.
    """
    rng = random.Random(seed)

    arrivals = []
    t = 0.0
    for _ in range(n_customers):
        t += rng.expovariate(arrival_rate)
        arrivals.append(t)

    records = []
    for arrival_time in arrivals:
        order_time = arrival_time + rng.expovariate(1.0 / mean_wait)
        leave_time = order_time + rng.expovariate(1.0 / mean_service)
        records.append({"arrival": arrival_time, "order": order_time, "leave": leave_time})

    return records


def simulate_customers_tandem(arrival_rate, rate_stage1, rate_stage2, n_customers=200, seed=None):
    """
    Discrete-event simulation of a TWO-STAGE TANDEM (series) queue —
    e.g. "one person checks the customer out, then a second person
    assembles the order" — instead of n independent, interchangeable
    servers.

    Every customer passes through BOTH stages IN SEQUENCE, and each
    stage has its own single server and its own queue:
      Stage 1 (e.g. checkout):        arrival -> order
      Stage 2 (e.g. sandwich-making): order   -> leave

    This is structurally different from simulate_customers(): there,
    a server can handle a customer start-to-finish alone; here, a
    customer must always go through stage 1 THEN stage 2, and can be
    held up waiting for stage 2 even after stage 1 is done.

    rate_stage1 / rate_stage2: service rate (customers/unit time) for
    each individual stage, estimated from real wait/service times.
    """
    rng = random.Random(seed)

    t = 0.0
    arrivals = []
    for _ in range(n_customers):
        t += rng.expovariate(arrival_rate)
        arrivals.append(t)

    server1_free = 0.0  # e.g. checkout
    server2_free = 0.0  # e.g. sandwich-making
    records = []
    for arrival_time in arrivals:
        start1 = max(arrival_time, server1_free)
        dur1 = rng.expovariate(rate_stage1)
        order_time = start1 + dur1
        server1_free = order_time

        # Stage 2 can only begin once stage 1 has finished for this
        # customer AND the stage-2 server is free.
        start2 = max(order_time, server2_free)
        dur2 = rng.expovariate(rate_stage2)
        leave_time = start2 + dur2
        server2_free = leave_time

        records.append({"arrival": arrival_time, "order": order_time, "leave": leave_time})

    return records


# =====================================================
# 2b. TRANSFORMER INTEGRATION (optional — only needed with
#     --transformer_checkpoint; the rest of this script works
#     without PyTorch installed)
# =====================================================

try:
    import torch  # noqa: F401
    from generate import load_model as _load_transformer_model
    from generate import generate as _transformer_generate
    HAVE_TRANSFORMER = True
except ImportError:
    HAVE_TRANSFORMER = False


def _customers_to_event_dt(customer_records, dt_key_order=("arrival", "order", "leave")):
    """
    Inverse of match_events_to_customers(): converts a list of customer
    records {arrival, order, leave} (real seconds OR datetimes) back
    into a flat, time-sorted [event_id, dt] sequence of floats — the
    format the Transformer expects as input. Used to build a REAL seed
    sequence to condition generation on, rather than an arbitrary
    hand-picked one.
    """
    customer_records = _to_seconds(customer_records)  # ensure floats, not datetimes
    events = []
    for c in customer_records:
        if "arrival" in c:
            events.append((0, c["arrival"]))
        if "order" in c:
            events.append((1, c["order"]))
        if "leave" in c:
            events.append((2, c["leave"]))
    events.sort(key=lambda x: x[1])

    seq = []
    prev_t = None
    for event_id, t in events:
        dt = 0.0 if prev_t is None else t - prev_t
        seq.append([event_id, dt])
        prev_t = t
    return seq


def match_events_to_customers(event_dt_sequence):
    """
    The Transformer does NOT track customer identity — it only predicts
    a flat stream of (event_id, dt) pairs, exactly like it was trained
    on. To compute per-customer metrics (wait/service/cycle time) from
    generated output, this function reconstructs an APPROXIMATE
    customer-level record by matching events in the order they occur,
    using a first-in-first-out (FIFO) rule at each stage:

      - An Arrival (0) opens a new customer, who is now "waiting to order"
      - An Ordering (1) event is matched to the customer who has been
        waiting to order the LONGEST (FIFO) — that customer moves to
        "waiting to leave"
      - A Departure (2) event is matched to the customer who has been
        waiting to leave the longest; if nobody has ordered yet but
        someone is still waiting to order, they're matched instead
        (representing a customer who left without ordering)

    This is a reasonable approximation when customers are served
    roughly in arrival order (true for a single queue), but is NOT
    exact — the Transformer's raw output has no ground-truth customer
    labels, so this matching is inherently a best-effort reconstruction,
    not a guaranteed-correct one. Treat results from generated data with
    this in mind; it's most meaningful in aggregate (distributions),
    less so for any single "customer's" reconstructed time.

    event_dt_sequence: list of [event_id, dt] as produced by generate.py
                        (0=Arrival, 1=Ordering, 2=Departure)
    Returns: list of {"arrival":..., "order":..., "leave":...} (seconds,
             float, cumulative from the start of the sequence)
    """
    t = 0.0
    waiting_to_order = []   # customers who arrived, no Order event yet (FIFO)
    waiting_to_leave = []   # customers who ordered, no Leave event yet (FIFO)
    records = []

    for event_id, dt in event_dt_sequence:
        t += dt
        if event_id == 0:  # Arrival
            waiting_to_order.append({"arrival": t})
        elif event_id == 1:  # Ordering
            if waiting_to_order:
                cust = waiting_to_order.pop(0)
                cust["order"] = t
                waiting_to_leave.append(cust)
            # else: an Order event with nobody waiting — a generation
            # artifact (the model predicted an implausible sequence).
            # Silently dropped; see the "unmatched events" count printed
            # by generate_transformer_customers() below.
        elif event_id == 2:  # Departure
            if waiting_to_leave:
                cust = waiting_to_leave.pop(0)
                cust["leave"] = t
                records.append(cust)
            elif waiting_to_order:
                # Interpreted as: customer left without ordering.
                cust = waiting_to_order.pop(0)
                cust["leave"] = t
                records.append(cust)
            # else: a Departure with nobody open — also a generation
            # artifact, silently dropped.

    return records


def generate_transformer_customers(checkpoint_path, real_seed_events, n_events=600, rng_seed=None,
                                    constrain=None):
    """
    Loads a trained Transformer checkpoint, autoregressively generates
    `n_events` steps continuing from a real seed sequence, and
    reconstructs approximate customer records via match_events_to_customers().

    real_seed_events: list of [event_id, dt] (real units) to condition
                       generation on — e.g. the first seq_len events of
                       a real session. Using a real seed (not
                       an arbitrary fixed one) means generation starts
                       from a realistic system state.
    """
    if not HAVE_TRANSFORMER:
        raise RuntimeError(
            "PyTorch / generate.py are not importable in this environment. "
            "Transformer-based comparisons need a machine with PyTorch "
            "installed (pip install torch) and generate.py + model.py "
            "in the same folder as this script."
        )

    model, dt_mean, dt_std, seq_len = _load_transformer_model(checkpoint_path)
    seed = real_seed_events[:seq_len] if len(real_seed_events) >= seq_len else real_seed_events

    generated = _transformer_generate(model, seed, dt_mean, dt_std, seq_len,
                                       steps=n_events, rng_seed=rng_seed,
                                       constrain=constrain)
    records = match_events_to_customers(generated)

    n_unmatched = (n_events + len(seed)) - (len(records) * 3) - len(
        [c for c in records if "order" not in c]
    )
    return records


# =====================================================
# 3. METRICS: arrival rate, service rate, wait/service/cycle time
# =====================================================

def _to_seconds(records):
    """Convert datetime-based records to float-seconds-from-first-arrival,
    so real and simulated data can be analysed with the same code."""
    if not records:
        return records
    if isinstance(records[0]["arrival"], (int, float)):
        return records  # already numeric (simulated data)
    t0 = min(r["arrival"] for r in records)
    out = []
    for r in records:
        out.append({k: (v - t0).total_seconds() for k, v in r.items()})
    return out


def _summarize(vals):
    if not vals:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(vals),
        "mean": statistics.mean(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


def compute_metrics(records):
    """
    records: list of {arrival, order, leave} (seconds, float)
    Returns a dict of summary statistics.

    NOTE: only valid for records drawn from a SINGLE, CONTIGUOUS
    observation session. For multiple sessions with gaps between them
    (e.g. Monday's lunch peak + Tuesday's dinner peak), use
    compute_pooled_metrics() instead — pooling records directly here
    would treat the gap between sessions as part of the observation
    span and silently produce a wrong (too-low) arrival rate.
    """
    records = _to_seconds(records)
    records = sorted(records, key=lambda r: r["arrival"])
    n = len(records)
    if n < 2:
        return None

    arrivals = [r["arrival"] for r in records]
    interarrival = [b - a for a, b in zip(arrivals[:-1], arrivals[1:])]
    wait = [r["order"] - r["arrival"] for r in records if "order" in r]
    service = [r["leave"] - r["order"] for r in records if "order" in r]
    cycle = [r["leave"] - r["arrival"] for r in records]

    span_seconds = arrivals[-1] - arrivals[0]
    span_minutes = span_seconds / 60.0 if span_seconds > 0 else float("nan")

    return {
        "n_customers": n,
        "observation_span_minutes": span_minutes,
        "arrival_rate_per_min": (n / span_minutes) if span_minutes else None,
        "service_rate_per_min": (60.0 / statistics.mean(service)) if service else None,
        "interarrival_seconds": _summarize(interarrival),
        "wait_seconds": _summarize(wait),
        "service_seconds": _summarize(service),
        "cycle_seconds": _summarize(cycle),
        "_raw_cycle": cycle,          # kept for statistical tests below
        "_raw_service": service,
        "_raw_wait": wait,
        "_raw_interarrival": interarrival,
    }


def compute_pooled_metrics(by_system):
    """
    Correctly combines metrics across MULTIPLE observation sessions that
    may be recorded on different days/times with gaps in between.

    Two things must NOT be computed naively by flattening all customers
    into one list and reusing compute_metrics():
      1. Arrival rate: (total customers) / (last arrival - first arrival)
         would divide by the full wall-clock range across all sessions,
         including the hours/days when nobody was being observed. This
         silently produces an arrival rate far too low. Instead, each
         session's own duration is summed, and the gaps between
         sessions are excluded entirely.
      2. Interarrival time: the gap between the LAST customer of one
         session and the FIRST customer of the next session is not a
         real interarrival time (it's just "how long until we started
         observing again") and must not be included.

    Wait / service / cycle times ARE valid to pool directly, since each
    is a property of a single customer within their own session and
    doesn't depend on session boundaries.
    """
    total_customers = 0
    total_span_minutes = 0.0
    all_interarrival, all_wait, all_service, all_cycle = [], [], [], []

    for sys_id, records in by_system.items():
        m = compute_metrics(records)
        if m is None:
            continue
        total_customers += m["n_customers"]
        total_span_minutes += m["observation_span_minutes"]
        all_interarrival.extend(m["_raw_interarrival"])
        all_wait.extend(m["_raw_wait"])
        all_service.extend(m["_raw_service"])
        all_cycle.extend(m["_raw_cycle"])

    if total_customers == 0 or total_span_minutes == 0:
        return None

    return {
        "n_customers": total_customers,
        "observation_span_minutes": total_span_minutes,
        "arrival_rate_per_min": total_customers / total_span_minutes,
        "service_rate_per_min": (60.0 / statistics.mean(all_service)) if all_service else None,
        "interarrival_seconds": _summarize(all_interarrival),
        "wait_seconds": _summarize(all_wait),
        "service_seconds": _summarize(all_service),
        "cycle_seconds": _summarize(all_cycle),
        "_raw_cycle": all_cycle,
        "_raw_service": all_service,
        "_raw_wait": all_wait,
        "_raw_interarrival": all_interarrival,
    }


# =====================================================
# 4. STATISTICAL COMPARISON: real vs simulated
# =====================================================

def compare_distributions(group_a_vals, group_b_vals, label, group_a_name="Real", group_b_name="Simulated"):
    print(f"\n--- {label}: {group_a_name} vs. {group_b_name} ---")
    if len(group_a_vals) < 2 or len(group_b_vals) < 2:
        print("  Not enough data points to run a statistical test (need >= 2 in each group).")
        return

    real_mean = statistics.mean(group_a_vals)
    sim_mean = statistics.mean(group_b_vals)
    print(f"  {group_a_name}:{' ' * max(1, 10 - len(group_a_name))}n={len(group_a_vals):4d}  mean={real_mean:8.2f}s  std={statistics.stdev(group_a_vals):8.2f}s")
    print(f"  {group_b_name}:{' ' * max(1, 10 - len(group_b_name))}n={len(group_b_vals):4d}  mean={sim_mean:8.2f}s  std={statistics.stdev(group_b_vals):8.2f}s")

    if not HAVE_SCIPY:
        print("  (install scipy to also get p-values: pip install scipy --break-system-packages)")
        return

    t_stat, t_p = scipy_stats.ttest_ind(group_a_vals, group_b_vals, equal_var=False)
    ks_stat, ks_p = scipy_stats.ks_2samp(group_a_vals, group_b_vals)

    print(f"  Welch's t-test:  t={t_stat:.3f}, p={t_p:.4f}  "
          f"({'means are NOT significantly different' if t_p > 0.05 else 'means ARE significantly different'} at alpha=0.05)")
    print(f"  KS test:         D={ks_stat:.3f}, p={ks_p:.4f}  "
          f"({'distributions look consistent' if ks_p > 0.05 else 'distributions look significantly different'} at alpha=0.05)")


# =====================================================
# 5. PARAMETER SWEEP ("plenty of numerical studies")
# =====================================================

def test_exponential_fit(vals, label):
    """
    Tests whether `vals` (e.g. real inter-arrival times or service times)
    are consistent with an EXPONENTIAL distribution — the core assumption
    behind classical M/M/n queueing theory (memoryless inter-event times).

    Fits the best-matching exponential (rate = 1/mean) to the data, then
    runs a one-sample Kolmogorov-Smirnov test against it. A small p-value
    means the data does NOT look exponential — i.e., direct numerical
    evidence against the classical queueing-theory assumption.
    """
    print(f"\n--- Exponential goodness-of-fit: {label} ---")
    if len(vals) < 8:
        print("  Not enough data points for a meaningful fit test (need >= 8).")
        return
    if not HAVE_SCIPY:
        print("  (install scipy to run this test: pip install scipy --break-system-packages)")
        return

    mean_val = statistics.mean(vals)
    rate = 1.0 / mean_val
    ks_stat, ks_p = scipy_stats.kstest(vals, "expon", args=(0, mean_val))
    verdict = "CONSISTENT with exponential" if ks_p > 0.05 else "NOT consistent with exponential"
    print(f"  n={len(vals)}, mean={mean_val:.1f}s, fitted rate={rate*60:.3f}/min")
    print(f"  KS test vs. Exponential(mean={mean_val:.1f}s): D={ks_stat:.3f}, p={ks_p:.4f}")
    print(f"  -> Data is {verdict} at alpha=0.05")
    return {"n": len(vals), "mean": mean_val, "ks_stat": ks_stat, "ks_p": ks_p, "consistent": ks_p > 0.05}


def parameter_sweep(base_arrival_rate, base_service_rate, n_servers_list=(1, 2, 3),
                     arrival_multipliers=(0.5, 1.0, 1.5), n_customers=300, seed=0):
    """
    Re-runs the simulation across a grid of settings and tabulates the
    resulting cycle time. This demonstrates how the system behaves under
    different loads (a classic "numerical study" in queueing analysis) —
    for example, showing that cycle time grows sharply as arrival rate
    approaches service capacity.
    """
    print("\n=== Parameter sweep: how cycle time changes with load ===")
    print(f"{'servers':>8} {'arrival/min':>12} {'utilisation':>12} {'mean cycle (s)':>16}")
    for n_servers in n_servers_list:
        for mult in arrival_multipliers:
            arrival_rate = base_arrival_rate * mult
            records = simulate_customers(n_servers, arrival_rate, base_service_rate,
                                          n_customers=n_customers, seed=seed)
            metrics = compute_metrics(records)
            utilisation = arrival_rate / (n_servers * base_service_rate)
            mean_cycle = metrics["cycle_seconds"]["mean"] if metrics else float("nan")
            print(f"{n_servers:>8} {arrival_rate*60:>12.2f} {utilisation:>12.2f} {mean_cycle:>16.1f}")
    print("\n('utilisation' = arrival rate / total service capacity. As it approaches 1.0,")
    print(" cycle time should grow sharply — this is the classic queueing 'knee' effect.)")


def tandem_parameter_sweep(arrival_rate, rate_stage1, rate_stage2_list, n_customers=300, seed=0):
    """
    Sweeps stage-2 capacity (e.g. sandwich-making speed) while holding
    arrival rate and stage-1 capacity fixed, to show how a single
    bottleneck stage dominates overall cycle time in a tandem system —
    a distinct dynamic from the parallel-server case, where adding ANY
    server helps. Here, only fixing the SPECIFIC bottleneck stage helps.
    """
    print("\n=== Tandem parameter sweep: effect of stage-2 (bottleneck) capacity ===")
    print(f"{'stage2 rate/min':>16} {'stage2 utilisation':>20} {'mean cycle (s)':>16}")
    for rate2 in rate_stage2_list:
        records = simulate_customers_tandem(arrival_rate, rate_stage1, rate2,
                                             n_customers=n_customers, seed=seed)
        metrics = compute_metrics(records)
        utilisation2 = arrival_rate / rate2
        mean_cycle = metrics["cycle_seconds"]["mean"] if metrics else float("nan")
        print(f"{rate2*60:>16.2f} {utilisation2:>20.2f} {mean_cycle:>16.1f}")
    print("\n(In a tandem system, speeding up a NON-bottleneck stage barely helps —")
    print(" only increasing capacity at the slowest stage reduces overall cycle time.)")


def repeated_comparison(real_vals, sim_fn, n_repeats=10, label=""):
    """
    Runs sim_fn() (a zero-argument callable returning a list of raw
    values, e.g. simulated cycle times) N_REPEATS times under different
    random seeds, and reports the mean simulated value ± its standard
    deviation across repeats, alongside the gap to the real mean.

    This directly answers "how much of the earlier single-run gap was
    just random chance?" — a single simulation run (one seed) could
    happen to land closer to or further from the real data purely by
    luck. Repeating it and reporting mean ± std is the standard way to
    report simulation output with proper uncertainty.
    """
    real_mean = statistics.mean(real_vals)
    sim_means = []
    for seed in range(n_repeats):
        vals = sim_fn(seed)
        if vals:
            sim_means.append(statistics.mean(vals))

    if not sim_means:
        print(f"\n--- {label}: no simulated data produced across {n_repeats} repeats ---")
        return None

    grand_mean = statistics.mean(sim_means)
    grand_std = statistics.stdev(sim_means) if len(sim_means) > 1 else 0.0
    gap = grand_mean - real_mean

    print(f"\n--- {label}: repeated over {len(sim_means)} random seeds ---")
    print(f"  Real mean:                {real_mean:8.1f}s")
    print(f"  Simulated mean (across {len(sim_means)} runs): {grand_mean:8.1f}s  ± {grand_std:.1f}s (1 std dev across runs)")
    print(f"  Gap:                      {gap:+8.1f}s")
    if HAVE_SCIPY and len(sim_means) > 1:
        t_stat, t_p = scipy_stats.ttest_1samp(sim_means, real_mean)
        print(f"  One-sample t-test (H0: simulated runs average to the real mean): "
              f"t={t_stat:.3f}, p={t_p:.4f}  "
              f"({'consistent with real mean' if t_p > 0.05 else 'simulated mean reliably differs from real mean'})")
    return {"real_mean": real_mean, "sim_grand_mean": grand_mean, "sim_std": grand_std, "gap": gap}


# =====================================================
# 6. REPORT PRINTING
# =====================================================

def print_metrics_table(label, metrics, n_servers=None):
    print(f"\n--- {label} ---")
    if metrics is None:
        print("  Not enough data (need at least 2 customers with both arrival and leave times).")
        return
    print(f"  Customers observed:      {metrics['n_customers']}")
    print(f"  Observation span:        {metrics['observation_span_minutes']:.1f} minutes")
    print(f"  Arrival rate:            {metrics['arrival_rate_per_min']:.3f} customers/min")
    if metrics['service_rate_per_min']:
        print(f"  Service rate:            {metrics['service_rate_per_min']:.3f} customers/min per server")
    for name, key in [("Wait time (Arrival->Order)", "wait_seconds"),
                       ("Service time (Order->Leave)", "service_seconds"),
                       ("Cycle time (Arrival->Leave)", "cycle_seconds")]:
        s = metrics[key]
        if s["n"]:
            print(f"  {name}: mean={s['mean']:.1f}s  std={s['std']:.1f}s  "
                  f"min={s['min']:.1f}s  max={s['max']:.1f}s  (n={s['n']})")

    # ---- Little's Law: L = lambda * W ----
    # This makes NO distributional assumption (not exponential, not
    # anything) — it holds for ANY stable queueing system. It's a
    # purely empirical cross-check computed directly from the observed
    # arrival rate and mean time-in-system.
    cycle_mean_min = metrics["cycle_seconds"]["mean"] / 60.0 if metrics["cycle_seconds"]["n"] else None
    if cycle_mean_min is not None:
        L = metrics["arrival_rate_per_min"] * cycle_mean_min
        print(f"  Little's Law (L = arrival_rate x mean cycle time): "
              f"~{L:.2f} customers in the system on average (waiting + being served)")

    # ---- Theoretical utilisation, if n_servers is known ----
    if n_servers and metrics["service_rate_per_min"]:
        rho = metrics["arrival_rate_per_min"] / (n_servers * metrics["service_rate_per_min"])
        print(f"  Theoretical utilisation with {n_servers} server(s): rho = {rho:.2f}", end="")
        if rho >= 1.0:
            print(" -- >= 1.0 means a classical M/M/n model predicts an EVER-GROWING queue.")
            print(f"     If real cycle times stayed bounded despite this, that's evidence the real")
            print(f"     system behaves differently from the classical independent-exponential-server")
            print(f"     assumption (e.g. staff working in parallel/pipeline, or speeding up under load).")
        else:
            print(" -- < 1.0, consistent with a stable classical queueing system at this server count.")


def save_summary_csv(all_metrics, output_path):
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "n_customers", "arrival_rate_per_min", "service_rate_per_min",
                    "mean_wait_s", "mean_service_s", "mean_cycle_s"])
        for label, m in all_metrics.items():
            if m is None:
                continue
            w.writerow([label, m["n_customers"], m["arrival_rate_per_min"], m["service_rate_per_min"],
                        m["wait_seconds"]["mean"], m["service_seconds"]["mean"], m["cycle_seconds"]["mean"]])
    print(f"\nSaved summary table to {output_path}")


# =====================================================
# 7. MAIN
# =====================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_csv", type=str, default=None,
                         help="Path to the CSV exported by data_logger.html "
                              "(columns: system_id, customer_id, event_type, timestamp)")
    parser.add_argument("--output", type=str, default="numerical_studies_summary.csv")
    parser.add_argument("--n_servers", type=int, default=1,
                         help="Number of servers/counters to assume for the simulation")
    parser.add_argument("--transformer_checkpoint", type=str, default=None,
                         help="Path to a checkpoint.pt from train.py. If given, the "
                              "trained Transformer's generated output is compared "
                              "against real data using the same statistical tests "
                              "as the parallel-server and tandem models.")
    parser.add_argument("--constrain", choices=["off"] + list(VALID_MODES), default="off",
                         help="Constrained generation for the Transformer (hard rule: forbid "
                              "events impossible in the current state). Use 'allow_balk' for "
                              "real data (a few customers leave without an Order event). "
                              "This is a hard rule, not something the model learned.")
    args = parser.parse_args()
    constrain_mode = None if args.constrain == "off" else args.constrain

    all_metrics = {}

    if args.real_csv:
        print(f"Loading real data from {args.real_csv} ...")
        by_system = load_real_customers(args.real_csv)
        print(f"Found {len(by_system)} system(s): {list(by_system.keys())}")

        # Pool all systems together for an overall real-data summary.
        # (uses compute_pooled_metrics, NOT compute_metrics, so that gaps
        # between sessions recorded on different days don't get counted
        # as part of the observation time — see its docstring for why.)
        real_metrics = compute_pooled_metrics(by_system)
        print_metrics_table("REAL DATA (all sessions pooled)", real_metrics, n_servers=args.n_servers)
        all_metrics["real_pooled"] = real_metrics

        # Also show each session individually.
        for sys_id, custs in by_system.items():
            m = compute_metrics(custs)
            print_metrics_table(f"REAL DATA - session '{sys_id}'", m, n_servers=args.n_servers)
            all_metrics[f"real_{sys_id}"] = m

        if real_metrics is None:
            print("\nNot enough real data to run comparisons or a parameter sweep. Exiting.")
            save_summary_csv(all_metrics, args.output)
            return

        # ---- Estimate matched simulation parameters FROM the real data ----
        arrival_rate_per_sec = real_metrics["arrival_rate_per_min"] / 60.0
        service_rate_per_sec = real_metrics["service_rate_per_min"] / 60.0 if real_metrics["service_rate_per_min"] else 1.0 / 60.0

        print(f"\nEstimated from real data: arrival_rate={arrival_rate_per_sec*60:.3f}/min, "
              f"service_rate={service_rate_per_sec*60:.3f}/min per server, n_servers={args.n_servers}")

        sim_records = simulate_customers(args.n_servers, arrival_rate_per_sec, service_rate_per_sec,
                                          n_customers=max(real_metrics["n_customers"], 100), seed=0)
        sim_metrics = compute_metrics(sim_records)
        print_metrics_table("SIMULATED DATA - PARALLEL SERVERS (classic M/M/n)", sim_metrics, n_servers=args.n_servers)
        all_metrics["simulated_parallel"] = sim_metrics

        # ---- Statistical comparison: classic parallel-server model ----
        print("\n#################################################")
        print("# PARALLEL-SERVER MODEL vs REAL DATA")
        print("#################################################")
        compare_distributions(real_metrics["_raw_cycle"], sim_metrics["_raw_cycle"], "Cycle time")
        compare_distributions(real_metrics["_raw_service"], sim_metrics["_raw_service"], "Service time")
        if real_metrics["_raw_wait"] and sim_metrics["_raw_wait"]:
            compare_distributions(real_metrics["_raw_wait"], sim_metrics["_raw_wait"], "Wait time")

        # ---- Tandem (series) structure: e.g. "one person checks out,
        #      a second person assembles the order" ----
        # rate_stage1 estimated from mean wait time (arrival->order),
        # rate_stage2 estimated from mean service time (order->leave) —
        # same rate estimates as the parallel model above, but plugged
        # into a fundamentally different queueing STRUCTURE. This
        # isolates whether structure (not parameters) explains the gap.
        # NOTE ON RATE ESTIMATION:
        # A naive estimate would use rate = 1 / mean(observed duration).
        # But the observed wait/service duration ALREADY INCLUDES its own
        # queueing delay (customers waiting for that stage's single
        # server to free up) — it is not pure processing time. Treating
        # it as pure processing time systematically UNDERESTIMATES the
        # server's true capacity, which can make the simulated stage
        # artificially unstable (arrivals outpace an underestimated
        # service rate) and wildly inflate simulated times.
        #
        # Instead, we invert the standard M/M/1 mean-time-in-system
        # formula  W = 1 / (mu - lambda)  to solve for the TRUE service
        # rate mu, given the observed mean sojourn time W and the known
        # arrival rate lambda into that stage:
        #     mu = lambda + 1/W
        # This guarantees mu > lambda (a stable queue) by construction,
        # and correctly separates "time spent queueing" from "time spent
        # actually being served."
        mean_wait = real_metrics["wait_seconds"]["mean"]
        mean_service = real_metrics["service_seconds"]["mean"]
        rate_stage1 = arrival_rate_per_sec + (1.0 / mean_wait) if mean_wait else arrival_rate_per_sec + 1.0
        rate_stage2 = arrival_rate_per_sec + (1.0 / mean_service) if mean_service else arrival_rate_per_sec + 1.0

        print(f"\nEstimated for tandem model: stage1 (e.g. checkout) rate={rate_stage1*60:.3f}/min, "
              f"stage2 (e.g. sandwich-making) rate={rate_stage2*60:.3f}/min")

        tandem_records = simulate_customers_tandem(arrival_rate_per_sec, rate_stage1, rate_stage2,
                                                     n_customers=max(real_metrics["n_customers"], 100), seed=0)
        tandem_metrics = compute_metrics(tandem_records)
        print_metrics_table("SIMULATED DATA - TANDEM / SERIES STRUCTURE (e.g. checkout -> sandwich-making)",
                             tandem_metrics)
        all_metrics["simulated_tandem"] = tandem_metrics

        print("\n#################################################")
        print("# TANDEM MODEL vs REAL DATA")
        print("#################################################")
        compare_distributions(real_metrics["_raw_cycle"], tandem_metrics["_raw_cycle"], "Cycle time")
        compare_distributions(real_metrics["_raw_service"], tandem_metrics["_raw_service"], "Service time")
        if real_metrics["_raw_wait"] and tandem_metrics["_raw_wait"]:
            compare_distributions(real_metrics["_raw_wait"], tandem_metrics["_raw_wait"], "Wait time")

        # =========================================================
        # TRANSFORMER MODEL (only if --transformer_checkpoint given)
        # =========================================================
        transformer_metrics = None
        if args.transformer_checkpoint:
            print("\n#################################################")
            print("# TRANSFORMER MODEL vs REAL DATA")
            print("#################################################")
            if not HAVE_TRANSFORMER:
                print("  PyTorch / generate.py not available in this environment — skipping. "
                      "Run this on a machine with `pip install torch` and generate.py/model.py present.")
            else:
                # Condition generation on a real seed sequence (the first
                # window's worth of events from the largest real session),
                # so generation starts from a realistic state rather than
                # an arbitrary hand-picked one.
                largest_session = max(by_system.values(), key=len)
                # convert that session's customer records into the flat
                # (event_id, dt) format the Transformer was trained on
                seed_events = _customers_to_event_dt(largest_session)

                n_gen_events = max(real_metrics["n_customers"], 50) * 3  # ~3 events/customer
                transformer_records = generate_transformer_customers(
                    args.transformer_checkpoint, seed_events,
                    n_events=n_gen_events, rng_seed=0, constrain=constrain_mode,
                )
                transformer_metrics = compute_metrics(transformer_records)
                print_metrics_table("SIMULATED DATA - TRANSFORMER (trained on the real data)",
                                     transformer_metrics)
                if transformer_metrics is None:
                    print("  Transformer generated too few complete (arrival+order+leave) "
                          "customer records to compute metrics — try a longer --steps / "
                          "more training epochs.")
                else:
                    all_metrics["simulated_transformer"] = transformer_metrics
                    compare_distributions(real_metrics["_raw_cycle"], transformer_metrics["_raw_cycle"], "Cycle time")
                    compare_distributions(real_metrics["_raw_service"], transformer_metrics["_raw_service"], "Service time")
                    if real_metrics["_raw_wait"] and transformer_metrics["_raw_wait"]:
                        compare_distributions(real_metrics["_raw_wait"], transformer_metrics["_raw_wait"], "Wait time")

                    print("\n--- Repeated Transformer generation across 10 random seeds ---")

                    def _transformer_cycle_times(rng_seed):
                        recs = generate_transformer_customers(
                            args.transformer_checkpoint, seed_events,
                            n_events=n_gen_events, rng_seed=rng_seed, constrain=constrain_mode)
                        m = compute_metrics(recs)
                        return m["_raw_cycle"] if m else []

                    repeated_comparison(
                        real_metrics["_raw_cycle"], _transformer_cycle_times,
                        n_repeats=10, label="Transformer: cycle time",
                    )

        print("\n#################################################")
        print("# SUMMARY: which structure fits better?")
        print("#################################################")
        real_cycle_mean = real_metrics["cycle_seconds"]["mean"]
        parallel_cycle_mean = sim_metrics["cycle_seconds"]["mean"]
        tandem_cycle_mean = tandem_metrics["cycle_seconds"]["mean"]
        print(f"  Real mean cycle time:                {real_cycle_mean:8.1f}s")
        print(f"  Parallel-server model mean cycle:    {parallel_cycle_mean:8.1f}s  "
              f"(off by {abs(parallel_cycle_mean - real_cycle_mean):8.1f}s)")
        print(f"  Tandem/series model mean cycle:      {tandem_cycle_mean:8.1f}s  "
              f"(off by {abs(tandem_cycle_mean - real_cycle_mean):8.1f}s)")
        candidates = {
            "parallel-server": abs(parallel_cycle_mean - real_cycle_mean),
            "tandem": abs(tandem_cycle_mean - real_cycle_mean),
        }
        if transformer_metrics is not None:
            transformer_cycle_mean = transformer_metrics["cycle_seconds"]["mean"]
            print(f"  Transformer model mean cycle:        {transformer_cycle_mean:8.1f}s  "
                  f"(off by {abs(transformer_cycle_mean - real_cycle_mean):8.1f}s)")
            candidates["Transformer"] = abs(transformer_cycle_mean - real_cycle_mean)
        closer = min(candidates, key=candidates.get)
        print(f"  -> The {closer} structure is closest to the real data on mean cycle time.")

        # =========================================================
        # STUDY 2: Repeated simulation across multiple random seeds
        # (turns the single-run "gap" number into mean +/- std dev,
        # so the comparison isn't resting on one lucky/unlucky draw)
        # =========================================================
        print("\n#################################################")
        print("# NUMERICAL STUDY 2: repeated runs (multiple random seeds)")
        print("#################################################")
        n_target = max(real_metrics["n_customers"], 100)

        repeated_comparison(
            real_metrics["_raw_cycle"],
            lambda seed: compute_metrics(
                simulate_customers(args.n_servers, arrival_rate_per_sec, service_rate_per_sec,
                                    n_customers=n_target, seed=seed))["_raw_cycle"],
            n_repeats=10, label="Parallel-server model: cycle time",
        )
        repeated_comparison(
            real_metrics["_raw_cycle"],
            lambda seed: compute_metrics(
                simulate_customers_tandem(arrival_rate_per_sec, rate_stage1, rate_stage2,
                                           n_customers=n_target, seed=seed))["_raw_cycle"],
            n_repeats=10, label="Tandem model: cycle time",
        )

        # =========================================================
        # STUDY 3: Does the real data even look exponential?
        # (tests the core assumption behind BOTH simulated models,
        # and behind the classical M/M/n / exponential-Markov baseline)
        # =========================================================
        print("\n#################################################")
        print("# NUMERICAL STUDY 3: is the real data actually exponential?")
        print("#################################################")
        test_exponential_fit(real_metrics["_raw_interarrival"], "Real inter-arrival times")
        test_exponential_fit(real_metrics["_raw_wait"], "Real wait times (Arrival->Order)")
        test_exponential_fit(real_metrics["_raw_service"], "Real service times (Order->Leave)")

        # =========================================================
        # STUDY 4: Tandem bottleneck sweep
        # =========================================================
        print("\n#################################################")
        print("# NUMERICAL STUDY 4: tandem bottleneck sensitivity")
        print("#################################################")
        tandem_parameter_sweep(
            arrival_rate_per_sec, rate_stage1,
            rate_stage2_list=[rate_stage2 * m for m in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)],
        )

        # =========================================================
        # STUDY 5: Peak vs. non-peak sessions (using the session
        # labels, e.g. "12:19-13:19 peak" vs "13:21-14:21 non peak")
        # =========================================================
        print("\n#################################################")
        print("# NUMERICAL STUDY 5: peak vs. non-peak sessions")
        print("#################################################")
        peak_systems = {k: v for k, v in by_system.items() if "non peak" not in k.lower() and "peak" in k.lower()}
        nonpeak_systems = {k: v for k, v in by_system.items() if "non peak" in k.lower()}
        if peak_systems and nonpeak_systems:
            peak_metrics = compute_pooled_metrics(peak_systems)
            nonpeak_metrics = compute_pooled_metrics(nonpeak_systems)
            print_metrics_table(f"PEAK sessions ({list(peak_systems.keys())})", peak_metrics)
            print_metrics_table(f"NON-PEAK sessions ({list(nonpeak_systems.keys())})", nonpeak_metrics)
            compare_distributions(peak_metrics["_raw_cycle"], nonpeak_metrics["_raw_cycle"],
                                   "Cycle time", group_a_name="Peak", group_b_name="Non-peak")
        else:
            print("  Skipped — no sessions in this CSV have 'peak' / 'non peak' in their name.")
            print("  (This study only runs on labelled sessions.)")

        # ---- Parameter sweep using the real-data-estimated baseline ----
        parameter_sweep(arrival_rate_per_sec, service_rate_per_sec)

    else:
        print("No --real_csv provided. Running a demo using simulated data only.\n")
        demo_records = simulate_customers(n_servers=2, arrival_rate=0.05, service_rate=0.03, n_customers=200, seed=0)
        demo_metrics = compute_metrics(demo_records)
        print_metrics_table("DEMO: simulated data only", demo_metrics, n_servers=2)
        all_metrics["demo_simulated"] = demo_metrics
        parameter_sweep(0.05, 0.03)
        print("\nTo compare against real data, run again with:")
        print("    python numerical_studies.py --real_csv real_data.csv")

    save_summary_csv(all_metrics, args.output)


if __name__ == "__main__":
    main()
