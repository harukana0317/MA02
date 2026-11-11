import random
import csv
from collections import defaultdict

import torch
from torch.utils.data import Dataset


# =====================================================
# 1. QUEUE SIMULATOR (G/G/n) — for pretraining/testing only
# =====================================================

def generate_queue_log(n_servers, arrival_rate, service_rate, n_customers=200):
    """
    Simulate a G/G/n queue and return a time-sorted list of (event, time).
    Three event types per customer: Arrival (A), Ordering (O), Departure (D).
    "Ordering" is placed at a random point between arrival and departure,
    representing when the customer places their order after joining the queue.
    """
    arrivals = []
    t = 0.0
    for _ in range(n_customers):
        t += random.expovariate(arrival_rate)
        arrivals.append(t)

    servers = [0.0] * n_servers
    events = []

    for arrival_time in arrivals:
        server_idx = min(range(n_servers), key=lambda i: servers[i])
        start_time = max(arrival_time, servers[server_idx])
        service_time = random.expovariate(service_rate)
        departure_time = start_time + service_time
        servers[server_idx] = departure_time

        order_time = arrival_time + (departure_time - arrival_time) * random.uniform(0.05, 0.6)

        events.append(("A", arrival_time))
        events.append(("O", order_time))
        events.append(("D", departure_time))

    events.sort(key=lambda x: x[1])
    return events


# Canonical event-type -> event_id mapping used throughout the project.
# 0 = Arrival, 1 = Ordering, 2 = Departure/Leave.
EVENT_ID = {"A": 0, "O": 1, "D": 2}
NUM_EVENT_TYPES = 3


def convert_to_deltat(events):
    """Convert a list of (event, absolute_time) into [event_id, dt] pairs."""
    data = []
    prev_time = None
    for event, t in events:
        event_id = EVENT_ID[event]
        dt = 0.0 if prev_time is None else t - prev_time
        data.append([event_id, dt])
        prev_time = t
    return data


def generate_simulated_systems(num_systems=50, n_customers=200):
    """
    Returns a LIST OF SYSTEMS (not flattened!), each a list of [event_id, dt].
    Keeping systems separate is required to avoid cross-system window leakage.
    """
    systems = []
    for _ in range(num_systems):
        n_servers = random.choice([1, 2, 3, 4])
        arrival_rate = random.uniform(0.3, 1.5)
        service_rate = random.uniform(0.5, 2.0)

        log = generate_queue_log(n_servers, arrival_rate, service_rate, n_customers)
        seq = convert_to_deltat(log)
        systems.append(seq)
    return systems


# =====================================================
# 1b. STEP-1 SANITY-CHECK GENERATOR — NO SERVER AT ALL
# =====================================================

def generate_random_event_log(arrival_rate, mean_wait, mean_service, n_customers=200):
    """
    The simplest possible synthetic event stream: NO server, NO queueing.
    Each customer's Arrival, Ordering, and Departure times are drawn
    independently — nobody waits for a server, and customers never
    interact with or block each other.

    arrival_rate:  customers per unit time (Poisson arrival process)
    mean_wait:     average Arrival -> Ordering gap (drawn exponentially)
    mean_service:  average Ordering -> Departure gap (drawn exponentially)
    """
    arrivals = []
    t = 0.0
    for _ in range(n_customers):
        t += random.expovariate(arrival_rate)
        arrivals.append(t)

    events = []
    for arrival_time in arrivals:
        order_time = arrival_time + random.expovariate(1.0 / mean_wait)
        leave_time = order_time + random.expovariate(1.0 / mean_service)
        events.append(("A", arrival_time))
        events.append(("O", order_time))
        events.append(("D", leave_time))

    events.sort(key=lambda x: x[1])
    return events


def generate_random_systems(num_systems=50, n_customers=200):
    """
    List-of-systems wrapper around generate_random_event_log(), matching
    the same shape as generate_simulated_systems() — a drop-in
    replacement usable via `train.py --data random`.
    """
    systems = []
    for _ in range(num_systems):
        # All three parameters share the same time unit (seconds).
        # Arrival rate is sampled as "customers per minute" for
        # readability, then converted to per-second to stay consistent
        # with mean_wait / mean_service, which are already in seconds.
        arrival_rate_per_min = random.uniform(0.2, 2.0)
        arrival_rate = arrival_rate_per_min / 60.0
        mean_wait = random.uniform(20, 120)
        mean_service = random.uniform(60, 300)

        log = generate_random_event_log(arrival_rate, mean_wait, mean_service, n_customers)
        seq = convert_to_deltat(log)
        systems.append(seq)
    return systems


# =====================================================
# 1c. STEP-2: G/G/1 QUEUE — ONE SERVER, GENERAL DISTRIBUTIONS
# =====================================================

def generate_gg1_event_log(arrival_shape, arrival_scale, service_shape, service_scale,
                            n_customers=200):
    """
    Single-server (G/G/1) queue with Gamma-distributed inter-arrival and
    service times. Mean inter-arrival time = arrival_shape * arrival_scale
    (same for service). shape=1 reduces to the exponential/memoryless
    case; shape>1 gives more regular (less bursty) timing; shape<1 gives
    burstier, heavier-tailed timing than exponential.
    """
    arrivals = []
    t = 0.0
    for _ in range(n_customers):
        t += random.gammavariate(arrival_shape, arrival_scale)
        arrivals.append(t)

    server_free = 0.0  # only ONE server -> genuine contention/waiting
    events = []
    for arrival_time in arrivals:
        start_time = max(arrival_time, server_free)
        service_time = random.gammavariate(service_shape, service_scale)
        departure_time = start_time + service_time
        server_free = departure_time

        order_time = arrival_time + (departure_time - arrival_time) * random.uniform(0.05, 0.3)

        events.append(("A", arrival_time))
        events.append(("O", order_time))
        events.append(("D", departure_time))

    events.sort(key=lambda x: x[1])
    return events


def sample_gg1_params(rho_range=None, max_tries=10000, fixed_service=None, fixed_arrival_shape=None):
    """
    Draw one random set of Gamma parameters for a G/G/1 system:
    (arrival_shape, arrival_scale, service_shape, service_scale).

    Utilisation rho = mean service time / mean inter-arrival time
                    = (service_shape * service_scale) / (arrival_shape * arrival_scale).
    rho >= 1 means the queue grows without bound.

    With the default ranges, about 23% of draws have rho >= 1 and about 28% have
    rho >= 0.9 (rho runs up to ~4.5). rho_range=(low, high) keeps only draws with
    low <= rho <= high (rejection sampling), e.g. (0.3, 0.9) for stable systems only.
    rho_range=None reproduces the original draws exactly (same random sequence).

    fixed_service=(shape, scale) keeps the service-time distribution fixed and varies only
    the arrival rate: utilisation is drawn uniformly from rho_range and the arrival scale is
    set to give exactly that utilisation (so rho_range is required). fixed_arrival_shape
    optionally fixes the arrival shape too (otherwise it is drawn from 1 to 3).
    """
    if fixed_arrival_shape is not None and fixed_service is None:
        raise ValueError("fixed_arrival_shape can only be used together with fixed_service")
    if fixed_service is not None:
        if rho_range is None:
            raise ValueError("fixed_service requires rho_range")
        low, high = rho_range
        if not (0 < low < high):
            raise ValueError(f"rho_range must satisfy 0 < low < high, got {rho_range}")
        service_shape, service_scale = fixed_service
        arrival_shape = fixed_arrival_shape if fixed_arrival_shape is not None else random.uniform(1.0, 3.0)
        rho = random.uniform(low, high)
        arrival_scale = (service_shape * service_scale) / (rho * arrival_shape)
        return arrival_shape, arrival_scale, service_shape, service_scale
    if rho_range is not None:
        low, high = rho_range
        if not (0 <= low < high):
            raise ValueError(f"rho_range must satisfy 0 <= low < high, got {rho_range}")
    for _ in range(max_tries):
        arrival_shape = random.uniform(1.0, 3.0)
        arrival_scale = random.uniform(25, 60)          # mean inter-arrival: shape*scale seconds
        service_shape = random.uniform(1.0, 3.0)
        service_scale = random.uniform(15, 40)          # mean service: shape*scale seconds
        if rho_range is None:
            return arrival_shape, arrival_scale, service_shape, service_scale
        rho = (service_shape * service_scale) / (arrival_shape * arrival_scale)
        if rho_range[0] <= rho <= rho_range[1]:
            return arrival_shape, arrival_scale, service_shape, service_scale
    raise ValueError(f"Could not draw a system with utilisation in {rho_range} after "
                     f"{max_tries} tries; rho can only range from about 0.08 to 4.8.")


def generate_gg1_systems(num_systems=50, n_customers=200, rho_range=None, fixed_service=None,
                         fixed_arrival_shape=None):
    """
    List-of-systems wrapper around generate_gg1_event_log(), matching the
    same shape as the other generate_*_systems() functions — a drop-in
    replacement usable via `train.py --data gg1`.

    Gamma parameters are randomised per system (see sample_gg1_params). NOTE: with
    the default ranges roughly a quarter of the systems are UNSTABLE (rho >= 1);
    pass rho_range=(low, high) to restrict this.
    """
    systems = []
    for _ in range(num_systems):
        arrival_shape, arrival_scale, service_shape, service_scale = sample_gg1_params(
            rho_range, fixed_service=fixed_service, fixed_arrival_shape=fixed_arrival_shape)

        log = generate_gg1_event_log(arrival_shape, arrival_scale,
                                      service_shape, service_scale, n_customers)
        seq = convert_to_deltat(log)
        systems.append(seq)
    return systems


# =====================================================
# 1d. STEP-2 WITH O2DES: THE SAME G/G/1 QUEUE, BUILT AS AN O2DES MODEL
# =====================================================
# Requires: pip3 install o2despy pandas sortedcontainers (see o2des_gg1.py).
# NOTE: Ordering is now the real moment service STARTS, not a random point
# between arrival and departure as in generate_gg1_event_log above. Do not mix
# data or results from the two generators.
# =====================================================

def generate_gg1_o2des_event_log(arrival_shape, arrival_scale, service_shape, service_scale,
                                  n_customers=200):
    """O2DES G/G/1 queue -> time-sorted list of ("A"/"O"/"D", time) events."""
    from o2des_gg1 import simulate_gg1_o2des  # imported here so o2despy is only needed when used

    records = simulate_gg1_o2des(arrival_shape, arrival_scale, service_shape, service_scale,
                                  n_customers=n_customers)
    events = []
    for r in records:
        events.append(("A", r["arrival"]))
        events.append(("O", r["order"]))
        events.append(("D", r["leave"]))
    # Stable sort: ties keep the order they were appended in, which is the causal
    # order (a customer's Arrival, then Ordering; one customer's Leave before the
    # next customer's Ordering).
    events.sort(key=lambda x: x[1])
    return events


def generate_gg1_o2des_systems(num_systems=50, n_customers=200, rho_range=None, fixed_service=None,
                               fixed_arrival_shape=None):
    """Same parameter ranges as generate_gg1_systems() (including rho_range), but each system comes from the O2DES model."""
    systems = []
    for _ in range(num_systems):
        arrival_shape, arrival_scale, service_shape, service_scale = sample_gg1_params(
            rho_range, fixed_service=fixed_service, fixed_arrival_shape=fixed_arrival_shape)

        log = generate_gg1_o2des_event_log(arrival_shape, arrival_scale,
                                            service_shape, service_scale, n_customers)
        systems.append(convert_to_deltat(log))
    return systems


# =====================================================
# 2. REAL DATA LOADER
# =====================================================
#
# Expected CSV format (header required):
#
#   system_id,event_type,timestamp
#   sys_1,A,0.00
#   sys_1,O,0.10
#   sys_1,D,0.55
#   sys_2,A,0.00
#   ...
#
# - system_id   : any string/int identifying an independent queueing system
#                 (e.g. one counter, one call center, one day of observation)
# - event_type  : "A"/"arrival" for arrivals, "O"/"ordering" for the moment
#                 an order is placed, "D"/"departure"/"leave" for departures
#                 (case-insensitive; also accepts 0/1/2)
# - timestamp   : a float number of seconds (or any consistent unit) OR an
#                 ISO datetime string (YYYY-MM-DD HH:MM:SS[.ffffff])
# =====================================================

_EVENT_MAP = {
    "a": 0, "arrival": 0, "arrive": 0, "0": 0,
    "o": 1, "order": 1, "ordering": 1, "1": 1,
    "d": 2, "departure": 2, "depart": 2, "leave": 2, "2": 2,
}


def _parse_timestamp(raw):
    """Accept either a float (seconds) or an ISO-ish datetime string."""
    try:
        return float(raw)
    except ValueError:
        pass
    from datetime import datetime
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(raw.strip(), fmt).timestamp()
        except ValueError:
            continue
    raise ValueError(f"Could not parse timestamp: {raw!r}")


def load_real_data(csv_path):
    """
    Load a real event-log CSV and return a LIST OF SYSTEMS, each a list of
    [event_id, dt], sorted by time within each system.
    """
    raw_by_system = defaultdict(list)

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"system_id", "event_type", "timestamp"}
        if not required.issubset(set(h.strip() for h in reader.fieldnames)):
            raise ValueError(
                f"CSV must have columns {required}, got {reader.fieldnames}"
            )

        for row in reader:
            sys_id = row["system_id"].strip()
            etype_raw = row["event_type"].strip().lower()
            if etype_raw not in _EVENT_MAP:
                raise ValueError(f"Unrecognized event_type: {row['event_type']!r}")
            event_id = _EVENT_MAP[etype_raw]
            t = _parse_timestamp(row["timestamp"])
            raw_by_system[sys_id].append((event_id, t))

    systems = []
    for sys_id, events in raw_by_system.items():
        events.sort(key=lambda x: x[1])
        seq = []
        prev_time = None
        for event_id, t in events:
            dt = 0.0 if prev_time is None else t - prev_time
            if dt < 0:
                raise ValueError(
                    f"Negative dt in system {sys_id!r} at t={t}. "
                    f"Check for duplicate/unsorted timestamps."
                )
            seq.append([event_id, dt])
            prev_time = t
        if len(seq) >= 2:
            systems.append(seq)

    if not systems:
        raise ValueError("No valid systems parsed from CSV — check the file.")

    return systems


# =====================================================
# 3. WINDOWED DATASET (per-system, no cross-boundary leakage)
# =====================================================

class QueueDataset(Dataset):
    """
    Builds sliding windows of length `seq_len` STRICTLY WITHIN each system.
    dt values are normalized using (dt_mean, dt_std) computed on the
    TRAIN split only, then reused for val/generation.
    """

    def __init__(self, systems, seq_len=16, dt_mean=None, dt_std=None, stride=1):
        if stride < 1:
            raise ValueError("stride must be at least 1")
        self.seq_len = seq_len

        # Compute (or reuse) dt normalization stats.
        if dt_mean is None or dt_std is None:
            all_dt = [pt[1] for seq in systems for pt in seq]
            dt_mean = sum(all_dt) / len(all_dt)
            var = sum((d - dt_mean) ** 2 for d in all_dt) / len(all_dt)
            dt_std = max(var ** 0.5, 1e-6)

        self.dt_mean = dt_mean
        self.dt_std = dt_std

        self.X = []
        self.Y = []

        for seq in systems:
            norm_seq = [[e, (dt - dt_mean) / dt_std] for e, dt in seq]
            for i in range(0, len(norm_seq) - seq_len, stride):
                self.X.append(norm_seq[i:i + seq_len])
                self.Y.append(norm_seq[i + 1:i + seq_len + 1])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.Y[idx], dtype=torch.float32),
        )


def split_systems(systems, val_fraction=0.15, seed=0):
    """
    Deprecated: use split_systems_three_way, which also provides a held-out
    test set that is never touched during model selection. Kept for
    quick/exploratory use.
    """
    rng = random.Random(seed)
    systems = systems[:]
    rng.shuffle(systems)
    n_val = max(1, int(len(systems) * val_fraction))
    return systems[n_val:], systems[:n_val]


def split_systems_three_way(systems, val_fraction=0.15, test_fraction=0.15, seed=0):
    """
    Split at the SYSTEM level into train/val/test.

    - train: used for gradient updates
    - val:   used for early stopping / hyperparameter choices during
             development; it may be evaluated repeatedly
    - test:  touched ONCE, at the very end, to report final numbers in
             the paper. Do not tune anything based on test performance —
             that turns it into a second validation set and invalidates
             the reported numbers.

    Splitting is done on whole systems (not on individual windows) so that
    overlapping sliding windows from the same system never appear in two
    different splits.
    """
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1.0")
    if len(systems) < 3:
        raise ValueError(
            f"Need at least 3 independent systems for a train/val/test split, "
            f"got {len(systems)}. Collect data from more independent "
            f"queues/sessions, or fall back to split_systems() for now."
        )

    rng = random.Random(seed)
    systems = systems[:]
    rng.shuffle(systems)

    n_test = max(1, int(len(systems) * test_fraction))
    n_val = max(1, int(len(systems) * val_fraction))

    test_systems = systems[:n_test]
    val_systems = systems[n_test:n_test + n_val]
    train_systems = systems[n_test + n_val:]

    if len(train_systems) == 0:
        raise ValueError(
            "train split is empty after removing val/test — more "
            "systems are needed, or a smaller val_fraction/test_fraction."
        )

    return train_systems, val_systems, test_systems
