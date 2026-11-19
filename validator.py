"""
One validator for event streams, used by every evaluation script.

An event stream is a list of [event_id, dt]: 0 = Arrival, 1 = Ordering (service starts),
2 = Leave (service ends); dt = seconds since the previous event.

Target system: G/G/1, one server, FIFO, no abandonment, initially empty (unless `initial`
says otherwise), non-idling (a waiting customer starts service the moment the server is free).

State tracked while the stream is read event by event:
    Q = arrivals - service starts   (customers waiting)       must stay >= 0
    B = service starts - leaves     (customers in service)    must be 0 or 1

Rules checked on EVERY event (the first failure is recorded and the stream is rejected):
    bad_gap             dt is not finite or is negative
    order_without_wait  Ordering while nobody is waiting (Q == 0)
    order_while_busy    Ordering while the server is already serving someone (B == 1)
    leave_without_service  Leave while nobody is in service (B == 0). This also covers a customer who
                        skips service and a waiting customer who abandons the queue.
    late_service_start  the server is idle and someone is waiting, but the next event is not an
                        immediate (gap <= tol) Ordering

FIFO identity: arrivals, service starts and leaves are anonymous tokens, so which customer an event
belongs to is a RECONSTRUCTION assumption. With one server and FIFO the k-th Ordering belongs to the
k-th Arrival and the k-th Leave to the k-th Ordering.
"""
import math

ARRIVAL, ORDERING, LEAVE = 0, 1, 2
CATEGORIES = ("bad_gap", "order_without_wait", "order_while_busy", "leave_without_service", "late_service_start")
DEFAULT_TOL = 1e-6   # seconds; a gap this small counts as "at once"


def validate_stream(stream, tol=DEFAULT_TOL, initial=(0, 0)):
    """
    stream: list of [event_id, dt]. initial: (Q, B) at the start of the stream.

    Returns a dict:
      valid            True if no rule is broken
      first_failure    None, or {"index", "category", "detail"} for the first broken rule
      n_events         number of events read (all of them, or up to and including the failure)
      customers        FIFO reconstruction up to the failure: list of {"arrival", "order", "leave"}
                       ("order"/"leave" are None while not yet reached) -- includes UNFINISHED customers
      n_completed      customers with a Leave
      n_unfinished     customers who arrived but have not left
      queue_end        waiting customers at the end   (Q)
      busy_end         1 if someone is in service at the end (B)
      end_time         time of the last event read
    Nothing is repaired or reinterpreted: a stream with a broken rule is simply reported as invalid.
    """
    t = 0.0
    Q, B = initial
    # FIFO queues of customer indices. Customers present at the start (initial) have unknown times.
    customers = []
    waiting, in_service = [], None
    for _ in range(Q):
        customers.append({"arrival": None, "order": None, "leave": None})
        waiting.append(len(customers) - 1)
    if B:
        customers.append({"arrival": None, "order": None, "leave": None})
        in_service = len(customers) - 1

    failure = None
    n_read = 0
    for i, (event, dt) in enumerate(stream):
        n_read = i + 1
        event = int(event)
        try:
            dt_value = float(dt)
        except (TypeError, ValueError):
            dt_value = float("nan")
        if not math.isfinite(dt_value) or dt_value < 0:
            failure = {"index": i, "category": "bad_gap", "detail": f"dt={dt!r}"}
            break
        dt = dt_value

        # 1) is the event itself possible in the current state?
        if event not in (ARRIVAL, ORDERING, LEAVE):
            failure = {"index": i, "category": "bad_gap", "detail": f"unknown event id {event}"}
            break
        if event == ORDERING and not waiting:
            failure = {"index": i, "category": "order_without_wait", "detail": "Ordering with nobody waiting"}
            break
        if event == ORDERING and in_service is not None:
            failure = {"index": i, "category": "order_while_busy", "detail": "Ordering while the server is busy"}
            break
        if event == LEAVE and in_service is None:
            failure = {"index": i, "category": "leave_without_service",
                       "detail": "Leave with nobody in service" + (f" ({len(waiting)} waiting)" if waiting else "")}
            break
        # 2) non-idling: idle server + someone waiting => the next event must come at once
        #    (another Arrival at the very same instant may still come first).
        if in_service is None and waiting and dt > tol:
            failure = {"index": i, "category": "late_service_start",
                       "detail": f"server idle with {len(waiting)} waiting but next event is {dt:.3f}s later"}
            break

        t += dt
        if event == ARRIVAL:
            customers.append({"arrival": t, "order": None, "leave": None})
            waiting.append(len(customers) - 1)
        elif event == ORDERING:
            in_service = waiting.pop(0)
            customers[in_service]["order"] = t
        else:
            customers[in_service]["leave"] = t
            in_service = None

    n_completed = sum(1 for c in customers if c["leave"] is not None)
    return {
        "valid": failure is None,
        "first_failure": failure,
        "n_events": n_read,
        "customers": customers,
        "n_completed": n_completed,
        "n_unfinished": len(customers) - n_completed,
        "queue_end": len(waiting),
        "busy_end": 0 if in_service is None else 1,
        "end_time": t,
    }


def completed_cycles(result, after_time=None):
    """
    Arrival->Leave times of customers who completed. With after_time, only customers who ARRIVED after
    that time count (so a conditioning history is not scored). Customers whose arrival is unknown are skipped.
    """
    out = []
    for c in result["customers"]:
        if c["leave"] is None or c["arrival"] is None:
            continue
        if after_time is not None and c["arrival"] <= after_time:
            continue
        out.append(c["leave"] - c["arrival"])
    return out


def time_at(stream, n_events):
    """Clock time of the n_events-th event of the stream (0 if n_events == 0)."""
    return sum(float(dt) for _, dt in stream[:n_events])
