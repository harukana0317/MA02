import contextlib
import datetime as dt
import random
from collections import deque

try:
    import numpy as _np
except ImportError:  # numpy is only needed to protect its global RNG state
    _np = None

try:
    from o2despy.sandbox import Sandbox
    HAVE_O2DES = True
    _IMPORT_ERROR = None
except ImportError as _err:  # also raised if pandas / sortedcontainers are missing
    Sandbox = object
    HAVE_O2DES = False
    _IMPORT_ERROR = _err


@contextlib.contextmanager
def _preserve_global_rng():
    """
    O2DESpy's Sandbox(seed=...) calls random.seed() and np.random.seed() on the
    GLOBAL generators every time one is created. Left alone, that would reset the
    random draws used elsewhere in this project (e.g. the per-system parameter
    sampling, or the seed set in train.py). The model below uses its own private
    random.Random, so we simply save and restore the global state around it.
    """
    py_state = random.getstate()
    np_state = _np.random.get_state() if _np is not None else None
    try:
        yield
    finally:
        random.setstate(py_state)
        if np_state is not None:
            _np.random.set_state(np_state)


class GG1Queue(Sandbox):
    """A single-server FIFO queue as an O2DES Sandbox."""

    def __init__(self, arrival_shape, arrival_scale, service_shape, service_scale,
                 n_customers, seed=0):
        super().__init__(seed=seed)
        self._rng = random.Random(seed)          # private generator (see _preserve_global_rng)
        self.arrival_shape = arrival_shape
        self.arrival_scale = arrival_scale
        self.service_shape = service_shape
        self.service_scale = service_scale
        self.n_customers = n_customers

        self.waiting = deque()      # customers in the queue, FIFO
        self.server_busy = False
        self.n_arrived = 0
        self.records = []           # finished customers: {"arrival", "order", "leave"} as datetimes

        self.schedule(self._arrive, self._arrival_gap())

    def _arrival_gap(self):
        return dt.timedelta(seconds=self._rng.gammavariate(self.arrival_shape, self.arrival_scale))

    def _service_time(self):
        return dt.timedelta(seconds=self._rng.gammavariate(self.service_shape, self.service_scale))

    # ---- the three events ----
    def _arrive(self):
        self.n_arrived += 1
        self.waiting.append({"arrival": self.clock_time})
        if self.n_arrived < self.n_customers:
            self.schedule(self._arrive, self._arrival_gap())
        self._try_start_service()

    def _try_start_service(self):
        if self.server_busy or not self.waiting:
            return
        customer = self.waiting.popleft()
        customer["order"] = self.clock_time          # Ordering = service starts
        self.server_busy = True
        self.schedule(self._depart, self._service_time(), customer=customer)

    def _depart(self, customer):
        customer["leave"] = self.clock_time
        self.records.append(customer)
        self.server_busy = False
        self._try_start_service()


def simulate_gg1_o2des(arrival_shape, arrival_scale, service_shape, service_scale,
                       n_customers=200, seed=None):
    """
    Run the O2DES G/G/1 model to completion and return one record per customer,
    {"arrival", "order", "leave"} in SECONDS from the start of the simulation,
    sorted by arrival (which is also departure order for a single FIFO server).

    seed=None draws a seed from Python's global random generator, so results
    follow whatever global seeding the caller set (as the other generators do).
    """
    if not HAVE_O2DES:
        raise RuntimeError(
            "O2DESpy is not available (" + str(_IMPORT_ERROR) + "). Install it with:\n"
            "    pip3 install o2despy pandas sortedcontainers")
    if seed is None:
        seed = random.randrange(2 ** 31)

    with _preserve_global_rng():
        sim = GG1Queue(arrival_shape, arrival_scale, service_shape, service_scale,
                       n_customers, seed=seed)
        while sim.run():
            pass

    t0 = dt.datetime.min
    records = [{k: (v - t0).total_seconds() for k, v in r.items()} for r in sim.records]
    records.sort(key=lambda r: r["arrival"])
    return records
