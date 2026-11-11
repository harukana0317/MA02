# strict         Ordering needs someone waiting to order; Leave needs someone who has ordered.
# allow_balk     As strict, but a customer who has not ordered may also leave.
# single_server  As strict, and Ordering is also forbidden while another customer is in service
#                (one server serves one customer at a time).
# derived_ordering  As single_server, and Ordering is not sampled at all: whenever the server is free
#                and someone is waiting, the next event is that customer's Ordering, with zero gap.
VALID_MODES = ("strict", "allow_balk", "single_server", "derived_ordering")


def advance_state(state, event):
    """Return the new [waiting_to_order, waiting_to_leave] after `event` happens."""
    waiting_to_order, waiting_to_leave = state
    if event == 0:
        waiting_to_order += 1
    elif event == 1:
        if waiting_to_order > 0:
            waiting_to_order -= 1
            waiting_to_leave += 1
    elif event == 2:
        if waiting_to_leave > 0:
            waiting_to_leave -= 1
        elif waiting_to_order > 0:
            waiting_to_order -= 1
    return [waiting_to_order, waiting_to_leave]


def allowed_events(state, mode):
    """[Arrival_ok, Ordering_ok, Leave_ok] for the current state. Arrival is always allowed."""
    if mode not in VALID_MODES:
        raise ValueError(f"constrain mode must be one of {VALID_MODES}, got {mode!r}")
    waiting_to_order, waiting_to_leave = state
    ordering_ok = waiting_to_order > 0
    if mode == "strict":
        leave_ok = waiting_to_leave > 0
    elif mode == "allow_balk":
        leave_ok = waiting_to_leave > 0 or waiting_to_order > 0
    else:  # single_server, derived_ordering
        ordering_ok = waiting_to_order > 0 and waiting_to_leave == 0
        leave_ok = waiting_to_leave > 0
    return [True, ordering_ok, leave_ok]


def forced_event(state, mode):
    """The event that must happen next with zero gap in this state, or None if the model decides."""
    waiting_to_order, waiting_to_leave = state
    if mode == "derived_ordering" and waiting_to_order > 0 and waiting_to_leave == 0:
        return 1
    return None
