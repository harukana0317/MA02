import argparse
import torch

from model import QueueTransformer
from queue_rules import advance_state, allowed_events, forced_event, VALID_MODES


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoint.pt")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--constrain", choices=["off"] + list(VALID_MODES), default="off",
                    help="Constrained generation: forbid events that are impossible "
                         "in the current state (e.g. a Leave when nobody is in the "
                         "shop). A hard rule applied at generation time only.")
    return p.parse_args()


def load_model(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt["config"]

    model = QueueTransformer(
        d_model=cfg["d_model"], nhead=cfg["nhead"], num_layers=cfg["num_layers"],
        max_len=max(cfg["seq_len"] + 1, 64),
        num_event_types=cfg.get("num_event_types", 3),
        time_mode=cfg.get("time_mode", "mse"),   # old checkpoints have no such key -> legacy head
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, ckpt["dt_mean"], ckpt["dt_std"], cfg["seq_len"]


def generate(model, seed, dt_mean, dt_std, seq_len, steps=20, rng_seed=None, constrain=None):
    """
    seed: list of [event_id, raw_dt] in REAL (unnormalized) units — the
          conditioning sequence generation continues from. (Not to be
          confused with rng_seed below — unfortunate naming clash kept
          for backwards compatibility with earlier scripts.)
    rng_seed: if given, fixes torch's random number generator before
          sampling, so repeated calls with the same rng_seed produce
          IDENTICAL output. Leave as None for a different random draw
          each call.
    constrain: None (default, unchanged behaviour) or one of the modes in queue_rules.py
          ("strict", "allow_balk", "single_server", "derived_ordering") —
          at every step, events that are impossible given who is currently in
          the system are given probability 0 and the rest are renormalised;
          "derived_ordering" also makes Ordering a zero-gap consequence of the queue state.
          This is a HARD RULE, not something the model learned.
    Returns: list of [event_id, raw_dt] in REAL units.
    """
    model.eval()
    if rng_seed is not None:
        torch.manual_seed(rng_seed)

    state = [0, 0]  # [waiting_to_order, waiting_to_leave], only used when constrain is set
    if constrain:
        if getattr(model, "num_event_types", 3) != 3:
            raise ValueError("constrain only supports the 3 event types Arrival/Ordering/Leave")
        for e, _ in seed:               # start from the state implied by the seed events
            state = advance_state(state, e)

    # normalize seed for the model
    norm_seq = [[e, (dt - dt_mean) / dt_std] for e, dt in seed]

    for _ in range(steps):
        forced = forced_event(state, constrain) if constrain else None
        if forced is not None:
            # a deterministic consequence of the queue state: it happens at once, no sampling
            norm_seq.append([forced, (0.0 - dt_mean) / dt_std])
            state = advance_state(state, forced)
            continue

        window = norm_seq[-seq_len:]
        x = torch.tensor([window], dtype=torch.float32)

        with torch.no_grad():
            event_logits, time_pred = model(x)

            probs = torch.softmax(event_logits[:, -1], dim=-1)
            if constrain:
                allowed = allowed_events(state, constrain)
                mask = torch.tensor([[1.0 if ok else 0.0 for ok in allowed]], dtype=probs.dtype)
                masked = probs * mask
                total = masked.sum()
                # If the model put (numerically) all its mass on forbidden events,
                # fall back to a uniform choice among the allowed ones.
                probs = masked / total if total.item() > 0 else mask / mask.sum()
            event = torch.multinomial(probs, 1).item()
            if constrain:
                state = advance_state(state, event)

            if model.time_mode == "lognormal":
                # dt is SAMPLED from the distribution belonging to the event
                # type we just drew, then z-scored so it can be fed back in.
                dt_sample = model.sample_dt(time_pred[:, -1],
                                            torch.tensor([event])).item()
                norm_dt = (dt_sample - dt_mean) / dt_std
            else:
                norm_dt = time_pred[:, -1].item()

        norm_seq.append([event, norm_dt])

    # denormalize + clip negative dt (cannot happen in a real event log)
    real_seq = []
    for e, norm_dt in norm_seq:
        real_dt = norm_dt * dt_std + dt_mean
        real_dt = max(real_dt, 0.0)
        real_seq.append([e, real_dt])

    return real_seq


EVENT_LABELS = {0: "A (Arrival)", 1: "O (Ordering)", 2: "D (Leave)"}


if __name__ == "__main__":
    args = parse_args()
    model, dt_mean, dt_std, seq_len = load_model(args.checkpoint)

    # Example seed in REAL units (seconds). Replace with a real snippet
    # from collected data to condition generation on it.
    seed = [
        [0, 0.00],  # Arrival
        [1, 0.10],  # Ordering
        [2, 0.55],  # Leave
        [0, 0.15],
        [1, 0.20],
    ]

    generated = generate(model, seed, dt_mean, dt_std, seq_len, steps=args.steps,
                         constrain=None if args.constrain == "off" else args.constrain)

    print("\nGenerated Queue Process:\n")
    for e, dt in generated:
        print("Event:", EVENT_LABELS.get(e, e), "| dt:", round(dt, 3))
