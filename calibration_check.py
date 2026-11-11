import argparse
import sys

try:
    import torch
    from torch.utils.data import DataLoader
    from data import QueueDataset
    from generate import load_model
except ImportError as err:
    sys.exit(f"PyTorch and the project files (data.py, generate.py, model.py) are required: {err}")

from numerical_studies import _customers_to_event_dt
from o2des_gg1 import simulate_gg1_o2des

NAMES = ["Arrival", "Ordering", "Leave"]


def summarise(count, gap_true, gap_pred, prob_mass, total):
    """Per event type: actual share, mean predicted probability, actual / predicted mean gap and the bias in percent."""
    rows = []
    for k in range(3):
        share = count[k] / total if total else None
        mean_prob = prob_mass[k] / total if total else None
        actual = gap_true[k] / count[k] if count[k] else None
        sampled = gap_pred[k] / count[k] if count[k] else None
        bias = 100.0 * (sampled / actual - 1.0) if actual and sampled is not None else None
        rows.append((NAMES[k], share, mean_prob, actual, sampled, bias))
    return rows


def fmt(v, width, digits=3):
    return " " * (width - 3) + "n/a" if v is None else f"{v:>{width}.{digits}f}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n_systems", type=int, default=40)
    p.add_argument("--n_customers", type=int, default=300)
    p.add_argument("--n_samples", type=int, default=20, help="gaps sampled per position, averaged to reduce noise")
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--arrival_shape", type=float, default=2.0)
    p.add_argument("--arrival_scale", type=float, default=40.0)
    p.add_argument("--service_shape", type=float, default=2.0)
    p.add_argument("--service_scale", type=float, default=25.0)
    return p.parse_args()


def main():
    args = parse_args()
    params = (args.arrival_shape, args.arrival_scale, args.service_shape, args.service_scale)
    model, dt_mean, dt_std, seq_len = load_model(args.checkpoint)
    model.eval()
    torch.manual_seed(args.seed)

    systems = [_customers_to_event_dt(simulate_gg1_o2des(*params, n_customers=args.n_customers, seed=100000 + args.seed + k))
               for k in range(args.n_systems)]
    dataset = QueueDataset(systems, seq_len=seq_len, dt_mean=dt_mean, dt_std=dt_std, stride=max(1, seq_len // 4))
    if len(dataset) == 0:
        sys.exit("No windows: raise --n_customers (the training window is longer than the generated systems).")
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    count, gap_true, gap_pred, prob_mass = [0] * 3, [0.0] * 3, [0.0] * 3, [0.0] * 3
    nll_event, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X, Y in loader:
            logits, time_out = model(X)
            probs = torch.softmax(logits[:, -1], dim=-1)                       # predicted next event, full context
            target = Y[:, -1, 0].long()                                         # what really came next
            true_gap = (Y[:, -1, 1] * dt_std + dt_mean).clamp(min=0.0)          # real gap in seconds
            if model.time_mode == "lognormal":
                draws = [model.sample_dt(time_out[:, -1], target) for _ in range(args.n_samples)]
                pred_gap = torch.stack(draws).mean(0)
            else:
                pred_gap = (time_out[:, -1, 0] * dt_std + dt_mean).clamp(min=0.0)
            p_true = probs.gather(1, target.unsqueeze(1)).squeeze(1)
            nll_event += -torch.log(p_true.clamp_min(1e-12)).sum().item()
            correct += (probs.argmax(1) == target).sum().item()
            total += target.numel()
            for k in range(3):
                m = target == k
                count[k] += int(m.sum().item())
                gap_true[k] += true_gap[m].sum().item()
                gap_pred[k] += pred_gap[m].sum().item()
                prob_mass[k] += probs[:, k].sum().item()

    print(f"Checkpoint {args.checkpoint} (training window {seq_len}, time head {model.time_mode})")
    print(f"One-step check on {total} windows from {args.n_systems} fresh ground-truth systems. "
          f"The model sees the TRUE history, never its own output.\n")
    print(f"{'next event':<12}{'actual share':>13}{'mean predicted':>16}{'next gap, actual':>18}{'next gap, model':>17}{'bias':>8}")
    print(f"{'':<12}{'':>13}{'probability':>16}{'(s)':>18}{'(s)':>17}{'%':>8}")
    for name, share, mean_prob, actual, sampled, bias in summarise(count, gap_true, gap_pred, prob_mass, total):
        print(f"{name:<12}{fmt(share, 13)}{fmt(mean_prob, 16)}{fmt(actual, 18, 1)}{fmt(sampled, 17, 1)}{fmt(bias, 8, 0)}")
    print(f"\nnext-event accuracy {correct / total:.3f}; mean event-type loss {nll_event / total:.3f} nats")
    print("\nHow to read it: compare 'bias' here with the free-running gaps in the 'Mean time gap BEFORE each event type' table of diagnose_generation.py and in audit_times.py.")
    print("  - small bias here, but free-running gaps 15-25% too long: the error builds up while the model feeds on its own output.")
    print("  - similar bias here: the one-step time prediction itself is off (a time-head / training problem).")
    print("  - 'mean predicted probability' far from 'actual share': the model's event probabilities are miscalibrated.")


if __name__ == "__main__":
    main()
