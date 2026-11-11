import argparse
import math

import torch
from torch.utils.data import DataLoader

from data import (
    generate_simulated_systems,
    generate_random_systems,
    generate_gg1_systems,
    generate_gg1_o2des_systems,
    load_real_data,
    QueueDataset,
    split_systems_three_way,
)
from model import QueueTransformer
from utils import set_seed, ExperimentLogger, summarize_runs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="simulate",
                    help="'random' (no server, pipeline sanity check), "
                         "'gg1_o2des' (the same G/G/1 queue built as an O2DES model, needs "
                         "pip3 install o2despy pandas sortedcontainers), "
                         "'gg1' (single-server queue, general/Gamma-distributed "
                         "timing), 'simulate' (G/G/n queue with multiple servers), "
                         "or a path to a real-data CSV")
    p.add_argument("--seq_len", type=int, default=16)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--time_head", choices=["mse", "lognormal"], default="mse",
                    help="'mse' = legacy point-estimate of dt (trained with MSE, "
                         "ignores which event type comes next). 'lognormal' = a "
                         "dt DISTRIBUTION per event type trained by negative "
                         "log-likelihood and SAMPLED at generation time.")
    p.add_argument("--num_event_types", type=int, default=3,
                    help="3 for Arrival/Ordering/Leave. Change only if the "
                         "event vocabulary differs from data.py's EVENT_ID map.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad_clip", type=float, default=0.0,
                    help="Clip the gradient norm to this value (0 = off). Helps against loss spikes with long sequences.")
    p.add_argument("--time_loss_weight", type=float, default=1.0)
    p.add_argument("--val_fraction", type=float, default=0.15)
    p.add_argument("--test_fraction", type=float, default=0.15)
    p.add_argument("--num_sim_systems", type=int, default=60,
                    help="Only used when --data simulate, random, gg1 or gg1_o2des")
    p.add_argument("--rho_range", type=float, nargs=2, metavar=("LOW", "HIGH"), default=None,
                    help="Only for --data gg1 / gg1_o2des. Keep only training systems whose "
                         "utilisation rho (mean service time / mean inter-arrival time) is "
                         "between LOW and HIGH, e.g. --rho_range 0.3 0.9 for stable systems "
                         "only. By default about 23%% of drawn systems have rho >= 1 (unstable).")
    p.add_argument("--fixed_service", type=float, nargs=2, metavar=("SHAPE", "SCALE"), default=None,
                    help="Only for --data gg1 / gg1_o2des, together with --rho_range. Keeps the Gamma service-time "
                         "distribution fixed and varies only the arrival rate, so that utilisation is uniform in "
                         "the --rho_range. Example: --fixed_service 2 25 --rho_range 0.3 0.9.")
    p.add_argument("--fixed_arrival_shape", type=float, default=None,
                    help="With --fixed_service: also fix the Gamma shape of the inter-arrival times "
                         "(otherwise it is drawn between 1 and 3 for each system).")
    p.add_argument("--n_customers_per_system", type=int, default=200,
                    help="Customers per generated system (3 events each). Raise it when --seq_len is large.")
    p.add_argument("--window_stride", type=int, default=1,
                    help="Step between consecutive training windows. 1 uses every window; a larger value "
                         "(e.g. 50 with --seq_len 400) keeps long-sequence training affordable.")
    p.add_argument("--regenerate_each_epoch", action="store_true",
                    help="Only for --data random/simulate. Generates a FRESH "
                         "batch of training systems every epoch instead of "
                         "reusing the same fixed batch, so the model can never "
                         "memorise a specific finite sample. This is the "
                         "principled way to exploit 'unlimited' synthetic "
                         "training data. Validation/test sets stay fixed for "
                         "a consistent benchmark across epochs.")
    p.add_argument("--checkpoint", type=str, default="checkpoint.pt")
    p.add_argument("--run_dir", type=str, default="runs")
    p.add_argument("--seed", type=int, default=0,
                    help="Ignored if --n_seeds > 1 (seeds 0..n_seeds-1 used instead)")
    p.add_argument("--n_seeds", type=int, default=1,
                    help="Run the whole experiment under multiple seeds and "
                         "report mean/std. Recommended >=3 for paper-grade claims.")
    return p.parse_args()


def evaluate(model, loader, event_loss_fn, time_loss_fn, device, num_event_types,
             dt_mean=0.0, dt_std=1.0):
    model.eval()
    total_event, total_time, n = 0.0, 0.0, 0
    with torch.no_grad():
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            pred_event, pred_time = model(X)

            le = event_loss_fn(pred_event.reshape(-1, num_event_types), Y[..., 0].long().reshape(-1))
            if model.time_mode == "lognormal":
                raw_dt = (Y[..., 1] * dt_std + dt_mean).clamp(min=0.0)
                lt = model.time_nll(pred_time, Y[..., 0].long(), raw_dt)
            else:
                lt = time_loss_fn(pred_time.squeeze(-1), Y[..., 1])

            total_event += le.item()
            total_time += lt.item()
            n += 1
    model.train()
    return total_event / max(n, 1), total_time / max(n, 1)


def run_single_seed(args, seed):
    set_seed(seed)  # fix randomness BEFORE any data generation or model init

    is_generative = args.data in ("random", "simulate", "gg1", "gg1_o2des")
    if args.rho_range is not None and args.data not in ("gg1", "gg1_o2des"):
        raise ValueError("--rho_range only applies to --data gg1 or --data gg1_o2des")
    if (args.fixed_service is not None or args.fixed_arrival_shape is not None) and args.data not in ("gg1", "gg1_o2des"):
        raise ValueError("--fixed_service / --fixed_arrival_shape only apply to --data gg1 or --data gg1_o2des")
    if args.fixed_service is not None and args.rho_range is None:
        raise ValueError("--fixed_service requires --rho_range")
    if args.fixed_arrival_shape is not None and args.fixed_service is None:
        raise ValueError("--fixed_arrival_shape requires --fixed_service")
    if args.regenerate_each_epoch and not is_generative:
        raise ValueError("--regenerate_each_epoch only makes sense with "
                          "--data random, simulate, gg1 or gg1_o2des "
                          "(real CSV data is fixed and can't be 'regenerated').")

    def _generate_systems(n):
        nc = args.n_customers_per_system
        if args.data == "random":
            return generate_random_systems(n, n_customers=nc)
        elif args.data == "gg1":
            return generate_gg1_systems(n, n_customers=nc, rho_range=args.rho_range,
                                        fixed_service=args.fixed_service, fixed_arrival_shape=args.fixed_arrival_shape)
        elif args.data == "gg1_o2des":
            return generate_gg1_o2des_systems(n, n_customers=nc, rho_range=args.rho_range,
                                              fixed_service=args.fixed_service,
                                              fixed_arrival_shape=args.fixed_arrival_shape)
        else:  # "simulate"
            return generate_simulated_systems(n, n_customers=nc)

    # ---- Load data as a list of independent "systems" ----
    if is_generative:
        label = {"random": "NO-SERVER random", "gg1": "G/G/1 single-server", "gg1_o2des": "O2DES G/G/1 single-server",
                  "simulate": "simulated G/G/n"}[args.data]
        print(f"[seed {seed}] Generating {args.num_sim_systems} {label} systems...")
        systems = _generate_systems(args.num_sim_systems)
    else:
        print(f"[seed {seed}] Loading real event log from {args.data}...")
        systems = load_real_data(args.data)

    train_systems, val_systems, test_systems = split_systems_three_way(
        systems, val_fraction=args.val_fraction, test_fraction=args.test_fraction,
        seed=seed,
    )
    print(f"[seed {seed}] train={len(train_systems)} val={len(val_systems)} "
          f"test={len(test_systems)} systems")

    train_ds = QueueDataset(train_systems, seq_len=args.seq_len, stride=args.window_stride)
    val_ds = QueueDataset(val_systems, seq_len=args.seq_len, stride=args.window_stride,
                           dt_mean=train_ds.dt_mean, dt_std=train_ds.dt_std)
    test_ds = QueueDataset(test_systems, seq_len=args.seq_len, stride=args.window_stride,
                            dt_mean=train_ds.dt_mean, dt_std=train_ds.dt_std)
    # Normalisation stats are fixed HERE, from the first training draw,
    # and reused for every subsequent regenerated batch (if any) — the
    # model must always see the same dt scale, or training destabilises.
    dt_mean, dt_std = train_ds.dt_mean, train_ds.dt_std

    if len(train_ds) == 0:
        raise ValueError(
            "No training windows produced — increase data per system or "
            "decrease --seq_len."
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False) \
        if len(val_ds) > 0 else None
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False) \
        if len(test_ds) > 0 else None

    if args.regenerate_each_epoch:
        print(f"[seed {seed}] --regenerate_each_epoch is ON: a fresh batch of "
              f"{args.num_sim_systems} training systems will be drawn every "
              f"epoch. Validation/test sets above stay fixed.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = QueueTransformer(
        d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
        max_len=max(args.seq_len + 1, 64), num_event_types=args.num_event_types,
        time_mode=args.time_head,
    ).to(device)

    if args.time_head == "lognormal":
        # Start the dt distribution at the data's typical scale (mean / std of
        # log(1+dt) over the first training draw) instead of at 0, so the first
        # epochs are not dominated by one huge, easily-fixed offset.
        ys = [math.log1p(max(step[1] * dt_std + dt_mean, 0.0))
              for window in train_ds.Y for step in window]
        y_mean = sum(ys) / len(ys)
        y_std = math.sqrt(sum((v - y_mean) ** 2 for v in ys) / len(ys)) + 1e-6
        with torch.no_grad():
            # bias layout is [mu_0, log_sigma_0, mu_1, log_sigma_1, ...]
            model.time_head.bias[0::2] = y_mean
            model.time_head.bias[1::2] = math.log(y_std)
        print(f"[seed {seed}] lognormal time head initialised at "
              f"mean(log(1+dt))={y_mean:.2f}, std={y_std:.2f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    event_loss_fn = torch.nn.CrossEntropyLoss()
    time_loss_fn = torch.nn.MSELoss()

    config = vars(args).copy()
    config["seed"] = seed
    config["n_train_systems"] = len(train_systems)
    config["n_val_systems"] = len(val_systems)
    config["n_test_systems"] = len(test_systems)
    logger = ExperimentLogger(config, run_dir=args.run_dir, tag=f"seed{seed}")

    ckpt_path = args.checkpoint if args.n_seeds == 1 else \
        args.checkpoint.replace(".pt", f"_seed{seed}.pt")

    def _save_checkpoint(path):
        torch.save({
            "model_state_dict": model.state_dict(),
            "dt_mean": dt_mean,
            "dt_std": dt_std,
            "config": {
                "d_model": args.d_model, "nhead": args.nhead,
                "num_layers": args.num_layers, "seq_len": args.seq_len,
                "num_event_types": args.num_event_types,
                "time_mode": args.time_head,
            },
        }, path)

    # Tracks the BEST validation performance seen so far. Whenever we
    # find a new best, we immediately overwrite the checkpoint file —
    # so no matter how many epochs are requested, the saved file
    # is always the best-validated model, not just "whatever the
    # last epoch happened to produce" (which can be badly overfit).
    best_val_event_loss = float("inf")
    best_epoch = None

    for epoch in range(args.epochs):
        if args.regenerate_each_epoch and epoch > 0:
            # Draw a completely fresh training set this epoch. Val/test
            # are NEVER regenerated, so the benchmark stays comparable
            # across epochs.
            fresh_systems = _generate_systems(args.num_sim_systems)
            fresh_train_systems, _, _ = split_systems_three_way(
                fresh_systems, val_fraction=args.val_fraction,
                test_fraction=args.test_fraction, seed=seed * 10_000 + epoch,
            )
            train_ds = QueueDataset(fresh_train_systems, seq_len=args.seq_len, stride=args.window_stride,
                                     dt_mean=dt_mean, dt_std=dt_std)
            if len(train_ds) == 0:
                continue
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

        total_loss = 0.0
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)

            optimizer.zero_grad()
            pred_event, pred_time = model(X)

            loss_event = event_loss_fn(
                pred_event.reshape(-1, args.num_event_types), Y[..., 0].long().reshape(-1)
            )
            if args.time_head == "lognormal":
                raw_dt = (Y[..., 1] * dt_std + dt_mean).clamp(min=0.0)
                loss_time = model.time_nll(pred_time, Y[..., 0].long(), raw_dt)
            else:
                loss_time = time_loss_fn(pred_time.squeeze(-1), Y[..., 1])
            loss = loss_event + args.time_loss_weight * loss_time

            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += loss.item()

        epoch_metrics = {"train_loss": total_loss / len(train_loader)}
        if val_loader is not None:
            ve, vt = evaluate(model, val_loader, event_loss_fn, time_loss_fn, device, args.num_event_types,
                              dt_mean, dt_std)
            epoch_metrics["val_event_loss"] = ve
            epoch_metrics["val_time_loss"] = vt

            # With the lognormal head the time part is a real likelihood, so the
            # best epoch is judged on event loss + time NLL (the full joint
            # NLL). With the legacy MSE head we keep the old event-loss rule.
            score = ve + args.time_loss_weight * vt if args.time_head == "lognormal" else ve
            if score < best_val_event_loss:
                best_val_event_loss = score
                best_epoch = epoch + 1
                _save_checkpoint(ckpt_path)

        logger.log_epoch(epoch, epoch_metrics)
        print(f"[seed {seed}] Epoch {epoch + 1}/{args.epochs}: {epoch_metrics}")

    if val_loader is None:
        # No validation set to select on — fall back to saving the final
        # epoch's model (the old behaviour).
        _save_checkpoint(ckpt_path)
        best_epoch = args.epochs
        print(f"[seed {seed}] No validation set available — saved the final "
              f"epoch's model instead of a best-validated one.")
    else:
        score_name = "event + time NLL" if args.time_head == "lognormal" else "event only"
        print(f"[seed {seed}] Best validation loss ({score_name}): {best_val_event_loss:.4f} "
              f"at epoch {best_epoch}/{args.epochs}. Checkpoint saved from that epoch "
              f"(not necessarily the last one).")

    # ---- Reload the BEST checkpoint before final test evaluation, so the
    # reported test numbers correspond to the model actually being kept ----
    if val_loader is not None:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    final_metrics = {}
    if test_loader is not None:
        te, tt = evaluate(model, test_loader, event_loss_fn, time_loss_fn, device, args.num_event_types,
                          dt_mean, dt_std)
        final_metrics = {"test_event_loss": te, "test_time_loss": tt, "best_epoch": best_epoch}
        print(f"[seed {seed}] FINAL TEST METRICS (touch once!): {final_metrics}")
    logger.log_final(final_metrics)
    print(f"[seed {seed}] Saved checkpoint to {ckpt_path}")

    return final_metrics


def main():
    args = parse_args()

    if args.n_seeds <= 1:
        run_single_seed(args, args.seed)
        return

    print(f"Running {args.n_seeds} seeds for paper-grade mean/std reporting...")
    for seed in range(args.n_seeds):
        run_single_seed(args, seed)

    for key in ["test_event_loss", "test_time_loss"]:
        summary = summarize_runs(run_dir=args.run_dir, metric_key=key)
        if summary["n_runs"] > 0:
            print(f"\n{key}: {summary['mean']:.4f} \u00b1 {summary['std']:.4f} "
                  f"(n={summary['n_runs']} seeds)")


if __name__ == "__main__":
    main()
