from queue_rules import VALID_MODES
import argparse

from numerical_studies import (
    simulate_customers_gg1,
    compute_metrics,
    compare_distributions,
    repeated_comparison,
    print_metrics_table,
    generate_transformer_customers,
    _customers_to_event_dt,
    HAVE_TRANSFORMER,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoint.pt")
    p.add_argument("--n_customers", type=int, default=300,
                    help="Size of the ground-truth comparison set")
    p.add_argument("--arrival_shape", type=float, default=2.0,
                    help="Gamma shape for inter-arrival times. Use similar "
                         "values to what train.py's gg1 data used "
                         "(sampled uniformly between 1.0 and 3.0).")
    p.add_argument("--arrival_scale", type=float, default=40.0,
                    help="Gamma scale for inter-arrival times, in seconds "
                         "(train.py samples this between 25 and 60). Mean "
                         "inter-arrival time = shape * scale.")
    p.add_argument("--service_shape", type=float, default=2.0,
                    help="Gamma shape for service times (train.py samples "
                         "between 1.0 and 3.0)")
    p.add_argument("--service_scale", type=float, default=25.0,
                    help="Gamma scale for service times, in seconds "
                         "(train.py samples between 15 and 40). Mean "
                         "service time = shape * scale.")
    p.add_argument("--n_repeats", type=int, default=10,
                    help="Number of random seeds to repeat generation over")
    p.add_argument("--truth", choices=["proxy", "o2des"], default="proxy",
                    help="Where the ground truth comes from. 'o2des' = the O2DES G/G/1 model "
                         "(use this for checkpoints trained with --data gg1_o2des); 'proxy' = the "
                         "earlier simulator (checkpoints trained with --data gg1).")
    p.add_argument("--constrain", choices=["off"] + list(VALID_MODES), default="off",
                    help="Constrained generation (hard rule: forbid events that are "
                         "impossible in the current state). Use 'strict' for this G/G/1 "
                         "data. This is a hard rule, not something the model learned.")
    return p.parse_args()


def main():
    args = parse_args()

    if not HAVE_TRANSFORMER:
        print("PyTorch / generate.py are not available in this environment. "
              "Install torch (pip install torch) and make sure model.py / "
              "generate.py are in the same folder as this script.")
        return

    print("=" * 60)
    print("STEP 2 SANITY CHECK: G/G/1 queue (single server, Gamma-distributed)")
    print("=" * 60)

    mean_interarrival = args.arrival_shape * args.arrival_scale
    mean_service = args.service_shape * args.service_scale
    utilisation = mean_service / mean_interarrival
    print(f"Ground-truth parameters: arrival ~ Gamma(shape={args.arrival_shape}, "
          f"scale={args.arrival_scale}s) [mean={mean_interarrival:.1f}s], "
          f"service ~ Gamma(shape={args.service_shape}, scale={args.service_scale}s) "
          f"[mean={mean_service:.1f}s]")
    print(f"Utilisation (rho) = {utilisation:.2f} "
          f"({'STABLE' if utilisation < 1.0 else 'UNSTABLE — expect runaway queue growth; lower service_scale/shape or raise arrival_scale/shape'})")

    # ---- Ground truth (fresh draw, not used in training) ----
    if args.truth == "o2des":
        from o2des_gg1 import simulate_gg1_o2des
        truth_fn = simulate_gg1_o2des
    else:
        truth_fn = simulate_customers_gg1
    print(f"Ground truth source: {args.truth}")
    ground_truth = truth_fn(
        args.arrival_shape, args.arrival_scale, args.service_shape, args.service_scale,
        n_customers=args.n_customers, seed=999,  # fixed seed, disjoint from training
    )
    gt_metrics = compute_metrics(ground_truth)
    print_metrics_table("GROUND TRUTH (fresh G/G/1 draw)", gt_metrics, n_servers=1)

    # ---- Transformer-generated data, conditioned on a ground-truth seed ----
    seed_events = _customers_to_event_dt(ground_truth[:20])
    n_gen_events = args.n_customers * 3

    constrain_mode = None if args.constrain == "off" else args.constrain
    print(f"Constrained generation: {args.constrain}")
    transformer_records = generate_transformer_customers(
        args.checkpoint, seed_events, n_events=n_gen_events, rng_seed=0,
        constrain=constrain_mode,
    )
    tf_metrics = compute_metrics(transformer_records)
    print_metrics_table("TRANSFORMER (trained on G/G/1 data)", tf_metrics)

    if tf_metrics is None:
        print("\nThe Transformer generated too few complete (arrival+order+leave) "
              "records to compute metrics. Try more training epochs or a longer "
              "generation length.")
        return

    print("\n#################################################")
    print("# TRANSFORMER vs GROUND TRUTH")
    print("#################################################")
    compare_distributions(gt_metrics["_raw_cycle"], tf_metrics["_raw_cycle"], "Cycle time",
                           group_a_name="Ground truth", group_b_name="Transformer")
    compare_distributions(gt_metrics["_raw_wait"], tf_metrics["_raw_wait"], "Wait time",
                           group_a_name="Ground truth", group_b_name="Transformer")
    compare_distributions(gt_metrics["_raw_service"], tf_metrics["_raw_service"], "Service time",
                           group_a_name="Ground truth", group_b_name="Transformer")

    print(f"\n--- Repeated over {args.n_repeats} random seeds ---")

    def _transformer_cycle_times(rng_seed):
        recs = generate_transformer_customers(
            args.checkpoint, seed_events, n_events=n_gen_events, rng_seed=rng_seed,
            constrain=constrain_mode)
        m = compute_metrics(recs)
        return m["_raw_cycle"] if m else []

    repeated_comparison(gt_metrics["_raw_cycle"], _transformer_cycle_times,
                         n_repeats=args.n_repeats, label="Transformer vs ground truth: cycle time")

    print("\n#################################################")
    print("# VERDICT")
    print("#################################################")
    print("If the tests above mostly show p > 0.05 ('consistent' / 'NOT significantly")
    print("different'), the model has learned useful structure from a real (if simple)")
    print("queueing process with genuine contention, and it is reasonable to move on")
    print("to Step 3 (parallel servers / tandem structures — already explored against")
    print("real data).")
    print("If they show strong, consistent significant differences but Step 1 passed,")
    print("the issue is specifically in learning QUEUEING dynamics (contention/waiting)")
    print("rather than the pipeline itself — consider more training, more data, or a")
    print("larger model (d_model / num_layers).")


if __name__ == "__main__":
    main()
