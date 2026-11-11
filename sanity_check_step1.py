import argparse
import random

from numerical_studies import (
    simulate_customers_random,
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
    p.add_argument("--arrival_rate", type=float, default=0.6,
                    help="Ground-truth arrival rate (customers/min). Use a "
                         "similar value to what train.py's random data used.")
    p.add_argument("--mean_wait", type=float, default=70.0,
                    help="Ground-truth mean Arrival->Order gap (seconds)")
    p.add_argument("--mean_service", type=float, default=180.0,
                    help="Ground-truth mean Order->Leave gap (seconds)")
    p.add_argument("--n_repeats", type=int, default=10,
                    help="Number of random seeds to repeat generation over")
    return p.parse_args()


def main():
    args = parse_args()

    if not HAVE_TRANSFORMER:
        print("PyTorch / generate.py are not available in this environment. "
              "Install torch (pip install torch) and make sure model.py / "
              "generate.py are in the same folder as this script.")
        return

    arrival_rate_per_sec = args.arrival_rate / 60.0

    print("=" * 60)
    print("STEP 1 SANITY CHECK: no-server random process")
    print("=" * 60)
    print(f"Ground-truth parameters: arrival_rate={args.arrival_rate}/min, "
          f"mean_wait={args.mean_wait}s, mean_service={args.mean_service}s")

    # ---- Ground truth (fresh draw, not used in training) ----
    ground_truth = simulate_customers_random(
        arrival_rate_per_sec, args.mean_wait, args.mean_service,
        n_customers=args.n_customers, seed=999,  # fixed seed, disjoint from training
    )
    gt_metrics = compute_metrics(ground_truth)
    print_metrics_table("GROUND TRUTH (fresh no-server random draw)", gt_metrics)

    # ---- Transformer-generated data, conditioned on a ground-truth seed ----
    seed_events = _customers_to_event_dt(ground_truth[:20])
    n_gen_events = args.n_customers * 3

    transformer_records = generate_transformer_customers(
        args.checkpoint, seed_events, n_events=n_gen_events, rng_seed=0,
    )
    tf_metrics = compute_metrics(transformer_records)
    print_metrics_table("TRANSFORMER (trained on no-server random data)", tf_metrics)

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
            args.checkpoint, seed_events, n_events=n_gen_events, rng_seed=rng_seed)
        m = compute_metrics(recs)
        return m["_raw_cycle"] if m else []

    repeated_comparison(gt_metrics["_raw_cycle"], _transformer_cycle_times,
                         n_repeats=args.n_repeats, label="Transformer vs ground truth: cycle time")

    print("\n#################################################")
    print("# VERDICT")
    print("#################################################")
    print("If the tests above mostly show p > 0.05 ('consistent' / 'NOT significantly")
    print("different'), the pipeline is working correctly on this trivial case, and it")
    print("is reasonable to move on to Step 2 (a G/G/1 queue with one server).")
    print("If they show strong, consistent significant differences, the issue is in")
    print("the pipeline itself (data representation, model, or training) and should")
    print("be debugged BEFORE testing on any queueing structure.")


if __name__ == "__main__":
    main()
