# Queue Transformer

Event-level modelling of a queue with a Transformer. The model is trained on logs of
three event types (Arrival, Ordering, Leave), each with the time since the previous
event. It is then run freely to generate new logs, which are compared with a
ground-truth simulation using system-level measures (cycle time, system occupancy).

## Status
- One-step prediction is good (validation event loss about 0.3-0.4, against 1.099 for
  guessing among three event types).
- Free-running generation (900 steps) is not yet reliable on the G/G/1 benchmark: most
  generated queues become more congested than the ground truth. Training on stable
  systems only (utilisation 0.3-0.9) lowered the distance to the ground truth in all
  three seeds tried, but the effect is small relative to seed-to-seed variation.
- Real Subway data (6 sessions, 151 customers): a two-stage tandem queue reproduces
  the observed cycle time much better than a parallel M/M/n model; the Transformer
  has not yet beaten either.

## Setup
    pip3 install -r requirements.txt
Run with Python 3.11. `numerical_studies.py` and `o2des_gg1.py` also run without PyTorch (the Transformer parts of
`numerical_studies.py` are then skipped); training, generation and the diagnostics need it.

## Files
| File | Purpose |
|---|---|
| `data.py` | Data generators (`random`, `gg1`, `gg1_o2des`, `simulate`), real-data CSV loader, windowed dataset, train/val/test split by system |
| `o2des_gg1.py` | G/G/1 queue as an O2DES model (O2DESpy). Ordering = the moment service starts |
| `model.py` | Transformer: positional encoding, causal mask, event head, time head (`mse` or `lognormal`) |
| `train.py` | Training, with best-checkpoint selection on validation loss and experiment logs in `runs/` |
| `generate.py` | Free-running generation from a checkpoint |
| `queue_rules.py` | Rules for `--constrain` (forbid events that are impossible in the current state) |
| `diagnose_generation.py` | Checks generated streams: event mix, customers in system, impossible events, W1 and KS distance to the ground truth |
| `horizon_analysis.py` | Finds for how many generated steps the model stays acceptable: each generation length is scored against what a perfect generator would score, and the length where performance drops is reported |
| `audit_times.py` | Splits cycle time into inter-arrival, wait and service time, reports the effective utilisation of the generated queue and whether the generated events obey the single-server rule |
| `calibration_check.py` | One-step check with the true history: are the predicted event probabilities and next-gap predictions accurate for each event type? Separates one-step bias from errors that build up during free-running generation |
| `run_seed_replicates.py` | Repeats training and diagnosis over seeds and prints one summary table |
| `numerical_studies.py` | Real-data analysis: arrival/service rates, cycle time, parallel vs tandem models, exponential fit tests, optional Transformer comparison |
| `sanity_check_step1.py` / `sanity_check_step2.py` | Server-less pipeline check / single-comparison check on G/G/1 |
| `baseline.py` | Historical-mean and exponential-Markov baselines |
| `utils.py` | Seeding and experiment logging |
| `data_logger.html` | Field tool used to record Arrival / Ordering / Leave per customer |
| `master_real_data.csv`, `example_data.csv` | Real data (6 sessions) and a format example |

## Data format
`system_id,customer_id,event_type,timestamp` (event_type: A, O or D; timestamp
`YYYY-MM-DD HH:MM:SS`). `train.py` needs only `system_id,event_type,timestamp`;
`numerical_studies.py` also needs `customer_id`.

Study 5 of `numerical_studies.py` (peak vs non-peak) groups sessions by the words `peak` / `non peak` in `system_id`.
In `master_real_data.csv` the 14:21-15:27 session is deliberately left unlabelled (`1421-1527`), so it is in the pooled
analyses but not in that comparison. Renaming it to `1421-1527 peak` adds it to the peak group and changes Study 5.

## Reproducing the main experiments
Run from this folder, one command at a time.

Step 1, server-less pipeline check:
    python3 train.py --data random --epochs 150 --num_sim_systems 20 --seed 0 --regenerate_each_epoch
    python3 sanity_check_step1.py --checkpoint checkpoint.pt --arrival_rate 1.0 --mean_wait 70 --mean_service 180

Step 2, G/G/1 built with O2DES (add `--rho_range 0.3 0.9` for stable systems only):
    python3 train.py --data gg1_o2des --epochs 100 --num_sim_systems 20 --seed 0 --regenerate_each_epoch --time_head lognormal --checkpoint checkpoint_o2des.pt
    python3 diagnose_generation.py --checkpoint checkpoint_o2des.pt --truth o2des
    python3 diagnose_generation.py --checkpoint checkpoint_o2des.pt --truth o2des --constrain strict

Generation length and time components (use `--constrain strict` for the rule-on setting):
    python3 horizon_analysis.py --band_only
    python3 horizon_analysis.py --checkpoint checkpoint_o2des.pt --out horizon.csv
    python3 audit_times.py --checkpoint checkpoint_o2des.pt
    python3 audit_times.py --checkpoint checkpoint_o2des.pt --constrain derived_ordering
    python3 calibration_check.py --checkpoint checkpoint_o2des.pt

Other training options: service-time distribution fixed with only the arrival rate varying, and long training sequences:
    python3 train.py --data gg1_o2des --epochs 100 --num_sim_systems 20 --seed 0 --regenerate_each_epoch --time_head lognormal --rho_range 0.3 0.9 --fixed_service 2 25 --fixed_arrival_shape 2 --checkpoint checkpoint_fixed_service.pt
    python3 train.py --data gg1_o2des --epochs 100 --num_sim_systems 80 --seed 0 --regenerate_each_epoch --time_head lognormal --rho_range 0.3 0.9 --seq_len 400 --n_customers_per_system 400 --window_stride 20 --checkpoint checkpoint_len400.pt

More seeds, both training configurations, one summary table (several hours; `--quick` first to test):
    python3 run_seed_replicates.py --quick
    python3 run_seed_replicates.py

Real data:
    python3 numerical_studies.py --real_csv master_real_data.csv --n_servers 2
    python3 train.py --data master_real_data.csv --epochs 60 --seed 0
    python3 numerical_studies.py --real_csv master_real_data.csv --n_servers 2 --transformer_checkpoint checkpoint.pt

## Design notes
- **Loss.** Training loss = cross-entropy of the next event type + negative log-likelihood of the
  next time gap (`--time_head lognormal`: a Gaussian over log(1+gap) for each event type, sampled at
  generation time). `--time_head mse` is the earlier point-estimate head, and is what the
  real-data Transformer results used.
- **Comparison metric.** Wasserstein-1 distance (seconds) between generated and ground-truth
  cycle-time distributions, reported together with the number of runs containing impossible events.
- **Data generators must not be mixed.** `gg1` draws Ordering at a random point between arrival and
  departure; `gg1_o2des` makes Ordering the start of service (its time gap is always 0). Their cycle
  times agree in distribution, their event sequences and wait/service split do not.
- `--regenerate_each_epoch` draws a fresh batch of systems every epoch (reusing one fixed batch of
  generated data also overfitted). `--rho_range LOW HIGH` keeps only systems whose utilisation lies in that range;
  by default about 23% of drawn systems are unstable (utilisation 1 or more).
- `--fixed_service SHAPE SCALE` (with `--rho_range`) keeps the service-time distribution fixed and draws utilisation uniformly in the
  range by varying the arrival rate. `--n_customers_per_system` sets the length of each generated system and `--window_stride` the step
  between training windows; with a long `--seq_len`, fewer windows per epoch means fewer gradient updates, so use more systems or a smaller stride.
- Only customers who arrive after the conditioning events are scored by `horizon_analysis.py`, so the true history given to the model does not count.
  `--grad_clip` limits the gradient norm during training (long `--seq_len` runs can show loss spikes without it).
- `horizon_analysis.py` takes the 95th percentile of the Wasserstein-1 distance that fresh ground-truth runs score at each horizon as the
  acceptance threshold for the model at that horizon; a run is acceptable if it is also free of impossible events.
- `--constrain` (generation) applies hard rules that the model did not learn; report results with the rule on and off.
  `strict`: no Ordering or Leave without someone to attach it to. `allow_balk`: as strict, but a customer may leave without ordering.
  `single_server`: as strict, and no Ordering while another customer is in service. `derived_ordering`: as single_server, and
  Ordering is not sampled but follows the queue state with zero gap (as if only Arrival and Leave were modelled).
  `strict` alone still allows service to start while the server is busy; `audit_times.py` shows this as early starts.
- Training and generation are seeded, but results can differ slightly between machines.

## Known limitations
- A single ground-truth system (utilisation 0.625) and three training seeds per configuration.
- Real-data Transformer results use the earlier MSE head and have not been re-checked with `diagnose_generation.py`.
- The cause of unstable free-running generation is not established; exposure bias is a hypothesis only.
- Trained checkpoints are not included.
