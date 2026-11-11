import math
from collections import defaultdict


class HistoricalMeanBaseline:
    """Predicts: next event type = most common type seen in training;
    next dt = mean dt seen in training. A sanity floor, not a serious model."""

    def fit(self, systems):
        counts = defaultdict(int)
        dts = []
        for seq in systems:
            for e, dt in seq:
                counts[e] += 1
                dts.append(dt)
        self.majority_event = max(counts, key=counts.get)
        self.mean_dt = sum(dts) / len(dts) if dts else 0.0
        var = sum((d - self.mean_dt) ** 2 for d in dts) / len(dts) if dts else 1.0
        self.mse_floor = var  # predicting the mean gives MSE == variance
        return self

    def event_accuracy(self, systems):
        correct, total = 0, 0
        for seq in systems:
            for e, _ in seq:
                correct += int(e == self.majority_event)
                total += 1
        return correct / total if total else 0.0

    def time_mse(self, systems):
        se, n = 0.0, 0
        for seq in systems:
            for _, dt in seq:
                se += (dt - self.mean_dt) ** 2
                n += 1
        return se / n if n else 0.0


class ExponentialMarkovBaseline:
    """
    The natural queueing-theory baseline:
      - event type transitions: first-order Markov chain, MLE = empirical
        transition frequencies
      - dt | event type: exponential distribution, MLE rate = 1 / mean(dt
        for that event type)

    This mirrors the classic assumption behind M/M/n queues (memoryless
    inter-event times), so beating this baseline is evidence the
    Transformer is capturing dependencies BEYOND the standard queueing
    model — which is the research claim being tested.
    """

    def fit(self, systems):
        # Transition counts: P(next_event | current_event)
        trans_counts = defaultdict(lambda: defaultdict(int))
        dt_by_event = defaultdict(list)

        for seq in systems:
            for i, (e, dt) in enumerate(seq):
                dt_by_event[e].append(max(dt, 1e-8))  # avoid log(0)
                if i > 0:
                    prev_e = seq[i - 1][0]
                    trans_counts[prev_e][e] += 1

        self.transition_probs = {}
        for prev_e, next_counts in trans_counts.items():
            total = sum(next_counts.values())
            self.transition_probs[prev_e] = {
                e: c / total for e, c in next_counts.items()
            }

        # exponential MLE rate = 1 / sample mean
        self.rates = {}
        for e, dts in dt_by_event.items():
            mean_dt = sum(dts) / len(dts)
            self.rates[e] = 1.0 / mean_dt

        # fallback for event types never seen transitioning from some state
        self.marginal_event_probs = {
            e: len(dts) / sum(len(d) for d in dt_by_event.values())
            for e, dts in dt_by_event.items()
        }
        return self

    def _event_prob(self, prev_e, next_e):
        if prev_e in self.transition_probs and next_e in self.transition_probs[prev_e]:
            return self.transition_probs[prev_e][next_e]
        return self.marginal_event_probs.get(next_e, 1e-8)

    def negative_log_likelihood(self, systems):
        """
        Average NLL per event, combining:
          - categorical NLL for event type (from Markov transition probs)
          - exponential NLL for dt (from per-event-type rate)
        Directly comparable to a properly-trained NLL-based neural model
        (NOT directly comparable to the current MSE-based Transformer —
        see note in README about switching the model's time head to an
        NLL-based distribution before doing this comparison fairly).
        """
        total_nll, n = 0.0, 0
        for seq in systems:
            for i, (e, dt) in enumerate(seq):
                if i == 0:
                    p_event = self.marginal_event_probs.get(e, 1e-8)
                else:
                    prev_e = seq[i - 1][0]
                    p_event = self._event_prob(prev_e, e)

                rate = self.rates.get(e, 1.0)
                dt_safe = max(dt, 1e-8)
                # exponential log-pdf: log(rate) - rate * dt
                log_pdf_dt = math.log(rate) - rate * dt_safe

                nll = -math.log(max(p_event, 1e-12)) - log_pdf_dt
                total_nll += nll
                n += 1
        return total_nll / n if n else float("nan")

    def event_accuracy(self, systems):
        """Predicts next event = argmax transition prob; compares to actual."""
        correct, total = 0, 0
        for seq in systems:
            for i in range(1, len(seq)):
                prev_e = seq[i - 1][0]
                actual_e = seq[i][0]
                probs = self.transition_probs.get(prev_e, self.marginal_event_probs)
                pred_e = max(probs, key=probs.get) if probs else 0
                correct += int(pred_e == actual_e)
                total += 1
        return correct / total if total else 0.0


if __name__ == "__main__":
    # Quick self-check on simulated data
    from data import generate_simulated_systems, split_systems_three_way

    systems = generate_simulated_systems(num_systems=30)
    train, val, test = split_systems_three_way(systems, seed=0)

    print("=== HistoricalMeanBaseline ===")
    hm = HistoricalMeanBaseline().fit(train)
    print(f"event_accuracy: {hm.event_accuracy(test):.3f}")
    print(f"time_mse: {hm.time_mse(test):.3f}")

    print("\n=== ExponentialMarkovBaseline ===")
    em = ExponentialMarkovBaseline().fit(train)
    print(f"event_accuracy: {em.event_accuracy(test):.3f}")
    print(f"nll: {em.negative_log_likelihood(test):.3f}")
