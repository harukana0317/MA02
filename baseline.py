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
    A deliberately simple reference model:
      - event types: first-order Markov chain. Every row P(next | previous) is a complete,
        normalised distribution over all event types (add-alpha smoothing), so each row sums to 1.
      - dt given the next event type: zero-inflated exponential. P(dt = 0) is the (smoothed) share
        of zero gaps for that event type; positive gaps follow an exponential whose rate is
        1 / mean(positive gaps). This is the same kind of distribution as the Transformer's censored
        time head (a point mass at 0 plus a density), so the two NLLs are comparable.

    It does NOT track queue length or server state, so it is not an M/M/1 simulator: in a real M/M/1
    queue those quantities change what happens next. It is a "no memory of the queue" reference, and
    beating it shows only that the model uses more than the previous event type.
    """

    def __init__(self, num_event_types=3, alpha=1.0, zero_tol=1e-3):
        self.K = num_event_types
        self.alpha = alpha
        self.zero_tol = zero_tol

    def fit(self, systems):
        K, a = self.K, self.alpha
        trans = [[0] * K for _ in range(K)]
        first = [0] * K
        marginal = [0] * K
        n_zero, n_pos, sum_pos = [0] * K, [0] * K, [0.0] * K

        for seq in systems:
            for i, (e, dt) in enumerate(seq):
                marginal[e] += 1
                if dt <= self.zero_tol:
                    n_zero[e] += 1
                else:
                    n_pos[e] += 1
                    sum_pos[e] += dt
                if i == 0:
                    first[e] += 1
                else:
                    trans[seq[i - 1][0]][e] += 1

        def normalise(counts):
            total = sum(counts) + a * K
            return [(c + a) / total for c in counts]

        self.marginal_event_probs = {e: p for e, p in enumerate(normalise(marginal))}
        self.first_event_probs = {e: p for e, p in enumerate(normalise(first))}
        # a previous event type never seen as a predecessor falls back to the (normalised) marginal row
        self.transition_probs = {
            prev: ({e: p for e, p in enumerate(normalise(trans[prev]))} if sum(trans[prev]) > 0
                   else dict(self.marginal_event_probs))
            for prev in range(K)
        }
        self.p_zero = {e: (n_zero[e] + a) / (n_zero[e] + n_pos[e] + 2 * a) for e in range(K)}
        self.rates = {e: (n_pos[e] / sum_pos[e]) if n_pos[e] > 0 else 1.0 for e in range(K)}
        return self

    def event_row(self, prev_e):
        """The full next-event distribution after prev_e (prev_e=None: first event). Always sums to 1."""
        return self.first_event_probs if prev_e is None else self.transition_probs[prev_e]

    def _event_prob(self, prev_e, next_e):
        return self.event_row(prev_e)[next_e]

    def _time_nll(self, e, dt):
        if dt <= self.zero_tol:
            return -math.log(self.p_zero[e])
        rate = self.rates[e]
        return -(math.log(1.0 - self.p_zero[e]) + math.log(rate) - rate * dt)

    def negative_log_likelihood(self, systems):
        """
        Average NLL per event = categorical NLL of the event type (Markov row) + NLL of dt under the
        zero-inflated exponential for that event type. The same probabilities are used for prediction
        and for the likelihood.
        """
        total_nll, n = 0.0, 0
        for seq in systems:
            for i, (e, dt) in enumerate(seq):
                prev_e = None if i == 0 else seq[i - 1][0]
                total_nll += -math.log(self._event_prob(prev_e, e)) + self._time_nll(e, dt)
                n += 1
        return total_nll / n if n else float("nan")

    def event_accuracy(self, systems):
        """Predicts next event = argmax of the Markov row; compares to actual."""
        correct, total = 0, 0
        for seq in systems:
            for i in range(1, len(seq)):
                row = self.transition_probs[seq[i - 1][0]]
                pred_e = max(row, key=row.get)
                correct += int(pred_e == seq[i][0])
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
