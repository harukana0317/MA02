import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1), :]


class EventTimeEmbedding(nn.Module):
    def __init__(self, d_model=64, num_event_types=3):
        super().__init__()
        self.event_embed = nn.Embedding(num_event_types, d_model // 2)
        self.time_proj = nn.Linear(1, d_model // 2)

    def forward(self, x):
        event = x[..., 0].long()
        dt = x[..., 1].unsqueeze(-1)
        e = self.event_embed(event)
        t = self.time_proj(dt)
        return torch.cat([e, t], dim=-1)


class QueueTransformer(nn.Module):
    """
    time_mode:
      "mse"       (legacy) the time head outputs ONE number: a point estimate
                  of the next (z-scored) dt, trained with MSE. At generation
                  time it does not know which event type was sampled.
      "lognormal" the time head outputs, FOR EACH event type, the mean and
                  log-std of a Gaussian over y = log(1 + dt). Training uses
                  the TRUE next event type to pick which pair to score
                  (negative log-likelihood); generation uses the SAMPLED
                  event type and then SAMPLES dt from that distribution.
                  So the model describes a proper joint distribution
                  p(event, dt | history) = p(event | history) * p(dt | event, history).
    """

    def __init__(self, d_model=64, nhead=4, num_layers=2, max_len=512, num_event_types=3,
                 time_mode="mse"):
        super().__init__()
        if time_mode not in ("mse", "lognormal"):
            raise ValueError(f"time_mode must be 'mse' or 'lognormal', got {time_mode!r}")
        self.time_mode = time_mode
        self.num_event_types = num_event_types

        self.embedding = EventTimeEmbedding(d_model, num_event_types=num_event_types)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.event_head = nn.Linear(d_model, num_event_types)  # A / O / D logits
        if time_mode == "mse":
            self.time_head = nn.Linear(d_model, 1)   # normalized dt (point estimate)
        else:
            self.time_head = nn.Linear(d_model, 2 * num_event_types)  # (mu, log_sigma) per event type

    def forward(self, x):
        # x: (batch, seq_len, 2) -> [event_id, normalized_dt]
        seq_len = x.size(1)

        h = self.embedding(x)
        h = self.pos_encoding(h)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        h = self.transformer(h, mask=causal_mask, is_causal=True)

        return self.event_head(h), self.time_head(h)

    # ---- helpers used only when time_mode == "lognormal" ----

    def _select_params(self, time_out, event):
        """time_out: (..., 2K); event: (...) long. Returns (mu, log_sigma), each (...)."""
        shape = time_out.shape[:-1]
        out = time_out.reshape(*shape, self.num_event_types, 2)
        idx = event.unsqueeze(-1).unsqueeze(-1).expand(*shape, 1, 2)
        sel = out.gather(-2, idx).squeeze(-2)
        mu = sel[..., 0]
        log_sigma = sel[..., 1].clamp(-5.0, 3.0)   # keeps training numerically stable
        return mu, log_sigma

    def time_nll(self, time_out, next_event, raw_dt, zero_tol=1e-3, reduction="mean"):
        """
        Negative log-likelihood (nats) of the observed inter-event times, in the ORIGINAL units of dt
        (seconds). It scores exactly the distribution that sample_dt() draws from:

            Y = mu + sigma * N(0, 1),   dt = max(expm1(Y), 0)

        so dt has a point mass at 0 and a density for dt > 0:
            dt == 0 (dt <= zero_tol):  -log P(Y <= 0) = -log Phi(-mu / sigma)
            dt  > 0:                   Gaussian density of y = log(1 + dt), plus the Jacobian term y
        Zero gaps are real in this queue (a service start follows the event that frees the server at
        once), so the point mass must be scored; scoring them with the density alone rewards a tiny sigma.
        The value is a likelihood with respect to a mixed (mass + density) reference measure, so compare it
        only with models scored the same way (see baseline.py).

        reduction: "mean" (default), "sum" or "none" (one value per target).
        """
        mu, log_sigma = self._select_params(time_out, next_event)
        sigma = torch.exp(log_sigma)
        y = torch.log1p(raw_dt)
        pos_nll = 0.5 * ((y - mu) / sigma) ** 2 + log_sigma + 0.5 * math.log(2 * math.pi) + y
        zero_nll = -torch.special.log_ndtr(-mu / sigma)          # numerically stable log-CDF
        nll = torch.where(raw_dt > zero_tol, pos_nll, zero_nll)
        if reduction == "mean":
            return nll.mean()
        if reduction == "sum":
            return nll.sum()
        if reduction == "none":
            return nll
        raise ValueError(f"reduction must be 'mean', 'sum' or 'none', got {reduction!r}")

    def sample_dt(self, time_out_last, event):
        """Draw dt (seconds, >= 0) for the given event type(s). Negative draws become exactly 0
        (the point mass that time_nll scores)."""
        mu, log_sigma = self._select_params(time_out_last, event)
        y = mu + torch.exp(log_sigma) * torch.randn_like(mu)
        return torch.expm1(y).clamp(min=0.0)
