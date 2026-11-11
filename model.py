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

    def time_nll(self, time_out, next_event, raw_dt):
        """
        Mean negative log-likelihood (nats) of the observed inter-event times
        under the model, in the ORIGINAL units of dt (seconds), including the
        change-of-variables term, so the number is a proper density and can be
        compared with other density models such as the exponential baseline.
        raw_dt must be >= 0 (zero gaps are fine: we model log(1 + dt)).
        """
        mu, log_sigma = self._select_params(time_out, next_event)
        y = torch.log1p(raw_dt)
        nll = 0.5 * ((y - mu) / torch.exp(log_sigma)) ** 2 + log_sigma \
              + 0.5 * math.log(2 * math.pi) + y        # "+ y" = log(1+dt), the Jacobian
        return nll.mean()

    def sample_dt(self, time_out_last, event):
        """Draw dt (seconds, >= 0) for the given event type(s)."""
        mu, log_sigma = self._select_params(time_out_last, event)
        y = mu + torch.exp(log_sigma) * torch.randn_like(mu)
        return torch.expm1(y).clamp(min=0.0)
