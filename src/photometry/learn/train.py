"""Training loop: on-the-fly sampled examples, Adam + warmup/cosine."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .data import GeometryPool, sample_batch
from .model import SpinNet, spin_loss


def train(pool: GeometryPool, shapes, out_dir: Path, steps: int = 3000,
          batch: int = 32, n_tokens: int = 256, width_s: float = 7200.0,
          lr: float = 1e-3, seed: int = 0, log_every: int = 100,
          model_kw: dict | None = None) -> tuple[SpinNet, list[dict]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SpinNet(**(model_kw or {}))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warm = max(1, steps // 20)

    def lr_at(s):
        if s < warm:
            return lr * (s + 1) / warm
        return lr * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))

    history, t0 = [], time.time()
    model.train()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        tok, mask, pole, logp, axis, ok = sample_batch(pool, shapes, rng, batch,
                                                       n_tokens=n_tokens,
                                                       width_s=width_s)
        pred = model(torch.from_numpy(tok), torch.from_numpy(mask))
        loss, parts = spin_loss(pred, torch.from_numpy(pole),
                                torch.from_numpy(logp), torch.from_numpy(axis),
                                torch.from_numpy(ok))
        parts["tier0_frac"] = float(ok.mean())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            rec = dict(step=step, loss=float(loss), **parts,
                       elapsed_s=time.time() - t0)
            history.append(rec)
            print(f"step {step:5d} loss {loss:.4f} pole {parts['pole']:.4f} "
                  f"period {parts['period']:.4f} axis {parts['axis']:.4f} "
                  f"[{rec['elapsed_s']:.0f}s]", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "spin_net.pt")
    with open(out_dir / "train_history.json", "w") as f:
        json.dump(dict(steps=steps, batch=batch, n_tokens=n_tokens,
                       width_s=width_s, history=history), f, indent=1)
    return model, history
