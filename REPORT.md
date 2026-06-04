# Lowest fineweb val_loss in 5 min of single-GPU training — Report

**Best val_loss so far: `3.8158`** (config `exp/arch_l6.py`, a 6-layer model).
Baseline (unmodified) was `4.1955`. → **−0.380 improvement.**

Reproduce the best:
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 exp/best.py > logs/best.out 2>&1
# best.py == arch_l6.py
```

## Environment note
The box exposed **only 1 H100** (not 4), so all experiments were run **sequentially**, one
single-GPU run at a time. Every reported number uses `--nproc_per_node=1`, `total_train_minutes=5.0`.
A persistent issue: `torch.compile` adds ~130s startup and occasionally a mid-run recompile;
these only cost wall-clock, **not** the 5-minute training budget (the timer is paused during
validation, where the recompiles occurred), so the metrics are clean.

## The story

### 1. Batch size: the first big win
The baseline uses `batch_size=512` with `device_batch_size=64` → **8 gradient-accumulation
microbatches per optimizer step**, costing ~1081 ms/step and yielding only **288 optimizer steps**
in 5 minutes. Throughput (~480k tok/s) is set by the hardware, so a smaller global batch buys
**more optimizer steps over the same number of tokens**.

| global batch | steps in 5 min | val_loss |
|---|---|---|
| 512 (baseline) | 288 | 4.1955 |
| 256 | — | (skipped) |
| 128 | 1098 | 3.8619 |
| 64  | 1956 | 3.8621 |

Dropping 512→128 gave a **huge −0.33**. 128 vs 64 was a **tie** → the benefit of more frequent
updates **saturates by batch≈128**; below that the token budget, not update count, is the limiter.

### 2. Learning rate: insensitive
Sweeping the (coupled AdamW+Muon) LR around the default `3.6e-3`:
`5.0e-3 → 3.8694`, `3.6e-3 → 3.8621`, `2.4e-3 ≈ tie`. Essentially flat — Muon's orthogonalized
update is fairly LR-robust. Not a productive lever here.

### 3. Model size: the second win (the model is *undertrained*)
In 5 min the 124M model sees ~145M tokens ≈ **1 token/parameter** — far below Chinchilla-optimal
(~20). Loss is bottlenecked by **compute/undertraining, not capacity**. So a **smaller, faster
model trains on more tokens and reaches lower loss**:

| n_layer | ms/step | steps | val_loss |
|---|---|---|---|
| 12 (default) | 154 | 1956 | 3.8621 |
| 10 | — | — | (pending/optional) |
| 8 | 113 | 2670 | 3.8252 |
| 6 | 96  | 3145 | **3.8158** |
| 4 | 71  | ~4400 | (running) |

The trend `12 > 8 > 6` confirms the hypothesis, though gains shrink (−0.037, then −0.009),
suggesting we're approaching the depth sweet spot. `n_layer=4` is being tested to find the floor.

## What's next / remaining levers
- Finish the depth floor (l4) and pick the best depth.
- One tuning run on the winner (warmdown fraction / minor width).
- Clean final re-verification run of the chosen best config.

## Plots
![trajectories](img/trajectories.png)
![final bar](img/final_bar.png)

See `RESULTS.md` for the full per-experiment table.
