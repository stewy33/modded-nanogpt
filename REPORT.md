# Lowest fineweb val_loss in 5 min of single-GPU training — Report

## Result

**Best val_loss = `3.8158`** (config `exp/best.py`, a **6-layer** model), vs the unmodified
baseline `4.1955`. → **−0.380** (a large improvement for this budget).

Reproduce:
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 exp/best.py > logs/best.out 2>&1
# best.py is identical to exp/arch_l6.py
# proof of original result: logs/arch_l6.out ; verification re-run: logs/best_verify.out
```

The two changes from baseline that matter:
1. **Global batch 512 → 64** (no gradient accumulation): ~4–8× more optimizer steps over the
   same tokens. (−0.33)
2. **Depth 12 → 6 layers**: a smaller/faster model trains on more tokens and reaches lower loss
   in this compute-limited regime. (−0.046)

## Environment note (important)
The machine exposed **only 1 H100**, not 4 — so all runs were **sequential**, one single-GPU run
at a time. Every reported number uses `--nproc_per_node=1`, `total_train_minutes=5.0`. `torch.compile`
adds ~130 s of startup and sometimes a mid-run recompile; these cost wall-clock only, **not** the
5-minute training budget — the budget timer is paused during validation, where the recompiles
happened — so all val_loss numbers are directly comparable.

## The story

### 1. Batch size — the first big win
Baseline uses `batch_size=512`, `device_batch_size=64` → **8 grad-accumulation microbatches per
optimizer step**, ~1081 ms/step, only **288 steps** in 5 min. Throughput (~480k tok/s) is fixed by
the hardware, so a smaller global batch buys **more optimizer steps over the same tokens**.

| global batch | steps | val_loss |
|---|---|---|
| 512 (baseline) | 288 | 4.1955 |
| 128 | 1098 | 3.8619 |
| 64  | 1956 | 3.8621 |

512→128 gave **−0.33**. 128 vs 64 **tied** → the benefit of more frequent updates **saturates by
batch ≈ 128**; below that, the token budget (not update count) is the limiter. We keep batch 64.

### 2. Learning rate — insensitive
Sweeping the coupled AdamW+Muon LR around the default `3.6e-3`: `5.0e-3 → 3.8694`,
`3.6e-3 → 3.8621`, `2.4e-3 ≈ tie`. Essentially flat — Muon's orthogonalized update is LR-robust.
Not a useful lever here; kept `3.6e-3`.

### 3. Model depth — the second win (the model is *undertrained*)
In 5 min the 124M model sees ~145M tokens ≈ **1 token/parameter**, far below Chinchilla-optimal
(~20). Loss is bottlenecked by **compute, not capacity** → a smaller, faster model wins:

| n_layer | ms/step | steps | val_loss |
|---|---|---|---|
| 12 | 154 | 1956 | 3.8621 |
| 8  | 113 | 2670 | 3.8252 |
| 6  | 96  | 3145 | **3.8158** |
| 5  | 86  | 3485 | 3.8242 |
| 4  | 76  | 3962 | 3.8393 |

Depth is **U-shaped with a clear minimum at 6 layers**. Smaller models lead *early* (more steps),
but capacity matters in the warmdown/late phase — by 300 s the ordering settles with 6L on top, and
4L hits a capacity floor (loses despite ~4000 steps). 6 layers is the sweet spot between
"more steps" and "enough capacity".

### 4. Schedule (warmdown) — already near-optimal
`warmdown_frac` 0.28 → 0.45 on the 6-layer model gave `3.8192` ≈ tie with `3.8158`. The default
trapezoidal schedule is already good; no gain.

## Conclusion
The winning recipe = baseline architecture/optimizer (Muon + AdamW, RoPE, QK-norm, ReLU² MLP) with
**batch_size = device_batch_size = 64** and **n_layer = 6**. Final val_loss **3.8158**.
The two effective levers both come from the same insight: in a fixed 5-minute, single-GPU,
heavily-undertrained budget, **maximize useful gradient updates and tokens-per-second** (small batch,
shallow model) up to the point where update-frequency saturates and capacity starts to bind.

## Plots
![trajectories](img/trajectories.png)
![final bar](img/final_bar.png)

Full per-experiment table: `RESULTS.md`.
