# Experiment Results — lowest fineweb val_loss in 5-min single-GPU

Environment: **1× H100 80GB (only 1 GPU available** despite the brief mentioning 4 — so all runs
were **sequential**). Budget: `total_train_minutes=5.0`, `--nproc_per_node=1`. Metric = final
`val_loss` at the 5-min (300 s) training stop. `step_avg` = ms per optimizer step (compile/recompile
time is excluded from the budget).

Sorted best-first:

| name | key changes vs baseline | final val_loss | steps | step_avg | notes |
|------|------------------------|---------------:|------:|---------:|-------|
| **best.py (= l6_bs128)** | **n_layer=6, batch=128, dbs=128** | **3.8041** | **1753** | **172ms** | **BEST — larger device batch, +13% tok/s via MFU, still above update-saturation; `logs/l6_bs128.out`** |
| narrow512 | 6L, **n_embd=512** (n_head=4), bs128 | 3.8108 | 2686 | 112ms | **near-tie** w/ best; narrower=much faster (+53% steps), capacity binds only in warmdown; `logs/narrow512.out` |
| arch_l6 (verified) | n_layer=6, batch=64 | 3.8125 | 3241 | 93ms | prior best; bs64 |
| arch_l6 | n_layer=6, batch=64 | 3.8158 | 3145 | 96ms | original best run; `logs/arch_l6.out` |
| l6_wd45 | 6L, warmdown_frac=0.45 | 3.8192 | 3099 | 97ms | ~tie; default warmdown already near-optimal |
| arch_l5 | n_layer=5, batch=64 | 3.8242 | 3485 | 86ms | confirms U-shaped depth curve |
| arch_l8 | n_layer=8, batch=64 | 3.8252 | 2670 | 113ms | smaller=faster=more tokens wins |
| arch_l4 | n_layer=4, batch=64 | 3.8393 | 3962 | 76ms | worse: capacity floor; 6L is the optimum |
| bs128 | batch=128, dbs=128, no grad-accum | 3.8619 | 1098 | 276ms | 1st big win: 512→128 (−0.33) |
| bs64 | batch=64, dbs=64, no grad-accum | 3.8621 | 1956 | 154ms | ties bs128 → batch benefit saturates ≤128 |
| bs64_lr0050 | batch=64, lr=5.0e-3 | 3.8694 | 1943 | 155ms | ~tie; LR insensitive |
| bs64_lr0024 | batch=64, lr=2.4e-3 | ~tie (stopped) | ~1400 | 153ms | tracked lr0036; stopped early (LR insensitive) |
| baseline | bs=512, dbs=64, lr=3.6e-3, wd=0, warmdown=0.28 | 4.1955 | 288 | 1081ms | reference (8 grad-accum microbatches/step) |

(`bs256` was prepared but its first launch crashed during the initial 4-way parallel attempt,
before we discovered only 1 GPU exists; not re-run since 128/64 already characterised the batch axis.)

### Session 2 — negative results (stopped early once clearly behind at 60s)
| name | change | val@60s vs best(4.5019) | verdict |
|------|--------|------------------------:|---------|
| width1024 | n_embd=1024, n_head=8 | 4.8342 (245ms/step, −42% steps) | LOSE — wider=slower, throughput dominates undertrained regime |
| softcap | tanh logit soft-cap @15 | 4.5596 (177ms/step) | LOSE — slower + behind; no quality gain here |

**Width sweep (6L, bs128):** 512→**3.8108**, 768→**3.8041** (best), 1024→lose. Width optimum is at/near 768;
512 nearly ties despite 53% more steps. Testing 640 to see if the in-between point edges out 768.

## Two effective levers
1. **Smaller global batch** (512→64): more optimizer steps per token. Saturates by batch≈128.
2. **Shallower model** (12→6 layers): in this heavily-undertrained 5-min budget, faster throughput
   (more tokens) beats extra capacity — until ~4 layers, where capacity binds. Optimum = **6 layers**.

LR and warmdown_frac were both ~insensitive. See `REPORT.md` for the full narrative and plots.
