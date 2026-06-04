# Experiment Results — lowest fineweb val_loss in 5-min single-GPU

Environment: **1× H100 80GB (only 1 GPU available** despite the brief mentioning 4 — so all runs
were **sequential**). Budget: `total_train_minutes=5.0`, `--nproc_per_node=1`. Metric = final
`val_loss` at the 5-min (300 s) training stop. `step_avg` = ms per optimizer step (compile/recompile
time is excluded from the budget).

Sorted best-first:

| name | key changes vs baseline | final val_loss | steps | step_avg | notes |
|------|------------------------|---------------:|------:|---------:|-------|
| **best.py (= narrow640)** | **6L, n_embd=640 (n_head=5), bs128** | **3.7967** | **2126** | **142ms** | **NEW BEST — width sweet spot; 3 clean runs {3.7967, 3.7975, 3.8014}, mean 3.7985; proof `logs/best.out`** |
| b_wd40 | w640 6L, **warmdown_frac=0.40** | 3.7971 | 2112 | 143ms | ties best (Δ0.0004); warmdown 0.28–0.40 flat; `logs/b_wd40.out` |
| l7_w640 | **n_layer=7**, n_embd=640, bs128 | 3.7983 | 1939 | 156ms | **ties best** (Δ0.0008, noise); depth flat at 6–7 for w640; `logs/l7_w640.out` |
| l6_bs128 (prev best) | n_layer=6, batch=128, n_embd=768 | 3.8041 | 1753 | 172ms | prior best; larger device batch; `logs/l6_bs128.out` |
| narrow512 | 6L, **n_embd=512** (n_head=4), bs128 | 3.8108 | 2686 | 112ms | near-tie; narrower=much faster (+53% steps), capacity binds in warmdown; `logs/narrow512.out` |
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
| b_untie | untie wte/lm_head (both in AdamW) | 4.7098 (143ms/step) | LOSE badly — tied embeddings benefit from shared gradients in undertrained regime |
| b_muonlr2x | Muon lr 0.1×→0.2× base | ~tie at 120s (stopped) | no gain — Muon LR already well-tuned |
| b_lr47 | LR 3.6e-3→4.7e-3 | 3.7996 | ~tie (slightly behind throughout) — LR insensitivity reconfirmed at w640 |
| b_mom98 | Muon momentum 0.95→0.98 | 4.6722 @60s (stopped) | LOSE badly — high momentum overshoots in short run; 0.95 optimal |
| b_wd1 | weight_decay 0→0.1 | 4.1829 @120s vs 4.1641 (stopped) | LOSE — nothing to regularize when undertrained; wd just slows learning; wd=0 optimal |
| b_bs64 | w640, batch 128→64 | 4.0919@180s vs 4.0391 (stopped) | LOSE — batch saturated; 64 adds gradient noise, no benefit |
| b_mlp3x | w640, MLP ratio 4×→3× | 3.8080 (134ms/step) | LOSE (Δ+0.010) — lost MLP capacity not recovered by extra steps; 4× is right |
| b_hd64 | w640, head_dim 128→64 (n_head 5→10) | 4.5517 @60s vs 4.4383 (stopped) | LOSE badly — head_dim 128 is optimal |

**Width sweep (6L, bs128):** 512→3.8108, **640→3.7975 (BEST)**, 768→3.8041, 1024→lose.
Width is U-shaped with optimum at **n_embd=640** — the same "maximize useful tokens up to where capacity
binds" tradeoff as depth, now on the width axis. 640 gets 2143 steps (vs 768's 1753) yet keeps enough
capacity that its warmdown (−0.131) matches the wider model's.

## Two effective levers
1. **Smaller global batch** (512→64): more optimizer steps per token. Saturates by batch≈128.
2. **Shallower model** (12→6 layers): in this heavily-undertrained 5-min budget, faster throughput
   (more tokens) beats extra capacity — until ~4 layers, where capacity binds. Optimum = **6 layers**.

LR and warmdown_frac were both ~insensitive. See `REPORT.md` for the full narrative and plots.
