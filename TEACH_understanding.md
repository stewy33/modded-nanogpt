# Understanding `train_gpt2.py` + `run.py` queue — running checklist

Goal: deep enough to present on it and answer hard questions.

## 1. The problem / motivation
- [ ] What the take-home actually asks (objective, constraints, hardware)
- [ ] Why a queue/orchestration system exists at all (4 GPUs, parallel experiments)
- [ ] Why "single-file snapshot per run" is the design choice

## 2. train_gpt2.py — the training script
### Model architecture
- [ ] Overall GPT-2 shape (embeddings, blocks, lm_head, weight tying)
- [ ] Block: pre-norm, residual, attn + MLP
- [ ] RMSNorm (vs LayerNorm), where it's applied
- [ ] Rotary embeddings (RoPE) — what/why, QK-norm
- [ ] MLP: relu² activation choice
- [ ] FlexAttention + document block mask (the big custom piece)
### Optimization
- [ ] Two optimizers: AdamW (lm_head) + Muon (transformer blocks) — why split
- [x] Muon: what it does (momentum → orthogonalize via Newton-Schulz)
- [x] Distributed Muon: sharding params across ranks, all_reduce (SUM + zero buffer)
- [x] Two-optimizer split (AdamW lm_head / Muon blocks) + why
- [x] Zero-init c_proj → blocks start as identity on residual
- [ ] Time-based trapezoidal LR schedule (warmup/flat/warmdown)
- [ ] Gradient accumulation + DDP no_sync
### Training-loop mechanics
- [ ] Time budget vs num_iterations; minutes / num_gpus logic
- [ ] Step-10 timer reset (compile warmup)
- [ ] Cross-rank clock broadcast (why — deadlock avoidance)
- [ ] Eval scheduling, val_loss all_reduce
### Data loading
- [ ] .bin shard format, header, uint16 tokens
- [ ] DistributedDataLoader sharding by rank, prefetch / CUDA streams
- [ ] token limits / budget
### Misc/perf
- [ ] torch.compile, bf16 autocast, tf32 logits
- [ ] Profiler path

## 3. run.py — the queue
- [ ] add/worker/cell/ls/rm/cancel lifecycle
- [ ] JSON files as the queue; atomic writes
- [ ] GPU-aware scheduler (backfill, parallel placement)
- [ ] tmux window per run; snapshot as source of truth
- [ ] cancellation / reaping

## 4. Broader context
- [ ] How the knobs you tuned (bs, lr, NS steps, flex-attn) map to results
- [ ] What you'd do next & why
