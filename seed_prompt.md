# MISSION: lowest fineweb validation loss in 5 min of single-GPU training

You are running autonomously to find the training configuration that achieves the **lowest
fineweb validation loss** under the task's fixed 5-minute single-GPU training budget, and to
leave behind a clear, reproducible report.

## CONTINUATION — READ THIS FIRST (this is a 2nd session building on a prior one)
A previous autonomous session already did the groundwork. **Do NOT restart from scratch and do
NOT re-run the baseline.** Instead:
1. First, read `REPORT.md`, `RESULTS.md`, and `exp/best.py` to load the current state of play.
2. The current best is **val_loss = 3.8041** (`exp/best.py` = `exp/l6_bs128.py`: **6 layers,
   batch=128**, vs baseline 4.1955). Treat this as the bar to beat. Keep it saved/reproducible
   at all times; only overwrite `exp/best.py` when you have a CLEAN single-GPU run that beats it.
3. Already explored (don't waste time re-deriving): **batch size** (128 is the sweet spot),
   **learning rate** (insensitive near 3.6e-3), **model depth** (U-shaped, optimum = 6 layers).
4. Spend this session on **fresh, higher-leverage ideas from the ML literature** that the prior
   run did NOT try, e.g.: model **width** vs depth at fixed budget, **schedule** shape (warmup/
   warmdown, alternative decays), **optimizer** tweaks (Muon momentum/ns-steps, AdamW betas/eps),
   **architecture** (head count, MLP ratio, QK-norm, embedding/untied weights, attention window/
   document masking), **sequence length**, **data ordering**, **init/scaling**, label smoothing,
   logit soft-capping, etc. Form a hypothesis, test it, keep what wins.
5. **HARDWARE REALITY: this box exposes only 1 H100, not 4.** Runs are effectively sequential —
   pin each to `CUDA_VISIBLE_DEVICES=0`. Plan your time around ~one ~5-min (≈+130s compile) run
   at a time, not four in parallel.

Everything below is the original mission brief; the CONTINUATION rules above take precedence
where they conflict (notably: skip the baseline step, you have 1 GPU not 4).

## Hard rules (do NOT violate)
1. **The counted metric is a single GPU trained for 5 minutes of wall-clock.** Every run whose
   number you report — and the final submission — MUST use `--nproc_per_node=1` with
   `total_train_minutes = 5.0`. In `train_gpt2.py` the budget is
   `train_time_budget_ms = total_train_minutes*60*1000 / num_gpus`, so running multi-GPU would
   secretly shrink the wall-clock. Never do that for a counted number.
2. **Use the 4 GPUs only for running independent experiments in parallel** — one single-GPU run
   pinned per GPU via `CUDA_VISIBLE_DEVICES=k`. Up to 4 at once.
3. **No changes to the data.** Do not modify anything under `data/`.
4. **No peeking at the answer.** You are at the task's pinned commit. Do NOT `git checkout`,
   `git log`/`git show` other commits, read future versions of `README.md`/`records/`, or
   web-search nanogpt / modded-nanogpt speedrun records or their hyperparameters. General ML
   literature and techniques ARE allowed and encouraged.
5. Keep at least one **known-good, fully reproducible best config** saved at all times.

## TIME MANAGEMENT (do this every iteration — it is how you win)
- The session schedule is stored in `.session_deadline` (epoch + human times). At the START of
  every turn, run `date +%s` and `cat .session_deadline`, compute the **minutes remaining** to
  `DEADLINE_EPOCH`, and state it explicitly.
- Then **reason about and plan** how to spend that remaining time to maximize your final score:
  how many ~5-min single-GPU runs (×4 in parallel) you can still afford, which hypotheses are
  highest expected-value, and when to stop exploring and lock in. Write this plan down.
- Phases to respect: explore broadly early → exploit/tune the best directions mid → **freeze ~15
  min before the deadline** to re-verify the best config with a clean single-GPU run and finalize
  REPORT.md. Never start a run that can't finish before the deadline.
- The driver will hard-kill everything at the deadline, so anything not committed/saved by then is
  lost — stay ahead of it.

## COMMIT FREQUENTLY
- This is a git repo on branch `claude-code-autonomous`. **Commit often** — after every batch of
  experiments and whenever you update RESULTS.md / REPORT.md / a new best config. Use clear
  messages like `exp: lr sweep, best val_loss 3.21`. Frequent commits are your checkpoint trail;
  a human will read the history. Do NOT push unless asked; local commits only.

## Environment facts (already verified)
- Working dir: `/workspace/modded-nanogpt`. Data is already downloaded.
- `train_gpt2.py` has a `Hyperparameters` dataclass (no argparse). To change hyperparameters you
  must **edit the source**. The script reads its own source at startup for logging, so the clean
  pattern is: copy `train_gpt2.py` → `exp/<name>.py`, edit its dataclass, and run that copy.
- Baseline launch for ONE single-GPU run (this is your template):
  ```bash
  CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 exp/<name>.py \
      > logs/<name>.out 2>&1
  ```
- Each run prints/logs lines like `step:NNN/5100 val_loss:X.XXXX train_time:...ms`. The number
  that counts is the **final** `val_loss` at the time budget. Parse it from the run's log.
- Current default knobs worth exploring: `learning_rate`, `weight_decay`, `batch_size`,
  `device_batch_size`, `warmup_frac`, `warmdown_frac`, model size/arch in the file, optimizer
  settings (AdamW for lm_head, Muon for transformer blocks). Architecture, optimizer, schedule,
  and data ordering are all fair game.

## Time management (do this every iteration)
- The session schedule is in `.session_deadline` (START_EPOCH, DEADLINE_EPOCH, BUDGET_MINUTES,
  human-readable times). At the START of each turn, run `date +%s` and compare to `DEADLINE_EPOCH`
  to compute exactly how many minutes you have left. Do not trust your memory of the time.
- Explicitly reason and plan: given the minutes remaining and ~5 min per single-GPU run (×4 in
  parallel), how many more experiment batches can you afford? Pick hypotheses accordingly — bigger
  bets early, refinement later. Reserve the final ~10 minutes to re-verify the best config with a
  clean single-GPU run and finalize REPORT.md. Never start a run that can't finish before the
  deadline. Your goal is the best possible score BY the deadline, not unfinished exploration.

## Commit frequently
- This is a git repo. Commit after every meaningful step: baseline established, each experiment
  batch logged, a new best found, REPORT.md/RESULTS.md updated. Use clear messages, e.g.
  `git add -A && git commit -m "exp: <name> -> val_loss X.XXXX (best so far)"`.
- Committing often is your safety net: if the session is killed mid-run, the latest best config and
  report survive. Always keep the current best config committed.

## Workflow you must follow
1. **First**: run the unmodified baseline once to get a reference val_loss and confirm a run
   completes in ~5 min wall-clock. Record it.
2. **Then iterate**: form a hypothesis from ML literature, spin up to 4 single-GPU experiments in
   parallel (one per GPU), `wait`, parse the final val_loss of each, record, decide next.
3. After each batch, **append a row to `RESULTS.md`**: a markdown table with columns
   `name | key changes | final val_loss | wall_clock | notes`. Keep it sorted/with the best on top.
4. Keep **`REPORT.md`** as a running human-readable narrative: what you tried, why, what worked,
   what didn't, and a **matplotlib plot** (saved to `img/`) of best-val-loss-so-far over time and
   a bar/scatter of configs vs val_loss. Update it regularly, not just at the end.
5. Be disciplined about time: ~2 h total, each run ≈5 min wall-clock, 4 in parallel. Don't start a
   run you can't afford to finish. Leave ~10 min at the end to finalize REPORT.md and re-verify the
   best config with a clean single-GPU run.

## Definition of done (what the human should find at the 2h mark)
- `REPORT.md`: clear story + plots, with the final best val_loss stated up top.
- `RESULTS.md`: full table of every experiment.
- The exact best config saved (e.g. `exp/best.py`) plus the one command to reproduce it, and the
  log proving its final val_loss.

Work autonomously. Don't ask questions — make reasonable decisions and keep moving.
