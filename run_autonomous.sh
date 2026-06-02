#!/usr/bin/env bash
# Autonomous 2-hour driver for the fineweb val-loss speedrun task.
# Runs Claude Code headless in a resume-loop with a hard wall-clock kill switch.
#
# Usage:
#   tmux new -s speedrun
#   ./run_autonomous.sh            # 2 hours (default)
#   ./run_autonomous.sh 90         # custom: 90 minutes
#   Ctrl-b d to detach; `tmux attach -t speedrun` to peek.

set -u
cd "$(dirname "$0")"

WALL_MIN="${1:-150}"                       # total wall-clock budget in minutes (default 2.5h)
START=$(date +%s)
DEADLINE=$(( START + WALL_MIN*60 ))
LOGDIR="runs/autonomous_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
SEED="$(dirname "$0")/seed_prompt.md"
LOG="$LOGDIR/claude.log"

# Persist the schedule so Claude can check the clock itself and plan its remaining time.
DEADLINE_FILE="$(dirname "$0")/.session_deadline"
{
  echo "START_EPOCH=$START"
  echo "DEADLINE_EPOCH=$DEADLINE"
  echo "BUDGET_MINUTES=$WALL_MIN"
  echo "START_HUMAN=$(date -d "@$START")"
  echo "DEADLINE_HUMAN=$(date -d "@$DEADLINE")"
} > "$DEADLINE_FILE"

echo "[driver] start $(date) | budget ${WALL_MIN}m | deadline $(date -d "@$DEADLINE") | logdir $LOGDIR" | tee -a "$LOG"

# Auto-launch a live transcript viewer in a separate tmux window (if we're inside tmux).
VIEWER="$(dirname "$0")/watch_transcript.sh"
if [ -n "${TMUX:-}" ]; then
  tmux new-window -n transcript "bash '$VIEWER' '$LOG'"
  tmux last-window 2>/dev/null   # return focus to the driver window
  echo "[driver] transcript viewer opened in tmux window 'transcript' (Ctrl-b n / Ctrl-b p to switch)" | tee -a "$LOG"
else
  echo "[driver] NOT inside tmux — to watch the transcript run in another shell:" | tee -a "$LOG"
  echo "         bash '$VIEWER' '$LOG'" | tee -a "$LOG"
fi

first_done=""
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  remaining=$(( (DEADLINE - $(date +%s)) / 60 ))
  [ "$remaining" -lt 1 ] && break

  if [ -z "$first_done" ]; then
    msg="$(cat "$SEED")

== LIVE CONTEXT ==
You have ~${remaining} minutes of WALL-CLOCK left for the entire session. Budget accordingly."
  else
    msg="Continue the speedrun. ~${remaining} minutes of wall-clock left.
Keep iterating to LOWER the validation loss. Log every run's result to RESULTS.md and keep
REPORT.md + the plot up to date. If you feel 'done', you are not — pull another idea from the
ML literature and test it. Always keep your best-known config saved and reproducible."
  fi

  # Per-iteration timeout is bounded by remaining time, so the only kill is at the deadline.
  # --continue resumes the SAME session so all prior context/results persist across iterations.
  timeout "${remaining}m" claude -p "$msg" \
      ${first_done:+--continue} \
      --dangerously-skip-permissions \
      --verbose --output-format stream-json \
      >> "$LOG" 2>&1

  first_done=1
  echo "[driver] iteration returned at $(date), ${remaining}m had remained" | tee -a "$LOG"
  sleep 2
done

# ---- HARD STOP: free all GPUs no matter what Claude left running ----
echo "[driver] DEADLINE hit at $(date) — killing any in-flight training" | tee -a "$LOG"
pkill -9 -f train_gpt2.py 2>/dev/null
pkill -9 -f torchrun      2>/dev/null
pkill -9 -f "exp/"        2>/dev/null
sleep 3
nvidia-smi --query-gpu=index,memory.used --format=csv | tee -a "$LOG"
echo "[driver] stopped. Logs in $LOGDIR ; report in REPORT.md ; results in RESULTS.md" | tee -a "$LOG"

# ---- FINALIZE: render a human-readable transcript, then add/commit/push everything ----
# NOTE: runs/ is gitignored, so write the committed transcript to the repo ROOT.
TRANSCRIPT="$(dirname "$0")/TRANSCRIPT.txt"
echo "[driver] rendering transcript -> $TRANSCRIPT" | tee -a "$LOG"
if command -v jq >/dev/null 2>&1; then
  # Same view as watch_transcript.sh but plain text (no ANSI) for a clean committed file.
  jq -rcR '
    fromjson? // empty |
    if .type=="assistant" then (.message.content[]? |
        if .type=="text" then "\n● "+.text
        elif .type=="tool_use" then "  ⏵ "+.name+" "+(.input|tojson|.[0:300])
        else empty end)
    elif .type=="user" then (.message.content[]? |
        if (.type?=="tool_result")
          then "  ⏴ "+((.content|if type=="array" then (.[0].text//"") else . end)|tostring|.[0:500])
          else empty end)
    elif .type=="result" then "\n━━ turn done: "+(.subtype//"ok")+" ━━"
    else empty end
  ' "$LOG" > "$TRANSCRIPT" 2>/dev/null
else
  cp "$LOG" "$TRANSCRIPT"
fi
# Keep the raw stream-json too, for completeness (also at repo root so it's not gitignored).
cp "$LOG" "$(dirname "$0")/TRANSCRIPT.stream-json.jsonl" 2>/dev/null

BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
echo "[driver] committing + pushing on branch '$BRANCH'" | tee -a "$LOG"
git add -A
git commit -m "autonomous speedrun session: report, results, and full transcript

Budget ${WALL_MIN}m. Final artifacts: REPORT.md, RESULTS.md, TRANSCRIPT.txt

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" 2>&1 | tee -a "$LOG"
git push origin "$BRANCH" 2>&1 | tee -a "$LOG" \
  || echo "[driver] WARN: push failed (check remote/auth) — commit is saved locally" | tee -a "$LOG"
echo "[driver] done at $(date)." | tee -a "$LOG"
