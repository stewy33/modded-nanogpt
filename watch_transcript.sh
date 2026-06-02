#!/usr/bin/env bash
# Pretty-prints a headless Claude stream-json log as a readable live transcript.
# Usage: ./watch_transcript.sh <path-to-claude.log>
set -u
LOG="${1:?usage: watch_transcript.sh <claude.log>}"

# Wait for the log to exist (driver may start us a beat early).
while [ ! -f "$LOG" ]; do sleep 0.5; done

echo "[transcript] following $LOG  (Ctrl-b then & to close this window)"
echo

if command -v jq >/dev/null 2>&1; then
  tail -n +1 -f "$LOG" | jq -rcR --unbuffered '
    def c(code): "[\(code)m";
    fromjson? // empty |
    if .type=="assistant" then (.message.content[]? |
        if .type=="text" then "\n"+c("36")+"● "+.text+c("0")
        elif .type=="tool_use" then c("33")+"  ⏵ "+.name+" "+(.input|tojson|.[0:160])+c("0")
        else empty end)
    elif .type=="user" then (.message.content[]? |
        if (.type?=="tool_result")
          then c("90")+"  ⏴ "+((.content|if type=="array" then (.[0].text//"") else . end)|tostring|.[0:200])+c("0")
          else empty end)
    elif .type=="result" then "\n"+c("32")+"━━ turn done: "+(.subtype//"ok")+" ━━"+c("0")
    else empty end
  '
else
  # Fallback: raw follow if jq is unavailable.
  tail -n +1 -f "$LOG"
fi
