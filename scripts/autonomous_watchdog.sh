#!/bin/bash
# Runs the monitor every 3 hours indefinitely
# To stop: pkill -f autonomous_watchdog

LOG="/Users/ember/dev/restrans/logs/watchdog.log"
mkdir -p "$(dirname "$LOG")"

echo "Watchdog started at $(date -Iseconds)" >> "$LOG"

while true; do
    bash /Users/ember/dev/restrans/scripts/monitor_and_log.sh
    echo "Cycle complete at $(date -Iseconds). Sleeping 3h..." >> "$LOG"
    sleep 10800
done
