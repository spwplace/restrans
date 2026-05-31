#!/bin/bash
# Periodic training monitor — run via cron or background task
# Logs status to a single file that accumulates over time

LOGFILE="/Users/ember/dev/restrans/logs/training_monitor.log"
mkdir -p "$(dirname "$LOGFILE")"

echo "========================================" >> "$LOGFILE"
echo "CHECK-IN: $(date -Iseconds)" >> "$LOGFILE"
echo "========================================" >> "$LOGFILE"

# Nextop status
echo "--- NEXTOP ---" >> "$LOGFILE"
if [ -f /Users/ember/dev/restrans/logs/webscale/std_alibi_20M_s42.pid ]; then
    NP=$(cat /Users/ember/dev/restrans/logs/webscale/std_alibi_20M_s42.pid)
    ps -p "$NP" -o pid,pcpu,etime 2>/dev/null >> "$LOGFILE" || echo "std_alibi finished" >> "$LOGFILE"
fi
if [ -f /Users/ember/dev/restrans/logs/webscale/phase_dyn_20M_s42.pid ]; then
    NP=$(cat /Users/ember/dev/restrans/logs/webscale/phase_dyn_20M_s42.pid)
    ps -p "$NP" -o pid,pcpu,etime 2>/dev/null >> "$LOGFILE" || echo "phase_dyn finished" >> "$LOGFILE"
fi
tail -n 3 /Users/ember/dev/restrans/logs/webscale/std_alibi_20M_s42.log >> "$LOGFILE" 2>/dev/null
tail -n 3 /Users/ember/dev/restrans/logs/webscale/phase_dyn_20M_s42.log >> "$LOGFILE" 2>/dev/null

# Storage
df -h /System/Volumes/Data | tail -n 1 >> "$LOGFILE"

# Persvati status (if reachable)
echo "--- PERSVATI ---" >> "$LOGFILE"
ssh -i ~/.ssh/id_aws -o ConnectTimeout=5 persvati.local "
if [ -f /home/ember/restrans-exp/logs/std_alibi_20M_s42.pid ]; then
    ps -p \"\$(cat /home/ember/restrans-exp/logs/std_alibi_20M_s42.pid)\" -o pid,pcpu,etime 2>/dev/null || echo 'std_alibi finished'
fi
if [ -f /home/ember/restrans-exp/logs/phase_dyn_20M_s42.pid ]; then
    ps -p \"\$(cat /home/ember/restrans-exp/logs/phase_dyn_20M_s42.pid)\" -o pid,pcpu,etime 2>/dev/null || echo 'phase_dyn finished'
fi
tail -n 3 /home/ember/restrans-exp/logs/webscale_std_alibi_20M_s42.log 2>/dev/null
tail -n 3 /home/ember/restrans-exp/logs/webscale_phase_dyn_20M_s42.log 2>/dev/null
df -h / | tail -n 1
" >> "$LOGFILE" 2>&1

echo "" >> "$LOGFILE"
echo "Next check: $(date -d '+3 hours' -Iseconds 2>/dev/null || date -v+3H -Iseconds)" >> "$LOGFILE"
echo "" >> "$LOGFILE"
