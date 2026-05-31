#!/bin/bash
# Auto-launcher for sequential webscale training on nextop
# Launches phase_dynamic when standard_alibi finishes

cd /Users/ember/dev/restrans
PYTHON=/Users/ember/dev/restrans/.venv/bin/python3
LOGDIR=logs/webscale
OUTDIR=outputs/webscale

# Wait for standard_alibi to finish
PIDFILE=$LOGDIR/std_alibi_20M_s42.pid
if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE")
    echo "Waiting for standard_alibi (PID=$PID) to finish..."
    while kill -0 "$PID" 2>/dev/null; do
        sleep 60
    done
    echo "standard_alibi finished at $(date)"
fi

# Launch phase_dynamic
nohup $PYTHON -u resonance/train_webscale.py \
  --condition phase_dynamic_qk_film \
  --scale 20M \
  --max_tokens 100000000 \
  --seq_len 512 \
  --batch_size 16 \
  --lr 3e-4 \
  --grad_clip 1.0 \
  --eval_every 2000 \
  --log_every 200 \
  --seed 42 \
  --device mps \
  --output_dir $OUTDIR \
  --save_name phase_dyn_20M_s42.pt \
  > $LOGDIR/phase_dyn_20M_s42.log 2>&1 &

NEWPID=$!
echo $NEWPID > $LOGDIR/phase_dyn_20M_s42.pid
echo "Launched phase_dynamic (PID=$NEWPID) at $(date)"
