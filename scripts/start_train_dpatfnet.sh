#!/bin/bash

echo "=========================================="
echo "Starting DPATFNet Training"
echo "=========================================="
echo ""

# Run training in background
nohup python3 train_dpatfnet.py > train_dpatfnet.log 2>&1 &

PID=$!
echo "Training started with PID: $PID"
echo "Log file: train_dpatfnet.log"
echo ""
echo "Monitor training:"
echo "  tail -f train_dpatfnet.log"
echo ""
echo "Stop training:"
echo "  pkill -f train_dpatfnet.py"
echo ""
