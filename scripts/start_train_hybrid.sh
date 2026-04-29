#!/bin/bash

echo "=========================================="
echo "Starting HybridTFNet Training (Corrected)"
echo "=========================================="
echo ""

# Run training in background
nohup python3 train_hybrid_corrected.py > train_hybrid_corrected.log 2>&1 &

PID=$!
echo "Training started with PID: $PID"
echo "Log file: train_hybrid_corrected.log"
echo ""
echo "Monitor training:"
echo "  tail -f train_hybrid_corrected.log"
echo ""
echo "Stop training:"
echo "  pkill -f train_hybrid_corrected.py"
echo ""
