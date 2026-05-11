#!/bin/bash

# 分階段訓練 HybridPhysicalLearned

cd /home/sbplab/frank/BinauralSpeechSynthesis

nohup python3 -u training/train_staged.py > logs/train_staged.log 2>&1 &

echo "Staged training started! PID: $!"
echo "Monitor with: tail -f logs/train_staged.log"
echo ""
echo "Training plan:"
echo "  Stage 1: 20 epochs (Magnitude only)"
echo "  Stage 2: 30 epochs (+ IPD)"
echo "  Stage 3: 50 epochs (L2 focused)"
echo "  Total: 100 epochs (~33 hours)"
