#!/bin/bash

# 啟動 HybridPhysicalLearned 訓練

cd /home/sbplab/frank/BinauralSpeechSynthesis

nohup python3 -u training/train_hybrid_physical.py > logs/train_hybrid_physical.log 2>&1 &

echo "Training started! PID: $!"
echo "Monitor with: tail -f logs/train_hybrid_physical.log"
