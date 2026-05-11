#!/bin/bash

cd /home/sbplab/frank/BinauralSpeechSynthesis

nohup python3 training/train_improved_residual.py > logs/train_improved_residual.log 2>&1 &

echo "Training started! PID: $!"
echo "Monitor with: tail -f logs/train_improved_residual.log"
