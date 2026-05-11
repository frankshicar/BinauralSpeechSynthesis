#!/bin/bash

cd /home/sbplab/frank/BinauralSpeechSynthesis

nohup python3 -u train_hybrid_ipd.py > train_hybrid_ipd.log 2>&1 &

echo "Training started! PID: $!"
echo "Monitor with: tail -f train_hybrid_ipd.log"
