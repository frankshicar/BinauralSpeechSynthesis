#!/bin/bash

# Magnitude-only Baseline

cd /home/sbplab/frank/BinauralSpeechSynthesis

nohup python3 -u training/train_magnitude_only.py > logs/train_magnitude_only.log 2>&1 &

echo "Magnitude-only baseline started! PID: $!"
echo "Monitor with: tail -f logs/train_magnitude_only.log"
echo ""
echo "This is the upper bound:"
echo "  - Magnitude: learned"
echo "  - Phase: mono + physical ITD"
echo "  - Epochs: 50 (~17 hours)"
