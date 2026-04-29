#!/bin/bash

# 啟動 WaveformSpatializer 訓練

nohup python3 -u train_waveform.py > train_waveform.log 2>&1 &

echo "Training started! PID: $!"
echo "Monitor with: tail -f train_waveform.log"
