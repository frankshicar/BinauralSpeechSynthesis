#!/bin/bash

# 啟動 GeoWarpFiLMNet 訓練

cd /home/sbplab/frank/BinauralSpeechSynthesis

# 確保輸出資料夾存在
mkdir -p geowarp_film

nohup python3 -u training/train_geowarp_film.py > geowarp_film/train_console.log 2>&1 &

echo "Training started! PID: $!"
echo "Monitor with: tail -f geowarp_film/train.log"
echo "Console log: tail -f geowarp_film/train_console.log"
