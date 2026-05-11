#!/bin/bash

# 啟動 GeoWarpFiLMNet v4 訓練

cd /home/sbplab/frank/BinauralSpeechSynthesis

# 確保輸出資料夾存在
mkdir -p geowarp_film_v4

nohup python3 -u training/train_geowarp_film_v4.py > geowarp_film_v4/train_console.log 2>&1 &

echo "Training started! PID: $!"
echo "Monitor with: tail -f geowarp_film_v4/train.log"
echo "Console log: tail -f geowarp_film_v4/train_console.log"
