#!/bin/bash
cd /home/sbplab/frank/BinauralSpeechSynthesis
mkdir -p geowarp_film_v5b

nohup /home/sbplab/anaconda3/bin/python3 training/train_geowarp_film_v5.py \
    > geowarp_film_v5b/train_console.log 2>&1 &

echo "Training started, PID: $!"
echo "Monitor: tail -f geowarp_film_v5b/train.log"
