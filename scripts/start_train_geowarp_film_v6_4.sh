#!/bin/bash
set -euo pipefail

cd /home/sbplab/frank/BinauralSpeechSynthesis
mkdir -p geowarp_film_v6_4
/home/sbplab/anaconda3/bin/python training/train_geowarp_film_v6_4.py 2>&1 | tee geowarp_film_v6_4/train_console.log
