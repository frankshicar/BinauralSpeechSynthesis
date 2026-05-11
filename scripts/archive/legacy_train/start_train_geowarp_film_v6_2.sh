#!/bin/bash
cd /home/sbplab/frank/BinauralSpeechSynthesis
mkdir -p geowarp_film_v6_2
/home/sbplab/anaconda3/bin/python training/train_geowarp_film_v6_2.py 2>&1 | tee geowarp_film_v6_2/train_console.log
