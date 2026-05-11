#!/bin/bash

echo "=========================================="
echo "開始訓練 v7 (180 epochs)"
echo "=========================================="
echo "  - Stage 1 (0-60):   L2=10.0, Phase=0.0, IPD=0.0"
echo "  - Stage 2 (60-120): L2=10.0, Phase=0.3, IPD=0.5"
echo "  - Stage 3 (120-180):L2=10.0, Phase=0.5, IPD=0.8"
echo "  - LR: CosineAnnealingWarmRestarts (T_0=60)"
echo "  - 預估訓練時間: ~40 小時"
echo "=========================================="

if [ ! -d "./dataset/train" ]; then
    echo "錯誤: 找不到 ./dataset/train"
    exit 1
fi

RESUME=${1:-""}  # 第一個參數為 resume suffix，例如 epoch-70

if [ -n "$RESUME" ]; then
    echo "從 checkpoint 繼續: $RESUME"
    RESUME_ARG="--resume $RESUME"
else
    RESUME_ARG=""
fi

nohup python train_v7.py \
    --dataset_directory ./dataset/train \
    --artifacts_directory ./outputs_v7 \
    --num_gpus 1 \
    --blocks 3 \
    $RESUME_ARG \
    >> train_v7.log 2>&1 &

PID=$!
echo "訓練已在背景啟動 (PID: $PID)"
echo ""
echo "監控指令:"
echo "  tail -f train_v7.log"
echo "  tail outputs_v7/training_logs/training_history.csv"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "停止訓練: kill $PID"
