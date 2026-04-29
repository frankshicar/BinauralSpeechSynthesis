#!/bin/bash

# v6 訓練啟動腳本
# 主要改進：L2 全程高權重 + Stage 切換 LR 重置 + 200 epochs

echo "=========================================="
echo "開始訓練 v6 (200 epochs)"
echo "=========================================="
echo "配置:"
echo "  - WaveNet blocks: 3 (大模型)"
echo "  - 總 epochs: 200"
echo "  - Stage 1 (0-80):    L2=10.0, Phase=0.0, IPD=0.0"
echo "  - Stage 2 (80-140):  L2=10.0, Phase=0.1, IPD=0.2"
echo "  - Stage 3 (140-200): L2=10.0, Phase=0.15, IPD=0.3"
echo "  - LR patience: 20"
echo "  - Stage 切換時 LR 重置為 0.0003"
echo "  - 預估訓練時間: ~22 小時"
echo "=========================================="
echo ""

if [ ! -d "./dataset_original/trainset" ]; then
    echo "錯誤: 找不到 ./dataset_original/trainset"
    exit 1
fi

nohup python train_v6.py \
    --dataset_directory ./dataset_original/trainset \
    --artifacts_directory ./outputs_v6 \
    --num_gpus 1 \
    --blocks 3 \
    > train_v6.log 2>&1 &

PID=$!
echo "訓練已在背景啟動 (PID: $PID)"
echo ""
echo "監控指令:"
echo "  tail -f train_v6.log"
echo "  tail outputs_v6/training_logs/training_history.csv"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "停止訓練: kill $PID"
