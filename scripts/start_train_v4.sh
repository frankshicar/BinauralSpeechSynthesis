#!/bin/bash

# 訓練腳本 v4 - 完整優化策略
# blocks=3 + ReduceLROnPlateau + 分階段訓練

echo "=========================================="
echo "開始訓練 v4 (完整優化策略)"
echo "=========================================="
echo "配置:"
echo "  - WaveNet blocks: 3 (大模型)"
echo "  - 學習率調度: ReduceLROnPlateau (patience=10)"
echo "  - 訓練策略: 分階段 (0-30, 30-70, 70-100)"
echo "  - Batch size: 16"
echo "  - GPU: 1"
echo "=========================================="
echo ""

# 檢查 dataset 路徑
if [ ! -d "./dataset_original/trainset" ]; then
    echo "錯誤: 找不到 ./dataset_original/trainset"
    echo "請確認資料集路徑正確"
    exit 1
fi

# 執行訓練
nohup python train_v4.py \
    --dataset_directory ./dataset_original/trainset \
    --artifacts_directory ./outputs_v4 \
    --num_gpus 1 \
    --blocks 3 \
    > train_v4.log 2>&1 &

PID=$!
echo "訓練已在背景啟動 (PID: $PID)"
echo "查看訓練日誌: tail -f train_v4.log"
echo "查看訓練進度: python monitor_training.py outputs_v4"
echo ""
echo "停止訓練: kill $PID"
