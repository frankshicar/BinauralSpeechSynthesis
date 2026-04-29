#!/bin/bash

# v5 訓練啟動腳本 - 150 epochs 長訓練版本
# 目標：通用模型的單角度精確定位 (< 2° MAA)

echo "=========================================="
echo "開始訓練 v5 (150 epochs 長訓練版本)"
echo "=========================================="
echo "配置:"
echo "  - WaveNet blocks: 3 (大模型)"
echo "  - 總 epochs: 150 (比 v4 多 50%)"
echo "  - LR patience: 15 (避免過早衰減)"
echo "  - Phase 權重: 0.1 → 0.2 (提高 20 倍)"
echo "  - IPD 權重: 0.0 → 0.4 (大幅提高)"
echo "  - 預估訓練時間: 15-16 小時 (batch_size=32 優化後)"
echo "=========================================="
echo ""

# 檢查 dataset 路徑
if [ ! -d "./dataset_original/trainset" ]; then
    echo "錯誤: 找不到 ./dataset_original/trainset"
    echo "請確認資料集路徑正確"
    exit 1
fi

# 執行訓練
nohup python train_v5_150ep.py \
    --dataset_directory ./dataset_original/trainset \
    --artifacts_directory ./outputs_v5 \
    --num_gpus 1 \
    --blocks 3 \
    > train_v5.log 2>&1 &

PID=$!
echo "訓練已在背景啟動 (PID: $PID)"
echo ""
echo "監控指令:"
echo "  查看訓練日誌: tail -f train_v5.log"
echo "  查看訓練進度: tail outputs_v5/training_logs/training_history.csv"
echo "  監控 GPU: watch -n 1 nvidia-smi"
echo ""
echo "停止訓練: kill $PID"
echo ""
echo "預期效果 (150 epochs):"
echo "  - L2: 0.18-0.20 (v4 是 0.175)"
echo "  - Phase: 0.75-0.85 (v4 是 1.127，大幅改善)"
echo "  - IPD: 0.30-0.38 (v4 是 0.605，大幅改善)"
echo "  - 角度誤差: 2-4° (v4 推估 5-8°)"
