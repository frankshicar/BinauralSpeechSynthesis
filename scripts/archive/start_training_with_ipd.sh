#!/bin/bash

# 使用 IPD Loss 開始訓練的快速腳本
# Quick script to start training with IPD Loss

echo "=========================================="
echo "開始使用 IPD Loss 訓練模型"
echo "Starting training with IPD Loss"
echo "=========================================="
echo ""

# 檢查訓練資料是否存在
if [ ! -d "./data/trainset" ]; then
    echo "❌ 錯誤：找不到訓練資料 ./data/trainset"
    echo "❌ Error: Training data not found at ./data/trainset"
    exit 1
fi

# 創建輸出目錄
OUTPUT_DIR="./outputs_with_ipd_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
echo "✅ 輸出目錄：$OUTPUT_DIR"
echo ""

# 顯示當前配置
echo "訓練配置 (Training Configuration):"
echo "  - 資料集 (Dataset): ./data/trainset"
echo "  - 輸出目錄 (Output): $OUTPUT_DIR"
echo "  - GPU 數量 (GPUs): 1"
echo "  - WaveNet Blocks: 3"
echo "  - Loss 權重 (Loss Weights):"
echo "    * L2: 1.0"
echo "    * Phase: 0.01"
echo "    * IPD: 0.1"
echo ""

# 詢問是否繼續
read -p "是否開始訓練？(y/n) Start training? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消 (Cancelled)"
    exit 0
fi

echo ""
echo "=========================================="
echo "開始訓練... (Starting training...)"
echo "=========================================="
echo ""

# 開始訓練
python train.py \
    --dataset_directory ./data/trainset \
    --artifacts_directory "$OUTPUT_DIR" \
    --num_gpus 1 \
    --blocks 3

# 檢查訓練是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 訓練完成！(Training completed!)"
    echo "=========================================="
    echo ""
    echo "模型檔案位置 (Model files location):"
    echo "  $OUTPUT_DIR"
    echo ""
    echo "下一步 (Next steps):"
    echo "  1. 檢查 ITD 是否在物理範圍內："
    echo "     python check_90deg_itd.py"
    echo ""
    echo "  2. 評估各角度的誤差："
    echo "     python evaluate_angles.py"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ 訓練失敗 (Training failed)"
    echo "=========================================="
    echo ""
    echo "請檢查錯誤訊息並參考 IPD_Loss_說明.md"
    echo "Please check error messages and refer to IPD_Loss_說明.md"
    exit 1
fi
