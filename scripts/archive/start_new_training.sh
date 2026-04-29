#!/bin/bash

# ============================================================================
# 使用新的 Geometric Warp ITD 和 Checkpoint 系統開始訓練
# ============================================================================

echo "=========================================="
echo "開始新的訓練"
echo "=========================================="
echo ""

# 配置參數
DATASET_DIR="./dataset/trainset"
OUTPUT_DIR="./outputs_with_new_geom_warp"
NUM_GPUS=1
BLOCKS=3
EPOCHS=100

# 檢查數據集是否存在
if [ ! -d "$DATASET_DIR" ]; then
    echo "❌ 錯誤: 數據集目錄不存在: $DATASET_DIR"
    exit 1
fi

echo "✅ 數據集目錄: $DATASET_DIR"
echo "✅ 輸出目錄: $OUTPUT_DIR"
echo "✅ GPU 數量: $NUM_GPUS"
echo "✅ WaveNet Blocks: $BLOCKS"
echo "✅ 總 Epochs: $EPOCHS"
echo ""

# 創建輸出目錄
mkdir -p "$OUTPUT_DIR"

# 開始訓練
echo "開始訓練..."
echo ""

python train.py \
    --dataset_directory "$DATASET_DIR" \
    --artifacts_directory "$OUTPUT_DIR" \
    --num_gpus $NUM_GPUS \
    --blocks $BLOCKS

echo ""
echo "=========================================="
echo "訓練完成或中斷"
echo "=========================================="
echo ""
echo "如需繼續訓練，請使用："
echo "python train.py --dataset_directory $DATASET_DIR --artifacts_directory $OUTPUT_DIR --num_gpus $NUM_GPUS --blocks $BLOCKS --resume epoch-XX"
echo ""
