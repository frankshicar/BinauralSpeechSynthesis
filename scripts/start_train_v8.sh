#!/bin/bash

# BinauralTFNet v8.3 訓練啟動腳本
# v8.3 修改：從 y_common 出發，SimpleDPAB，調整 stage 劃分
# Stage 1 (0–60):    CommonBranch，L2（純 L2，ITD 只記錄）
# Stage 2 (60–160):  SpecificBranch（凍結 CommonBranch），Phase + IPD
# Stage 3 (160–180): 全部，L2×100 + Phase + IPD

DATASET="./dataset/train"
OUTPUT="./outputs_v8"
RESUME=""   # 若要從 checkpoint 繼續，改為例如 "epoch-80"

echo "=========================================="
echo "BinauralTFNet v8.3 訓練"
echo "=========================================="
echo "  Dataset:  $DATASET"
echo "  Output:   $OUTPUT"
echo "  Epochs:   180 (Stage1: 0-60, Stage2: 60-160, Stage3: 160-180)"
echo "  Loss:"
echo "    Stage 1: L2（純 L2，ITD 只記錄）"
echo "    Stage 2: Phase + IPD"
echo "    Stage 3: L2×100 + Phase + IPD"
echo ""
echo "  v8.3 修改："
echo "    - 從 y_common 出發（不是 warped）"
echo "    - SimpleDPAB（移除 Cross-attention）"
echo "    - Stage 1: 80 → 60 epochs"
echo "    - Stage 2: 80 → 100 epochs"
echo "  預估時間: ~24-30 小時"
echo "    Stage 3: L2×100 + Phase + IPD"
echo "  預估時間: ~22 小時 (依 GPU 而定)"
echo "=========================================="

if [ ! -d "$DATASET" ]; then
    echo "錯誤: 找不到 $DATASET"
    exit 1
fi

RESUME_ARG=""
if [ -n "$RESUME" ]; then
    RESUME_ARG="--resume $RESUME"
    echo "從 checkpoint 繼續: $RESUME"
fi

nohup python3 train_v8.py \
    --dataset_directory "$DATASET" \
    --artifacts_directory "$OUTPUT" \
    --num_gpus 1 \
    $RESUME_ARG \
    > train_v8.log 2>&1 &

PID=$!
echo ""
echo "訓練已在背景啟動 (PID: $PID)"
echo ""
echo "監控指令:"
echo "  tail -f train_v8.log"
echo "  tail -f $OUTPUT/training_logs/training_history.json | python3 -c \"import sys,json; [print(e) for e in json.load(sys.stdin)[-3:]]\""
echo "  watch -n 5 nvidia-smi"
echo ""
echo "停止訓練: kill $PID"
