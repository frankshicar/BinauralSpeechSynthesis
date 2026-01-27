#!/bin/bash
# 快速驗證修復效果的腳本 (Quick verification script)
# 2026-01-26 建立 (Created 2026-01-26)

echo "========================================="
echo "雙耳語音合成 - 噪音修復驗證"
echo "Binaural Speech Synthesis - Noise Fix Verification"
echo "========================================="
echo ""

# 設定路徑 (Set paths)
DATASET_DIR="./dataset/testset"
MODEL_FILE="./outputs/binaural_network.net"
OUTPUT_DIR="./outputs_verification"

# 檢查必要檔案 (Check required files)
echo "1. 檢查必要檔案... (Checking required files...)"
if [ ! -f "$MODEL_FILE" ]; then
    echo "   ❌ 錯誤: 找不到模型檔案 (Error: Model file not found)"
    echo "   路徑 (Path): $MODEL_FILE"
    exit 1
fi

if [ ! -d "$DATASET_DIR" ]; then
    echo "   ❌ 錯誤: 找不到測試資料集 (Error: Test dataset not found)"
    echo "   路徑 (Path): $DATASET_DIR"
    exit 1
fi

echo "   ✅ 模型檔案存在 (Model file exists)"
echo "   ✅ 測試資料集存在 (Test dataset exists)"
echo ""

# 建立輸出目錄 (Create output directory)  
echo "2. 建立輸出目錄... (Creating output directory...)"
mkdir -p "$OUTPUT_DIR"
echo "   ✅ 輸出目錄: $OUTPUT_DIR"
echo ""

# 運行評估 (Run evaluation)
echo "3. 開始評估... (Starting evaluation...)"
echo "   這可能需要幾分鐘 (This may take a few minutes)"
echo "========================================="
echo ""

python evaluate.py \
    --dataset_directory "$DATASET_DIR" \
    --model_file "$MODEL_FILE" \
    --artifacts_directory "$OUTPUT_DIR" \
    --blocks 3

EXIT_CODE=$?

echo ""
echo "========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 評估完成！(Evaluation completed!)"
    echo ""
    echo "輸出檔案位置 (Output files location):"
    echo "   $OUTPUT_DIR/"
    echo ""
    echo "下一步 (Next steps):"
    echo "   1. 使用音訊播放器聽輸出檔案，確認噪音是否消失"
    echo "      Listen to output files to confirm noise is gone"
    echo "   2. 檢查評估指標是否在合理範圍"
    echo "      Check if metrics are in reasonable range"
    echo "   3. (可選) 使用 Audacity 視覺化波形"
    echo "      (Optional) Visualize waveforms using Audacity"
else
    echo "❌ 評估失敗 (Evaluation failed)"
    echo "   退出代碼 (Exit code): $EXIT_CODE"
fi
echo "========================================="
