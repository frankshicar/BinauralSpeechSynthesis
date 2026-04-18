# 開始訓練 - 快速指南

## ✅ 系統檢查完成

所有檢查都通過了：
- ✅ 數據集: `./dataset/trainset` (7 個 subjects)
- ✅ CUDA: NVIDIA GeForce RTX 5090
- ✅ Geometric Warp ITD 修改已應用
- ✅ Python 環境正常

## 🚀 開始訓練

### 方法 1: 使用腳本（推薦）

```bash
./start_new_training.sh
```

### 方法 2: 直接命令

```bash
python train.py \
    --dataset_directory ./dataset/trainset \
    --artifacts_directory ./outputs_with_new_geom_warp \
    --num_gpus 1 \
    --blocks 3
```

## 📊 訓練配置

當前配置（在 `train.py` 中）：
- **Learning Rate**: 0.001 (初始)
- **Batch Size**: 32
- **Epochs**: 100
- **Save Frequency**: 每 10 個 epoch
- **Loss Weights**:
  - L2: 1.0
  - Phase: 0.01
  - IPD: 0.1

## 📁 輸出文件

訓練會在 `./outputs_with_new_geom_warp/` 創建：

### 每 10 個 epoch 保存：
- `binaural_network_checkpoint.epoch-10.pth` - 完整 checkpoint
- `binaural_network.epoch-10.net` - 模型權重

### 訓練記錄（實時更新）：
- `training_logs/training_history.json` - JSON 格式
- `training_logs/training_history.csv` - CSV 格式
- `training_logs/training_config.json` - 訓練配置

## 🔄 如果訓練中斷

假設訓練到 epoch 40 時中斷：

```bash
python train.py \
    --dataset_directory ./dataset/trainset \
    --artifacts_directory ./outputs_with_new_geom_warp \
    --num_gpus 1 \
    --blocks 3 \
    --resume epoch-40
```

系統會自動：
- ✅ 恢復模型權重
- ✅ 恢復 learning rate
- ✅ 恢復 NewbobAdam 狀態
- ✅ 從 epoch 41 繼續

## 📈 監控訓練

### 在另一個終端運行：

```bash
# 實時監控
python monitor_training.py

# 或者可視化
python visualize_training.py
```

### 檢查 checkpoint 狀態：

```bash
# 列出所有 checkpoint
python check_checkpoint.py --artifacts_dir ./outputs_with_new_geom_warp --list

# 查看訓練歷史
python check_checkpoint.py --artifacts_dir ./outputs_with_new_geom_warp --history

# 檢查特定 checkpoint
python check_checkpoint.py --artifacts_dir ./outputs_with_new_geom_warp --checkpoint epoch-40
```

## ⏱️ 預估時間

根據你之前的訓練記錄：
- 每個 epoch: ~13 分鐘 (800 秒)
- 100 個 epoch: ~22 小時

建議：
- 可以先訓練 10-20 個 epoch 測試
- 然後暫停檢查結果
- 再繼續訓練

## 🎯 新功能優勢

### 1. 改進的 Geometric Warp
- 分別計算左右耳距離
- ITD 由物理模型提供
- 預期 L_warp loss 更低

### 2. 完整的 Checkpoint 系統
- 可以隨時暫停/恢復
- Learning rate 正確保持
- 訓練歷史連續

## 📝 訓練建議

### 第一階段：測試（10 epochs）
```bash
# 修改 train.py 中的 epochs
config = {
    "epochs": 10,  # 先測試 10 個 epoch
    # ...
}

python train.py --dataset_directory ./dataset/trainset --artifacts_directory ./outputs_test --num_gpus 1 --blocks 3
```

### 第二階段：完整訓練（100 epochs）
確認測試正常後，開始完整訓練：
```bash
python train.py --dataset_directory ./dataset/trainset --artifacts_directory ./outputs_with_new_geom_warp --num_gpus 1 --blocks 3
```

### 第三階段：分段訓練
如果需要分段：
```bash
# 第一段：0-50
python train.py --artifacts_directory ./outputs_with_new_geom_warp --num_gpus 1 --blocks 3
# ... 訓練到 epoch 50 ...

# 第二段：50-100
python train.py --artifacts_directory ./outputs_with_new_geom_warp --num_gpus 1 --blocks 3 --resume epoch-50
```

## 🔍 關鍵指標

訓練時注意觀察：

### Loss 值
- `accumulated_loss`: 總損失（應該下降）
- `l2`: L2 損失
- `phase`: 相位損失
- `ipd`: IPD 損失

### Learning Rate
- 初始: 0.001
- NewbobAdam 會自動調整
- 如果 loss 上升，會衰減 0.5 倍

### 預期結果
根據你之前的訓練（epoch 39）：
- Accumulated Loss: ~0.095
- Learning Rate: ~6.25e-05 (已衰減多次)

新的 geometric warp 應該能：
- 更快收斂
- 更低的最終 loss
- 更好的 ITD 預測

## ⚠️ 注意事項

1. **磁碟空間**
   - 每個 checkpoint ~60-120 MB
   - 100 epochs = ~1-2 GB

2. **GPU 記憶體**
   - RTX 5090 有足夠記憶體
   - 如果 OOM，減少 batch_size

3. **訓練時間**
   - 可以隨時 Ctrl+C 中斷
   - 使用 --resume 恢復

4. **備份**
   - 定期備份重要的 checkpoint
   - 特別是 loss 最低的

## 🎉 準備好了！

現在可以開始訓練了：

```bash
./start_new_training.sh
```

或者

```bash
python train.py \
    --dataset_directory ./dataset/trainset \
    --artifacts_directory ./outputs_with_new_geom_warp \
    --num_gpus 1 \
    --blocks 3
```

祝訓練順利！🚀
