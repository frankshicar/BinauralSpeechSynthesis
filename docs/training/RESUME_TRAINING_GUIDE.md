# 接續訓練使用指南

## 問題說明

之前的實現只保存模型權重，不保存 optimizer 狀態，導致：
- ❌ Learning rate 重置回初始值 (0.001)
- ❌ NewbobAdam 的歷史記錄丟失
- ❌ 訓練進度無法正確恢復

## 新的解決方案

現在系統會保存**完整的 checkpoint**，包括：
- ✅ 模型權重
- ✅ Optimizer 狀態（包括當前 learning rate）
- ✅ NewbobAdam 的 `last_epoch_loss` 和 `total_decay`
- ✅ 訓練歷史記錄
- ✅ Epoch 計數

## 使用方法

### 1. 開始新訓練

```bash
python train.py \
    --dataset_directory ./dataset/trainset \
    --artifacts_directory ./outputs_new \
    --num_gpus 1 \
    --blocks 3
```

這會在每 10 個 epoch 保存兩種文件：
- `binaural_network_checkpoint.epoch-10.pth` - **完整 checkpoint**（新格式）
- `binaural_network.epoch-10.net` - 模型權重（舊格式，向後兼容）

### 2. 從 checkpoint 恢復訓練

假設你訓練到 epoch 40 後暫停了：

```bash
python train.py \
    --dataset_directory ./dataset/trainset \
    --artifacts_directory ./outputs_new \
    --num_gpus 1 \
    --blocks 3 \
    --resume epoch-40
```

系統會：
1. 載入 `binaural_network_checkpoint.epoch-40.pth`
2. 恢復模型權重
3. **恢復 optimizer 狀態**（包括 learning rate）
4. **恢復 NewbobAdam 狀態**
5. 載入訓練歷史
6. 從 epoch 41 繼續訓練

### 3. 查看恢復的狀態

當你使用 `--resume` 時，會看到類似輸出：

```
============================================================
從 checkpoint 恢復訓練: epoch-40
============================================================
✅ Checkpoint 已載入: ./outputs_new/binaural_network_checkpoint.epoch-40.pth
✅ Optimizer 狀態已恢復
✅ NewbobAdam 狀態已恢復:
   - last_epoch_loss: 0.094984
   - total_decay: 0.125
   - current_lr: 0.000125
✅ 將從 epoch 41 繼續訓練
✅ 已載入訓練歷史: 40 個 epoch

上次訓練狀態:
  - Epoch: 40
  - Learning Rate: 0.000125
  - Accumulated Loss: 0.094984
============================================================
```

## 文件說明

### Checkpoint 文件 (.pth)

完整的訓練狀態，包含：
```python
{
    'model_state_dict': ...,      # 模型權重
    'optimizer_state_dict': ...,  # Optimizer 狀態
    'epoch': 40,                   # Epoch 計數
    'newbob_state': {
        'last_epoch_loss': 0.094984,
        'total_decay': 0.125,
    },
    'training_state': {
        'training_history': [...],
        'total_iters': 12800,
    }
}
```

### 模型文件 (.net)

只包含模型權重（向後兼容舊代碼）

### 訓練記錄

- `training_logs/training_history.json` - 每個 epoch 的詳細記錄
- `training_logs/training_history.csv` - CSV 格式
- `training_logs/training_summary.json` - 訓練總結

## 實際使用範例

### 場景 1：訓練被中斷

```bash
# 開始訓練
python train.py --artifacts_directory ./outputs --epochs 100

# ... 訓練到 epoch 30 時被中斷 ...

# 從 epoch 30 繼續
python train.py --artifacts_directory ./outputs --epochs 100 --resume epoch-30
```

### 場景 2：分段訓練

```bash
# 第一階段：訓練 50 個 epoch
python train.py --artifacts_directory ./outputs --epochs 50

# 第二階段：從 epoch 50 繼續訓練到 100
python train.py --artifacts_directory ./outputs --epochs 100 --resume epoch-50
```

### 場景 3：Learning Rate 已經衰減

如果你的訓練已經進行了多次 NewbobAdam 衰減：

```bash
# 查看當前 learning rate
python -c "
import json
with open('./outputs/training_logs/training_history.json') as f:
    history = json.load(f)
    print(f'Last LR: {history[-1][\"learning_rate\"]}')
"

# 從最後的 checkpoint 恢復（會保持衰減後的 learning rate）
python train.py --artifacts_directory ./outputs --resume epoch-40
```

## 向後兼容性

### 從舊的 .net 文件恢復

如果你有舊的 `.net` 文件但沒有 `.pth` checkpoint：

```bash
# 這會載入模型權重，但 learning rate 會重置
python train.py --resume epoch-40
```

⚠️ **注意**：這種情況下 learning rate 會重置回初始值，建議手動調整 `config["learning_rate"]`。

### 手動設置 Learning Rate

如果需要手動設置 learning rate，修改 `train.py` 中的 config：

```python
config = {
    "learning_rate": 0.000125,  # 手動設置為上次的 learning rate
    # ... 其他配置 ...
}
```

## 監控訓練

使用現有的監控工具：

```bash
# 實時監控
python monitor_training.py

# 可視化
python visualize_training.py
```

## 常見問題

### Q1: 如何知道上次訓練到哪個 epoch？

查看 `training_logs/training_history.json` 的最後一條記錄。

### Q2: 如何知道當前的 learning rate？

查看 `training_logs/training_history.json` 中的 `learning_rate` 欄位。

### Q3: 可以從任意 epoch 恢復嗎？

可以，只要該 epoch 有保存 checkpoint（預設每 10 個 epoch 保存一次）。

### Q4: 舊的 .net 文件還能用嗎？

可以用於推理（synthesize），但用於恢復訓練時會丟失 optimizer 狀態。

### Q5: 如何更改保存頻率？

修改 `train.py` 中的 `config["save_frequency"]`：

```python
config = {
    "save_frequency": 5,  # 每 5 個 epoch 保存一次
    # ...
}
```

## 最佳實踐

1. **定期保存**：建議 `save_frequency` 設為 5-10
2. **保留多個 checkpoint**：不要只保留最新的
3. **監控 learning rate**：確保 NewbobAdam 正常工作
4. **備份重要 checkpoint**：特別是 loss 最低的那些

## 技術細節

### NewbobAdam 狀態恢復

NewbobAdam 需要恢復兩個關鍵狀態：
- `last_epoch_loss`: 上一個 epoch 的 loss（用於判斷是否需要衰減）
- `total_decay`: 累積的衰減倍數（用於判斷是否達到 max_decay）

如果不恢復這些狀態，NewbobAdam 會誤判並可能：
- 過早衰減 learning rate
- 或者不衰減（因為 loss 比較基準錯誤）

### Checkpoint 文件大小

完整 checkpoint 比單純的模型權重稍大：
- `.net` 文件：~50-100 MB（僅模型權重）
- `.pth` checkpoint：~60-120 MB（包含 optimizer 狀態）

## 總結

使用新的 checkpoint 系統，你可以：
- ✅ 隨時暫停和恢復訓練
- ✅ Learning rate 正確保持
- ✅ NewbobAdam 狀態完整恢復
- ✅ 訓練歷史連續記錄
- ✅ 向後兼容舊的 .net 文件
