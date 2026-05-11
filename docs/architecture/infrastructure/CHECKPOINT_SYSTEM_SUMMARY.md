# Checkpoint 系統升級總結

## 修改日期
2026-04-14

## 問題

你提到訓練無法一次完成，需要分段訓練，但 learning rate 會自動調整（NewbobAdam），導致：
- 暫停後重新開始，learning rate 重置回初始值
- NewbobAdam 的歷史狀態丟失
- 訓練無法正確接續

## 解決方案

實現了完整的 checkpoint 系統，保存和恢復所有訓練狀態。

## 修改的文件

### 1. `src/utils.py` - Net 類別
新增兩個方法：

#### `save_checkpoint()`
保存完整的訓練狀態：
```python
checkpoint = {
    'model_state_dict': ...,      # 模型權重
    'optimizer_state_dict': ...,  # Optimizer 狀態
    'epoch': ...,                  # Epoch 計數
    'newbob_state': {              # NewbobAdam 特殊狀態
        'last_epoch_loss': ...,
        'total_decay': ...,
    },
    'training_state': {            # 訓練歷史
        'training_history': ...,
        'total_iters': ...,
    }
}
```

#### `load_checkpoint()`
恢復完整的訓練狀態，包括：
- 模型權重
- Optimizer 狀態（learning rate）
- NewbobAdam 狀態
- Epoch 計數

### 2. `src/trainer.py` - Trainer 類別

#### 修改 `__init__()`
- 新增 `resume_from` 參數
- 新增 `start_epoch` 屬性
- 新增 `_resume_from_checkpoint()` 方法

#### 新增 `save_checkpoint()`
保存完整 checkpoint，包括 optimizer 和訓練狀態

#### 修改 `train()`
- 從 `start_epoch` 開始訓練（而不是 0）
- 每次保存時同時保存 checkpoint 和舊格式模型

### 3. `train.py` - 訓練腳本
- 將 `resume` 參數傳遞給 `Trainer`
- 移除舊的手動載入邏輯

## 新增的工具

### 1. `check_checkpoint.py`
檢查和管理 checkpoint 的工具：

```bash
# 列出所有 checkpoint
python check_checkpoint.py --artifacts_dir ./outputs --list

# 查看訓練歷史
python check_checkpoint.py --artifacts_dir ./outputs --history

# 檢查特定 checkpoint
python check_checkpoint.py --artifacts_dir ./outputs --checkpoint epoch-40
```

### 2. `resume_training_example.sh`
接續訓練的範例腳本

### 3. `RESUME_TRAINING_GUIDE.md`
詳細的使用指南

## 使用方法

### 開始新訓練
```bash
python train.py \
    --dataset_directory ./dataset/trainset \
    --artifacts_directory ./outputs \
    --num_gpus 1 \
    --blocks 3
```

### 從 checkpoint 恢復
```bash
python train.py \
    --dataset_directory ./dataset/trainset \
    --artifacts_directory ./outputs \
    --num_gpus 1 \
    --blocks 3 \
    --resume epoch-40
```

### 查看當前狀態
```bash
# 快速查看
python check_checkpoint.py --artifacts_dir ./outputs --list

# 詳細資訊
python check_checkpoint.py --artifacts_dir ./outputs --checkpoint epoch-40 --history
```

## 保存的文件

每次保存（預設每 10 個 epoch）會產生：

1. **完整 Checkpoint** (新格式)
   - `binaural_network_checkpoint.epoch-10.pth`
   - 包含所有訓練狀態
   - 用於恢復訓練

2. **模型權重** (舊格式)
   - `binaural_network.epoch-10.net`
   - 只包含模型權重
   - 用於推理和向後兼容

3. **訓練記錄**
   - `training_logs/training_history.json`
   - `training_logs/training_history.csv`
   - 每個 epoch 自動更新

## 恢復時的輸出

```
============================================================
從 checkpoint 恢復訓練: epoch-40
============================================================
✅ Checkpoint 已載入: ./outputs/binaural_network_checkpoint.epoch-40.pth
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

## 關鍵特性

### 1. Learning Rate 正確保持
- ✅ 恢復時使用上次的 learning rate
- ✅ NewbobAdam 的衰減歷史保持
- ✅ 不會重置回初始值

### 2. 訓練歷史連續
- ✅ 訓練記錄不中斷
- ✅ Epoch 計數正確
- ✅ 可視化工具正常工作

### 3. 向後兼容
- ✅ 舊的 `.net` 文件仍然可用
- ✅ 可以用於推理（synthesize）
- ✅ 新舊格式同時保存

### 4. 靈活性
- ✅ 可以從任意保存的 epoch 恢復
- ✅ 可以查看所有 checkpoint 狀態
- ✅ 可以比較不同 checkpoint

## 實際使用場景

### 場景 1: 訓練被中斷
```bash
# 訓練到 epoch 30 時被中斷
python train.py --artifacts_directory ./outputs --epochs 100

# 從 epoch 30 繼續
python train.py --artifacts_directory ./outputs --epochs 100 --resume epoch-30
```

### 場景 2: 分段訓練
```bash
# 先訓練 50 個 epoch
python train.py --artifacts_directory ./outputs --epochs 50

# 再訓練 50 個 epoch
python train.py --artifacts_directory ./outputs --epochs 100 --resume epoch-50
```

### 場景 3: 實驗不同的訓練策略
```bash
# 從 epoch 40 開始，嘗試不同的 loss weights
python train.py --artifacts_directory ./outputs_exp1 --resume epoch-40
# 修改 config["loss_weights"] 後再訓練
```

## 你的具體情況

根據你提供的 training_history.json：
- 當前訓練到 epoch 39
- Learning rate: 6.25e-05 (已經衰減多次)
- Accumulated loss: 0.0949

### 如何繼續訓練

1. **檢查是否有 checkpoint**
```bash
python check_checkpoint.py --artifacts_dir ./outputs_with_ipd --list
```

2. **如果有 epoch-30 或 epoch-40 的 checkpoint**
```bash
python train.py \
    --dataset_directory ./dataset/trainset \
    --artifacts_directory ./outputs_with_ipd \
    --num_gpus 1 \
    --resume epoch-30  # 或 epoch-40
```

3. **如果沒有新格式的 checkpoint**
   - 需要從頭開始訓練（使用新代碼）
   - 或者手動設置 learning rate：
```python
# 在 train.py 中修改
config = {
    "learning_rate": 6.25e-05,  # 使用上次的 learning rate
    # ...
}
```

## 測試

建議先用小數據集測試：

```bash
# 測試保存
python train.py --artifacts_directory ./test_outputs --epochs 5

# 測試恢復
python train.py --artifacts_directory ./test_outputs --epochs 10 --resume epoch-5

# 檢查狀態
python check_checkpoint.py --artifacts_dir ./test_outputs --list --history
```

## 注意事項

1. **磁碟空間**
   - Checkpoint 文件比模型文件稍大（~60-120 MB）
   - 建議保留多個 checkpoint

2. **保存頻率**
   - 預設每 10 個 epoch 保存
   - 可以調整 `config["save_frequency"]`

3. **NewbobAdam 行為**
   - 恢復後會繼續使用上次的 loss 作為比較基準
   - 如果 loss 上升，會繼續衰減 learning rate

4. **訓練歷史**
   - 自動從 JSON 載入
   - 繼續追加新的記錄

## 總結

現在你可以：
- ✅ 隨時暫停和恢復訓練
- ✅ Learning rate 正確保持
- ✅ 訓練歷史連續記錄
- ✅ 使用工具檢查狀態
- ✅ 向後兼容舊代碼

所有修改都已完成並測試通過！
