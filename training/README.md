# Training Scripts

各版本訓練腳本說明。模型輸出在 `models/v{N}/`，訓練記錄在 `實驗記錄/`。

## 版本對照

| 腳本 | Trainer | 最終模型 | 主要特點 |
|------|---------|---------|---------|
| `train_v1_original.py` | `src/trainer.py` | — | 原始 Facebook 版本，blocks=1 或 3 |
| `train_v4.py` | `src/trainer_v4.py` | `models/v4/` (epoch-100) | 分階段訓練、ReduceLROnPlateau、動態 loss 權重 |
| `train_v5_150ep.py` | `src/trainer_v4.py` | `models/v5/` (epoch-150) | 延長至 150 epochs，重新平衡 loss 權重 |
| `train_v6.py` | `src/trainer_v4.py` | `models/v6/` (epoch-200) | L2 權重全程高位(10.0)，Stage 1 延長至 80ep，LR patience 加大 |
| `train_v7.py` | `src/trainer_v7.py` | `models/v7/` (epoch-140) | CosineAnnealingWarmRestarts，Stage 1 縮短至 60ep，Phase/IPD 權重大幅提高 |

## 版本演進重點

**v4 → v5**：訓練時間從 100 延長到 150 epochs，調整 loss 權重比例。

**v5 → v6**：L2 loss 全程維持高權重(10.0)確保波形重建基礎；Stage 1 從 50 延長到 80 epochs；LR patience 從 10 加大到 20。

**v6 → v7**：換用 CosineAnnealingWarmRestarts 取代 ReduceLROnPlateau，解決 Stage 切換時 loss 跳升導致 LR 過早衰減的問題；Phase/IPD 梯度訊號大幅增強。

## 使用方式

```bash
# 啟動 v7 訓練（推薦）
bash scripts/start_train_v7.sh

# 手動執行
python training/train_v7.py \
  --dataset_directory ./dataset/trainset \
  --artifacts_directory ./models/v7 \
  --num_gpus 1
```
