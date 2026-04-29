# 訓練 v4 - 完整優化策略

## 改進內容

### 1. 模型容量提升
- **WaveNet blocks: 1 → 3**
- 參數量: 8.5M → ~20M
- Receptive field 增加 3 倍

### 2. 改進的學習率調度
- 替換 NewbobAdam 為 **Adam + ReduceLROnPlateau**
- Patience: 10 epochs（原本 ~3 epochs）
- 衰減因子: 0.5
- 最小學習率: 1e-5

### 3. 分階段訓練策略

| 階段 | Epoch 範圍 | L2 權重 | Phase 權重 | IPD 權重 | 目標 |
|------|-----------|---------|-----------|----------|------|
| Stage 1 | 0-29 | 10.0 | 0.01 | 0.0 | 專注時域波形重建 |
| Stage 2 | 30-69 | 10.0 | 0.01 | 0.1 | 加入空間音訊特徵 |
| Stage 3 | 70-100 | 15.0 | 0.005 | 0.05 | 全面優化 + 微調 |

### 4. 動態 Loss 權重
- L2 權重從 1.0 提升到 10.0-15.0
- 強制模型優化時域波形
- 後期降低 Phase/IPD 權重避免過度優化頻域

## 使用方法

### 啟動訓練
```bash
./start_train_v4.sh
```

### 查看訓練進度
```bash
# 即時日誌
tail -f train_v4.log

# 訓練歷史
python monitor_training.py outputs_v4

# 或直接查看 CSV
tail outputs_v4/training_logs/training_history.csv
```

### 從 checkpoint 恢復
```bash
python train_v4.py \
    --dataset_directory ./dataset/trainset \
    --artifacts_directory ./outputs_v4 \
    --num_gpus 1 \
    --blocks 3 \
    --resume epoch-20
```

## 預期效果

| 指標 | v3 (epoch 31) | v4 目標 (epoch 100) | 論文大模型 |
|------|---------------|---------------------|-----------|
| L2 (×10³) | 0.536 | **0.15-0.18** | 0.144 |
| Phase | 0.993 | **0.80-0.85** | 0.804 |
| IPD | 0.408 | **0.35-0.40** | — |

## 訓練時間估算

- 單 epoch: ~13-20 分鐘（3 blocks 比 1 block 慢約 1.5 倍）
- 100 epochs: ~22-33 小時
- 建議使用 `nohup` 或 `screen` 在背景執行

## 監控重點

### 階段 1 (Epoch 0-29)
- L2 應快速下降到 0.0003 以下
- Phase 下降到 1.0 左右
- IPD 權重為 0，不需關注

### 階段 2 (Epoch 30-69)
- L2 繼續下降到 0.0002 左右
- IPD 開始優化，應降到 0.4 以下
- 學習率可能觸發第一次衰減

### 階段 3 (Epoch 70-100)
- L2 目標: 0.00015-0.00018
- Phase 目標: 0.80-0.85
- 微調階段，改善幅度較小

## 故障排除

### GPU 記憶體不足
```bash
# 降低 batch size
# 修改 train_v4.py 第 35 行
"batch_size": 8,  # 從 16 改為 8
```

### 訓練不收斂
- 檢查 L2 是否在 epoch 30 前降到 0.0003 以下
- 如果沒有，可能需要調整 Stage 1 的 L2 權重到 15.0

### 學習率衰減太快
- 增加 patience 到 15
- 修改 train_v4.py 第 43 行

## 與 v2/v3 的差異

| 特性 | v2/v3 | v4 |
|------|-------|-----|
| Blocks | 1 | 3 |
| 參數量 | 8.5M | ~20M |
| LR 調度 | NewbobAdam (patience~3) | ReduceLROnPlateau (patience=10) |
| Loss 權重 | 固定 | 動態分階段 |
| L2 權重 | 1.0 | 10.0-15.0 |
| 預期 L2 | 0.52-0.54 | 0.15-0.18 |

## 檔案結構

```
outputs_v4/
├── training_logs/
│   ├── training_config.json      # 訓練配置
│   ├── training_history.json     # 完整訓練歷史
│   └── training_history.csv      # CSV 格式
├── binaural_network.epoch-10.net # 模型權重
├── checkpoint.epoch-10.pth       # 完整 checkpoint
└── ...
```

## 注意事項

1. **記憶體需求**: 3 blocks 需要約 12-16GB GPU 記憶體（batch_size=16）
2. **訓練時間**: 比 v2/v3 慢約 1.5-2 倍
3. **Checkpoint**: 每 10 epochs 自動保存，包含 optimizer 和 scheduler 狀態
4. **中斷恢復**: 使用 `--resume epoch-XX` 可完整恢復訓練狀態

## 參考

- 論文基準: L2=0.144, Phase=0.804 (3 blocks, 4 GPUs)
- v2 結果: L2=0.527, Phase=0.862 (1 block, 1 GPU, 59 epochs)
- v3 結果: L2=0.536, Phase=0.993 (1 block, 1 GPU, 31 epochs, 進行中)
