# 專案結構整理

## 核心檔案

### 模型定義
- `src/models_hybrid.py` - HybridTFNet 架構（待改進為 IPD-only）
- `src/models_dpatfnet.py` - DPATFNet 參考實現
- `src/models_v2.py` - v8.3 模型（baseline）

### 訓練腳本
- `train_hybrid_corrected.py` - HybridTFNet 訓練（當前版本）
- `train_dpatfnet.py` - DPATFNet 訓練
- `train_v8.py` - v8.3 訓練

### 啟動腳本
- `start_train_hybrid.sh` - 啟動 HybridTFNet
- `start_train_dpatfnet.sh` - 啟動 DPATFNet
- `start_train_v8.sh` - 啟動 v8.3

### 資料處理
- `src/dataset.py` - 資料載入
- `src/warping.py` - Time-domain warping
- `src/losses.py` - Loss functions

## 實驗記錄

### 重要文件
- `實驗記錄/20260427_HybridTFNet失敗分析與DPATFNet實現.md` - 完整實驗記錄
- `ai/discussion_l2_loss.md` - Sub-agent 討論記錄

## 待刪除/歸檔

### 舊版本（可歸檔）
- `training/train_v*.py` - 舊版本訓練腳本
- `src/trainer_v*.py` - 舊版本 trainer
- `experiments/train_*.py` - 實驗性訓練腳本

### 實驗工具（保留但不常用）
- `experiments/*.py` - 各種診斷和實驗工具
- `scripts/*.py` - 監控和可視化工具

## 下一步

### 即將創建
- `src/models_hybrid_ipd.py` - IPD-only 版本
- `train_hybrid_ipd.py` - IPD-only 訓練腳本
- `start_train_ipd.sh` - 啟動腳本
