# Python 檔案清理建議

**分析日期**：2026-04-12

---

## 檔案分類分析

### 核心檔案（必須保留）

#### 主要腳本
- `train.py` - 訓練腳本 ✅
- `evaluate.py` - 評估腳本 ✅
- `synthesize.py` - 合成腳本 ✅
- `finetune.py` - 微調腳本 ✅

#### 監控與測試工具
- `monitor_training.py` - 即時監控訓練 ✅
- `visualize_training.py` - 視覺化訓練記錄 ✅

#### src/ 核心模組
- `src/dataset.py` - 資料集處理 ✅
- `src/models.py` - 模型定義 ✅
- `src/trainer.py` - 訓練器 ✅
- `src/losses.py` - 損失函數（包含 IPDLoss）✅
- `src/utils.py` - 工具函數 ✅
- `src/synthesis_utils.py` - 合成工具 ✅
- `src/doa.py` - DOA 估計方法 ✅
- `src/alignment.py` - 時間對齊 ✅
- `src/hyperconv.py` - Hyperconvolution 層 ✅
- `src/warping.py` - Time Warping ✅
- `src/finetuner.py` - 微調器 ✅

---

## 可以刪除的檔案

### 1. 重複的 regenerate_tx.py 檔案 ❌

**問題**：有 4 個幾乎相同的 regenerate_tx.py 檔案
```
experiments/regenerate_tx.py           # 實驗版本
dataset/testset/regenerate_tx.py       # testset 版本
dataset/testset_org/regenerate_tx.py   # testset_org 版本
dataset/testset_KEMAR/regenerate_tx.py # KEMAR 版本
```

**建議**：
- 保留 `experiments/regenerate_tx.py`（最完整）
- 刪除其他 3 個重複檔案

### 2. 過時的測試檔案 ❌

#### test_ipd_loss.py
- **用途**：測試 IPD Loss 基本功能
- **狀態**：已被 `test_ipd_gradient.py` 取代
- **建議**：刪除（功能重複且較舊）

#### test_training_logs.py
- **用途**：測試訓練記錄功能
- **狀態**：功能已穩定，測試檔案不再需要
- **建議**：移到 experiments/ 或刪除

### 3. 診斷工具（可選刪除）⚠️

#### check_90deg_ild.py
- **用途**：檢查 ±90° 的 ILD 值
- **狀態**：一次性診斷工具
- **建議**：移到 experiments/ 或刪除

#### check_90deg_itd.py
- **用途**：檢查 ±90° 的 ITD 值
- **狀態**：一次性診斷工具
- **建議**：移到 experiments/ 或刪除

#### diagnose_itd_all_angles.py
- **用途**：診斷所有角度的 ITD
- **狀態**：一次性診斷工具，功能較完整
- **建議**：移到 experiments/

---

## 建議的清理動作

### 立即刪除（重複檔案）

```bash
# 刪除重複的 regenerate_tx.py
rm dataset/testset/regenerate_tx.py
rm dataset/testset_org/regenerate_tx.py
rm dataset/testset_KEMAR/regenerate_tx.py

# 刪除過時的測試檔案
rm test_ipd_loss.py
```

### 移動到 experiments/（診斷工具）

```bash
# 移動診斷工具到 experiments
mv check_90deg_ild.py experiments/
mv check_90deg_itd.py experiments/
mv diagnose_itd_all_angles.py experiments/
mv test_training_logs.py experiments/
```

### 保留但可考慮移動

#### test_ipd_gradient.py
- **建議**：移到 experiments/（測試工具）
- **原因**：不是日常使用的腳本

---

## 檔案依賴關係分析

### 高依賴檔案（不能刪除）
- `src/` 下的所有模組：被多個腳本依賴
- `train.py`, `evaluate.py`, `synthesize.py`：主要功能腳本

### 獨立檔案（可安全刪除）
- `test_ipd_loss.py`：只測試 IPDLoss，無其他依賴
- `check_90deg_*.py`：獨立的診斷工具
- 重複的 `regenerate_tx.py`：功能重複

### 實驗性檔案（已在 experiments/）
- experiments/ 下的所有檔案都是實驗性的
- 可以保留但不影響核心功能

---

## backup_original/ 資料夾

### 狀態
- 包含原始程式碼的完整備份
- 7 個檔案：dataset.py, hyperconv.py, losses.py, models.py, trainer.py, utils.py, warping.py, train.py

### 建議
- **保留**：作為歷史參考和回滾用途
- 不需要清理，已經在獨立資料夾中

---

## 清理後的檔案結構

### 根目錄 Python 檔案（8個）
```
train.py                    # 訓練腳本
evaluate.py                 # 評估腳本
synthesize.py               # 合成腳本
finetune.py                 # 微調腳本
monitor_training.py         # 監控工具
visualize_training.py       # 視覺化工具
test_ipd_gradient.py        # IPD 測試（可移到 experiments/）
```

### src/ 模組（11個）
```
src/alignment.py
src/dataset.py
src/doa.py
src/finetuner.py
src/hyperconv.py
src/losses.py
src/models.py
src/synthesis_utils.py
src/trainer.py
src/utils.py
src/warping.py
```

### experiments/ 實驗腳本（約20個）
- 包含所有實驗性和診斷工具
- 移入的診斷工具：check_90deg_*.py, diagnose_itd_all_angles.py

---

## 執行清理的命令

```bash
# 1. 刪除重複檔案
rm dataset/testset/regenerate_tx.py
rm dataset/testset_org/regenerate_tx.py  
rm dataset/testset_KEMAR/regenerate_tx.py

# 2. 刪除過時測試檔案
rm test_ipd_loss.py

# 3. 移動診斷工具到 experiments
mv check_90deg_ild.py experiments/
mv check_90deg_itd.py experiments/
mv diagnose_itd_all_angles.py experiments/
mv test_training_logs.py experiments/

# 4. 可選：移動測試工具
mv test_ipd_gradient.py experiments/
```

---

## 清理效果

### 清理前
- 根目錄 Python 檔案：13個
- 重複檔案：4個 regenerate_tx.py
- 過時測試檔案：2個

### 清理後
- 根目錄 Python 檔案：7個（減少 46%）
- 重複檔案：0個
- 所有診斷工具整合到 experiments/

### 好處
1. **根目錄更簡潔**：只保留核心功能腳本
2. **消除重複**：避免維護多個相同檔案
3. **分類清晰**：診斷工具統一放在 experiments/
4. **降低混淆**：減少過時檔案的干擾

---

## 注意事項

1. **備份重要**：執行刪除前先確認 Git 狀態
2. **測試功能**：清理後測試核心功能是否正常
3. **文檔更新**：更新相關文檔中的檔案路徑
4. **漸進式清理**：可以分批執行，先移動再刪除

---

**建議執行順序**：
1. 先移動檔案到 experiments/
2. 測試核心功能
3. 確認無問題後刪除重複檔案
4. 更新文檔

**風險評估**：低風險
- 刪除的都是重複或過時檔案
- 核心功能不受影響
- 可以從 Git 歷史恢復