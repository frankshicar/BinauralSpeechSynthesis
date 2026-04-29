# Binaural Speech Synthesis — 評估分析報告

生成時間：2026-04-16

---

## 1. 各 Subject 實際角度整理

> 標稱角度來自評估腳本硬編碼的 `SUBJECT_ANGLES` 字典；實際角度由 `tx_positions.txt` 計算方位角後統計。

| Subject | 標稱角度 (°) | 實際 Median 角度 (°) | Std (°) | 是否靜止 |
|---|---|---|---|---|
| subject1 | -90 | -12.08 | ~90+ | ❌ 移動 |
| subject2 | — | +37.40 | ~90+ | ❌ 移動 |
| subject3 | — | -2.66 | ~90+ | ❌ 移動 |
| subject4 | — | -17.95 | 71.16 | ❌ 移動 |
| subject5 | — | -1.81 | ~90+ | ❌ 移動 |
| subject6 | — | -9.28 | ~90+ | ❌ 移動 |
| subject7 | — | -15.46 | ~90+ | ❌ 移動 |
| subject8 | — | +9.21 | ~90+ | ❌ 移動 |
| validation_sequence | 不計算 | +1.29 | 106.43 | ❌ 移動 |

**關鍵發現：** 所有 subject 的方位角 std 均遠超 30° 閾值（最小為 71.16°），且方位角範圍幾乎橫跨 ±180°，代表說話者在測試序列中做了大範圍環繞移動，並非固定在某一角度。

---

## 2. 評估腳本 Angle Error 計算邏輯

### 2.1 整體流程

```
compute_metrics(pred, gt, ground_truth_angle)
    │
    ├─ 若 ground_truth_angle is None → 跳過角度計算
    │
    └─ 呼叫 src/doa.gcc_phat_estimate(pred)
           │
           ├─ 前處理：找最後一個振幅 > 1e-4 的 sample，再乘 0.95 截短
           │
           ├─ GCC-PHAT：
           │     FFT(左聲道) × conj(FFT(右聲道))
           │     → PHAT 加權（除以交叉功率譜絕對值）
           │     → 300–8000 Hz 頻率遮罩
           │     → 反 FFT → 在 ±48 samples 範圍找峰值 τ
           │
           ├─ ITD → 角度：θ = -arcsin(c × τ / d)
           │         c = 343 m/s，d = 0.215 m（固定耳距）
           │
           └─ 誤差 = min(|θ_pred - θ_gt|, 360 - |θ_pred - θ_gt|)
                    → 按樣本數加權平均
```

### 2.2 GT 角度決定方式

評估腳本使用**硬編碼**的 `SUBJECT_ANGLES` 字典指定每個 subject 的 GT 角度（例如 subject1 = -90°），**完全不從 `tx_positions.txt` 動態計算**。不在字典中的序列（如 `validation_sequence`）直接跳過，不納入角度誤差統計。

---

## 3. 模型誤差偏大的原因分析

原始論文大模型（3 blocks）的基準指標：

| 指標 | 論文基準 |
|---|---|
| L2 (×10³) | 0.144 |
| Amplitude | 0.036 |
| Phase | 0.804 |

以下從四個面向分析本次訓練模型誤差偏大的可能原因。

### 3.1 tx_positions 的角度分布問題

- **說話者持續移動**：所有測試序列的方位角 std 均超過 71°，範圍橫跨 ±180°，說話者並非靜止在某固定角度。
- **評估腳本假設靜止**：`SUBJECT_ANGLES` 字典為每個 subject 指定單一固定角度（如 subject1 = -90°），與實際動態移動的 tx_positions 嚴重脫鉤。
- **後果**：即使模型預測完全正確，GCC-PHAT 估計出的瞬時角度也會因說話者移動而與硬編碼 GT 角度產生大幅偏差，導致 Angle Error 虛高。

### 3.2 評估腳本的 GT 角度設定方式是否正確

- **根本問題**：GT 角度應逐幀從 `tx_positions.txt` 動態計算，而非使用單一靜態值。
- **subject1 的極端案例**：標稱角度為 -90°，但實際 median 為 -12.08°，偏差高達 ~78°，arcsin 在 ±90° 附近的非線性特性會進一步放大此誤差。
- **subject2 的偏移**：實際 median 為 +37.40°，若字典中標稱角度與此差距大，誤差貢獻將非常顯著。
- **GCC-PHAT 的輸入是預測輸出**：角度估計基於模型預測的雙耳訊號，而非 reference 錄音，若模型預測品質差，GCC-PHAT 估計本身就不準確，形成雙重誤差。

### 3.3 訓練資料與測試資料的差異

- **角度覆蓋不對稱**：若訓練集中說話者的角度分布與測試集不同（例如訓練集以正面為主，測試集包含大量側面或後方），模型對未見角度的泛化能力不足。
- **動態 vs. 靜態場景**：測試序列說話者持續移動，模型若在訓練時未充分學習動態角度變化的 HRTF 轉換，推論時會出現系統性誤差。
- **資料量與多樣性**：若本次訓練使用的資料子集較小，或未涵蓋完整的空間角度範圍，模型的空間音訊建模能力會受限。

### 3.4 模型本身的問題（訓練 loss 收斂情況）

- **訓練不足**：若訓練 epoch 數不夠或 learning rate 設定不當，模型可能尚未收斂至最佳解。
- **Batch size 影響**：在 GPU 記憶體受限的情況下縮小 batch size，可能導致梯度估計不穩定，影響收斂品質。
- **Blocks 數量**：使用 1 block（輕量模型）而非論文的 3 blocks，模型容量不足以捕捉複雜的空間音訊特徵，L2/Amplitude/Phase 指標均會較差。
- **耳距假設誤差**：GCC-PHAT 固定使用 d = 0.215 m，但資料集使用 mannequin，實際耳距可能不同，導致 ITD → 角度的換算系統性偏移。

---

## 4. 建議改善方向

### 4.1 修正 GT 角度計算方式（最高優先）

將評估腳本的靜態 `SUBJECT_ANGLES` 字典改為從 `tx_positions.txt` 動態計算每幀的方位角：

```python
import numpy as np

def compute_gt_angle_from_positions(tx_pos_file):
    """從 tx_positions.txt 計算每幀方位角（度）"""
    positions = np.loadtxt(tx_pos_file)  # shape: (N, 7)
    x, y = positions[:, 0], positions[:, 1]
    angles = np.degrees(np.arctan2(y, x))  # 方位角
    return angles
```

並在 `compute_metrics()` 中改為逐幀比對，或至少使用序列的 median 角度作為 GT。

### 4.2 確認 Mannequin 耳距

量測或查閱資料集使用的 mannequin 實際耳距，替換 GCC-PHAT 中固定的 `d = 0.215 m`，減少 ITD → 角度換算的系統性誤差。

### 4.3 使用完整 3-block 模型訓練

- 確保使用 `--blocks 3` 訓練，以達到論文基準的模型容量。
- 在有足夠 GPU 記憶體的環境下，恢復原始 batch size（論文使用 4 GPU）。

### 4.4 檢查訓練資料的角度覆蓋

統計訓練集 `tx_positions.txt` 的角度分布，確認是否覆蓋測試集中出現的所有角度範圍（尤其是 ±90° 附近的極端角度）。

### 4.5 改善 GCC-PHAT 角度估計的穩健性

- 考慮對整段序列做滑動窗口 GCC-PHAT，取中位數角度而非單一估計值。
- 或改用動態 GT 角度序列，計算逐幀誤差後再平均，避免靜態假設帶來的系統性偏差。

---

## 總結

本次評估的 Angle Error 偏高，**主要根源在於評估腳本的 GT 角度設定方式與實際測試資料嚴重不符**：所有 subject 的說話者均在測試中大範圍移動（std > 71°），但評估腳本仍以單一靜態角度作為 GT。L2 / Amplitude / Phase 指標的偏差則可能同時來自模型容量不足（blocks 數量）、訓練資料覆蓋不完整，以及訓練收斂程度。建議優先修正 GT 角度計算邏輯，再針對模型訓練條件進行調整。
