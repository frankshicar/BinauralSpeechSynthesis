# HybridTFNet 失敗分析與 DPATFNet 實現

**日期**: 2026-04-27  
**目標**: 解決 BinauralTFNet v8.3 的 Phase/ITD 學習失敗問題  
**結果**: HybridTFNet 創新失敗，改為實現 DPATFNet baseline

---

## 一、HybridTFNet 設計理念

### 核心創新
1. **Time-Frequency 分離**：Time Branch 學 Phase，Freq Branch 學 Magnitude
2. **Geometric ITD + Neural Correction**：用 Woodworth 公式提供幾何先驗
3. **Phase Difference 學習**：學習相對於 mono 的 phase 變化，而非 absolute phase
4. **FiLM 調制**：用 Feature-wise Linear Modulation 注入位置資訊

### 架構演進

#### 版本 1：Geometric ITD + Frequency-Dependent Delay
```
TimeBranch:
  GeometricITD (Woodworth formula, fixed)
  → FrequencyDependentDelay (learnable, per-frequency delay)
  → Phase_L, Phase_R
```
**結果**: ❌ Phase error 3.28（接近 uniform random）  
**問題**: Delay-based 方法無法建模複雜的 HRTF phase response

---

#### 版本 2：直接預測 Phase（時域 broadcast）
```
TimeBranch:
  shared_feat (B×128×T)
  → time_to_freq (B×64×T)
  → Cross-Attention(position)
  → expand to (B×64×F×T)  # broadcast 到所有頻率
  → phase_net (Conv2d)
  → Phase_L, Phase_R
```
**結果**: ❌ Phase error 3.28  
**問題**: `expand` 讓所有頻率 bin 共享同一份數據，Conv2d 學不到頻率相關資訊

---

#### 版本 3：STFT 域預測 Phase
```
TimeBranch:
  STFT(mono) → real + imag (B×2×F×T)
  → stft_encoder (Conv2d)
  → Cross-Attention(position)
  → phase_net (Conv2d, 3 layers)
  → Phase_L, Phase_R
```
**結果**: ❌ Phase error 3.28  
**問題**: Cross-Attention 的 residual 連接讓 position 資訊被淹沒（梯度 1e-5 vs 1e-2）

---

#### 版本 4：FiLM 調制
```
TimeBranch:
  STFT(mono) → stft_encoder
  → FiLM(position)  # gamma * feat + beta
  → phase_net
  → Phase_L, Phase_R
```
**結果**: ❌ Phase error 3.28  
**問題**: 
- Position 梯度提升到 0.3（成功），但 Phase error 不動
- Tanh 激活限制輸出範圍，移除後仍無改善

---

#### 版本 5：Phase Difference 學習
```
Loss:
  Phase_diff_L_gt = angle(Y_L_gt / Y_mono)
  Phase_diff_R_gt = angle(Y_R_gt / Y_mono)
  loss = wrapped_mse(Phase_diff_L_pred, Phase_diff_L_gt)
```
**結果**: ❌ Phase error 3.28  
**Sanity Check**:
```
Phase_diff_L GT - std: 1.82 (接近 uniform π/√3 ≈ 1.81)
Phase_diff_R GT - std: 1.82
```
**結論**: Phase difference 本身就接近 uniform，不是可學習的目標

---

#### 版本 6：低頻 Phase Loss
```
Loss: 只在 <1.5kHz 計算 phase loss
```
**結果**: ❌ 低頻 Phase error 仍是 3.28  
**Sanity Check**:
```
Low-freq (<1.5kHz) Phase_diff_L - std: 1.83 (跟全頻段一樣)
```
**結論**: 即使在低頻，phase difference 也是 uniform

---

## 二、根本問題診斷

### 實驗 A：GT Magnitude + Mono Phase 重建
```python
Y_L_recon = Mag_L_gt * exp(1j * Phase_mono)
L2_error = mse(recon, target)
```
**結果**: L2 = 0.000695（很小）  
**結論**: ✅ Mono phase 可以用來重建 binaural（理論上限）

---

### 實驗 B：模型預測的 Magnitude 統計
```
Mag_L_pred - mean: 0.69, std: 0.0012 (幾乎是常數！)
Mag_L_gt   - mean: 0.07, std: 0.48 (動態範圍很大)
```
**結論**: ❌ 模型輸出的 magnitude 沒有動態範圍，完全沒在學

---

### 實驗 C：時域端到端 Baseline
```python
class SimpleBaseline:
    Conv1d(mono + view → binaural)
    直接在時域預測波形，不用 STFT
```
**結果**: L2 從 0.000552 → 0.000547（30 epochs，幾乎沒動）  
**結論**: ❌ 即使最簡單的時域模型也學不動

---

## 三、失敗原因總結

### 1. Phase Difference 不是可學習的目標
- `angle(Y_L / Y_mono)` 的 std ≈ 1.82（接近 uniform）
- Wrapping 太嚴重，即使在低頻也是
- Mono 和 target 的 STFT phase 雖然內容相同，但微小的 delay 和 HRTF 的 all-pass 特性導致 phase difference 看起來是隨機的

### 2. 模型容量不足
- Channels = 64, Blocks = 4（為了避免 OOM）
- 相比 DPATFNet 的 256 channels, 8+ blocks，表達能力不足

### 3. 缺少長程依賴建模
- 只有 3 層 Conv2d，感受野太小
- 沒有 Self-Attention，無法捕捉頻率間的相關性

### 4. 訓練設定可能有問題
- L2 loss 對 phase 很敏感
- 可能需要 multi-scale STFT loss 或 perceptual loss
- 音訊長度（3 秒）可能不夠

---

## 四、DPATFNet 實現

### 架構
```
Input: mono (B×1×T) + view (B×7)
  ↓
STFT → Magnitude (B×1×F×T)
  ↓
Encoder (Conv2d: 1→64→128→256)
  ↓
DPAB × N (Dual-Path Attention Block)
  ├─ Intra-frame Self-Attention (across frequency)
  ├─ Inter-frame Self-Attention (across time)
  ├─ Position Cross-Attention
  └─ Feed-Forward
  ↓
Decoder (Conv2d: 256→128→64→4)
  ↓
Output: real_L, imag_L, real_R, imag_R (B×4×F×T)
  ↓
iSTFT → binaural (B×2×T)
```

### 關鍵差異
| 特性 | HybridTFNet | DPATFNet |
|------|-------------|----------|
| 分支數 | 2 (Time + Freq) | 1 (統一) |
| Phase 學習 | Phase difference | Complex spectrum (real + imag) |
| Attention | Cross-Attention only | Self + Cross Attention |
| Dual-Path | ❌ | ✅ Intra/Inter-frame |
| 輸出 | Phase + Magnitude | Complex spectrum |

### 實現細節
- **檔案**: `src/models_dpatfnet.py`
- **參數**: 1.17M (channels=128, num_dpab=3)
- **Loss**: Complex MSE + 10 × Time-domain L2
- **Optimizer**: Adam (lr=3e-4) + ReduceLROnPlateau

---

## 五、清理的檔案

### 刪除（失敗的實驗）
- `src/models_hybrid.py` - HybridTFNet 各版本
- `train_hybrid.py` - HybridTFNet 訓練腳本
- `train_baseline.py` - Freq Branch only baseline
- `train_simple_baseline.py` - 時域端到端 baseline
- `test_*.py` - 各種診斷腳本
- `experiments/test_*.py` - Toy experiments

### 保留
- `src/models_dpatfnet.py` - DPATFNet 實現
- `train_dpatfnet.py` - DPATFNet 訓練腳本
- `src/dataset.py` - Dataset (不變)
- `實驗記錄/` - 所有實驗記錄

---

## 六、下一步

### 1. 訓練 DPATFNet
```bash
python train_dpatfnet.py
```

### 2. 監控指標
- Train/Val Loss（Complex MSE + Time L2）
- 每 10 epochs 儲存 checkpoint
- 使用 ReduceLROnPlateau 自動調整學習率

### 3. 如果成功
- 評估 test set 性能
- 與 v8.3 比較
- 考慮加入部分創新（如 FiLM）

### 4. 如果失敗
- 檢查資料處理（normalization, augmentation）
- 嘗試 multi-scale STFT loss
- 增加音訊長度（3秒 → 5秒）
- 參考 DPATFNet 論文的完整訓練設定

---

## 七、經驗教訓

1. **先跑通 baseline，再做創新**
   - 花了大量時間在創新架構上，但連 baseline 都沒跑通
   - 應該先複現 DPATFNet，確認資料和訓練設定正確

2. **Sanity check 很重要**
   - Phase difference std = 1.82 這個數字早該發現
   - GT magnitude + mono phase 的重建實驗應該更早做

3. **不要過度相信指標**
   - Phase error 3.28 可能本身就不是正確的評估指標
   - 應該多看其他指標（L2, IPD, perceptual metrics）

4. **記憶體限制影響模型設計**
   - Channels 64 vs 256 是巨大的能力差異
   - 應該優先用 gradient checkpointing 而不是砍模型容量

5. **創新需要建立在穩固的基礎上**
   - Phase difference 的想法不錯，但實現上有根本問題
   - 應該先確認學習目標是可學習的（通過 sanity check）
