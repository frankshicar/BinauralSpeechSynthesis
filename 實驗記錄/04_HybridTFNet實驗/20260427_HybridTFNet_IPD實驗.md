# 實驗記錄：HybridTFNet-IPD

**日期**：2026-04-27  
**作者**：AI Engineer Agent  
**目標**：解決 HybridTFNet Phase 學習失敗問題

---

## 背景

### 問題回顧

經過 6 個版本的 HybridTFNet 實驗，發現 **Phase Difference 學習目標有根本缺陷**：

```python
# 之前的做法
Phase_diff_L = angle(Y_L_gt / Y_mono)
Phase_diff_R = angle(Y_R_gt / Y_mono)
```

**問題**：
1. Phase difference 的 GT 分佈接近 uniform（std ≈ 1.82 ≈ π/√3）
2. Wrapped MSE 對 uniform distribution 的梯度很弱
3. 模型無法學到有意義的映射

**證據**：
- 訓練 9 epochs 後：
  - Train Loss 下降：6.47 → 6.35 ✅
  - L2 不動：0.000719 ≈ 理論上限 0.000695 ❌
  - Phase error 不動：3.18 ≈ uniform random ❌

**結論**：模型學到了 GT magnitude + mono phase，Phase Branch 完全無效。

---

## 新方法：IPD-only Learning

### 核心想法

**只學習 IPD (Interaural Phase Difference)**，而不是分別學習 Phase_L 和 Phase_R。

```python
# 新做法
IPD = Phase_L - Phase_R  # 只預測這個
Phase_L = Phase_mono + IPD / 2  # 對稱生成
Phase_R = Phase_mono - IPD / 2
```

### 為什麼這樣可能有效？

1. **IPD 的變異性更低**
   - Phase_L 和 Phase_R 各自接近 uniform
   - 但 IPD = Phase_L - Phase_R 的變異性較低
   - 之前實驗顯示 IPD error 有在下降（2.85 → 2.78）

2. **更符合人耳感知**
   - 人耳主要依賴 IPD 來定位聲源（尤其低頻）
   - 絕對 phase 不重要，相對 phase 才重要

3. **減少學習目標複雜度**
   - 從 2 個輸出（Phase_L, Phase_R）減少到 1 個（IPD）
   - 參數更少，更容易學習

4. **保持創新性**
   - DPATFNet 學習完整的 phase
   - 我們只學習 IPD → 仍然是創新

---

## 實現細節

### 模型架構

**檔案**：`src/models_hybrid_ipd.py`

```python
class TimeBranchIPD(nn.Module):
    """只預測 IPD"""
    def forward(self, mono, shared_feat, view):
        # 1. STFT
        Y_mono = torch.stft(mono, ...)
        
        # 2. 位置編碼
        pos_feat = self.position_encoder(view)
        
        # 3. STFT 特徵編碼
        stft_feat = torch.stack([Y_mono.real, Y_mono.imag], dim=1)
        phase_feat = self.stft_encoder(stft_feat)
        
        # 4. FiLM 調制
        gamma = self.gamma_net(pos_feat)
        beta = self.beta_net(pos_feat)
        phase_feat = gamma * phase_feat + beta
        
        # 5. IPD prediction
        IPD = self.output_ipd(self.ipd_net(phase_feat))
        
        # 6. 對稱生成 Phase_L 和 Phase_R
        Phase_mono = torch.angle(Y_mono)
        Phase_L = Phase_mono + IPD / 2
        Phase_R = Phase_mono - IPD / 2
        
        return IPD, Phase_L, Phase_R
```

**關鍵改進**：
1. 只有一個輸出頭 `output_ipd`（之前有 `output_L` 和 `output_R`）
2. 從 IPD 對稱生成 Phase_L 和 Phase_R
3. 保證 `Phase_L - Phase_R = IPD`（數學上一致）

### 模型參數

```
Total parameters: 2,179,971 (2.2M)
```

**對比**：
- HybridTFNet (Phase difference): 11.5M
- HybridTFNet-IPD: 2.2M ✅ (減少 80%)
- DPATFNet: 515K

**說明**：
- 減少參數是因為：
  - `tf_channels`: 256 → 128
  - `tf_blocks`: 8 → 4
  - Time Branch 只有 1 個輸出頭（之前 2 個）
- 仍然比 DPATFNet 大，因為保留了 Time-Freq 分離架構

### Loss Function

```python
def compute_losses(y_pred, y_gt, outputs, mono, config):
    # GT IPD
    IPD_gt = Phase_L_gt - Phase_R_gt
    
    # 1. L2 Loss (time-domain)
    loss_l2 = F.mse_loss(y_pred, y_gt)
    
    # 2. IPD Loss (wrapped MSE)
    loss_ipd = wrapped_mse(outputs['IPD'], IPD_gt)
    
    # 3. Magnitude Loss
    loss_mag = (F.mse_loss(outputs['Mag_L'], Mag_L_gt) + 
                F.mse_loss(outputs['Mag_R'], Mag_R_gt)) / 2
    
    # Total
    loss = 10 * loss_l2 + 1 * loss_ipd + 1 * loss_mag
    
    return loss
```

**權重設計**：
- L2: 10 → 確保波形質量
- IPD: 1 → 提供 phase 監督
- Mag: 1 → 提供 magnitude 監督

### 訓練配置

**檔案**：`train_hybrid_ipd.py`

```python
config = {
    # Model
    'sample_rate': 48000,  # ✅ 正確的 sample rate
    'n_fft': 1024,
    'hop_size': 64,
    'tf_channels': 128,
    'tf_blocks': 4,
    'use_checkpointing': True,  # ✅ 節省記憶體
    
    # Training
    'batch_size': 16,
    'learning_rate': 3e-4,
    'num_epochs': 100,
    'gradient_clip': 5.0,
    
    # Data
    'chunk_size_ms': 200,
    
    # Loss weights
    'lambda_l2': 10.0,
    'lambda_ipd': 1.0,
    'lambda_mag': 1.0,
    
    # Early stopping
    'patience': 10,
    'min_delta': 1e-5,
    
    # Random seed
    'seed': 42,  # ✅ 確保可重現性
}
```

**改進**：
1. ✅ **Gradient Checkpointing**：節省 40-60% 記憶體
2. ✅ **Random Seed**：確保可重現性
3. ✅ **Early Stopping**：避免浪費計算資源
4. ✅ **正確的 sample_rate**：48000 Hz

---

## 實驗設計

### 驗證目標

**成功標準**：
1. **IPD error < 1.5**（從 2.5 降到 1.5）
2. **L2 < 0.0003**（從 0.0007 降到 0.0003）
3. **20 epochs 內看到明顯改善**

**對比基準**：
- HybridTFNet (Phase difference): L2 = 0.000719, Phase = 3.18
- 理論上限（GT mag + mono phase）: L2 = 0.000695

### 監控指標

**Loss**：
- `l2`: Time-domain L2 loss
- `ipd`: IPD wrapped MSE
- `mag`: Magnitude MSE

**Metrics**：
- `ipd`: IPD error (RMSE)
- `phase_L`: Phase_L error (RMSE)
- `phase_R`: Phase_R error (RMSE)
- `mag_L`: Magnitude_L error (RMSE)
- `mag_R`: Magnitude_R error (RMSE)
- `l2`: Time-domain L2 error (RMSE)

### 預期結果

**如果成功**：
- IPD error 應該從 ~2.5 降到 <1.5
- L2 應該從 0.0007 降到 <0.0003
- Phase_L 和 Phase_R error 也應該下降（因為從 IPD 生成）

**如果失敗**：
- IPD error 停在 ~2.5 不動
- L2 停在 0.0007 不動
- 說明 IPD 本身也是 unlearnable

---

## 與之前版本的對比

| 版本 | 學習目標 | 參數量 | L2 | Phase Error | 問題 |
|------|----------|--------|----|-----------|----|
| v1 | Geometric ITD | 11.5M | - | 3.28 | Delay 無法建模 HRTF |
| v2 | Phase (time broadcast) | 11.5M | - | 3.28 | Conv2d 學不到東西 |
| v3 | Phase (STFT domain) | 11.5M | - | 3.28 | Cross-Attn residual 太強 |
| v4 | Phase (FiLM) | 11.5M | - | 3.28 | Phase 仍學不到 |
| v5 | Phase difference | 11.5M | 0.000719 | 3.18 | Phase diff 接近 uniform |
| v6 | Low-freq phase | 11.5M | 0.000719 | 3.18 | 低頻 phase diff 也 uniform |
| **v7 (IPD)** | **IPD only** | **2.2M** | **?** | **?** | **待驗證** |

---

## 與 DPATFNet 的差異

| 項目 | DPATFNet | HybridTFNet-IPD |
|------|----------|-----------------|
| 架構 | Dual-Path Attention | Time-Freq 分離 + FiLM |
| 學習目標 | 完整 phase | 只學 IPD |
| 參數量 | 515K | 2.2M |
| Batch size | 8 | 16 |
| 創新性 | 複製論文 | 原創設計 |

**我們的優勢**：
1. Time-Freq 分離更符合人耳感知
2. IPD-only 更簡單、更容易學習
3. Batch size 更大，訓練更快

---

## 檔案清單

### 新增檔案
- `src/models_hybrid_ipd.py` - IPD-only 模型
- `train_hybrid_ipd.py` - 訓練腳本
- `start_train_ipd.sh` - 啟動腳本
- `實驗記錄/20260427_HybridTFNet_IPD實驗.md` - 本文件

### 相關檔案
- `src/models_hybrid.py` - 舊版 HybridTFNet（Phase difference）
- `train_hybrid_corrected.py` - 舊版訓練腳本
- `實驗記錄/20260427_HybridTFNet失敗分析與DPATFNet實現.md` - 失敗分析
- `ai/code_review.md` - Code review 記錄

---

## 下一步

### 立即執行
```bash
# 停止舊的訓練
pkill -f train_hybrid_corrected.py

# 啟動新訓練
cd /home/sbplab/frank/BinauralSpeechSynthesis
./start_train_ipd.sh

# 監控
tail -f train_hybrid_ipd.log
```

### 驗證計畫

**第一階段（5 epochs）**：
- 檢查 IPD error 是否下降
- 檢查 L2 是否突破 0.0007

**第二階段（20 epochs）**：
- 如果 IPD error < 2.0，繼續訓練
- 如果 IPD error 停在 2.5，停止並分析

**第三階段（100 epochs）**：
- 如果持續改善，訓練到收斂
- 目標：IPD < 1.5, L2 < 0.0003

---

## 風險與備案

### 風險 1：IPD 也學不到

**可能原因**：
- IPD 的 GT 分佈仍然太 uniform
- Wrapped MSE 仍然梯度太弱

**備案**：
- 嘗試其他 phase representation（Group Delay）
- 或者放棄 phase learning，只學 magnitude

### 風險 2：記憶體不足

**可能原因**：
- Gradient checkpointing 沒有正常工作
- Batch size 16 仍然太大

**備案**：
- 減少 batch size 到 8
- 減少 `tf_channels` 到 64

### 風險 3：訓練太慢

**可能原因**：
- Gradient checkpointing 增加計算時間
- 200ms audio 太長

**備案**：
- 減少 `chunk_size_ms` 到 100ms
- 減少 validation 頻率

---

## 總結

**核心假設**：IPD 比 Phase difference 更容易學習

**驗證方法**：訓練 20 epochs，觀察 IPD error 和 L2

**成功標準**：IPD < 1.5, L2 < 0.0003

**時間估計**：
- 5 epochs: ~2 小時（初步驗證）
- 20 epochs: ~8 小時（完整驗證）
- 100 epochs: ~40 小時（訓練到收斂）

**預期結果**：
- 樂觀：IPD 學習成功，L2 顯著下降
- 中性：IPD 部分改善，但不夠理想
- 悲觀：IPD 也學不到，需要改變方向

---

**實驗開始時間**：2026-04-27 14:15  
**預計完成時間**：2026-04-27 22:00（初步結果）
