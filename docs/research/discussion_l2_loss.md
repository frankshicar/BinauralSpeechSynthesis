# Sub-Agent Discussion: Why is L2 Loss Not Decreasing?

## Context
HybridTFNet 訓練 4 個 epochs：
- Phase_L: 3.24 → 3.18 (微弱下降)
- Phase_R: 3.23 → 3.26 (震盪)
- **L2: 0.000719 → 0.000719 (完全不動)**

Loss function: `loss = loss_phase + 10 × loss_l2`

---

## Agent A (Optimist): L2 應該會下降

**論點 1: Phase 和 L2 是相關的**
- 如果 Phase 學得更好，重建的波形應該更接近 GT
- Phase_L 從 3.24 降到 3.18，雖然幅度小，但方向是對的
- L2 應該跟著下降，即使很慢

**論點 2: 可能需要更多 epochs**
- 4 個 epochs 太少，模型還在初期探索
- Phase 的改善可能需要累積到一定程度，L2 才會明顯下降
- 建議至少觀察到 Epoch 20

**論點 3: Loss weight 的影響**
- `10 × loss_l2` 的權重很大
- 模型可能優先在學 L2，但因為 Phase 不準，L2 改善很慢
- 這是正常的，需要耐心

**結論**: 繼續訓練，L2 會下降

---

## Agent B (Skeptic): L2 不動代表模型沒在學

**反駁論點 1: Phase 的微弱變化可能是噪聲**
- Phase_L 降了 0.06，但 Phase_R 升了 0.03
- 這個幅度太小（<2%），可能只是隨機波動
- 如果模型真的在學，應該兩個都下降

**反駁論點 2: L2 = 0.000719 太穩定了**
- 4 個 epochs，L2 一點都沒變（小數點後 6 位都一樣）
- 這不是"下降很慢"，而是"完全沒動"
- 即使模型在隨機探索，L2 也應該有波動

**反駁論點 3: 回顧之前的實驗**
- 之前所有 HybridTFNet 版本的 L2 都卡在 ~0.0007
- 即使換了 sample_rate，L2 還是 0.000719
- 這說明問題不在 sample_rate，而在架構本身

**關鍵觀察: GT magnitude + mono phase 的理論上限**
```python
# 之前的 sanity check
Y_L_recon = Mag_L_gt * exp(1j * Phase_mono)
L2_error = 0.000695
```

**當前 L2 = 0.000719，幾乎等於理論上限 0.000695！**

這說明：
1. 模型預測的 Magnitude 接近 GT（這部分學到了）
2. 但 Phase 完全沒學到，還是用 mono phase
3. 所以 L2 卡在理論上限，無法再降

**結論**: 模型的 Phase Branch 完全沒在工作

---

## Agent A 的回應

**承認部分觀點**:
- L2 = 0.000719 ≈ 理論上限 0.000695 這個觀察很關鍵
- 確實說明 Phase 沒有真正學到

**但提出新假設**:
- 可能 Magnitude Branch 學得太快，Phase Branch 還沒開始學
- Loss = Phase + 10×L2，如果 L2 主導了梯度，Phase 的梯度被淹沒
- 建議調整 loss weight: `loss = 10 × loss_phase + loss_l2`

**或者**:
- 分階段訓練：先只訓練 Phase Branch，再訓練 Magnitude Branch
- 這樣 Phase 不會被 L2 的梯度干擾

---

## Agent B 的最終論證

**即使調整 loss weight 也沒用**:

回顧訓練腳本：
```python
loss_phase = wrapped_mse(Phase_diff_L, Phase_diff_L_gt) + 
             wrapped_mse(Phase_diff_R, Phase_diff_R_gt)
loss_l2 = F.mse_loss(pred, target)
loss = loss_phase + 10 × loss_l2
```

**問題在於 Phase difference 本身**:
1. GT Phase difference std = 1.82 (接近 uniform)
2. 模型預測的 Phase difference 也是 ~3.2 (uniform)
3. Wrapped MSE 對兩個 uniform distribution 的梯度很弱

**更根本的問題**:
```python
# 模型的 fusion
Phase_L = Phase_mono + Phase_diff_L_pred
Y_L = Mag_L * exp(1j * Phase_L)
```

如果 `Phase_diff_L_pred` 是隨機的（uniform），那：
- `Phase_L = Phase_mono + random` 還是隨機的
- 最終的 L2 就是 GT magnitude + mono phase 的上限

**這就是為什麼 L2 卡在 0.000719**

**結論**: 
- Phase difference 這個學習目標從根本上就有問題
- 不是 loss weight 的問題，不是訓練不夠久的問題
- 需要換一個學習目標（例如直接預測 complex spectrum）

---

## 共識

**兩個 Agent 達成共識**:

1. **L2 = 0.000719 ≈ 理論上限 0.000695** 是關鍵證據
2. **Phase difference 的 GT 分布接近 uniform**，很難學
3. **當前架構無法讓 L2 下降**

**建議**:
1. 停止當前訓練（已經證明無效）
2. 改用 DPATFNet 的方法：直接預測 complex spectrum (real + imag)
3. 或者參考 Phase-aware 論文，用不同的 phase 表示方法

---

## 給用戶的建議

**Agent B 的論證更有說服力**。數據清楚顯示：
- L2 卡在理論上限
- Phase 沒有真正學到
- 繼續訓練不會有改善

**建議立即停止訓練，切換到 DPATFNet 或其他方法。**
