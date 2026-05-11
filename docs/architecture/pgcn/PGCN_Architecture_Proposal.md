# PhysicsGuidedComplexNet (PGCN) 架構提案

**日期**：2026-04-27  
**狀態**：設計階段  
**目標**：L2 和 Phase 都超越 DPATFNet，固定角度誤差 < 人類 MAA（1-2°）

---

## 執行摘要

經過 5 個 AI agent（2 位架構師 + 2 位審查者 + 1 位整合者）的討論，我們提出 **PhysicsGuidedComplexNet (PGCN)**：一個結合物理先驗和複數域建模的雙耳音訊合成架構。

### 核心創新

1. **Fourier Position Encoder (L=10)**：提供 0.1° 角度精度
2. **Complex-valued 全程建模**：避免 phase wrapping 問題
3. **64-band FiLM Modulation**：精細的頻率調制
4. **Physics-Constrained Head**：注入 ITD/ILD 物理先驗
5. **Minimum-Phase Enforcer**：確保因果性

### 預期效果

| 指標 | DPATFNet 論文 | 你的 DPATFNet 實作 | PGCN 目標 |
|------|--------------|-------------------|----------|
| L2 (×10⁻³) | ~0.144 | 0.180 | **< 0.14** |
| Phase | 0.70 | 3.28 | **< 0.65** |
| 角度誤差 | ? | 46.6° | **< 2°** |
| 記憶體 | ? | OOM (hop=64) | < 16GB (hop=256) |

### 實作時間：10 天

---

## 一、問題診斷與失敗案例分析

### 1.1 你的失敗案例總結

| 版本 | 核心問題 | Phase 結果 | 角度誤差 |
|------|---------|-----------|---------|
| **v8 系列** | DPAB 實作錯誤，從 warped 出發限制修正能力 | 改善 8-13% | 46.6° |
| **HybridTFNet** | Phase difference (angle(Y_L/Y_mono)) std≈1.82，接近 uniform，不可學習 | 3.28 | - |
| **DPATFNet 實作** | hop_size=64 導致記憶體爆炸，砍到 channels=64/num_dpab=2 | 3.28 | - |

### 1.2 根本問題

1. **Phase wrapping 問題**
   - Phase difference 的 std≈1.82 ≈ π/√3（uniform distribution）
   - MSE loss 對 wrapped phase 不敏感
   - 解決方案：改用 complex representation (real + imag)

2. **Position encoding 精度不足**
   - 120Hz 採樣 = 8.33ms 離散化
   - 簡單的 MLP 或 1-query Cross-Attention 無法提供足夠精度
   - 解決方案：Multi-scale Fourier Features

3. **記憶體限制導致容量不足**
   - hop_size=64 → T_stft 過長 → 記憶體爆炸 → 砍 channels
   - 解決方案：hop_size=256，降低 4 倍

4. **缺乏物理先驗**
   - 純 data-driven 容易學到不合理的預測
   - 解決方案：Physics-constrained head

---

## 二、PGCN 架構設計

### 2.1 整體架構

```
輸入：
  mono:     B × 1 × T (48000 samples = 1s)
  position: B × 7 × K (K = T/400, 120Hz)

┌─────────────────────────────────────────────────────────┐
│ 1. Fourier Position Encoder (L=10)                     │
│    position (B×7×K) → pos_feat (B×256)                 │
│    Multi-scale: [2^0, 2^1, ..., 2^9] × π               │
│    精度：0.1° (vs 8.33ms 離散化)                        │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 2. STFT (n_fft=1024, hop=256)                          │
│    mono → Y_mono: B × F × T_stft (F=513, T_stft≈188)  │
│    Complex representation: real + imag                  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 3. Complex Encoder (Conv2d)                            │
│    Y_mono (B×2×F×T) → feat (B×256×F×T)                │
│    2 channels: [real, imag]                             │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 4. Complex DualPath Blocks × 4                         │
│    ├─ FreqAttention (across F, within each T)          │
│    ├─ TimeAttention (across T, within each F)          │
│    ├─ 64-band FiLM (pos_feat → gamma/beta per band)   │
│    └─ Complex FFN                                       │
│    保持 complex-valued 全程                             │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 5. Physics-Constrained Head                            │
│    ├─ Woodworth ITD formula (geometric prior)          │
│    ├─ Frequency-dependent ILD (6dB @ 1kHz baseline)   │
│    └─ Learnable residual correction                    │
│    Output: Y_L, Y_R (B×F×T_stft, complex)             │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 6. Minimum-Phase Enforcer                              │
│    Causal loss + Hilbert transform constraint          │
│    確保物理可實現性                                      │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 7. iSTFT                                                │
│    Y_L, Y_R → y_L, y_R (B×2×T)                         │
└─────────────────────────────────────────────────────────┘
```

### 2.2 參數配置

```python
# STFT
n_fft = 1024          # F = 513
hop_size = 256        # T_stft ≈ 188 (for 1s audio)
                      # 記憶體降低 4 倍 vs hop=64

# Model
channels = 256        # 足夠容量 (vs 你的實作 64)
num_dpab = 4          # 足夠深度 (vs 你的實作 2)
num_heads = 8
fourier_L = 10        # 2^10 = 1024 個 basis

# Training
batch_size = 16       # 可行 (hop=256)
chunk_size = 48000    # 1 秒
learning_rate = 3e-4
epochs = 200
```

---

詳細設計請見後續章節...
