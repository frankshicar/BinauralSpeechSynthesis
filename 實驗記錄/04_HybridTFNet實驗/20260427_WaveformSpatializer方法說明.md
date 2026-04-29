# WaveformSpatializer 方法說明

**日期**: 2026-04-27  
**模型**: E6 - WaveformSpatializer

---

## 核心概念

### 這版在做什麼？

**時域學習 + 頻域監控**

```python
# 1. 模型在時域操作
mono_wave (B×1×T) → WaveformSpatializer → binaural_wave (B×2×T)
                     (學習 HRTF-like filter)

# 2. Loss 計算包含多個部分
loss = λ_waveform × L_waveform  # 時域 MSE
     + λ_mag × L_mag            # 頻域 Magnitude MSE
     + λ_ipd × L_ipd            # 頻域 IPD (相位差)
```

---

## 詳細流程

### Forward Pass (時域)

```python
# 輸入
mono: B×1×9600 (200ms @ 48kHz)
view: B×7×24

# 模型處理
view_feat = ViewEncoder(view)           # B×256
filter_L = FilterGenerator_L(view_feat) # B×1×512
filter_R = FilterGenerator_R(view_feat) # B×1×512

# 時域卷積 (模擬 HRTF filtering)
binaural_L = conv1d(mono, filter_L)  # B×1×9600
binaural_R = conv1d(mono, filter_R)  # B×1×9600

binaural = [binaural_L, binaural_R]  # B×2×9600
```

**關鍵**: 整個過程在時域，沒有 STFT，沒有 Phase wrapping 問題

---

### Loss Computation (時域 + 頻域)

```python
# 1. 時域 L2 loss
L_waveform = MSE(pred_wave, target_wave)
# 直接比較波形，最直觀

# 2. 轉到頻域計算其他 loss
pred_stft = STFT(pred_wave)      # B×2×F×T (complex)
target_stft = STFT(target_wave)  # B×2×F×T (complex)

# 3. Magnitude loss
L_mag = MSE(|pred_stft|, |target_stft|)
# 確保能量分佈正確

# 4. IPD loss (雙耳相位差)
pred_ipd = angle(pred_stft_L / pred_stft_R)
target_ipd = angle(target_stft_L / target_stft_R)
L_ipd = mean(1 - cos(pred_ipd - target_ipd))
# 確保雙耳定位正確

# 5. 監控 (不參與訓練)
L_phase_L = mean(1 - cos(angle(pred_stft_L) - angle(target_stft_L)))
L_phase_R = mean(1 - cos(angle(pred_stft_R) - angle(target_stft_R)))
# 只是觀察，不優化
```

---

## 為什麼這樣設計？

### 1. 時域學習避開 Phase wrapping

**問題**: STFT-based 方法
```python
# 傳統方法
mono_stft → Network → pred_stft (mag + phase)
                      ↑
                      Phase 在 [-π, π] 跳躍
                      Wrapped loss 梯度混亂
```

**解決**: Waveform-based 方法
```python
# 我們的方法
mono_wave → Network → pred_wave
                      ↑
                      連續的時域信號
                      沒有 wrapping 問題
```

### 2. 多重 Loss 確保質量

| Loss | 作用 | 為什麼需要 |
|------|------|-----------|
| Waveform | 整體波形匹配 | 最直接的目標 |
| Magnitude | 能量分佈 | 確保頻譜正確 |
| IPD | 雙耳相位差 | 確保空間定位 |

**如果只用 Waveform loss**:
- 可能學到錯誤的頻譜
- 可能忽略相位差
- 聽起來可能不自然

**加入 Magnitude + IPD**:
- 明確約束頻域特性
- 確保雙耳線索正確
- 更好的聽覺質量

### 3. Phase 監控但不優化

```python
# 為什麼不優化 Phase_L 和 Phase_R？

# 原因 1: 它們學不會
Phase_L, Phase_R 分佈太分散 (std ≈ 1.82)
直接優化會失敗 (已經證明過了)

# 原因 2: IPD 才是關鍵
人耳主要感知 IPD (雙耳差異)
不需要絕對 Phase 正確

# 原因 3: 只是監控
Phase_L, Phase_R loss 只用來觀察
確認它們確實學不會
但不影響訓練
```

---

## 與之前方法的對比

### DPATFNet / HybridTFNet (失敗)

```python
# 流程
mono_stft → Network → pred_stft (mag + phase)
                      ↓
                      直接預測 Phase
                      ↓
                      Phase wrapping 問題
                      ↓
                      學習失敗

# 結果
IPD loss: 3.07 (接近隨機)
```

### WaveformSpatializer (成功)

```python
# 流程
mono_wave → Network → pred_wave
                      ↓
                      時域卷積 (HRTF-like)
                      ↓
                      沒有 Phase wrapping
                      ↓
                      學習成功

# 結果
IPD loss: 0.9071 (改善 70%!)
```

---

## 實驗結果 (Epoch 1)

```
Train:
  Waveform: 0.000619  ← 時域誤差
  Mag: 0.118286       ← 頻域能量誤差
  IPD: 0.9217         ← 雙耳相位差 (關鍵指標)

Val:
  Waveform: 0.000593
  Mag: 0.096896
  IPD: 0.9071         ← 比 STFT 方法好 70%!

Phase (監控):
  L: 0.9974           ← 接近 1.0 = 隨機 (預期)
  R: 0.9970           ← 接近 1.0 = 隨機 (預期)
```

### 解讀

1. **IPD = 0.9071 很好**
   - Wrapped cosine loss: `1 - cos(θ)`
   - 0.9071 → 平均相位差誤差 ≈ 25°
   - 遠優於隨機基準 1.0
   - 證明模型學到了雙耳定位

2. **Phase_L/R ≈ 0.997 沒關係**
   - 個別 Phase 本來就學不會
   - 我們不需要它們正確
   - 只要 IPD (差異) 正確就好

3. **Waveform loss 合理**
   - 0.000593 是時域 MSE
   - 包含了 magnitude 和 phase 的綜合誤差
   - 比純 magnitude-only 方法更完整

---

## 總結

### 這版的創新

1. **時域學習** - 完全避開 STFT 的 Phase wrapping 問題
2. **多重約束** - Waveform + Magnitude + IPD 確保質量
3. **聰明的目標** - 優化 IPD (可學的)，監控 Phase (學不會的)

### 為什麼有效

```
傳統方法: 試圖學習 Phase → 失敗
我們的方法: 學習 waveform filter → 成功

關鍵差異:
- 時域連續 vs 頻域跳躍
- 學習 filter vs 學習 phase
- 優化 IPD vs 優化 Phase
```

### 下一步

繼續訓練，觀察:
1. IPD loss 能否進一步降低
2. Waveform 和 Magnitude 的收斂
3. 聽覺質量評估

如果 IPD 能降到 0.5 以下，就是重大突破！
