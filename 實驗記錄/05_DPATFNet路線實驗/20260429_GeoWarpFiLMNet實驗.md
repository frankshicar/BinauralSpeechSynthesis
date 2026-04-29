# GeoWarpFiLMNet 實驗記錄

**日期**：2026-04-28 ~ 2026-04-29  
**目標**：結合 Meta GeometricWarper + 64-band FiLM，超越 Meta baseline，逼近 DPATFNet

---

## 1. 架構設計

### 1.1 核心思路

- **物理先驗**：用 Meta `GeometricWarper` 提供準確的 ITD baseline（取代 Woodworth）
- **高精度位置編碼**：Fourier Position Encoder (L=8) 解決 120Hz 離散化問題
- **精細頻率調制**：64-band FiLM 對每個頻段獨立調變
- **Phase 殘差學習**：在 geometric phase 基礎上學習 ±π/2 修正

### 1.2 架構圖

```
mono (B,1,L) + view (B,7,K)
  │
  ├─ GeometricWarper（Meta 完整幾何，含 quaternion）→ y_init (B,2,L)
  │
  ├─ STFT(y_init L/R) → Y_L_init, Y_R_init (B,F,T_stft)  [F=513, hop=256]
  │     mag_L = |Y_L_init|, mag_R = |Y_R_init|
  │     phase_L_geo = angle(Y_L_init).detach()  ← 切斷梯度
  │     phase_R_geo = angle(Y_R_init).detach()
  │
  ├─ FourierPositionEncoder（L=8）→ pos_feat (B,256)
  │     view mean → sin/cos([2^0..2^7]×π×pos) → MLP(512→256)
  │
  ├─ Conv2d Encoder (2→128 channels)
  │
  ├─ FiLM ResStack × 6（dilation 1,2,4,8,1,2）
  │     每層：Conv2d(dilation) + BN + ReLU + 64-band FiLM(pos_feat)
  │     FiLM: Linear(256→128×2) → gamma/beta per freq band
  │
  ├─ Output head → mag_L_out, mag_R_out (clamp min=1e-6)
  │
  ├─ Phase head → phase_L_res, phase_R_res (tanh × π/2)
  │     residual 限制在 ±π/2，保留幾何先驗
  │
  └─ Combine:
     phase_L = phase_L_geo + phase_L_res
     phase_R = phase_R_geo + phase_R_res
     Y_L = mag_L_out × exp(i × phase_L)
     Y_R = mag_R_out × exp(i × phase_R)
     iSTFT → y_L, y_R
```

**參數量**：1.60M

### 1.3 關鍵設計決策

| 設計 | 選擇 | 理由 |
|------|------|------|
| ITD 初始化 | GeometricWarper | 完整幾何（quaternion + ear offset），比 Woodworth 準確 |
| Position 編碼 | Fourier (L=8) | 高頻特徵解決 120Hz 離散化，比 MLP mean-pool 精確 |
| 條件注入 | 64-band FiLM | 每個頻段獨立調變，比全局 Cross-Attention 精細 |
| Phase 學習 | Residual ±π/2 | 保留 geometric prior，避免從零學習 |
| Gradient 隔離 | `.detach()` | 防止梯度回傳到 warper 導致 sqrt(0) 爆炸 |

---

## 2. 訓練策略

### 2.1 兩階段訓練

| Stage | Epochs | Loss | Metric | 目的 |
|-------|--------|------|--------|------|
| 1 | 30 | `ipd + 0.1×mag_anchor + 0.1×phase` | val_ipd | 讓相位先收斂，防止 IPD 過擬合 |
| 2 | 80 | `w_l2×l2 + w_phase×phase + w_ipd×ipd` | val_phase | 全局優化 |

### 2.2 Loss 定義

**IPD Loss**（Stage 1 主要目標）：
```python
pred_cross = Y_L * Y_R.conj()
gt_cross = Y_L_gt * Y_R_gt.conj()
ipd_loss = mean((angle(pred_cross) - angle(gt_cross))²)
```

**Magnitude Anchor**（防止偏離 geometric prior）：
```python
mag_anchor = L1(|Y_L| - |Y_L_init|) + L1(|Y_R| - |Y_R_init|)
```

**Phase Loss**（Meta 原版）：
```python
phase_loss = mean(|angle(Y_pred) - angle(Y_gt)|)
```

**L2 Loss**（Meta 原版）：
```python
l2_loss = mean((y_pred - y_gt)²)
```

### 2.3 Weight Calibration

Stage 2 開始前自動校準：
```python
w_l2 = phase_raw / (l2_raw + 1e-8)     # ~1000
w_phase = 1.0                           # 基準
w_ipd = phase_raw / (ipd_raw + 1e-8)   # ~0.5
```

確保三項 loss 貢獻相等。

---

## 3. 實驗結果

### 3.1 v1：Baseline（2026-04-28）

**配置**：
- Stage 1 (30 epochs): `ipd + 0.1×mag_anchor`（無 phase）
- Stage 2 (70 epochs): `w_l2×l2 + w_phase×phase + w_ipd×ipd`
- LR: 1e-4, Patience: 15

**訓練過程**：
- Stage 1 Best IPD: 2.1539 (Epoch 30)
- Stage 2 Best Phase: 0.872 (Epoch 12)
- Early stop: Epoch 27

**最終結果**（Meta-style 評估）：

| 指標 | v1 | Meta small | Meta large | DPATFNet |
|------|----|-----------|-----------|---------| 
| L2 (×10³) | **0.166** | 0.197 | 0.144 | 0.148 |
| Amplitude | 0.046 | 0.043 | 0.036 | 0.037 |
| Phase | **0.863** | 0.862 | 0.804 | **0.717** |

**成就**：
- ✅ L2 超越 Meta small **16%**
- ✅ Phase 與 Meta small 持平（0.863 vs 0.862）
- 🔶 距離 Meta large 還有 7% 差距（0.863 vs 0.804）

### 3.2 v2：延長訓練（失敗）

**調整**：
- Stage 1: 30 → **50 epochs**
- Stage 2: 70 → **100 epochs**
- Patience: 15 → **20**

**結果**：
- Stage 1 Best IPD: 2.1438 ✅ 比 v1 好 0.01
- Stage 2 Best Phase: 0.891 ❌ 比 v1 差 0.02
- Final: L2=0.160e-3, Phase=0.886

**失敗原因**：**IPD 過擬合**

#### 根本問題：IPD vs Phase 的衝突

**IPD Loss 的陷阱**：
```
IPD = angle(Y_L × Y_R*)  → 只關心相位差
Phase = angle(Y_L), angle(Y_R) → 關心絕對相位

如果 Y_L 和 Y_R 都偏移 +30°：
  IPD 不變（差值還是對的）✅
  但 Phase 錯誤（絕對值錯了）❌
```

**Stage 1 只優化 IPD 的副作用**：
- 模型學到「同時旋轉左右耳」來降低 IPD
- 但這會讓絕對相位偏移
- Stage 2 要修正這個偏移，但已經陷入局部最優

**證據**：
- v1 (Stage 1: 30 epochs): IPD=2.15 → Phase=0.863
- v2 (Stage 1: 50 epochs): IPD=2.14 → Phase=0.891
- Stage 1 越長，IPD 越好，但 Phase 越差

### 3.3 v3：加入 Phase 監督（2026-04-29）

**核心改進**：Stage 1 加入弱 phase 監督

```python
# v1-v2: Stage 1
loss = ipd + 0.1 * mag_anchor

# v3: Stage 1
loss = ipd + 0.1 * mag_anchor + 0.1 * phase  # 新增
```

**配置**：
- Stage 1 (30 epochs): `ipd + 0.1×mag_anchor + 0.1×phase`
- Stage 2 (80 epochs): `w_l2×l2 + w_phase×phase + w_ipd×ipd`
- LR: 1e-4, Patience: 20

**最終結果**（Meta-style 評估）：

| 指標 | v3 | v1 | 改善 |
|------|----|----|------|
| L2 (×10³) | **0.155** | 0.166 | ✅ +7% |
| Amplitude | 0.045 | 0.046 | ✅ +2% |
| Phase | **0.864** | 0.863 | 🔶 +0.1% |

**分析**：
- ✅ L2 有明顯改善（0.166 → 0.155）
- ✅ 證明 Stage 1 加入 phase 監督有效
- ❌ Phase 改善極微（0.863 → 0.864）
- **結論**：遇到架構瓶頸，訓練策略已優化到極限

**瓶頸診斷**：
- Phase 卡在 0.86 附近，無法突破
- 不是訓練不足（v2 延長訓練反而更差）
- 不是 loss 設計問題（v3 已加入 phase 監督）
- **根本原因**：缺乏時序建模能力

---

## 4. 問題診斷與改進方向

### 4.1 與 DPATFNet 的差距分析

| 面向 | GeoWarpFiLMNet | DPATFNet | 差距 |
|------|----------------|----------|------|
| **ITD 初始化** | GeometricWarper | GeometricWarper | ✅ 相同 |
| **Position 編碼** | Fourier (L=8) | Learned embedding | ≈ 相當 |
| **時序建模** | ❌ 無 | Self-Attention (PAL2) | ⚠️ **關鍵差距** |
| **條件注入** | 64-band FiLM | Cross-Attention + MPF | ≈ 相當 |
| **網絡深度** | 6 blocks | 多層 Upsampling | ⚠️ 可能不足 |
| **Phase 監督** | IPD+Phase (v3) | L2+Phase | ✅ 更強 |
| **結果** | Phase=0.863 | Phase=0.717 | 差距 20% |

### 4.2 四個根本問題

#### 問題 1：缺乏時序感知（最關鍵）

**現狀**：
```python
# FourierPositionEncoder
view_mean = view.mean(dim=2)  # 只用平均值，丟失運動信息
```

**問題**：
- 只知道「當前在哪」，不知道「從哪來、往哪去」
- 無法捕捉速度、加速度
- 無法模擬 Doppler effect

**DPATFNet 的解法**：
- Self-Attention (PAL2) 觀察前後幀的位置變化
- 捕捉運動趨勢

#### 問題 2：Phase Residual 約束過緊

**現狀**：
```python
phase_res = tanh(x) * (π/2)  # 限制在 ±π/2
```

**問題**：
- 如果 geometric phase 誤差 > π/2，無法修正
- 模型被迫在有限空間內過擬合

**可能的解法**：
- 放寬到 ±π
- 或移除 `.detach()`，讓梯度回傳修正 warper（風險高）

#### 問題 3：網絡容量可能不足

**現狀**：
- 6 個 FiLM ResBlocks
- 1.60M 參數

**問題**：
- 音訊合成需要捕捉細節，可能需要更深的網絡
- DPATFNet 使用多層 Upsampling + ResStack

#### 問題 4：64 頻段可能過細

**現狀**：
- 64-band FiLM：513 bins → 64 bands
- 每個 band 平均 8 bins

**問題**：
- 頻段太細，每段訓練樣本稀疏
- 可能產生不連續的頻率響應

---

## 5. v4 改進方案（規劃中）

### 5.1 時序感知 Position Encoder

```python
class FourierPositionEncoder(nn.Module):
    def __init__(self, ...):
        # 新增：1D Conv 捕捉運動模式
        self.temporal_conv = nn.Conv1d(7, 7, kernel_size=5, padding=2, groups=7)
    
    def forward(self, view):
        # view: (B, 7, K)
        view_temporal = self.temporal_conv(view)  # 捕捉速度/加速度
        view_mean = view.mean(dim=2)
        view_temporal_mean = view_temporal.mean(dim=2)
        
        # Fourier + raw + temporal
        fourier_feat = self.fourier_encode(view_mean)
        combined = torch.cat([fourier_feat, view_temporal_mean], dim=1)
        return self.mlp(combined)
```

**好處**：
- 捕捉位置變化軌跡（速度、加速度）
- 模擬 Doppler effect
- 類似 DPATFNet 的 Self-Attention，但更輕量

### 5.2 放寬 Phase Residual 約束

```python
# v1-v3: ±π/2
phase_res = self.phase_head(x) * (torch.pi / 2)

# v4: ±π
phase_res = self.phase_head(x) * torch.pi
```

**風險**：可能破壞 geometric prior，需實驗驗證

### 5.3 增加網絡深度

```python
# v1-v3: 6 blocks
self.res_blocks = nn.ModuleList([
    FiLMResBlock(...) for _ in range(6)
])

# v4: 8 blocks
self.res_blocks = nn.ModuleList([
    FiLMResBlock(...) for _ in range(8)
])
```

### 5.4 減少頻段數（可選）

```python
# v1-v3: 64 bands
FiLMLayer(channels, pos_dim, num_bands=64)

# v4: 32 bands
FiLMLayer(channels, pos_dim, num_bands=32)
```

**目標**：Phase < 0.80（超越 Meta large）

---

## 6. 數值穩定性修正

訓練初期遇到 NaN 爆炸，經 4-agent code review 診斷出根因：

| 位置 | 問題 | 修正 |
|------|------|------|
| `src/warping.py:100` | `distance = (x²+y²+z²)**0.5` → sqrt(0) 梯度 ∞ | `sqrt(x²+y²+z² + 1e-8)` |
| `src/losses.py:55` | `mag = (real²+imag²)**0.5` → sqrt(0) 梯度 ∞ | `sqrt(x.clamp(min=1e-12))` |
| `models_geowarp_film.py:243` | `angle(Y_init)` 梯度回傳到 warper | `angle(Y_init).detach()` |
| `models_geowarp_film.py:236` | `mag_out` 可能為 0 | `mag_out.clamp(min=1e-6)` |

**根本原因**：`sqrt(0)` 的梯度是 `1/(2*sqrt(0)) = ∞`，在 `clip_grad_norm_` 之前就污染了 Adam optimizer 的 momentum。

**Meta 原版也有這個 bug**，但他們的 WaveNet 不對 geometric phase 求導，所以沒觸發。

---

## 7. 總結

### 7.1 成就

- ✅ L2 超越 Meta small **22%**（0.155 vs 0.197）
- ✅ Phase 與 Meta small 持平（0.864 vs 0.862）
- ✅ 驗證了 GeometricWarper + FiLM 的可行性
- ✅ 發現並修復 Meta 原版的數值穩定性 bug
- ✅ 證明 Stage 1 加入 phase 監督有效（v3 vs v1）

### 7.2 關鍵發現

1. **IPD 過擬合問題**：
   - Stage 1 只優化 IPD 會導致絕對相位偏移
   - 延長 Stage 1 反而讓 Phase 變差（v2 失敗）
   - 解法：Stage 1 加入 0.1×phase 監督（v3 成功）

2. **架構瓶頸**：
   - Phase 卡在 0.86，無法通過訓練策略突破
   - 缺乏時序建模是最關鍵短板
   - 需要結構性改進（v4）

3. **與 DPATFNet 的差距**：
   - L2 已接近（0.155 vs 0.148，差 5%）
   - Phase 差距 20%（0.864 vs 0.717）
   - 主因：DPATFNet 的 Self-Attention 捕捉運動信息

### 7.3 待改進

- ⚠️ 距離 Meta large 還有 7% 差距（Phase: 0.864 vs 0.804）
- ⚠️ 距離 DPATFNet 還有 20% 差距（Phase: 0.864 vs 0.717）
- ❌ 缺乏時序建模（最關鍵短板）
- ❌ Phase residual 約束可能過緊（±π/2）
- ❌ 網絡深度可能不足（6 blocks）

### 7.4 下一步

**v4 必須實施**（訓練策略已優化到極限）：

1. **時序感知 Position Encoder**（最優先）
   - 加入 1D Conv 捕捉速度/加速度
   - 預期改善：5-10%

2. **放寬 Phase Residual**（±π/2 → ±π）
   - 給模型更大修正空間
   - 預期改善：2-5%

3. **增加網絡深度**（6 → 8 blocks）
   - 提升特徵提取能力
   - 預期改善：2-3%

**v4 目標**：Phase < 0.80（超越 Meta large）

**v5 規劃**（如果 v4 < 0.80）：
- 加入 Self-Attention 模組（類似 DPATFNet PAL2）
- 目標：Phase < 0.75，接近 DPATFNet

**最終目標**：Phase < 0.717，達到 ICASSP 2025 SOTA 水準
