# BinauralTFNet：時頻域分離雙耳音訊合成架構設計

**日期**：2026-04-24
**狀態**：設計階段（未實作）
**基於**：Meta WarpNet + BinauralGrad + DPATFNet 的綜合改進

---

## 一、動機與問題分析

### 1.1 現有架構（Meta WarpNet）的瓶頸

目前使用的 WarpNet 架構（Richard et al., ICLR 2021）：

```
mono → Warpnet（時域 warp）→ WaveNet（時域合成）→ 雙耳輸出
```

**問題 1：L2 和 Phase/IPD 梯度衝突**

WaveNet 在時域生成波形，L2 loss 對波形能量敏感，Phase/IPD loss 對相位敏感。
兩者的梯度方向經常衝突，導致 Stage 3 的 Phase/IPD 改善僅 1.5-1.9%（v6 實驗數據）。

**問題 2：時域模型對 Phase 不敏感**

WaveNet 的 dilated convolution 在時域操作，無法直接感知頻域的相位結構。
Phase/IPD 的改善需要大量 epoch，且容易被 L2 梯度壓制。

**問題 3：單一模型同時學習兩種不同性質的特徵**

- 共同特徵（距離感、殘響、音質）：兩耳相似，適合時域 L2 優化
- 差異特徵（ITD、ILD、Phase 差）：左右耳不同，適合頻域 Phase/IPD 優化

強迫單一模型同時學習這兩種特徵，導致優化困難。

### 1.2 相關工作的啟發

**BinauralGrad（Leng et al., NeurIPS 2022）：**

核心洞見：將雙耳音訊分解為共同部分和差異部分：
```
y_L = ȳ + δ_L
y_R = ȳ + δ_R
```
Stage 1 學共同部分 ȳ，Stage 2 學差異部分 δ。
問題：使用 Diffusion model，推論需要 200 步，速度慢。

**DPATFNet（He et al., ICASSP 2025）：**

核心洞見：在 STFT 頻域處理 Phase/IPD，比時域更直接有效。
使用 Dual Position Attention Block（DPAB）捕捉 Doppler 效果。
結果：Phase-L2 達到 0.717，是目前最佳。
問題：L2 和 Phase loss 仍然一起訓練，存在梯度衝突。

---

## 二、BinauralTFNet 架構設計

### 2.1 核心概念

結合 BinauralGrad 的兩階段分解 + DPATFNet 的頻域處理，用確定性模型取代 Diffusion：

```
Stage 1（時域）：學習兩耳共同部分 → 用 L2 監督
Stage 2（頻域）：學習左右耳差異部分 → 用 Phase + IPD 監督
最終輸出：y_L = y_common + δ_L,  y_R = y_common + δ_R
```

### 2.2 整體架構

```
輸入：
  mono:     B × 1 × T
  position: B × 7 × K  (K = T/400, 120Hz 位置資訊)

┌──────────────────────────────────────────────────────────────┐
│  Stage 1：Common Branch（時域，學共同部分）                    │
│                                                              │
│  Warpnet(mono, position) → warpfield: B × 2 × T             │
│    → warped_L: B × 1 × T  （完整左右耳，給 Stage 2 用）      │
│    → warped_R: B × 1 × T                                    │
│                                                              │
│  warpfield_common = mean(warpfield_L, warpfield_R)          │
│  warped_common = warp(mono, warpfield_common): B × 1 × T    │
│  （在 warpfield 域平均，避免信號域平均的 comb filtering）     │
│                                                              │
│  WaveNet(warped_common, position)                            │
│    → y_common: B × 1 × T                                    │
│                                                              │
│  監督目標：mean(y_L_gt, y_R_gt)                               │
│  Loss：L2(y_common, mean(y_L_gt, y_R_gt))                   │
└──────────────────────────────────────────────────────────────┘
                          ↓ y_common
┌──────────────────────────────────────────────────────────────┐
│  Stage 2：Specific Branch（頻域，學差異部分）                  │
│                                                              │
│  STFT(warped_L) → Y_warp_L: B × F × T_stft × 2             │
│  STFT(warped_R) → Y_warp_R: B × F × T_stft × 2             │
│                                                              │
│  concat([Y_warp_L, Y_warp_R]) → Y_input: B × F × T_stft × 4│
│  （移除 Y_common，避免冗餘；Y_common 只在輸出端做殘差加法）  │
│                                                              │
│  DPAB(position) → cond: B × T × C                           │
│  （Dual Position Attention，捕捉 Doppler 效果）               │
│                                                              │
│  TF-ResStack(Y_input, cond) → Y_delta_L, Y_delta_R          │
│  （頻域殘差學習，直接優化 Phase/IPD）                          │
│                                                              │
│  iSTFT(Y_warp_L + Y_delta_L) → y_L: B × 1 × T              │
│  iSTFT(Y_warp_R + Y_delta_R) → y_R: B × 1 × T              │
│  （從 warped_L/R 出發，保留 Warpnet 的 ITD；                 │
│   delta 只修正 Phase 細節，分工清晰）                         │
│                                                              │
│  監督目標：y_L_gt, y_R_gt                                    │
│  Loss：Phase(y_L, y_L_gt) + Phase(y_R, y_R_gt)              │
│        + IPD(y_L, y_R, y_L_gt, y_R_gt)                      │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│  最終輸出                                                     │
│  output = (y_L, y_R): B × 2 × T                             │
│  Loss（Stage 3 fine-tune）：                                  │
│    L2(y_L, y_L_gt) + L2(y_R, y_R_gt)                        │
│    + Phase(y_L, y_L_gt) + Phase(y_R, y_R_gt)                │
│    + IPD(y_L, y_R, y_L_gt, y_R_gt)                          │
└──────────────────────────────────────────────────────────────┘
```

### 2.3 Stage 1：Common Branch 詳細設計

與現有 WarpNet 架構相同，但輸出目標改為兩耳平均：

```python
class CommonBranch(nn.Module):
    def __init__(self, view_dim=7, warpnet_layers=4, warpnet_channels=64,
                 wavenet_blocks=3, layers_per_block=10, wavenet_channels=64):
        self.warper = Warpnet(warpnet_layers, warpnet_channels)
        self.input = nn.Conv1d(2, wavenet_channels, kernel_size=1)
        self.hyperconv_wavenet = HyperConvWavenet(view_dim, wavenet_channels,
                                                   wavenet_blocks, layers_per_block)
        self.output_net = WaveoutBlock(wavenet_channels)

    def forward(self, mono, view):
        # 計算完整 warpfield（geometric + neural），shape: B × 2 × T
        geo_warpfield = self.warper.geometric_warper._warpfield(view, mono.shape[-1])
        neural_warpfield = self.warper.neural_warpfield(view, mono.shape[-1])
        warpfield = geo_warpfield + neural_warpfield
        warpfield = -F.relu(-warpfield)  # 確保 causality

        # 完整左右耳 warp（給 Stage 2 使用）
        warped = self.warper.neural_warper(th.cat([mono, mono], dim=1), warpfield)  # B × 2 × T

        # 選項 B：在 warpfield 域平均，再 warp，避免信號域平均造成的 comb filtering
        # 物理意義：聲源到頭部中心的平均延遲
        warpfield_common = warpfield.mean(dim=1, keepdim=True).expand(-1, 2, -1)  # B × 2 × T
        warped_common = self.warper.neural_warper(
            th.cat([mono, mono], dim=1), warpfield_common
        )[:, 0:1, :]  # B × 1 × T

        # WaveNet 合成共同部分
        x = self.input(warped_common.repeat(1, 2, 1))
        _, skips = self.hyperconv_wavenet(x, view)
        y_common = self.output_net(skips[-1])  # B × 1 × T

        return y_common, warped  # warped 傳給 Stage 2 使用
```

**監督目標：**
```python
y_common_gt = (y_L_gt + y_R_gt) / 2  # 兩耳平均
loss_stage1 = L2Loss(y_common, y_common_gt)
```

### 2.4 Stage 2：Specific Branch 詳細設計

#### 2.4.1 STFT 特徵提取

```python
# STFT 參數（參考 DPATFNet）
fft_size = 1024
hop_size = 64
win_length = 1024

# 輸入特徵
Y_common = STFT(y_common)    # B × F × T_stft × 2 (real/imag)
Y_warp_L = STFT(warped_L)    # B × F × T_stft × 2
Y_warp_R = STFT(warped_R)    # B × F × T_stft × 2

# 拼接：提供共同部分和幾何 warp 的資訊
Y_input = concat([Y_common, Y_warp_L, Y_warp_R])  # B × F × T_stft × 6
```

#### 2.4.2 Dual Position Attention Block（DPAB）

直接採用 DPATFNet 的設計：

```
position (B × 7 × K)
    ↓
PAL1（Cross-Attention）：融合位置和方向資訊
    → cond1 (B × T × 3D)
    ↓
PAL2（Self-Attention + Mask）：捕捉位置變化時刻（Doppler）
    → cond2 (B × T × 3D)
    ↓
Concat(Conv(cond2), position) → cond (B × 10T)
```

**PAL1（Cross-Attention）：**
- Query：位置資訊 P（sinusoidal encoding）
- Key/Value：方向資訊 O（sinusoidal encoding）
- 輸出：融合位置和方向的條件特徵 cond1

**PAL2（Masked Self-Attention）：**
- 對 cond1 做 self-attention
- Mask：每個時刻只關注前後一個時刻（t-1, t+1）
- 目的：捕捉位置變化（Doppler 效果）

#### 2.4.3 TF-ResStack

在頻域做殘差學習，輸出左右耳的差異部分：

```python
class TFResStack(nn.Module):
    def __init__(self, in_channels, channels, num_layers=3):
        # 每個 ResStack 包含 3 個 dilated conv
        # 條件注入：cond 通過 Conv1 + ReLU 後加到每個 dilated conv 後面

    def forward(self, Y_input, cond):
        # Y_input: B × F × T_stft × 6
        # cond: B × 10T（位置條件）
        # 輸出: Y_delta_L, Y_delta_R (B × F × T_stft × 2)
```

**上採樣設計（參考 DPATFNet）：**
```
Y_input → Conv1 → ResStack → ConvTranspose(×4) → ResStack
                           → ConvTranspose(×4) → ResStack
                           → ConvTranspose(×2) → ResStack
                           → ConvTranspose(×2) → ResStack
                           → Conv1 → Tanh → Y_delta_L, Y_delta_R
```

#### 2.4.4 輸出合成

```python
# 在頻域加上差異部分
Y_L = Y_common + Y_delta_L
Y_R = Y_common + Y_delta_R

# 轉回時域
y_L = iSTFT(Y_L)  # B × 1 × T
y_R = iSTFT(Y_R)  # B × 1 × T
```

---

## 三、訓練策略

### 3.1 三階段訓練

| 階段 | Epochs | 訓練參數 | Loss | 目標 |
|------|--------|---------|------|------|
| Stage 1 | 0–80 | Common Branch | L2 | 學好共同部分 |
| Stage 2 | 80–160 | Specific Branch（凍結 Common）| Phase + IPD | 學好差異部分 |
| Stage 3 | 160–180 | 全部 | L2(×1) + Phase + IPD | 端對端 fine-tune（修正 iSTFT 誤差，監控 Phase-L2 不退步）|

**Stage 2 可以凍結 Common Branch 的原因：**

Stage 2 有自己獨立的參數（TF-ResStack + DPAB），Phase/IPD 的梯度只需要更新 Stage 2 的參數，不需要流回 Stage 1。這與之前失敗的「凍結 WaveNet」不同——之前是同一個網路中凍結中間層，切斷了梯度路徑；現在是兩個獨立的網路，Stage 2 有完整的梯度路徑。

### 3.2 Loss 函數設計

```python
# 各 loss 的真實數值範圍（來自 v7 實驗）：
#   L2    ~ 0.0001   （比 Phase/IPD 小約 8000 倍）
#   Phase ~ 0.80
#   IPD   ~ 0.62

# Stage 1：單一 loss，不需要 weight
loss_s1 = L2Loss(y_common, mean(y_L_gt, y_R_gt))

# Stage 2：Phase + IPD 量級相近，weight 設 1.0 即可
# 不使用小 weight（那是為了配合 L2 量級設計的）
loss_s2 = PhaseLoss(y_L, y_L_gt) * 1.0 \
        + PhaseLoss(y_R, y_R_gt) * 1.0 \
        + IPDLoss(y_L, y_R, y_L_gt, y_R_gt) * 1.0

# Stage 3（端對端 fine-tune，只跑 10–20 epoch 修正 iSTFT 重建誤差）
# L2 weight=100 讓 L2 貢獻約 Phase 的 1/8，有梯度但不壓制 Phase
# accumulated_loss ≈ 0.02 + 0.02 + 0.80 + 0.80 + 0.62 = 2.26
# L2 佔 ~1.8%，Phase 佔 ~70%，IPD 佔 ~27%
loss_s3 = L2Loss(y_L, y_L_gt) * 100.0 \
        + L2Loss(y_R, y_R_gt) * 100.0 \
        + PhaseLoss(y_L, y_L_gt) * 1.0 \
        + PhaseLoss(y_R, y_R_gt) * 1.0 \
        + IPDLoss(y_L, y_R, y_L_gt, y_R_gt) * 1.0
```

### 3.3 LR 調度

使用 CosineAnnealingWarmRestarts（v7 已驗證 Stage 1 效果良好）：

```python
CosineAnnealingWarmRestarts(T_0=80, eta_min=5e-6)
```

Stage 切換時限制 LR ≤ 0.0003，避免 warm restart 造成震盪。

---

## 四、與現有架構的比較

### 4.1 架構對比

| 項目 | 現有 WarpNet | BinauralGrad | DPATFNet | BinauralTFNet（新）|
|------|------------|-------------|---------|------------------|
| 時域處理 | WaveNet | Diffusion | TDW only | WaveNet（Stage 1）|
| 頻域處理 | 無 | 無 | TF-ResStack | TF-ResStack（Stage 2）|
| 位置處理 | HyperConv | FC | DPAB | DPAB（Stage 2）|
| 兩階段分解 | 無 | ✅ | 無 | ✅ |
| 推論速度 | 快 | 慢（200步）| 快 | 快 |
| 梯度衝突 | 有 | 無（分開訓練）| 有 | 無（Stage 分離）|

### 4.2 預期性能

基於三篇論文的結果推估：

| 指標 | 現有 v6 | BinauralGrad | DPATFNet | BinauralTFNet 預期 |
|------|---------|-------------|---------|------------------|
| Wave-L2 (×10³) | 0.157 | **0.128** | 0.148 | ~0.130 |
| Amplitude-L2 | **0.034** | 0.030 | 0.037 | ~0.030 |
| Phase-L2 | 0.800 | 0.837 | **0.717** | ~0.730 |
| IPD-L2 | 0.630* | - | **1.020** | - |
| 推論速度 | 快 | 慢 | 快 | 快 |

*不同計算方式，無法直接比較

**預期優勢：**
- Wave-L2 接近 BinauralGrad（Stage 1 專注 L2）
- Phase-L2 接近 DPATFNet（Stage 2 在頻域直接優化）
- 推論速度比 BinauralGrad 快（無 Diffusion 迭代）

---

## 五、實作計畫

### 5.1 新增檔案

```
src/
├── models_v2.py          # BinauralTFNet 模型定義
│   ├── CommonBranch      # Stage 1：時域共同部分
│   ├── DPAB              # Dual Position Attention Block
│   ├── TFResStack        # 頻域殘差堆疊
│   └── SpecificBranch    # Stage 2：頻域差異部分
├── trainer_v8.py         # 三階段訓練邏輯
└── losses.py             # 已有，無需修改
train_v8.py               # 訓練腳本
```

### 5.2 實作順序

1. **實作 CommonBranch**（基於現有 WarpNet，改輸出目標）
2. **實作 DPAB**（參考 DPATFNet 論文）
3. **實作 TFResStack**（頻域殘差，參考 MelGAN）
4. **實作 SpecificBranch**（整合 DPAB + TFResStack）
5. **實作 BinauralTFNet**（整合兩個 Branch）
6. **實作 trainer_v8**（三階段訓練邏輯）
7. **驗證 Stage 1**（確認 y_common 品質）
8. **驗證 Stage 2**（確認 Phase/IPD 改善）
9. **端對端 fine-tune**

### 5.3 關鍵超參數

| 超參數 | 值 | 說明 |
|--------|-----|------|
| STFT fft_size | 1024 | 參考 DPATFNet |
| STFT hop_size | 64 | 參考 DPATFNet |
| DPAB attention_dim | 64 | 每個 channel 的維度 |
| TFResStack layers | 3 | 每個 ResStack 的 dilated conv 數 |
| ConvTranspose 上採樣 | [4,4,2,2] | 參考 DPATFNet，總上採樣 64x |
| Stage 1 WaveNet blocks | 3 | 與現有架構相同 |

---

## 六、風險評估

### 6.1 主要風險

**風險 1：Stage 1 的 y_common 品質不足**

BinauralGrad 論文指出 Stage 1 是整個框架的瓶頸。如果 y_common 品質差，Stage 2 的上限也受限。

**應對：** Stage 1 使用現有 WarpNet 架構（已驗證有效），監督目標改為 `mean(y_L_gt, y_R_gt)`，這比原本的雙耳輸出更容易學習。

**風險 2：Stage 2 輸入特徵冗餘**

原設計 `concat([Y_common, Y_warp_L, Y_warp_R])` 中，`Y_common` 是由 `warpfield_common = mean(warpfield_L, warpfield_R)` 生成的，與 `Y_warp_L`、`Y_warp_R` 高度相關，提供的額外資訊有限，反而增加模型參數量和記憶體。

**應對：** Stage 2 輸入改為只用 `Y_warp_L` 和 `Y_warp_R`（4 channels），讓 Stage 2 自己從左右耳差異中學習 delta，`Y_common` 只在輸出端做殘差加法：

```python
Y_input = concat([Y_warp_L, Y_warp_R])  # B × F × T_stft × 4

# 輸出端
Y_L = Y_common + Y_delta_L
Y_R = Y_common + Y_delta_R
```

**風險 3：Stage 2 的頻域特徵記憶體過大**

STFT 後的特徵維度為 `B × F × T_stft × 4`（F=513, T_stft≈T/64），以 batch_size=8、T=48000 估算約 5 GB，加上 Stage 1 的 WaveNet 和梯度，GPU 記憶體壓力大。

**應對：** 初步驗證時先用 `hop_size=128`（T_stft≈T/128，記憶體減半）或 `fft_size=512`（F=257），確認訓練流程正確後再調回 DPATFNet 的參數。TF-ResStack 的 F 維度作為 channel 處理（flatten），不使用 Conv2d。

**風險 4：Stage 3 的 L2 weight 過高導致 Phase 退步**

Stage 3 原設計 L2 weight=10.0、Phase weight=0.1，比例 100:1，與 v6 遇到的梯度衝突問題相同，可能讓 Stage 2 學到的 Phase/IPD 成果在 fine-tune 後退步。

**應對：** Stage 3 的 L2 weight 降低，並縮短 fine-tune epoch 數：

```python
# Stage 3（端對端 fine-tune）—— 修正後
# L2 原始值 ~0.0001，weight=100 讓其貢獻約 Phase 的 1/8，有梯度但不壓制 Phase
loss_s3 = L2Loss(y_L, y_L_gt) * 100.0 \
        + L2Loss(y_R, y_R_gt) * 100.0 \
        + PhaseLoss(y_L, y_L_gt) * 1.0 \
        + PhaseLoss(y_R, y_R_gt) * 1.0 \
        + IPDLoss(y_L, y_R, y_L_gt, y_R_gt) * 1.0
```

Stage 3 epoch 數從 40 縮短為 10–20，目的只是修正 iSTFT 重建誤差，不應大幅改變 Stage 2 的 Phase 結果。訓練時監控 Phase-L2，若相比 Stage 2 退步超過 5% 則提前停止。

### 6.2 與現有工作的差異性

本架構的創新點：
1. **BinauralGrad 的分解概念 + 確定性模型**：避免 Diffusion 的推論慢問題
2. **DPATFNet 的頻域處理 + 梯度分離訓練**：避免 L2/Phase 梯度衝突
3. **Stage 2 可以真正凍結 Stage 1**：因為兩個 Branch 是獨立的網路

---

## 七、參考文獻

1. **Richard et al. (2021).** "Neural Synthesis of Binaural Speech from Mono Audio." *ICLR 2021.*
   - 本研究的基礎架構（WarpNet）

2. **Leng et al. (2022).** "BinauralGrad: A Two-Stage Conditional Diffusion Probabilistic Model for Binaural Audio Synthesis." *NeurIPS 2022.*
   - 兩階段分解概念的來源
   - Wave-L2: 0.128（目前最佳）

3. **He et al. (2025).** "Dual Position Attention Time-Frequency Network for Binaural Audio Synthesis." *ICASSP 2025.*
   - DOI: 10.1109/ICASSP49660.2025.10890826
   - 頻域處理和 DPAB 的來源
   - Phase-L2: 0.717（目前最佳）

4. **Kumar et al. (2019).** "MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis." *NeurIPS 2019.*
   - TF-ResStack 和 ConvTranspose 上採樣設計的參考

5. **Vaswani et al. (2017).** "Attention is All You Need." *NeurIPS 2017.*
   - DPAB 中 Cross-Attention 和 Self-Attention 的基礎

6. **Loshchilov & Hutter (2017).** "SGDR: Stochastic Gradient Descent with Warm Restarts." *ICLR 2017.*
   - LR 調度策略

7. **Yu et al. (2020).** "Gradient Surgery for Multi-Task Learning." *NeurIPS 2020.*
   - 梯度衝突問題的理論依據
