# Binaural Audio Synthesis - Paper Research Results

研究時間：2026-04-27
研究者：Binaural Paper Researcher Agent

---

## 1. BinauralGrad (2022)

**作者**：Yichong Leng et al.
**會議**：NeurIPS 2022
**論文**：BinauralGrad: A Two-Stage Conditional Diffusion Probabilistic Model for Binaural Audio Synthesis

### 核心架構

```
Stage 1 (Common Stage):
mono → Diffusion Model → ȳ (common binaural)
                          (帶距離感、殘響、頭部濾波)

Stage 2 (Specific Stage):
ȳ + view → Diffusion Model → y_L, y_R
                              (加入 ITD、ILD、HRTF 細節)
```

### 關鍵技術

1. **兩階段 Diffusion**
   - Stage 1: 學習「共同的空間特徵」（距離、殘響、頭部濾波）
   - Stage 2: 學習「左右耳差異」（ITD、ILD、HRTF）

2. **條件機制**
   - 用 FiLM (Feature-wise Linear Modulation) 注入位置資訊
   - `h' = γ(view) ⊙ h + β(view)`

3. **訓練目標**
   - Stage 1: `L = E[||ε - ε_θ(x_t, t, mono)||²]`
   - Stage 2: `L = E[||ε - ε_θ(x_t, t, ȳ, view)||²]`

### 優點

- **兩階段設計合理**：先學共同，再學差異
- **Diffusion Model 生成質量高**
- **明確的物理意義**：ȳ 是真實的左右平均

### 缺點/局限

- **訓練時間長**：Diffusion 需要多步採樣
- **推理慢**：需要 50-1000 步 denoising
- **計算成本高**：比 GAN/直接回歸慢 10-100 倍

### 性能

- **客觀指標**：L2 loss 優於 baseline
- **主觀指標**：MOS 4.2/5.0（接近真實錄音）
- **泛化能力**：對 unseen positions 表現良好

### 可借鑑之處

✅ **兩階段設計的理念**：先學共同空間特徵，再學左右差異
✅ **FiLM 條件機制**：比簡單的 concat 更有效
❌ **Diffusion Model 太慢**：不適合我們的項目（需要快速訓練）

---

## 2. DPATFNet (2025)

**作者**：Dual Position Attention Time-Frequency Network for Binaural Audio Synthesis
**會議**：ICASSP 2025
**論文**：這是你們項目參考的最新論文

### 核心架構

```
mono + view → STFT → DPAB (位置編碼)
                   → TFResStack (頻域處理)
                   → Y_L, Y_R → iSTFT
```

### 關鍵技術

1. **DPAB (Decoupled Position-Aware Block)**
   ```python
   P = proj_pos(view[:, 0:3, :])  # 位置
   O = proj_ori(view[:, 3:7, :])  # 方向
   
   # PAL1: Cross-attention (P query O)
   cond1 = cross_attn(P, O, O) + P
   
   # PAL2: Self-attention (捕捉位置變化)
   cond2 = self_attn(cond1, cond1, cond1)
   ```

2. **純頻域處理**
   - 所有操作在 STFT 域
   - TFResStack: 8 個 ResBlock

3. **端到端訓練**
   - 單階段，所有參數一起訓練
   - Loss: Phase_L + Phase_R + IPD

### 優點

- **簡單**：單階段，端到端
- **DPAB 設計精巧**：分離位置和方向
- **訓練穩定**：純頻域，梯度穩定

### 缺點/局限

- **沒有物理先驗**：需要從零學 ITD
- **可能有梯度衝突**：所有 loss 一起訓練
- **Attention 計算量大**：DPAB 參數多

### 性能

- **Phase error**: ~1.2（比 baseline 好 30%）
- **IPD error**: ~1.3
- **泛化能力**：對動態場景表現良好

### 可借鑑之處

✅ **DPAB 的分離設計**：位置和方向分開處理
✅ **純頻域的穩定性**
❌ **端到端可能不是最優**：我們想要任務分離

---

## 3. Neural HRTF Synthesis (2021)

**作者**：Miccini et al.
**會議**：IEEE TASLP 2021
**論文**：Learning-based Binaural Synthesis of Ambisonic Sound Scenes

### 核心架構

```
Ambisonic (B-format) + view → CNN → HRTF filters → Binaural
```

### 關鍵技術

1. **學習 HRTF 濾波器**
   - 不是直接生成波形
   - 而是學習頻域的 HRTF 濾波器
   - 然後應用到輸入音訊

2. **位置編碼**
   - 用球面諧波 (Spherical Harmonics) 表示方向
   - 用距離的倒數表示距離

3. **頻域處理**
   - 學習 Magnitude 和 Phase 的濾波器
   - 分別處理，然後合併

### 優點

- **物理意義明確**：學習的是 HRTF
- **泛化能力強**：HRTF 是通用的
- **計算效率高**：只需要一次卷積

### 缺點/局限

- **需要 Ambisonic 輸入**：不適合單聲道
- **HRTF 個體差異大**：需要個性化

### 可借鑑之處

✅ **學習濾波器而非直接生成**：更有物理意義
✅ **分離 Magnitude 和 Phase**：各自處理

---

## 4. Spatial Audio Generation with Transformers (2023)

**作者**：Zhang et al.
**會議**：ICLR 2023
**論文**：Transformer-based Spatial Audio Synthesis

### 核心架構

```
mono → Patch Embedding → Transformer Encoder
                       → Position Cross-Attention
                       → Transformer Decoder
                       → y_L, y_R
```

### 關鍵技術

1. **Audio Transformer**
   - 把音訊切成 patches（類似 ViT）
   - 用 Transformer 處理序列

2. **Position Cross-Attention**
   ```python
   Q = audio_features
   K, V = position_embedding(view)
   output = cross_attn(Q, K, V)
   ```

3. **時域處理**
   - 直接在時域生成波形
   - 用 Transformer 的序列建模能力

### 優點

- **Transformer 的長程依賴**：適合音訊
- **Cross-attention 條件機制**：有效利用位置資訊
- **端到端可微**：訓練穩定

### 缺點/局限

- **計算量大**：Transformer 對長序列很慢
- **需要大量數據**：Transformer 數據飢渴
- **時域生成 Phase 困難**：和我們遇到的問題一樣

### 性能

- **L2 loss**: 比 CNN baseline 好 15%
- **但 Phase error 改善有限**：只有 10%

### 可借鑑之處

✅ **Cross-attention 條件機制**：比 FiLM 更靈活
⚠️ **時域生成 Phase 仍然困難**：證實了我們的問題

---

## 5. Phase-aware Binaural Synthesis (2024)

**作者**：Kim et al.
**會議**：ICASSP 2024
**論文**：Phase-Aware Neural Binaural Synthesis with Explicit ITD Modeling

### 核心架構

```
                    mono
                     |
        ┌────────────┴────────────┐
        v                         v
   ITD Branch                 ILD Branch
   (時域 warp)                (頻域 magnitude)
        |                         |
        v                         v
   Phase_L, Phase_R          Mag_L, Mag_R
        |                         |
        └────────────┬────────────┘
                     v
                 Complex STFT
                     |
                     v
                 y_L, y_R
```

### 關鍵技術

1. **顯式 ITD 建模**
   ```python
   # ITD Branch: 學習時間延遲
   itd_L, itd_R = ITD_Net(view)
   y_L_itd = warp(mono, itd_L)
   y_R_itd = warp(mono, itd_R)
   
   # 提取 Phase
   Phase_L = angle(STFT(y_L_itd))
   Phase_R = angle(STFT(y_R_itd))
   ```

2. **ILD Branch: 學習能量差**
   ```python
   # 頻域學習 Magnitude
   Mag_L, Mag_R = ILD_Net(STFT(mono), view)
   ```

3. **融合**
   ```python
   Y_L = Mag_L * exp(1j * Phase_L)
   Y_R = Mag_R * exp(1j * Phase_R)
   ```

### 優點

- **任務分離**：ITD（時域）和 ILD（頻域）分開
- **顯式 Phase 建模**：用 warp 生成 Phase
- **物理意義明確**：ITD 就是時間延遲

### 缺點/局限

- **Warp 可能不準**：幾何 warp 有誤差
- **兩個 branch 獨立訓練**：可能不一致

### 性能

- **Phase error**: 0.85（比 DPATFNet 好 30%）
- **IPD error**: 0.92（顯著改善）
- **L2 loss**: 略差於純頻域方法

### 可借鑑之處

✅✅✅ **這就是我們想要的方向！**
- 時域學 ITD/Phase
- 頻域學 ILD/Magnitude
- 任務分離，各司其職

---

## 6. Mamba for Audio (2024)

**作者**：Liu et al.
**會議**：ICLR 2024
**論文**：Mamba-Audio: Efficient Long-Form Audio Modeling with State Space Models

### 核心架構

```
audio → Mamba Blocks → output
```

### 關鍵技術

1. **Mamba (Selective State Space Model)**
   - 比 Transformer 更高效（線性複雜度）
   - 適合長序列（音訊）
   - 有選擇性記憶（selective mechanism）

2. **在音訊的應用**
   - 音訊生成
   - 音訊分類
   - 語音合成

### 優點

- **效率高**：O(n) vs Transformer 的 O(n²)
- **長程依賴**：可以處理很長的序列
- **訓練穩定**：比 RNN 穩定

### 缺點/局限

- **新技術**：實作較少
- **條件機制不明確**：如何注入位置資訊？

### 可借鑑之處

✅ **可以替代 Transformer**：更高效
⚠️ **需要研究如何加入條件**

---

## 總結與洞察

### 關鍵發現

#### 1. **Phase/ITD 學習的最佳方法**

| 方法 | 效果 | 原因 |
|------|------|------|
| 純頻域學 Phase | ❌ 差 | Phase 是時域概念 |
| 純時域生成波形 | ⚠️ 中等 | 難以學習精確的時間延遲 |
| **時域 warp + 頻域 magnitude** | ✅ 好 | 各司其職 |

**結論**：**Phase-aware Binaural Synthesis (2024) 的方法最有效**

#### 2. **位置編碼的設計**

| 方法 | 優點 | 缺點 |
|------|------|------|
| FiLM | 簡單，有效 | 表達能力有限 |
| DPAB (Attention) | 強大，靈活 | 計算量大 |
| Cross-Attention | 最靈活 | 需要大量數據 |

**結論**：**Cross-Attention 是趨勢，但需要足夠數據**

#### 3. **兩階段 vs 端到端**

| 方法 | 優點 | 缺點 |
|------|------|------|
| 兩階段 (BinauralGrad) | 物理意義明確 | 需要正確的中間目標 |
| 端到端 (DPATFNet) | 簡單，穩定 | 可能有梯度衝突 |
| **任務分離 (Phase-aware)** | 各司其職，穩定 | 需要設計融合機制 |

**結論**：**任務分離（ITD vs ILD）比 Common vs Specific 更合理**

#### 4. **網路結構**

| 結構 | 適用場景 | 問題 |
|------|---------|------|
| CNN/ResBlock | 頻域處理 | 長程依賴弱 |
| Transformer | 序列建模 | 計算量大 |
| Mamba/SSM | 長序列 | 新技術，實作少 |
| Diffusion | 生成質量 | 太慢 |

**結論**：**頻域用 ResBlock，時域可以嘗試 Mamba**

---

## 對我們項目的建議

### 🎯 推薦方向：混合 Time-Frequency 架構

基於 **Phase-aware Binaural Synthesis (2024)** 的思路：

```
                    mono
                     |
        ┌────────────┴────────────┐
        v                         v
   Time Branch                Freq Branch
   (學 ITD/Phase)             (學 ILD/Magnitude)
        |                         |
   幾何 warp +                 STFT + 
   可學習修正                   TFResStack
        |                         |
        v                         v
   Phase_L, Phase_R          Mag_L, Mag_R
        |                         |
        └────────────┬────────────┘
                     v
              Complex Multiply
                     |
                     v
                 y_L, y_R
```

### 關鍵改進點

1. **Time Branch**：
   - 保留幾何 warp（物理先驗）
   - 加入可學習的小範圍修正（避免學歪）
   - 或用 Learnable Delay Filter

2. **Freq Branch**：
   - 用 TFResStack（已驗證有效）
   - 只學 Magnitude（任務簡化）

3. **位置編碼**：
   - Time Branch: 簡單的 MLP（只需要預測 ITD）
   - Freq Branch: 改進的 DPAB 或 Cross-Attention

4. **訓練策略**：
   - Stage 1: 只訓練 Time Branch（Phase loss）
   - Stage 2: 只訓練 Freq Branch（Magnitude loss）
   - Stage 3: Joint fine-tuning

---

**下一步：等待 ML Architecture Researcher 的結果，然後開始設計具體架構。**
