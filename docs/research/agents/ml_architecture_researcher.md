# Modern ML Architecture Research Agent

## 任務
研究最新的深度學習架構和技術，找出可應用於雙耳音訊合成的創新方法。

## 研究領域

### 1. Attention Mechanisms
- **Multi-head attention** variants
- **Cross-attention** for conditioning
- **Sparse attention** for efficiency
- **Rotary Position Embedding (RoPE)**
- **ALiBi** (Attention with Linear Biases)

### 2. Transformer Architectures
- **Vision Transformer (ViT)** → Audio Transformer
- **Swin Transformer** (shifted windows)
- **Perceiver** (cross-attention to latent space)
- **Conformer** (Convolution + Transformer)

### 3. Diffusion Models
- **DDPM** (Denoising Diffusion Probabilistic Models)
- **DDIM** (faster sampling)
- **Latent Diffusion** (efficiency)
- **Conditional Diffusion** (class/position conditioning)

### 4. State Space Models (SSMs)
- **Mamba** (selective state space)
- **S4** (Structured State Space)
- 比 Transformer 更高效的序列建模

### 5. Hybrid Architectures
- **ConvNeXt** (modern CNN)
- **MobileViT** (CNN + Transformer)
- **MetaFormer** (general architecture framework)

### 6. Audio-Specific Architectures
- **WaveNet** variants
- **HiFi-GAN** (vocoder)
- **EnCodec** (neural audio codec)
- **AudioLM** (audio language model)
- **MusicGen** (music generation)

### 7. Conditioning Mechanisms
- **FiLM** (Feature-wise Linear Modulation)
- **AdaIN** (Adaptive Instance Normalization)
- **Cross-attention conditioning**
- **Hypernetworks** (generate weights dynamically)

### 8. Multi-Task Learning
- **Task-specific heads**
- **Shared backbone + specialized branches**
- **Gradient balancing** techniques

## 研究重點

### 對每種技術，請分析：

1. **核心原理**
   - 如何工作
   - 數學公式（簡要）

2. **優勢**
   - 比傳統方法好在哪
   - 適合什麼場景

3. **劣勢**
   - 計算成本
   - 訓練難度
   - 局限性

4. **在音訊領域的應用**
   - 是否有成功案例
   - 如何適配音訊數據

5. **可應用於雙耳合成的方式**
   - 如何處理位置資訊
   - 如何處理時域/頻域
   - 如何學習 Phase/ITD

## 輸出格式

```markdown
## 技術名稱
**類別**：Attention / Transformer / Diffusion / etc.
**提出時間**：202x
**代表論文**：xxx

### 核心原理
[簡要說明]

### 優勢
- 優勢 1
- 優勢 2

### 劣勢
- 劣勢 1
- 劣勢 2

### 音訊應用案例
- 案例 1: [論文名稱]
- 案例 2: [論文名稱]

### 應用於雙耳合成的可能性
[具體建議]

### 實作難度
- 複雜度：低/中/高
- 訓練成本：低/中/高
- 是否有現成實作：是/否
```

## 研究目標

找出：
1. **比 ResBlock 更強的特徵提取器**
2. **比 SimpleDPAB 更好的位置編碼**
3. **比分階段訓練更穩定的方法**
4. **專門處理 Phase 的技術**
5. **時域-頻域融合的創新方法**

## 重點關注

### 對我們項目特別有用的技術：
- **處理序列的高效方法**（音訊是長序列）
- **條件生成**（位置 → 音訊）
- **多模態融合**（時域 + 頻域）
- **Phase 建模**（目前的痛點）

## 開始研究

請從以下方向開始：
1. **2023-2024 年的 SOTA 架構**（最新技術）
2. **音訊生成領域的突破**（TTS, music generation, audio synthesis）
3. **條件生成的創新方法**（如何更好地利用條件資訊）
4. **序列建模的新範式**（Mamba, SSM 等 Transformer 替代品）
