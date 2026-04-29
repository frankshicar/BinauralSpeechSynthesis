# Binaural Audio Paper Research Agent

## 任務
研究雙耳音訊合成（Binaural Audio Synthesis）領域的學術論文，分析現有方法的架構、優缺點和創新點。

## 研究範圍

### 1. 經典方法
- HRTF-based methods
- Geometric acoustic methods
- Signal processing approaches

### 2. 深度學習方法
- End-to-end neural networks
- GAN-based approaches
- Diffusion models
- Transformer-based methods

### 3. 重點論文
請特別關注以下論文及其引用：
- **BinauralGrad** (Diffusion-based, 2-stage)
- **DPATFNet** (DPAB + TF-domain)
- **Neural HRTF synthesis**
- **Spatial audio generation**

## 研究重點

### 對每篇論文，請分析：

1. **架構設計**
   - 時域 vs 頻域 vs 混合
   - 單階段 vs 多階段
   - 網路結構（CNN, RNN, Transformer, Diffusion, etc.）

2. **位置編碼方式**
   - 如何處理 3D 位置 (x, y, z)
   - 如何處理方向（orientation）
   - 是否有 attention mechanism

3. **訓練策略**
   - Loss functions (L2, Phase, IPD, Magnitude, etc.)
   - 分階段訓練 vs 端到端
   - 數據增強方法

4. **性能指標**
   - 客觀指標（L2, Phase error, IPD error）
   - 主觀指標（MOS, listening tests）
   - 泛化能力（unseen positions, dynamic scenes）

5. **創新點**
   - 與前人工作的差異
   - 解決了什麼問題
   - 有什麼局限性

## 輸出格式

請將研究結果整理成：

```markdown
## 論文名稱
**作者**：xxx
**年份**：202x
**會議/期刊**：xxx

### 核心架構
[簡要描述架構圖]

### 關鍵技術
- 技術點 1
- 技術點 2

### 優點
- 優點 1
- 優點 2

### 缺點/局限
- 缺點 1
- 缺點 2

### 性能
- 指標 1: xxx
- 指標 2: xxx

### 可借鑑之處
[對我們的項目有什麼啟發]
```

## 研究目標

找出：
1. **ITD/Phase 學習的最佳方法**（我們目前的痛點）
2. **位置編碼的創新設計**（DPAB 之外的選擇）
3. **時域-頻域混合架構**（是否有先例）
4. **分階段訓練的成功案例**（如何避免我們遇到的問題）

## 開始研究

請從以下關鍵詞開始搜索：
- "binaural audio synthesis"
- "spatial audio generation"
- "HRTF neural network"
- "3D audio deep learning"
- "binaural speech synthesis"
- "interaural time difference learning"

重點關注 **2020-2026 年的最新論文**。
