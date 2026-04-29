# Sub-Agent 協作計畫

## 目標
設計一個創新的雙耳音訊合成架構，解決 BinauralTFNet v8.3 的 Phase/ITD 學習問題。

---

## 三個 Sub-Agent

### 1. Binaural Paper Researcher
**檔案**：`ai/agents/binaural_paper_researcher.md`

**任務**：研究雙耳音訊合成領域的學術論文

**輸出**：
- 現有方法的架構分析
- ITD/Phase 學習的成功案例
- 位置編碼的設計方法
- 訓練策略的最佳實踐

**重點關注**：
- BinauralGrad, DPATFNet 等代表性工作
- 2020-2026 年的最新論文
- 如何處理 Phase/ITD 的方法

---

### 2. ML Architecture Researcher
**檔案**：`ai/agents/ml_architecture_researcher.md`

**任務**：研究最新的深度學習架構和技術

**輸出**：
- 最新的 Attention/Transformer 變體
- Diffusion Models 的應用
- State Space Models (Mamba, S4)
- 音訊領域的創新架構
- 條件生成的方法

**重點關注**：
- 2023-2024 年的 SOTA 架構
- 音訊生成領域的突破
- 序列建模的新範式
- Phase 建模的技術

---

### 3. Architecture Synthesizer
**檔案**：`ai/agents/architecture_synthesizer.md`

**任務**：綜合前兩個 agent 的研究，設計新架構

**輸出**：
- 完整的架構設計文件
- 訓練策略
- 實作計畫
- 風險評估

**設計原則**：
- 不能和 DPATFNet 一樣
- 必須能學好 Phase/ITD
- 架構要合理
- 訓練要穩定
- 可實作

---

## 工作流程

```
Step 1: Binaural Paper Researcher 開始研究
        ↓
        輸出：論文分析報告
        ↓
Step 2: ML Architecture Researcher 開始研究
        ↓
        輸出：技術分析報告
        ↓
Step 3: Architecture Synthesizer 綜合設計
        ↓
        輸出：新架構設計文件
        ↓
Step 4: 主 Agent 審查並決定是否實作
```

---

## 當前項目背景

### BinauralTFNet v8.3 的問題

**架構**：
- CommonBranch（時域）：學習「共同部分」
- SpecificBranch（頻域）：學習「左右差異」
- 分三階段訓練

**失敗原因**：
- Stage 2 Phase 改善只有 13.9%（目標 >40%）
- IPD 反而變差（-6.8%）
- CommonBranch 可能沒學到空間感
- SpecificBranch 任務太重

**已嘗試的修改**：
- v8: ITD loss 主導，L2 幾乎不訓練
- v8.2: 增加模型容量（fft_size, tf_channels, tf_blocks）
- v8.3: 重新設計 CommonBranch（BinauralGrad 方式）

**都失敗了。**

### 對比：DPATFNet

**架構**：
- 單階段，端到端
- 純頻域
- DPAB（Cross + Self Attention）
- TFResStack

**我們不想用 DPATFNet 的原因**：
- 希望有創新
- 希望有物理先驗（Warpnet）
- 希望有任務分離（Common + Specific）

---

## 研究重點

### 必須回答的問題：

1. **如何學好 Phase/ITD？**
   - 時域 vs 頻域 vs 混合？
   - 有沒有專門處理 Phase 的技術？
   - 分階段訓練是對的嗎？

2. **位置編碼怎麼做？**
   - DPAB 之外有什麼選擇？
   - Attention 是必須的嗎？
   - 如何更好地利用位置資訊？

3. **任務分離是對的嗎？**
   - Common + Specific 的設計有問題嗎？
   - 還是應該端到端？
   - 或者有其他分離方式（Time + Freq）？

4. **有沒有更好的網路結構？**
   - ResBlock 之外的選擇？
   - Transformer？Diffusion？Mamba？
   - 音訊領域有什麼新突破？

---

## 成功標準

新架構必須：
1. **理論上合理**（有物理/數學依據）
2. **創新**（不同於 DPATFNet 和 v8.3）
3. **可實作**（1-2 週內完成）
4. **預期有效**（能解決 Phase/ITD 問題）

---

## 開始工作

請三個 sub-agent 按順序開始工作：

1. **Binaural Paper Researcher**：先開始，預計 1-2 天
2. **ML Architecture Researcher**：可以同時進行，預計 1-2 天
3. **Architecture Synthesizer**：等前兩個完成後開始，預計 1 天

**總預計時間：2-3 天**

---

## 輸出位置

所有研究結果和設計文件放在：
```
ai/agents/
├── binaural_paper_researcher.md (任務說明)
├── ml_architecture_researcher.md (任務說明)
├── architecture_synthesizer.md (任務說明)
├── research_results/
│   ├── binaural_papers.md (論文研究結果)
│   ├── ml_architectures.md (技術研究結果)
│   └── new_architecture_design.md (最終設計)
└── coordination.md (本文件)
```

---

## 注意事項

1. **不要被現有方法限制**：大膽創新
2. **但要有理論依據**：不能亂設計
3. **考慮實作成本**：太複雜的不要
4. **多提方案**：至少 2-3 個候選架構
5. **風險評估**：每個方案都要評估風險

---

**準備好了嗎？讓我們開始吧！**
