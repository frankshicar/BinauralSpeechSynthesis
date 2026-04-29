# PGCN 完整報告索引

**PhysicsGuidedComplexNet (PGCN)** - 雙耳音訊合成架構提案

**日期**：2026-04-27  
**狀態**：設計階段，待實作  
**預期完成時間**：10 天

---

## 報告結構

本報告分為 4 個文件，涵蓋完整的設計、實作和評估計畫：

### 1. [架構提案](./PGCN_Architecture_Proposal.md)
- 執行摘要
- 問題診斷與失敗案例分析
- PGCN 整體架構設計
- 參數配置

**關鍵內容**：
- 5 個核心創新點
- 完整架構圖
- 與失敗案例的對比

---

### 2. [模組詳細設計](./PGCN_Module_Design.md)
- FourierPositionEncoder（0.1° 精度）
- ComplexDualPathBlock（64-band FiLM）
- PhysicsConstrainedHead（ITD/ILD 先驗）
- MinimumPhaseEnforcer（因果性約束）
- 完整模型代碼框架

**關鍵內容**：
- 每個模組的 PyTorch 實作框架
- 理論依據和參考文獻
- 參數量估算（~24M）

---

### 3. [訓練策略](./PGCN_Training_Strategy.md)
- Multi-objective Loss 設計（6 項 loss）
- Single-stage + Curriculum Learning
- 學習率調度和數據處理
- 評估指標和成功標準
- 與失敗案例的對比

**關鍵內容**：
- Loss 權重設計理由
- Curriculum learning 策略
- 訓練監控與 debug 指南

---

### 4. [實作計畫](./PGCN_Implementation_Plan.md)
- 10 天詳細時間表
- 每日任務和檢查點
- 風險評估與備選方案（Plan B/C/D）
- 成功標準與決策樹
- 預期結果總結

**關鍵內容**：
- Day-by-day 實作計畫
- 4 個備選方案
- 決策樹（何時切換 Plan）

---

## 快速導航

### 如果你想了解...

**「為什麼 PGCN 能解決我的問題？」**
→ 閱讀 [架構提案](./PGCN_Architecture_Proposal.md) 第一、二章

**「核心模組怎麼實作？」**
→ 閱讀 [模組詳細設計](./PGCN_Module_Design.md) 第三章

**「Loss 怎麼設計？為什麼這樣設計？」**
→ 閱讀 [訓練策略](./PGCN_Training_Strategy.md) 第五章

**「10 天的實作計畫是什麼？」**
→ 閱讀 [實作計畫](./PGCN_Implementation_Plan.md) 第十章

**「如果失敗了怎麼辦？」**
→ 閱讀 [實作計畫](./PGCN_Implementation_Plan.md) 第十一章（風險與備選）

---

## 核心創新總結

| 創新點 | 解決的問題 | 預期效果 |
|--------|-----------|---------|
| **Fourier Features (L=10)** | v8 的 46.6° 角度誤差 | 0.1° 精度 |
| **Complex-valued 全程** | HybridTFNet 的 Phase wrapping | Phase < 0.65 |
| **64-band FiLM** | DPATFNet 的 1-query 不足 | 精細頻率控制 |
| **Physics Constraints** | v8 的預測角度固定 | 物理合理性 |
| **hop_size=256** | 你的 DPATFNet OOM | 記憶體 < 16GB |

---

## 預期效果

| 指標 | DPATFNet 論文 | 你的實作 | PGCN 目標 | 改善 |
|------|--------------|---------|----------|------|
| L2 (×10⁻³) | ~0.144 | 0.180 | **< 0.14** | -22% |
| Phase | 0.70 | 3.28 | **< 0.65** | -80% |
| Angle (°) | ? | 46.6 | **< 2** | -96% |

---

## 實作時間表

- **Day 1-3**：核心模組（FourierEncoder, DualPath, PhysicsHead, Loss）
- **Day 4**：模型整合 + 訓練腳本
- **Day 5**：Warm-up 訓練（Epoch 1-50）
- **Day 6-7**：Full space 訓練（Epoch 51-150）
- **Day 8**：Fine-tuning（Epoch 151-200）
- **Day 9-10**：評估與調優

---

## 風險與備選

| 風險 | 備選方案 |
|------|---------|
| Physics constraints 過強 | Plan B: 移除 physics，純 data-driven |
| Fourier Features 不夠 | 增加 L (10→12) 或 Learnable Fourier |
| 記憶體不足 | Gradient checkpointing + 降低 batch size |
| Complex 不穩定 | Plan C: Magnitude + Unwrapped Phase |
| 性能接近但不夠 | Plan D: 加 U-Net refinement stage |

---

## 成功標準

### 最低標準（可接受）
- L2 < 0.16×10⁻³
- Phase < 1.5
- Angle < 10°

### 目標標準（良好）
- L2 < 0.14×10⁻³
- Phase < 0.8
- Angle < 5°

### 理想標準（優秀）
- L2 < 0.14×10⁻³
- Phase < 0.65
- Angle < 2°（人類 MAA）

---

## 下一步

1. **閱讀完整報告**（4 個文件）
2. **確認理解所有設計決策**
3. **開始實作**（從 Day 1 開始）
4. **按照檢查點監控進度**
5. **遇到問題時參考備選方案**

---

## 參考文獻

### 核心論文
1. DPATFNet (ICASSP 2025): Dual-Path Attention
2. BinauralGrad (NeurIPS 2022): 兩階段分解
3. Phase-aware (ICASSP 2024): 時頻混合
4. Meta WarpNet (ICLR 2021): Geometric warp

### 技術參考
5. Tancik et al. (2020): Fourier Features
6. Oppenheim & Schafer (2009): Minimum-phase
7. Woodworth (1938): ITD formula
8. Algazi et al. (2001): CIPIC HRTF Database

---

## 聯絡與支援

如果在實作過程中遇到問題：

1. **檢查對應章節**：每個問題都有詳細的解決方案
2. **參考備選方案**：Plan B/C/D 涵蓋主要風險
3. **查看決策樹**：何時切換策略
4. **記錄實驗結果**：方便後續分析

---

**祝實作順利！預期 10 天後看到優秀的結果。**
