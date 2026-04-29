# HybridTFNet 訓練記錄

**日期**：2026-04-27  
**模型**：HybridTFNet  
**目標**：解決 v8.3 的 Phase/ITD 學習問題

---

## 架構設計

### 核心創新

1. **時頻混合 + 任務分離**
   - Time Branch：時域學 ITD/Phase
   - Freq Branch：頻域學 ILD/Magnitude
   - Fusion：複數乘法融合

2. **幾何先驗 + 神經修正**
   - GeometricITD：Woodworth 公式（固定）
   - FrequencyDependentDelay：頻率相關的 phase shift（可學習）

3. **Shared Encoder 部分共享**
   - Low-level 特徵共享（Conv1d: 1→64→128）
   - High-level 特徵獨立（Time 和 Freq 各自的 encoder）

4. **Cross-Attention 位置編碼**
   - Position query Audio
   - 比 SimpleDPAB 強，比 DPAB 簡潔

### 模型參數

- **總參數量**：11,460,484 (約 11.5M)
- Shared Encoder: ~0.2M
- Time Branch: ~2M
- Freq Branch: ~9M

### 訓練策略

| Stage | Epochs | 訓練對象 | Loss | 學習率 |
|-------|--------|---------|------|--------|
| **Stage 1** | 0-60 | Time Branch + Shared Encoder | Phase_L + Phase_R + IPD | 3e-4 |
| **Stage 2** | 60-160 | Freq Branch + Shared Encoder (小 lr) | Mag_L + Mag_R + ILD | 3e-4 / 3e-5 |
| **Stage 3** | 160-200 | 全部 | 100×L2 + Phase + Mag | 3e-4 |

---

## 與 v8.3 的差異

| 維度 | v8.3 | HybridTFNet |
|------|------|-------------|
| 任務分離 | Common + Specific | **Time + Freq** |
| Time Branch | WaveNet 學「共同部分」| **Warp + Learnable Delay 學 ITD** |
| Freq Branch | 學「所有差異」| **只學 ILD + HRTF** |
| 位置編碼 | SimpleDPAB（太弱）| **Cross-Attention** |
| Stage 1 目標 | L2（錯誤）| **Phase（正確）** |
| Stage 2 目標 | Phase（從 Y_common）| **Magnitude（從 Y_mono）** |

---

## 成功標準

### Stage 1（epoch 60）
- ✅ Phase error < 1.3 rad²（比 v8.3 的 1.43 好）
- ✅ IPD error < 1.3（比 v8.3 的 1.35 好）

### Stage 2（epoch 160）
- ✅ Phase error < 0.85 rad²（改善 >40%）
- ✅ Magnitude error 正常下降
- ✅ L2 loss < 0.00005

### Stage 3（epoch 200）
- ✅ Phase error < 0.7 rad²
- ✅ L2 loss < 0.00004
- ✅ 主觀聽感接近真實

---

## 訓練記錄

### 2026-04-27 - 開始訓練

**配置**：
- batch_size: 16
- learning_rate: 3e-4
- n_fft: 1024
- hop_size: 256
- tf_channels: 256
- tf_blocks: 8

**狀態**：訓練中...

---

## 風險與備選方案

### 主要風險

1. **FrequencyDependentDelay 可能學不到精確 delay**
   - 緩解：max_delay 設為 32 samples
   - 備選：回退到 Warpnet

2. **Cross-Attention 可能過擬合**
   - 緩解：Dropout 0.1
   - 備選：回退到 FiLM

3. **兩個 Branch 可能不一致**
   - 緩解：Stage 3 的 joint fine-tuning
   - 備選：加入 consistency loss（如果需要）

### 備選方案

- **Plan B**：如果 Stage 2 改善 < 20%，簡化 Freq Branch
- **Plan C**：如果整個架構失敗，回退到單 Branch（端到端）

---

## 下一步

1. **監控 Stage 1**（epoch 60）
   - 檢查 Phase error 是否 < 1.3
   - 如果 > 1.4，停止並分析

2. **監控 Stage 2**（epoch 100, 160）
   - Epoch 100: Phase error 應該 < 1.0
   - Epoch 160: Phase error 應該 < 0.85

3. **評估最終效果**（epoch 200）
   - 在 test_13angles 和 test_dynamic 上評估
   - 與 DPATFNet 和 v8.3 對比
