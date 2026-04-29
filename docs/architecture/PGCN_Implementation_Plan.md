# PGCN 實作計畫

## 十、實作路線圖（10 天）

### Phase 1：核心模組實作（Day 1-3）

#### Day 1：基礎模組

**上午（4 小時）**：
- [ ] `FourierPositionEncoder`
  - Multi-scale Fourier Features
  - 單元測試：輸入 B×7×K，輸出 B×256
  - 驗證：不同 position 的輸出應該明顯不同

- [ ] `ComplexLayerNorm`
  - Complex-valued layer normalization
  - 單元測試：保持 complex 性質

**下午（4 小時）**：
- [ ] `ComplexConv2d`
  - Complex-valued convolution
  - 單元測試：forward + backward

- [ ] `ComplexMultiheadAttention`
  - 基於 PyTorch 的 MultiheadAttention
  - 單元測試：attention weights 的合理性

**晚上（2 小時）**：
- [ ] 整合測試：所有基礎模組
- [ ] 記錄遇到的問題

---

#### Day 2：DualPath Block

**上午（4 小時）**：
- [ ] `ComplexDualPathBlock`
  - FreqAttention + TimeAttention
  - 64-band FiLM Modulation
  - 單元測試：輸入 B×C×F×T，輸出同 shape

**下午（4 小時）**：
- [ ] `ComplexFFN`
  - Complex feed-forward network
  - 單元測試：非線性變換

- [ ] 整合測試：完整的 DualPath Block
  - 驗證：梯度流動正常
  - 驗證：記憶體使用合理（< 2GB for batch=1）

**晚上（2 小時）**：
- [ ] 性能測試：forward/backward 時間
- [ ] 記憶體 profiling

---

#### Day 3：Physics Head + Loss

**上午（4 小時）**：
- [ ] `PhysicsConstrainedHead`
  - Woodworth ITD formula
  - Frequency-dependent ILD
  - 單元測試：輸出的 ITD/ILD 是否合理

**下午（4 小時）**：
- [ ] Loss functions
  - `complex_stft_loss`
  - `multi_resolution_stft_loss`
  - `physics_constraints_loss`
  - `minimum_phase_loss`
  - `lipschitz_regularization`
  - 單元測試：每個 loss 的數值範圍

**晚上（2 小時）**：
- [ ] 整合測試：完整的 loss 計算
- [ ] 驗證：loss 權重平衡

---

### Phase 2：整合與訓練（Day 4-8）

#### Day 4：模型整合

**上午（4 小時）**：
- [ ] `PhysicsGuidedComplexNet`
  - 整合所有模組
  - 單元測試：完整 forward pass
  - 驗證：輸出 shape 正確

**下午（4 小時）**：
- [ ] 訓練腳本 `train_pgcn.py`
  - DataLoader
  - Optimizer + Scheduler
  - Training loop
  - Logging

**晚上（2 小時）**：
- [ ] 小規模測試（10 samples）
  - 驗證：loss 下降
  - 驗證：沒有 NaN/Inf
  - 驗證：記憶體使用 < 16GB

---

#### Day 5：Warm-up 訓練（Epoch 1-50）

**全天**：
- [ ] 開始訓練（Frontal only）
- [ ] 監控指標：
  - Loss 曲線
  - Phase error
  - Angle error（frontal angles）
- [ ] 調整超參數（如果需要）

**檢查點（Epoch 10）**：
- [ ] Loss 開始下降？
- [ ] Phase error < 2.0？
- [ ] 沒有 NaN/Inf？

**檢查點（Epoch 50）**：
- [ ] Frontal angles 的 angle error < 5°？
- [ ] Phase error < 1.5？
- [ ] ITD error < 300μs？

---

#### Day 6-7：Full space 訓練（Epoch 51-150）

**Day 6**：
- [ ] 繼續訓練（Epoch 51-100）
- [ ] 監控所有角度的性能
- [ ] 調整 curriculum（如果需要）

**檢查點（Epoch 100）**：
- [ ] All angles 的 angle error < 10°？
- [ ] Phase error < 1.0？
- [ ] ITD error < 200μs？

**Day 7**：
- [ ] 繼續訓練（Epoch 101-150）
- [ ] 重點監控困難角度（±90°, ±180°）

**檢查點（Epoch 150）**：
- [ ] Phase error < 0.8？
- [ ] Angle error < 5°？

---

#### Day 8：Fine-tuning（Epoch 151-200）

**全天**：
- [ ] 提高 physics loss 權重（0.10 → 0.15）
- [ ] 繼續訓練
- [ ] 監控所有指標

**檢查點（Epoch 200）**：
- [ ] L2 < 0.14×10⁻³？
- [ ] Phase < 0.65？
- [ ] Angle error < 2°？

---

### Phase 3：評估與調優（Day 9-10）

#### Day 9：完整評估

**上午（4 小時）**：
- [ ] 在 test_13angles 上評估
  - 每個角度的詳細指標
  - 生成音訊樣本
  - 可視化（ITD/ILD 曲線）

**下午（4 小時）**：
- [ ] 在 testset（連續移動）上評估
  - 整體指標
  - 與 DPATFNet 對比
  - 主觀聽感測試

**晚上（2 小時）**：
- [ ] 分析失敗案例
  - 哪些角度表現差？
  - 哪些頻段有問題？

---

#### Day 10：調優與報告

**上午（4 小時）**：
- [ ] 根據 Day 9 的分析調優
  - 調整 loss 權重？
  - 增加困難樣本？
  - Fine-tune 特定角度？

**下午（4 小時）**：
- [ ] 最終評估
- [ ] 撰寫實驗報告
  - 訓練曲線
  - 評估結果
  - 與 baseline 對比
  - 失敗案例分析

**晚上（2 小時）**：
- [ ] 整理代碼
- [ ] 撰寫 README
- [ ] 提交結果

---

## 十一、風險評估與備選方案

### 11.1 主要風險

#### 風險 1：Physics constraints 過強（可能性：中）

**症狀**：
- Loss 下降緩慢
- 模型輸出過於接近 Woodworth formula
- 無法學習個體化的 HRTF

**解決方案**：
- 降低 physics loss 權重（0.10 → 0.05）
- 增加 residual 權重（0.1 → 0.3）
- 或完全移除 physics constraints（Plan B）

---

#### 風險 2：Fourier Features 不夠精細（可能性：低）

**症狀**：
- Angle error 仍然 > 5°
- 固定角度的性能不穩定

**解決方案**：
- 增加 L（10 → 12 或 15）
- 或改用 Learnable Fourier Features
- 或加入 Positional Encoding（Transformer-style）

---

#### 風險 3：記憶體仍然不足（可能性：低）

**症狀**：
- Batch size 16 仍然 OOM
- 無法訓練

**解決方案**：
- 降低 batch size（16 → 8）
- 使用 gradient checkpointing
- 降低 channels（256 → 192）
- 或降低 num_dpab（4 → 3）

---

#### 風險 4：Complex 建模不穩定（可能性：中）

**症狀**：
- 訓練過程中出現 NaN/Inf
- Phase 震盪

**解決方案**：
- 降低 learning rate（3e-4 → 1e-4）
- 增加 gradient clipping（5.0 → 1.0）
- 或改用 Magnitude + Phase 分離建模（Plan C）

---

### 11.2 備選方案

#### Plan B：移除 Physics Constraints

如果 physics constraints 導致性能下降：

```python
# 移除 PhysicsConstrainedHead
# 改用簡單的 decoder

class SimpleHead(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.decoder = nn.Sequential(
            ComplexConv2d(channels, 128, 3, padding=1),
            ComplexReLU(),
            ComplexConv2d(128, 64, 3, padding=1),
            ComplexReLU(),
            ComplexConv2d(64, 2, 3, padding=1)  # [Y_L, Y_R]
        )
    
    def forward(self, x):
        out = self.decoder(x)  # B×2×F×T (complex)
        Y_L = out[:, 0, :, :]
        Y_R = out[:, 1, :, :]
        return Y_L, Y_R
```

**預期效果**：
- 更靈活，可能學到更好的 HRTF
- 但可能出現不合理的預測（需要更多數據）

---

#### Plan C：Magnitude + Phase 分離

如果 complex 建模不穩定：

```python
# 分別預測 magnitude 和 phase
# 但用 unwrapped phase（避免 wrapping）

class MagPhaseHead(nn.Module):
    def forward(self, x):
        # Magnitude
        mag_L = self.mag_decoder_L(x)  # B×F×T
        mag_R = self.mag_decoder_R(x)
        
        # Unwrapped phase (用 cumsum)
        phase_diff = self.phase_decoder(x)  # B×F×T
        phase_L = torch.cumsum(phase_diff[:, 0, :, :], dim=1)
        phase_R = torch.cumsum(phase_diff[:, 1, :, :], dim=1)
        
        # Combine
        Y_L = mag_L * torch.exp(1j * phase_L)
        Y_R = mag_R * torch.exp(1j * phase_R)
        
        return Y_L, Y_R
```

**預期效果**：
- 更穩定（magnitude 和 phase 分開學習）
- 但可能失去 phase coherence

---

#### Plan D：增加 U-Net Refinement

如果性能接近但不夠好：

```python
# 在 PhysicsConstrainedHead 後加一個 U-Net refinement stage

class RefinementUNet(nn.Module):
    def forward(self, Y_L, Y_R):
        # U-Net 結構
        # Encoder: downsample
        # Decoder: upsample + skip connections
        # 精細修正 Y_L, Y_R
        
        Y_L_refined = self.unet(Y_L)
        Y_R_refined = self.unet(Y_R)
        
        return Y_L_refined, Y_R_refined
```

**預期效果**：
- 進一步提升性能（+5-10%）
- 但增加參數量和訓練時間

---

## 十二、成功標準與決策樹

### 12.1 成功標準（分級）

#### 最低標準（可接受）
- L2 < 0.16×10⁻³（比你的 DPATFNet 實作好）
- Phase < 1.5（比你的實作好 54%）
- Angle error < 10°（比你的實作好 78%）

**決策**：如果達到，繼續 fine-tune

---

#### 目標標準（良好）
- L2 < 0.14×10⁻³（比 Meta large 好）
- Phase < 0.8（比 DPATFNet 論文好 14%）
- Angle error < 5°（接近人類 MAA）

**決策**：如果達到，嘗試 Plan D（U-Net refinement）

---

#### 理想標準（優秀）
- L2 < 0.14×10⁻³
- Phase < 0.65（比 DPATFNet 論文好 7%）
- Angle error < 2°（達到人類 MAA）

**決策**：成功！撰寫論文

---

### 12.2 決策樹

```
Day 5 (Epoch 50)
├─ Phase < 1.5? 
│  ├─ Yes → 繼續
│  └─ No → 檢查 complex loss 權重，提高到 0.5
│
Day 7 (Epoch 150)
├─ Phase < 0.8?
│  ├─ Yes → 繼續 fine-tuning
│  └─ No → 分析失敗原因
│     ├─ Physics constraints 過強? → Plan B
│     ├─ Position encoding 不夠? → 增加 Fourier L
│     └─ Complex 不穩定? → Plan C
│
Day 8 (Epoch 200)
├─ 達到目標標準?
│  ├─ Yes → 嘗試 Plan D (U-Net refinement)
│  └─ No → 分析並調優
│
Day 10
├─ 達到理想標準?
│  ├─ Yes → 成功！
│  └─ No → 評估是否達到最低標準
│     ├─ Yes → 可接受，撰寫報告
│     └─ No → 需要重新設計
```

---

## 十三、預期結果總結

### 13.1 定量指標

| 指標 | DPATFNet 論文 | 你的 DPATFNet | PGCN 目標 | 改善幅度 |
|------|--------------|--------------|----------|---------|
| L2 (×10⁻³) | ~0.144 | 0.180 | **< 0.14** | -22% |
| Phase | 0.70 | 3.28 | **< 0.65** | -80% |
| ITD (μs) | ~200 | 772 | **< 150** | -81% |
| ILD (dB) | ~2 | 6.03 | **< 1.8** | -70% |
| Angle (°) | ? | 46.6 | **< 2** | -96% |
| 參數量 | ~30M | 0.5M | **~24M** | +4700% |
| 記憶體 | ? | OOM | **< 16GB** | 可訓練 |

### 13.2 定性改進

1. **解決 Phase wrapping 問題**
   - Complex representation 避免 HybridTFNet 的失敗

2. **解決角度精度問題**
   - Fourier Features 提供 0.1° 精度

3. **解決記憶體問題**
   - hop_size=256 降低 4 倍記憶體

4. **解決物理合理性問題**
   - Physics constraints 避免不合理預測

5. **解決訓練穩定性問題**
   - Single-stage + curriculum 避免梯度衝突

---

## 十四、後續工作（如果成功）

1. **個體化 HRTF**
   - 加入 subject ID embedding
   - Few-shot learning for new subjects

2. **Real-time 推論**
   - 模型壓縮（pruning, quantization）
   - ONNX 導出

3. **更複雜的場景**
   - 多音源
   - 殘響環境

4. **論文撰寫**
   - 投稿 ICASSP 2027 或 INTERSPEECH 2027

---

**總結**：PGCN 的實作計畫詳細且可執行，預期可以在 10 天內完成並達到目標指標。
