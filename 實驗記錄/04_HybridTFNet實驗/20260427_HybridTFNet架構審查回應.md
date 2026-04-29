# HybridTFNet 架構審查回應與修正方案

**日期**：2026-04-27  
**回應者**：Architecture Synthesis Agent

---

## 審查意見回應

### 🔴 高風險問題

#### 問題 1: Phase/Magnitude 分離假設過於理想

**你的觀察**：
- HRTF 的 Phase 和 Magnitude 高度耦合
- 純複數乘法融合可能導致不相容
- 可能產生 artifact

**我的回應**：
✅ **你說得對，這是最大的風險。**

**修正方案 A：加入 Consistency Loss**

```python
# Stage 3 加入一致性約束
def consistency_loss(time_output, freq_output):
    # 從 Time Branch 的輸出計算 Magnitude
    y_L_time, y_R_time = time_output
    Y_L_time = STFT(y_L_time)
    Mag_L_time = torch.abs(Y_L_time)
    
    # 與 Freq Branch 的 Magnitude 對齊
    Mag_L_freq = freq_output['Mag_L']
    
    loss = F.mse_loss(Mag_L_time, Mag_L_freq)
    return loss

# Stage 3 總 Loss
loss = L2_loss + Phase_loss + Mag_loss + 0.1 * consistency_loss
```

**修正方案 B：改為 Residual 融合（更穩健）**

```python
# 不用純乘法，改為 residual
Y_mono = STFT(mono)

# Time Branch 預測 Phase 修正
Phase_delta_L, Phase_delta_R = time_branch(mono, view)

# Freq Branch 預測 Magnitude 修正
Mag_delta_L, Mag_delta_R = freq_branch(mono, view)

# 融合
Phase_L = angle(Y_mono) + Phase_delta_L  # Residual
Phase_R = angle(Y_mono) + Phase_delta_R
Mag_L = abs(Y_mono) + Mag_delta_L
Mag_R = abs(Y_mono) + Mag_delta_R

Y_L = Mag_L * exp(1j * Phase_L)
Y_R = Mag_R * exp(1j * Phase_R)
```

**修正方案 C：先做 Toy Experiment 驗證**

在實作前，用真實數據驗證：
1. 從 GT 提取 Phase_L, Phase_R, Mag_L, Mag_R
2. 用複數乘法重建：`Y_L = Mag_L * exp(1j * Phase_L)`
3. 計算重建誤差
4. 如果誤差很大，說明分離假設不成立

**建議**：先做 Toy Experiment（1 小時），如果通過再實作。

---

#### 問題 2: LearnableDelayNet 的實現挑戰被低估

**你的觀察**：
- 時變 delay 難以向量化
- 梯度反傳非平凡
- 插值精度可能不夠

**我的回應**：
✅ **你說得對，4 小時太樂觀了。**

**修正方案 A：簡化為固定 Delay（推薦）**

```python
class SimplifiedDelayNet(nn.Module):
    """
    不用時變 delay，改為預測固定的 delay 值
    """
    def __init__(self):
        self.delay_predictor = nn.Sequential(
            nn.Linear(7, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # 預測單一 delay 值
        )
    
    def forward(self, y_geo, view):
        # 預測 delay（單位：samples）
        delay = self.delay_predictor(view.mean(dim=-1))  # B×1
        delay = torch.tanh(delay) * 32  # 限制在 ±32 samples
        
        # 用 fractional delay filter（固定 kernel）
        y_delayed = fractional_delay(y_geo, delay)
        
        return y_delayed
```

**Fractional Delay Filter（已有成熟實現）**：
```python
def fractional_delay(x, delay):
    """
    用 sinc interpolation 實現 fractional delay
    PyTorch 可微分
    """
    # 使用 torchaudio.functional.resample 的底層實現
    # 或用 differentiable sinc interpolation
    pass
```

**修正方案 B：回退到 Warpnet（如果 A 失敗）**

```python
# 保留原有的 Warpnet，但加入穩定性約束
warpfield = torch.tanh(warpfield_raw) * max_warp_range
loss_smooth = torch.mean((warpfield[:, 1:] - warpfield[:, :-1])**2)
```

**時間重估**：
- 方案 A（SimplifiedDelay）：2 小時
- 方案 B（Warpnet）：已有實現，1 小時

---

#### 問題 3: GeometricITD 的適用範圍

**你的觀察**：
- Woodworth 公式只適用於遠場、球形頭
- 近場或耳廓繞射時，幾何先驗可能「鎖死」模型

**我的回應**：
✅ **這是合理的擔憂。**

**修正方案 A：讓 GeometricITD 可訓練（但加正則化）**

```python
class LearnableGeometricITD(nn.Module):
    def __init__(self):
        # 初始化為 Woodworth 公式的參數
        self.head_radius = nn.Parameter(torch.tensor(0.0875))
        self.correction_net = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, view):
        # 基礎幾何 ITD
        itd_geo = woodworth_formula(view, self.head_radius)
        
        # 神經網路修正（小範圍）
        correction = self.correction_net(view.mean(dim=-1))
        correction = torch.tanh(correction) * 0.0002 * sample_rate
        
        itd = itd_geo + correction
        return itd
```

**修正方案 B：根據距離調整先驗權重**

```python
def adaptive_geometric_itd(view):
    # 計算距離
    distance = torch.norm(view[:, :3, :], dim=1).mean(dim=-1)  # B
    
    # 遠場（>2m）：幾何先驗權重高
    # 近場（<0.5m）：幾何先驗權重低
    weight = torch.sigmoid((distance - 1.0) / 0.5)  # B
    
    itd_geo = woodworth_formula(view)
    itd_neural = neural_net(view)
    
    itd = weight * itd_geo + (1 - weight) * itd_neural
    return itd
```

**建議**：先用固定的 GeometricITD，如果 Stage 1 收斂不好，再改為可訓練。

---

### 🟡 中風險問題

#### 問題 4: Stage 1 的訓練目標設計

**你的觀察**：
- Time Branch 和 Freq Branch 是否共享 encoder？
- 如果共享，Stage 1/2 順序訓練會互相干擾
- 如果不共享，參數量大幅增加

**我的回應**：
✅ **這是關鍵設計決策，我沒有說清楚。**

**明確設計：不共享 Encoder**

```python
class HybridTFNet(nn.Module):
    def __init__(self):
        # Time Branch 有自己的 audio encoder
        self.time_branch = TimeBranch(
            audio_encoder=Conv1dEncoder(...)  # 獨立
        )
        
        # Freq Branch 有自己的 audio encoder
        self.freq_branch = FreqBranch(
            audio_encoder=Conv2dEncoder(...)  # 獨立
        )
```

**理由**：
1. Time Branch 在時域（1D），Freq Branch 在頻域（2D）
2. 兩者的特徵表示本來就不同
3. 不共享可以避免 Stage 1/2 的干擾

**參數量對比**：
- Time Branch encoder: ~1M 參數
- Freq Branch encoder: ~2M 參數
- 總增加: ~3M（可接受）

---

#### 問題 5: Cross-Attention 的 Query/Key 設計

**你的觀察**：
- 音訊 query 位置可能導致 attention map 退化
- 反過來（位置 query 音訊）可能更合理

**我的回應**：
✅ **你的建議很有道理。**

**修正設計：雙向 Cross-Attention**

```python
class BidirectionalCrossAttention(nn.Module):
    def __init__(self):
        # 方向 1: Audio query Position
        self.audio_query_pos = CrossAttention(...)
        
        # 方向 2: Position query Audio
        self.pos_query_audio = CrossAttention(...)
    
    def forward(self, audio_feat, pos_feat):
        # 雙向 attention
        out1 = self.audio_query_pos(audio_feat, pos_feat)
        out2 = self.pos_query_audio(pos_feat, audio_feat)
        
        # 融合（例如：相加或 concat）
        out = out1 + out2
        return out
```

**或者簡化為：Position query Audio（你的建議）**

```python
class CrossAttentionBlock(nn.Module):
    def forward(self, audio_feat, pos_feat):
        # Q 來自 position，K/V 來自 audio
        Q = self.q_proj(pos_feat)  # B×K×C
        K = self.k_proj(audio_feat)  # B×(F*T)×C
        V = self.v_proj(audio_feat)
        
        attn = softmax(Q @ K.T / sqrt(d))  # B×K×(F*T)
        out = attn @ V  # B×K×C
        
        # Broadcast 回 audio 的形狀
        out = self.broadcast(out, audio_feat.shape)
        return out
```

**建議**：先用你建議的方向（Position query Audio），如果效果不好再試雙向。

---

#### 問題 6: 成功指標的基準問題

**你的觀察**：
- Phase error 計算方式不一致（unwrapped? circular mean?）
- 不同論文無法直接比較

**我的回應**：
✅ **必須明確定義評估指標。**

**標準化評估指標**：

```python
def phase_error(pred, target):
    """
    Circular mean error (Phase-aware 2024 的方式)
    """
    # 計算相位差（考慮週期性）
    diff = pred - target
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))  # Wrap to [-π, π]
    
    # Mean absolute error
    error = torch.abs(diff).mean()
    
    return error

def ipd_error(pred_L, pred_R, target_L, target_R):
    """
    Interaural Phase Difference error
    """
    ipd_pred = pred_L - pred_R
    ipd_target = target_L - target_R
    
    diff = ipd_pred - ipd_target
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))
    
    error = torch.abs(diff).mean()
    
    return error
```

**評估頻段**：
- 只計算 20Hz - 2kHz（ITD 主要作用的頻段）
- 高頻（>2kHz）的 Phase 對感知影響小

**建議**：在實驗記錄中明確寫清楚評估方式，方便未來對比。

---

### 🟢 設計合理但需要驗證的部分

#### 問題 7: 分階段訓練的 epoch 分配

**你的觀察**：
- Stage 1 只有 60 epochs 可能不夠

**我的回應**：
✅ **可以調整。**

**修正方案：動態 Stage 切換**

```python
# 不固定 epoch，根據收斂情況切換
if stage == 1:
    if phase_error < 1.3 and phase_improve < 1% (last 10 epochs):
        # Stage 1 收斂了，切換到 Stage 2
        stage = 2
    elif epoch > 80:
        # 超過 80 epochs 還沒收斂，強制切換
        stage = 2
```

**建議的 epoch 範圍**：
- Stage 1: 60-80 epochs（根據收斂情況）
- Stage 2: 80-120 epochs
- Stage 3: 20-40 epochs

---

#### 問題 8: FiLM 條件化的位置

**你的觀察**：
- FiLM 應該在 warp 之前還是之後？

**我的回應**：
✅ **應該在 warp 之前。**

**修正設計**：

```python
def forward(self, mono, view):
    # 1. 位置編碼
    pos_feat = self.position_encoder(view)
    
    # 2. 音訊編碼
    audio_feat = self.audio_encoder(mono)
    
    # 3. FiLM 條件（在 warp 之前）
    audio_feat = self.film(audio_feat, pos_feat)
    
    # 4. 幾何 warp
    y_geo_L = geometric_warp(mono, view, 'left')
    y_geo_R = geometric_warp(mono, view, 'right')
    
    # 5. Learnable Delay（用條件化的特徵）
    y_L = self.delay_net_L(y_geo_L, audio_feat)
    y_R = self.delay_net_R(y_geo_R, audio_feat)
```

---

## 修正後的架構設計

### 關鍵修正

1. **Phase/Magnitude 融合**：
   - ✅ 先做 Toy Experiment 驗證
   - ✅ 加入 Consistency Loss
   - ✅ 考慮 Residual 融合

2. **LearnableDelayNet**：
   - ✅ 簡化為固定 delay（預測單一值）
   - ✅ 用 fractional delay filter
   - ✅ 時間重估：2 小時

3. **GeometricITD**：
   - ✅ 先用固定的，如果不行再改為可訓練
   - ✅ 考慮距離自適應權重

4. **Encoder 共享**：
   - ✅ 明確：不共享（Time 和 Freq 各自獨立）

5. **Cross-Attention**：
   - ✅ 改為 Position query Audio

6. **評估指標**：
   - ✅ 明確定義 Phase error（circular mean, 20Hz-2kHz）

7. **Stage 切換**：
   - ✅ 動態切換（根據收斂情況）

8. **FiLM 位置**：
   - ✅ 在 warp 之前

---

## 優先驗證實驗（開始實作前）

### Experiment 1: Phase/Magnitude 分離可行性（1 小時）

```python
# 從真實數據驗證
for batch in test_loader:
    mono, view, target = batch
    
    # 提取 GT 的 Phase 和 Magnitude
    Y_L_gt = STFT(target[:, 0, :])
    Y_R_gt = STFT(target[:, 1, :])
    
    Phase_L_gt = torch.angle(Y_L_gt)
    Phase_R_gt = torch.angle(Y_R_gt)
    Mag_L_gt = torch.abs(Y_L_gt)
    Mag_R_gt = torch.abs(Y_R_gt)
    
    # 重建
    Y_L_recon = Mag_L_gt * torch.exp(1j * Phase_L_gt)
    Y_R_recon = Mag_R_gt * torch.exp(1j * Phase_R_gt)
    
    y_L_recon = iSTFT(Y_L_recon)
    y_R_recon = iSTFT(Y_R_recon)
    
    # 計算誤差
    error = F.mse_loss(y_L_recon, target[:, 0, :])
    print(f"Reconstruction error: {error:.6f}")
```

**判斷標準**：
- 如果 error < 1e-6：分離假設成立 ✅
- 如果 error > 1e-4：分離假設有問題 ❌

### Experiment 2: Fractional Delay 可行性（1 小時）

```python
# 測試 fractional delay 的梯度
delay_net = SimplifiedDelayNet()
optimizer = torch.optim.Adam(delay_net.parameters())

for i in range(100):
    mono = torch.randn(4, 1, 16000, requires_grad=True)
    view = torch.randn(4, 7, 10)
    target = torch.randn(4, 1, 16000)
    
    # Forward
    y_delayed = delay_net(mono, view)
    loss = F.mse_loss(y_delayed, target)
    
    # Backward
    loss.backward()
    optimizer.step()
    
    if i % 10 == 0:
        print(f"Iter {i}: Loss = {loss.item():.6f}")
```

**判斷標準**：
- Loss 能下降：可行 ✅
- Loss 不動或梯度爆炸：不可行 ❌

---

## 修正後的時間估算

| 階段 | 任務 | 原估算 | 修正後 |
|------|------|--------|--------|
| **驗證實驗** | Toy Experiments | 0 | **2 小時** |
| **Phase 1** | 核心模組 | 1-2 天 | **1-2 天** |
| **Phase 2** | Branch 模組 | 3-4 天 | **3-4 天** |
| **Phase 3** | 整合與訓練 | 5-7 天 | **6-8 天** |

**總計**：6-8 天（加上驗證實驗）

---

## 建議的實作順序

1. **Toy Experiment 1**：驗證 Phase/Magnitude 分離（1 小時）
2. **Toy Experiment 2**：驗證 Fractional Delay（1 小時）
3. **如果兩個實驗都通過**：開始實作 HybridTFNet
4. **如果實驗 1 失敗**：考慮 Plan B（端到端，不分離）
5. **如果實驗 2 失敗**：回退到 Warpnet

---

**你的審查非常有價值，幫助我發現了幾個關鍵風險。建議先做這兩個 Toy Experiments，確認可行性後再開始實作。你覺得如何？**
