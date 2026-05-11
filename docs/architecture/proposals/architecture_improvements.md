# 架構改進方案：提升 Phase 和 L2

**目標**: 在 Meta-Audio 基礎上，通過架構改進提升 Phase 學習和 L2 性能

**背景**: 
- Loss function 調整已有成效
- DPATFNet 還在訓練中
- 需要更好的架構來學習 Phase

---

## 方案 1: Multi-Scale Phase Learning ⭐⭐⭐

### 核心思想
**不同頻率的 Phase 有不同特性**：
- 低頻 (< 1.5kHz): ITD 主導，Phase 變化大
- 高頻 (> 1.5kHz): ILD 主導，Phase 變化小

**分頻處理**：
```python
class MultiScalePhaseNet(nn.Module):
    def __init__(self):
        # 低頻 Phase 網路 (0-1.5kHz)
        self.low_freq_phase = PhaseNet(
            freq_bins=72,  # 0-1.5kHz
            hidden_dim=256,
            use_itd_prior=True  # 強調 ITD
        )
        
        # 中頻 Phase 網路 (1.5-8kHz)
        self.mid_freq_phase = PhaseNet(
            freq_bins=312,  # 1.5-8kHz
            hidden_dim=128,
            use_itd_prior=True,
            use_ild_prior=True  # ITD + ILD
        )
        
        # 高頻 Phase 網路 (8-24kHz)
        self.high_freq_phase = PhaseNet(
            freq_bins=129,  # 8-24kHz
            hidden_dim=64,
            use_ild_prior=True  # 強調 ILD
        )
    
    def forward(self, mono_phase, view):
        # 分頻
        low = mono_phase[:, :72]
        mid = mono_phase[:, 72:384]
        high = mono_phase[:, 384:]
        
        # 各自處理
        phase_L_low, phase_R_low = self.low_freq_phase(low, view)
        phase_L_mid, phase_R_mid = self.mid_freq_phase(mid, view)
        phase_L_high, phase_R_high = self.high_freq_phase(high, view)
        
        # 合併
        phase_L = torch.cat([phase_L_low, phase_L_mid, phase_L_high], dim=1)
        phase_R = torch.cat([phase_R_low, phase_R_mid, phase_R_high], dim=1)
        
        return phase_L, phase_R
```

### 優點
- ✅ 符合聽覺理論 (Duplex theory)
- ✅ 低頻專注 ITD，高頻專注 ILD
- ✅ 減少參數衝突

### 實現難度
- 🟢 簡單：只需修改網路結構
- 訓練時間：和原本差不多

---

## 方案 2: Attention-based Phase Refinement ⭐⭐⭐⭐

### 核心思想
**Phase 和 Magnitude 應該互相關聯**：
- Magnitude 大的頻率，Phase 更重要
- Magnitude 小的頻率，Phase 可以忽略

**Cross-attention 機制**：
```python
class AttentionPhaseNet(nn.Module):
    def __init__(self):
        self.magnitude_encoder = MagnitudeEncoder(513, 256)
        self.phase_encoder = PhaseEncoder(513, 256)
        
        # Cross-attention: Phase attends to Magnitude
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8
        )
        
        self.phase_decoder = PhaseDecoder(256, 513)
    
    def forward(self, mono_mag, mono_phase, view):
        # Encode
        mag_feat = self.magnitude_encoder(mono_mag, view)  # (B, T, 256)
        phase_feat = self.phase_encoder(mono_phase, view)  # (B, T, 256)
        
        # Cross-attention: Phase 關注 Magnitude
        refined_phase, attention_weights = self.cross_attention(
            query=phase_feat,      # Phase 作為 query
            key=mag_feat,          # Magnitude 作為 key
            value=mag_feat         # Magnitude 作為 value
        )
        
        # Decode
        phase_L, phase_R = self.phase_decoder(refined_phase, view)
        
        return phase_L, phase_R, attention_weights
```

### 優點
- ✅ Magnitude 引導 Phase 學習
- ✅ 自動學習頻率重要性
- ✅ Attention weights 可視化

### 實現難度
- 🟡 中等：需要加入 Transformer
- 訓練時間：增加 20-30%

---

## 方案 3: Residual Phase Learning ⭐⭐⭐⭐⭐

### 核心思想
**不要直接預測 Phase，預測 Phase 的修正量**：
```
Phase_pred = Phase_mono + Physical_ITD + Learned_Residual
```

**分解學習**：
```python
class ResidualPhaseNet(nn.Module):
    def __init__(self):
        # Physical ITD (不需要學習)
        self.physical_itd = PhysicalITD(
            head_radius=0.0875,  # 8.75cm
            sound_speed=343.0    # m/s
        )
        
        # Learned residual (小的修正)
        self.residual_net = ResidualNet(
            input_dim=513,
            hidden_dim=128,
            output_dim=513,
            num_layers=3
        )
    
    def forward(self, mono_phase, view, freq_bins):
        # 1. Physical ITD (幾何)
        physical_itd = self.physical_itd(view)  # (B, 1)
        physical_phase_shift = 2 * np.pi * freq_bins * physical_itd  # (B, 513)
        
        # 2. Learned residual (頻率相關修正)
        residual = self.residual_net(mono_phase, view)  # (B, 513)
        
        # 3. 合併
        phase_L = mono_phase + (physical_phase_shift + residual) / 2
        phase_R = mono_phase - (physical_phase_shift + residual) / 2
        
        # 4. Wrap to [-π, π]
        phase_L = torch.atan2(torch.sin(phase_L), torch.cos(phase_L))
        phase_R = torch.atan2(torch.sin(phase_R), torch.cos(phase_R))
        
        return phase_L, phase_R
```

### 優點
- ✅ Physical prior 提供基礎
- ✅ Residual 只需學習小的修正
- ✅ 更容易收斂
- ✅ 符合物理直覺

### 實現難度
- 🟢 簡單：已經在 HybridPhysical 實現過
- 訓練時間：和原本一樣

### 改進點
**之前失敗的原因**：
- Residual 網路太大，學習太多
- 沒有約束 residual 的大小

**新的改進**：
```python
# 1. 限制 residual 大小
residual = torch.tanh(self.residual_net(...)) * 0.5  # 限制在 [-0.5, 0.5]

# 2. Regularization
residual_loss = torch.mean(residual ** 2)  # 懲罰大的 residual
total_loss = waveform_loss + 0.01 * residual_loss

# 3. 頻率相關的 residual
# 低頻允許大的 residual，高頻限制小的 residual
freq_mask = torch.linspace(1.0, 0.1, 513)  # 低頻→高頻
residual = residual * freq_mask
```

---

## 方案 4: Temporal Consistency ⭐⭐⭐⭐

### 核心思想
**Phase 應該在時間上連續**：
- 相鄰 frame 的 Phase 不應該跳變
- 加入 temporal smoothness constraint

**LSTM/GRU 處理時間序列**：
```python
class TemporalPhaseNet(nn.Module):
    def __init__(self):
        self.phase_encoder = PhaseEncoder(513, 256)
        
        # Bidirectional LSTM for temporal modeling
        self.temporal_lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        self.phase_decoder = PhaseDecoder(256, 513)
    
    def forward(self, mono_phase, view):
        # Encode each frame
        B, T, F = mono_phase.shape
        phase_feat = self.phase_encoder(mono_phase, view)  # (B, T, 256)
        
        # Temporal modeling
        temporal_feat, _ = self.temporal_lstm(phase_feat)  # (B, T, 256)
        
        # Decode
        phase_L, phase_R = self.phase_decoder(temporal_feat, view)
        
        return phase_L, phase_R
```

### Temporal Smoothness Loss
```python
def temporal_smoothness_loss(phase_pred):
    # Phase difference between adjacent frames
    phase_diff = phase_pred[:, 1:] - phase_pred[:, :-1]
    
    # Wrap to [-π, π]
    phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
    
    # Penalize large jumps
    smoothness = torch.mean(phase_diff ** 2)
    
    return smoothness

# Total loss
total_loss = waveform_loss + 0.1 * temporal_smoothness_loss(phase_L) + 0.1 * temporal_smoothness_loss(phase_R)
```

### 優點
- ✅ 時間連續性
- ✅ 減少 Phase 跳變
- ✅ 更自然的聲音

### 實現難度
- 🟡 中等：需要處理序列
- 訓練時間：增加 30-40%

---

## 方案 5: Perceptual Phase Loss ⭐⭐⭐⭐⭐

### 核心思想
**不要直接優化 Phase，優化聽覺感知**：
- Phase 本身不重要，重要的是聽起來對不對
- 使用 perceptual loss

**Multi-resolution STFT Loss**：
```python
class PerceptualPhaseLoss(nn.Module):
    def __init__(self):
        # Multiple STFT resolutions
        self.fft_sizes = [512, 1024, 2048]
        self.hop_sizes = [128, 256, 512]
        self.win_sizes = [512, 1024, 2048]
    
    def forward(self, pred, target):
        loss = 0
        
        for fft_size, hop_size, win_size in zip(
            self.fft_sizes, self.hop_sizes, self.win_sizes
        ):
            # STFT
            pred_stft = torch.stft(pred, n_fft=fft_size, hop_length=hop_size, 
                                   win_length=win_size, return_complex=True)
            target_stft = torch.stft(target, n_fft=fft_size, hop_length=hop_size,
                                     win_length=win_size, return_complex=True)
            
            # Magnitude loss
            mag_loss = F.l1_loss(pred_stft.abs(), target_stft.abs())
            
            # Phase loss (weighted by magnitude)
            phase_pred = torch.angle(pred_stft)
            phase_target = torch.angle(target_stft)
            mag_weight = target_stft.abs() / (target_stft.abs().max() + 1e-8)
            phase_loss = F.l1_loss(
                torch.sin(phase_pred) * mag_weight,
                torch.sin(phase_target) * mag_weight
            ) + F.l1_loss(
                torch.cos(phase_pred) * mag_weight,
                torch.cos(phase_target) * mag_weight
            )
            
            loss += mag_loss + 0.1 * phase_loss
        
        return loss / len(self.fft_sizes)
```

### 優點
- ✅ 直接優化聽覺感知
- ✅ Magnitude 大的地方 Phase 更重要
- ✅ Multi-resolution 捕捉不同尺度

### 實現難度
- 🟢 簡單：只是 loss function
- 訓練時間：和原本一樣

---

## 方案 6: Hybrid: 結合多個方案 ⭐⭐⭐⭐⭐

### 最佳組合
```python
class ImprovedBinauralNet(nn.Module):
    def __init__(self):
        # 1. Multi-scale magnitude
        self.magnitude_net = MultiScaleMagnitudeNet()
        
        # 2. Residual phase with physical prior
        self.physical_itd = PhysicalITD()
        self.residual_phase_net = ResidualPhaseNet()
        
        # 3. Cross-attention between magnitude and phase
        self.cross_attention = CrossAttention(256, num_heads=8)
        
        # 4. Temporal modeling
        self.temporal_lstm = nn.LSTM(256, 128, 2, bidirectional=True)
    
    def forward(self, mono, view):
        # STFT
        mono_stft = torch.stft(mono, ...)
        mono_mag = mono_stft.abs()
        mono_phase = torch.angle(mono_stft)
        
        # 1. Magnitude (multi-scale)
        mag_L, mag_R = self.magnitude_net(mono_mag, view)
        
        # 2. Physical ITD
        physical_itd = self.physical_itd(view)
        physical_phase_shift = 2 * np.pi * freq_bins * physical_itd
        
        # 3. Learned residual (with attention)
        mag_feat = self.magnitude_net.get_features()
        phase_feat = self.residual_phase_net.get_features(mono_phase, view)
        refined_phase_feat = self.cross_attention(phase_feat, mag_feat)
        
        # 4. Temporal modeling
        temporal_feat, _ = self.temporal_lstm(refined_phase_feat)
        
        # 5. Decode residual
        residual = self.residual_phase_net.decode(temporal_feat)
        residual = torch.tanh(residual) * 0.5  # Limit residual
        
        # 6. Combine
        phase_L = mono_phase + (physical_phase_shift + residual) / 2
        phase_R = mono_phase - (physical_phase_shift + residual) / 2
        
        # 7. Reconstruct
        Y_L = mag_L * torch.exp(1j * phase_L)
        Y_R = mag_R * torch.exp(1j * phase_R)
        
        return Y_L, Y_R
```

### Loss Function
```python
def compute_loss(pred, target, residual):
    # 1. Perceptual loss (multi-resolution STFT)
    perceptual_loss = perceptual_phase_loss(pred, target)
    
    # 2. Waveform L2
    waveform_loss = F.mse_loss(pred, target)
    
    # 3. Residual regularization
    residual_reg = torch.mean(residual ** 2)
    
    # 4. Temporal smoothness
    temporal_smooth = temporal_smoothness_loss(pred)
    
    # Total
    total_loss = (
        10.0 * waveform_loss +
        5.0 * perceptual_loss +
        0.01 * residual_reg +
        0.1 * temporal_smooth
    )
    
    return total_loss
```

---

## 推薦實施順序

### Phase 1: 快速驗證 (1-2 天)

**實驗 E9: Residual Phase + Regularization**
```python
# 改進 HybridPhysical
# 1. 限制 residual 大小
# 2. 加入 residual regularization
# 3. 頻率相關的 residual mask
```

**預期**:
- 如果成功 → L2 < 0.00080
- 快速驗證 residual 方法的潛力

### Phase 2: 中等改進 (3-5 天)

**實驗 E10: Multi-scale + Attention**
```python
# 1. Multi-scale phase learning
# 2. Cross-attention between magnitude and phase
# 3. Perceptual loss
```

**預期**:
- 如果成功 → L2 < 0.00070
- 接近或超越 DPATFNet

### Phase 3: 完整方案 (1-2 週)

**實驗 E11: Full Hybrid**
```python
# 結合所有改進
# 1. Multi-scale
# 2. Residual with physical prior
# 3. Cross-attention
# 4. Temporal modeling
# 5. Perceptual loss
```

**預期**:
- 如果成功 → L2 < 0.00065
- 顯著超越 DPATFNet

---

## 立即行動

### 今天可以開始

**實驗 E9: Improved Residual Phase**

1. **修改 `src/models_hybrid_physical.py`**
```python
# 加入 residual 限制和 regularization
```

2. **修改 `training/train_hybrid_physical.py`**
```python
# 加入新的 loss terms
```

3. **訓練**
```bash
python training/train_hybrid_physical.py
```

**預計時間**: 1-2 天訓練

---

## 總結

### 最有潛力的方案

1. **Residual Phase + Regularization** ⭐⭐⭐⭐⭐
   - 簡單、快速、有理論支持
   - 之前失敗是因為沒有約束

2. **Perceptual Loss** ⭐⭐⭐⭐⭐
   - 直接優化聽覺感知
   - 只需改 loss，不改架構

3. **Cross-attention** ⭐⭐⭐⭐
   - Magnitude 引導 Phase
   - 符合直覺

### 建議策略

**先快後慢**:
1. 先試 Residual + Perceptual Loss (2-3 天)
2. 如果有效，加入 Attention (1 週)
3. 最後整合 Multi-scale + Temporal (2 週)

**並行論文**:
- 如果改進成功 → 寫 positive results
- 如果改進失敗 → 寫 negative results
- 無論如何都有貢獻

你想先試哪個方案？
