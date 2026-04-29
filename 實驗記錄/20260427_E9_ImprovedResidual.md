# E9: Improved Residual Phase Learning

**日期**: 2026-04-27  
**狀態**: 🔄 準備中  
**目標**: 通過約束和 Perceptual Loss 改進 Residual Phase 學習

---

## 背景

### E7/E8 失敗分析

**E7 (HybridPhysical)**:
```
L2: 0.000830
Phase L/R: 0.999
→ Residual 沒有學到有用的東西
```

**E8 (Staged Training)**:
```
L2: 0.000864
Phase L/R: 0.999
→ 分階段訓練也沒幫助
```

**失敗原因**:
1. ❌ Residual 網路太大，學習太多無用的東西
2. ❌ 沒有約束 residual 的大小
3. ❌ 高頻和低頻用同樣的 residual scale
4. ❌ Loss function 只有 L2，不夠好

---

## 改進策略

### 1. 限制 Residual 大小

**問題**: Residual 可以任意大，導致學習不穩定

**解決**:
```python
# 之前 (E7/E8)
residual = self.residual_net(features)  # 無限制

# 現在 (E9)
residual = torch.tanh(self.residual_net(features)) * 0.5  # 限制在 [-0.5, 0.5]
```

**原理**:
- Physical ITD 已經提供基礎
- Residual 只需要小的修正
- 限制大小避免過度修正

### 2. Residual Regularization

**問題**: 沒有懲罰大的 residual

**解決**:
```python
# L2 regularization on residual
residual_reg = torch.mean(residual ** 2)
total_loss += 0.01 * residual_reg
```

**原理**:
- 鼓勵 residual 盡量小
- 只在必要時才修正
- 避免 overfitting

### 3. 頻率相關的 Residual Mask

**問題**: 低頻和高頻用同樣的 residual scale

**解決**:
```python
# 低頻允許大的 residual，高頻限制小的 residual
freq_mask = torch.linspace(1.0, 0.1, 513, device=residual.device)
# freq_mask: [1.0, 0.998, ..., 0.102, 0.1]

residual = residual * freq_mask.unsqueeze(0).unsqueeze(0)
```

**原理** (Duplex Theory):
- 低頻 (< 1.5kHz): ITD 主導，需要大的 phase 修正
- 高頻 (> 1.5kHz): ILD 主導，phase 修正應該小

### 4. Perceptual Loss

**問題**: 只有 L2 loss，不夠好

**解決**: Multi-resolution STFT Loss
```python
class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self):
        self.fft_sizes = [512, 1024, 2048]
        self.hop_sizes = [128, 256, 512]
        self.win_sizes = [512, 1024, 2048]
    
    def forward(self, pred, target):
        loss = 0
        for fft_size, hop_size, win_size in zip(...):
            # STFT
            pred_stft = torch.stft(pred, n_fft=fft_size, ...)
            target_stft = torch.stft(target, n_fft=fft_size, ...)
            
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

**原理**:
- Multi-resolution 捕捉不同尺度的特徵
- Magnitude 大的地方 phase 更重要
- 直接優化聽覺感知

### 5. Temporal Smoothness

**問題**: Phase 在時間上可能跳變

**解決**:
```python
def temporal_smoothness_loss(phase):
    # Phase difference between adjacent frames
    phase_diff = phase[:, :, 1:] - phase[:, :, :-1]
    
    # Wrap to [-π, π]
    phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
    
    # Penalize large jumps
    return torch.mean(phase_diff ** 2)

total_loss += 0.1 * temporal_smoothness_loss(phase_L)
total_loss += 0.1 * temporal_smoothness_loss(phase_R)
```

**原理**:
- Phase 應該連續
- 減少跳變
- 更自然的聲音

---

## 模型架構

### ImprovedResidualPhaseNet

```python
class ImprovedResidualPhaseNet(nn.Module):
    def __init__(self):
        # View encoder
        self.view_encoder = ViewEncoder(
            view_dim=7,
            hidden_dim=128,
            output_dim=256
        )
        
        # Magnitude network
        self.magnitude_net = MagnitudeNet(
            freq_bins=513,
            view_dim=256,
            hidden_dim=512
        )
        
        # Residual ITD network (smaller than before)
        self.residual_itd_net = ResidualITDNet(
            freq_bins=513,
            view_dim=256,
            hidden_dim=128,  # 減小 (之前是 256)
            num_layers=2     # 減少層數 (之前是 3)
        )
        
        # Physical parameters
        self.head_radius = 0.0875  # 8.75cm
        self.sound_speed = 343.0   # m/s
        
        # Frequency mask (learnable or fixed)
        self.register_buffer(
            'freq_mask',
            torch.linspace(1.0, 0.1, 513)
        )
    
    def forward(self, mono, view):
        # STFT
        mono_stft = torch.stft(mono, n_fft=1024, hop_length=64, 
                               win_length=1024, return_complex=True)
        mono_mag = mono_stft.abs()
        mono_phase = torch.angle(mono_stft)
        
        # View encoding
        view_feat = self.view_encoder(view)
        
        # 1. Magnitude (learned)
        mag_L, mag_R = self.magnitude_net(mono_mag, view_feat)
        
        # 2. Physical ITD
        azimuth = torch.atan2(view[:, 1], view[:, 0])  # y/x
        physical_itd = (self.head_radius / self.sound_speed) * (
            torch.sin(azimuth) + azimuth
        )
        
        # 3. Learned residual (constrained)
        residual = self.residual_itd_net(mono_mag, view_feat)
        residual = torch.tanh(residual) * 0.5  # Limit to [-0.5, 0.5]
        residual = residual * self.freq_mask.unsqueeze(0).unsqueeze(0)  # Freq-dependent
        
        # 4. Total ITD
        total_itd = physical_itd.unsqueeze(-1) + residual  # (B, 1, 513)
        
        # 5. Phase shift
        freq_bins = torch.fft.rfftfreq(1024, 1/48000).to(mono.device)
        phase_shift = 2 * np.pi * freq_bins.unsqueeze(0).unsqueeze(0) * total_itd
        
        # 6. Binaural phase
        phase_L = mono_phase + phase_shift / 2
        phase_R = mono_phase - phase_shift / 2
        
        # 7. Wrap to [-π, π]
        phase_L = torch.atan2(torch.sin(phase_L), torch.cos(phase_L))
        phase_R = torch.atan2(torch.sin(phase_R), torch.cos(phase_R))
        
        # 8. Reconstruct
        Y_L = mag_L * torch.exp(1j * phase_L)
        Y_R = mag_R * torch.exp(1j * phase_R)
        
        # 9. iSTFT
        y_L = torch.istft(Y_L, n_fft=1024, hop_length=64, win_length=1024)
        y_R = torch.istft(Y_R, n_fft=1024, hop_length=64, win_length=1024)
        
        return y_L, y_R, {
            'mag_L': mag_L,
            'mag_R': mag_R,
            'phase_L': phase_L,
            'phase_R': phase_R,
            'residual': residual,
            'physical_itd': physical_itd
        }
```

**參數數量**: ~1.8M (比 E7/E8 的 2.0M 少)

---

## Loss Function

### Complete Loss

```python
def compute_loss(pred, target, outputs, config):
    y_L_pred, y_R_pred = pred
    y_L_gt, y_R_gt = target
    
    # 1. Waveform L2
    waveform_loss = (
        F.mse_loss(y_L_pred, y_L_gt) +
        F.mse_loss(y_R_pred, y_R_gt)
    ) / 2
    
    # 2. Perceptual loss (multi-resolution STFT)
    perceptual_loss = (
        multi_resolution_stft_loss(y_L_pred, y_L_gt) +
        multi_resolution_stft_loss(y_R_pred, y_R_gt)
    ) / 2
    
    # 3. Residual regularization
    residual = outputs['residual']
    residual_reg = torch.mean(residual ** 2)
    
    # 4. Temporal smoothness
    phase_L = outputs['phase_L']
    phase_R = outputs['phase_R']
    temporal_smooth = (
        temporal_smoothness_loss(phase_L) +
        temporal_smoothness_loss(phase_R)
    ) / 2
    
    # 5. Magnitude loss (auxiliary)
    mag_L_pred = outputs['mag_L']
    mag_R_pred = outputs['mag_R']
    mag_L_gt = torch.stft(y_L_gt, ...).abs()
    mag_R_gt = torch.stft(y_R_gt, ...).abs()
    mag_loss = (
        F.l1_loss(mag_L_pred, mag_L_gt) +
        F.l1_loss(mag_R_pred, mag_R_gt)
    ) / 2
    
    # Total loss
    total_loss = (
        10.0 * waveform_loss +      # 主要目標
        5.0 * perceptual_loss +      # 聽覺感知
        1.0 * mag_loss +             # 輔助
        0.01 * residual_reg +        # 正則化
        0.1 * temporal_smooth        # 時間連續性
    )
    
    return total_loss, {
        'waveform': waveform_loss.item(),
        'perceptual': perceptual_loss.item(),
        'magnitude': mag_loss.item(),
        'residual_reg': residual_reg.item(),
        'temporal_smooth': temporal_smooth.item(),
        'total': total_loss.item()
    }
```

---

## 訓練配置

```python
config = {
    # Model
    'model': 'ImprovedResidualPhaseNet',
    
    # Training
    'epochs': 100,
    'batch_size': 16,
    'learning_rate': 3e-4,
    'optimizer': 'Adam',
    'scheduler': 'ReduceLROnPlateau',
    'early_stopping': 15,
    
    # Loss weights
    'loss_weights': {
        'waveform': 10.0,
        'perceptual': 5.0,
        'magnitude': 1.0,
        'residual_reg': 0.01,
        'temporal_smooth': 0.1
    },
    
    # Data
    'chunk_size_ms': 200,
    'overlap': 0.5,
    'sample_rate': 48000,
    
    # STFT
    'n_fft': 1024,
    'hop_length': 64,
    'win_length': 1024,
    
    # Multi-resolution STFT
    'stft_resolutions': [
        {'n_fft': 512, 'hop': 128, 'win': 512},
        {'n_fft': 1024, 'hop': 256, 'win': 1024},
        {'n_fft': 2048, 'hop': 512, 'win': 2048}
    ],
    
    # Regularization
    'gradient_clip': 1.0,
    'weight_decay': 1e-5,
    
    # Logging
    'log_interval': 10,
    'save_interval': 5
}
```

---

## 評估指標

### 主要指標

1. **Waveform L2** (主要目標)
   - 目標: < 0.00070

2. **Phase Correlation** (關鍵改進)
   - 目標: < 0.95 (之前都是 0.999)

3. **IPD Loss**
   - 監控用

4. **Magnitude Loss**
   - 應該和之前一樣好

### 輔助指標

5. **Residual Magnitude**
   - 監控 residual 的大小
   - 應該很小 (< 0.3)

6. **Temporal Smoothness**
   - 監控 phase 跳變
   - 應該很小

7. **Perceptual Loss**
   - 聽覺感知
   - 應該持續下降

---

## 預期結果

### 樂觀情況 (成功)

```
Waveform L2: < 0.00070
Phase L: < 0.95
Phase R: < 0.95
Residual: 0.1-0.3
```

**意義**:
- ✅ 超越 DPATFNet
- ✅ Phase 有明顯改善
- ✅ Residual 學到有用的東西

### 中等情況 (部分成功)

```
Waveform L2: 0.00070-0.00080
Phase L: 0.95-0.98
Phase R: 0.95-0.98
Residual: 0.05-0.15
```

**意義**:
- ⚠️ 接近 DPATFNet
- ⚠️ Phase 有小幅改善
- ⚠️ Residual 有一些作用

### 悲觀情況 (失敗)

```
Waveform L2: > 0.00080
Phase L: > 0.99
Phase R: > 0.99
Residual: < 0.05
```

**意義**:
- ❌ 仍然失敗
- ❌ Phase 沒有改善
- ❌ Residual 沒有學到東西

---

## 成功標準

### 最低標準 (可接受)

- Waveform L2 < 0.00080 (比 Magnitude-only 好)
- Phase L/R < 0.99 (有一些改善)
- Residual 有非零的值

### 理想標準 (論文亮點)

- Waveform L2 < 0.00070 (持平或超越 DPATFNet)
- Phase L/R < 0.95 (明顯改善)
- Residual 在合理範圍 (0.1-0.3)

### 突破標準 (重大創新)

- Waveform L2 < 0.00065 (顯著超越 DPATFNet)
- Phase L/R < 0.90 (大幅改善)
- 聽覺評估明顯更好

---

## 時間規劃

### Day 1: 實現 (今天)

- ✅ 實現 ImprovedResidualPhaseNet
- ✅ 實現 MultiResolutionSTFTLoss
- ✅ 實現 training script
- ✅ 測試 forward pass

### Day 2-3: 訓練

- 🔄 開始訓練
- 🔄 監控指標
- 🔄 調整 hyperparameters (如果需要)

### Day 4: 分析

- 📊 分析結果
- 📊 對比 E0-E8
- 📊 決定下一步

---

## 如果成功

### 下一步

1. **聽覺評估**
   - 主觀評價
   - 對比 DPATFNet

2. **消融實驗**
   - 哪個改進最重要？
   - Residual constraint?
   - Perceptual loss?
   - Frequency mask?

3. **進一步改進**
   - 加入 Attention (方案 2)
   - 加入 Multi-scale (方案 1)
   - 加入 Temporal LSTM (方案 4)

4. **撰寫論文**
   - Positive results
   - 方法創新
   - 顯著改進

---

## 如果失敗

### 分析

1. **為什麼失敗？**
   - Residual 仍然沒學到？
   - Perceptual loss 不夠？
   - 約束太強？

2. **還有什麼可以試？**
   - 放鬆約束？
   - 不同的 loss？
   - 完全不同的方法？

3. **是否該放棄 Phase 學習？**
   - 9 種方法都失敗
   - 可能真的不可能
   - 轉向 Magnitude-only

### 論文策略

- 寫負面結果論文
- 9 種方法的系統性失敗
- 資訊理論解釋
- 為社群節省時間

---

## 立即行動

### 今天完成

1. ✅ 實現 `src/models_improved_residual.py`
2. ✅ 實現 `src/losses_perceptual.py`
3. ✅ 實現 `training/train_improved_residual.py`
4. ✅ 測試模型
5. ✅ 開始訓練

### 明天檢查

- 訓練是否正常
- Loss 是否下降
- Phase 是否改善

---

**Let's do this!** 🚀
