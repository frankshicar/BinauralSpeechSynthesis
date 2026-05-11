# PGCN 訓練策略

## 五、Loss 設計

### 5.1 Multi-objective Loss

```python
total_loss = (
    0.15 × L_time +           # 時域 L2
    0.40 × L_complex_stft +   # Complex STFT (主要)
    0.20 × L_multi_res +      # Multi-resolution STFT
    0.10 × L_physics +        # Physics constraints
    0.10 × L_causal +         # Minimum-phase
    0.05 × L_lipschitz        # Lipschitz regularization
)
```

### 5.2 各項 Loss 詳細說明

#### 1. 時域 L2 (0.15)

```python
def time_domain_loss(pred, target):
    return F.mse_loss(pred, target)
```

**權重理由**：
- 保證基本的波形重建質量
- 權重較低（0.15），避免壓制 phase 學習
- 對比：你的 DPATFNet 用 10.0（過高）

---

#### 2. Complex STFT Loss (0.40) ⭐ 主要

```python
def complex_stft_loss(Y_pred, Y_gt):
    """
    Complex MSE: |Y_pred - Y_gt|^2
    = (real_pred - real_gt)^2 + (imag_pred - imag_gt)^2
    
    優點：
    - 同時約束 magnitude 和 phase
    - 避免 phase wrapping 問題
    - 對 phase 敏感度高
    """
    loss_real = F.mse_loss(Y_pred.real, Y_gt.real)
    loss_imag = F.mse_loss(Y_pred.imag, Y_gt.imag)
    return loss_real + loss_imag
```

**權重理由**：
- 0.40 是主要 loss（vs 你的 DPATFNet 用 1.0 complex + 10.0 time）
- 直接約束 complex spectrum，對 phase 最敏感
- 避免 HybridTFNet 的 phase difference 問題

---

#### 3. Multi-resolution STFT Loss (0.20)

```python
def multi_resolution_stft_loss(pred, target):
    """
    多尺度 STFT loss，捕捉不同時頻分辨率
    
    n_fft = [512, 1024, 2048]
    hop_size = [128, 256, 512]
    """
    loss = 0
    configs = [
        (512, 128),
        (1024, 256),
        (2048, 512)
    ]
    
    for n_fft, hop_size in configs:
        window = torch.hann_window(n_fft).to(pred.device)
        
        Y_pred = torch.stft(pred, n_fft, hop_size, window=window, return_complex=True)
        Y_gt = torch.stft(target, n_fft, hop_size, window=window, return_complex=True)
        
        # Magnitude + Phase
        loss += F.mse_loss(torch.abs(Y_pred), torch.abs(Y_gt))
        loss += F.mse_loss(torch.angle(Y_pred), torch.angle(Y_gt))
    
    return loss / len(configs)
```

**權重理由**：
- 0.20：補充不同尺度的資訊
- 低頻用大 n_fft（2048），高頻用小 n_fft（512）

---

#### 4. Physics Constraints Loss (0.10)

```python
def physics_constraints_loss(Y_L, Y_R, position):
    """
    物理約束：
    1. ITD 應該符合 Woodworth formula
    2. ILD 應該隨頻率增加
    3. Cone of confusion: 前後方向的 ILD 應該不同
    """
    # 1. ITD constraint
    gcc_phat = compute_gcc_phat(Y_L, Y_R)  # B×T_stft
    itd_pred = extract_itd_from_gcc(gcc_phat)  # B
    
    azimuth = extract_azimuth(position)  # B
    itd_gt = woodworth_formula(azimuth)  # B
    
    loss_itd = F.mse_loss(itd_pred, itd_gt)
    
    # 2. ILD constraint (應該隨頻率增加)
    ild = 20 * torch.log10(torch.abs(Y_L) / (torch.abs(Y_R) + 1e-8))  # B×F×T
    ild_mean_freq = ild.mean(dim=2)  # B×F
    
    # ILD 應該單調遞增（低頻小，高頻大）
    ild_diff = ild_mean_freq[:, 1:] - ild_mean_freq[:, :-1]
    loss_ild_monotonic = F.relu(-ild_diff).mean()  # 懲罰遞減
    
    # 3. Cone of confusion (前後方向的 spectral cue 不同)
    # 6-16kHz 的 notch 位置應該不同
    elevation = extract_elevation(position)  # B
    notch_freq_pred = find_notch_frequency(Y_L, Y_R)  # B
    notch_freq_gt = elevation_to_notch(elevation)  # B (empirical)
    
    loss_notch = F.mse_loss(notch_freq_pred, notch_freq_gt)
    
    return loss_itd + 0.5 * loss_ild_monotonic + 0.5 * loss_notch
```

**權重理由**：
- 0.10：提供物理先驗，避免不合理預測
- 不能太高，否則限制模型學習能力

---

#### 5. Minimum-Phase Loss (0.10)

```python
def minimum_phase_loss(Y_pred):
    """
    確保預測的 HRTF 是 minimum-phase（因果）
    """
    # Log magnitude
    log_mag = torch.log(torch.abs(Y_pred) + 1e-8)
    
    # Hilbert transform (via FFT)
    phase_minphase = -torch.imag(torch.fft.fft(log_mag))
    phase_actual = torch.angle(Y_pred)
    
    # Wrap to [-π, π]
    phase_diff = torch.angle(torch.exp(1j * (phase_actual - phase_minphase)))
    
    return F.mse_loss(phase_diff, torch.zeros_like(phase_diff))
```

**權重理由**：
- 0.10：確保物理可實現性
- 避免預測出非因果的 HRTF

---

#### 6. Lipschitz Regularization (0.05)

```python
def lipschitz_regularization(model):
    """
    限制模型的 Lipschitz constant，提升訓練穩定性
    
    對 position 的微小變化，輸出不應該劇烈變化
    """
    # Spectral normalization on weights
    loss = 0
    for module in model.modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            # Spectral norm of weight matrix
            W = module.weight
            sigma = torch.svd(W.view(W.size(0), -1))[1].max()
            loss += F.relu(sigma - 1.0)  # 懲罰 sigma > 1
    
    return loss
```

**權重理由**：
- 0.05：提升訓練穩定性
- 避免對 position 的微小變化過度敏感

---

## 六、訓練策略

### 6.1 Single-stage + Curriculum Learning

**為什麼不用多階段？**
- v6/v7/v8 的多階段訓練有梯度衝突問題
- Stage 切換時 loss 跳升，LR 調度混亂
- Single-stage 更簡單，更穩定

**Curriculum Learning**：
```python
# Epoch 1-50: Frontal only (warm-up)
# 只用 azimuth ∈ [-30°, +30°] 的數據
# 目標：學好基本的 ITD/ILD

# Epoch 51-150: Full space
# 60% 均勻採樣 + 40% 困難角度
# 困難角度：±90°, ±180°（cone of confusion）

# Epoch 151-200: Fine-tuning
# Physics loss 權重提升：0.10 → 0.15
# 確保物理合理性
```

### 6.2 學習率調度

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=50,      # 每 50 epochs 一個 cycle
    T_mult=1,
    eta_min=1e-6
)
```

**理由**：
- Cosine decay 比 ReduceLROnPlateau 穩定
- Warm restart 幫助跳出 local minima
- v7 已驗證有效

### 6.3 數據處理

```python
# STFT
n_fft = 1024
hop_size = 256
sample_rate = 48000

# Training
batch_size = 16
chunk_size = 48000  # 1 秒
num_workers = 4

# Data augmentation
augmentations = [
    RandomGain(min_gain=-3, max_gain=3),      # ±3dB
    RandomTimeShift(max_shift=0.01),          # ±10ms
    # 不做 pitch shift（會改變 HRTF）
]
```

### 6.4 記憶體優化

```python
# Gradient checkpointing
use_checkpoint = True  # 在 DualPath blocks

# Mixed precision
use_amp = True  # FP16

# 預期記憶體：
# batch_size=16, chunk=48000 (1s)
# hop_size=256 → T_stft ≈ 188
# channels=256, num_dpab=4
# → ~14GB (vs 你的 hop=64 會 OOM)
```

---

## 七、評估指標

### 7.1 訓練時監控

```python
metrics = {
    # Loss components
    'loss_time': ...,
    'loss_complex_stft': ...,
    'loss_multi_res': ...,
    'loss_physics': ...,
    'loss_causal': ...,
    'loss_lipschitz': ...,
    
    # Phase metrics
    'phase_L_error': ...,
    'phase_R_error': ...,
    'ipd_error': ...,
    
    # Spatial metrics
    'itd_error': ...,      # μs
    'ild_error': ...,      # dB
    'angle_error': ...,    # degrees
}
```

### 7.2 測試集評估

```python
# test_13angles: 13 個固定角度
angles = [-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90]

for angle in angles:
    # 1. L2 loss
    l2 = F.mse_loss(pred, target)
    
    # 2. Phase error
    phase_L_err = F.mse_loss(torch.angle(Y_L_pred), torch.angle(Y_L_gt))
    phase_R_err = F.mse_loss(torch.angle(Y_R_pred), torch.angle(Y_R_gt))
    
    # 3. ITD error (via GCC-PHAT)
    itd_pred = compute_itd(Y_L_pred, Y_R_pred)
    itd_gt = compute_itd(Y_L_gt, Y_R_gt)
    itd_err = abs(itd_pred - itd_gt)  # μs
    
    # 4. ILD error
    ild_pred = 20 * log10(|Y_L_pred| / |Y_R_pred|)
    ild_gt = 20 * log10(|Y_L_gt| / |Y_R_gt|)
    ild_err = abs(ild_pred - ild_gt)  # dB
    
    # 5. Angle error (via DOA estimation)
    angle_pred = estimate_doa(Y_L_pred, Y_R_pred)
    angle_err = abs(angle_pred - angle)  # degrees
```

### 7.3 成功標準

| 指標 | DPATFNet 論文 | 你的實作 | PGCN 目標 | 備註 |
|------|--------------|---------|----------|------|
| L2 (×10⁻³) | ~0.144 | 0.180 | **< 0.14** | 比 Meta large 好 |
| Phase | 0.70 | 3.28 | **< 0.65** | 比論文好 7% |
| ITD (μs) | ~200 | 772 | **< 150** | 比論文好 25% |
| ILD (dB) | ~2 | 6.03 | **< 1.8** | 比論文好 10% |
| 角度誤差 (°) | ? | 46.6 | **< 2** | 人類 MAA |

---

## 八、訓練監控與 Debug

### 8.1 關鍵檢查點

**Epoch 10**（Warm-up 結束）：
- [ ] Loss 開始下降？
- [ ] Phase error < 2.0？
- [ ] 沒有 NaN/Inf？

**Epoch 50**（Frontal only 結束）：
- [ ] Frontal angles (±30°) 的 angle error < 5°？
- [ ] Phase error < 1.5？
- [ ] ITD error < 300μs？

**Epoch 100**（Full space 中期）：
- [ ] All angles 的 angle error < 10°？
- [ ] Phase error < 1.0？
- [ ] ITD error < 200μs？

**Epoch 150**（Full space 結束）：
- [ ] Phase error < 0.8？
- [ ] Angle error < 5°？

**Epoch 200**（Fine-tuning 結束）：
- [ ] 達到目標指標？

### 8.2 常見問題與解決

| 問題 | 可能原因 | 解決方案 |
|------|---------|---------|
| Loss 不下降 | LR 太大/太小 | 調整 LR（3e-4 → 1e-4 或 1e-3） |
| Phase error 卡在 2.0 | Complex loss 權重不足 | 提高到 0.5 |
| Angle error 很大 | Fourier Features 不夠 | 增加 L（10 → 12） |
| 記憶體 OOM | Batch size 太大 | 降到 8 或用 gradient checkpointing |
| Physics loss 爆炸 | 權重太高 | 降到 0.05 |
| 預測角度固定 | Position encoding 失效 | 檢查 FourierEncoder 的梯度 |

---

## 九、與失敗案例的對比

| 失敗案例 | 問題 | PGCN 的解決方案 |
|---------|------|----------------|
| **v8 系列** | DPAB 實作錯誤，從 warped 出發 | ✅ 正確的 DualPath + 從 mono 出發 |
| **HybridTFNet** | Phase difference 不可學習（std≈1.82） | ✅ Complex representation，避免 wrapping |
| **DPATFNet 實作** | hop=64 記憶體爆炸，容量不足 | ✅ hop=256，channels=256 |
| **v8 角度誤差** | Position encoding 精度不足（46.6°） | ✅ Fourier Features (L=10)，0.1° 精度 |
| **v8 預測固定** | 缺乏物理先驗 | ✅ Physics-constrained head |
| **所有版本** | 多階段梯度衝突 | ✅ Single-stage + curriculum |

---

**總結**：PGCN 的訓練策略針對所有已知失敗模式進行了設計，預期可以達到目標指標。
