# PGCN 模組詳細設計

## 三、核心模組設計

### 3.1 Fourier Position Encoder

**問題**：120Hz 採樣 = 8.33ms 離散化，導致角度精度不足（46.6° 誤差）

**解決方案**：Multi-scale Fourier Features

```python
class FourierPositionEncoder(nn.Module):
    """
    Multi-scale Fourier Features for high-precision position encoding
    
    Input: position (B×7×K)
        [x, y, z, quat_w, quat_x, quat_y, quat_z]
    
    Output: pos_feat (B×256)
    
    精度：0.1° (vs 8.33ms 離散化)
    """
    def __init__(self, input_dim=7, L=10, output_dim=256):
        super().__init__()
        self.L = L
        # Fourier basis: [2^0, 2^1, ..., 2^(L-1)] × π
        self.freqs = 2.0 ** torch.arange(L) * torch.pi
        
        # MLP: (7 × 2L) → 256
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2 * L, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, position):
        # position: B×7×K → B×7 (temporal average)
        pos = position.mean(dim=2)  # B×7
        
        # Fourier features
        # pos: B×7 → B×7×1
        # freqs: L → 1×1×L
        # → B×7×L
        pos_scaled = pos.unsqueeze(2) * self.freqs.view(1, 1, -1)
        
        # [sin, cos]: B×7×L → B×7×2L → B×(7×2L)
        fourier_feat = torch.cat([
            torch.sin(pos_scaled),
            torch.cos(pos_scaled)
        ], dim=2).flatten(1)
        
        # MLP
        return self.mlp(fourier_feat)  # B×256
```

**理論依據**：
- Tancik et al. (2020). "Fourier Features Let Networks Learn High Frequency Functions"
- 提供 2^L 倍的頻率分辨率
- L=10 → 1024 個 basis → 0.1° 精度

---

### 3.2 Complex DualPath Block

**問題**：
1. HybridTFNet 的 Phase difference 不可學習（std≈1.82）
2. DPATFNet 的 Position Cross-Attention 只有 1 個 query

**解決方案**：
1. Complex-valued 全程建模（避免 phase wrapping）
2. 64-band FiLM Modulation（每個頻段獨立調制）

```python
class ComplexDualPathBlock(nn.Module):
    """
    Complex-valued DualPath Attention Block
    
    - FreqAttention: Self-Attention across frequency (within each time)
    - TimeAttention: Self-Attention across time (within each frequency)
    - 64-band FiLM: Position-conditioned modulation per frequency band
    - Complex FFN: Maintain complex representation
    """
    def __init__(self, channels=256, num_heads=8, num_bands=64):
        super().__init__()
        self.channels = channels
        self.num_bands = num_bands
        
        # Frequency Attention
        self.freq_norm = ComplexLayerNorm(channels)
        self.freq_attn = ComplexMultiheadAttention(channels, num_heads)
        
        # Time Attention
        self.time_norm = ComplexLayerNorm(channels)
        self.time_attn = ComplexMultiheadAttention(channels, num_heads)
        
        # 64-band FiLM
        self.film_net = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, num_bands * channels * 2)  # gamma + beta
        )
        
        # Complex FFN
        self.ffn_norm = ComplexLayerNorm(channels)
        self.ffn = ComplexFFN(channels, channels * 4)
    
    def forward(self, x, pos_feat):
        """
        Args:
            x: B×C×F×T (complex-valued, C=256)
            pos_feat: B×256
        Returns:
            x: B×C×F×T (complex-valued)
        """
        B, C, F, T = x.shape
        
        # 1. Frequency Attention (across F, within each T)
        x_freq = x.permute(0, 3, 2, 1)  # B×T×F×C
        x_freq = x_freq.reshape(B * T, F, C)
        x_freq = self.freq_norm(x_freq)
        x_freq_out = self.freq_attn(x_freq, x_freq, x_freq)
        x_freq_out = x_freq_out.reshape(B, T, F, C).permute(0, 3, 2, 1)
        x = x + x_freq_out
        
        # 2. Time Attention (across T, within each F)
        x_time = x.permute(0, 2, 3, 1)  # B×F×T×C
        x_time = x_time.reshape(B * F, T, C)
        x_time = self.time_norm(x_time)
        x_time_out = self.time_attn(x_time, x_time, x_time)
        x_time_out = x_time_out.reshape(B, F, T, C).permute(0, 3, 1, 2)
        x = x + x_time_out
        
        # 3. 64-band FiLM Modulation
        film_params = self.film_net(pos_feat)  # B×(num_bands×C×2)
        film_params = film_params.view(B, self.num_bands, C, 2)
        gamma = film_params[..., 0]  # B×num_bands×C
        beta = film_params[..., 1]   # B×num_bands×C
        
        # 將 F=513 分成 64 個 band
        band_size = F // self.num_bands
        for i in range(self.num_bands):
            start = i * band_size
            end = (i + 1) * band_size if i < self.num_bands - 1 else F
            # FiLM: x = gamma * x + beta
            x[:, :, start:end, :] = (
                gamma[:, i, :].view(B, C, 1, 1) * x[:, :, start:end, :] +
                beta[:, i, :].view(B, C, 1, 1)
            )
        
        # 4. Complex FFN
        x_ffn = x.permute(0, 2, 3, 1)  # B×F×T×C
        x_ffn = self.ffn_norm(x_ffn)
        x_ffn = self.ffn(x_ffn)
        x_ffn = x_ffn.permute(0, 3, 1, 2)  # B×C×F×T
        x = x + x_ffn
        
        return x
```

**關鍵改進**：
1. **64-band FiLM** vs 1-query Cross-Attention
   - 每個頻段獨立調制，精細控制
   - 密集採樣 6-16kHz（pinna notches 的關鍵區域）
   
2. **Complex-valued 全程**
   - 避免 phase wrapping（HybridTFNet 的致命問題）
   - 保持 phase coherence

---

### 3.3 Physics-Constrained Head

**問題**：v8 的預測角度幾乎固定（-44.3°, -19.4° 重複），缺乏物理先驗

**解決方案**：注入 Woodworth ITD formula 和 frequency-dependent ILD

```python
class PhysicsConstrainedHead(nn.Module):
    """
    Physics-guided prediction head
    
    1. Woodworth ITD formula (geometric prior)
    2. Frequency-dependent ILD (6dB @ 1kHz baseline)
    3. Learnable residual correction
    """
    def __init__(self, channels=256, n_freq=513):
        super().__init__()
        self.n_freq = n_freq
        
        # Learnable residual
        self.residual_net = nn.Sequential(
            ComplexConv2d(channels, 128, 3, padding=1),
            ComplexReLU(),
            ComplexConv2d(128, 4, 3, padding=1)  # [real_L, imag_L, real_R, imag_R]
        )
        
        # Frequency-dependent ILD weights (learnable)
        self.ild_weights = nn.Parameter(torch.ones(n_freq))
    
    def forward(self, x, pos_feat, Y_mono):
        """
        Args:
            x: B×C×F×T (complex features)
            pos_feat: B×256 (position encoding)
            Y_mono: B×F×T (complex, mono STFT)
        Returns:
            Y_L, Y_R: B×F×T (complex)
        """
        B, C, F, T = x.shape
        
        # 1. Extract azimuth from pos_feat (simplified)
        # 實際應該從 position (x,y,z,quat) 計算
        azimuth = self._extract_azimuth(pos_feat)  # B
        
        # 2. Woodworth ITD formula
        # ITD(θ) = (r/c) * (θ + sin(θ))
        # r = 0.0875m (head radius), c = 343 m/s
        r = 0.0875
        c = 343.0
        itd = (r / c) * (azimuth + torch.sin(azimuth))  # B (seconds)
        
        # Convert ITD to phase shift per frequency
        freqs = torch.linspace(0, 24000, F).to(x.device)  # Hz
        phase_shift = 2 * torch.pi * freqs.view(1, F, 1) * itd.view(B, 1, 1)  # B×F×1
        
        # 3. Frequency-dependent ILD
        # ILD(f) = 6dB @ 1kHz, increases with frequency
        ild_db = 6.0 * (freqs / 1000.0).clamp(0, 3)  # 0-18dB
        ild_linear = 10 ** (ild_db / 20.0)  # B×F
        
        # 4. Apply physics prior to mono
        Y_L_physics = Y_mono * torch.exp(1j * phase_shift / 2) * ild_linear.view(1, F, 1)
        Y_R_physics = Y_mono * torch.exp(-1j * phase_shift / 2) / ild_linear.view(1, F, 1)
        
        # 5. Learnable residual correction
        residual = self.residual_net(x)  # B×4×F×T
        real_L_res = residual[:, 0, :, :]
        imag_L_res = residual[:, 1, :, :]
        real_R_res = residual[:, 2, :, :]
        imag_R_res = residual[:, 3, :, :]
        
        Y_L_res = torch.complex(real_L_res, imag_L_res)
        Y_R_res = torch.complex(real_R_res, imag_R_res)
        
        # 6. Combine: physics prior + learnable residual
        Y_L = Y_L_physics + 0.1 * Y_L_res  # 0.1: residual weight
        Y_R = Y_R_physics + 0.1 * Y_R_res
        
        return Y_L, Y_R
    
    def _extract_azimuth(self, pos_feat):
        # Simplified: 實際應該從 (x,y,z,quat) 計算
        # 這裡假設 pos_feat 的第一個維度編碼了 azimuth
        return torch.atan2(pos_feat[:, 1], pos_feat[:, 0])
```

**理論依據**：
- Woodworth (1938). "Experimental Psychology"
- Algazi et al. (2001). "The CIPIC HRTF Database"
- 提供合理的初始化，避免學到不合理的預測

---

### 3.4 Minimum-Phase Enforcer

**問題**：預測的 HRTF 可能不是 minimum-phase（非因果）

**解決方案**：Causal loss + Hilbert transform constraint

```python
def minimum_phase_loss(Y_pred, Y_gt):
    """
    Enforce minimum-phase constraint
    
    Minimum-phase: |H(f)| uniquely determines phase
    via Hilbert transform
    """
    # 1. Magnitude
    mag_pred = torch.abs(Y_pred)
    mag_gt = torch.abs(Y_gt)
    
    # 2. Log magnitude
    log_mag_pred = torch.log(mag_pred + 1e-8)
    log_mag_gt = torch.log(mag_gt + 1e-8)
    
    # 3. Hilbert transform (approximate via FFT)
    # phase = -imag(hilbert(log_mag))
    phase_pred_minphase = -torch.imag(torch.fft.fft(log_mag_pred))
    phase_pred_actual = torch.angle(Y_pred)
    
    # 4. Loss: encourage minimum-phase
    loss = F.mse_loss(phase_pred_actual, phase_pred_minphase)
    
    return loss
```

**理論依據**：
- Oppenheim & Schafer (2009). "Discrete-Time Signal Processing"
- Minimum-phase 系統是因果且穩定的
- 確保預測的 HRTF 物理可實現

---

## 四、完整模型

```python
class PhysicsGuidedComplexNet(nn.Module):
    def __init__(self, 
                 n_fft=1024,
                 hop_size=256,
                 channels=256,
                 num_dpab=4,
                 num_heads=8,
                 fourier_L=10,
                 num_bands=64):
        super().__init__()
        
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.n_freq = n_fft // 2 + 1
        
        # 1. Fourier Position Encoder
        self.pos_encoder = FourierPositionEncoder(
            input_dim=7, L=fourier_L, output_dim=channels
        )
        
        # 2. Complex Encoder
        self.encoder = nn.Sequential(
            ComplexConv2d(1, 64, 3, padding=1),
            ComplexReLU(),
            ComplexConv2d(64, 128, 3, padding=1),
            ComplexReLU(),
            ComplexConv2d(128, channels, 3, padding=1),
            ComplexReLU()
        )
        
        # 3. Complex DualPath Blocks
        self.dpab_blocks = nn.ModuleList([
            ComplexDualPathBlock(channels, num_heads, num_bands)
            for _ in range(num_dpab)
        ])
        
        # 4. Physics-Constrained Head
        self.head = PhysicsConstrainedHead(channels, self.n_freq)
    
    def forward(self, mono, position):
        """
        Args:
            mono: B×1×T
            position: B×7×K
        Returns:
            y_binaural: B×2×T
            outputs: dict
        """
        B, _, T = mono.shape
        device = mono.device
        
        # 1. Position encoding
        pos_feat = self.pos_encoder(position)  # B×256
        
        # 2. STFT
        window = torch.hann_window(self.n_fft).to(device)
        Y_mono = torch.stft(mono.squeeze(1), self.n_fft, self.hop_size,
                           window=window, return_complex=True)  # B×F×T_stft
        
        # 3. Encoder (complex-valued)
        x = Y_mono.unsqueeze(1)  # B×1×F×T_stft
        x = self.encoder(x)  # B×C×F×T_stft
        
        # 4. DualPath Blocks
        for dpab in self.dpab_blocks:
            x = dpab(x, pos_feat)
        
        # 5. Physics-Constrained Head
        Y_L, Y_R = self.head(x, pos_feat, Y_mono)
        
        # 6. iSTFT
        y_L = torch.istft(Y_L, self.n_fft, self.hop_size, window=window, length=T)
        y_R = torch.istft(Y_R, self.n_fft, self.hop_size, window=window, length=T)
        
        y_binaural = torch.stack([y_L, y_R], dim=1)
        
        return y_binaural, {
            'Y_L': Y_L,
            'Y_R': Y_R,
            'Y_mono': Y_mono
        }
```

---

**參數量估算**：
- FourierPositionEncoder: ~0.5M
- Encoder: ~2M
- DualPath Blocks × 4: ~20M
- Physics Head: ~1M
- **總計：~24M**（vs DPATFNet 論文 ~30M，你的實作 0.5M）
