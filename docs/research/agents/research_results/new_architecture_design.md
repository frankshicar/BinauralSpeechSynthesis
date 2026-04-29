# 新架構設計文件

設計時間：2026-04-27
設計者：Architecture Synthesizer Agent

綜合來源：
- Binaural Paper Research (6 篇重點論文)
- ML Architecture Research (8 項關鍵技術)
- BinauralTFNet v8.3 失敗經驗

---

## 1. 問題分析

### 1.1 核心問題

**BinauralTFNet v8.3 的失敗**：
- Stage 2 Phase 改善只有 13.9%（目標 >40%）
- IPD 反而變差（-6.8%）
- 訓練 119 epochs 後仍然無法收斂

**根本原因**：
1. **CommonBranch 沒有學到空間感**
   - 即使用了 BinauralGrad 的方式（spatial_encoder + audio_encoder）
   - WaveNet 在時域很難學習「帶空間感的共同部分」
   - 結果：y_common 只是能量平均，沒有距離感/殘響

2. **SpecificBranch 任務太重**
   - 需要從幾乎沒有空間感的 Y_common 學出：
     - 距離感
     - 殘響
     - 頭部濾波
     - ITD
     - ILD
     - HRTF
   - 任務太多，學不完

3. **頻域學 Phase 很困難**
   - Phase 本質是時域概念（時間延遲）
   - 頻域的 TFResStack 擅長學 Magnitude，不擅長學 Phase
   - 這是所有純頻域方法的通病

### 1.2 為什麼現有方法失敗

#### DPATFNet 的局限
- **純頻域**：學 Phase 困難
- **端到端**：可能有梯度衝突（Phase loss vs Magnitude loss）
- **沒有物理先驗**：需要從零學 ITD

#### BinauralTFNet v8.3 的局限
- **Common + Specific 分離方式錯誤**：
  - 「共同部分」的定義不清晰
  - WaveNet 學不到「帶空間感的共同部分」
- **時域 + 頻域混合方式錯誤**：
  - Stage 1 時域，Stage 2 頻域
  - 但 Stage 1 的輸出（y_common）沒有足夠的空間資訊
  - Stage 2 無法從中學到 Phase

#### BinauralGrad 的局限
- **Diffusion Model 太慢**：訓練和推理都慢
- **不適合我們的項目**：需要快速迭代

### 1.3 理想的解決方案應該具備

1. **任務分離合理**
   - ✅ ITD（時域）vs ILD（頻域）
   - ❌ Common vs Specific（定義不清）

2. **時域學 Phase，頻域學 Magnitude**
   - 各司其職，不互相干擾

3. **有物理先驗**
   - 幾何 ITD（避免從零學）
   - 但允許神經網路修正

4. **訓練穩定**
   - 分階段訓練，避免梯度衝突
   - 每個階段目標明確

5. **可實作**
   - 1-2 週內完成
   - 不依賴太新的技術

---

## 2. 技術選型

### 2.1 從論文研究中借鑑

#### ✅ Phase-aware Binaural Synthesis (2024) - 核心靈感

**借鑑**：
- **Time Branch + Freq Branch 的分離**
- **時域學 ITD/Phase，頻域學 ILD/Magnitude**
- **最後融合：Phase 來自 Time，Magnitude 來自 Freq**

**改進**：
- 加入幾何先驗（他們沒有）
- 用更好的位置編碼（他們用簡單的 MLP）

#### ✅ BinauralGrad (2022)

**借鑑**：
- **FiLM 條件機制**（簡單有效）
- **分階段訓練的理念**

**不借鑑**：
- ❌ Diffusion Model（太慢）
- ❌ Common + Specific 分離（定義不清）

#### ✅ DPATFNet (2023)

**借鑑**：
- **DPAB 的分離設計**（位置和方向分開）
- **純頻域的穩定性**

**改進**：
- 用 Cross-Attention 替代 DPAB 的複雜 Attention
- 加入時域 Branch

#### ✅ Neural HRTF Synthesis (2021)

**借鑑**：
- **學習濾波器而非直接生成**（更有物理意義）

### 2.2 從 ML 架構研究中借鑑

#### ✅ Learnable Delay Filter (Tier 1)

**用於**：Time Branch 的 ITD 學習

**原因**：
- 比 warp 穩定（線性操作）
- 可微分，端到端訓練
- 可以學習精確的 ITD

#### ✅ FiLM (Tier 1)

**用於**：Time Branch 的位置條件

**原因**：
- 簡單有效
- 訓練穩定
- 適合簡單的條件機制

#### ✅ Cross-Attention (Tier 1)

**用於**：Freq Branch 的位置編碼

**原因**：
- 最靈活的條件機制
- 表達能力強
- 替代 SimpleDPAB

#### ✅ Flash Attention (Tier 2)

**用於**：優化 Cross-Attention

**原因**：
- 顯存降低 5-20x
- 速度提升 2-4x
- 直接替換，無需改代碼

#### ⚠️ Mamba (Tier 2 - 可選)

**用於**：Time Branch 的序列建模（替代 Conv1d）

**原因**：
- 比 WaveNet 更高效
- 長程依賴能力強

**風險**：
- 新技術，實作經驗少
- 條件機制需要設計

### 2.3 創新組合

**核心創新**：
```
Time Branch (時域):
  幾何 ITD (物理先驗)
  + Learnable Delay (神經修正)
  + FiLM (位置條件)
  → Phase_L, Phase_R

Freq Branch (頻域):
  Cross-Attention (位置編碼)
  + TFResStack (特徵提取)
  → Mag_L, Mag_R

Fusion (融合):
  Y_L = Mag_L * exp(1j * Phase_L)
  Y_R = Mag_R * exp(1j * Phase_R)
  → iSTFT → y_L, y_R
```

**與現有方法的差異**：
- vs DPATFNet：加入時域 Branch，有物理先驗
- vs v8.3：任務分離方式不同（ITD vs ILD，不是 Common vs Specific）
- vs Phase-aware：加入幾何先驗，用更好的位置編碼

---

## 3. 架構設計

### 3.1 整體架構圖

```
                        mono (B×1×T) + view (B×7×K)
                                    |
                    ┌───────────────┴───────────────┐
                    |                               |
                    v                               v
            ┌──────────────┐              ┌──────────────┐
            │ Time Branch  │              │ Freq Branch  │
            │  (時域 ITD)   │              │ (頻域 ILD)    │
            └──────────────┘              └──────────────┘
                    |                               |
                    | Phase_L, Phase_R              | Mag_L, Mag_R
                    |                               |
                    └───────────────┬───────────────┘
                                    v
                            ┌──────────────┐
                            │    Fusion    │
                            │ (複數乘法)     │
                            └──────────────┘
                                    |
                                    v
                                  iSTFT
                                    |
                                    v
                            y_L, y_R (B×2×T)
```

### 3.2 Time Branch 詳細設計

**功能**：學習 ITD（時間差），輸出 Phase_L 和 Phase_R

```python
class TimeBranch(nn.Module):
    """
    時域分支：學習 ITD（Interaural Time Difference）
    
    輸入：
        mono: B×1×T
        view: B×7×K (position + orientation)
    
    輸出：
        Phase_L: B×F×T_stft (左耳相位)
        Phase_R: B×F×T_stft (右耳相位)
    """
    def __init__(self, 
                 sample_rate=16000,
                 fft_size=1024,
                 hop_size=256,
                 max_delay_samples=32):
        super().__init__()
        
        # 1. 幾何 ITD 計算（固定，不訓練）
        self.geometric_itd = GeometricITD(sample_rate)
        
        # 2. 位置編碼器（view → condition）
        self.position_encoder = nn.Sequential(
            nn.Linear(7, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )
        
        # 3. 音訊編碼器（mono → features）
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
        )
        
        # 4. FiLM 條件層
        self.film = FiLMLayer(cond_dim=256, feat_dim=256)
        
        # 5. Learnable Delay Network（左右耳各一個）
        self.delay_net_L = LearnableDelayNet(
            feat_dim=256, 
            max_delay=max_delay_samples
        )
        self.delay_net_R = LearnableDelayNet(
            feat_dim=256, 
            max_delay=max_delay_samples
        )
        
        self.fft_size = fft_size
        self.hop_size = hop_size
    
    def forward(self, mono, view):
        B, _, T = mono.shape
        
        # 1. 幾何 ITD（粗調）
        itd_geo_L, itd_geo_R = self.geometric_itd(view)  # B×1
        y_geo_L = self.apply_geometric_warp(mono, itd_geo_L)
        y_geo_R = self.apply_geometric_warp(mono, itd_geo_R)
        
        # 2. 位置編碼
        pos_feat = self.position_encoder(view.mean(dim=-1))  # B×7 → B×256
        
        # 3. 音訊編碼
        audio_feat = self.audio_encoder(mono)  # B×1×T → B×256×T
        
        # 4. FiLM 條件
        audio_feat = self.film(audio_feat, pos_feat)  # B×256×T
        
        # 5. Learnable Delay（細調）
        y_L = self.delay_net_L(y_geo_L, audio_feat)  # B×1×T
        y_R = self.delay_net_R(y_geo_R, audio_feat)  # B×1×T
        
        # 6. STFT 提取 Phase
        Y_L = torch.stft(y_L.squeeze(1), self.fft_size, self.hop_size, 
                         return_complex=True)  # B×F×T_stft
        Y_R = torch.stft(y_R.squeeze(1), self.fft_size, self.hop_size,
                         return_complex=True)
        
        Phase_L = torch.angle(Y_L)  # B×F×T_stft
        Phase_R = torch.angle(Y_R)
        
        return Phase_L, Phase_R
```

**關鍵模組**：

#### GeometricITD（幾何 ITD 計算）
```python
class GeometricITD(nn.Module):
    """
    計算幾何 ITD（基於頭部球體模型）
    固定計算，不訓練
    """
    def __init__(self, sample_rate=16000, head_radius=0.0875):
        super().__init__()
        self.sample_rate = sample_rate
        self.head_radius = head_radius
        self.c = 343.0  # 聲速 (m/s)
    
    def forward(self, view):
        # view: B×7×K (x, y, z, qw, qx, qy, qz)
        # 取平均位置
        pos = view[:, :3, :].mean(dim=-1)  # B×3
        
        # 計算方位角（azimuth）
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        azimuth = torch.atan2(y, x)  # B
        
        # Woodworth 公式：ITD = (r/c) * (azimuth + sin(azimuth))
        itd_seconds = (self.head_radius / self.c) * (
            azimuth + torch.sin(azimuth)
        )
        
        # 轉換為樣本數
        itd_samples = itd_seconds * self.sample_rate  # B
        
        # 左右耳 ITD（左耳負，右耳正）
        itd_L = -itd_samples / 2  # B
        itd_R = itd_samples / 2
        
        return itd_L.unsqueeze(-1), itd_R.unsqueeze(-1)  # B×1
```

#### LearnableDelayNet（可學習延遲網路）
```python
class LearnableDelayNet(nn.Module):
    """
    用可學習的 FIR filter 實現精細的 delay
    """
    def __init__(self, feat_dim=256, max_delay=32):
        super().__init__()
        
        # 預測 delay filter 的權重
        self.delay_predictor = nn.Sequential(
            nn.Conv1d(feat_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, max_delay, kernel_size=1)
        )
        
        self.max_delay = max_delay
    
    def forward(self, y_geo, audio_feat):
        # y_geo: B×1×T (幾何 warp 後的音訊)
        # audio_feat: B×256×T (音訊特徵)
        
        # 1. 預測 delay weights
        weights = self.delay_predictor(audio_feat)  # B×max_delay×T
        weights = F.softmax(weights, dim=1)  # 歸一化
        
        # 2. 對每個時間步應用 delay
        B, _, T = y_geo.shape
        y_delayed = []
        
        for t in range(T):
            # 取當前時間步的 weights
            w = weights[:, :, t]  # B×max_delay
            
            # 取當前時間步周圍的音訊
            start = max(0, t - self.max_delay // 2)
            end = min(T, t + self.max_delay // 2)
            y_window = y_geo[:, :, start:end]  # B×1×window
            
            # Pad 到 max_delay
            if y_window.shape[-1] < self.max_delay:
                y_window = F.pad(y_window, 
                    (0, self.max_delay - y_window.shape[-1]))
            
            # 加權求和
            y_t = (w.unsqueeze(1) * y_window).sum(dim=-1)  # B×1
            y_delayed.append(y_t)
        
        y_delayed = torch.stack(y_delayed, dim=-1)  # B×1×T
        
        return y_delayed
```

#### FiLMLayer（特徵調制層）
```python
class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation
    """
    def __init__(self, cond_dim, feat_dim):
        super().__init__()
        self.gamma_net = nn.Linear(cond_dim, feat_dim)
        self.beta_net = nn.Linear(cond_dim, feat_dim)
    
    def forward(self, feat, cond):
        # feat: B×feat_dim×T
        # cond: B×cond_dim
        
        gamma = self.gamma_net(cond).unsqueeze(-1)  # B×feat_dim×1
        beta = self.beta_net(cond).unsqueeze(-1)
        
        return gamma * feat + beta
```

### 3.3 Freq Branch 詳細設計

**功能**：學習 ILD（能量差）和 HRTF（頻譜差），輸出 Mag_L 和 Mag_R

```python
class FreqBranch(nn.Module):
    """
    頻域分支：學習 ILD（Interaural Level Difference）和 HRTF
    
    輸入：
        mono: B×1×T
        view: B×7×K
    
    輸出：
        Mag_L: B×F×T_stft (左耳幅度)
        Mag_R: B×F×T_stft (右耳幅度)
    """
    def __init__(self,
                 fft_size=1024,
                 hop_size=256,
                 tf_channels=256,
                 tf_blocks=8,
                 num_heads=8):
        super().__init__()
        
        self.fft_size = fft_size
        self.hop_size = hop_size
        
        # 1. 位置編碼器（Cross-Attention 的 K, V）
        self.position_encoder = PositionEncoder(
            view_dim=7,
            hidden_dim=128,
            output_dim=tf_channels,
            num_layers=3
        )
        
        # 2. 音訊編碼器（STFT → initial features）
        self.audio_encoder = nn.Conv2d(
            2,  # Real + Imag
            tf_channels,
            kernel_size=1
        )
        
        # 3. Cross-Attention（注入位置資訊）
        self.cross_attn = CrossAttentionBlock(
            dim=tf_channels,
            num_heads=num_heads
        )
        
        # 4. TFResStack（頻域特徵提取）
        self.tf_resstack = TFResStack(
            channels=tf_channels,
            num_blocks=tf_blocks
        )
        
        # 5. 輸出層（分別預測左右耳 Magnitude）
        self.output_L = nn.Conv2d(tf_channels, 1, kernel_size=1)
        self.output_R = nn.Conv2d(tf_channels, 1, kernel_size=1)
    
    def forward(self, mono, view):
        B, _, T = mono.shape
        
        # 1. STFT
        Y_mono = torch.stft(mono.squeeze(1), self.fft_size, self.hop_size,
                           return_complex=True)  # B×F×T_stft
        
        # 轉換為 real + imag
        Y_real = Y_mono.real.unsqueeze(1)  # B×1×F×T_stft
        Y_imag = Y_mono.imag.unsqueeze(1)
        Y_input = torch.cat([Y_real, Y_imag], dim=1)  # B×2×F×T_stft
        
        # 2. 音訊編碼
        audio_feat = self.audio_encoder(Y_input)  # B×C×F×T_stft
        
        # 3. 位置編碼
        pos_feat = self.position_encoder(view)  # B×K×C
        
        # 4. Cross-Attention（注入位置資訊）
        audio_feat = self.cross_attn(audio_feat, pos_feat)  # B×C×F×T_stft
        
        # 5. TFResStack（特徵提取）
        feat = self.tf_resstack(audio_feat)  # B×C×F×T_stft
        
        # 6. 預測左右耳 Magnitude
        Mag_L = self.output_L(feat).squeeze(1)  # B×F×T_stft
        Mag_R = self.output_R(feat).squeeze(1)
        
        # 確保 Magnitude 非負
        Mag_L = F.softplus(Mag_L)
        Mag_R = F.softplus(Mag_R)
        
        return Mag_L, Mag_R
```

**關鍵模組**：

#### PositionEncoder（位置編碼器）
```python
class PositionEncoder(nn.Module):
    """
    將 view (B×7×K) 編碼為位置特徵 (B×K×C)
    用於 Cross-Attention 的 K, V
    """
    def __init__(self, view_dim=7, hidden_dim=128, 
                 output_dim=256, num_layers=3):
        super().__init__()
        
        layers = []
        in_dim = view_dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Conv1d(in_dim, output_dim, kernel_size=1))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, view):
        # view: B×7×K
        feat = self.encoder(view)  # B×C×K
        return feat.transpose(1, 2)  # B×K×C
```

#### CrossAttentionBlock（交叉注意力）
```python
class CrossAttentionBlock(nn.Module):
    """
    Cross-Attention: audio features query position features
    使用 Flash Attention 優化
    """
    def __init__(self, dim=256, num_heads=8):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Q 來自 audio，K/V 來自 position
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)
        
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, audio_feat, pos_feat):
        # audio_feat: B×C×F×T
        # pos_feat: B×K×C
        
        B, C, F, T = audio_feat.shape
        K = pos_feat.shape[1]
        
        # 1. Project
        Q = self.q_proj(audio_feat)  # B×C×F×T
        Q = Q.view(B, self.num_heads, self.head_dim, F*T)
        Q = Q.transpose(2, 3)  # B×H×(F*T)×D
        
        K = self.k_proj(pos_feat)  # B×K×C
        K = K.view(B, self.num_heads, K, self.head_dim)
        K = K.transpose(2, 3)  # B×H×D×K
        
        V = self.v_proj(pos_feat)  # B×K×C
        V = V.view(B, self.num_heads, K, self.head_dim)
        
        # 2. Attention (可以用 Flash Attention 優化)
        attn = torch.matmul(Q, K) / (self.head_dim ** 0.5)  # B×H×(F*T)×K
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, V)  # B×H×(F*T)×D
        
        # 3. Reshape and project
        out = out.transpose(2, 3).contiguous()  # B×H×D×(F*T)
        out = out.view(B, C, F, T)
        out = self.out_proj(out)
        
        # 4. Residual
        return audio_feat + out
```


### 3.4 Fusion（融合層）

**功能**：將 Time Branch 的 Phase 和 Freq Branch 的 Magnitude 融合

```python
class HybridTFNet(nn.Module):
    """
    混合時頻網路：Time Branch + Freq Branch + Fusion
    """
    def __init__(self, 
                 sample_rate=16000,
                 fft_size=1024,
                 hop_size=256,
                 tf_channels=256,
                 tf_blocks=8):
        super().__init__()
        
        self.time_branch = TimeBranch(
            sample_rate=sample_rate,
            fft_size=fft_size,
            hop_size=hop_size
        )
        
        self.freq_branch = FreqBranch(
            fft_size=fft_size,
            hop_size=hop_size,
            tf_channels=tf_channels,
            tf_blocks=tf_blocks
        )
        
        self.fft_size = fft_size
        self.hop_size = hop_size
    
    def forward(self, mono, view):
        # 1. Time Branch: 學習 Phase
        Phase_L, Phase_R = self.time_branch(mono, view)  # B×F×T_stft
        
        # 2. Freq Branch: 學習 Magnitude
        Mag_L, Mag_R = self.freq_branch(mono, view)  # B×F×T_stft
        
        # 3. 融合：複數乘法
        Y_L = Mag_L * torch.exp(1j * Phase_L)  # B×F×T_stft (complex)
        Y_R = Mag_R * torch.exp(1j * Phase_R)
        
        # 4. iSTFT
        y_L = torch.istft(Y_L, self.fft_size, self.hop_size)  # B×T
        y_R = torch.istft(Y_R, self.fft_size, self.hop_size)
        
        # 5. Stack
        y_binaural = torch.stack([y_L, y_R], dim=1)  # B×2×T
        
        return y_binaural, {
            'Phase_L': Phase_L,
            'Phase_R': Phase_R,
            'Mag_L': Mag_L,
            'Mag_R': Mag_R
        }
```

### 3.5 數據流總結

```
輸入：mono (B×1×T), view (B×7×K)

Time Branch:
  mono → 幾何 warp → y_geo_L, y_geo_R
       → audio_encoder → audio_feat (B×256×T)
       → FiLM(audio_feat, pos_feat) → conditioned_feat
       → LearnableDelay → y_L, y_R (B×1×T)
       → STFT → Phase_L, Phase_R (B×F×T_stft)

Freq Branch:
  mono → STFT → Y_mono (B×F×T_stft)
       → audio_encoder → audio_feat (B×C×F×T_stft)
       → CrossAttn(audio_feat, pos_feat) → conditioned_feat
       → TFResStack → feat (B×C×F×T_stft)
       → output_L/R → Mag_L, Mag_R (B×F×T_stft)

Fusion:
  Y_L = Mag_L * exp(1j * Phase_L)
  Y_R = Mag_R * exp(1j * Phase_R)
  → iSTFT → y_L, y_R (B×2×T)
```

---

## 4. 訓練策略

### 4.1 Loss Functions

#### Stage 1: Time Branch（學習 Phase/ITD）

```python
def stage1_loss(pred, target, outputs):
    Phase_L_pred = outputs['Phase_L']
    Phase_R_pred = outputs['Phase_R']
    
    # 計算 GT 的 Phase
    Y_L_gt = torch.stft(target[:, 0, :], fft_size, hop_size, 
                        return_complex=True)
    Y_R_gt = torch.stft(target[:, 1, :], fft_size, hop_size,
                        return_complex=True)
    Phase_L_gt = torch.angle(Y_L_gt)
    Phase_R_gt = torch.angle(Y_R_gt)
    
    # Phase Loss（用 cosine distance）
    loss_phase_L = 1 - torch.cos(Phase_L_pred - Phase_L_gt).mean()
    loss_phase_R = 1 - torch.cos(Phase_R_pred - Phase_R_gt).mean()
    
    # IPD Loss
    IPD_pred = Phase_L_pred - Phase_R_pred
    IPD_gt = Phase_L_gt - Phase_R_gt
    loss_ipd = 1 - torch.cos(IPD_pred - IPD_gt).mean()
    
    # 總 Loss
    loss = loss_phase_L + loss_phase_R + loss_ipd
    
    return loss
```

**訓練參數**：
- 只訓練 `time_branch` 的參數
- 凍結 `freq_branch`
- Epochs: 0-60

#### Stage 2: Freq Branch（學習 Magnitude/ILD）

```python
def stage2_loss(pred, target, outputs):
    Mag_L_pred = outputs['Mag_L']
    Mag_R_pred = outputs['Mag_R']
    
    # 計算 GT 的 Magnitude
    Y_L_gt = torch.stft(target[:, 0, :], fft_size, hop_size,
                        return_complex=True)
    Y_R_gt = torch.stft(target[:, 1, :], fft_size, hop_size,
                        return_complex=True)
    Mag_L_gt = torch.abs(Y_L_gt)
    Mag_R_gt = torch.abs(Y_R_gt)
    
    # Magnitude Loss（L1）
    loss_mag_L = F.l1_loss(Mag_L_pred, Mag_L_gt)
    loss_mag_R = F.l1_loss(Mag_R_pred, Mag_R_gt)
    
    # ILD Loss（能量差）
    ILD_pred = torch.log(Mag_L_pred + 1e-8) - torch.log(Mag_R_pred + 1e-8)
    ILD_gt = torch.log(Mag_L_gt + 1e-8) - torch.log(Mag_R_gt + 1e-8)
    loss_ild = F.mse_loss(ILD_pred, ILD_gt)
    
    # L2 Loss（時域）
    loss_l2 = F.mse_loss(pred, target)
    
    # 總 Loss
    loss = loss_mag_L + loss_mag_R + loss_ild + 0.1 * loss_l2
    
    return loss
```

**訓練參數**：
- 凍結 `time_branch`
- 只訓練 `freq_branch` 的參數
- Epochs: 60-160

#### Stage 3: Joint Fine-tuning（全局微調）

```python
def stage3_loss(pred, target, outputs):
    # 結合 Stage 1 和 Stage 2 的 loss
    loss_phase = stage1_loss(pred, target, outputs)
    loss_mag = stage2_loss(pred, target, outputs)
    
    # L2 Loss（主要目標）
    loss_l2 = F.mse_loss(pred, target)
    
    # 總 Loss（L2 權重較大）
    loss = 100 * loss_l2 + loss_phase + loss_mag
    
    return loss
```

**訓練參數**：
- 訓練所有參數
- Epochs: 160-200

### 4.2 訓練配置

```python
# 模型參數
model = HybridTFNet(
    sample_rate=16000,
    fft_size=1024,
    hop_size=256,
    tf_channels=256,
    tf_blocks=8
)

# 訓練參數
batch_size = 16
learning_rate = 3e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 學習率調度
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=200, 
    eta_min=1e-6
)

# Stage 劃分
stages = {
    'stage1': (0, 60),      # Time Branch
    'stage2': (60, 160),    # Freq Branch
    'stage3': (160, 200)    # Joint Fine-tuning
}
```

### 4.3 為什麼這樣設計能避免 v8.3 的問題

#### 問題 1: CommonBranch 學不到空間感

**v8.3 的問題**：
- CommonBranch 用 WaveNet 在時域學「共同部分」
- 但「共同部分」定義不清晰
- 結果只學到能量平均

**新架構的解決**：
- ❌ 不再有「共同部分」的概念
- ✅ 改為「ITD vs ILD」的任務分離
- ✅ Time Branch 只學時間差（任務明確）
- ✅ Freq Branch 只學能量差（任務明確）

#### 問題 2: SpecificBranch 任務太重

**v8.3 的問題**：
- SpecificBranch 要從 Y_common 學出所有差異
- 包括：距離、殘響、ITD、ILD、HRTF
- 任務太多，學不完

**新架構的解決**：
- ✅ Time Branch 專門學 ITD（有幾何先驗）
- ✅ Freq Branch 專門學 ILD + HRTF
- ✅ 任務分離，各司其職

#### 問題 3: 頻域學 Phase 困難

**v8.3 的問題**：
- SpecificBranch 在頻域用 TFResStack 學 Phase
- Phase 是時域概念，頻域很難學

**新架構的解決**：
- ✅ Time Branch 在時域學 Phase（用 warp + learnable delay）
- ✅ Freq Branch 在頻域學 Magnitude（擅長的任務）
- ✅ 最後融合（Phase 來自時域，Magnitude 來自頻域）

#### 問題 4: 訓練不穩定

**v8.3 的問題**：
- Stage 1 學 L2，Stage 2 學 Phase
- 但 Stage 1 的輸出（y_common）沒有足夠的空間資訊
- Stage 2 無法從中學到 Phase

**新架構的解決**：
- ✅ Stage 1 直接學 Phase（目標明確）
- ✅ Stage 2 直接學 Magnitude（目標明確）
- ✅ 兩個 Branch 獨立訓練，不互相依賴
- ✅ Stage 3 才一起微調

---

## 5. 創新點總結

### 5.1 與 DPATFNet 的差異

| 維度 | DPATFNet | HybridTFNet（新架構）|
|------|----------|---------------------|
| **時域/頻域** | 純頻域 | 混合（時域學 Phase，頻域學 Mag）|
| **任務分離** | 無 | 有（Time Branch + Freq Branch）|
| **物理先驗** | 無 | 有（幾何 ITD）|
| **位置編碼** | DPAB（複雜 Attention）| Cross-Attention（更簡潔）|
| **訓練方式** | 端到端 | 分階段（避免梯度衝突）|
| **Phase 學習** | 頻域（困難）| 時域（容易）|

**核心差異**：
- DPATFNet 是純頻域，端到端
- HybridTFNet 是時頻混合，任務分離

### 5.2 與 BinauralTFNet v8.3 的差異

| 維度 | v8.3 | HybridTFNet（新架構）|
|------|------|---------------------|
| **任務分離** | Common + Specific | Time + Freq |
| **Common 定義** | 「共同部分」（不清晰）| 無（改為 ITD vs ILD）|
| **Time Branch** | WaveNet 學「共同部分」| Warp + Learnable Delay 學 ITD |
| **Freq Branch** | TFResStack 學「所有差異」| TFResStack 學 ILD + HRTF |
| **位置編碼** | SimpleDPAB（太弱）| Cross-Attention（更強）|
| **Stage 1 目標** | L2（y_common）| Phase（ITD）|
| **Stage 2 目標** | Phase（從 Y_common）| Magnitude（從 Y_mono）|

**核心差異**：
- v8.3 的任務分離是「共同 vs 差異」（定義不清）
- HybridTFNet 的任務分離是「ITD vs ILD」（物理明確）

### 5.3 與 Phase-aware Binaural Synthesis 的差異

| 維度 | Phase-aware (2024) | HybridTFNet（新架構）|
|------|-------------------|---------------------|
| **物理先驗** | 無 | 有（幾何 ITD）|
| **ITD 學習** | 純神經網路 warp | 幾何 warp + Learnable Delay |
| **位置編碼** | 簡單 MLP | Cross-Attention |
| **Time Branch** | 簡單 CNN | Conv1d + FiLM |
| **Freq Branch** | 簡單 ResBlock | TFResStack + Cross-Attention |

**核心差異**：
- Phase-aware 沒有物理先驗，完全從零學
- HybridTFNet 有幾何 ITD，神經網路只需要學修正

### 5.4 核心創新

#### 創新 1: 時頻混合 + 任務分離

**創新點**：
- Time Branch 在時域學 ITD/Phase
- Freq Branch 在頻域學 ILD/Magnitude
- 最後融合（複數乘法）

**為什麼創新**：
- 現有方法要麼純時域（WaveNet），要麼純頻域（DPATFNet）
- 我們是第一個明確分離 Phase 和 Magnitude 學習的架構

**為什麼有效**：
- Phase 是時域概念（時間延遲）→ 時域學習更自然
- Magnitude 是頻域概念（能量分布）→ 頻域學習更自然
- 各司其職，不互相干擾

#### 創新 2: 幾何先驗 + 神經修正

**創新點**：
- 用幾何 ITD 提供粗略的時間延遲
- 用 Learnable Delay 提供精細修正

**為什麼創新**：
- Phase-aware 用純神經網路 warp（容易學歪）
- v8.3 用 Warpnet（幾何 + 神經，但還是 warp）
- 我們用 Learnable Delay Filter（更穩定）

**為什麼有效**：
- 幾何 ITD 保證方向正確（不會學歪）
- Learnable Delay 是線性操作（梯度穩定）
- 神經網路只需要學小範圍修正（搜索空間小）

#### 創新 3: Cross-Attention 位置編碼

**創新點**：
- 用 Cross-Attention 注入位置資訊
- Audio features query Position features

**為什麼創新**：
- DPATFNet 用 DPAB（Cross + Self Attention，複雜）
- v8.3 用 SimpleDPAB（3 層 Conv1d，太弱）
- 我們用單純的 Cross-Attention（簡潔有效）

**為什麼有效**：
- Cross-Attention 是最靈活的條件機制
- 可以學習複雜的位置-音頻交互
- 比 DPAB 簡單，比 SimpleDPAB 強

#### 創新 4: 分階段訓練策略

**創新點**：
- Stage 1: 只訓練 Time Branch（Phase loss）
- Stage 2: 只訓練 Freq Branch（Magnitude loss）
- Stage 3: Joint fine-tuning（L2 + Phase + Magnitude）

**為什麼創新**：
- DPATFNet 端到端（可能梯度衝突）
- v8.3 分階段，但 Stage 1 目標錯誤（L2 而非 Phase）
- 我們的 Stage 1 直接學 Phase（目標正確）

**為什麼有效**：
- 避免 Phase loss 和 Magnitude loss 的梯度衝突
- 每個 Stage 目標明確
- 兩個 Branch 獨立訓練，不互相依賴

---


## 6. 實作計畫

### 6.1 需要實作的模組

#### 模組 1: GeometricITD
- **功能**：計算幾何 ITD（Woodworth 公式）
- **難度**：低
- **預計時間**：1 小時
- **依賴**：無

#### 模組 2: LearnableDelayNet
- **功能**：用 FIR filter 實現可學習的 delay
- **難度**：中
- **預計時間**：4 小時
- **依賴**：無
- **挑戰**：如何高效實現時變的 delay

#### 模組 3: FiLMLayer
- **功能**：Feature-wise Linear Modulation
- **難度**：低
- **預計時間**：30 分鐘
- **依賴**：無

#### 模組 4: TimeBranch
- **功能**：整合 GeometricITD + LearnableDelay + FiLM
- **難度**：中
- **預計時間**：6 小時
- **依賴**：模組 1, 2, 3

#### 模組 5: PositionEncoder
- **功能**：將 view 編碼為位置特徵
- **難度**：低
- **預計時間**：1 小時
- **依賴**：無

#### 模組 6: CrossAttentionBlock
- **功能**：Cross-Attention 注入位置資訊
- **難度**：中
- **預計時間**：4 小時
- **依賴**：無
- **可選優化**：使用 Flash Attention

#### 模組 7: FreqBranch
- **功能**：整合 PositionEncoder + CrossAttention + TFResStack
- **難度**：中
- **預計時間**：6 小時
- **依賴**：模組 5, 6, TFResStack（已有）

#### 模組 8: HybridTFNet
- **功能**：整合 TimeBranch + FreqBranch + Fusion
- **難度**：低
- **預計時間**：2 小時
- **依賴**：模組 4, 7

#### 模組 9: 訓練腳本
- **功能**：分階段訓練邏輯
- **難度**：中
- **預計時間**：4 小時
- **依賴**：模組 8

### 6.2 實作順序

#### Phase 1: 核心模組（第 1-2 天）

1. **GeometricITD**（1 小時）
   - 實作 Woodworth 公式
   - 單元測試：驗證 ITD 範圍（±0.7ms）

2. **FiLMLayer**（30 分鐘）
   - 簡單的線性調制
   - 單元測試：驗證輸出形狀

3. **PositionEncoder**（1 小時）
   - Conv1d stack
   - 單元測試：驗證輸出形狀

4. **LearnableDelayNet**（4 小時）
   - 實作 FIR filter delay
   - 單元測試：驗證 delay 效果
   - **挑戰**：時變 delay 的高效實現

#### Phase 2: Branch 模組（第 3-4 天）

5. **CrossAttentionBlock**（4 小時）
   - 實作 Cross-Attention
   - 單元測試：驗證 attention weights
   - 可選：整合 Flash Attention

6. **TimeBranch**（6 小時）
   - 整合 GeometricITD + LearnableDelay + FiLM
   - 單元測試：驗證 Phase 輸出範圍（-π 到 π）

7. **FreqBranch**（6 小時）
   - 整合 PositionEncoder + CrossAttention + TFResStack
   - 單元測試：驗證 Magnitude 非負

#### Phase 3: 整合與訓練（第 5-7 天）

8. **HybridTFNet**（2 小時）
   - 整合兩個 Branch
   - 整合測試：完整 forward pass

9. **訓練腳本**（4 小時）
   - 實作分階段訓練邏輯
   - 實作 loss functions

10. **小規模訓練測試**（1 天）
    - 用 100 個樣本測試收斂性
    - 驗證梯度流動
    - 調試 bug

11. **完整訓練**（1 天啟動）
    - 啟動完整訓練
    - 監控 Stage 1 收斂

**總預計時間**：5-7 天

### 6.3 測試計畫

#### 單元測試

```python
# test_geometric_itd.py
def test_geometric_itd():
    model = GeometricITD(sample_rate=16000)
    
    # 測試正前方（ITD 應該接近 0）
    view_front = torch.tensor([[[1.0], [0.0], [0.0], 
                                [1.0], [0.0], [0.0], [0.0]]])
    itd_L, itd_R = model(view_front)
    assert torch.abs(itd_L).item() < 1.0  # < 1 sample
    assert torch.abs(itd_R).item() < 1.0
    
    # 測試正左方（ITD 應該最大）
    view_left = torch.tensor([[[0.0], [1.0], [0.0],
                               [1.0], [0.0], [0.0], [0.0]]])
    itd_L, itd_R = model(view_left)
    assert itd_L.item() < -5.0  # 左耳延遲
    assert itd_R.item() > 5.0   # 右耳提前
    
    print("✓ GeometricITD test passed")

# test_learnable_delay.py
def test_learnable_delay():
    model = LearnableDelayNet(feat_dim=256, max_delay=32)
    
    # 測試輸入
    y_geo = torch.randn(2, 1, 16000)  # B×1×T
    audio_feat = torch.randn(2, 256, 16000)  # B×256×T
    
    # Forward
    y_delayed = model(y_geo, audio_feat)
    
    # 驗證形狀
    assert y_delayed.shape == y_geo.shape
    
    # 驗證可微分
    loss = y_delayed.sum()
    loss.backward()
    assert model.delay_predictor[0].weight.grad is not None
    
    print("✓ LearnableDelayNet test passed")

# test_time_branch.py
def test_time_branch():
    model = TimeBranch(sample_rate=16000, fft_size=1024, hop_size=256)
    
    # 測試輸入
    mono = torch.randn(2, 1, 16000)
    view = torch.randn(2, 7, 10)
    
    # Forward
    Phase_L, Phase_R = model(mono, view)
    
    # 驗證形狀
    assert Phase_L.shape[0] == 2  # Batch
    assert Phase_L.shape[1] == 1024 // 2 + 1  # Freq bins
    
    # 驗證 Phase 範圍（-π 到 π）
    assert Phase_L.min() >= -np.pi
    assert Phase_L.max() <= np.pi
    
    print("✓ TimeBranch test passed")

# test_freq_branch.py
def test_freq_branch():
    model = FreqBranch(fft_size=1024, hop_size=256, 
                       tf_channels=256, tf_blocks=8)
    
    # 測試輸入
    mono = torch.randn(2, 1, 16000)
    view = torch.randn(2, 7, 10)
    
    # Forward
    Mag_L, Mag_R = model(mono, view)
    
    # 驗證形狀
    assert Mag_L.shape[0] == 2
    assert Mag_L.shape[1] == 1024 // 2 + 1
    
    # 驗證 Magnitude 非負
    assert Mag_L.min() >= 0
    assert Mag_R.min() >= 0
    
    print("✓ FreqBranch test passed")
```

#### 整合測試

```python
# test_hybrid_tfnet.py
def test_hybrid_tfnet():
    model = HybridTFNet(sample_rate=16000, fft_size=1024, 
                        hop_size=256, tf_channels=256, tf_blocks=8)
    
    # 測試輸入
    mono = torch.randn(2, 1, 16000)
    view = torch.randn(2, 7, 10)
    
    # Forward
    y_binaural, outputs = model(mono, view)
    
    # 驗證形狀
    assert y_binaural.shape == (2, 2, 16000)  # B×2×T
    
    # 驗證輸出
    assert 'Phase_L' in outputs
    assert 'Phase_R' in outputs
    assert 'Mag_L' in outputs
    assert 'Mag_R' in outputs
    
    # 驗證可微分
    loss = y_binaural.sum()
    loss.backward()
    
    print("✓ HybridTFNet test passed")

# test_training.py
def test_small_scale_training():
    """用 100 個樣本測試訓練"""
    model = HybridTFNet(...)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    # 生成假數據
    dataset = [(torch.randn(1, 16000), torch.randn(7, 10), 
                torch.randn(2, 16000)) for _ in range(100)]
    
    # 訓練 10 epochs
    for epoch in range(10):
        total_loss = 0
        for mono, view, target in dataset:
            optimizer.zero_grad()
            pred, outputs = model(mono.unsqueeze(0), view.unsqueeze(0))
            loss = stage1_loss(pred, target.unsqueeze(0), outputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch}: Loss = {total_loss/100:.6f}")
    
    # 驗證 loss 下降
    assert total_loss < initial_loss * 0.5
    
    print("✓ Small-scale training test passed")
```

### 6.4 實作檢查清單

#### 代碼實作
- [ ] GeometricITD
- [ ] LearnableDelayNet
- [ ] FiLMLayer
- [ ] TimeBranch
- [ ] PositionEncoder
- [ ] CrossAttentionBlock
- [ ] FreqBranch
- [ ] HybridTFNet
- [ ] 訓練腳本（分階段邏輯）
- [ ] Loss functions

#### 測試
- [ ] 單元測試：GeometricITD
- [ ] 單元測試：LearnableDelayNet
- [ ] 單元測試：TimeBranch
- [ ] 單元測試：FreqBranch
- [ ] 整合測試：HybridTFNet forward pass
- [ ] 整合測試：梯度流動
- [ ] 小規模訓練測試（100 樣本）

#### 文檔
- [ ] 代碼註釋
- [ ] 架構圖
- [ ] 訓練指南
- [ ] 實驗記錄模板

---

## 7. 風險評估

### 7.1 潛在風險

#### 風險 1: LearnableDelayNet 可能學不到精確的 delay

**描述**：
- FIR filter 的分辨率有限（受 max_delay 限制）
- 可能無法學到亞樣本級別的精確 delay

**可能性**：中
**影響**：高（如果學不到精確 ITD，Phase 會不準）

**緩解方案**：
1. 增加 max_delay（例如 64 或 128）
2. 用 fractional delay filter（更精確）
3. 如果還是不行，回退到 warp（但要加入穩定性約束）

#### 風險 2: Cross-Attention 可能過擬合

**描述**：
- Cross-Attention 參數多，表達能力強
- 如果數據不夠，可能過擬合

**可能性**：中
**影響**：中（泛化能力差）

**緩解方案**：
1. 加入 Dropout（0.1-0.2）
2. 減少 num_heads（8 → 4）
3. 如果還是過擬合，回退到 FiLM

#### 風險 3: 兩個 Branch 可能不一致

**描述**：
- Time Branch 和 Freq Branch 獨立訓練
- 融合後可能不一致（例如：Phase 和 Magnitude 不匹配）

**可能性**：低
**影響**：中（音質可能有問題）

**緩解方案**：
1. Stage 3 的 Joint Fine-tuning 應該能解決
2. 如果還是不一致，加入 consistency loss：
   ```python
   # 從 Time Branch 的輸出計算 Magnitude
   Y_L_time = torch.stft(y_L_time, ...)
   Mag_L_time = torch.abs(Y_L_time)
   
   # 與 Freq Branch 的 Magnitude 對齊
   loss_consistency = F.mse_loss(Mag_L_time, Mag_L_freq)
   ```

#### 風險 4: 訓練時間可能很長

**描述**：
- 兩個 Branch 都比較複雜
- 訓練可能比 v8.3 慢

**可能性**：高
**影響**：低（只是時間成本）

**緩解方案**：
1. 使用 Flash Attention（降低顯存，提升速度）
2. 減少 tf_blocks（8 → 6）
3. 使用混合精度訓練（AMP）

#### 風險 5: 幾何 ITD 可能不準

**描述**：
- Woodworth 公式是簡化模型（球體頭部）
- 真實頭部形狀更複雜

**可能性**：中
**影響**：低（LearnableDelay 可以修正）

**緩解方案**：
1. LearnableDelay 的修正範圍要足夠大（±32 samples）
2. 如果幾何 ITD 誤差太大，可以讓它也可訓練（但要加正則化）

### 7.2 成功標準

#### 最低標準（必須達到）

- **Stage 1 結束（epoch 60）**：
  - Phase error < 1.3（比 v8.3 的 1.43 好）
  - IPD error < 1.3（比 v8.3 的 1.35 好）

- **Stage 2 結束（epoch 160）**：
  - Phase error < 1.0（改善 >30%）
  - L2 loss < 0.00005

#### 目標標準（期望達到）

- **Stage 2 結束（epoch 160）**：
  - Phase error < 0.85（接近 Phase-aware 的水平）
  - IPD error < 0.9
  - L2 loss < 0.00004

#### 優秀標準（超出預期）

- **Stage 3 結束（epoch 200）**：
  - Phase error < 0.7（超越 Phase-aware）
  - IPD error < 0.8
  - L2 loss < 0.00003
  - 主觀聽感：接近真實錄音

### 7.3 備選方案

#### Plan B: 如果 LearnableDelayNet 失敗

**方案**：回退到 Warpnet，但加入穩定性約束

```python
# 限制 warpfield 的範圍
warpfield = torch.tanh(warpfield_raw) * max_warp_range

# 加入平滑性約束
loss_smooth = torch.mean((warpfield[:, 1:] - warpfield[:, :-1])**2)
```

#### Plan C: 如果 Cross-Attention 過擬合

**方案**：回退到 FiLM

```python
# 替換 CrossAttentionBlock
self.condition = FiLMLayer(cond_dim=256, feat_dim=256)
```

#### Plan D: 如果整個架構失敗

**方案**：簡化為單 Branch（純頻域）

```python
# 只保留 Freq Branch
# 放棄 Time Branch
# 回到類似 DPATFNet 的架構，但用更好的位置編碼
```

---

## 8. 總結

### 8.1 核心設計理念

**HybridTFNet = Time Branch (ITD) + Freq Branch (ILD) + Fusion**

- **Time Branch**：時域學 Phase（幾何 ITD + Learnable Delay + FiLM）
- **Freq Branch**：頻域學 Magnitude（Cross-Attention + TFResStack）
- **Fusion**：複數乘法融合

### 8.2 為什麼這個架構會成功

1. **任務分離合理**：ITD vs ILD（物理明確）
2. **時域學 Phase，頻域學 Magnitude**：各司其職
3. **有物理先驗**：幾何 ITD（不會學歪）
4. **訓練穩定**：分階段，目標明確
5. **借鑑最佳實踐**：Phase-aware + BinauralGrad + DPATFNet

### 8.3 預期效果

- **Stage 1**：Phase error 降到 1.3（比 v8.3 好 10%）
- **Stage 2**：Phase error 降到 0.85（比 v8.3 好 40%+）
- **Stage 3**：Phase error 降到 0.7（超越現有方法）

### 8.4 下一步

1. **審查架構設計**（主 Agent）
2. **開始實作**（按照實作計畫）
3. **小規模測試**（100 樣本）
4. **完整訓練**（監控 Stage 1 收斂）

---

**設計完成！等待主 Agent 審查和決定是否實作。**
