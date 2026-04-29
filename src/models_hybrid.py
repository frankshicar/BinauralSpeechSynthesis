"""
HybridTFNet: 混合時頻網路
Time Branch 學習 ITD/Phase，Freq Branch 學習 ILD/Magnitude

作者：Architecture Synthesis Agent
日期：2026-04-27
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ==================== Utility Modules ====================

class GeometricITD(nn.Module):
    """
    計算幾何 ITD（基於 Woodworth 公式）
    固定計算，不訓練
    """
    def __init__(self, sample_rate=16000, head_radius=0.0875):
        super().__init__()
        self.sample_rate = sample_rate
        self.head_radius = head_radius
        self.c = 343.0  # 聲速 (m/s)
    
    def forward(self, view):
        """
        Args:
            view: B×7 or B×7×K
        Returns:
            itd_L, itd_R: B×1 (左右耳 ITD，單位：samples)
        """
        # 如果是時間序列，先平均
        if view.dim() == 3:
            view = view.mean(dim=-1)  # B×7×K → B×7
        
        # 取位置
        pos = view[:, :3]  # B×3
        
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
    
    def apply_warp(self, mono, itd):
        """
        應用幾何 warp（簡單的 shift）
        
        Args:
            mono: B×1×T
            itd: B×1 (delay in samples)
        Returns:
            warped: B×1×T
        """
        B, _, T = mono.shape
        device = mono.device
        
        # 用 roll 實現簡單的 shift
        # 注意：這是近似，真實的 fractional delay 需要插值
        itd_int = itd.squeeze(-1).round().long()  # B
        
        warped = []
        for i in range(B):
            shift = itd_int[i].item()
            if shift > 0:
                # 右移（延遲）
                w = torch.cat([torch.zeros(1, abs(shift), device=device), 
                              mono[i, :, :-abs(shift)]], dim=-1)
            elif shift < 0:
                # 左移（提前）
                w = torch.cat([mono[i, :, abs(shift):], 
                              torch.zeros(1, abs(shift), device=device)], dim=-1)
            else:
                w = mono[i]
            warped.append(w)
        
        return torch.stack(warped, dim=0)  # B×1×T


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation
    用位置資訊調制音訊特徵
    """
    def __init__(self, cond_dim, feat_dim):
        super().__init__()
        self.gamma_net = nn.Linear(cond_dim, feat_dim)
        self.beta_net = nn.Linear(cond_dim, feat_dim)
    
    def forward(self, feat, cond):
        """
        Args:
            feat: B×feat_dim×T
            cond: B×cond_dim
        Returns:
            modulated: B×feat_dim×T
        """
        gamma = self.gamma_net(cond).unsqueeze(-1)  # B×feat_dim×1
        beta = self.beta_net(cond).unsqueeze(-1)
        
        return gamma * feat + beta


class FrequencyDependentDelay(nn.Module):
    """
    頻率相關的 delay（用 phase shift 實現）
    """
    def __init__(self, n_fft=1024, hop_size=256, sample_rate=16000):
        super().__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        self.n_freq = n_fft // 2 + 1
        
        # 預測每個頻率的 delay
        self.delay_predictor = nn.Sequential(
            nn.Linear(128 + 7, 256),  # audio_feat_pooled + view
            nn.ReLU(),
            nn.Linear(256, self.n_freq)
        )
    
    def forward(self, y_geo, audio_feat, view):
        """
        Args:
            y_geo: B×1×T (幾何 warp 後的波形)
            audio_feat: B×128×T (條件特徵)
            view: B×7 or B×7×K (位置資訊)
        Returns:
            Phase: B×F×T_stft
        """
        B, _, T = y_geo.shape
        
        # 如果 view 是時間序列，先平均
        if view.dim() == 3:
            view = view.mean(dim=-1)  # B×7×K → B×7
        
        # 1. 從 audio_feat 預測 delay
        feat_pooled = F.adaptive_avg_pool1d(audio_feat, 1).squeeze(-1)  # B×128
        feat_combined = torch.cat([feat_pooled, view], dim=1)  # B×(128+7)
        
        delays = self.delay_predictor(feat_combined)  # B×n_freq
        delays = torch.tanh(delays) * 32  # 限制在 ±32 samples
        
        # 2. STFT
        window = torch.hann_window(self.n_fft).to(y_geo.device)
        Y = torch.stft(y_geo.squeeze(1), self.n_fft, self.hop_size,
                      window=window, return_complex=True)  # B×F×T_stft
        
        # 3. 計算 phase shift
        freqs = torch.fft.rfftfreq(self.n_fft, 1/self.sample_rate).to(Y.device)
        phase_shift = -2 * np.pi * freqs.unsqueeze(0).unsqueeze(-1) * \
                      (delays.unsqueeze(-1) / self.sample_rate)  # B×F×T_stft
        
        # 4. 應用 phase shift 並提取 Phase
        Y_delayed = Y * torch.exp(1j * phase_shift)
        Phase = torch.angle(Y_delayed)  # B×F×T_stft
        
        return Phase


# ==================== Position Encoder ====================

class PositionEncoder(nn.Module):
    """
    將 view 編碼為位置特徵
    如果 view 是時間序列 (B×7×T)，先平均到 B×7
    """
    def __init__(self, view_dim=7, hidden_dim=128, output_dim=256, num_layers=3):
        super().__init__()
        
        layers = []
        in_dim = view_dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, view):
        """
        Args:
            view: B×7×K (時間序列) or B×7
        Returns:
            feat: B×output_dim
        """
        # Debug: print view shape
        # print(f"PositionEncoder input view shape: {view.shape}")
        
        # 如果是 B×7×K，平均到 B×7
        if view.dim() == 3:
            view = view.mean(dim=-1)  # B×7×K → B×7
        elif view.dim() == 2 and view.shape[1] > 7:
            # B×T 格式，reshape 成 B×7×(T//7) 再平均
            B, T = view.shape
            T_new = (T // 7) * 7
            view = view[:, :T_new].view(B, 7, T_new // 7).mean(dim=-1)  # B×7
        elif view.dim() == 2 and view.shape[1] < 7:
            raise ValueError(f"view shape {view.shape} is invalid, expected B×7 or B×7×K")
        
        return self.encoder(view)


if __name__ == "__main__":
    # 測試
    print("Testing utility modules...")
    
    # GeometricITD
    geo_itd = GeometricITD()
    view = torch.randn(4, 7)
    itd_L, itd_R = geo_itd(view)
    print(f"GeometricITD: itd_L shape = {itd_L.shape}, itd_R shape = {itd_R.shape}")
    
    mono = torch.randn(4, 1, 16000)
    warped = geo_itd.apply_warp(mono, itd_L)
    print(f"Warped shape = {warped.shape}")
    
    # FiLMLayer
    film = FiLMLayer(cond_dim=7, feat_dim=128)
    feat = torch.randn(4, 128, 16000)
    modulated = film(feat, view)
    print(f"FiLM: modulated shape = {modulated.shape}")
    
    # FrequencyDependentDelay
    delay_net = FrequencyDependentDelay()
    audio_feat = torch.randn(4, 128, 16000)
    phase = delay_net(mono, audio_feat, view)
    print(f"FrequencyDependentDelay: phase shape = {phase.shape}")
    
    # PositionEncoder
    pos_enc = PositionEncoder()
    pos_feat = pos_enc(view)
    print(f"PositionEncoder: pos_feat shape = {pos_feat.shape}")
    


# ==================== Time Branch ====================

class TimeBranch(nn.Module):
    """
    時域分支：直接在 STFT 域預測 Phase_L 和 Phase_R
    """
    def __init__(self, 
                 sample_rate=16000,
                 n_fft=1024,
                 hop_size=256,
                 channels=128):
        super().__init__()
        
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.n_freq = n_fft // 2 + 1
        
        # 1. 位置編碼器
        self.position_encoder = PositionEncoder(
            view_dim=7,
            hidden_dim=128,
            output_dim=channels
        )
        
        # 2. STFT 特徵編碼（real + imag → channels）
        self.stft_encoder = nn.Conv2d(2, channels, kernel_size=3, padding=1)
        
        # 3. FiLM 調制（替代 Cross-Attention）
        self.gamma_net = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels)
        )
        self.beta_net = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels)
        )
        
        # 4. Phase Predictor (Conv2d on STFT features)
        self.phase_net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        
        # 5. 輸出層
        self.output_L = nn.Conv2d(channels, 1, kernel_size=1)
        self.output_R = nn.Conv2d(channels, 1, kernel_size=1)
    
    def forward(self, mono, shared_feat, view):
        """
        Args:
            mono: B×1×T
            shared_feat: B×128×T (unused, kept for interface compatibility)
            view: B×7 or B×7×K
        Returns:
            Phase_L: B×F×T_stft
            Phase_R: B×F×T_stft
        """
        # 1. STFT (real + imag as 2 channels)
        window = torch.hann_window(self.n_fft).to(mono.device)
        Y_mono = torch.stft(mono.squeeze(1), self.n_fft, self.hop_size,
                           window=window, return_complex=True)  # B×F×T_stft
        stft_feat = torch.stack([Y_mono.real, Y_mono.imag], dim=1)  # B×2×F×T_stft
        
        # 2. 位置編碼
        pos_feat = self.position_encoder(view)  # B×channels
        
        # 3. STFT 特徵編碼
        phase_feat = self.stft_encoder(stft_feat)  # B×channels×F×T_stft
        
        # 4. FiLM 調制（注入位置資訊）
        gamma = self.gamma_net(pos_feat).unsqueeze(-1).unsqueeze(-1)  # B×C×1×1
        beta = self.beta_net(pos_feat).unsqueeze(-1).unsqueeze(-1)  # B×C×1×1
        phase_feat = gamma * phase_feat + beta  # B×channels×F×T_stft
        
        # 5. Phase prediction
        phase_feat = self.phase_net(phase_feat)  # B×channels×F×T_stft
        
        # 6. 輸出 Phase Difference（不限制範圍，wrapped MSE 會處理）
        Phase_diff_L = self.output_L(phase_feat).squeeze(1)  # B×F×T_stft
        Phase_diff_R = self.output_R(phase_feat).squeeze(1)
        
        return Phase_diff_L, Phase_diff_R


# ==================== Freq Branch ====================

class SimpleTFResStack(nn.Module):
    """
    簡化的頻域殘差堆疊
    直接處理 B×C×F×T 的特徵
    """
    def __init__(self, channels=256, num_blocks=8):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            )
            for _ in range(num_blocks)
        ])
    
    def forward(self, x):
        """
        Args:
            x: B×C×F×T
        Returns:
            out: B×C×F×T
        """
        for block in self.blocks:
            residual = x
            x = block(x)
            x = x + residual  # Residual connection
        return x


class CrossAttentionBlock(nn.Module):
    """
    Cross-Attention: Position query Audio
    """
    def __init__(self, dim=256, num_heads=8):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Q 來自 position，K/V 來自 audio
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)
        
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, audio_feat, pos_feat):
        """
        Args:
            audio_feat: B×C×T (時域) or B×C×F×T (頻域)
            pos_feat: B×C
        Returns:
            out: same shape as audio_feat
        """
        B, C = pos_feat.shape
        
        # 判斷輸入維度
        if audio_feat.dim() == 3:
            # 時域：B×C×T
            _, _, T = audio_feat.shape
            F_bins, T_stft = 1, T
            audio_feat_4d = audio_feat.unsqueeze(2)  # B×C×1×T
        else:
            # 頻域：B×C×F×T
            _, _, F_bins, T_stft = audio_feat.shape
            audio_feat_4d = audio_feat
        
        # 1. Project
        Q = self.q_proj(pos_feat)  # B×C
        Q = Q.view(B, self.num_heads, self.head_dim, 1)  # B×H×D×1
        
        K = self.k_proj(audio_feat_4d)  # B×C×F×T
        K = K.view(B, self.num_heads, self.head_dim, F_bins*T_stft)  # B×H×D×(F*T)
        
        V = self.v_proj(audio_feat_4d)  # B×C×F×T
        V = V.view(B, self.num_heads, self.head_dim, F_bins*T_stft)  # B×H×D×(F*T)
        
        # 2. Attention
        attn = torch.matmul(Q.transpose(2, 3), K) / (self.head_dim ** 0.5)  # B×H×1×(F*T)
        attn = torch.nn.functional.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, V.transpose(2, 3))  # B×H×1×D
        
        # 3. Reshape and project
        out = out.transpose(2, 3).contiguous()  # B×H×D×1
        out = out.view(B, C, 1, 1)  # B×C×1×1
        out = out.expand(-1, -1, F_bins, T_stft)  # B×C×F×T
        out = self.out_proj(out)
        
        # 4. Residual
        if audio_feat.dim() == 3:
            return audio_feat + out.squeeze(2)  # B×C×T
        else:
            return audio_feat + out  # B×C×F×T


class FreqBranch(nn.Module):
    """
    頻域分支：學習 ILD（Interaural Level Difference）和 HRTF
    輸出 Mag_L 和 Mag_R
    """
    def __init__(self,
                 n_fft=1024,
                 hop_size=256,
                 tf_channels=256,
                 tf_blocks=8,
                 num_heads=8):
        super().__init__()
        
        self.n_fft = n_fft
        self.hop_size = hop_size
        
        # 1. 將時域特徵轉換為頻域維度
        self.time_to_freq = nn.Conv1d(128, tf_channels, kernel_size=1)
        
        # 2. 頻域專用的 encoder
        self.freq_encoder = nn.Sequential(
            nn.Conv2d(tf_channels, tf_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(tf_channels, tf_channels, kernel_size=3, padding=1),
        )
        
        # 3. 位置編碼器
        self.position_encoder = PositionEncoder(
            view_dim=7,
            hidden_dim=128,
            output_dim=tf_channels
        )
        
        # 4. Cross-Attention
        self.cross_attn = CrossAttentionBlock(dim=tf_channels, num_heads=num_heads)
        
        # 5. TFResStack
        self.tf_resstack = SimpleTFResStack(
            channels=tf_channels,
            num_blocks=tf_blocks
        )
        
        # 6. 輸出層
        self.output_L = nn.Conv2d(tf_channels, 1, kernel_size=1)
        self.output_R = nn.Conv2d(tf_channels, 1, kernel_size=1)
    
    def forward(self, shared_feat, mono, view):
        """
        Args:
            shared_feat: B×128×T (來自 shared_encoder)
            mono: B×1×T (用於 STFT)
            view: B×7 (位置資訊)
        Returns:
            Mag_L: B×F×T_stft
            Mag_R: B×F×T_stft
        """
        B, _, T = mono.shape
        
        # 1. STFT (從 mono)
        window = torch.hann_window(self.n_fft).to(mono.device)
        Y_mono = torch.stft(mono.squeeze(1), self.n_fft, self.hop_size,
                           window=window, return_complex=True)  # B×F×T_stft
        
        # 2. 將 shared_feat 轉換到頻域維度
        shared_feat_freq = self.time_to_freq(shared_feat)  # B×128×T → B×256×T
        shared_feat_freq = F.interpolate(
            shared_feat_freq, 
            size=Y_mono.shape[-1],
            mode='linear',
            align_corners=False
        )  # B×256×T_stft
        
        # 3. Broadcast 到頻率維度
        F_bins = Y_mono.shape[1]
        shared_feat_freq = shared_feat_freq.unsqueeze(2).expand(
            -1, -1, F_bins, -1
        )  # B×256×F×T_stft
        
        # 4. 頻域 encoder
        freq_feat = self.freq_encoder(shared_feat_freq)  # B×256×F×T_stft
        
        # 5. 位置編碼
        pos_feat = self.position_encoder(view)  # B×256
        
        # 6. Cross-Attention（注入位置資訊）
        freq_feat = self.cross_attn(freq_feat, pos_feat)  # B×256×F×T_stft
        
        # 7. TFResStack
        feat = self.tf_resstack(freq_feat)  # B×256×F×T_stft
        
        # 8. 輸出 Magnitude
        Mag_L = F.softplus(self.output_L(feat).squeeze(1))  # B×F×T_stft
        Mag_R = F.softplus(self.output_R(feat).squeeze(1))
        
        return Mag_L, Mag_R


# ==================== HybridTFNet ====================

class HybridTFNet(nn.Module):
    """
    混合時頻網路：Time Branch + Freq Branch + Fusion
    
    Time Branch: 學習 ITD/Phase（時域）
    Freq Branch: 學習 ILD/Magnitude（頻域）
    Fusion: 複數乘法融合
    """
    def __init__(self,
                 sample_rate=16000,
                 n_fft=1024,
                 hop_size=256,
                 tf_channels=256,
                 tf_blocks=8,
                 use_cuda=True):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.use_cuda = use_cuda
        
        # ===== Shared Encoder =====
        self.shared_encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        
        # ===== Time Branch =====
        self.time_branch = TimeBranch(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_size=hop_size,
            channels=tf_channels
        )
        
        # ===== Freq Branch =====
        self.freq_branch = FreqBranch(
            n_fft=n_fft,
            hop_size=hop_size,
            tf_channels=tf_channels,
            tf_blocks=tf_blocks
        )
        
        if use_cuda and torch.cuda.is_available():
            self.cuda()
    
    def forward(self, mono, view):
        """
        Args:
            mono: B×1×T
            view: B×7 (position + orientation)
        Returns:
            y_binaural: B×2×T
            outputs: dict with Phase_L, Phase_R, Mag_L, Mag_R
        """
        B, _, T = mono.shape
        
        # 1. Shared encoding
        shared_feat = self.shared_encoder(mono)  # B×128×T
        
        # 2. Time Branch: 學習 Phase Difference（相對於 mono）
        Phase_diff_L, Phase_diff_R = self.time_branch(mono, shared_feat, view)  # B×F×T_stft
        
        # 3. Freq Branch: 學習 Magnitude
        Mag_L, Mag_R = self.freq_branch(shared_feat, mono, view)  # B×F×T_stft
        
        # 4. 計算 mono 的 phase
        window = torch.hann_window(self.n_fft).to(mono.device)
        Y_mono = torch.stft(mono.squeeze(1), self.n_fft, self.hop_size,
                           window=window, return_complex=True)  # B×F×T_stft
        Phase_mono = torch.angle(Y_mono)
        
        # 5. Fusion: Phase = Phase_mono + Phase_diff
        Phase_L = Phase_mono + Phase_diff_L
        Phase_R = Phase_mono + Phase_diff_R
        Y_L = Mag_L * torch.exp(1j * Phase_L)  # B×F×T_stft (complex)
        Y_R = Mag_R * torch.exp(1j * Phase_R)
        
        # 6. iSTFT
        y_L = torch.istft(Y_L, self.n_fft, self.hop_size, 
                         window=window, length=T)  # B×T
        y_R = torch.istft(Y_R, self.n_fft, self.hop_size,
                         window=window, length=T)
        
        # 7. Stack
        y_binaural = torch.stack([y_L, y_R], dim=1)  # B×2×T
        
        return y_binaural, {
            'Phase_L': Phase_diff_L,  # 返回 phase difference 用於 loss 計算
            'Phase_R': Phase_diff_R,
            'Mag_L': Mag_L,
            'Mag_R': Mag_R
        }
    
    def get_num_params(self):
        """計算參數量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 測試
    print("=" * 60)
    print("Testing HybridTFNet")
    print("=" * 60)
    print()
    
    # 測試完整的 HybridTFNet
    print("Testing HybridTFNet...")
    model = HybridTFNet(use_cuda=False)
    
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Forward pass
    mono = torch.randn(2, 1, 16000)
    view = torch.randn(2, 7)
    
    y_binaural, outputs = model(mono, view)
    
    print(f"\nInput mono shape:  {mono.shape}")
    print(f"Input view shape:  {view.shape}")
    print(f"Output shape:      {y_binaural.shape}")
    print(f"Phase_L shape:     {outputs['Phase_L'].shape}")
    print(f"Phase_R shape:     {outputs['Phase_R'].shape}")
    print(f"Mag_L shape:       {outputs['Mag_L'].shape}")
    print(f"Mag_R shape:       {outputs['Mag_R'].shape}")
    
    # 驗證輸出
    assert y_binaural.shape == (2, 2, 16000), f"Output shape mismatch: {y_binaural.shape}"
    assert outputs['Phase_L'].shape == outputs['Phase_R'].shape
    assert outputs['Mag_L'].shape == outputs['Mag_R'].shape
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
