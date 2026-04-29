"""
HybridTFNet-IPD: 只學習 IPD (Interaural Phase Difference)

改進：
1. Time Branch 只預測 IPD，不預測 Phase_L 和 Phase_R
2. 從 IPD 對稱地生成 Phase_L 和 Phase_R
3. 使用 gradient checkpointing 節省記憶體

作者：AI Engineer Agent
日期：2026-04-27
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ==================== Position Encoder ====================

class PositionEncoder(nn.Module):
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
        if view.dim() == 3:
            view = view.mean(dim=-1)
        return self.encoder(view)


# ==================== Time Branch (IPD-only) ====================

class TimeBranchIPD(nn.Module):
    """
    只預測 IPD (Interaural Phase Difference)
    Phase_L = Phase_mono + IPD/2
    Phase_R = Phase_mono - IPD/2
    """
    def __init__(self, 
                 sample_rate=48000,
                 n_fft=1024,
                 hop_size=64,
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
        
        # 3. FiLM 調制
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
        
        # 4. IPD Predictor
        self.ipd_net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        
        # 5. 輸出層（只有一個 IPD）
        self.output_ipd = nn.Conv2d(channels, 1, kernel_size=1)
    
    def forward(self, mono, shared_feat, view):
        """
        Args:
            mono: B×1×T
            shared_feat: B×128×T (unused)
            view: B×7 or B×7×K
        Returns:
            IPD: B×F×T_stft
            Phase_L: B×F×T_stft
            Phase_R: B×F×T_stft
        """
        # 1. STFT
        window = torch.hann_window(self.n_fft).to(mono.device)
        Y_mono = torch.stft(mono.squeeze(1), self.n_fft, self.hop_size,
                           window=window, return_complex=True)
        stft_feat = torch.stack([Y_mono.real, Y_mono.imag], dim=1)
        
        # 2. 位置編碼
        pos_feat = self.position_encoder(view)
        
        # 3. STFT 特徵編碼
        phase_feat = self.stft_encoder(stft_feat)
        
        # 4. FiLM 調制
        gamma = self.gamma_net(pos_feat).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_net(pos_feat).unsqueeze(-1).unsqueeze(-1)
        phase_feat = gamma * phase_feat + beta
        
        # 5. IPD prediction
        phase_feat = self.ipd_net(phase_feat)
        IPD = self.output_ipd(phase_feat).squeeze(1)  # B×F×T_stft
        
        # 6. 從 IPD 對稱生成 Phase_L 和 Phase_R
        Phase_mono = torch.angle(Y_mono)
        Phase_L = Phase_mono + IPD / 2
        Phase_R = Phase_mono - IPD / 2
        
        return IPD, Phase_L, Phase_R


# ==================== Freq Branch ====================

class SimpleTFResStack(nn.Module):
    def __init__(self, channels=128, num_blocks=4):
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
        for block in self.blocks:
            residual = x
            x = block(x)
            x = x + residual
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim=128, num_heads=4):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)
        
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, audio_feat, pos_feat):
        B, C = pos_feat.shape
        
        if audio_feat.dim() == 3:
            _, _, T = audio_feat.shape
            F_bins, T_stft = 1, T
            audio_feat_4d = audio_feat.unsqueeze(2)
        else:
            _, _, F_bins, T_stft = audio_feat.shape
            audio_feat_4d = audio_feat
        
        Q = self.q_proj(pos_feat)
        Q = Q.view(B, self.num_heads, self.head_dim, 1)
        
        K = self.k_proj(audio_feat_4d)
        K = K.view(B, self.num_heads, self.head_dim, F_bins*T_stft)
        
        V = self.v_proj(audio_feat_4d)
        V = V.view(B, self.num_heads, self.head_dim, F_bins*T_stft)
        
        attn = torch.matmul(Q.transpose(2, 3), K) / (self.head_dim ** 0.5)
        attn = torch.nn.functional.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, V.transpose(2, 3))
        out = out.transpose(2, 3).contiguous()
        out = out.view(B, C, 1, 1)
        out = out.expand(-1, -1, F_bins, T_stft)
        out = self.out_proj(out)
        
        if audio_feat.dim() == 3:
            return audio_feat + out.squeeze(2)
        else:
            return audio_feat + out


class FreqBranch(nn.Module):
    def __init__(self,
                 n_fft=1024,
                 hop_size=64,
                 tf_channels=128,
                 tf_blocks=4,
                 num_heads=4):
        super().__init__()
        
        self.n_fft = n_fft
        self.hop_size = hop_size
        
        self.time_to_freq = nn.Conv1d(128, tf_channels, kernel_size=1)
        
        self.freq_encoder = nn.Sequential(
            nn.Conv2d(tf_channels, tf_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(tf_channels, tf_channels, kernel_size=3, padding=1),
        )
        
        self.position_encoder = PositionEncoder(
            view_dim=7,
            hidden_dim=128,
            output_dim=tf_channels
        )
        
        self.cross_attn = CrossAttentionBlock(dim=tf_channels, num_heads=num_heads)
        
        self.tf_resstack = SimpleTFResStack(
            channels=tf_channels,
            num_blocks=tf_blocks
        )
        
        self.output_L = nn.Conv2d(tf_channels, 1, kernel_size=1)
        self.output_R = nn.Conv2d(tf_channels, 1, kernel_size=1)
    
    def forward(self, shared_feat, mono, view):
        B, _, T = mono.shape
        
        window = torch.hann_window(self.n_fft).to(mono.device)
        Y_mono = torch.stft(mono.squeeze(1), self.n_fft, self.hop_size,
                           window=window, return_complex=True)
        
        shared_feat_freq = self.time_to_freq(shared_feat)
        shared_feat_freq = F.interpolate(
            shared_feat_freq, 
            size=Y_mono.shape[-1],
            mode='linear',
            align_corners=False
        )
        
        F_bins = Y_mono.shape[1]
        shared_feat_freq = shared_feat_freq.unsqueeze(2).expand(
            -1, -1, F_bins, -1
        )
        
        freq_feat = self.freq_encoder(shared_feat_freq)
        
        pos_feat = self.position_encoder(view)
        
        freq_feat = self.cross_attn(freq_feat, pos_feat)
        
        feat = self.tf_resstack(freq_feat)
        
        Mag_L = F.softplus(self.output_L(feat).squeeze(1))
        Mag_R = F.softplus(self.output_R(feat).squeeze(1))
        
        return Mag_L, Mag_R


# ==================== HybridTFNet-IPD ====================

class HybridTFNetIPD(nn.Module):
    """
    HybridTFNet-IPD: 只學習 IPD
    
    改進：
    1. Time Branch 只預測 IPD
    2. 使用 gradient checkpointing
    3. 減少模型容量（channels=128, blocks=4）
    """
    def __init__(self,
                 sample_rate=48000,
                 n_fft=1024,
                 hop_size=64,
                 tf_channels=128,
                 tf_blocks=4,
                 use_checkpointing=True,
                 use_cuda=True):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.use_checkpointing = use_checkpointing
        self.use_cuda = use_cuda
        
        # Shared Encoder
        self.shared_encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        
        # Time Branch (IPD-only)
        self.time_branch = TimeBranchIPD(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_size=hop_size,
            channels=tf_channels
        )
        
        # Freq Branch
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
            view: B×7
        Returns:
            y_binaural: B×2×T
            outputs: dict with IPD, Phase_L, Phase_R, Mag_L, Mag_R
        """
        B, _, T = mono.shape
        
        # 1. Shared encoding
        if self.use_checkpointing and self.training:
            shared_feat = checkpoint(self.shared_encoder, mono, use_reentrant=False)
        else:
            shared_feat = self.shared_encoder(mono)
        
        # 2. Time Branch: 學習 IPD
        IPD, Phase_L, Phase_R = self.time_branch(mono, shared_feat, view)
        
        # 3. Freq Branch: 學習 Magnitude
        Mag_L, Mag_R = self.freq_branch(shared_feat, mono, view)
        
        # 4. Fusion
        Y_L = Mag_L * torch.exp(1j * Phase_L)
        Y_R = Mag_R * torch.exp(1j * Phase_R)
        
        # 5. iSTFT
        window = torch.hann_window(self.n_fft).to(mono.device)
        y_L = torch.istft(Y_L, self.n_fft, self.hop_size, 
                         window=window, length=T)
        y_R = torch.istft(Y_R, self.n_fft, self.hop_size,
                         window=window, length=T)
        
        # 6. Stack
        y_binaural = torch.stack([y_L, y_R], dim=1)
        
        return y_binaural, {
            'IPD': IPD,
            'Phase_L': Phase_L,
            'Phase_R': Phase_R,
            'Mag_L': Mag_L,
            'Mag_R': Mag_R
        }
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing HybridTFNet-IPD")
    print("=" * 60)
    
    model = HybridTFNetIPD(use_cuda=False, use_checkpointing=False)
    
    print(f"Model parameters: {model.get_num_params():,}")
    
    mono = torch.randn(2, 1, 9600)  # 200ms @ 48kHz
    view = torch.randn(2, 7)
    
    y_binaural, outputs = model(mono, view)
    
    print(f"\nInput mono shape:  {mono.shape}")
    print(f"Input view shape:  {view.shape}")
    print(f"Output shape:      {y_binaural.shape}")
    print(f"IPD shape:         {outputs['IPD'].shape}")
    print(f"Phase_L shape:     {outputs['Phase_L'].shape}")
    print(f"Phase_R shape:     {outputs['Phase_R'].shape}")
    print(f"Mag_L shape:       {outputs['Mag_L'].shape}")
    print(f"Mag_R shape:       {outputs['Mag_R'].shape}")
    
    print("\n✅ All tests passed!")
