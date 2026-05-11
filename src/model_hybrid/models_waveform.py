"""
WaveformSpatializer: 時域直接學習雙耳空間化

完全避開 STFT 和 Phase wrapping 問題
直接在時域學習 HRTF-like filter
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViewEncoder(nn.Module):
    """視角編碼器"""
    def __init__(self, view_dim=7, hidden_dim=128, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(view_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, view):
        # view: B×7×K → B×7 (取平均)
        view_mean = view.mean(dim=-1)
        return self.net(view_mean)  # B×output_dim


class FilterGenerator(nn.Module):
    """生成時域 filter"""
    def __init__(self, view_dim=256, filter_len=512, hidden_dim=512):
        super().__init__()
        self.filter_len = filter_len
        
        self.net = nn.Sequential(
            nn.Linear(view_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, filter_len),
            nn.Tanh(),  # 限制 filter 幅度
        )
    
    def forward(self, view_feat):
        # view_feat: B×view_dim
        filter_weights = self.net(view_feat)  # B×filter_len
        # 歸一化
        filter_weights = filter_weights / (filter_weights.abs().sum(dim=-1, keepdim=True) + 1e-8)
        return filter_weights.unsqueeze(1)  # B×1×filter_len


class WaveformSpatializer(nn.Module):
    """
    時域雙耳空間化模型
    
    輸入:
        mono: B×1×T (時域波形)
        view: B×7×K (視角序列)
    
    輸出:
        binaural: B×2×T (雙耳波形)
    """
    def __init__(
        self,
        view_dim=7,
        view_hidden=128,
        view_output=256,
        filter_len=512,
        filter_hidden=512,
    ):
        super().__init__()
        
        self.view_encoder = ViewEncoder(view_dim, view_hidden, view_output)
        self.filter_gen_L = FilterGenerator(view_output, filter_len, filter_hidden)
        self.filter_gen_R = FilterGenerator(view_output, filter_len, filter_hidden)
        
        self.filter_len = filter_len
    
    def forward(self, mono, view):
        """
        Args:
            mono: B×1×T
            view: B×7×K
        
        Returns:
            binaural: B×2×T
        """
        B, _, T = mono.shape
        
        # 編碼視角
        view_feat = self.view_encoder(view)  # B×view_output
        
        # 生成 filter
        filter_L = self.filter_gen_L(view_feat)  # B×1×filter_len
        filter_R = self.filter_gen_R(view_feat)  # B×1×filter_len
        
        # 時域卷積 (模擬 HRTF filtering)
        # Padding 保持長度
        pad_left = self.filter_len // 2
        pad_right = self.filter_len - pad_left - 1
        mono_padded = F.pad(mono, (pad_left, pad_right), mode='constant', value=0)
        
        # 對每個樣本單獨卷積
        binaural_L = []
        binaural_R = []
        for i in range(B):
            # F.conv1d 需要 filter 是 out_channels×in_channels×kernel_size
            # 這裡 out=1, in=1, kernel=filter_len
            out_L = F.conv1d(
                mono_padded[i:i+1],  # 1×1×(T+filter_len-1)
                filter_L[i:i+1],     # 1×1×filter_len
            )
            out_R = F.conv1d(
                mono_padded[i:i+1],
                filter_R[i:i+1],
            )
            binaural_L.append(out_L)
            binaural_R.append(out_R)
        
        binaural_L = torch.cat(binaural_L, dim=0)[:, :, :T]  # B×1×T (截斷到原長度)
        binaural_R = torch.cat(binaural_R, dim=0)[:, :, :T]  # B×1×T
        
        # 合併
        binaural = torch.cat([binaural_L, binaural_R], dim=1)  # B×2×T
        
        return binaural
    
    def get_filters(self, view):
        """獲取生成的 filter (用於分析)"""
        view_feat = self.view_encoder(view)
        filter_L = self.filter_gen_L(view_feat)
        filter_R = self.filter_gen_R(view_feat)
        return filter_L, filter_R


def test_model():
    """測試模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 創建模型
    model = WaveformSpatializer(
        view_dim=7,
        view_hidden=128,
        view_output=256,
        filter_len=512,
        filter_hidden=512,
    ).to(device)
    
    # 計算參數量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # 測試前向傳播
    B, T, K = 4, 9600, 24  # 200ms @ 48kHz
    mono = torch.randn(B, 1, T).to(device)
    view = torch.randn(B, 7, K).to(device)
    
    print(f"\nInput shapes:")
    print(f"  mono: {mono.shape}")
    print(f"  view: {view.shape}")
    
    with torch.no_grad():
        binaural = model(mono, view)
    
    print(f"\nOutput shape:")
    print(f"  binaural: {binaural.shape}")
    
    # 測試 filter 獲取
    with torch.no_grad():
        filter_L, filter_R = model.get_filters(view)
    
    print(f"\nFilter shapes:")
    print(f"  filter_L: {filter_L.shape}")
    print(f"  filter_R: {filter_R.shape}")
    
    print("\n✅ Model test passed!")


if __name__ == '__main__':
    test_model()
