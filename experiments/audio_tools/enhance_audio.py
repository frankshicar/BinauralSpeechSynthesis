#!/usr/bin/env python3
"""
音質後處理增強模組
無需重新訓練，直接改善雙耳音訊品質
"""

import torch
import numpy as np
from scipy import signal


def enhance_binaural_audio(audio, sr=48000, config=None):
    """
    後處理增強音質
    
    Args:
        audio: torch.Tensor [2, T] (L/R channels) 或 [T] (mono)
        sr: 採樣率
        config: 增強參數配置
    
    Returns:
        enhanced: torch.Tensor 增強後的音訊
    """
    if config is None:
        config = {
            'high_freq_boost': 0.15,      # 高頻增強強度 (0-0.3)
            'compression_strength': 1.2,   # 動態壓縮強度 (1.0-1.5)
            'denoise_threshold': -60,      # 去噪閾值 dB (-70 to -50)
            'dc_remove': True,             # 移除直流偏移
        }
    
    # 處理單聲道或雙聲道
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    
    enhanced = []
    
    for channel in audio:
        # 1. 去除直流偏移
        if config['dc_remove']:
            channel = channel - channel.mean()
        
        # 2. 高頻增強（提升清晰度）
        if config['high_freq_boost'] > 0:
            channel = boost_high_frequency(
                channel, sr, 
                boost=config['high_freq_boost']
            )
        
        # 3. 動態範圍壓縮（減少失真）
        if config['compression_strength'] > 1.0:
            channel = dynamic_range_compression(
                channel, 
                strength=config['compression_strength']
            )
        
        # 4. 頻譜去噪
        channel = denoise_spectral(
            channel, sr, 
            noise_floor=config['denoise_threshold']
        )
        
        enhanced.append(channel)
    
    enhanced = torch.stack(enhanced)
    
    # 如果輸入是單聲道，返回單聲道
    if audio.shape[0] == 1:
        enhanced = enhanced.squeeze(0)
    
    return enhanced


def boost_high_frequency(audio, sr, cutoff=4000, boost=0.15):
    """
    高頻增強
    
    Args:
        audio: torch.Tensor [T]
        sr: 採樣率
        cutoff: 高通濾波器截止頻率 (Hz)
        boost: 增強強度 (0-0.3)
    """
    # 設計高通濾波器
    sos = signal.butter(4, cutoff, 'hp', fs=sr, output='sos')
    
    # 提取高頻
    high_freq = signal.sosfilt(sos, audio.cpu().numpy())
    high_freq = torch.from_numpy(high_freq).float().to(audio.device)
    
    # 混合
    enhanced = audio + boost * high_freq
    
    return enhanced


def dynamic_range_compression(audio, strength=1.2):
    """
    動態範圍壓縮（軟削波）
    
    Args:
        audio: torch.Tensor [T]
        strength: 壓縮強度 (1.0-1.5)
    """
    # 使用 tanh 進行軟壓縮
    compressed = torch.tanh(audio * strength) / strength
    
    return compressed


def denoise_spectral(audio, sr, noise_floor=-60, n_fft=2048, hop_length=512):
    """
    頻譜去噪（軟閾值）
    
    Args:
        audio: torch.Tensor [T]
        sr: 採樣率
        noise_floor: 噪音閾值 (dB)
        n_fft: FFT 大小
        hop_length: 跳躍長度
    """
    # STFT
    stft = torch.stft(
        audio, 
        n_fft=n_fft, 
        hop_length=hop_length,
        window=torch.hann_window(n_fft).to(audio.device),
        return_complex=True
    )
    
    # 計算噪音閾值
    mag = stft.abs()
    threshold = mag.max() * (10 ** (noise_floor / 20))
    
    # 軟閾值遮罩
    mask = torch.sigmoid((mag - threshold) * 10)
    stft_clean = stft * mask
    
    # iSTFT
    audio_clean = torch.istft(
        stft_clean, 
        n_fft=n_fft, 
        hop_length=hop_length,
        window=torch.hann_window(n_fft).to(audio.device)
    )
    
    # 長度對齊
    if audio_clean.shape[0] < audio.shape[0]:
        audio_clean = torch.nn.functional.pad(
            audio_clean, 
            (0, audio.shape[0] - audio_clean.shape[0])
        )
    else:
        audio_clean = audio_clean[:audio.shape[0]]
    
    return audio_clean


def diagnose_audio_quality(pred, target, sr=48000):
    """
    診斷音質問題
    
    Args:
        pred: torch.Tensor [T] 預測音訊
        target: torch.Tensor [T] 目標音訊
        sr: 採樣率
    
    Returns:
        dict: 診斷結果
    """
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    
    # 1. 頻譜對比
    pred_spec = np.fft.rfft(pred_np)
    target_spec = np.fft.rfft(target_np)
    
    spectral_diff = np.abs(pred_spec - target_spec).mean()
    
    # 2. 高頻能量比
    mid_point = len(pred_spec) // 2
    high_freq_pred = np.abs(pred_spec[mid_point:]).mean()
    high_freq_target = np.abs(target_spec[mid_point:]).mean()
    high_freq_ratio = high_freq_pred / (high_freq_target + 1e-8)
    
    # 3. 信噪比
    noise = pred_np - target_np
    snr = 10 * np.log10(np.sum(target_np**2) / (np.sum(noise**2) + 1e-8))
    
    # 4. 相位連續性
    pred_stft = np.fft.rfft(pred_np.reshape(-1, 512), axis=1)
    phase_diff = np.diff(np.angle(pred_stft), axis=0)
    phase_discontinuity = np.abs(phase_diff).mean()
    
    results = {
        'spectral_diff': float(spectral_diff),
        'high_freq_ratio': float(high_freq_ratio),
        'snr_db': float(snr),
        'phase_discontinuity': float(phase_discontinuity)
    }
    
    # 診斷建議
    print("\n=== 音質診斷結果 ===")
    print(f"頻譜差異: {spectral_diff:.4f}")
    print(f"高頻能量比: {high_freq_ratio:.2f} ", end="")
    if high_freq_ratio < 0.7:
        print("⚠️  高頻不足，聲音會悶")
    elif high_freq_ratio > 1.3:
        print("⚠️  高頻過強，可能刺耳")
    else:
        print("✓ 正常")
    
    print(f"信噪比: {snr:.2f} dB ", end="")
    if snr < 20:
        print("⚠️  噪音較大")
    elif snr > 40:
        print("✓ 優秀")
    else:
        print("✓ 良好")
    
    print(f"相位不連續性: {phase_discontinuity:.4f} ", end="")
    if phase_discontinuity > 1.0:
        print("⚠️  相位不連續，可能有金屬音")
    else:
        print("✓ 正常")
    
    return results


if __name__ == "__main__":
    # 測試範例
    print("音質增強模組測試")
    
    # 生成測試信號
    sr = 48000
    duration = 2.0
    t = torch.linspace(0, duration, int(sr * duration))
    
    # 模擬雙耳音訊（帶噪音）
    audio_L = torch.sin(2 * np.pi * 440 * t) + 0.1 * torch.randn_like(t)
    audio_R = torch.sin(2 * np.pi * 440 * t + 0.1) + 0.1 * torch.randn_like(t)
    binaural = torch.stack([audio_L, audio_R])
    
    print(f"輸入音訊形狀: {binaural.shape}")
    
    # 增強
    enhanced = enhance_binaural_audio(binaural, sr=sr)
    
    print(f"增強後音訊形狀: {enhanced.shape}")
    print("✓ 測試通過")
