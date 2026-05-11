#!/usr/bin/env python3
"""
檢查 ±90° 樣本的實際 ITD 值
看看是否超過物理極限（被 clip 了）
"""

import numpy as np
import wave
from src.synthesis_utils import load_binaural_net, angle_to_tx_positions, chunked_forwarding

def calculate_raw_itd(binaural_audio, sample_rate=48000):
    """
    計算原始 ITD（不做 clipping）
    """
    if binaural_audio.shape[0] != 2:
        binaural_audio = binaural_audio.T
    
    left = binaural_audio[0].astype(np.float64)
    right = binaural_audio[1].astype(np.float64)
    
    # FFT
    n = len(left)
    nfft = 1 << int(np.ceil(np.log2(n)))
    
    left_fft = np.fft.rfft(left, n=nfft)
    right_fft = np.fft.rfft(right, n=nfft)
    
    # GCC-PHAT
    cross_spectrum = left_fft * np.conj(right_fft)
    magnitude = np.abs(cross_spectrum)
    magnitude[magnitude < 1e-10] = 1e-10
    gcc_phat = cross_spectrum / magnitude
    
    # 頻率遮罩 (300-8000 Hz)
    freqs = np.fft.rfftfreq(nfft, 1.0 / sample_rate)
    freq_mask = (freqs >= 300) & (freqs <= 8000)
    gcc_phat[~freq_mask] = 0
    
    # 反 FFT
    correlation = np.fft.irfft(gcc_phat, n=nfft)
    
    # 搜尋峰值
    max_lag = int(1.0e-3 * sample_rate)
    positive_lags = correlation[0:max_lag + 1]
    negative_lags = correlation[nfft - max_lag:]
    search_region = np.concatenate([negative_lags, positive_lags])
    
    peak_idx = np.argmax(search_region)
    delay_samples = peak_idx - max_lag
    
    # 計算 sin_theta（不做 clip）
    itd_seconds = delay_samples / sample_rate
    sound_speed = 343.0
    ear_distance = 0.215
    sin_theta = (sound_speed * itd_seconds) / ear_distance
    
    # 計算角度（如果 sin_theta > 1.0，會得到 NaN）
    if abs(sin_theta) <= 1.0:
        azimuth_deg = np.degrees(np.arcsin(sin_theta))
        azimuth_deg = -azimuth_deg
        clipped = False
    else:
        # 被 clip 了
        azimuth_deg = 90.0 if sin_theta > 0 else -90.0
        azimuth_deg = -azimuth_deg
        clipped = True
    
    return {
        'delay_samples': delay_samples,
        'itd_seconds': itd_seconds * 1e6,  # 轉成微秒
        'sin_theta': sin_theta,
        'azimuth_deg': azimuth_deg,
        'clipped': clipped
    }

def main():
    import torch as th
    
    print("檢查 ±90° 樣本的實際 ITD 值")
    print("=" * 80)
    
    # 載入模型
    net = load_binaural_net("outputs/binaural_network.newbob.net", blocks=3)
    
    # 載入 mono
    with wave.open("dataset/testset/subject1/mono.wav", 'r') as f:
        mono_data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    mono = th.from_numpy(mono_data.astype(np.float32) / 32768.0).unsqueeze(0)
    
    # 對齊到 400 的倍數
    target_samples = (mono.shape[-1] // 400) * 400
    mono = mono[:, :target_samples]
    num_frames = target_samples // 400
    
    # 測試角度
    test_angles = [-90, -75, -60, 60, 75, 90]
    
    for angle in test_angles:
        print(f"\n角度: {angle:+4d}°")
        print("-" * 80)
        
        # 生成 tx_positions
        view = th.from_numpy(angle_to_tx_positions(angle, distance=1.0, num_frames=num_frames))
        
        # 合成
        binaural = chunked_forwarding(net, mono, view)
        bio = binaural.numpy()
        
        # 計算原始 ITD
        result = calculate_raw_itd(bio)
        
        print(f"  ITD 延遲:     {result['delay_samples']:+6.1f} samples")
        print(f"  ITD 時間:     {result['itd_seconds']:+8.2f} μs")
        print(f"  sin(θ):       {result['sin_theta']:+7.4f}")
        print(f"  估計角度:     {result['azimuth_deg']:+7.2f}°")
        print(f"  是否被 clip:  {'是 ⚠️' if result['clipped'] else '否'}")
        
        if result['clipped']:
            # 計算「真實」角度（如果沒有 clip）
            # 這只是一個估計，因為超過 90° 後物理模型不再適用
            implied_angle = np.degrees(np.arcsin(np.clip(result['sin_theta'], -1, 1)))
            print(f"  ⚠️  ITD 對應的 sin(θ) = {result['sin_theta']:.4f} > 1.0")
            print(f"  ⚠️  這表示模型輸出的 ITD 超過了物理極限！")
    
    print("\n" + "=" * 80)
    print("結論:")
    print("  如果 ±90° 的 sin(θ) > 1.0，表示模型輸出的 ITD 超過物理極限")
    print("  GCC-PHAT 會將其 clip 到 ±90°，導致誤差顯示為 0.0")
    print("  這是一個「假的零誤差」，實際上模型在極端角度的表現可能很差")

if __name__ == "__main__":
    main()
