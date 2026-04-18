"""
Direction of Arrival (DOA) Estimation Module
方位角估計模組

使用 GCC-PHAT (Generalized Cross-Correlation with Phase Transform) 算法
從雙耳音訊中估計聲源方位角

Author: 2026-02-16
Updated: 2026-02-19 - 使用完整音訊的 FFT 取代分段方式，解決效能問題
"""

import numpy as np
import torch as th


def gcc_phat_estimate(binaural_audio, sample_rate=48000, freq_range=(300, 8000)):
    """
    使用 GCC-PHAT 算法估計聲源方位角
    Estimate sound source azimuth using GCC-PHAT algorithm
    
    參數 Parameters:
        binaural_audio: 雙耳音訊 (2 x T) 或 (T x 2) numpy array or torch tensor
        sample_rate: 採樣率 (Hz)
        freq_range: 頻率範圍 (Hz)，僅使用此範圍內的頻率以提高穩定性
    
    返回 Returns:
        azimuth: 估計的方位角（度），範圍 [-90, +90]
                 0° = 正前方，+angle = 左方，-angle = 右方
    """
    # ====== 格式轉換 ======
    if isinstance(binaural_audio, th.Tensor):
        binaural_audio = binaural_audio.detach().cpu().numpy()
    
    if binaural_audio.ndim == 1:
        raise ValueError(f"Expected 2-channel audio, got 1D array shape {binaural_audio.shape}")
    
    if binaural_audio.shape[0] != 2:
        binaural_audio = binaural_audio.T
    
    if binaural_audio.shape[0] != 2:
        raise ValueError(f"Expected 2 channels, got shape {binaural_audio.shape}")
    
    left = binaural_audio[0].astype(np.float64)
    right = binaural_audio[1].astype(np.float64)
    
    # ====== 用完整訊號做 FFT（使用 next power of 2 加速）======
    n = len(left)
    nfft = 1 << int(np.ceil(np.log2(n)))  # next power of 2
    
    left_fft = np.fft.rfft(left, n=nfft)
    right_fft = np.fft.rfft(right, n=nfft)
    
    # ====== 互功率譜 + PHAT 加權 ======
    cross_spectrum = left_fft * np.conj(right_fft)
    magnitude = np.abs(cross_spectrum)
    magnitude[magnitude < 1e-10] = 1e-10
    gcc_phat = cross_spectrum / magnitude
    
    # ====== 頻率遮罩 ======
    freqs = np.fft.rfftfreq(nfft, 1.0 / sample_rate)
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    gcc_phat[~freq_mask] = 0
    
    # ====== 反 FFT 得到互相關 ======
    correlation = np.fft.irfft(gcc_phat, n=nfft)
    
    # ====== 在合理的 ITD 範圍內搜尋峰值 ======
    # 人類最大 ITD ≈ 0.8ms，搜尋 1.0ms
    max_lag = int(1.0e-3 * sample_rate)  # 48 samples @ 48kHz
    
    # correlation[0] = lag 0
    # correlation[1..max_lag] = positive lags (left leads right)
    # correlation[-max_lag...-1] = negative lags (right leads left)
    positive_lags = correlation[0:max_lag + 1]
    negative_lags = correlation[nfft - max_lag:]
    
    # 組合 [-max_lag, ..., -1, 0, +1, ..., +max_lag]
    search_region = np.concatenate([negative_lags, positive_lags])
    
    peak_idx = np.argmax(search_region)
    delay_samples = peak_idx - max_lag  # 中心 = max_lag
    
    # ====== ITD → 方位角 ======
    azimuth = itd_to_azimuth(delay_samples, sample_rate)
    
    return azimuth


def itd_to_azimuth(itd_samples, sample_rate=48000, ear_distance=0.215):
    """
    將 ITD (樣本數) 轉換為方位角（度）
    Convert ITD (in samples) to azimuth angle (in degrees)
    
    公式: θ = arcsin(c * τ / d)
    """
    itd_seconds = itd_samples / sample_rate
    sound_speed = 343.0
    
    sin_theta = (sound_speed * itd_seconds) / ear_distance
    sin_theta = np.clip(sin_theta, -1.0, 1.0)
    
    azimuth_deg = np.degrees(np.arcsin(sin_theta))
    
    # 修正：根據歷史數據，符號應該是正的
    # 之前的實現沒有負號，現在改回來以保持一致性
    # azimuth_deg = -azimuth_deg  # 這行導致所有角度反向
    
    return azimuth_deg


def validate_angle(angle):
    """驗證估計角度是否在合理範圍"""
    is_valid = (-90 <= angle <= 90)
    clamped_angle = np.clip(angle, -90, 90)
    if not is_valid:
        print(f"Warning: Estimated angle {angle:.1f}° out of range, clamped to {clamped_angle:.1f}°")
    return is_valid, clamped_angle


def ild_estimate(binaural_audio, sample_rate=48000, freq_range=(1000, 8000), method='rms'):
    """
    使用 ILD (Interaural Level Difference) 估計聲源方位角
    Estimate sound source azimuth using ILD
    
    ILD 在高頻（>1kHz）較為明顯，因為頭部遮蔽效應（head shadow effect）
    對高頻聲波的衰減更顯著
    
    參數 Parameters:
        binaural_audio: 雙耳音訊 (2 x T) 或 (T x 2) numpy array or torch tensor
        sample_rate: 採樣率 (Hz)
        freq_range: 頻率範圍 (Hz)，建議使用高頻段 (1000-8000 Hz)
        method: 能量計算方法
            'rms': RMS (Root Mean Square) 能量
            'peak': 峰值能量
            'spectral': 頻譜能量（在指定頻率範圍內）
    
    返回 Returns:
        azimuth: 估計的方位角（度），範圍 [-90, +90]
                 0° = 正前方，+angle = 左方，-angle = 右方
    """
    # ====== 格式轉換 ======
    if isinstance(binaural_audio, th.Tensor):
        binaural_audio = binaural_audio.detach().cpu().numpy()
    
    if binaural_audio.ndim == 1:
        raise ValueError(f"Expected 2-channel audio, got 1D array shape {binaural_audio.shape}")
    
    if binaural_audio.shape[0] != 2:
        binaural_audio = binaural_audio.T
    
    if binaural_audio.shape[0] != 2:
        raise ValueError(f"Expected 2 channels, got shape {binaural_audio.shape}")
    
    left = binaural_audio[0].astype(np.float64)
    right = binaural_audio[1].astype(np.float64)
    
    # ====== 計算能量 ======
    if method == 'rms':
        left_energy = np.sqrt(np.mean(left ** 2))
        right_energy = np.sqrt(np.mean(right ** 2))
    
    elif method == 'peak':
        left_energy = np.max(np.abs(left))
        right_energy = np.max(np.abs(right))
    
    elif method == 'spectral':
        # 使用頻譜能量（在指定頻率範圍內）
        n = len(left)
        nfft = 1 << int(np.ceil(np.log2(n)))
        
        left_fft = np.fft.rfft(left, n=nfft)
        right_fft = np.fft.rfft(right, n=nfft)
        
        freqs = np.fft.rfftfreq(nfft, 1.0 / sample_rate)
        freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        
        left_energy = np.sqrt(np.mean(np.abs(left_fft[freq_mask]) ** 2))
        right_energy = np.sqrt(np.mean(np.abs(right_fft[freq_mask]) ** 2))
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # ====== 計算 ILD (dB) ======
    # 避免除以零
    if left_energy < 1e-10 or right_energy < 1e-10:
        return 0.0
    
    ild_db = 20 * np.log10(left_energy / right_energy)
    
    # ====== ILD → 方位角 ======
    azimuth = ild_to_azimuth(ild_db)
    
    return azimuth


def ild_to_azimuth(ild_db, max_ild=20.0, lookup_table=None):
    """
    將 ILD (dB) 轉換為方位角（度）
    Convert ILD (in dB) to azimuth angle (in degrees)
    
    參數 Parameters:
        ild_db: ILD 值（dB），正值表示左耳較響（聲源在左側）
        max_ild: 最大 ILD 值（dB），對應 ±90°（僅在 lookup_table=None 時使用）
                 典型值：15-25 dB（取決於頻率和頭部大小）
        lookup_table: 可選的查找表 (ild_values, angle_values)
                     如果提供，將使用插值而非線性映射
    
    ⚠️ 警告：線性映射是粗糙的近似，實際 ILD-角度關係是非線性的
    建議使用 diagnose_ild.py 測量實際 ILD 值，然後建立查找表
    
    使用簡化的線性映射（不推薦）：
    - ILD = 0 dB → 0°（正前方）
    - ILD = +max_ild dB → +90°（左側）
    - ILD = -max_ild dB → -90°（右側）
    
    注意：實際的 ILD-角度關係是非線性的，且與頻率相關
    更精確的方法需要使用 HRTF 數據庫建立查找表
    """
    if lookup_table is not None:
        # 使用查找表插值
        from scipy.interpolate import interp1d
        ild_values, angle_values = lookup_table
        interp_func = interp1d(
            ild_values, angle_values, 
            kind='cubic', 
            fill_value='extrapolate',
            bounds_error=False
        )
        azimuth_deg = float(interp_func(ild_db))
    else:
        # 線性映射（不準確）
        azimuth_deg = (ild_db / max_ild) * 90.0
    
    # 限制範圍
    azimuth_deg = np.clip(azimuth_deg, -90.0, 90.0)
    
    return azimuth_deg


def hybrid_estimate(binaural_audio, sample_rate=48000, 
                   itd_weight=0.7, ild_weight=0.3,
                   freq_range_itd=(300, 3000), 
                   freq_range_ild=(2000, 8000)):
    """
    混合 ITD 和 ILD 的方位角估計
    Hybrid azimuth estimation combining ITD and ILD
    
    ITD 在低頻較準確，ILD 在高頻較準確
    混合方法可以提高整體估計的穩定性
    
    參數 Parameters:
        binaural_audio: 雙耳音訊 (2 x T)
        sample_rate: 採樣率 (Hz)
        itd_weight: ITD 估計的權重
        ild_weight: ILD 估計的權重
        freq_range_itd: ITD 使用的頻率範圍（低頻）
        freq_range_ild: ILD 使用的頻率範圍（高頻）
    
    返回 Returns:
        azimuth: 估計的方位角（度）
    """
    # ITD 估計（使用低頻）
    azimuth_itd = gcc_phat_estimate(binaural_audio, sample_rate, freq_range=freq_range_itd)
    
    # ILD 估計（使用高頻）
    azimuth_ild = ild_estimate(binaural_audio, sample_rate, 
                               freq_range=freq_range_ild, method='spectral')
    
    # 加權平均
    azimuth = itd_weight * azimuth_itd + ild_weight * azimuth_ild
    
    return np.clip(azimuth, -90.0, 90.0)


if __name__ == "__main__":
    print("Testing DOA estimation methods...")
    print("=" * 50)
    
    sr = 48000
    t = np.linspace(0, 0.5, sr // 2, endpoint=False)
    sig = np.sin(2 * np.pi * 1000 * t)
    
    # ====== Test 1: GCC-PHAT (ITD) ======
    print("\n1. GCC-PHAT (ITD-based):")
    
    # 0°
    a = gcc_phat_estimate(np.stack([sig, sig]), sr)
    print(f"  0° -> {a:+.1f}°")
    
    # ~-30° (right, left leads by 15 samples)
    d = 15
    a = gcc_phat_estimate(np.stack([sig, np.concatenate([np.zeros(d), sig[:-d]])]), sr)
    print(f"  ~-30° (right) -> {a:+.1f}°")
    
    # ~+30° (left, right leads by 15 samples)
    a = gcc_phat_estimate(np.stack([np.concatenate([np.zeros(d), sig[:-d]]), sig]), sr)
    print(f"  ~+30° (left) -> {a:+.1f}°")
    
    # ====== Test 2: ILD ======
    print("\n2. ILD (Intensity-based):")
    
    # 0° (equal level)
    a = ild_estimate(np.stack([sig, sig]), sr, method='rms')
    print(f"  0° (equal) -> {a:+.1f}°")
    
    # +45° (left louder)
    left_loud = sig * 1.5
    right_quiet = sig * 0.8
    a = ild_estimate(np.stack([left_loud, right_quiet]), sr, method='rms')
    print(f"  +45° (left louder) -> {a:+.1f}°")
    
    # -45° (right louder)
    left_quiet = sig * 0.8
    right_loud = sig * 1.5
    a = ild_estimate(np.stack([left_quiet, right_loud]), sr, method='rms')
    print(f"  -45° (right louder) -> {a:+.1f}°")
    
    # ====== Test 3: Hybrid ======
    print("\n3. Hybrid (ITD + ILD):")
    
    # Simulate +30° with both ITD and ILD
    d = 15
    left_sig = np.concatenate([np.zeros(d), sig[:-d]]) * 1.3
    right_sig = sig * 0.9
    a = hybrid_estimate(np.stack([left_sig, right_sig]), sr)
    print(f"  ~+30° (hybrid) -> {a:+.1f}°")
    
    print("\n" + "=" * 50)
    print("Done.")
