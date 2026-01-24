"""
時間對齊工具模組 (Temporal Alignment Utilities)
用於雙耳語音合成 (Binaural Speech Synthesis)

此模組提供函數來偵測並修正 mono 和 binaural 音訊信號之間的時間偏移
使用互相關 (cross-correlation) 方法

作者: Antigravity AI Assistant
日期: 2026-01-24
目的: 診斷並修正 Phase Error 高的問題
"""

import numpy as np
import torch as th


def find_alignment_offset(mono, binaural, max_shift=2400, sample_rate=48000):
    """
    使用互相關找到 mono 和 binaural 信號之間的時間偏移
    Find the temporal offset between mono and binaural signals using cross-correlation.
    
    參數 Args:
        mono: Mono 音訊信號，形狀 (1, T) 或 (T,)
        binaural: Binaural 音訊信號，形狀 (2, T) 或 (T,)
        max_shift: 最大搜尋範圍（samples），預設 2400 = 50ms @ 48kHz
        sample_rate: 採樣率 Hz，預設 48000
    
    回傳 Returns:
        offset: 需要移動的 samples 數（正數 = mono 延遲）
        correlation: 最大互相關係數
    
    2026-01-24: 創建此函數以診斷 Phase Error 問題
    """
    # 轉換為 numpy (如果需要) - Convert to numpy if needed
    if isinstance(mono, th.Tensor):
        mono = mono.cpu().numpy()
    if isinstance(binaural, th.Tensor):
        binaural = binaural.cpu().numpy()
    
    # 確保 mono 是 1D - Ensure mono is 1D
    if mono.ndim > 1:
        mono = mono.flatten()
    
    # 使用 binaural 的左聲道作為參考 - Use left channel of binaural for correlation
    if binaural.ndim > 1:
        binaural_ref = binaural[0, :] if binaural.shape[0] == 2 else binaural[:, 0]
    else:
        binaural_ref = binaural
    
    # 裁剪到相同長度 - Trim to same length for correlation
    min_len = min(len(mono), len(binaural_ref))
    mono = mono[:min_len]
    binaural_ref = binaural_ref[:min_len]
    
    # 正規化信號（零均值、單位方差）- Normalize signals (zero-mean, unit-variance)
    mono = (mono - np.mean(mono)) / (np.std(mono) + 1e-8)
    binaural_ref = (binaural_ref - np.mean(binaural_ref)) / (np.std(binaural_ref) + 1e-8)
    
    # 使用 FFT 計算互相關（提高效率）- Compute cross-correlation using FFT for efficiency
    # 只使用前 10 秒的信號以加快速度 - Use only first 10 seconds for speed
    max_samples = min(min_len, sample_rate * 10)
    mono_segment = mono[:max_samples]
    binaural_segment = binaural_ref[:max_samples]
    
    # 透過 FFT 進行互相關 - Cross-correlation via FFT (2026-01-24)
    from scipy import signal
    correlation = signal.correlate(binaural_segment, mono_segment, mode='same', method='fft')
    
    # 在搜尋範圍內找到最大相關性的 lag - Find the lag with maximum correlation within search range
    center = len(correlation) // 2
    search_start = max(0, center - max_shift)
    search_end = min(len(correlation), center + max_shift)
    
    search_region = correlation[search_start:search_end]
    max_idx = np.argmax(search_region)
    
    # 轉換為偏移量（正數 = mono 延遲）- Convert to offset (positive = mono is delayed)
    offset = (search_start + max_idx) - center
    max_correlation = search_region[max_idx] / len(mono_segment)
    
    return int(offset), float(max_correlation)


def align_signals(mono, binaural, offset):
    """
    根據偵測到的偏移量對齊 mono 和 binaural 信號
    Align mono and binaural signals based on detected offset.
    
    參數 Args:
        mono: Mono 音訊信號，形狀 (1, T) 或 (C, T)
        binaural: Binaural 音訊信號，形狀 (2, T)
        offset: 需要移動的 samples 數（來自 find_alignment_offset）
    
    回傳 Returns:
        aligned_mono: 對齊後的 mono 信號
        aligned_binaural: 對齊後的 binaural 信號（裁剪以匹配長度）
    
    2026-01-24: 創建此函數以實作時間對齊
    """
    is_tensor = isinstance(mono, th.Tensor)
    
    if is_tensor:
        device = mono.device
        mono_np = mono.cpu().numpy()
        binaural_np = binaural.cpu().numpy()
    else:
        mono_np = mono
        binaural_np = binaural
    
    # 應用偏移量 - Apply offset (2026-01-24)
    if offset > 0:
        # Mono 延遲，裁剪 mono 的開頭 - Mono is delayed, trim beginning of mono
        aligned_mono = mono_np[:, offset:] if mono_np.ndim > 1 else mono_np[offset:]
        aligned_binaural = binaural_np
    elif offset < 0:
        # Binaural 延遲，裁剪 binaural 的開頭 - Binaural is delayed, trim beginning of binaural
        aligned_mono = mono_np
        aligned_binaural = binaural_np[:, -offset:] if binaural_np.ndim > 1 else binaural_np[-offset:]
    else:
        # 沒有偏移 - No offset
        aligned_mono = mono_np
        aligned_binaural = binaural_np
    
    # 裁剪到相同長度 - Trim to same length
    if aligned_mono.ndim > 1:
        min_len = min(aligned_mono.shape[-1], aligned_binaural.shape[-1])
        aligned_mono = aligned_mono[:, :min_len]
        aligned_binaural = aligned_binaural[:, :min_len]
    else:
        min_len = min(len(aligned_mono), len(aligned_binaural))
        aligned_mono = aligned_mono[:min_len]
        aligned_binaural = aligned_binaural[:min_len]
    
    # Convert back to tensor if needed
    if is_tensor:
        aligned_mono = th.from_numpy(aligned_mono).to(device)
        aligned_binaural = th.from_numpy(aligned_binaural).to(device)
    
    return aligned_mono, aligned_binaural


def diagnose_alignment(mono, binaural, offset, correlation, sample_rate=48000):
    """
    輸出對齊的診斷資訊
    Print diagnostic information about the alignment.
    
    參數 Args:
        mono: 原始 mono 信號
        binaural: 原始 binaural 信號
        offset: 偵測到的偏移量（samples）
        correlation: 互相關係數
        sample_rate: 採樣率 Hz
    
    2026-01-24: 創建此函數以輸出對齊診斷資訊
    """
    offset_ms = (offset / sample_rate) * 1000  # 轉換為毫秒 - Convert to milliseconds
    
    # Compute correlation before alignment
    if isinstance(mono, th.Tensor):
        mono_np = mono.cpu().numpy().flatten()
        binaural_np = binaural.cpu().numpy()[0, :] if binaural.ndim > 1 else binaural.cpu().numpy()
    else:
        mono_np = mono.flatten() if mono.ndim > 1 else mono
        binaural_np = binaural[0, :] if binaural.ndim > 1 else binaural
    
    min_len = min(len(mono_np), len(binaural_np))
    mono_segment = mono_np[:min_len]
    binaural_segment = binaural_np[:min_len]
    
    # Normalize
    mono_segment = (mono_segment - np.mean(mono_segment)) / (np.std(mono_segment) + 1e-8)
    binaural_segment = (binaural_segment - np.mean(binaural_segment)) / (np.std(binaural_segment) + 1e-8)
    
    # Correlation at zero lag
    corr_before = np.corrcoef(mono_segment, binaural_segment)[0, 1]
    
    print(f"  [ALIGNMENT] Detected offset: {offset:+d} samples ({offset_ms:+.2f} ms)")
    print(f"  [ALIGNMENT] Correlation: before={corr_before:.3f}, after={correlation:.3f}", end="")
    
    if abs(correlation - corr_before) > 0.05:
        print(" ✓ IMPROVED")
    else:
        print("")
    
    return {
        'offset_samples': offset,
        'offset_ms': offset_ms,
        'correlation_before': corr_before,
        'correlation_after': correlation
    }
