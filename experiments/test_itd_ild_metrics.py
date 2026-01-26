"""
測試 ITD 和 ILD 指標的正確性
Test correctness of ITD and ILD metrics

2026-01-25: 創建此腳本以驗證新實作的 ITD/ILD 指標
"""

import numpy as np
import torch as th
from src.losses import ITDLoss, ILDLoss


def create_synthetic_binaural_with_itd(duration_sec=1.0, sample_rate=48000, itd_ms=0.5):
    """
    創建具有已知 ITD 的合成雙耳信號
    Create synthetic binaural signal with known ITD
    
    :param duration_sec: 信號時長（秒）
    :param sample_rate: 採樣率
    :param itd_ms: ITD 值（毫秒）
    :return: 雙耳信號 (1, 2, T)
    """
    t = np.linspace(0, duration_sec, int(duration_sec * sample_rate))
    
    # 創建一個包含多個頻率的信號（300 Hz + 500 Hz）
    # Create signal with multiple frequencies (300 Hz + 500 Hz) for better ITD detection
    mono_signal = np.sin(2 * np.pi * 300 * t) + 0.5 * np.sin(2 * np.pi * 500 * t)
    
    # 計算 ITD 對應的樣本數
    itd_samples = int(itd_ms * sample_rate / 1000)
    
    # 創建左右聲道
    left = mono_signal
    right = np.zeros_like(mono_signal)
    
    if itd_samples > 0:
        # 正 ITD: 右聲道延遲
        right[itd_samples:] = mono_signal[:-itd_samples]
    elif itd_samples < 0:
        # 負 ITD: 左聲道延遲
        left[-itd_samples:] = mono_signal[:itd_samples]
    else:
        # 0 ITD: 左右聲道相同
        right = mono_signal
    
    # 組合成雙耳信號 (1, 2, T)
    stereo = np.stack([left, right], axis=0)
    return th.from_numpy(stereo).float().unsqueeze(0)


def create_synthetic_binaural_with_ild(duration_sec=1.0, sample_rate=48000, ild_db=6.0):
    """
    創建具有已知 ILD 的合成雙耳信號
    Create synthetic binaural signal with known ILD
    
    :param duration_sec: 信號時長（秒）
    :param sample_rate: 採樣率
    :param ild_db: ILD 值（dB）, ILD = 10*log10(E_left/E_right)
    :return: 雙耳信號 (1, 2, T)
    """
    t = np.linspace(0, duration_sec, int(duration_sec * sample_rate))
    
    # 使用高頻信號，因為 ILD 在高頻最有效
    # Use high-frequency signal since ILD is most effective at high frequencies
    # 使用 2 kHz 和 4 kHz
    mono_signal = np.sin(2 * np.pi * 2000 * t) + 0.5 * np.sin(2 * np.pi * 4000 * t)
    
    # 根據 ILD 計算右聲道的縮放因子
    # ILD = 10*log10(E_left/E_right) => E_right/E_left = 10^(-ILD/10)
    right_scale = np.sqrt(10 ** (-ild_db / 10))
    
    left = mono_signal
    right = mono_signal * right_scale
    
    # 組合成雙耳信號 (1, 2, T)
    stereo = np.stack([left, right], axis=0)
    return th.from_numpy(stereo).float().unsqueeze(0)


def test_itd_zero_error():
    """測試 ITD: 相同信號應該產生 0 誤差"""
    print("\n" + "="*60)
    print("測試 1: ITD 零誤差測試 (Zero Error Test)")
    print("="*60)
    
    # 創建兩個相同的信號
    signal1 = create_synthetic_binaural_with_itd(duration_sec=0.5, itd_ms=0.3)
    signal2 = signal1.clone()
    
    itd_loss = ITDLoss(sample_rate=48000, max_shift_ms=1.0)
    error = itd_loss(signal1, signal2).item()
    
    print(f"ITD 誤差 (相同信號): {error:.3f} μs")
    print(f"預期: 接近 0 μs")
    print(f"結果: {'✓ 通過' if error < 10 else '✗ 失敗'}")  # 允許 10μs 誤差
    return error < 10


def test_itd_known_delay():
    """測試 ITD: 已知延遲檢測"""
    print("\n" + "="*60)
    print("測試 2: ITD 已知延遲測試 (Known Delay Test)")
    print("="*60)
    
    # 創建兩個具有不同 ITD 的信號
    true_itd1_ms = 0.3  # 300 μs
    true_itd2_ms = 0.5  # 500 μs
    expected_diff_us = abs(true_itd1_ms - true_itd2_ms) * 1000  # 200 μs
    
    signal1 = create_synthetic_binaural_with_itd(duration_sec=0.5, itd_ms=true_itd1_ms)
    signal2 = create_synthetic_binaural_with_itd(duration_sec=0.5, itd_ms=true_itd2_ms)
    
    itd_loss = ITDLoss(sample_rate=48000, max_shift_ms=1.0)
    error = itd_loss(signal1, signal2).item()
    
    print(f"信號 1 ITD: {true_itd1_ms} ms")
    print(f"信號 2 ITD: {true_itd2_ms} ms")
    print(f"預期 ITD 差異: {expected_diff_us:.1f} μs")
    print(f"測量 ITD 誤差: {error:.1f} μs")
    
    # 允許 20% 誤差範圍
    tolerance = 0.2 * expected_diff_us
    passed = abs(error - expected_diff_us) < tolerance
    print(f"結果: {'✓ 通過' if passed else '✗ 失敗'} (容差: ±{tolerance:.1f} μs)")
    return passed


def test_ild_zero_error():
    """測試 ILD: 相同信號應該產生 0 誤差"""
    print("\n" + "="*60)
    print("測試 3: ILD 零誤差測試 (Zero Error Test)")
    print("="*60)
    
    # 創建兩個相同的信號
    signal1 = create_synthetic_binaural_with_ild(duration_sec=0.5, ild_db=3.0)
    signal2 = signal1.clone()
    
    ild_loss = ILDLoss(sample_rate=48000)
    error = ild_loss(signal1, signal2).item()
    
    print(f"ILD 誤差 (相同信號): {error:.3f} dB")
    print(f"預期: 接近 0 dB")
    print(f"結果: {'✓ 通過' if error < 0.5 else '✗ 失敗'}")  # 允許 0.5dB 誤差
    return error < 0.5


def test_ild_known_difference():
    """測試 ILD: 已知強度差檢測"""
    print("\n" + "="*60)
    print("測試 4: ILD 已知強度差測試 (Known Level Difference Test)")
    print("="*60)
    
    # 創建兩個具有不同 ILD 的信號
    true_ild1_db = 3.0
    true_ild2_db = 6.0
    expected_diff_db = abs(true_ild1_db - true_ild2_db)
    
    signal1 = create_synthetic_binaural_with_ild(duration_sec=0.5, ild_db=true_ild1_db)
    signal2 = create_synthetic_binaural_with_ild(duration_sec=0.5, ild_db=true_ild2_db)
    
    ild_loss = ILDLoss(sample_rate=48000)
    error = ild_loss(signal1, signal2).item()
    
    print(f"信號 1 ILD: {true_ild1_db} dB")
    print(f"信號 2 ILD: {true_ild2_db} dB")
    print(f"預期 ILD 差異: {expected_diff_db:.1f} dB")
    print(f"測量 ILD 誤差: {error:.1f} dB")
    
    # 允許 20% 誤差範圍
    tolerance = 0.2 * expected_diff_db + 0.5  # 加上基礎容差
    passed = abs(error - expected_diff_db) < tolerance
    print(f"結果: {'✓ 通過' if passed else '✗ 失敗'} (容差: ±{tolerance:.1f} dB)")
    return passed


def main():
    print("\n" + "="*60)
    print("ITD 和 ILD 指標單元測試")
    print("Unit Tests for ITD and ILD Metrics")
    print("="*60)
    
    results = []
    
    # 運行所有測試
    results.append(("ITD 零誤差測試", test_itd_zero_error()))
    results.append(("ITD 已知延遲測試", test_itd_known_delay()))
    results.append(("ILD 零誤差測試", test_ild_zero_error()))
    results.append(("ILD 已知強度差測試", test_ild_known_difference()))
    
    # 總結
    print("\n" + "="*60)
    print("測試總結 (Test Summary)")
    print("="*60)
    for test_name, passed in results:
        status = "✓ 通過" if passed else "✗ 失敗"
        print(f"{test_name}: {status}")
    
    total_passed = sum(1 for _, p in results if p)
    print(f"\n總計: {total_passed}/{len(results)} 測試通過")
    
    return all(p for _, p in results)


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
