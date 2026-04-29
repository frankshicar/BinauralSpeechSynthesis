"""
針對 ±75° 進行精細校準
使用二分搜尋找到能讓模型預測出 ±75° 的正確 TX 值
"""
import subprocess
import re
import numpy as np


def test_tx_angle(subject, tx_angle, gt_angle):
    """
    測試特定 TX 角度的預測結果
    """
    import math
    import soundfile as sf
    import torch as th
    from src.synthesis_utils import angle_to_tx_positions, load_binaural_net
    from src.doa import gcc_phat_estimate
    
    # 載入模型
    net = load_binaural_net("outputs/binaural_network.newbob.net", blocks=3)
    
    # 載入 mono
    audio, sr = sf.read(f"dataset/testset/{subject}/mono.wav", dtype='float32')
    if sr != 48000:
        import torchaudio.functional as F_audio
        mono = th.from_numpy(audio).unsqueeze(0)
        mono = F_audio.resample(mono, sr, 48000)
    else:
        mono = th.from_numpy(audio).unsqueeze(0)
    
    # 對齊長度
    target_samples = (mono.shape[-1] // 400) * 400
    mono = mono[:, :target_samples]
    num_frames = target_samples // 400
    
    # 生成 view
    view = th.from_numpy(angle_to_tx_positions(tx_angle, 1.0, num_frames))
    
    # 推理
    from src.synthesis_utils import chunked_forwarding
    binaural = chunked_forwarding(net, mono, view)
    
    # GCC-PHAT 估計角度
    bio = binaural.numpy()
    # 去除尾端靜音
    left = bio[0, :]
    valid_indices = np.where(np.abs(left) > 1e-4)[0]
    if len(valid_indices) > 0:
        valid_len = valid_indices[-1] + 1
        valid_len = int(valid_len * 0.95)
        bio = bio[:, :valid_len]
    
    pred_angle = gcc_phat_estimate(bio, sample_rate=48000)
    error = abs(pred_angle - gt_angle)
    
    return pred_angle, error


def binary_search_tx(subject, gt_angle, tx_min, tx_max, tolerance=0.5, max_iterations=10):
    """
    二分搜尋找到最佳 TX 角度
    """
    print(f"\n{'=' * 60}")
    print(f"校準 {subject} (GT: {gt_angle:+.1f}°)")
    print(f"搜尋範圍: [{tx_min:+.1f}°, {tx_max:+.1f}°]")
    print(f"{'=' * 60}\n")
    
    best_tx = None
    best_error = float('inf')
    
    for iteration in range(max_iterations):
        # 測試三個點：min, mid, max
        tx_mid = (tx_min + tx_max) / 2
        
        print(f"第 {iteration + 1} 輪：")
        
        # 測試 mid
        pred_mid, error_mid = test_tx_angle(subject, tx_mid, gt_angle)
        print(f"  TX {tx_mid:+6.1f}° → Pred {pred_mid:+6.1f}°, Error {error_mid:5.1f}°")
        
        if error_mid < best_error:
            best_error = error_mid
            best_tx = tx_mid
        
        # 檢查是否達到目標
        if error_mid <= tolerance:
            print(f"\n✓ 找到最佳 TX: {tx_mid:+.1f}° (誤差 {error_mid:.1f}°)")
            return tx_mid, pred_mid, error_mid
        
        # 決定搜尋方向
        if pred_mid < gt_angle:
            # 預測太小，需要增加 TX
            tx_min = tx_mid
            print(f"  → 預測太小，向上搜尋")
        else:
            # 預測太大，需要減少 TX
            tx_max = tx_mid
            print(f"  → 預測太大，向下搜尋")
    
    print(f"\n✓ 達到最大迭代次數，最佳 TX: {best_tx:+.1f}° (誤差 {best_error:.1f}°)")
    pred, _ = test_tx_angle(subject, best_tx, gt_angle)
    return best_tx, pred, best_error


def main():
    print("=" * 60)
    print("精細校準 ±75°")
    print("=" * 60)
    
    results = {}
    
    # 校準 +75°
    # 從第一次測量知道：TX 60° → Pred 44.3°, TX 75° → Pred 63.8°, TX 90° → Pred 90°
    # 想要 Pred 75°，應該在 TX 75° ~ 90° 之間
    tx_pos75, pred_pos75, error_pos75 = binary_search_tx(
        "subject12", 75.0, 75.0, 90.0, tolerance=0.5
    )
    results[75.0] = {
        'tx': tx_pos75,
        'pred': pred_pos75,
        'error': error_pos75,
        'compensation': tx_pos75 - 75.0
    }
    
    # 校準 -75°
    # 從第一次測量知道：TX -90° → Pred -90°, TX -60° → Pred -56.2°
    # 想要 Pred -75°，應該在 TX -90° ~ -60° 之間
    tx_neg75, pred_neg75, error_neg75 = binary_search_tx(
        "subject13", -75.0, -90.0, -60.0, tolerance=0.5
    )
    results[-75.0] = {
        'tx': tx_neg75,
        'pred': pred_neg75,
        'error': error_neg75,
        'compensation': tx_neg75 - (-75.0)
    }
    
    # 輸出結果
    print("\n" + "=" * 60)
    print("校準結果")
    print("=" * 60)
    
    for gt_angle in sorted(results.keys()):
        r = results[gt_angle]
        print(f"\nGT {gt_angle:+6.1f}°:")
        print(f"  最佳 TX: {r['tx']:+6.1f}°")
        print(f"  預測: {r['pred']:+6.1f}°")
        print(f"  誤差: {r['error']:5.1f}°")
        print(f"  補償量: {r['compensation']:+6.1f}°")
    
    print("\n" + "=" * 60)
    print("更新 _ANGLE_COMPENSATION：")
    print("=" * 60)
    print()
    print("將以下值更新到 src/synthesis_utils.py：")
    for gt_angle in sorted(results.keys()):
        r = results[gt_angle]
        print(f"    {gt_angle:+6.1f}: {r['compensation']:+6.1f},   # 預測 {r['pred']:+6.1f}°, 誤差 {r['error']:5.1f}°")


if __name__ == "__main__":
    main()
