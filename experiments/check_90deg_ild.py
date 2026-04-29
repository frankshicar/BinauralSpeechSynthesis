#!/usr/bin/env python3
"""
檢查 ±90° 樣本的 ILD（Interaural Level Difference）
看看 ILD 是否正確，這可能解釋為什麼 ITD 過大但聽起來還是在側邊
"""

import numpy as np
import wave
import torch as th
from src.synthesis_utils import load_binaural_net, angle_to_tx_positions, chunked_forwarding

def calculate_ild(binaural_audio):
    """
    計算 ILD（dB）
    正值 = 左耳較大聲（聲源在左）
    負值 = 右耳較大聲（聲源在右）
    """
    if binaural_audio.shape[0] != 2:
        binaural_audio = binaural_audio.T
    
    left = binaural_audio[0]
    right = binaural_audio[1]
    
    # RMS 能量
    left_rms = np.sqrt(np.mean(left**2))
    right_rms = np.sqrt(np.mean(right**2))
    
    # ILD (dB)
    ild_db = 20 * np.log10(left_rms / (right_rms + 1e-10))
    
    return ild_db, left_rms, right_rms

def main():
    print("檢查 ±90° 樣本的 ILD")
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
    test_angles = [-90, -75, -60, -30, 0, 30, 60, 75, 90]
    
    print("\n理論參考：")
    print("  - 0° (正前方): ILD ≈ 0 dB")
    print("  - ±30°: ILD ≈ ±3-5 dB")
    print("  - ±60°: ILD ≈ ±8-12 dB")
    print("  - ±90° (側邊): ILD ≈ ±15-20 dB (高頻可達 ±30 dB)")
    print("  - 正值 = 左耳較大聲（聲源在左）")
    print("  - 負值 = 右耳較大聲（聲源在右）")
    print()
    
    for angle in test_angles:
        print(f"\n角度: {angle:+4d}°")
        print("-" * 80)
        
        # 生成 tx_positions
        view = th.from_numpy(angle_to_tx_positions(angle, distance=1.0, num_frames=num_frames))
        
        # 合成
        binaural = chunked_forwarding(net, mono, view)
        bio = binaural.numpy()
        
        # 計算 ILD
        ild_db, left_rms, right_rms = calculate_ild(bio)
        
        print(f"  左耳 RMS:  {left_rms:.6f}")
        print(f"  右耳 RMS:  {right_rms:.6f}")
        print(f"  ILD:       {ild_db:+7.2f} dB")
        
        # 判斷是否合理
        if angle == 0:
            if abs(ild_db) < 2.0:
                print(f"  ✓ 正前方的 ILD 接近 0 dB")
            else:
                print(f"  ⚠️  正前方的 ILD 應該接近 0 dB")
        elif angle > 0:  # 左側
            if ild_db > 0:
                print(f"  ✓ 左側聲源，左耳較大聲")
            else:
                print(f"  ✗ 左側聲源，但右耳較大聲（錯誤）")
        else:  # 右側
            if ild_db < 0:
                print(f"  ✓ 右側聲源，右耳較大聲")
            else:
                print(f"  ✗ 右側聲源，但左耳較大聲（錯誤）")
    
    print("\n" + "=" * 80)
    print("結論:")
    print("  如果 ±90° 的 ILD 是正確的（±15-20 dB），")
    print("  即使 ITD 過大，人耳仍會感知為在側邊，")
    print("  因為在極端角度，ILD 比 ITD 更重要。")

if __name__ == "__main__":
    main()
