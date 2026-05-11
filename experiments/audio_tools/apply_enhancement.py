#!/usr/bin/env python3
"""
對已生成的雙耳音訊應用後處理增強
使用方式：
    python apply_enhancement.py --input_dir eval_results/results_audio --output_dir eval_results/enhanced
"""

import argparse
import os
import torch
import soundfile as sf
from tqdm import tqdm
from enhance_audio import enhance_binaural_audio, diagnose_audio_quality


def main():
    parser = argparse.ArgumentParser(description='對雙耳音訊應用後處理增強')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='輸入音訊目錄')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='輸出音訊目錄')
    parser.add_argument('--high_freq_boost', type=float, default=0.15,
                        help='高頻增強強度 (0-0.3, 預設 0.15)')
    parser.add_argument('--compression_strength', type=float, default=1.2,
                        help='動態壓縮強度 (1.0-1.5, 預設 1.2)')
    parser.add_argument('--denoise_threshold', type=float, default=-60,
                        help='去噪閾值 dB (-70 to -50, 預設 -60)')
    parser.add_argument('--diagnose', action='store_true',
                        help='診斷音質問題（需要 ground truth）')
    parser.add_argument('--gt_dir', type=str, default=None,
                        help='Ground truth 音訊目錄（用於診斷）')
    
    args = parser.parse_args()
    
    # 創建輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 增強配置
    config = {
        'high_freq_boost': args.high_freq_boost,
        'compression_strength': args.compression_strength,
        'denoise_threshold': args.denoise_threshold,
        'dc_remove': True,
    }
    
    print("=== 音質增強配置 ===")
    print(f"高頻增強: {config['high_freq_boost']}")
    print(f"動態壓縮: {config['compression_strength']}")
    print(f"去噪閾值: {config['denoise_threshold']} dB")
    print()
    
    # 獲取所有音訊文件
    audio_files = [f for f in os.listdir(args.input_dir) 
                   if f.endswith('.wav')]
    
    if len(audio_files) == 0:
        print(f"錯誤：在 {args.input_dir} 中沒有找到 .wav 文件")
        return
    
    print(f"找到 {len(audio_files)} 個音訊文件")
    print()
    
    # 處理每個文件
    for filename in tqdm(audio_files, desc="增強音訊"):
        input_path = os.path.join(args.input_dir, filename)
        output_path = os.path.join(args.output_dir, filename)
        
        # 讀取音訊
        audio, sr = sf.read(input_path)
        audio = torch.from_numpy(audio.T).float()  # [2, T]
        
        # 應用增強
        enhanced = enhance_binaural_audio(audio, sr=sr, config=config)
        
        # 保存
        enhanced_np = enhanced.cpu().numpy().T  # [T, 2]
        sf.write(output_path, enhanced_np, sr)
        
        # 診斷（如果提供 ground truth）
        if args.diagnose and args.gt_dir:
            gt_path = os.path.join(args.gt_dir, filename)
            if os.path.exists(gt_path):
                gt_audio, _ = sf.read(gt_path)
                gt_audio = torch.from_numpy(gt_audio.T).float()
                
                print(f"\n診斷 {filename}:")
                print("原始音訊:")
                diagnose_audio_quality(audio[0], gt_audio[0], sr)
                print("\n增強後:")
                diagnose_audio_quality(enhanced[0], gt_audio[0], sr)
    
    print(f"\n✓ 完成！增強後的音訊保存在: {args.output_dir}")
    print("\n建議：")
    print("1. 聽一下增強前後的對比")
    print("2. 如果聲音太亮/刺耳，降低 --high_freq_boost (例如 0.10)")
    print("3. 如果聲音太悶，提高 --high_freq_boost (例如 0.20)")
    print("4. 如果有失真，降低 --compression_strength (例如 1.1)")
    print("5. 如果背景噪音明顯，降低 --denoise_threshold (例如 -65)")


if __name__ == "__main__":
    main()
