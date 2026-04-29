"""
重新生成所有 testset subject 的 tx_positions.txt（套用角度補償）
使用 synthesis_utils.py 中的 _ANGLE_COMPENSATION 表
"""
import math
import os
import numpy as np
import soundfile as sf

# 從 synthesis_utils.py 導入補償表
from src.synthesis_utils import _ANGLE_COMPENSATION

# 定義所有 subject 的角度（ground truth 角度）
SUBJECTS = {
    "subject1": -90.0,
    "subject2": -60.0,
    "subject3": -30.0,
    "subject4": 0.0,
    "subject5": 30.0,
    "subject6": 60.0,
    "subject7": 90.0,
    "subject8": -45.0,
    "subject9": 45.0,
    "subject10": -15.0,
    "subject11": 15.0,
    "subject12": 75.0,
    "subject13": -75.0,
}

DISTANCE = 1.0  # 距離（公尺）
TESTSET_DIR = "dataset/testset"


def angle_to_position(azimuth_deg, distance):
    """
    將方位角轉換為 (x, y) 座標
    0° = 正前方 (1, 0)
    +90° = 左方 (0, -1)
    -90° = 右方 (0, +1)
    """
    rad = math.radians(azimuth_deg)
    x = distance * math.cos(rad)
    y = -distance * math.sin(rad)  # 注意：Y 軸定義為正右方，所以要取負
    return x, y


def generate_tx_positions(azimuth_deg, distance, num_frames):
    """
    生成 tx_positions.txt 的內容
    格式：每行 7 個值 [x, y, z, qx, qy, qz, qw]
    """
    x, y = angle_to_position(azimuth_deg, distance)
    z = 0.0
    qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0  # 四元數（無旋轉）
    
    # 生成所有幀（每幀相同）
    row = [x, y, z, qx, qy, qz, qw]
    return np.array([row] * num_frames, dtype=np.float32)


def get_num_frames(subject_dir):
    """
    從 mono.wav 獲取幀數（考慮 resample 到 48kHz）
    """
    mono_path = os.path.join(subject_dir, "mono.wav")
    if not os.path.exists(mono_path):
        return None
    
    try:
        audio, sr = sf.read(mono_path)
        
        # 如果不是 48kHz，計算 resample 後的樣本數
        if sr != 48000:
            num_samples = len(audio) if audio.ndim == 1 else audio.shape[0]
            num_samples_48k = num_samples * 48000 // sr
        else:
            num_samples_48k = len(audio) if audio.ndim == 1 else audio.shape[0]
        
        num_frames = num_samples_48k // 400  # 每 400 個樣本一幀
        return num_frames
    except Exception as e:
        print(f"  錯誤：無法讀取 {mono_path}: {e}")
        return None


def main():
    print("=" * 60)
    print("重新生成所有 testset subject 的 tx_positions.txt")
    print("套用角度補償（來自 synthesis_utils.py）")
    print("=" * 60)
    print()
    
    # 處理所有 subject
    success_count = 0
    fail_count = 0
    
    for subject_name in sorted(SUBJECTS.keys(), key=lambda x: int(x.replace("subject", ""))):
        gt_angle = SUBJECTS[subject_name]
        subject_dir = os.path.join(TESTSET_DIR, subject_name)
        
        # 查表取得補償量
        compensation = _ANGLE_COMPENSATION.get(gt_angle, 0.0)
        compensated_angle = gt_angle + compensation
        
        print(f"處理 {subject_name}...")
        print(f"  GT 角度: {gt_angle:+.1f}°, 補償: {compensation:+.1f}°, TX 角度: {compensated_angle:+.1f}°")
        
        if not os.path.exists(subject_dir):
            print(f"  警告：資料夾不存在，跳過")
            fail_count += 1
            continue
        
        # 獲取幀數
        num_frames = get_num_frames(subject_dir)
        if num_frames is None:
            print(f"  錯誤：無法獲取幀數，跳過")
            fail_count += 1
            continue
        
        # 生成 tx_positions（使用補償後的角度）
        tx_positions = generate_tx_positions(compensated_angle, DISTANCE, num_frames)
        
        # 儲存
        tx_path = os.path.join(subject_dir, "tx_positions.txt")
        np.savetxt(tx_path, tx_positions, fmt="%.7f")
        
        x, y = angle_to_position(compensated_angle, DISTANCE)
        print(f"  ✓ 已生成 tx_positions.txt")
        print(f"    座標: ({x:.2f}, {y:.2f}), 幀數: {num_frames}")
        success_count += 1
    
    print()
    print("=" * 60)
    print(f"完成！成功: {success_count}, 失敗: {fail_count}")
    print("=" * 60)
    print()
    print("現在可以執行 evaluate.py 來驗證補償效果：")
    print("python evaluate.py --dataset_directory ./dataset/testset \\")
    print("  --model_file outputs/binaural_network.newbob.net \\")
    print("  --artifacts_directory results_audio --blocks 3")


if __name__ == "__main__":
    main()
