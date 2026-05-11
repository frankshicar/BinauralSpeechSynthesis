"""
重新生成所有 testset subject 的 tx_positions.txt（使用未校正的原始角度）
並建立新的 subject13 (-75°)
"""
import math
import os
import numpy as np
import soundfile as sf
import shutil

# 定義所有 subject 的角度（未校正的原始角度）
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
    "subject13": -75.0,  # 新增
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
            print(f"  原始: {num_samples} samples @ {sr}Hz -> Resample: {num_samples_48k} samples @ 48000Hz")
        else:
            num_samples_48k = len(audio) if audio.ndim == 1 else audio.shape[0]
        
        num_frames = num_samples_48k // 400  # 每 400 個樣本一幀
        return num_frames
    except Exception as e:
        print(f"  錯誤：無法讀取 {mono_path}: {e}")
        return None


def create_subject13():
    """
    建立新的 subject13 資料夾（-75°）
    複製 subject1 的音訊檔案作為模板
    """
    subject13_dir = os.path.join(TESTSET_DIR, "subject13")
    
    if os.path.exists(subject13_dir):
        print(f"subject13 已存在，跳過建立")
        return True
    
    # 複製 subject1 作為模板
    subject1_dir = os.path.join(TESTSET_DIR, "subject1")
    if not os.path.exists(subject1_dir):
        print(f"錯誤：找不到 {subject1_dir}，無法建立 subject13")
        return False
    
    print(f"建立 subject13 (-75°)...")
    os.makedirs(subject13_dir, exist_ok=True)
    
    # 複製 mono.wav 和 binaural.wav（如果存在）
    for filename in ["mono.wav", "binaural.wav"]:
        src = os.path.join(subject1_dir, filename)
        dst = os.path.join(subject13_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  已複製 {filename}")
    
    # 複製 rx_positions.txt（如果存在）
    rx_src = os.path.join(subject1_dir, "rx_positions.txt")
    rx_dst = os.path.join(subject13_dir, "rx_positions.txt")
    if os.path.exists(rx_src):
        shutil.copy2(rx_src, rx_dst)
        print(f"  已複製 rx_positions.txt")
    
    print(f"  subject13 建立完成")
    return True


def main():
    print("=" * 60)
    print("重新生成所有 testset subject 的 tx_positions.txt")
    print("使用未校正的原始角度")
    print("=" * 60)
    print()
    
    # 先建立 subject13
    if not create_subject13():
        print("無法建立 subject13，繼續處理其他 subject...")
    print()
    
    # 處理所有 subject
    success_count = 0
    fail_count = 0
    
    for subject_name in sorted(SUBJECTS.keys(), key=lambda x: int(x.replace("subject", ""))):
        angle = SUBJECTS[subject_name]
        subject_dir = os.path.join(TESTSET_DIR, subject_name)
        
        print(f"處理 {subject_name} ({angle:+.1f}°)...")
        
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
        
        # 生成 tx_positions
        tx_positions = generate_tx_positions(angle, DISTANCE, num_frames)
        
        # 儲存
        tx_path = os.path.join(subject_dir, "tx_positions.txt")
        np.savetxt(tx_path, tx_positions, fmt="%.7f")  # 不轉置，直接儲存
        
        x, y = angle_to_position(angle, DISTANCE)
        print(f"  ✓ 已生成 tx_positions.txt")
        print(f"    角度: {angle:+.1f}°, 座標: ({x:.2f}, {y:.2f}), 幀數: {num_frames}")
        success_count += 1
    
    print()
    print("=" * 60)
    print(f"完成！成功: {success_count}, 失敗: {fail_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
