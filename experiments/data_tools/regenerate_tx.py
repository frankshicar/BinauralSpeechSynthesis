import numpy as np
import os

def regenerate_tx():
    root = "./dataset/testset"
    
    # 2026-01-24: 修正座標系統定義（依據 README）
    # Mapping based on user's LATEST correction (2026-01-22)
    # Coordinate System (per README):
    # x = Forward (Front) - X 軸指向前方
    # y = Right - Y 軸指向右方
    # z = Up - Z 軸指向上方
    # 
    # Azimuth Convention - 方位角慣例:
    # 0 deg = Front (+x direction) - 0° 為正前方
    # Positive angle = Counter-clockwise from front = LEFT side (negative y) - 正角度 = 左方（負 y）
    # Negative angle = Clockwise from front = RIGHT side (positive y) - 負角度 = 右方（正 y）
    
    angle_map = {
        # Subject 1: Left Front 30 -> +30 deg -> (x>0, y<0) - 左前方 30 度
        "subject1": 30.0,
        # Subject 2: Left Front 60 -> +60 deg -> (x>0, y<0) - 左前方 60 度
        "subject2": 60.0,
        # Subject 3: Left -> +90 deg -> (x=0, y<0) - 正左方
        "subject3": 90.0,
        # Subject 4: Front -> 0 deg -> (x>0, y=0) - 正前方
        "subject4": 0.0,
        # Subject 5: Right Front 30 -> -30 deg -> (x>0, y>0) - 右前方 30 度
        "subject5": -30.0,
        # Subject 6: Right Front 60 -> -60 deg -> (x>0, y>0) - 右前方 60 度
        "subject6": -60.0
    }
    
    dist = 1.0 # 1 Meter - 1 公尺
    
    for subj, az_deg in angle_map.items():
        subj_dir = os.path.join(root, subj)
        if not os.path.exists(subj_dir):
            continue
            
        print(f"Regenerating {subj} at {az_deg} deg...")
        
        # 2026-01-24: 計算位置（注意 y 需要取負以符合 README 座標系）
        # Calculate Position
        # Note: y is negated because positive azimuth = left = negative y
        az_rad = np.radians(az_deg)
        x = dist * np.cos(az_rad)
        y = -dist * np.sin(az_rad)  # Negative to align with "y points right" - 取負以符合「y 指向右方」
        z = 0.0
        
        # Orientation: Face Forward (Identity)
        qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
        
        # Create Data (N frames)
        existing_path = os.path.join(subj_dir, "tx_positions.txt")
        if os.path.exists(existing_path):
            existing_data = np.loadtxt(existing_path)
            n_frames = existing_data.shape[0] if existing_data.ndim > 1 else 1
        else:
            n_frames = 100 
            
        new_data = np.zeros((n_frames, 7))
        new_data[:, 0] = x
        new_data[:, 1] = y
        new_data[:, 2] = z
        new_data[:, 3] = qx
        new_data[:, 4] = qy
        new_data[:, 5] = qz
        new_data[:, 6] = qw
        
        np.savetxt(existing_path, new_data, fmt='%.7f')
        print(f"  -> Saved {subj} at ({x:.2f}, {y:.2f}, {z:.2f})")

if __name__ == "__main__":
    regenerate_tx()
