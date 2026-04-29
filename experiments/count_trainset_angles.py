#!/usr/bin/env python3
"""統計 trainset 中各角度的樣本數"""

import os
import numpy as np
import math
from collections import Counter

trainset_path = "/home/sbplab/frank/BinauralSpeechSynthesis/dataset_original/trainset"

angle_counter = Counter()
total_samples = 0

def position_to_angle(x, y):
    """從 x, y 座標計算方位角（度）"""
    # atan2(y, x) 返回弧度，轉換為度
    # 注意：根據 synthesis_utils.py 的 angle_to_tx_positions 函數
    # x = distance * cos(rad), y = -distance * sin(rad)
    # 所以反推: angle = -atan2(y, x)
    angle_rad = math.atan2(-y, x)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

# 遍歷所有 subject
for subject in sorted(os.listdir(trainset_path)):
    subject_path = os.path.join(trainset_path, subject)
    if not os.path.isdir(subject_path):
        continue
    
    tx_file = os.path.join(subject_path, "tx_positions.txt")
    if not os.path.exists(tx_file):
        print(f"警告: {subject} 沒有 tx_positions.txt")
        continue
    
    # 讀取位置資料
    try:
        tx_data = np.loadtxt(tx_file)
        # tx_data 的格式應該是每行 7 個數字 (x, y, z, qw, qx, qy, qz)
        # 我們只需要 x, y 來計算方位角
        for row in tx_data:
            x, y = row[0], row[1]
            angle = position_to_angle(x, y)
            # 四捨五入到整數度
            angle_rounded = round(angle)
            angle_counter[angle_rounded] += 1
            total_samples += 1
    except Exception as e:
        print(f"讀取 {subject} 時出錯: {e}")

# 輸出統計結果
print("=" * 60)
print(f"Trainset 角度樣本統計")
print("=" * 60)
print(f"總樣本數: {total_samples}\n")

print("各角度樣本數:")
print("-" * 60)
for angle in sorted(angle_counter.keys()):
    count = angle_counter[angle]
    percentage = (count / total_samples * 100) if total_samples > 0 else 0
    print(f"角度 {angle:6.1f}°: {count:4d} 個樣本 ({percentage:5.2f}%)")

print("=" * 60)
