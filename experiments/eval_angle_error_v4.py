"""
評估 v4 模型的角度誤差
用於判斷是否適合聽損空間感復健應用
"""

import torch as th
import numpy as np
from src.models import BinauralNetwork
from src.dataset import BinauralDataset
from src.doa import gcc_phat_estimate
from torch.utils.data import DataLoader

# 載入模型
net = BinauralNetwork(
    view_dim=7,
    warpnet_layers=4,
    warpnet_channels=64,
    wavenet_blocks=3,
    layers_per_block=10,
    wavenet_channels=64
)

net.load('./outputs_v4', suffix='epoch-70')
net.eval()
net.cuda()

# 載入測試資料
dataset = BinauralDataset(
    dataset_directory='./dataset_original/testset',
    chunk_size_ms=200,
    overlap=0.0
)

dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# 評估角度誤差
angle_errors = []

print("評估角度誤差...")
print("="*60)

with th.no_grad():
    for i, (mono, binaural_gt, view) in enumerate(dataloader):
        if i >= 100:  # 測試 100 個樣本
            break
        
        mono = mono.cuda()
        view = view.cuda()
        
        # 預測
        binaural_pred = net(mono, view)
        
        # 計算 ground truth 角度
        x, y = view[0, 0, 0].item(), view[0, 1, 0].item()
        gt_angle = np.degrees(np.arctan2(-y, x))
        
        # 估計預測角度
        try:
            pred_np = binaural_pred[0].cpu().numpy()
            pred_angle = gcc_phat_estimate(pred_np, sample_rate=48000)
            
            # 計算誤差
            error = abs(pred_angle - gt_angle)
            error = min(error, 360 - error)
            angle_errors.append(error)
            
            if i < 10:  # 顯示前 10 個
                print(f"Sample {i+1}: GT={gt_angle:6.1f}°  Pred={pred_angle:6.1f}°  Error={error:5.1f}°")
        except:
            angle_errors.append(90.0)  # 失敗時記為最大誤差

print("="*60)
print(f"\n角度誤差統計 (n={len(angle_errors)}):")
print(f"  平均誤差: {np.mean(angle_errors):.2f}°")
print(f"  中位數誤差: {np.median(angle_errors):.2f}°")
print(f"  標準差: {np.std(angle_errors):.2f}°")
print(f"  最大誤差: {np.max(angle_errors):.2f}°")
print(f"  < 2° (MAA): {np.sum(np.array(angle_errors) < 2)}/{len(angle_errors)} ({np.sum(np.array(angle_errors) < 2)/len(angle_errors)*100:.1f}%)")
print(f"  < 5°: {np.sum(np.array(angle_errors) < 5)}/{len(angle_errors)} ({np.sum(np.array(angle_errors) < 5)/len(angle_errors)*100:.1f}%)")
print(f"  < 10°: {np.sum(np.array(angle_errors) < 10)}/{len(angle_errors)} ({np.sum(np.array(angle_errors) < 10)/len(angle_errors)*100:.1f}%)")
print()
print("評估結論:")
if np.mean(angle_errors) < 2:
    print("  ✅ 適合聽損空間感復健（平均誤差 < MAA）")
elif np.mean(angle_errors) < 5:
    print("  ⚠️  勉強可用，但建議改進（平均誤差 2-5°）")
else:
    print("  ❌ 不適合聽損空間感復健（平均誤差 > 5°）")
    print("  建議：重新訓練，提高 Phase/IPD 權重")
