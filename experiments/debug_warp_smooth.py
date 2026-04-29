"""
調試 warp_smooth loss 為什麼不變
"""
import torch as th
import numpy as np
from src.models import BinauralNetwork
from src.dataset import BinauralDataset
from src.losses import WarpSmoothnessLoss

# 載入模型
print("載入模型...")
net = BinauralNetwork(
    view_dim=7,
    warpnet_layers=4,
    warpnet_channels=64,
    wavenet_blocks=3,
    layers_per_block=10,
    wavenet_channels=64,
    use_cuda=True
)

# 載入 checkpoint
try:
    net.load_checkpoint('./outputs_with_warp_loss', suffix='epoch-10', optimizer=None)
    print("✓ 載入 epoch-10 checkpoint")
except:
    print("⚠ 無法載入 checkpoint，使用隨機初始化")

net.eval()

# 載入數據
print("\n載入數據...")
dataset = BinauralDataset(
    dataset_directory='./dataset/trainset',
    chunk_size_ms=200,
    overlap=0.5
)

# 取一個 batch
mono, binaural, view = dataset[0]
mono = th.from_numpy(mono).unsqueeze(0).cuda() if isinstance(mono, np.ndarray) else mono.unsqueeze(0).cuda()
view = th.from_numpy(view).unsqueeze(0).cuda() if isinstance(view, np.ndarray) else view.unsqueeze(0).cuda()

print(f"mono shape: {mono.shape}")
print(f"view shape: {view.shape}")

# Forward pass
print("\nForward pass...")
with th.no_grad():
    prediction = net(mono, view, return_warpfields=True)

warpfields = prediction['warpfields']

print(f"\n=== Warpfield 統計 ===")
print(f"Geometric warpfield:")
print(f"  - shape: {warpfields['geometric'].shape}")
print(f"  - mean: {warpfields['geometric'].mean().item():.6f}")
print(f"  - std: {warpfields['geometric'].std().item():.6f}")
print(f"  - min: {warpfields['geometric'].min().item():.6f}")
print(f"  - max: {warpfields['geometric'].max().item():.6f}")

print(f"\nNeural warpfield:")
print(f"  - shape: {warpfields['neural'].shape}")
print(f"  - mean: {warpfields['neural'].mean().item():.6f}")
print(f"  - std: {warpfields['neural'].std().item():.6f}")
print(f"  - min: {warpfields['neural'].min().item():.6f}")
print(f"  - max: {warpfields['neural'].max().item():.6f}")

print(f"\nTotal warpfield:")
print(f"  - shape: {warpfields['total'].shape}")
print(f"  - mean: {warpfields['total'].mean().item():.6f}")
print(f"  - std: {warpfields['total'].std().item():.6f}")
print(f"  - min: {warpfields['total'].min().item():.6f}")
print(f"  - max: {warpfields['total'].max().item():.6f}")

# 計算 smoothness
print(f"\n=== Smoothness 分析 ===")

def compute_smoothness(warpfield):
    """計算 warpfield 的平滑度"""
    temporal_diff = warpfield[:, :, 1:] - warpfield[:, :, :-1]
    return {
        'mean_abs_diff': th.mean(th.abs(temporal_diff)).item(),
        'mean_squared_diff': th.mean(temporal_diff ** 2).item(),
        'max_abs_diff': th.max(th.abs(temporal_diff)).item(),
        'std_diff': th.std(temporal_diff).item(),
    }

geom_smooth = compute_smoothness(warpfields['geometric'])
neural_smooth = compute_smoothness(warpfields['neural'])
total_smooth = compute_smoothness(warpfields['total'])

print(f"\nGeometric warpfield smoothness:")
for k, v in geom_smooth.items():
    print(f"  - {k}: {v:.10f}")

print(f"\nNeural warpfield smoothness:")
for k, v in neural_smooth.items():
    print(f"  - {k}: {v:.10f}")

print(f"\nTotal warpfield smoothness:")
for k, v in total_smooth.items():
    print(f"  - {k}: {v:.10f}")

# 使用 WarpSmoothnessLoss 計算
print(f"\n=== WarpSmoothnessLoss 計算 ===")

warp_smooth_loss = WarpSmoothnessLoss(lambda_smooth=0.001)
loss_value = warp_smooth_loss(warpfields['total'])

print(f"WarpSmoothnessLoss (lambda=0.001): {loss_value.item():.10f}")

# 不帶權重的計算
temporal_diff = warpfields['total'][:, :, 1:] - warpfields['total'][:, :, :-1]
raw_smoothness = th.mean(temporal_diff ** 2).item()
print(f"Raw smoothness (不帶權重): {raw_smoothness:.10f}")
print(f"Raw smoothness * 0.001: {raw_smoothness * 0.001:.10f}")

# 檢查 warpfield 是否真的在變化
print(f"\n=== 檢查 Warpfield 時間變化 ===")
print(f"前 10 個時間步的 total warpfield (左耳):")
print(warpfields['total'][0, 0, :10].cpu().numpy())

print(f"\n相鄰時間步的差異 (前 10 個):")
diff = warpfields['total'][0, 0, 1:11] - warpfields['total'][0, 0, :10]
print(diff.cpu().numpy())

# 檢查是否是常數
is_constant = th.allclose(
    warpfields['total'][:, :, 1:], 
    warpfields['total'][:, :, :-1],
    atol=1e-8
)
print(f"\nWarpfield 是否為常數（時間不變）: {is_constant}")

if is_constant:
    print("⚠️ 警告：Warpfield 在時間上幾乎不變！")
    print("這可能是因為：")
    print("  1. View 數據（位置/方向）在時間上變化很小")
    print("  2. Geometric warp 計算有問題")
    print("  3. Neural warp 學習到了常數輸出")
