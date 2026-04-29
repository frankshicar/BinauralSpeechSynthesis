"""
v7 訓練腳本
主要改進：
1. CosineAnnealingWarmRestarts 取代 ReduceLROnPlateau
   - 每個 Stage 做一次 cosine decay (0.001 → 5e-6)
   - Stage 切換時自動 warm restart，不受 loss 跳升干擾
2. Stage 1 縮短至 60 epochs（v6 分析：epoch 30 後 L2 改善 < 1%/10epoch）
3. Phase/IPD 權重大幅提高（v6 Stage 3 改善僅 1.5-1.9%，需更強梯度訊號）
"""

import os
import argparse
from src.dataset import BinauralDataset
from src.models import BinauralNetwork
from src.trainer_v7 import TrainerV7

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_directory", type=str, default="./dataset_original/trainset")
parser.add_argument("--artifacts_directory", type=str, default="./outputs_v7")
parser.add_argument("--num_gpus", type=int, default=1)
parser.add_argument("--blocks", type=int, default=3)
parser.add_argument("--resume", type=str, default=None)
args = parser.parse_args()

config = {
    "artifacts_dir": args.artifacts_directory,
    "learning_rate": 0.001,
    "batch_size": 32,
    "mask_beginning": 1024,
    "save_frequency": 10,
    "epochs": 200,
    "num_gpus": args.num_gpus,

    "lr_scheduler": {
        "type": "CosineAnnealingWarmRestarts",
        "T_0": 70,
        "T_mult": 1,
        "eta_min": 5e-6
    },

    "training_stages": {
        "stage1": {
            "epochs": [0, 70],
            "loss_weights": {"l2": 10.0, "phase": 0.0, "ipd": 0.0}
        },
        "stage2": {
            "epochs": [70, 155],     # 延長至 155，讓 LR 在 cosine 中段時才切換
            "loss_weights": {"l2": 10.0, "phase": 0.15, "ipd": 0.3}
        },
        "stage3": {
            "epochs": [155, 200],    # 從 cosine 中段開始（LR ~0.0003），weight 增幅放緩
            "loss_weights": {"l2": 10.0, "phase": 0.20, "ipd": 0.40}
        }
    },

    "lambda_warp": 0.0,
    "lambda_smooth": 0.0,
    # Resume 時強制把 LR 設為此值，避免從 warm restart 高點開始
    # epoch 140 的 cosine 中段約 0.0003，設 0.0002 保守一點
    "resume_lr_override": 0.0002,
}

os.makedirs(config["artifacts_dir"], exist_ok=True)

dataset = BinauralDataset(
    dataset_directory=args.dataset_directory,
    chunk_size_ms=200,
    overlap=0.5
)

net = BinauralNetwork(
    view_dim=7,
    warpnet_layers=4,
    warpnet_channels=64,
    wavenet_blocks=args.blocks,
    layers_per_block=10,
    wavenet_channels=64
)

# 預期 Stage 2 實際貢獻（基於 v6 epoch 80 數值：l2=0.000089, phase=0.969, ipd=0.876）
l2_v, ph_v, ipd_v = 0.000089, 0.969, 0.876
c = [10.0*l2_v, 0.3*ph_v, 0.5*ipd_v]
t = sum(c)

print(f"{'='*60}")
print(f"v7 訓練配置")
print(f"  - WaveNet blocks: {args.blocks}")
print(f"  - 參數量: {net.num_trainable_parameters():,}")
print(f"  - 總 epochs: 180")
print(f"  - LR: CosineAnnealingWarmRestarts (T_0=70, eta_min=5e-6)")
print(f"  - Stage 1 (0-70):   L2=10.0, Phase=0.0, IPD=0.0")
print(f"  - Stage 2 (70-140): L2=10.0, Phase=0.15, IPD=0.3")
print(f"  - Stage 3 (140-180):L2=10.0, Phase=0.25, IPD=0.5")
print(f"  預期 Stage 2 實際貢獻:")
print(f"    L2:{c[0]:.4f}({c[0]/t*100:.1f}%) Phase:{c[1]:.4f}({c[1]/t*100:.1f}%) IPD:{c[2]:.4f}({c[2]/t*100:.1f}%)")
print(f"{'='*60}\n")

trainer = TrainerV7(config, net, dataset, resume_from=args.resume)
trainer.train()
