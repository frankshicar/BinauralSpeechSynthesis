"""
v6 訓練腳本
主要改進：
1. L2 權重全程保持高位 (10.0)，先打好波形重建基礎
2. Stage 1 延長至 80 epochs，確保 L2 充分收斂
3. Stage 切換時重置 LR scheduler，避免新 loss 項加入後 LR 過早衰減
4. LR patience 加大至 20，配合更長的 Stage 1
"""

import os
import argparse
from src.dataset import BinauralDataset
from src.models import BinauralNetwork
from src.trainer_v4 import TrainerV4

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_directory", type=str, default="./dataset_original/trainset")
parser.add_argument("--artifacts_directory", type=str, default="./outputs_v6")
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
        "type": "ReduceLROnPlateau",
        "patience": 20,   # v5 是 15，加大避免 Stage 1 純 L2 期過早衰減
        "factor": 0.5,
        "min_lr": 5e-6
    },

    # 三階段訓練策略
    # Stage 1: 純 L2，讓模型先學好波形重建
    # Stage 2: 加入 Phase + IPD，學習空間感知
    # Stage 3: 提高 Phase/IPD 權重，精修空間定位
    "training_stages": {
        "stage1": {
            "epochs": [0, 80],        # v5 是 50，延長讓 L2 充分收斂
            "loss_weights": {"l2": 10.0, "phase": 0.0, "ipd": 0.0}
        },
        "stage2": {
            "epochs": [80, 140],
            "loss_weights": {"l2": 10.0, "phase": 0.1, "ipd": 0.2}
        },
        "stage3": {
            "epochs": [140, 200],
            "loss_weights": {"l2": 10.0, "phase": 0.15, "ipd": 0.3}
        }
    },

    # Stage 切換時重置 LR 到此值
    "stage_switch_lr": 0.0003,

    "lambda_warp": 0.01,
    "lambda_smooth": 0.001,
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

print(f"{'='*60}")
print(f"v6 訓練配置")
print(f"  - WaveNet blocks: {args.blocks}")
print(f"  - 參數量: {net.num_trainable_parameters():,}")
print(f"  - 總 epochs: 200")
print(f"  - Stage 1 (0-80):   L2=10.0, Phase=0.0, IPD=0.0")
print(f"  - Stage 2 (80-140): L2=10.0, Phase=0.1, IPD=0.2")
print(f"  - Stage 3 (140-200):L2=10.0, Phase=0.15, IPD=0.3")
print(f"  - LR patience: 20 (v5 是 15)")
print(f"  - Stage 切換時 LR 重置為 0.0003")
print(f"{'='*60}\n")

trainer = TrainerV4(config, net, dataset, resume_from=args.resume)
trainer.train()
