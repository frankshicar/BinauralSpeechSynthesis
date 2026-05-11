"""
v5 快速驗證版本
只修改 loss 權重，專注 Phase/IPD 優化
目標：通用模型的單角度精確定位
"""

import os
import argparse
from src.dataset import BinauralDataset
from src.models import BinauralNetwork
from src.trainer_v4 import TrainerV4

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_directory", type=str, default="./dataset_original/trainset")
parser.add_argument("--artifacts_directory", type=str, default="./outputs_v5_quick")
parser.add_argument("--num_gpus", type=int, default=1)
parser.add_argument("--blocks", type=int, default=3)
parser.add_argument("--resume", type=str, default=None)
args = parser.parse_args()

# 關鍵修改：重新平衡 loss 權重
config = {
    "artifacts_dir": args.artifacts_directory,
    "learning_rate": 0.001,
    "batch_size": 16,
    "mask_beginning": 1024,
    "save_frequency": 10,
    "epochs": 100,
    "num_gpus": args.num_gpus,
    
    "lr_scheduler": {
        "type": "ReduceLROnPlateau",
        "patience": 10,
        "factor": 0.5,
        "min_lr": 1e-5
    },
    
    # 核心改進：平衡 L2/Phase/IPD
    "training_stages": {
        "stage1": {
            "epochs": [0, 40],
            "loss_weights": {
                "l2": 3.0,      # 降低 L2 (v4 是 10.0)
                "phase": 0.1,   # 提高 Phase (v4 是 0.01)
                "ipd": 0.0
            }
        },
        "stage2": {
            "epochs": [40, 80],
            "loss_weights": {
                "l2": 3.0,
                "phase": 0.15,
                "ipd": 0.3      # 高 IPD 權重
            }
        },
        "stage3": {
            "epochs": [80, 100],
            "loss_weights": {
                "l2": 2.0,      # 進一步降低
                "phase": 0.2,   # 最高 Phase
                "ipd": 0.4      # 最高 IPD
            }
        }
    },
    
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
print(f"v5 快速驗證版本 - 通用模型單角度優化")
print(f"  - WaveNet blocks: {args.blocks}")
print(f"  - 參數量: {net.num_trainable_parameters():,}")
print(f"  - 關鍵改進: Phase/IPD 權重大幅提高")
print(f"{'='*60}\n")

trainer = TrainerV4(config, net, dataset, resume_from=args.resume)
trainer.train()
