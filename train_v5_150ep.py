"""
v5 - 150 epochs 版本
延長訓練時間 + 重新平衡 loss 權重
目標：通用模型的單角度精確定位
"""

import os
import argparse
from src.dataset import BinauralDataset
from src.models import BinauralNetwork
from src.trainer_v4 import TrainerV4

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_directory", type=str, default="./dataset_original/trainset")
parser.add_argument("--artifacts_directory", type=str, default="./outputs_v5")
parser.add_argument("--num_gpus", type=int, default=1)
parser.add_argument("--blocks", type=int, default=3)
parser.add_argument("--resume", type=str, default=None)
args = parser.parse_args()

config = {
    "artifacts_dir": args.artifacts_directory,
    "learning_rate": 0.001,
    "batch_size": 16,
    "mask_beginning": 1024,
    "save_frequency": 10,
    "epochs": 150,  # 增加到 150 epochs
    "num_gpus": args.num_gpus,
    
    "lr_scheduler": {
        "type": "ReduceLROnPlateau",
        "patience": 15,  # 增加 patience，避免過早衰減
        "factor": 0.5,
        "min_lr": 5e-6   # 降低最小 LR，允許更長時間優化
    },
    
    # 延長訓練 + 重新平衡 loss 權重
    "training_stages": {
        "stage1": {
            "epochs": [0, 50],  # 延長 Stage 1
            "loss_weights": {
                "l2": 3.0,      # 降低 L2 (v4 是 10.0)
                "phase": 0.1,   # 提高 Phase (v4 是 0.01)
                "ipd": 0.0
            }
        },
        "stage2": {
            "epochs": [50, 110],  # 延長 Stage 2
            "loss_weights": {
                "l2": 3.0,
                "phase": 0.15,
                "ipd": 0.3      # 高 IPD 權重
            }
        },
        "stage3": {
            "epochs": [110, 150],  # 延長 Stage 3
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
print(f"v5 - 150 epochs 長訓練版本")
print(f"  - WaveNet blocks: {args.blocks}")
print(f"  - 參數量: {net.num_trainable_parameters():,}")
print(f"  - 總 epochs: 150 (比 v4 多 50%)")
print(f"  - LR patience: 15 (比 v4 多 50%)")
print(f"  - 關鍵改進: Phase/IPD 權重提高 + 延長訓練")
print(f"{'='*60}\n")

trainer = TrainerV4(config, net, dataset, resume_from=args.resume)
trainer.train()
