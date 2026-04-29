"""
針對靜止角度優化的訓練腳本
基於原始 train.py，但加入角度損失和靜止角度專用的訓練策略
"""

import os
import json
import time
import argparse
import numpy as np
import torch as th
from torch.utils.data import DataLoader

from src.dataset import BinauralDataset
from src.losses import L2Loss, AmplitudeLoss, PhaseLoss, ITDLoss, ILDLoss
from src.models_static import StaticAngleBinauralNetwork
from src.doa import gcc_phat_estimate


def angle_accuracy_loss(pred_binaural, view, sample_rate=48000):
    """
    角度準確度損失：直接優化 GCC-PHAT 估計的角度誤差
    """
    batch_size = pred_binaural.shape[0]
    angle_errors = []
    
    for i in range(batch_size):
        # 提取目標角度
        x, y = view[i, 0, 0].item(), view[i, 1, 0].item()
        target_angle = np.degrees(np.arctan2(-y, x))
        
        # GCC-PHAT 估計
        try:
            binaural_np = pred_binaural[i].detach().cpu().numpy()
            pred_angle = gcc_phat_estimate(binaural_np, sample_rate=sample_rate)
            
            # 計算角度誤差（處理 ±180° 邊界）
            error = abs(pred_angle - target_angle)
            error = min(error, 360 - error)
            angle_errors.append(error)
        except:
            angle_errors.append(90.0)  # 失敗時給最大誤差
    
    return th.tensor(np.mean(angle_errors), device=pred_binaural.device)


def train_static_angle_model():
    # 訓練配置
    config = {
        "dataset_directory": "./dataset_original/trainset",
        "artifacts_directory": "./outputs_static",
        "num_gpus": 1,
        "batch_size": 16,  # 減小 batch size 以適應新的損失計算
        "learning_rate": 0.001,
        "epochs": 50,
        "save_frequency": 5,
        
        # 損失權重
        "loss_weights": {
            "l2": 1.0,
            "phase": 0.01,
            "ipd": 0.1,
            "angle": 0.5,  # 新增：角度損失權重
        },
        
        # 模型參數
        "wavenet_blocks": 3,
        "warpnet_layers": 4,
        "warpnet_channels": 64,
        "wavenet_channels": 64,
    }
    
    # 創建輸出目錄
    os.makedirs(config["artifacts_directory"], exist_ok=True)
    
    # 保存配置
    with open(f"{config['artifacts_directory']}/config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # 載入資料集
    print("Loading dataset...")
    dataset = BinauralDataset(config["dataset_directory"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    
    # 創建模型
    print("Creating model...")
    model = StaticAngleBinauralNetwork(
        view_dim=7,
        wavenet_blocks=config["wavenet_blocks"],
        warpnet_layers=config["warpnet_layers"],
        warpnet_channels=config["warpnet_channels"],
        wavenet_channels=config["wavenet_channels"],
    )
    
    if th.cuda.is_available():
        model.cuda()
    
    # 優化器
    optimizer = th.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 損失函數
    l2_loss = L2Loss()
    phase_loss = PhaseLoss(sample_rate=48000, ignore_below=0.2)
    ipd_loss = ITDLoss(sample_rate=48000, max_shift_ms=1.0)
    
    # 訓練循環
    print("Starting training...")
    training_history = []
    
    for epoch in range(config["epochs"]):
        model.train()
        epoch_losses = {"total": 0, "l2": 0, "phase": 0, "ipd": 0, "angle": 0}
        num_batches = 0
        
        start_time = time.time()
        
        for batch_idx, (mono, binaural, view) in enumerate(dataloader):
            if th.cuda.is_available():
                mono, binaural, view = mono.cuda(), binaural.cuda(), view.cuda()
            
            optimizer.zero_grad()
            
            # 前向傳播
            output = model(mono, view, return_angle_loss=True)
            pred_binaural = output["output"]
            
            # 計算各項損失
            l2 = l2_loss(pred_binaural, binaural)
            phase = phase_loss(pred_binaural, binaural)
            ipd = ipd_loss(pred_binaural, binaural)
            
            # 角度損失（每 5 個 batch 計算一次，避免太慢）
            if batch_idx % 5 == 0:
                angle = angle_accuracy_loss(pred_binaural, view)
            else:
                angle = th.tensor(0.0, device=pred_binaural.device)
            
            # 總損失
            total_loss = (
                config["loss_weights"]["l2"] * l2 +
                config["loss_weights"]["phase"] * phase +
                config["loss_weights"]["ipd"] * ipd +
                config["loss_weights"]["angle"] * angle
            )
            
            # 反向傳播
            total_loss.backward()
            th.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 記錄損失
            epoch_losses["total"] += total_loss.item()
            epoch_losses["l2"] += l2.item()
            epoch_losses["phase"] += phase.item()
            epoch_losses["ipd"] += ipd.item()
            epoch_losses["angle"] += angle.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}: "
                      f"L2={l2.item():.4f}, Phase={phase.item():.4f}, "
                      f"IPD={ipd.item():.4f}, Angle={angle.item():.2f}°")
        
        # 平均損失
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{config['epochs']} completed in {epoch_time:.1f}s")
        print(f"  L2: {epoch_losses['l2']:.4f}")
        print(f"  Phase: {epoch_losses['phase']:.4f}")
        print(f"  IPD: {epoch_losses['ipd']:.4f}")
        print(f"  Angle: {epoch_losses['angle']:.2f}°")
        print(f"  Total: {epoch_losses['total']:.4f}")
        
        # 學習率調整
        scheduler.step(epoch_losses["total"])
        
        # 記錄歷史
        training_history.append({
            "epoch": epoch + 1,
            "losses": epoch_losses,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_time": epoch_time
        })
        
        # 保存模型
        if (epoch + 1) % config["save_frequency"] == 0:
            model_path = f"{config['artifacts_directory']}/static_model_epoch_{epoch+1}.pth"
            th.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
    
    # 保存最終模型
    final_model_path = f"{config['artifacts_directory']}/static_binaural_network.pth"
    th.save(model.state_dict(), final_model_path)
    
    # 保存訓練歷史
    with open(f"{config['artifacts_directory']}/training_history.json", "w") as f:
        json.dump(training_history, f, indent=2)
    
    print(f"Training completed! Final model saved to {final_model_path}")


if __name__ == "__main__":
    train_static_angle_model()