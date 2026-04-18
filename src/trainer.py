"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import tqdm
import time
import torch as th
from torch.utils.data import DataLoader
import os
import json
import csv
from datetime import datetime

from src.utils import NewbobAdam
from src.losses import L2Loss, PhaseLoss, IPDLoss


class Trainer:
    def __init__(self, config, net, dataset, resume_from=None):
        '''
        :param config: a dict containing parameters
        :param net: the network to be trained, must be of type src.utils.Net
        :param dataset: the dataset to be trained on
        :param resume_from: (str) checkpoint suffix to resume from (e.g., 'epoch-40')
        '''
        self.config = config
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=8, pin_memory=True)
        gpus = [i for i in range(config["num_gpus"])]
        self.net = th.nn.DataParallel(net, gpus)
        weights = filter(lambda x: x.requires_grad, net.parameters())
        self.optimizer = NewbobAdam(weights,
                                    net,
                                    artifacts_dir=config["artifacts_dir"],
                                    initial_learning_rate=config["learning_rate"],
                                    decay=config["newbob_decay"],
                                    max_decay=config["newbob_max_decay"])
        self.l2_loss = L2Loss(mask_beginning=config["mask_beginning"])
        self.phase_loss = PhaseLoss(sample_rate=48000, mask_beginning=config["mask_beginning"])
        self.ipd_loss = IPDLoss(sample_rate=48000, mask_beginning=config["mask_beginning"])
        
        # Warp losses
        from src.losses import WarpLoss, WarpSmoothnessLoss
        self.warp_loss = WarpLoss(lambda_warp=config.get("lambda_warp", 0.01))
        self.warp_smoothness_loss = WarpSmoothnessLoss(lambda_smooth=config.get("lambda_smooth", 0.001))
        
        self.total_iters = 0
        
        # 訓練記錄 (Training logs)
        self.training_history = []
        self.log_dir = os.path.join(config["artifacts_dir"], "training_logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 起始 epoch
        self.start_epoch = 0
        
        # 如果是從 checkpoint 恢復
        if resume_from:
            self._resume_from_checkpoint(resume_from)
        else:
            # 記錄訓練配置 (Log training configuration)
            self._save_config()
        
        # switch to training mode
        self.net.train()
    
    def _resume_from_checkpoint(self, suffix):
        """
        從 checkpoint 恢復訓練
        :param suffix: checkpoint 後綴 (e.g., 'epoch-40')
        """
        import torch as th
        
        print(f"\n{'='*60}")
        print(f"從 checkpoint 恢復訓練: {suffix}")
        print(f"{'='*60}")
        
        # 載入模型權重
        self.net.module.load(self.config["artifacts_dir"], suffix=suffix)
        print(f"✅ 已載入模型權重")
        
        # 載入 checkpoint 資訊
        if suffix == "":
            checkpoint_file = f"{self.config['artifacts_dir']}/checkpoint.pth"
        else:
            checkpoint_file = f"{self.config['artifacts_dir']}/checkpoint.{suffix}.pth"
        
        if os.path.exists(checkpoint_file):
            checkpoint_data = th.load(checkpoint_file)
            
            # 恢復 optimizer 狀態
            if 'optimizer_state_dict' in checkpoint_data:
                self.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                print(f"✅ 已載入 optimizer 狀態")
            
            # 恢復 epoch 計數
            if 'epoch' in checkpoint_data:
                self.start_epoch = checkpoint_data['epoch']
                print(f"✅ 將從 epoch {self.start_epoch + 1} 繼續訓練")
            
            # 恢復訓練歷史
            if 'training_history' in checkpoint_data:
                self.training_history = checkpoint_data['training_history']
                print(f"✅ 已載入訓練歷史: {len(self.training_history)} 個 epoch")
            
            # 恢復 total_iters
            if 'total_iters' in checkpoint_data:
                self.total_iters = checkpoint_data['total_iters']
                print(f"✅ 已載入 total_iters: {self.total_iters}")
            
            # 顯示最後的訓練狀態
            if self.training_history:
                last_entry = self.training_history[-1]
                print(f"\n上次訓練狀態:")
                print(f"  - Epoch: {last_entry['epoch']}")
                print(f"  - Learning Rate: {last_entry['learning_rate']}")
                print(f"  - Accumulated Loss: {last_entry['accumulated_loss']:.6f}")
        else:
            print(f"⚠️  找不到 checkpoint 檔案: {checkpoint_file}")
            print(f"⚠️  僅載入模型權重，optimizer 狀態將重新初始化")
        
        print(f"{'='*60}\n")

    def save(self, suffix=""):
        """保存模型權重（舊格式，向後兼容）"""
        self.net.module.save(self.config["artifacts_dir"], suffix)
    
    def save_checkpoint(self, epoch, suffix=""):
        """保存完整的 checkpoint，包括 optimizer 狀態"""
        import torch as th
        import os
        
        # 保存模型權重
        self.net.module.save(self.config["artifacts_dir"], suffix)
        
        # 保存 checkpoint 資訊（optimizer 和 training state）
        checkpoint_data = {
            'epoch': epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'total_iters': self.total_iters,
        }
        
        if suffix == "":
            checkpoint_file = f"{self.config['artifacts_dir']}/checkpoint.pth"
        else:
            checkpoint_file = f"{self.config['artifacts_dir']}/checkpoint.{suffix}.pth"
        
        th.save(checkpoint_data, checkpoint_file)
        print(f"Saved checkpoint: {checkpoint_file}")
    
    def _save_config(self):
        """保存訓練配置到 JSON 檔案"""
        config_path = os.path.join(self.log_dir, "training_config.json")
        config_to_save = {
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "batch_size": self.config["batch_size"],
            "learning_rate": self.config["learning_rate"],
            "epochs": self.config["epochs"],
            "loss_weights": self.config["loss_weights"],
            "mask_beginning": self.config["mask_beginning"],
            "newbob_decay": self.config["newbob_decay"],
            "newbob_max_decay": self.config["newbob_max_decay"],
            "save_frequency": self.config["save_frequency"],
            "num_gpus": self.config["num_gpus"],
            "dataset_size": len(self.dataset.chunks),
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_to_save, f, indent=2, ensure_ascii=False)
        print(f"✅ 訓練配置已保存到: {config_path}")
    
    def _log_epoch(self, epoch, loss_stats, epoch_time, learning_rate):
        """記錄每個 epoch 的訓練資訊"""
        # 準備記錄資料
        log_entry = {
            "epoch": epoch + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "learning_rate": learning_rate,
            "epoch_time_seconds": epoch_time,
            "epoch_time_formatted": time.strftime('%H:%M:%S', time.gmtime(epoch_time)),
        }
        
        # 加入所有 loss 值
        for k, v in loss_stats.items():
            log_entry[k] = float(v)
        
        # 加入到歷史記錄
        self.training_history.append(log_entry)
        
        # 即時保存到 JSON（每個 epoch 都更新）
        json_path = os.path.join(self.log_dir, "training_history.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, indent=2, ensure_ascii=False)
        
        # 即時保存到 CSV（每個 epoch 都更新）
        csv_path = os.path.join(self.log_dir, "training_history.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            if self.training_history:
                writer = csv.DictWriter(f, fieldnames=self.training_history[0].keys())
                writer.writeheader()
                writer.writerows(self.training_history)
    
    def _save_final_summary(self):
        """保存訓練總結"""
        if not self.training_history:
            return
        
        summary = {
            "total_epochs": len(self.training_history),
            "total_time_seconds": sum(entry["epoch_time_seconds"] for entry in self.training_history),
            "start_time": self.training_history[0]["timestamp"],
            "end_time": self.training_history[-1]["timestamp"],
            "final_losses": {
                k: v for k, v in self.training_history[-1].items() 
                if k not in ["epoch", "timestamp", "learning_rate", "epoch_time_seconds", "epoch_time_formatted"]
            },
            "best_epoch": {
                "epoch": min(self.training_history, key=lambda x: x["accumulated_loss"])["epoch"],
                "accumulated_loss": min(entry["accumulated_loss"] for entry in self.training_history),
            },
            "loss_improvement": {
                "initial_loss": self.training_history[0]["accumulated_loss"],
                "final_loss": self.training_history[-1]["accumulated_loss"],
                "improvement": self.training_history[0]["accumulated_loss"] - self.training_history[-1]["accumulated_loss"],
                "improvement_percent": (
                    (self.training_history[0]["accumulated_loss"] - self.training_history[-1]["accumulated_loss"]) 
                    / self.training_history[0]["accumulated_loss"] * 100
                ),
            }
        }
        
        summary_path = os.path.join(self.log_dir, "training_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"✅ 訓練總結已保存到: {summary_path}")
        print(f"{'='*60}")
        print(f"總訓練時間: {time.strftime('%H:%M:%S', time.gmtime(summary['total_time_seconds']))}")
        print(f"最佳 epoch: {summary['best_epoch']['epoch']} (loss: {summary['best_epoch']['accumulated_loss']:.6f})")
        print(f"Loss 改善: {summary['loss_improvement']['improvement']:.6f} ({summary['loss_improvement']['improvement_percent']:.2f}%)")
        print(f"{'='*60}\n")

    def train(self):
        print(f"\n{'='*60}")
        print(f"開始訓練 (Starting training)")
        print(f"訓練記錄將保存到: {self.log_dir}")
        if self.start_epoch > 0:
            print(f"從 epoch {self.start_epoch + 1} 繼續訓練")
        print(f"{'='*60}\n")
        
        for epoch in range(self.start_epoch, self.config["epochs"]):
            t_start = time.time()
            loss_stats = {}
            data_pbar = tqdm.tqdm(self.dataloader)
            for data in data_pbar:
                loss_new = self.train_iteration(data)
                # logging
                for k, v in loss_new.items():
                    loss_stats[k] = loss_stats[k]+v if k in loss_stats else v
                data_pbar.set_description(f"loss: {loss_new['accumulated_loss'].item():.7f}")
            for k in loss_stats:
                loss_stats[k] /= len(self.dataloader)
            
            # 更新學習率並獲取當前學習率
            self.optimizer.update_lr(loss_stats["accumulated_loss"])
            current_lr = self.optimizer.param_groups[0]['lr']
            
            t_end = time.time()
            epoch_time = t_end - t_start
            
            # 記錄到檔案
            self._log_epoch(epoch, loss_stats, epoch_time, current_lr)
            
            # 顯示訓練進度
            loss_str = "    ".join([f"{k}:{v:.4}" for k, v in loss_stats.items()])
            time_str = f"({time.strftime('%H:%M:%S', time.gmtime(epoch_time))})"
            lr_str = f"lr:{current_lr:.6f}"
            print(f"epoch {epoch+1} " + loss_str + "    " + lr_str + "    " + time_str)
            
            # Save checkpoint (包含 optimizer 狀態)
            if self.config["save_frequency"] > 0 and (epoch + 1) % self.config["save_frequency"] == 0:
                self.save_checkpoint(epoch, suffix='epoch-' + str(epoch+1))
                # 同時保存舊格式（向後兼容）
                self.save(suffix='epoch-' + str(epoch+1))
                print("Saved checkpoint and model")
        
        # Save final checkpoint
        self.save_checkpoint(self.config["epochs"] - 1, suffix='final')
        self.save(suffix='final')
        
        # 保存訓練總結
        self._save_final_summary()

    def train_iteration(self, data):
        '''
        one optimization step
        :param data: tuple of tensors containing mono, binaural, and quaternion data
        :return: dict containing values for all different losses
        '''
        # forward
        self.optimizer.zero_grad()

        mono, binaural, quats = data
        mono, binaural, quats = mono.cuda(), binaural.cuda(), quats.cuda()
        
        # Forward pass with warpfields
        prediction = self.net.forward(mono, quats, return_warpfields=True)
        
        # Audio losses
        l2 = self.l2_loss(prediction["output"], binaural)
        phase = self.phase_loss(prediction["output"], binaural)
        ipd = self.ipd_loss(prediction["output"], binaural)
        
        # Warp losses
        warpfields = prediction["warpfields"]
        warp_loss = self.warp_loss(warpfields['neural'], warpfields['geometric'])
        
        # Warp smoothness loss - 只對 neural warp 計算
        # Geometric warp 已經很平滑（由插值保證），主要約束 neural warp
        warp_smooth_loss = self.warp_smoothness_loss(warpfields['neural'])
        
        # 處理中間層輸出
        intermediate_binaural = th.cat([binaural] * len(prediction["intermediate"]), dim=1)
        intermediate_prediction = th.cat(prediction["intermediate"], dim=1)
        intermediate_l2 = self.l2_loss(intermediate_prediction, intermediate_binaural)
        intermediate_phase = self.phase_loss(intermediate_prediction, intermediate_binaural)
        
        # 分別計算每個中間層的 IPD loss
        intermediate_ipd = 0.0
        for intermediate_output in prediction["intermediate"]:
            intermediate_ipd += self.ipd_loss(intermediate_output, binaural)
        intermediate_ipd /= len(prediction["intermediate"])  # 平均

        # Total loss
        loss = (l2 + intermediate_l2) * self.config["loss_weights"]["l2"] + \
               (phase + intermediate_phase) * self.config["loss_weights"]["phase"] + \
               (ipd + intermediate_ipd) * self.config["loss_weights"]["ipd"] + \
               warp_loss + warp_smooth_loss

        # update model parameters
        loss.backward()
        self.optimizer.step()
        self.total_iters += 1

        return {
            "l2": l2,
            "phase": phase,
            "ipd": ipd,
            "warp": warp_loss,
            "warp_smooth": warp_smooth_loss,
            "intermediate_l2": intermediate_l2,
            "intermediate_phase": intermediate_phase,
            "intermediate_ipd": intermediate_ipd,
            "accumulated_loss": loss,
        }
