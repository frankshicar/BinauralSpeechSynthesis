"""
改進版 Trainer v4
- 使用 ReduceLROnPlateau 替代 NewbobAdam
- 分階段訓練策略
- 動態 loss 權重
"""

import tqdm
import time
import torch as th
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import json
import csv
from datetime import datetime

from src.losses import L2Loss, PhaseLoss, IPDLoss, WarpLoss, WarpSmoothnessLoss


class TrainerV4:
    def __init__(self, config, net, dataset, resume_from=None):
        self.config = config
        self.dataset = dataset
        self.dataloader = DataLoader(
            dataset, 
            batch_size=config["batch_size"], 
            shuffle=True, 
            num_workers=16,  # 提高到 16，改善 I/O 
            pin_memory=True
        )
        
        # 多 GPU
        gpus = [i for i in range(config["num_gpus"])]
        self.net = th.nn.DataParallel(net, gpus)
        
        # 使用標準 Adam + ReduceLROnPlateau
        weights = filter(lambda x: x.requires_grad, net.parameters())
        self.optimizer = Adam(weights, lr=config["learning_rate"])
        
        # 學習率調度器
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config["lr_scheduler"]["factor"],
            patience=config["lr_scheduler"]["patience"],
            min_lr=config["lr_scheduler"]["min_lr"]
        )
        
        # Loss functions
        self.l2_loss = L2Loss(mask_beginning=config["mask_beginning"])
        self.phase_loss = PhaseLoss(sample_rate=48000, mask_beginning=config["mask_beginning"])
        self.ipd_loss = IPDLoss(sample_rate=48000, mask_beginning=config["mask_beginning"])
        self.warp_loss = WarpLoss(lambda_warp=config.get("lambda_warp", 0.01))
        self.warp_smoothness_loss = WarpSmoothnessLoss(lambda_smooth=config.get("lambda_smooth", 0.001))
        
        self.total_iters = 0
        self.training_history = []
        self.log_dir = os.path.join(config["artifacts_dir"], "training_logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.start_epoch = 0
        
        if resume_from:
            self._resume_from_checkpoint(resume_from)
        else:
            self._save_config()
        
        self.net.train()
    
    def _get_loss_weights(self, epoch):
        """根據訓練階段返回動態 loss 權重"""
        stages = self.config["training_stages"]
        
        for stage_name, stage_config in stages.items():
            start, end = stage_config["epochs"]
            if start <= epoch < end:
                return stage_config["loss_weights"]
        
        # 默認使用最後階段的權重
        return stages["stage3"]["loss_weights"]
    
    def _save_config(self):
        """保存訓練配置"""
        config_path = os.path.join(self.log_dir, "training_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({
                "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "batch_size": self.config["batch_size"],
                "learning_rate": self.config["learning_rate"],
                "epochs": self.config["epochs"],
                "lr_scheduler": self.config["lr_scheduler"],
                "training_stages": self.config["training_stages"],
                "mask_beginning": self.config["mask_beginning"],
                "lambda_warp": self.config.get("lambda_warp", 0.01),
                "lambda_smooth": self.config.get("lambda_smooth", 0.001),
                "num_gpus": self.config["num_gpus"],
                "dataset_size": len(self.dataset.chunks)
            }, f, indent=2, ensure_ascii=False)
    
    def _resume_from_checkpoint(self, suffix):
        """從 checkpoint 恢復訓練"""
        print(f"\n{'='*60}")
        print(f"從 checkpoint 恢復訓練: {suffix}")
        print(f"{'='*60}")
        
        # 載入模型權重
        self.net.module.load(self.config["artifacts_dir"], suffix=suffix)
        print(f"✅ 已載入模型權重")
        
        # 載入 checkpoint
        if suffix == "":
            checkpoint_file = f"{self.config['artifacts_dir']}/checkpoint.pth"
        else:
            checkpoint_file = f"{self.config['artifacts_dir']}/checkpoint.{suffix}.pth"
        
        if os.path.exists(checkpoint_file):
            checkpoint_data = th.load(checkpoint_file)
            
            if 'optimizer_state_dict' in checkpoint_data:
                self.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                print(f"✅ 已載入 optimizer 狀態")
            
            if 'scheduler_state_dict' in checkpoint_data:
                self.scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
                print(f"✅ 已載入 scheduler 狀態")
            
            if 'epoch' in checkpoint_data:
                self.start_epoch = checkpoint_data['epoch']
                print(f"✅ 將從 epoch {self.start_epoch + 1} 繼續訓練")
            
            if 'training_history' in checkpoint_data:
                self.training_history = checkpoint_data['training_history']
                print(f"✅ 已載入訓練歷史: {len(self.training_history)} 個 epoch")
            
            if 'total_iters' in checkpoint_data:
                self.total_iters = checkpoint_data['total_iters']
        
        print(f"{'='*60}\n")
    
    def save_checkpoint(self, epoch, suffix=""):
        """保存完整 checkpoint"""
        # 保存模型權重
        self.net.module.save(self.config["artifacts_dir"], suffix)
        
        # 保存 checkpoint 資訊
        checkpoint_data = {
            'epoch': epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_history': self.training_history,
            'total_iters': self.total_iters,
        }
        
        if suffix == "":
            checkpoint_file = f"{self.config['artifacts_dir']}/checkpoint.pth"
        else:
            checkpoint_file = f"{self.config['artifacts_dir']}/checkpoint.{suffix}.pth"
        
        th.save(checkpoint_data, checkpoint_file)
    
    def _log_epoch(self, epoch, loss_stats, epoch_time, current_lr, loss_weights):
        """記錄 epoch 訓練結果"""
        entry = {
            "epoch": epoch + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "learning_rate": current_lr,
            "epoch_time_seconds": epoch_time,
            "epoch_time_formatted": time.strftime('%H:%M:%S', time.gmtime(epoch_time)),
            "loss_weights": loss_weights,
        }
        
        for k, v in loss_stats.items():
            entry[k] = float(v)
        
        self.training_history.append(entry)
        
        # 保存 JSON
        json_path = os.path.join(self.log_dir, "training_history.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, indent=2, ensure_ascii=False)
        
        # 保存 CSV
        csv_path = os.path.join(self.log_dir, "training_history.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            if self.training_history:
                # 排除 loss_weights 字典，只保留數值欄位
                fieldnames = [k for k in self.training_history[0].keys() if k != 'loss_weights']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for entry in self.training_history:
                    row = {k: v for k, v in entry.items() if k != 'loss_weights'}
                    writer.writerow(row)
    
    def train(self):
        print(f"\n{'='*60}")
        print(f"開始訓練 v4 (改進版)")
        print(f"訓練記錄: {self.log_dir}")
        if self.start_epoch > 0:
            print(f"從 epoch {self.start_epoch + 1} 繼續")
        print(f"{'='*60}\n")
        
        for epoch in range(self.start_epoch, self.config["epochs"]):
            # 獲取當前階段的 loss 權重
            loss_weights = self._get_loss_weights(epoch)
            
            # 顯示訓練階段
            if epoch == 0 or self._get_loss_weights(epoch-1) != loss_weights:
                print(f"\n{'='*60}")
                print(f"訓練階段切換 (Epoch {epoch+1})")
                print(f"Loss 權重: {loss_weights}")
                print(f"{'='*60}\n")
            
            t_start = time.time()
            loss_stats = {}
            data_pbar = tqdm.tqdm(self.dataloader)
            
            for data in data_pbar:
                loss_new = self.train_iteration(data, loss_weights)
                for k, v in loss_new.items():
                    loss_stats[k] = loss_stats[k] + v if k in loss_stats else v
                data_pbar.set_description(f"loss: {loss_new['accumulated_loss'].item():.7f}")
            
            for k in loss_stats:
                loss_stats[k] /= len(self.dataloader)
            
            # 更新學習率
            self.scheduler.step(loss_stats["accumulated_loss"])
            current_lr = self.optimizer.param_groups[0]['lr']
            
            t_end = time.time()
            epoch_time = t_end - t_start
            
            # 記錄
            self._log_epoch(epoch, loss_stats, epoch_time, current_lr, loss_weights)
            
            # 顯示
            loss_str = "    ".join([f"{k}:{v:.4}" for k, v in loss_stats.items()])
            time_str = f"({time.strftime('%H:%M:%S', time.gmtime(epoch_time))})"
            lr_str = f"lr:{current_lr:.6f}"
            print(f"epoch {epoch+1} " + loss_str + "    " + lr_str + "    " + time_str)
            
            # 保存 checkpoint
            if self.config["save_frequency"] > 0 and (epoch + 1) % self.config["save_frequency"] == 0:
                self.save_checkpoint(epoch, suffix='epoch-' + str(epoch+1))
                print("Saved checkpoint")
        
        # 保存最終模型
        self.save_checkpoint(self.config["epochs"] - 1, suffix='final')
        print("\n訓練完成!")
    
    def train_iteration(self, data, loss_weights):
        """單次訓練迭代"""
        self.optimizer.zero_grad()
        
        mono, binaural, quats = data
        mono, binaural, quats = mono.cuda(), binaural.cuda(), quats.cuda()
        
        # Forward pass
        prediction = self.net.forward(mono, quats)
        
        # Audio losses
        l2 = self.l2_loss(prediction["output"], binaural)
        phase = self.phase_loss(prediction["output"], binaural)
        ipd = self.ipd_loss(prediction["output"], binaural)
        
        # Warp losses (設為 0，因為 BinauralNetwork 不返回 warpfields)
        warp_loss = 0.0
        warp_smooth_loss = 0.0
        
        # Intermediate losses
        intermediate_binaural = th.cat([binaural] * len(prediction["intermediate"]), dim=1)
        intermediate_prediction = th.cat(prediction["intermediate"], dim=1)
        intermediate_l2 = self.l2_loss(intermediate_prediction, intermediate_binaural)
        intermediate_phase = self.phase_loss(intermediate_prediction, intermediate_binaural)
        
        intermediate_ipd = 0.0
        for intermediate_output in prediction["intermediate"]:
            intermediate_ipd += self.ipd_loss(intermediate_output, binaural)
        intermediate_ipd /= len(prediction["intermediate"])
        
        # Total loss with dynamic weights
        loss = (l2 + intermediate_l2) * loss_weights["l2"] + \
               (phase + intermediate_phase) * loss_weights["phase"] + \
               (ipd + intermediate_ipd) * loss_weights["ipd"] + \
               warp_loss + warp_smooth_loss
        
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
