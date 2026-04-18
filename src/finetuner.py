"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import tqdm
import time
import torch as th
import numpy as np

from src.utils import NewbobAdam
from src.utils import NewbobAdam
from src.losses import PhaseLoss, MultiResolutionSTFTLoss


class FineTuner:
    def __init__(self, config, net, train_loader, val_loader):
        '''
        :param config: a dict containing parameters
        :param net: the network to be trained, must be of type src.utils.Net
        :param train_loader: DataLoader for training set
        :param val_loader: DataLoader for validation set
        '''
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        gpus = [i for i in range(config["num_gpus"])]
        self.net = th.nn.DataParallel(net, gpus)
        weights = filter(lambda x: x.requires_grad, net.parameters())
        self.optimizer = NewbobAdam(weights,
                                    net,
                                    artifacts_dir=config["artifacts_dir"],
                                    initial_learning_rate=config["learning_rate"],
                                    decay=config["newbob_decay"],
                                    max_decay=config["newbob_max_decay"])
        self.stft_loss = MultiResolutionSTFTLoss()
        self.phase_loss = PhaseLoss(sample_rate=48000, mask_beginning=config["mask_beginning"])
        self.total_iters = 0
        
        # Early stopping parameters
        self.patience = config.get("early_stopping_patience", 5)
        self.best_val_loss = np.inf
        self.patience_counter = 0

    def save(self, suffix=""):
        self.net.module.save(self.config["artifacts_dir"], suffix)

    def compute_loss(self, data):
        mono, binaural, quats = data
        mono, binaural, quats = mono.cuda(), binaural.cuda(), quats.cuda()
        prediction = self.net.forward(mono, quats)
        
        # Main Loss: MultiResolutionSTFTLoss (Spectral Convergence + L1 Magnitude)
        # This replaces simple L2 loss to avoid over-smoothing and Phase Smearing
        stft = self.stft_loss(prediction["output"], binaural)
        
        # Legacy Phase Loss (kept but maybe weight lowered?)
        # User said "Switch to STFT Loss", implied L2 is bad.
        # We will use STFT as primary loss.
        phase = self.phase_loss(prediction["output"], binaural)
        
        # Intermediate Losses
        intermediate_binaural = th.cat([binaural] * len(prediction["intermediate"]), dim=1)
        intermediate_prediction = th.cat(prediction["intermediate"], dim=1)
        
        stft_intermediate = self.stft_loss(intermediate_prediction, intermediate_binaural)
        phase_intermediate = self.phase_loss(intermediate_prediction, intermediate_binaural)

        # STFT loss is typically larger scale, L2 was ~0.03. STFT might be ~1.0-5.0. 
        # Weights: Default L2=1.0. User suggested prioritizing STFT.
        # Let's assume passed weights "l2" key now controls "stft" contribution for backward compat
        loss = (stft + stft_intermediate) * self.config["loss_weights"].get("l2", 1.0) + \
               (phase + phase_intermediate) * self.config["loss_weights"].get("phase", 0.01)

        return {
            "stft": stft,
            "phase": phase,
            "accumulated_loss": loss,
        }

    def train_iteration(self, data):
        '''
        one optimization step
        :param data: tuple of tensors containing mono, binaural, and quaternion data
        :return: dict containing values for all different losses
        '''
        # forward
        self.optimizer.zero_grad()
        loss_dict = self.compute_loss(data)
        loss = loss_dict["accumulated_loss"]
        
        # update model parameters
        loss.backward()
        self.optimizer.step()
        self.total_iters += 1

        return loss_dict

    def validate(self):
        self.net.eval()
        total_loss = 0
        count = 0
        with th.no_grad():
            for data in self.val_loader:
                loss_dict = self.compute_loss(data)
                total_loss += loss_dict["accumulated_loss"].item()
                count += 1
        self.net.train()
        return total_loss / count if count > 0 else np.inf

    def train(self):
        self.net.train()
        print(f"Starting Fine-Tuning for {self.config['epochs']} epochs...")
        print(f"Early Stopping Patience: {self.patience}")
        
        for epoch in range(self.config["epochs"]):
            t_start = time.time()
            loss_stats = {}
            data_pbar = tqdm.tqdm(self.train_loader)
            
            # Training Loop
            for data in data_pbar:
                loss_new = self.train_iteration(data)
                # logging
                for k, v in loss_new.items():
                    loss_stats[k] = loss_stats[k]+v if k in loss_stats else v
                data_pbar.set_description(f"loss: {loss_new['accumulated_loss'].item():.7f}")
            
            for k in loss_stats:
                loss_stats[k] /= len(self.train_loader)
            
            # Note: We might NOT want default Newbob decay on training loss if we are doing fine-tuning with fixed low LR
            # But the user asked for low LR. The scheduler logic in NewbobAdam might still be useful or we can just keep it.
            # However, if validation loss is the metric, we should maybe decay based on validation loss?
            # The original code updates LR based on training loss passed to update_lr.
            # I will keep using training loss for consistency with original scheduler, or fine-tuning strategy might interfere.
            # Given the request is "Set Learning Rate extremely low", aggressive decay might not be needed, but won't hurt.
            self.optimizer.update_lr(loss_stats["accumulated_loss"])
            
            # Validation Loop
            val_loss = self.validate()
            
            t_end = time.time()
            loss_str = "    ".join([f"{k}:{v:.4}" for k, v in loss_stats.items()])
            time_str = f"({time.strftime('%H:%M:%S', time.gmtime(t_end-t_start))})"
            print(f"epoch {epoch+1} " + loss_str + f"    val_loss:{val_loss:.4f}    " + time_str)
            
            # Early Stopping Check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save(suffix='best')
                print("Saved best model.")
            else:
                self.patience_counter += 1
                print(f"Validation loss did not improve. Counter: {self.patience_counter}/{self.patience}")
                if self.patience_counter >= self.patience:
                    print("Early stopping triggered.")
                    break
            
            # Save periodic model
            if self.config["save_frequency"] > 0 and (epoch + 1) % self.config["save_frequency"] == 0:
                self.save(suffix='epoch-' + str(epoch+1))
                print("Saved periodic model")
        
        # Save final model
        self.save(suffix='final')
