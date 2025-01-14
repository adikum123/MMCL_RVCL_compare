import json
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import mmcl.utils as utils
import rocl.data_loader as data_loader
from mmcl.losses import MMCL_PGD as MMCL_pgd


class MMCL_Encoder(nn.Module):

    def __init__(self, hparams, device):
        super(MMCL_Encoder, self).__init__()
        self.hparams = hparams
        self.crit = MMCL_pgd(
            sigma=self.hparams.kernel_gamma,
            batch_size=self.hparams.batch_size,
            anchor_count=2,
            C=self.hparams.C,
            solver_type=self.hparams.solver_type,
            use_norm=self.hparams.use_norm,
            device=device,
            kernel=self.hparams.kernel_type,
            eta=self.hparams.svm_lr,
        )
        self.device = device
        self.model = utils.load_model_contrastive(
            args=self.hparams, weights_loaded=False
        ).to(self.device)
        self.trainloader, self.traindst, self.testloader, self.testdst = (
            data_loader.get_dataset(self.hparams)
        )
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.hparams.encoder_lr,
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.hparams.step_size,
            gamma=self.hparams.scheduler_gamma,
        )

    def step(self, batch):
        original_batch, transformed_batch, transformed_batch_2, target = batch
        z = self.model(batch)
        loss, kxz, kyz, sparsity, num_zero, acc = self.criterion(z)
        return {
            "loss": loss,
            "contrast_acc": acc,
            "kxz": kxz,
            "kyz": kyz,
            "sparsity": sparsity,
            "num_zero": num_zero,
        }

    def forward(self, x):
        return self.model(x)

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    def train_step(self, batch, it=None):
        logs = self.step(batch)

        if self.hparams.dist == "ddp":
            self.trainsampler.set_epoch(it)
        if it is not None:
            logs["epoch"] = it / len(self.batch_trainsampler)
        return logs

    def train(self):
        for epoch in range(self.hparams.encoder_num_iters):
            self.model.train()  # Set the model to training mode

            total_loss, total_num = 0.0, 0
            train_bar = tqdm(self.trainloader, desc=f"Epoch {epoch+1}")

            for iii, (ori_image, pos_1, pos_2, target) in enumerate(train_bar):
                # Move data to device
                pos_1, pos_2 = pos_1.to(self.device, non_blocking=True), pos_2.to(
                    self.device, non_blocking=True
                )

                # Forward pass through the model
                feature_1 = self.model(pos_1)
                feature_2 = self.model(pos_2)
                features = torch.cat(
                    [feature_1.unsqueeze(1), feature_2.unsqueeze(1)], dim=1
                )
                # Compute loss
                loss, _, _, _, _, _ = self.crit(features)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # Update metrics
                batch_size = pos_1.size(0)
                total_num += batch_size
                total_loss += loss.item() * batch_size
                # Update progress bar description
                train_bar.set_description(
                    "Train Epoch: [{}/{}] Total Loss: {:.4e}".format(
                        epoch + 1,
                        self.hparams.encoder_num_iters,
                        total_loss / total_num,
                    )
                )
            self.scheduler.step()
            metrics = {
                "total_loss": total_loss / total_num,
                "epoch": epoch,
                "lr": self.get_lr(),
            }
            print(f"Epoch: {epoch+1}, Metrics: {metrics}")
