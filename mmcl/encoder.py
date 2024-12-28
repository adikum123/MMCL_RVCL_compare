import json
from typing import Union

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import mmcl.utils as utils
import rocl.data_loader as data_loader
from mmcl.svm_losses import MMCL_inv, MMCL_pgd


class MMCL_Encoder(nn.Module):

    def __init__(self, hparams, device):
        super(MMCL_Encoder, self).__init__()
        self.hparams = hparams
        if self.hparams.criterion_to_use == 'mmcl_inv':
            self.crit = MMCL_inv(
                sigma=self.hparams.gamma,
                batch_size=self.hparams.batch_size,
                anchor_count=2,
                C=self.hparams.C
            )
        elif self.hparams.criterion_to_use == 'mmcl_pgd':
            self.crit = MMCL_pgd(
                sigma=self.hparams.gamma,
                batch_size=self.hparams.batch_size,
                anchor_count=2,
                C=self.hparams.C,
                solver_type=self.hparams.solver_type,
                use_norm=self.hparams.use_norm
            )
        self.device = device
        self.model = utils.load_model_contrastive(args=self.hparams, weights_loaded=False)
        self.trainloader, self.traindst, self.testloader, self.testdst = data_loader.get_dataset(self.hparams)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=1e-6
        )
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma)

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

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def train_step(self, batch, it=None):
        logs = self.step(batch)

        if self.hparams.dist == "ddp":
            self.trainsampler.set_epoch(it)
        if it is not None:
            logs["epoch"] = it / len(self.batch_trainsampler)
        return logs

    def train_epoch(self, epoch):
        self.model.train()
        total_loss, total_num, train_bar = 0.0, 0, tqdm(self.trainloader)
        kxz_losses,kyz_losses = 0.0, 0.0
        for iii, (ori_image, pos_1, pos_2, target) in enumerate(train_bar):
            pos_1, pos_2 = pos_1.to(self.device,non_blocking=True), pos_2.to(self.device,non_blocking=True)
            feature_1 = self.model(pos_1)
            feature_2 = self.model(pos_2)

            features = torch.cat([feature_1.unsqueeze(1), feature_2.unsqueeze(1)], dim=1)

            loss = self.crit(features)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_num += self.hparams.batch_size
            total_loss += loss.item() * self.hparams.batch_size
            train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4e}'.format(epoch, self.hparams.num_iters, total_loss / total_num))
        self.scheduler.step()
        metrics = {
            'total_loss':total_loss / total_num,
            'epoch': epoch
        }
        return metrics

    def train(self):
        for epoch in range(self.hparams.num_iters):
            metrics = self.train_epoch(epoch=epoch)
            print(f'Epoch: {epoch+1}, metrics: {json.dumps(metrics, indent=4)}')