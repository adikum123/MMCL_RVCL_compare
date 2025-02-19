import json
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import mmcl.utils as utils
import rocl.data_loader as data_loader
from mmcl.losses import MMCL_PGD as MMCL_pgd


class RegularCLModel(nn.Module):

    def __init__(self, hparams, device):
        super(RegularCLModel, self).__init__()
        self.hparams = hparams
        self.device = device
        if 'regular_cl_checkpoint' in vars(self.hparams) and self.hparams.regular_cl_checkpoint:
            ckpt = self.hparams.regular_cl_checkpoint
            self.model = utils.load_model_contrastive_test(model=self.hparams.model, model_path=ckpt, device=device)
        else:
            self.model = utils.load_model_contrastive(
                args=self.hparams, weights_loaded=False
            ).to(self.device)
        # if self.hparams.use_validation:
        #     (
        #         self.trainloader,
        #         self.traindst,
        #         self.valloader,
        #         self.valdst,
        #         self.testloader,
        #         self.testdst,
        #     ) = data_loader.get_train_val_test_dataset(self.hparams)
        # else:
        self.trainloader, self.traindst, self.testloader, self.testdst = (
            data_loader.get_dataset(self.hparams)
        )
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.hparams.lr
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.hparams.step_size,
            gamma=self.hparams.scheduler_gamma,
        )
        self.best_model_saved = False

    def forward(self, x):
        return self.model(x)

    def info_nce_loss(self, f1, f2, temperature=0.5):
        """
        Computes the InfoNCE loss given two sets of representations f1 and f2.
        Args:
            f1: Tensor of shape (N, d) from first augmented view.
            f2: Tensor of shape (N, d) from second augmented view.
            temperature: Temperature scaling factor.
        Returns:
            A scalar InfoNCE loss.
        """
        batch_size = f1.size(0)
        # Normalize the representations
        f1 = F.normalize(f1, dim=1)
        f2 = F.normalize(f2, dim=1)
        # Concatenate along the batch dimension -> (2N, d)
        features = torch.cat([f1, f2], dim=0)
        # Compute similarity matrix (2N x 2N)
        similarity_matrix = torch.matmul(features, features.T)
        # Scale by temperature
        logits = similarity_matrix / temperature
        # To avoid trivial self-comparison, mask out the diagonal
        mask = torch.eye(2 * batch_size, device=logits.device).bool()
        logits.masked_fill_(mask, -1e9)

        # For each sample i in the batch, the positive example is at index (i+batch_size) % (2N)
        positive_indices = (torch.arange(2 * batch_size, device=logits.device) + batch_size) % (2 * batch_size)
        # Compute cross entropy loss using the positive indices as targets.
        loss = F.cross_entropy(logits, positive_indices)
        return loss

    def train(self):
        """
        Trains the model using InfoNCE loss. The train loader is assumed to return
        (original image, augmented view 1, augmented view 2, target). Only the two
        augmented views are used for the contrastive loss.
        """
        # You may want to set a temperature in hparams (defaulting to 0.5 if not provided)
        temperature = getattr(self.hparams, 'temperature', 0.5)
        best_running_loss = float("inf")
        for epoch in range(self.hparams.num_iters):
            self.model.train()  # Set the model to training mode
            total_loss, total_samples = 0.0, 0
            train_bar = tqdm(self.trainloader, desc=f"Epoch {epoch + 1}")

            for iii, (ori_image, pos_1, pos_2, target) in enumerate(train_bar):
                # Move augmented images to device
                pos_1 = pos_1.to(self.device)
                pos_2 = pos_2.to(self.device)
                # Forward pass for both augmented views
                f1 = self.model(pos_1)
                f2 = self.model(pos_2)
                # Compute InfoNCE loss
                loss = self.info_nce_loss(f1, f2, temperature)

                # Backpropagation and optimizer step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Logging
                batch_size = pos_1.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                train_bar.set_postfix(loss=f"{total_loss / total_samples:.4f}")

            # Step the learning rate scheduler at the end of each epoch
            self.scheduler.step()
            running_loss = total_loss/total_samples
            if running_loss < best_running_loss:
                print(f"Best running loss improved from {best_running_loss} to {running_loss}")
                best_running_loss = running_loss
                self.best_model_saved = False
                self.save()
                self.best_model_saved = True
            print(f"Epoch [{epoch+1}/{self.hparams.num_iters}] - Loss: {total_loss/total_samples:.4f}")

    def save(self):
        if not self.best_model_saved:
            save_dir = f'models/regular_cl/{self.hparams.kernel_type}'
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, self.get_model_save_name() + ".pkl")
            torch.save(self.model, save_path)
