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


class ContrastiveLoss(nn.Module):

    def __init__(self, loss_type="info_nce", temperature=0.5, lambda_param=5e-3):
        """
        Initializes the ContrastiveLoss module.
        Args:
            loss_type: String indicating which loss to use: "info_nce", "nce", "cosine", or "barlow".
            temperature: Temperature parameter for applicable losses.
            lambda_param: Lambda parameter for the Barlow Twins loss.
        """
        super(ContrastiveLoss, self).__init__()
        self.loss_type = loss_type
        self.temperature = temperature
        self.lambda_param = lambda_param

    def forward(self, f1, f2):
        if self.loss_type == "info_nce":
            return self.info_nce_loss(f1, f2)
        elif self.loss_type == "nce":
            return self.nce_loss(f1, f2)
        elif self.loss_type == "cosine":
            return self.cosine_similarity_loss(f1, f2)
        elif self.loss_type == "barlow":
            return self.barlow_twins_loss(f1, f2)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def info_nce_loss(self, f1, f2):
        """
        Computes the InfoNCE (NT-Xent) loss.
        """
        batch_size = f1.size(0)
        f1 = F.normalize(f1, dim=1)
        f2 = F.normalize(f2, dim=1)
        features = torch.cat([f1, f2], dim=0)  # (2N, d)
        similarity_matrix = torch.matmul(features, features.T)  # (2N, 2N)
        logits = similarity_matrix / self.temperature
        # Mask self-similarity by setting diagonal to a very low value
        mask = torch.eye(2 * batch_size, device=logits.device).bool()
        logits.masked_fill_(mask, -1e9)
        # For each sample, the positive is the corresponding augmented view.
        positive_indices = (torch.arange(2 * batch_size, device=logits.device) + batch_size) % (2 * batch_size)
        loss = F.cross_entropy(logits, positive_indices)
        return loss

    def nce_loss(self, f1, f2):
        """
        Computes the Noise Contrastive Estimation (NCE) loss.
        """
        batch_size = f1.size(0)
        f1 = F.normalize(f1, dim=1)
        f2 = F.normalize(f2, dim=1)
        logits = torch.matmul(f1, f2.T) / self.temperature  # (N, N) similarity matrix
        positives = torch.diag(logits)
        logsumexp = torch.logsumexp(logits, dim=1)
        loss = - (positives - logsumexp).mean()
        return loss

    def cosine_similarity_loss(self, f1, f2):
        """
        Computes a loss based on the negative cosine similarity.
        """
        f1 = F.normalize(f1, dim=1)
        f2 = F.normalize(f2, dim=1)
        loss = - (f1 * f2).sum(dim=1).mean()
        return loss

    def off_diagonal(self, x):
        """
        Returns a flattened view of the off-diagonal elements of a square matrix.
        """
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def barlow_twins_loss(self, f1, f2):
        """
        Computes the Barlow Twins loss.
        """
        batch_size = f1.size(0)
        f1_norm = (f1 - f1.mean(0)) / f1.std(0)
        f2_norm = (f2 - f2.mean(0)) / f2.std(0)
        c = torch.mm(f1_norm.T, f2_norm) / batch_size  # Cross-correlation matrix
        # On-diagonal: encourage values to be 1
        on_diag = torch.diagonal(c).add_(-1).pow(2).sum()
        # Off-diagonal: encourage values to be 0
        off_diag = self.off_diagonal(c).pow(2).sum()
        loss = on_diag + self.lambda_param * off_diag
        return loss


class RegularCLModel(nn.Module):

    def __init__(self, hparams, device):
        super(RegularCLModel, self).__init__()
        self.hparams = hparams
        self.device = device
        if 'regular_cl_checkpoint' in vars(self.hparams) and self.hparams.regular_cl_checkpoint:
            ckpt = self.hparams.regular_cl_checkpoint
            print(f"Loading regular cl model with checkpoint: {ckpt}")
            self.model = utils.load_model_contrastive_mmcl(model=self.hparams.model, model_path=ckpt, device=device)
        else:
            self.model = utils.load_model_contrastive(
                args=self.hparams, weights_loaded=False
            ).to(self.device)
        if self.hparams.use_validation:
            (
                self.trainloader,
                self.traindst,
                self.valloader,
                self.valdst,
                self.testloader,
                self.testdst,
            ) = data_loader.get_train_val_test_dataset(self.hparams)
        else:
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
        self.crit = ContrastiveLoss(
            loss_type=self.hparams.loss_type,
            temperature=self.hparams.temperature,
            lambda_param=self.hparams.lambda_param,
        )
        self.best_model_saved = False

    def forward(self, x):
        return self.model(x)

    def set_train(self):
        self.model.train()

    def set_eval(self):
        self.model.eval()

    def train(self):
        best_val_loss = float("inf")
        patience_counter = 0
        max_patience = 5  # Number of epochs to wait before stopping

        train_losses = []
        val_losses = []
        temperature = getattr(self.hparams, 'temperature', 0.5)
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
                loss = self.crit(f1, f2)

                # Backpropagation and optimizer step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Logging
                batch_size = pos_1.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                train_bar.set_postfix(loss=f"{total_loss / total_samples:.4f}")
            train_loss = total_loss/len(self.traindst)
            train_losses.append(train_loss)
            print(f"Epoch [{epoch+1}/{self.hparams.num_iters}] - Train Loss: {train_loss:.4f}")
            # validation
            if self.hparams.use_validation:
                val_bar = tqdm(self.valloader, desc=f"Epoch {epoch + 1}")
                # Validation Phase
                self.model.eval()  # Set the model to evaluation mode
                total_loss, total_num = 0.0, 0
                with torch.no_grad():
                    for iii, (ori_image, pos_1, pos_2, target) in enumerate(val_bar):
                        # Move augmented images to device
                        pos_1 = pos_1.to(self.device)
                        pos_2 = pos_2.to(self.device)
                        # Forward pass for both augmented views
                        f1 = self.model(pos_1)
                        f2 = self.model(pos_2)
                        # Compute InfoNCE loss
                        loss = self.info_nce_loss(f1, f2, temperature)
                        batch_size = pos_1.size(0)
                        total_num += batch_size
                        total_loss += loss.item() * batch_size
                        val_bar.set_description(
                            "Val Epoch: [{}/{}] Running Loss: {:.4e}".format(
                                epoch + 1,
                                self.hparams.num_iters,
                                total_loss / total_num,
                            )
                        )
                val_loss = total_loss / len(self.valdst)
                val_losses.append(val_loss)
                # Early Stopping Check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    print(
                        f"\nValidation loss improved to {val_loss:.4e}. Saving model..."
                    )
                    self.best_model_saved = False
                    self.save()
                    self.best_model_saved = True
                else:
                    patience_counter += 1
                    print(
                        f"\nValidation loss did not improve. Patience: {patience_counter}/{max_patience}"
                    )
                if patience_counter >= max_patience:
                    print("\nEarly stopping triggered. Training terminated.")
                    break
            # Step the learning rate scheduler at the end of each epoch
            self.scheduler.step()
        # Plot and save the training and validation loss
        save_dir = "plots/encoder"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir, f"{self.get_model_save_name()}_{time.time()}.png"
        )

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
        if self.hparams.use_validation:
            plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        print(f"\nLoss plot saved to {save_path}")

    def get_model_save_name(self):
        if self.hparams.model_save_name:
            return self.hparams.model_save_name
        return f"regular_cl_bs_{self.hparams.batch_size}_lr_{self.hparams.lr}.pkl"


    def save(self):
        if not self.best_model_saved:
            save_dir = f'models/regular_cl'
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, self.get_model_save_name() + ".pkl")
            torch.save(self.model, save_path)

    def save_finetune(self):
        save_dir = f'models/regular_cl'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{self.get_model_save_name()}_finetune.pkl")
        torch.save(self.model, save_path)
