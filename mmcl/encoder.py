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
        (
            self.trainloader,
            self.traindst,
            self.valloader,
            self.valdst,
            self.testloader,
            self.testdst,
        ) = data_loader.get_train_val_test_dataset(self.hparams)
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=self.hparams.encoder_lr, momentum=0.9
        )
        print(f"Step size: {self.hparams.step_size}")
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.hparams.step_size,
            gamma=self.hparams.scheduler_gamma,
        )

    def set_eval(self):
        self.model.eval()

    def forward(self, x):
        return self.model(x)

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    def train(self):
        best_val_loss = float("inf")
        patience_counter = 0
        max_patience = 5  # Number of epochs to wait before stopping

        for epoch in range(self.hparams.encoder_num_iters):
            # Training Phase
            self.model.train()  # Set the model to training mode
            total_loss, total_num = 0.0, 0
            train_bar = tqdm(self.trainloader, desc=f"Epoch {epoch + 1}")
            val_bar = tqdm(self.valloader, desc=f"Epoch {epoch + 1}")

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
            # Validation Phase
            self.model.eval()  # Set the model to evaluation mode
            val_loss, val_num = 0.0, 0
            with torch.no_grad():
                for iii, (ori_image, pos_1, pos_2, target) in enumerate(self.valloader):
                    # Move data to device
                    pos_1, pos_2 = pos_1.to(self.device, non_blocking=True), pos_2.to(
                        self.device, non_blocking=True
                    )

                    feature_1 = self.model(pos_1)
                    feature_2 = self.model(pos_2)
                    features = torch.cat(
                        [feature_1.unsqueeze(1), feature_2.unsqueeze(1)], dim=1
                    )

                    loss, _, _, _, _, _ = self.crit(features)
                    batch_size = pos_1.size(0)
                    val_num += batch_size
                    val_loss += loss.item() * batch_size

            val_loss /= val_num

            # Scheduler step
            self.scheduler.step()

            # Logging metrics
            train_loss = total_loss / total_num
            metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch": epoch + 1,
                "lr": self.get_lr(),
            }
            print(f"Epoch: {epoch+1}, Metrics: {metrics}")

            # Early Stopping Check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                print(f"Validation loss improved to {val_loss:.4e}. Saving model...")
                torch.save(self.model.state_dict(), "best_model.pth")
            else:
                patience_counter += 1
                print(
                    f"Validation loss did not improve. Patience: {patience_counter}/{max_patience}"
                )

            if patience_counter >= max_patience:
                print("Early stopping triggered. Training terminated.")
                break
