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


class MMCL_Encoder(nn.Module):

    def __init__(self, hparams, device):
        super(MMCL_Encoder, self).__init__()
        self.hparams = hparams
        try:
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
        except Exception:
            self.crit = None
        self.device = device
        if 'mmcl_checkpoint' in vars(self.hparams):
            ckpt = self.hparams.mmcl_checkpoint
            print(f"Loaded model with ckpt: {ckpt}")
            self.model = utils.load_model_contrastive_mmcl(model=self.hparams.model, model_path=ckpt, device=device)
        else:
            self.model = utils.load_model_contrastive(
                args=self.hparams, weights_loaded=False
            ).to(self.device)
        print('Loading dataset')
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
        print('Dataset loaded')
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.hparams.lr
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.hparams.step_size,
            gamma=self.hparams.scheduler_gamma,
        )
        self.best_model_saved = False
        self.min_epochs = 60

    def set_eval(self):
        self.model.eval()

    def set_train(self):

        self.model.train()
        if self.crit is None:
            print("Warning: Loss function (crit) is not initialized. Check MMCL_PGD initialization.")
        if hasattr(self, "optimizer") and self.optimizer is not None:
            self.optimizer.zero_grad()  # Reset gradients
        print("Encoder set to training mode.")

    def forward(self, x):
        return self.model(x)

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    def train(self):
        best_val_loss = float("inf")
        patience_counter = 0
        max_patience = 5  # Number of epochs to wait before stopping

        train_losses = []
        val_losses = []

        for epoch in range(self.hparams.num_iters):
            # Training Phase
            self.model.train()  # Set the model to training mode
            total_loss, total_num = 0.0, 0
            train_bar = tqdm(self.trainloader, desc=f"Train Epoch {epoch + 1}")

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
                    "Train Epoch: [{}/{}] Running Loss: {:.4e}".format(
                        epoch + 1,
                        self.hparams.num_iters,
                        total_loss / total_num,
                    )
                )

            # Save training loss for the epoch
            train_loss = total_loss / len(self.traindst)
            train_losses.append(train_loss)

            val_loss = None
            if self.hparams.use_validation:
                val_bar = tqdm(self.valloader, desc=f"Val Epoch {epoch + 1}")
                # Validation Phase
                self.model.eval()  # Set the model to evaluation mode
                total_loss, total_num = 0.0, 0
                with torch.no_grad():
                    for iii, (ori_image, pos_1, pos_2, target) in enumerate(val_bar):
                        # Move data to device
                        pos_1, pos_2 = pos_1.to(
                            self.device, non_blocking=True
                        ), pos_2.to(self.device, non_blocking=True)

                        feature_1 = self.model(pos_1)
                        feature_2 = self.model(pos_2)
                        features = torch.cat(
                            [feature_1.unsqueeze(1), feature_2.unsqueeze(1)], dim=1
                        )

                        loss, _, _, _, _, _ = self.crit(features)
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
                    if epoch + 1 <= self.min_epochs:
                        print("Min number of epohcs not reached yet. Continue training.")
                        continue
                    break
            # Scheduler step
            self.scheduler.step()
            # Logging metrics
            metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch": epoch + 1,
                "lr": self.get_lr(),
            }
            print(f"\nEpoch: {epoch+1}, Metrics: {json.dumps(metrics, indent=4)}\n")
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
        def get_train_data_desc(hparams):
            if hparams.clean:
                return "clean"
            if hparams.adv_img:
                return "adversarial"
            if hparams.trans:
                return "transformed"

        if self.hparams.model_save_name:
            return self.hparams.model_save_name
        return (
            self.hparams.model
            + "_C_"
            + str(self.hparams.C)
            + "_kernel_type_"
            + str(self.hparams.kernel_type)
            + "_train_data_"
            + (get_train_data_desc(self.hparams))
        )

    def save(self):
        if not self.best_model_saved:
            save_dir = f'models/mmcl/{self.hparams.kernel_type}'
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, self.get_model_save_name() + ".pkl")
            torch.save(self.model, save_path)
