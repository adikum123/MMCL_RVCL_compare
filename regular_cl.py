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
from cl_utils import ContrastiveLoss, RobustContrastiveLoss


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
            self.model.parameters(),
            lr=self.hparams.lr if "lr" in vars(self.hparams) else self.hparams.lr_encoder
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.hparams.step_size,
            gamma=self.hparams.scheduler_gamma,
        )
        self.set_loss_fn()
        self.best_model_saved = False
        self.min_epochs = 80

    def forward(self, x):
        return self.model(x)

    def set_train(self):
        self.model.train()

    def set_eval(self):
        self.model.eval()

    def set_loss_fn(self):
        try:
            if self.hparams.adversarial:
                self.crit = RobustContrastiveLoss(base_model=self)
            else:
                self.crit = ContrastiveLoss(loss_type=self.hparams.loss_type)
        except Exception as e:
            print(f"Error in setting loss function: {e}")
            self.crit = None

    def set_zero_grad(self):
        self.model.zero_grad()

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
                loss = self.crit(pos_1, pos_2)
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
                        loss = self.crit(pos_1, pos_2)
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
                if patience_counter >= max_patience and epoch >= self.min_epochs - 1:
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
        if self.hparams.adversarial:
            return f"adv_regular_cl_{self.hparams.loss_type}_bs_{self.hparams.batch_size}_lr_{self.hparams.lr}.pkl"
        return f"regular_cl_{self.hparams.loss_type}_bs_{self.hparams.batch_size}_lr_{self.hparams.lr}.pkl"

    def save(self):
        if not self.best_model_saved:
            save_dir = f'models/regular_cl'
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, self.get_model_save_name() + ".pkl")
            torch.save(self.model, save_path)

    def save_finetune(self, model_name, prefix=""):
        if model_name.endswith(".pkl"):
            model_name = model_name[:-4]
        save_dir = f"models/regular_cl"
        os.makedirs(save_dir, exist_ok=True)
        save_name = (
            f"{prefix}finetune_{model_name}" if not model_name.startswith("finetune_") else f"{prefix}{model_name}"
        )
        save_path = os.path.join(save_dir, f"{save_name}.pkl")
        torch.save(self.model, save_path)