import json
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import tqdm

import rocl.data_loader as data_loader
from mmcl.losses import MMCL_PGD as MMCL_pgd
from regular_cl import ContrastiveLoss
from resnet.resnet import ResNet50


class ResnetEncoder(nn.Module):

    def __init__(self, hparams, device):
        super().__init__()
        self.hparams = hparams
        self.device = device
        self.set_covnet_and_projection()
        self.set_data_loader()
        self.set_loss_fn()
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        ).to(device)
        self.optimizer = optim.Adam(
            list(self.convnet.parameters()) + list(self.projection.parameters()),
            lr=self.hparams.lr
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.hparams.step_size,
            gamma=self.hparams.scheduler_gamma,
        )
        self.min_epochs = 80

    def set_covnet_and_projection(self):
        if "resnet_encoder_ckpt" not in vars(self.hparams):
            resnet = ResNet50(cifar_head=True)
            feat_dim = resnet.fc.in_features if hasattr(resnet, "fc") else 2048
            projection_dim = 128
            resnet.fc = nn.Identity()
            self.convnet = resnet
            self.projection = nn.Sequential(
                nn.Linear(feat_dim, feat_dim, bias=False),
                nn.BatchNorm1d(feat_dim),
                nn.ReLU(),
                nn.Linear(feat_dim, projection_dim, bias=False),
                nn.BatchNorm1d(projection_dim, affine=False)
            )
            self.projection[0].name = "fc1"
            self.projection[1].name = "bn1"
            self.projection[3].name = "fc2"
            self.projection[4].name = "bn2"
            for param in self.convnet.parameters():
                param.requires_grad = True
            for param in self.projection.parameters():
                param.requires_grad = True
            self.convnet.to(self.device)
            self.projection.to(self.device)
            return None
        self.load_checkpoint()

    def set_data_loader(self):
        try:
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
            print("Data loaded!!!")
        except Exception as e:
            print(f"Could not load data due to: {e}")
            return None

    def set_loss_fn(self):
        """
            self.hparams.loss_type can be in [mmcl, info_nce, nce, cosine, barlow]
        """
        try:
            if self.hparams.loss_type == "mmcl":
                self.crit = MMCL_pgd(
                    sigma=self.hparams.kernel_gamma,
                    batch_size=self.hparams.batch_size,
                    anchor_count=2,
                    C=self.hparams.C,
                    solver_type=self.hparams.solver_type,
                    use_norm=self.hparams.use_norm,
                    device=self.device,
                    kernel=self.hparams.kernel_type,
                    eta=self.hparams.svm_lr,
                )
            else:
                self.crit = ContrastiveLoss(loss_type=self.hparams.loss_type)
        except Exception:
            return None

    def forward(self, x):
        return self.projection(self.convnet(x))

    def load_checkpoint(self):
        print(f"[Info] Loading checkpoint from {self.hparams.resnet_encoder_ckpt}")

        # === Initialize convnet and projection ===
        resnet = ResNet50(cifar_head=True)
        feat_dim = resnet.fc.in_features if hasattr(resnet, "fc") else 2048
        projection_dim = 128
        resnet.fc = nn.Identity()
        self.convnet = resnet
        self.projection = nn.Sequential(
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, projection_dim, bias=False),
            nn.BatchNorm1d(projection_dim, affine=False)
        )
        self.projection[0].name = "fc1"
        self.projection[1].name = "bn1"
        self.projection[3].name = "fc2"
        self.projection[4].name = "bn2"
        # Move models to device
        self.convnet.to(self.device)
        self.projection.to(self.device)
        # === Load weights ===
        checkpoint = torch.load(self.hparams.resnet_encoder_ckpt, map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            if k.startswith("projection."):
                parts = k.split(".")
                if parts[1] == "fc1":
                    parts[1] = "0"
                elif parts[1] == "bn1":
                    parts[1] = "1"
                elif parts[1] == "fc2":
                    parts[1] = "3"
                elif parts[1] == "bn2":
                    parts[1] = "4"
                k = ".".join(parts)
            new_state_dict[k] = v

        self.load_state_dict(new_state_dict, strict=False)
        print("[Info] Checkpoint loaded successfully.")

    def set_eval(self):
        self.convnet.eval()
        self.projection.eval()

    def set_train(self):
        self.convnet.train()
        self.projection.train()

    def get_finetune_params(self):
        return list(self.projection.parameters()) + list(self.convnet.layer4.parameters())

    def compute_loss(self, pos_1, pos_2):
        pos_1, pos_2 = pos_1.to(self.device), pos_2.to(self.device)
        feature_1 = self.forward(pos_1)
        feature_2 = self.forward(pos_2)
        if self.hparams.loss_type == "mmcl":
            features = torch.cat(
                [feature_1.unsqueeze(1), feature_2.unsqueeze(1)], dim=1
            )
            loss, _, _, _, _, _ = self.crit(features)
            return loss
        return self.crit(feature_1, feature_2)

    def train(self):
        best_val_loss = float("inf")
        patience_counter = 0
        max_patience = 5  # Number of epochs to wait before stopping
        train_losses = []
        val_losses = []
        for epoch in range(self.hparams.num_iters):
            # Training Phase
            self.set_train()  # Set the model to training mode
            total_loss, total_num = 0.0, 0
            train_bar = tqdm(self.trainloader, desc=f"Train Epoch {epoch + 1}")
            for iii, (ori_image, pos_1, pos_2, target) in enumerate(train_bar):
                loss = self.compute_loss(pos_1, pos_2)
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
                self.set_eval()  # Set the model to evaluation mode
                total_loss, total_num = 0.0, 0
                with torch.no_grad():
                    for iii, (ori_image, pos_1, pos_2, target) in enumerate(val_bar):
                        loss = self.compute_loss(pos_1, pos_2)
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
                    self.save()
                else:
                    patience_counter += 1
                    print(
                        f"\nValidation loss did not improve. Patience: {patience_counter}/{max_patience}"
                    )
                if patience_counter >= max_patience and epoch >= self.min_epochs - 1:
                    print("\nEarly stopping triggered. Training terminated.")
                    break
            # Scheduler step
            self.scheduler.step()
        # Plot and save the training and validation loss
        save_dir = "plots/encoder"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir, f"{self.get_model_save_name()}.png"
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
        if self.hparams.loss_type == "mmcl":
            return f"resnet_{self.hparams.loss_type}_{self.hparams.kernel_type}_C_{self.hparams.C}_bs_{self.hparams.batch_size}_lr_{self.hparams.lr}"
        return f"resnet_{self.hparams.loss_type}_bs_{self.hparams.batch_size}_lr_{self.hparams.lr}"

    def save(self):
        save_dir = os.path.join("models", "resnet")
        os.makedirs(save_dir, exist_ok=True)
        save_name = f"{self.get_model_save_name()}.pt"
        save_path = os.path.join(save_dir, save_name)
        # Save model weights only
        torch.save(self.state_dict(), save_path)
        print(f"[Info] Model weights saved to: {save_path}")
        # Optionally save hyperparams to match model
        with open(save_path.replace('.pt', '_hparams.json'), 'w') as f:
            json.dump(vars(self.hparams), f, indent=4)

    def save_finetune(self, num_layers=0):
        save_dir = os.path.join("models", "resnet")
        os.makedirs(save_dir, exist_ok=True)
        save_name = f"finetune_{self.get_model_save_name()}.pt"
        save_path = os.path.join(save_dir, save_name)
        # Save model weights only
        torch.save(self.state_dict(), save_path)
        print(f"[Info] Finetuned model weights saved to: {save_path}")
        # Optionally save hyperparams
        with open(save_path.replace('.pt', '_hparams.json'), 'w') as f:
            json.dump(vars(self.hparams), f, indent=4)