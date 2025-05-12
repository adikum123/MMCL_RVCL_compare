import json
import math
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from vision_transformers.vit import ViT


# Based on: https://github.com/omihub777/ViT-CIFAR/tree/main
class VisionTransformerModel(nn.Module):

    def __init__(self, hparams, device):
        super(VisionTransformerModel, self).__init__()
        self.hparams = hparams
        self.device = device
        self.set_model()
        # set data loader
        self.set_data_loader()
        # set loss fn with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        # set optimizer with weight decay
        self.set_optimizer()
        self.set_scheduler()

    def set_model(self):
        # set model
        self.model = ViT(
            in_c=3,             # RGB images
            num_classes=10,     # CIFAR-10
            img_size=32,        # 32×32 pixels
            patch=8,            # 8×8 patches → (32/8)²=16 tokens
            dropout=0.0,        # no dropout
            num_layers=7,       # transformer depth
            hidden=384,         # embedding dim
            mlp_hidden=384*4,   # 1536
            head=12,            # attention heads
            is_cls_token=True   # use [CLS] token
        )
        print(f"Model: {self.model} with param number: {sum(p.numel() for p in self.model.parameters())}")
        if hasattr(self.hparams, "vit_ckpt") and self.hparams.vit_ckpt:
            ckpt = torch.load(self.hparams.vit_ckpt, map_location=self.device)
            self.model.load_state_dict(ckpt)
        self.model.to(self.device)
        return None

    def set_optimizer(self):
        try:
            # set optimizer with weight decay
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=self.hparams.lr, weight_decay=5e-5
            )
        except Exception:
            return None

    def set_scheduler(self):
        try:
            # learning rate schedule: warmup + cosine annealing
            warmup_epochs = 5
            total_epochs = self.hparams.num_iters
            cosine_epochs = total_epochs - warmup_epochs
            self.scheduler = optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[
                    optim.lr_scheduler.LinearLR(
                        self.optimizer,
                        start_factor=1e-6,
                        total_iters=warmup_epochs
                    ),
                    optim.lr_scheduler.CosineAnnealingLR(
                        self.optimizer,
                        T_max=cosine_epochs,
                        eta_min=1e-5
                    )
                ],
                milestones=[warmup_epochs]
            )
            # minimum train epochs before early stopping
            self.min_epochs = 50
        except Exception:
            return None

    def set_data_loader(self):
        transform_train = transforms.Compose([
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            ),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            ),
        ])

        full_train_set = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train
        )
        test_set = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test
        )

        self.testloader = DataLoader(
            test_set, batch_size=self.hparams.batch_size, shuffle=False, num_workers=2
        )

        if self.hparams.use_validation:
            total_size = len(full_train_set)
            val_size = 2000
            train_size = total_size - val_size
            train_set, val_set = random_split(full_train_set, [train_size, val_size])
            self.trainloader = DataLoader(
                train_set, batch_size=self.hparams.batch_size, shuffle=True, num_workers=2
            )
            self.valloader = DataLoader(
                val_set, batch_size=self.hparams.batch_size, shuffle=False, num_workers=2
            )
        else:
            self.trainloader = DataLoader(
                full_train_set, batch_size=self.hparams.batch_size, shuffle=True, num_workers=2
            )

    def forward(self, x):
        return self.model(x)

    def set_eval(self):
        self.model.eval()

    def train(self):
        best_val_loss = float("inf")
        patience_counter = 0
        max_patience = 5  # Number of epochs to wait before stopping

        train_losses = []
        val_losses = []
        for epoch in range(self.hparams.num_iters):
            self.model.train()
            total_loss, total_num = 0.0, 0
            train_bar = tqdm(self.trainloader, desc=f"Train Epoch {epoch + 1}")
            for images, targets in train_bar:
                images, targets = images.to(self.device), targets.to(self.device)
                logits = self(images)
                loss = self.criterion(logits, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                total_num += batch_size
                train_bar.set_postfix(loss=f"{total_loss / total_num:.4f}")
            train_loss = total_loss / total_num
            train_losses.append(train_loss)
            print(f"Epoch [{epoch+1}/{self.hparams.num_iters}] - Train Loss: {train_loss:.4f}")

            if self.hparams.use_validation:
                self.model.eval()
                total_loss, total_num = 0.0, 0
                val_bar = tqdm(self.valloader, desc=f"Val Epoch {epoch + 1}")
                with torch.no_grad():
                    for images, targets in val_bar:
                        images, targets = images.to(self.device), targets.to(self.device)
                        logits = self(images)
                        loss = self.criterion(logits, targets)
                        batch_size = images.size(0)
                        total_loss += loss.item() * batch_size
                        total_num += batch_size
                val_loss = total_loss / total_num
                val_losses.append(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    print(f"\nValidation loss improved to {val_loss:.4e}. Saving model...")
                    self.save()
                else:
                    patience_counter += 1
                    print(f"\nValidation loss did not improve. Patience: {patience_counter}/{max_patience}")
                if patience_counter >= max_patience and epoch >= self.min_epochs - 1:
                    print("\nEarly stopping triggered. Training terminated.")
                    break

            self.scheduler.step()

        # Plot and save the training and validation loss
        save_dir = "plots/vision_transformer"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{self.get_model_save_name()}.png")
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

    def test(self):
        """Evaluate the model on the test dataset."""
        self.set_eval()
        total_correct, total_num = 0, 0
        total_loss = 0.0
        test_bar = tqdm(self.testloader, desc="Testing")
        with torch.no_grad():
            for images, targets in test_bar:
                images, targets = images.to(self.device), targets.to(self.device)
                logits = self(images)
                loss = self.criterion(logits, targets)
                predictions = torch.argmax(logits, dim=1)
                correct = (predictions == targets).sum().item()
                total_correct += correct
                total_num += targets.size(0)
                total_loss += loss.item() * targets.size(0)
                test_bar.set_description(
                    "Test: Loss: {:.4e}, Acc: {:.2f}%".format(
                        total_loss / total_num, 100 * total_correct / total_num
                    )
                )
        metrics = {
            "accuracy": total_correct / total_num,
            "loss": total_loss / total_num,
            "total_correct": total_correct,
            "total_num": total_num,
        }
        print(f"Test Results: {json.dumps(metrics, indent=4)}")
        return metrics

    def get_model_save_name(self):
        return f"vision_transformer_bs_{self.hparams.batch_size}_lr_{self.hparams.lr}"

    def save(self):
        save_dir = "models/vision_transformer"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{self.get_model_save_name()}.pth")
        torch.save(self.model.state_dict(), save_path)
