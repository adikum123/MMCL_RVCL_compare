import json
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F  # Add this line
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import mmcl.utils as utils
import rocl.data_loader as data_loader
from resnet.resnet import ResNet50


class ResnetSupervised(nn.Module):

    def __init__(self, hparams, device):
        super(ResnetSupervised, self).__init__()
        self.hparams = hparams
        self.device = device
        self.set_model()
        self.set_data_loader()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.hparams.lr
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.hparams.step_size,
            gamma=self.hparams.scheduler_gamma,
        )
        self.best_model_saved = False
        self.min_epochs = 50

    def set_model(self):
        self.model = ResNet50(cifar_head=True)
        self.model.fc = nn.Linear(2048, 10)

    def set_data_loader(self):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
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

    def get_model_save_name(self):
        return f"resnet_supervised_bs_{self.hparams.batch_size}_lr_{self.hparams.lr}"

    def train(self):
        best_val_loss = float("inf")
        patience_counter = 0
        max_patience = 5  # Number of epochs to wait before stopping

        train_losses = []
        val_losses = []
        for epoch in range(self.hparams.num_iters):
            self.model.train()  # Set the model to training mode
            total_loss, total_num = 0.0, 0
            train_bar = tqdm(self.trainloader, desc=f"Train Epoch {epoch + 1}")
            for iii, (images, targets) in enumerate(train_bar):
                images, targets = images.to(self.device), targets.to(self.device)
                logits = self.forward(x=images)
                loss = self.criterion(logits, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # Logging
                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                total_num += batch_size
                train_bar.set_postfix(loss=f"{total_loss / total_num:.4f}")
            train_loss = total_loss / total_num
            train_losses.append(train_loss)
            print(f"Epoch [{epoch+1}/{self.hparams.num_iters}] - Train Loss: {train_loss:.4f}")
            if self.hparams.use_validation:
                val_bar = tqdm(self.valloader, desc=f"Val Epoch {epoch + 1}")
                self.model.eval()
                total_loss, total_num = 0.0, 0
                with torch.no_grad():
                    for iii, (images, targets) in enumerate(val_bar):
                        images, targets = images.to(self.device), targets.to(self.device)
                        logits = self.forward(x=images)
                        loss = self.criterion(logits, targets)
                        batch_size = images.size(0)
                        total_num += batch_size
                        total_loss += loss.item() * batch_size
                val_loss = total_loss / total_num
                val_losses.append(val_loss)
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
            self.scheduler.step()
        save_dir = "plots/resnet_supervised"
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

    def test(self):
        """Evaluate the model on the test dataset."""
        self.model.eval()
        total_correct, total_num = 0, 0
        total_loss = 0.0
        test_bar = tqdm(self.testloader, desc="Testing")
        with torch.no_grad():
            for images, targets in test_bar:
                images, targets = images.to(self.device), targets.to(self.device)
                logits = self.forward(images)
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
        # Final metrics
        metrics = {
            "accuracy": total_correct / total_num,
            "loss": total_loss / total_num,
            "total_correct": total_correct,
            "total_num": total_num,
        }
        print(f"Test Results: {json.dumps(metrics, indent=4)}")
        return metrics

    def save(self):
        if not self.best_model_saved:
            save_dir = f"models/resnet"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, self.get_model_save_name() + ".pt")
            torch.save(self.model.state_dict(), save_path)  # Save state_dict instead