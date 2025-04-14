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


class SimCLRModel(nn.Module):

    def __init__(
        self,
        base_model='resnet50',
        projection_dim=128,
        checkpoint_path="models/resnet_pretrained_models/resnet50_cifar10_bs1024_epochs1000.pth.tar"
    ):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        resnet = ResNet50(cifar_head=True)
        feat_dim = resnet.fc.in_features if hasattr(resnet, 'fc') else 2048
        resnet.fc = nn.Identity()
        self.convnet = resnet
        self.projection = nn.Sequential(
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, projection_dim, bias=False),
            nn.BatchNorm1d(projection_dim, affine=False)
        )
        self.checkpoint_path = checkpoint_path
        self.projection[0].name = 'fc1'
        self.projection[1].name = 'bn1'
        self.projection[3].name = 'fc2'
        self.projection[4].name = 'bn2'
        if self.checkpoint_path is not None and os.path.isfile(self.checkpoint_path):
            self.load_checkpoint()
        else:
            print(f"[Warning] No checkpoint found at: {self.checkpoint_path}")

    def forward(self, x):
        features = self.convnet(x)
        projections = self.projection(features)
        return projections

    def load_checkpoint(self):
        print(f"[Info] Loading checkpoint from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            if k.startswith("projection."):
                parts = k.split('.')
                if parts[1] == 'fc1':
                    parts[1] = '0'
                elif parts[1] == 'bn1':
                    parts[1] = '1'
                elif parts[1] == 'fc2':
                    parts[1] = '3'
                elif parts[1] == 'bn2':
                    parts[1] = '4'
                k = '.'.join(parts)
            new_state_dict[k] = v
        self.load_state_dict(new_state_dict, strict=False)
        print("[Info] Checkpoint loaded successfully.")

    def set_eval(self):
        self.convnet.eval()
        self.projection.eval()

    def set_train(self):
        self.convnet.train()
        self.projection.train()

    def save_finetune(self, num_layers):
        state_dict = self.state_dict()
        new_state_dict = {}
        for key in state_dict:
            new_key = key
            if key.startswith("projection."):
                parts = key.split(".")
                if parts[1] == "0":
                    parts[1] = "fc1"
                elif parts[1] == "1":
                    parts[1] = "bn1"
                elif parts[1] == "3":
                    parts[1] = "fc2"
                elif parts[1] == "4":
                    parts[1] = "bn2"
                new_key = ".".join(parts)
            new_state_dict[new_key] = state_dict[key]
        save_dir = os.path.join("models", "resnet_pretrained_models")
        os.makedirs(save_dir, exist_ok=True)
        save_name = f"finetune_{num_layers}_{os.path.basename(self.checkpoint_path)}"
        save_path = os.path.join(save_dir, save_name)
        torch.save({"state_dict": new_state_dict}, save_path)
        print(f"[Info] Finetuned model saved to: {save_path}")



class ResnetUnsupervised(nn.Module):

    def __init__(self, hparams, device):
        super(ResnetUnsupervised, self).__init__()
        self.hparams = hparams
        self.device = device
        self.encoder = SimCLRModel()
        self.set_classifier()
        self.set_data_loader()
        self.criterion = nn.CrossEntropyLoss()
        self.set_optimizer()
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.hparams.step_size,
            gamma=self.hparams.scheduler_gamma,
        )
        self.best_model_saved = False
        self.min_epochs = 80


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

    def set_optimizer(self):
        if "finetune" in vars(self.hparams):
            assert "finetune_num_layers" in vars(self.hparams), "Number of finetune layers not provided"
        lr = self.hparams.lr
        params_to_optimize = []
        if self.hparams.finetune:
            finetune_num_layers = self.hparams.finetune_num_layers
            if finetune_num_layers >= 1:
                params_to_optimize += list(self.encoder.projection.parameters())
            if finetune_num_layers >= 2:
                if hasattr(self.encoder.convnet, 'layer4'):
                    params_to_optimize += list(self.encoder.convnet.layer4.parameters())
                else:
                    print("[Warning] ConvNet does not have a 'layer4'. Skipping this layer.")
            if finetune_num_layers >= 3:
                if hasattr(self.encoder.convnet, 'layer3'):
                    params_to_optimize += list(self.encoder.convnet.layer3.parameters())
                else:
                    print("[Warning] ConvNet does not have a 'layer3'. Skipping this layer.")
            for name, param in self.encoder.named_parameters():
                if param not in params_to_optimize:
                    param.requires_grad = False
        else:
            for param in self.encoder.parameters():
                param.requires_grad = False
        params_to_optimize += list(self.classifier.parameters())
        self.optimizer = optim.Adam(
            params_to_optimize,
            lr=lr
        )

    def set_classifier(self):
        if self.hparams.relu_layer:
            self.classifier = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            ).to(self.device)
        else:
            self.classifier = nn.Linear(128, 10).to(self.device)


    def forward(self, x):
        if self.hparams.finetune:
            features = self.encoder(x)
        else:
            with torch.no_grad():
                features = self.encoder(x)
        logits = self.classifier(features)
        return logits

    def train(self):
        best_val_loss = float("inf")
        patience_counter = 0
        max_patience = 5  # Number of epochs to wait before stopping

        train_losses = []
        val_losses = []
        for epoch in range(self.hparams.num_iters):
            self.classifier.train()  # Set the model to training mode
            self.encoder.set_train()
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
            # validation
            if self.hparams.use_validation:
                val_bar = tqdm(self.valloader, desc=f"Val Epoch {epoch + 1}")
                self.classifier.eval()
                self.encoder.set_eval()
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
        save_dir = "plots/resnet_unsupervised"
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
        if self.hparams.finetune:
            self.encoder.save_finetune(num_layers=self.hparams.finetune_num_layers)

    def test(self):
        """Evaluate the model on the test dataset."""
        self.classifier.eval()
        self.encoder.eval()
        total_correct, total_num = 0, 0
        total_loss = 0.0
        test_bar = tqdm(self.testloader, desc="Testing")

        with torch.no_grad():  # No gradients required during evaluation
            for images, targets in test_bar:
                # Move data to device
                images, targets = images.to(self.device), targets.to(self.device)
                # Forward pass
                logits = self.forward(images)
                loss = self.criterion(logits, targets)
                # Predictions
                predictions = torch.argmax(logits, dim=1)
                correct = (predictions == targets).sum().item()
                # Update metrics
                total_correct += correct
                total_num += targets.size(0)
                total_loss += loss.item() * targets.size(0)
                # Update progress bar
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

    def get_model_save_name(self):
        encoder_name = self.hparams.resnet_unsupervised_ckpt.split("/")[-1]
        prefix = "relu_" if self.hparams.relu_layer else ""
        postfix = "finetune_" if self.hparams.finetune else ""
        return f"{prefix}linear_{postfix}{encoder_name}"

    def save(self):
        if not self.best_model_saved:
            save_dir = "models/linear_evaluate"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{self.get_model_save_name()}")
            torch.save(self.classifier.state_dict(), save_path)
