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


class ResnetUnsupervised(nn.Module):

    def __init__(self, hparams, device):
        super(ResnetUnsupervised, self).__init__()
        self.hparams = hparams
        self.device = device
        self.set_encoder_and_classifier()
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

    def set_encoder_and_classifier(self):
        self.encoder = self.load_resnet_encoder_from_ckpt(self.hparams.resnet_unsupervised_ckpt)
        self.encoder.to(self.device)
        if self.hparams.relu_layer:
            self.classifier = nn.Sequential(
                nn.Linear(2048, 1024), # reduce size in few steps
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 10),
            ).to(self.device)
        else:
            self.classifier = nn.Linear(2048, 10).to(self.device)

    def load_resnet_encoder_from_ckpt(self, ckpt):
        checkpoint = torch.load(ckpt, map_location=self.device)
        state_dict = checkpoint['state_dict']
        new_state_dict = {k.replace("convnet.", ""): v for k, v in state_dict.items() if not k.startswith("projection.")}
        model = models.resnet50(pretrained=False)
        model.load_state_dict(new_state_dict, strict=True)
        model.fc = torch.nn.Identity()
        return model

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
        if self.hparams.finetune:
            for param in self.encoder.layer4.parameters():
                param.requires_grad = True
            self.optimizer = optim.Adam(
                list(self.classifier.parameters()) + list(self.encoder.layer4.parameters()),
                lr=self.hparams.lr
            )
        else:
            self.optimizer = optim.Adam(
                self.classifier.parameters(), lr=self.hparams.lr
            )

    def forward(self, x):
        # Upsample the input images to 224x224 using bilinear interpolation
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        if self.hparams.finetune:
            features = self.encoder(x)
        else:
            with torch.no_grad():
                features = self.encoder(x)
        return self.classifier(features)

    def set_eval(self):
        self.classifier.eval()

    def get_model_save_name(self):
        encoder_name = self.hparams.resnet_unsupervised_ckpt.split("/")[-1]
        prefix = "relu_" if self.hparams.relu_layer else ""
        postfix = "finetune_" if self.hparams.finetune else ""
        return f"{prefix}linear_{postfix}{encoder_name}"

    def train(self):
        best_val_loss = float("inf")
        patience_counter = 0
        max_patience = 5  # Number of epochs to wait before stopping

        train_losses = []
        val_losses = []
        for epoch in range(self.hparams.num_iters):
            self.classifier.train()  # Set the model to training mode
            if self.hparams.finetune:
                self.encoder.train()
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
                self.encoder.eval()
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
            self.save_encoder_finetune()

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

    def save_encoder_finetune(self):
        save_dir = "models/resnet_pretrained_models"
        os.makedirs(save_dir, exist_ok=True)
        encoder_name = self.hparams.resnet_unsupervised_ckpt.split("/")[-1]
        prefix = "finetune_"
        save_path = os.path.join(save_dir, f"{prefix}{encoder_name}")
        torch.save(self.encoder.state_dict(), save_path)

    def save(self):
        if not self.best_model_saved:
            save_dir = "models/linear_evaluate"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{self.get_model_save_name()}")
            torch.save(self.classifier.state_dict(), save_path)
