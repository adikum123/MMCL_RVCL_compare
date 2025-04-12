import json
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import mmcl.utils as utils
import rocl.data_loader as data_loader


class MultiClassLoss(nn.Module):

    def __init__(self, loss_type='cross_entropy', margin=1.0):
        super(MultiClassLoss, self).__init__()
        self.loss_type = loss_type.lower()
        self.margin = margin

        if self.loss_type == 'cross_entropy':
            self.loss_fn = nn.CrossEntropyLoss()
        elif self.loss_type == 'nll':
            self.loss_fn = nn.NLLLoss()
        elif self.loss_type == 'kl':
            self.loss_fn = nn.KLDivLoss(reduction='batchmean')
        elif self.loss_type == 'hinge':
            self.loss_fn = None  # we'll implement hinge manually
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def forward(self, logits, targets):
        """
        logits: raw model outputs (before softmax), shape (batch_size, num_classes)
        targets:
            - For CE/NLL: LongTensor of shape (batch_size,)
            - For KL: one-hot or soft targets of shape (batch_size, num_classes)
        """
        if self.loss_type == 'cross_entropy':
            return self.loss_fn(logits, targets)
        if self.loss_type == 'nll':
            log_probs = F.log_softmax(logits, dim=1)
            return self.loss_fn(log_probs, targets)
        if self.loss_type == 'kl':
            log_probs = F.log_softmax(logits, dim=1)
            # targets must be soft or one-hot (float)
            if targets.dtype != torch.float:
                targets = F.one_hot(targets, num_classes=logits.shape[1]).float()
            return self.loss_fn(log_probs, targets)
        if self.loss_type == 'hinge':
            # Convert targets to one-hot encoding
            target_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).float()
            correct_scores = torch.sum(logits * target_one_hot, dim=1, keepdim=True)
            margins = self.margin + logits - correct_scores
            margins = torch.clamp(margins, min=0)
            margins = margins * (1 - target_one_hot)  # zero out the correct class
            loss = torch.sum(margins) / logits.size(0)
            return loss



class SupervisedModel(nn.Module):

    def __init__(self, hparams, device):
        super(SupervisedModel, self).__init__()
        self.hparams = hparams
        self.device = device
        self.set_classifier()
        self.set_data_loader()
        self.criterion = MultiClassLoss(loss_type=self.hparams.loss_type)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.hparams.lr
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.hparams.step_size,
            gamma=self.hparams.scheduler_gamma,
        )
        self.best_model_saved = False
        self.min_epochs = 80

    def set_classifier(self):
        if self.hparams.relu_layer:
            self.model = nn.Sequential(
                nn.ZeroPad2d((1, 2, 1, 2)),
                nn.Conv2d(3, 32, (5, 5), stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 128, (4, 4), stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(8192, 100),
                nn.ReLU(),
                nn.Linear(100, 200), # adds additional relu layer
                nn.ReLU(),
                nn.Linear(200, 10)
            ).to(self.device)
        else:
            self.model = nn.Sequential(
                nn.ZeroPad2d((1, 2, 1, 2)),
                nn.Conv2d(3, 32, (5, 5), stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 128, (4, 4), stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(8192, 100),
                nn.ReLU(),
                nn.Linear(100, 10)
            ).to(self.device)

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
            train_size = 55000
            val_size = 5000
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
        prefix = "relu_" if self.hparams.relu_layer else ""
        return f"{prefix}supervised_{self.hparams.loss_type}_bs_{self.hparams.batch_size}_lr_{self.hparams.lr}"

    def get_total_images_and_targets(self, ori_image, trans1, trans2, targets):
        if self.hparams.trans:
            self.ori_image, trans1, trans2, targets = (
                ori_image.to(self.device),
                trans1.to(self.device),
                trans2.to(self.device),
                targets.to(self.device),
            )
            return torch.cat([ori_image, trans1, trans2], dim=0), torch.cat([targets, targets, targets], dim=0)
        ori_image, targets = (
            ori_image.to(self.device),
            targets.to(self.device),
        )
        return ori_image, targets

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
            train_bar = tqdm(self.trainloader, desc=f"Train Epoch {epoch + 1}")
            for iii, (images, targets) in enumerate(train_bar):
                images, targets = images.to(self.device), targets.to(self.device)
                logits = self.forward(x=images)
                # Backpropagation and optimizer step
                loss = self.criterion(logits, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # Logging
                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                train_bar.set_postfix(loss=f"{total_loss / total_samples:.4f}")
            train_loss = total_loss/ total_samples
            train_losses.append(train_loss)
            print(f"Epoch [{epoch+1}/{self.hparams.num_iters}] - Train Loss: {train_loss:.4f}")
            # validation
            if self.hparams.use_validation:
                val_bar = tqdm(self.valloader, desc=f"Val Epoch {epoch + 1}")
                # Validation Phase
                self.model.eval()  # Set the model to evaluation mode
                total_loss, total_samples = 0.0, 0
                with torch.no_grad():
                    for iii, (images, targets) in enumerate(val_bar):
                        images, targets = images.to(self.device), targets.to(self.device)
                        logits = self.forward(x=images)
                        loss = self.criterion(logits, targets)
                        batch_size = images.size(0)
                        total_samples += batch_size
                        total_loss += loss.item() * batch_size
                val_loss = total_loss / total_samples
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
        save_dir = "plots/supervised"
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
        total_correct, total_samples = 0, 0
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
                total_samples += targets.size(0)
                total_loss += loss.item() * targets.size(0)
                # Update progress bar
                test_bar.set_description(
                    "Test: Loss: {:.4e}, Acc: {:.2f}%".format(
                        total_loss / total_samples, 100 * total_correct / total_samples
                    )
                )
        # Final metrics
        metrics = {
            "accuracy": total_correct / total_samples,
            "loss": total_loss / total_samples,
            "total_correct": total_correct,
            "total_samples": total_samples,
        }
        print(f"Test Results: {json.dumps(metrics, indent=4)}")
        return metrics

    def save(self):
        if not self.best_model_saved:
            save_dir = f'models/supervised'
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, self.get_model_save_name() + ".pkl")
            torch.save(self.model, save_path)