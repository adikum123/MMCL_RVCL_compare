import json
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F  # Add this line
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm

import mmcl.utils as utils
import rocl.data_loader as data_loader


class ResnetUnsupervised(nn.Module):

    def __init__(self, hparams, device):
        super(ResnetUnsupervised, self).__init__()
        self.hparams = hparams
        self.device = device
        print(f"Using device: {self.device}")
        self.encoder = self.load_resnet_encoder_from_ckpt(self.hparams.resnet_unsupervised_ckpt)
        self.encoder.to(self.device)
        if self.hparams.relu_layer:
            self.classifier = nn.Sequential(
                nn.Linear(2048, 4096),
                nn.ReLU(),
                nn.Linear(4096, 10),
            ).to(self.device)
        else:
            self.classifier = nn.Linear(2048, 10).to(self.device)
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
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.classifier.parameters(), lr=self.hparams.lr
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.hparams.step_size,
            gamma=self.hparams.scheduler_gamma,
        )
        self.best_model_saved = False
        self.min_epochs = 150

    def load_resnet_encoder_from_ckpt(self, ckpt):
        checkpoint = torch.load(ckpt, map_location=self.device)
        state_dict = checkpoint['state_dict']
        new_state_dict = {k.replace("convnet.", ""): v for k, v in state_dict.items() if not k.startswith("projection.")}
        model = models.resnet50(pretrained=False)
        model.load_state_dict(new_state_dict, strict=True)
        model.fc = torch.nn.Identity()
        return model

    def forward(self, x):
        # Upsample the input images to 224x224 using bilinear interpolation
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        with torch.no_grad():
            features = self.encoder(x)
        return self.classifier(features)

    def set_eval(self):
        self.classifier.eval()

    def get_model_save_name(self):
        prefix = "relu_" if self.hparams.relu_layer else ""
        return f"{prefix}linear_eval_resnet_unsupervised_bs_{self.hparams.batch_size}_lr_{self.hparams.lr}"

    def train(self):
        best_val_loss = float("inf")
        patience_counter = 0
        max_patience = 5  # Number of epochs to wait before stopping

        train_losses = []
        val_losses = []
        for epoch in range(self.hparams.num_iters):
            self.classifier.train()  # Set the model to training mode
            total_loss, total_samples = 0.0, 0
            train_bar = tqdm(self.trainloader, desc=f"Train Epoch {epoch + 1}")
            for iii, (ori_image, pos_1, pos_2, target) in enumerate(train_bar):
                ori_image, pos_1, pos_2, target = (
                    ori_image.to(self.device),
                    pos_1.to(self.device),
                    pos_2.to(self.device),
                    target.to(self.device),
                )
                # Combine images and corresponding targets
                images = torch.cat([ori_image, pos_1, pos_2], dim=0)
                logits = self.forward(x=images)
                targets = torch.cat([target, target, target], dim=0)
                # Backpropagation and optimizer step
                loss = self.criterion(logits, targets)
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
                val_bar = tqdm(self.valloader, desc=f"Val Epoch {epoch + 1}")
                # Validation Phase
                self.classifier.eval()  # Set the model to evaluation mode
                total_loss, total_num = 0.0, 0
                with torch.no_grad():
                    for iii, (ori_image, pos_1, pos_2, target) in enumerate(val_bar):
                        ori_image, pos_1, pos_2, target = (
                            ori_image.to(self.device),
                            pos_1.to(self.device),
                            pos_2.to(self.device),
                            target.to(self.device),
                        )
                        # Combine images and corresponding targets
                        images = torch.cat([ori_image, pos_1, pos_2], dim=0)
                        logits = self.forward(x=images)
                        targets = torch.cat([target, target, target], dim=0)
                        # Compute InfoNCE loss
                        loss = self.criterion(logits, targets)
                        batch_size = pos_1.size(0)
                        total_num += batch_size
                        total_loss += loss.item() * batch_size
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

    def test(self):
        """Evaluate the model on the test dataset."""
        self.classifier.eval()
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
            save_dir = "models/linear_evaluate"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{self.get_model_save_name()}.pt")
            torch.save(self.classifier.state_dict(), save_path)
