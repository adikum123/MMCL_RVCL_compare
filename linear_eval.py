import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import rocl.data_loader as data_loader
from mmcl.utils import min_max_value
from rocl.attack_lib import FastGradientSignUntargeted
from rocl.loss import NT_xent, pairwise_similarity


class LinearEval(nn.Module):

    def __init__(
        self,
        hparams,
        encoder,
        device,
        feature_dim=100,
        num_classes=10,
    ):
        super(LinearEval, self).__init__()
        self.hparams = hparams
        self.device = device
        self.encoder = encoder.to(self.device)
        if self.hparams.finetune:
            for param in self.encoder.parameters():
                param.requires_grad = True  # Unfreeze encoder for fine-tuning
        else:
            for param in self.encoder.parameters():
                param.requires_grad = False  # Keep encoder frozen
        if self.hparams.relu_layer:
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, 2 * feature_dim),  # Hidden layer (2x feature_dim)
                nn.ReLU(),  # Activation function
                nn.Linear(2 * feature_dim, num_classes)  # Output layer
            ).to(self.device)
        else:
            self.classifier = nn.Linear(feature_dim, num_classes).to(self.device)
        print(f"Classifier in downstream task:\n{self.classifier}")
        self.criterion = nn.CrossEntropyLoss()
        (
            self.trainloader,
            self.traindst,
            self.valloader,
            self.valdst,
            self.testloader,
            self.testdst,
        ) = data_loader.get_train_val_test_dataset(self.hparams)
        if self.hparams.finetune:
            self.optimizer = optim.Adam([
                {"params": self.encoder.parameters(), "lr": self.hparams.lr_encoder},
                {"params": self.classifier.parameters(), "lr": self.hparams.lr_classifier}
            ])
        else:
            self.optimizer = optim.Adam(self.classifier.parameters(), lr=self.hparams.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.hparams.step_size,
            gamma=self.hparams.scheduler_gamma
        )
        self.best_model_saved = False
        self.min_epochs = 150

    def forward(self, x):
        if self.hparams.finetune:
            features = self.encoder(x)  # Allow gradient updates
        else:
            with torch.no_grad():
                features = self.encoder(x)  # Keep frozen
        return self.classifier(features)

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
            self.classifier.train()
            if self.hparams.finetune:
                self.encoder.set_train()  # Ensure encoder is in training mode
            total_loss, total_num = 0.0, 0
            train_bar = tqdm(self.trainloader, desc=f"Epoch {epoch + 1}")

            for i, (ori_image, _, _, targets) in enumerate(train_bar):
                ori_image, targets = (
                    ori_image.to(self.device),
                    targets.to(self.device)
                )
                logits = self.forward(x=ori_image)
                loss = self.criterion(logits, targets)
                # Backpropagation after full training phase
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # Update metrics
                batch_size = ori_image.size(0)
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

            # Compute final training loss for the epoch
            train_loss = total_loss / total_num
            train_losses.append(train_loss)

            val_loss = None
            if self.hparams.use_validation:
                val_bar = tqdm(self.valloader, desc=f"Epoch {epoch + 1}")
                self.encoder.set_eval()  # Set encoder to evaluation mode
                val_loss, val_num = 0.0, 0

                with torch.no_grad():
                    for i, (ori_image, _, _, targets) in enumerate(val_bar):
                        ori_image, _, _, targets = (
                            ori_image.to(self.device),
                            targets.to(self.device)
                        )
                        logits = self.forward(x=ori_image)
                        loss = self.criterion(logits, targets)
                        batch_size = ori_image.size(0)
                        val_num += batch_size
                        val_loss += loss.item() * batch_size

                    val_loss /= val_num
                    val_losses.append(val_loss)

                # Early Stopping Check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    print(f"Validation loss improved to {val_loss:.4e}. Saving model...")
                    self.best_model_saved = False
                    self.save()
                    self.best_model_saved = True
                else:
                    patience_counter += 1
                    print(f"Validation loss did not improve. Patience: {patience_counter}/{max_patience}")

                if patience_counter >= max_patience and epoch >= self.min_epochs - 1:
                    print("\nEarly stopping triggered. Training terminated.")
                    break

            # Scheduler step
            self.scheduler.step()
            # Logging metrics
            metrics = {
                "total_train_loss": train_loss,
                "total_val_loss": val_loss,
                "epoch": epoch + 1,
                "lr": self.get_lr(),
            }

        # Load best saved model as the classifier
        if self.hparams.use_validation:
            self.classifier = torch.load(os.path.join("models/linear_evaluate", self.get_model_save_name()), map_location=self.device)
        if self.hparams.finetune:
            self.save_encoder_finetune()

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

    def get_encoder_ckpt(self):
        if 'mmcl_checkpoint' in vars(self.hparams):
            return self.hparams.mmcl_checkpoint
        if 'rvcl_checkpoint' in vars(self.hparams):
            return self.hparams.rvcl_checkpoint
        if 'regular_cl_checkpoint' in vars(self.hparams):
            return self.hparams.regular_cl_checkpoint
        raise ValueError("Checkpoint not found in hparams.")

    def get_model_save_name(self):
        prefix = ""
        ckpt = self.get_encoder_ckpt()
        model_name = ckpt.split("/")[-1]
        model_name = (
            model_name.replace("finetune_", "") if model_name.startswith("finetune_") else model_name
        )
        if self.hparams.relu_layer:
            prefix = "relu_"
        if self.hparams.finetune:
            return f"{prefix}linear_finetune_{ckpt.split('/')[-1]}"
        return f"{prefix}linear_{ckpt.split('/')[-1]}"

    def save(self):
        if not self.best_model_saved:
            save_dir = "models/linear_evaluate"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, self.get_model_save_name())
            torch.save(self.classifier, save_path)

    def save_encoder_finetune(self):
        ckpt = self.get_encoder_ckpt()
        model_name = ckpt.split("/")[-1]
        self.encoder.save_finetune(model_name=model_name, prefix="relu_" if self.hparams.relu_layer else "")