import json

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import rocl.data_loader as data_loader


class LinearEval(nn.Module):

    def __init__(
        self,
        hparams,
        encoder,
        device,
        feature_dim=100,
        num_classes=10,
        freeze_encoder=True,
    ):
        super(LinearEval, self).__init__()
        self.hparams = hparams
        self.encoder = encoder
        self.device = device
        if freeze_encoder:
            self.freeze_encoder()
        self.classifier = nn.Linear(feature_dim, num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.trainloader, self.traindst, self.testloader, self.testdst = (
            data_loader.get_dataset(self.hparams)
        )
        self.optimizer = optim.Adam(
            self.classifier.parameters(),
            lr=self.hparams.linear_eval_lr,
            weight_decay=1e-6,
            betas=(0.9, 0.999),  # Default values for the Adam optimizer
            eps=1e-8,  # Small value to prevent division by zero
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.hparams.step_size,
            gamma=self.hparams.scheduler_gamma,
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        return self.classifier(features)

    def unfreeze_encoder(self):
        """Unfreeze the encoder for fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def freeze_encoder(self):
        """Freeze the encoder to train only the classifier."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def train_epoch(self, epoch):
        self.classifier.train()
        total_loss, total_num, train_bar = 0.0, 0, tqdm(self.trainloader)
        for i, (ori_image, _, _, target) in enumerate(train_bar):
            ori_image, target = ori_image.to(self.device), target.to(self.device)
            # compute logits and loss
            logits = self.forward(x=ori_image).to(self.device)
            loss = nn.CrossEntropyLoss()(logits, target)
            # do optimizer step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # rest ...
            total_num += self.hparams.batch_size
            total_loss += loss.item() * self.hparams.batch_size
            train_bar.set_description(
                "Train Epoch: [{}/{}] Total Loss: {:.4e}, loss: {}".format(
                    epoch,
                    self.hparams.linear_eval_num_iters,
                    total_loss / total_num,
                    loss,
                )
            )
        self.scheduler.step()
        metrics = {"total_loss": total_loss / total_num, "epoch": epoch}
        return metrics

    def train(self):
        for epoch in range(self.hparams.linear_eval_num_iters):
            metrics = self.train_epoch(epoch=epoch)
            print(f"Epoch: {epoch+1}, metrics: {json.dumps(metrics, indent=4)}")

    def test(self):
        """Evaluate the model on the test dataset."""
        self.classifier.eval()  # Set the classifier to evaluation mode
        self.linear.eval()
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
