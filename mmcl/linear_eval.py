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
        self.encoder = encoder
        self.device = device
        self.classifier = nn.Linear(feature_dim, num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        (
            self.trainloader,
            self.traindst,
            self.valloader,
            self.valdst,
            self.testloader,
            self.testdst,
        ) = data_loader.get_train_val_test_dataset(self.hparams)
        self.optimizer = optim.SGD(
            self.classifier.parameters(), lr=self.hparams.lr, momentum=0.9
        )
        print(f"Step size: {self.hparams.step_size}")
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.hparams.step_size,
            gamma=self.hparams.scheduler_gamma,
        )
        # attacker info
        attacker = None
        save_name = ""
        if self.hparams.name != "":
            save_name = self.hparams.name + "_"
        if self.hparams.adv_img:
            attack_info = (
                "Adv_train_epsilon_"
                + str(self.hparams.epsilon)
                + "_alpha_"
                + str(self.hparams.alpha)
                + "_max_iters_"
                + str(self.hparams.k)
                + "_type_"
                + str(self.hparams.attack_type)
                + "_randomstart_"
                + str(self.hparams.random_start)
            )
            save_name += attack_info + "_"
            print("Adversarial training info...")
            print(attack_info)
            img_clip = min_max_value(self.hparams)
            self.attacker = FastGradientSignUntargeted(
                self.encoder,
                linear="None",
                epsilon=self.hparams.epsilon,
                alpha=self.hparams.alpha,
                min_val=img_clip["min"].to(self.device),
                max_val=img_clip["max"].to(self.device),
                max_iters=self.hparams.k,
                device=self.device,
                _type=self.hparams.attack_type,
            )
        self.best_model_saved = False

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        return self.classifier(features)

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    def get_total_inputs_and_targets(self, ori_image, input1, input2, targets):
        # Move data to device
        ori_image, input1, input2, targets = (
            ori_image.to(self.device),
            input1.to(self.device),
            input2.to(self.device),
            targets.to(self.device),
        )
        # Configure inputs based on args
        if self.hparams.trans:
            inputs = input1
        else:
            inputs = ori_image
        input_flag = False
        if self.hparams.adv_img:
            advinputs = self.attacker.perturb(
                original_images=inputs,
                labels=targets,
                random_start=self.hparams.random_start,
            )
        if self.hparams.clean:
            total_inputs = inputs
            total_targets = targets
            input_flag = True
        if self.hparams.ss:
            total_inputs = torch.cat((inputs, input2))
            total_targets = torch.cat((targets, targets))
        if self.hparams.adv_img:
            if input_flag:
                total_inputs = torch.cat((total_inputs, advinputs))
                total_targets = torch.cat((total_targets, targets))
            else:
                total_inputs = advinputs
                total_targets = targets
        return total_inputs, total_targets

    def train(self):
        best_val_loss = float("inf")
        patience_counter = 0
        max_patience = 5  # Number of epochs to wait before stopping
        for epoch in range(self.hparams.num_iters):
            self.classifier.train()
            total_loss, total_num = 0.0, 0
            train_bar = tqdm(self.trainloader, desc=f"Epoch {epoch+1}")
            for i, (ori_image, input1, input2, targets) in enumerate(train_bar):
                total_inputs, total_targets = self.get_total_inputs_and_targets(
                    ori_image, input1, input2, targets
                )
                # Forward pass
                logits = self.forward(x=total_inputs)
                loss = self.criterion(logits, total_targets)
                # Backpropagation and optimizer step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # Update metrics
                batch_size = total_inputs.size(0)
                total_num += batch_size
                total_loss += loss.item() * batch_size
                # log metrics for specific batch
                train_bar.set_description(
                    "Train Epoch: [{}/{}] Total Loss: {:.4e}".format(
                        epoch + 1,
                        self.hparams.num_iters,
                        total_loss / total_num,
                    )
                )
            val_loss = None
            if self.hparams.use_validation:
                val_bar = tqdm(self.valloader, desc=f"Epoch {epoch+1}")
                self.encoder.set_eval()
                val_loss, val_num = 0.0, 0
                with torch.no_grad():
                    for i, (ori_image, input1, input2, targets) in enumerate(val_bar):
                        total_inputs, total_targets = self.get_total_inputs_and_targets(
                            ori_image, input1, input2, targets
                        )
                        # Compute val loss
                        logits = self.forward(x=total_inputs)
                        loss = self.criterion(logits, total_targets)
                        batch_size = input1.size(0)
                        val_num += batch_size
                        val_loss += loss.item() * batch_size
                    val_loss /= val_num
                self.scheduler.step()
                # Early Stopping Check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    print(
                        f"Validation loss improved to {val_loss:.4e}. Saving model..."
                    )
                    self.best_model_saved = False
                    self.save()
                    self.best_model_saved = True
                else:
                    patience_counter += 1
                    print(
                        f"Validation loss did not improve. Patience: {patience_counter}/{max_patience}"
                    )

                if patience_counter >= max_patience:
                    print("Early stopping triggered. Training terminated.")
                    break
            metrics = {
                "total_train_loss": total_loss / total_num,
                "total_val_loss": val_loss,
                "epoch": epoch + 1,
                "lr": self.get_lr(),
            }
        # load best saved model as the classifier
        if self.hparams.use_validation:
            self.classifier = torch.load(os.path.join("models/linear_evaluate", self.get_model_save_name()), map_location=self.device)

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

    def get_model_save_name(self):
        return f"linear_{self.hparams.load_checkpoint.split('/')[-1]}"

    def save(self):
        if not self.best_model_saved:
            save_path = os.path.join("models/linear_evaluate", self.get_model_save_name())
            torch.save(self.classifier, save_path)