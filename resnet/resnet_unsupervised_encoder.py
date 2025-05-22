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
        self.set_optimizer()
        self.set_scheduler()
        self.min_epochs = 80
        self.lambda_weight = 1/256

    def set_optimizer(self):
        try:
            self.optimizer = optim.Adam(
                list(self.convnet.parameters()) + list(self.projection.parameters()),
                lr=self.hparams.lr
            )
        except Exception as e:
            self.optimizer = None

    def set_scheduler(self):
        try:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.hparams.step_size,
                gamma=self.hparams.scheduler_gamma,
            )
        except Exception as e:
            self.scheduler = None

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

    def set_eval(self):
        self.convnet.eval()
        self.projection.eval()

    def set_train(self):
        self.convnet.train()
        self.projection.train()

    def is_in_train_mode(self):
        return self.convnet.training and self.projection.training

    def get_finetune_params(self):
        return (
            list(self.projection.parameters())
            + list(self.convnet.layer4.parameters())
            + list(self.convnet.layer2.parameters())
            + list(self.convnet.layer3.parameters())
        )

    def compute_loss_submethod(self, pos_1, pos_2):
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

    def generate_adversarial_example(self, anchor_img, positive_img, negatives, epsilon=4/255, alpha=1/255, num_iter=10):
        """
        Generates an instance-wise adversarial example from anchor_img (t(x))
        by maximizing the contrastive loss against positive_img (t0(x)) and negatives.

        Args:
            anchor_img (Tensor): t(x), the anchor view to perturb
            positive_img (Tensor): t0(x), the positive view of the same instance
            negatives (Tensor): Batch of negative embeddings
            epsilon (float): Perturbation bound
            alpha (float): Step size
            num_iter (int): Number of PGD steps

        Returns:
            adv_img (Tensor): The generated adversarial version of anchor_img
        """
        anchor_img = anchor_img.to(self.device)
        positive_img = positive_img.to(self.device)
        negatives = negatives.to(self.device)
        # Initial adversarial image is a copy of anchor
        adv_img = anchor_img.clone().detach().requires_grad_(True)
        # Compute target positive embeddings (no grad)
        with torch.no_grad():
            target_pos = self.forward(positive_img)
        for _ in range(num_iter):
            adv_embed = self.forward(adv_img)
            # Compute contrastive loss w.r.t. target_pos and negatives
            loss = self.contrastive_loss(adv_embed, positives=[target_pos], negatives=negatives)
            # Backprop and take a PGD step
            self.convnet.zero_grad()
            self.projection.zero_grad()
            loss.backward()

            # Update adversarial image
            grad = adv_img.grad.data
            adv_img = adv_img.detach() + alpha * torch.sign(grad)
            eta = torch.clamp(adv_img - anchor_img, min=-epsilon, max=epsilon)
            adv_img = torch.clamp(anchor_img + eta, 0, 1).detach()
            adv_img.requires_grad_()
        return adv_img

    def compute_loss(self, pos_1, pos_2):
        """
        Compute contrastive loss using clean or adversarial views.
        """
        if self.hparams.adversarial and self.is_in_train_mode():
            return self.compute_loss_adv(pos_1, pos_2)
        return self.compute_loss_submethod(pos_1, pos_2)

    def compute_loss_adv(self, pos_1, pos_2):
        pos_1, pos_2 = pos_1.to(self.device), pos_2.to(self.device)
        z1 = self.forward(pos_1)
        z2 = self.forward(pos_2)
        # Use z1 and z2 to build negatives (before generating adv)
        with torch.no_grad():
            negatives = torch.cat([z1, z2], dim=0).detach()
        # Generate adversarial example from pos_1
        adv_pos_1 = self.generate_adversarial_example(pos_1, pos_2, negatives)
        # Get embeddings for adversarial examples
        z1_adv = self.forward(adv_pos_1)
        # RoCL main loss (anchor: z1, positives: z2, z1_adv)
        loss_rocl = self.contrastive_loss(z1, positives=[z2, z1_adv], negatives=negatives)
        # Regularization loss (anchor: z1_adv, positive: z2)
        loss_reg = self.contrastive_loss(z1_adv, positives=[z2], negatives=negatives)
        total_loss = loss_rocl + self.lambda_weight * loss_reg
        return total_loss

    def contrastive_loss(self, anchor, positives, negatives, temperature=0.5):
        """
        Compute contrastive loss for an anchor embedding with sets of positive and negative embeddings.

        Args:
            anchor (Tensor): Anchor embeddings of shape (batch_size, dim)
            positives (List[Tensor]): List of positive embeddings (each of shape (batch_size, dim))
            negatives (Tensor): Negative embeddings of shape (num_negatives, dim)

        Returns:
            loss (Tensor): Scalar contrastive loss
        """
        # Normalize all embeddings
        anchor = F.normalize(anchor, dim=1)
        positives = [F.normalize(p, dim=1) for p in positives]
        negatives = F.normalize(negatives, dim=1)
        # Compute similarities
        sim_pos = torch.cat([torch.sum(anchor * p, dim=1, keepdim=True) for p in positives], dim=1)  # (batch, n_pos)
        sim_neg = anchor @ negatives.T  # (batch, n_neg)
        # Temperature scaling
        sim_pos /= temperature
        sim_neg /= temperature
        # Numerator: exp of positive similarities (sum across positive set)
        numerator = torch.sum(torch.exp(sim_pos), dim=1)
        # Denominator: exp of all similarities (positive + negative)
        denominator = numerator + torch.sum(torch.exp(sim_neg), dim=1)
        # Contrastive loss
        loss = -torch.log(numerator / denominator)
        return loss.mean()

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
        if self.hparams.adversarial:
            return f"adv_resnet_{self.hparams.loss_type}_bs_{self.hparams.batch_size}_lr_{self.hparams.lr}"
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

    def save_finetune(self):
        # get save dir
        save_dir = os.path.join("models", "resnet")
        os.makedirs(save_dir, exist_ok=True)
        save_name = f"finetune_{os.path.basename(self.hparams.resnet_encoder_ckpt)}"
        save_path = os.path.join(save_dir, save_name)
        # Save model weights only
        torch.save(self.state_dict(), save_path)
        print(f"[Info] Finetuned model weights saved to: {save_path}")